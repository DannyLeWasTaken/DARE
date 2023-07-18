//! Asset loaders for Gltf
use crate::assets::gltf_asset_loader::GltfAssetLoader;
use crate::assets::AttributeView;
use crate::utils;
use crate::utils::handle_storage::{Handle, Storage};
use crate::utils::memory;
use crate::{app, assets};
use anyhow::Result;
use ash::vk;
use gltf;
use gltf::json::validation::Checked;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufReader, Read};

/// Context containing any necessary data and/or information about the struct
pub struct GltfContext;

/// Define virtual types for later usage
mod loader_structs {
    use crate::assets::gltf_asset_loader2::AccessorSemantic;

    pub struct Buffer {
        pub data: Option<Vec<u8>>,
        pub index: usize,
        pub format: AccessorSemantic,
        pub dimension:
    }
    pub struct VBufferViewer {}
    pub struct VAccessor {}
    pub struct BufferType {}
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum AccessorSemantic {
    Gltf(gltf::Semantic),
    Index,
}

impl GltfContext {
    /// Responsible for deserializing any given Gltf file path
    fn deserialize(
        path: &std::path::Path,
    ) -> Result<(gltf::Document, Option<std::borrow::Cow<'static, [u8]>>)> {
        let (document, bin) = match path
            .extension()
            .unwrap_or("".as_ref())
            .to_str()
            .unwrap_or("")
        {
            "Gltf" => {
                let gltf = gltf::Gltf::open(path)?;
                (gltf.document, None)
            }
            "glb" => {
                let reader = BufReader::new(std::fs::File::open(path)?);
                let glb = gltf::Glb::from_reader(reader)?;
                let cursor = std::io::Cursor::new(&glb.json);
                let gltf = gltf::Gltf::from_reader(cursor)?;
                (gltf.document, glb.bin)
            }
            _ => panic!("Invalid path extension"),
        };

        Ok((document, bin))
    }

    /// Turn the scene tree into a flat array of primitives with their transformations
    fn flatten_scene_meshes(
        scene: &gltf::json::Root,
        nodes: &[gltf::json::Node],
    ) -> Vec<(gltf::json::Mesh, usize, glam::Mat4)> {
        let mut meshes: Vec<(gltf::json::Mesh, usize, glam::Mat4)> = Vec::new();
        let mut seen_meshes: HashSet<usize> = HashSet::new();

        let mut nodes: Vec<(gltf::json::Node, glam::Mat4)> = nodes
            .iter()
            .map(|x| {
                (
                    x.clone(),
                    glam::Mat4::from_cols_array(
                        &x.matrix.unwrap_or(glam::Mat4::IDENTITY.to_cols_array()),
                    ),
                )
            })
            .collect();

        while let Some((node, transform)) = nodes.pop() {
            let node_transform = transform
                * glam::Mat4::from_cols_array(
                    &node.matrix.unwrap_or(glam::Mat4::IDENTITY.to_cols_array()),
                );
            if let Some(index) = node.mesh {
                if let Some(mesh) = scene.meshes.get(index.value()) {
                    if !seen_meshes.contains(&index.value()) {
                        seen_meshes.insert(index.value());
                        meshes.push((mesh.clone(), index.value(), node_transform));
                    }
                }
            }
            for child in node.children.iter() {
                for node_index in child {
                    if let Some(child) = scene.nodes.get(node_index.value()) {
                        nodes.push((child.clone(), node_transform));
                    }
                }
            }
        }
        meshes
    }

    /// HOW THE LOADING PROCESS WORKS
    /// Goals: We wish to load Gltf scenes in, but need to be able to perform processing on the
    /// meshes such as changing their types, dimensions, or generating normals
    /// Therefore, our process follows
    /// # Process
    /// 1. Serialize Gltf information, but not data itself (load buffer info, but not buffer)
    /// this process includes flattening the node tree of the scene
    /// 2. Process the scene. We are still not loading yet, but instead are creating virtual
    /// buffers for future buffers such as generated normals
    /// 3. Load scene & generate buffers -> Load in all data and generate the necessary buffers
    /// this will also process any data and transform them into the necessary types
    /// 4. Create handles & upload to host
    pub fn load_scene(ctx: &mut app::Context, path: &std::path::Path) -> Result<assets::Scene> {
        let (document, gltf_buffers_content, gltf_images_content) = gltf::import(path)?;
        let mut document = document.into_json();

        // Load meshes after flattening them
        let mut gltf_meshes: Vec<(gltf::json::Mesh, usize, glam::Mat4)> =
            GltfContext::flatten_scene_meshes(
                &document,
                document
                    .scenes
                    .get(document.scene.unwrap().value())
                    .unwrap()
                    .nodes
                    .iter()
                    .map(|x| document.nodes.get(x.value()).unwrap().clone())
                    .collect::<Vec<gltf::json::Node>>()
                    .as_slice(),
            );
        // Organize it so the meshes are now in order
        gltf_meshes.sort_by(|x, y| x.1.cmp(&y.1));
        let mut gltf_meshes = gltf_meshes
            .into_iter()
            .map(|x| (x.0, x.2))
            .collect::<Vec<(gltf::json::Mesh, glam::Mat4)>>();

        let mut scene_buffers: BTreeMap<AccessorSemantic, BTreeMap<usize, Option<Vec<u8>>>> =
            BTreeMap::new();
        scene_buffers.insert(AccessorSemantic::Index, BTreeMap::new()); // Index
        scene_buffers.insert(
            AccessorSemantic::Gltf(gltf::Semantic::Positions),
            BTreeMap::new(),
        ); // Vertex
        scene_buffers.insert(
            AccessorSemantic::Gltf(gltf::Semantic::Normals),
            BTreeMap::new(),
        ); // Normals
        scene_buffers.insert(
            AccessorSemantic::Gltf(gltf::Semantic::TexCoords(0)),
            BTreeMap::new(),
        ); // Texture coordinate (only first set is support as of now)
           // TODO: support multiple texture coordinates

        // Find all used buffers, put them into the monolithic buffer
        // If required some do not exist, generate them (normals)
        let mut primitive_index: usize = 0;
        for (index, (gltf_mesh, transform)) in gltf_meshes.iter_mut().enumerate() {
            for gltf_primitive in gltf_mesh.primitives.iter_mut() {
                primitive_index += 1;
                if gltf_primitive.indices.is_none() {
                    continue;
                }
                let mut primitive_buffers: HashMap<AccessorSemantic, loader_structs::Buffer> =
                    HashMap::new();
                primitive_buffers.insert(
                    AccessorSemantic::Index,
                    loader_structs::Buffer {
                        data: GltfContext::access_accessor_contents(
                            &document,
                            document
                                .accessors
                                .get(gltf_primitive.indices.unwrap().value())
                                .unwrap(),
                            &gltf_buffers_content,
                        ),
                        index: primitive_buffers.len(),
                        format: Semantic::Positions,
                    },
                );
                for (semantic, accessor_index) in gltf_primitive.attributes.iter() {
                    if let Some(accessor) = document.accessors.get(accessor_index.value()) {
                        if let Checked::Valid(semantic) = semantic {
                            if let Some(view_index) = accessor.buffer_view {
                                let view = document.buffer_views.get(view_index.value()).unwrap();
                                match semantic {
                                    gltf::Semantic::Positions
                                    | gltf::Semantic::Normals
                                    | gltf::Semantic::TexCoords(0) => {
                                        primitive_buffers.insert(
                                            AccessorSemantic::Gltf(semantic.clone()),
                                            loader_structs::Buffer {
                                                data: GltfContext::access_accessor_contents(
                                                & document,
                                                accessor,
                                                &gltf_buffers_content,
                                                ),
                                                index: primitive_buffers.len(),
                                                format: AccessorSemantic::Gltf(semantic.clone())
                                            },
                                        );
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                // Check for absolute requirements
                if primitive_buffers
                    .get(&AccessorSemantic::Gltf(gltf::Semantic::Positions))
                    .is_none()
                {
                    continue;
                }
                {
                    // Convert all indices to u32
                    let mut index_buffer = primitive_buffers.get_mut(&AccessorSemantic::Index).unwrap();
                    let indices: &[u16] = bytemuck::cast_slice(primitive_buffers.get(&AccessorSemantic::Index).unwrap().data.unwrap().as_slice());
                    index_buffer.data = Some(bytemuck::bytes_of(&indices.into_iter().map(|x| {
                        *x as u32
                    }).collect::<Vec<u32>>()).to_vec());
                }

                if primitive_buffers
                    .get(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                    .is_none()
                {
                    // We can generate this one
                    let vertex_buffer = primitive_buffers.get(&AccessorSemantic::Gltf(gltf::Semantic::Positions)).unwrap();
                    let index_buffer = primitive_buffers.get(&AccessorSemantic::Index).unwrap();
                    let vertices: &[f32] = bytemuck::cast_slice(vertex_buffer.data.unwrap().as_slice());
                    let indices: &[u32] = bytemuck::cast_slice(index_buffer.data.unwrap().as_slice());
                    let vertices = vertices
                        .chunks(3)
                        .map(|chunk| glam::Vec3::new(chunk[0], chunk[1], chunk[2]))
                        .collect::<Vec<glam::Vec3>>();
                    let generated_normal_buffer = assets::utils::calculate_normals(&vertices, indices);

                    let mut bytes: Vec<u8> = Vec::with_capacity(generated_normal_buffer.len() * 12);
                    for normal in generated_normal_buffer.into_iter() {
                        bytes.extend_from_slice(&normal.x.to_ne_bytes());
                        bytes.extend_from_slice(&normal.y.to_ne_bytes());
                        bytes.extend_from_slice(&normal.z.to_ne_bytes());
                    }
                    document.buffers.push(gltf::json::buffer::Buffer {
                        byte_length: bytes.len() as u32,
                        name: Some(String::from("Normal buffer")),
                        uri: None,
                        extensions: None,
                        extras: Default::default(),
                    });
                    document.buffer_views.push(gltf::json::buffer::View {
                        buffer: gltf::json::Index::new((document.buffers.len() - 1) as u32),
                        byte_length: bytes.len() as u32,
                        byte_offset: None,
                        byte_stride: None,
                        name: None,
                        target: None,
                        extensions: None,
                        extras: Default::default(),
                    });
                    document.accessors.push(gltf::json::Accessor {
                        buffer_view: Some(gltf::json::Index::new(
                            (document.buffer_views.len() - 1) as u32,
                        )),
                        byte_offset: 0,
                        count: normal_count as u32,
                        component_type: Checked::Valid(gltf::json::accessor::GenericComponentType(
                            gltf::json::accessor::ComponentType::F32,
                        )),
                        extensions: None,
                        extras: Default::default(),
                        type_: Checked::Valid(gltf::json::accessor::Type::Vec3),
                        min: None,
                        max: None,
                        name: None,
                        normalized: false,
                        sparse: None,
                    });
                    gltf_primitive.attributes.insert(
                        Checked::Valid(gltf::Semantic::Normals),
                        gltf::json::Index::new((document.accessors.len() - 1) as u32),
                    );
                    normal_buffer = Some(bytes);
                }
                // Put buffers into monolithic global buffers
                scene_buffers
                    .get_mut(&AccessorSemantic::Index)
                    .unwrap()
                    .insert(primitive_index, index_buffer.clone());
                scene_buffers
                    .get_mut(&AccessorSemantic::Gltf(gltf::Semantic::Positions))
                    .unwrap()
                    .insert(primitive_index, vertex_buffer);
                scene_buffers
                    .get_mut(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                    .unwrap()
                    .insert(primitive_index, normal_buffer);
                scene_buffers
                    .get_mut(&AccessorSemantic::Gltf(gltf::Semantic::TexCoords(0)))
                    .unwrap()
                    .insert(primitive_index, tex_buffer);
            }
        }

        // Upload all data into buffers
        let mut scene_buffer_offsets: HashMap<AccessorSemantic, usize> = HashMap::new();
        let mut monolithic_buffer: Vec<u8> = Vec::new();
        {
            let mut buffer_offsets: usize = 0;
            for (semantic, scene_buffer) in scene_buffers.iter() {
                scene_buffer_offsets.insert(semantic.clone(), buffer_offsets);
                for (index, buffer) in scene_buffer {
                    if let Some(buffer) = buffer {
                        buffer_offsets += buffer.len();
                        monolithic_buffer.extend_from_slice(buffer.as_slice());
                    }
                }
            }
        }
        let scene_buffer =
            memory::make_transfer_buffer(ctx, monolithic_buffer.as_slice(), None, "Scene buffer")
                .expect("Failed to allocate scene buffer");

        // Create scene struct
        let mut asset_scene: assets::Scene = assets::Scene {
            meshes_storage: Storage::new(),
            buffer_storage: Storage::new(),
            attributes_storage: Storage::new(),
            image_storage: Storage::new(),
            sampler_storage: Storage::new(),
            texture_storage: Storage::new(),
            material_storage: Storage::new(),
            buffers: vec![],
            images: vec![],
            samplers: vec![],
            meshes: vec![],
            attributes: vec![],
            textures: vec![],
            materials: vec![],
            material_buffer: None,
        };
        asset_scene
            .buffers
            .push(asset_scene.buffer_storage.insert(scene_buffer));
        let scene_buffer = asset_scene
            .buffer_storage
            .get_immutable(asset_scene.buffers.first().unwrap())
            .unwrap();

        // Update accessors to reflect the compacted changes
        primitive_index = 0;
        let asset_meshes: Vec<assets::Mesh> = Vec::new();
        for (gltf_mesh, transform) in gltf_meshes.iter_mut() {
            for (mesh_prim_index, gltf_primitive) in gltf_mesh.primitives.iter().enumerate() {
                primitive_index += 1;
                if let (
                    Some(Some(index_buffer)),
                    Some(Some(vertex_buffer)),
                    Some(Some(normal_buffer)),
                    Some(Some(tex_buffer)),
                ) = (
                    scene_buffers
                        .get(&AccessorSemantic::Index)
                        .unwrap()
                        .get(&primitive_index),
                    scene_buffers
                        .get(&AccessorSemantic::Gltf(gltf::Semantic::Positions))
                        .unwrap()
                        .get(&primitive_index),
                    scene_buffers
                        .get(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                        .unwrap()
                        .get(&primitive_index),
                    scene_buffers
                        .get(&AccessorSemantic::Gltf(gltf::Semantic::TexCoords(0)))
                        .unwrap()
                        .get(&primitive_index),
                ) {
                    let mut primitive_accessors: HashMap<
                        AccessorSemantic,
                        Option<Handle<assets::AttributeView>>,
                    > = HashMap::new();
                    // Include index
                    {
                        let accessor = document
                            .accessors
                            .get(gltf_primitive.indices.unwrap().value())
                            .unwrap();
                        let attribute = GltfContext::add_attribute_to_scene(
                            &mut asset_scene,
                            &document,
                            &scene_buffer_offsets,
                            accessor,
                            &AccessorSemantic::Index,
                        );
                        primitive_accessors.insert(AccessorSemantic::Index, attribute);
                    }

                    // Query all attributes
                    for (valid_index, valid_attribute) in gltf_primitive.attributes.iter() {
                        if let (Checked::Valid(valid_index), valid_attribute) =
                            (valid_index, valid_attribute)
                        {
                            if let (
                                &gltf::Semantic::Positions
                                | &gltf::Semantic::Normals
                                | &gltf::Semantic::TexCoords(_),
                                attribute,
                            ) = (valid_index, valid_attribute)
                            {
                                let m = document.accessors.get(attribute.value()).unwrap();
                                let accessor =
                                    document.accessors.get(attribute.value()).unwrap().clone();
                                let attribute = GltfContext::add_attribute_to_scene(
                                    &mut asset_scene,
                                    &document,
                                    &scene_buffer_offsets,
                                    &accessor,
                                    &AccessorSemantic::Gltf(valid_index.clone()),
                                );
                                primitive_accessors
                                    .insert(AccessorSemantic::Gltf(valid_index.clone()), attribute);
                            }
                        }
                    }

                    let mesh = assets::Mesh {
                        name: Some(format![
                            "{} primitive {}",
                            gltf_mesh.name.as_ref().unwrap_or(&String::from("Unnamed")),
                            mesh_prim_index
                        ]),
                        vertex_buffer: primitive_accessors
                            .get(&AccessorSemantic::Gltf(gltf::Semantic::Positions))
                            .unwrap()
                            .unwrap(),
                        index_buffer: primitive_accessors
                            .get(&AccessorSemantic::Index)
                            .unwrap()
                            .unwrap(),
                        normal_buffer: *primitive_accessors
                            .get(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                            .unwrap(),
                        tangent_buffer: None,
                        tex_buffer: None,
                        material: 0,
                        transform: *transform,
                    };
                    primitive_index += 1;
                }
            }
        }
        Ok(asset_scene)
    }

    /// Adds any given attribute onto the scene and returns a handle
    fn add_attribute_to_scene(
        scene: &mut assets::Scene,
        document: &gltf::json::Root,
        offsets: &HashMap<AccessorSemantic, usize>,
        accessor: &gltf::json::Accessor,
        accessor_type: &AccessorSemantic,
    ) -> Option<Handle<AttributeView>> {
        let view = document.buffer_views.get(accessor.buffer_view?.value())?;
        let gltf_buffer = document.buffers.get(view.buffer.value())?;
        let component_size = GltfContext::get_component_size(accessor);
        let buffer_size = component_size * accessor.count as usize;
        // Get buffer storage. We can hardcode this to the first as we're always making a monolithic buffer
        let asset_buffer = scene
            .buffer_storage
            .get_immutable(scene.buffers.first()?)
            .unwrap();
        let buffer_view = asset_buffer
            .view(
                *offsets.get(accessor_type)? as vk::DeviceSize,
                buffer_size as vk::DeviceSize,
            )
            .ok()?;

        let attribute_view = assets::AttributeView {
            buffer_view,
            stride: view.byte_stride.unwrap_or(component_size as u32) as u64,
            format: Default::default(),
            count: accessor.count as u64,
            component_size: component_size as u64,
        };
        let attribute_view = scene.attributes_storage.insert(attribute_view);
        scene.attributes.push(attribute_view);
        Some(attribute_view)
    }

    /// Get the offset in a vector
    fn get_offset_in_vector(vec: &BTreeMap<usize, Vec<u8>>, index: usize) -> usize {
        vec.iter().take(index).map(|x| x.1.len()).sum()
    }

    /// Get the offset in a vector
    fn get_offset_in_optional_vector(
        vec: &BTreeMap<usize, Option<Vec<u8>>>,
        index: usize,
    ) -> usize {
        vec.iter().take(index).flat_map(|v| v.1).map(Vec::len).sum()
    }

    /// Accesses the contents of an accessor given
    fn access_accessor_contents(
        scene: &gltf::json::Root,
        accessor: &gltf::json::Accessor,
        data: &[gltf::buffer::Data],
    ) -> Option<Vec<u8>> {
        let view = scene.buffer_views.get(accessor.buffer_view?.value())?;
        let buffer_data = data.get(view.buffer.value())?.0.as_slice();
        let buffer = scene.buffers.get(view.buffer.value())?;
        let total_offset = (accessor.byte_offset + view.byte_offset.unwrap_or(0)) as usize;
        let stride =
            view.byte_stride
                .unwrap_or(GltfContext::get_component_size(accessor) as u32) as usize;
        let acessor_size = stride * (accessor.count as usize);
        assert!(acessor_size <= view.byte_length as usize);
        assert!(total_offset <= buffer.byte_length as usize);
        assert!((total_offset + acessor_size) <= buffer.byte_length as usize);
        Some(
            buffer_data
                .get(total_offset..(total_offset + acessor_size))
                .unwrap()
                .to_vec(),
        )
    }

    fn get_component_size(accessor: &gltf::json::Accessor) -> usize {
        accessor.component_type.unwrap().0.size() * accessor.type_.unwrap().multiplicity()
    }
}
