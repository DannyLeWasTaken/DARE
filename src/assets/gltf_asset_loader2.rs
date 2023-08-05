//! Asset loader for Gltf
use crate::assets::AttributeView;
use crate::utils::handle_storage::{Handle, Storage};
use crate::utils::memory;
use crate::utils::memory::make_transfer_buffer;
use crate::{app, assets};
use anyhow::Result;
use ash::vk;
use ash::vk::Format;
use gltf;
use gltf::json::validation::Checked;
use phobos::{IncompleteCmdBuffer, TransferCmdBuffer};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufReader, Read};
use std::ptr;
use std::sync::{Arc, RwLock};

/// Context containing any necessary data and/or information about the struct
pub struct GltfContext;

/// Define virtual types for later usage
mod loader_structs {
    use crate::assets::gltf_asset_loader2::AccessorSemantic;

    pub struct Buffer {
        pub data: Option<Vec<u8>>,
        pub index: usize,
        pub format: AccessorSemantic,
    }
    pub struct VBufferViewer {}
    pub struct VAccessor {}
    pub struct BufferType {}
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessorSemantic {
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
    pub fn load_scene(
        ctx: Arc<RwLock<app::Context>>,
        path: &std::path::Path,
    ) -> Result<assets::Scene> {
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
                            &document.accessors[gltf_primitive.indices.unwrap().value()],
                            &gltf_buffers_content,
                        ),
                        index: primitive_buffers.len(),
                        format: AccessorSemantic::Index,
                    },
                );
                document.accessors[gltf_primitive.indices.unwrap().value()].name = Some(format!(
                    "{} {} {:?}",
                    gltf_mesh.name.clone().unwrap_or("Unnamed".parse()?),
                    primitive_index,
                    AccessorSemantic::Index,
                ));
                for (semantic, accessor_index) in gltf_primitive.attributes.iter() {
                    if let Some(accessor) = document.accessors.get(accessor_index.value()) {
                        if let Checked::Valid(semantic) = semantic {
                            if let Some(view_index) = accessor.buffer_view {
                                let view = &document.buffer_views[view_index.value()];
                                match semantic {
                                    gltf::Semantic::Positions
                                    | gltf::Semantic::Normals
                                    | gltf::Semantic::TexCoords(0) => {
                                        // Accessors
                                        primitive_buffers.insert(
                                            AccessorSemantic::Gltf(semantic.clone()),
                                            loader_structs::Buffer {
                                                data: GltfContext::access_accessor_contents(
                                                    &document,
                                                    accessor,
                                                    &gltf_buffers_content,
                                                ),
                                                index: primitive_buffers.len(),
                                                format: AccessorSemantic::Gltf(semantic.clone()),
                                            },
                                        );
                                        document
                                            .accessors
                                            .get_mut(accessor_index.value())
                                            .unwrap()
                                            .name = Some(format!(
                                            "{} {} {:?}",
                                            gltf_mesh.name.clone().unwrap_or("Unnamed".parse()?),
                                            primitive_index,
                                            AccessorSemantic::Gltf(semantic.clone())
                                        ));
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
                    let mut index_buffer =
                        primitive_buffers.get_mut(&AccessorSemantic::Index).unwrap();
                    println!("Index is: {:?}", index_buffer.format);
                    let indices: &[u16] =
                        bytemuck::cast_slice(index_buffer.data.as_ref().unwrap().as_slice());
                    index_buffer.data = Some(
                        bytemuck::cast_slice(
                            &indices.iter().map(|x| *x as u32).collect::<Vec<u32>>(),
                        )
                        .to_vec(),
                    );
                }

                // Generate normal buffer if normal exists, if not stop.
                if primitive_buffers
                    .get(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                    .is_some()
                {
                    println!("Generating normals!");
                    // We can generate this one
                    let vertex_buffer =
                        &primitive_buffers[&AccessorSemantic::Gltf(gltf::Semantic::Positions)];
                    let index_buffer = &primitive_buffers[&AccessorSemantic::Index];
                    let vertices: &[f32] =
                        bytemuck::cast_slice(vertex_buffer.data.as_ref().unwrap().as_slice());
                    let indices: &[u32] =
                        bytemuck::cast_slice(index_buffer.data.as_ref().unwrap().as_slice());
                    let vertices = vertices
                        .chunks(3)
                        .map(|chunk| glam::Vec3::new(chunk[0], chunk[1], chunk[2]))
                        .collect::<Vec<glam::Vec3>>();
                    let generated_normal_buffer =
                        assets::utils::calculate_normals(&vertices, &indices.to_vec());
                    let normal_count = generated_normal_buffer.len();
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
                        name: Some(format!(
                            "{} {} {:?}",
                            gltf_mesh.name.clone().unwrap_or("Unnamed".parse()?),
                            primitive_index,
                            AccessorSemantic::Gltf(gltf::Semantic::Normals)
                        )),
                        normalized: false,
                        sparse: None,
                    });
                    gltf_primitive.attributes.insert(
                        Checked::Valid(gltf::Semantic::Normals),
                        gltf::json::Index::new((document.accessors.len() - 1) as u32),
                    );
                    primitive_buffers.insert(
                        AccessorSemantic::Gltf(gltf::Semantic::Normals),
                        loader_structs::Buffer {
                            data: Some(bytes),
                            index: primitive_buffers.len(),
                            format: AccessorSemantic::Gltf(gltf::Semantic::Normals),
                        },
                    );
                }
                // Put buffers into monolithic scene buffers
                scene_buffers
                    .get_mut(&AccessorSemantic::Index)
                    .unwrap()
                    .insert(
                        primitive_index,
                        primitive_buffers[&AccessorSemantic::Index].data.clone(),
                    );
                scene_buffers
                    .get_mut(&AccessorSemantic::Gltf(gltf::Semantic::Positions))
                    .unwrap()
                    .insert(
                        primitive_index,
                        primitive_buffers[&AccessorSemantic::Gltf(gltf::Semantic::Positions)]
                            .data
                            .clone(),
                    );
                scene_buffers
                    .get_mut(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                    .unwrap()
                    .insert(
                        primitive_index,
                        primitive_buffers[&AccessorSemantic::Gltf(gltf::Semantic::Normals)]
                            .data
                            .clone(),
                    );
                if let Some(primitive_buffer) =
                    primitive_buffers.get(&AccessorSemantic::Gltf(gltf::Semantic::TexCoords(0)))
                {
                    scene_buffers
                        .get_mut(&AccessorSemantic::Gltf(gltf::Semantic::TexCoords(0)))
                        .unwrap()
                        .insert(primitive_index, primitive_buffer.data.clone());
                }
            }
        }

        // Upload all data into host
        let mut scene_buffer_offsets: HashMap<AccessorSemantic, usize> = HashMap::new();
        let mut monolithic_buffer: Vec<u8> = Vec::new();
        {
            let mut buffer_offsets: usize = 0;
            for (semantic, scene_buffer) in scene_buffers.iter() {
                scene_buffer_offsets.insert(semantic.clone(), monolithic_buffer.len());
                for (index, buffer) in scene_buffer {
                    if let Some(buffer) = buffer {
                        buffer_offsets += buffer.len();
                        monolithic_buffer.extend_from_slice(buffer.as_slice());
                    }
                }
            }
        }
        let monolithic_buffer = make_transfer_buffer(
            ctx.clone(),
            monolithic_buffer.as_slice(),
            None,
            "Scene buffer",
        )
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
            .push(asset_scene.buffer_storage.insert(monolithic_buffer));
        let scene_buffer = asset_scene
            .buffer_storage
            .get_immutable(asset_scene.buffers.first().unwrap())
            .unwrap();
        // Load images
        let mut transfer_images: Vec<(phobos::Buffer, phobos::Image)> = Vec::new();

        // Load images first
        for (index, image) in document.images.iter().enumerate() {
            let image_data = gltf_images_content.get(index).unwrap();
            let pixels = GltfContext::convert_image_types_to_rgba(
                image_data.format,
                gltf_images_content.get(index).unwrap().pixels.as_slice(),
            );
            let transfer_buffer = make_transfer_buffer(
                ctx.clone(),
                pixels.as_slice(),
                None,
                image
                    .name
                    .as_deref()
                    .unwrap_or(format!("Unnamed Image {}", index).as_str()),
            )
            .unwrap();
            let mut ctx_write = ctx.write().unwrap();
            let image_phobos = phobos::Image::new(
                ctx_write.device.clone(),
                &mut ctx_write.allocator,
                phobos::image::ImageCreateInfo {
                    width: image_data.width,
                    height: image_data.height,
                    depth: 1,
                    usage: vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED,
                    format: get_image_type_rgba(image_data.format).unwrap(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    mip_levels: 1,
                    layers: 1,
                    memory_type: phobos::MemoryType::GpuOnly,
                },
            )
            .unwrap();
            transfer_images.push((transfer_buffer, image_phobos));
        }
        // Submit execution for image transfers
        {
            let ctx_read = ctx.read().unwrap();
            let temp_exec_manager = ctx_read.execution_manager.clone();
            let mut image_commands = temp_exec_manager
                .on_domain::<phobos::domain::Graphics>()
                .unwrap();
            for (staging_buffer, image) in transfer_images.iter() {
                // Transition layout
                {
                    let src_access_mask = vk::AccessFlags2::empty();
                    let dst_access_mask = vk::AccessFlags2::TRANSFER_WRITE;
                    let source_stage = vk::PipelineStageFlags2::TOP_OF_PIPE;
                    let destination_stage = vk::PipelineStageFlags2::TRANSFER;

                    let image_barrier = vk::ImageMemoryBarrier2 {
                        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                        p_next: ptr::null(),
                        src_stage_mask: source_stage,
                        src_access_mask,
                        dst_stage_mask: destination_stage,
                        dst_access_mask,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: unsafe { image.handle() },
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    };

                    image_commands = image_commands.pipeline_barrier(&vk::DependencyInfo {
                        s_type: vk::StructureType::DEPENDENCY_INFO,
                        p_next: ptr::null(),
                        dependency_flags: vk::DependencyFlags::empty(),
                        memory_barrier_count: 0,
                        p_memory_barriers: ptr::null(),
                        buffer_memory_barrier_count: 0,
                        p_buffer_memory_barriers: ptr::null(),
                        image_memory_barrier_count: 1,
                        p_image_memory_barriers: &image_barrier.clone(),
                    });
                }

                image_commands = image_commands
                    .copy_buffer_to_image(
                        &staging_buffer.view_full(),
                        &image.whole_view(vk::ImageAspectFlags::COLOR).unwrap(),
                    )
                    .unwrap();

                image_commands = image_commands.memory_barrier(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                );

                // Mipmap barrier
                let mut image_barrier = vk::ImageMemoryBarrier2 {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: ptr::null(),
                    src_stage_mask: vk::PipelineStageFlags2::BLIT
                        | vk::PipelineStageFlags2::TRANSFER,
                    src_access_mask: vk::AccessFlags2::empty(),
                    dst_stage_mask: vk::PipelineStageFlags2::BLIT
                        | vk::PipelineStageFlags2::TRANSFER,
                    dst_access_mask: vk::AccessFlags2::empty(),
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::UNDEFINED,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: unsafe { image.handle() },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };
            }

            // Submit image transfers for submission
            ctx_read
                .execution_manager
                .submit(image_commands.finish().unwrap())
                .unwrap()
                .wait()
                .unwrap();

            for (buffer, image) in transfer_images {
                asset_scene
                    .images
                    .push(asset_scene.image_storage.insert(image));
            }
        }

        // Load textures
        // Create a universal sampler
        {
            let ctx_read = ctx.read().unwrap();
            asset_scene.samplers.push(
                asset_scene.sampler_storage.insert(
                    phobos::Sampler::new(
                        ctx_read.device.clone(),
                        vk::SamplerCreateInfo {
                            mag_filter: vk::Filter::LINEAR,
                            min_filter: vk::Filter::LINEAR,
                            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                            address_mode_u: vk::SamplerAddressMode::REPEAT,
                            address_mode_v: vk::SamplerAddressMode::REPEAT,
                            address_mode_w: vk::SamplerAddressMode::REPEAT,
                            mip_lod_bias: 0.0,
                            anisotropy_enable: vk::FALSE,
                            max_anisotropy: 16.0,
                            compare_enable: vk::FALSE,
                            compare_op: vk::CompareOp::ALWAYS,
                            min_lod: 0.0,
                            max_lod: vk::LOD_CLAMP_NONE,
                            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
                            unnormalized_coordinates: vk::FALSE,
                            ..std::default::Default::default()
                        },
                    )
                    .unwrap(),
                ),
            );
        }

        for (index, texture) in document.textures.iter().enumerate() {
            asset_scene.textures.push(
                asset_scene.texture_storage.insert(assets::Texture {
                    name: Option::from(
                        texture
                            .name
                            .clone()
                            .unwrap_or(format!("Unnamed texture {}", index)),
                    ),
                    source: asset_scene.images[texture.source.value()].clone(),
                    sampler: asset_scene.samplers.get(0).unwrap().clone(),
                }),
            );
        }

        // Load materials
        for material in document.materials.iter() {
            let pbr = material.pbr_metallic_roughness.clone();
            let albedo_texture = pbr
                .base_color_texture
                .as_ref()
                .and_then(|x| asset_scene.textures.get(x.index.value()).cloned())
                .clone();
            if albedo_texture.is_none() {
                println!(
                    "No albedo texture found! Expected one at: {:?}",
                    pbr.base_color_texture
                        .and_then(|x| Option::from(x.index.value() as i32))
                        .unwrap_or(999i32)
                );
            }
            asset_scene
                .materials
                .push(asset_scene.material_storage.insert(assets::Material {
                    albedo_texture,
                    albedo_color: glam::Vec3::from_slice(pbr.base_color_factor.0.as_slice()),
                }));
        }

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
                            vk::Format::R32_UINT,
                        );
                        primitive_accessors.insert(AccessorSemantic::Index, attribute);
                    }

                    // Query all attributes
                    for (valid_index, valid_attribute) in gltf_primitive.attributes.iter() {
                        if let (Checked::Valid(valid_index), valid_attribute) =
                            (valid_index, valid_attribute)
                        {
                            use gltf::json::accessor::{ComponentType, Type};
                            if let (
                                &gltf::Semantic::Positions
                                | &gltf::Semantic::Normals
                                | &gltf::Semantic::TexCoords(0),
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
                                    match (
                                        accessor.component_type.unwrap().0,
                                        accessor.type_.unwrap(),
                                    ) {
                                        (ComponentType::F32, Type::Vec4) => {
                                            vk::Format::R32G32B32A32_SFLOAT
                                        }
                                        (ComponentType::F32, Type::Vec3) => {
                                            vk::Format::R32G32B32_SFLOAT
                                        }
                                        (ComponentType::F32, Type::Vec2) => {
                                            vk::Format::R32G32_SFLOAT
                                        }
                                        (ComponentType::F32, Type::Scalar) => {
                                            vk::Format::R32_SFLOAT
                                        }

                                        (ComponentType::U32, Type::Vec4) => {
                                            vk::Format::R32G32B32A32_UINT
                                        }
                                        (ComponentType::U32, Type::Vec3) => {
                                            vk::Format::R32G32B32_UINT
                                        }
                                        (ComponentType::U32, Type::Vec2) => vk::Format::R32G32_UINT,
                                        (ComponentType::U32, Type::Scalar) => vk::Format::R32_UINT,

                                        (ComponentType::U16, Type::Vec4) => {
                                            vk::Format::R16G16B16A16_UINT
                                        }
                                        (ComponentType::U16, Type::Vec3) => {
                                            vk::Format::R16G16B16_UINT
                                        }
                                        (ComponentType::U16, Type::Vec2) => vk::Format::R16G16_UINT,
                                        (ComponentType::U16, Type::Scalar) => vk::Format::R16_UINT,

                                        (ComponentType::U8, Type::Vec4) => {
                                            vk::Format::R8G8B8A8_UINT
                                        }
                                        (ComponentType::U8, Type::Vec3) => vk::Format::R8G8B8_UINT,
                                        (ComponentType::U8, Type::Vec2) => vk::Format::R8G8_UINT,
                                        (ComponentType::U8, Type::Scalar) => vk::Format::R16_UINT,

                                        (ComponentType::I16, Type::Vec4) => {
                                            vk::Format::R16G16B16A16_SINT
                                        }
                                        (ComponentType::I16, Type::Vec3) => {
                                            vk::Format::R16G16B16_SINT
                                        }
                                        (ComponentType::I16, Type::Vec2) => vk::Format::R16G16_SINT,
                                        (ComponentType::I16, Type::Scalar) => vk::Format::R16_SINT,

                                        (ComponentType::I8, Type::Vec4) => {
                                            vk::Format::R8G8B8A8_SINT
                                        }
                                        (ComponentType::I8, Type::Vec3) => vk::Format::R8G8B8_SINT,
                                        (ComponentType::I8, Type::Vec2) => vk::Format::R8G8_SINT,
                                        (ComponentType::I8, Type::Scalar) => vk::Format::R8_SINT,

                                        _ => vk::Format::UNDEFINED,
                                    },
                                );
                                primitive_accessors
                                    .insert(AccessorSemantic::Gltf(valid_index.clone()), attribute);
                            }
                        }
                    }
                    let normal = primitive_accessors
                        .get(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                        .unwrap()
                        .clone();
                    let tex = primitive_accessors
                        .get(&AccessorSemantic::Gltf(gltf::Semantic::TexCoords(0)))
                        .unwrap()
                        .clone();
                    if let Some(normal) = normal {
                        println!("Found normal!");
                    }
                    if let Some(tex) = tex {
                        println!("Found tex!");
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
                            .clone()
                            .unwrap(),
                        index_buffer: primitive_accessors
                            .get(&AccessorSemantic::Index)
                            .unwrap()
                            .clone()
                            .unwrap(),
                        normal_buffer: primitive_accessors
                            .get(&AccessorSemantic::Gltf(gltf::Semantic::Normals))
                            .unwrap()
                            .clone(),
                        tangent_buffer: None,
                        tex_buffer: primitive_accessors
                            .get(&AccessorSemantic::Gltf(gltf::Semantic::TexCoords(0)))
                            .unwrap()
                            .clone(),
                        material: gltf_primitive
                            .material
                            .map(|x| x.value() as i32)
                            .unwrap_or(-1),
                        transform: *transform,
                    };
                    asset_scene
                        .meshes
                        .push(asset_scene.meshes_storage.insert(mesh));
                }
            }
        }

        // Upload materials buffer
        let mut materials: Vec<assets::CMaterial> = Vec::with_capacity(asset_scene.materials.len());
        for material in asset_scene.materials.iter() {
            let material = asset_scene.material_storage.get_clone(material).unwrap();
            materials.push(material.to_c_struct(&asset_scene));
        }
        if !materials.is_empty() {
            asset_scene.material_buffer = Some(
                memory::make_transfer_buffer(ctx, materials.as_slice(), None, "Materials").unwrap(),
            );
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
        format: Format,
    ) -> Option<Handle<AttributeView>> {
        let view = document
            .buffer_views
            .get(accessor.buffer_view.unwrap().value())
            .unwrap();
        let gltf_buffer = document.buffers.get(view.buffer.value()).unwrap();
        let component_size = GltfContext::get_component_size(accessor);
        let buffer_size = component_size * accessor.count as usize;
        let total_offset = view.byte_offset.unwrap_or(0) + accessor.byte_offset;
        // Get buffer storage. We can hardcode this to the first as we're always making a monolithic buffer
        let asset_buffer = scene
            .buffer_storage
            .get_immutable(scene.buffers.first().unwrap())
            .unwrap();
        let buffer_view = asset_buffer
            .view(
                *offsets.get(accessor_type).unwrap_or(&0) as vk::DeviceSize,
                buffer_size as vk::DeviceSize,
            )
            .unwrap();

        let attribute_view = assets::AttributeView {
            buffer_view,
            stride: view.byte_stride.unwrap_or(component_size as u32) as u64,
            format,
            count: accessor.count as u64,
            component_size: component_size as u64,
        };
        println!(
            "{}: {:?}",
            accessor
                .name
                .clone()
                .unwrap_or(String::from("Unnamed accessor")),
            attribute_view.format
        );
        let attribute_view = scene.attributes_storage.insert(attribute_view);
        scene.attributes.push(attribute_view.clone());
        Some(attribute_view)
    }

    /// Get the Offset in a vector
    fn get_offset_in_vector(vec: &BTreeMap<usize, Vec<u8>>, index: usize) -> usize {
        vec.iter().take(index).map(|x| x.1.len()).sum()
    }

    /// Get the Offset in a vector
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

    /// Converts all incoming image types to texels
    fn convert_image_types_to_rgba(
        convert_from_format: gltf::image::Format,
        data: &[u8],
    ) -> Vec<u8> {
        use gltf::image::Format;
        let (component_size, component_count) = match convert_from_format {
            Format::R8 => (1, 1),
            Format::R8G8 => (1, 2),
            Format::R8G8B8 => (1, 3),
            Format::R8G8B8A8 => (1, 4),
            Format::R16 => (2, 1),
            Format::R16G16 => (2, 2),
            Format::R16G16B16 => (2, 3),
            Format::R16G16B16A16 => (2, 4),
            Format::R32G32B32FLOAT => (4, 3),
            Format::R32G32B32A32FLOAT => (4, 4),
        };

        // Split up the data into chunks, slice into it by the component Size:
        // If component does not exist, default to zero expect for the alpha channel

        let max_component_value = match component_size {
            1 => u8::MAX as f32,
            2 => u16::MAX as f32,
            4 => f32::MAX,
            _ => 1.0, // Default value for unknown component Size
        };

        let mut result = Vec::with_capacity(data.len() / component_size * 4);
        for pixel in data.chunks(component_size * component_count) {
            let r = pixel.get(0..component_size);
            let g = pixel.get(component_size..(2 * component_size));
            let b = pixel.get((component_size * 2)..(3 * component_size));
            let a = pixel.get((component_size * 3)..(4 * component_size));
            result.push(
                r.map(|x| GltfContext::normalize_component(x, max_component_value))
                    .unwrap_or(0),
            );
            result.push(
                g.map(|x| GltfContext::normalize_component(x, max_component_value))
                    .unwrap_or(0),
            );
            result.push(
                b.map(|x| GltfContext::normalize_component(x, max_component_value))
                    .unwrap_or(0),
            );
            result.push(
                a.map(|x| GltfContext::normalize_component(x, max_component_value))
                    .unwrap_or(0),
            );
        }
        result
    }

    /// Normalize a single component value
    fn normalize_component(component: &[u8], max_value: f32) -> u8 {
        let value = match component.len() {
            1 => component[0] as f32,
            2 => u16::from_be_bytes([component[0], component[1]]) as f32,
            4 => f32::from_be_bytes([component[0], component[1], component[2], component[3]]),
            _ => 0.0, // Default value for unknown component Size
        };

        (value / max_value * 255.0).round().clamp(0.0, 255.0) as u8
    }
}

/// Converts the image type from gltf to Vulkan format
fn get_image_type_rgba(in_format: gltf::image::Format) -> Option<vk::Format> {
    match in_format {
        gltf::image::Format::R8
        | gltf::image::Format::R8G8
        | gltf::image::Format::R8G8B8
        | gltf::image::Format::R8G8B8A8 => Some(vk::Format::R8G8B8A8_UNORM),
        gltf::image::Format::R16
        | gltf::image::Format::R16G16
        | gltf::image::Format::R16G16B16
        | gltf::image::Format::R16G16B16A16 => Some(vk::Format::R16G16B16A16_UNORM),
        _ => None,
    }
}
