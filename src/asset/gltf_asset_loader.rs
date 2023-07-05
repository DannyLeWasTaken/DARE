//! Helper struct to load in gltf files and parse them into [`asset::Scene`] objects
//!
//! [`asset::Scene`]: crate::asset::Scene

use crate::app;
use crate::asset;
use crate::utils::handle_storage;
use anyhow;
use anyhow::Result;
use gltf::accessor::DataType;
use gltf::Semantic;
use phobos::vk;
use phobos::vk::Handle;
use std::collections::HashMap;
use std::path::Path;

pub struct GltfAssetLoader {}

impl GltfAssetLoader {
    pub fn new() -> Self {
        Self {}
    }

    /// Loads any gltf asset given a file
    pub fn load_asset_from_file(&self, gltf_path: &Path, ctx: &mut app::Context) -> asset::Scene {
        let mut scene = asset::Scene {
            meshes: HashMap::new(),
            buffers: HashMap::new(),
            attributes: HashMap::new(),
            meshes_storage: handle_storage::Storage::new(),
            buffer_storage: handle_storage::Storage::new(),
            attributes_storage: handle_storage::Storage::new(),
        };
        let gltf = gltf::Gltf::open(gltf_path).unwrap();
        let (document, buffers, images) = gltf::import(gltf_path).unwrap();

        // Quick, store it all into memory!
        for gltf_buffer in document.buffers() {
            let buffer = buffers.get(gltf_buffer.index()).unwrap();
            let buffer_data = &*buffer.0;
            let gpu_buffer = phobos::Buffer::new(
                ctx.device.clone(),
                &mut ctx.allocator,
                gltf_buffer.length() as u64,
                vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                phobos::MemoryType::CpuToGpu,
            )
            .unwrap();
            gpu_buffer
                .view_full()
                .mapped_slice::<u8>()
                .unwrap() // Stored as bytes
                .copy_from_slice(buffer_data);

            // Debug naming
            ctx.device
                .set_name(&gpu_buffer, gltf_buffer.name().unwrap_or("Unnamed"))
                .expect("Unable to debug name");

            println!("[gltf]: New buffer!");
            println!("Buffer size: {}", gpu_buffer.size());
            println!("Buffer data size in bytes: {}", buffer_data.len());

            // Store buffer in scene
            let buffer_handle = scene.buffer_storage.insert(gpu_buffer);
            scene
                .buffers
                .insert(gltf_buffer.index() as u64, buffer_handle);
        }

        // Load accessors first then load meshes second
        for gltf_mesh in document.meshes() {
            for gltf_primitive in gltf_mesh.primitives() {
                // Only accept primitives which have indices
                if gltf_primitive.indices().is_none() {
                    println!(
                        "{}'s primitive #{} does not have indices. Skipping loading primitive.",
                        gltf_mesh.name().unwrap(),
                        gltf_primitive.index()
                    );
                    continue;
                }

                // Gets the handle for any given accessor
                let get_attribute_handle =
                    |accessor: gltf::Accessor,
                     semantic: Option<gltf::Semantic>|
                     -> handle_storage::Handle<asset::AttributeView> {
                        // Create a buffer view
                        let accessor_viewer = &accessor.view().unwrap();
                        let total_offset = accessor_viewer.offset() + accessor.offset();
                        let component_size = accessor.size();
                        let stride = accessor_viewer.stride().unwrap_or(component_size);
                        let buffer_size = stride * (accessor.count() - 1) + component_size;

                        // Get buffer which is referenced and store attribute in scene
                        let buffer = scene
                            .buffer_storage
                            .get_immutable(
                                scene
                                    .buffers
                                    .get(&(accessor_viewer.buffer().index() as u64))
                                    .unwrap(),
                            )
                            .unwrap();

                        println!("Name: {}", accessor.name().unwrap_or("unnamed"));
                        println!(
                            "Semantic: {:#?}",
                            semantic.unwrap_or(gltf::Semantic::Weights(0))
                        );
                        println!("Buffer Start Address: {}", buffer.address());
                        println!(
                            "Buffer offset: {} + {} = {}",
                            accessor_viewer.offset(),
                            accessor.offset(),
                            total_offset
                        );
                        println!("Buffer End Address: {}", total_offset + buffer_size);
                        println!("Buffer stride: {}", stride);
                        println!("View size: {}", buffer_size);
                        println!("Component size: {}", component_size);
                        println!("Data type: {:?}", accessor.data_type());
                        println!("Dimension: {:?}", accessor.dimensions());
                        println!("\n\n");

                        use gltf::accessor::Dimensions;

                        scene.attributes_storage.insert(asset::AttributeView {
                            buffer_view: buffer
                                .view(total_offset as u64, buffer_size as u64)
                                .unwrap(),
                            stride: stride as u64,
                            count: accessor.count() as u64,
                            component_size: accessor.size() as u64,
                            // Oh god
                            format: match (accessor.data_type(), accessor.dimensions()) {
                                (DataType::F32, Dimensions::Vec4) => {
                                    vk::Format::R32G32B32A32_SFLOAT
                                }
                                (DataType::F32, Dimensions::Vec3) => vk::Format::R32G32B32_SFLOAT,
                                (DataType::F32, Dimensions::Vec2) => vk::Format::R32G32_SFLOAT,
                                (DataType::F32, Dimensions::Scalar) => vk::Format::R32_SFLOAT,

                                (DataType::U32, Dimensions::Vec4) => vk::Format::R32G32B32A32_UINT,
                                (DataType::U32, Dimensions::Vec3) => vk::Format::R32G32B32_UINT,
                                (DataType::U32, Dimensions::Vec2) => vk::Format::R32G32_UINT,
                                (DataType::U32, Dimensions::Scalar) => vk::Format::R32_UINT,

                                (DataType::U16, Dimensions::Vec4) => vk::Format::R16G16B16A16_UINT,
                                (DataType::U16, Dimensions::Vec3) => vk::Format::R16G16B16_UINT,
                                (DataType::U16, Dimensions::Vec2) => vk::Format::R16G16_UINT,
                                (DataType::U16, Dimensions::Scalar) => vk::Format::R16_UINT,

                                (DataType::U8, Dimensions::Vec4) => vk::Format::R8G8B8A8_UINT,
                                (DataType::U8, Dimensions::Vec3) => vk::Format::R8G8B8_UINT,
                                (DataType::U8, Dimensions::Vec2) => vk::Format::R8G8_UINT,
                                (DataType::U8, Dimensions::Scalar) => vk::Format::R8_UINT,

                                (DataType::I16, Dimensions::Vec4) => vk::Format::R16G16B16A16_SINT,
                                (DataType::I16, Dimensions::Vec3) => vk::Format::R16G16B16_SINT,
                                (DataType::I16, Dimensions::Vec2) => vk::Format::R16G16_SINT,
                                (DataType::I16, Dimensions::Scalar) => vk::Format::R16_SINT,

                                (DataType::I8, Dimensions::Vec4) => vk::Format::R8G8B8A8_SINT,
                                (DataType::I8, Dimensions::Vec3) => vk::Format::R8G8B8_SINT,
                                (DataType::I8, Dimensions::Vec2) => vk::Format::R8G8_SINT,
                                (DataType::I8, Dimensions::Scalar) => vk::Format::R8_SINT,

                                _ => vk::Format::UNDEFINED,
                            },
                        })
                    };

                // Indices
                {
                    let accessor = gltf_primitive.indices().unwrap();
                    if scene.attributes.get(&(accessor.index() as u64)).is_none() {
                        scene.attributes.insert(
                            accessor.index() as u64,
                            get_attribute_handle(accessor, None),
                        );
                    }
                }

                // Get everything else!
                for gltf_attribute in gltf_primitive.attributes() {
                    match gltf_attribute.0 {
                        gltf::Semantic::Positions
                        | gltf::Semantic::Normals
                        | gltf::Semantic::Tangents => {
                            // Check if accessor exists already
                            let accessor = gltf_attribute.1;
                            if scene.attributes.get(&(accessor.index() as u64)).is_some() {
                                continue;
                            }

                            // Create an attribute handle
                            scene.attributes.insert(
                                accessor.index() as u64,
                                get_attribute_handle(accessor, Some(gltf_attribute.0)),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        // Select just one scene for now
        for gltf_node in document.scenes().next().unwrap().nodes() {
            let gltf_mat = glam::Mat4::from_cols(
                glam::Vec4::from(gltf_node.transform().matrix()[0]),
                glam::Vec4::from(gltf_node.transform().matrix()[1]),
                glam::Vec4::from(gltf_node.transform().matrix()[2]),
                glam::Vec4::from(gltf_node.transform().matrix()[3]),
            );
            let asset_meshes = self.flatten_meshes(&scene, gltf_node, gltf_mat, 1);
            for mesh in asset_meshes {
                scene
                    .meshes
                    .insert(scene.meshes.len() as u64, scene.meshes_storage.insert(mesh));
            }
        }
        println!("[gltf]: Scene has {} meshes", scene.meshes.len());
        scene
    }

    /// Recursively creates a vector of all the meshes in the scene from the nodes.
    /// Adds the meshes' transformation.
    fn flatten_meshes(
        &self,
        scene: &asset::Scene,
        node: gltf::Node,
        mut transform: glam::Mat4,
        layer: u32,
    ) -> Vec<asset::Mesh> {
        transform *= glam::Mat4::from_cols(
            glam::Vec4::from(node.transform().matrix()[0]),
            glam::Vec4::from(node.transform().matrix()[1]),
            glam::Vec4::from(node.transform().matrix()[2]),
            glam::Vec4::from(node.transform().matrix()[3]),
        );

        // Flatten all the meshes
        let mut meshes: Vec<asset::Mesh> = Vec::new();
        if node.mesh().is_some() {
            let gltf_mesh = node.mesh().unwrap();
            // Mesh exists in node
            meshes.append(&mut self.load_mesh(
                scene,
                gltf_mesh,
                // convert from gltf to vulkan coordinates
                transform
                    * glam::Mat4::from_cols_array(&[
                        1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
                        1.0,
                    ]),
            ));
        }
        println!("Found {} meshes on layer {}", meshes.len(), layer);
        for child_node in node.children() {
            meshes.append(&mut self.flatten_meshes(scene, child_node, transform, layer + 1));
        }
        meshes
    }

    /// Creates a [`Mesh`] object and binds it to the proper handles for its respective handles
    /// from a given [`gltf::Mesh`].
    ///
    /// [`Mesh`]: asset::Mesh
    /// [`gltf::Mesh`]: gltf::Mesh
    fn load_mesh(
        &self,
        scene: &asset::Scene,
        mesh: gltf::Mesh,
        transform: glam::Mat4,
    ) -> Vec<asset::Mesh> {
        let mut asset_meshes = Vec::new();
        for gltf_primitive in mesh.primitives() {
            let mut position_index: i32 = -1;
            let mut normal_index: i32 = -1;
            let mut index_index: i32 = gltf_primitive.indices().unwrap().index() as i32;
            for gltf_attribute in gltf_primitive.attributes() {
                match gltf_attribute.0 {
                    Semantic::Positions => {
                        position_index = gltf_attribute.1.index() as i32;
                    }
                    Semantic::Normals => {
                        normal_index = gltf_attribute.1.index() as i32;
                    }
                    _ => {}
                }
            }
            if position_index < 0 || normal_index < 0 {
                continue;
            }
            let vertex_buffer = scene.attributes.get(&(position_index as u64));
            let index_buffer = scene.attributes.get(&(index_index as u64));
            if vertex_buffer.is_some() && index_buffer.is_some() {
                asset_meshes.push(asset::Mesh {
                    vertex_buffer: *vertex_buffer.unwrap(),
                    index_buffer: *index_buffer.unwrap(),
                    transform,
                    name: Some(String::from(mesh.name().unwrap_or("Unnamed"))),
                });
            } else {
                println!(
                    "\"{}\" mesh does not have a valid index or vertex buffer",
                    mesh.name().unwrap_or("Unnamed")
                );
            }
        }
        asset_meshes
    }
}
