use crate::app;
use crate::asset;
use crate::utils::handle_storage;
use anyhow;
use anyhow::Result;
use phobos::vk;
use phobos::vk::Handle;
use std::path::Path;

pub struct GltfAssetLoader {}

impl GltfAssetLoader {
    pub fn new() -> Self {
        Self {}
    }

    /// Loads any gltf asset given a file
    pub fn load_asset_from_file(&self, gltf_path: &Path, ctx: &mut app::Context) -> asset::Scene {
        let mut scene = asset::Scene {
            meshes: Vec::new(),
            buffers: Vec::new(),
            attributes: Vec::new(),
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
                vk::BufferUsageFlags::TRANSFER_DST,
                phobos::MemoryType::CpuToGpu,
            )
            .unwrap();
            gpu_buffer
                .view_full()
                .mapped_slice::<u8>()
                .unwrap() // Stored as bytes
                .copy_from_slice(buffer_data);

            // Debug naming
            let buffer_name =
                std::ffi::CString::new(gltf_buffer.name().unwrap_or("unnamed")).unwrap();
            let name_info = phobos::prelude::vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::BUFFER)
                .object_handle(unsafe { gpu_buffer.handle().as_raw() })
                .object_name(&buffer_name)
                .build();

            unsafe {
                ctx.debug_utils
                    .set_debug_utils_object_name(ctx.device.handle().handle(), &name_info)
                    .expect("Failed to set object name!");
            };

            // Store buffer in scene
            let buffer_handle = scene.buffer_storage.insert(gpu_buffer);
            scene.buffers.insert(gltf_buffer.index(), buffer_handle);
        }

        // Load accessors
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
                        let stride = accessor_viewer.stride().unwrap_or(0);
                        let component_size = accessor.size();

                        // https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_005_BuffersBufferViewsAccessors.md#buffers
                        let buffer_size = if stride > 0 {
                            stride * (accessor.count() - 1) + component_size
                        } else {
                            component_size * accessor.count()
                        };

                        // Get buffer which is referenced and store attribute in scene
                        let buffer = scene
                            .buffer_storage
                            .get_immutable(
                                scene.buffers.get(accessor_viewer.buffer().index()).unwrap(),
                            )
                            .unwrap();

                        println!("Name: {}", accessor.name().unwrap_or("unnamed"));
                        println!(
                            "Semantic: {:#?}",
                            semantic.unwrap_or(gltf::Semantic::Weights(0))
                        );
                        println!("Buffer Address: {}", buffer.address());
                        println!("Buffer offset: {}", total_offset);
                        println!("Buffer stride: {}", stride);
                        println!("View size: {}", buffer_size);
                        println!("Component size: {}", component_size);
                        println!("Data type: {:?}", accessor.data_type());
                        println!("Dimension: {:?}", accessor.dimensions());
                        println!("\n\n");

                        scene.attributes_storage.insert(asset::AttributeView {
                            buffer_view: buffer
                                .view(total_offset as u64, buffer_size as u64)
                                .unwrap(),
                            stride,
                        })
                    };

                // Indices
                {
                    let accessor = gltf_primitive.indices().unwrap();
                    if scene.attributes.get(accessor.index()).is_none() {
                        scene
                            .attributes
                            .insert(accessor.index(), get_attribute_handle(accessor, None))
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
                            if scene.attributes.get(accessor.index()).is_some() {
                                continue;
                            }

                            // Create an attribute handle
                            scene.attributes.insert(
                                accessor.index(),
                                get_attribute_handle(accessor, Some(gltf_attribute.0)),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        scene
    }

    fn load_mesh(&self, mesh: gltf::Mesh) {}
}
