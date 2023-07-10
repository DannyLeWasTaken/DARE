//! Helper struct to load in gltf files and parse them into [`asset::Scene`] objects
//!
//! [`asset::Scene`]: crate::asset::Scene

use crate::app;
use crate::asset;
use crate::utils::handle_storage;
use crate::utils::handle_storage::Storage;
use crate::utils::memory;
use anyhow;
use anyhow::Result;
use bytemuck;
use gltf::accessor::DataType;
use gltf::Semantic;
use phobos::vk::Handle;
use phobos::{vk, IncompleteCmdBuffer, TransferCmdBuffer};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
            images: HashMap::new(),
            textures: HashMap::new(),
            attributes: HashMap::new(),

            meshes_storage: Storage::new(),
            buffer_storage: Storage::new(),
            attributes_storage: Storage::new(),
            image_storage: Storage::new(),
            texture_storage: Storage::new(),
        };
        let gltf = gltf::Gltf::open(gltf_path).unwrap();
        let (document, buffers, images) = gltf::import(gltf_path).unwrap();

        // Store buffers onto GPU memory rather than host visible memory
        {
            let temp_exec_manager = ctx.execution_manager.clone();
            let mut staging_buffers: Vec<(phobos::Buffer, Arc<phobos::Buffer>)> = Vec::new();
            let mut transfer_commands = temp_exec_manager
                .on_domain::<phobos::domain::Compute>()
                .unwrap();

            for gltf_buffer in document.buffers() {
                let buffer = buffers.get(gltf_buffer.index()).unwrap();
                let buffer_data = &*buffer.0;
                let staging_buffer = phobos::Buffer::new(
                    ctx.device.clone(),
                    &mut ctx.allocator,
                    gltf_buffer.length() as u64,
                    phobos::MemoryType::CpuToGpu,
                )
                .unwrap();
                staging_buffer
                    .view_full()
                    .mapped_slice::<u8>()
                    .unwrap() // Stored as bytes
                    .copy_from_slice(buffer_data);

                let gpu_buffer = phobos::Buffer::new(
                    ctx.device.clone(),
                    &mut ctx.allocator,
                    staging_buffer.size(),
                    phobos::MemoryType::GpuOnly,
                )
                .unwrap();

                // Debug naming
                ctx.device
                    .set_name(&gpu_buffer, gltf_buffer.name().unwrap_or("Unnamed"))
                    .expect("Unable to debug name");

                println!("[gltf]: New buffer!");
                println!("Buffer size: {}", staging_buffer.size());
                println!("Buffer data size in bytes: {}", buffer_data.len());

                // Store buffer in scene
                let buffer_handle = scene.buffer_storage.insert(gpu_buffer);
                scene
                    .buffers
                    .insert(gltf_buffer.index() as u64, buffer_handle.clone());
                staging_buffers.push((
                    staging_buffer,
                    scene.buffer_storage.get_immutable(&buffer_handle).unwrap(),
                ));
            }

            for (staging_buffer, gpu_buffer) in staging_buffers {
                transfer_commands = transfer_commands
                    .copy_buffer(&staging_buffer.view_full(), &gpu_buffer.view_full())
                    .unwrap();
            }

            ctx.execution_manager
                .submit(transfer_commands.finish().unwrap())
                .unwrap()
                .wait()
                .unwrap();
        }

        {
            let temp_exec_manager = ctx.execution_manager.clone();
            let mut image_commands = temp_exec_manager
                .on_domain::<phobos::domain::Compute>()
                .unwrap();
            let mut transfer_images: Vec<(phobos::Buffer, Arc<phobos::Image>)> = Vec::new();

            // Load images
            for gltf_image in document.images() {
                let image = images.get(gltf_image.index()).unwrap();
                let image_data = image.pixels.as_slice();
                let image_phobos = phobos::image::Image::new(
                    ctx.device.clone(),
                    &mut ctx.allocator,
                    image.width,
                    image.height,
                    vk::ImageUsageFlags::TRANSFER_DST,
                    get_image_type_rgba(image.format),
                    vk::SampleCountFlags::TYPE_1,
                )
                .unwrap();
                let image_data = convert_image_types_to_rgba(image.format, image_data);

                // Store the image on GPU memory
                let staging_buffer = memory::make_transfer_buffer(
                    ctx,
                    image_data.as_slice(),
                    None,
                    gltf_image.name().unwrap_or(
                        format!("Unnamed image buffer {:?}", image_phobos.format()).as_str(),
                    ),
                )
                .unwrap();
                ctx.device
                    .set_name(&image_phobos, gltf_image.name().unwrap_or("Unnamed Image"))
                    .expect("TODO: panic message");

                println!("[gltf]: New image!");
                println!(
                    "Image format: {:?} {:?}",
                    image_phobos.format(),
                    image.format
                );

                let image_phobos_handle = scene.image_storage.insert(image_phobos);
                scene
                    .images
                    .insert(gltf_image.index() as u64, image_phobos_handle.clone());

                transfer_images.push((
                    staging_buffer,
                    scene
                        .image_storage
                        .get_immutable(&image_phobos_handle)
                        .unwrap(),
                ))
            }

            for (staging_buffer, image) in transfer_images {
                image_commands = image_commands
                    .copy_buffer_to_image(
                        &staging_buffer.view_full(),
                        &image.view(vk::ImageAspectFlags::COLOR).unwrap(),
                    )
                    .unwrap();
            }

            ctx.execution_manager
                .submit(image_commands.finish().unwrap())
                .unwrap()
                .wait()
                .unwrap();
        }

        // Create samplers
        // NOTE: This does not deal with default samplers.
        {
            for gltf_sampler in document.samplers() {
                if gltf_sampler.index().is_none() {
                    continue;
                }
                let phobos_sampler = phobos::Sampler::new(
                    ctx.device.clone(),
                    vk::SamplerCreateInfo {
                        mag_filter: translate_filter(
                            gltf_sampler
                                .mag_filter()
                                .unwrap_or(gltf::texture::MagFilter::Linear),
                        ),
                        min_filter: translate_filter(
                            gltf_sampler
                                .mag_filter()
                                .unwrap_or(gltf::texture::MagFilter::Linear),
                        ),
                        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                        address_mode_u: translate_wrap_mode(gltf_sampler.wrap_s()),
                        address_mode_v: translate_wrap_mode(gltf_sampler.wrap_t()),
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
                .unwrap();
                scene.textures.insert(
                    gltf_sampler.index().unwrap() as u64,
                    scene.texture_storage.insert(phobos_sampler),
                );
            }
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
                        | gltf::Semantic::Tangents
                        | gltf::Semantic::TexCoords(_) => {
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
            let gltf_mat = glam::Mat4::from_cols_array_2d(&gltf_node.transform().matrix());
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
            meshes.append(&mut self.load_mesh(scene, gltf_mesh, transform));
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

use gltf::accessor::Dimensions;
use gltf::json::accessor::{ComponentType, Type};
use winit::window::CursorIcon::Default;

fn get_dimension_size(dimension: Dimensions) -> Option<u32> {
    match dimension {
        Type::Scalar => Some(1),
        Type::Vec2 => Some(2),
        Type::Vec3 => Some(3),
        Type::Vec4 => Some(4),
        _ => None,
    }
}

fn get_component_size(component: DataType) -> u32 {
    (match component {
        ComponentType::I8 => std::mem::size_of::<i8>(),
        ComponentType::U8 => std::mem::size_of::<u8>(),
        ComponentType::I16 => std::mem::size_of::<i16>(),
        ComponentType::U16 => std::mem::size_of::<u16>(),
        ComponentType::U32 => std::mem::size_of::<u32>(),
        ComponentType::F32 => std::mem::size_of::<f32>(),
    } as u32)
}

fn convert_slice_type<T: bytemuck::Pod, U: bytemuck::Pod, F: Fn(T) -> U>(
    input: &[u8],
    convert: F,
) -> Vec<u8> {
    let typed_input: &[T] = bytemuck::cast_slice(input);
    let output: Vec<U> = typed_input.into_iter().map(|&x| convert(x)).collect();
    bytemuck::cast_slice(&output).to_vec()
}

/// I hate this
fn convert_component_type(src_type: DataType, dst_type: DataType, data: &[u8]) -> Option<Vec<u8>> {
    if src_type == dst_type {
        return Some(data.to_vec());
    } else {
        match (src_type, dst_type) {
            (DataType::U8, DataType::I8) => {
                Some(convert_slice_type::<u8, i8, _>(data, |x| x as i8))
            }
            (DataType::U8, DataType::I16) => {
                Some(convert_slice_type::<u8, i16, _>(data, |x| x as i16))
            }
            (DataType::U8, DataType::U16) => {
                Some(convert_slice_type::<u8, u16, _>(data, |x| x as u16))
            }
            (DataType::U8, DataType::U32) => {
                Some(convert_slice_type::<u8, u32, _>(data, |x| x as u32))
            }
            (DataType::U8, DataType::F32) => {
                Some(convert_slice_type::<u8, f32, _>(data, |x| x as f32))
            }

            (DataType::I8, DataType::U8) => {
                Some(convert_slice_type::<i8, u8, _>(data, |x| x as u8))
            }
            (DataType::I8, DataType::I16) => {
                Some(convert_slice_type::<i8, i16, _>(data, |x| x as i16))
            }
            (DataType::I8, DataType::U16) => {
                Some(convert_slice_type::<i8, u16, _>(data, |x| x as u16))
            }
            (DataType::I8, DataType::U32) => {
                Some(convert_slice_type::<i8, u32, _>(data, |x| x as u32))
            }
            (DataType::I8, DataType::F32) => {
                Some(convert_slice_type::<i8, f32, _>(data, |x| x as f32))
            }

            (DataType::I16, DataType::I8) => {
                Some(convert_slice_type::<i16, i8, _>(data, |x| x as i8))
            }
            (DataType::I16, DataType::U8) => {
                Some(convert_slice_type::<i16, u8, _>(data, |x| x as u8))
            }
            (DataType::I16, DataType::U16) => {
                Some(convert_slice_type::<i16, u16, _>(data, |x| x as u16))
            }
            (DataType::I16, DataType::U32) => {
                Some(convert_slice_type::<i16, u32, _>(data, |x| x as u32))
            }
            (DataType::I16, DataType::F32) => {
                Some(convert_slice_type::<i16, f32, _>(data, |x| x as f32))
            }

            (DataType::U16, DataType::I8) => {
                Some(convert_slice_type::<u16, i8, _>(data, |x| x as i8))
            }
            (DataType::U16, DataType::U8) => {
                Some(convert_slice_type::<u16, u8, _>(data, |x| x as u8))
            }
            (DataType::U16, DataType::I16) => {
                Some(convert_slice_type::<u16, i16, _>(data, |x| x as i16))
            }
            (DataType::U16, DataType::U32) => {
                Some(convert_slice_type::<u16, u32, _>(data, |x| x as u32))
            }
            (DataType::U16, DataType::F32) => {
                Some(convert_slice_type::<u16, f32, _>(data, |x| x as f32))
            }

            (DataType::U32, DataType::I8) => {
                Some(convert_slice_type::<u32, i8, _>(data, |x| x as i8))
            }
            (DataType::U32, DataType::U8) => {
                Some(convert_slice_type::<u32, u8, _>(data, |x| x as u8))
            }
            (DataType::U32, DataType::I16) => {
                Some(convert_slice_type::<u32, i16, _>(data, |x| x as i16))
            }
            (DataType::U32, DataType::U16) => {
                Some(convert_slice_type::<u32, u16, _>(data, |x| x as u16))
            }
            (DataType::U32, DataType::F32) => {
                Some(convert_slice_type::<u32, f32, _>(data, |x| x as f32))
            }

            (DataType::F32, DataType::I8) => {
                Some(convert_slice_type::<f32, i8, _>(data, |x| x as i8))
            }
            (DataType::F32, DataType::U8) => {
                Some(convert_slice_type::<f32, u8, _>(data, |x| x as u8))
            }
            (DataType::F32, DataType::I16) => {
                Some(convert_slice_type::<f32, i16, _>(data, |x| x as i16))
            }
            (DataType::F32, DataType::U16) => {
                Some(convert_slice_type::<f32, u16, _>(data, |x| x as u16))
            }
            (DataType::F32, DataType::U32) => {
                Some(convert_slice_type::<f32, u32, _>(data, |x| x as u32))
            }

            _ => None,
        }
    }
}

fn get_zero_as_bytes(dst_type: DataType) -> &'static [u8] {
    match dst_type {
        ComponentType::I8 => bytemuck::bytes_of(&0i8),
        ComponentType::U8 => bytemuck::bytes_of(&0u8),
        ComponentType::I16 => bytemuck::bytes_of(&0i16),
        ComponentType::U16 => bytemuck::bytes_of(&0u16),
        ComponentType::U32 => bytemuck::bytes_of(&0u32),
        ComponentType::F32 => bytemuck::bytes_of(&0f32),
    }
}

/// Literally does that.
/// Converts the component type (u16, f32, f64, etc.) to the desired one
/// Then converts dimensions
fn convert_data_type(
    src_dimension: Dimensions,
    dst_type: DataType,
    src_type: DataType,
    dst_dimension: Dimensions,
    data: &[u8],
) -> Option<Vec<u8>> {
    let src_dimension = get_dimension_size(src_dimension);
    let dst_dimension = get_dimension_size(dst_dimension);
    let src_component_size = get_component_size(src_type);
    let dst_component_size = get_component_size(dst_type);
    if src_dimension.is_none() || dst_dimension.is_none() {
        return None;
    }
    let src_dimension = src_dimension.unwrap();
    let dst_dimension = dst_dimension.unwrap();
    Some(
        data.chunks((src_dimension * src_component_size) as usize)
            .flat_map(|data| {
                let mut out: Vec<u8> = Vec::new();
                // Iterate over each component
                for (dimension, ..) in (1..=src_dimension)
                    .take(dst_dimension as usize)
                    .zip(1..=dst_dimension)
                {
                    if dimension <= dst_dimension {
                        // Get component at current dimension
                        let section = data.get(
                            ((src_component_size * (src_dimension - 1)) as usize)
                                ..((src_component_size * src_dimension) as usize),
                        );
                        // welcome to hell my friend, it's not hot, but fucking boring
                        let section = match section {
                            None => get_zero_as_bytes(dst_type).to_vec(),
                            Some(section) => {
                                match convert_component_type(src_type, dst_type, section) {
                                    Some(section) => section,
                                    None => get_zero_as_bytes(dst_type).to_vec(),
                                }
                            }
                        };
                        out.extend_from_slice(section.as_slice());
                    } else {
                        // src_dimension is too big
                        // vec4 -> vec2 examples
                        break;
                    }
                }
                out.into_iter()
            })
            .collect::<Vec<u8>>(),
    )
}

/// Converts the image type from gltf to Vulkan format
fn get_image_type_rgba(in_format: gltf::image::Format) -> vk::Format {
    match in_format {
        gltf::image::Format::R8
        | gltf::image::Format::R8G8
        | gltf::image::Format::R8G8B8
        | gltf::image::Format::R8G8B8A8 => vk::Format::R8G8B8A8_SRGB,
        gltf::image::Format::R16
        | gltf::image::Format::R16G16
        | gltf::image::Format::R16G16B16
        | gltf::image::Format::R16G16B16A16 => vk::Format::R16G16B16A16_UINT,
        gltf::image::Format::R32G32B32FLOAT => vk::Format::R32G32B32A32_UINT,
        gltf::image::Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
    }
}

/// Converts all incoming image types to texels
fn convert_image_types_to_rgba(convert_from_format: gltf::image::Format, data: &[u8]) -> Vec<u8> {
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
    // Data is already in rgba
    if component_size == 4 {
        return data.to_vec();
    }

    // Split up the data into chunks, slice into it by the component size:
    // If component does not exist, default to zero expect for the alpha channel
    data.chunks(component_size * component_count)
        .flat_map(|p| {
            vec![
                p.get(0..component_size).unwrap_or(&[0]).to_vec(),
                p.get((component_size)..(2 * component_size))
                    .unwrap_or(&[0])
                    .to_vec(),
                p.get((component_size * 2)..(3 * component_size))
                    .unwrap_or(&[0])
                    .to_vec(),
                match component_size {
                    2 => (u16::MAX).to_be_bytes().to_vec(),
                    4 => (f32::MAX).to_be_bytes().to_vec(),
                    _ => vec![u8::MAX],
                },
            ]
            .into_iter()
            .flatten()
        })
        .collect::<Vec<u8>>()
}

fn translate_filter(filter: gltf::texture::MagFilter) -> vk::Filter {
    match filter {
        gltf::texture::MagFilter::Nearest => vk::Filter::NEAREST,
        gltf::texture::MagFilter::Linear => vk::Filter::LINEAR,
    }
}

fn translate_wrap_mode(wrap: gltf::texture::WrappingMode) -> vk::SamplerAddressMode {
    match wrap {
        gltf::texture::WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        gltf::texture::WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        gltf::texture::WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}
