//! Helper struct to load in gltf files and parse them into [`asset::Scene`] objects
//!
//! [`asset::Scene`]: crate::asset::Scene

use crate::app;
use crate::asset;
use crate::utils::handle_storage::{Handle, Storage};
use crate::utils::memory;
use anyhow;
use bytemuck;
use gltf::accessor::DataType;
use gltf::Semantic;
use phobos::{vk, IncompleteCmdBuffer, TransferCmdBuffer};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

pub struct GltfAssetLoader {}

// Create a type that includes indices in accessor type
#[derive(Debug, Clone)]
enum AccessorType {
    Gltf(Semantic),
    Index,
}

#[derive(Debug, Clone)]
struct Accessor<'a> {
    handle: gltf::Accessor<'a>,
    data_type: DataType,
    dimension: Dimensions,
}

/// Converts and given hashmap to a vector and returns the vector + a lookup hashmap of old indices
/// mapped to the new ones
fn compress_hashmap<T, U>(hash: HashMap<T, U>) -> (Vec<U>, HashMap<T, usize>)
where
    T: Eq + std::hash::Hash + Ord + Into<usize> + Copy,
    U: Clone,
{
    let mut lookup_table: HashMap<T, usize> = HashMap::new();
    let mut vector: Vec<U> = Vec::new();

    // Collect keys + sort
    let mut keys: Vec<T> = hash.keys().cloned().collect();
    keys.sort();

    for (new_index, old_index) in keys.into_iter().enumerate() {
        lookup_table.insert(old_index, new_index);
        vector.insert(new_index, hash[&old_index].clone());
    }
    (vector, lookup_table)
}

/// A structure to hold all lookup tables
#[derive(Debug)]
struct SceneLookup {
    meshes: HashMap<usize, usize>,
    buffers: HashMap<usize, usize>,
    images: HashMap<usize, usize>,
    samplers: HashMap<usize, usize>,
    attributes: HashMap<usize, usize>,
    textures: HashMap<usize, usize>,
    materials: HashMap<usize, usize>,
}

impl GltfAssetLoader {
    pub fn new() -> Self {
        Self {}
    }

    /// Loads any gltf asset given a file
    pub fn load_asset_from_file(&self, gltf_path: &Path, ctx: &mut app::Context) -> asset::Scene {
        let mut scene = asset::Scene {
            meshes: Vec::new(),
            buffers: Vec::new(),
            images: Vec::new(),
            samplers: Vec::new(),
            attributes: Vec::new(),
            textures: Vec::new(),
            materials: Vec::new(),

            meshes_storage: Storage::new(),
            buffer_storage: Storage::new(),
            attributes_storage: Storage::new(),
            image_storage: Storage::new(),
            sampler_storage: Storage::new(),
            texture_storage: Storage::new(),
            material_storage: Storage::new(),
        };
        let mut scene_lookup = SceneLookup {
            meshes: HashMap::new(),
            images: HashMap::new(),
            buffers: HashMap::new(),
            samplers: HashMap::new(),
            attributes: HashMap::new(),
            textures: HashMap::new(),
            materials: HashMap::new(),
        };
        let gltf = gltf::Gltf::open(gltf_path).unwrap();
        let (document, buffers, images) = gltf::import(gltf_path).unwrap();

        {
            let mut image_hashmap: HashMap<usize, Handle<phobos::Image>> = HashMap::new();
            let temp_exec_manager = ctx.execution_manager.clone();
            let mut image_commands = temp_exec_manager
                .on_domain::<phobos::domain::Graphics>()
                .unwrap();
            let mut transfer_images: Vec<(phobos::Buffer, Arc<phobos::Image>)> = Vec::new();

            // Load images
            for gltf_image in document.images() {
                let image = images.get(gltf_image.index()).unwrap();
                let image_data = image.pixels.as_slice();
                let mip_levels: u32 =
                    f32::floor(f32::log2(f32::max(image.width as f32, image.height as f32))) as u32
                        + 1u32;

                let image_phobos = phobos::image::Image::new(
                    ctx.device.clone(),
                    &mut ctx.allocator,
                    phobos::image::ImageCreateInfo {
                        width: image.width,
                        height: image.height,
                        depth: 1,
                        usage: vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                        format: get_image_type_rgba(image.format).unwrap(),
                        samples: vk::SampleCountFlags::TYPE_1,
                        mip_levels: 1,
                        layers: 1,
                    },
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

                let image_phobos_handle = scene.image_storage.insert(image_phobos);
                image_hashmap.insert(gltf_image.index(), image_phobos_handle.clone());

                transfer_images.push((
                    staging_buffer,
                    scene
                        .image_storage
                        .get_immutable(&image_phobos_handle)
                        .unwrap(),
                ));
            }

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

                let mut mip_width = image.width() as i32;
                let mut mip_height = image.height() as i32;
                /*
                for i in 1..image.mip_levels() {
                    image_barrier.subresource_range.base_mip_level = i - 1;
                    image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    image_barrier.src_access_mask = vk::AccessFlags2::TRANSFER_WRITE;
                    image_barrier.dst_access_mask = vk::AccessFlags2::TRANSFER_READ;
                    image_barrier.src_stage_mask = vk::PipelineStageFlags2::TRANSFER;
                    image_barrier.dst_stage_mask = vk::PipelineStageFlags2::TRANSFER;

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

                    let image_view = image.whole_view(vk::ImageAspectFlags::COLOR).unwrap();
                    let src_view = image
                        .view(phobos::image::ImageViewCreateInfo {
                            aspect: vk::ImageAspectFlags::COLOR,
                            view_type: vk::ImageViewType::TYPE_2D,
                            base_mip_level: i - 1,
                            level_count: Some(1),
                            base_layer: 0,
                            layers: Some(1),
                        })
                        .unwrap();
                    let dst_view = image
                        .view(phobos::image::ImageViewCreateInfo {
                            aspect: vk::ImageAspectFlags::COLOR,
                            view_type: vk::ImageViewType::TYPE_2D,
                            base_mip_level: i,
                            level_count: Some(1),
                            base_layer: 0,
                            layers: Some(1),
                        })
                        .unwrap();

                    image_barrier.subresource_range.base_mip_level = i - 1;
                    image_barrier.subresource_range.base_array_layer = 0;
                    image_barrier.subresource_range.layer_count = 1;

                    image_commands = image_commands.blit_image(
                        &src_view,
                        &dst_view,
                        &[
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: mip_width,
                                y: mip_height,
                                z: 1,
                            },
                        ],
                        &[
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: i32::max(mip_width / 2, 1),
                                y: i32::max(mip_height / 2, 1),
                                z: 1,
                            },
                        ],
                        vk::Filter::LINEAR,
                    );

                    image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    image_barrier.src_access_mask = vk::AccessFlags2::TRANSFER_WRITE;
                    image_barrier.dst_access_mask = vk::AccessFlags2::SHADER_READ;
                    image_barrier.src_stage_mask = vk::PipelineStageFlags2::TRANSFER;
                    image_barrier.dst_stage_mask = vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;

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
                    mip_width = i32::max(mip_width / 2, 1);
                    mip_height = i32::max(mip_height / 2, 1);
                }
                */
                /*
                image_barrier.subresource_range.base_mip_level = image.mip_levels() - 1;
                image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                image_barrier.src_access_mask = vk::AccessFlags2::TRANSFER_WRITE;
                image_barrier.dst_access_mask = vk::AccessFlags2::SHADER_READ;
                image_barrier.src_stage_mask = vk::PipelineStageFlags2::TRANSFER;
                image_barrier.dst_stage_mask = vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;

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
                */
            }

            ctx.execution_manager
                .submit(image_commands.finish().unwrap())
                .unwrap()
                .wait()
                .unwrap();
            let (images, lookup) = compress_hashmap(image_hashmap);
            scene.images = images;
            scene_lookup.images = lookup;
        }

        // Create samplers
        // NOTE: This does not deal with default samplers.
        {
            let mut sampler_hashmap: HashMap<usize, Handle<phobos::Sampler>> = HashMap::new();
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
                sampler_hashmap.insert(
                    gltf_sampler.index().unwrap(),
                    scene.sampler_storage.insert(phobos_sampler),
                );
            }

            // For now we will force one default sampler
            let (samplers, lookup) = compress_hashmap(sampler_hashmap);
            let lookup: HashMap<usize, usize> = lookup.keys().map(|&key| (key, 0)).collect();
            scene.samplers = vec![scene.sampler_storage.insert(
                phobos::Sampler::new(
                    ctx.device.clone(),
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
            )];
            scene_lookup.samplers = lookup;
        }
        {
            // Indicate every attribute that has been found in the meshes
            let mut attributes: Vec<(AccessorType, gltf::Accessor)> = Vec::new();
            let mut attribute_hashmap: HashMap<usize, Handle<asset::AttributeView>> =
                HashMap::new();
            {
                let mut indices: HashSet<u64> = HashSet::new();
                document
                    .meshes()
                    .flat_map(|x| x.primitives())
                    .filter_map(|gltf_primitive| {
                        let mut primitive_attributes: Vec<(AccessorType, gltf::Accessor)> =
                            Vec::with_capacity(gltf_primitive.attributes().len() + 1);
                        primitive_attributes
                            .push(((AccessorType::Index), gltf_primitive.indices()?));
                        for gltf_attribute in gltf_primitive.attributes() {
                            primitive_attributes
                                .push((AccessorType::Gltf(gltf_attribute.0), gltf_attribute.1));
                        }
                        primitive_attributes
                            .retain(|x| indices.get(&(x.1.index() as u64)).is_none());
                        for attribute in primitive_attributes.iter() {
                            indices.insert(attribute.1.index() as u64);
                        }
                        primitive_attributes.retain(|x| match x.0 {
                            AccessorType::Gltf(Semantic::Positions) => true,
                            AccessorType::Gltf(Semantic::Normals) => true,
                            AccessorType::Gltf(Semantic::TexCoords(_)) => true,
                            AccessorType::Index => true,
                            _ => false,
                        });
                        Some(primitive_attributes)
                    })
                    .for_each(|mut x| {
                        attributes.append(&mut x);
                    });
            }

            let mut monolithic_buffer_sizes: Vec<u64> = Vec::new();
            let mut monolithic_buffer: Vec<u8> = Vec::new();

            let attributes: Vec<(AccessorType, Accessor)> = attributes
                .into_iter()
                .map(|attribute| {
                    let gltf_buffer = attribute.1.view().unwrap().buffer().index();
                    let (mut compacted_buffer, data_type, dimension) =
                        GltfAssetLoader::compact_attribute_buffer(
                            &scene,
                            &buffers,
                            attribute.1.clone(),
                            attribute.0.clone(),
                        );
                    monolithic_buffer_sizes.push(compacted_buffer.len() as u64);
                    monolithic_buffer.append(&mut compacted_buffer);
                    (
                        attribute.0,
                        Accessor {
                            handle: attribute.1,
                            data_type,
                            dimension,
                        },
                    )
                })
                .collect();

            let staging_buffer = phobos::Buffer::new(
                ctx.device.clone(),
                &mut ctx.allocator,
                monolithic_buffer.len() as u64,
                phobos::MemoryType::CpuToGpu,
            )
            .unwrap();
            staging_buffer
                .view_full()
                .mapped_slice::<u8>()
                .unwrap()
                .copy_from_slice(&monolithic_buffer);

            let monolithic_buffer: phobos::Buffer = phobos::Buffer::new(
                ctx.device.clone(),
                &mut ctx.allocator,
                staging_buffer.size(),
                phobos::MemoryType::GpuOnly,
            )
            .unwrap();
            ctx.execution_manager
                .submit(
                    ctx.execution_manager
                        .on_domain::<Compute>()
                        .unwrap()
                        .copy_buffer(&staging_buffer.view_full(), &monolithic_buffer.view_full())
                        .unwrap()
                        .finish()
                        .unwrap(),
                )
                .unwrap()
                .wait()
                .unwrap();
            ctx.device
                .set_name(&monolithic_buffer, "Scene buffer")
                .expect("Failed to name buffer");
            let monolithic_buffer = scene.buffer_storage.insert(monolithic_buffer);
            scene.buffers.insert(0, monolithic_buffer);

            let mut total_monolithic_offset: u64 = 0;
            for (index, attribute) in attributes.iter().enumerate() {
                let accessor = &attribute.1;
                let accessor_viewer = &accessor.handle.view().unwrap();
                let total_offset = accessor_viewer.offset() + accessor.handle.offset();
                let stride = accessor_viewer.stride().unwrap_or(accessor.handle.size());
                let buffer_size = stride * accessor.handle.count();
                let buffer_view: phobos::BufferView = scene
                    .buffer_storage
                    .get_immutable(
                        scene
                            .buffers
                            .get(0usize) // One monolithic buffer
                            .unwrap(),
                    )
                    .unwrap()
                    .view(
                        total_monolithic_offset,
                        *monolithic_buffer_sizes.get(index).unwrap(),
                    )
                    .unwrap();

                // Output some cool debug information
                println!("Base accessor information:");
                println!("Name: {}", accessor.handle.name().unwrap_or("unnamed"));
                println!("Semantic: {:#?}", attribute.0,);
                println!("Buffer stride: {}", stride);
                println!("View size: {}", buffer_size);
                println!("Data type: {:?}", accessor.handle.data_type());
                println!("Dimension: {:?}", accessor.handle.dimensions());
                println!("Count: {}", accessor.handle.count());
                println!("\nNon-compacted:");
                println!(
                    "Buffer offset: {} + {} = {}",
                    accessor_viewer.offset(),
                    accessor.handle.offset(),
                    total_offset
                );
                println!("Buffer End Address: {}", total_offset + buffer_size);
                let stride = get_component_size(accessor.data_type)
                    * get_dimension_size(accessor.dimension).unwrap();
                println!("\nCompacted:");
                println!("View size: {}", buffer_view.size());
                println!("Buffer offset: {}", buffer_view.offset());
                println!(
                    "Buffer End Address: {}",
                    buffer_view.offset() + buffer_view.size(),
                );
                println!("Buffer stride: {}", stride);
                println!(
                    "Data type conversion: {:?} -> {:?}",
                    accessor.handle.data_type(),
                    accessor.data_type
                );
                println!(
                    "Dimension conversion: {:?} -> {:?}",
                    accessor.handle.dimensions(),
                    accessor.dimension
                );
                println!("\n\n");

                attribute_hashmap.insert(
                    accessor.handle.index(),
                    scene.attributes_storage.insert(asset::AttributeView {
                        buffer_view,
                        stride: stride as u64,
                        format: match (accessor.data_type, accessor.dimension) {
                            (DataType::F32, Dimensions::Vec4) => vk::Format::R32G32B32A32_SFLOAT,
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
                            (DataType::U8, Dimensions::Scalar) => vk::Format::R16_UINT,

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
                        count: accessor.handle.count() as u64,
                        component_size: accessor.handle.size() as u64,
                    }),
                );
                total_monolithic_offset += *monolithic_buffer_sizes.get(index).unwrap();
            }
            let (attributes, lookup) = compress_hashmap(attribute_hashmap);
            scene.attributes = attributes;
            scene_lookup.attributes = lookup;
        }
        {
            // Textures
            let mut texture_hashmap: HashMap<usize, Handle<asset::Texture>> = HashMap::new();
            for gltf_texture in document.textures() {
                let source = scene
                    .images
                    .get(
                        *scene_lookup
                            .images
                            .get(&gltf_texture.source().index())
                            .unwrap(),
                    )
                    .unwrap()
                    .clone();
                let sampler = scene
                    .samplers
                    .get(
                        *scene_lookup
                            .samplers
                            .get(&gltf_texture.sampler().index().unwrap_or(0))
                            .unwrap(),
                    )
                    .unwrap()
                    .clone();
                texture_hashmap.insert(
                    gltf_texture.index(),
                    scene.texture_storage.insert(asset::Texture {
                        name: Some(String::from(
                            gltf_texture.name().unwrap_or("Unnamed texture"),
                        )),
                        source,
                        sampler,
                    }),
                );
            }
            let (textures, lookup) = compress_hashmap(texture_hashmap);
            scene.textures = textures;
            scene_lookup.textures = lookup;
            println!("Scene has {} textures", scene.textures.len());
        }
        {
            // Materials
            let mut material_hashmap: HashMap<usize, Handle<asset::Material>> = HashMap::new();
            for gltf_material in document.materials() {
                if let Some(gltf_material_index) = gltf_material.index() {
                    let pbr_metallic = gltf_material.pbr_metallic_roughness();
                    let albedo: Option<Handle<asset::Texture>> =
                        pbr_metallic.base_color_texture().and_then(|x| {
                            scene_lookup
                                .textures
                                .get(&x.texture().index())
                                .and_then(|x| scene.textures.get(*x).cloned())
                        });
                    if albedo.is_none() {
                        println!("Something went very wrong");
                    }
                    material_hashmap.insert(
                        gltf_material_index,
                        scene.material_storage.insert(asset::Material {
                            albedo_texture: albedo,
                            albedo_color: glam::Vec3::new(0.0, 0.0, 0.0),
                        }),
                    );
                } else {
                    println!("This material expected the default");
                }
            }
            let (materials, lookup) = compress_hashmap(material_hashmap);
            scene.materials = materials;
            scene_lookup.materials = lookup;
            println!("[gltf]: Scene has {} materials", scene.materials.len());
        }

        {
            let mut mesh_hashmap: HashMap<usize, Handle<asset::Mesh>> = HashMap::new();
            // Select just one scene for now
            for gltf_node in document.scenes().next().unwrap().nodes() {
                let gltf_mat = glam::Mat4::from_cols_array_2d(&gltf_node.transform().matrix());
                let transform_matrix = glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0));
                let rotation_matrix = glam::Mat4::from_quat(glam::Quat::from_rotation_y(
                    -std::f32::consts::FRAC_PI_2,
                ));
                let transformed_matrix = transform_matrix * gltf_mat;
                let asset_meshes =
                    self.flatten_meshes(&scene, &scene_lookup, gltf_node, transformed_matrix, 1);
                for (index, mesh) in asset_meshes.into_iter().enumerate() {
                    mesh_hashmap.insert(index, scene.meshes_storage.insert(mesh));
                }
            }
            let (meshes, lookup) = compress_hashmap(mesh_hashmap);
            scene.meshes = meshes;
            scene_lookup.meshes = lookup;
        }
        println!("[gltf]: Scene has {} meshes", scene.meshes.len());
        println!("[gltf]: Lookup scene: {:?}", scene_lookup);
        scene
    }

    /// Compacts any given attribute's corresponding data
    fn compact_attribute_buffer(
        scene: &asset::Scene,
        buffers: &Vec<gltf::buffer::Data>,
        accessor: gltf::Accessor,
        semantic: AccessorType,
    ) -> (Vec<u8>, DataType, Dimensions) {
        // Create a buffer view
        let accessor_viewer = &accessor.view().unwrap();
        let total_offset = accessor_viewer.offset() + accessor.offset();
        let stride = accessor_viewer.stride().unwrap_or(accessor.size());
        let buffer_size = accessor_viewer.length();
        let buffer_size = stride * accessor.count();

        let dst_type = match accessor.data_type() {
            DataType::U32 | DataType::U16 | DataType::U8 => DataType::U32, // Horrible for perf, but allows shader to have a predictable index size
            _ => DataType::F32,
        };
        let dst_dimension = match accessor.dimensions() {
            Dimensions::Scalar => Dimensions::Scalar,
            Dimensions::Mat4 => Dimensions::Mat4,
            Dimensions::Mat3 => Dimensions::Mat3,
            Dimensions::Mat2 => Dimensions::Mat2,
            Dimensions::Vec4 => Dimensions::Vec4,
            Dimensions::Vec2 => Dimensions::Vec2,
            _ => Dimensions::Vec3,
        };

        // Compact the buffer & convert to a standard size
        let compact_buffer: Vec<u8> = convert_data_type(
            accessor.data_type(),
            accessor.dimensions(),
            dst_type,
            dst_dimension,
            view_buffer(
                buffers
                    .get(accessor_viewer.buffer().index())
                    .unwrap()
                    .0
                    .as_slice(),
                accessor.size(),
                total_offset,
                total_offset + buffer_size,
                stride,
            )
            .as_slice(),
        )
        .unwrap();

        match dst_type {
            ComponentType::U32 => {
                let float_buffer: Vec<u32> = bytemuck::cast_slice(&compact_buffer).to_vec();
                println!(
                    "{}: {:?}",
                    accessor.name().unwrap_or("Unnamed"),
                    float_buffer
                );
            }
            ComponentType::F32 => {
                let float_buffer: Vec<f32> = bytemuck::cast_slice(&compact_buffer).to_vec();
                println!(
                    "{}: {:?}",
                    accessor.name().unwrap_or("Unnamed"),
                    float_buffer
                );
            }
            _ => {}
        }

        (compact_buffer, dst_type, dst_dimension)
    }

    /// Recursively creates a vector of all the meshes in the scene from the nodes.
    /// Adds the meshes' transformation.
    fn flatten_meshes(
        &self,
        scene: &asset::Scene,
        scene_lookup: &SceneLookup,
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
            meshes.append(&mut self.load_mesh(scene, scene_lookup, gltf_mesh, transform));
        }
        println!("Found {} meshes on layer {}", meshes.len(), layer);
        for child_node in node.children() {
            meshes.append(&mut self.flatten_meshes(
                scene,
                scene_lookup,
                child_node,
                transform,
                layer + 1,
            ));
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
        scene_lookup: &SceneLookup,
        mesh: gltf::Mesh,
        transform: glam::Mat4,
    ) -> Vec<asset::Mesh> {
        let mut asset_meshes = Vec::new();
        for gltf_primitive in mesh.primitives() {
            let mut position_index: i32 = -1;
            let mut normal_index: i32 = -1;
            let mut tex_index: i32 = -1;
            let index_index: i32 = gltf_primitive.indices().unwrap().index() as i32;
            for gltf_attribute in gltf_primitive.attributes() {
                match gltf_attribute.0 {
                    Semantic::Positions => {
                        position_index = gltf_attribute.1.index() as i32;
                    }
                    Semantic::Normals => {
                        normal_index = gltf_attribute.1.index() as i32;
                    }
                    Semantic::TexCoords(_) => {
                        tex_index = gltf_attribute.1.index() as i32;
                    }
                    _ => {}
                }
            }
            if position_index < 0 || normal_index < 0 {
                println!(
                    "Could not add {} mesh due to having no normals or vertices",
                    mesh.name().unwrap_or("Unnamed")
                );
                continue;
            }
            let vertex_buffer = scene.attributes.get(
                *scene_lookup
                    .attributes
                    .get(&(position_index as usize))
                    .unwrap(),
            );
            let index_buffer = scene.attributes.get(
                *scene_lookup
                    .attributes
                    .get(&(index_index as usize))
                    .unwrap(),
            );
            let normal_buffer = scene.attributes.get(
                *scene_lookup
                    .attributes
                    .get(&(normal_index as usize))
                    .unwrap_or(&usize::MAX),
            );
            let tex_buffer = scene.attributes.get(
                *scene_lookup
                    .attributes
                    .get(&(tex_index as usize))
                    .unwrap_or(&usize::MAX),
            );
            if tex_buffer.is_none() {
                println!("Tex buffer is none!");
            }
            println!(
                "Material id: {}",
                scene_lookup
                    .materials
                    .get(&gltf_primitive.material().index().unwrap())
                    .unwrap()
            );
            if vertex_buffer.is_some() && index_buffer.is_some() {
                asset_meshes.push(asset::Mesh {
                    vertex_buffer: *vertex_buffer.unwrap(),
                    index_buffer: *index_buffer.unwrap(),
                    normal_buffer: normal_buffer.copied(),
                    tangent_buffer: None,
                    tex_buffer: tex_buffer.copied(),
                    transform,
                    name: Some(String::from(mesh.name().unwrap_or("Unnamed"))),
                    /*
                    material: *scene_lookup
                        .materials
                        .get(&gltf_primitive.material().index().unwrap())
                        .unwrap() as i32,
                     */
                    material: gltf_primitive.material().index().unwrap() as i32,
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

/// Gets the buffer information given an offset and stride
/// SIZE REFERS TO SIZE OF COMPONENTS. NOT SIZE OF BUFFER
fn view_buffer(
    buffer: &[u8],
    size: usize,
    byte_offset: usize,
    byte_end: usize,
    stride: usize,
) -> Vec<u8> {
    if byte_offset >= buffer.len() {
        return Vec::new();
    }

    println!(
        "[Buffer]: {} - {}, [Buffer size]: {}",
        byte_offset,
        byte_end,
        buffer.len(),
    );
    buffer
        .get(byte_offset..byte_end)
        .unwrap()
        .chunks(stride)
        .flat_map(|chunk| chunk.get(0..size).unwrap().to_vec())
        .collect::<Vec<u8>>()
}

use crate::graphics::acceleration_structure::create_blas_from_scene;
use gltf::accessor::Dimensions;
use gltf::buffer::Data;
use gltf::json::accessor::{ComponentType, Type};
use phobos::domain::{Compute, Graphics};

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
    src_type: DataType,
    src_dimension: Dimensions,
    dst_type: DataType,
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
    /*
    {
        match src_type {
            DataType::F32 => {
                let to_float: Vec<f32> = bytemuck::cast_slice(data).to_vec();
                println!("INPUT: {:?}", to_float);
            }
            DataType::I16 => {
                let to_float: Vec<i16> = bytemuck::cast_slice(data).to_vec();
                println!("INPUT: {:?}", to_float);
            }
            DataType::U16 => {
                let to_float: Vec<u16> = bytemuck::cast_slice(data).to_vec();
                println!("INPUT: {:?}", to_float);
            }
            DataType::U32 => {
                let to_float: Vec<u32> = bytemuck::cast_slice(data).to_vec();
                println!("INPUT: {:?}", to_float);
            }
            _ => {}
        }
    }
    */
    Some(
        data.chunks((src_dimension * src_component_size) as usize)
            .flat_map(|data| {
                let mut out: Vec<u8> = Vec::new();
                // Iterate over each component
                for dimension in 1..=dst_dimension {
                    if dimension <= dst_dimension {
                        // Get component at current dimension
                        let section = data.get(
                            ((src_component_size * (dimension - 1)) as usize)
                                ..((src_component_size * dimension) as usize),
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

    let max_component_value = match component_size {
        1 => u8::MAX as f32,
        2 => u16::MAX as f32,
        4 => f32::MAX,
        _ => 1.0, // Default value for unknown component size
    };

    data.chunks(component_size * component_count)
        .flat_map(|p| {
            vec![
                normalize_component(
                    p.get(0..component_size).unwrap_or(&[0]),
                    max_component_value,
                ),
                normalize_component(
                    p.get(component_size..(2 * component_size)).unwrap_or(&[0]),
                    max_component_value,
                ),
                normalize_component(
                    p.get((component_size * 2)..(3 * component_size))
                        .unwrap_or(&[0]),
                    max_component_value,
                ),
                vec![u8::MAX], // Alpha channel is set to maximum value
            ]
            .into_iter()
            .flatten()
        })
        .collect::<Vec<u8>>()
}

/// Normalize a single component value
fn normalize_component(component: &[u8], max_value: f32) -> Vec<u8> {
    let value = match component.len() {
        1 => component[0] as f32,
        2 => u16::from_be_bytes([component[0], component[1]]) as f32,
        4 => f32::from_be_bytes([component[0], component[1], component[2], component[3]]),
        _ => 0.0, // Default value for unknown component size
    };

    let normalized_value = (value / max_value * 255.0).round().clamp(0.0, 255.0) as u8;

    vec![normalized_value]
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
