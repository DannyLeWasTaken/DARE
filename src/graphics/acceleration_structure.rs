//! Acceleration structure handler
//! Large parts thanks to: https://github.com/NotAPenguin0/Andromeda/blob/master/include/andromeda/graphics/backend/rtx.hpp
#![warn(missing_docs)]

use crate::utils::handle_storage::Handle;
use crate::utils::memory;
use crate::utils::types;
use crate::{asset, utils};

use anyhow::Result;

use phobos::domain::{All, Compute};
use phobos::{vk, ComputeCmdBuffer, IncompleteCmdBuffer};

pub struct AllocatedAS {
    buffer: phobos::BufferView,
    scratch: phobos::BufferView,
    handle: phobos::AccelerationStructure,
}

pub struct AccelerationStructureResources {
    buffer: phobos::Buffer,
    scratch: Option<phobos::Buffer>,
}

pub struct AccelerationStructure {
    resources: AccelerationStructureResources,
    instances: Vec<AllocatedAS>,
}

pub struct SceneAccelerationStructure {
    tlas: AccelerationStructure,
    blas: AccelerationStructure,
    instances: phobos::Buffer,
}

struct AccelerationStructureBuildInfo<'a> {
    // phobos::AccelerationStructureBuildInfo covers:
    // AccelerationStructureBuildGeometryInfoKHR
    // AccelerationStructureBuildRangeInfoKHR
    handle: phobos::AccelerationStructureBuildInfo<'a>,

    // Size information of the BLAS
    size_info: Option<phobos::AccelerationStructureBuildSize>,

    /// Offset of this BLAS entry in the main blas buffer
    buffer_offset: u64,

    /// Offset of this BLAS entry in the scratch buffer
    scratch_offset: u64,
}

/// Gets the build information of the BLASes based off of the scene
fn get_blas_entries<'a>(
    meshes: &Vec<Handle<asset::Mesh>>,
    scene: &asset::Scene,
) -> Vec<Result<phobos::AccelerationStructureBuildInfo<'a>>> {
    let mut build_infos: Vec<Result<phobos::AccelerationStructureBuildInfo<'a>>> =
        Vec::with_capacity(meshes.len());

    for mesh_handle in meshes {
        let mesh = scene.meshes_storage.get_immutable(mesh_handle);
        if mesh.is_none() {
            build_infos.push(Err(anyhow::anyhow!("No mesh found")));
            continue;
        }
        // Retrieve the mesh's vertex and index buffer
        let mesh = mesh.unwrap();
        let vertex_buffer = scene.attributes_storage.get_immutable(&mesh.vertex_buffer);
        let index_buffer = scene.attributes_storage.get_immutable(&mesh.index_buffer);
        if vertex_buffer.is_none() || index_buffer.is_none() {
            build_infos.push(Err(anyhow::anyhow!("No vertex or index buffer found")));
            continue;
        }
        let vertex_buffer = vertex_buffer.unwrap();
        let index_buffer = index_buffer.unwrap();
        let index_type = types::convert_scalar_format_to_index(index_buffer.format);
        if index_type.is_none() {
            build_infos.push(Err(anyhow::anyhow!("No index type found")));
            continue;
        }

        // Create the build information for the mesh
        let index_type = index_type.unwrap();
        build_infos.push(Ok(
            phobos::AccelerationStructureBuildInfo::new_build()
                .flags(
                    vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
                        | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                )
                .set_type(phobos::AccelerationStructureType::BottomLevel)
                .push_triangles(
                    phobos::AccelerationStructureGeometryTrianglesData::default()
                        .format(vertex_buffer.format)
                        .vertex_data(vertex_buffer.buffer_view.address())
                        .stride(vertex_buffer.component_size + vertex_buffer.stride)
                        .max_vertex(vertex_buffer.count as u32)
                        .index_data(index_type, index_buffer.buffer_view.address())
                        .flags(
                            vk::GeometryFlagsKHR::OPAQUE
                                | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
                        ),
                )
                .push_range((index_buffer.count / 3) as u32, 0, 0, 0),
            // first_vertex in push_range could be a concern thanks to first_index
        ));
    }
    build_infos
}

/// Gets primarily the size information about the BLAS entry
fn get_blas_build_infos<'a>(
    ctx: &mut crate::app::Context,
    mut entries: Vec<AccelerationStructureBuildInfo<'a>>,
) -> Vec<AccelerationStructureBuildInfo<'a>> {
    let mut buffer_offset: u64 = 0;
    let mut scratch_offset: u64 = 0;

    for mut entry in entries.iter_mut() {
        match phobos::query_build_size(
            &ctx.device,
            phobos::AccelerationStructureBuildType::Device,
            &entry.handle,
            std::slice::from_ref(&(entry.handle.as_vulkan().1.get(0).unwrap().primitive_count)),
        ) {
            Err(e) => {
                println!("Failed to get size of mesh due to: {}", e);
            }
            Ok(mut build_size) => {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureCreateInfoKHR.html/
                build_size.size =
                    memory::align_size(build_size.size, phobos::AccelerationStructure::alignment());
                build_size.build_scratch_size = memory::align_size(
                    build_size.build_scratch_size,
                    ctx.device
                        .acceleration_structure_properties()
                        .unwrap()
                        .min_acceleration_structure_scratch_offset_alignment
                        as u64,
                );
                entry.buffer_offset = buffer_offset;
                entry.scratch_offset = scratch_offset;

                buffer_offset += build_size.size;
                scratch_offset += build_size.build_scratch_size;

                entry.size_info = Some(build_size);
            }
        }
    }
    entries
}

/// Get total amount of scratch memory needed for the BLAS
fn total_scratch_memory(build_infos: &Vec<AccelerationStructureBuildInfo>) -> u64 {
    let mut size: u64 = 0;
    for build_info in build_infos {
        size += build_info.size_info.unwrap().build_scratch_size;
    }
    size
}

/// Get total amount of memory
fn total_blas_memory(build_infos: &Vec<AccelerationStructureBuildInfo>) -> u64 {
    let mut size: u64 = 0;
    for build_info in build_infos {
        size += build_info.size_info.unwrap().size;
    }
    size
}

/// Create the acceleration structure itself alongside the buffers
fn create_acceleration_structure(
    ctx: &mut crate::app::Context,
    build_infos: &Vec<AccelerationStructureBuildInfo<'_>>,
) -> (AccelerationStructureResources, Vec<AllocatedAS>) {
    // Create memory for all the BLASes
    let mut entries: Vec<AllocatedAS> = Vec::new();
    let buffer_size = total_blas_memory(build_infos);
    let scratch_size = total_scratch_memory(build_infos);
    let as_buffer = phobos::Buffer::new_device_local(
        ctx.device.clone(),
        &mut ctx.allocator,
        buffer_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
    )
    .unwrap();
    let scratch_buffer = phobos::Buffer::new_device_local(
        ctx.device.clone(),
        &mut ctx.allocator,
        scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )
    .unwrap();

    ctx.device.set_name(&as_buffer, "AS memory").unwrap();
    ctx.device
        .set_name(&scratch_buffer, "Scratch memory")
        .unwrap();
    // Allocate the scratch buffer for building the acceleration structure
    entries.reserve(build_infos.len());

    // Iterate of each entry
    for build_info in build_infos.iter() {
        // Suballocate the buffers
        let buffer_view =
            as_buffer.view(build_info.buffer_offset, build_info.size_info.unwrap().size);
        let scratch_view = scratch_buffer.view(
            build_info.scratch_offset,
            build_info.size_info.unwrap().build_scratch_size,
        );
        if buffer_view.is_err() || scratch_view.is_err() {
            //entries.push(Err(buffer_view.err().unwrap()));
            continue;
        }
        let buffer_view = buffer_view.unwrap();
        let scratch_view = scratch_view.unwrap();

        let acceleration_structure = phobos::AccelerationStructure::new(
            ctx.device.clone(),
            build_info.handle.ty(),
            buffer_view,
            vk::AccelerationStructureCreateFlagsKHR::default(),
        );
        if acceleration_structure.is_err() {
            //entries.push(Err(acceleration_structure.err().unwrap()));
            continue;
        }
        let acceleration_structure = acceleration_structure.unwrap();
        ctx.device
            .set_name(&acceleration_structure, "Acceleration structure")
            .unwrap();
        let entry = AllocatedAS {
            buffer: buffer_view,
            scratch: scratch_view,
            handle: acceleration_structure,
        };
        entries.push(entry);
    }

    (
        AccelerationStructureResources {
            buffer: as_buffer,
            scratch: Some(scratch_buffer),
        },
        entries,
    )
}

/// Builds all the BLAS structures in the given BLAS
fn build_blas(
    ctx: &mut crate::app::Context,
    entries: &Vec<AllocatedAS>,
    build_infos: Vec<AccelerationStructureBuildInfo>,
) -> Result<Vec<u64>> {
    let mut build_infos: Vec<phobos::AccelerationStructureBuildInfo> =
        build_infos.into_iter().map(|x| x.handle).collect();

    // Indicate the instance address + scratch
    build_infos = entries
        .iter()
        .zip(build_infos.into_iter())
        .map(|(allocated_as, mut build_info)| {
            build_info
                .dst(&allocated_as.handle)
                .scratch_data(allocated_as.scratch.address())
        })
        .collect::<Vec<phobos::AccelerationStructureBuildInfo>>();

    // Create a query pool
    let mut query_pool = phobos::QueryPool::<phobos::AccelerationStructureCompactedSizeQuery>::new(
        ctx.device.clone(),
        phobos::QueryPoolCreateInfo {
            count: build_infos.len() as u32,
            statistic_flags: None,
        },
    )?;
    let mut fences: Vec<phobos::pool::Pooled<phobos::Fence>> =
        Vec::with_capacity(build_infos.len());

    for (entry, build_info) in entries.iter().zip(build_infos.iter()) {
        let command = ctx
            .execution_manager
            .on_domain::<Compute>()?
            .build_acceleration_structure(build_info)?
            .memory_barrier(
                phobos::PipelineStage::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                phobos::PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            )
            .write_acceleration_structure_properties(&entry.handle, &mut query_pool)?
            .finish()?;
        let fence = ctx.execution_manager.submit(command).unwrap();
        fences.push(fence);
    }

    // Wait for all the fences to finish up
    for mut fence in fences {
        fence.wait()?;
    }
    query_pool.wait_for_all_results()
}

/// Create the final compacted acceleration structures
fn compact_blases(
    ctx: &mut crate::app::Context,
    as_resource: &AccelerationStructureResources,
    entries: &Vec<AllocatedAS>,
    mut compacted_sizes: Vec<u64>,
) -> (Vec<AllocatedAS>, AccelerationStructureResources) {
    let mut new_entries: Vec<AllocatedAS> = Vec::with_capacity(entries.len());

    compacted_sizes = compacted_sizes
        .iter()
        .map(|&x| memory::align_size(x, phobos::AccelerationStructure::alignment()))
        .collect::<Vec<u64>>();
    let total_compacted_size: u64 = compacted_sizes.iter().sum();

    let compacted_buffer = phobos::Buffer::new_device_local(
        ctx.device.clone(),
        &mut ctx.allocator,
        total_compacted_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
    )
    .unwrap();
    ctx.device
        .set_name(&as_resource.buffer, "BLAS Buffer")
        .unwrap();

    // Update the allocatedAS alongside
    let mut as_buffer_offset: u64 = 0;
    for (index, entry) in entries.iter().enumerate() {
        let buffer_view: phobos::BufferView = compacted_buffer
            .view(as_buffer_offset, *compacted_sizes.get(index).unwrap())
            .unwrap();
        println!(
            "Size before compactation: {}. Size after: {}",
            entry.buffer.size(),
            buffer_view.size()
        );
        let ty = entry.handle.ty();
        println!("YOOO our type is: {:?}", ty);
        let new_as = phobos::AccelerationStructure::new(
            ctx.device.clone(),
            entry.handle.ty(),
            buffer_view,
            vk::AccelerationStructureCreateFlagsKHR::default(),
        );
        if new_as.is_err() {
            println!("We fucked up so badly");
        }
        let new_as = new_as.unwrap();
        as_buffer_offset += buffer_view.size();

        let cmd = ctx
            .execution_manager
            .on_domain::<Compute>()
            .unwrap()
            .compact_acceleration_structure(&entry.handle, &new_as)
            .unwrap()
            .memory_barrier(
                phobos::PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
                phobos::PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ,
            )
            .finish()
            .unwrap();
        ctx.execution_manager.submit(cmd).unwrap().wait().unwrap();
        new_entries.push(AllocatedAS {
            buffer: buffer_view,
            scratch: entry.scratch.clone(),
            handle: new_as,
        });
    }
    (
        new_entries,
        AccelerationStructureResources {
            buffer: compacted_buffer,
            scratch: None,
        },
    )
}

/// Creates the instance buffer from all the created entries
pub fn make_instances_buffer(
    ctx: &mut crate::app::Context,
    entries: &Vec<AllocatedAS>,
) -> Result<phobos::Buffer> {
    let mut instances: Vec<phobos::AccelerationStructureInstance> = Vec::new();
    for entry in entries {
        instances.push(
            phobos::AccelerationStructureInstance::default()
                .mask(0xFF)
                .flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE)
                .sbt_record_offset(0)
                .unwrap()
                .custom_index(0)
                .unwrap()
                .transform(phobos::TransformMatrix::identity())
                .acceleration_structure(
                    &entry.handle,
                    phobos::AccelerationStructureBuildType::Device,
                )
                .unwrap(),
        );
    }

    memory::make_transfer_buffer(
        ctx,
        instances.as_slice(),
        Default::default(),
        Some(16),
        "Instance Buffer",
    )
}

/// Gets the build sizes with alignment included for the size and returns all the build sizes
fn get_acceleration_structures_build_sizes(
    ctx: &mut crate::app::Context,
    build_infos: &Vec<AccelerationStructureBuildInfo>,
    prim_counts: &[u32],
) -> Vec<Result<phobos::AccelerationStructureBuildSize>> {
    let mut sizes: Vec<Result<phobos::AccelerationStructureBuildSize>> = Vec::new();
    for (index, build_info) in build_infos.iter().enumerate() {
        sizes.push(phobos::query_build_size(
            &ctx.device,
            phobos::AccelerationStructureBuildType::Device,
            &build_info.handle,
            &prim_counts[index..index + 1],
        ));
    }
    for size_info in sizes.iter_mut().flatten() {
        size_info.build_scratch_size = memory::align_size(
            size_info.build_scratch_size,
            ctx.device
                .acceleration_structure_properties()
                .unwrap()
                .min_acceleration_structure_scratch_offset_alignment as u64,
        );
        size_info.size =
            memory::align_size(size_info.size, phobos::AccelerationStructure::alignment());
    }
    sizes
}

/// Get the build sizes for each build_infos and bind it
fn set_acceleration_structure_build_sizes<'a>(
    ctx: &mut crate::app::Context,
    mut build_infos: Vec<AccelerationStructureBuildInfo>,
    prim_counts: &[u32],
) -> Vec<Result<AccelerationStructureBuildInfo<'a>>> {
    let sizes = get_acceleration_structures_build_sizes(ctx, &build_infos, prim_counts);
    let mut out_build_infos: Vec<Result<AccelerationStructureBuildInfo>> = Vec::new();
    for (mut build_info, size) in build_infos.into_iter().zip(sizes.into_iter()) {
        match size {
            Err(E) => out_build_infos.push(Err(E)),
            Ok(size) => {
                build_info.size_info = Some(size);
                out_build_infos.push(Ok(build_info));
            }
        }
    }
    out_build_infos
}

/// Suballocates all given build_infos by setting their buffer_offset and scratch_offset. If you pass in build infos without a
/// size_info, it will leave the AS blank with an error
fn suballocate_acceleration_structures<'a>(
    ctx: &mut crate::app::Context,
    mut build_infos: Vec<AccelerationStructureBuildInfo>,
) -> Vec<Result<AccelerationStructureBuildInfo<'a>>> {
    let mut out_build_infos: Vec<Result<AccelerationStructureBuildInfo>> = Vec::new();
    let mut buffer_offset: u64 = 0;
    let mut scratch_offset: u64 = 0;
    for mut build_info in build_infos.into_iter() {
        match build_info.size_info {
            None => out_build_infos.push(Err(anyhow::anyhow!("size_info expected, found None"))),
            Some(size_info) => {
                build_info.buffer_offset = buffer_offset;
                build_info.scratch_offset = scratch_offset;

                buffer_offset += size_info.size;
                scratch_offset += size_info.build_scratch_size;

                out_build_infos.push(Ok(build_info));
            }
        }
    }
    out_build_infos
}

// Gets all the build information for the related TLAS
fn get_tlas_build_infos<'a>(
    instances: &phobos::Buffer,
    instance_count: u32,
) -> AccelerationStructureBuildInfo<'a> {
    AccelerationStructureBuildInfo {
        handle: phobos::AccelerationStructureBuildInfo::new_build()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .set_type(phobos::AccelerationStructureType::TopLevel)
            .push_instances(phobos::AccelerationStructureGeometryInstancesData {
                data: instances.address().into(),
                flags: vk::GeometryFlagsKHR::OPAQUE
                    | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
            })
            .push_range(instance_count, 0, 0, 0),
        size_info: None,
        buffer_offset: 0,
        scratch_offset: 0,
    }
}

pub fn convert_scene_to_blas(
    ctx: &mut crate::app::Context,
    scene: &asset::Scene,
) -> SceneAccelerationStructure {
    println!("Scene has total # of meshes: {}", scene.meshes.len());
    let mut blas_build_infos: Vec<AccelerationStructureBuildInfo> =
        get_blas_entries(&scene.meshes, scene)
            .into_iter()
            .filter_map(|x| {
                if x.is_err() {
                    println!(
                        "A mesh failed to load properly due to error {}!",
                        x.err().unwrap()
                    );
                    None
                } else {
                    Some(AccelerationStructureBuildInfo {
                        handle: x.unwrap(),
                        size_info: None,
                        buffer_offset: 0,
                        scratch_offset: 0,
                    })
                }
            })
            .collect();

    blas_build_infos = get_blas_build_infos(ctx, blas_build_infos);
    let (as_resources, entries) = create_acceleration_structure(ctx, &blas_build_infos);
    let compact_sizes = build_blas(ctx, &entries, blas_build_infos).expect("TODO: panic message");
    let (new_entries, new_as_resources) =
        compact_blases(ctx, &as_resources, &entries, compact_sizes);
    // Make TLAS

    //  TODO: REDO THIS UNHOLY MESS
    //  TODO: LEARN HOW TO CODE

    let instance_buffer = make_instances_buffer(ctx, &new_entries).unwrap();
    let mut tlas_build_info = get_tlas_build_infos(&instance_buffer, new_entries.len() as u32);
    let build_size = get_tlas_build_size(ctx, &tlas_build_info);
    tlas_build_info.size_info = Some(build_size);
    let mut tlas_build_infos: Vec<AccelerationStructureBuildInfo> = vec![tlas_build_info];

    let (tlas_resources, tlas_entries) = create_acceleration_structure(ctx, &tlas_build_infos);
    let mut tlas_build_info = tlas_build_infos.remove(0);
    tlas_build_info.handle = tlas_build_info
        .handle
        .dst(&tlas_entries.first().unwrap().handle)
        .scratch_data(tlas_resources.scratch.as_ref().unwrap().address());

    println!("You should be tlas: {:?}", tlas_build_info.handle.ty());
    let cmd = ctx
        .execution_manager
        .on_domain::<Compute>()
        .unwrap()
        .memory_barrier(
            phobos::PipelineStage::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            phobos::PipelineStage::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ,
        )
        .build_acceleration_structure(&tlas_build_info.handle)
        .unwrap()
        .finish()
        .unwrap();
    ctx.execution_manager.submit(cmd).unwrap().wait().unwrap();

    SceneAccelerationStructure {
        tlas: AccelerationStructure {
            resources: tlas_resources,
            instances: tlas_entries,
        },
        blas: AccelerationStructure {
            resources: new_as_resources,
            instances: new_entries,
        },
        instances: instance_buffer,
    }
}

// TODO: FOR THE SEGFAULTING MEMORY ERROR LOOK INTO
// - KEEPING THE LIFETIME OF THE OLD BUFFERS ALIVE
// - KEEPING THE LIFETIME OF THE OLD AS ALIVE
