//! Acceleration structure handler
//! Large parts thanks to: https://github.com/NotAPenguin0/Andromeda/blob/master/include/andromeda/graphics/backend/rtx.hpp
#![warn(missing_docs)]

use crate::asset;
use crate::utils::handle_storage::Handle;
use crate::utils::memory;
use crate::utils::types;

use anyhow::Result;

use phobos::domain::Compute;
use phobos::{vk, ComputeCmdBuffer, IncompleteCmdBuffer};

pub struct AllocatedAS {
    buffer: phobos::BufferView,
    scratch: phobos::BufferView,
    handle: usize,
}

pub struct AccelerationStructureResources {
    buffer: Option<phobos::Buffer>,
    scratch: Option<phobos::Buffer>,
    acceleration_structures: Vec<phobos::AccelerationStructure>,
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

    let mut total_primitive_count = 0;

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
        // Check if the number of indices can make a triangle
        assert_eq!((index_buffer.count % 3), 0);
        assert!(
            vertex_buffer.stride > 0,
            "[BLAS]: Given vertex buffer has a stride of zero"
        );
        total_primitive_count += index_buffer.count / 3;
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
                        .stride(vertex_buffer.stride)
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
    let mut entries: Vec<AllocatedAS> = Vec::with_capacity(build_infos.len());
    let mut instances: Vec<phobos::AccelerationStructure> = Vec::with_capacity(build_infos.len());
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
        instances.push(acceleration_structure);
        let entry = AllocatedAS {
            buffer: buffer_view,
            scratch: scratch_view,
            handle: instances.len() - 1,
        };
        entries.push(entry);
    }

    (
        AccelerationStructureResources {
            buffer: Some(as_buffer),
            scratch: Some(scratch_buffer),
            acceleration_structures: instances,
        },
        entries,
    )
}

/// Builds all the BLAS structures in the given BLAS
fn build_blas(
    ctx: &mut crate::app::Context,
    acceleration_structures_resource: &AccelerationStructureResources,
    entries: &[AllocatedAS],
    build_infos: Vec<AccelerationStructureBuildInfo>,
) -> Result<Vec<u64>> {
    // Make sure we have the right amounts
    assert_eq!(build_infos.len(), entries.len());
    assert_eq!(
        entries.len(),
        acceleration_structures_resource
            .acceleration_structures
            .len()
    );

    let build_infos: Vec<phobos::AccelerationStructureBuildInfo> =
        build_infos.into_iter().map(|x| x.handle).collect();
    assert_eq!(
        build_infos.len(),
        acceleration_structures_resource
            .acceleration_structures
            .len()
    );
    assert_eq!(build_infos.len(), entries.len());
    let build_infos = entries
        .iter()
        .zip(
            acceleration_structures_resource
                .acceleration_structures
                .iter(),
        )
        .zip(build_infos.into_iter())
        .map(|((allocated_as, acceleration_structure), build_info)| {
            build_info
                .dst(acceleration_structure)
                .scratch_data(allocated_as.scratch.address())
        })
        .collect::<Vec<phobos::AccelerationStructureBuildInfo>>();

    let mut query_pool = phobos::QueryPool::<phobos::AccelerationStructureCompactedSizeQuery>::new(
        ctx.device.clone(),
        phobos::QueryPoolCreateInfo {
            count: build_infos.len() as u32,
            statistic_flags: None,
        },
    )?;

    let mut command = ctx
        .execution_manager
        .on_domain::<Compute>()?
        .build_acceleration_structures(build_infos.as_slice())?
        .memory_barrier(
            phobos::PipelineStage::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            phobos::PipelineStage::ALL_COMMANDS,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        )
        .write_acceleration_structures_properties(
            acceleration_structures_resource
                .acceleration_structures
                .as_slice(),
            &mut query_pool,
        )?
        .finish()?;

    ctx.execution_manager.submit(command)?.wait()?;

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
    let mut new_structures: Vec<phobos::AccelerationStructure> = Vec::with_capacity(entries.len());
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
        .set_name(&compacted_buffer, "BLAS Buffer")
        .unwrap();

    // Update the allocatedAS alongside
    let mut as_buffer_offset: u64 = 0;
    assert_eq!(entries.len(), compacted_sizes.len());
    for (index, entry) in entries.iter().enumerate() {
        let buffer_view: phobos::BufferView = compacted_buffer
            .view(as_buffer_offset, *compacted_sizes.get(index).unwrap())
            .unwrap();
        /**
        println!(
            "Size before compactation: {}. Size after: {}",
            entry.buffer.size(),
            buffer_view.size()
        );
        **/
        let new_as = phobos::AccelerationStructure::new(
            ctx.device.clone(),
            phobos::AccelerationStructureType::BottomLevel,
            buffer_view,
            vk::AccelerationStructureCreateFlagsKHR::default(),
        );
        let new_as = new_as.unwrap();
        as_buffer_offset += buffer_view.size();

        let cmd = ctx
            .execution_manager
            .on_domain::<Compute>()
            .unwrap()
            .compact_acceleration_structure(
                as_resource
                    .acceleration_structures
                    .get(entry.handle)
                    .unwrap(),
                &new_as,
            )
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
        new_structures.push(new_as);
        new_entries.push(AllocatedAS {
            buffer: buffer_view,
            scratch: entry.scratch,
            handle: new_structures.len() - 1,
        });
    }
    (
        new_entries,
        AccelerationStructureResources {
            buffer: Some(compacted_buffer),
            scratch: None,
            acceleration_structures: new_structures,
        },
    )
}

/// Creates the instance buffer from all the created entries
pub fn make_instances_buffer(
    ctx: &mut crate::app::Context,
    as_resources: &AccelerationStructureResources,
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
                    as_resources
                        .acceleration_structures
                        .get(entry.handle)
                        .unwrap(),
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
fn get_build_info_sizes(
    ctx: &mut crate::app::Context,
    build_infos: &[AccelerationStructureBuildInfo],
    prim_counts: &[u32],
) -> Vec<Result<phobos::AccelerationStructureBuildSize>> {
    let mut sizes: Vec<Result<phobos::AccelerationStructureBuildSize>> = Vec::new();
    assert_eq!(prim_counts.len(), build_infos.len());
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
fn set_build_infos_sizes<'a>(
    ctx: &mut crate::app::Context,
    build_infos: Vec<AccelerationStructureBuildInfo<'a>>,
    prim_counts: &[u32],
) -> Vec<Result<AccelerationStructureBuildInfo<'a>>> {
    let sizes = get_build_info_sizes(ctx, &build_infos, prim_counts);
    let mut out_build_infos: Vec<Result<AccelerationStructureBuildInfo>> = Vec::new();
    assert_eq!(build_infos.len(), sizes.len());
    for (mut build_info, size) in build_infos.into_iter().zip(sizes.into_iter()) {
        match size {
            Err(e) => out_build_infos.push(Err(e)),
            Ok(size) => {
                build_info.size_info = Some(size);
                out_build_infos.push(Ok(build_info));
            }
        }
    }
    out_build_infos
}

/// Suballocates all given build_infos by setting their buffer_offset and scratch_offset. If you pass in build infos without a
/// size_info, it will leave the AS blank with an error.
fn suballocate_build_infos(
    build_infos: Vec<AccelerationStructureBuildInfo>,
) -> Vec<Result<AccelerationStructureBuildInfo>> {
    let mut out_build_infos: Vec<Result<AccelerationStructureBuildInfo>> =
        Vec::with_capacity(build_infos.len());
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
    println!(
        "[acceleration_structure]: Scene has total # of meshes: {}",
        scene.meshes.len()
    );
    let meshes_selected: Vec<Handle<asset::Mesh>> = scene.meshes.values().cloned().collect();
    let mut blas_build_infos: Vec<AccelerationStructureBuildInfo> =
        get_blas_entries(&meshes_selected, scene)
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

    //let blas_build_infos: Vec<Result<AccelerationStructureBuildInfo>>;
    {
        let primitive_counts: Vec<u32> = blas_build_infos
            .iter()
            .map(|build_info| {
                build_info
                    .handle
                    .as_vulkan()
                    .1
                    .get(0)
                    .unwrap()
                    .primitive_count
            })
            .collect::<Vec<u32>>();
        blas_build_infos =
            set_build_infos_sizes(ctx, blas_build_infos, primitive_counts.as_slice())
                .into_iter()
                .filter_map(|build_info| {
                    build_info
                        .map_err(|e| println!("Failed to bind size_info due to: {}", e))
                        .ok()
                })
                .collect::<Vec<AccelerationStructureBuildInfo>>();
        // Suballocate it
        blas_build_infos = suballocate_build_infos(blas_build_infos)
            .into_iter()
            .filter_map(|build_info| {
                build_info
                    .map_err(|e| println!("Couldn't suballocate AS due to: {}", e))
                    .ok()
            })
            .collect::<Vec<AccelerationStructureBuildInfo>>();
    }
    let (as_resources, entries) = create_acceleration_structure(ctx, &blas_build_infos);
    let compact_sizes =
        build_blas(ctx, &as_resources, &entries, blas_build_infos).expect("Failed to build BLAS");
    let (new_entries, new_as_resources) =
        compact_blases(ctx, &as_resources, &entries, compact_sizes);
    // Make TLAS

    //  TODO: REDO THIS UNHOLY MESS
    //  TODO: LEARN HOW TO CODE

    // as_resources and entries should not be used beyond this point
    assert_eq!(
        new_entries.len(),
        new_as_resources.acceleration_structures.len()
    );
    let instance_buffer = make_instances_buffer(ctx, &new_as_resources, &new_entries)
        .map_err(|e| println!("Could not make instance buffer: {}", e))
        .ok()
        .unwrap();
    let tlas_build_infos = get_tlas_build_infos(&instance_buffer, new_entries.len() as u32);
    let mut tlas_build_infos =
        set_build_infos_sizes(ctx, vec![tlas_build_infos], &[new_entries.len() as u32])
            .into_iter()
            .filter_map(|build_info| {
                build_info
                    .map_err(|e| println!("Failed to make TLAS due to: {}", e))
                    .ok()
            })
            .collect::<Vec<AccelerationStructureBuildInfo>>();
    //assert_eq!(tlas_build_infos.len(), new_entries.len());
    tlas_build_infos = suballocate_build_infos(tlas_build_infos)
        .into_iter()
        .filter_map(|build_info| {
            build_info
                .map_err(|e| println!("Unable suballocate TLAS due to: {}", e))
                .ok()
        })
        .collect::<Vec<AccelerationStructureBuildInfo>>();

    let (tlas_resources, tlas_entries) = create_acceleration_structure(ctx, &tlas_build_infos);
    let mut tlas_build_info = tlas_build_infos.remove(0);
    tlas_build_info.handle = tlas_build_info
        .handle
        .dst(
            tlas_resources
                .acceleration_structures
                .get(tlas_entries.get(0).unwrap().handle)
                .unwrap(),
        )
        .scratch_data(tlas_resources.scratch.as_ref().unwrap().address());

    // Submit tlas command
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
