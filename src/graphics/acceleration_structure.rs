//! Acceleration structure handler
//! Large parts thanks to: https://github.com/NotAPenguin0/Andromeda/blob/master/include/andromeda/graphics/backend/rtx.hpp
#![warn(missing_docs)]

use crate::assets;
use crate::utils::handle_storage::Handle;
use crate::utils::memory;
use crate::utils::types;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};

use phobos::domain::Compute;
use phobos::{vk, AccelerationStructureType, ComputeCmdBuffer, IncompleteCmdBuffer};

/// Represents the stored acceleration structure
pub struct AllocatedAS {
    /// Location of the acceleration structure in the buffer
    buffer: phobos::BufferView,

    /// The scratch buffer
    scratch: phobos::BufferView,

    /// The index in the [`AccelerationStructureResources`] which the [`phobos::AccelerationStructure`] is held
    ///
    /// [`AccelerationStructureResources`]: AccelerationStructureResources
    /// [`phobos::AccelerationStructure`]: phobos::AccelerationStructure
    handle: usize,

    /// Transformation of the mesh
    transformation: glam::Mat4,

    /// Name of the acceleration structure
    name: Option<String>,

    /// Handle to the original mesh
    mesh_handle: Option<Handle<assets::mesh::Mesh>>,
}

/// Contains all the resources used by [`AllocatedAS`]
///
/// [`AllocatedAS`]: AllocatedAS
pub struct AccelerationStructureResources {
    /// The buffer which acceleration structures are stored in
    pub(crate) buffer: Option<phobos::Buffer>,

    /// The scratch buffer
    pub(crate) scratch: Option<phobos::Buffer>,

    /// Contains all the acceleration structures
    pub(crate) acceleration_structures: Vec<phobos::AccelerationStructure>,
}

/// Contains both the [`AllocatedAS`] and the [`AccelerationStructureResources`] to represent all
/// the acceleration structures and their respective resources
///
/// [`AllocatedAS`]: AllocatedAS
/// [`AccelerationStructureResources`]: AccelerationStructureResources
pub struct AccelerationStructure {
    pub(crate) resources: AccelerationStructureResources,
    pub(crate) instances: Vec<AllocatedAS>,
}

/// A struct that represents a BLAS's buffers to be used by the shader
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BLASBufferAddresses {
    pub(crate) vertex_buffer: u64,
    pub(crate) index_buffer: u64,
    pub(crate) normal_buffer: u64,
    pub(crate) tex_buffer: u64,
}

/// All the acceleration structures used in a scene
pub struct SceneAccelerationStructure {
    pub(crate) tlas: AccelerationStructure,
    pub(crate) blas: AccelerationStructure,
    pub(crate) instances: phobos::Buffer,
    pub(crate) addresses: Vec<assets::mesh::CMesh>,
    pub(crate) addresses_buffer: Option<phobos::Buffer>,
}

/// Contains additional information to build acceleration structures
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

    /// Name of the AS
    name: Option<String>,

    /// Transformation
    transformation: glam::Mat4,

    /// Mesh handle
    mesh_handle: Option<Handle<assets::mesh::Mesh>>,
}

/// Gets the build information of the BLASes based off of the scene
/// # Errors
/// This function will panic if:
/// * The given mesh is not a triangle such as not having indices which are divisible by 3
/// * Mesh stride is zero
fn get_blas_entries<'a>(
    meshes: &Vec<Handle<assets::mesh::Mesh>>,
    scene: &assets::scene::Scene,
) -> Vec<(
    Result<phobos::AccelerationStructureBuildInfo<'a>>,
    Handle<assets::mesh::Mesh>,
    Option<assets::mesh::Mesh>,
)> {
    let mut build_infos: Vec<(
        Result<phobos::AccelerationStructureBuildInfo<'a>>,
        Handle<assets::mesh::Mesh>,
        Option<assets::mesh::Mesh>,
    )> = Vec::with_capacity(meshes.len());

    for mesh_handle in meshes {
        let mesh = scene.meshes_storage.get_immutable(mesh_handle);
        if mesh.is_none() {
            println!("Mesh not found!");
            build_infos.push((
                Err(anyhow::anyhow!("No mesh found")),
                mesh_handle.clone(),
                None,
            ));
            continue;
        }
        // Retrieve the mesh's vertex and index buffer
        let mesh = mesh.unwrap();
        let vertex_buffer = scene.attributes_storage.get_immutable(&mesh.vertex_buffer);
        let index_buffer = scene.attributes_storage.get_immutable(&mesh.index_buffer);
        if vertex_buffer.is_none() || index_buffer.is_none() {
            println!("Failed to find vertex / index buffer");
            build_infos.push((
                Err(anyhow::anyhow!("No vertex or index buffer found")),
                mesh_handle.clone(),
                Some(mesh.as_ref().clone()),
            ));
            continue;
        }
        let vertex_buffer = vertex_buffer.unwrap();
        let index_buffer = index_buffer.unwrap();
        let index_type = types::convert_scalar_format_to_index(index_buffer.format);
        if index_type.is_none() {
            build_infos.push((
                Err(anyhow::anyhow!(
                    "No index type found. Found: {:?}",
                    index_buffer.format
                )),
                mesh_handle.clone(),
                Some(mesh.as_ref().clone()),
            ));
            continue;
        }

        // Create the build information for the mesh
        let index_type = index_type.unwrap();
        // Check if the number of indices can make a triangle
        assert_eq!(
            (index_buffer.count % 3),
            0,
            "[acceleration_structure]: # of indices in the mesh exceeds 3000."
        );
        assert!(
            vertex_buffer.stride > 0,
            "[acceleration_structure]: Given vertex buffer has a stride of zero"
        );
        build_infos.push((
            Ok(
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
            ),
            mesh_handle.clone(),
            Some(mesh.as_ref().clone()),
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
    ctx: Arc<RwLock<crate::app::Context>>,
    build_infos: &Vec<AccelerationStructureBuildInfo<'_>>,
) -> (AccelerationStructureResources, Vec<AllocatedAS>) {
    // Create memory for all the BLASes
    let mut ctx = ctx.write().unwrap();
    let mut entries: Vec<AllocatedAS> = Vec::with_capacity(build_infos.len());
    let mut instances: Vec<phobos::AccelerationStructure> = Vec::with_capacity(build_infos.len());
    let buffer_size = total_blas_memory(build_infos);
    let scratch_size = total_scratch_memory(build_infos);
    let as_buffer =
        phobos::Buffer::new_device_local(ctx.device.clone(), &mut ctx.allocator, buffer_size)
            .unwrap();
    let scratch_buffer =
        phobos::Buffer::new_device_local(ctx.device.clone(), &mut ctx.allocator, scratch_size)
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
            .set_name(
                &acceleration_structure,
                &format!(
                    "{:?} {}",
                    build_info.name.as_ref().unwrap_or(&String::from("Unnamed")),
                    match build_info.handle.ty() {
                        phobos::AccelerationStructureType::TopLevel => "TLAS",
                        phobos::AccelerationStructureType::BottomLevel => "BLAS",
                        _ => "",
                    },
                ),
            )
            .unwrap();
        instances.push(acceleration_structure);
        let entry = AllocatedAS {
            buffer: buffer_view,
            scratch: scratch_view,
            handle: instances.len() - 1,
            transformation: build_info.transformation,
            name: build_info.name.clone(),
            mesh_handle: build_info.mesh_handle.clone(),
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
    ctx: Arc<RwLock<crate::app::Context>>,
    acceleration_structures_resource: &AccelerationStructureResources,
    entries: &[AllocatedAS],
    build_infos: Vec<AccelerationStructureBuildInfo>,
    compact_blas: bool,
) -> Result<Option<Vec<u64>>> {
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

    let ctx_read = ctx.read().unwrap();
    let mut command = ctx_read
        .execution_manager
        .on_domain::<Compute>()?
        .build_acceleration_structures(build_infos.as_slice())?
        .memory_barrier(
            phobos::PipelineStage::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            phobos::PipelineStage::ALL_COMMANDS,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );

    let mut query_pool: Option<phobos::QueryPool<phobos::AccelerationStructureCompactedSizeQuery>> =
        None;

    // Optional compaction
    if compact_blas {
        query_pool = Some(phobos::QueryPool::<
            phobos::AccelerationStructureCompactedSizeQuery,
        >::new(
            ctx_read.device.clone(),
            phobos::QueryPoolCreateInfo {
                count: build_infos.len() as u32,
                statistic_flags: None,
            },
        )?);
        command = command.write_acceleration_structures_properties(
            acceleration_structures_resource
                .acceleration_structures
                .as_slice(),
            query_pool.as_mut().unwrap(),
        )?;
    }
    // Indicate command submission is finished
    let command = command.finish()?;

    ctx_read.execution_manager.submit(command)?.wait()?;

    if let Some(mut query_pool) = query_pool {
        match query_pool.wait_for_all_results() {
            Ok(query_pool) => Ok(Some(query_pool)),
            Err(e) => Err(e),
        }
    } else {
        Ok(None)
    }
}

/// Create the final compacted acceleration structures
fn compact_blases(
    ctx: Arc<RwLock<crate::app::Context>>,
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

    let mut ctx_read = ctx.write().unwrap();
    let compacted_buffer = phobos::Buffer::new_device_local(
        ctx_read.device.clone(),
        &mut ctx_read.allocator,
        total_compacted_size,
    )
    .unwrap();
    ctx_read
        .device
        .set_name(&compacted_buffer, "BLAS Buffer")
        .unwrap();

    // Update the allocatedAS alongside
    let mut as_buffer_offset: u64 = 0;
    assert_eq!(entries.len(), compacted_sizes.len());
    for (index, entry) in entries.iter().enumerate() {
        let buffer_view: phobos::BufferView = compacted_buffer
            .view(as_buffer_offset, *compacted_sizes.get(index).unwrap())
            .unwrap();
        let new_as = phobos::AccelerationStructure::new(
            ctx_read.device.clone(),
            phobos::AccelerationStructureType::BottomLevel,
            buffer_view,
            vk::AccelerationStructureCreateFlagsKHR::default(),
        );
        let new_as = new_as.unwrap();
        as_buffer_offset += buffer_view.size();

        let cmd = ctx_read
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
        ctx_read
            .execution_manager
            .submit(cmd)
            .unwrap()
            .wait()
            .unwrap();
        ctx_read
            .device
            .set_name(
                &new_as,
                &format!(
                    "{} BLAS",
                    entry.name.as_ref().unwrap_or(&String::from("Unnamed"))
                ),
            )
            .unwrap();
        new_structures.push(new_as);
        new_entries.push(AllocatedAS {
            buffer: buffer_view,
            scratch: entry.scratch,
            handle: new_structures.len() - 1,
            transformation: entry.transformation,
            name: entry.name.clone(),
            mesh_handle: entry.mesh_handle.clone(),
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
    ctx: Arc<RwLock<crate::app::Context>>,
    as_resources: &AccelerationStructureResources,
    scene: &assets::scene::Scene,
    entries: &Vec<AllocatedAS>,
) -> (Result<phobos::Buffer>, Vec<assets::mesh::CMesh>) {
    let mut instances: Vec<phobos::AccelerationStructureInstance> = Vec::new();
    let mut bindless_mesh_info: Vec<assets::mesh::CMesh> = Vec::new();
    let mut mesh_index: u32 = 0;
    for entry in entries {
        let m = entry.transformation;
        let transformation_matrix = phobos::TransformMatrix::from_elements(&[
            m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x, // First row
            m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y, // Second row
            m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z, // Third row
        ]);
        let mesh = scene
            .meshes_storage
            .get_immutable(entry.mesh_handle.as_ref().unwrap())
            .unwrap();
        instances.push(
            phobos::AccelerationStructureInstance::default()
                .mask(0xFF)
                .flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE)
                .sbt_record_offset(0)
                .unwrap()
                .custom_index(mesh_index)
                .unwrap()
                .transform(transformation_matrix)
                .acceleration_structure(
                    as_resources
                        .acceleration_structures
                        .get(entry.handle)
                        .unwrap(),
                    phobos::AccelerationStructureBuildType::Device,
                )
                .unwrap(),
        );
        let vertex_buffer = scene
            .attributes_storage
            .get_immutable(&mesh.vertex_buffer)
            .unwrap();
        let index_buffer = scene
            .attributes_storage
            .get_immutable(&mesh.index_buffer)
            .unwrap();

        let get_buffer_address =
            |buffer: &Option<Handle<assets::buffer_view::AttributeView<u8>>>| -> u64 {
                if let Some(buffer) = buffer {
                    if let Some(buffer) = scene.attributes_storage.get_immutable(buffer) {
                        return buffer.buffer_view.address();
                    }
                }
                return 0u64;
            };

        bindless_mesh_info.push(mesh.to_c_struct(scene));
        mesh_index += 1;
        println!("IT'S GOBLIN TIME");
        if mesh.tex_buffer.is_some() {
            println!(
                "{} - {:?}",
                bindless_mesh_info.last().unwrap().tex_buffer,
                scene
                    .attributes_storage
                    .get_immutable(mesh.tex_buffer.as_ref().unwrap())
                    .unwrap()
                    .name
            );
        }
        mesh_index &= 0xFFFFFF;
    }

    (
        memory::make_transfer_buffer(
            ctx.clone(),
            instances.as_slice(),
            Some(16),
            "Instance Buffer",
        ),
        bindless_mesh_info,
    )
}

/// Gets the build sizes with alignment included for the Size and returns all the build sizes
fn get_build_info_sizes(
    ctx: Arc<RwLock<crate::app::Context>>,
    build_infos: &[AccelerationStructureBuildInfo],
    prim_counts: &[u32],
) -> Vec<Result<phobos::AccelerationStructureBuildSize>> {
    let mut sizes: Vec<Result<phobos::AccelerationStructureBuildSize>> = Vec::new();
    assert_eq!(prim_counts.len(), build_infos.len());
    let ctx_read = ctx.read().unwrap();
    for (index, build_info) in build_infos.iter().enumerate() {
        sizes.push(phobos::query_build_size(
            &ctx_read.device,
            phobos::AccelerationStructureBuildType::Device,
            &build_info.handle,
            &prim_counts[index..index + 1],
        ));
    }
    for size_info in sizes.iter_mut().flatten() {
        size_info.build_scratch_size = memory::align_size(
            size_info.build_scratch_size,
            ctx_read
                .device
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
    ctx: Arc<RwLock<crate::app::Context>>,
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

/// Gets all the [`AccelerationStructureBuildInfo`] for the related TLAS
/// # Errors
/// This function will panic if:
/// * Number of `instances_buffers` is not the same as the number of `instance_counts`
///
/// [`AccelerationStructureBuildInfo`]: AccelerationStructureBuildInfo
fn get_tlas_build_infos<'a>(
    instance_buffers: &[&phobos::Buffer],
    instance_counts: &[u32],
) -> Vec<AccelerationStructureBuildInfo<'a>> {
    let mut out_vec: Vec<AccelerationStructureBuildInfo> =
        Vec::with_capacity(instance_buffers.len());
    assert_eq!(
        instance_buffers.len(),
        instance_counts.len(),
        "[acceleration_structure]: Given instances count must be the same as number of instances"
    );
    for (instance_buffer, instance_count) in instance_buffers.iter().zip(instance_counts) {
        out_vec.push(AccelerationStructureBuildInfo {
            handle: phobos::AccelerationStructureBuildInfo::new_build()
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .set_type(phobos::AccelerationStructureType::TopLevel)
                .push_instances(phobos::AccelerationStructureGeometryInstancesData {
                    data: instance_buffer.address().into(),
                    flags: vk::GeometryFlagsKHR::OPAQUE
                        | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
                })
                .push_range(*instance_count, 0, 0, 0),
            size_info: None,
            buffer_offset: 0,
            scratch_offset: 0,
            name: Some(String::from("TLAS")),
            transformation: glam::Mat4::IDENTITY,
            mesh_handle: None,
        });
    }
    out_vec
}

/// Gets the [`AccelerationStructure`] from the passed in [`assets::Scene`]
///
/// [`AccelerationStructure`]: AccelerationStructure
/// [`assets::Scene`]: assets::Scene
pub fn create_blas_from_scene(
    ctx: Arc<RwLock<crate::app::Context>>,
    scene: &assets::scene::Scene,
    do_compaction: bool,
) -> AccelerationStructure {
    // Get build information from the scene
    let meshes_selected: Vec<Handle<assets::mesh::Mesh>> = scene.meshes.clone();
    let mut blas_build_infos: Vec<AccelerationStructureBuildInfo> =
        get_blas_entries(&meshes_selected, scene)
            .into_iter()
            .filter_map(|(x, mesh_handle, mesh)| match x {
                Ok(x) => Some(AccelerationStructureBuildInfo {
                    handle: x,
                    size_info: None,
                    buffer_offset: 0,
                    scratch_offset: 0,
                    name: (mesh.as_ref().unwrap()).name.clone(),
                    transformation: mesh.as_ref().unwrap().transform,
                    mesh_handle: Some(mesh_handle),
                }),
                Err(E) => {
                    println!(
                        "[acceleration_structure]: Failed to load mesh properly: {:?}",
                        E
                    );
                    None
                }
            })
            .collect();
    // Fill the remaining information left over

    // Collect all primitive counts for later usage
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

    // Fill in size_info about the build
    blas_build_infos =
        set_build_infos_sizes(ctx.clone(), blas_build_infos, primitive_counts.as_slice())
            .into_iter()
            .filter_map(|build_info| {
                build_info
                    .map_err(|e| println!("Failed to bind size_info due to: {}", e))
                    .ok()
            })
            .collect::<Vec<AccelerationStructureBuildInfo>>();
    // Suballocate the BLAS with known pre-compact sizes
    blas_build_infos = suballocate_build_infos(blas_build_infos)
        .into_iter()
        .filter_map(|build_info| {
            build_info
                .map_err(|e| {
                    println!(
                        "[acceleration_structure]: Couldn't suballocate AS due to: {}",
                        e
                    )
                })
                .ok()
        })
        .collect::<Vec<AccelerationStructureBuildInfo>>();

    // Creates the acceleration structures
    let (mut as_resources, mut allocated_structures) =
        create_acceleration_structure(ctx.clone(), &blas_build_infos);
    // Builds BLAS with compact
    let compact_sizes = build_blas(
        ctx.clone(),
        &as_resources,
        &allocated_structures,
        blas_build_infos,
        do_compaction,
    )
    .expect("[acceleration_structure]: Failed to build BLAS");

    if do_compaction {
        // Override existing BLAS with compacted BLAS
        let compact_sizes = compact_sizes
            .expect("[acceleration_structure]: Expected sizes from compaction, found None");
        let (new_allocated_structures, new_as_resources) =
            compact_blases(ctx, &as_resources, &allocated_structures, compact_sizes);
        return AccelerationStructure {
            resources: new_as_resources,
            instances: new_allocated_structures,
        };
    }
    AccelerationStructure {
        resources: as_resources,
        instances: allocated_structures,
    }
}

/// Creates a top level acceleration structure based off of the given blas passed in
/// # Errors
/// This function will return an error if:
/// * Command buffer submission failed
/// This function will panic if:
/// * The number of BLAS [`AllocatedAS`] is not the same as the number of instances
pub fn create_tlas(
    ctx: Arc<RwLock<crate::app::Context>>,
    scene: &assets::scene::Scene,
    blas: &AccelerationStructure,
) -> Result<(
    AccelerationStructure,
    phobos::Buffer,
    Vec<assets::mesh::CMesh>,
)> {
    assert_eq!(
        blas.resources.acceleration_structures.len(),
        blas.instances.len(),
        "BLAS stored structures is not the same Size that the number of instances"
    );
    let (instance_buffer, bindless_meshes) =
        make_instances_buffer(ctx.clone(), &blas.resources, scene, &blas.instances);
    let instance_buffer = instance_buffer
        .map_err(|e| {
            println!(
                "[acceleration_structure]: Couldn't make instance buffer for TLAS: {}",
                e
            )
        })
        .ok()
        .unwrap();
    // Get TLAS build information
    let mut tlas_build_infos =
        get_tlas_build_infos(&[&instance_buffer], &[blas.instances.len() as u32]);
    tlas_build_infos = set_build_infos_sizes(
        ctx.clone(),
        tlas_build_infos,
        &[blas.instances.len() as u32],
    )
    .into_iter()
    .filter_map(|build_info| {
        build_info
            .map_err(|e| {
                println!(
                    "[acceleration_structure]: Failed to make TLAS due to: {}",
                    e
                )
            })
            .ok()
    })
    .collect::<Vec<AccelerationStructureBuildInfo>>();
    // suballocate build information (pretty useless, but just in case)
    tlas_build_infos = suballocate_build_infos(tlas_build_infos)
        .into_iter()
        .filter_map(|build_info| {
            build_info
                .map_err(|e| println!("[acceleration_structure]: Failed to make TLAS: {}", e))
                .ok()
        })
        .collect::<Vec<AccelerationStructureBuildInfo>>();

    let (as_resources, allocated_structures) =
        create_acceleration_structure(ctx.clone(), &tlas_build_infos);
    // Assign the as and scratch address
    let mut tlas_build_info: AccelerationStructureBuildInfo = tlas_build_infos.remove(0);
    tlas_build_info.handle = tlas_build_info
        .handle
        .dst(
            as_resources
                .acceleration_structures
                .get(allocated_structures.first().unwrap().handle)
                .unwrap(),
        )
        .scratch_data(as_resources.scratch.as_ref().unwrap().address());

    // Submit tlas commands
    let ctx_read = ctx.read().unwrap();
    let command = ctx_read
        .execution_manager
        .on_domain::<Compute>()
        .unwrap()
        .memory_barrier(
            phobos::PipelineStage::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            phobos::PipelineStage::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ,
        )
        .build_acceleration_structure(&tlas_build_info.handle)?
        .finish()?;
    ctx_read.execution_manager.submit(command)?.wait()?;
    Ok((
        AccelerationStructure {
            resources: as_resources,
            instances: allocated_structures,
        },
        instance_buffer,
        bindless_meshes,
    ))
}

pub fn convert_scene_to_blas(
    ctx: Arc<RwLock<crate::app::Context>>,
    scene: &assets::scene::Scene,
) -> SceneAccelerationStructure {
    println!(
        "[acceleration_structure]: Scene has total # of meshes: {}",
        scene.meshes.len()
    );
    let do_compaction: bool = true; // Whether or not to do compaction
    let blas: AccelerationStructure = create_blas_from_scene(ctx.clone(), scene, do_compaction);
    let (tlas, tlas_instance_buffer, bindless_meshes) =
        create_tlas(ctx.clone(), scene, &blas).expect("Failed to build TLAS");
    let address_buffer = memory::make_transfer_buffer(
        ctx.clone(),
        bindless_meshes.as_slice(),
        None,
        "Object description",
    )
    .unwrap();
    let address_buffer =
        memory::copy_buffer_to_gpu_buffer(ctx.clone(), address_buffer, "Addresses").unwrap();

    SceneAccelerationStructure {
        tlas,
        blas,
        instances: tlas_instance_buffer,
        addresses: bindless_meshes,
        addresses_buffer: Some(address_buffer),
    }
}

// TODO: FOR THE SEGFAULTING MEMORY ERROR LOOK INTO
// - KEEPING THE LIFETIME OF THE OLD BUFFERS ALIVE
// - KEEPING THE LIFETIME OF THE OLD AS ALIVE
