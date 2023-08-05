use crate::assets::loader::LoadableAsset;
use crate::utils::handle_storage::{Handle, Storage};
use crate::utils::memory;
use crate::{app, assets};
use anyhow::Result;
use ash::vk;
use gltf;
use gltf::image::Format;
use phobos::{IncompleteCmdBuffer, TransferCmdBuffer};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ptr;
use std::sync::{Arc, RwLock};

mod structs {
    use crate::assets;
    use std::collections::{BTreeMap, HashMap};

    pub struct GltfPrimitive {
        pub name: Option<String>,
        pub handle: Option<gltf::json::mesh::Primitive>,
        pub index: usize,
        pub transform: glam::Mat4,
        pub accessors: HashMap<AccessorSemantic, usize>,
        pub material: Option<usize>,
    }

    #[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Clone, Debug)]
    pub enum AccessorSemantic {
        Gltf(gltf::Semantic),
        Index,
    }

    #[derive(Clone, Debug)]
    pub struct Accessor {
        pub name: Option<String>,
        pub handle: Option<gltf::json::Accessor>,
        pub semantic: AccessorSemantic,
        pub component_type: gltf::accessor::DataType,
        pub dimension: gltf::accessor::Dimensions,
        pub component_size: usize,
        pub index: usize,
        pub offset: usize,
        pub size: usize,
    }

    pub struct BufferView {
        pub handle: Option<gltf::json::buffer::View>,
        pub index: usize,
    }

    pub struct GltfBuffer {
        pub handle: Option<gltf::json::Buffer>,
        pub index: usize,
    }

    pub struct CpuBuffer {
        pub handle: Option<assets::buffer::CpuBuffer>,
        pub index: usize,
    }
}

/// Deserialize imported gltf file
fn deserialize(
    path: std::path::PathBuf,
) -> Result<(
    gltf::json::Root,
    Vec<gltf::buffer::Data>,
    Vec<gltf::image::Data>,
)> {
    let (document, buffers, images) = gltf::import(path)?;
    let mut document_json = document.clone().into_json();
    Ok((document_json, buffers, images))
}

/// Turn scene graph into a flat vector
fn flatten_primitives(
    root_nodes: Vec<&gltf::json::scene::Node>,
    document: &gltf::json::Root,
) -> (Vec<structs::GltfPrimitive>, Vec<(usize, glam::Mat4)>) {
    let get_transformation =
        |translation: Option<[f32; 3]>, rotation: Option<[f32; 4]>, scale: Option<[f32; 3]>| {
            glam::Mat4::from_scale(scale.map(glam::Vec3::from).unwrap_or(glam::Vec3::ONE))
                * glam::Mat4::from_quat(
                    rotation
                        .map(glam::Quat::from_array)
                        .unwrap_or(glam::Quat::IDENTITY),
                )
                * glam::Mat4::from_translation(
                    translation
                        .map(glam::Vec3::from)
                        .unwrap_or(glam::Vec3::ZERO),
                )
        };

    //convert.y_axis.y *= -1f32;
    //let convert = glam::Mat4::IDENTITY;
    let mut nodes: Vec<(gltf::json::Node, glam::Mat4)> = root_nodes
        .into_iter()
        .map(|node| {
            (
                node.clone(),
                node.matrix
                    .map(|x| glam::Mat4::from_cols_array(&x))
                    .unwrap_or(get_transformation(
                        node.translation,
                        node.rotation.map(|x| x.0),
                        node.scale,
                    )),
            )
        })
        .collect();
    let mut unique_primitives: Vec<structs::GltfPrimitive> = Vec::new(); // Vector containing unique primitives
    let mut primitive_indices: Vec<(usize, glam::Mat4)> = Vec::new(); // Vector that contains indices pointing into the vector with unique indices
    let mut seen_primitives: HashMap<(usize, usize), usize> = HashMap::new();
    while !nodes.is_empty() {
        let (node, transform) = nodes.pop().unwrap();
        if let Some(mesh_index) = node.mesh {
            let mesh = &document.meshes[mesh_index.value()];
            for (index, primitive) in mesh.primitives.iter().enumerate() {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    seen_primitives.entry((mesh_index.value(), index))
                {
                    unique_primitives.push(structs::GltfPrimitive {
                        name: Some(format!(
                            "{} {}",
                            mesh.name.as_ref().unwrap_or(&String::from("Unnamed")),
                            index
                        )),
                        handle: Some(primitive.clone()),
                        index: unique_primitives.len(),
                        transform,
                        accessors: HashMap::new(),
                        material: primitive.material.map(|x| x.value()),
                    });
                    println!(
                        "Adding mesh: {} with transform: {:?}",
                        mesh.name.as_ref().unwrap_or(&String::from("Unnamed")),
                        transform
                    );
                    e.insert(unique_primitives.len() - 1);
                    primitive_indices.push(((unique_primitives.len() - 1), transform));
                } else {
                    primitive_indices.push((
                        *seen_primitives.get(&(mesh_index.value(), index)).unwrap(),
                        transform,
                    ));
                }
            }
        }
        if let Some(node_childrens) = node.children {
            for child_node_index in node_childrens {
                let mut child_transform = transform;
                let child_node = document.nodes[child_node_index.value()].clone();
                if let Some(matrix) = child_node.matrix {
                    child_transform *= glam::Mat4::from_cols_array(&matrix);
                } else {
                    child_transform *= get_transformation(
                        child_node.translation,
                        child_node.rotation.map(|x| x.0),
                        child_node.scale,
                    );
                }
                nodes.push((child_node, child_transform));
            }
        }
    }
    unique_primitives.sort_by(|x, y| x.index.cmp(&y.index));

    (unique_primitives, primitive_indices)
}

/// Gets all gltf buffers
fn get_buffers() -> Vec<structs::GltfBuffer> {
    todo!()
}

/// Generate normals
fn generate_normals(
    document: &gltf::json::Root,
    monolithic_buffer: &[u8],
    vertex_accessor: &structs::Accessor,
    index_accessor: &structs::Accessor,
) -> Vec<u8> {
    todo!()
}

/// Convert a slice of bytes as another
fn cast_bytes_slice_type<
    T: 'static + num_traits::ToPrimitive + bytemuck::Pod + Sync,
    U: 'static + num_traits::NumCast + bytemuck::Pod + Send,
>(
    contents: &[u8],
) -> Vec<u8> {
    // If same type, return early
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>() {
        println!(
            "I AM NOT CONVERTING: {} -> {}",
            std::any::type_name::<T>(),
            std::any::type_name::<U>()
        );
        return contents.to_vec();
    }
    let from_size = std::mem::size_of::<T>();
    let target_size = std::mem::size_of::<U>();
    println!(
        "I AM NOW CONVERTING: {} -> {}",
        std::any::type_name::<T>(),
        std::any::type_name::<U>()
    );

    assert_eq!(contents.len() % from_size, 0); // Must be perfectly sliced

    // Cast contents (bytes) -> T -> Cast to U -> Reinterpret back to bytes
    let intermediate: Vec<U> = bytemuck::cast_slice::<u8, T>(contents)
        .iter()
        .map(|x| num_traits::NumCast::from(*x).unwrap())
        .collect();

    assert_eq!(intermediate.len(), contents.len() / from_size);

    bytemuck::cast_slice::<U, u8>(intermediate.as_slice()).to_vec()
}

/// Normalizes the slice of one bytes to another
fn normalize_bytes_slice_type<
    T: 'static
        + num_traits::ToPrimitive
        + bytemuck::Pod
        + Sync
        + num_traits::Bounded
        + std::ops::Div
        + std::ops::Mul,
    U: 'static + num_traits::NumCast + bytemuck::Pod + Send + num_traits::Bounded,
>(
    contents: &[u8],
) -> Vec<u8> {
    // If same type, return early
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>() {
        println!(
            "I AM NOT CONVERTING: {} -> {}",
            std::any::type_name::<T>(),
            std::any::type_name::<U>()
        );
        return contents.to_vec();
    }
    println!(
        "I AM NOW CONVERTING: {} -> {}",
        std::any::type_name::<T>(),
        std::any::type_name::<U>()
    );

    let from_size = std::mem::size_of::<T>();
    let target_size = std::mem::size_of::<U>();

    assert_eq!(contents.len() % from_size, 0); // Must be perfectly sliced

    // Cast contents (bytes) -> T -> Cast to U -> Reinterpret back to bytes
    let contents: Vec<U> = bytemuck::cast_slice::<u8, T>(contents)
        .par_iter()
        .map(|x: &T| {
            let max_value = T::max_value().to_f32().unwrap();
            let value = x.to_f32().unwrap();
            num_traits::NumCast::from(
                (value / max_value * U::max_value().to_f32().unwrap())
                    .round()
                    .clamp(
                        U::min_value().to_f32().unwrap(),
                        U::max_value().to_f32().unwrap(),
                    ),
            )
            .unwrap()
        })
        .collect();
    bytemuck::cast_slice::<U, u8>(contents.as_slice()).to_vec()
}

/// Convert the dimensions of a given slice to another
fn convert_dimensions<T: bytemuck::Zeroable + bytemuck::Pod + Sync + Send>(
    from_components: usize,
    to_components: usize,
    contents: &[u8],
) -> Vec<u8> {
    if from_components == to_components {
        return contents.to_vec();
    }
    let component_size = std::mem::size_of::<T>();
    assert_eq!(contents.len() % component_size, 0); // Right component size
    assert_eq!((contents.len() / component_size) % from_components, 0); // Correct from size

    let components: Vec<T> = bytemuck::cast_slice::<u8, T>(contents).to_vec();
    let mut result: Vec<T> = Vec::new();

    // Quickly convert dimensions
    let intermediate = components
        .chunks(from_components)
        .map(|chunk| {
            let mut inner: Vec<T> = Vec::with_capacity(to_components);
            if from_components > to_components {
                inner.extend_from_slice(&chunk[0..to_components]);
            } else {
                inner.extend_from_slice(chunk);
                for _ in chunk.len()..to_components {
                    inner.push(T::zeroed());
                }
            }
            inner
        })
        .collect::<Vec<Vec<T>>>();
    for vec in intermediate {
        result.extend(vec);
    }

    bytemuck::cast_slice::<T, u8>(&result).to_vec()
}

#[cfg(test)]
mod tests {
    use crate::assets::scene::loader::gltf::*;
    use bytemuck;

    #[test]
    fn dimension_conversion() {
        let values = [0u8, 1u8, 3u8, 0u8, 4u8, 3u8];

        // Expanding
        let converted_values = convert_dimensions::<u8>(3, 4, values.as_slice());
        assert_eq!(converted_values, [0u8, 1u8, 3u8, 0u8, 0u8, 4u8, 3u8, 0u8]);

        // Shortening
        let converted_values = convert_dimensions::<u8>(3, 2, &values);
        assert_eq!(converted_values, [0u8, 1u8, 0u8, 4u8]);

        let converted_values = convert_dimensions::<u8>(3, 1, &values);
        assert_eq!(converted_values, [0u8, 0u8]);

        // Same
        let converted_values = convert_dimensions::<u8>(3, 3, &values);
        assert_eq!(converted_values, [0u8, 1u8, 3u8, 0u8, 4u8, 3u8]);
    }

    #[test]
    fn normalize_byte_slice() {
        let values = [f32::MAX, f32::MAX, 0f32];
        let normalized_values =
            normalize_bytes_slice_type::<f32, u8>(bytemuck::cast_slice(values.as_slice()));
        assert_eq!(normalized_values, [u8::MAX, u8::MAX, 0u8]);
    }

    #[test]
    fn cast_byte_slice() {
        let values = [0u16, 4, 8, 2];
        let casted_values = cast_bytes_slice_type::<u16, u32>(bytemuck::cast_slice(&values));
        let casted_values = bytemuck::cast_slice::<u8, u32>(&casted_values);
        assert_eq!(casted_values, [0u32, 4, 8, 2]);
    }
}

/// Get all accessor information
fn get_accessors_from_primitives(
    context: Arc<RwLock<app::Context>>,
    primitives: &mut [structs::GltfPrimitive],
    document: &gltf::json::Root,
    buffers: &[gltf::buffer::Data],
) -> (Vec<assets::buffer_view::AttributeView<u8>>, phobos::Buffer) {
    // COLLECT ALL ACCESSORS' CONTENTS INTO MONOLITHIC BUFFER
    let mut monolithic_buffer: Vec<u8> = Vec::new();

    // TODO: Primitives can reference the same accessor with our current method,
    // we may end up with duplicates as accessors may refer to the same data

    let mut accessor_struct_views: Vec<structs::Accessor> = Vec::new();
    for primitive in primitives.iter_mut() {
        // Lambda function to process accessors
        let mut process_accessor = |accessor_index: usize, semantic: &structs::AccessorSemantic| {
            let accessor = &document.accessors[accessor_index];
            let view = &document.buffer_views[accessor.buffer_view.unwrap().value()];
            use gltf::json::accessor::{ComponentType, Type};
            use gltf::Semantic;
            // Do type conversions
            // Create the desired dimension + type
            // Convert dimensions first
            // Cast components

            let target_component_type: ComponentType = match &semantic {
                structs::AccessorSemantic::Gltf(gltf) => match gltf {
                    Semantic::Positions
                    | Semantic::Normals
                    | Semantic::TexCoords(_)
                    | Semantic::Tangents => ComponentType::F32,
                    _ => ComponentType::U32,
                },
                structs::AccessorSemantic::Index => ComponentType::U32,
            };

            let target_dimension: Type = match &semantic {
                structs::AccessorSemantic::Gltf(gltf) => match gltf {
                    Semantic::Positions | Semantic::Normals => Type::Vec3,
                    Semantic::Tangents => Type::Vec4,
                    Semantic::TexCoords(_) => Type::Vec2,
                    _ => Type::Scalar,
                },
                structs::AccessorSemantic::Index => Type::Scalar,
            };

            // Get accessors contents
            let accessor_contents = get_accessors_contents(accessor, view, document, buffers);
            if accessor.type_.unwrap().multiplicity() == 2 {
                println!("{:?}", bytemuck::cast_slice::<u8, f32>(&accessor_contents));
            }
            // Convert accessor dimensions first
            let accessor_contents = match accessor.component_type.unwrap().0 {
                ComponentType::I8 => convert_dimensions::<i8>(
                    accessor.type_.unwrap().multiplicity(),
                    target_dimension.multiplicity(),
                    accessor_contents.as_slice(),
                ),
                ComponentType::U8 => convert_dimensions::<u8>(
                    accessor.type_.unwrap().multiplicity(),
                    target_dimension.multiplicity(),
                    accessor_contents.as_slice(),
                ),
                ComponentType::I16 => convert_dimensions::<i16>(
                    accessor.type_.unwrap().multiplicity(),
                    target_dimension.multiplicity(),
                    accessor_contents.as_slice(),
                ),
                ComponentType::U16 => convert_dimensions::<u16>(
                    accessor.type_.unwrap().multiplicity(),
                    target_dimension.multiplicity(),
                    accessor_contents.as_slice(),
                ),
                ComponentType::U32 => convert_dimensions::<u32>(
                    accessor.type_.unwrap().multiplicity(),
                    target_dimension.multiplicity(),
                    accessor_contents.as_slice(),
                ),
                ComponentType::F32 => convert_dimensions::<f32>(
                    accessor.type_.unwrap().multiplicity(),
                    target_dimension.multiplicity(),
                    accessor_contents.as_slice(),
                ),
            };
            // Now convert types (i hate this code)
            let mut accessor_contents =
                match (accessor.component_type.unwrap().0, target_component_type) {
                    // Same types
                    (ComponentType::I8, ComponentType::I8) => accessor_contents,
                    (ComponentType::U8, ComponentType::U8) => accessor_contents,
                    (ComponentType::I16, ComponentType::I16) => accessor_contents,
                    (ComponentType::U16, ComponentType::U16) => accessor_contents,
                    (ComponentType::U32, ComponentType::U32) => accessor_contents,
                    (ComponentType::F32, ComponentType::F32) => accessor_contents,
                    // i8 -> *
                    (ComponentType::I8, ComponentType::U8) => {
                        cast_bytes_slice_type::<i8, u8>(&accessor_contents)
                    }
                    (ComponentType::I8, ComponentType::I16) => {
                        cast_bytes_slice_type::<i8, i16>(&accessor_contents)
                    }
                    (ComponentType::I8, ComponentType::U16) => {
                        cast_bytes_slice_type::<i8, u16>(&accessor_contents)
                    }
                    (ComponentType::I8, ComponentType::U32) => {
                        cast_bytes_slice_type::<i8, u32>(&accessor_contents)
                    }
                    (ComponentType::I8, ComponentType::F32) => {
                        cast_bytes_slice_type::<i8, f32>(&accessor_contents)
                    }
                    // u8 -> *
                    (ComponentType::U8, ComponentType::I8) => {
                        cast_bytes_slice_type::<u8, i8>(&accessor_contents)
                    }
                    (ComponentType::U8, ComponentType::I16) => {
                        cast_bytes_slice_type::<u8, i16>(&accessor_contents)
                    }
                    (ComponentType::U8, ComponentType::U16) => {
                        cast_bytes_slice_type::<u8, u16>(&accessor_contents)
                    }
                    (ComponentType::U8, ComponentType::U32) => {
                        cast_bytes_slice_type::<u8, u32>(&accessor_contents)
                    }
                    (ComponentType::U8, ComponentType::F32) => {
                        cast_bytes_slice_type::<u8, f32>(&accessor_contents)
                    }
                    // i16 -> *
                    (ComponentType::I16, ComponentType::I8) => {
                        cast_bytes_slice_type::<i16, i8>(&accessor_contents)
                    }
                    (ComponentType::I16, ComponentType::U8) => {
                        cast_bytes_slice_type::<i16, u8>(&accessor_contents)
                    }
                    (ComponentType::I16, ComponentType::U16) => {
                        cast_bytes_slice_type::<i16, u16>(&accessor_contents)
                    }
                    (ComponentType::I16, ComponentType::U32) => {
                        cast_bytes_slice_type::<i16, u32>(&accessor_contents)
                    }
                    (ComponentType::I16, ComponentType::F32) => {
                        cast_bytes_slice_type::<i16, f32>(&accessor_contents)
                    }
                    // u16 -> *
                    (ComponentType::U16, ComponentType::I8) => {
                        cast_bytes_slice_type::<u16, i8>(&accessor_contents)
                    }
                    (ComponentType::U16, ComponentType::U8) => {
                        cast_bytes_slice_type::<u16, u8>(&accessor_contents)
                    }
                    (ComponentType::U16, ComponentType::I16) => {
                        cast_bytes_slice_type::<u16, i16>(&accessor_contents)
                    }
                    (ComponentType::U16, ComponentType::U32) => {
                        cast_bytes_slice_type::<u16, u32>(&accessor_contents)
                    }
                    (ComponentType::U16, ComponentType::F32) => {
                        cast_bytes_slice_type::<u16, f32>(&accessor_contents)
                    }
                    // u32 -> *
                    (ComponentType::U32, ComponentType::I8) => {
                        cast_bytes_slice_type::<u32, i8>(&accessor_contents)
                    }
                    (ComponentType::U32, ComponentType::U8) => {
                        cast_bytes_slice_type::<u32, u8>(&accessor_contents)
                    }
                    (ComponentType::U32, ComponentType::I16) => {
                        cast_bytes_slice_type::<u32, i16>(&accessor_contents)
                    }
                    (ComponentType::U32, ComponentType::U16) => {
                        cast_bytes_slice_type::<u32, u16>(&accessor_contents)
                    }
                    (ComponentType::U32, ComponentType::F32) => {
                        cast_bytes_slice_type::<u32, f32>(&accessor_contents)
                    }
                    // f32 -> *
                    (ComponentType::F32, ComponentType::I8) => {
                        cast_bytes_slice_type::<f32, i8>(&accessor_contents)
                    }
                    (ComponentType::F32, ComponentType::U8) => {
                        cast_bytes_slice_type::<f32, u8>(&accessor_contents)
                    }
                    (ComponentType::F32, ComponentType::I16) => {
                        cast_bytes_slice_type::<f32, i16>(&accessor_contents)
                    }
                    (ComponentType::F32, ComponentType::U16) => {
                        cast_bytes_slice_type::<f32, u16>(&accessor_contents)
                    }
                    (ComponentType::F32, ComponentType::U32) => {
                        cast_bytes_slice_type::<f32, u32>(&accessor_contents)
                    }
                    (_, _) => {
                        panic!("Unsupported component type conversion!");
                    }
                };
            println!(
                "Transforming CType: {:?} -> {:?}",
                accessor.component_type.unwrap().0,
                target_component_type
            );
            println!(
                "Transforming dimension: {:?} -> {:?}",
                accessor.type_.unwrap().multiplicity(),
                target_dimension.multiplicity()
            );
            // Use a temporary entry struct as the asset version requires a buffer view which we
            // do not have
            let mut entry = structs::Accessor {
                name: Some(format![
                    "{} accessor {:?} {}",
                    primitive.name.as_ref().unwrap(),
                    semantic.clone(),
                    accessor_index,
                ]),
                handle: Some(accessor.clone()),
                semantic: semantic.clone(),
                component_type: target_component_type,
                dimension: target_dimension,
                component_size: target_component_type.size(),
                index: accessor_index,
                size: 0,
                offset: 0,
            };

            // Get contents of the accessor, add to the monolithic buffer, update offsets
            entry.size = accessor_contents.len();
            entry.offset = monolithic_buffer.len();
            monolithic_buffer.append(&mut accessor_contents);
            println!("{:?}", entry);

            accessor_struct_views.push(entry);
            // Put index of the accessor struct in the vector
            primitive
                .accessors
                .insert(semantic.clone(), accessor_struct_views.len() - 1);
        };

        // Handle indices separately as for some reason they're not included
        if let Some(indices) = primitive.handle.as_ref().unwrap().indices {
            process_accessor(indices.value(), &structs::AccessorSemantic::Index);
        } else {
            // All primitives must have indices
            println!(
                "{:?} does not have indices. Skipping",
                primitive.name.clone()
            );
            continue;
        }

        let primitive_accessors: HashMap<structs::AccessorSemantic, Vec<u8>> = HashMap::new();
        for (semantic, accessor_index) in primitive.handle.as_ref().unwrap().attributes.iter() {
            let semantic = semantic.clone().unwrap();
            process_accessor(
                accessor_index.value(),
                &structs::AccessorSemantic::Gltf(semantic.clone()),
            );
        }
    }

    // Store the monolithic buffer into device memory
    let monolithic_buffer = memory::make_transfer_buffer(
        context.clone(),
        monolithic_buffer.as_slice(),
        None,
        "Scene monolithic transfer buffer",
    )
    .unwrap();
    let monolithic_buffer = memory::copy_buffer_to_gpu_buffer(
        context.clone(),
        monolithic_buffer,
        "Scene monolithic buffer",
    )
    .unwrap();

    // Create structs + buffer_views
    let mut accessor_asset_views: Vec<assets::buffer_view::AttributeView<u8>> =
        Vec::with_capacity(accessor_struct_views.len());
    for (index, struct_entry) in accessor_struct_views.iter().enumerate() {
        let accessor = &document.accessors[struct_entry.index];

        let view = &document.buffer_views[accessor.buffer_view.unwrap().value()];
        let entry: assets::buffer_view::AttributeView<u8> = assets::buffer_view::AttributeView {
            name: struct_entry.name.clone(),
            buffer_view: monolithic_buffer
                .view(
                    struct_entry.offset as vk::DeviceSize,
                    struct_entry.size as vk::DeviceSize,
                )
                .unwrap(),
            stride: (struct_entry.dimension.multiplicity() * struct_entry.component_type.size())
                as u64,
            format: get_vk_format(struct_entry.component_type, struct_entry.dimension),
            count: accessor.count as u64,
            component_size: struct_entry.component_type.size() as u64,
            _marker: Default::default(),
            size: struct_entry.size as u64,
            offset: struct_entry.offset as u64,
        };
        println!(
            "Name: {:?} CType: {:?} Dimension: {:?} Format: {:?} Index: {} Stride: {} Range: {}-{} Address: {}",
            struct_entry.name,
            struct_entry.component_type,
            struct_entry.dimension,
            entry.format,
            index,
            entry.stride,
            entry.offset,
            entry.offset + entry.size,
            entry.buffer_view.address(),
        );
        accessor_asset_views.push(entry);
    }

    (accessor_asset_views, monolithic_buffer)
}

fn get_image_type_rgba(image_format: gltf::image::Format) -> gltf::image::Format {
    match image_format {
        Format::R8 | Format::R16 => Format::R8,
        Format::R8G8 | Format::R16G16 => Format::R8G8,
        Format::R8G8B8 | Format::R16G16B16 | Format::R32G32B32FLOAT => Format::R8G8B8,
        Format::R8G8B8A8 | Format::R16G16B16A16 | Format::R32G32B32A32FLOAT => Format::R8G8B8A8,
    }
}

fn load_images(
    context: Arc<RwLock<app::Context>>,
    scene: &mut assets::scene::Scene,
    document: &gltf::json::Root,
    images: &[gltf::image::Data],
) {
    {
        let ctx_read = context.read().unwrap();
        let images: Vec<assets::image::Image> = Vec::new();
        // One universal sampler
        let samplers: Vec<phobos::Sampler> = vec![phobos::Sampler::new(
            ctx_read.device.clone(),
            vk::SamplerCreateInfo {
                s_type: vk::StructureType::SAMPLER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::SamplerCreateFlags::empty(),
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
            },
        )
        .unwrap()];
        for sampler in samplers.into_iter() {
            scene.samplers.push(scene.sampler_storage.insert(sampler));
        }
    }

    let mut vk_images: Vec<phobos::Image> = Vec::new();
    let mut transfer_buffers: Vec<phobos::Buffer> = Vec::new();

    {
        // Make transfer buffers
        for (index, image) in document.images.iter().enumerate() {
            let image_data = &images[index];
            // Do some type conversions first
            // All types should be 8_UNORM
            let image_contents = image_data.pixels.as_slice();
            let dimension_size = match image_data.format {
                Format::R8 | Format::R16 => 1,
                Format::R8G8 | Format::R16G16 => 2,
                Format::R8G8B8 | Format::R16G16B16 | Format::R32G32B32FLOAT => 3,
                Format::R8G8B8A8 | Format::R16G16B16A16 | Format::R32G32B32A32FLOAT => 4,
            };
            let image_contents = match image_data.format {
                Format::R8 | Format::R8G8 | Format::R8G8B8 | Format::R8G8B8A8 => {
                    convert_dimensions::<u8>(dimension_size, 4, image_contents)
                }
                Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
                    convert_dimensions::<u8>(
                        dimension_size,
                        4,
                        normalize_bytes_slice_type::<u16, u8>(image_contents).as_slice(),
                    )
                }

                Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => convert_dimensions::<u8>(
                    dimension_size,
                    4,
                    normalize_bytes_slice_type::<f32, u8>(image_contents).as_slice(),
                ),
            };
            transfer_buffers.push(
                memory::make_transfer_buffer(
                    context.clone(),
                    image_contents.as_slice(),
                    None,
                    image
                        .name
                        .as_ref()
                        .unwrap_or(&String::from("Unnamed"))
                        .as_str(),
                )
                .unwrap(),
            );
        }
    }
    {
        let mut ctx_write = context.write().unwrap();
        let temp_exec_manager = ctx_write.execution_manager.clone();
        let mut image_cmds = temp_exec_manager
            .on_domain::<phobos::domain::Transfer>()
            .unwrap();
        for (index, image) in document.images.iter().enumerate() {
            let image_data = &images[index];
            let vk_image = phobos::Image::new(
                ctx_write.device.clone(),
                &mut ctx_write.allocator,
                phobos::image::ImageCreateInfo {
                    width: image_data.width,
                    height: image_data.height,
                    depth: 1,
                    usage: vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED,
                    format: match image_data.format {
                        Format::R8 | Format::R16 => vk::Format::R8G8B8A8_UNORM,
                        Format::R8G8 | Format::R16G16 => vk::Format::R8G8B8A8_UNORM,
                        Format::R8G8B8 | Format::R16G16B16 | Format::R32G32B32FLOAT => {
                            vk::Format::R8G8B8A8_UNORM
                        }
                        Format::R8G8B8A8 | Format::R16G16B16A16 | Format::R32G32B32A32FLOAT => {
                            vk::Format::R8G8B8A8_UNORM
                        }
                    },
                    samples: vk::SampleCountFlags::TYPE_1,
                    mip_levels: 1,
                    layers: 1,
                    memory_type: phobos::MemoryType::GpuOnly,
                },
            )
            .unwrap();
            vk_images.push(vk_image);
            let vk_image = vk_images.last().unwrap();

            // Transition layout
            {
                let src_access_mask = vk::AccessFlags2::empty();
                let dst_access_mask = vk::AccessFlags2::TRANSFER_WRITE;
                let source_stage = vk::PipelineStageFlags2::TOP_OF_PIPE;
                let destination_stage = vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;

                let image_barrier = vk::ImageMemoryBarrier2 {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: ptr::null(),
                    src_stage_mask: source_stage,
                    src_access_mask,
                    dst_stage_mask: destination_stage,
                    dst_access_mask,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::READ_ONLY_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: unsafe { vk_image.handle() },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };

                image_cmds = image_cmds.pipeline_barrier(&vk::DependencyInfo {
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

            // Copy data into image
            image_cmds = image_cmds
                .copy_buffer_to_image(
                    &transfer_buffers[index].view_full(),
                    &vk_image.whole_view(vk::ImageAspectFlags::COLOR).unwrap(),
                )
                .unwrap();
            println!(
                "Name: {}, Index image: {} - {}",
                image
                    .name
                    .clone()
                    .unwrap_or(format!("Unnamed image {}", index)),
                index,
                vk_images.len() - 1
            );
        }
        let image_cmds = image_cmds.finish().unwrap();
        temp_exec_manager
            .submit(image_cmds)
            .unwrap()
            .wait()
            .unwrap();
        // Store images into scene
        for (index, image) in vk_images.into_iter().enumerate() {
            ctx_write
                .device
                .set_name(&image, &format!("Unnamed image {}", index))
                .unwrap();
            scene.images.push(
                scene.image_storage.insert(assets::image::Image {
                    name: Some(
                        document.images[index]
                            .name
                            .clone()
                            .unwrap_or(format!("Unnamed image {}", index)),
                    ),
                    image,
                }),
            );
        }
    }
}

fn load_materials_from_primitives(
    context: Arc<RwLock<app::Context>>,
    scene: &mut assets::scene::Scene,
    document: &gltf::json::Root,
    primitives: &mut [structs::GltfPrimitive],
) -> Vec<assets::material::Material> {
    let mut materials: Vec<assets::material::Material> = Vec::new();
    println!("[gltf]: Scene has: {} materials", document.materials.len());

    for primitive in primitives.iter_mut() {
        let get_texture = |index: usize| -> Option<Handle<assets::texture::Texture>> {
            scene.textures.get(index).cloned()
        };
        let primitive_material =
            match primitive.handle.as_ref().unwrap().material {
                None => assets::material::Material {
                    albedo_texture: None,
                    albedo_color: glam::Vec3::new(75f32 / 255f32, 0f32, 130f32 / 255f32),
                },
                Some(index) => {
                    let gltf_material = &document.materials[index.value()];
                    let pbr_material = &gltf_material.pbr_metallic_roughness;
                    println!(
                    "Primitive name: {:?}. Image name: {:?}. Albedo name: {:?}. Texture index: {}.",
                    primitive.name,
                    pbr_material.base_color_texture.as_ref().and_then(|x| {
                        get_texture(x.index.value()).and_then(|x| {
                            scene.texture_storage.get_immutable(&x).and_then(|x| {
                                scene
                                    .image_storage
                                    .get_immutable(&x.image)
                                    .unwrap()
                                    .name
                                    .clone()
                            })
                        })
                    }),
                    pbr_material.base_color_texture.as_ref().and_then(|x| {
                        get_texture(x.index.value()).and_then(|x| {
                            scene.image_storage.get_immutable(
                            &scene.texture_storage.get_immutable(&x).unwrap()
                                .image).unwrap().name.clone()
                        })
                    }),
                    pbr_material
                        .base_color_texture
                        .as_ref()
                        .map(|x| {
                            x.index
                            .value() as i32
                        }).unwrap_or(-1),
                );
                    if pbr_material.base_color_texture.is_some() {
                        assert_eq!(
                            pbr_material.base_color_texture.as_ref().unwrap().tex_coord,
                            0
                        );
                    }
                    assets::material::Material {
                        albedo_texture: pbr_material
                            .base_color_texture
                            .as_ref()
                            .and_then(|x| get_texture(x.index.value())),
                        albedo_color: glam::Vec3::from_slice(&pbr_material.base_color_factor.0),
                    }
                }
            };
        materials.push(primitive_material);
        primitive.material = Some(materials.len() - 1);
    }
    materials
}

/// Converts gltf json format to vk format
fn get_vk_format(
    component_type: gltf::json::accessor::ComponentType,
    component_dimension: gltf::json::accessor::Type,
) -> vk::Format {
    use gltf::json::accessor::{ComponentType, Type};
    match (component_type, component_dimension) {
        (ComponentType::F32, Type::Vec4) => vk::Format::R32G32B32A32_SFLOAT,
        (ComponentType::F32, Type::Vec3) => vk::Format::R32G32B32_SFLOAT,
        (ComponentType::F32, Type::Vec2) => vk::Format::R32G32_SFLOAT,
        (ComponentType::F32, Type::Scalar) => vk::Format::R32_SFLOAT,

        (ComponentType::U32, Type::Vec4) => vk::Format::R32G32B32A32_UINT,
        (ComponentType::U32, Type::Vec3) => vk::Format::R32G32B32_UINT,
        (ComponentType::U32, Type::Vec2) => vk::Format::R32G32_UINT,
        (ComponentType::U32, Type::Scalar) => vk::Format::R32_UINT,

        (ComponentType::U16, Type::Vec4) => vk::Format::R16G16B16A16_UINT,
        (ComponentType::U16, Type::Vec3) => vk::Format::R16G16B16_UINT,
        (ComponentType::U16, Type::Vec2) => vk::Format::R16G16_UINT,
        (ComponentType::U16, Type::Scalar) => vk::Format::R16_UINT,

        (ComponentType::U8, Type::Vec4) => vk::Format::R8G8B8A8_UINT,
        (ComponentType::U8, Type::Vec3) => vk::Format::R8G8B8_UINT,
        (ComponentType::U8, Type::Vec2) => vk::Format::R8G8_UINT,
        (ComponentType::U8, Type::Scalar) => vk::Format::R16_UINT,

        (ComponentType::I16, Type::Vec4) => vk::Format::R16G16B16A16_SINT,
        (ComponentType::I16, Type::Vec3) => vk::Format::R16G16B16_SINT,
        (ComponentType::I16, Type::Vec2) => vk::Format::R16G16_SINT,
        (ComponentType::I16, Type::Scalar) => vk::Format::R16_SINT,

        (ComponentType::I8, Type::Vec4) => vk::Format::R8G8B8A8_SINT,
        (ComponentType::I8, Type::Vec3) => vk::Format::R8G8B8_SINT,
        (ComponentType::I8, Type::Vec2) => vk::Format::R8G8_SINT,
        (ComponentType::I8, Type::Scalar) => vk::Format::R8_SINT,
        _ => vk::Format::UNDEFINED,
    }
}

/// Loads the textures into the scene into the dcument
fn load_texture(scene: &mut assets::scene::Scene, document: &gltf::json::Root) {
    for (index, texture) in document.textures.iter().enumerate() {
        let asset_texture = assets::texture::Texture {
            name: Some(
                texture
                    .name
                    .clone()
                    .unwrap_or(format!("Unnamed texture {}", index)),
            ),
            image: scene.images[texture.source.value()].clone(),
            sampler: scene.samplers[0].clone(),
            format: vk::Format::R8G8B8A8_UNORM,
        };
        scene
            .textures
            .push(scene.texture_storage.insert(asset_texture));
    }
}

/// Get the contents of an accessor
fn get_accessors_contents(
    accessor: &gltf::json::Accessor,
    buffer_view: &gltf::json::buffer::View,
    document: &gltf::json::Root,
    buffers: &[gltf::buffer::Data],
) -> Vec<u8> {
    let total_byte_offset = (accessor.byte_offset + buffer_view.byte_offset.unwrap_or(0)) as usize;
    let entry_size =
        accessor.component_type.unwrap().0.size() * accessor.type_.unwrap().multiplicity();
    let view_stride = buffer_view
        .byte_stride
        .map(|x| x as usize)
        .unwrap_or(entry_size);
    let accessor_length = view_stride * (accessor.count as usize);
    assert!(buffer_view.byte_length as usize >= accessor_length); // Miscalculated Size of accessor
    assert!(accessor_length >= entry_size); // Entry Size is somehow bigger?
    assert_ne!(view_stride, 0); // View stride is not zero

    let buffer = &buffers[buffer_view.buffer.value()];
    let buffer_contents =
        &buffer.0[total_byte_offset..(total_byte_offset + view_stride * (accessor.count as usize))];
    let mut new_contents: Vec<u8> = Vec::with_capacity(entry_size * (accessor.count as usize));

    // Temporary form for parallelization
    let intermediate: Vec<Option<&[u8]>> = buffer_contents
        .chunks_exact(view_stride)
        .map(|x| x.get(0..entry_size))
        .collect();

    for vec in intermediate {
        new_contents.extend_from_slice(vec.unwrap());
    }

    new_contents
}

pub fn gltf_load(
    context: Arc<RwLock<crate::app::Context>>,
    path: std::path::PathBuf,
) -> Result<assets::scene::Scene> {
    let (document, buffers, images) = deserialize(path)?;
    let mut scene = assets::scene::Scene {
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
    for scene in document.nodes.iter() {
        println!("Scene has: {:?}", &scene.matrix);
    }

    let (mut unique_primitives, gltf_meshes) = flatten_primitives(
        document.scenes[document.scene.map(|x| x.value()).unwrap_or(0)]
            .nodes
            .iter()
            .map(|x| &document.nodes[x.value()])
            .collect::<Vec<&gltf::json::Node>>(),
        &document,
    );

    // Iterate over each primitive's accessors and throw its contents into a monolithic buffer
    let (accessors, monolithic_buffer) =
        get_accessors_from_primitives(context.clone(), &mut unique_primitives, &document, &buffers);
    scene
        .buffers
        .push(scene.buffer_storage.insert(monolithic_buffer));

    // Input accessors
    for accessor in accessors.into_iter() {
        scene
            .attributes
            .push(scene.attributes_storage.insert(accessor));
    }
    // Input images
    load_images(context.clone(), &mut scene, &document, images.as_slice());

    // Input textures
    load_texture(&mut scene, &document);

    // Create materials
    let materials = load_materials_from_primitives(
        context.clone(),
        &mut scene,
        &document,
        &mut unique_primitives,
    );
    for material in materials {
        scene
            .materials
            .push(scene.material_storage.insert(material));
    }

    {
        // Make material buffer
        let mut c_materials: Vec<assets::material::CMaterial> = Vec::new();
        for mat in scene.materials.iter() {
            let mat = scene.material_storage.get_immutable(mat).unwrap();
            c_materials.push(mat.to_c_struct(&scene));
        }

        let transfer_material = memory::make_transfer_buffer(
            context.clone(),
            c_materials.as_slice(),
            None,
            "Material buffer",
        )
        .unwrap();
        let material_buffer = memory::copy_buffer_to_gpu_buffer(
            context.clone(),
            transfer_material,
            "Material buffer",
        )
        .unwrap();
        scene.material_buffer = Some(material_buffer);
    }

    // Finally, add the glorious mesh
    {
        for (index, transform) in gltf_meshes {
            let primitive = &unique_primitives[index];
            let get_accessor = |semantic: structs::AccessorSemantic| {
                primitive
                    .accessors
                    .get(&semantic)
                    .and_then(|x| scene.attributes.get(*x).cloned())
            };
            println!(
                "Mesh name: {:?} material index at: {}",
                primitive.name.clone(),
                index
            );
            let asset_mesh = assets::mesh::Mesh {
                name: Some(format!(
                    "{} {}",
                    primitive.name.clone().unwrap_or("Unnamed mesh".parse()?),
                    index
                )),
                vertex_buffer: get_accessor(structs::AccessorSemantic::Gltf(
                    gltf::Semantic::Positions,
                ))
                .unwrap(),
                index_buffer: get_accessor(structs::AccessorSemantic::Index).unwrap(),
                normal_buffer: get_accessor(structs::AccessorSemantic::Gltf(
                    gltf::Semantic::Normals,
                )),
                tangent_buffer: None,
                tex_buffer: get_accessor(structs::AccessorSemantic::Gltf(
                    gltf::Semantic::TexCoords(0),
                )),
                material: scene.materials.get(index).cloned(),
                transform,
            };
            if asset_mesh.tex_buffer.is_some() {
                println!(
                    "Tex buffer used by: {:?}. Index: {:?}. Name: {:?}",
                    primitive.name.clone(),
                    primitive
                        .handle
                        .as_ref()
                        .unwrap()
                        .attributes
                        .get(&gltf::json::validation::Checked::Valid(
                            gltf::Semantic::TexCoords(0)
                        ))
                        .unwrap()
                        .value(),
                    scene
                        .attributes_storage
                        .get_immutable(
                            &get_accessor(structs::AccessorSemantic::Gltf(
                                gltf::Semantic::TexCoords(0),
                            ))
                            .unwrap()
                        )
                        .unwrap()
                        .name
                );
            }

            scene.meshes.push(scene.meshes_storage.insert(asset_mesh));
        }
    }

    Ok(scene)
}
