pub mod gltf_asset_loader;

use crate::utils::handle_storage::{Handle, Storage};
use std::sync::Arc;

/// Similar to phobos' [`BufferView`], however it includes additional information about
/// the attributes
///
/// [`BufferView`]: phobos::BufferView
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct AttributeView {
    pub buffer_view: phobos::BufferView,

    /// Stride information
    pub stride: u64,

    /// Format of the attribute
    pub format: phobos::vk::Format,

    /// Number of components in the attribute
    pub count: u64,

    /// Size of the component
    pub component_size: u64,
}

/// An object that holds information about the Mesh object in a scene and its' resource
/// [`handles`]
///
/// [`handles`]: Handle
#[derive(Copy, Clone, PartialEq)]
pub struct Mesh {
    pub vertex_buffer: Handle<AttributeView>,
    pub index_buffer: Handle<AttributeView>,
    pub transform: glam::Mat4,
}

pub struct Scene {
    pub meshes_storage: Storage<Mesh>,
    pub buffer_storage: Storage<phobos::Buffer>,
    pub attributes_storage: Storage<AttributeView>,

    pub buffers: Vec<Handle<phobos::Buffer>>,
    pub meshes: Vec<Handle<Mesh>>,
    pub attributes: Vec<Handle<AttributeView>>,
}
