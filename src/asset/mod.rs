pub mod gltf_asset_loader;

use crate::utils::handle_storage::{Handle, Storage};

/// Similar to phobos' [`BufferView`], however it includes additional information about
/// the attributes
///
pub struct AttributeView {
    buffer_view: phobos::BufferView,
    stride: usize,
}

pub struct Mesh {
    vertex_buffer: Handle<AttributeView>,
    index_buffer: Handle<AttributeView>,
}

pub struct Scene {
    meshes_storage: Storage<Mesh>,
    buffer_storage: Storage<phobos::Buffer>,
    attributes_storage: Storage<AttributeView>,

    buffers: Vec<Handle<phobos::Buffer>>,
    meshes: Vec<Handle<Mesh>>,
    attributes: Vec<Handle<AttributeView>>,
}
