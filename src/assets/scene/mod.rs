//! Representation of the scene
pub mod loader;
pub use loader::SceneLoadInfo;

use crate::assets;
use crate::assets::buffer_view::attribute_view::AttributeView;
use crate::assets::mesh::Mesh;
use crate::utils::handle_storage::{Handle, Storage};

pub struct Scene {
    pub meshes_storage: Storage<Mesh>,
    pub buffer_storage: Storage<phobos::Buffer>,
    pub attributes_storage: Storage<AttributeView<u8>>,
    pub image_storage: Storage<assets::image::Image>,
    pub sampler_storage: Storage<phobos::Sampler>,
    pub texture_storage: Storage<assets::texture::Texture>,
    pub material_storage: Storage<assets::material::Material>,

    pub buffers: Vec<Handle<phobos::Buffer>>,
    pub images: Vec<Handle<assets::image::Image>>,
    pub samplers: Vec<Handle<phobos::Sampler>>,
    pub meshes: Vec<Handle<Mesh>>,
    pub attributes: Vec<Handle<AttributeView<u8>>>,
    pub textures: Vec<Handle<assets::texture::Texture>>,
    pub materials: Vec<Handle<assets::material::Material>>,
    pub material_buffer: Option<phobos::Buffer>,
}
