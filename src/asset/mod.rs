pub mod gltf_asset_loader;

use crate::utils::handle_storage::{Handle, Storage};
use std::collections::HashMap;
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
#[derive(Clone)]
pub struct Mesh {
    pub name: Option<String>,
    pub vertex_buffer: Handle<AttributeView>,
    pub index_buffer: Handle<AttributeView>,
    pub normal_buffer: Option<Handle<AttributeView>>,
    pub tangent_buffer: Option<Handle<AttributeView>>,
    pub tex_buffer: Option<Handle<AttributeView>>,
    pub material: i32,
    pub transform: glam::Mat4,
}

/// C-like representation of the mesh mainly for use in shader
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CMesh {
    pub vertex_buffer: u64,
    pub index_buffer: u64,
    pub normal_buffer: u64,
    pub tex_buffer: u64,
    pub material: i32,
}

impl Mesh {
    pub fn to_c_struct(&self, scene: &Scene) -> CMesh {
        CMesh {
            vertex_buffer: scene
                .attributes_storage
                .get_immutable(&self.vertex_buffer)
                .and_then(|x| {
                    return Some(x.buffer_view.address());
                })
                .unwrap_or(0u64),
            index_buffer: scene
                .attributes_storage
                .get_immutable(&self.index_buffer)
                .and_then(|x| {
                    return Some(x.buffer_view.address() as u64);
                })
                .unwrap_or(0u64),
            normal_buffer: self
                .normal_buffer
                .and_then(|buffer| scene.attributes_storage.get_immutable(&buffer))
                .map(|x| x.buffer_view.address())
                .unwrap_or(0u64),
            tex_buffer: self
                .tex_buffer
                .and_then(|buffer| scene.attributes_storage.get_immutable(&buffer))
                .map(|x| x.buffer_view.address())
                .unwrap_or(0u64),
            material: self.material,
        }
    }
}

#[derive(Clone)]
pub struct Material {
    pub albedo_texture: Option<Handle<Texture>>,
    pub albedo_color: glam::Vec3,
}

impl Material {
    /// Returns the c compatible version of the material
    pub fn to_c_struct(&self, scene: &Scene) -> CMaterial {
        CMaterial {
            albedo_texture: self
                .albedo_texture
                .as_ref()
                .map(|texture| {
                    if let Some(index) = scene.textures.iter().position(|x| x == texture) {
                        return index as i32;
                    } else {
                        return -1;
                    }
                })
                .unwrap_or(-1),
            albedo: self.albedo_color.to_array(),
        }
    }
}

/// Version of material but for C
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CMaterial {
    pub albedo_texture: i32,
    pub albedo: [f32; 3],
}

/// Abstraction for textures including their source + sampler
#[derive(Clone, PartialEq, Eq)]
pub struct Texture {
    pub name: Option<String>,
    pub source: Handle<phobos::Image>,
    pub sampler: Handle<phobos::Sampler>,
}

pub struct Scene {
    pub meshes_storage: Storage<Mesh>,
    pub buffer_storage: Storage<phobos::Buffer>,
    pub attributes_storage: Storage<AttributeView>,
    pub image_storage: Storage<phobos::Image>,
    pub sampler_storage: Storage<phobos::Sampler>,
    pub texture_storage: Storage<Texture>,
    pub material_storage: Storage<Material>,

    pub buffers: Vec<Handle<phobos::Buffer>>,
    pub images: Vec<Handle<phobos::Image>>,
    pub samplers: Vec<Handle<phobos::Sampler>>,
    pub meshes: Vec<Handle<Mesh>>,
    pub attributes: Vec<Handle<AttributeView>>,
    pub textures: Vec<Handle<Texture>>,
    pub materials: Vec<Handle<Material>>,
}
