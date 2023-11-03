//! Representation of a mesh

use crate::assets;
use crate::utils::handle_storage::Handle;

/// An object that holds information about the Mesh object in a scene and its' resources
/// [`handles`]
///
/// [`handles`]: Handle
#[derive(Clone, PartialEq)]
pub struct Mesh {
    pub name: Option<String>,
    pub vertex_buffer: Handle<assets::buffer_view::AttributeView<u8>>,
    pub index_buffer: Handle<assets::buffer_view::AttributeView<u8>>,
    pub normal_buffer: Option<Handle<assets::buffer_view::AttributeView<u8>>>,
    pub tangent_buffer: Option<Handle<assets::buffer_view::AttributeView<u8>>>,
    pub tex_buffer: Option<Handle<assets::buffer_view::AttributeView<u8>>>,
    pub material: Option<Handle<assets::material::Material>>,
    pub transform: glam::Mat4,
}

/// C-like representation of the mesh mainly for use in shader
#[repr(C, align(4))]
#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct CMesh {
    pub vertex_buffer: u64,
    pub index_buffer: u64,
    pub normal_buffer: u64,
    pub tangent_buffer: u64,
    pub tex_buffer: u64,
    pub material: i32,
}

impl Mesh {
    pub fn to_c_struct(&self, scene: &assets::scene::Scene) -> CMesh {
        let mesh = CMesh {
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
                .clone()
                .and_then(|buffer| scene.attributes_storage.get_immutable(&buffer))
                .map(|x| x.buffer_view.address())
                .unwrap_or(0u64),
            tangent_buffer: self
                .tangent_buffer
                .clone()
                .and_then(|buffer| scene.attributes_storage.get_immutable(&buffer))
                .map(|x| x.buffer_view.address())
                .unwrap_or(0u64),
            tex_buffer: self
                .tex_buffer
                .clone()
                .and_then(|buffer| scene.attributes_storage.get_immutable(&buffer))
                .map(|x| x.buffer_view.address())
                .unwrap_or(0u64),
            material: self
                .material
                .as_ref()
                .and_then(|x| {
                    scene
                        .materials
                        .iter()
                        .position(|y| y == x)
                        .map(|x| x as i32)
                })
                .unwrap_or(-1),
        };
        mesh
    }
}

impl Eq for Mesh {}
