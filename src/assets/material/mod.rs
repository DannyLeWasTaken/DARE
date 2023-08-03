use crate::assets;
use crate::assets::Asset;
use crate::utils::handle_storage::Handle;

/// Represents the materials
#[derive(Clone, PartialEq)]
pub struct Material {
    pub albedo_texture: Option<Handle<assets::texture::Texture>>,
    pub albedo_color: glam::Vec3,
}

impl Eq for Material {}
impl Asset for Material {}

impl Material {
    pub fn to_c_struct(&self, scene: &assets::scene::Scene) -> CMaterial {
        let get_index = |texture: &Option<Handle<assets::texture::Texture>>| -> i32 {
            texture
                .as_ref()
                .map(|texture: &Handle<assets::texture::Texture>| {
                    scene
                        .textures
                        .iter()
                        .position(|r| r == texture)
                        .map(|x| x as i32)
                        .unwrap_or(-1)
                })
                .unwrap_or(-1)
        };
        let material = CMaterial {
            albedo_texture: get_index(&self.albedo_texture),
            albedo_color: self.albedo_color.to_array(),
        };
        material
    }
}

/// Represents the material c-struct for use in shaders
#[repr(C)]
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub struct CMaterial {
    pub albedo_texture: i32,
    pub albedo_color: [f32; 3],
}
