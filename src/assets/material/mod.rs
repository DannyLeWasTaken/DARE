use crate::assets;
use crate::assets::Asset;
use crate::utils::handle_storage::Handle;

/// Represents the materials
#[derive(Clone, PartialEq)]
pub struct Material {
    pub albedo_texture: Option<Handle<assets::texture::Texture>>,
    pub albedo_color: glam::Vec3,
    pub normal_texture: Option<Handle<assets::texture::Texture>>,
    pub emissive_texture: Option<Handle<assets::texture::Texture>>,
    pub emissive_factor: glam::Vec3,
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
                        .texture_storage
                        .get_immutable(texture)
                        .map(|x| {
                            scene
                                .images
                                .iter()
                                .position(|p| *p == x.image)
                                .map(|x| x as i32)
                                .unwrap_or(-1i32)
                        })
                        .unwrap_or(-1i32)
                })
                .unwrap_or(-1i32)
        };
        let merge = |value: Option<glam::Vec3>, index: i32| -> [f32; 4] {
            let mut vec4 = [0.0; 4];
            vec4[0..3].copy_from_slice(&value.map(|x| x.to_array()).unwrap_or([0f32, 0f32, 0f32]));
            vec4[3] = index as f32;
            vec4
        };
        let material = CMaterial {
            albedo: merge(Some(self.albedo_color), get_index(&self.albedo_texture)),
            normal: merge(None, get_index(&self.normal_texture)),
            emissive: merge(
                Some(self.emissive_factor),
                get_index(&self.emissive_texture),
            ),
        };
        //println!("I think normal is: {}", material.normal[3]);
        material
    }
}

/// Represents the material c-struct for use in shaders
#[repr(C, align(16))]
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub struct CMaterial {
    pub albedo: [f32; 4],
    pub normal: [f32; 4],
    pub emissive: [f32; 4],
}
// Ensure that CMaterial is compatible with bytemuck
unsafe impl bytemuck::Pod for CMaterial {}
unsafe impl bytemuck::Zeroable for CMaterial {}
