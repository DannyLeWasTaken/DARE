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

    pub diffuse_factor: glam::Vec4,
    pub diffuse_texture: Option<Handle<assets::texture::Texture>>,
    pub specular_factor: glam::Vec3,
    pub glossiness_factor: f32,
    pub specular_glossiness_texture: Option<Handle<assets::texture::Texture>>,

    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<Handle<assets::texture::Texture>>,
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
        let merge = |value: Option<glam::Vec3>, index: f32| -> [f32; 4] {
            let mut vec4 = [0.0; 4];
            vec4[0..3].copy_from_slice(&value.map(|x| x.to_array()).unwrap_or([0f32, 0f32, 0f32]));
            vec4[3] = index;
            vec4
        };
        let material = CMaterial {
            albedo: merge(
                Some(self.albedo_color),
                get_index(&self.albedo_texture) as f32,
            ),
            normal: merge(None, get_index(&self.normal_texture) as f32),
            emissive: merge(
                Some(self.emissive_factor),
                get_index(&self.emissive_texture) as f32,
            ),
            diffuse_factor: self.diffuse_factor.to_array(),
            specular_glossiness_factor: merge(Some(self.specular_factor), self.glossiness_factor),
            specular_glossiness_diffuse_texture: [
                get_index(&self.specular_glossiness_texture) as f32,
                get_index(&self.diffuse_texture) as f32,
                0f32,
                0f32,
            ],
            metallic_roughness: [
                get_index(&self.metallic_roughness_texture) as f32,
                self.roughness_factor,
                self.metallic_factor,
                0f32,
            ],
        };
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
    pub diffuse_factor: [f32; 4],
    pub specular_glossiness_factor: [f32; 4], // rgb -> specular factor, a -> glossiness factor
    pub specular_glossiness_diffuse_texture: [f32; 4], // r channel is only used sadly :(
    pub metallic_roughness: [f32; 4],         // r -> texture, g -> roughness. b -> metallic
}
// Ensure that CMaterial is compatible with bytemuck
unsafe impl bytemuck::Pod for CMaterial {}
unsafe impl bytemuck::Zeroable for CMaterial {}
