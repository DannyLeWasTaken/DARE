use crate::assets;
use crate::utils::handle_storage::{Handle, Storage};
use phobos::vk;

pub struct TextureLoadInfo<'a> {
    /// Name of the texture
    pub name: Option<String>,

    /// Storage of where all images are held
    pub image_storage: &'a Storage<assets::image::Image>,

    /// Image handle
    pub image_handle: Handle<assets::image::Image>,

    /// Sampler storage
    pub sampler_storage: &'a Storage<phobos::Sampler>,

    /// Sampler handle
    pub image_sampler: Handle<phobos::Sampler>,

    /// Mip map
    pub mip_mapping: Option<u32>,
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Texture {
    pub name: Option<String>,
    pub image: Handle<assets::image::Image>,
    pub sampler: Handle<phobos::Sampler>,
    pub format: vk::Format,
}

impl assets::Asset for Texture {}

impl<'a> assets::loader::LoadableAsset<'a> for Texture {
    type LoadInfo = TextureLoadInfo<'a>;

    fn load(info: Self::LoadInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            name: info.name,
            image: info.image_handle,
            sampler: info.image_sampler,
            format: Default::default(),
        })
    }
}
