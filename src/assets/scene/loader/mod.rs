mod gltf;
mod utility;

use crate::assets;
use anyhow::Result;
use std::sync::{Arc, RwLock};

pub enum SceneLoadInfo {
    /// Loads in a scene assuming it is a gltf file
    gltf {
        context: Arc<RwLock<crate::app::Context>>,
        path: std::path::PathBuf,
    },
}

impl<'a> assets::loader::LoadableAsset<'a> for assets::scene::Scene {
    type LoadInfo = SceneLoadInfo;

    fn load(info: Self::LoadInfo) -> Result<Self>
    where
        Self: Sized,
    {
        match info {
            SceneLoadInfo::gltf { context, path } => gltf::gltf_load(context, path),
        }
    }
}
