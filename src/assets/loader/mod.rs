//! Everything regarding loader for assets
use anyhow::Result;

pub trait LoadableAsset<'a> {
    type LoadInfo: 'a;

    fn load(info: Self::LoadInfo) -> Result<Self>
    where
        Self: Sized;
}
