use crate::assets;
use crate::assets::data_source::{DataSource, DataViewInfo};
use anyhow::Result;
use ash::vk;
use phobos::{IncompleteCmdBuffer, TransferCmdBuffer};
use std::sync::{Arc, RwLock};

pub enum ImageViewInfo {
    full(vk::ImageAspectFlags),
    view_info(phobos::image::ImageViewCreateInfo),
}

pub enum ImageLoadInfo<'a> {
    FromFile {
        context: Arc<RwLock<crate::app::Context>>,
        name: Option<String>,
        buffer_source: DataSource<'a, u8>,
        buffer_view: DataViewInfo<u64>,
        image_create_info: phobos::image::ImageCreateInfo,
        image_view_info: ImageViewInfo,
    },
    FromImageBufferView {
        context: Arc<RwLock<crate::app::Context>>,
        name: Option<String>,
        buffer_view: phobos::BufferView,
        image: phobos::Image,
        image_view: phobos::ImageView,
    },
}

pub struct Image {
    pub name: Option<String>,
    pub image: phobos::Image,
}

impl<'a> assets::loader::LoadableAsset<'a> for Image {
    type LoadInfo = ImageLoadInfo<'a>;

    fn load(info: Self::LoadInfo) -> Result<Self>
    where
        Self: Sized,
    {
        match info {
            ImageLoadInfo::FromFile {
                context,
                name,
                buffer_source,
                buffer_view,
                image_view_info,
                image_create_info,
            } => match &buffer_source {
                DataSource::FromFile { .. } | DataSource::FromSlice { .. } => {
                    let cpu_buffer =
                        assets::buffer::CpuBuffer::load(assets::buffer::CpuBufferLoadInfo {
                            name: name.clone(),
                            view: buffer_view,
                            source: buffer_source.clone(),
                        })
                        .unwrap();
                    let buffer_view = cpu_buffer
                        .upload(
                            context.clone(),
                            phobos::MemoryType::CpuToGpu,
                            None,
                            name.clone(),
                        )?
                        .view_full();
                    let mut ctx_write = context.write().unwrap();
                    let image = phobos::Image::new(
                        ctx_write.device.clone(),
                        &mut ctx_write.allocator,
                        image_create_info,
                    )?;
                    let image_view: phobos::ImageView = match image_view_info {
                        ImageViewInfo::full(flag) => image.whole_view(flag)?,
                        ImageViewInfo::view_info(image_view_info) => image.view(image_view_info)?,
                    };
                    let cmd = ctx_write
                        .execution_manager
                        .on_domain::<phobos::domain::Transfer>()?
                        .copy_buffer_to_image(&buffer_view, &image_view)?
                        .finish()?;
                    ctx_write.execution_manager.submit(cmd)?.wait()?;
                    Ok(Image { name, image })
                }
            },
            ImageLoadInfo::FromImageBufferView {
                context,
                name,
                buffer_view,
                image,
                image_view,
            } => {
                let ctx_write = context.write().unwrap();
                let cmd = ctx_write
                    .execution_manager
                    .on_domain::<phobos::domain::Transfer>()?
                    .copy_buffer_to_image(&buffer_view, &image_view)?
                    .finish()?;
                ctx_write.execution_manager.submit(cmd)?.wait()?;
                Ok(Image { name, image })
            }
        }
    }
}

/*
impl assets::loader::LoadableAsset for Texture {
    type LoadInfo = TextureLoadInfo;

    fn load(ctx: &mut crate::app::Context, info: Self::LoadInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        // Load image from buffer
        let load_info = assets::buffer::cpu_buffer::CpuBufferLoadInfo::FromDisk {
            path: info.path,
            Offset: 0,
            Size: None,
        };
        let buffer = assets::buffer::cpu_buffer::CpuBuffer::load(load_info)?;
        if buffer.blob.is_none() {
            return Err(anyhow!("No blob"));
        }
        // Process image
        phobos::Image::new(
            ctx.device.clone(),
            phobos::image::ImageCreateInfo {
                width: 0,
                height: 0,
                depth: 0,
                usage: Default::default(),
                format: Default::default(),
                samples: Default::default(),
                mip_levels: 0,
                layers: 0,
            }
        );

        todo!()
    }
}
*/
