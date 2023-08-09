use crate::app;
use crate::assets;
use crate::assets::buffer::Buffer;
use crate::assets::loader::LoadableAsset;
use crate::utils::memory;
use anyhow::{anyhow, Result};
use ash::vk;
use std::sync::{Arc, RwLock};
use std::{fs, path};

pub struct CpuBuffer {
    /// Binary large object
    pub blob: Vec<u8>,

    /// Name
    pub name: Option<String>,
}

pub struct CpuBufferLoadInfo<'a, T> {
    pub name: Option<String>,
    pub view: assets::data_source::DataViewInfo<u64>,
    pub source: assets::data_source::DataSource<'a, T>,
}

impl CpuBuffer {
    /// Uploads data buffer into Vulkan.
    /// MemoryType::GpuOnly will **do a host -> device memory copy**.
    /// If this such behavior is not
    /// desired, refer to [`reserve()`]
    ///
    /// [`reserve()`]: CpuBuffer::reserve
    pub fn upload(
        &self,
        ctx: Arc<RwLock<app::Context>>,
        location: phobos::MemoryType,
        alignment: Option<u64>,
        name: Option<String>,
    ) -> Result<phobos::Buffer> {
        match location {
            phobos::MemoryType::GpuOnly => {
                let transfer_buffer = memory::make_transfer_buffer(
                    ctx.clone(),
                    self.blob.as_slice(),
                    alignment,
                    name.clone()
                        .unwrap_or(String::from("Unnamed transfer buffer"))
                        .as_str(),
                )?;
                memory::copy_buffer_to_gpu_buffer(
                    ctx.clone(),
                    transfer_buffer,
                    name.unwrap_or(String::from("Unnamed buffer")).as_str(),
                )
            }
            phobos::MemoryType::CpuToGpu => memory::make_transfer_buffer(
                ctx.clone(),
                self.blob.as_slice(),
                alignment,
                name.unwrap_or(String::from("Unnamed buffer")).as_str(),
            ),
            phobos::MemoryType::GpuToCpu => panic!("Unsupported location"),
        }
    }

    /// Reserves space for the buffer. **Does not upload blob**
    pub fn reserve(
        &self,
        ctx: &mut app::Context,
        location: phobos::MemoryType,
        name: Option<&str>,
    ) -> Result<phobos::Buffer> {
        match location {
            phobos::MemoryType::GpuOnly => {
                let buffer = phobos::Buffer::new_device_local(
                    ctx.device.clone(),
                    &mut ctx.allocator,
                    self.blob.len() as vk::DeviceSize,
                )?;
                ctx.device
                    .set_name(&buffer, name.unwrap_or("Unnamed buffer"))?;
                Ok(buffer)
            }
            phobos::MemoryType::CpuToGpu => {
                let buffer = phobos::Buffer::new(
                    ctx.device.clone(),
                    &mut ctx.allocator,
                    self.blob.len() as vk::DeviceSize,
                    location,
                )?;
                ctx.device
                    .set_name(&buffer, name.unwrap_or("Unnamed buffer"))?;
                Ok(buffer)
            }
            phobos::MemoryType::GpuToCpu => panic!("Unsupported location!"),
        }
    }
}

impl<'a> LoadableAsset<'a> for CpuBuffer {
    type LoadInfo = CpuBufferLoadInfo<'a, u8>;

    /// Loads in data given a [`LoadInfo`]
    ///
    /// [`LoadInfo`]: CpuBufferLoadInfo
    fn load(info: Self::LoadInfo) -> Result<Self>
    where
        Self: Sized,
    {
        match info.source {
            assets::data_source::DataSource::FromFile {
                path,
                cpu_postprocess,
            } => {
                let offset = info.view.offset as usize;
                let size = info.view.length;
                let mut file_contents = fs::read(path)?;
                let file_range =
                    offset..(offset + size.map(|x| x as usize).unwrap_or(file_contents.len()));
                if let Some(cpu_postprocess) = cpu_postprocess {
                    file_contents = cpu_postprocess(file_contents);
                }
                return Ok(CpuBuffer {
                    blob: file_contents
                        .get(file_range)
                        .map(|x| x.to_vec())
                        .ok_or(anyhow!("Error: range not in file_contents"))?,
                    name: info.name,
                });
            }
            assets::data_source::DataSource::FromSlice { slice } => {
                // No real need for this yet
                // Should make this into a callback
                return Ok(CpuBuffer {
                    blob: slice.to_vec(),
                    name: info.name,
                });
            }
        }
    }
}

impl Buffer for CpuBuffer {}
