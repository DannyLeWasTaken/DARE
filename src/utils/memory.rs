//! Memory utility functions

use anyhow::Result;

use phobos::vk;

/// Gets the total size that any given slice of data would take up
pub fn get_size<T: Copy>(data: &[T]) -> u64 {
    (data.len() * std::mem::size_of::<T>()) as u64
}

/// Quick utility function to create transfer buffers mainly for input
/// https://github.com/NotAPenguin0/phobos-rs/blob/master/examples/03_raytracing/main.rs
pub fn make_transfer_buffer<T: Copy>(
    ctx: &mut crate::app::Context,
    data: &[T],
    usage: vk::BufferUsageFlags,
    alignment: Option<u64>,
    name: &str,
) -> Result<phobos::Buffer> {
    let buffer = match alignment {
        None => phobos::Buffer::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            (data.len() * std::mem::size_of::<T>()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::TRANSFER_DST
                | usage,
            phobos::MemoryType::CpuToGpu,
        )?,
        Some(alignment) => phobos::Buffer::new_aligned(
            ctx.device.clone(),
            &mut ctx.allocator,
            (data.len() * std::mem::size_of::<T>()) as u64,
            alignment,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::TRANSFER_DST
                | usage,
            phobos::MemoryType::CpuToGpu,
        )?,
    };
    buffer
        .view_full()
        .mapped_slice::<T>()?
        .copy_from_slice(data);
    ctx.device.set_name(&buffer, name)?;
    Ok(buffer)
}
