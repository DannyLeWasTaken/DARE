//! Memory utility functions

use anyhow::Result;

use phobos::vk;

/// Gets the total size that any given slice of data would take up
pub fn get_size<T: Copy>(data: &[T]) -> u64 {
    std::mem::size_of_val(data) as u64
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
            get_size(&data),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::TRANSFER_DST
                | usage,
            phobos::MemoryType::CpuToGpu,
        )?,
        Some(alignment) => phobos::Buffer::new_aligned(
            ctx.device.clone(),
            &mut ctx.allocator,
            get_size(&data),
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

pub fn vector_to_array<T: Clone + Copy, const N: usize>(v: Vec<T>) -> Option<[T; N]> {
    if v.len() == N {
        let mut arr = [v[0].clone(); N];
        arr.clone_from_slice(&v);
        Some(arr)
    } else {
        None
    }
}

use std::ops::{Add, Rem, Sub};

/// Aligns any given size and alignment to the correct size accounting for alignment
pub fn align_size<T: Add<Output = T> + Sub<Output = T> + Rem<Output = T> + Copy>(
    size: T,
    alignment: T,
) -> T {
    size + alignment - (size % alignment)
}

pub fn bytes_to_mib(size: f64) -> f64 {
    return size / (1024.0 * 1024.0);
}
