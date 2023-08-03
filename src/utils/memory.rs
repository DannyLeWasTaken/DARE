//! Memory utility functions

use anyhow::Result;

use phobos::{vk, IncompleteCmdBuffer, TransferCmdBuffer};

/// Gets the total Size that any given slice of data would take up
pub fn get_size<T: Copy>(data: &[T]) -> u64 {
    std::mem::size_of_val(data) as u64
}

/// Quick utility function to create transfer buffers mainly for input
/// https://github.com/NotAPenguin0/phobos-rs/blob/master/examples/03_raytracing/main.rs
pub fn make_transfer_buffer<T: Copy>(
    ctx: Arc<RwLock<crate::app::Context>>,
    data: &[T],
    alignment: Option<u64>,
    name: &str,
) -> Result<phobos::Buffer> {
    let mut ctx = ctx.write().unwrap();
    let buffer = match alignment {
        None => phobos::Buffer::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            get_size(&data),
            phobos::MemoryType::CpuToGpu,
        )?,
        Some(alignment) => phobos::Buffer::new_aligned(
            ctx.device.clone(),
            &mut ctx.allocator,
            get_size(data),
            alignment,
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

/// Copies any buffer to the GPU buffer
pub fn copy_buffer_to_gpu_buffer(
    ctx: Arc<RwLock<crate::app::Context>>,
    in_buffer: phobos::Buffer,
    name: &str,
) -> Result<phobos::Buffer> {
    // Create a new buffer that is on the gpu only
    let mut ctx = ctx.write().unwrap();
    let gpu_buffer =
        phobos::Buffer::new_device_local(ctx.device.clone(), &mut ctx.allocator, in_buffer.size())?;
    ctx.device.set_name(&gpu_buffer, name).unwrap();

    let cmds = ctx
        .execution_manager
        .on_domain::<phobos::domain::Compute>()?
        .copy_buffer(&in_buffer.view_full(), &gpu_buffer.view_full())?
        .finish()?;
    ctx.execution_manager.submit(cmds)?.wait()?;

    Ok(gpu_buffer)
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
use std::sync::{Arc, RwLock};

/// Aligns any given Size and alignment to the correct Size accounting for alignment
pub fn align_size<T: Add<Output = T> + Sub<Output = T> + Rem<Output = T> + Copy>(
    size: T,
    alignment: T,
) -> T {
    size + alignment - (size % alignment)
}

pub fn bytes_to_mib(size: f64) -> f64 {
    return size / (1024.0 * 1024.0);
}
