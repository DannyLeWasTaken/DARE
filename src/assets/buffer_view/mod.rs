pub mod attribute_view;
pub use attribute_view::AttributeView;

use crate::assets;
use crate::assets::Asset;
use anyhow::Result;
use ash::vk;
use std::marker::PhantomData;

pub trait BufferView {
    /// Offset into the buffer in bytes
    type Offset: Into<u64>;

    /// Size into the buffer in bytes
    type Size: Into<u64>;
}

/// A view into any given [`Buffer`]
///
/// [`Buffer`]: crate::assets::buffer::Buffer
pub struct BufferViewCpu<'a, T>
where
    T: Sized + Copy,
{
    pub handle: Option<&'a [T]>,
}

impl<'a, T: Copy> BufferView for BufferViewCpu<'a, T> {
    type Offset = u64;
    type Size = u64;
}
impl<'a, T: Copy> Asset for BufferViewCpu<'a, T> {}

/// Vulkan buffer view into a [`VulkenBuffer`]
///
/// [`VulkanBuffer`]: VulkanBuffer
#[derive(Clone, Eq, PartialEq)]
pub struct BufferViewVulkan<T: Sized> {
    /// Handle of the buffer view
    pub handle: phobos::BufferView,

    /// Phantom marker to ensure type safety
    pub _marker: PhantomData<T>,
}

impl<T: Sized> BufferViewVulkan<T> {
    /// Get a view into a given buffer
    fn new<A: Into<u64>>(
        buffer: &phobos::Buffer,
        view: assets::data_source::DataViewInfo<u64>,
    ) -> Result<Self> {
        match view.length {
            None => Ok(Self {
                handle: buffer.view_full(),
                _marker: PhantomData::default(),
            }),
            Some(size) => Ok(Self {
                handle: buffer.view(view.offset, size)?,
                _marker: PhantomData::default(),
            }),
        }
    }
}

impl<T: Sized> BufferView for BufferViewVulkan<T> {
    type Offset = u64;
    type Size = u64;
}
impl<T> Asset for BufferViewVulkan<T> {}
