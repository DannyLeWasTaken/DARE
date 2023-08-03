use crate::assets;
use crate::assets::buffer_view;
use crate::utils::handle_storage::Handle;
use ash::vk;
use std::sync::{Arc, Weak};

/// Similar to phobos' [`BufferView`] or [`BufferViewVulkan`], however it includes
/// additional information about the attributes
///
/// [`BufferView`]: phobos::BufferView
/// [`BufferViewVulkan`]: BufferViewVulkan
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct AttributeView<T: Sized> {
    pub buffer_view: phobos::BufferView,

    /// Stride information
    pub stride: u64,

    /// Format of the attribute
    pub format: vk::Format,

    /// Number of components in the attribute
    pub count: u64,

    /// Size of the component
    pub component_size: u64,

    /// Type marker for type-safety
    pub _marker: std::marker::PhantomData<T>,

    /// Size of the attribute view
    pub size: u64,

    /// Size of the offset
    pub offset: u64,
}

impl<T: Sized> buffer_view::BufferView for AttributeView<T> {
    type Offset = u64;
    type Size = u64;
}
impl<T> assets::Asset for AttributeView<T> {}
