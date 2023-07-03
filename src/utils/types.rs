//! Utility function to help convert between types

use phobos;
use phobos::vk;

/// Convert any scalar type to an index type as long as it is an unsigned integer scalar
pub fn convert_scalar_format_to_index(
    scalar_type: phobos::vk::Format,
) -> Option<phobos::vk::IndexType> {
    match scalar_type {
        phobos::vk::Format::R32_UINT => Some(phobos::vk::IndexType::UINT32),
        phobos::vk::Format::R16_UINT => Some(phobos::vk::IndexType::UINT16),
        phobos::vk::Format::R8_UINT => Some(phobos::vk::IndexType::UINT8_EXT),
        _ => None,
    }
}
