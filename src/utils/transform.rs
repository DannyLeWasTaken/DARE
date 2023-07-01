//! Helper functions for glam

use phobos::vk;

pub fn glam_to_vulkan(mat: glam::Mat4) {
    let mat = mat.transpose();
    let (scale_rotation, translation) = mat.to_scale_rotation_translation();
}
