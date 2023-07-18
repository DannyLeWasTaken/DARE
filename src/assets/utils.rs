//! Internal utility functions that may commonly come up when loading assets
use glam;

/// Calculate normal buffer from vertex + index buffer. **Only accepts triangles**
/// # Fail conditions
/// - If # of indices is not a multiple of 3
/// - Indices are not lined up with vertices
pub fn calculate_normals(vertices: &Vec<glam::Vec3>, indices: &Vec<u32>) -> Vec<glam::Vec3> {
    let mut normals: Vec<glam::Vec3> = vec![glam::Vec3::ZERO; vertices.len()];
    assert_eq!(indices.len() % 3, 0); // Only accept triangles
    let _ = indices.chunks(3).map(|triangle| {
        let i0 = triangle[0] as usize;
        let i1 = triangle[1] as usize;
        let i2 = triangle[2] as usize;
        let v0 = vertices.get(i0).unwrap();
        let v1 = vertices.get(i1).unwrap();
        let v2 = vertices.get(i2).unwrap();
        let n = glam::Vec3::normalize(glam::Vec3::cross(
            *v1 - *v0, *v2 - *v0
        ));
        normals[i0] += n;
        normals[i1] += n;
        normals[i2] += n;
    });

    normals.into_iter().map(glam::Vec3::normalize).collect() 
}
