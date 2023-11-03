use std::ops::Add;
use rayon::prelude::*;
use num_traits;
use num_traits::{Float, PrimInt};

/*
// Automatically generate normals
// TODO: deal with component types!!
pub fn generate_normals<T, U>(indices: &Vec<T>, positions: &Vec<U>) -> Vec<U>
where
	T: PrimInt + Copy + Send + Sync,
	U: Float + Add + Copy + Default + Send + Sync + Into<usize>,
{
	let normals: Vec<U> = vec![U::default(); positions.len()];
	let position_chunked = positions.chunks_exact(3);
	let _ = indices.par_chunks_exact(3)
	               .map(|indices_face| {
		               let v0 = [
			               positions.get(indices_face[0].into()),
			               positions.get(indices_face[0].into() + 1usize),
			               positions.get(indices_face[0].into() + 2usize)
		               ];
		               let v1 = [
			               positions.get(indices_face[1].into()),
			               positions.get(indices_face[1].into() + 1usize),
			               positions.get(indices_face[1].into() + 2usize)
		               ];
		               let v2 = [
			               positions.get(indices_face[2].into()),
			               positions.get(indices_face[2].into() + 1usize),
			               positions.get(indices_face[2].into() + 2usize)
		               ];
	               });

	normals
}
*/
// Automatically generate tangents
pub fn generate_tangents() {

}

