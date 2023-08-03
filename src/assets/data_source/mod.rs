use std::ops::{Add, AddAssign};

/// Struct to describe the view
#[derive(Clone, Ord, Hash, PartialOrd, Eq, PartialEq)]
pub struct DataViewInfo<T: Into<u64> + Copy> {
    pub offset: T,
    pub length: Option<T>,
}

impl<T: Into<u64> + Copy + Add<Output = T>> DataViewInfo<T> {
    /// Get the end Offset length
    pub fn end(&self) -> T {
        self.offset + self.length.unwrap()
    }
}

/// Describes where the data is being sourced from
#[derive(Clone, Ord, Hash, PartialOrd, Eq, PartialEq)]
pub enum DataSource<'a, T> {
    /// Data is generated from a file
    FromFile {
        /// Path to the file
        path: std::path::PathBuf,

        /// Callback function
        cpu_postprocess: Option<fn(Vec<u8>) -> Vec<u8>>,
    },
    /// Data is generated from the CPU
    FromSlice {
        /// Callback function to get the generated data on the cpu
        slice: &'a [T],
    },
}
