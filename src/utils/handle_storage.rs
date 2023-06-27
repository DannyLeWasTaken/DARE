use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

static ATOMIC_STORAGE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Handle to abstract away data that it represents
///
/// # Validity
/// When we say [`Handle<T>`] is valid, it means that there is still an active
/// corresponding data that the [`Handle<T>`] represents in the [`Handle<T>`]'s
/// [`Storage<T>`] that it belongs to.\
/// \
/// If something is not valid, it means that either:
/// - The handle was given to the wrong [`Storage<T>`] that it didn't belong to
/// - [`Handle<T>`]'s underlying data has been removed from [`Storage<T>`]
///
/// [`Handle<T>`]: Handle
/// [`Storage<T>`]: Storage
#[derive(Hash, PartialEq, Eq)]
pub struct Handle<T> {
	/// Index in the `Storage<T>`'s `storage` HashMap.
	index: usize,

	/// Unique id of the handle to help validate if the handle is outdated or not
	handle_id: usize,

	/// Storage ID that the handle corresponds to
	storage_id: usize,

	// Phantom marker to help during compile time for lifetimes and type-safety
	_marker: std::marker::PhantomData<*const T>,
}

impl<T> Clone for Handle<T> {
	fn clone(&self) -> Self {
		Handle {
			index: self.index,
			handle_id: self.handle_id,
			storage_id: self.storage_id,
			_marker: self._marker.clone(),
		}
	}
}

/// Storage for all the handles\
/// Responsible for handling, removing, and modifying all handles.
///
pub struct Storage<T> {
	/// HashMap containing K: `usize` and value: `T` of the storage.
	/// Effectively holds storage of `T`
	storage: Arc<RwLock<HashMap<usize, T>>>,

	/// Contains all storage handles including active and inactive.
	/// Used to check if any incoming handles for methods are still active
	/// or invalid
	handles: Arc<RwLock<HashMap<usize, Handle<T>>>>,

	/// Responsible for ensuring handles are kept up to date. If
	/// a [`Handle<T>`]'s `handle_id` does not correspond to the same one found
	/// in [`Handle<T>`], the handle is considered invalid
	///
	/// [`Handle<T>`]: Handle
	handle_id: Arc<RwLock<usize>>,

	/// Unique identifier for [`Storage<T>`] to allow it to distinguish, it's
	/// handles from other [`Storage<T>`]s.
	///
	/// [`Storage<T>`]: Storage
	storage_id: Arc<RwLock<usize>>,
}

impl<T> Storage<T> {

	/// Create a new storage class
	///
	pub fn new() -> Self {
		let id = ATOMIC_STORAGE_COUNTER.fetch_add(1, Ordering::SeqCst);
		Self {
			storage: Arc::new(RwLock::new(HashMap::new())),
			handles: Arc::new(RwLock::new(HashMap::new())),
			handle_id: Arc::new(RwLock::new(0)),
			storage_id: Arc::new(RwLock::new(id)),
		}
	}

	/// Transfer management of the `data` to [`Storage<T>`]
	///
	/// # Arguments
	///
	/// * `data` - Data to be transferred over to [`Storage<T>`]
	///
	/// ['Storage<T>']: Storage
	pub fn insert(&self, data: T) -> Handle<T> {
		let mut handle_id = self.handle_id.write().unwrap();
		let mut storage = self.storage.write().unwrap();
		let mut handles = self.handles.write().unwrap();
		let next_index = storage.len() + 1;
		*handle_id += 1;

		// Create new handle
		let handle = Handle {
			index: next_index,
			handle_id: *handle_id,
			storage_id: *self.storage_id.read().unwrap(),
			_marker: std::marker::PhantomData,
		};

		// Insert handles
		storage.insert(next_index, data);
		handles.insert(next_index, (&handle).clone());
		handle
	}

	/// Gets the data underlying the handle as a clone
	///
	/// # Arguments
	///
	/// * `handle` - The [`Handle<T>`] to get underlying data from
	///
	/// [`Handle<T>`]: Handle
	pub fn get_clone(&self, handle: &Handle<T>) -> Option<T>
	                 where
		T: (Clone) {
		if !self.is_valid_handle(handle) {
			return None;
		}

		self.storage.read().unwrap().get(&handle.index).cloned()
	}

	/// Remove data from `Storage<T>` using the handle and invalidates
	/// all handles with the data. Returns the data back
	///
	/// # Arguments
	/// * `handle` - The [`Handle<T>`] to remove from storage
	///
	/// [`Handle<T>`]: Handle
	pub fn remove(&self, handle: &Handle<T>) -> Option<T> {
		if !self.is_valid_handle(handle) {
			return None;
		}

		self.handles.write().unwrap().remove(&handle.handle_id);
		self.storage.write().unwrap().remove(&handle.handle_id)
	}

	/// Same thing as `remove()` with the added bonus of a drop
	///
	/// # Arguments
	/// * `handle` - The [`Handle<T>`]'s underlying data to remove from
	/// storage and destroy
	///
	/// [`Handle<T>`]: Handle
	pub fn destroy(&self, handle: &Handle<T>) {
		drop(self.remove(handle))
	}

	/// Check if the handle belongs to the storage struct
	/// & if the handle_id matches up
	///
	/// # Arguments
	///
	/// * `handle` - [`Handle<T>`] to check
	///
	/// [`Handle<T>`]: Handle
	fn is_valid_handle(&self, handle: &Handle<T>) -> bool {
		return handle.storage_id == *self.handle_id.read().unwrap() &&
			self.handles.read().unwrap().get(&handle.handle_id)
				.is_some_and(|x| x.handle_id == handle.handle_id);
	}
}