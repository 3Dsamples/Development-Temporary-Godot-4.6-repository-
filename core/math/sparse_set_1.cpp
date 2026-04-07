--- START OF FILE core/math/sparse_set.cpp ---

#include "core/math/sparse_set.h"
#include "core/os/memory.h"

/**
 * erase()
 * 
 * Removes an entity from the set while maintaining dense array contiguity.
 * 1. Locates the target entity's dense index.
 * 2. Swaps it with the last entity in the dense array.
 * 3. Updates the sparse paging pointers for the moved entity.
 * 4. Invalidates the sparse pointer for the removed entity.
 */
void SparseSet::erase(const BigIntCore &p_entity) {
	if (unlikely(!contains(p_entity))) {
		return;
	}

	uint64_t h_removed = p_entity.hash();
	uint64_t page_removed = h_removed / PAGE_SIZE;
	uint64_t offset_removed = h_removed % PAGE_SIZE;
	
	// Get the index of the entity we are removing
	uint64_t dense_idx = static_cast<uint64_t>(sparse_pages[page_removed]->indices[offset_removed]);
	
	// Identify the last entity in the stream to perform the swap
	uint64_t last_dense_idx = static_cast<uint64_t>(dense.size()) - 1;
	BigIntCore last_entity = dense[last_dense_idx];

	// Step 1: Overwrite the target in the dense array with the last entity
	dense.ptrw()[dense_idx] = last_entity;

	// Step 2: Update the sparse mapping for the entity that was moved
	uint64_t h_moved = last_entity.hash();
	uint64_t page_moved = h_moved / PAGE_SIZE;
	uint64_t offset_moved = h_moved % PAGE_SIZE;
	sparse_pages[page_moved]->indices[offset_moved] = static_cast<int64_t>(dense_idx);

	// Step 3: Invalidate the sparse mapping for the removed entity
	sparse_pages[page_removed]->indices[offset_removed] = -1;

	// Step 4: Remove the duplicate tail element
	dense.remove_at(last_dense_idx);
}

/**
 * clear()
 * 
 * Performs a deterministic purge of the sparse set.
 * Iterates through the virtual pages and releases memory back to the 
 * Scale-Aware memory manager to ensure zero fragmentation.
 */
void SparseSet::clear() {
	// Dense vector handles its own memory via CowData
	dense.clear();

	// Explicitly free all virtual index pages
	for (auto it = sparse_pages.begin(); it.is_valid(); it.next()) {
		Page* p = it.value();
		if (p) {
			memdelete(p);
		}
	}

	sparse_pages.clear();
}

/**
 * find_dense_index()
 * 
 * High-speed lookup for Warp kernels.
 * Returns the raw offset in the SoA stream for a BigIntCore handle.
 */
int64_t SparseSet::find_dense_index(const BigIntCore &p_entity) const {
	uint64_t h = p_entity.hash();
	uint64_t page_id = h / PAGE_SIZE;
	uint64_t offset = h % PAGE_SIZE;

	if (!sparse_pages.has(page_id)) {
		return -1;
	}

	int64_t idx = sparse_pages[page_id]->indices[offset];
	
	// Validation check: ensure the dense entry actually points back to our entity
	if (idx != -1 && static_cast<uint64_t>(idx) < dense.size() && dense[idx] == p_entity) {
		return idx;
	}

	return -1;
}

/**
 * get_memory_footprint()
 * 
 * ETEngine Telemetry Integration.
 * Calculates the total byte weight of the sparse set including paging overhead.
 */
BigIntCore SparseSet::get_memory_footprint() const {
	uint64_t dense_bytes = dense.size() * sizeof(BigIntCore);
	uint64_t sparse_bytes = sparse_pages.size() * (sizeof(uint64_t) + sizeof(Page));
	
	return BigIntCore(static_cast<int64_t>(dense_bytes + sparse_bytes));
}

--- END OF FILE core/math/sparse_set.cpp ---
