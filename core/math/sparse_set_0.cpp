--- START OF FILE core/math/sparse_set.cpp ---

#include "core/math/sparse_set.h"
#include "core/os/memory.h"

namespace UniversalSolver {

/**
 * emplace()
 * 
 * Registers a BigIntCore entity into the sparse set.
 * If the entity already exists, it returns the current dense index.
 * Otherwise, it appends the entity to the dense array and updates the sparse mapping.
 */
uint64_t SparseSet::emplace(const BigIntCore &p_entity) {
	int64_t &sparse_idx = _get_sparse_index(p_entity);
	
	if (sparse_idx != -1 && static_cast<uint64_t>(sparse_idx) < dense.size() && dense[sparse_idx] == p_entity) {
		return static_cast<uint64_t>(sparse_idx);
	}

	uint64_t new_dense_idx = dense.size();
	dense.push_back(p_entity);
	sparse_idx = static_cast<int64_t>(new_dense_idx);
	
	return new_dense_idx;
}

/**
 * erase()
 * 
 * Removes an entity using the swap-and-pop mechanism.
 * This ensures that the dense array remains contiguous for Warp-style parallel sweeps.
 * The last element in the dense array is moved to the position of the deleted element.
 */
void SparseSet::erase(const BigIntCore &p_entity) {
	int64_t &sparse_idx_ref = _get_sparse_index(p_entity);
	int64_t idx = sparse_idx_ref;

	if (unlikely(idx == -1 || static_cast<uint64_t>(idx) >= dense.size() || !(dense[idx] == p_entity))) {
		return;
	}

	BigIntCore last_entity = dense[dense.size() - 1];
	uint64_t last_idx = dense.size() - 1;

	// Swap the deleted element with the last element in the dense array
	dense.ptrw()[idx] = last_entity;
	
	// Update the sparse index for the moved element
	_get_sparse_index(last_entity) = idx;
	
	// Invalidate the sparse index for the erased entity
	sparse_idx_ref = -1;

	// Remove the last element
	dense.remove_at(last_idx);
}

/**
 * clear()
 * 
 * Resets the sparse set by clearing the dense array and freeing paged sparse memory.
 * Essential for simulation resets where trillions of entity handles must be purged.
 */
void SparseSet::clear() {
	dense.clear();
	
	// Iterate through the hash map and free all sparse index pages
	for (auto it = sparse_pages.begin(); it.is_valid(); it.next()) {
		memdelete(it.value());
	}
	sparse_pages.clear();
}

/**
 * find()
 * 
 * Returns the dense index of an entity if it exists, otherwise returns -1.
 * Optimized for high-frequency Warp kernel lookup.
 */
int64_t SparseSet::find(const BigIntCore &p_entity) const {
	// Const-casting is required internally for paged lookup if not present,
	// however, in a 'find' scenario, we only check for existence.
	uint64_t h = p_entity.hash();
	uint64_t page_id = h / 4096;
	uint64_t offset = h % 4096;

	if (!sparse_pages.has(page_id)) {
		return -1;
	}

	int64_t idx = sparse_pages[page_id]->indices[offset];
	if (idx != -1 && static_cast<uint64_t>(idx) < dense.size() && dense[idx] == p_entity) {
		return idx;
	}

	return -1;
}

} // namespace UniversalSolver

--- END OF FILE core/math/sparse_set.cpp ---
