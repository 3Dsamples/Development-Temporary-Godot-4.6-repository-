--- START OF FILE core/math/sparse_set.h ---

#ifndef SPARSE_SET_H
#define SPARSE_SET_H

#include "core/typedefs.h"
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * SparseSet
 * 
 * Manages Entity IDs (BigIntCore) in a dual-array system.
 * Dense Array: Contiguous storage of Entity IDs for SIMD-friendly iteration.
 * Sparse Array: Paged lookup mapping Entity IDs to Dense indices.
 * Engineered for 120 FPS simulation throughput in galactic-scale environments.
 */
class ET_ALIGN_32 SparseSet {
private:
	// Dense storage for Entity Handles (BigIntCore)
	// Aligned to 32 bytes for Warp kernel iteration efficiency
	Vector<BigIntCore> dense;

	// Sparse mapping logic
	// Since IDs are BigIntCore, we use a paged sparse array approach to handle 
	// astronomical ID ranges without massive contiguous memory allocation.
	struct Page {
		int64_t indices[4096];
		Page() {
			for (int i = 0; i < 4096; i++) {
				indices[i] = -1; // -1 indicates null mapping
			}
		}
	};

	HashMap<uint64_t, Page*> sparse_pages;

	_FORCE_INLINE_ int64_t& _get_sparse_index(const BigIntCore &p_entity) {
		uint64_t h = p_entity.hash();
		uint64_t page_id = h / 4096;
		uint64_t offset = h % 4096;

		if (unlikely(!sparse_pages.has(page_id))) {
			sparse_pages[page_id] = memnew(Page);
		}
		return sparse_pages[page_id]->indices[offset];
	}

public:
	// ------------------------------------------------------------------------
	// Entity Management (O(1) Operations)
	// ------------------------------------------------------------------------

	/**
	 * contains()
	 * Returns true if the entity handle exists in this set.
	 */
	_FORCE_INLINE_ bool contains(const BigIntCore &p_entity) {
		int64_t idx = _get_sparse_index(p_entity);
		return (idx != -1 && static_cast<uint64_t>(idx) < dense.size() && dense[idx] == p_entity);
	}

	/**
	 * emplace()
	 * Adds an entity to the set. Optimized for 120 FPS batch insertion.
	 */
	uint64_t emplace(const BigIntCore &p_entity) {
		if (contains(p_entity)) {
			return static_cast<uint64_t>(_get_sparse_index(p_entity));
		}

		uint64_t idx = dense.size();
		dense.push_back(p_entity);
		_get_sparse_index(p_entity) = static_cast<int64_t>(idx);
		return idx;
	}

	/**
	 * erase()
	 * Removes an entity using the "Swap-and-Pop" technique.
	 * Maintains dense array contiguity for Warp kernel SoA sweeps.
	 */
	void erase(const BigIntCore &p_entity) {
		if (!contains(p_entity)) return;

		int64_t idx = _get_sparse_index(p_entity);
		BigIntCore last_entity = dense[dense.size() - 1];

		// Overwrite target with last element to maintain density
		dense.ptrw()[idx] = last_entity;
		_get_sparse_index(last_entity) = idx;
		_get_sparse_index(p_entity) = -1;

		dense.remove_at(dense.size() - 1);
	}

	// ------------------------------------------------------------------------
	// Data Accessors
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ uint64_t size() const { return dense.size(); }
	_FORCE_INLINE_ bool is_empty() const { return dense.is_empty(); }

	/**
	 * get_dense_ptr()
	 * Provides the raw pointer for Warp kernels to perform zero-copy
	 * parallel sweeps over the entity list.
	 */
	_FORCE_INLINE_ const BigIntCore* get_dense_ptr() const { return dense.ptr(); }

	_FORCE_INLINE_ const BigIntCore& operator[](uint64_t p_idx) const {
		return dense[p_idx];
	}

	void clear() {
		dense.clear();
		for (auto &E : sparse_pages) {
			memdelete(E.value);
		}
		sparse_pages.clear();
	}

	SparseSet() {}
	~SparseSet() { clear(); }
};

#endif // SPARSE_SET_H

--- END OF FILE core/math/sparse_set.h ---
