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
 * The master data organizer for the "Who" in the Universal Solver.
 * Manages Entity IDs (BigIntCore) and maps them to dense indices.
 * 
 * Features:
 * - Paged Sparse Array: Supports trillion-entity ranges.
 * - Dense Contiguous Array: Aligned for Warp kernel SIMD sweeps.
 * - Swap-and-Pop: Constant time erasure while maintaining density.
 */
class ET_ALIGN_32 SparseSet {
public:
	static const uint64_t PAGE_SIZE = 4096;

private:
	// Dense storage for Entity Handles (BigIntCore)
	Vector<BigIntCore> dense;

	// Sparse paging structure to handle BigInt ranges
	struct Page {
		int64_t indices[PAGE_SIZE];
		Page() {
			for (uint64_t i = 0; i < PAGE_SIZE; i++) {
				indices[i] = -1; // -1 indicates null mapping
			}
		}
	};

	// Map of Page IDs to paged index buffers
	HashMap<uint64_t, Page*> sparse_pages;

	/**
	 * _get_or_create_page()
	 * Resolves the virtual page for a specific BigIntCore entity.
	 */
	_FORCE_INLINE_ Page* _get_or_create_page(const BigIntCore &p_entity) {
		uint64_t h = p_entity.hash();
		uint64_t page_id = h / PAGE_SIZE;
		if (unlikely(!sparse_pages.has(page_id))) {
			sparse_pages[page_id] = memnew(Page);
		}
		return sparse_pages[page_id];
	}

public:
	// ------------------------------------------------------------------------
	// Entity Management API (O(1))
	// ------------------------------------------------------------------------

	/**
	 * contains()
	 * Bit-perfect validation of entity existence in the set.
	 */
	_FORCE_INLINE_ bool contains(const BigIntCore &p_entity) const {
		uint64_t h = p_entity.hash();
		uint64_t page_id = h / PAGE_SIZE;
		uint64_t offset = h % PAGE_SIZE;

		if (!sparse_pages.has(page_id)) return false;
		
		int64_t idx = sparse_pages[page_id]->indices[offset];
		return (idx != -1 && static_cast<uint64_t>(idx) < dense.size() && dense[idx] == p_entity);
	}

	/**
	 * emplace()
	 * Registers a new entity. Returns its index in the dense SoA stream.
	 */
	uint64_t emplace(const BigIntCore &p_entity) {
		if (contains(p_entity)) {
			uint64_t h = p_entity.hash();
			return static_cast<uint64_t>(sparse_pages[h / PAGE_SIZE]->indices[h % PAGE_SIZE]);
		}

		uint64_t new_idx = static_cast<uint64_t>(dense.size());
		dense.push_back(p_entity);

		Page* p = _get_or_create_page(p_entity);
		p->indices[p_entity.hash() % PAGE_SIZE] = static_cast<int64_t>(new_idx);

		return new_idx;
	}

	/**
	 * erase()
	 * Removes an entity using Swap-and-Pop.
	 * Maintains dense array contiguity for 120 FPS Warp sweeps.
	 */
	void erase(const BigIntCore &p_entity);

	// ------------------------------------------------------------------------
	// Accessors (Zero-Copy)
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ BigIntCore size() const { return BigIntCore(static_cast<int64_t>(dense.size())); }
	_FORCE_INLINE_ bool is_empty() const { return dense.is_empty(); }

	_FORCE_INLINE_ const BigIntCore& operator[](uint64_t p_idx) const { return dense[p_idx]; }

	/**
	 * get_dense_ptr()
	 * Direct access for Warp Kernels to perform parallel entity processing.
	 */
	_FORCE_INLINE_ const BigIntCore* get_dense_ptr() const { return dense.ptr(); }

	/**
	 * index_of()
	 * Returns the dense index for a BigIntCore handle.
	 */
	_FORCE_INLINE_ uint64_t index_of(const BigIntCore &p_entity) const {
		uint64_t h = p_entity.hash();
		return static_cast<uint64_t>(sparse_pages[h / PAGE_SIZE]->indices[h % PAGE_SIZE]);
	}

	// ------------------------------------------------------------------------
	// Lifecycle
	// ------------------------------------------------------------------------

	void clear();

	SparseSet() {}
	~SparseSet() { clear(); }
};

#endif // SPARSE_SET_H

--- END OF FILE core/math/sparse_set.h ---
