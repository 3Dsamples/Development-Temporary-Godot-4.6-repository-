--- START OF FILE core/variant/array.h ---

#ifndef ARRAY_H
#define ARRAY_H

#include "core/typedefs.h"
#include "core/templates/vector.h"
#include "core/templates/safe_refcount.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

class Variant;

/**
 * Array
 * 
 * An ordered dynamic container of Variants.
 * Features thread-safe COW (Copy-On-Write) and atomic reference counting.
 * Aligned to 32 bytes for zero-copy EnTT SoA integration and Warp sweeps.
 */
class ET_ALIGN_32 Array {
	struct ArrayData {
		SafeRefCount refcount;
		Vector<Variant> vector;
	};

	ArrayData *_data = nullptr;

	void _ref(const Array &p_from);
	void _unref();

public:
	// ------------------------------------------------------------------------
	// Accessors & Operators
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ const Variant &operator[](const BigIntCore &p_index) const;
	_FORCE_INLINE_ Variant &operator[](const BigIntCore &p_index);

	_FORCE_INLINE_ BigIntCore size() const {
		return _data ? BigIntCore(static_cast<int64_t>(_data->vector.size())) : BigIntCore(0LL);
	}

	_FORCE_INLINE_ bool is_empty() const {
		return !_data || _data->vector.is_empty();
	}

	// ------------------------------------------------------------------------
	// Modification API (Scale-Aware)
	// ------------------------------------------------------------------------

	void clear();
	void push_back(const Variant &p_value);
	void push_front(const Variant &p_value);
	void insert(const BigIntCore &p_pos, const Variant &p_value);
	void remove_at(const BigIntCore &p_index);
	void resize(const BigIntCore &p_size);

	/**
	 * duplicate()
	 * Creates a bit-perfect copy of the array. Essential for snapshotting
	 * deterministic physics states in the Universal Solver.
	 */
	Array duplicate(bool p_deep = false) const;

	// ------------------------------------------------------------------------
	// Deterministic Analysis API
	// ------------------------------------------------------------------------

	/**
	 * sum()
	 * Performs a bit-perfect summation of all elements. 
	 * Automatically handles BigIntCore and FixedMathCore promotion.
	 */
	Variant sum() const;

	/**
	 * min() / max()
	 * Deterministic extreme value search for 120 FPS culling logic.
	 */
	Variant min() const;
	Variant max() const;

	// ------------------------------------------------------------------------
	// Lifecycle
	// ------------------------------------------------------------------------

	void operator=(const Array &p_from);

	_FORCE_INLINE_ bool operator==(const Array &p_other) const {
		return _data == p_other._data; // COW pointer equality
	}

	_FORCE_INLINE_ bool operator!=(const Array &p_other) const {
		return _data != p_other._data;
	}

	uint32_t hash() const;

	Array();
	Array(const Array &p_from);
	~Array();
};

#endif // ARRAY_H

--- END OF FILE core/variant/array.h ---
