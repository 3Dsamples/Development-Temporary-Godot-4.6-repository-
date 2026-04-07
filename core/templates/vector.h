--- START OF FILE core/templates/vector.h ---

#ifndef VECTOR_H
#define VECTOR_H

#include "core/typedefs.h"
#include "core/templates/cowdata.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Vector<T>
 * 
 * High-performance dynamic array with COW (Copy-On-Write) semantics.
 * Aligned to 32 bytes to ensure EnTT component pools are SIMD-ready.
 * Optimized for Warp-style zero-copy kernel sweeps.
 */
template <typename T>
class ET_ALIGN_32 Vector {
	CowData<T> _cowdata;

public:
	// ------------------------------------------------------------------------
	// Accessors (Zero-Copy for Warp Kernels)
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ const T *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ T *ptrw() { return _cowdata.ptrw(); }

	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	_FORCE_INLINE_ bool is_empty() const { return _cowdata.size() == 0; }

	_FORCE_INLINE_ const T &operator[](int p_index) const {
		return _cowdata.get(p_index);
	}

	_FORCE_INLINE_ T &operator[](int p_index) {
		return _cowdata.get_m(p_index);
	}

	// ------------------------------------------------------------------------
	// Modification API (EnTT Compatible)
	// ------------------------------------------------------------------------

	Error push_back(const T &p_val) {
		return _cowdata.insert(size(), p_val);
	}

	void remove_at(int p_index) {
		_cowdata.remove_at(p_index);
	}

	void clear() {
		_cowdata.resize(0);
	}

	Error resize(int p_size) {
		return _cowdata.resize(p_size);
	}

	/**
	 * fill()
	 * Batch-initialization for simulation buffers.
	 */
	void fill(const T &p_val) {
		int s = size();
		T *w = ptrw();
		for (int i = 0; i < s; i++) {
			w[i] = p_val;
		}
	}

	// ------------------------------------------------------------------------
	// Operators
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ void operator=(const Vector &p_from) {
		_cowdata = p_from._cowdata;
	}

	_FORCE_INLINE_ bool operator==(const Vector &p_other) const {
		int s = size();
		if (s != p_other.size()) return false;
		const T *r1 = ptr();
		const T *r2 = p_other.ptr();
		for (int i = 0; i < s; i++) {
			if (!(r1[i] == r2[i])) return false;
		}
		return true;
	}

	Vector() {}
	Vector(const Vector &p_from) { _cowdata = p_from._cowdata; }
	~Vector() {}
};

#endif // VECTOR_H

--- END OF FILE core/templates/vector.h ---
