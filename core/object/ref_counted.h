--- START OF FILE core/object/ref_counted.h ---

#ifndef REF_COUNTED_H
#define REF_COUNTED_H

#include "core/object/object.h"
#include "core/templates/safe_refcount.h"
#include "src/fixed_math_core.h"

/**
 * RefCounted
 * 
 * Base class for objects managed by smart pointers (Ref<T>).
 * Features atomic reference counting for thread-safe simulation sweeps.
 * Aligned to 32 bytes to ensure that EnTT SoA pools of reference handles
 * remain SIMD-optimized and cache-local.
 */
class ET_ALIGN_32 RefCounted : public Object {
	GDCLASS(RefCounted, Object);

	mutable SafeRefCount ref_count;
	mutable SafeRefCount ref_count_init;

protected:
	static void _bind_methods();

public:
	/**
	 * reference()
	 * Increments the atomic counter. Thread-safe for multi-threaded Warp kernels.
	 */
	_FORCE_INLINE_ bool reference() const {
		if (ref_count_init.get() > 0) {
			if (ref_count_init.unref()) {
				ref_count.set(1);
			}
			return true;
		}

		ref_count.ref();
		return true;
	}

	/**
	 * unreference()
	 * Decrements the atomic counter. Returns true if the object should be deleted.
	 */
	_FORCE_INLINE_ bool unreference() const {
		if (ref_count_init.get() > 0) {
			if (ref_count_init.unref()) {
				return true;
			} else {
				return false;
			}
		}

		if (ref_count.unref()) {
			return true;
		}

		return false;
	}

	_FORCE_INLINE_ int get_reference_count() const {
		return ref_count.get();
	}

	/**
	 * init_ref()
	 * Internal initialization to bridge raw pointers to the Ref<T> system.
	 */
	_FORCE_INLINE_ bool init_ref() {
		if (ref_count_init.get() > 0 && ref_count.get() == 0 && ref_count_init.unref()) {
			ref_count.set(1);
			return true;
		} else {
			return false;
		}
	}

	RefCounted();
	virtual ~RefCounted();
};

/**
 * Ref<T>
 * 
 * High-performance smart pointer for the Universal Solver.
 * Provides zero-copy access to the underlying FixedMathCore or BigIntCore data.
 */
template <typename T>
class Ref {
	T *referenceptr = nullptr;

	void ref(const Ref &p_from) {
		if (p_from.referenceptr == referenceptr) {
			return;
		}

		unref();

		referenceptr = p_from.referenceptr;
		if (referenceptr) {
			referenceptr->reference();
		}
	}

	void unref() {
		if (referenceptr && referenceptr->unreference()) {
			memdelete(referenceptr);
		}
		referenceptr = nullptr;
	}

public:
	_FORCE_INLINE_ bool is_valid() const { return referenceptr != nullptr; }
	_FORCE_INLINE_ bool is_null() const { return referenceptr == nullptr; }

	_FORCE_INLINE_ T *ptr() const { return referenceptr; }
	_FORCE_INLINE_ T *operator->() const { return referenceptr; }
	_FORCE_INLINE_ T &operator*() const { return *referenceptr; }

	Ref &operator=(const Ref &p_from) {
		ref(p_from);
		return *this;
	}

	template <typename T_Other>
	Ref &operator=(const Ref<T_Other> &p_from) {
		T *p_other = dynamic_cast<T *>(p_from.ptr());
		if (p_other == referenceptr) {
			return *this;
		}
		unref();
		referenceptr = p_other;
		if (referenceptr) {
			referenceptr->reference();
		}
		return *this;
	}

	_FORCE_INLINE_ void instantiate() {
		unref();
		referenceptr = memnew(T);
		referenceptr->init_ref();
	}

	_FORCE_INLINE_ bool operator==(const Ref &p_other) const { return referenceptr == p_other.referenceptr; }
	_FORCE_INLINE_ bool operator!=(const Ref &p_other) const { return referenceptr != p_other.referenceptr; }
	_FORCE_INLINE_ bool operator<(const Ref &p_other) const { return referenceptr < p_other.referenceptr; }

	Ref(T *p_ptr) {
		if (p_ptr) {
			referenceptr = p_ptr;
			referenceptr->reference();
		}
	}

	Ref(const Ref &p_from) {
		ref(p_from);
	}

	Ref() {}

	~Ref() {
		unref();
	}
};

#endif // REF_COUNTED_H

--- END OF FILE core/object/ref_counted.h ---
