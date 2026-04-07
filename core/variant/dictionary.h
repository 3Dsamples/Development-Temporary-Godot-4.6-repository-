--- START OF FILE core/variant/dictionary.h ---

#ifndef DICTIONARY_H
#define DICTIONARY_H

#include "core/typedefs.h"
#include "core/templates/hash_map.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/variant.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * Dictionary
 * 
 * A high-performance associative container for the Universal Solver.
 * Features Reference Counting and Copy-On-Write for thread-safe simulation data.
 * Aligned to 32 bytes for SIMD-accelerated metadata traversal.
 */
class ET_ALIGN_32 Dictionary {
	struct DictionaryData {
		SafeRefCount refcount;
		HashMap<Variant, Variant> map;
	};

	DictionaryData *_data = nullptr;

	void _ref(const Dictionary &p_from);
	void _unref();

public:
	// ------------------------------------------------------------------------
	// Accessors & Operators
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ Variant &operator[](const Variant &p_key);
	_FORCE_INLINE_ const Variant &operator[](const Variant &p_key) const;

	_FORCE_INLINE_ BigIntCore size() const {
		return _data ? BigIntCore(static_cast<int64_t>(_data->map.size().to_int())) : BigIntCore(0LL);
	}

	_FORCE_INLINE_ bool is_empty() const {
		return !_data || _data->map.is_empty();
	}

	// ------------------------------------------------------------------------
	// Modification API (Warp-Kernel Friendly)
	// ------------------------------------------------------------------------

	void clear();
	bool has(const Variant &p_key) const;
	bool erase(const Variant &p_key);

	/**
	 * get()
	 * Retrieves a value or returns a default. Optimized for zero-copy 
	 * lookups during high-frequency simulation sweeps.
	 */
	Variant get(const Variant &p_key, const Variant &p_default = Variant()) const;

	/**
	 * keys() / values()
	 * Returns EnTT-compatible arrays of keys or values.
	 */
	Array keys() const;
	Array values() const;

	/**
	 * duplicate()
	 * Performs a deep or shallow copy. Used to snapshot simulation states
	 * for deterministic rollback logic.
	 */
	Dictionary duplicate(bool p_deep = false) const;

	// ------------------------------------------------------------------------
	// Lifecycle
	// ------------------------------------------------------------------------

	void operator=(const Dictionary &p_from);

	_FORCE_INLINE_ bool operator==(const Dictionary &p_other) const {
		return _data == p_other._data; // Pointer comparison for COW efficiency
	}

	_FORCE_INLINE_ bool operator!=(const Dictionary &p_other) const {
		return _data != p_other._data;
	}

	uint32_t hash() const;

	Dictionary();
	Dictionary(const Dictionary &p_from);
	~Dictionary();
};

#endif // DICTIONARY_H

--- END OF FILE core/variant/dictionary.h ---
