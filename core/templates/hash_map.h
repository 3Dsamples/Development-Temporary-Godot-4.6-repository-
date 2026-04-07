--- START OF FILE core/templates/hash_map.h ---

#ifndef HASH_MAP_H
#define HASH_MAP_H

#include "core/os/memory.h"
#include "core/templates/hash_funcs.h"
#include "core/typedefs.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * HashMap Template
 * 
 * Associative container using open-addressing with linear probing.
 * Engineered for cache-locality to support high-frequency EnTT registry queries.
 * Power-of-two sizing ensures fast bitwise indexing for Warp kernels.
 */
template <typename K, typename V, typename Hasher = HashMapHasherDefault<K>, typename Comparator = DefaultComparator<K>>
class ET_ALIGN_32 HashMap {
public:
	struct Element {
		K key;
		V value;
		bool active = false;
	};

private:
	Element *_entries = nullptr;
	uint32_t _capacity = 0;
	BigIntCore _size;

	_FORCE_INLINE_ uint32_t _get_slot(const K &p_key) const {
		uint32_t h = Hasher::hash(p_key);
		uint32_t slot = h & (_capacity - 1);
		while (_entries[slot].active && !Comparator::compare(_entries[slot].key, p_key)) {
			slot = (slot + 1) & (_capacity - 1);
		}
		return slot;
	}

	void _rehash(uint32_t p_new_capacity) {
		Element *old_entries = _entries;
		uint32_t old_capacity = _capacity;

		_capacity = p_new_capacity;
		_entries = (Element *)Memory::alloc_static(sizeof(Element) * _capacity, ET_MEM_DYNAMIC);
		for (uint32_t i = 0; i < _capacity; i++) {
			memnew_placement(&_entries[i], Element);
		}

		_size = BigIntCore(0LL);
		if (old_entries) {
			for (uint32_t i = 0; i < old_capacity; i++) {
				if (old_entries[i].active) {
					insert(old_entries[i].key, old_entries[i].value);
				}
				old_entries[i].~Element();
			}
			Memory::free_static(old_entries);
		}
	}

public:
	// ------------------------------------------------------------------------
	// Modification API
	// ------------------------------------------------------------------------

	/**
	 * insert()
	 * Maps a key to a value. Triggers deterministic rehash if load factor > 0.75.
	 * Zero-copy compatible for FixedMathCore and BigIntCore values.
	 */
	void insert(const K &p_key, const V &p_value) {
		if (unlikely(_capacity == 0 || (_size + BigIntCore(1LL)) * BigIntCore(4LL) > BigIntCore(static_cast<int64_t>(_capacity)) * BigIntCore(3LL))) {
			_rehash(_capacity == 0 ? 8 : _capacity * 2);
		}

		uint32_t slot = _get_slot(p_key);
		if (!_entries[slot].active) {
			_entries[slot].active = true;
			_entries[slot].key = p_key;
			_size += BigIntCore(1LL);
		}
		_entries[slot].value = p_value;
	}

	_FORCE_INLINE_ bool has(const K &p_key) const {
		if (_capacity == 0) return false;
		uint32_t slot = _get_slot(p_key);
		return _entries[slot].active;
	}

	_FORCE_INLINE_ V &operator[](const K &p_key) {
		uint32_t slot = _get_slot(p_key);
		if (!_entries[slot].active) {
			insert(p_key, V());
			slot = _get_slot(p_key);
		}
		return _entries[slot].value;
	}

	bool erase(const K &p_key) {
		if (_capacity == 0) return false;
		uint32_t slot = _get_slot(p_key);
		if (!_entries[slot].active) return false;

		_entries[slot].active = false;
		_entries[slot].value = V();
		_size -= BigIntCore(1LL);

		// Rehash the cluster to maintain linear probing integrity
		uint32_t next = (slot + 1) & (_capacity - 1);
		while (_entries[next].active) {
			K k = _entries[next].key;
			V v = _entries[next].value;
			_entries[next].active = false;
			_size -= BigIntCore(1LL);
			insert(k, v);
			next = (next + 1) & (_capacity - 1);
		}
		return true;
	}

	// ------------------------------------------------------------------------
	// Accessors
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ BigIntCore size() const { return _size; }
	_FORCE_INLINE_ bool is_empty() const { return _size == BigIntCore(0LL); }

	void clear() {
		if (_entries) {
			for (uint32_t i = 0; i < _capacity; i++) {
				_entries[i].~Element();
			}
			Memory::free_static(_entries);
			_entries = nullptr;
		}
		_capacity = 0;
		_size = BigIntCore(0LL);
	}

	HashMap() : _size(0LL) {}
	~HashMap() { clear(); }
};

#endif // HASH_MAP_H

--- END OF FILE core/templates/hash_map.h ---
