--- START OF FILE core/templates/hash_set.h ---

#ifndef HASH_SET_H
#define HASH_SET_H

#include "core/os/memory.h"
#include "core/templates/hash_funcs.h"
#include "core/typedefs.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * HashSet Template
 * 
 * A high-performance unique collection using open-addressing with linear probing.
 * Optimized for cache-locality and zero-copy integration with EnTT Sparse Sets.
 * Power-of-two capacity allows for high-speed bitwise masking in Warp kernels.
 */
template <typename T, typename Hasher = HashMapHasherDefault<T>, typename Comparator = DefaultComparator<T>>
class ET_ALIGN_32 HashSet {
public:
	struct Element {
		T value;
		bool active = false;
	};

private:
	Element *_entries = nullptr;
	uint32_t _capacity = 0;
	BigIntCore _size;

	_FORCE_INLINE_ uint32_t _get_slot(const T &p_value) const {
		uint32_t h = Hasher::hash(p_value);
		uint32_t slot = h & (_capacity - 1);
		while (_entries[slot].active && !Comparator::compare(_entries[slot].value, p_value)) {
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
					insert(old_entries[i].value);
				}
				old_entries[i].~Element();
			}
			Memory::free_static(old_entries);
		}
	}

public:
	// ------------------------------------------------------------------------
	// Modification API (Deterministic)
	// ------------------------------------------------------------------------

	/**
	 * insert()
	 * Adds a unique element. Triggers rehash if load factor exceeds 0.75.
	 * Guarantees bit-perfect collection state across TIER_DETERMINISTIC nodes.
	 */
	bool insert(const T &p_value) {
		if (unlikely(_capacity == 0 || (_size + BigIntCore(1LL)) * BigIntCore(4LL) > BigIntCore(static_cast<int64_t>(_capacity)) * BigIntCore(3LL))) {
			_rehash(_capacity == 0 ? 8 : _capacity * 2);
		}

		uint32_t slot = _get_slot(p_value);
		if (!_entries[slot].active) {
			_entries[slot].active = true;
			_entries[slot].value = p_value;
			_size += BigIntCore(1LL);
			return true;
		}
		return false;
	}

	_FORCE_INLINE_ bool has(const T &p_value) const {
		if (_capacity == 0) return false;
		uint32_t slot = _get_slot(p_value);
		return _entries[slot].active;
	}

	bool erase(const T &p_value) {
		if (_capacity == 0) return false;
		uint32_t slot = _get_slot(p_value);
		if (!_entries[slot].active) return false;

		_entries[slot].active = false;
		_entries[slot].value = T();
		_size -= BigIntCore(1LL);

		// Re-insert following elements in the cluster to maintain linear probing sequence
		uint32_t next = (slot + 1) & (_capacity - 1);
		while (_entries[next].active) {
			T val = _entries[next].value;
			_entries[next].active = false;
			_size -= BigIntCore(1LL);
			insert(val);
			next = (next + 1) & (_capacity - 1);
		}
		return true;
	}

	// ------------------------------------------------------------------------
	// Accessors & Lifecycle
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

	HashSet() : _size(0LL) {}
	~HashSet() { clear(); }

	// ------------------------------------------------------------------------
	// Iteration Support (EnTT Friendly)
	// ------------------------------------------------------------------------

	class Iterator {
		const HashSet *set = nullptr;
		uint32_t index = 0;
		friend class HashSet;

	public:
		_FORCE_INLINE_ bool is_valid() const { return set && index < set->_capacity; }
		_FORCE_INLINE_ const T &get() const { return set->_entries[index].value; }
		_FORCE_INLINE_ const T &operator*() const { return set->_entries[index].value; }

		_FORCE_INLINE_ void next() {
			if (!set) return;
			index++;
			while (index < set->_capacity && !set->_entries[index].active) {
				index++;
			}
		}

		_FORCE_INLINE_ bool operator==(const Iterator &p_other) const { return index == p_other.index; }
		_FORCE_INLINE_ bool operator!=(const Iterator &p_other) const { return index != p_other.index; }
	};

	Iterator begin() const {
		Iterator it;
		it.set = this;
		it.index = 0;
		while (it.index < _capacity && !_entries[it.index].active) {
			it.index++;
		}
		return it;
	}

	Iterator end() const {
		Iterator it;
		it.set = this;
		it.index = _capacity;
		return it;
	}
};

#endif // HASH_SET_H

--- END OF FILE core/templates/hash_set.h ---
