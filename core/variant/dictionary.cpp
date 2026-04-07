--- START OF FILE core/variant/dictionary.cpp ---

#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/os/memory.h"

// ============================================================================
// Internal Memory Management (COW Logic)
// ============================================================================

void Dictionary::_unref() {
	if (_data && _data->refcount.unref()) {
		memdelete(_data);
	}
	_data = nullptr;
}

void Dictionary::_ref(const Dictionary &p_from) {
	if (_data == p_from._data) {
		return;
	}
	_unref();
	_data = p_from._data;
	if (_data) {
		_data->refcount.ref();
	}
}

// ============================================================================
// Lifecycle & Operators
// ============================================================================

Dictionary::Dictionary() {
	_data = memnew(DictionaryData);
	_data->refcount.init();
}

Dictionary::Dictionary(const Dictionary &p_from) {
	_ref(p_from);
}

Dictionary::~Dictionary() {
	_unref();
}

void Dictionary::operator=(const Dictionary &p_from) {
	_ref(p_from);
}

Variant &Dictionary::operator[](const Variant &p_key) {
	// ETEngine: Trigger copy-on-write if shared
	if (_data->refcount.get() > 1) {
		DictionaryData *new_data = memnew(DictionaryData);
		new_data->refcount.init();
		new_data->map = _data->map;
		_unref();
		_data = new_data;
	}
	return _data->map[p_key];
}

const Variant &Dictionary::operator[](const Variant &p_key) const {
	static Variant nil;
	if (!_data || !_data->map.has(p_key)) {
		return nil;
	}
	return _data->map[p_key];
}

// ============================================================================
// Modification API
// ============================================================================

void Dictionary::clear() {
	if (_data && _data->refcount.get() == 1) {
		_data->map.clear();
	} else {
		_unref();
		_data = memnew(DictionaryData);
		_data->refcount.init();
	}
}

bool Dictionary::has(const Variant &p_key) const {
	return _data && _data->map.has(p_key);
}

bool Dictionary::erase(const Variant &p_key) {
	if (!_data) return false;
	if (_data->refcount.get() > 1) {
		DictionaryData *new_data = memnew(DictionaryData);
		new_data->refcount.init();
		new_data->map = _data->map;
		_unref();
		_data = new_data;
	}
	return _data->map.erase(p_key);
}

Variant Dictionary::get(const Variant &p_key, const Variant &p_default) const {
	if (_data && _data->map.has(p_key)) {
		return _data->map[p_key];
	}
	return p_default;
}

// ============================================================================
// Data Extraction & Batching
// ============================================================================

Array Dictionary::keys() const {
	Array a;
	if (!_data) return a;
	for (auto it = _data->map.begin(); it.is_valid(); it.next()) {
		a.push_back(it.key());
	}
	return a;
}

Array Dictionary::values() const {
	Array a;
	if (!_data) return a;
	for (auto it = _data->map.begin(); it.is_valid(); it.next()) {
		a.push_back(it.value());
	}
	return a;
}

/**
 * duplicate()
 * 
 * Deep copy implementation for simulation snapshots.
 * Ensures that all internal FixedMathCore and BigIntCore components are 
 * bit-perfectly cloned for deterministic rollback or state logging.
 */
Dictionary Dictionary::duplicate(bool p_deep) const {
	Dictionary d;
	if (!_data) return d;
	
	for (auto it = _data->map.begin(); it.is_valid(); it.next()) {
		Variant val = it.value();
		if (p_deep && val.get_type() == Variant::DICTIONARY) {
			d[it.key()] = val.operator Dictionary().duplicate(true);
		} else {
			d[it.key()] = val;
		}
	}
	return d;
}

uint32_t Dictionary::hash() const {
	if (!_data || _data->map.is_empty()) return 0;
	// Deterministic hash based on BigInt/FixedMath key-value pairs
	uint32_t h = hash_murmur3_one_32(static_cast<uint32_t>(_data->map.size().to_int()));
	for (auto it = _data->map.begin(); it.is_valid(); it.next()) {
		h = hash_murmur3_one_32(it.key().hash(), h);
		h = hash_murmur3_one_32(it.value().hash(), h);
	}
	return h;
}

--- END OF FILE core/variant/dictionary.cpp ---
