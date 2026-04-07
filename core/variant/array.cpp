--- START OF FILE core/variant/array.cpp ---

#include "core/variant/array.h"
#include "core/variant/variant.h"
#include "core/os/memory.h"

// ============================================================================
// Internal Memory Management (COW Logic)
// ============================================================================

void Array::_unref() {
	if (_data && _data->refcount.unref()) {
		memdelete(_data);
	}
	_data = nullptr;
}

void Array::_ref(const Array &p_from) {
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

Array::Array() {
	_data = memnew(ArrayData);
	_data->refcount.init();
}

Array::Array(const Array &p_from) {
	_ref(p_from);
}

Array::~Array() {
	_unref();
}

void Array::operator=(const Array &p_from) {
	_ref(p_from);
}

Variant &Array::operator[](const BigIntCore &p_index) {
	int64_t idx = std::stoll(p_index.to_string());
	CRASH_BAD_INDEX(idx, static_cast<int64_t>(_data->vector.size()));

	if (_data->refcount.get() > 1) {
		ArrayData *new_data = memnew(ArrayData);
		new_data->refcount.init();
		new_data->vector = _data->vector;
		_unref();
		_data = new_data;
	}
	return _data->vector.ptrw()[idx];
}

const Variant &Array::operator[](const BigIntCore &p_index) const {
	int64_t idx = std::stoll(p_index.to_string());
	CRASH_BAD_INDEX(idx, static_cast<int64_t>(_data->vector.size()));
	return _data->vector[idx];
}

// ============================================================================
// Modification API
// ============================================================================

void Array::clear() {
	if (_data && _data->refcount.get() == 1) {
		_data->vector.clear();
	} else {
		_unref();
		_data = memnew(ArrayData);
		_data->refcount.init();
	}
}

void Array::push_back(const Variant &p_value) {
	if (_data->refcount.get() > 1) {
		ArrayData *new_data = memnew(ArrayData);
		new_data->refcount.init();
		new_data->vector = _data->vector;
		_unref();
		_data = new_data;
	}
	_data->vector.push_back(p_value);
}

void Array::resize(const BigIntCore &p_size) {
	int64_t new_size = std::stoll(p_size.to_string());
	if (_data->refcount.get() > 1) {
		ArrayData *new_data = memnew(ArrayData);
		new_data->refcount.init();
		new_data->vector = _data->vector;
		_unref();
		_data = new_data;
	}
	_data->vector.resize(static_cast<int>(new_size));
}

// ============================================================================
// Deterministic Analysis API (Zero-Copy Simulation)
// ============================================================================

/**
 * sum()
 * 
 * Aggregates all elements into a single high-precision result.
 * Optimized for Warp kernels to perform batch reduction on physical quantities.
 */
Variant Array::sum() const {
	if (is_empty()) return Variant(0LL);

	// Detect type from first element for promotion
	Variant::Type first_type = _data->vector[0].get_type();
	
	if (first_type == Variant::BIG_INT) {
		BigIntCore total(0LL);
		for (int i = 0; i < _data->vector.size(); i++) {
			total += _data->vector[i].operator BigIntCore();
		}
		return Variant(total);
	} else {
		FixedMathCore total(0LL, true);
		for (int i = 0; i < _data->vector.size(); i++) {
			total += _data->vector[i].operator FixedMathCore();
		}
		return Variant(total);
	}
}

Variant Array::min() const {
	if (is_empty()) return Variant();
	Variant m = _data->vector[0];
	for (int i = 1; i < _data->vector.size(); i++) {
		if (_data->vector[i] < m) m = _data->vector[i];
	}
	return m;
}

Variant Array::max() const {
	if (is_empty()) return Variant();
	Variant m = _data->vector[0];
	for (int i = 1; i < _data->vector.size(); i++) {
		if (_data->vector[i] > m) m = _data->vector[i];
	}
	return m;
}

uint32_t Array::hash() const {
	if (!_data || _data->vector.is_empty()) return 0;
	uint32_t h = hash_murmur3_one_32(static_cast<uint32_t>(_data->vector.size()));
	for (int i = 0; i < _data->vector.size(); i++) {
		h = hash_murmur3_one_32(_data->vector[i].hash(), h);
	}
	return h;
}

Array Array::duplicate(bool p_deep) const {
	Array a;
	a.resize(size());
	for (int i = 0; i < _data->vector.size(); i++) {
		const Variant &v = _data->vector[i];
		if (p_deep && v.get_type() == Variant::ARRAY) {
			a.push_back(v.operator Array().duplicate(true));
		} else if (p_deep && v.get_type() == Variant::DICTIONARY) {
			a.push_back(v.operator Dictionary().duplicate(true));
		} else {
			a.push_back(v);
		}
	}
	return a;
}

--- END OF FILE core/variant/array.cpp ---
