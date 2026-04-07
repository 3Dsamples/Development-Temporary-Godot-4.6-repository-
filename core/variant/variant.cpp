--- START OF FILE core/variant/variant.cpp ---

#include "core/variant/variant.h"
#include "core/os/memory.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

// ============================================================================
// Memory Management (Internal)
// ============================================================================

void Variant::_clear() {
	switch (type) {
		case STRING:
			memdelete(_data.string);
			break;
		case STRING_NAME:
			memdelete(_data.string_name);
			break;
		case BIG_INT:
			memdelete(_data.big_int);
			break;
		case VECTOR2:
			memdelete(_data.v2);
			break;
		case VECTOR3:
			memdelete(_data.v3);
			break;
		case VECTOR4:
			memdelete(_data.v4);
			break;
		case RECT2:
			memdelete(_data.rect2);
			break;
		case AABB:
			memdelete(_data.aabb);
			break;
		case BASIS:
			memdelete(_data.basis);
			break;
		case TRANSFORM2D:
			memdelete(_data.t2d);
			break;
		case TRANSFORM3D:
			memdelete(_data.t3d);
			break;
		case QUATERNION:
			memdelete(_data.quat);
			break;
		case COLOR:
			memdelete(_data.color);
			break;
		case PROJECTION:
			memdelete(_data.proj);
			break;
		case RID:
			memdelete(_data.rid);
			break;
		default:
			break;
	}
	type = NIL;
}

void Variant::_copy(const Variant &p_other) {
	_clear();
	type = p_other.type;

	switch (type) {
		case BOOL: _data._bool = p_other._data._bool; break;
		case INT: _data._int = p_other._data._int; break;
		case FIXED_P: _data._fixed_raw = p_other._data._fixed_raw; break;
		
		case STRING: _data.string = memnew(String(*p_other._data.string)); break;
		case BIG_INT: _data.big_int = memnew(BigIntCore(*p_other._data.big_int)); break;
		case VECTOR2: _data.v2 = memnew(Vector2f(*p_other._data.v2)); break;
		case VECTOR3: _data.v3 = memnew(Vector3f(*p_other._data.v3)); break;
		case VECTOR4: _data.v4 = memnew(Vector4f(*p_other._data.v4)); break;
		case RECT2: _data.rect2 = memnew(Rect2f(*p_other._data.rect2)); break;
		case AABB: _data.aabb = memnew(AABBf(*p_other._data.aabb)); break;
		case BASIS: _data.basis = memnew(Basisf(*p_other._data.basis)); break;
		case TRANSFORM2D: _data.t2d = memnew(Transform2Df(*p_other._data.t2d)); break;
		case TRANSFORM3D: _data.t3d = memnew(Transform3Df(*p_other._data.t3d)); break;
		case QUATERNION: _data.quat = memnew(Quaternionf(*p_other._data.quat)); break;
		case COLOR: _data.color = memnew(Colorf(*p_other._data.color)); break;
		case PROJECTION: _data.proj = memnew(Projectionf(*p_other._data.proj)); break;
		case RID: _data.rid = memnew(::RID(*p_other._data.rid)); break;
		case OBJECT: _data.object = p_other._data.object; break;
		default: break;
	}
}

// ============================================================================
// Constructors (Scale-Aware)
// ============================================================================

Variant::Variant(const Variant &p_other) : type(NIL) {
	_copy(p_other);
}

Variant::Variant(bool p_bool) : type(BOOL) { _data._bool = p_bool; }
Variant::Variant(int64_t p_int) : type(INT) { _data._int = p_int; }

Variant::Variant(const FixedMathCore &p_fixed) : type(FIXED_P) {
	_data._fixed_raw = p_fixed.get_raw();
}

Variant::Variant(const BigIntCore &p_big_int) : type(BIG_INT) {
	_data.big_int = memnew(BigIntCore(p_big_int));
}

Variant::Variant(const String &p_string) : type(STRING) {
	_data.string = memnew(String(p_string));
}

Variant::Variant(const Vector3f &p_v3) : type(VECTOR3) {
	_data.v3 = memnew(Vector3f(p_v3));
}

Variant::Variant(const Transform3Df &p_t3d) : type(TRANSFORM3D) {
	_data.t3d = memnew(Transform3Df(p_t3d));
}

Variant::Variant(const RID &p_rid) : type(RID) {
	_data.rid = memnew(::RID(p_rid));
}

Variant::Variant(const Object *p_object) : type(OBJECT) {
	_data.object = const_cast<Object *>(p_object);
}

Variant::~Variant() {
	_clear();
}

// ============================================================================
// Cast Operators (Deterministic)
// ============================================================================

Variant::operator bool() const {
	if (type == BOOL) return _data._bool;
	if (type == INT) return _data._int != 0;
	if (type == FIXED_P) return _data._fixed_raw != 0;
	return type != NIL;
}

Variant::operator int64_t() const {
	if (type == INT) return _data._int;
	if (type == FIXED_P) return FixedMathCore(_data._fixed_raw, true).to_int();
	if (type == BIG_INT) return std::stoll(_data.big_int->to_string());
	return 0;
}

Variant::operator FixedMathCore() const {
	if (type == FIXED_P) return FixedMathCore(_data._fixed_raw, true);
	if (type == INT) return FixedMathCore(_data._int);
	return FixedMathCore(0LL, true);
}

Variant::operator BigIntCore() const {
	if (type == BIG_INT) return *_data.big_int;
	if (type == INT) return BigIntCore(_data._int);
	if (type == FIXED_P) return BigIntCore(FixedMathCore(_data._fixed_raw, true).to_int());
	return BigIntCore(0LL);
}

Variant::operator Vector3f() const {
	if (type == VECTOR3) return *_data.v3;
	return Vector3f();
}

Variant::operator Transform3Df() const {
	if (type == TRANSFORM3D) return *_data.t3d;
	return Transform3Df();
}

Variant::operator Object *() const {
	if (type == OBJECT) return _data.object;
	return nullptr;
}

// ============================================================================
// Operators & Utilities
// ============================================================================

Variant &Variant::operator=(const Variant &p_other) {
	if (this != &p_other) {
		_copy(p_other);
	}
	return *this;
}

bool Variant::operator==(const Variant &p_other) const {
	if (type != p_other.type) return false;
	switch (type) {
		case NIL: return true;
		case BOOL: return _data._bool == p_other._data._bool;
		case INT: return _data._int == p_other._data._int;
		case FIXED_P: return _data._fixed_raw == p_other._data._fixed_raw;
		case BIG_INT: return *_data.big_int == *p_other._data.big_int;
		case STRING: return *_data.string == *p_other._data.string;
		case VECTOR3: return *_data.v3 == *p_other._data.v3;
		case OBJECT: return _data.object == p_other._data.object;
		default: return false;
	}
}

bool Variant::operator!=(const Variant &p_other) const {
	return !(*this == p_other);
}

uint32_t Variant::hash() const {
	switch (type) {
		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return hash_murmur3_one_64(_data._int);
		case FIXED_P: return hash_murmur3_one_64(_data._fixed_raw);
		case BIG_INT: return _data.big_int->hash();
		case STRING: return _data.string->hash();
		case VECTOR3: return _data.v3->x.hash() ^ _data.v3->y.hash() ^ _data.v3->z.hash();
		case OBJECT: return (uint32_t)(uintptr_t)_data.object;
		default: return 0;
	}
}

String Variant::str() const {
	switch (type) {
		case NIL: return "Nil";
		case BOOL: return _data._bool ? "True" : "False";
		case INT: return String::num_int64(_data._int);
		case FIXED_P: return String(FixedMathCore(_data._fixed_raw, true));
		case BIG_INT: return String(*_data.big_int);
		case STRING: return *_data.string;
		case VECTOR3: return (String)*_data.v3;
		default: return "Variant";
	}
}

--- END OF FILE core/variant/variant.cpp ---
