--- START OF FILE core/variant/variant.h ---

#ifndef VARIANT_H
#define VARIANT_H

#include "core/typedefs.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/rect2.h"
#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/math/quaternion.h"
#include "core/math/color.h"
#include "core/math/projection.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

class Object;
class RefCounted;

/**
 * Variant
 * 
 * The universal data bridge for the Scale-Aware pipeline.
 * Strictly deterministic: all continuous values use FixedMathCore (Q32.32).
 * All discrete large-scale values use BigIntCore.
 * Optimized for Warp-Kernel dispatch and 120 FPS simulation heartbeats.
 */
class ET_ALIGN_32 Variant {
public:
	enum Type {
		NIL,

		// Atomic Types
		BOOL,
		INT,          // 64-bit Integer
		FIXED_P,      // FixedMathCore (Deterministic Continuous)
		BIG_INT,      // BigIntCore (Arbitrary Precision)

		// Math Types (Strictly Deterministic Fixed-Point versions)
		STRING,
		VECTOR2,      // Vector2f
		RECT2,        // Rect2f
		VECTOR3,      // Vector3f
		TRANSFORM2D,  // Transform2Df
		VECTOR4,      // Vector4f
		PLANE,        // Planef
		QUATERNION,   // Quaternionf
		AABB,         // AABBf
		BASIS,        // Basisf
		TRANSFORM3D,  // Transform3Df
		PROJECTION,   // Projectionf

		// Misc Types
		COLOR,        // Colorf
		STRING_NAME,
		NODE_PATH,
		RID,
		OBJECT,
		CALLABLE,
		SIGNAL,
		DICTIONARY,
		ARRAY,

		VARIANT_MAX
	};

	enum Operator {
		OP_EQUAL,
		OP_NOT_EQUAL,
		OP_LESS,
		OP_LESS_EQUAL,
		OP_GREATER,
		OP_GREATER_EQUAL,
		
		OP_ADD,
		OP_SUBTRACT,
		OP_MULTIPLY,
		OP_DIVIDE,
		OP_NEGATE,
		OP_POSITIVE,
		OP_MODULE,
		OP_POWER,

		OP_SHIFT_LEFT,
		OP_SHIFT_RIGHT,
		OP_BIT_AND,
		OP_BIT_OR,
		OP_BIT_XOR,
		OP_BIT_NEGATE,

		OP_AND,
		OP_OR,
		OP_XOR,
		OP_NOT,

		OP_IN,
		OP_MAX
	};

private:
	Type type = NIL;

	// Internal data union aligned for SIMD/Warp performance
	union ET_ALIGN_32 Data {
		bool _bool;
		int64_t _int;
		int64_t _fixed_raw; // Direct raw storage for FixedMathCore

		// Pointers for larger/heap types to maintain Variant size
		BigIntCore *big_int;
		String *string;
		Vector2f *v2;
		Vector3f *v3;
		Vector4f *v4;
		Rect2f *rect2;
		AABBf *aabb;
		Basisf *basis;
		Transform2Df *t2d;
		Transform3Df *t3d;
		Quaternionf *quat;
		Colorf *color;
		Projectionf *proj;

		::RID *rid;
		Object *object;
		StringName *string_name;

		uint8_t _mem[24]; // Buffer for small-data optimization
	} _data;

	void _clear();
	void _copy(const Variant &p_other);

public:
	// ------------------------------------------------------------------------
	// Constructors (Scale-Aware Only)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Variant() : type(NIL) {}
	Variant(const Variant &p_other);

	Variant(bool p_bool);
	Variant(int64_t p_int);
	Variant(const FixedMathCore &p_fixed);
	Variant(const BigIntCore &p_big_int);
	Variant(const String &p_string);
	Variant(const Vector3f &p_v3);
	Variant(const Transform3Df &p_t3d);
	Variant(const RID &p_rid);
	Variant(const Object *p_object);

	~Variant();

	// ------------------------------------------------------------------------
	// Type Conversion & Logic
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Type get_type() const { return type; }
	static String get_type_name(Type p_type);
	
	bool is_num() const { return type == INT || type == FIXED_P || type == BIG_INT; }
	bool is_shared_data() const;

	operator bool() const;
	operator int64_t() const;
	operator FixedMathCore() const;
	operator BigIntCore() const;
	operator String() const;
	operator Vector3f() const;
	operator Transform3Df() const;
	operator RID() const;
	operator Object*() const;

	// ------------------------------------------------------------------------
	// Operators (Deterministic Evaluation)
	// ------------------------------------------------------------------------
	Variant &operator=(const Variant &p_other);
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const;
	bool operator<(const Variant &p_other) const;

	static void evaluate(const Operator &p_op, const Variant &p_a, const Variant &p_b, Variant &r_ret, bool &r_valid);

	// ------------------------------------------------------------------------
	// Dynamic Interaction (Warp/EnTT Bridges)
	// ------------------------------------------------------------------------
	void call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error);
	bool has_method(const StringName &p_method) const;

	uint32_t hash() const;
	String str() const;
};

#endif // VARIANT_H

--- END OF FILE core/variant/variant.h ---
