--- START OF FILE core/variant/variant_call.cpp ---

#include "core/variant/variant.h"
#include "core/object/object.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * call()
 * 
 * Dynamic method invocation engine. 
 * Optimized to handle Scale-Aware primitives natively.
 * Uses interned StringName comparisons to maintain 120 FPS performance
 * during dynamic script execution.
 */
void Variant::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	// 1. Handle Object-based calls (Native Classes / Scripts)
	if (type == OBJECT) {
		if (unlikely(!_data.object)) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}
		r_ret = _data.object->callp(p_method, p_args, p_argcount, r_error);
		return;
	}

	// 2. TIER_DETERMINISTIC: FixedMathCore Built-ins
	if (type == FIXED_P) {
		FixedMathCore v = FixedMathCore(_data._fixed_raw, true);

		if (p_method == SNAME("sin")) { r_ret = Variant(v.sin()); return; }
		if (p_method == SNAME("cos")) { r_ret = Variant(v.cos()); return; }
		if (p_method == SNAME("tan")) { r_ret = Variant(v.tan()); return; }
		if (p_method == SNAME("sqrt")) { r_ret = Variant(v.square_root()); return; }
		if (p_method == SNAME("abs")) { r_ret = Variant(v.absolute()); return; }
		if (p_method == SNAME("to_int")) { r_ret = v.to_int(); return; }
	}

	// 3. TIER_MACRO_ECONOMY: BigIntCore Built-ins
	if (type == BIG_INT) {
		const BigIntCore &v = *_data.big_int;

		if (p_method == SNAME("to_scientific")) { r_ret = String(v.to_scientific().c_str()); return; }
		if (p_method == SNAME("to_aa")) { r_ret = String(v.to_aa_notation().c_str()); return; }
		if (p_method == SNAME("abs")) { r_ret = Variant(v.absolute()); return; }
		
		if (p_method == SNAME("pow") && p_argcount == 1) {
			r_ret = Variant(v.power(p_args[0]->operator BigIntCore()));
			return;
		}
	}

	// 4. Vector3f (Deterministic Spatial) Built-ins
	if (type == VECTOR3) {
		Vector3f *v = _data.v3;

		if (p_method == SNAME("length")) { r_ret = Variant(v->length()); return; }
		if (p_method == SNAME("length_squared")) { r_ret = Variant(v->length_squared()); return; }
		if (p_method == SNAME("normalized")) { r_ret = Variant(v->normalized()); return; }
		
		if (p_method == SNAME("dot") && p_argcount == 1) {
			r_ret = Variant(v->dot(p_args[0]->operator Vector3f()));
			return;
		}
		if (p_method == SNAME("cross") && p_argcount == 1) {
			r_ret = Variant(v->cross(p_args[0]->operator Vector3f()));
			return;
		}
	}

	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
}

/**
 * has_method()
 * 
 * High-speed validation for dynamic calls.
 */
bool Variant::has_method(const StringName &p_method) const {
	if (type == OBJECT) {
		return _data.object ? _data.object->has_method(p_method) : false;
	}

	if (type == FIXED_P) {
		return (p_method == SNAME("sin") || p_method == SNAME("cos") || p_method == SNAME("tan") || 
				p_method == SNAME("sqrt") || p_method == SNAME("abs") || p_method == SNAME("to_int"));
	}

	if (type == BIG_INT) {
		return (p_method == SNAME("to_scientific") || p_method == SNAME("to_aa") || 
				p_method == SNAME("abs") || p_method == SNAME("pow"));
	}

	if (type == VECTOR3) {
		return (p_method == SNAME("length") || p_method == SNAME("length_squared") || 
				p_method == SNAME("normalized") || p_method == SNAME("dot") || p_method == SNAME("cross"));
	}

	return false;
}

--- END OF FILE core/variant/variant_call.cpp ---
