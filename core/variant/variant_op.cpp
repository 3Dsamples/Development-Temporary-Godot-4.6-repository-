--- START OF FILE core/variant/variant_op.cpp ---

#include "core/variant/variant.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * evaluate()
 * 
 * The master dispatcher for Variant operations.
 * Ported to strictly use Software-Defined Arithmetic (SDA).
 * Optimized for Zero-Copy data flow between EnTT Sparse Sets and Warp Kernels.
 */
void Variant::evaluate(const Operator &p_op, const Variant &p_a, const Variant &p_b, Variant &r_ret, bool &r_valid) {
	r_valid = true;
	Type type_a = p_a.get_type();
	Type type_b = p_b.get_type();

	// 1. TIER_DETERMINISTIC: FixedMathCore Arithmetic
	if (type_a == FIXED_P || type_b == FIXED_P) {
		FixedMathCore a = p_a.operator FixedMathCore();
		FixedMathCore b = p_b.operator FixedMathCore();

		switch (p_op) {
			case OP_EQUAL: r_ret = (a == b); return;
			case OP_NOT_EQUAL: r_ret = (a != b); return;
			case OP_LESS: r_ret = (a < b); return;
			case OP_LESS_EQUAL: r_ret = (a <= b); return;
			case OP_GREATER: r_ret = (a > b); return;
			case OP_GREATER_EQUAL: r_ret = (a >= b); return;
			case OP_ADD: r_ret = Variant(a + b); return;
			case OP_SUBTRACT: r_ret = Variant(a - b); return;
			case OP_MULTIPLY: r_ret = Variant(a * b); return;
			case OP_DIVIDE: 
				if (unlikely(b.get_raw() == 0)) { r_valid = false; return; }
				r_ret = Variant(a / b); return;
			case OP_MODULE: 
				if (unlikely(b.get_raw() == 0)) { r_valid = false; return; }
				r_ret = Variant(a % b); return;
			default: break;
		}
	}

	// 2. TIER_MACRO_ECONOMY: BigIntCore Arithmetic
	if (type_a == BIG_INT || type_b == BIG_INT) {
		BigIntCore a = p_a.operator BigIntCore();
		BigIntCore b = p_b.operator BigIntCore();

		switch (p_op) {
			case OP_EQUAL: r_ret = (a == b); return;
			case OP_NOT_EQUAL: r_ret = (a != b); return;
			case OP_LESS: r_ret = (a < b); return;
			case OP_LESS_EQUAL: r_ret = (a <= b); return;
			case OP_GREATER: r_ret = (a > b); return;
			case OP_GREATER_EQUAL: r_ret = (a >= b); return;
			case OP_ADD: r_ret = Variant(a + b); return;
			case OP_SUBTRACT: r_ret = Variant(a - b); return;
			case OP_MULTIPLY: r_ret = Variant(a * b); return;
			case OP_DIVIDE: 
				if (unlikely(b.is_zero())) { r_valid = false; return; }
				r_ret = Variant(a / b); return;
			case OP_MODULE: 
				if (unlikely(b.is_zero())) { r_valid = false; return; }
				r_ret = Variant(a % b); return;
			default: break;
		}
	}

	// 3. Integer Arithmetic (Standard 64-bit)
	if (type_a == INT && type_b == INT) {
		int64_t a = p_a.operator int64_t();
		int64_t b = p_b.operator int64_t();

		switch (p_op) {
			case OP_EQUAL: r_ret = (a == b); return;
			case OP_NOT_EQUAL: r_ret = (a != b); return;
			case OP_LESS: r_ret = (a < b); return;
			case OP_LESS_EQUAL: r_ret = (a <= b); return;
			case OP_GREATER: r_ret = (a > b); return;
			case OP_GREATER_EQUAL: r_ret = (a >= b); return;
			case OP_ADD: r_ret = a + b; return;
			case OP_SUBTRACT: r_ret = a - b; return;
			case OP_MULTIPLY: r_ret = a * b; return;
			case OP_DIVIDE: 
				if (unlikely(b == 0)) { r_valid = false; return; }
				r_ret = a / b; return;
			case OP_BIT_AND: r_ret = a & b; return;
			case OP_BIT_OR: r_ret = a | b; return;
			case OP_BIT_XOR: r_ret = a ^ b; return;
			case OP_SHIFT_LEFT: r_ret = a << b; return;
			case OP_SHIFT_RIGHT: r_ret = a >> b; return;
			default: break;
		}
	}

	// 4. Boolean Logic
	if (type_a == BOOL && type_b == BOOL) {
		bool a = p_a.operator bool();
		bool b = p_b.operator bool();
		switch (p_op) {
			case OP_EQUAL: r_ret = (a == b); return;
			case OP_NOT_EQUAL: r_ret = (a != b); return;
			case OP_AND: r_ret = (a && b); return;
			case OP_OR: r_ret = (a || b); return;
			case OP_XOR: r_ret = (a ^ b); return;
			default: break;
		}
	}

	r_valid = false;
	r_ret = Variant();
}

--- END OF FILE core/variant/variant_op.cpp ---
