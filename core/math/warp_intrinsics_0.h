--- START OF FILE core/math/warp_intrinsics.h ---

#ifndef WARP_INTRINSICS_H
#define WARP_INTRINSICS_H

#include "core/typedefs.h"
#include "core/math/math_funcs.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * wp Namespace (Warp-style Intrinsics)
 * 
 * Provides a functional math API for use within Warp Kernels.
 * Strictly uses FixedMathCore for continuous values and BigIntCore for discrete logic.
 */
namespace wp {

// ============================================================================
// Scalar Math Intrinsics
// ============================================================================

static _FORCE_INLINE_ FixedMathCore abs(FixedMathCore p_a) { return p_a.absolute(); }
static _FORCE_INLINE_ BigIntCore abs(BigIntCore p_a) { return p_a.absolute(); }

static _FORCE_INLINE_ FixedMathCore min(FixedMathCore p_a, FixedMathCore p_b) { return p_a < p_b ? p_a : p_b; }
static _FORCE_INLINE_ BigIntCore min(BigIntCore p_a, BigIntCore p_b) { return p_a < p_b ? p_a : p_b; }

static _FORCE_INLINE_ FixedMathCore max(FixedMathCore p_a, FixedMathCore p_b) { return p_a > p_b ? p_a : p_b; }
static _FORCE_INLINE_ BigIntCore max(BigIntCore p_a, BigIntCore p_b) { return p_a > p_b ? p_a : p_b; }

static _FORCE_INLINE_ FixedMathCore clamp(FixedMathCore p_val, FixedMathCore p_min, FixedMathCore p_max) {
	return max(p_min, min(p_max, p_val));
}

static _FORCE_INLINE_ FixedMathCore lerp(FixedMathCore p_a, FixedMathCore p_b, FixedMathCore p_t) {
	return p_a + (p_b - p_a) * p_t;
}

static _FORCE_INLINE_ FixedMathCore step(FixedMathCore p_edge, FixedMathCore p_x) {
	return p_x < p_edge ? FixedMathCore(0LL, true) : FixedMathCore(1LL, false);
}

static _FORCE_INLINE_ FixedMathCore sqrt(FixedMathCore p_a) { return p_a.square_root(); }
static _FORCE_INLINE_ FixedMathCore sin(FixedMathCore p_a) { return p_a.sin(); }
static _FORCE_INLINE_ FixedMathCore cos(FixedMathCore p_a) { return p_a.cos(); }
static _FORCE_INLINE_ FixedMathCore tan(FixedMathCore p_a) { return p_a.tan(); }
static _FORCE_INLINE_ FixedMathCore atan2(FixedMathCore p_y, FixedMathCore p_x) { return p_y.atan2(p_x); }

// ============================================================================
// Vector Math Intrinsics (Deterministic TIER_DETERMINISTIC)
// ============================================================================

static _FORCE_INLINE_ FixedMathCore dot(const Vector2f &p_a, const Vector2f &p_b) { return p_a.dot(p_b); }
static _FORCE_INLINE_ FixedMathCore dot(const Vector3f &p_a, const Vector3f &p_b) { return p_a.dot(p_b); }
static _FORCE_INLINE_ FixedMathCore dot(const Vector4f &p_a, const Vector4f &p_b) { return p_a.dot(p_b); }

static _FORCE_INLINE_ Vector3f cross(const Vector3f &p_a, const Vector3f &p_b) { return p_a.cross(p_b); }

static _FORCE_INLINE_ FixedMathCore length(const Vector3f &p_a) { return p_a.length(); }
static _FORCE_INLINE_ FixedMathCore length_sq(const Vector3f &p_a) { return p_a.length_squared(); }

static _FORCE_INLINE_ Vector3f normalize(const Vector3f &p_a) { return p_a.normalized(); }

static _FORCE_INLINE_ Vector3f lerp(const Vector3f &p_a, const Vector3f &p_b, FixedMathCore p_t) {
	return p_a.lerp(p_b, p_t);
}

// ============================================================================
// Relational & Logic (Zero-Copy)
// ============================================================================

static _FORCE_INLINE_ bool all(const Vector3f &p_a) {
	return p_a.x.get_raw() != 0 && p_a.y.get_raw() != 0 && p_a.z.get_raw() != 0;
}

static _FORCE_INLINE_ bool any(const Vector3f &p_a) {
	return p_a.x.get_raw() != 0 || p_a.y.get_raw() != 0 || p_a.z.get_raw() != 0;
}

static _FORCE_INLINE_ FixedMathCore select(bool p_cond, FixedMathCore p_true, FixedMathCore p_false) {
	return p_cond ? p_true : p_false;
}

static _FORCE_INLINE_ BigIntCore select(bool p_cond, BigIntCore p_true, BigIntCore p_false) {
	return p_cond ? p_true : p_false;
}

} // namespace wp

#endif // WARP_INTRINSICS_H

--- END OF FILE core/math/warp_intrinsics.h ---
