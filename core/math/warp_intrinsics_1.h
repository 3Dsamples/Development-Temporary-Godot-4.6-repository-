--- START OF FILE core/math/warp_intrinsics.h ---

#ifndef WARP_INTRINSICS_H
#define WARP_INTRINSICS_H

#include "core/typedefs.h"
#include "core/math/math_funcs.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/quaternion.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * wp Namespace (Warp-style Deterministic Intrinsics)
 * 
 * Provides the functional math API for Warp Kernels.
 * strictly uses Software-Defined Arithmetic to ensure 100% bit-perfection.
 * Optimized for SIMD-unrolling and zero-copy data flow.
 */
namespace wp {

// ============================================================================
// Scalar Functional Primitives
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
	return p_x < p_edge ? FixedMathCore(0LL, true) : FixedMathCore(FixedMathCore::ONE_RAW, true);
}

static _FORCE_INLINE_ FixedMathCore smoothstep(FixedMathCore p_edge0, FixedMathCore p_edge1, FixedMathCore p_x) {
	FixedMathCore t = clamp((p_x - p_edge0) / (p_edge1 - p_edge0), FixedMathCore(0LL, true), FixedMathCore(FixedMathCore::ONE_RAW, true));
	return t * t * (FixedMathCore(3LL) - (t * FixedMathCore(2LL)));
}

// ============================================================================
// Transcendental Functional API
// ============================================================================

static _FORCE_INLINE_ FixedMathCore sqrt(FixedMathCore p_a) { return p_a.square_root(); }
static _FORCE_INLINE_ FixedMathCore sin(FixedMathCore p_a) { return p_a.sin(); }
static _FORCE_INLINE_ FixedMathCore cos(FixedMathCore p_a) { return p_a.cos(); }
static _FORCE_INLINE_ FixedMathCore exp(FixedMathCore p_a) { return p_a.exp(); }
static _FORCE_INLINE_ FixedMathCore log(FixedMathCore p_a) { return p_a.log(); }
static _FORCE_INLINE_ FixedMathCore pow(FixedMathCore p_base, int32_t p_exp) { return p_base.power(p_exp); }

// ============================================================================
// Vector Interaction Kernels
// ============================================================================

static _FORCE_INLINE_ FixedMathCore dot(const Vector3f &p_a, const Vector3f &p_b) { return p_a.dot(p_b); }
static _FORCE_INLINE_ Vector3f cross(const Vector3f &p_a, const Vector3f &p_b) { return p_a.cross(p_b); }
static _FORCE_INLINE_ Vector3f normalize(const Vector3f &p_a) { return p_a.normalized(); }
static _FORCE_INLINE_ FixedMathCore length(const Vector3f &p_a) { return p_a.length(); }

/**
 * reflect()
 * R = I - 2 * dot(N, I) * N
 */
static _FORCE_INLINE_ Vector3f reflect(const Vector3f &p_i, const Vector3f &p_n) {
	return p_i - p_n * (p_n.dot(p_i) * FixedMathCore(2LL));
}

// ============================================================================
// Sophisticated Relativistic & Physics Kernels
// ============================================================================

/**
 * lorentz_gamma()
 * Calculates the relativistic time dilation factor.
 * gamma = 1 / sqrt(1 - v^2/c^2)
 */
static _FORCE_INLINE_ FixedMathCore lorentz_gamma(const Vector3f &p_vel, const FixedMathCore &p_c) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore v2 = p_vel.length_squared();
	FixedMathCore c2 = p_c * p_c;
	FixedMathCore beta2 = clamp(v2 / c2, FixedMathCore(0LL, true), FixedMathCore(4294967290LL, true)); // Max 0.99999
	return one / (one - beta2).square_root();
}

/**
 * calculate_galactic_relative_pos()
 * Resolves bit-perfect distance between two points in different BigInt sectors.
 */
static _FORCE_INLINE_ Vector3f calculate_galactic_relative_pos(
		const Vector3f &p_pos_a, const BigIntCore &p_sx_a, const BigIntCore &p_sy_a, const BigIntCore &p_sz_a,
		const Vector3f &p_pos_b, const BigIntCore &p_sx_b, const BigIntCore &p_sy_b, const BigIntCore &p_sz_b,
		const FixedMathCore &p_sector_size) {

	BigIntCore dx = p_sx_b - p_sx_a;
	BigIntCore dy = p_sy_b - p_sy_a;
	BigIntCore dz = p_sz_b - p_sz_a;

	FixedMathCore off_x = FixedMathCore(static_cast<int64_t>(std::stoll(dx.to_string()))) * p_sector_size;
	FixedMathCore off_y = FixedMathCore(static_cast<int64_t>(std::stoll(dy.to_string()))) * p_sector_size;
	FixedMathCore off_z = FixedMathCore(static_cast<int64_t>(std::stoll(dz.to_string()))) * p_sector_size;

	return (p_pos_b + Vector3f(off_x, off_y, off_z)) - p_pos_a;
}

// ============================================================================
// Geometric Intersection Kernels
// ============================================================================

/**
 * intersect_sphere()
 * Deterministic ray-sphere intersection. Used for light-transport and sensors.
 */
static _FORCE_INLINE_ bool intersect_sphere(
		const Vector3f &p_ray_o, const Vector3f &p_ray_d, 
		const Vector3f &p_sph_o, const FixedMathCore &p_sph_r, 
		FixedMathCore &r_t) {
	
	Vector3f l = p_sph_o - p_ray_o;
	FixedMathCore tca = l.dot(p_ray_d);
	if (tca.get_raw() < 0) return false;
	
	FixedMathCore d2 = l.dot(l) - tca * tca;
	FixedMathCore r2 = p_sph_r * p_sph_r;
	if (d2 > r2) return false;
	
	FixedMathCore thc = (r2 - d2).square_root();
	r_t = tca - thc;
	return true;
}

} // namespace wp

#endif // WARP_INTRINSICS_H

--- END OF FILE core/math/warp_intrinsics.h ---
