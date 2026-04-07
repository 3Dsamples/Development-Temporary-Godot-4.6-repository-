--- START OF FILE core/math/transform.h ---

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "core/math/basis.h"
#include "core/math/vector3.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Transform Template
 * 
 * A 3x4 matrix representing a 3D transformation.
 * Optimized for Scale-Aware pipelines and batch-oriented math.
 * Aligned for SIMD-accelerated Warp kernels over EnTT component streams.
 */
template <typename T>
struct ET_ALIGN_32 Transform {
	Basis<T> basis;
	Vector3<T> origin;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Transform() {}

	ET_SIMD_INLINE Transform(const Basis<T> &p_basis, const Vector3<T> &p_origin) :
			basis(p_basis),
			origin(p_origin) {}

	ET_SIMD_INLINE Transform(
			T p_xx, T p_xy, T p_xz, 
			T p_yx, T p_yy, T p_yz, 
			T p_zx, T p_zy, T p_zz, 
			T p_ox, T p_oy, T p_oz) :
			basis(p_xx, p_xy, p_xz, p_yx, p_yy, p_yz, p_zx, p_zy, p_zz),
			origin(p_ox, p_oy, p_oz) {}

	// ------------------------------------------------------------------------
	// Deterministic Operators (Batch-Oriented)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3<T> xform(const Vector3<T> &p_v) const {
		return Vector3<T>(
				basis[0].dot(p_v) + origin.x,
				basis[1].dot(p_v) + origin.y,
				basis[2].dot(p_v) + origin.z);
	}

	ET_SIMD_INLINE Vector3<T> xform_inv(const Vector3<T> &p_v) const {
		Vector3<T> v = p_v - origin;
		return Vector3<T>(
				basis[0][0] * v.x + basis[1][0] * v.y + basis[2][0] * v.z,
				basis[0][1] * v.x + basis[1][1] * v.y + basis[2][1] * v.z,
				basis[0][2] * v.x + basis[1][2] * v.y + basis[2][2] * v.z);
	}

	ET_SIMD_INLINE void operator*=(const Transform<T> &p_transform) {
		origin = xform(p_transform.origin);
		basis *= p_transform.basis;
	}

	ET_SIMD_INLINE Transform<T> operator*(const Transform<T> &p_transform) const {
		Transform<T> t = *this;
		t *= p_transform;
		return t;
	}

	ET_SIMD_INLINE bool operator==(const Transform<T> &p_transform) const {
		return (basis == p_transform.basis && origin == p_transform.origin);
	}

	ET_SIMD_INLINE bool operator!=(const Transform<T> &p_transform) const {
		return (basis != p_transform.basis || origin != p_transform.origin);
	}

	// ------------------------------------------------------------------------
	// Deterministic Simulation API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE void affine_invert() {
		basis = basis.inverse();
		origin = basis.xform(-origin);
	}

	ET_SIMD_INLINE Transform<T> affine_inverted() const {
		Transform<T> t = *this;
		t.affine_invert();
		return t;
	}

	ET_SIMD_INLINE void rotate(const Vector3<T> &p_axis, T p_angle) {
		Transform<T> r(Basis<T>(p_axis, p_angle), Vector3<T>());
		*this = r * (*this);
	}

	ET_SIMD_INLINE void orthonormalize() {
		basis.orthonormalize();
	}

	/**
	 * interpolate_with()
	 * Deterministic SLERP for Basis and LERP for origin.
	 * Essential for 60/120 FPS Heartbeat synchronization.
	 */
	ET_SIMD_INLINE Transform<T> interpolate_with(const Transform<T> &p_to, T p_weight) const {
		// Basis uses SLERP internally via Quaternions
		return Transform<T>(basis.slerp(p_to.basis, p_weight), origin.lerp(p_to.origin, p_weight));
	}

	// Godot UI/Debug Conversion
	operator String() const {
		return "[Basis: " + (String)basis + ", Origin: " + (String)origin + "]";
	}
};

// Simulation Tier Aliases
typedef Transform<FixedMathCore> Transformf; // Bit-perfect local/planetary physics
typedef Transform<BigIntCore> Transformb;    // Discrete Macro-Logic transformations

#endif // TRANSFORM_H

--- END OF FILE core/math/transform.h ---
