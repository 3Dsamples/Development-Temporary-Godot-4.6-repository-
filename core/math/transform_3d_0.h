--- START OF FILE core/math/transform_3d.h ---

#ifndef TRANSFORM_3D_H
#define TRANSFORM_3D_H

#include "core/math/basis.h"
#include "core/math/vector3.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Transform3D Template
 * 
 * The foundational 3D spatial transformation (TRS) for the Universal Solver.
 * Aligned to 32 bytes to ensure that EnTT component streams are SIMD-optimized
 * for high-frequency Warp kernel processing across all scales.
 */
template <typename T>
struct ET_ALIGN_32 Transform3D {
	Basis<T> basis;
	Vector3<T> origin;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Transform3D() {}

	ET_SIMD_INLINE Transform3D(const Basis<T> &p_basis, const Vector3<T> &p_origin) :
			basis(p_basis),
			origin(p_origin) {}

	ET_SIMD_INLINE Transform3D(
			T p_xx, T p_xy, T p_xz, 
			T p_yx, T p_yy, T p_yz, 
			T p_zx, T p_zy, T p_zz, 
			T p_ox, T p_oy, T p_oz) {
		basis = Basis<T>(p_xx, p_xy, p_xz, p_yx, p_yy, p_yz, p_zx, p_zy, p_zz);
		origin = Vector3<T>(p_ox, p_oy, p_oz);
	}

	// ------------------------------------------------------------------------
	// Operators (Batch Transformation Logic)
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

	ET_SIMD_INLINE void operator*=(const Transform3D<T> &p_transform) {
		origin = xform(p_transform.origin);
		basis *= p_transform.basis;
	}

	ET_SIMD_INLINE Transform3D<T> operator*(const Transform3D<T> &p_transform) const {
		Transform3D<T> t = *this;
		t *= p_transform;
		return t;
	}

	ET_SIMD_INLINE bool operator==(const Transform3D<T> &p_transform) const {
		return (basis == p_transform.basis && origin == p_transform.origin);
	}

	ET_SIMD_INLINE bool operator!=(const Transform3D<T> &p_transform) const {
		return (basis != p_transform.basis || origin != p_transform.origin);
	}

	// ------------------------------------------------------------------------
	// Deterministic Transformation API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE void affine_invert() {
		basis = basis.inverse();
		origin = basis.xform(-origin);
	}

	ET_SIMD_INLINE Transform3D<T> affine_inverted() const {
		Transform3D<T> t = *this;
		t.affine_invert();
		return t;
	}

	ET_SIMD_INLINE void rotate(const Vector3<T> &p_axis, T p_angle) {
		*this = rotated(p_axis, p_angle);
	}

	ET_SIMD_INLINE Transform3D<T> rotated(const Vector3<T> &p_axis, T p_angle) const {
		return Transform3D<T>(Basis<T>(p_axis, p_angle), Vector3<T>()) * (*this);
	}

	ET_SIMD_INLINE void scale(const Vector3<T> &p_scale) {
		basis.scale(p_scale);
		origin *= p_scale;
	}

	ET_SIMD_INLINE Transform3D<T> scaled(const Vector3<T> &p_scale) const {
		Transform3D<T> t = *this;
		t.scale(p_scale);
		return t;
	}

	ET_SIMD_INLINE void translate(const Vector3<T> &p_translation) {
		origin += basis.xform(p_translation);
	}

	ET_SIMD_INLINE Transform3D<T> translated(const Vector3<T> &p_translation) const {
		Transform3D<T> t = *this;
		t.translate(p_translation);
		return t;
	}

	ET_SIMD_INLINE void orthonormalize() {
		basis.orthonormalize();
	}

	void set_look_at(const Vector3<T> &p_eye, const Vector3<T> &p_target, const Vector3<T> &p_up) {
		Vector3<T> v_z = (p_eye - p_target).normalized();
		Vector3<T> v_x = p_up.cross(v_z).normalized();
		Vector3<T> v_y = v_z.cross(v_x).normalized();

		basis.set_column(0, v_x);
		basis.set_column(1, v_y);
		basis.set_column(2, v_z);
		origin = p_eye;
	}

	ET_SIMD_INLINE bool is_equal_approx(const Transform3D<T> &p_transform) const {
		return basis.is_equal_approx(p_transform.basis) && origin.is_equal_approx(p_transform.origin);
	}

	// Godot UI Conversion
	operator String() const {
		return "[Basis: " + (String)basis + ", Origin: " + (String)origin + "]";
	}
};

// Simulation Type Aliases
typedef Transform3D<FixedMathCore> Transform3Df; // Primary Deterministic Physics
typedef Transform3D<BigIntCore> Transform3Db;    // Macro-Scale Universe Grids

#endif // TRANSFORM_3D_H

--- END OF FILE core/math/transform_3d.h ---
