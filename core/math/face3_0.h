--- START OF FILE core/math/face3.h ---

#ifndef FACE3_H
#define FACE3_H

#include "core/math/vector3.h"
#include "core/math/plane.h"
#include "src/fixed_math_core.h"

/**
 * Face3 Template
 * 
 * Foundational 3D triangle primitive for the Universal Solver.
 * Aligned to 32 bytes for EnTT SoA batching and SIMD-accelerated Warp kernels.
 * Provides deterministic geometric queries for collision and ray-casting.
 */
template <typename T>
struct ET_ALIGN_32 Face3 {
	Vector3<T> vertex[3];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Face3() {}
	ET_SIMD_INLINE Face3(const Vector3<T> &p_v1, const Vector3<T> &p_v2, const Vector3<T> &p_v3) {
		vertex[0] = p_v1;
		vertex[1] = p_v2;
		vertex[2] = p_v3;
	}

	// ------------------------------------------------------------------------
	// Geometric Queries (Deterministic)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3<T> get_normal() const {
		return (vertex[1] - vertex[0]).cross(vertex[2] - vertex[0]).normalized();
	}

	ET_SIMD_INLINE Plane<T> get_plane() const {
		return Plane<T>(vertex[0], vertex[1], vertex[2]);
	}

	ET_SIMD_INLINE Vector3<T> get_median() const {
		T third = MathConstants<T>::one() / FixedMathCore(3LL, false);
		return (vertex[0] + vertex[1] + vertex[2]) * third;
	}

	ET_SIMD_INLINE T get_area() const {
		return (vertex[1] - vertex[0]).cross(vertex[2] - vertex[0]).length() * MathConstants<T>::half();
	}

	// ------------------------------------------------------------------------
	// Collision & Intersection API (Batch Optimized)
	// ------------------------------------------------------------------------

	/**
	 * intersects_ray()
	 * Bit-perfect Moller-Trumbore intersection. 
	 * Essential for deterministic ray-casting in galactic-scale voxel grids.
	 */
	bool intersects_ray(const Vector3<T> &p_from, const Vector3<T> &p_dir, Vector3<T> *p_intersection = nullptr) const {
		Vector3<T> edge1 = vertex[1] - vertex[0];
		Vector3<T> edge2 = vertex[2] - vertex[0];
		Vector3<T> h = p_dir.cross(edge2);
		T a = edge1.dot(h);

		if (Math::abs(a) < T(42949LL, true)) { // Deterministic epsilon
			return false;
		}

		T f = MathConstants<T>::one() / a;
		Vector3<T> s = p_from - vertex[0];
		T u = f * s.dot(h);

		if (u < MathConstants<T>::zero() || u > MathConstants<T>::one()) {
			return false;
		}

		Vector3<T> q = s.cross(edge1);
		T v = f * p_dir.dot(q);

		if (v < MathConstants<T>::zero() || u + v > MathConstants<T>::one()) {
			return false;
		}

		T t = f * edge2.dot(q);
		if (t > T(42949LL, true)) {
			if (p_intersection) {
				*p_intersection = p_from + p_dir * t;
			}
			return true;
		}

		return false;
	}

	/**
	 * get_closest_point()
	 * Returns the point on the triangle closest to p_point.
	 * Optimized for high-frequency Warp kernel physics solvers.
	 */
	Vector3<T> get_closest_point(const Vector3<T> &p_point) const;

	// Godot UI Conversion
	operator String() const {
		return "(" + (String)vertex[0] + "), (" + (String)vertex[1] + "), (" + (String)vertex[2] + ")";
	}
};

typedef Face3<FixedMathCore> Face3f;

#endif // FACE3_H

--- END OF FILE core/math/face3.h ---
