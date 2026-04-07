--- START OF FILE core/math/plane.h ---

#ifndef PLANE_H
#define PLANE_H

#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Plane Template
 * 
 * Represents a 3D plane in Hessian Normal Form (Normal + Distance).
 * Aligned to 32 bytes to ensure that EnTT component streams are SIMD-optimized
 * for high-frequency Warp kernel clipping and culling sweeps.
 */
template <typename T>
struct ET_ALIGN_32 Plane {
	Vector3<T> normal;
	T d;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Plane() : d(T()) {}
	
	ET_SIMD_INLINE Plane(T p_a, T p_b, T p_c, T p_d) : 
		normal(p_a, p_b, p_c), d(p_d) {}

	ET_SIMD_INLINE Plane(const Vector3<T> &p_normal, T p_d) : 
		normal(p_normal), d(p_d) {}

	ET_SIMD_INLINE Plane(const Vector3<T> &p_point, const Vector3<T> &p_normal) : 
		normal(p_normal), d(p_normal.dot(p_point)) {}

	inline Plane(const Vector3<T> &p_v1, const Vector3<T> &p_v2, const Vector3<T> &p_v3) {
		normal = (p_v1 - p_v3).cross(p_v1 - p_v2);
		normal.normalize();
		d = normal.dot(p_v1);
	}

	// ------------------------------------------------------------------------
	// Deterministic Geometric API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE void normalize() {
		T l = normal.length();
		if (l == MathConstants<T>::zero()) {
			normal = Vector3<T>();
			d = MathConstants<T>::zero();
		} else {
			T inv_l = MathConstants<T>::one() / l;
			normal *= inv_l;
			d *= inv_l;
		}
	}

	ET_SIMD_INLINE Plane<T> normalized() const {
		Plane<T> p = *this;
		p.normalize();
		return p;
	}

	ET_SIMD_INLINE Vector3<T> project(const Vector3<T> &p_point) const {
		return p_point - normal * (normal.dot(p_point) - d);
	}

	ET_SIMD_INLINE T distance_to(const Vector3<T> &p_point) const {
		return normal.dot(p_point) - d;
	}

	ET_SIMD_INLINE bool has_point(const Vector3<T> &p_point, T p_epsilon) const {
		return Math::abs(normal.dot(p_point) - d) <= p_epsilon;
	}

	ET_SIMD_INLINE bool is_point_over(const Vector3<T> &p_point) const {
		return normal.dot(p_point) > d;
	}

	// ------------------------------------------------------------------------
	// Intersection API (Batch Optimized)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE bool intersects_ray(const Vector3<T> &p_from, const Vector3<T> &p_dir, Vector3<T> *p_intersection) const {
		T den = normal.dot(p_dir);
		if (Math::abs(den) < T(42949LL, true)) { // Deterministic epsilon
			return false;
		}
		T dist = (d - normal.dot(p_from)) / den;
		if (dist < MathConstants<T>::zero()) {
			return false;
		}
		if (p_intersection) {
			*p_intersection = p_from + p_dir * dist;
		}
		return true;
	}

	ET_SIMD_INLINE bool intersects_segment(const Vector3<T> &p_begin, const Vector3<T> &p_end, Vector3<T> *p_intersection) const {
		Vector3<T> segment = p_end - p_begin;
		T den = normal.dot(segment);
		if (Math::abs(den) < T(42949LL, true)) {
			return false;
		}
		T dist = (d - normal.dot(p_begin)) / den;
		if (dist < MathConstants<T>::zero() || dist > MathConstants<T>::one()) {
			return false;
		}
		if (p_intersection) {
			*p_intersection = p_begin + segment * dist;
		}
		return true;
	}

	// ------------------------------------------------------------------------
	// Operators
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE bool operator==(const Plane &p_plane) const { return normal == p_plane.normal && d == p_plane.d; }
	ET_SIMD_INLINE bool operator!=(const Plane &p_plane) const { return normal != p_plane.normal || d != p_plane.d; }
	ET_SIMD_INLINE Plane operator-() const { return Plane(-normal, -d); }

	// Godot UI Conversion
	operator String() const {
		return "[N: " + (String)normal + ", D: " + String(d.to_string().c_str()) + "]";
	}
};

typedef Plane<FixedMathCore> Planef; // Deterministic Culling/Physics
typedef Plane<BigIntCore> Planeb;    // Macro-Scale Boundary Mapping

#endif // PLANE_H

--- END OF FILE core/math/plane.h ---
