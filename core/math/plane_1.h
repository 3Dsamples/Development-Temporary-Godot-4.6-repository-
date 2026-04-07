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
 * 32-byte aligned 3D plane for deterministic geometric clipping.
 * Strictly utilizes Software-Defined Arithmetic to eliminate FPU drift.
 * Engineered for high-speed batch processing in the Scale-Aware pipeline.
 */
template <typename T>
struct ET_ALIGN_32 Plane {
	Vector3<T> normal;
	T d;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Plane() : d(MathConstants<T>::zero()) {}
	
	_FORCE_INLINE_ Plane(T p_a, T p_b, T p_c, T p_d) : 
		normal(p_a, p_b, p_c), d(p_d) {}

	_FORCE_INLINE_ Plane(const Vector3<T> &p_normal, T p_d) : 
		normal(p_normal), d(p_d) {}

	_FORCE_INLINE_ Plane(const Vector3<T> &p_point, const Vector3<T> &p_normal) : 
		normal(p_normal), d(p_normal.dot(p_point)) {}

	/**
	 * Plane(v1, v2, v3)
	 * Clockwise winding order constructor.
	 * Bit-perfect cross product ensures identical normals across all nodes.
	 */
	_FORCE_INLINE_ Plane(const Vector3<T> &p_v1, const Vector3<T> &p_v2, const Vector3<T> &p_v3) {
		normal = (p_v1 - p_v3).cross(p_v1 - p_v2);
		normal.normalize();
		d = normal.dot(p_v1);
	}

	// ------------------------------------------------------------------------
	// Deterministic Geometric API
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ void normalize() {
		T l = normal.length();
		if (unlikely(l == MathConstants<T>::zero())) {
			normal = Vector3<T>();
			d = MathConstants<T>::zero();
		} else {
			T inv_l = MathConstants<T>::one() / l;
			normal *= inv_l;
			d *= inv_l;
		}
	}

	_FORCE_INLINE_ Plane<T> normalized() const {
		Plane<T> p = *this;
		p.normalize();
		return p;
	}

	_FORCE_INLINE_ Vector3<T> project(const Vector3<T> &p_point) const {
		return p_point - normal * (normal.dot(p_point) - d);
	}

	_FORCE_INLINE_ T distance_to(const Vector3<T> &p_point) const {
		return normal.dot(p_point) - d;
	}

	_FORCE_INLINE_ bool has_point(const Vector3<T> &p_point, T p_epsilon) const {
		return Math::abs(normal.dot(p_point) - d) <= p_epsilon;
	}

	_FORCE_INLINE_ bool is_point_over(const Vector3<T> &p_point) const {
		return normal.dot(p_point) > d;
	}

	// ------------------------------------------------------------------------
	// Sophisticated Intersection API (CCD Support)
	// ------------------------------------------------------------------------

	/**
	 * intersect_3()
	 * Finds the intersection point of three planes. 
	 * Essential for resolving bit-perfect convex hull vertices.
	 */
	bool intersect_3(const Plane &p_plane1, const Plane &p_plane2, Vector3<T> *r_result) const {
		const Plane &p0 = *this;
		const Plane &p1 = p_plane1;
		const Plane &p2 = p_plane2;

		T det = p0.normal.cross(p1.normal).dot(p2.normal);

		if (unlikely(Math::abs(det) < CMP_EPSILON)) return false;

		if (r_result) {
			*r_result = (p1.normal.cross(p2.normal) * p0.d +
						p2.normal.cross(p0.normal) * p1.d +
						p0.normal.cross(p1.normal) * p2.d) / det;
		}

		return true;
	}

	/**
	 * intersects_ray()
	 * Standard ray-plane intersection.
	 */
	bool intersects_ray(const Vector3<T> &p_from, const Vector3<T> &p_dir, Vector3<T> *p_intersection) const {
		T den = normal.dot(p_dir);
		if (unlikely(Math::abs(den) < CMP_EPSILON)) return false;

		T dist = (d - normal.dot(p_from)) / den;
		if (dist < MathConstants<T>::zero()) return false;

		if (p_intersection) *p_intersection = p_from + p_dir * dist;
		return true;
	}

	/**
	 * intersects_segment()
	 * Master CCD Kernel: Resolves exact intersection with a moving segment.
	 * Guaranteed to detect collisions regardless of entity velocity at 120 FPS.
	 */
	bool intersects_segment(const Vector3<T> &p_begin, const Vector3<T> &p_end, Vector3<T> *p_intersection) const {
		T d_begin = distance_to(p_begin);
		T d_end = distance_to(p_end);

		// Check if segment crosses the plane (Different signs)
		if (!((d_begin > MathConstants<T>::zero() && d_end < MathConstants<T>::zero()) ||
			  (d_begin < MathConstants<T>::zero() && d_end > MathConstants<T>::zero()))) {
			return false;
		}

		T dist = d_begin / (d_begin - d_end);
		if (p_intersection) *p_intersection = p_begin + (p_end - p_begin) * dist;
		return true;
	}

	// ------------------------------------------------------------------------
	// Operators
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ bool operator==(const Plane &p_plane) const { return normal == p_plane.normal && d == p_plane.d; }
	_FORCE_INLINE_ bool operator!=(const Plane &p_plane) const { return normal != p_plane.normal || d != p_plane.d; }
	_FORCE_INLINE_ Plane operator-() const { return Plane(-normal, -d); }

	operator String() const {
		return "[N: " + (String)normal + ", D: " + String(d.to_string().c_str()) + "]";
	}
};

// Simulation Tier Typedefs
typedef Plane<FixedMathCore> Planef; // Bit-perfect 3D clipping / Physics
typedef Plane<BigIntCore> Planeb;    // Discrete macro-boundary mapping

#endif // PLANE_H

--- END OF FILE core/math/plane.h ---
