// File 40: modules/gaia/src/utility/math_utils.h

#ifndef GAIA_UTILITY_MATH_UTILS_H
#define GAIA_UTILITY_MATH_UTILS_H

#include "core/math/vector3.h"
#include "core/math/vector2.h"
#include "core/typedefs.h"

namespace gaia::math_utils {

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

// Clamp value between lo and hi.
inline real_t clamp(real_t value, real_t lo, real_t hi) {
	return CLAMP(value, lo, hi);
}

// Linear interpolation between a and b.
inline real_t lerp(real_t a, real_t b, real_t t) {
	return Math::lerp(a, b, t);
}

// Smoothstep (Hermite interpolation) – returns 0..1 smooth curve.
inline real_t smoothstep(real_t edge0, real_t edge1, real_t x) {
	real_t t = CLAMP((x - edge0) / (edge1 - edge0), 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}

// Sign function: returns -1, 0, or +1.
inline real_t sign(real_t x) {
	return (x > 0.0) - (x < 0.0);
}

// Step function: 0 if x < edge else 1.
inline real_t step(real_t edge, real_t x) {
	return x < edge ? 0.0 : 1.0;
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

// Component-wise clamp for Vector2.
inline Vector2 clamp_vec2(const Vector2 &v, const Vector2 &lo, const Vector2 &hi) {
	return Vector2(
		CLAMP(v.x, lo.x, hi.x),
		CLAMP(v.y, lo.y, hi.y)
	);
}

// Component-wise clamp for Vector3.
inline Vector3 clamp_vec3(const Vector3 &v, const Vector3 &lo, const Vector3 &hi) {
	return Vector3(
		CLAMP(v.x, lo.x, hi.x),
		CLAMP(v.y, lo.y, hi.y),
		CLAMP(v.z, lo.z, hi.z)
	);
}

// Lerp for Vector3.
inline Vector3 lerp_vec3(const Vector3 &a, const Vector3 &b, real_t t) {
	return a.lerp(b, t);
}

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

// Compute triangle area (2D cross product version for Vector2 or 3D using edge lengths).
// We'll provide a 3D triangle area from three points.
inline real_t triangle_area(const Vector3 &a, const Vector3 &b, const Vector3 &c) {
	return 0.5 * (b - a).cross(c - a).length();
}

// Signed angle between two 2D vectors (in radians).
inline real_t signed_angle_2d(const Vector2 &from, const Vector2 &to) {
	real_t cross = from.x * to.y - from.y * to.x;
	real_t dot = from.dot(to);
	return Math::atan2(cross, dot);
}

// Project a point onto a line segment, returning the parameter t in [0,1].
inline real_t closest_point_on_segment_t(const Vector3 &point,
										 const Vector3 &seg_a, const Vector3 &seg_b) {
	Vector3 ab = seg_b - seg_a;
	real_t len_sq = ab.length_squared();
	if (len_sq < CMP_EPSILON) return 0.0;
	return CLAMP((point - seg_a).dot(ab) / len_sq, 0.0, 1.0);
}

// Closest point on a segment.
inline Vector3 closest_point_on_segment(const Vector3 &point,
										const Vector3 &seg_a, const Vector3 &seg_b) {
	real_t t = closest_point_on_segment_t(point, seg_a, seg_b);
	return seg_a + (seg_b - seg_a) * t;
}

// Distance from point to segment.
inline real_t point_segment_distance(const Vector3 &point,
									 const Vector3 &seg_a, const Vector3 &seg_b) {
	return point.distance_to(closest_point_on_segment(point, seg_a, seg_b));
}

// ---------------------------------------------------------------------------
// Approximate comparison helpers
// ---------------------------------------------------------------------------

// Check if two reals are nearly equal.
inline bool approx_equal(real_t a, real_t b, real_t epsilon = CMP_EPSILON) {
	return Math::abs(a - b) <= epsilon;
}

// Check if two Vector3s are nearly equal.
inline bool approx_equal_vec3(const Vector3 &a, const Vector3 &b, real_t epsilon = CMP_EPSILON) {
	return a.distance_squared_to(b) <= epsilon * epsilon;
}

// ---------------------------------------------------------------------------
// Transformation helpers (extras beyond Godot's built-in)
// ---------------------------------------------------------------------------

// Rotate a vector around an arbitrary axis (Rodrigues formula).
inline Vector3 rotate_around_axis(const Vector3 &v, const Vector3 &axis, real_t angle) {
	real_t cos_a = Math::cos(angle);
	real_t sin_a = Math::sin(angle);
	return v * cos_a + axis.cross(v) * sin_a + axis * axis.dot(v) * (1.0 - cos_a);
}

} // namespace gaia::math_utils

#endif // GAIA_UTILITY_MATH_UTILS_H