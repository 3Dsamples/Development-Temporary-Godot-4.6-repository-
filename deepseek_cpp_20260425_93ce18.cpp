// File 04: modules/gaia/src/bvh/query.h

#ifndef GAIA_BVH_QUERY_H
#define GAIA_BVH_QUERY_H

#include "aabb.h"

#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace gaia::bvh {

// ---------------------------------------------------------------------------
// Ray – AABB intersection (slab method)
// Returns true if the ray hits the box, and outputs the entry/exit t values.
// Ray origin: `origin`, direction: `dir` (normalised or not; works unnormalised).
// t_min/t_max are the clipping limits of the ray; typically (0, INFINITY).
// ---------------------------------------------------------------------------
inline bool intersect_ray_aabb(const Vector3 &origin, const Vector3 &dir,
		const AABB &box, real_t t_min, real_t t_max,
		real_t &r_t_entry, real_t &r_t_exit) {
	real_t t1 = (box.position.x - origin.x) / dir.x;
	real_t t2 = (box.position.x + box.size.x - origin.x) / dir.x;
	if (dir.x < 0) SWAP(t1, t2);
	real_t t_enter = MAX(t1, t_min);
	real_t t_exit = MIN(t2, t_max);
	if (t_enter > t_exit) return false;

	t1 = (box.position.y - origin.y) / dir.y;
	t2 = (box.position.y + box.size.y - origin.y) / dir.y;
	if (dir.y < 0) SWAP(t1, t2);
	t_enter = MAX(t_enter, t1);
	t_exit = MIN(t_exit, t2);
	if (t_enter > t_exit) return false;

	t1 = (box.position.z - origin.z) / dir.z;
	t2 = (box.position.z + box.size.z - origin.z) / dir.z;
	if (dir.z < 0) SWAP(t1, t2);
	t_enter = MAX(t_enter, t1);
	t_exit = MIN(t_exit, t2);
	if (t_enter > t_exit) return false;

	r_t_entry = t_enter;
	r_t_exit = t_exit;
	return true;
}

// ---------------------------------------------------------------------------
// Ray – Triangle intersection (Möller–Trumbore)
// Returns true if the ray intersects the triangle, and outputs the t parameter
// and barycentric coordinates u, v (w = 1-u-v).
// ---------------------------------------------------------------------------
inline bool intersect_ray_triangle(const Vector3 &origin, const Vector3 &dir,
		const Vector3 &v0, const Vector3 &v1, const Vector3 &v2,
		real_t &r_t, real_t &r_u, real_t &r_v) {
	Vector3 edge1 = v1 - v0;
	Vector3 edge2 = v2 - v0;
	Vector3 h = dir.cross(edge2);
	real_t a = edge1.dot(h);
	if (ABS(a) < CMP_EPSILON) return false; // ray parallel to triangle

	real_t f = 1.0 / a;
	Vector3 s = origin - v0;
	r_u = f * s.dot(h);
	if (r_u < 0.0 || r_u > 1.0) return false;

	Vector3 q = s.cross(edge1);
	r_v = f * dir.dot(q);
	if (r_v < 0.0 || r_u + r_v > 1.0) return false;

	r_t = f * edge2.dot(q);
	return r_t > CMP_EPSILON; // intersection at positive t
}

// ---------------------------------------------------------------------------
// Closest point on triangle to a given point
// Returns the point on the triangle (including edges/vertices) that is closest to `p`.
// Outputs barycentric coordinates if needed.
// ---------------------------------------------------------------------------
inline Vector3 closest_point_on_triangle(const Vector3 &p, const Vector3 &a,
		const Vector3 &b, const Vector3 &c,
		real_t *r_u = nullptr, real_t *r_v = nullptr) {
	Vector3 ab = b - a;
	Vector3 ac = c - a;
	Vector3 ap = p - a;
	real_t d1 = ab.dot(ap);
	real_t d2 = ac.dot(ap);
	if (d1 <= 0.0 && d2 <= 0.0) {
		if (r_u) *r_u = 0.0;
		if (r_v) *r_v = 0.0;
		return a;
	}

	Vector3 bp = p - b;
	real_t d3 = ab.dot(bp);
	real_t d4 = ac.dot(bp);
	if (d3 >= 0.0 && d4 <= d3) {
		if (r_u) *r_u = 1.0;
		if (r_v) *r_v = 0.0;
		return b;
	}

	real_t vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
		real_t v = d1 / (d1 - d3);
		if (r_u) *r_u = v;
		if (r_v) *r_v = 0.0;
		return a + ab * v;
	}

	Vector3 cp = p - c;
	real_t d5 = ab.dot(cp);
	real_t d6 = ac.dot(cp);
	if (d6 >= 0.0 && d5 <= d6) {
		if (r_u) *r_u = 0.0;
		if (r_v) *r_v = 1.0;
		return c;
	}

	real_t vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
		real_t w = d2 / (d2 - d6);
		if (r_u) *r_u = 0.0;
		if (r_v) *r_v = w;
		return a + ac * w;
	}

	real_t va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
		real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		if (r_u) *r_u = 1.0 - w;
		if (r_v) *r_v = w;
		return b + (c - b) * w;
	}

	// Inside face
	real_t denom = 1.0 / (va + vb + vc);
	real_t v = vb * denom;
	real_t w = vc * denom;
	if (r_u) *r_u = v;
	if (r_v) *r_v = w;
	return a + ab * v + ac * w;
}

// ---------------------------------------------------------------------------
// Squared distance from point to triangle
// ---------------------------------------------------------------------------
inline real_t point_triangle_distance_squared(const Vector3 &p,
		const Vector3 &a, const Vector3 &b, const Vector3 &c) {
	Vector3 closest = closest_point_on_triangle(p, a, b, c);
	return p.distance_squared_to(closest);
}

} // namespace gaia::bvh

#endif // GAIA_BVH_QUERY_H