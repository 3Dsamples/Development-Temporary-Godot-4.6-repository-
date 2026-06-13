// aabb.h – Gaia BVH module for Godot 4.6
// Adapted from Gaia’s aabb.cuh to work with Godot’s core math types.

#ifndef GAIA_BVH_AABB_H
#define GAIA_BVH_AABB_H

#include "core/math/aabb.h"
#include "core/math/vector3.h"

// Gaia-style AABB operations implemented as free functions that accept Godot’s AABB.
// This keeps the original Gaia algorithms intact but uses Godot’s well-optimized AABB class.

namespace gaia::bvh {

/**
 * Test if two Godot AABBs intersect (same logic as Gaia’s CUDA version).
 */
inline bool intersects(const AABB &p_lhs, const AABB &p_rhs) {
	// Original Gaia code:
	// return lhs.upper.x >= rhs.lower.x && rhs.upper.x >= lhs.lower.x && ...
	return p_lhs.position.x + p_lhs.size.x >= p_rhs.position.x &&
	       p_rhs.position.x + p_rhs.size.x >= p_lhs.position.x &&
	       p_lhs.position.y + p_lhs.size.y >= p_rhs.position.y &&
	       p_rhs.position.y + p_rhs.size.y >= p_lhs.position.y &&
	       p_lhs.position.z + p_lhs.size.z >= p_rhs.position.z &&
	       p_rhs.position.z + p_rhs.size.z >= p_lhs.position.z;
}

/**
 * Merge two AABBs, returning the minimal AABB that encloses both.
 */
inline AABB merge(const AABB &p_lhs, const AABB &p_rhs) {
	AABB merged;
	merged.position.x = MIN(p_lhs.position.x, p_rhs.position.x);
	merged.position.y = MIN(p_lhs.position.y, p_rhs.position.y);
	merged.position.z = MIN(p_lhs.position.z, p_rhs.position.z);

	Vector3 lhs_end = p_lhs.position + p_lhs.size;
	Vector3 rhs_end = p_rhs.position + p_rhs.size;

	merged.size.x = MAX(lhs_end.x, rhs_end.x) - merged.position.x;
	merged.size.y = MAX(lhs_end.y, rhs_end.y) - merged.position.y;
	merged.size.z = MAX(lhs_end.z, rhs_end.z) - merged.position.z;
	return merged;
}

/**
 * Minimum distance from a point to an AABB (original db_MIN_DIST from Gaia).
 */
inline real_t min_dist(const AABB &p_box, const Vector3 &p_point) {
	Vector3 lower = p_box.position;
	Vector3 upper = p_box.position + p_box.size;

	real_t dx = MAX(lower.x - p_point.x, MIN(upper.x - p_point.x, 0.0));
	real_t dy = MAX(lower.y - p_point.y, MIN(upper.y - p_point.y, 0.0));
	real_t dz = MAX(lower.z - p_point.z, MIN(upper.z - p_point.z, 0.0));
	return dx * dx + dy * dy + dz * dz;
}

/**
 * Squared min-max distance (used in nearest neighbor queries, from Roussopoulos et al. 1995).
 */
inline real_t min_max_dist(const AABB &p_box, const Vector3 &p_point) {
	Vector3 lower = p_box.position;
	Vector3 upper = p_box.position + p_box.size;

	real_t dx = MIN(ABS(p_point.x - lower.x), ABS(p_point.x - upper.x));
	real_t dy = MIN(ABS(p_point.y - lower.y), ABS(p_point.y - upper.y));
	real_t dz = MIN(ABS(p_point.z - lower.z), ABS(p_point.z - upper.z));
	return dx * dx + dy * dy + dz * dz;
}

} // namespace gaia::bvh

#endif // GAIA_BVH_AABB_H