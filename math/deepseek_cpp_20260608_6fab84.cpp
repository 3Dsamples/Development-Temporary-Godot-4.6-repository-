// File 243: modules/newton/src/world/newton_world_query.h
// NewtonWorldQuery – performs spatial queries against all bodies in the
// NewtonWorld: ray-casts, shape sweeps, sphere overlaps, and AABB overlaps.
// Uses Gaia BVH for broad‑phase and GJK for exact convex‑shape tests.

#ifndef NEWTON_WORLD_QUERY_H
#define NEWTON_WORLD_QUERY_H

#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../bodies/newton_body.h"
#include "../collision/newton_collision.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h"
#include "core/templates/local_vector.h"

namespace newton {

class NewtonWorld;

class NewtonWorldQuery {
public:
	// Single ray-cast result.
	struct RayHit {
		body_id body;
		vec3 point;
		vec3 normal;
		real_t distance;
	};

	// Sweep result.
	struct SweepHit {
		body_id body;
		real_t fraction;         // [0,1] fraction along the sweep where hit occurred
		vec3 point;
		vec3 normal;
	};

	// Overlap result.
	struct OverlapResult {
		body_id body;
		int shape_count;
	};

	// Ray-cast against all active bodies.
	static LocalVector<RayHit> ray_cast(const NewtonWorld *p_world,
										const vec3 &p_origin, const vec3 &p_direction,
										real_t p_max_distance = INFINITY);

	// Sweep a convex shape through the world.
	static LocalVector<SweepHit> sweep_shape(const NewtonWorld *p_world,
											 const NewtonCollision &p_shape,
											 const mat4 &p_start_transform,
											 const vec3 &p_end_translation,
											 real_t p_margin = 0.0);

	// Overlap test with a sphere.
	static LocalVector<OverlapResult> sphere_overlap(const NewtonWorld *p_world,
													 const vec3 &p_center, real_t p_radius);

	// Overlap test with an AABB.
	static LocalVector<OverlapResult> aabb_overlap(const NewtonWorld *p_world,
												   const AABB &p_aabb);
};

} // namespace newton

#endif // NEWTON_WORLD_QUERY_H