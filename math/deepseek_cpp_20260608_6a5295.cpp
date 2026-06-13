// File 226: modules/newton/src/utils/newton_ray_cast.h
// Ray-cast utility against Newton world bodies using Gaia BVH for acceleration.
// Returns hit information, body ID, contact point, and normal.

#ifndef NEWTON_UTILS_RAY_CAST_H
#define NEWTON_UTILS_RAY_CAST_H

#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../bodies/newton_body.h"
#include "../world/newton_world.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/query.h"

namespace newton {

class RayCastResult {
public:
	bool hit = false;
	body_id body_id = 0;
	vec3 point;               // world-space hit point
	vec3 normal;              // surface normal at hit point (pointing outward from body)
	real_t distance;          // distance from ray origin to hit
};

class NewtonRayCast {
public:
	// Perform a ray-cast against all active bodies in the world.
	static RayCastResult cast_ray(NewtonWorld *p_world, const vec3 &p_origin, const vec3 &p_direction, real_t p_max_distance = INFINITY) {
		RayCastResult result;
		result.distance = p_max_distance;

		// Build Gaia BVH from all body AABBs (using existing NewtonBody AABBs)
		gaia::bvh::BVH bvh;
		LocalVector<body_id> body_ids = p_world->get_body_ids();
		LocalVector<AABB> aabbs;
		LocalVector<body_id> id_map; // maps bvh primitive index to body ID
		for (body_id id : body_ids) {
			Ref<NewtonBody> body = p_world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;
			aabbs.push_back(body->get_aabb());
			id_map.push_back(id);
		}
		if (aabbs.is_empty()) return result;
		bvh.build_final(aabbs);

		// Normalised direction
		vec3 dir = p_direction.normalized();

		// Query BVH for intersection with ray
		AABB ray_aabb(p_origin, vec3());
		vec3 ray_end = p_origin + dir * p_max_distance;
		ray_aabb.expand_to(ray_end);

		real_t best_t = p_max_distance;
		bvh.query_intersect(ray_aabb, [&](int prim_idx) {
			if (prim_idx < 0 || prim_idx >= id_map.size()) return;
			body_id id = id_map[prim_idx];
			Ref<NewtonBody> body = p_world->get_body(id);
			if (body.is_null()) return;

			const AABB &box = aabbs[prim_idx];
			real_t t_entry, t_exit;
			// Ray-AABB intersection test
			if (gaia::bvh::intersect_ray_aabb(p_origin, dir, box, 0.0, best_t, t_entry, t_exit)) {
				if (t_entry < best_t) {
					best_t = t_entry;
					result.hit = true;
					result.body_id = id;
					result.point = p_origin + dir * t_entry;
					// Compute normal from AABB: find which face was hit.
					// Simplified: use the normal of the closest face.
					vec3 local_point = body->get_transform().affine_inverse().xform(result.point);
					vec3 half_extents = box.size * 0.5;
					vec3 center = box.position + half_extents;
					vec3 d = local_point - center;
					real_t min_dist = INFINITY;
					vec3 best_normal(0, 0, 0);
					// Check faces
					for (int i = 0; i < 3; ++i) {
						real_t dist = Math::abs(half_extents[i] - Math::abs(d[i]));
						if (dist < min_dist) {
							min_dist = dist;
							best_normal = vec3();
							best_normal[i] = (d[i] > 0) ? 1.0f : -1.0f;
						}
					}
					result.normal = body->get_rotation().xform(best_normal);
					result.distance = best_t;
				}
			}
		});

		return result;
	}
};

} // namespace newton

#endif // NEWTON_UTILS_RAY_CAST_H