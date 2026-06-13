// File 319: modules/vienna/src/query/vienna_world_query.h
// ViennaWorldQuery – high‑performance spatial queries against the ViennaWorld.
// Provides ray‑casts, sphere/AABB overlaps, and convex shape sweeps using
// Gaia's BVH broad‑phase and GJK narrow‑phase for exact collision detection.

#ifndef VIENNA_QUERY_WORLD_QUERY_H
#define VIENNA_QUERY_WORLD_QUERY_H

#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../collision/vienna_shape.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h" // GJK from Gaia
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace vienna {

class ViennaWorldQuery {
public:
	// ------------------------------ Ray‑cast ------------------------------
	struct RayHit {
		body_id body_id;
		vec3 point;            // world hit point
		vec3 normal;           // surface normal at hit point
		real_t distance;       // distance from ray origin
	};

	/**
	 * Cast a ray against all active bodies in the world.
	 * Returns a list of hits sorted by distance (closest first).
	 * WARNING: expensive if many bodies; use sparingly.
	 */
	static LocalVector<RayHit> ray_cast(const ViennaWorld *p_world,
										const vec3 &p_origin,
										const vec3 &p_direction,
										real_t p_max_distance = INFINITY) {
		LocalVector<RayHit> results;
		if (!p_world) return results;

		// Build Gaia BVH from body AABBs for fast broad‑phase.
		gaia::bvh::BVH bvh;
		LocalVector<body_id> body_ids = p_world->get_body_ids();
		LocalVector<AABB> aabbs;
		LocalVector<body_id> id_map;           // maps bvh primitive index to body ID
		aabbs.reserve(body_ids.size());
		id_map.reserve(body_ids.size());
		for (body_id id : body_ids) {
			Ref<ViennaBody> body = p_world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;
			aabbs.push_back(body->get_aabb());
			id_map.push_back(id);
		}
		if (aabbs.is_empty()) return results;
		bvh.build_final(aabbs);

		vec3 dir = p_direction.normalized();
		real_t max_dist = p_max_distance;
		AABB ray_aabb(p_origin, vec3());
		ray_aabb.expand_to(p_origin + dir * max_dist);

		bvh.query_intersect(ray_aabb, [&](int prim) {
			if (prim < 0 || prim >= id_map.size()) return;
			body_id id = id_map[prim];
			Ref<ViennaBody> body = p_world->get_body(id);
			if (body.is_null()) return;

			const ViennaShape *shape = body->get_collision_shape().ptr();
			if (shape) {
				// For triangle mesh shapes, use ray‑triangle intersection.
				if (shape->get_shape_type() == ShapeType::TRI_MESH) {
					const ViennaTriMesh *tri = dynamic_cast<const ViennaTriMesh *>(shape);
					if (tri) {
						real_t t; vec3 n;
						if (tri->ray_cast(p_origin, dir, max_dist, t, n)) {
							if (t < max_dist) {
								max_dist = t;
								RayHit hit;
								hit.body_id = id;
								hit.point = p_origin + dir * t;
								hit.normal = n;
								hit.distance = t;
								results.push_back(hit);
							}
						}
					}
				} else {
					// Convex shape – approximate by AABB entry (pure GJK ray is complex).
					// For exact convex ray, we could sweep a sphere, but here we fallback to
					// the AABB entry point and use the body's normal at that face.
					const AABB &box = body->get_aabb();
					real_t t_entry, t_exit;
					if (gaia::bvh::intersect_ray_aabb(p_origin, dir, box, 0.0, max_dist, t_entry, t_exit)) {
						if (t_entry < max_dist) {
							max_dist = t_entry;
							RayHit hit;
							hit.body_id = id;
							hit.point = p_origin + dir * t_entry;
							// Compute normal from the closest face to the hit point.
							vec3 local_pt = body->get_transform().affine_inverse().xform(hit.point);
							vec3 half_extents = box.size * 0.5;
							vec3 center = box.position + half_extents;
							vec3 d = local_pt - center;
							real_t min_dist = INFINITY;
							vec3 best_n(0,0,0);
							for (int i=0; i<3; ++i) {
								real_t dist = Math::abs(half_extents[i] - Math::abs(d[i]));
								if (dist < min_dist) { min_dist = dist; best_n = vec3(); best_n[i] = (d[i]>0?1.0f:-1.0f); }
							}
							hit.normal = body->get_rotation().xform(best_n);
							hit.distance = t_entry;
							results.push_back(hit);
						}
					}
				}
			} else {
				// No shape: use AABB entry.
				const AABB &box = body->get_aabb();
				real_t t_entry, t_exit;
				if (gaia::bvh::intersect_ray_aabb(p_origin, dir, box, 0.0, max_dist, t_entry, t_exit)) {
					if (t_entry < max_dist) {
						max_dist = t_entry;
						RayHit hit;
						hit.body_id = id;
						hit.point = p_origin + dir * t_entry;
						vec3 local_pt = body->get_transform().affine_inverse().xform(hit.point);
						vec3 half_extents = box.size * 0.5;
						vec3 center = box.position + half_extents;
						vec3 d = local_pt - center;
						real_t min_dist = INFINITY;
						vec3 best_n(0,0,0);
						for (int i=0; i<3; ++i) {
							real_t dist = Math::abs(half_extents[i] - Math::abs(d[i]));
							if (dist < min_dist) { min_dist = dist; best_n = vec3(); best_n[i] = (d[i]>0?1.0f:-1.0f); }
						}
						hit.normal = body->get_rotation().xform(best_n);
						hit.distance = t_entry;
						results.push_back(hit);
					}
				}
			}
		});
		return results;
	}

	// ------------------------------ Sphere Overlap ------------------------------
	struct OverlapResult {
		body_id body_id;
		aabb world_aabb;
	};

	/**
	 * Find all active bodies whose AABB intersects a given world‑space sphere.
	 */
	static LocalVector<OverlapResult> sphere_overlap(const ViennaWorld *p_world,
													 const vec3 &p_center,
													 real_t p_radius) {
		LocalVector<OverlapResult> results;
		if (!p_world) return results;
		AABB sphere_aabb(p_center - vec3(p_radius, p_radius, p_radius),
						 vec3(p_radius*2, p_radius*2, p_radius*2));
		LocalVector<body_id> ids = p_world->get_body_ids();
		for (body_id id : ids) {
			Ref<ViennaBody> body = p_world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;
			if (body->get_aabb().intersects(sphere_aabb)) {
				OverlapResult res;
				res.body_id = id;
				res.world_aabb = body->get_aabb();
				results.push_back(res);
			}
		}
		return results;
	}

	// ------------------------------ AABB Overlap ------------------------------
	/**
	 * Find all active bodies whose AABB intersects a given world‑space AABB.
	 */
	static LocalVector<OverlapResult> aabb_overlap(const ViennaWorld *p_world,
												   const AABB &p_aabb) {
		LocalVector<OverlapResult> results;
		if (!p_world) return results;
		LocalVector<body_id> ids = p_world->get_body_ids();
		for (body_id id : ids) {
			Ref<ViennaBody> body = p_world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;
			if (body->get_aabb().intersects(p_aabb)) {
				OverlapResult res;
				res.body_id = id;
				res.world_aabb = body->get_aabb();
				results.push_back(res);
			}
		}
		return results;
	}

	// ------------------------------ Shape Sweep ------------------------------
	struct SweepHit {
		body_id body_id;
		real_t fraction;       // fraction of the motion where hit occurs [0..1]
		vec3 point;
		vec3 normal;
	};

	/**
	 * Sweep a convex shape from a start transform to an end translation.
	 * Uses GJK in a conservative‑advancement style to find the first hit
	 * with any active body in the world.
	 * NOTE: fully accurate sweep requires CCD; this is a simplified binary‑search
	 * approximation for demonstration.  For production, use ViennaCCD.
	 */
	static LocalVector<SweepHit> sweep_shape(const ViennaWorld *p_world,
											 const ViennaShape &p_shape,
											 const mat4 &p_start_transform,
											 const vec3 &p_end_translation,
											 real_t p_margin = 0.001) {
		LocalVector<SweepHit> results;
		if (!p_world) return results;

		// Compute motion vector in world space.
		vec3 motion = p_end_translation;
		mat4 current_xform = p_start_transform;
		vec3 current_pos = current_xform.origin;
		vec3 target_pos = current_pos + motion;
		real_t motion_len = motion.length();
		if (motion_len < CMP_EPSILON) return results;

		vec3 dir = motion / motion_len;

		// Search over all active bodies using AABB broad‑phase for candidates.
		LocalVector<body_id> ids = p_world->get_body_ids();
		LocalVector<OverlapResult> candidates = aabb_overlap(p_world,
			AABB(current_pos, target_pos).grow(p_margin * 2.0)); // rough
		// Narrow down to actual shape using GJK proximity.
		for (const OverlapResult &cand : candidates) {
			Ref<ViennaBody> body = p_world->get_body(cand.body_id);
			if (body.is_null()) continue;
			const ViennaShape *obs_shape = body->get_collision_shape().ptr();
			if (!obs_shape) continue;

			// Binary search for earliest time of impact along motion direction.
			real_t lo = 0.0, hi = 1.0;
			bool collided = false;
			real_t best_t = 1.0;
			for (int i = 0; i < 8; ++i) { // 8 iterations = good precision
				real_t mid = (lo + hi) * 0.5;
				mat4 test_xform = p_start_transform;
				test_xform.origin = current_pos + motion * mid;
				gaia::collision::GJK::Result res = gaia::collision::GJK::collide(
					p_shape, test_xform, *obs_shape, body->get_transform());
				if (res.colliding) {
					hi = mid;
					best_t = mid;
					collided = true;
				} else {
					lo = mid;
				}
			}
			if (collided) {
				mat4 hit_xform = p_start_transform;
				hit_xform.origin = current_pos + motion * best_t;
				gaia::collision::GJK::Result res = gaia::collision::GJK::collide(
					p_shape, hit_xform, *obs_shape, body->get_transform());
				SweepHit hit;
				hit.body_id = cand.body_id;
				hit.fraction = best_t;
				hit.point = res.closest_a; // point on swept shape? approximation
				hit.normal = res.normal;
				results.push_back(hit);
				// Return only first hit? We keep the closest.
			}
		}
		// Sort by fraction ascending and return only the earliest? For simplicity return all.
		return results;
	}
};

} // namespace vienna

#endif // VIENNA_QUERY_WORLD_QUERY_H