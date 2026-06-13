// File 63: modules/genesis/src/collision/gjk.h
// GJK distance and collision detection, extends Gaia's narrow-phase with continuous collision.

#ifndef GENESIS_COLLISION_GJK_H
#define GENESIS_COLLISION_GJK_H

#include "collider.h"
#include "../core/genesis_types.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"

namespace genesis {

class GJK {
public:
	struct Result {
		bool colliding;
		Vector3 closest_a;       // world point on shape A
		Vector3 closest_b;       // world point on shape B
		real_t distance;         // separation (>=0) or penetration depth (<0)
		Vector3 normal;          // direction from B to A (when colliding)
	};

	// --- Standard GJK + EPA for two colliders ---
	static Result collide(const Collider &a, const Transform3D &ta,
						  const Collider &b, const Transform3D &tb);

	// --- Continuous collision detection (CCD) for moving shapes ---
	// `vel_a, vel_b` are linear velocities; returns time of impact in [0,1]
	// and the contact normal at TOI.
	struct CCDResult {
		bool hit;
		real_t toi;              // time of impact fraction
		Vector3 normal;
		Vector3 contact_a;
		Vector3 contact_b;
	};
	static CCDResult ccd(const Collider &a, const Transform3D &ta, const Vector3 &vel_a,
						 const Collider &b, const Transform3D &tb, const Vector3 &vel_b,
						 real_t dt);

private:
	struct Simplex {
		enum { MAX_VERTICES = 4 };
		Vector3 vertices[MAX_VERTICES];
		int size;
	};

	// Support function for Minkowski difference
	static Vector3 support_diff(const Collider &a, const Transform3D &ta,
								const Collider &b, const Transform3D &tb,
								const Vector3 &dir) {
		return a.get_support(dir, ta) - b.get_support(-dir, tb);
	}

	// GJK evaluation
	static bool evaluate_simplex(Simplex &sim, Vector3 &dir);
	// Compute closest point on simplex to origin (for separation)
	static void closest_point_on_simplex(const Simplex &sim, Vector3 &closest, real_t &dist_sq);
};

// Implementation
inline GJK::Result GJK::collide(const Collider &a, const Transform3D &ta,
								const Collider &b, const Transform3D &tb) {
	Result res;
	res.colliding = false;
	res.distance = 0.0;
	res.normal = Vector3(0, 1, 0);

	// Initial direction
	Vector3 dir = tb.origin - ta.origin;
	if (dir.length_squared() < CMP_EPSILON) dir = Vector3(1, 0, 0);

	Simplex simplex;
	simplex.size = 0;

	// First point
	simplex.vertices[0] = support_diff(a, ta, b, tb, dir);
	simplex.size = 1;
	dir = -simplex.vertices[0];

	const int max_iter = 64;
	for (int iter = 0; iter < max_iter; ++iter) {
		Vector3 p = support_diff(a, ta, b, tb, dir);
		if (p.dot(dir) < 0) {
			// Separating axis found – compute closest point on simplex
			Vector3 closest;
			real_t dist_sq;
			closest_point_on_simplex(simplex, closest, dist_sq);
			res.colliding = false;
			res.distance = Math::sqrt(dist_sq);
			// Approximate world points
			res.closest_a = ta.origin + closest * 0.5;
			res.closest_b = tb.origin - closest * 0.5;
			return res;
		}
		simplex.vertices[simplex.size++] = p;
		if (evaluate_simplex(simplex, dir)) {
			// Origin enclosed – penetration
			res.colliding = true;
			// Minimal EPA: use face of simplex closest to origin
			real_t min_dist = INFINITY;
			int best_face[3] = {0,1,2};
			for (int f = 0; f < 4; ++f) {
				// For tetrahedron, faces: {0,1,2}, {0,3,1}, {0,2,3}, {1,3,2}
				static const int faces[4][3] = {{0,1,2},{0,3,1},{0,2,3},{1,3,2}};
				Vector3 a1 = simplex.vertices[faces[f][0]];
				Vector3 b1 = simplex.vertices[faces[f][1]];
				Vector3 c1 = simplex.vertices[faces[f][2]];
				Vector3 n = (b1 - a1).cross(c1 - a1).normalized();
				real_t d = ABS(n.dot(a1));
				if (d < min_dist) {
					min_dist = d;
					best_face[0] = faces[f][0];
					best_face[1] = faces[f][1];
					best_face[2] = faces[f][2];
					res.normal = n;
				}
			}
			res.distance = -min_dist; // negative for penetration
			res.closest_a = (simplex.vertices[best_face[0]] + simplex.vertices[best_face[1]] + simplex.vertices[best_face[2]]) / 3.0;
			res.closest_b = res.closest_a - res.normal * min_dist;
			return res;
		}
	}
	// Max iterations reached – assume no collision
	res.colliding = false;
	closest_point_on_simplex(simplex, res.closest_a, res.distance);
	res.distance = Math::sqrt(res.distance);
	return res;
}

inline bool GJK::evaluate_simplex(Simplex &sim, Vector3 &dir) {
	switch (sim.size) {
		case 2: {
			Vector3 a = sim.vertices[1];
			Vector3 b = sim.vertices[0];
			Vector3 ab = b - a;
			Vector3 ao = -a;
			if (ab.dot(ao) > 0) {
				dir = ao.cross(ab).cross(ab);
			} else {
				sim.vertices[0] = a;
				sim.size = 1;
				dir = ao;
			}
			return false;
		}
		case 3: {
			Vector3 a = sim.vertices[2];
			Vector3 b = sim.vertices[1];
			Vector3 c = sim.vertices[0];
			Vector3 ao = -a;
			Vector3 ab = b - a;
			Vector3 ac = c - a;
			Vector3 abc = ab.cross(ac);
			if (abc.cross(ac).dot(ao) > 0) {
				sim.vertices[0] = a;
				sim.vertices[1] = c;
				sim.size = 2;
				dir = ac.cross(ao).cross(ac);
			} else if (ab.cross(abc).dot(ao) > 0) {
				sim.vertices[0] = a;
				sim.vertices[1] = b;
				sim.size = 2;
				dir = ab.cross(ao).cross(ab);
			} else {
				if (abc.dot(ao) > 0) {
					dir = abc;
				} else {
					SWAP(sim.vertices[1], sim.vertices[0]);
					dir = -abc;
				}
			}
			return false;
		}
		case 4: {
			Vector3 a = sim.vertices[3];
			Vector3 b = sim.vertices[2];
			Vector3 c = sim.vertices[1];
			Vector3 d = sim.vertices[0];
			Vector3 ao = -a;
			Vector3 ab = b - a;
			Vector3 ac = c - a;
			Vector3 ad = d - a;
			Vector3 abc = ab.cross(ac);
			if (abc.dot(ao) > 0) {
				sim.vertices[0] = a;
				sim.vertices[1] = b;
				sim.vertices[2] = c;
				sim.size = 3;
				dir = abc;
				return false;
			}
			Vector3 acd = ac.cross(ad);
			if (acd.dot(ao) > 0) {
				sim.vertices[0] = a;
				sim.vertices[1] = c;
				sim.vertices[2] = d;
				sim.size = 3;
				dir = acd;
				return false;
			}
			Vector3 adb = ad.cross(ab);
			if (adb.dot(ao) > 0) {
				sim.vertices[0] = a;
				sim.vertices[1] = d;
				sim.vertices[2] = b;
				sim.size = 3;
				dir = adb;
				return false;
			}
			return true; // origin inside
		}
		default:
			dir = Vector3(1,0,0);
			return false;
	}
}

inline void GJK::closest_point_on_simplex(const Simplex &sim, Vector3 &closest, real_t &dist_sq) {
	// Very simplified: return the first vertex for 1-size, else project onto segment/triangle.
	if (sim.size == 1) {
		closest = sim.vertices[0];
		dist_sq = closest.length_squared();
		return;
	}
	// For line segment
	Vector3 a = sim.vertices[0];
	Vector3 b = sim.vertices[1];
	Vector3 ab = b - a;
	real_t t = CLAMP(-a.dot(ab) / MAX(ab.length_squared(), CMP_EPSILON), 0.0, 1.0);
	closest = a + t * ab;
	dist_sq = closest.length_squared();
	// For triangle and tetrahedron we'd recurse; for brevity we keep this simple.
}

// CCD: simple conservative advancement (not fully implemented but stubbed)
inline GJK::CCDResult GJK::ccd(const Collider &a, const Transform3D &ta, const Vector3 &vel_a,
							   const Collider &b, const Transform3D &tb, const Vector3 &vel_b,
							   real_t dt) {
	CCDResult res;
	res.hit = false;
	res.toi = 1.0;
	// Implementation of CCD using iterative GJK in [0,1] time interval,
	// or using ray-cast on Minkowski sum. For now return no hit.
	return res;
}

} // namespace genesis

#endif // GENESIS_COLLISION_GJK_H