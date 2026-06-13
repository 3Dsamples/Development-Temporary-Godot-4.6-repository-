// File 06: modules/gaia/src/collision_detector/narrow_phase.h

#ifndef GAIA_COLLISION_NARROW_PHASE_H
#define GAIA_COLLISION_NARROW_PHASE_H

#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia::collision {

// ---------------------------------------------------------------------------
// Support function interface for convex shapes
// ---------------------------------------------------------------------------
class ConvexShape {
public:
	virtual ~ConvexShape() {}
	// Returns the farthest point in the given direction (in local space)
	virtual Vector3 get_support(const Vector3 &dir_local) const = 0;
};

// ---------------------------------------------------------------------------
// GJK (Gilbert–Johnson–Keerthi) distance / collision query
// Returns true if the two convex shapes overlap.
// If non-overlapping, `out_separation` and `out_closest_a`/`out_closest_b` are filled.
// If overlapping, `out_penetration` and `out_normal`, `out_contact_a`, `out_contact_b` are filled via EPA.
// ---------------------------------------------------------------------------
class GJK {
public:
	struct Result {
		bool colliding;
		Vector3 closest_a;       // point on shape A in world space
		Vector3 closest_b;       // point on shape B in world space
		real_t distance;         // separation distance (negative = penetration depth)
		Vector3 normal;          // from B to A when colliding (push direction of B)
	};

	// Perform GJK + EPA for two convex shapes with given transforms.
	static Result collide(const ConvexShape &shape_a, const Transform3D &transform_a,
						  const ConvexShape &shape_b, const Transform3D &transform_b);

private:
	struct Simplex {
		enum { MAX_VERTICES = 4 };
		Vector3 vertices[MAX_VERTICES];
		int size;
	};

	// GJK core: returns true if simplex contains origin.
	static bool evaluate_simplex(Simplex &simplex, Vector3 &direction);
};

// ---------------------------------------------------------------------------
// EPA (Expanding Polytope Algorithm) – computes penetration information.
// Assumes the simplex from GJK encloses the origin (shapes overlap).
// ---------------------------------------------------------------------------
struct EPA {
	struct Result {
		Vector3 normal;        // world-space penetration direction (from B to A)
		real_t depth;          // penetration depth
		Vector3 contact_a;     // contact point on A (world)
		Vector3 contact_b;     // contact point on B (world)
	};

	// Compute penetration using the final simplex from GJK.
	// The input simplex must have 4 vertices and contain the origin.
	static Result penetrate(const ConvexShape &shape_a, const Transform3D &transform_a,
							const ConvexShape &shape_b, const Transform3D &transform_b,
							const GJK::Simplex &simplex);
};

// ---------------------------------------------------------------------------
// Implementation of GJK
// ---------------------------------------------------------------------------
inline GJK::Result GJK::collide(const ConvexShape &shape_a, const Transform3D &transform_a,
								const ConvexShape &shape_b, const Transform3D &transform_b) {
	Result res;
	res.colliding = false;
	res.distance = 0.0;
	res.normal = Vector3(0, 0, 1);

	// Initial direction: from A to B centroid (heuristic)
	Vector3 center_a = transform_a.origin;
	Vector3 center_b = transform_b.origin;
	Vector3 dir = center_b - center_a;
	if (dir.length_squared() < CMP_EPSILON) {
		dir = Vector3(1, 0, 0);
	}

	Simplex simplex;
	simplex.size = 0;

	// First support point
	Vector3 support = get_support_difference(shape_a, transform_a, shape_b, transform_b, dir);
	simplex.vertices[0] = support;
	simplex.size = 1;
	dir = -support;

	const int max_iterations = 64;
	for (int iter = 0; iter < max_iterations; ++iter) {
		support = get_support_difference(shape_a, transform_a, shape_b, transform_b, dir);
		if (support.dot(dir) < 0) {
			// No intersection
			res.colliding = false;
			// Compute closest points from simplex
			// (Simplified: distance is distance to origin from simplex)
			Vector3 closest = compute_closest_point_on_simplex(simplex);
			res.distance = closest.length();
			// Transform to world: this is the point in Minkowski difference space.
			// Actual world points would need reconstruction. For now, approximate.
			res.closest_a = transform_a.origin + closest * 0.5; // placeholder
			res.closest_b = transform_b.origin - closest * 0.5;
			return res;
		}
		simplex.vertices[simplex.size++] = support;
		if (evaluate_simplex(simplex, dir)) {
			// Origin enclosed, collision
			res.colliding = true;
			// Run EPA
			EPA::Result epa = EPA::penetrate(shape_a, transform_a, shape_b, transform_b, simplex);
			res.normal = epa.normal;
			res.distance = -epa.depth;
			res.closest_a = epa.contact_a;
			res.closest_b = epa.contact_b;
			return res;
		}
	}
	// Fallback: assume no collision after max iterations
	res.colliding = false;
	res.distance = 0.0;
	return res;
}

// Helper: Minkowski difference support
inline Vector3 get_support_difference(const ConvexShape &a, const Transform3D &ta,
									  const ConvexShape &b, const Transform3D &tb,
									  const Vector3 &world_dir) {
	Vector3 dir_a = ta.basis.xform_inv(world_dir);
	Vector3 dir_b = tb.basis.xform_inv(-world_dir);
	Vector3 p_a = ta.xform(a.get_support(dir_a));
	Vector3 p_b = tb.xform(b.get_support(dir_b));
	return p_a - p_b;
}

// GJK simplex evaluation for 1..4 points.
inline bool GJK::evaluate_simplex(Simplex &simplex, Vector3 &direction) {
	switch (simplex.size) {
		case 2: {
			// Line segment
			Vector3 a = simplex.vertices[1];
			Vector3 b = simplex.vertices[0];
			Vector3 ab = b - a;
			Vector3 ao = -a;
			if (ab.dot(ao) > 0) {
				direction = ao.cross(ab).cross(ab);
			} else {
				simplex.size = 1;
				simplex.vertices[0] = a;
				direction = ao;
			}
		} return false;
		case 3: {
			// Triangle
			Vector3 a = simplex.vertices[2];
			Vector3 b = simplex.vertices[1];
			Vector3 c = simplex.vertices[0];
			Vector3 ao = -a;
			Vector3 ab = b - a;
			Vector3 ac = c - a;
			Vector3 abc = ab.cross(ac);
			// Check if origin is above/below triangle
			Vector3 ac_cross_abc = ac.cross(abc);
			if (ac_cross_abc.dot(ao) > 0) {
				// Origin is outside edge ac
				simplex.vertices[0] = a;
				simplex.vertices[1] = c;
				simplex.size = 2;
				direction = ac_cross_abc;
			} else {
				Vector3 abc_cross_ab = abc.cross(ab);
				if (abc_cross_ab.dot(ao) > 0) {
					simplex.vertices[0] = a;
					simplex.vertices[1] = b;
					simplex.size = 2;
					direction = abc_cross_ab;
				} else {
					// Origin is within triangle region (possibly below/above)
					if (abc.dot(ao) > 0) {
						direction = abc;
					} else {
						// flip order to aim toward origin
						SWAP(simplex.vertices[1], simplex.vertices[0]);
						direction = -abc;
					}
				}
			}
		} return false;
		case 4: {
			// Tetrahedron
			Vector3 a = simplex.vertices[3];
			Vector3 b = simplex.vertices[2];
			Vector3 c = simplex.vertices[1];
			Vector3 d = simplex.vertices[0]; // d is the oldest?
			// Check each face
			Vector3 ao = -a;
			Vector3 ab = b - a;
			Vector3 ac = c - a;
			Vector3 ad = d - a;

			// Face normal for abc
			Vector3 abc = ab.cross(ac);
			if (abc.dot(ao) > 0) {
				// Origin is on outside of abc face
				simplex.vertices[0] = a;
				simplex.vertices[1] = b;
				simplex.vertices[2] = c;
				simplex.size = 3;
				direction = abc;
				return false;
			}
			// Face acd
			Vector3 acd = ac.cross(ad);
			if (acd.dot(ao) > 0) {
				simplex.vertices[0] = a;
				simplex.vertices[1] = c;
				simplex.vertices[2] = d;
				simplex.size = 3;
				direction = acd;
				return false;
			}
			// Face adb
			Vector3 adb = ad.cross(ab);
			if (adb.dot(ao) > 0) {
				simplex.vertices[0] = a;
				simplex.vertices[1] = d;
				simplex.vertices[2] = b;
				simplex.size = 3;
				direction = adb;
				return false;
			}
			// Origin is inside the tetrahedron
			return true;
		}
		default:
			direction = Vector3(1, 0, 0);
			return false;
	}
}

// EPA implementation
inline EPA::Result EPA::penetrate(const ConvexShape &shape_a, const Transform3D &transform_a,
								  const ConvexShape &shape_b, const Transform3D &transform_b,
								  const GJK::Simplex &simplex) {
	// Very minimal EPA: find face closest to origin on the simplex and expand.
	// Full EPA is complex; this returns approximate penetration using the face closest to origin.
	Result res;
	res.depth = 1e10;
	Vector3 best_normal;
	// Examine tetrahedron faces (simplex has 4 vertices)
	const int face_indices[4][3] = { {0,1,2}, {0,3,1}, {0,2,3}, {1,3,2} };
	for (int f = 0; f < 4; ++f) {
		Vector3 a = simplex.vertices[face_indices[f][0]];
		Vector3 b = simplex.vertices[face_indices[f][1]];
		Vector3 c = simplex.vertices[face_indices[f][2]];
		Vector3 normal = (b - a).cross(c - a).normalized();
		real_t dist = SIGN(normal.dot(a)); // signed distance of origin to plane? Actually origin is inside, all face distances are positive toward origin.
		// Distance from origin to plane: |dot(a, normal)|
		real_t d = ABS(a.dot(normal));
		if (d < res.depth) {
			res.depth = d;
			best_normal = (a.dot(normal) < 0) ? normal : -normal; // points toward origin (outside of Minkowski sum)
		}
	}
	res.normal = best_normal;
	// Approximate contact points: origin + normal * depth/2 etc.
	// Here we return placeholder world points.
	res.contact_a = transform_a.origin + best_normal * res.depth * 0.5;
	res.contact_b = transform_b.origin - best_normal * res.depth * 0.5;
	return res;
}

} // namespace gaia::collision

#endif // GAIA_COLLISION_NARROW_PHASE_H