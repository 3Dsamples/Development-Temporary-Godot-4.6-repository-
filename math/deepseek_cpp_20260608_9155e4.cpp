// File 07: modules/gaia/src/collision_detector/contact.h

#ifndef GAIA_COLLISION_CONTACT_H
#define GAIA_COLLISION_CONTACT_H

#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include <cfloat>

namespace gaia::collision {

/**
 * Contact point structure.
 */
struct ContactPoint {
	Vector3 point_a;        // contact point on body A (world space)
	Vector3 point_b;        // contact point on body B (world space)
	Vector3 normal;         // from B to A
	real_t penetration;     // positive = interpenetration depth
	real_t distance;        // separation distance (negative = penetration)
};

/**
 * Contact manifold – a collection of up to 4 contact points.
 */
struct Manifold {
	static const int MAX_CONTACTS = 4;
	ContactPoint contacts[MAX_CONTACTS];
	int count;

	Manifold() : count(0) {}

	void add_contact(const ContactPoint &cp) {
		if (count < MAX_CONTACTS) {
			contacts[count++] = cp;
		}
	}

	void clear() { count = 0; }

	// Remove duplicate or nearly collinear contacts.
	void reduce(const Vector3 &centroid_a, const Vector3 &centroid_b);
};

// ---------------------------------------------------------------------------
// Helper: project a point onto a plane defined by origin and normal, and
// return the barycentric coordinates of the projection onto the given triangle.
// Used for clipping and manifold optimization.
// ---------------------------------------------------------------------------
inline real_t closest_pt_segment_segment(const Vector3 &p1, const Vector3 &q1,
										 const Vector3 &p2, const Vector3 &q2,
										 Vector3 &c1, Vector3 &c2) {
	// Original Gaia code: find closest points between two segments.
	Vector3 d1 = q1 - p1;
	Vector3 d2 = q2 - p2;
	Vector3 r = p1 - p2;
	real_t a = d1.dot(d1); // always >= 0
	real_t e = d2.dot(d2); // always >= 0
	real_t f = d2.dot(r);

	real_t s, t;
	if (a <= CMP_EPSILON && e <= CMP_EPSILON) {
		s = 0.0;
		t = 0.0;
		c1 = p1;
		c2 = p2;
		return (c1 - c2).length_squared();
	}
	if (a <= CMP_EPSILON) {
		s = 0.0;
		t = CLAMP(f / e, 0.0, 1.0);
	} else {
		real_t c = d1.dot(r);
		if (e <= CMP_EPSILON) {
			t = 0.0;
			s = CLAMP(-c / a, 0.0, 1.0);
		} else {
			real_t b = d1.dot(d2);
			real_t denom = a * e - b * b; // always nonnegative
			if (denom != 0.0) {
				s = CLAMP((b * f - c * e) / denom, 0.0, 1.0);
			} else {
				s = 0.0;
			}
			t = (b * s + f) / e;
			if (t < 0.0) {
				t = 0.0;
				s = CLAMP(-c / a, 0.0, 1.0);
			} else if (t > 1.0) {
				t = 1.0;
				s = CLAMP((b - c) / a, 0.0, 1.0);
			}
		}
	}
	c1 = p1 + d1 * s;
	c2 = p2 + d2 * t;
	return (c1 - c2).length_squared();
}

// ---------------------------------------------------------------------------
// Contact generation for two colliding convex shapes, given GJK/EPA results.
// Produces a set of contact points (up to 4) for a manifold.
// ---------------------------------------------------------------------------
void generate_contacts(const Vector3 &normal, real_t penetration,
					   const Vector3 &closest_a, const Vector3 &closest_b,
					   const ConvexShape *shape_a, const Transform3D &transform_a,
					   const ConvexShape *shape_b, const Transform3D &transform_b,
					   Manifold &manifold);

// Manifold reduction: keep only the contacts that best represent the contact area.
inline void Manifold::reduce(const Vector3 &centroid_a, const Vector3 &centroid_b) {
	if (count <= 2) return;

	// Keep the deepest contact
	real_t deepest = -INFINITY;
	int deepest_idx = 0;
	for (int i = 0; i < count; ++i) {
		if (contacts[i].penetration > deepest) {
			deepest = contacts[i].penetration;
			deepest_idx = i;
		}
	}
	SWAP(contacts[0], contacts[deepest_idx]);

	// For the remaining, keep farthest from deepest contact (up to 4)
	while (count > 2) {
		real_t max_dist = -INFINITY;
		int max_idx = 1;
		for (int i = 1; i < count; ++i) {
			real_t d = contacts[i].point_a.distance_squared_to(contacts[0].point_a);
			if (d > max_dist) {
				max_dist = d;
				max_idx = i;
			}
		}
		if (max_dist < CMP_EPSILON) {
			// All contacts are essentially the same point; keep only deepest.
			count = 1;
			return;
		}
		SWAP(contacts[1], contacts[max_idx]);
		// Now we have two contacts (0 and 1). For 3rd and 4th, they must be far from the edges.
		// But reduce loop will try to keep up to 4 farthest. Simplify: keep 2 contacts.
		count = 2;
		return;
	}
}

// ---------------------------------------------------------------------------
// Contact generation for two spheres (specialised).
// ---------------------------------------------------------------------------
void generate_sphere_sphere_contacts(const Vector3 &center_a, real_t radius_a,
									 const Vector3 &center_b, real_t radius_b,
									 Manifold &manifold) {
	Vector3 delta = center_b - center_a;
	real_t dist_sq = delta.length_squared();
	real_t sum_radii = radius_a + radius_b;
	if (dist_sq > sum_radii * sum_radii) {
		manifold.clear();
		return;
	}
	real_t dist = math::sqrt(dist_sq);
	Vector3 normal;
	real_t penetration;
	if (dist < CMP_EPSILON) {
		normal = Vector3(0, 1, 0); // arbitrary
		penetration = sum_radii;
	} else {
		normal = delta / dist;
		penetration = sum_radii - dist;
	}
	ContactPoint cp;
	cp.normal = normal;
	cp.penetration = penetration;
	cp.distance = -penetration;
	cp.point_a = center_a + normal * radius_a;
	cp.point_b = center_b - normal * radius_b;
	manifold.add_contact(cp);
}

} // namespace gaia::collision

#endif // GAIA_COLLISION_CONTACT_H