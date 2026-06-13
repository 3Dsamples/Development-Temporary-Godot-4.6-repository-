// File 62: modules/genesis/src/collision/collider.h
// Collider: wraps a convex shape and provides support functions for GJK/EPA.

#ifndef GENESIS_COLLISION_COLLIDER_H
#define GENESIS_COLLISION_COLLIDER_H

#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"

namespace genesis {

/**
 * Base collider for a convex shape.
 * Provides a support function used by GJK.
 */
class Collider {
public:
	virtual ~Collider() {}

	// Return the world-space support point in the given direction.
	virtual Vector3 get_support(const Vector3 &dir_world, const Transform3D &transform) const = 0;

	virtual GeometryType get_type() const = 0;
};

// --- Sphere collider ---
class SphereCollider : public Collider {
public:
	real_t radius;

	SphereCollider(real_t r) : radius(r) {}

	virtual Vector3 get_support(const Vector3 &dir_world, const Transform3D &transform) const override {
		Vector3 dir_local = transform.basis.xform_inv(dir_world).normalized();
		return transform.xform(dir_local * radius);
	}

	virtual GeometryType get_type() const override { return GeometryType::SPHERE; }
};

// --- Box collider ---
class BoxCollider : public Collider {
public:
	Vector3 half_extents;

	BoxCollider(const Vector3 &he) : half_extents(he) {}

	virtual Vector3 get_support(const Vector3 &dir_world, const Transform3D &transform) const override {
		Vector3 dir_local = transform.basis.xform_inv(dir_world);
		Vector3 support_local(
			(dir_local.x >= 0) ? half_extents.x : -half_extents.x,
			(dir_local.y >= 0) ? half_extents.y : -half_extents.y,
			(dir_local.z >= 0) ? half_extents.z : -half_extents.z
		);
		return transform.xform(support_local);
	}

	virtual GeometryType get_type() const override { return GeometryType::BOX; }
};

// --- Capsule collider (aligned along local Y axis by default) ---
class CapsuleCollider : public Collider {
public:
	real_t radius;
	real_t height;

	CapsuleCollider(real_t r, real_t h) : radius(r), height(h) {}

	virtual Vector3 get_support(const Vector3 &dir_world, const Transform3D &transform) const override {
		Vector3 dir_local = transform.basis.xform_inv(dir_world).normalized();
		// capsule: support is either a hemisphere point or a cylinder segment.
		// determine which hemisphere to use based on direction's vertical component.
		real_t half_h = height * 0.5 - radius;
		if (half_h < 0) half_h = 0;
		Vector3 center(0, 0, 0);
		if (dir_local.y > 0) center.y = half_h;
		else if (dir_local.y < 0) center.y = -half_h;
		// else center stays at 0

		Vector3 support_local = center + dir_local * radius;
		return transform.xform(support_local);
	}

	virtual GeometryType get_type() const override { return GeometryType::CAPSULE; }
};

// --- Cylinder collider (aligned Y) ---
class CylinderCollider : public Collider {
public:
	real_t radius;
	real_t height;

	CylinderCollider(real_t r, real_t h) : radius(r), height(h) {}

	virtual Vector3 get_support(const Vector3 &dir_world, const Transform3D &transform) const override {
		Vector3 dir_local = transform.basis.xform_inv(dir_world);
		real_t half_h = height * 0.5;
		// Decide cap center
		real_t cap_y = (dir_local.y >= 0) ? half_h : -half_h;
		// Radial support
		Vector2 radial_dir(dir_local.x, dir_local.z);
		real_t len = radial_dir.length();
		Vector3 support_local;
		if (len < CMP_EPSILON) {
			support_local = Vector3(0, cap_y, 0);
		} else {
			Vector3 radial = Vector3(radial_dir.x / len, 0, radial_dir.y / len) * radius;
			support_local = radial;
			support_local.y = cap_y;
		}
		return transform.xform(support_local);
	}

	virtual GeometryType get_type() const override { return GeometryType::CYLINDER; }
};

// --- Convex mesh collider (simplistic, assumes a set of vertices) ---
class ConvexMeshCollider : public Collider {
public:
	LocalVector<Vector3> vertices; // in local space

	ConvexMeshCollider(const LocalVector<Vector3> &p_verts) : vertices(p_verts) {}

	virtual Vector3 get_support(const Vector3 &dir_world, const Transform3D &transform) const override {
		Vector3 dir_local = transform.basis.xform_inv(dir_world).normalized();
		int best_idx = 0;
		real_t best_dot = -INFINITY;
		for (int i = 0; i < vertices.size(); ++i) {
			real_t d = vertices[i].dot(dir_local);
			if (d > best_dot) {
				best_dot = d;
				best_idx = i;
			}
		}
		return transform.xform(vertices[best_idx]);
	}

	virtual GeometryType get_type() const override { return GeometryType::CONVEX_MESH; }
};

} // namespace genesis

#endif // GENESIS_COLLISION_COLLIDER_H