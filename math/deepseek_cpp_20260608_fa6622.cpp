// File 179: modules/newton/src/collision/newton_collision.h
// Newton collision shape base class – wraps a convex collision primitive
// (sphere, box, capsule, cylinder, cone, convex hull) and supports
// support-point queries for GJK narrow‑phase.

#ifndef NEWTON_COLLISION_NEWTON_COLLISION_H
#define NEWTON_COLLISION_NEWTON_COLLISION_H

#include "core/object/ref_counted.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonCollision : public RefCounted {
	GDCLASS(NewtonCollision, RefCounted);

public:
	NewtonCollision() {}
	virtual ~NewtonCollision() {}

	virtual ShapeType get_shape_type() const = 0;

	// Return the world‑space support point in the given direction.
	// `p_dir` must be normalized.
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const = 0;

	// Compute inertia tensor for a unit density (scaled by mass later).
	virtual mat3 compute_inertia(real_t p_mass) const = 0;

	// Compute axis‑aligned bounding box in local space.
	virtual aabb get_local_aabb() const = 0;

protected:
	static void _bind_methods() {}
};

// -----------------------------------------------------------------------
// Sphere collision shape
// -----------------------------------------------------------------------
class NewtonCollisionSphere : public NewtonCollision {
	GDCLASS(NewtonCollisionSphere, NewtonCollision);
public:
	NewtonCollisionSphere(real_t p_radius = 0.5) : radius(p_radius) {}

	virtual ShapeType get_shape_type() const override { return ShapeType::SPHERE; }

	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		// Transform direction to local space, scale by radius, transform back.
		vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
		return p_transform.xform(local_dir * radius);
	}

	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t I = (2.0 / 5.0) * p_mass * radius * radius;
		return mat3().scaled(vec3(I, I, I));
	}

	virtual aabb get_local_aabb() const override {
		return aabb(vec3(-radius, -radius, -radius), vec3(radius * 2, radius * 2, radius * 2));
	}

	real_t get_radius() const { return radius; }
	void set_radius(real_t p_r) { radius = MAX(p_r, 0.0); }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NewtonCollisionSphere::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &NewtonCollisionSphere::get_radius);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
	}

private:
	real_t radius;
};

// -----------------------------------------------------------------------
// Box collision shape
// -----------------------------------------------------------------------
class NewtonCollisionBox : public NewtonCollision {
	GDCLASS(NewtonCollisionBox, NewtonCollision);
public:
	NewtonCollisionBox(const vec3 &p_half_extents = vec3(0.5, 0.5, 0.5))
		: half_extents(p_half_extents) {}

	virtual ShapeType get_shape_type() const override { return ShapeType::BOX; }

	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		vec3 support_local(
			(local_dir.x >= 0) ? half_extents.x : -half_extents.x,
			(local_dir.y >= 0) ? half_extents.y : -half_extents.y,
			(local_dir.z >= 0) ? half_extents.z : -half_extents.z
		);
		return p_transform.xform(support_local);
	}

	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t x = half_extents.x, y = half_extents.y, z = half_extents.z;
		real_t Ix = (1.0 / 12.0) * p_mass * (y * y + z * z);
		real_t Iy = (1.0 / 12.0) * p_mass * (x * x + z * z);
		real_t Iz = (1.0 / 12.0) * p_mass * (x * x + y * y);
		return mat3().scaled(vec3(Ix, Iy, Iz));
	}

	virtual aabb get_local_aabb() const override {
		return aabb(-half_extents, half_extents * 2.0);
	}

	const vec3 &get_half_extents() const { return half_extents; }
	void set_half_extents(const vec3 &p_ext) { half_extents = p_ext.abs(); }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_half_extents", "extents"), &NewtonCollisionBox::set_half_extents);
		ClassDB::bind_method(D_METHOD("get_half_extents"), &NewtonCollisionBox::get_half_extents);
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "half_extents"), "set_half_extents", "get_half_extents");
	}

private:
	vec3 half_extents;
};

// -----------------------------------------------------------------------
// Capsule collision shape (axis along Y in local)
// -----------------------------------------------------------------------
class NewtonCollisionCapsule : public NewtonCollision {
	GDCLASS(NewtonCollisionCapsule, NewtonCollision);
public:
	NewtonCollisionCapsule(real_t p_radius = 0.5, real_t p_height = 1.0)
		: radius(p_radius), height(p_height) {}

	virtual ShapeType get_shape_type() const override { return ShapeType::CAPSULE; }

	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
		real_t half_h = MAX(height * 0.5 - radius, 0.0);
		vec3 center(0, 0, 0);
		if (local_dir.y > 0.0) center.y = half_h;
		else if (local_dir.y < 0.0) center.y = -half_h;
		return p_transform.xform(center + local_dir * radius);
	}

	virtual mat3 compute_inertia(real_t p_mass) const override {
		// Approximation: cylinder + two hemispheres.  Use known formula.
		real_t r2 = radius * radius;
		real_t h2 = height * height;
		real_t I_xy = p_mass * (3.0 * r2 + h2) / 12.0;
		real_t I_z  = p_mass * r2 * 0.5;
		return mat3().scaled(vec3(I_xy, I_xy, I_z));
	}

	virtual aabb get_local_aabb() const override {
		real_t half_h = height * 0.5;
		return aabb(vec3(-radius, -half_h, -radius), vec3(radius * 2, height, radius * 2));
	}

	real_t get_radius() const { return radius; }
	void set_radius(real_t p_r) { radius = MAX(p_r, 0.0); }
	real_t get_height() const { return height; }
	void set_height(real_t p_h) { height = MAX(p_h, 0.0); }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_radius", "r"), &NewtonCollisionCapsule::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &NewtonCollisionCapsule::get_radius);
		ClassDB::bind_method(D_METHOD("set_height", "h"), &NewtonCollisionCapsule::set_height);
		ClassDB::bind_method(D_METHOD("get_height"), &NewtonCollisionCapsule::get_height);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
	}

private:
	real_t radius;
	real_t height;
};

} // namespace newton

#endif // NEWTON_COLLISION_NEWTON_COLLISION_H