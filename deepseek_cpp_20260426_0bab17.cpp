// File 180: modules/newton/src/collision/newton_collision_cylinder.h
// Cylinder collision shape (aligned along local Y axis).
// Provides support point, inertia, and local AABB for GJK narrow‑phase.

#ifndef NEWTON_COLLISION_NEWTON_CYLINDER_H
#define NEWTON_COLLISION_NEWTON_CYLINDER_H

#include "newton_collision.h"

namespace newton {

class NewtonCollisionCylinder : public NewtonCollision {
	GDCLASS(NewtonCollisionCylinder, NewtonCollision);

public:
	NewtonCollisionCylinder(real_t p_radius = 0.5, real_t p_height = 1.0)
		: radius(p_radius), height(p_height) {}

	virtual ShapeType get_shape_type() const override { return ShapeType::CYLINDER; }

	// Support point: pick the cap centre (top / bottom) based on direction's Y sign,
	// then add radial component scaled by radius.
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		real_t half_h = height * 0.5;
		real_t cap_y = (local_dir.y >= 0) ? half_h : -half_h;
		// Radial direction (ignore Y component)
		real_t rad_len = Math::sqrt(local_dir.x * local_dir.x + local_dir.z * local_dir.z);
		vec3 support_local(0, cap_y, 0);
		if (rad_len > CMP_EPSILON) {
			real_t s = radius / rad_len;
			support_local.x += local_dir.x * s;
			support_local.z += local_dir.z * s;
		}
		return p_transform.xform(support_local);
	}

	virtual mat3 compute_inertia(real_t p_mass) const override {
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
		ClassDB::bind_method(D_METHOD("set_radius", "r"), &NewtonCollisionCylinder::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &NewtonCollisionCylinder::get_radius);
		ClassDB::bind_method(D_METHOD("set_height", "h"), &NewtonCollisionCylinder::set_height);
		ClassDB::bind_method(D_METHOD("get_height"), &NewtonCollisionCylinder::get_height);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
	}

private:
	real_t radius;
	real_t height;
};

} // namespace newton

#endif // NEWTON_COLLISION_NEWTON_CYLINDER_H