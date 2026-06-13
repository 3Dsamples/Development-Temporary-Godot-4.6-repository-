// File 210: modules/newton/src/collision/newton_collision_cone.h
// Cone collision shape (aligned along local Y axis, tip at origin + height/2).
// Provides support point, inertia tensor, and local AABB for GJK narrow-phase.

#ifndef NEWTON_COLLISION_NEWTON_CONE_H
#define NEWTON_COLLISION_NEWTON_CONE_H

#include "newton_collision.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonCollisionCone : public NewtonCollision {
	GDCLASS(NewtonCollisionCone, NewtonCollision);

public:
	NewtonCollisionCone(real_t p_radius = 0.5, real_t p_height = 1.0)
		: radius(p_radius), height(p_height) {}

	virtual ShapeType get_shape_type() const override { return ShapeType::CONE; }

	// Support point for a cone.
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
		real_t half_h = height * 0.5f;
		// Cone tip is at (0, half_h, 0), base circle at y = -half_h.
		// If direction has a positive Y component, the tip is the support.
		// Otherwise, find the support on the base circle edge or the conical surface.
		if (local_dir.y > 0.999f) {
			return p_transform.xform(vec3(0, half_h, 0));
		}
		// Radial direction
		real_t rad_len = Math::sqrt(local_dir.x * local_dir.x + local_dir.z * local_dir.z);
		if (rad_len < CMP_EPSILON) {
			// Direction points straight down; support is a point on the base circle (any).
			return p_transform.xform(vec3(radius, -half_h, 0));
		}
		// Compute the point on the conical surface.
		// The cone surface normal at base is outward and downward; for a cone, 
		// the supporting point for a direction can be computed by intersecting
		// the ray from tip to base edge with the plane perpendicular to direction...
		// Simplification: clamp between tip and base edge.
		real_t s = radius / rad_len;
		vec3 base_point(s * local_dir.x, -half_h, s * local_dir.z);
		// If direction is nearly perpendicular to the cone axis, the support
		// point lies on the rim of the base.
		// If direction has a more upward component, the tip is better.
		// We use a simple heuristic: if local_dir.y > 0, use tip; otherwise
		// use projection of tip+base edge.
		if (local_dir.y > 0.0f) {
			// Linear combination of tip and base point that maximizes dot.
			vec3 tip(0, half_h, 0);
			real_t dot_tip = tip.dot(local_dir);
			real_t dot_base = base_point.dot(local_dir);
			if (dot_tip > dot_base) return p_transform.xform(tip);
			else return p_transform.xform(base_point);
		}
		return p_transform.xform(base_point);
	}

	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t r2 = radius * radius;
		real_t h2 = height * height;
		real_t I_xy = p_mass * (3.0f * r2 + h2) / 20.0f;   // approximate
		real_t I_z  = p_mass * r2 * 0.3f;
		return mat3().scaled(vec3(I_xy, I_xy, I_z));
	}

	virtual aabb get_local_aabb() const override {
		real_t half_h = height * 0.5f;
		return aabb(vec3(-radius, -half_h, -radius), vec3(radius * 2.0f, height, radius * 2.0f));
	}

	real_t get_radius() const { return radius; }
	void set_radius(real_t p_r) { radius = MAX(p_r, 0.0); }
	real_t get_height() const { return height; }
	void set_height(real_t p_h) { height = MAX(p_h, 0.0); }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_radius", "r"), &NewtonCollisionCone::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &NewtonCollisionCone::get_radius);
		ClassDB::bind_method(D_METHOD("set_height", "h"), &NewtonCollisionCone::set_height);
		ClassDB::bind_method(D_METHOD("get_height"), &NewtonCollisionCone::get_height);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
	}

private:
	real_t radius;
	real_t height;
};

} // namespace newton

#endif // NEWTON_COLLISION_NEWTON_CONE_H