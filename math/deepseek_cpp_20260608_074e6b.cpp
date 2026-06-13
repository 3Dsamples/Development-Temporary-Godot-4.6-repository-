// File 274: modules/vienna/src/collision/vienna_shape.h
// Base class and primitive collision shapes for ViennaPhysicsEngine.
// Sphere, box, capsule, cylinder, cone, and convex hull with support point,
// inertia, and local AABB.

#ifndef VIENNA_COLLISION_SHAPE_H
#define VIENNA_COLLISION_SHAPE_H

#include "core/object/ref_counted.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/templates/local_vector.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaShape : public RefCounted {
	GDCLASS(ViennaShape, RefCounted);

public:
	ViennaShape() {}
	virtual ~ViennaShape() {}

	virtual ShapeType get_shape_type() const = 0;
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const = 0;
	virtual mat3 compute_inertia(real_t p_mass) const = 0;
	virtual aabb get_local_aabb() const = 0;

protected:
	static void _bind_methods() {}
};

// --- Sphere ---
class ViennaShapeSphere : public ViennaShape {
	GDCLASS(ViennaShapeSphere, ViennaShape);
	real_t radius = 0.5;
public:
	ViennaShapeSphere(real_t p_r = 0.5) : radius(MAX(p_r, 0.0)) {}
	virtual ShapeType get_shape_type() const override { return ShapeType::SPHERE; }
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
		return p_transform.xform(local_dir * radius);
	}
	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t I = (2.0f / 5.0f) * p_mass * radius * radius;
		return mat3().scaled(vec3(I, I, I));
	}
	virtual aabb get_local_aabb() const override {
		return aabb(vec3(-radius, -radius, -radius), vec3(radius * 2, radius * 2, radius * 2));
	}
	void set_radius(real_t p_r) { radius = MAX(p_r, 0.0); }
	real_t get_radius() const { return radius; }
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_radius", "radius"), &ViennaShapeSphere::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &ViennaShapeSphere::get_radius);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
	}
};

// --- Box ---
class ViennaShapeBox : public ViennaShape {
	GDCLASS(ViennaShapeBox, ViennaShape);
	vec3 half_extents = vec3(0.5, 0.5, 0.5);
public:
	ViennaShapeBox(const vec3 &p_he = vec3(0.5,0.5,0.5)) : half_extents(p_he.abs()) {}
	virtual ShapeType get_shape_type() const override { return ShapeType::BOX; }
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		vec3 support_local(
			(local_dir.x >= 0) ? half_extents.x : -half_extents.x,
			(local_dir.y >= 0) ? half_extents.y : -half_extents.y,
			(local_dir.z >= 0) ? half_extents.z : -half_extents.z);
		return p_transform.xform(support_local);
	}
	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t x = half_extents.x, y = half_extents.y, z = half_extents.z;
		real_t Ix = (1.0f/12.0f) * p_mass * (y*y + z*z);
		real_t Iy = (1.0f/12.0f) * p_mass * (x*x + z*z);
		real_t Iz = (1.0f/12.0f) * p_mass * (x*x + y*y);
		return mat3().scaled(vec3(Ix, Iy, Iz));
	}
	virtual aabb get_local_aabb() const override {
		return aabb(-half_extents, half_extents * 2.0f);
	}
	void set_half_extents(const vec3 &p_he) { half_extents = p_he.abs(); }
	const vec3 &get_half_extents() const { return half_extents; }
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_half_extents", "extents"), &ViennaShapeBox::set_half_extents);
		ClassDB::bind_method(D_METHOD("get_half_extents"), &ViennaShapeBox::get_half_extents);
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "half_extents"), "set_half_extents", "get_half_extents");
	}
};

// --- Capsule ---
class ViennaShapeCapsule : public ViennaShape {
	GDCLASS(ViennaShapeCapsule, ViennaShape);
	real_t radius = 0.5;
	real_t height = 1.0;
public:
	ViennaShapeCapsule(real_t r=0.5, real_t h=1.0) : radius(MAX(r,0.0)), height(MAX(h,0.0)) {}
	virtual ShapeType get_shape_type() const override { return ShapeType::CAPSULE; }
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
		real_t half_h = MAX(height * 0.5f - radius, 0.0f);
		vec3 center(0,0,0);
		if (local_dir.y > 0) center.y = half_h;
		else if (local_dir.y < 0) center.y = -half_h;
		return p_transform.xform(center + local_dir * radius);
	}
	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t r2 = radius * radius;
		real_t h2 = height * height;
		real_t I_xy = p_mass * (3.0f * r2 + h2) / 12.0f;
		real_t I_z  = p_mass * r2 * 0.5f;
		return mat3().scaled(vec3(I_xy, I_xy, I_z));
	}
	virtual aabb get_local_aabb() const override {
		real_t half_h = height * 0.5f;
		return aabb(vec3(-radius, -half_h, -radius), vec3(radius*2, height, radius*2));
	}
	void set_radius(real_t r) { radius = MAX(r,0.0); }
	real_t get_radius() const { return radius; }
	void set_height(real_t h) { height = MAX(h,0.0); }
	real_t get_height() const { return height; }
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_radius","r"), &ViennaShapeCapsule::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &ViennaShapeCapsule::get_radius);
		ClassDB::bind_method(D_METHOD("set_height","h"), &ViennaShapeCapsule::set_height);
		ClassDB::bind_method(D_METHOD("get_height"), &ViennaShapeCapsule::get_height);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"radius"),"set_radius","get_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"height"),"set_height","get_height");
	}
};

// --- Cylinder ---
class ViennaShapeCylinder : public ViennaShape {
	GDCLASS(ViennaShapeCylinder, ViennaShape);
	real_t radius = 0.5;
	real_t height = 1.0;
public:
	ViennaShapeCylinder(real_t r=0.5, real_t h=1.0) : radius(MAX(r,0.0)), height(MAX(h,0.0)) {}
	virtual ShapeType get_shape_type() const override { return ShapeType::CYLINDER; }
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		real_t half_h = height * 0.5f;
		real_t cap_y = (local_dir.y >= 0) ? half_h : -half_h;
		real_t rad_len = Math::sqrt(local_dir.x*local_dir.x + local_dir.z*local_dir.z);
		vec3 support_local(0, cap_y, 0);
		if (rad_len > CMP_EPSILON) {
			real_t s = radius / rad_len;
			support_local.x += local_dir.x * s;
			support_local.z += local_dir.z * s;
		}
		return p_transform.xform(support_local);
	}
	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t r2 = radius*radius, h2 = height*height;
		real_t I_xy = p_mass*(3.0f*r2 + h2)/12.0f;
		real_t I_z = p_mass * r2 * 0.5f;
		return mat3().scaled(vec3(I_xy, I_xy, I_z));
	}
	virtual aabb get_local_aabb() const override {
		real_t half_h = height*0.5f;
		return aabb(vec3(-radius, -half_h, -radius), vec3(radius*2, height, radius*2));
	}
	void set_radius(real_t r) { radius = MAX(r,0.0); }
	real_t get_radius() const { return radius; }
	void set_height(real_t h) { height = MAX(h,0.0); }
	real_t get_height() const { return height; }
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_radius","r"), &ViennaShapeCylinder::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &ViennaShapeCylinder::get_radius);
		ClassDB::bind_method(D_METHOD("set_height","h"), &ViennaShapeCylinder::set_height);
		ClassDB::bind_method(D_METHOD("get_height"), &ViennaShapeCylinder::get_height);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"radius"),"set_radius","get_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"height"),"set_height","get_height");
	}
};

// --- Cone ---
class ViennaShapeCone : public ViennaShape {
	GDCLASS(ViennaShapeCone, ViennaShape);
	real_t radius = 0.5;
	real_t height = 1.0;
public:
	ViennaShapeCone(real_t r=0.5, real_t h=1.0) : radius(MAX(r,0.0)), height(MAX(h,0.0)) {}
	virtual ShapeType get_shape_type() const override { return ShapeType::CONE; }
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir).normalized();
		real_t half_h = height * 0.5f;
		if (local_dir.y > 0.999f) return p_transform.xform(vec3(0, half_h, 0));
		real_t rad_len = Math::sqrt(local_dir.x*local_dir.x + local_dir.z*local_dir.z);
		vec3 base_point(0, -half_h, 0);
		if (rad_len > CMP_EPSILON) {
			real_t s = radius / rad_len;
			base_point.x = local_dir.x * s;
			base_point.z = local_dir.z * s;
		}
		vec3 tip(0, half_h, 0);
		real_t dot_tip = tip.dot(local_dir);
		real_t dot_base = base_point.dot(local_dir);
		return p_transform.xform(dot_tip > dot_base ? tip : base_point);
	}
	virtual mat3 compute_inertia(real_t p_mass) const override {
		real_t r2 = radius*radius, h2 = height*height;
		real_t I_xy = p_mass*(3.0f*r2 + h2)/20.0f;
		real_t I_z = p_mass * r2 * 0.3f;
		return mat3().scaled(vec3(I_xy, I_xy, I_z));
	}
	virtual aabb get_local_aabb() const override {
		real_t half_h = height*0.5f;
		return aabb(vec3(-radius, -half_h, -radius), vec3(radius*2, height, radius*2));
	}
	void set_radius(real_t r) { radius = MAX(r,0.0); }
	real_t get_radius() const { return radius; }
	void set_height(real_t h) { height = MAX(h,0.0); }
	real_t get_height() const { return height; }
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_radius","r"), &ViennaShapeCone::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &ViennaShapeCone::get_radius);
		ClassDB::bind_method(D_METHOD("set_height","h"), &ViennaShapeCone::set_height);
		ClassDB::bind_method(D_METHOD("get_height"), &ViennaShapeCone::get_height);
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"radius"),"set_radius","get_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"height"),"set_height","get_height");
	}
};

// --- Convex hull ---
class ViennaShapeConvexHull : public ViennaShape {
	GDCLASS(ViennaShapeConvexHull, ViennaShape);
	LocalVector<vec3> vertices;
	aabb local_aabb;
public:
	ViennaShapeConvexHull() {}
	void add_vertex(const vec3 &p_v) {
		vertices.push_back(p_v);
		if (vertices.size()==1) local_aabb = aabb(p_v, vec3());
		else local_aabb.expand_to(p_v);
	}
	void clear_vertices() { vertices.clear(); local_aabb = aabb(); }
	virtual ShapeType get_shape_type() const override { return ShapeType::CONVEX_HULL; }
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		real_t best_dot = -INFINITY;
		int best = 0;
		for (int i=0; i<vertices.size(); ++i) {
			real_t d = vertices[i].dot(local_dir);
			if (d > best_dot) { best_dot = d; best = i; }
		}
		return p_transform.xform(vertices[best]);
	}
	virtual mat3 compute_inertia(real_t p_mass) const override {
		if (vertices.is_empty()) return mat3().scaled(vec3(1,1,1));
		real_t inv_n = 1.0f / (real_t)vertices.size();
		real_t point_mass = p_mass * inv_n;
		mat3 I; I.scale(vec3(0,0,0));
		for (const vec3 &v : vertices) {
			real_t x=v.x, y=v.y, z=v.z;
			vec3 diag(y*y+z*z, x*x+z*z, x*x+y*y);
			mat3 contrib; contrib.set(diag.x,0,0, 0,diag.y,0, 0,0,diag.z);
			contrib[0][1] = -point_mass*x*y; contrib[1][0] = -point_mass*x*y;
			contrib[0][2] = -point_mass*x*z; contrib[2][0] = -point_mass*x*z;
			contrib[1][2] = -point_mass*y*z; contrib[2][1] = -point_mass*y*z;
			I += contrib;
		}
		return I;
	}
	virtual aabb get_local_aabb() const override { return local_aabb; }
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_vertex","vertex"), &ViennaShapeConvexHull::add_vertex);
		ClassDB::bind_method(D_METHOD("clear_vertices"), &ViennaShapeConvexHull::clear_vertices);
	}
};

} // namespace vienna

#endif // VIENNA_COLLISION_SHAPE_H