// File 275: modules/vienna/src/collision/vienna_compound_shape.h
// Vienna Compound Shape – a collection of sub‑shapes with local transforms,
// aggregated support point, inertia via parallel axis theorem, and local AABB.

#ifndef VIENNA_COLLISION_COMPOUND_H
#define VIENNA_COLLISION_COMPOUND_H

#include "vienna_shape.h"
#include "core/templates/local_vector.h"

namespace vienna {

class ViennaCompoundShape : public ViennaShape {
	GDCLASS(ViennaCompoundShape, ViennaShape);

public:
	struct SubShape {
		Ref<ViennaShape> shape;
		mat4 local_transform;      // relative to compound origin
	};

	ViennaCompoundShape() {}

	void add_sub_shape(const Ref<ViennaShape> &p_shape, const mat4 &p_local = mat4()) {
		ERR_FAIL_COND(p_shape.is_null());
		sub_shapes.push_back({p_shape, p_local});
	}

	void remove_sub_shape(int p_idx) {
		ERR_FAIL_INDEX(p_idx, sub_shapes.size());
		sub_shapes.remove_at(p_idx);
	}

	void clear_sub_shapes() { sub_shapes.clear(); }
	int get_sub_shape_count() const { return sub_shapes.size(); }

	virtual ShapeType get_shape_type() const override { return ShapeType::CONVEX_HULL; }

	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_compound_xform) const override {
		vec3 best_point;
		real_t best_dot = -INFINITY;
		for (const SubShape &sub : sub_shapes) {
			mat4 world_xform = p_compound_xform * sub.local_transform;
			vec3 pt = sub.shape->get_support(p_dir, world_xform);
			real_t d = pt.dot(p_dir);
			if (d > best_dot) { best_dot = d; best_point = pt; }
		}
		return best_point;
	}

	virtual mat3 compute_inertia(real_t p_mass) const override {
		if (sub_shapes.is_empty()) return mat3().scaled(vec3(1,1,1));
		// Compute total volume for mass distribution
		real_t total_volume = 0.0;
		for (const SubShape &sub : sub_shapes) {
			aabb local_aabb = sub.shape->get_local_aabb();
			vec3 size = local_aabb.size;
			total_volume += size.x * size.y * size.z;
		}
		if (total_volume < CMP_EPSILON) return mat3().scaled(vec3(1,1,1));

		// Compute compound centre of mass
		vec3 com(0,0,0);
		for (const SubShape &sub : sub_shapes) {
			aabb local_aabb = sub.shape->get_local_aabb();
			real_t vol = local_aabb.size.x * local_aabb.size.y * local_aabb.size.z;
			com += sub.local_transform.origin * vol;
		}
		com /= total_volume;

		// Accumulate inertia
		mat3 I; I.scale(vec3(0,0,0));
		for (const SubShape &sub : sub_shapes) {
			aabb local_aabb = sub.shape->get_local_aabb();
			real_t vol = local_aabb.size.x * local_aabb.size.y * local_aabb.size.z;
			real_t sub_mass = p_mass * vol / total_volume;
			mat3 local_inertia = sub.shape->compute_inertia(sub_mass);
			// Rotate to compound frame
			mat3 rot = sub.local_transform.basis;
			mat3 rotated = rot * local_inertia * rot.transposed();
			// Parallel axis theorem
			vec3 d = sub.local_transform.origin - com;
			real_t d2 = d.length_squared();
			mat3 offset;
			offset.set(d2 - d.x*d.x, -d.x*d.y, -d.x*d.z,
					  -d.y*d.x, d2 - d.y*d.y, -d.y*d.z,
					  -d.z*d.x, -d.z*d.y, d2 - d.z*d.z);
			// I += rotated + sub_mass * (d2*I - d*d^T)
			for (int r=0; r<3; ++r)
				for (int c=0; c<3; ++c)
					I[r][c] += rotated[r][c] + sub_mass * offset[r][c];
		}
		return I;
	}

	virtual aabb get_local_aabb() const override {
		if (sub_shapes.is_empty()) return aabb();
		aabb result;
		bool first = true;
		for (const SubShape &sub : sub_shapes) {
			aabb local = sub.shape->get_local_aabb();
			mat4 world = sub.local_transform;
			vec3 corners[8];
			vec3 min_local = local.position;
			vec3 max_local = local.position + local.size;
			corners[0] = world.xform(min_local);
			corners[1] = world.xform(vec3(max_local.x, min_local.y, min_local.z));
			corners[2] = world.xform(vec3(max_local.x, max_local.y, min_local.z));
			corners[3] = world.xform(vec3(min_local.x, max_local.y, min_local.z));
			corners[4] = world.xform(vec3(min_local.x, min_local.y, max_local.z));
			corners[5] = world.xform(vec3(max_local.x, min_local.y, max_local.z));
			corners[6] = world.xform(max_local);
			corners[7] = world.xform(vec3(min_local.x, max_local.y, max_local.z));
			for (int i=0; i<8; ++i) {
				if (first) { result.set_position(corners[i]); result.set_size(vec3()); first = false; }
				else result.expand_to(corners[i]);
			}
		}
		return result;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_sub_shape","shape","local_transform"), &ViennaCompoundShape::add_sub_shape, DEFVAL(mat4()));
		ClassDB::bind_method(D_METHOD("remove_sub_shape","index"), &ViennaCompoundShape::remove_sub_shape);
		ClassDB::bind_method(D_METHOD("clear_sub_shapes"), &ViennaCompoundShape::clear_sub_shapes);
		ClassDB::bind_method(D_METHOD("get_sub_shape_count"), &ViennaCompoundShape::get_sub_shape_count);
	}

private:
	LocalVector<SubShape> sub_shapes;
};

} // namespace vienna

#endif // VIENNA_COLLISION_COMPOUND_H