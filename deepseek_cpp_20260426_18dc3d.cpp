// File 181: modules/newton/src/collision/newton_compound_collision.h
// Compound collision shape – a collection of convex sub‑shapes each with
// a local transform. Support point queries iterate over all sub‑shapes
// and return the farthest. Inertia is computed as sum of sub‑inertias
// using the parallel axis theorem.

#ifndef NEWTON_COLLISION_COMPOUND_H
#define NEWTON_COLLISION_COMPOUND_H

#include "newton_collision.h"
#include "core/templates/local_vector.h"

namespace newton {

class NewtonCompoundCollision : public NewtonCollision {
	GDCLASS(NewtonCompoundCollision, NewtonCollision);

public:
	struct SubShape {
		Ref<NewtonCollision> shape;   // sub‑shape (sphere, box, etc.)
		mat4 local_transform;         // relative to compound origin

		SubShape() {}
		SubShape(const Ref<NewtonCollision> &p_shape, const mat4 &p_local)
			: shape(p_shape), local_transform(p_local) {}
	};

	NewtonCompoundCollision() {}

	// Add a sub‑shape.
	void add_sub_shape(const Ref<NewtonCollision> &p_shape, const mat4 &p_local = mat4()) {
		ERR_FAIL_COND(p_shape.is_null());
		sub_shapes.push_back(SubShape(p_shape, p_local));
	}

	void remove_sub_shape(int p_index) {
		ERR_FAIL_INDEX(p_index, sub_shapes.size());
		sub_shapes.remove_at(p_index);
	}

	void clear_sub_shapes() { sub_shapes.clear(); }

	int get_sub_shape_count() const { return sub_shapes.size(); }
	const SubShape &get_sub_shape(int p_idx) const { return sub_shapes[p_idx]; }

	virtual ShapeType get_shape_type() const override { return ShapeType::CONVEX_HULL; } // treat as convex hull

	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_compound_transform) const override {
		vec3 best_point;
		real_t best_dot = -INFINITY;
		for (const SubShape &sub : sub_shapes) {
			// World transform of sub‑shape = compound_world * local
			mat4 sub_world = p_compound_transform * sub.local_transform;
			vec3 p = sub.shape->get_support(p_dir, sub_world);
			real_t d = p.dot(p_dir);
			if (d > best_dot) {
				best_dot = d;
				best_point = p;
			}
		}
		return best_point;
	}

	virtual mat3 compute_inertia(real_t p_mass) const override {
		// Compute total volume for mass proportion, then sum inertia tensors
		// using parallel axis theorem.
		real_t total_volume = 0.0;
		for (const SubShape &sub : sub_shapes) {
			aabb local_aabb = sub.shape->get_local_aabb();
			vec3 size = local_aabb.size;
			total_volume += size.x * size.y * size.z; // approximate volume
		}
		if (total_volume < CMP_EPSILON) return mat3().scaled(vec3(1,1,1));

		mat3 compound_inertia;
		compound_inertia.scale(vec3(0,0,0));
		vec3 compound_com; // center of mass of compound (approximate)
		// First pass: compute compound center of mass
		compound_com = vec3();
		for (const SubShape &sub : sub_shapes) {
			vec3 sub_com = sub.local_transform.origin;
			aabb local_aabb = sub.shape->get_local_aabb();
			real_t sub_volume = local_aabb.size.x * local_aabb.size.y * local_aabb.size.z;
			compound_com += sub_com * sub_volume;
		}
		compound_com /= total_volume;

		// Second pass: accumulate inertia
		for (const SubShape &sub : sub_shapes) {
			vec3 sub_com = sub.local_transform.origin;
			aabb local_aabb = sub.shape->get_local_aabb();
			real_t sub_volume = local_aabb.size.x * local_aabb.size.y * local_aabb.size.z;
			real_t sub_mass = p_mass * sub_volume / total_volume;
			mat3 local_inertia = sub.shape->compute_inertia(sub_mass);
			// Rotate local inertia to compound frame
			mat3 rot = sub.local_transform.basis;
			mat3 rotated_inertia = rot * local_inertia * rot.transposed();
			// Parallel axis theorem: I += rotated + m * (d²*I - d*d^T)
			vec3 d = sub_com - compound_com;
			real_t d2 = d.length_squared();
			mat3 d_outer;
			d_outer.set(d.x*d.x, d.x*d.y, d.x*d.z,
						d.y*d.x, d.y*d.y, d.y*d.z,
						d.z*d.x, d.z*d.y, d.z*d.z);
			for (int i = 0; i < 3; ++i)
				d_outer[i][i] -= d2; // actually d²*I - d*d^T = -(d_outer - d²*I)? We'll use formula: I += I_local + m * (dot(d,d)*I - d*d^T)
			// Manual accumulation
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					real_t I_add = rotated_inertia[r][c];
					// parallel axis: m * (d²*δ_rc - d[r]*d[c])
					if (r == c)
						I_add += sub_mass * (d2 - d[r] * d[c]);
					else
						I_add -= sub_mass * d[r] * d[c];
					compound_inertia[r][c] += I_add;
				}
			}
		}
		return compound_inertia;
	}

	virtual aabb get_local_aabb() const override {
		if (sub_shapes.is_empty()) return aabb();
		aabb result;
		bool first = true;
		for (const SubShape &sub : sub_shapes) {
			aabb local = sub.shape->get_local_aabb();
			// Transform local AABB corners to compound frame (approximate by rotating extents)
			mat4 t = sub.local_transform;
			vec3 corners[8];
			vec3 min_local = local.position;
			vec3 max_local = local.position + local.size;
			corners[0] = t.xform(min_local);
			corners[1] = t.xform(vec3(max_local.x, min_local.y, min_local.z));
			corners[2] = t.xform(vec3(max_local.x, max_local.y, min_local.z));
			corners[3] = t.xform(vec3(min_local.x, max_local.y, min_local.z));
			corners[4] = t.xform(vec3(min_local.x, min_local.y, max_local.z));
			corners[5] = t.xform(vec3(max_local.x, min_local.y, max_local.z));
			corners[6] = t.xform(max_local);
			corners[7] = t.xform(vec3(min_local.x, max_local.y, max_local.z));
			for (int i = 0; i < 8; ++i) {
				if (first) {
					result.set_position(corners[i]);
					result.set_size(vec3());
					first = false;
				} else {
					result.expand_to(corners[i]);
				}
			}
		}
		return result;
	}

	bool is_empty() const { return sub_shapes.is_empty(); }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_sub_shape", "shape", "local_transform"), &NewtonCompoundCollision::add_sub_shape, DEFVAL(mat4()));
		ClassDB::bind_method(D_METHOD("remove_sub_shape", "index"), &NewtonCompoundCollision::remove_sub_shape);
		ClassDB::bind_method(D_METHOD("clear_sub_shapes"), &NewtonCompoundCollision::clear_sub_shapes);
		ClassDB::bind_method(D_METHOD("get_sub_shape_count"), &NewtonCompoundCollision::get_sub_shape_count);
		ClassDB::bind_method(D_METHOD("get_sub_shape", "index"), &NewtonCompoundCollision::get_sub_shape);
	}

private:
	LocalVector<SubShape> sub_shapes;
};

} // namespace newton

#endif // NEWTON_COLLISION_COMPOUND_H