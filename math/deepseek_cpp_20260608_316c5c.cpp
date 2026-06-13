// File 242: modules/newton/src/collision/newton_collision_aggregate.h
// Collision aggregate – a collection of collision shapes that move together
// (e.g., parts of a compound object). The aggregate computes a combined
// AABB and provides a fast broad‑phase pruning for internal pairs.
// Supports adding static terrain, dynamic sub‑shapes, and disabling collision
// between certain pairs.

#ifndef NEWTON_COLLISION_AGGREGATE_H
#define NEWTON_COLLISION_AGGREGATE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "newton_collision.h"
#include "../bodies/newton_body.h"

namespace newton {

class NewtonCollisionAggregate : public RefCounted {
	GDCLASS(NewtonCollisionAggregate, RefCounted);

public:
	struct SubShape {
		Ref<NewtonCollision> shape;
		body_id owner_body;      // the body this shape belongs to
		mat4 local_transform;    // relative to the body's origin
		bool enabled;
	};

	NewtonCollisionAggregate() {}

	void add_shape(const Ref<NewtonCollision> &p_shape, body_id p_owner, const mat4 &p_local_transform = mat4()) {
		ERR_FAIL_COND(p_shape.is_null());
		SubShape sub;
		sub.shape = p_shape;
		sub.owner_body = p_owner;
		sub.local_transform = p_local_transform;
		sub.enabled = true;
		sub_shapes.push_back(sub);
	}

	void remove_shape(int p_index) {
		ERR_FAIL_INDEX(p_index, sub_shapes.size());
		sub_shapes.remove_at(p_index);
	}

	void clear() { sub_shapes.clear(); }

	int get_shape_count() const { return sub_shapes.size(); }
	const SubShape &get_shape(int p_idx) const { return sub_shapes[p_idx]; }

	// Disable collision between two specific shapes (by their indices).
	void disable_pair(int p_idx_a, int p_idx_b) {
		if (p_idx_a > p_idx_b) SWAP(p_idx_a, p_idx_b);
		disabled_pairs.insert(std::make_pair(p_idx_a, p_idx_b));
	}

	// Check if a pair is disabled.
	bool is_pair_disabled(int p_idx_a, int p_idx_b) const {
		if (p_idx_a > p_idx_b) SWAP(p_idx_a, p_idx_b);
		return disabled_pairs.has(std::make_pair(p_idx_a, p_idx_b));
	}

	// Compute the world‑space AABB of all shapes in the aggregate.
	aabb get_world_aabb() const {
		if (sub_shapes.is_empty()) return aabb();
		aabb result;
		bool first = true;
		for (const SubShape &sub : sub_shapes) {
			aabb local_aabb = sub.shape->get_local_aabb();
			mat4 world_xform = get_world_transform(sub.owner_body) * sub.local_transform;
			vec3 corners[8];
			vec3 min_local = local_aabb.position;
			vec3 max_local = local_aabb.position + local_aabb.size;
			corners[0] = world_xform.xform(min_local);
			corners[1] = world_xform.xform(vec3(max_local.x, min_local.y, min_local.z));
			corners[2] = world_xform.xform(vec3(max_local.x, max_local.y, min_local.z));
			corners[3] = world_xform.xform(vec3(min_local.x, max_local.y, min_local.z));
			corners[4] = world_xform.xform(vec3(min_local.x, min_local.y, max_local.z));
			corners[5] = world_xform.xform(vec3(max_local.x, min_local.y, max_local.z));
			corners[6] = world_xform.xform(max_local);
			corners[7] = world_xform.xform(vec3(min_local.x, max_local.y, max_local.z));
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

	// Set the transform of a body (used to compute world AABB).
	void set_body_transform(body_id p_id, const mat4 &p_xform) {
		body_transforms[p_id] = p_xform;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_shape", "shape", "owner_body", "local_transform"), &NewtonCollisionAggregate::add_shape, DEFVAL(mat4()));
		ClassDB::bind_method(D_METHOD("remove_shape", "index"), &NewtonCollisionAggregate::remove_shape);
		ClassDB::bind_method(D_METHOD("clear"), &NewtonCollisionAggregate::clear);
		ClassDB::bind_method(D_METHOD("get_shape_count"), &NewtonCollisionAggregate::get_shape_count);
		ClassDB::bind_method(D_METHOD("get_shape", "index"), &NewtonCollisionAggregate::get_shape);
		ClassDB::bind_method(D_METHOD("disable_pair", "idx_a", "idx_b"), &NewtonCollisionAggregate::disable_pair);
		ClassDB::bind_method(D_METHOD("is_pair_disabled", "idx_a", "idx_b"), &NewtonCollisionAggregate::is_pair_disabled);
		ClassDB::bind_method(D_METHOD("get_world_aabb"), &NewtonCollisionAggregate::get_world_aabb);
	}

private:
	mat4 get_world_transform(body_id p_owner) const {
		HashMap<body_id, mat4>::ConstIterator it = body_transforms.find(p_owner);
		if (it) return it->value;
		return mat4(); // identity
	}

	LocalVector<SubShape> sub_shapes;
	HashMap<body_id, mat4> body_transforms;
	HashSet<std::pair<int, int>> disabled_pairs;
};

} // namespace newton

#endif // NEWTON_COLLISION_AGGREGATE_H