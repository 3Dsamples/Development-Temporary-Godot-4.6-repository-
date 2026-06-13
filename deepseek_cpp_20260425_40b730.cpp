// File 08: modules/gaia/src/collision_detector/collision_object.h

#ifndef GAIA_COLLISION_OBJECT_H
#define GAIA_COLLISION_OBJECT_H

#include "narrow_phase.h"     // for ConvexShape
#include "core/math/aabb.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace gaia::collision {

/**
 * A collision object that holds a convex shape, its transform, and a cached
 * world-space AABB. It exposes an interface for use in broad and narrow phases.
 *
 * The shape is stored as a raw pointer; lifetime must be managed externally (or
 * via a Godot resource wrapper later).
 */
class CollisionObject {
public:
	CollisionObject() :
			shape(nullptr),
			handle(0),
			active(true),
			aabb_dirty(true) {}

	void set_shape(ConvexShape *p_shape) {
		shape = p_shape;
		aabb_dirty = true;
	}
	ConvexShape *get_shape() const { return shape; }

	void set_transform(const Transform3D &p_transform) {
		transform = p_transform;
		aabb_dirty = true;
	}
	const Transform3D &get_transform() const { return transform; }

	void set_handle(uint32_t p_handle) { handle = p_handle; }
	uint32_t get_handle() const { return handle; }

	void set_active(bool p_active) { active = p_active; }
	bool is_active() const { return active; }

	// Recompute the world-space AABB from the shape's arbitrary support points.
	void update_aabb() {
		ERR_FAIL_COND(!shape);
		// Build AABB from sampling the shape along the six axis-aligned directions
		// (a conservative but fast method).
		Vector3 min_point(INFINITY, INFINITY, INFINITY);
		Vector3 max_point(-INFINITY, -INFINITY, -INFINITY);

		// Directions for an axis-aligned box: ±x, ±y, ±z
		const Vector3 dirs[6] = {
			Vector3(1, 0, 0), Vector3(-1, 0, 0),
			Vector3(0, 1, 0), Vector3(0, -1, 0),
			Vector3(0, 0, 1), Vector3(0, 0, -1)
		};

		for (int i = 0; i < 6; ++i) {
			Vector3 local_dir = transform.basis.xform_inv(dirs[i]);
			Vector3 sup_local = shape->get_support(local_dir);
			Vector3 sup_world = transform.xform(sup_local);
			min_point = min_point.min(sup_world);
			max_point = max_point.max(sup_world);
		}

		aabb.position = min_point;
		aabb.size = max_point - min_point;
		aabb_dirty = false;
	}

	const AABB &get_aabb() const {
		return aabb;
	}

	bool is_aabb_dirty() const { return aabb_dirty; }

private:
	ConvexShape *shape;
	Transform3D transform;
	uint32_t handle;
	bool active;
	AABB aabb;
	bool aabb_dirty;
};

} // namespace gaia::collision

#endif // GAIA_COLLISION_OBJECT_H