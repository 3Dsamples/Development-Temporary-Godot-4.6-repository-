// File 05: modules/gaia/src/collision_detector/broad_phase.h

#ifndef GAIA_COLLISION_BROAD_PHASE_H
#define GAIA_COLLISION_BROAD_PHASE_H

#include "../bvh/bvh.h"
#include "../bvh/aabb.h"

#include "core/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace gaia::collision {

/**
 * Broad-phase collision detector.
 * Maintains a set of axis-aligned bounding boxes (AABBs) associated with
 * integer handles and efficiently reports all overlapping pairs.
 *
 * This implementation uses the Gaia BVH for static and dynamic objects.
 * Handles that are marked as inactive are ignored.
 */

class BroadPhase {
public:
	struct Object {
		AABB aabb;
		uint32_t handle; // external identifier
		bool active;
		Object() : active(true) {}
	};

	typedef void (*PairCallback)(uint32_t handle_a, uint32_t handle_b, void *userdata);

private:
	HashMap<uint32_t, int32_t> handle_to_index; // maps external handle -> internal index
	LocalVector<Object> objects;
	LocalVector<uint32_t> handles; // parallel to objects
	LocalVector<AABB> aabbs;       // for BVH rebuild
	bvh::BVH bvh;
	bool bvh_dirty;

public:
	BroadPhase() : bvh_dirty(true) {}

	// Add a new object. Returns false if handle already exists.
	bool add_object(uint32_t p_handle, const AABB &p_aabb) {
		if (handle_to_index.has(p_handle)) return false;
		int32_t idx = objects.size();
		Object obj;
		obj.handle = p_handle;
		obj.aabb = p_aabb;
		obj.active = true;
		objects.push_back(obj);
		handles.push_back(p_handle);
		handle_to_index[p_handle] = idx;
		bvh_dirty = true;
		return true;
	}

	// Remove an object by handle.
	bool remove_object(uint32_t p_handle) {
		HashMap<uint32_t, int32_t>::Iterator it = handle_to_index.find(p_handle);
		if (!it) return false;
		int32_t idx = it->value;
		// Swap with last and pop
		int32_t last_idx = objects.size() - 1;
		if (idx != last_idx) {
			objects[idx] = objects[last_idx];
			handles[idx] = handles[last_idx];
			handle_to_index[handles[idx]] = idx;
		}
		objects.resize(last_idx);
		handles.resize(last_idx);
		handle_to_index.remove(it);
		bvh_dirty = true;
		return true;
	}

	// Update an object's AABB and active flag.
	bool update_object(uint32_t p_handle, const AABB &p_aabb, bool p_active = true) {
		HashMap<uint32_t, int32_t>::Iterator it = handle_to_index.find(p_handle);
		if (!it) return false;
		int32_t idx = it->value;
		objects[idx].aabb = p_aabb;
		objects[idx].active = p_active;
		bvh_dirty = true;
		return true;
	}

	// Get object count (including inactive).
	int32_t get_object_count() const { return objects.size(); }

	// Rebuild internal BVH if needed.
	void rebuild_if_needed() {
		if (!bvh_dirty) return;

		aabbs.clear();
		for (int32_t i = 0; i < objects.size(); ++i) {
			if (objects[i].active) {
				aabbs.push_back(objects[i].aabb);
			}
		}
		bvh.build_final(aabbs);
		bvh_dirty = false;
	}

	// Find all overlapping active pairs and invoke callback.
	void find_pairs(PairCallback p_callback, void *p_userdata) {
		rebuild_if_needed();
		// Brute-force over active objects for correctness; could be enhanced with BVH traversal but we
		// must ensure we capture all pairs, including those not in BVH due to rebuild.
		for (int32_t i = 0; i < objects.size(); ++i) {
			if (!objects[i].active) continue;
			const Object &obj_a = objects[i];
			// Query BVH for overlapping AABBs, but we also need to check other objects that may not be in BVH
			// (BVH contains only active objects). Because we rebuilt from aabbs list which excludes inactive,
			// we need to map returned primitive indices back to actual objects.
			// Instead, we rely on the BVH's intersect query but we must handle index mapping carefully.
			// We'll store a parallel mapping from BVH primitive index -> object index after rebuild.
		}
		// Better approach: after rebuild, we know which objects are active and can directly brute-force over active list,
		// which is O(n^2) but simple; for large n BVH can accelerate. We'll implement BVH-accelerated pair detection.
		_find_pairs_bvh(p_callback, p_userdata);
	}

private:
	// BVH-accelerated pair finding.
	void _find_pairs_bvh(PairCallback p_callback, void *p_userdata) {
		// After rebuild, the BVH holds all active AABBs. The primitive index in BVH corresponds to the
		// order in `aabbs` array. We must map that back to original object indices.
		LocalVector<int32_t> active_indices; // maps BVH primitive -> objects index
		for (int32_t i = 0; i < objects.size(); ++i) {
			if (objects[i].active) {
				active_indices.push_back(i);
			}
		}

		// For each active object, query BVH and report pairs with larger index to avoid duplicates.
		for (int32_t i = 0; i < active_indices.size(); ++i) {
			int32_t obj_idx = active_indices[i];
			const Object &obj_a = objects[obj_idx];
			bvh.query_intersect(obj_a.aabb, [&](int32_t prim_idx) {
				if (prim_idx <= i) return; // avoid self and duplicate pairs (i < j ensures each pair once)
				int32_t other_obj_idx = active_indices[prim_idx];
				const Object &obj_b = objects[other_obj_idx];
				if (bvh::intersects(obj_a.aabb, obj_b.aabb)) {
					p_callback(obj_a.handle, obj_b.handle, p_userdata);
				}
			});
		}
	}
};

} // namespace gaia::collision

#endif // GAIA_COLLISION_BROAD_PHASE_H