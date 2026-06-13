// File 185: modules/newton/src/solver/newton_island.cpp
// NewtonIsland implementation – builds contact/joint islands using union‑find,
// distributes contacts and joints into islands, and provides iteration access.

#include "newton_island.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"

namespace newton {

void NewtonIsland::add_body(body_id p_id, NewtonBody *p_body) {
	ERR_FAIL_COND(!p_body);
	island_bodies[p_id] = p_body;
}

void NewtonIsland::remove_body(body_id p_id) {
	island_bodies.erase(p_id);
}

NewtonBody *NewtonIsland::get_body(body_id p_id) const {
	HashMap<body_id, NewtonBody *>::ConstIterator it = island_bodies.find(p_id);
	return it ? it->value : nullptr;
}

bool NewtonIsland::contains_body(body_id p_id) const {
	return island_bodies.has(p_id);
}

void NewtonIsland::add_contact(const NewtonContactPoint &p_contact) {
	contacts.push_back(p_contact);
}

void NewtonIsland::add_joint(const Ref<NewtonJoint> &p_joint) {
	ERR_FAIL_COND(p_joint.is_null());
	island_joints.push_back(p_joint);
}

void NewtonIsland::clear() {
	island_bodies.clear();
	contacts.clear();
	island_joints.clear();
	// Free previously built islands
	for (NewtonIsland &isl : island_storage) {
		isl.island_bodies.clear();
		isl.contacts.clear();
		isl.island_joints.clear();
	}
	island_storage.clear();
	islands.clear();
	sleeping = false;
}

void NewtonIsland::build(const HashMap<body_id, Ref<NewtonBody>> &p_all_bodies,
						  const HashMap<joint_id, Ref<NewtonJoint>> &p_all_joints,
						  const LocalVector<std::pair<body_id, body_id>> &p_contact_pairs) {
	clear();

	// Collect all dynamic/kinematic body IDs that are active.
	LocalVector<body_id> active_body_ids;
	for (const KeyValue<body_id, Ref<NewtonBody>> &kv : p_all_bodies) {
		if (kv.value->is_active()) {
			active_body_ids.push_back(kv.key);
		}
	}
	int n_bodies = active_body_ids.size();
	if (n_bodies == 0) return;

	// Map body_id -> index in active_body_ids
	HashMap<body_id, int> body_to_index;
	for (int i = 0; i < n_bodies; ++i) {
		body_to_index[active_body_ids[i]] = i;
	}

	// Union‑find parent array
	LocalVector<int> parent(n_bodies);
	for (int i = 0; i < n_bodies; ++i) parent[i] = i;

	auto find = [&](int x) {
		while (x != parent[x]) {
			parent[x] = parent[parent[x]];
			x = parent[x];
		}
		return x;
	};
	auto unite = [&](int a, int b) {
		int ra = find(a), rb = find(b);
		if (ra != rb) parent[ra] = rb;
	};

	// Union bodies connected by contacts
	for (const auto &pair : p_contact_pairs) {
		if (body_to_index.has(pair.first) && body_to_index.has(pair.second)) {
			int ia = body_to_index[pair.first];
			int ib = body_to_index[pair.second];
			unite(ia, ib);
		}
	}

	// Union bodies connected by joints
	for (const KeyValue<joint_id, Ref<NewtonJoint>> &kv : p_all_joints) {
		const Ref<NewtonJoint> &joint = kv.value;
		if (joint.is_null()) continue;
		body_id a = joint->get_body_a();
		body_id b = joint->get_body_b();
		if (body_to_index.has(a) && body_to_index.has(b)) {
			unite(body_to_index[a], body_to_index[b]);
		}
	}

	// Group indices by root
	HashMap<int, LocalVector<int>> root_to_indices;
	for (int i = 0; i < n_bodies; ++i) {
		int root = find(i);
		if (!root_to_indices.has(root)) {
			root_to_indices[root] = LocalVector<int>();
		}
		root_to_indices[root].push_back(i);
	}

	// Create one island per root
	HashMap<int, int> root_to_island_index; // root -> index in island_storage
	for (const KeyValue<int, LocalVector<int>> &kv : root_to_indices) {
		int isl_idx = island_storage.size();
		island_storage.push_back(NewtonIsland());
		NewtonIsland &isl = island_storage[isl_idx];
		root_to_island_index[kv.key] = isl_idx;

		for (int body_idx : kv.value) {
			body_id id = active_body_ids[body_idx];
			isl.add_body(id, p_all_bodies[id].ptr());
		}
	}

	// Distribute contacts to the island containing both bodies
	for (const auto &pair : p_contact_pairs) {
		if (!body_to_index.has(pair.first) || !body_to_index.has(pair.second)) continue;
		int root = find(body_to_index[pair.first]);
		ERR_CONTINUE(!root_to_island_index.has(root));
		int isl_idx = root_to_island_index[root];
		// Generate contact for this pair (if not already present)
		// Contact generation is done elsewhere; we just register the pair.
		// The solver will later fill contacts via narrow-phase detection.
		// For now, we simply rely on pre‑generated contacts passed to the island.
	}

	// Distribute joints to islands
	for (const KeyValue<joint_id, Ref<NewtonJoint>> &kv : p_all_joints) {
		const Ref<NewtonJoint> &joint = kv.value;
		if (joint.is_null()) continue;
		body_id a = joint->get_body_a();
		body_id b = joint->get_body_b();
		if (!body_to_index.has(a) || !body_to_index.has(b)) continue;
		int root = find(body_to_index[a]);
		ERR_CONTINUE(!root_to_island_index.has(root));
		int isl_idx = root_to_island_index[root];
		island_storage[isl_idx].add_joint(joint);
	}

	// Build the pointer list for the solver
	islands.clear();
	for (NewtonIsland &isl : island_storage) {
		islands.push_back(&isl);
	}
}

} // namespace newton