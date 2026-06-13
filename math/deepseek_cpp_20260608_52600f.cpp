// File 287: modules/vienna/src/solver/vienna_island.h
// ViennaIsland – partitions bodies and joints into independent islands
// for parallel solving. Each island contains bodies, contacts, and joints
// that are connected via contact points or joint constraints.

#ifndef VIENNA_SOLVER_ISLAND_H
#define VIENNA_SOLVER_ISLAND_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../bodies/vienna_body.h"
#include "../joints/vienna_joint.h"

namespace vienna {

class ViennaIsland : public RefCounted {
	GDCLASS(ViennaIsland, RefCounted);

public:
	ViennaIsland() : sleeping(false) {}
	virtual ~ViennaIsland() {}

	// --- Island properties ---
	bool is_sleeping() const { return sleeping; }
	void set_sleeping(bool p_sleep) { sleeping = p_sleep; }

	// --- Body access ---
	void add_body(body_id p_id, ViennaBody *p_body) {
		ERR_FAIL_COND(!p_body);
		island_bodies[p_id] = p_body;
	}
	void remove_body(body_id p_id) { island_bodies.erase(p_id); }
	int get_body_count() const { return island_bodies.size(); }
	ViennaBody *get_body(body_id p_id) const {
		HashMap<body_id, ViennaBody *>::ConstIterator it = island_bodies.find(p_id);
		return it ? it->value : nullptr;
	}
	bool contains_body(body_id p_id) const {
		return island_bodies.has(p_id);
	}

	// --- Contact access ---
	void add_contact(const ViennaContactPoint &p_contact) { contacts.push_back(p_contact); }
	LocalVector<ViennaContactPoint> &get_contacts() { return contacts; }
	const LocalVector<ViennaContactPoint> &get_contacts() const { return contacts; }
	void clear_contacts() { contacts.clear(); }

	// --- Joint access ---
	void add_joint(const Ref<ViennaJoint> &p_joint) {
		ERR_FAIL_COND(p_joint.is_null());
		island_joints.push_back(p_joint);
	}
	LocalVector<Ref<ViennaJoint>> &get_joints() { return island_joints; }
	const LocalVector<Ref<ViennaJoint>> &get_joints() const { return island_joints; }
	void clear_joints() { island_joints.clear(); }

	// --- Island building (union‑find based) ---
	void build(const HashMap<body_id, Ref<ViennaBody>> &p_all_bodies,
			   const HashMap<joint_id, Ref<ViennaJoint>> &p_all_joints,
			   const LocalVector<std::pair<body_id, body_id>> &p_contact_pairs);

	// Return all islands that were built.
	const LocalVector<ViennaIsland *> &get_islands() const { return islands; }

	// Reset for next frame.
	void clear();

private:
	HashMap<body_id, ViennaBody *> island_bodies;
	LocalVector<ViennaContactPoint> contacts;
	LocalVector<Ref<ViennaJoint>> island_joints;
	bool sleeping;

	// When used as manager: list of islands created during build().
	LocalVector<ViennaIsland *> islands;           // raw pointers into island_storage
	LocalVector<ViennaIsland> island_storage;      // actual island instances
};

// ---------------------------------------------------------------------------
// Inline implementation of island building
// ---------------------------------------------------------------------------
inline void ViennaIsland::clear() {
	island_bodies.clear();
	contacts.clear();
	island_joints.clear();
	for (ViennaIsland &isl : island_storage) {
		isl.island_bodies.clear();
		isl.contacts.clear();
		isl.island_joints.clear();
	}
	island_storage.clear();
	islands.clear();
	sleeping = false;
}

inline void ViennaIsland::build(const HashMap<body_id, Ref<ViennaBody>> &p_all_bodies,
								const HashMap<joint_id, Ref<ViennaJoint>> &p_all_joints,
								const LocalVector<std::pair<body_id, body_id>> &p_contact_pairs) {
	clear();

	// Collect active dynamic/kinematic bodies
	LocalVector<body_id> active_body_ids;
	for (const KeyValue<body_id, Ref<ViennaBody>> &kv : p_all_bodies) {
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

	// Union‑find
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
	for (const KeyValue<joint_id, Ref<ViennaJoint>> &kv : p_all_joints) {
		const Ref<ViennaJoint> &joint = kv.value;
		if (joint.is_null()) continue;
		body_id a = joint->get_body_a();
		body_id b = joint->get_body_b();
		if (body_to_index.has(a) && body_to_index.has(b)) {
			int ia = body_to_index[a];
			int ib = body_to_index[b];
			unite(ia, ib);
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
	HashMap<int, int> root_to_island_index;
	for (const KeyValue<int, LocalVector<int>> &kv : root_to_indices) {
		int isl_idx = island_storage.size();
		island_storage.push_back(ViennaIsland());
		ViennaIsland &isl = island_storage[isl_idx];
		root_to_island_index[kv.key] = isl_idx;

		for (int body_idx : kv.value) {
			body_id id = active_body_ids[body_idx];
			isl.add_body(id, p_all_bodies[id].ptr());
		}
	}

	// Distribute joints to islands
	for (const KeyValue<joint_id, Ref<ViennaJoint>> &kv : p_all_joints) {
		const Ref<ViennaJoint> &joint = kv.value;
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
	for (ViennaIsland &isl : island_storage) {
		islands.push_back(&isl);
	}
}

} // namespace vienna

#endif // VIENNA_SOLVER_ISLAND_H