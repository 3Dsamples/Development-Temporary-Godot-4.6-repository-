// File 348: modules/wicked/src/solver/wicked_island.cpp
// WickedIsland builder – partitions bodies and joints into independent islands
// using union‑find.  Bodies connected by contacts or joints are placed into
// the same island.  Each island's contacts and joints are then distributed
// accordingly.  The build method is fully implemented here for clarity,
// while the island body/contact/joint storage is accessed via inline methods
// defined in the header.

#include "wicked_island.h"
#include "../bodies/wicked_body.h"
#include "../joints/wicked_joint.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/typedefs.h"

namespace wicked {

void WickedIsland::add_body(body_id p_id, WickedBody *p_body) {
    ERR_FAIL_COND(!p_body);
    island_bodies[p_id] = p_body;
}

void WickedIsland::remove_body(body_id p_id) {
    island_bodies.erase(p_id);
}

WickedBody *WickedIsland::get_body(body_id p_id) const {
    HashMap<body_id, WickedBody *>::ConstIterator it = island_bodies.find(p_id);
    return it ? it->value : nullptr;
}

bool WickedIsland::contains_body(body_id p_id) const {
    return island_bodies.has(p_id);
}

void WickedIsland::add_contact(const WickedContactPoint &p_contact) {
    contacts.push_back(p_contact);
}

void WickedIsland::add_joint(const Ref<WickedJoint> &p_joint) {
    ERR_FAIL_COND(p_joint.is_null());
    island_joints.push_back(p_joint);
}

void WickedIsland::clear() {
    island_bodies.clear();
    contacts.clear();
    island_joints.clear();
    // Release previously built islands
    for (WickedIsland &isl : island_storage) {
        isl.island_bodies.clear();
        isl.contacts.clear();
        isl.island_joints.clear();
    }
    island_storage.clear();
    islands.clear();
    sleeping = false;
}

void WickedIsland::build(const HashMap<body_id, WickedBody *> &p_all_bodies,
                          const HashMap<joint_id, Ref<WickedJoint>> &p_all_joints,
                          const LocalVector<std::pair<body_id, body_id>> &p_contact_pairs) {
    clear();

    // Collect active dynamic/kinematic bodies (those that can be part of an island).
    LocalVector<body_id> active_body_ids;
    for (const KeyValue<body_id, WickedBody *> &kv : p_all_bodies) {
        if (kv.value && kv.value->get_activation_state() == ActivationState::ACTIVE_TAG) {
            active_body_ids.push_back(kv.key);
        }
    }
    int n_bodies = active_body_ids.size();
    if (n_bodies == 0) return;

    // Map body ID -> index in active_body_ids
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
            unite(body_to_index[pair.first], body_to_index[pair.second]);
        }
    }

    // Union bodies connected by joints
    for (const KeyValue<joint_id, Ref<WickedJoint>> &kv : p_all_joints) {
        const Ref<WickedJoint> &joint = kv.value;
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
    HashMap<int, int> root_to_island_index;
    for (const KeyValue<int, LocalVector<int>> &kv : root_to_indices) {
        int isl_idx = island_storage.size();
        island_storage.push_back(WickedIsland());
        WickedIsland &isl = island_storage[isl_idx];
        root_to_island_index[kv.key] = isl_idx;

        for (int body_idx : kv.value) {
            body_id id = active_body_ids[body_idx];
            isl.add_body(id, p_all_bodies[id]);
        }
    }

    // Distribute joints to islands
    for (const KeyValue<joint_id, Ref<WickedJoint>> &kv : p_all_joints) {
        const Ref<WickedJoint> &joint = kv.value;
        if (joint.is_null()) continue;
        body_id a = joint->get_body_a();
        body_id b = joint->get_body_b();
        if (!body_to_index.has(a) || !body_to_index.has(b)) continue;
        int root = find(body_to_index[a]);
        ERR_CONTINUE(!root_to_island_index.has(root));
        int isl_idx = root_to_island_index[root];
        island_storage[isl_idx].add_joint(joint);
    }

    // Build pointer list for the solver
    islands.clear();
    for (WickedIsland &isl : island_storage) {
        islands.push_back(&isl);
    }
}

} // namespace wicked