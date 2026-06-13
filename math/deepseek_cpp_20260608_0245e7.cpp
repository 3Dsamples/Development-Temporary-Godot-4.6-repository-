// File 338: modules/wicked/src/solver/wicked_island.h
// WickedIsland – partitions bodies and joints into independent islands
// for parallel solving. Each island contains bodies, contacts, and joints
// that are connected via contact points or joint constraints. Union‑find
// algorithm is used for fast island construction.

#ifndef WICKED_SOLVER_ISLAND_H
#define WICKED_SOLVER_ISLAND_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../core/wicked_types.h"
#include "../core/wicked_constants.h"
#include "../bodies/wicked_body.h"
#include "../joints/wicked_joint.h"

namespace wicked {

// Forward declaration for the contact point used during solving
struct WickedContactPoint;

class WickedIsland : public RefCounted {
    GDCLASS(WickedIsland, RefCounted);

public:
    WickedIsland() : sleeping(false) {}
    virtual ~WickedIsland() {}

    // --- Island properties ---
    bool is_sleeping() const { return sleeping; }
    void set_sleeping(bool p_sleep) { sleeping = p_sleep; }

    // --- Body access ---
    void add_body(body_id p_id, WickedBody *p_body);
    void remove_body(body_id p_id);
    int get_body_count() const { return island_bodies.size(); }
    WickedBody *get_body(body_id p_id) const;
    bool contains_body(body_id p_id) const;

    // --- Contact access ---
    void add_contact(const WickedContactPoint &p_contact);
    LocalVector<WickedContactPoint> &get_contacts() { return contacts; }
    const LocalVector<WickedContactPoint> &get_contacts() const { return contacts; }
    void clear_contacts() { contacts.clear(); }

    // --- Joint access ---
    void add_joint(const Ref<WickedJoint> &p_joint);
    LocalVector<Ref<WickedJoint>> &get_joints() { return island_joints; }
    const LocalVector<Ref<WickedJoint>> &get_joints() const { return island_joints; }
    void clear_joints() { island_joints.clear(); }

    // --- Island building (union‑find based) ---
    void build(const HashMap<body_id, WickedBody *> &p_all_bodies,
               const HashMap<joint_id, Ref<WickedJoint>> &p_all_joints,
               const LocalVector<std::pair<body_id, body_id>> &p_contact_pairs);

    // Return all islands that were built.
    const LocalVector<WickedIsland *> &get_islands() const { return islands; }

    // Reset for next frame.
    void clear();

private:
    HashMap<body_id, WickedBody *> island_bodies;
    LocalVector<WickedContactPoint> contacts;
    LocalVector<Ref<WickedJoint>> island_joints;
    bool sleeping;

    // When used as manager: list of islands created during build().
    LocalVector<WickedIsland *> islands;           // raw pointers into island_storage
    LocalVector<WickedIsland> island_storage;      // actual island instances
};

} // namespace wicked

#endif // WICKED_SOLVER_ISLAND_H