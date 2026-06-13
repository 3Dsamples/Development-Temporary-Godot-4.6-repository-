// File 184: modules/newton/src/solver/newton_island.h
// NewtonIsland – partitions bodies and joints into independent islands
// for parallel solving. Each island contains bodies, contacts, and joints
// that are connected via contact points or joint constraints. Islands
// that are entirely static or sleeping are skipped by the solver.

#ifndef NEWTON_SOLVER_NEWTON_ISLAND_H
#define NEWTON_SOLVER_NEWTON_ISLAND_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"

namespace newton {

// Forward declare NewtonContactPoint (defined in newton_solver.h or separately)
struct NewtonContactPoint;

class NewtonIsland : public RefCounted {
	GDCLASS(NewtonIsland, RefCounted);

public:
	NewtonIsland() : sleeping(false) {}
	virtual ~NewtonIsland() {}

	// --- Island properties ---
	bool is_sleeping() const { return sleeping; }
	void set_sleeping(bool p_sleep) { sleeping = p_sleep; }

	// --- Body access ---
	void add_body(body_id p_id, NewtonBody *p_body);
	void remove_body(body_id p_id);
	int get_body_count() const { return island_bodies.size(); }
	NewtonBody *get_body(body_id p_id) const;
	bool contains_body(body_id p_id) const;

	// --- Contact access ---
	void add_contact(const NewtonContactPoint &p_contact);
	const LocalVector<NewtonContactPoint> &get_contacts() const { return contacts; }
	void clear_contacts() { contacts.clear(); }

	// --- Joint access ---
	void add_joint(const Ref<NewtonJoint> &p_joint);
	const LocalVector<Ref<NewtonJoint>> &get_joints() const { return island_joints; }
	void clear_joints() { island_joints.clear(); }

	// --- Island builder (union‑find based) ---
	void build(const HashMap<body_id, Ref<NewtonBody>> &p_all_bodies,
			   const HashMap<joint_id, Ref<NewtonJoint>> &p_all_joints,
			   const LocalVector<std::pair<body_id, body_id>> &p_contact_pairs);

	// Return all islands that were built.
	const LocalVector<NewtonIsland *> &get_islands() const { return islands; }

	// Reset for next frame.
	void clear();

private:
	// Bodies belonging to this island (one island per instance when used as container)
	HashMap<body_id, NewtonBody *> island_bodies;
	LocalVector<NewtonContactPoint> contacts;
	LocalVector<Ref<NewtonJoint>> island_joints;
	bool sleeping;

	// When used as manager: list of islands created during build().
	LocalVector<NewtonIsland *> islands; // raw pointers; memory managed by the manager itself.
	LocalVector<NewtonIsland> island_storage; // actual island instances.
};

} // namespace newton

#endif // NEWTON_SOLVER_NEWTON_ISLAND_H