// File 260: modules/integration/unified_physics_server_3d.h
// UnifiedPhysicsServer3D – a Godot PhysicsServer3DExtension that replaces
// the default physics engine with a combination of Newton Dynamics 4.0
// (rigid bodies), Genesis (soft bodies, fluids, MPM), and Gaia (broad‑phase
// BVH and narrow‑phase). This server acts as the single point of entry for
// all physics simulation, managing spaces, bodies, joints, and queries.

#ifndef UNIFIED_PHYSICS_SERVER_3D_H
#define UNIFIED_PHYSICS_SERVER_3D_H

#include "servers/physics_server_3d.h"
#include "servers/physics_server_3d_helpers.h"

// Newton
#include "../newton/src/world/newton_world.h"
#include "../newton/src/bodies/newton_body.h"
#include "../newton/src/collision/newton_collision.h"

// Genesis
#include "../genesis/src/genesis_world.h"
#include "../genesis/src/entities/rigid_entity.h"
#include "../genesis/src/entities/fem_entity.h"
#include "../genesis/src/entities/mpm_entity.h"

// Gaia (broad‑phase)
#include "../gaia/src/collision_detector/broad_phase.h"

namespace unified {

class UnifiedPhysicsServer3D : public PhysicsServer3DExtension {
	GDCLASS(UnifiedPhysicsServer3D, PhysicsServer3DExtension);

	// Internal structures
	struct SpaceData {
		RID self;
		newton::NewtonWorld *newton_world;
		genesis::GenesisWorld *genesis_world;
		gaia::collision::BroadPhase *broad_phase;
		Vector3 gravity;
		real_t step_size;
		bool active;
	};

	struct NewtonBodyData {
		RID self;
		newton::body_id nid;
		bool active;
	};

	struct GenesisFEMData {
		RID self;
		genesis::entity_id_t gid;
		bool active;
	};

	struct GenesisMPMData {
		RID self;
		genesis::entity_id_t gid;
		bool active;
	};

	// Storage
	HashMap<RID, SpaceData> space_map;
	HashMap<RID, NewtonBodyData> newton_body_map;
	HashMap<RID, GenesisFEMData> genesis_fem_map;
	HashMap<RID, GenesisMPMData> genesis_mpm_map;

	// Shape storage (shared)
	HashMap<RID, Ref<newton::NewtonCollision>> shape_map;
	HashMap<RID, Ref<genesis::BaseEntity>> genesis_shape_map;

	// Active space for queries (the server uses a single global space by default)
	RID active_space;

public:
	UnifiedPhysicsServer3D();
	virtual ~UnifiedPhysicsServer3D();

	// PhysicsServer3D overrides
	virtual bool is_flushing_queries() const override;
	virtual int get_process_info(ProcessInfo p_info) override;

	virtual RID space_create() override;
	virtual RID area_create() override;
	virtual RID body_create() override;
	virtual RID soft_body_create() override;
	virtual RID shape_create(ShapeType p_type) override;

	virtual void body_set_space(RID p_body, RID p_space) override;
	virtual void body_set_mode(RID p_body, BodyMode p_mode) override;
	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_value) override;
	virtual Variant body_get_state(RID p_body, BodyState p_state) override;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false) override;
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) override;
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) override;

	virtual void physics_step(real_t p_step) override;
	virtual void set_active(bool p_active) override;
	virtual bool is_active() const override;

	// Additional management
	void set_space_gravity(RID p_space, const Vector3 &p_gravity);
	Vector3 get_space_gravity(RID p_space) const;

private:
	void _step_space(SpaceData &p_space, real_t p_dt);
	void _sync_newton_to_godot(const SpaceData &p_space);
	void _sync_genesis_to_godot(const SpaceData &p_space);
};

} // namespace unified

#endif // UNIFIED_PHYSICS_SERVER_3D_H