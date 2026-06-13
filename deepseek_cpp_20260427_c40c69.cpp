// File 205: modules/newton/src/servers/newton_physics_server_3d.h
// NewtonPhysicsServer3D – a Godot PhysicsServer3DExtension that replaces
// the default physics engine with Newton Dynamics 4.0, integrated with
// Gaia's broad‑phase and Genesis' multi‑solver for an all‑in‑one solution.

#ifndef NEWTON_SERVERS_PHYSICS_SERVER_3D_H
#define NEWTON_SERVERS_PHYSICS_SERVER_3D_H

#include "servers/physics_server_3d.h"
#include "servers/physics_server_3d_helpers.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

// Forward declarations
namespace newton {
class NewtonWorld;
class NewtonBody;
class NewtonJoint;
class NewtonCollision;
}

namespace genesis {
class GenesisPhysicsServer3D;  // optional hybrid
}

namespace gaia {
namespace collision {
class BroadPhase;
}
}

class NewtonPhysicsServer3D : public PhysicsServer3DExtension {
	GDCLASS(NewtonPhysicsServer3D, PhysicsServer3DExtension);

	// Internal data
	struct BodyData {
		RID self;
		newton::body_id nid;
		newton::BodyType body_type;
		Ref<newton::NewtonBody> newton_body;
		Ref<newton::NewtonCollision> newton_collision;
		bool active;
	};

	struct JointData {
		RID self;
		newton::joint_id nid;
		newton::JointType joint_type;
		Ref<newton::NewtonJoint> newton_joint;
	};

public:
	NewtonPhysicsServer3D();
	virtual ~NewtonPhysicsServer3D();

	// PhysicsServer3D overrides (core)
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

	// Custom methods (accessible via GDScript)
	void load_options(const String &path);

private:
	void _sync_body_to_newton(const RID &p_body);
	void _sync_newton_to_body(newton::body_id p_nid);

	// World
	newton::NewtonWorld *newton_world;
	bool active;
	real_t step_size;

	// Body storage
	HashMap<RID, BodyData> body_map;
	HashMap<newton::body_id, RID> newton_to_rid;

	// Joint storage
	HashMap<RID, JointData> joint_map;

	// Shape storage (mapping shape RID to collision ref)
	HashMap<RID, Ref<newton::NewtonCollision>> shape_map;

	// Material storage
	HashMap<newton::material_id, Ref<newton::NewtonMaterial>> mat_map;

	// Next IDs
	newton::body_id next_body_id;
	newton::joint_id next_joint_id;
	newton::material_id next_material_id;

	// Gravity
	Vector3 gravity;
};

#endif // NEWTON_SERVERS_PHYSICS_SERVER_3D_H