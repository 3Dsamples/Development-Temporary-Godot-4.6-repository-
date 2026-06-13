// File 300: modules/vienna/src/servers/vienna_physics_server_3d.h
// ViennaPhysicsServer3D – a Godot PhysicsServer3DExtension that replaces
// the default physics engine with the ViennaPhysicsEngine, integrated with
// Gaia's broad‑phase BVH and GJK narrow‑phase for high‑performance simulations.

#ifndef VIENNA_SERVERS_PHYSICS_SERVER_3D_H
#define VIENNA_SERVERS_PHYSICS_SERVER_3D_H

#include "servers/physics_server_3d.h"
#include "servers/physics_server_3d_helpers.h"

// Vienna core
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

// Vienna world and components
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../collision/vienna_shape.h"
#include "../collision/vienna_compound_shape.h"
#include "../joints/vienna_joint.h"
#include "../joints/vienna_ball_joint.h"
#include "../joints/vienna_hinge_joint.h"
#include "../joints/vienna_slider_joint.h"
#include "../joints/vienna_fixed_joint.h"
#include "../joints/vienna_distance_joint.h"
#include "../joints/vienna_rope_joint.h"
#include "../materials/vienna_material.h"

// Gaia broad‑phase for contact detection
#include "../../../gaia/src/collision_detector/broad_phase.h"

namespace vienna {

class ViennaPhysicsServer3D : public PhysicsServer3DExtension {
	GDCLASS(ViennaPhysicsServer3D, PhysicsServer3DExtension);

	// Internal structures for tracking bones (bodies) and joints
	struct BodyInfo {
		RID self;
		body_id id;
		BodyType type;
		bool active;
		Ref<ViennaBody> vienna_body;
		Ref<ViennaShape> shape;
		material_id material;
	};

	struct JointInfo {
		RID self;
		joint_id id;
		JointType type;
		bool active;
		Ref<ViennaJoint> vienna_joint;
	};

public:
	ViennaPhysicsServer3D();
	virtual ~ViennaPhysicsServer3D();

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

	// Additional: load options from a JSON file
	void load_options(const String &path);

private:
	void _step_vienna_world(real_t dt);
	void _sync_bodies_to_vienna();
	void _sync_vienna_to_bodies();

	ViennaWorld *vienna_world;
	gaia::collision::BroadPhase broad_phase;
	vec3 gravity;
	real_t step_size;
	bool active;
	int solver_iterations;

	// Containers
	HashMap<RID, BodyInfo> body_map;
	HashMap<RID, JointInfo> joint_map;
	HashMap<body_id, RID> id_to_rid;   // maps Vienna’s body/joint IDs back to RIDs
	HashMap<material_id, Ref<ViennaMaterial>> mat_map;

	// Shape mapping
	HashMap<RID, Ref<ViennaShape>> shape_map;

	// Next IDs
	body_id next_body_id;
	joint_id next_joint_id;
	material_id next_material_id;
};

} // namespace vienna

#endif // VIENNA_SERVERS_PHYSICS_SERVER_3D_H