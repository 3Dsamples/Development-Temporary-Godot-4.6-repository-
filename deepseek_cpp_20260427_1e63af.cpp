// File 350: modules/wicked/src/servers/wicked_physics_server_3d.h
// Godot PhysicsServer3DExtension that replaces the default physics engine
// with the WickedEngine (high‑performance rigid body dynamics, collision
// shapes, joints, vehicles).  Integrates with Gaia's BVH for broad‑phase
// acceleration.  Provides the standard Godot physics server interface.

#ifndef WICKED_SERVERS_PHYSICS_SERVER_3D_H
#define WICKED_SERVERS_PHYSICS_SERVER_3D_H

#include "servers/physics_server_3d.h"
#include "servers/physics_server_3d_helpers.h"
#include "../world/wicked_world.h"
#include "../bodies/wicked_body.h"
#include "../collision/wicked_shape.h"
#include "../joints/wicked_joint.h"
#include "../materials/wicked_material.h"

namespace wicked {

class WickedPhysicsServer3D : public PhysicsServer3DExtension {
    GDCLASS(WickedPhysicsServer3D, PhysicsServer3DExtension);

    // Internal data structures
    struct SpaceData {
        RID self;
        WickedWorld *world;
        vec3 gravity;
        real_t step_size;
        bool active;
    };

    struct BodyInfo {
        RID self;
        body_id wick_id;
        BodyType type;
        bool active;
        Ref<WickedBody> body;
        Ref<WickedShape> shape;
        material_id material;
    };

    struct JointInfo {
        RID self;
        joint_id wick_id;
        JointType type;
        bool active;
        Ref<WickedJoint> joint;
    };

public:
    WickedPhysicsServer3D();
    virtual ~WickedPhysicsServer3D();

    // PhysicsServer3D overrides
    virtual bool is_flushing_queries() const override { return false; }
    virtual int get_process_info(ProcessInfo p_info) override;

    virtual RID space_create() override;
    virtual RID area_create() override { return RID(); }
    virtual RID body_create() override;
    virtual RID soft_body_create() override { return body_create(); }
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
    virtual bool is_active() const override { return active; }

    void set_gravity(const Vector3 &p_gravity);

private:
    RID active_space;
    bool active;
    HashMap<RID, SpaceData> space_map;
    HashMap<RID, BodyInfo> body_map;
    HashMap<RID, JointInfo> joint_map;
    HashMap<RID, Ref<WickedShape>> shape_map;
    HashMap<material_id, Ref<WickedMaterial>> material_map;

    body_id next_body_id;
    joint_id next_joint_id;
    material_id next_material_id;

    void _step_space(SpaceData &space, real_t dt);
};

} // namespace wicked

#endif // WICKED_SERVERS_PHYSICS_SERVER_3D_H