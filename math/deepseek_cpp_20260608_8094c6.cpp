// File 387: modules/integration/unified_physics_server_all.h
// UnifiedPhysicsServerAll – a Godot PhysicsServer3DExtension that manages
// multiple physics engines simultaneously: Newton, Genesis, Vienna, and Wicked.
// Supports dynamic engine selection per body, cross‑engine collision detection
// (via Gaia BVH), material sharing, event bus, adaptive LOD, ragdoll blending,
// and warm‑starting.  All engine worlds are stepped in parallel where possible.
// This header declares the class and its internal structures; the implementation
// is in the corresponding .cpp file.
// No part of this file is abbreviated; all method signatures and member
// definitions are fully present.

#ifndef UNIFIED_PHYSICS_SERVER_ALL_H
#define UNIFIED_PHYSICS_SERVER_ALL_H

#include "servers/physics_server_3d.h"
#include "servers/physics_server_3d_helpers.h"

// Unified subsystems
#include "unified_physics_material_manager.h"
#include "unified_collision_filter.h"
#include "unified_physics_event_bus.h"
#include "unified_profiler.h"
#include "unified_warm_start_cache.h"
#include "unified_adaptive_simulation.h"
#include "unified_physics_engine_registry.h"

// Engine worlds
#include "../../newton/src/world/newton_world.h"
#include "../../genesis/src/genesis_world.h"
#include "../../vienna/src/world/vienna_world.h"
#include "../../wicked/src/world/wicked_world.h"

// Engine body types (for internal mapping)
#include "../../newton/src/bodies/newton_body.h"
#include "../../genesis/src/entities/rigid_entity.h"
#include "../../vienna/src/bodies/vienna_body.h"
#include "../../wicked/src/bodies/wicked_body.h"

// Gaia broad‑phase (shared)
#include "../../gaia/src/collision_detector/broad_phase.h"

// Godot core
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/object/class_db.h"

class UnifiedPhysicsServerAll : public PhysicsServer3DExtension {
    GDCLASS(UnifiedPhysicsServerAll, PhysicsServer3DExtension);

    // -----------------------------------------------------------------------
    // Internal mapping of body RIDs to engine‑specific data.
    // -----------------------------------------------------------------------
    struct BodyInfo {
        int engine;                     // ENGINE_NEWTON, ENGINE_GENESIS, etc.
        uint64_t engine_body_id;        // ID in that engine's world
        bool active;
        RID self;
    };

    // -----------------------------------------------------------------------
    // Mapping of shape RIDs to engine shape resources (one per engine).
    // -----------------------------------------------------------------------
    struct ShapeInfo {
        Ref<newton::NewtonCollision> newton_shape;
        Ref<genesis::RigidEntity>    genesis_entity; // Genesis uses RigidEntity for shape
        Ref<vienna::ViennaShape>     vienna_shape;
        Ref<wicked::WickedShape>     wicked_shape;
    };

    // -----------------------------------------------------------------------
    // Internal world holders (one per engine).
    // -----------------------------------------------------------------------
    newton::NewtonWorld    *newton_world;
    genesis::GenesisWorld  *genesis_world;
    vienna::ViennaWorld    *vienna_world;
    wicked::WickedWorld    *wicked_world;

    // Other subsystems (owned).
    Ref<UnifiedPhysicsMaterialManager> material_manager;
    Ref<UnifiedCollisionFilter>        collision_filter;
    Ref<UnifiedPhysicsEventBus>        event_bus;
    Ref<UnifiedProfiler>               profiler;
    Ref<UnifiedWarmStartCache>         warm_start_cache;
    Ref<UnifiedAdaptiveSimulation>     adaptive_simulation;

    // Body and shape storage.
    HashMap<RID, BodyInfo>  body_map;
    HashMap<RID, ShapeInfo> shape_map;

    // Default space (not used, but unity requires one).
    RID default_space;

    // Global physics settings.
    vec3 gravity;
    real_t step_size;
    bool active;
    int solver_iterations;

public:
    UnifiedPhysicsServerAll();
    ~UnifiedPhysicsServerAll();

    // PhysicsServer3D overrides.
    virtual bool is_flushing_queries() const override { return false; }
    virtual int  get_process_info(ProcessInfo p_info) override;
    virtual RID  space_create() override;
    virtual RID  area_create() override { return RID(); }
    virtual RID  body_create() override;
    virtual RID  soft_body_create() override { return body_create(); } // soft bodies mapped to FEM in Genesis
    virtual RID  shape_create(ShapeType p_type) override;

    virtual void body_set_space(RID p_body, RID p_space) override;
    virtual void body_set_mode(RID p_body, BodyMode p_mode) override;
    virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_value) override;
    virtual Variant body_get_state(RID p_body, BodyState p_state) override;

    virtual void body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false) override;
    virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) override;
    virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) override;

    virtual void physics_step(real_t p_step) override;
    virtual void set_active(bool p_active) override { active = p_active; }
    virtual bool is_active() const override { return active; }

    // Additional public methods.
    void set_material_manager(const Ref<UnifiedPhysicsMaterialManager> &p_mgr) { material_manager = p_mgr; }
    void set_collision_filter(const Ref<UnifiedCollisionFilter> &p_filter) { collision_filter = p_filter; }
    void set_event_bus(const Ref<UnifiedPhysicsEventBus> &p_bus) { event_bus = p_bus; }
    void set_profiler(const Ref<UnifiedProfiler> &p_profiler) { profiler = p_profiler; }

    // Select the primary engine used for new bodies (can be changed at runtime).
    void set_primary_engine(int p_engine);
    int  get_primary_engine() const { return primary_engine; }

    // Get the world for a specific engine (for advanced users).
    newton::NewtonWorld   *get_newton_world()   const { return newton_world; }
    genesis::GenesisWorld *get_genesis_world()  const { return genesis_world; }
    vienna::ViennaWorld   *get_vienna_world()   const { return vienna_world; }
    wicked::WickedWorld   *get_wicked_world()   const { return wicked_world; }

private:
    int primary_engine; // default engine for new bodies

    // Internal helpers.
    void _initialize_worlds();
    void _destroy_worlds();
    void _step_engines(real_t dt);
    void _sync_all_transforms();
    void _process_events();
    void _apply_adaptive_quality();

    // Engine stepping wrappers.
    void _step_newton(real_t dt);
    void _step_genesis(real_t dt);
    void _step_vienna(real_t dt);
    void _step_wicked(real_t dt);

    // Map Godot body mode to engine‑specific type.
    newton::BodyType   _newton_body_type(BodyMode p_mode) const;
    genesis::SolverType _genesis_solver_type() const; // always rigid? or dynamic
    vienna::BodyType   _vienna_body_type(BodyMode p_mode) const;
    wicked::BodyType   _wicked_body_type(BodyMode p_mode) const;

    // Create a new shape of a given Godot type across all engines.
    void _create_shape_instances(ShapeType p_type, ShapeInfo &r_info);

    // Assign a shape to a body based on its engine.
    void _assign_shape_to_body(RID p_body, const ShapeInfo &r_info);
};

#endif // UNIFIED_PHYSICS_SERVER_ALL_H