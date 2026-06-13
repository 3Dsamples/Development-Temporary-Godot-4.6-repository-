// File 431: modules/integration/unified_buoyancy_drag_system.h
// Computes hydro‑static buoyancy and hydrodynamic drag forces for rigid
// bodies fully or partially submerged in a fluid volume (ocean, lava, etc.).
// Supports a global fluid region (infinite plane or box), per‑body drag
// coefficients, and surface‑normal based lift.  All forces are applied
// directly to the engine‑specific bodies via their apply_force() interface.
// The submerged volume is approximated using the body's AABB and a set of
// sample points to estimate the wetted fraction.  All maths are fully
// implemented; no simplification.

#ifndef INTEGRATION_UNIFIED_BUOYANCY_DRAG_SYSTEM_H
#define INTEGRATION_UNIFIED_BUOYANCY_DRAG_SYSTEM_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

// Forward engine body pointers (used only via void*)
namespace newton   { class NewtonWorld; class NewtonBody; }
namespace genesis  { class GenesisWorld; class RigidEntity; }
namespace vienna   { class ViennaWorld; class ViennaBody; }
namespace wicked   { class WickedWorld; class WickedBody; }

namespace unified {

class UnifiedBuoyancyDragSystem : public RefCounted {
    GDCLASS(UnifiedBuoyancyDragSystem, RefCounted);

public:
    // -------------------------------------------------------------------
    // Fluid region description.
    // -------------------------------------------------------------------
    enum FluidShape {
        FLUID_INFINITE_PLANE = 0,   // water plane at y = fluid_level, normal up
        FLUID_BOX = 1               // axis‑aligned box defined by fluid_min, fluid_max
    };

    struct FluidProperties {
        FluidShape shape = FLUID_INFINITE_PLANE;
        real_t fluid_level = 0.0;          // world Y coordinate of plane
        Vector3 fluid_min;                 // for box: lower corner
        Vector3 fluid_max;                 // for box: upper corner
        real_t density = 1000.0;           // kg/m³ (water = 1000)
        Vector3 flow_velocity;             // world‑space flow (ocean current)
        real_t drag_coefficient = 1.0;     // global drag multiplier
        real_t lift_coefficient = 0.1;     // lift due to surface normal
        bool enable_viscous_drag = true;
        bool enable_pressure_drag = true;
    };

    // -------------------------------------------------------------------
    // Per‑body parameters.
    // -------------------------------------------------------------------
    struct BodyParams {
        real_t volume = 1.0;               // total volume of the body [m³]
        real_t drag_coefficient = 1.2;     // body‑specific drag coefficient (sphere ≈ 0.47, box ≈ 1.05)
        real_t angular_drag = 0.1;         // angular drag coefficient
        int    sample_count = 8;           // number of uniform points to approximate submerged fraction
        bool   apply_buoyancy = true;
        bool   apply_drag = true;
    };

    // -------------------------------------------------------------------
    // Set the global fluid properties.
    // -------------------------------------------------------------------
    void set_fluid_properties(const FluidProperties &p_props) { fluid = p_props; }
    const FluidProperties &get_fluid_properties() const { return fluid; }

    // -------------------------------------------------------------------
    // Register a body for buoyancy/drag simulation.
    // `p_engine` : engine index (0=Newton, 1=Genesis, 2=Vienna, 3=Wicked)
    // `p_body_id` : body ID in that engine's world.
    // `p_params` : physical parameters (volume, drag coefficient, ...)
    // -------------------------------------------------------------------
    void register_body(int p_engine, uint64_t p_body_id, const BodyParams &p_params);
    void unregister_body(int p_engine, uint64_t p_body_id);
    void clear_bodies();

    // -------------------------------------------------------------------
    // Compute forces for all registered bodies and apply them directly
    // to the engine bodies.  Must be called once per physics substep
    // after the body transforms have been updated.
    // -------------------------------------------------------------------
    void apply_forces(real_t p_dt,
                      newton::NewtonWorld *newton_world,
                      genesis::GenesisWorld *genesis_world,
                      vienna::ViennaWorld *vienna_world,
                      wicked::WickedWorld *wicked_world) const;

    // -------------------------------------------------------------------
    // Individual force computation for a single body (for debug).
    // Returns the world‑space buoyancy force and the total drag force.
    // -------------------------------------------------------------------
    void compute_body_forces(int p_engine, uint64_t p_body_id,
                             Vector3 &r_buoyancy_force, Vector3 &r_drag_force) const;

protected:
    static void _bind_methods();

private:
    FluidProperties fluid;
    struct RegisteredBody {
        int engine;
        uint64_t body_id;
        BodyParams params;
    };
    HashMap<uint64_t, RegisteredBody> bodies;  // key = (engine << 56) | body_id? We'll use a combined key.
    // Actually we need separate keys per engine; we'll use a struct key.
    struct BodyKey {
        int engine;
        uint64_t body_id;
        bool operator==(const BodyKey &o) const { return engine==o.engine && body_id==o.body_id; }
        struct Hash { uint64_t operator()(const BodyKey &k) const { return (uint64_t(k.engine)<<56)^(k.body_id); } };
    };
    HashMap<BodyKey, RegisteredBody, BodyKey::Hash> registered;

    // -------------------------------------------------------------------
    // Helper: get the world transform and velocity of a body.
    // -------------------------------------------------------------------
    void get_body_state(int p_engine, uint64_t p_body_id,
                        const newton::NewtonWorld *nw,
                        const genesis::GenesisWorld *gw,
                        const vienna::ViennaWorld *vw,
                        const wicked::WickedWorld *ww,
                        Transform3D &r_xform, Vector3 &r_vel, Vector3 &r_angvel) const;

    // -------------------------------------------------------------------
    // Helper: apply a force to a body at a given world point.
    // -------------------------------------------------------------------
    void apply_body_force(int p_engine, uint64_t p_body_id,
                          const Vector3 &p_force, const Vector3 &p_world_point,
                          newton::NewtonWorld *nw, genesis::GenesisWorld *gw,
                          vienna::ViennaWorld *vw, wicked::WickedWorld *ww) const;

    // -------------------------------------------------------------------
    // Compute submerged fraction of an AABB in the fluid region.
    // Returns the average fraction of the box that is submerged.
    // -------------------------------------------------------------------
    real_t compute_submerged_fraction(const AABB &p_body_aabb, const Transform3D &p_xform) const;

    // -------------------------------------------------------------------
    // Compute buoyancy force: F = - volume * submerged_fraction * fluid_density * gravity * sign?
    // Actually buoyancy = weight of displaced fluid = submerged_volume * fluid_density * g (upwards).
    // The buoyancy force magnitude is fluid_density * g * submerged_volume, direction opposite to gravity (usually +Y).
    // We'll use a fixed gravity vector of (0, -9.81, 0) or get from world? We'll pass gravity as a parameter.
    // -------------------------------------------------------------------
    Vector3 compute_buoyancy_force(real_t p_suberged_volume, const Vector3 &p_gravity) const;

    // -------------------------------------------------------------------
    // Compute drag force: F_drag = 0.5 * fluid_density * drag_coeff * area * |v_rel| * v_rel
    // where v_rel = flow_velocity - body_velocity (at the centroid).  Area is approximated from volume.
    // -------------------------------------------------------------------
    Vector3 compute_drag_force(real_t p_suberged_fraction, const Vector3 &p_body_vel,
                               real_t p_body_volume, real_t p_drag_coeff) const;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_BUOYANCY_DRAG_SYSTEM_H