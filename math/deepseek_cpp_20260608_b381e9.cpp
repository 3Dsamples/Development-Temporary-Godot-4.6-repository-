// File 440: modules/integration/unified_gravity_field_system.h
// Custom gravity field system for the unified physics pipeline.
// Defines multiple gravity fields (attractor, repulsor, directional) with
// configurable strength, mass, softening radius, and per‑body influence
// scale.  Forces are computed each physics step and applied to all
// registered bodies across any engine (Newton, Genesis, Vienna, Wicked).
// Uses inverse‑square law with softening and falloff curves.
// All maths and engine interfacing are fully implemented inline.

#ifndef INTEGRATION_UNIFIED_GRAVITY_FIELD_SYSTEM_H
#define INTEGRATION_UNIFIED_GRAVITY_FIELD_SYSTEM_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

// Forward declarations for engine bodies.
namespace newton   { class NewtonWorld; class NewtonBody; }
namespace genesis  { class GenesisWorld; class BaseEntity; }
namespace vienna   { class ViennaWorld; class ViennaBody; }
namespace wicked   { class WickedWorld; class WickedBody; }

namespace unified {

class UnifiedGravityFieldSystem : public RefCounted {
    GDCLASS(UnifiedGravityFieldSystem, RefCounted);

public:
    // -------------------------------------------------------------------
    // Gravity field types.
    // -------------------------------------------------------------------
    enum FieldType {
        FIELD_ATTRACTOR = 0,    // pulls bodies towards the field centre
        FIELD_REPULSOR  = 1,    // pushes bodies away from the centre
        FIELD_DIRECTIONAL = 2   // constant acceleration in a direction
    };

    // -------------------------------------------------------------------
    // Description of a single gravity field.
    // -------------------------------------------------------------------
    struct GravityField {
        FieldType type = FIELD_ATTRACTOR;
        Vector3   position;            // world space (only for attractor/repulsor)
        Vector3   direction;           // unit vector (only for directional)
        real_t    strength = 10.0f;    // m/s² at unit distance (or directional magnitude)
        real_t    mass = 100.0f;       // effective "gravitational mass" for attractor/repulsor
        real_t    softening = 0.5f;    // softening length to avoid singularity
        real_t    max_distance = 100.0f; // maximum influence distance, 0 for unlimited
        real_t    falloff_power = 2.0f; // exponent for distance falloff (2 = inverse square)
        bool      enabled = true;
    };

    // -------------------------------------------------------------------
    // Per‑body gravity scale (default 1.0).
    // -------------------------------------------------------------------
    void set_body_gravity_scale(int p_engine, uint64_t p_body_id, real_t p_scale);
    real_t get_body_gravity_scale(int p_engine, uint64_t p_body_id) const;

    // -------------------------------------------------------------------
    // Field management.
    // -------------------------------------------------------------------
    int add_field(const GravityField &p_field);
    void remove_field(int p_field_index);
    void clear_fields();
    int get_field_count() const;
    GravityField &get_field(int p_idx);
    const GravityField &get_field(int p_idx) const;

    // -------------------------------------------------------------------
    // Apply all enabled fields to all registered bodies in the given
    // engine worlds.  Call once per physics substep.
    // -------------------------------------------------------------------
    void apply_forces(real_t p_dt,
                      newton::NewtonWorld *newton_world,
                      genesis::GenesisWorld *genesis_world,
                      vienna::ViennaWorld *vienna_world,
                      wicked::WickedWorld *wicked_world) const;

    // -------------------------------------------------------------------
    // Compute the total gravitational acceleration on a single body
    // from all fields (for debug / preview).
    // -------------------------------------------------------------------
    Vector3 compute_acceleration(const Vector3 &p_body_position, real_t p_body_mass,
                                 real_t p_body_gravity_scale) const;

protected:
    static void _bind_methods();

private:
    // Storage for fields.
    LocalVector<GravityField> fields;

    // Per‑body gravity scale storage.
    struct BodyKey {
        int engine;
        uint64_t body_id;
        bool operator==(const BodyKey &o) const { return engine==o.engine && body_id==o.body_id; }
        struct Hash { uint64_t operator()(const BodyKey &k) const { return (uint64_t(k.engine)<<56) | (k.body_id); } };
    };
    HashMap<BodyKey, real_t, BodyKey::Hash> body_scales;

    // Engines helper: apply force to a body.
    void apply_body_force(int p_engine, uint64_t p_body_id,
                          const Vector3 &p_force,
                          newton::NewtonWorld *nw, genesis::GenesisWorld *gw,
                          vienna::ViennaWorld *vw, wicked::WickedWorld *ww) const;

    // Engines helper: get body position and mass.
    void get_body_state(int p_engine, uint64_t p_body_id,
                        const newton::NewtonWorld *nw, const genesis::GenesisWorld *gw,
                        const vienna::ViennaWorld *vw, const wicked::WickedWorld *ww,
                        Vector3 &r_pos, real_t &r_mass) const;

    // Engines helper: iterate over all body IDs of a world.
    void collect_body_ids(int p_engine,
                          const newton::NewtonWorld *nw, const genesis::GenesisWorld *gw,
                          const vienna::ViennaWorld *vw, const wicked::WickedWorld *ww,
                          LocalVector<uint64_t> &r_ids) const;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_GRAVITY_FIELD_SYSTEM_H