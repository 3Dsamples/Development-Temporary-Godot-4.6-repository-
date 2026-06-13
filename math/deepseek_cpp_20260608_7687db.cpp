// File 442: modules/integration/unified_magnetic_field_system.h
// Unified magnetic field system: defines external magnetic fields (uniform,
// dipole, or point‑source) and applies Lorentz‑like dipole forces and torques
// to rigid bodies carrying a magnetic moment vector.  The force on a dipole
// in a non‑uniform field is F = ∇(m·B), and the torque is τ = m × B.
// All field types and body interactions are fully implemented inline.

#ifndef INTEGRATION_UNIFIED_MAGNETIC_FIELD_SYSTEM_H
#define INTEGRATION_UNIFIED_MAGNETIC_FIELD_SYSTEM_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

// Forward engine body pointers.
namespace newton   { class NewtonWorld; class NewtonBody; }
namespace genesis  { class GenesisWorld; class BaseEntity; }
namespace vienna   { class ViennaWorld; class ViennaBody; }
namespace wicked   { class WickedWorld; class WickedBody; }

namespace unified {

class UnifiedMagneticFieldSystem : public RefCounted {
    GDCLASS(UnifiedMagneticFieldSystem, RefCounted);

public:
    // -------------------------------------------------------------------
    // Field source types
    // -------------------------------------------------------------------
    enum FieldSourceType {
        SOURCE_UNIFORM = 0,     // constant B everywhere
        SOURCE_DIPOLE  = 1,     // field of a magnetic dipole at a point
        SOURCE_CURRENT_LOOP = 2 // approximate circular loop field (simplified on‑axis)
    };

    // -------------------------------------------------------------------
    // Description of a single field source.
    // -------------------------------------------------------------------
    struct FieldSource {
        FieldSourceType type = SOURCE_UNIFORM;
        Vector3 position;          // dipole/loop centre (world)
        Vector3 direction;         // dipole moment direction (unit) or B direction for uniform
        real_t   strength = 1.0f;  // magnitude of B at 1 m (or uniform strength in Tesla)
        bool     enabled = true;
    };

    // -------------------------------------------------------------------
    // Per‑body magnetic moment (A·m²).  Default (0,0,0) = no magnetic interaction.
    // -------------------------------------------------------------------
    void set_body_magnetic_moment(int p_engine, uint64_t p_body_id, const Vector3 &p_moment);
    Vector3 get_body_magnetic_moment(int p_engine, uint64_t p_body_id) const;

    // -------------------------------------------------------------------
    // Field source management.
    // -------------------------------------------------------------------
    int add_field_source(const FieldSource &p_source);
    void remove_field_source(int p_index);
    void clear_field_sources();
    int get_field_source_count() const;
    FieldSource &get_field_source(int p_idx);
    const FieldSource &get_field_source(int p_idx) const;

    // -------------------------------------------------------------------
    // Compute the magnetic field B at a given world position [Tesla].
    // -------------------------------------------------------------------
    Vector3 compute_field(const Vector3 &p_world_pos) const;

    // -------------------------------------------------------------------
    // Apply magnetic forces and torques to all registered bodies with
    // non‑zero magnetic moments.  Called each physics substep.
    // -------------------------------------------------------------------
    void apply_forces_and_torques(real_t p_dt,
                                  newton::NewtonWorld *newton_world,
                                  genesis::GenesisWorld *genesis_world,
                                  vienna::ViennaWorld *vienna_world,
                                  wicked::WickedWorld *wicked_world) const;

    // -------------------------------------------------------------------
    // Compute the magnetic dipole force and torque on a single body
    // given its position, moment, and the local field gradient.
    // Returns force and torque vectors.
    // -------------------------------------------------------------------
    void compute_dipole_forces(const Vector3 &p_position,
                               const Vector3 &p_moment,
                               Vector3 &r_force, Vector3 &r_torque) const;

protected:
    static void _bind_methods();

private:
    LocalVector<FieldSource> sources;

    struct BodyKey {
        int engine;
        uint64_t body_id;
        bool operator==(const BodyKey &o) const { return engine==o.engine && body_id==o.body_id; }
        struct Hash { uint64_t operator()(const BodyKey &k) const { return (uint64_t(k.engine)<<56) | k.body_id; } };
    };
    HashMap<BodyKey, Vector3, BodyKey::Hash> body_moments;

    // Engine helpers
    void apply_body_force_and_torque(int p_engine, uint64_t p_body_id,
                                     const Vector3 &p_force, const Vector3 &p_torque,
                                     newton::NewtonWorld *nw, genesis::GenesisWorld *gw,
                                     vienna::ViennaWorld *vw, wicked::WickedWorld *ww) const;

    void get_body_state(int p_engine, uint64_t p_body_id,
                        const newton::NewtonWorld *nw, const genesis::GenesisWorld *gw,
                        const vienna::ViennaWorld *vw, const wicked::WickedWorld *ww,
                        Vector3 &r_pos, Vector3 &r_moment) const;

    void collect_body_ids(int p_engine,
                          const newton::NewtonWorld *nw, const genesis::GenesisWorld *gw,
                          const vienna::ViennaWorld *vw, const wicked::WickedWorld *ww,
                          LocalVector<uint64_t> &r_ids) const;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_MAGNETIC_FIELD_SYSTEM_H