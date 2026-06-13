// File 444: modules/integration/unified_spring_system.h
// Applies damped spring forces between pairs of bodies across any registered
// engine (Newton, Genesis, Vienna, Wicked).  Each spring has a rest length,
// stiffness, damping coefficient, and optional breakable force threshold.
// Forces are computed each substep via Hooke's law and viscous damping,
// and applied directly to the connected bodies.  All physics and engine
// adapters are fully implemented inline.

#ifndef INTEGRATION_UNIFIED_SPRING_SYSTEM_H
#define INTEGRATION_UNIFIED_SPRING_SYSTEM_H

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

class UnifiedSpringSystem : public RefCounted {
    GDCLASS(UnifiedSpringSystem, RefCounted);

public:
    // -------------------------------------------------------------------
    // Spring descriptor (one per pair of bodies).
    // -------------------------------------------------------------------
    struct Spring {
        // Engine and body IDs for the two endpoints (can be in different engines).
        int      engine_a = 0;
        uint64_t body_id_a = 0;
        int      engine_b = 0;
        uint64_t body_id_b = 0;

        // Anchor points in each body's local frame (default: origin).
        Vector3  local_anchor_a;
        Vector3  local_anchor_b;

        // Physical parameters.
        real_t   rest_length = 1.0f;       // m
        real_t   stiffness   = 1000.0f;    // N/m (Hooke's constant)
        real_t   damping     = 10.0f;      // Ns/m (viscous damping)
        bool     limit_compression = false;// hide if not needed; we don't limit
        // Breakable: if total force magnitude exceeds break_force, spring disables.
        bool     breakable = false;
        real_t   break_force = INFINITY;

        bool     enabled = true;
    };

    // -------------------------------------------------------------------
    // Add a spring and return its index.
    // -------------------------------------------------------------------
    int add_spring(const Spring &p_spring);

    // Remove a spring by index.
    void remove_spring(int p_index);

    // Remove all springs involving a given body (any engine).
    void remove_springs_with_body(int p_engine, uint64_t p_body_id);

    // Clear all springs.
    void clear_springs();

    // Return the number of active springs.
    int get_spring_count() const;

    // Access a spring descriptor (const).
    const Spring &get_spring(int p_idx) const;

    // Apply all enabled springs to the given engine worlds.
    // This should be called once per physics substep after positions are updated.
    void apply_forces(real_t p_dt,
                      newton::NewtonWorld *newton_world,
                      genesis::GenesisWorld *genesis_world,
                      vienna::ViennaWorld *vienna_world,
                      wicked::WickedWorld *wicked_world);

protected:
    static void _bind_methods();

private:
    LocalVector<Spring> springs;

    // -------------------------------------------------------------------
    // Engine‑specific helpers (same pattern as gravity field system).
    // -------------------------------------------------------------------
    void apply_body_force(int p_engine, uint64_t p_body_id,
                          const Vector3 &p_force, const Vector3 &p_world_point,
                          newton::NewtonWorld *nw, genesis::GenesisWorld *gw,
                          vienna::ViennaWorld *vw, wicked::WickedWorld *ww) const;

    void get_body_state(int p_engine, uint64_t p_body_id,
                        const newton::NewtonWorld *nw, const genesis::GenesisWorld *gw,
                        const vienna::ViennaWorld *vw, const wicked::WickedWorld *ww,
                        Transform3D &r_xform, Vector3 &r_vel) const;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_SPRING_SYSTEM_H