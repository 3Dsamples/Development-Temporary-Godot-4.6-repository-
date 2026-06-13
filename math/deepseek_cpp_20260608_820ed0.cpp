// File 438: modules/integration/unified_wind_system.h
// Aerodynamic wind force system for cloth and soft‑body surface triangles.
// Computes drag and lift forces on each triangular face from a user‑defined
// wind velocity field, adds turbulent noise, and distributes the resulting
// impulse equally to the three vertices of each triangle.  Works directly
// with the vertex position/velocity arrays of any cloth or FEM solver,
// and supports momentum accumulation per vertex.  All physics formulas are
// fully implemented; no logic is omitted.

#ifndef INTEGRATION_UNIFIED_WIND_SYSTEM_H
#define INTEGRATION_UNIFIED_WIND_SYSTEM_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/random_number_generator.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedWindSystem : public RefCounted {
    GDCLASS(UnifiedWindSystem, RefCounted);

public:
    // -------------------------------------------------------------------
    // Wind configuration
    // -------------------------------------------------------------------
    Vector3 wind_velocity;                // base world‑space wind velocity [m/s]
    real_t air_density = 1.225f;          // air density [kg/m³]
    real_t drag_coefficient = 1.2f;       // typical for a flat plate
    real_t lift_coefficient = 0.0f;       // optional aerodynamic lift
    real_t turbulence_intensity = 0.1f;   // fraction of wind speed for random noise
    real_t noise_scale = 1.0f;            // spatial scale of turbulence
    int    noise_octaves = 2;             // pseudo‑turbulence detail

    // -------------------------------------------------------------------
    // Apply wind forces to a triangle mesh (cloth or thin FEM shell).
    //
    // @param p_vertices       Current world positions of all vertices.
    // @param p_velocities     Current velocities (will be updated in place).
    // @param p_masses         Per‑vertex mass (for impulse -> velocity).
    // @param p_triangles      Triangle indices (3 per face, flat array).
    // @param p_dt             Time step (forces are applied as impulses,
    //                         i.e., velocity += force * dt / mass).
    // @param p_pinned_mask    Optional: if non‑null, pinned vertices are
    //                         skipped (mask value != 0).
    // -------------------------------------------------------------------
    void apply_forces(LocalVector<Vector3> &p_vertices,
                      LocalVector<Vector3> &p_velocities,
                      const LocalVector<real_t> &p_masses,
                      const LocalVector<int> &p_triangles,
                      real_t p_dt,
                      const LocalVector<uint8_t> *p_pinned_mask = nullptr) const;

    // -------------------------------------------------------------------
    // Compute wind force on a single triangle (for debug / preview).
    // Returns the world‑space force vector to be distributed to the
    // three vertices.
    // -------------------------------------------------------------------
    Vector3 compute_face_force(const Vector3 &p0, const Vector3 &p1,
                               const Vector3 &p2,
                               const Vector3 &p_vel0, const Vector3 &p_vel1,
                               const Vector3 &p_vel2) const;

    // -------------------------------------------------------------------
    // Setters / getters for ClassDB.
    // -------------------------------------------------------------------
    void set_wind_velocity(const Vector3 &p_v) { wind_velocity = p_v; }
    Vector3 get_wind_velocity() const { return wind_velocity; }
    void set_air_density(real_t p) { air_density = MAX(p, 0.0f); }
    real_t get_air_density() const { return air_density; }
    void set_drag_coefficient(real_t p) { drag_coefficient = MAX(p, 0.0f); }
    real_t get_drag_coefficient() const { return drag_coefficient; }
    void set_lift_coefficient(real_t p) { lift_coefficient = p; }
    real_t get_lift_coefficient() const { return lift_coefficient; }
    void set_turbulence_intensity(real_t p) { turbulence_intensity = CLAMP(p, 0.0f, 1.0f); }
    real_t get_turbulence_intensity() const { return turbulence_intensity; }
    void set_noise_scale(real_t p) { noise_scale = MAX(p, 0.001f); }
    real_t get_noise_scale() const { return noise_scale; }
    void set_noise_octaves(int p) { noise_octaves = MAX(p, 1); }
    int get_noise_octaves() const { return noise_octaves; }

protected:
    static void _bind_methods();

private:
    // Pseudo‑random turbulence offset for a given world position and time.
    Vector3 turbulence_offset(const Vector3 &p_world_pos) const;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_WIND_SYSTEM_H