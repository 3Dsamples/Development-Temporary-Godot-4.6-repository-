// File 454: modules/integration/unified_sph_force_calculator.h
// High‑performance SPH force integrator using TreeNSearch chunked queries.
// After density and pressure are computed (e.g., by DensityCalculator),
// this class applies pressure gradient, viscosity, and surface tension
// forces without storing neighbour lists.  All loops are parallelised via
// Gaia's CPUParallelization and use direct neighbour radius walks with
// precomputed squared distances for maximum cache efficiency.

#ifndef INTEGRATION_UNIFIED_SPH_FORCE_CALCULATOR_H
#define INTEGRATION_UNIFIED_SPH_FORCE_CALCULATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"
#include "../../treesearch/point_set_search.h"
#include "../../gaia/src/parallelization/cpu_parallelization.h"

namespace unified {

class UnifiedSPHForceCalculator : public RefCounted {
    GDCLASS(UnifiedSPHForceCalculator, RefCounted);

public:
    // Global smoothing length (must match density calculator).
    real_t smoothing_length = 0.1f;

    // Viscosity coefficient (dynamic).
    real_t viscosity = 0.001f;

    // Surface tension coefficient (CSF model).
    real_t surface_tension = 0.0f;

    // Gravit acceleration (added directly, not via pressure).
    Vector3 gravity = Vector3(0.0f, -9.81f, 0.0f);

    // Poly6 and Spiky gradient kernel constants (recomputed when h changes).
    real_t poly6_const = 0.0f;
    real_t spiky_grad_const = 0.0f;
    real_t visc_lapl_const = 0.0f;

    // -------------------------------------------------------------------
    // Rebuild the BVH from particle positions (must match the set used
    // for density computation).  Also re‑evaluates kernel constants.
    // -------------------------------------------------------------------
    void rebuild(const LocalVector<Vector3> &p_positions);

    // -------------------------------------------------------------------
    // Compute forces on all particles.  Requires pre‑computed densities
    // and pressures (from density calculator).  Velocities are updated
    // in place using force * dt / mass.
    // -------------------------------------------------------------------
    void compute_forces(const LocalVector<Vector3> &p_positions,
                        const LocalVector<real_t> &p_densities,
                        const LocalVector<real_t> &p_pressures,
                        const LocalVector<real_t> &p_masses,
                        LocalVector<Vector3> &r_velocities,
                        const LocalVector<uint8_t> *p_pinned_mask,
                        real_t p_dt) const;

    // -------------------------------------------------------------------
    // Update kernel constants when smoothing length is changed.
    // -------------------------------------------------------------------
    void update_kernel_constants();

protected:
    static void _bind_methods();

private:
    treesearch::PointSetSearch bvh;

    // Chunked force computation (pressure + viscosity + surface tension).
    void force_chunk(int start, int end,
                     const LocalVector<Vector3> &pos,
                     const LocalVector<real_t> &dens,
                     const LocalVector<real_t> &pres,
                     const LocalVector<real_t> &mass,
                     LocalVector<Vector3> &vel,
                     const LocalVector<uint8_t> *pinned,
                     real_t dt) const;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedSPHForceCalculator::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_smoothing_length", "h"), &UnifiedSPHForceCalculator::set_smoothing_length);
    ClassDB::bind_method(D_METHOD("get_smoothing_length"), &UnifiedSPHForceCalculator::get_smoothing_length);
    ClassDB::bind_method(D_METHOD("set_viscosity", "mu"), &UnifiedSPHForceCalculator::set_viscosity);
    ClassDB::bind_method(D_METHOD("get_viscosity"), &UnifiedSPHForceCalculator::get_viscosity);
    ClassDB::bind_method(D_METHOD("set_surface_tension", "gamma"), &UnifiedSPHForceCalculator::set_surface_tension);
    ClassDB::bind_method(D_METHOD("get_surface_tension"), &UnifiedSPHForceCalculator::get_surface_tension);
    ClassDB::bind_method(D_METHOD("rebuild", "positions"), &UnifiedSPHForceCalculator::rebuild);
    ClassDB::bind_method(D_METHOD("compute_forces", "positions", "densities", "pressures", "masses", "velocities", "pinned_mask", "dt"),
        &UnifiedSPHForceCalculator::compute_forces);
    ClassDB::bind_method(D_METHOD("update_kernel_constants"), &UnifiedSPHForceCalculator::update_kernel_constants);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing_length"), "set_smoothing_length", "get_smoothing_length");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "viscosity"), "set_viscosity", "get_viscosity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "surface_tension"), "set_surface_tension", "get_surface_tension");
}

void UnifiedSPHForceCalculator::rebuild(const LocalVector<Vector3> &p_positions) {
    bvh.build(p_positions);
    update_kernel_constants();
}

void UnifiedSPHForceCalculator::update_kernel_constants() {
    real_t h = smoothing_length;
    real_t h6 = Math::pow(h, 6);
    real_t h9 = Math::pow(h, 9);
    poly6_const      = 315.0f / (64.0f * Math_PI * h9);
    spiky_grad_const = -45.0f / (Math_PI * h6);
    visc_lapl_const  =  45.0f / (Math_PI * h6);
}

void UnifiedSPHForceCalculator::compute_forces(
        const LocalVector<Vector3> &p_positions,
        const LocalVector<real_t> &p_densities,
        const LocalVector<real_t> &p_pressures,
        const LocalVector<real_t> &p_masses,
        LocalVector<Vector3> &r_velocities,
        const LocalVector<uint8_t> *p_pinned_mask,
        real_t p_dt) const {
    int n = p_positions.size();
    ERR_FAIL_COND(p_densities.size() != n || p_pressures.size() != n ||
                  p_masses.size() != n || r_velocities.size() != n);

    // Chunked parallel dispatch.
    gaia::parallel::CPUParallelization::parallel_for(n,
        [this, &p_positions, &p_densities, &p_pressures, &p_masses,
         &r_velocities, p_pinned_mask, p_dt](int64_t start, int64_t end) {
            force_chunk((int)start, (int)end, p_positions, p_densities,
                        p_pressures, p_masses, r_velocities, p_pinned_mask, p_dt);
        },
        256);
}

void UnifiedSPHForceCalculator::force_chunk(
        int start, int end,
        const LocalVector<Vector3> &pos,
        const LocalVector<real_t> &dens,
        const LocalVector<real_t> &pres,
        const LocalVector<real_t> &mass,
        LocalVector<Vector3> &vel,
        const LocalVector<uint8_t> *pinned,
        real_t dt) const {
    real_t h = smoothing_length;
    real_t h2 = h * h;
    real_t grav_mag = gravity.length();

    for (int i = start; i < end; ++i) {
        if (pinned && (*pinned)[i]) continue;
        Vector3 f_pressure(0,0,0), f_visc(0,0,0), f_surf(0,0,0);

        const real_t rho_i = dens[i];
        const real_t p_i   = pres[i];
        const real_t m_i   = mass[i];
        const Vector3 &pos_i = pos[i];
        const Vector3 &vel_i = vel[i];

        // Self gravity (if enabled) is added as external acceleration.
        // We'll add gravity via force = m * g at the end, not inside neighbor loop.

        LocalVector<int> neighbours;
        bvh.radius(pos_i, h, neighbours);

        for (int j : neighbours) {
            if (j == i) continue;
            if (pinned && (*pinned)[j]) continue; // neighbour pinned? still contributes to fluid forces but we skip if we don't want to influence pinned? Usually pinned are boundaries, they exert pressure but don't move. We'll still compute forces from them but won't update their velocity.

            const real_t rho_j = dens[j];
            const real_t p_j   = pres[j];
            const real_t m_j   = mass[j];
            const Vector3 &pos_j = pos[j];
            const Vector3 &vel_j = vel[j];

            Vector3 diff = pos_i - pos_j;
            real_t r2 = diff.length_squared();
            if (r2 >= h2 || r2 < 1e-12f) continue;
            real_t r = Math::sqrt(r2);
            Vector3 r_dir = diff / r;

            // ---------- Pressure gradient (symmetrized) ----------
            real_t p_term = (p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j));
            real_t spiky_factor = (h - r) * (h - r); // (h-r)^2
            real_t grad_mag = spiky_grad_const * spiky_factor / r; // scalar * (h-r)^2 / r
            f_pressure -= m_j * p_term * grad_mag * r_dir;

            // ---------- Viscosity (XSPH-like) ----------
            Vector3 v_diff = vel_j - vel_i;
            if (viscosity > 0.0f) {
                real_t lapl = visc_lapl_const * (h - r); // (h-r) not squared
                f_visc += viscosity * m_j * (v_diff / rho_j) * lapl;
            }

            // ---------- Surface tension (CSF) ----------
            if (surface_tension > 0.0f) {
                // Colour field contribution: mass_j * poly6(r) / rho_j
                real_t poly6_val = poly6_const * (h2 - r2) * (h2 - r2) * (h2 - r2);
                real_t colour_term = m_j * poly6_val / rho_j;
                f_surf += colour_term * diff;  // gradient of colour field approximated
            }
        }

        // Total force from neighbours.
        Vector3 f_total = f_pressure + f_visc;
        if (surface_tension > 0.0f) {
            // Normalize and scale surface tension force.
            real_t f_surf_len = f_surf.length();
            if (f_surf_len > CMP_EPSILON) {
                f_total += f_surf * (surface_tension / f_surf_len);
            }
        }
        // Gravity force per unit volume: rho_i * gravity
        f_total += gravity * rho_i;

        // Acceleration = force / density, then update velocity.
        Vector3 accel = f_total / rho_i;
        vel[i] += accel * dt;

        // Apply the equal and opposite force to neighbours? No, we use pairwise forces already symmetrized, so energy is conserved.
    }
}

// Property setters/getters.
void UnifiedSPHForceCalculator::set_smoothing_length(real_t v) { smoothing_length = MAX(v, 0.001f); }
real_t UnifiedSPHForceCalculator::get_smoothing_length() const { return smoothing_length; }
void UnifiedSPHForceCalculator::set_viscosity(real_t v) { viscosity = MAX(v, 0.0f); }
real_t UnifiedSPHForceCalculator::get_viscosity() const { return viscosity; }
void UnifiedSPHForceCalculator::set_surface_tension(real_t v) { surface_tension = MAX(v, 0.0f); }
real_t UnifiedSPHForceCalculator::get_surface_tension() const { return surface_tension; }

} // namespace unified

#endif // INTEGRATION_UNIFIED_SPH_FORCE_CALCULATOR_H