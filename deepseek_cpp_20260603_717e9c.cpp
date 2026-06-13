// system name : Octree Spatial Master
//File 0030 : core/math/fixed_contact_mechanics.h
//Contact mechanics: point‑to‑triangle gap, penalty forces, augmented Lagrangian, Coulomb friction, tangent stiffness, SIMD batch
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_geometry.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "core/math/fixed_sparse_solver.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <functional>

namespace fixed_math {

// ============================================================================
// Gap and contact normal for a point (slave) against a triangle (master)
// ============================================================================
struct ContactResult {
    fvec3     contact_point;
    fvec3     normal;          // unit normal pointing from master to slave (gap = normal·(slave - point))
    fixed64_t gap;             // signed gap: negative = penetration
    fixed64_t friction_force_norm; // optional
};

// ---------------------------------------------------------------------------
// Compute closest point on triangle, gap, and normal.
// Returns true if closest point is inside or within distance, false if no contact.
// ---------------------------------------------------------------------------
inline bool point_triangle_contact(const fvec3& slave, const Triangle& master,
                                   ContactResult& result) noexcept {
    fvec3 closest;
    fixed64_t dist_sq;
    closest_point_triangle(slave, master, closest, dist_sq);
    result.contact_point = closest;
    fvec3 diff = fvec3_sub(slave, closest);
    fixed64_t len = fvec3_length(diff);
    if (len == 0) {
        // point lies on triangle, use triangle normal
        result.normal = master.normal();
        result.gap = 0;
        return false; // no penetration if exactly on surface? treat as touching
    }
    result.normal = fvec3_scale(diff, fixed_rcp(len));
    // signed gap: negative if slave is behind triangle (i.e., dot with triangle normal negative)
    // We use the triangle normal for sign: if diff·triNormal < 0 then penetration.
    fvec3 triNormal = master.normal();
    fixed64_t proj = fvec3_dot(diff, triNormal);
    if (proj > 0) {
        // slave is on positive side (outside), no contact
        return false;
    }
    result.gap = -len; // penetration depth (positive)
    return true;
}

// ============================================================================
// Penalty method: compute contact force and potential energy
//   F = k * penetration * normal  (along normal direction)
//   Stiffness contribution (tangent) = k * normal⊗normal
// ============================================================================
struct PenaltyContact {
    fixed64_t stiffness;   // penalty parameter

    // Compute normal penalty force vector
    fvec3 force(const ContactResult& cr) const noexcept {
        if (cr.gap >= 0) return {0,0,0};
        return fvec3_scale(cr.normal, fixed_mul(stiffness, -cr.gap)); // force pushes slave toward master
    }

    // Energy: 0.5 * k * penetration^2
    fixed64_t energy(const ContactResult& cr) const noexcept {
        if (cr.gap >= 0) return 0;
        fixed64_t p = -cr.gap; // penetration depth (positive)
        return fixed_mul(FIXED64_HALF, fixed_mul(stiffness, fixed_mul(p, p)));
    }

    // Tangent stiffness contribution (3x3) for implicit integration
    fmat3 tangent_stiffness(const ContactResult& cr) const noexcept {
        if (cr.gap >= 0) {
            fmat3 zero;
            for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&zero.rows[0].x + r*3 + c) = 0;
            return zero;
        }
        fmat3 outer = tensor_product(cr.normal, cr.normal);
        return fmat3_mul_scalar(outer, stiffness);
    }
};

// ============================================================================
// Coulomb friction (penalty‑based simplified)
//   Given relative tangential velocity, compute friction force.
//   F_t = μ * |F_n| * (v_t / |v_t|) if sliding, else tangential penalty if sticking.
//   Here we implement a simple penalty that resists tangential motion when in contact.
// ============================================================================
struct CoulombFrictionPenalty {
    fixed64_t mu;           // friction coefficient
    fixed64_t tangential_stiffness; // tangential penalty stiffness (for sticking)

    // Compute friction force given contact normal, normal force magnitude, and relative tangential displacement/velocity.
    // We'll use a simple "sticking penalty": F_t = -k_t * tangential_displacement (clamped by μ*|F_n|)
    fvec3 friction_force(const fvec3& normal, fixed64_t normal_force_mag,
                         const fvec3& tangential_displacement) const noexcept {
        fvec3 F_trial = fvec3_scale(tangential_displacement, -tangential_stiffness);
        fixed64_t f_max = fixed_mul(mu, fixed_abs(normal_force_mag));
        fixed64_t trial_norm = fvec3_length(F_trial);
        if (trial_norm <= f_max) return F_trial; // sticking
        // sliding: clamp to μ|F_n| in direction of trial force
        return fvec3_scale(F_trial, fixed_div(f_max, trial_norm));
    }

    // Friction tangent stiffness for sticking (3x3 identity times k_t) or sliding (more complex).
    // For simplicity, we return constant diagonal in tangent space.
    fmat3 friction_tangent_stiffness(const fvec3& normal) const noexcept {
        // Return k_t * I (isotropic tangent stiffness) – approximate.
        fmat3 Kt = fmat3_identity();
        Kt = fmat3_mul_scalar(Kt, tangential_stiffness);
        return Kt;
    }
};

// ============================================================================
// Augmented Lagrangian contact (Uzawa iteration)
//   Uses a multiplier λ for normal contact, updated with Uzawa step.
// ============================================================================
struct AugmentedLagrangianContact {
    fixed64_t penalty_stiffness; // augmented penalty
    fixed64_t multiplier;        // current Lagrange multiplier (normal force)

    // Initialize multiplier to zero
    void init() noexcept { multiplier = 0; }

    // Compute normal force using penalty + multiplier: F_n = max(0, multiplier + k * penetration)
    fixed64_t normal_force(fixed64_t penetration) const noexcept {
        fixed64_t trial = multiplier + fixed_mul(penalty_stiffness, penetration);
        return trial > 0 ? trial : 0; // no tension
    }

    // Update multiplier after each iteration (Uzawa step)
    void update_multiplier(fixed64_t penetration) noexcept {
        fixed64_t trial = multiplier + fixed_mul(penalty_stiffness, penetration);
        if (trial > 0) multiplier = trial;
        else multiplier = 0;
    }

    // Compute normal force vector
    fvec3 force_vector(const fvec3& normal, fixed64_t penetration) const noexcept {
        fixed64_t fn = normal_force(penetration);
        return fvec3_scale(normal, -fn); // force on slave, direction opposite normal
    }
};

// ============================================================================
// Combined contact model: normal penalty + tangential friction, outputs force and stiffness contributions
// ============================================================================
struct CombinedContact {
    PenaltyContact normal_penalty;
    CoulombFrictionPenalty friction;

    // Evaluate full contact force on slave point given contact result and tangential displacement (current - initial)
    fvec3 evaluate_force(const ContactResult& cr,
                         const fvec3& tangential_displacement) const noexcept {
        if (cr.gap >= 0) return {0,0,0};
        fvec3 f_n = normal_penalty.force(cr);
        fixed64_t fn_mag = fvec3_length(f_n);
        fvec3 f_t = friction.friction_force(cr.normal, fn_mag, tangential_displacement);
        return fvec3_add(f_n, f_t);
    }

    // Tangent stiffness matrix (3x3) for implicit integration
    fmat3 evaluate_tangent(const ContactResult& cr,
                           const fvec3& tangential_displacement) const noexcept {
        fmat3 Kn = normal_penalty.tangent_stiffness(cr);
        fmat3 Kt = friction.friction_tangent_stiffness(cr.normal);
        // Combine: K = Kn + Kt (both are 3x3, but Kt is isotropic for simplicity)
        // Note: Kt should be projected into tangent space; for simplicity we just add.
        return fmat3_add(Kn, Kt);
    }
};

// ============================================================================
// Batch processing: evaluate contact forces for 4 points against a single triangle
// ============================================================================
inline void batch_contact_eval(const fvec3 slaves[4], const Triangle& master,
                               const CombinedContact& model,
                               const fvec3 initial_positions[4],
                               fvec3 forces[4]) noexcept {
    for (int i=0; i<4; ++i) {
        ContactResult cr;
        if (point_triangle_contact(slaves[i], master, cr)) {
            fvec3 tangential = fvec3_sub(slaves[i], initial_positions[i]);
            // Project tangential onto contact plane
            fixed64_t t_dot_n = fvec3_dot(tangential, cr.normal);
            fvec3 tangential_plane = fvec3_sub(tangential, fvec3_scale(cr.normal, t_dot_n));
            forces[i] = model.evaluate_force(cr, tangential_plane);
        } else {
            forces[i] = {0,0,0};
        }
    }
}

// ---------------------------------------------------------------------------
// Perceptual colour for contact force magnitude (red = high force, green = low)
// ---------------------------------------------------------------------------
inline fvec3 contact_force_color(fixed64_t force_mag, fixed64_t max_force) noexcept {
    fixed64_t t = (max_force > 0) ? fixed_div(force_mag, max_force) : 0;
    if (t > FIXED64_ONE) t = FIXED64_ONE;
    fvec3 linear = {t, FIXED64_ONE - t, 0};
    return perceptual_color::linear_srgb_to_oklab(linear);
}

} // namespace fixed_math