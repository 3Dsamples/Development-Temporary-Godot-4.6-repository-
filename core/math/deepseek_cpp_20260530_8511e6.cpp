// File 0040 : core/math/contact_solver.h
// Sequential impulse solver with friction, warm starting, and Baumgarte stabilisation for rigid body contacts.

#pragma once

#include "vec3.h"
#include "mat3.h"
#include "quat.h"
#include "constants.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace wp {

// ── Solver body state ────────────────────────────────────────────────
template <typename T>
struct SolverBody {
    vec3<T>      position;
    quat<T>      orientation;
    vec3<T>      linear_velocity;
    vec3<T>      angular_velocity;
    T            inv_mass;
    mat3<T>      inv_inertia_world;

    constexpr SolverBody() noexcept
        : position(T(0)), orientation(), linear_velocity(T(0)), angular_velocity(T(0)),
          inv_mass(T(0)), inv_inertia_world(T(1)) {}
};

// ── Contact point with impulse accumulators ──────────────────────────
template <typename T>
struct ContactPoint {
    vec3<T>   point_a;                // world space point on body A
    vec3<T>   point_b;                // world space point on body B
    vec3<T>   normal;                 // from A to B
    T         penetration;            // positive = overlapping
    T         combined_restitution;   // typically min(A.restitution, B.restitution)
    T         combined_friction;      // sqrt(muA * muB) or similar
    int32     body_a = -1;
    int32     body_b = -1;

    // Warm‑start impulses
    T         normal_impulse = T(0);
    T         tangent_impulse1 = T(0);
    T         tangent_impulse2 = T(0);

    // Cached Jacobian diagonal (effective mass)
    T         effective_mass_normal = T(0);
    T         effective_mass_tangent1 = T(0);
    T         effective_mass_tangent2 = T(0);
};

// ── Contact constraint solver (sequential impulse) ──────────────────
template <typename T>
class ContactSolver {
public:
    // Set bodies and contacts; computes effective masses for each contact.
    void prepare(std::vector<SolverBody<T>>& bodies,
                 std::vector<ContactPoint<T>>& contacts) {
        m_bodies = &bodies;
        m_contacts = &contacts;

        for (auto& cp : *m_contacts) {
            const auto& bodyA = (*m_bodies)[cp.body_a];
            const auto& bodyB = (*m_bodies)[cp.body_b];

            vec3<T> rA = cp.point_a - bodyA.position;
            vec3<T> rB = cp.point_b - bodyB.position;
            vec3<T> n  = cp.normal;

            // Normal effective mass: 1 / (J * M⁻¹ * Jᵀ)
            T K = bodyA.inv_mass + bodyB.inv_mass;
            K += dot(cross(rA, n), mul(bodyA.inv_inertia_world, cross(rA, n)));
            K += dot(cross(rB, n), mul(bodyB.inv_inertia_world, cross(rB, n)));
            cp.effective_mass_normal = (K > epsilon<T>) ? T(1) / K : T(0);

            // Tangent directions (two vectors orthogonal to normal)
            vec3<T> t1, t2;
            compute_tangents(n, t1, t2);

            // Tangent 1 effective mass
            K = bodyA.inv_mass + bodyB.inv_mass;
            K += dot(cross(rA, t1), mul(bodyA.inv_inertia_world, cross(rA, t1)));
            K += dot(cross(rB, t1), mul(bodyB.inv_inertia_world, cross(rB, t1)));
            cp.effective_mass_tangent1 = (K > epsilon<T>) ? T(1) / K : T(0);

            // Tangent 2 effective mass
            K = bodyA.inv_mass + bodyB.inv_mass;
            K += dot(cross(rA, t2), mul(bodyA.inv_inertia_world, cross(rA, t2)));
            K += dot(cross(rB, t2), mul(bodyB.inv_inertia_world, cross(rB, t2)));
            cp.effective_mass_tangent2 = (K > epsilon<T>) ? T(1) / K : T(0);
        }
    }

    // Solve contacts with given time step and parameters.
    // `velocity_iterations` – number of sequential impulse passes.
    // `baumgarte_factor` – fraction of penetration error corrected per step (0..0.2).
    void solve(T dt, int velocity_iterations = 8, T baumgarte_factor = T(0.2)) {
        if (!m_bodies || !m_contacts) return;

        for (int iter = 0; iter < velocity_iterations; ++iter) {
            for (auto& cp : *m_contacts) {
                if (cp.body_a < 0 || cp.body_b < 0) continue;
                auto& bodyA = (*m_bodies)[cp.body_a];
                auto& bodyB = (*m_bodies)[cp.body_b];

                vec3<T> rA = cp.point_a - bodyA.position;
                vec3<T> rB = cp.point_b - bodyB.position;
                vec3<T> n  = cp.normal;

                // ── Normal impulse ──
                // Relative velocity at contact
                vec3<T> vA = bodyA.linear_velocity + cross(bodyA.angular_velocity, rA);
                vec3<T> vB = bodyB.linear_velocity + cross(bodyB.angular_velocity, rB);
                vec3<T> v_rel = vB - vA;

                T vn = dot(v_rel, n);
                T bias = (baumgarte_factor / dt) * std::max(cp.penetration, T(0));
                T restitution = cp.combined_restitution;

                T delta_lambda = (-(T(1) + restitution) * vn - bias) * cp.effective_mass_normal;
                T lambda_old = cp.normal_impulse;
                T lambda_new = std::max(T(0), lambda_old + delta_lambda);
                delta_lambda = lambda_new - lambda_old;
                cp.normal_impulse = lambda_new;

                // Apply impulse
                vec3<T> impulse = n * delta_lambda;
                bodyA.linear_velocity  -= impulse * bodyA.inv_mass;
                bodyA.angular_velocity -= mul(bodyA.inv_inertia_world, cross(rA, impulse));
                bodyB.linear_velocity  += impulse * bodyB.inv_mass;
                bodyB.angular_velocity += mul(bodyB.inv_inertia_world, cross(rB, impulse));

                // ── Friction impulses ──
                // Recompute relative velocity after normal impulse
                vA = bodyA.linear_velocity + cross(bodyA.angular_velocity, rA);
                vB = bodyB.linear_velocity + cross(bodyB.angular_velocity, rB);
                v_rel = vB - vA;
                T vn_after = dot(v_rel, n);
                vec3<T> vt = v_rel - n * vn_after;
                T vt_len = length(vt);
                if (vt_len > epsilon<T>) {
                    vec3<T> t_dir = vt / vt_len;

                    T K_t = bodyA.inv_mass + bodyB.inv_mass;
                    K_t += dot(cross(rA, t_dir), mul(bodyA.inv_inertia_world, cross(rA, t_dir)));
                    K_t += dot(cross(rB, t_dir), mul(bodyB.inv_inertia_world, cross(rB, t_dir)));
                    T eff_mass_t = (K_t > epsilon<T>) ? T(1) / K_t : T(0);

                    T delta_tangent = -vt_len * eff_mass_t;
                    T max_friction = cp.combined_friction * cp.normal_impulse;
                    T lambda_t_old = cp.tangent_impulse1; // use single tangent for demonstration (in practice 2)
                    T lambda_t_new = std::max(-max_friction, std::min(max_friction, lambda_t_old + delta_tangent));
                    delta_tangent = lambda_t_new - lambda_t_old;
                    cp.tangent_impulse1 = lambda_t_new;

                    vec3<T> friction_impulse = t_dir * delta_tangent;
                    bodyA.linear_velocity  -= friction_impulse * bodyA.inv_mass;
                    bodyA.angular_velocity -= mul(bodyA.inv_inertia_world, cross(rA, friction_impulse));
                    bodyB.linear_velocity  += friction_impulse * bodyB.inv_mass;
                    bodyB.angular_velocity += mul(bodyB.inv_inertia_world, cross(rB, friction_impulse));
                }
            }
        }
    }

private:
    std::vector<SolverBody<T>>*   m_bodies   = nullptr;
    std::vector<ContactPoint<T>>* m_contacts = nullptr;

    // Create two orthonormal vectors perpendicular to n
    static void compute_tangents(const vec3<T>& n, vec3<T>& t1, vec3<T>& t2) noexcept {
        if (std::abs(n.x) < T(0.9)) {
            t1 = normalize(cross(n, vec3<T>(T(1), T(0), T(0))));
        } else {
            t1 = normalize(cross(n, vec3<T>(T(0), T(1), T(0))));
        }
        t2 = normalize(cross(n, t1));
    }
};

} // namespace wp