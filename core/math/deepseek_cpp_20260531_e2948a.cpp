// File 0047 : core/math/spring_damper.h
// Spring‑damper dynamics (Hooke's law) for scalar and 3D systems, including chain of springs.

#pragma once

#include "vec3.h"
#include "constants.h"
#include <vector>
#include <cmath>

namespace wp {

// ── Scalar spring‑damper ────────────────────────────────────────────
template <typename T>
class ScalarSpringDamper {
public:
    T stiffness;
    T damping;
    T rest_length;
    T current_length;
    T velocity;        // rate of change of length

    ScalarSpringDamper(T k = T(10), T d = T(0.5), T rl = T(1))
        : stiffness(k), damping(d), rest_length(rl), current_length(rl), velocity(T(0)) {}

    // Set state (length and its derivative)
    void set_state(T len, T vel) noexcept { current_length = len; velocity = vel; }

    // Compute spring force (positive = tension extending)
    T compute_force() const noexcept {
        T stretch = current_length - rest_length;
        return -stiffness * stretch - damping * velocity;
    }

    // Update length and velocity using Euler integration of the internal dynamics
    // (mass attached to the spring, not part of this class)
    void integrate_length(T force, T inv_mass, T dt) noexcept {
        T accel = force * inv_mass;
        velocity += accel * dt;
        current_length += velocity * dt;
    }
};

// ── 3D spring‑damper (between two points) ───────────────────────────
template <typename T>
class SpringDamper3D {
public:
    vec3<T> anchor_a;      // one endpoint (world)
    vec3<T> anchor_b;      // other endpoint (world)
    T       rest_length;
    T       stiffness;
    T       damping;

    SpringDamper3D(const vec3<T>& a = vec3<T>(T(0)),
                   const vec3<T>& b = vec3<T>(T(1), T(0), T(0)),
                   T k = T(10), T d = T(0.5))
        : anchor_a(a), anchor_b(b), rest_length(distance(a,b)), stiffness(k), damping(d) {}

    // Compute the force vector on anchor_a (opposite on b) given current positions
    vec3<T> force_on_a(const vec3<T>& pos_a, const vec3<T>& pos_b,
                       const vec3<T>& vel_a, const vec3<T>& vel_b) const noexcept {
        vec3<T> ab = pos_b - pos_a;
        T len = length(ab);
        if (len < epsilon<T>) return vec3<T>(T(0));
        vec3<T> dir = ab / len;
        T stretch = len - rest_length;
        // Relative velocity along spring direction
        T rel_vel = dot(vel_b - vel_a, dir);
        T force_mag = stiffness * stretch + damping * rel_vel;
        return dir * force_mag;   // force on a is towards b
    }

    // Apply forces to both endpoints (modify forces and positions? Not modifying, just compute)
};

// ── Chain of springs (1D line) ──────────────────────────────────────
template <typename T>
class SpringChain {
public:
    std::vector<ScalarSpringDamper<T>> springs;
    std::vector<T> positions;       // positions of nodes (size = springs.size() + 1)
    std::vector<T> velocities;      // velocities of nodes
    std::vector<T> masses;          // mass of each node

    SpringChain(int num_nodes, T stiffness, T damping, T total_length)
        : springs(num_nodes - 1, ScalarSpringDamper<T>(stiffness, damping, total_length / T(num_nodes - 1))),
          positions(num_nodes, T(0)),
          velocities(num_nodes, T(0)),
          masses(num_nodes, T(1)) {
        T rest_len = total_length / T(num_nodes - 1);
        for (int i = 0; i < num_nodes; ++i)
            positions[i] = T(i) * rest_len;
    }

    // Integrate one step with external forces (zero for free vibration)
    void step(T dt, const std::vector<T>* external_forces = nullptr) {
        int n = static_cast<int>(positions.size());
        std::vector<T> forces(n, T(0));

        // Spring forces between adjacent nodes
        for (int i = 0; i < n - 1; ++i) {
            T len = positions[i+1] - positions[i];
            T vel_rel = velocities[i+1] - velocities[i];
            T force = springs[i].stiffness * (len - springs[i].rest_length) + springs[i].damping * vel_rel;
            forces[i]   += force;   // pull right on i
            forces[i+1] -= force;   // pull left on i+1
        }

        // External forces
        if (external_forces) {
            for (int i = 0; i < n; ++i)
                forces[i] += (*external_forces)[i];
        }

        // Integrate (symplectic Euler)
        for (int i = 0; i < n; ++i) {
            T accel = forces[i] / masses[i];
            velocities[i] += accel * dt;
            positions[i]  += velocities[i] * dt;
        }
    }
};

} // namespace wp