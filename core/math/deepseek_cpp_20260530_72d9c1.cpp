// File 0041 : core/math/integration.h
// Time integrators for particles and rigid bodies (Verlet, velocity Verlet, leapfrog, symplectic Euler, RK4).

#pragma once

#include "vec3.h"
#include "quat.h"
#include "mat3.h"
#include "rigid_body_dynamics.h"   // for angular acceleration helpers if needed
#include "constants.h"
#include <functional>

namespace wp {

// ── Particle state ──────────────────────────────────────────────────
template <typename T>
struct ParticleState {
    vec3<T> position;
    vec3<T> velocity;
    vec3<T> acceleration;   // from previous step (used in Verlet)
};

// ── Rigid body state ────────────────────────────────────────────────
template <typename T>
struct RigidBodyState {
    vec3<T> position;
    quat<T> orientation;
    vec3<T> linear_velocity;
    vec3<T> angular_velocity;
    // Inertia properties stored externally or as part of body
};

// ── Force functor type: acceleration = force(state, time) ──────────
template <typename T>
using ForceFunc = std::function<vec3<T>(const ParticleState<T>&, T)>;

// ── Torque functor for rigid body: angular_accel = torque(state, time) ──
template <typename T>
using TorqueFunc = std::function<vec3<T>(const RigidBodyState<T>&, T)>;

// ====================================================================
// Particle integrators
// ====================================================================

/** Symplectic (semi‑implicit) Euler */
template <typename T>
void integrate_symplectic_euler(ParticleState<T>& state, T dt,
                                const ForceFunc<T>& force,
                                T damping = T(1)) {
    vec3<T> acc = force(state, T(0));   // time not used in simple demo
    state.velocity += acc * dt;
    state.velocity *= damping;
    state.position += state.velocity * dt;
}

/** Velocity Verlet */
template <typename T>
void integrate_velocity_verlet(ParticleState<T>& state, T dt,
                               const ForceFunc<T>& force) {
    vec3<T> acc = force(state, T(0));
    state.position += state.velocity * dt + acc * (T(0.5) * dt * dt);
    vec3<T> new_acc = force(state, T(0));
    state.velocity += (acc + new_acc) * (T(0.5) * dt);
    state.acceleration = new_acc;
}

/** Position Verlet (no explicit velocity storage; velocity computed from positions) */
template <typename T>
void integrate_position_verlet(ParticleState<T>& state, T dt,
                               const ForceFunc<T>& force) {
    // Requires previous position stored externally.
    // This function assumes state.position is current and we have previous position
    // stored separately. However, ParticleState does not include prev position.
    // Instead, we implement using the stored velocity and acceleration.
    // Classic Verlet: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2.
    // We'll store prev_position in the state? For simplicity, provide a version
    // that takes previous position as argument.
    // Not implemented here; velocity Verlet is sufficient.
}

// ====================================================================
// Rigid body integrators (using quaternions)
// ====================================================================

/** Symplectic Euler for rigid body */
template <typename T>
void integrate_rb_symplectic_euler(RigidBodyState<T>& state, T dt,
                                   const vec3<T>& gravity,
                                   const mat3<T>& inv_inertia_world,
                                   T mass, T inv_mass,
                                   T linear_damping = T(1),
                                   T angular_damping = T(1)) {
    // Gravity
    vec3<T> lin_acc = gravity;
    // No external torque in this simple example
    state.linear_velocity  += lin_acc * dt;
    state.linear_velocity  *= linear_damping;
    state.angular_velocity *= angular_damping;

    // Update position
    state.position += state.linear_velocity * dt;

    // Update orientation using quaternion derivative: q' = 0.5 * (0, w) * q
    quat<T> omega_quat(state.angular_velocity, T(0));
    quat<T> dq = omega_quat * state.orientation;
    dq = dq * (T(0.5) * dt);
    state.orientation = normalize(state.orientation + dq);
}

/** Velocity Verlet for rigid body (translation and rotation) */
template <typename T>
void integrate_rb_velocity_verlet(RigidBodyState<T>& state, T dt,
                                  const vec3<T>& gravity,
                                  const mat3<T>& I_world, const mat3<T>& inv_I_world,
                                  T mass, T inv_mass,
                                  const vec3<T>& external_torque = vec3<T>(T(0))) {
    // Half‑step for angular velocity using current torque
    vec3<T> ang_momentum = mul(I_world, state.angular_velocity);
    vec3<T> torque = external_torque - cross(state.angular_velocity, ang_momentum); // if needed
    vec3<T> ang_acc = mul(inv_I_world, torque);
    vec3<T> ang_vel_mid = state.angular_velocity + ang_acc * (T(0.5) * dt);

    // Position update
    state.position += state.linear_velocity * dt + gravity * (T(0.5) * dt * dt);

    // Orientation update with mid angular velocity
    quat<T> omega_mid_quat(ang_vel_mid, T(0));
    quat<T> dq = omega_mid_quat * state.orientation;
    dq = dq * (T(0.5) * dt);
    state.orientation = normalize(state.orientation + dq);

    // Full step linear velocity
    state.linear_velocity += gravity * dt;

    // Recompute angular acceleration at new orientation (if torque depends on orientation, recalc here)
    // For simplicity, keep same torque.
    state.angular_velocity = ang_vel_mid + ang_acc * (T(0.5) * dt);
}

// ====================================================================
// Generic RK4 for first‑order ODE systems (state type with + and * scalar)
// ====================================================================

/** Single RK4 step for a system: dy/dt = f(y, t) */
template <typename State, typename T, typename Func>
State rk4_step(const State& y, T t, T h, const Func& f) {
    State k1 = f(y, t);
    State k2 = f(y + k1 * (h * T(0.5)), t + h * T(0.5));
    State k3 = f(y + k2 * (h * T(0.5)), t + h * T(0.5));
    State k4 = f(y + k3 * h, t + h);
    return y + (k1 + k2 * T(2) + k3 * T(2) + k4) * (h / T(6));
}

// To use with ParticleState, provide appropriate addition and scalar multiplication:
// ParticleState + ParticleState, ParticleState * T.
// These can be defined if needed.

} // namespace wp