// File 0041 : core/math/integration.h
// Time integrators: Symplectic Euler, Velocity Verlet, Position Verlet, Leapfrog, RK4 for particles and rigid bodies with quaternions.

#pragma once

#include "vec3.h"
#include "quat.h"
#include "mat3.h"
#include "constants.h"
#include <functional>

namespace wp {

// ── Particle state ──────────────────────────────────────────────────
template <typename T>
struct ParticleState {
    vec3<T> position;
    vec3<T> velocity;
    vec3<T> acceleration;       // current or previous step acceleration (for Verlet)
    vec3<T> prev_position;      // used by position Verlet
};

// ── Rigid body state ────────────────────────────────────────────────
template <typename T>
struct RigidBodyState {
    vec3<T> position;
    quat<T> orientation;
    vec3<T> linear_velocity;
    vec3<T> angular_velocity;
};

// ── Force / torque function types ───────────────────────────────────
template <typename T>
using ForceFunc = std::function<vec3<T>(const ParticleState<T>&, T time)>;

template <typename T>
using TorqueFunc = std::function<vec3<T>(const RigidBodyState<T>&, T time)>;

// =========================================================================
// PARTICLE INTEGRATORS
// =========================================================================

/** Symplectic (semi‑implicit) Euler */
template <typename T>
void integrate_symplectic_euler(ParticleState<T>& state, T dt,
                                const ForceFunc<T>& force,
                                T time, T damping = T(1)) {
    vec3<T> acc = force(state, time);
    state.velocity = state.velocity * damping + acc * dt;
    state.position = state.position + state.velocity * dt;
}

/** Velocity Verlet */
template <typename T>
void integrate_velocity_verlet(ParticleState<T>& state, T dt,
                                const ForceFunc<T>& force, T time) {
    vec3<T> a0 = state.acceleration;          // acceleration at current step (from previous call)
    state.position = state.position + state.velocity * dt + a0 * (T(0.5) * dt * dt);
    vec3<T> a1 = force(state, time + dt);     // new acceleration
    state.velocity = state.velocity + (a0 + a1) * (T(0.5) * dt);
    state.acceleration = a1;
}

/** Position Verlet (using previous position stored in state) */
template <typename T>
void integrate_position_verlet(ParticleState<T>& state, T dt,
                                const ForceFunc<T>& force, T time) {
    vec3<T> a = force(state, time);
    vec3<T> new_pos = state.position * T(2) - state.prev_position + a * (dt * dt);
    state.velocity = (new_pos - state.prev_position) * (T(0.5) / dt);
    state.prev_position = state.position;
    state.position = new_pos;
}

/** Leapfrog (kick‑drift‑kick) */
template <typename T>
void integrate_leapfrog(ParticleState<T>& state, T dt,
                        const ForceFunc<T>& force, T time) {
    // half‑kick
    state.velocity = state.velocity + force(state, time) * (T(0.5) * dt);
    // drift
    state.position = state.position + state.velocity * dt;
    // half‑kick
    state.velocity = state.velocity + force(state, time + dt) * (T(0.5) * dt);
}

// =========================================================================
// RK4 for particle state (requires + and * operators on state)
// =========================================================================

/** RK4 step for a generic state type supporting + and *scalar */
template <typename State, typename T, typename Func>
State rk4_step(const State& y, T t, T h, const Func& f) {
    State k1 = f(y, t);
    State k2 = f(y + k1 * (h * T(0.5)), t + h * T(0.5));
    State k3 = f(y + k2 * (h * T(0.5)), t + h * T(0.5));
    State k4 = f(y + k3 * h, t + h);
    return y + (k1 + k2 * T(2) + k3 * T(2) + k4) * (h / T(6));
}

// Define necessary operators for ParticleState to use with rk4_step
template <typename T>
ParticleState<T> operator+(const ParticleState<T>& a, const ParticleState<T>& b) {
    return { a.position + b.position, a.velocity + b.velocity, a.acceleration + b.acceleration, a.prev_position + b.prev_position };
}
template <typename T>
ParticleState<T> operator*(const ParticleState<T>& a, T s) {
    return { a.position * s, a.velocity * s, a.acceleration * s, a.prev_position * s };
}

// The derivative function for RK4: given state and time, return d(state)/dt = (velocity, acceleration, ...)
template <typename T>
ParticleState<T> particle_derivative(const ParticleState<T>& s, T t, const ForceFunc<T>& force) {
    vec3<T> acc = force(s, t);
    return { s.velocity, acc, vec3<T>(T(0)), vec3<T>(T(0)) };   // derivative of position = velocity, derivative of velocity = acceleration
}

/** RK4 for particle using the above operators */
template <typename T>
void integrate_rk4(ParticleState<T>& state, T dt, const ForceFunc<T>& force, T time) {
    auto deriv_func = [&force](const ParticleState<T>& s, T t) {
        return particle_derivative(s, t, force);
    };
    state = rk4_step(state, time, dt, deriv_func);
}

// =========================================================================
// RIGID BODY INTEGRATORS (quaternions)
// =========================================================================

/** Symplectic Euler for rigid body */
template <typename T>
void integrate_rb_symplectic_euler(RigidBodyState<T>& state, T dt,
                                   const vec3<T>& gravity,
                                   const mat3<T>& inv_inertia_world,
                                   T mass, T inv_mass,
                                   const vec3<T>& external_torque = vec3<T>(T(0)),
                                   T linear_damping = T(1),
                                   T angular_damping = T(1)) {
    // Linear acceleration from gravity (no other forces in this simple example)
    vec3<T> lin_acc = gravity;
    state.linear_velocity  = state.linear_velocity * linear_damping + lin_acc * dt;
    state.angular_velocity = state.angular_velocity * angular_damping;

    // Position update
    state.position = state.position + state.linear_velocity * dt;

    // Orientation update: q(t+dt) = q(t) + 0.5 * dt * (0, w) * q(t)
    vec3<T> w = state.angular_velocity;
    quat<T> w_quat(w, T(0));
    quat<T> dq = mul(w_quat, state.orientation) * (T(0.5) * dt);
    state.orientation = normalize(state.orientation + dq);
}

/** Velocity Verlet for rigid body translation and rotation */
template <typename T>
void integrate_rb_velocity_verlet(RigidBodyState<T>& state, T dt,
                                  const vec3<T>& gravity,
                                  const mat3<T>& I_world, const mat3<T>& inv_I_world,
                                  T mass, T inv_mass,
                                  const vec3<T>& external_torque = vec3<T>(T(0))) {
    // Half-step angular velocity (use torque at current state)
    vec3<T> ang_momentum = mul(I_world, state.angular_velocity);
    vec3<T> torque = external_torque - cross(state.angular_velocity, ang_momentum);
    vec3<T> ang_acc = mul(inv_I_world, torque);
    vec3<T> ang_vel_half = state.angular_velocity + ang_acc * (T(0.5) * dt);

    // Position update (translational)
    state.position = state.position + state.linear_velocity * dt + gravity * (T(0.5) * dt * dt);

    // Orientation update using half-step angular velocity
    quat<T> omega_half_quat(ang_vel_half, T(0));
    quat<T> dq = mul(omega_half_quat, state.orientation) * (T(0.5) * dt);
    state.orientation = normalize(state.orientation + dq);

    // Full-step linear velocity
    state.linear_velocity = state.linear_velocity + gravity * dt;

    // Recompute angular acceleration at new orientation (if torque depends on orientation, recalc here)
    // For consistency, compute torque at new orientation:
    ang_momentum = mul(I_world, ang_vel_half);   // approximate
    torque = external_torque - cross(ang_vel_half, ang_momentum);
    ang_acc = mul(inv_I_world, torque);
    state.angular_velocity = ang_vel_half + ang_acc * (T(0.5) * dt);
}

// =========================================================================
// Auxiliary: initialize velocity Verlet acceleration
// =========================================================================
/** Call once before starting velocity Verlet to set initial acceleration */
template <typename T>
void init_velocity_verlet(ParticleState<T>& state, const ForceFunc<T>& force, T time) {
    state.acceleration = force(state, time);
}

} // namespace wp