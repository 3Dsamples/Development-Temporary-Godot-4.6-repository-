// system name : onetbb-warp
// File 0014 : core/math/ode.h
// Description : Numerical ODE solvers (Euler, RK4, Verlet) for simulation dynamics.

#ifndef __TBB_WARP_CORE_MATH_ODE_H
#define __TBB_WARP_CORE_MATH_ODE_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/quaternion.h"
#include <cmath>
#include <type_traits>
#include <functional>

namespace tbb {
namespace core {
namespace math {
namespace ode {

// ============================================================
// Forward Euler (explicit)
// ============================================================

template<typename State, typename DerivativeFunc>
State euler_step(const State& y, float h, DerivativeFunc f) {
    return y + f(y) * h;
}

template<typename State, typename DerivativeFunc>
State euler_step_time(const State& y, float t, float h, DerivativeFunc f) {
    return y + f(t, y) * h;
}

// ============================================================
// Explicit Midpoint (Runge-Kutta 2)
// ============================================================

template<typename State, typename DerivativeFunc>
State midpoint_step(const State& y, float h, DerivativeFunc f) {
    auto k1 = f(y);
    auto y_mid = y + k1 * (0.5f * h);
    auto k2 = f(y_mid);
    return y + k2 * h;
}

template<typename State, typename DerivativeFunc>
State midpoint_step_time(const State& y, float t, float h, DerivativeFunc f) {
    auto k1 = f(t, y);
    auto y_mid = y + k1 * (0.5f * h);
    auto k2 = f(t + 0.5f * h, y_mid);
    return y + k2 * h;
}

// ============================================================
// Heun's method (improved Euler)
// ============================================================

template<typename State, typename DerivativeFunc>
State heun_step(const State& y, float h, DerivativeFunc f) {
    auto k1 = f(y);
    auto y_pred = y + k1 * h;
    auto k2 = f(y_pred);
    return y + (k1 + k2) * (0.5f * h);
}

template<typename State, typename DerivativeFunc>
State heun_step_time(const State& y, float t, float h, DerivativeFunc f) {
    auto k1 = f(t, y);
    auto y_pred = y + k1 * h;
    auto k2 = f(t + h, y_pred);
    return y + (k1 + k2) * (0.5f * h);
}

// ============================================================
// Classic Runge-Kutta 4 (RK4)
// ============================================================

template<typename State, typename DerivativeFunc>
State rk4_step(const State& y, float h, DerivativeFunc f) {
    auto k1 = f(y);
    auto y2 = y + k1 * (0.5f * h);
    auto k2 = f(y2);
    auto y3 = y + k2 * (0.5f * h);
    auto k3 = f(y3);
    auto y4 = y + k3 * h;
    auto k4 = f(y4);
    return y + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (h / 6.0f);
}

template<typename State, typename DerivativeFunc>
State rk4_step_time(const State& y, float t, float h, DerivativeFunc f) {
    auto k1 = f(t, y);
    auto y2 = y + k1 * (0.5f * h);
    auto k2 = f(t + 0.5f * h, y2);
    auto y3 = y + k2 * (0.5f * h);
    auto k3 = f(t + 0.5f * h, y3);
    auto y4 = y + k3 * h;
    auto k4 = f(t + h, y4);
    return y + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (h / 6.0f);
}

// ============================================================
// Runge-Kutta-Fehlberg 4(5) with adaptive step size
// ============================================================

template<typename State, typename DerivativeFunc, typename ErrorFunc>
float rkf45_step(State& y, float t, float& h, DerivativeFunc f, ErrorFunc error_fn,
                 float tolerance = 1e-6f, float min_h = 1e-8f, float max_h = 0.1f) {
    // Butcher tableau for RKF45
    auto k1 = f(t, y) * h;
    auto y2 = y + k1 * (1.0f / 4.0f);
    auto k2 = f(t + h * 0.25f, y2) * h;
    auto y3 = y + k1 * (3.0f / 32.0f) + k2 * (9.0f / 32.0f);
    auto k3 = f(t + h * (3.0f / 8.0f), y3) * h;
    auto y4 = y + k1 * (1932.0f / 2197.0f) - k2 * (7200.0f / 2197.0f) + k3 * (7296.0f / 2197.0f);
    auto k4 = f(t + h * (12.0f / 13.0f), y4) * h;
    auto y5 = y + k1 * (439.0f / 216.0f) - k2 * 8.0f + k3 * (3680.0f / 513.0f) - k4 * (845.0f / 4104.0f);
    auto k5 = f(t + h, y5) * h;
    auto y6 = y - k1 * (8.0f / 27.0f) + k2 * 2.0f - k3 * (3544.0f / 2565.0f) + k4 * (1859.0f / 4104.0f) - k5 * (11.0f / 40.0f);
    auto k6 = f(t + h * 0.5f, y6) * h;
    // 4th order estimate
    State y4_est = y + k1 * (25.0f / 216.0f) + k3 * (1408.0f / 2565.0f) + k4 * (2197.0f / 4101.0f) - k5 * (1.0f / 5.0f);
    // 5th order estimate
    State y5_est = y + k1 * (16.0f / 135.0f) + k3 * (6656.0f / 12825.0f) + k4 * (28561.0f / 56430.0f) - k5 * (9.0f / 50.0f) + k6 * (2.0f / 55.0f);
    float error = error_fn(y5_est, y4_est);
    float safety = 0.9f;
    float h_new = h * safety * std::pow(tolerance / (error + 1e-12f), 0.2f);
    h_new = clamp(h_new, min_h, max_h);
    y = y5_est;
    float old_h = h;
    h = h_new;
    return old_h;
}

// ============================================================
// Verlet integration (position‑only, velocity implicit)
// ============================================================

template<typename State, typename AccelerationFunc>
void verlet_step(State& x, State& x_prev, float h, AccelerationFunc a_fn) {
    State x_new = x * 2.0f - x_prev + a_fn(x) * (h * h);
    x_prev = x;
    x = x_new;
}

// ============================================================
// Velocity Verlet (explicit velocity)
// ============================================================

template<typename State, typename AccelerationFunc>
void velocity_verlet_step(State& x, State& v, float h, AccelerationFunc a_fn) {
    State a_current = a_fn(x);
    State x_new = x + v * h + a_current * (0.5f * h * h);
    State a_new = a_fn(x_new);
    State v_new = v + (a_current + a_new) * (0.5f * h);
    x = x_new;
    v = v_new;
}

template<typename State, typename AccelerationFunc>
void velocity_verlet_step_time(State& x, State& v, float t, float h, AccelerationFunc a_fn) {
    State a_current = a_fn(t, x);
    State x_new = x + v * h + a_current * (0.5f * h * h);
    State a_new = a_fn(t + h, x_new);
    State v_new = v + (a_current + a_new) * (0.5f * h);
    x = x_new;
    v = v_new;
}

// ============================================================
// Leapfrog (staggered Verlet)
// ============================================================

template<typename State, typename AccelerationFunc>
void leapfrog_step(State& x, State& v, float h, AccelerationFunc a_fn) {
    State a_half = a_fn(x);
    State v_half = v + a_half * (0.5f * h);
    x = x + v_half * h;
    State a_full = a_fn(x);
    v = v_half + a_full * (0.5f * h);
}

// ============================================================
// Symplectic Euler (semi‑implicit)
// ============================================================

template<typename State, typename AccelerationFunc>
void symplectic_euler_step(State& x, State& v, float h, AccelerationFunc a_fn) {
    v = v + a_fn(x) * h;
    x = x + v * h;
}

template<typename State, typename AccelerationFunc>
void symplectic_euler_step_time(State& x, State& v, float t, float h, AccelerationFunc a_fn) {
    v = v + a_fn(t, x) * h;
    x = x + v * h;
}

// ============================================================
// Rotational integration for quaternions (angular velocity)
// ============================================================

inline quaternion<float> integrate_angular_velocity(
    const quaternion<float>& q, const vector3<float>& omega, float dt)
{
    float len_sq = length_sq(omega);
    if (len_sq < FLOAT_EPSILON) return q;
    float theta = std::sqrt(len_sq) * dt * 0.5f;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    float inv_len = 1.0f / std::sqrt(len_sq);
    quaternion<float> delta(
        omega.x * inv_len * sin_theta,
        omega.y * inv_len * sin_theta,
        omega.z * inv_len * sin_theta,
        cos_theta
    );
    return normalize(q * delta);
}

inline quaternion<double> integrate_angular_velocity(
    const quaternion<double>& q, const vector3<double>& omega, double dt)
{
    double len_sq = length_sq(omega);
    if (len_sq < DOUBLE_EPSILON) return q;
    double theta = std::sqrt(len_sq) * dt * 0.5;
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double inv_len = 1.0 / std::sqrt(len_sq);
    quaternion<double> delta(
        omega.x * inv_len * sin_theta,
        omega.y * inv_len * sin_theta,
        omega.z * inv_len * sin_theta,
        cos_theta
    );
    return normalize(q * delta);
}

// ============================================================
// Rigid body state integration (position, orientation, velocity)
// ============================================================

template<typename T>
struct rigid_body_state {
    vector3<T> position;
    vector3<T> linear_velocity;
    quaternion<T> orientation;
    vector3<T> angular_velocity;
};

template<typename T, typename ForceTorqueFunc>
void rigid_body_symplectic_euler(rigid_body_state<T>& body, T mass, T dt, ForceTorqueFunc ft_fn) {
    // ft_fn returns pair<force, torque> in world frame
    auto [force, torque] = ft_fn(body.position, body.orientation, body.linear_velocity, body.angular_velocity);
    vector3<T> linear_accel = force * (T(1) / mass);
    body.linear_velocity = body.linear_velocity + linear_accel * dt;
    body.position = body.position + body.linear_velocity * dt;
    // Angular: torque -> angular acceleration (simplified: ignore inertia tensor)
    vector3<T> angular_accel = torque; // assume unit inertia
    body.angular_velocity = body.angular_velocity + angular_accel * dt;
    body.orientation = integrate_angular_velocity(body.orientation, body.angular_velocity, dt);
}

// ============================================================
// Runge‑Kutta 4 for second‑order ODE (position + velocity)
// ============================================================

template<typename State, typename AccelerationFunc>
void rk4_second_order(State& x, State& v, float h, AccelerationFunc a_fn) {
    auto a1 = a_fn(x);
    auto x2 = x + v * (0.5f * h) + a1 * (0.125f * h * h);
    auto v2 = v + a1 * (0.5f * h);
    auto a2 = a_fn(x2);
    auto x3 = x + v * (0.5f * h) + a2 * (0.125f * h * h);
    auto v3 = v + a2 * (0.5f * h);
    auto a3 = a_fn(x3);
    auto x4 = x + v3 * h + a3 * (0.5f * h * h);
    auto v4 = v + a3 * h;
    auto a4 = a_fn(x4);
    x = x + (v + (v2 + v3) * 2.0f + v4) * (h / 6.0f);
    v = v + (a1 + a2 * 2.0f + a3 * 2.0f + a4) * (h / 6.0f);
}

// ============================================================
// Stormer‑Verlet for Hamiltonian systems
// ============================================================

template<typename State, typename ForceFunc>
void stormer_verlet_step(State& x, State& v, float h, ForceFunc force_fn) {
    auto force = force_fn(x);
    v = v + force * (0.5f * h);
    x = x + v * h;
    force = force_fn(x);
    v = v + force * (0.5f * h);
}

// ============================================================
// Gear predictor‑corrector (4th order)
// ============================================================

template<typename State>
struct gear_4_state {
    State y;     // position
    State dy;    // velocity * h
    State d2y;   // acceleration * h^2 / 2
    State d3y;   // jerk * h^3 / 6
};

template<typename State, typename AccelerationFunc>
void gear_4_predict(gear_4_state<State>& s) {
    s.y  = s.y + s.dy + s.d2y + s.d3y;
    s.dy = s.dy + s.d2y * 2.0f + s.d3y * 3.0f;
    s.d2y = s.d2y + s.d3y * 3.0f;
    s.d3y = s.d3y;
}

template<typename State, typename AccelerationFunc>
void gear_4_correct(gear_4_state<State>& s, float h, AccelerationFunc a_fn) {
    State a_new = a_fn(s.y) * (h * h * 0.5f);
    State error = a_new - s.d2y;
    const float coeff[4] = {19.0f/90.0f, 3.0f/4.0f, 1.0f/1.0f, 1.0f/2.0f};
    s.y  = s.y  + error * coeff[0];
    s.dy = s.dy + error * coeff[1];
    s.d2y = a_new;
    s.d3y = s.d3y + error * coeff[3];
}

// ============================================================
// Runge-Kutta-Nyström (for second‑order ODE without velocity)
// ============================================================

template<typename State, typename AccelerationFunc>
State rkn4_step(const State& x, float h, AccelerationFunc a_fn) {
    auto k1 = a_fn(x) * (h * h);
    auto k2 = a_fn(x + k1 * 0.125f) * (h * h);
    auto k3 = a_fn(x + k1 * 0.25f) * (h * h);
    auto k4 = a_fn(x + k1 * 0.5f - k2 * 0.5f + k3) * (h * h);
    return x + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (1.0f / 6.0f);
}

} // namespace ode
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_ODE_H