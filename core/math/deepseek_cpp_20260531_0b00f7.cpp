//File 0035 : core/math/ode_integrators.h
//Complete suite of ODE integrators: explicit Euler, Heun, RK4, RKF45 (adaptive), DOPRI5, Cash‑Karp, symplectic Euler, velocity Verlet, leapfrog, implicit Euler (Newton), and specialized SIMD 3D versions.
#ifndef CORE_MATH_ODE_INTEGRATORS_H
#define CORE_MATH_ODE_INTEGRATORS_H

#include "vector_math.h"
#include <functional>
#include <cmath>
#include <limits>
#include <utility>

namespace SimulationMath {
namespace ode {

// ====================================================================
// Generic scalar / vector‑like integrators (use concept of arithmetic)
// ====================================================================

// ---------- 1. Explicit Euler ----------
template <typename State>
State euler_step(const State& y, float t, float dt,
                 const std::function<State(float, const State&)>& f) {
    return y + dt * f(t, y);
}

// ---------- 2. Heun (improved Euler) ----------
template <typename State>
State heun_step(const State& y, float t, float dt,
                const std::function<State(float, const State&)>& f) {
    State k1 = f(t, y);
    State k2 = f(t + dt, y + dt * k1);
    return y + (dt * 0.5f) * (k1 + k2);
}

// ---------- 3. Classic RK4 ----------
template <typename State>
State rk4_step(const State& y, float t, float dt,
               const std::function<State(float, const State&)>& f) {
    State k1 = f(t, y);
    State k2 = f(t + dt * 0.5f, y + (dt * 0.5f) * k1);
    State k3 = f(t + dt * 0.5f, y + (dt * 0.5f) * k2);
    State k4 = f(t + dt, y + dt * k3);
    return y + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
}

// ---------- 4. RKF45 adaptive (Fehlberg coefficients) ----------
template <typename State>
std::pair<State, float> rkf45_step(const State& y, float t, float dt,
                                   const std::function<State(float, const State&)>& f,
                                   float tolerance = 1e-6f) {
    const float a2 = 1.0f/4.0f;
    const float a3 = 3.0f/8.0f;
    const float a4 = 12.0f/13.0f;
    // a5 = 1.0f, a6 = 1/2
    State k1 = f(t, y);
    State k2 = f(t + a2*dt, y + dt * ((1.0f/4.0f) * k1));
    State k3 = f(t + a3*dt, y + dt * ((3.0f/32.0f)*k1 + (9.0f/32.0f)*k2));
    State k4 = f(t + a4*dt, y + dt * ((1932.0f/2197.0f)*k1 - (7200.0f/2197.0f)*k2 + (7296.0f/2197.0f)*k3));
    State k5 = f(t + dt, y + dt * ((439.0f/216.0f)*k1 - 8.0f*k2 + (3680.0f/513.0f)*k3 - (845.0f/4104.0f)*k4));
    State k6 = f(t + 0.5f*dt, y + dt * ((-8.0f/27.0f)*k1 + 2.0f*k2 - (3544.0f/2565.0f)*k3 + (1859.0f/4104.0f)*k4 - (11.0f/40.0f)*k5));
    // 4th order solution
    State y4 = y + dt * ((25.0f/216.0f)*k1 + (1408.0f/2565.0f)*k3 + (2197.0f/4104.0f)*k4 - (1.0f/5.0f)*k5);
    // 5th order solution (error estimator)
    State y5 = y + dt * ((16.0f/135.0f)*k1 + (6656.0f/12825.0f)*k3 + (28561.0f/56430.0f)*k4 - (9.0f/50.0f)*k5 + (2.0f/55.0f)*k6);
    float error = (y5 - y4).norm();
    float rel_error = error / (std::max(y4.norm(), 1e-12f) + 1e-12f);
    const float safety = 0.9f;
    const float exp = (rel_error > tolerance) ? 0.2f : 0.25f;
    float new_dt = dt * safety * std::pow(tolerance / std::max(rel_error, 1e-14f), exp);
    new_dt = std::min(new_dt, 2.0f * dt);
    new_dt = std::max(new_dt, 0.1f * dt);
    return {y4, new_dt};
}

// ---------- 5. DOPRI5 (Dormand‑Prince 5(4)) – optimized for low memory ----------
template <typename State>
std::pair<State, float> dopri5_step(const State& y, float t, float dt,
                                    const std::function<State(float, const State&)>& f,
                                    float tolerance = 1e-6f) {
    // Butcher tableau for DOPRI5 with FSAL property
    const float c2 = 1.0f/5.0f;
    const float c3 = 3.0f/10.0f;
    const float c4 = 4.0f/5.0f;
    const float c5 = 8.0f/9.0f;
    const float c6 = 1.0f;
    const float c7 = 1.0f; // FSAL

    State k1 = f(t, y);
    State k2 = f(t + c2*dt, y + dt * (1.0f/5.0f * k1));
    State k3 = f(t + c3*dt, y + dt * (3.0f/40.0f * k1 + 9.0f/40.0f * k2));
    State k4 = f(t + c4*dt, y + dt * (44.0f/45.0f * k1 - 56.0f/15.0f * k2 + 32.0f/9.0f * k3));
    State k5 = f(t + c5*dt, y + dt * (19372.0f/6561.0f * k1 - 25360.0f/2187.0f * k2 + 64448.0f/6561.0f * k3 - 212.0f/729.0f * k4));
    State k6 = f(t + c6*dt, y + dt * (9017.0f/3168.0f * k1 - 355.0f/33.0f * k2 + 46732.0f/5247.0f * k3 + 49.0f/176.0f * k4 - 5103.0f/18656.0f * k5));
    // 5th order solution
    State y5 = y + dt * (35.0f/384.0f * k1 + 500.0f/1113.0f * k3 + 125.0f/192.0f * k4 - 2187.0f/6784.0f * k5 + 11.0f/84.0f * k6);
    // 4th order estimate (FSAL, uses k2=k1 of next step, but here we compute explicitly)
    State k7 = f(t + dt, y5); // actual k7 for error
    State y4 = y + dt * (5179.0f/57600.0f * k1 + 7571.0f/16695.0f * k3 + 393.0f/640.0f * k4 - 92097.0f/339200.0f * k5 + 187.0f/2100.0f * k6 + 1.0f/40.0f * k7);
    float error = (y5 - y4).norm();
    float rel_error = error / (std::max(y5.norm(), 1e-12f) + 1e-12f);
    const float safety = 0.9f;
    float new_dt;
    if (rel_error > tolerance) {
        new_dt = dt * safety * std::pow(tolerance / rel_error, 0.2f);
        new_dt = std::max(new_dt, 0.1f * dt);
        return {y5, new_dt}; // reject step (but return y5 anyway; caller must check error)
    } else {
        new_dt = dt * safety * std::pow(tolerance / rel_error, 0.25f);
        new_dt = std::min(new_dt, 2.0f * dt);
        return {y5, new_dt};
    }
}

// ---------- 6. Cash‑Karp (RK5) adaptive ----------
template <typename State>
std::pair<State, float> cash_karp_step(const State& y, float t, float dt,
                                       const std::function<State(float, const State&)>& f,
                                       float tolerance = 1e-6f) {
    // Butcher tableau for Cash‑Karp
    const float b2 = 1.0f/5.0f;
    const float b3 = 3.0f/10.0f;
    const float b4 = 3.0f/5.0f;
    const float b5 = 1.0f;
    const float b6 = 7.0f/8.0f;

    State k1 = f(t, y);
    State k2 = f(t + b2*dt, y + dt * (1.0f/5.0f * k1));
    State k3 = f(t + b3*dt, y + dt * (3.0f/40.0f * k1 + 9.0f/40.0f * k2));
    State k4 = f(t + b4*dt, y + dt * (3.0f/10.0f * k1 - 9.0f/10.0f * k2 + 6.0f/5.0f * k3));
    State k5 = f(t + dt, y + dt * (-11.0f/54.0f * k1 + 5.0f/2.0f * k2 - 70.0f/27.0f * k3 + 35.0f/27.0f * k4));
    State k6 = f(t + b6*dt, y + dt * (1631.0f/55296.0f * k1 + 175.0f/512.0f * k2 + 575.0f/13824.0f * k3 + 44275.0f/110592.0f * k4 + 253.0f/4096.0f * k5));
    // 5th order solution
    State y5 = y + dt * (37.0f/378.0f * k1 + 250.0f/621.0f * k3 + 125.0f/594.0f * k4 + 512.0f/1771.0f * k6);
    // 4th order solution (embedded)
    State y4 = y + dt * (2825.0f/27648.0f * k1 + 18575.0f/48384.0f * k3 + 13525.0f/55296.0f * k4 + 277.0f/14336.0f * k5 + 1.0f/4.0f * k6);
    float error = (y5 - y4).norm();
    float rel_error = error / (std::max(y5.norm(), 1e-12f) + 1e-12f);
    const float safety = 0.9f;
    float new_dt = dt * safety * std::pow(tolerance / std::max(rel_error, 1e-14f), (rel_error > tolerance) ? 0.2f : 0.25f);
    new_dt = std::min(new_dt, 2.0f * dt);
    new_dt = std::max(new_dt, 0.1f * dt);
    return {y5, new_dt};
}

// ====================================================================
// Specialized second‑order ODE integrators for (position, velocity) systems
// ====================================================================

// ---------- 7. Symplectic Euler (kick‑drift for separable systems) ----------
template <typename Position, typename Velocity, typename AccelerationFunc>
void symplectic_euler_step(Position& x, Velocity& v, const AccelerationFunc& accel, float dt) {
    // Assumes acceleration depends only on position (and possibly time), not velocity.
    Velocity a = accel(x);       // current acceleration
    v = v + a * dt;              // full kick
    x = x + v * dt;              // drift with updated velocity
}

// ---------- 8. Velocity Verlet (second‑order symplectic) ----------
template <typename Position, typename Velocity, typename AccelerationFunc>
void velocity_verlet_step(Position& x, Velocity& v, AccelerationFunc& accel, float dt) {
    // accel returns acceleration given position.
    Velocity a0 = accel(x);      // current acceleration
    x = x + v * dt + (0.5f * dt * dt) * a0;
    Velocity a1 = accel(x);
    v = v + (dt * 0.5f) * (a0 + a1);
}

// ---------- 9. Leapfrog (drift‑kick‑drift / kick‑drift‑kick) ----------
template <typename Position, typename Velocity, typename AccelerationFunc>
void leapfrog_step(Position& x, Velocity& v, AccelerationFunc& accel, float dt) {
    // Classic leapfrog: half‑kick, full drift, half‑kick
    Velocity a0 = accel(x);
    v = v + (dt * 0.5f) * a0;   // half‑kick
    x = x + v * dt;              // full drift
    Velocity a1 = accel(x);      // recompute acceleration after drift
    v = v + (dt * 0.5f) * a1;   // second half‑kick
}

// ====================================================================
// Implicit integrators (for stiff problems)
// ====================================================================

// ---------- 10. Implicit Euler with Newton‑Raphson (generic) ----------
template <typename State>
State implicit_euler_step(const State& y, float t, float dt,
                          const std::function<State(float, const State&)>& f,
                          const std::function<State(float, const State&, const State&)>& solve_linear, // solves (I - dt*J) * delta = -r
                          int max_iter = 10, float tol = 1e-6f) {
    State yn1 = y;  // initial guess
    for (int iter = 0; iter < max_iter; ++iter) {
        State r = yn1 - y - dt * f(t + dt, yn1); // residual
        State delta;
        // The user‑provided function solves the linear system: (I - dt * J) * delta = -r
        // where J is the Jacobian of f evaluated at yn1.
        solve_linear(t + dt, yn1, r, delta); // should set delta = -(I-dt*J)^{-1} r
        yn1 = yn1 + delta;
        if (delta.norm() < tol * std::max(yn1.norm(), 1.0f)) break;
    }
    return yn1;
}

// ---------- 11. Backward Differentiation Formula (BDF2, fixed‑step) ----------
template <typename State>
State bdf2_step(const State& y, const State& y_prev, float t, float dt,
                const std::function<State(float, const State&)>& f,
                const std::function<State(float, const State&, const State&)>& solve_linear) {
    // BDF2: y_{n+2} = (4/3) y_{n+1} - (1/3) y_n + (2/3) dt f(t_{n+2}, y_{n+2})
    State yn2_guess = 2.0f * y - y_prev; // linear extrapolation
    const float a0 = 1.0f;
    const float a1 = -4.0f/3.0f;
    const float a2 = 1.0f/3.0f;
    const float beta = 2.0f/3.0f;
    // Solve a0 * y_{n+2} + a1 * y + a2 * y_prev = dt * beta * f(t+dt, y_{n+2})
    State yn2 = yn2_guess;
    for (int iter = 0; iter < 10; ++iter) {
        State r = yn2 - (4.0f/3.0f) * y + (1.0f/3.0f) * y_prev - dt * beta * f(t + dt, yn2);
        State delta;
        // Solve (I - dt*beta * J) * delta = -r
        solve_linear(t + dt, yn2, r, delta);
        yn2 = yn2 + delta;
        if (delta.norm() < 1e-6f) break;
    }
    return yn2;
}

// ====================================================================
// 3D SIMD explicit integrators (using DirectXMath for position/velocity)
// ====================================================================

// Helper: assume acceleration function returns DirectX::XMVECTOR
using SimdVec = DirectX::XMVECTOR;

inline SimdVec simd_scale(SimdVec v, float s) { return DirectX::XMVectorScale(v, s); }
inline SimdVec simd_add(SimdVec a, SimdVec b) { return DirectX::XMVectorAdd(a, b); }

// ---------- 12. SIMD Euler step for 3D ----------
inline void simd_euler_step(SimdVec& pos, SimdVec& vel,
                             const std::function<SimdVec(SimdVec)>& accel, float dt) {
    SimdVec a = accel(pos);
    vel = simd_add(vel, simd_scale(a, dt));
    pos = simd_add(pos, simd_scale(vel, dt));
}

// ---------- 13. SIMD Heun ----------
inline void simd_heun_step(SimdVec& pos, SimdVec& vel,
                            const std::function<SimdVec(SimdVec)>& accel, float dt) {
    SimdVec a0 = accel(pos);
    SimdVec vel1 = simd_add(vel, simd_scale(a0, dt));
    SimdVec pos1 = simd_add(pos, simd_scale(vel, dt));
    SimdVec a1 = accel(pos1);
    vel = simd_add(vel, simd_scale(simd_add(a0, a1), 0.5f * dt));
    pos = simd_add(pos, simd_scale(simd_add(vel, vel1), 0.5f * dt));
}

// ---------- 14. SIMD RK4 for position (second‑order ODE) ----------
inline void simd_rk4_step(SimdVec& pos, SimdVec& vel,
                           const std::function<SimdVec(SimdVec)>& accel, float dt) {
    SimdVec v0 = vel;
    SimdVec a0 = accel(pos);

    SimdVec v1 = simd_add(vel, simd_scale(a0, 0.5f * dt));
    SimdVec p1 = simd_add(pos, simd_scale(v0, 0.5f * dt));
    SimdVec a1 = accel(p1);

    SimdVec v2 = simd_add(vel, simd_scale(a1, 0.5f * dt));
    SimdVec p2 = simd_add(pos, simd_scale(v1, 0.5f * dt));
    SimdVec a2 = accel(p2);

    SimdVec v3 = simd_add(vel, simd_scale(a2, dt));
    SimdVec p3 = simd_add(pos, simd_scale(v2, dt));
    SimdVec a3 = accel(p3);

    vel = simd_add(vel, simd_scale(simd_add(simd_add(a0, simd_scale(a1, 2.0f)), simd_add(simd_scale(a2, 2.0f), a3)), dt / 6.0f));
    pos = simd_add(pos, simd_scale(simd_add(simd_add(v0, simd_scale(v1, 2.0f)), simd_add(simd_scale(v2, 2.0f), v3)), dt / 6.0f));
}

// ---------- 15. SIMD Velocity Verlet ----------
inline void simd_velocity_verlet_step(SimdVec& pos, SimdVec& vel,
                                      const std::function<SimdVec(SimdVec)>& accel, float dt) {
    SimdVec a0 = accel(pos);
    pos = simd_add(pos, simd_add(simd_scale(vel, dt), simd_scale(a0, 0.5f * dt * dt)));
    SimdVec a1 = accel(pos);
    vel = simd_add(vel, simd_scale(simd_add(a0, a1), 0.5f * dt));
}

// ---------- 16. SIMD Leapfrog ----------
inline void simd_leapfrog_step(SimdVec& pos, SimdVec& vel,
                                const std::function<SimdVec(SimdVec)>& accel, float dt) {
    SimdVec a0 = accel(pos);
    vel = simd_add(vel, simd_scale(a0, 0.5f * dt));
    pos = simd_add(pos, simd_scale(vel, dt));
    SimdVec a1 = accel(pos);
    vel = simd_add(vel, simd_scale(a1, 0.5f * dt));
}

} // namespace ode
} // namespace SimulationMath

#endif // CORE_MATH_ODE_INTEGRATORS_H