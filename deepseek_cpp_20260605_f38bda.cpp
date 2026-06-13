//File 0031 : core/xode.hpp
//ODE solvers: explicit Runge-Kutta (RK4, adaptive RK45), Adams-Bashforth multistep, symplectic integrators for simulation with SIMD state updates.
#ifndef XTENSOR_XODE_HPP
#define XTENSOR_XODE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xeval.hpp"
#include "xreducer.hpp"
#include "xlinalg.hpp"
#include "xstatistics.hpp"
#include "xinterpolate.hpp"

namespace xt {
namespace ode {

    using state_type = xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
    using ode_function = std::function<state_type(double, const state_type&)>;

    /*********************************************
     * RK4 (Fixed-step, 4th order)
     *********************************************/
    /**
     * Perform one step of classic 4th-order Runge-Kutta.
     */
    inline void rk4_step(ode_function f, double t, const state_type& y, double dt, state_type& y_next) {
        auto k1 = f(t, y);
        auto k2 = f(t + 0.5*dt, y + 0.5*dt * k1);
        auto k3 = f(t + 0.5*dt, y + 0.5*dt * k2);
        auto k4 = f(t + dt, y + dt * k3);
        y_next = y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
    }

    /**
     * Integrate ODE using fixed-step RK4 over a time span, returning trajectory.
     */
    inline std::tuple<std::vector<double>, std::vector<state_type>>
    rk4_integrate(ode_function f, const state_type& y0, double t0, double t_end, double dt) {
        std::vector<double> times;
        std::vector<state_type> states;
        double t = t0;
        state_type y = y0;
        times.push_back(t);
        states.push_back(y);
        while (t < t_end) {
            double h = std::min(dt, t_end - t);
            state_type y_next(y0.shape());
            rk4_step(f, t, y, h, y_next);
            t += h;
            y = std::move(y_next);
            times.push_back(t);
            states.push_back(y);
        }
        return {times, states};
    }

    /*********************************************
     * RK45 (Dormand-Prince adaptive step)
     *********************************************/
    namespace detail {
        // Butcher tableau for Dormand-Prince 5(4)
        constexpr double rk45_a[7][7] = {
            {0,0,0,0,0,0,0},
            {1.0/5,0,0,0,0,0,0},
            {3.0/40,9.0/40,0,0,0,0,0},
            {44.0/45,-56.0/15,32.0/9,0,0,0,0},
            {19372.0/6561,-25360.0/2187,64448.0/6561,-212.0/729,0,0,0},
            {9017.0/3168,-355.0/33,46732.0/5247,49.0/176,-5103.0/18656,0,0},
            {35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0}
        };
        constexpr double rk45_b5[7] = {35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0};
        constexpr double rk45_b4[7] = {5179.0/57600,0,7571.0/16695,393.0/640,-92097.0/339200,187.0/2100,1.0/40};
    }

    /**
     * Perform one adaptive RK45 step; returns new y and suggested next step size.
     */
    inline std::pair<state_type, double>
    rk45_step(ode_function f, double t, const state_type& y, double dt, double tol = 1e-12) {
        constexpr int s = 7;
        state_type k[s];
        for (int i = 0; i < s; ++i) {
            state_type sum = y;
            for (int j = 0; j < i; ++j)
                sum = sum + dt * detail::rk45_a[i][j] * k[j];
            k[i] = f(t + (i==0?0.0: (i==1?1.0/5: i==2?3.0/10: i==3?4.0/5: i==4?8.0/9: 1.0)), sum);
        }
        state_type y5 = y;
        state_type y4 = y;
        for (int j = 0; j < s; ++j) {
            y5 = y5 + dt * detail::rk45_b5[j] * k[j];
            y4 = y4 + dt * detail::rk45_b4[j] * k[j];
        }
        double error = xt::norm::norm_linf(y5 - y4)();
        double safety = 0.9;
        double dt_new = dt * safety * std::pow(tol / (error + 1e-15), 1.0/5.0);
        return {y5, dt_new};
    }

    /**
     * Integrate ODE using adaptive RK45 over time span, returning trajectory at given output times.
     */
    inline std::tuple<std::vector<double>, std::vector<state_type>>
    rk45_integrate(ode_function f, const state_type& y0, double t0, double t_end,
                   double dt_initial = 1e-3, double tol = 1e-9,
                   const std::vector<double>& output_times = {}) {
        std::vector<double> times;
        std::vector<state_type> states;
        double t = t0;
        state_type y = y0;
        double dt = dt_initial;
        times.push_back(t);
        states.push_back(y);
        std::size_t out_idx = 0;
        while (t < t_end) {
            if (!output_times.empty() && out_idx < output_times.size() && t >= output_times[out_idx]) {
                // Interpolate? We'll just use dense output via cubic Hermite (simplified: take current)
                times.push_back(t);
                states.push_back(y);
                ++out_idx;
            }
            double h = std::min(dt, t_end - t);
            auto [y_new, dt_new] = rk45_step(f, t, y, h, tol);
            t += h;
            y = std::move(y_new);
            dt = std::min(dt_new, t_end - t);
            times.push_back(t);
            states.push_back(y);
        }
        return {times, states};
    }

    /*********************************************
     * Adams-Bashforth (4-step, fixed-step)
     *********************************************/
    /**
     * Adams-Bashforth 4-step method: requires starting values from RK4.
     */
    inline void adams_bashforth_step(ode_function f, double t, const std::vector<state_type>& y_history,
                                      double dt, state_type& y_next) {
        // y_history[0] = y_n-3, [1] = y_n-2, [2] = y_n-1, [3] = y_n
        const auto& fn3 = f(t - 3*dt, y_history[0]);
        const auto& fn2 = f(t - 2*dt, y_history[1]);
        const auto& fn1 = f(t - dt, y_history[2]);
        const auto& fn  = f(t, y_history[3]);
        y_next = y_history[3] + (dt/24.0) * (55.0*fn - 59.0*fn1 + 37.0*fn2 - 9.0*fn3);
    }

    /**
     * Integrate using Adams-Bashforth 4-step with RK4 initialization.
     */
    inline std::tuple<std::vector<double>, std::vector<state_type>>
    adams_bashforth_integrate(ode_function f, const state_type& y0, double t0, double t_end, double dt) {
        std::vector<double> times;
        std::vector<state_type> states;
        double t = t0;
        state_type y = y0;
        // generate first 4 steps with RK4
        for (int i = 0; i < 4 && t < t_end; ++i) {
            double h = std::min(dt, t_end - t);
            state_type y_next(y0.shape());
            rk4_step(f, t, y, h, y_next);
            t += h;
            y = std::move(y_next);
            times.push_back(t);
            states.push_back(y);
        }
        while (t < t_end) {
            double h = std::min(dt, t_end - t);
            state_type y_next(y0.shape());
            adams_bashforth_step(f, t, &states[states.size()-4], h, y_next);
            t += h;
            y = std::move(y_next);
            times.push_back(t);
            states.push_back(y);
        }
        return {times, states};
    }

    /*********************************************
     * Symplectic Integrators (Velocity Verlet)
     *********************************************/
    /**
     * Velocity Verlet step for second-order ODE y'' = acc(t,y).
     */
    template <class AccFunc>
    inline void velocity_verlet_step(AccFunc acc, double t, state_type& pos, state_type& vel, double dt) {
        auto a0 = acc(t, pos);
        pos = pos + dt * vel + 0.5 * dt * dt * a0;
        auto a1 = acc(t + dt, pos);
        vel = vel + 0.5 * dt * (a0 + a1);
    }

    /**
     * Integrate with Velocity Verlet over time, recording positions and velocities.
     */
    template <class AccFunc>
    inline auto velocity_verlet_integrate(AccFunc acc, const state_type& pos0, const state_type& vel0,
                                          double t0, double t_end, double dt) {
        std::vector<double> times;
        std::vector<state_type> positions, velocities;
        double t = t0;
        state_type pos = pos0, vel = vel0;
        times.push_back(t);
        positions.push_back(pos);
        velocities.push_back(vel);
        while (t < t_end) {
            double h = std::min(dt, t_end - t);
            velocity_verlet_step(acc, t, pos, vel, h);
            t += h;
            times.push_back(t);
            positions.push_back(pos);
            velocities.push_back(vel);
        }
        return std::make_tuple(times, positions, velocities);
    }

    /*********************************************
     * Utility: Generate time grid
     *********************************************/
    /**
     * Generate a vector of evenly spaced time points.
     */
    inline std::vector<double> time_grid(double t0, double t_end, double dt) {
        std::vector<double> times;
        double t = t0;
        while (t <= t_end + 1e-14) {
            times.push_back(t);
            t += dt;
        }
        return times;
    }

} // namespace ode
} // namespace xt

#endif // XTENSOR_XODE_HPP