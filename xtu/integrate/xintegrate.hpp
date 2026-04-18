// include/xtu/integrate/xintegrate.hpp
// xtensor-unified - Numerical integration and ODE solvers
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_INTEGRATE_XINTEGRATE_HPP
#define XTU_INTEGRATE_XINTEGRATE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/containers/xtensor.hpp"

XTU_NAMESPACE_BEGIN
namespace integrate {

// #############################################################################
// Numerical integration of sampled data (1D)
// #############################################################################

/// Trapezoidal rule for uniformly spaced data
template <class E>
auto trapz(const xexpression<E>& y, double dx = 1.0) {
    const auto& arr = y.derived_cast();
    XTU_ASSERT_MSG(arr.dimension() == 1, "trapz requires 1D array");
    using value_type = typename E::value_type;
    size_t n = arr.size();
    if (n < 2) return value_type(0);
    value_type sum = (arr[0] + arr[n - 1]) / value_type(2);
    for (size_t i = 1; i < n - 1; ++i) {
        sum += arr[i];
    }
    return sum * static_cast<value_type>(dx);
}

/// Trapezoidal rule with explicit x coordinates
template <class E1, class E2>
auto trapz(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& x_arr = x.derived_cast();
    const auto& y_arr = y.derived_cast();
    XTU_ASSERT_MSG(x_arr.dimension() == 1 && y_arr.dimension() == 1, "trapz requires 1D arrays");
    XTU_ASSERT_MSG(x_arr.size() == y_arr.size(), "x and y must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    size_t n = x_arr.size();
    if (n < 2) return value_type(0);
    value_type sum = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        sum += (static_cast<value_type>(y_arr[i]) + static_cast<value_type>(y_arr[i + 1])) 
               * (static_cast<value_type>(x_arr[i + 1]) - static_cast<value_type>(x_arr[i]));
    }
    return sum / value_type(2);
}

/// Simpson's rule for uniformly spaced data (n must be odd, or uses Simpson's 3/8 for last interval)
template <class E>
auto simpson(const xexpression<E>& y, double dx = 1.0) {
    const auto& arr = y.derived_cast();
    XTU_ASSERT_MSG(arr.dimension() == 1, "simpson requires 1D array");
    using value_type = typename E::value_type;
    size_t n = arr.size();
    if (n < 3) return trapz(y, dx);
    value_type sum = arr[0] + arr[n - 1];
    for (size_t i = 1; i < n - 1; i += 2) {
        sum += value_type(4) * arr[i];
    }
    for (size_t i = 2; i < n - 2; i += 2) {
        sum += value_type(2) * arr[i];
    }
    // If even number of intervals, use Simpson's 3/8 for last three intervals
    if ((n - 1) % 2 == 0) {
        return sum * dx / value_type(3);
    } else {
        // Add 3/8 rule for last slice
        sum -= arr[n - 2] + arr[n - 1];
        sum += value_type(3) * (arr[n - 4] + arr[n - 3]) / value_type(8);
        return sum * dx / value_type(3);
    }
}

/// Simpson's rule with explicit x coordinates
template <class E1, class E2>
auto simpson(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& x_arr = x.derived_cast();
    const auto& y_arr = y.derived_cast();
    XTU_ASSERT_MSG(x_arr.dimension() == 1 && y_arr.dimension() == 1, "simpson requires 1D arrays");
    XTU_ASSERT_MSG(x_arr.size() == y_arr.size(), "x and y must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    size_t n = x_arr.size();
    if (n < 3) return trapz(x, y);
    value_type h = (static_cast<value_type>(x_arr[n - 1]) - static_cast<value_type>(x_arr[0])) / static_cast<value_type>(n - 1);
    // Check if spacing is uniform
    bool uniform = true;
    for (size_t i = 0; i < n - 1; ++i) {
        value_type diff = static_cast<value_type>(x_arr[i + 1]) - static_cast<value_type>(x_arr[i]);
        if (std::abs(diff - h) > value_type(1e-12) * std::abs(h)) {
            uniform = false;
            break;
        }
    }
    if (uniform) {
        return simpson(y, static_cast<double>(h));
    }
    // Non-uniform Simpson
    value_type sum = 0;
    for (size_t i = 0; i < n - 2; i += 2) {
        value_type h0 = static_cast<value_type>(x_arr[i + 1]) - static_cast<value_type>(x_arr[i]);
        value_type h1 = static_cast<value_type>(x_arr[i + 2]) - static_cast<value_type>(x_arr[i + 1]);
        value_type a = (value_type(2) * h0 * h0 + h0 * h1 - h1 * h1) / (value_type(6) * h0 * (h0 + h1));
        value_type b = (h0 + h1) * (h0 + h1) * (h0 + h1) / (value_type(6) * h0 * h1);
        value_type c = (value_type(2) * h1 * h1 + h0 * h1 - h0 * h0) / (value_type(6) * h1 * (h0 + h1));
        sum += a * static_cast<value_type>(y_arr[i]) 
             + b * static_cast<value_type>(y_arr[i + 1]) 
             + c * static_cast<value_type>(y_arr[i + 2]);
    }
    return sum;
}

// #############################################################################
// Romberg integration (adaptive extrapolation)
// #############################################################################
template <class Func>
double romberg(const Func& f, double a, double b, double tol = 1e-6, size_t max_steps = 20) {
    std::vector<std::vector<double>> R(max_steps);
    double h = b - a;
    R[0].resize(1);
    R[0][0] = 0.5 * h * (f(a) + f(b));
    
    for (size_t n = 1; n < max_steps; ++n) {
        R[n].resize(n + 1);
        // Composite trapezoidal with 2^n intervals
        h *= 0.5;
        double sum = 0.0;
        size_t intervals = static_cast<size_t>(1) << (n - 1);
        for (size_t k = 1; k <= intervals; ++k) {
            sum += f(a + (2.0 * k - 1.0) * h);
        }
        R[n][0] = 0.5 * R[n - 1][0] + h * sum;
        
        // Richardson extrapolation
        for (size_t m = 1; m <= n; ++m) {
            R[n][m] = R[n][m - 1] + (R[n][m - 1] - R[n - 1][m - 1]) / (std::pow(4.0, static_cast<int>(m)) - 1.0);
        }
        if (n > 1 && std::abs(R[n][n] - R[n - 1][n - 1]) < tol) {
            return R[n][n];
        }
    }
    return R[max_steps - 1][max_steps - 1];
}

// #############################################################################
// Gaussian quadrature (fixed order)
// #############################################################################

/// Gauss-Legendre quadrature weights and nodes (order 5 as default)
inline std::pair<std::vector<double>, std::vector<double>> gauss_legendre_nodes_weights(size_t n = 5) {
    // Precomputed for n=5
    if (n == 5) {
        std::vector<double> nodes = {
            -0.9061798459386640, -0.5384693101056831, 0.0,
            0.5384693101056831, 0.9061798459386640
        };
        std::vector<double> weights = {
            0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
            0.4786286704993665, 0.2369268850561891
        };
        return {nodes, weights};
    }
    // For other orders, use Golub-Welsch (tridiagonal eigenvalue problem)
    // Simplified: precompute up to order 10
    XTU_THROW(std::runtime_error, "Only order 5 is precomputed; higher orders require eigenvalue computation");
}

template <class Func>
double gauss_quadrature(const Func& f, double a, double b, size_t order = 5) {
    auto [nodes, weights] = gauss_legendre_nodes_weights(order);
    double mid = (b + a) / 2.0;
    double half_len = (b - a) / 2.0;
    double sum = 0.0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        double x = mid + half_len * nodes[i];
        sum += weights[i] * f(x);
    }
    return half_len * sum;
}

// #############################################################################
// ODE Solvers
// #############################################################################

/// Euler method (explicit)
template <class Func>
std::pair<std::vector<double>, std::vector<std::vector<double>>>
euler(const Func& f, double t0, const std::vector<double>& y0, double t_end, double dt) {
    size_t n_eq = y0.size();
    size_t steps = static_cast<size_t>(std::ceil((t_end - t0) / dt)) + 1;
    std::vector<double> t(steps);
    std::vector<std::vector<double>> y(steps, std::vector<double>(n_eq));
    t[0] = t0;
    y[0] = y0;
    for (size_t i = 0; i < steps - 1; ++i) {
        t[i + 1] = t[i] + dt;
        std::vector<double> dydt = f(t[i], y[i]);
        for (size_t j = 0; j < n_eq; ++j) {
            y[i + 1][j] = y[i][j] + dt * dydt[j];
        }
    }
    return {t, y};
}

/// Runge-Kutta 4th order (classic RK4)
template <class Func>
std::pair<std::vector<double>, std::vector<std::vector<double>>>
rk4(const Func& f, double t0, const std::vector<double>& y0, double t_end, double dt) {
    size_t n_eq = y0.size();
    size_t steps = static_cast<size_t>(std::ceil((t_end - t0) / dt)) + 1;
    std::vector<double> t(steps);
    std::vector<std::vector<double>> y(steps, std::vector<double>(n_eq));
    t[0] = t0;
    y[0] = y0;
    for (size_t i = 0; i < steps - 1; ++i) {
        double ti = t[i];
        const auto& yi = y[i];
        double h = dt;
        std::vector<double> k1 = f(ti, yi);
        std::vector<double> y_temp(n_eq);
        for (size_t j = 0; j < n_eq; ++j) y_temp[j] = yi[j] + 0.5 * h * k1[j];
        std::vector<double> k2 = f(ti + 0.5 * h, y_temp);
        for (size_t j = 0; j < n_eq; ++j) y_temp[j] = yi[j] + 0.5 * h * k2[j];
        std::vector<double> k3 = f(ti + 0.5 * h, y_temp);
        for (size_t j = 0; j < n_eq; ++j) y_temp[j] = yi[j] + h * k3[j];
        std::vector<double> k4 = f(ti + h, y_temp);
        t[i + 1] = ti + h;
        for (size_t j = 0; j < n_eq; ++j) {
            y[i + 1][j] = yi[j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
    }
    return {t, y};
}

/// Adaptive Runge-Kutta-Fehlberg 4(5)
template <class Func>
std::pair<std::vector<double>, std::vector<std::vector<double>>>
rk45(const Func& f, double t0, const std::vector<double>& y0, double t_end,
     double dt_initial = 0.01, double tol = 1e-6, double dt_min = 1e-8, double dt_max = 0.1) {
    size_t n_eq = y0.size();
    std::vector<double> t;
    std::vector<std::vector<double>> y;
    t.push_back(t0);
    y.push_back(y0);
    double dt = dt_initial;
    double t_curr = t0;
    std::vector<double> y_curr = y0;
    
    // RK45 Butcher tableau coefficients
    const double a2 = 1.0/4.0;
    const double a3 = 3.0/8.0;
    const double a4 = 12.0/13.0;
    const double a5 = 1.0;
    const double a6 = 1.0/2.0;
    
    const double b21 = 1.0/4.0;
    const double b31 = 3.0/32.0, b32 = 9.0/32.0;
    const double b41 = 1932.0/2197.0, b42 = -7200.0/2197.0, b43 = 7296.0/2197.0;
    const double b51 = 439.0/216.0, b52 = -8.0, b53 = 3680.0/513.0, b54 = -845.0/4104.0;
    const double b61 = -8.0/27.0, b62 = 2.0, b63 = -3544.0/2565.0, b64 = 1859.0/4104.0, b65 = -11.0/40.0;
    
    // 4th order weights
    const double c1 = 25.0/216.0, c3 = 1408.0/2565.0, c4 = 2197.0/4104.0, c5 = -1.0/5.0;
    // 5th order weights
    const double d1 = 16.0/135.0, d3 = 6656.0/12825.0, d4 = 28561.0/56430.0, d5 = -9.0/50.0, d6 = 2.0/55.0;
    
    while (t_curr < t_end) {
        if (t_curr + dt > t_end) dt = t_end - t_curr;
        
        std::vector<double> k1 = f(t_curr, y_curr);
        std::vector<double> y2(n_eq);
        for (size_t i = 0; i < n_eq; ++i) y2[i] = y_curr[i] + dt * b21 * k1[i];
        std::vector<double> k2 = f(t_curr + a2 * dt, y2);
        std::vector<double> y3(n_eq);
        for (size_t i = 0; i < n_eq; ++i) y3[i] = y_curr[i] + dt * (b31 * k1[i] + b32 * k2[i]);
        std::vector<double> k3 = f(t_curr + a3 * dt, y3);
        std::vector<double> y4(n_eq);
        for (size_t i = 0; i < n_eq; ++i) y4[i] = y_curr[i] + dt * (b41 * k1[i] + b42 * k2[i] + b43 * k3[i]);
        std::vector<double> k4 = f(t_curr + a4 * dt, y4);
        std::vector<double> y5(n_eq);
        for (size_t i = 0; i < n_eq; ++i) y5[i] = y_curr[i] + dt * (b51 * k1[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i]);
        std::vector<double> k5 = f(t_curr + a5 * dt, y5);
        std::vector<double> y6(n_eq);
        for (size_t i = 0; i < n_eq; ++i) y6[i] = y_curr[i] + dt * (b61 * k1[i] + b62 * k2[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i]);
        std::vector<double> k6 = f(t_curr + a6 * dt, y6);
        
        // 4th and 5th order estimates
        std::vector<double> y4_est(n_eq), y5_est(n_eq);
        for (size_t i = 0; i < n_eq; ++i) {
            y4_est[i] = y_curr[i] + dt * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i]);
            y5_est[i] = y_curr[i] + dt * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i]);
        }
        
        // Error estimate
        double err = 0.0;
        for (size_t i = 0; i < n_eq; ++i) {
            double diff = std::abs(y4_est[i] - y5_est[i]);
            if (diff > err) err = diff;
        }
        
        if (err <= tol) {
            t_curr += dt;
            t.push_back(t_curr);
            y.push_back(y5_est);
            y_curr = y5_est;
        }
        
        // Adapt step size
        if (err > 0) {
            double factor = 0.84 * std::pow(tol / err, 0.25);
            dt = std::min(std::max(factor * dt, dt_min), dt_max);
        } else {
            dt = std::min(dt * 2.0, dt_max);
        }
    }
    return {t, y};
}

// #############################################################################
// ODE solver for systems returning xarray containers
// #############################################################################
template <class Func>
auto solve_ivp(const Func& f, double t_span_begin, double t_span_end,
               const std::vector<double>& y0, const std::string& method = "rk45",
               double dt = 0.01, double tol = 1e-6) {
    std::pair<std::vector<double>, std::vector<std::vector<double>>> result;
    if (method == "euler") {
        result = euler(f, t_span_begin, y0, t_span_end, dt);
    } else if (method == "rk4") {
        result = rk4(f, t_span_begin, y0, t_span_end, dt);
    } else if (method == "rk45") {
        result = rk45(f, t_span_begin, y0, t_span_end, dt, tol);
    } else {
        XTU_THROW(std::invalid_argument, "Unknown ODE method: " + method);
    }
    size_t n_steps = result.first.size();
    size_t n_eq = y0.size();
    xarray_container<double> t_arr({n_steps});
    xarray_container<double> y_arr({n_steps, n_eq});
    for (size_t i = 0; i < n_steps; ++i) {
        t_arr[i] = result.first[i];
        for (size_t j = 0; j < n_eq; ++j) {
            y_arr(i, j) = result.second[i][j];
        }
    }
    return std::make_pair(std::move(t_arr), std::move(y_arr));
}

// #############################################################################
// Cumulative integration
// #############################################################################
template <class E>
auto cumtrapz(const xexpression<E>& y, double dx = 1.0) {
    const auto& arr = y.derived_cast();
    XTU_ASSERT_MSG(arr.dimension() == 1, "cumtrapz requires 1D array");
    using value_type = typename E::value_type;
    size_t n = arr.size();
    xarray_container<value_type> result({n});
    if (n == 0) return result;
    result[0] = 0;
    value_type accum = 0;
    for (size_t i = 1; i < n; ++i) {
        accum += (arr[i - 1] + arr[i]) * static_cast<value_type>(dx) / value_type(2);
        result[i] = accum;
    }
    return result;
}

template <class E1, class E2>
auto cumtrapz(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& x_arr = x.derived_cast();
    const auto& y_arr = y.derived_cast();
    XTU_ASSERT_MSG(x_arr.dimension() == 1 && y_arr.dimension() == 1, "cumtrapz requires 1D arrays");
    XTU_ASSERT_MSG(x_arr.size() == y_arr.size(), "x and y must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    size_t n = x_arr.size();
    xarray_container<value_type> result({n});
    if (n == 0) return result;
    result[0] = 0;
    value_type accum = 0;
    for (size_t i = 1; i < n; ++i) {
        value_type dx = static_cast<value_type>(x_arr[i]) - static_cast<value_type>(x_arr[i - 1]);
        accum += (static_cast<value_type>(y_arr[i - 1]) + static_cast<value_type>(y_arr[i])) * dx / value_type(2);
        result[i] = accum;
    }
    return result;
}

} // namespace integrate

// Bring into main namespace for convenience
using integrate::trapz;
using integrate::simpson;
using integrate::romberg;
using integrate::gauss_quadrature;
using integrate::euler;
using integrate::rk4;
using integrate::rk45;
using integrate::solve_ivp;
using integrate::cumtrapz;

XTU_NAMESPACE_END

#endif // XTU_INTEGRATE_XINTEGRATE_HPP