// include/xtu/interp/xinterp.hpp
// xtensor-unified - Interpolation routines (linear, nearest, cubic, spline, polynomial)
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_INTERP_XINTERP_HPP
#define XTU_INTERP_XINTERP_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/containers/xtensor.hpp"
#include "xtu/math/xsorting.hpp"
#include "xtu/math/xlinalg.hpp"

XTU_NAMESPACE_BEGIN
namespace interp {

// #############################################################################
// Interpolation method types
// #############################################################################
enum class interp_method {
    nearest,
    linear,
    quadratic,
    cubic,
    spline,
    akima,
    pchip
};

enum class extrapolate_mode {
    constant,   // use boundary value
    linear,     // linear extrapolation
    nearest,    // nearest boundary value
    nan,        // return NaN
    error       // throw exception
};

// #############################################################################
// 1D Interpolation base functions
// #############################################################################

/// Find interval index for a given x value (assuming x is sorted ascending)
template <class T>
size_t find_interval(const std::vector<T>& x, T xi, bool extrapolate = false) {
    if (xi < x.front()) {
        if (extrapolate) return 0;
        XTU_THROW(std::out_of_range, "Value below interpolation range");
    }
    if (xi > x.back()) {
        if (extrapolate) return x.size() - 2;
        XTU_THROW(std::out_of_range, "Value above interpolation range");
    }
    auto it = std::upper_bound(x.begin(), x.end(), xi);
    if (it == x.begin()) return 0;
    size_t idx = static_cast<size_t>(std::distance(x.begin(), it) - 1);
    return std::min(idx, x.size() - 2);
}

/// Nearest-neighbor interpolation (1D)
template <class T>
T interp_nearest(const std::vector<T>& x, const std::vector<T>& y, T xi, 
                 extrapolate_mode extrap = extrapolate_mode::constant) {
    XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
    XTU_ASSERT_MSG(x.size() >= 1, "At least one point required");
    if (x.size() == 1) return y[0];
    
    if (xi <= x.front()) {
        switch (extrap) {
            case extrapolate_mode::constant: return y.front();
            case extrapolate_mode::nearest: return y.front();
            case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
            default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
        }
    }
    if (xi >= x.back()) {
        switch (extrap) {
            case extrapolate_mode::constant: return y.back();
            case extrapolate_mode::nearest: return y.back();
            case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
            default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
        }
    }
    
    size_t idx = 0;
    T min_dist = std::abs(xi - x[0]);
    for (size_t i = 1; i < x.size(); ++i) {
        T dist = std::abs(xi - x[i]);
        if (dist < min_dist) {
            min_dist = dist;
            idx = i;
        }
    }
    return y[idx];
}

/// Linear interpolation (1D)
template <class T>
T interp_linear(const std::vector<T>& x, const std::vector<T>& y, T xi,
                extrapolate_mode extrap = extrapolate_mode::linear) {
    XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
    XTU_ASSERT_MSG(x.size() >= 2, "At least two points required for linear interpolation");
    
    if (xi < x.front()) {
        switch (extrap) {
            case extrapolate_mode::constant: return y.front();
            case extrapolate_mode::linear: {
                T slope = (y[1] - y[0]) / (x[1] - x[0]);
                return y[0] + slope * (xi - x[0]);
            }
            case extrapolate_mode::nearest: return y[0];
            case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
            default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
        }
    }
    if (xi > x.back()) {
        size_t n = x.size();
        switch (extrap) {
            case extrapolate_mode::constant: return y.back();
            case extrapolate_mode::linear: {
                T slope = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]);
                return y[n-1] + slope * (xi - x[n-1]);
            }
            case extrapolate_mode::nearest: return y.back();
            case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
            default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
        }
    }
    
    size_t idx = find_interval(x, xi, false);
    T t = (xi - x[idx]) / (x[idx + 1] - x[idx]);
    return y[idx] * (1 - t) + y[idx + 1] * t;
}

/// Quadratic interpolation (1D, using three points)
template <class T>
T interp_quadratic(const std::vector<T>& x, const std::vector<T>& y, T xi,
                   extrapolate_mode extrap = extrapolate_mode::constant) {
    XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
    XTU_ASSERT_MSG(x.size() >= 3, "At least three points required for quadratic interpolation");
    
    if (xi < x.front() || xi > x.back()) {
        switch (extrap) {
            case extrapolate_mode::constant: {
                if (xi < x.front()) return y.front();
                return y.back();
            }
            case extrapolate_mode::nearest: {
                if (xi < x.front()) return y.front();
                return y.back();
            }
            case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
            default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
        }
    }
    
    size_t idx = find_interval(x, xi, false);
    // Use idx-1, idx, idx+1 if possible, else adjust
    size_t i0 = (idx > 0) ? idx - 1 : idx;
    size_t i1 = idx;
    size_t i2 = (idx + 1 < x.size() - 1) ? idx + 1 : idx;
    if (i0 == i1) i0 = i1;
    if (i2 == i1) i2 = i1;
    
    // Lagrange quadratic interpolation
    T x0 = x[i0], x1 = x[i1], x2 = x[i2];
    T y0 = y[i0], y1 = y[i1], y2 = y[i2];
    T result = y0 * ((xi - x1) * (xi - x2)) / ((x0 - x1) * (x0 - x2))
             + y1 * ((xi - x0) * (xi - x2)) / ((x1 - x0) * (x1 - x2))
             + y2 * ((xi - x0) * (xi - x1)) / ((x2 - x0) * (x2 - x1));
    return result;
}

/// Cubic interpolation (1D, using four points)
template <class T>
T interp_cubic(const std::vector<T>& x, const std::vector<T>& y, T xi,
               extrapolate_mode extrap = extrapolate_mode::constant) {
    XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
    XTU_ASSERT_MSG(x.size() >= 4, "At least four points required for cubic interpolation");
    
    if (xi < x.front() || xi > x.back()) {
        switch (extrap) {
            case extrapolate_mode::constant: {
                if (xi < x.front()) return y.front();
                return y.back();
            }
            case extrapolate_mode::nearest: {
                if (xi < x.front()) return y.front();
                return y.back();
            }
            case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
            default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
        }
    }
    
    size_t idx = find_interval(x, xi, false);
    size_t i1 = idx;
    size_t i0 = (i1 > 0) ? i1 - 1 : i1;
    size_t i2 = i1 + 1;
    size_t i3 = (i2 + 1 < x.size()) ? i2 + 1 : i2;
    
    T x0 = x[i0], x1 = x[i1], x2 = x[i2], x3 = x[i3];
    T y0 = y[i0], y1 = y[i1], y2 = y[i2], y3 = y[i3];
    
    // Lagrange cubic interpolation
    T result = y0 * ((xi - x1) * (xi - x2) * (xi - x3)) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
             + y1 * ((xi - x0) * (xi - x2) * (xi - x3)) / ((x1 - x0) * (x1 - x2) * (x1 - x3))
             + y2 * ((xi - x0) * (xi - x1) * (xi - x3)) / ((x2 - x0) * (x2 - x1) * (x2 - x3))
             + y3 * ((xi - x0) * (xi - x1) * (xi - x2)) / ((x3 - x0) * (x3 - x1) * (x3 - x2));
    return result;
}

// #############################################################################
// Cubic Spline Interpolation (natural spline)
// #############################################################################
template <class T>
class cubic_spline {
private:
    std::vector<T> m_x, m_y, m_a, m_b, m_c, m_d;
    size_t m_n;
    
public:
    cubic_spline() = default;
    
    cubic_spline(const std::vector<T>& x, const std::vector<T>& y) {
        fit(x, y);
    }
    
    void fit(const std::vector<T>& x, const std::vector<T>& y) {
        XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
        XTU_ASSERT_MSG(x.size() >= 3, "At least three points required for cubic spline");
        m_n = x.size();
        m_x = x;
        m_y = y;
        m_a.resize(m_n);
        m_b.resize(m_n - 1);
        m_c.resize(m_n);
        m_d.resize(m_n - 1);
        
        // Natural spline boundary conditions: second derivative = 0 at ends
        std::vector<T> h(m_n - 1);
        for (size_t i = 0; i < m_n - 1; ++i) {
            h[i] = x[i + 1] - x[i];
            m_a[i] = y[i];
        }
        m_a[m_n - 1] = y[m_n - 1];
        
        // Set up tridiagonal system
        std::vector<T> alpha(m_n - 1);
        for (size_t i = 1; i < m_n - 1; ++i) {
            alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
        }
        
        std::vector<T> l(m_n), mu(m_n), z(m_n);
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;
        
        for (size_t i = 1; i < m_n - 1; ++i) {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        l[m_n - 1] = 1.0;
        z[m_n - 1] = 0.0;
        m_c[m_n - 1] = 0.0;
        
        for (int i = static_cast<int>(m_n) - 2; i >= 0; --i) {
            m_c[static_cast<size_t>(i)] = z[static_cast<size_t>(i)] - mu[static_cast<size_t>(i)] * m_c[static_cast<size_t>(i) + 1];
            m_b[static_cast<size_t>(i)] = (y[i + 1] - y[i]) / h[i] - h[i] * (m_c[i + 1] + 2.0 * m_c[i]) / 3.0;
            m_d[static_cast<size_t>(i)] = (m_c[i + 1] - m_c[i]) / (3.0 * h[i]);
        }
    }
    
    T operator()(T xi, extrapolate_mode extrap = extrapolate_mode::constant) const {
        if (xi < m_x.front()) {
            switch (extrap) {
                case extrapolate_mode::constant: return m_y.front();
                case extrapolate_mode::linear: {
                    T slope = m_b[0];
                    return m_y[0] + slope * (xi - m_x[0]);
                }
                case extrapolate_mode::nearest: return m_y.front();
                case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
                default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
            }
        }
        if (xi > m_x.back()) {
            switch (extrap) {
                case extrapolate_mode::constant: return m_y.back();
                case extrapolate_mode::linear: {
                    T slope = m_b.back() + 2.0 * m_c.back() * (m_x.back() - m_x[m_n-2]) 
                              + 3.0 * m_d.back() * std::pow(m_x.back() - m_x[m_n-2], 2);
                    return m_y.back() + slope * (xi - m_x.back());
                }
                case extrapolate_mode::nearest: return m_y.back();
                case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
                default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
            }
        }
        
        size_t idx = find_interval(m_x, xi, false);
        T dx = xi - m_x[idx];
        return m_a[idx] + m_b[idx] * dx + m_c[idx] * dx * dx + m_d[idx] * dx * dx * dx;
    }
    
    const std::vector<T>& x() const { return m_x; }
    const std::vector<T>& y() const { return m_y; }
};

// #############################################################################
// Vectorized interpolation (apply to array of xi)
// #############################################################################
template <class T>
xarray_container<T> interp1d(const std::vector<T>& x, const std::vector<T>& y,
                              const xarray_container<T>& xi, interp_method method = interp_method::linear,
                              extrapolate_mode extrap = extrapolate_mode::constant) {
    xarray_container<T> result(xi.shape());
    switch (method) {
        case interp_method::nearest:
            for (size_t i = 0; i < xi.size(); ++i) {
                result.flat(i) = interp_nearest(x, y, xi.flat(i), extrap);
            }
            break;
        case interp_method::linear:
            for (size_t i = 0; i < xi.size(); ++i) {
                result.flat(i) = interp_linear(x, y, xi.flat(i), extrap);
            }
            break;
        case interp_method::quadratic:
            for (size_t i = 0; i < xi.size(); ++i) {
                result.flat(i) = interp_quadratic(x, y, xi.flat(i), extrap);
            }
            break;
        case interp_method::cubic:
            for (size_t i = 0; i < xi.size(); ++i) {
                result.flat(i) = interp_cubic(x, y, xi.flat(i), extrap);
            }
            break;
        case interp_method::spline: {
            cubic_spline<T> spline(x, y);
            for (size_t i = 0; i < xi.size(); ++i) {
                result.flat(i) = spline(xi.flat(i), extrap);
            }
            break;
        }
        default:
            XTU_THROW(std::invalid_argument, "Unsupported interpolation method");
    }
    return result;
}

// #############################################################################
// Gridded 2D interpolation (bilinear)
// #############################################################################
template <class T>
T interp_bilinear(const std::vector<T>& x, const std::vector<T>& y,
                  const xarray_container<T>& z, T xi, T yi,
                  extrapolate_mode extrap = extrapolate_mode::constant) {
    XTU_ASSERT_MSG(z.dimension() == 2, "z must be 2D");
    XTU_ASSERT_MSG(x.size() == z.shape()[1], "x size must match z columns");
    XTU_ASSERT_MSG(y.size() == z.shape()[0], "y size must match z rows");
    
    size_t nx = x.size();
    size_t ny = y.size();
    
    size_t ix = find_interval(x, xi, (extrap != extrapolate_mode::error));
    size_t iy = find_interval(y, yi, (extrap != extrapolate_mode::error));
    
    if (xi < x.front() || xi > x.back() || yi < y.front() || yi > y.back()) {
        switch (extrap) {
            case extrapolate_mode::constant: {
                if (xi < x.front() && yi < y.front()) return z(0, 0);
                if (xi < x.front() && yi > y.back()) return z(ny-1, 0);
                if (xi > x.back() && yi < y.front()) return z(0, nx-1);
                if (xi > x.back() && yi > y.back()) return z(ny-1, nx-1);
                if (xi < x.front()) return z(iy, 0);
                if (xi > x.back()) return z(iy, nx-1);
                if (yi < y.front()) return z(0, ix);
                if (yi > y.back()) return z(ny-1, ix);
                break;
            }
            case extrapolate_mode::nearest: {
                size_t ix_near = 0, iy_near = 0;
                T min_dist_x = std::abs(xi - x[0]);
                for (size_t i = 1; i < nx; ++i) {
                    T d = std::abs(xi - x[i]);
                    if (d < min_dist_x) { min_dist_x = d; ix_near = i; }
                }
                T min_dist_y = std::abs(yi - y[0]);
                for (size_t i = 1; i < ny; ++i) {
                    T d = std::abs(yi - y[i]);
                    if (d < min_dist_y) { min_dist_y = d; iy_near = i; }
                }
                return z(iy_near, ix_near);
            }
            case extrapolate_mode::nan:
                return std::numeric_limits<T>::quiet_NaN();
            default:
                XTU_THROW(std::out_of_range, "Extrapolation not allowed");
        }
    }
    
    T x1 = x[ix], x2 = x[ix + 1];
    T y1 = y[iy], y2 = y[iy + 1];
    T q11 = z(iy, ix);
    T q21 = z(iy, ix + 1);
    T q12 = z(iy + 1, ix);
    T q22 = z(iy + 1, ix + 1);
    
    T tx = (xi - x1) / (x2 - x1);
    T ty = (yi - y1) / (y2 - y1);
    
    T r1 = q11 * (1 - tx) + q21 * tx;
    T r2 = q12 * (1 - tx) + q22 * tx;
    return r1 * (1 - ty) + r2 * ty;
}

// #############################################################################
// Polynomial interpolation (Lagrange)
// #############################################################################
template <class T>
T interp_polynomial(const std::vector<T>& x, const std::vector<T>& y, T xi) {
    XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
    size_t n = x.size();
    T result = 0;
    for (size_t i = 0; i < n; ++i) {
        T term = y[i];
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                term *= (xi - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }
    return result;
}

// #############################################################################
// Akima interpolation (continuously differentiable, local)
// #############################################################################
template <class T>
class akima_spline {
private:
    std::vector<T> m_x, m_y, m_b, m_c, m_d;
    size_t m_n;
    
public:
    akima_spline() = default;
    
    akima_spline(const std::vector<T>& x, const std::vector<T>& y) {
        fit(x, y);
    }
    
    void fit(const std::vector<T>& x, const std::vector<T>& y) {
        XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
        XTU_ASSERT_MSG(x.size() >= 5, "Akima requires at least 5 points");
        m_n = x.size();
        m_x = x;
        m_y = y;
        m_b.resize(m_n - 1);
        m_c.resize(m_n - 1);
        m_d.resize(m_n - 1);
        
        std::vector<T> h(m_n - 1);
        std::vector<T> delta(m_n - 1);
        for (size_t i = 0; i < m_n - 1; ++i) {
            h[i] = x[i + 1] - x[i];
            delta[i] = (y[i + 1] - y[i]) / h[i];
        }
        
        std::vector<T> d(m_n);
        for (size_t i = 2; i < m_n - 2; ++i) {
            T w1 = std::abs(delta[i + 1] - delta[i]);
            T w2 = std::abs(delta[i - 1] - delta[i - 2]);
            if (w1 + w2 == 0) {
                d[i] = (delta[i - 1] + delta[i]) / 2.0;
            } else {
                d[i] = (w1 * delta[i - 1] + w2 * delta[i]) / (w1 + w2);
            }
        }
        // Endpoint derivatives using quadratic extrapolation
        d[0] = delta[0] + (delta[0] - delta[1]) * h[0] / (h[0] + h[1]);
        d[1] = (3.0 * delta[0] - d[0]) / 2.0;
        d[m_n - 2] = delta[m_n - 2] + (delta[m_n - 2] - delta[m_n - 3]) * h[m_n - 2] / (h[m_n - 2] + h[m_n - 3]);
        d[m_n - 1] = (3.0 * delta[m_n - 2] - d[m_n - 2]) / 2.0;
        
        for (size_t i = 0; i < m_n - 1; ++i) {
            m_b[i] = d[i];
            m_c[i] = (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / h[i];
            m_d[i] = (d[i] + d[i + 1] - 2.0 * delta[i]) / (h[i] * h[i]);
        }
    }
    
    T operator()(T xi, extrapolate_mode extrap = extrapolate_mode::constant) const {
        if (xi < m_x.front() || xi > m_x.back()) {
            switch (extrap) {
                case extrapolate_mode::constant: {
                    if (xi < m_x.front()) return m_y.front();
                    return m_y.back();
                }
                case extrapolate_mode::linear: {
                    if (xi < m_x.front()) {
                        T slope = m_b[0];
                        return m_y[0] + slope * (xi - m_x[0]);
                    }
                    T slope = m_b.back() + 2.0 * m_c.back() * (m_x.back() - m_x[m_n-2])
                              + 3.0 * m_d.back() * std::pow(m_x.back() - m_x[m_n-2], 2);
                    return m_y.back() + slope * (xi - m_x.back());
                }
                case extrapolate_mode::nearest: {
                    if (xi < m_x.front()) return m_y.front();
                    return m_y.back();
                }
                case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
                default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
            }
        }
        
        size_t idx = find_interval(m_x, xi, false);
        T dx = xi - m_x[idx];
        return m_y[idx] + m_b[idx] * dx + m_c[idx] * dx * dx + m_d[idx] * dx * dx * dx;
    }
};

// #############################################################################
// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
// #############################################################################
template <class T>
class pchip_interpolator {
private:
    std::vector<T> m_x, m_y, m_b, m_c, m_d;
    size_t m_n;
    
public:
    pchip_interpolator() = default;
    
    pchip_interpolator(const std::vector<T>& x, const std::vector<T>& y) {
        fit(x, y);
    }
    
    void fit(const std::vector<T>& x, const std::vector<T>& y) {
        XTU_ASSERT_MSG(x.size() == y.size(), "x and y must have same size");
        XTU_ASSERT_MSG(x.size() >= 2, "At least two points required");
        m_n = x.size();
        m_x = x;
        m_y = y;
        m_b.resize(m_n - 1);
        m_c.resize(m_n - 1);
        m_d.resize(m_n - 1);
        
        std::vector<T> h(m_n - 1);
        std::vector<T> delta(m_n - 1);
        for (size_t i = 0; i < m_n - 1; ++i) {
            h[i] = x[i + 1] - x[i];
            delta[i] = (y[i + 1] - y[i]) / h[i];
        }
        
        std::vector<T> d(m_n);
        for (size_t i = 1; i < m_n - 1; ++i) {
            if (delta[i - 1] * delta[i] <= 0) {
                d[i] = 0;
            } else {
                T w1 = 2.0 * h[i] + h[i - 1];
                T w2 = h[i] + 2.0 * h[i - 1];
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }
        // Endpoint derivatives
        d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
        if (d[0] * delta[0] < 0) d[0] = 0;
        else if (delta[0] * delta[1] <= 0 && std::abs(d[0]) > std::abs(3.0 * delta[0])) {
            d[0] = 3.0 * delta[0];
        }
        
        size_t last = m_n - 1;
        d[last] = ((2.0 * h[last - 1] + h[last - 2]) * delta[last - 1] - h[last - 1] * delta[last - 2]) 
                  / (h[last - 1] + h[last - 2]);
        if (d[last] * delta[last - 1] < 0) d[last] = 0;
        else if (delta[last - 1] * delta[last - 2] <= 0 && std::abs(d[last]) > std::abs(3.0 * delta[last - 1])) {
            d[last] = 3.0 * delta[last - 1];
        }
        
        for (size_t i = 0; i < m_n - 1; ++i) {
            m_b[i] = d[i];
            m_c[i] = (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / h[i];
            m_d[i] = (d[i] + d[i + 1] - 2.0 * delta[i]) / (h[i] * h[i]);
        }
    }
    
    T operator()(T xi, extrapolate_mode extrap = extrapolate_mode::constant) const {
        if (xi < m_x.front() || xi > m_x.back()) {
            switch (extrap) {
                case extrapolate_mode::constant: {
                    if (xi < m_x.front()) return m_y.front();
                    return m_y.back();
                }
                case extrapolate_mode::nearest: {
                    if (xi < m_x.front()) return m_y.front();
                    return m_y.back();
                }
                case extrapolate_mode::nan: return std::numeric_limits<T>::quiet_NaN();
                default: XTU_THROW(std::out_of_range, "Extrapolation not allowed");
            }
        }
        
        size_t idx = find_interval(m_x, xi, false);
        T dx = xi - m_x[idx];
        return m_y[idx] + m_b[idx] * dx + m_c[idx] * dx * dx + m_d[idx] * dx * dx * dx;
    }
};

} // namespace interp

// Bring into main namespace for convenience
using interp::interp_method;
using interp::extrapolate_mode;
using interp::interp_nearest;
using interp::interp_linear;
using interp::interp_quadratic;
using interp::interp_cubic;
using interp::cubic_spline;
using interp::akima_spline;
using interp::pchip_interpolator;
using interp::interp1d;
using interp::interp_bilinear;
using interp::interp_polynomial;

XTU_NAMESPACE_END

#endif // XTU_INTERP_XINTERP_HPP