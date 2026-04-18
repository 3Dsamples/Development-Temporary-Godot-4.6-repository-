// math/xinterp.hpp

#ifndef XTENSOR_XINTERP_HPP
#define XTENSOR_XINTERP_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xsorting.hpp"
#include "../math/xstats.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <complex>
#include <map>
#include <queue>
#include <tuple>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace interp
        {
            // --------------------------------------------------------------------
            // 1D Interpolation base and helpers
            // --------------------------------------------------------------------
            enum class InterpKind
            {
                Linear,
                Nearest,
                Zero,
                Slinear,
                Quadratic,
                Cubic,
                Previous,
                Next
            };

            enum class ExtrapolateMode
            {
                Constant,   // Use boundary values
                Linear,     // Linear extrapolation
                Nearest,    // Nearest boundary
                None        // Return NaN
            };

            namespace detail
            {
                // Find interval index for a given x value in sorted array
                template <class E>
                inline size_t find_interval(const xexpression<E>& x, double xi, bool extrapolate = false)
                {
                    const auto& x_arr = x.derived_cast();
                    if (x_arr.size() == 0) return 0;
                    if (xi <= x_arr(0)) return extrapolate ? 0 : 0;
                    if (xi >= x_arr(x_arr.size() - 1)) return extrapolate ? x_arr.size() - 2 : x_arr.size() - 2;
                    auto it = std::upper_bound(x_arr.begin(), x_arr.end(), xi);
                    return static_cast<size_t>(std::distance(x_arr.begin(), it) - 1);
                }

                // Compute slopes for cubic Hermite spline (monotonic)
                inline std::vector<double> compute_cubic_slopes(const std::vector<double>& x,
                                                                const std::vector<double>& y,
                                                                bool monotonic = true)
                {
                    size_t n = x.size();
                    std::vector<double> m(n, 0.0);
                    if (n < 2) return m;
                    // Compute secant slopes (delta)
                    std::vector<double> delta(n - 1);
                    for (size_t i = 0; i < n - 1; ++i)
                        delta[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]);
                    // Interior slopes
                    for (size_t i = 1; i < n - 1; ++i)
                    {
                        double h0 = x[i] - x[i-1];
                        double h1 = x[i+1] - x[i];
                        double d0 = delta[i-1];
                        double d1 = delta[i];
                        if (monotonic)
                        {
                            // Fritsch-Carlson monotonic cubic
                            if (d0 * d1 <= 0.0)
                            {
                                m[i] = 0.0;
                            }
                            else
                            {
                                double w0 = 2.0 * h1 + h0;
                                double w1 = h1 + 2.0 * h0;
                                m[i] = (w0 + w1) / (w0 / d0 + w1 / d1);
                            }
                        }
                        else
                        {
                            // Standard Catmull-Rom (weighted average)
                            double w0 = h1 / (h0 + h1);
                            double w1 = h0 / (h0 + h1);
                            m[i] = w0 * d0 + w1 * d1;
                        }
                    }
                    // Endpoint slopes (using one-sided differences)
                    m[0] = delta[0];
                    m[n-1] = delta[n-2];
                    if (monotonic)
                    {
                        // Enforce monotonicity at boundaries
                        if (delta[0] != 0.0)
                        {
                            double d = delta[0];
                            double h = x[1] - x[0];
                            if (m[0] * d <= 0.0) m[0] = 0.0;
                            else if (std::abs(m[0]) > 3.0 * std::abs(d)) m[0] = 3.0 * d;
                        }
                        if (n > 2 && delta[n-2] != 0.0)
                        {
                            double d = delta[n-2];
                            double h = x[n-1] - x[n-2];
                            if (m[n-1] * d <= 0.0) m[n-1] = 0.0;
                            else if (std::abs(m[n-1]) > 3.0 * std::abs(d)) m[n-1] = 3.0 * d;
                        }
                    }
                    return m;
                }

                // Linear interpolation between two points
                inline double lerp(double x0, double y0, double x1, double y1, double x)
                {
                    double t = (x - x0) / (x1 - x0);
                    return y0 * (1.0 - t) + y1 * t;
                }

                // Cubic Hermite interpolation
                inline double cubic_hermite(double x0, double y0, double m0,
                                            double x1, double y1, double m1,
                                            double x)
                {
                    double h = x1 - x0;
                    double t = (x - x0) / h;
                    double t2 = t * t;
                    double t3 = t2 * t;
                    double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                    double h10 = t3 - 2.0 * t2 + t;
                    double h01 = -2.0 * t3 + 3.0 * t2;
                    double h11 = t3 - t2;
                    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
                }
            }

            // --------------------------------------------------------------------
            // 1D Interpolator class
            // --------------------------------------------------------------------
            template <class T = double>
            class Interpolator1D
            {
            public:
                using value_type = T;
                using x_type = std::vector<T>;
                using y_type = std::vector<T>;

                Interpolator1D() = default;

                Interpolator1D(const x_type& x, const y_type& y,
                               InterpKind kind = InterpKind::Linear,
                               ExtrapolateMode extrap = ExtrapolateMode::Constant)
                    : m_x(x), m_y(y), m_kind(kind), m_extrap(extrap)
                {
                    if (m_x.size() != m_y.size())
                        XTENSOR_THROW(std::invalid_argument, "Interpolator1D: x and y must have same size");
                    if (m_x.size() < 2)
                        XTENSOR_THROW(std::invalid_argument, "Interpolator1D: at least 2 points required");
                    // Ensure x is sorted
                    if (!std::is_sorted(m_x.begin(), m_x.end()))
                    {
                        // Sort together
                        std::vector<std::pair<T, T>> pairs;
                        pairs.reserve(m_x.size());
                        for (size_t i = 0; i < m_x.size(); ++i)
                            pairs.emplace_back(m_x[i], m_y[i]);
                        std::sort(pairs.begin(), pairs.end());
                        m_x.resize(pairs.size());
                        m_y.resize(pairs.size());
                        for (size_t i = 0; i < pairs.size(); ++i)
                        {
                            m_x[i] = pairs[i].first;
                            m_y[i] = pairs[i].second;
                        }
                    }
                    // Precompute slopes for cubic
                    if (m_kind == InterpKind::Cubic)
                    {
                        m_slopes = detail::compute_cubic_slopes(m_x, m_y, true);
                    }
                }

                T operator()(T x) const
                {
                    size_t n = m_x.size();
                    if (x < m_x[0])
                    {
                        return extrapolate_left(x);
                    }
                    else if (x > m_x[n-1])
                    {
                        return extrapolate_right(x);
                    }
                    else
                    {
                        size_t idx = detail::find_interval(m_x, x, false);
                        return interpolate_segment(idx, x);
                    }
                }

                // Batch interpolation
                template <class E>
                xarray_container<T> operator()(const xexpression<E>& xi) const
                {
                    const auto& x_arr = xi.derived_cast();
                    xarray_container<T> result(x_arr.shape());
                    for (size_t i = 0; i < x_arr.size(); ++i)
                        result.flat(i) = (*this)(static_cast<T>(x_arr.flat(i)));
                    return result;
                }

                // Accessors
                const x_type& x() const { return m_x; }
                const y_type& y() const { return m_y; }
                InterpKind kind() const { return m_kind; }
                ExtrapolateMode extrap_mode() const { return m_extrap; }

            private:
                x_type m_x;
                y_type m_y;
                std::vector<T> m_slopes; // for cubic
                InterpKind m_kind = InterpKind::Linear;
                ExtrapolateMode m_extrap = ExtrapolateMode::Constant;

                T interpolate_segment(size_t idx, T x) const
                {
                    T x0 = m_x[idx];
                    T x1 = m_x[idx + 1];
                    T y0 = m_y[idx];
                    T y1 = m_y[idx + 1];
                    switch (m_kind)
                    {
                        case InterpKind::Linear:
                            return detail::lerp(x0, y0, x1, y1, x);
                        case InterpKind::Nearest:
                            return (x - x0 < x1 - x) ? y0 : y1;
                        case InterpKind::Zero:
                            return y0;
                        case InterpKind::Slinear:
                            // Same as linear for now
                            return detail::lerp(x0, y0, x1, y1, x);
                        case InterpKind::Quadratic:
                            // Not fully implemented, fallback to linear
                            return detail::lerp(x0, y0, x1, y1, x);
                        case InterpKind::Cubic:
                            {
                                T m0 = m_slopes[idx];
                                T m1 = m_slopes[idx + 1];
                                return detail::cubic_hermite(x0, y0, m0, x1, y1, m1, x);
                            }
                        case InterpKind::Previous:
                            return y0;
                        case InterpKind::Next:
                            return y1;
                        default:
                            return detail::lerp(x0, y0, x1, y1, x);
                    }
                }

                T extrapolate_left(T x) const
                {
                    if (m_extrap == ExtrapolateMode::None)
                        return std::numeric_limits<T>::quiet_NaN();
                    if (m_extrap == ExtrapolateMode::Constant)
                        return m_y[0];
                    if (m_extrap == ExtrapolateMode::Nearest)
                        return m_y[0];
                    // Linear extrapolation using first two points
                    if (m_x.size() >= 2)
                        return detail::lerp(m_x[0], m_y[0], m_x[1], m_y[1], x);
                    return m_y[0];
                }

                T extrapolate_right(T x) const
                {
                    size_t n = m_x.size();
                    if (m_extrap == ExtrapolateMode::None)
                        return std::numeric_limits<T>::quiet_NaN();
                    if (m_extrap == ExtrapolateMode::Constant)
                        return m_y[n-1];
                    if (m_extrap == ExtrapolateMode::Nearest)
                        return m_y[n-1];
                    // Linear extrapolation using last two points
                    if (n >= 2)
                        return detail::lerp(m_x[n-2], m_y[n-2], m_x[n-1], m_y[n-1], x);
                    return m_y[n-1];
                }
            };

            // --------------------------------------------------------------------
            // Convenience functions for 1D interpolation
            // --------------------------------------------------------------------
            template <class E1, class E2>
            inline auto interp1d(const xexpression<E1>& x, const xexpression<E2>& y,
                                 const xexpression<E1>& xi,
                                 const std::string& kind = "linear",
                                 const std::string& extrapolate = "constant")
            {
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                std::vector<value_type> x_vec(x.derived_cast().begin(), x.derived_cast().end());
                std::vector<value_type> y_vec(y.derived_cast().begin(), y.derived_cast().end());
                InterpKind k = InterpKind::Linear;
                if (kind == "linear") k = InterpKind::Linear;
                else if (kind == "nearest") k = InterpKind::Nearest;
                else if (kind == "zero") k = InterpKind::Zero;
                else if (kind == "cubic") k = InterpKind::Cubic;
                else if (kind == "previous") k = InterpKind::Previous;
                else if (kind == "next") k = InterpKind::Next;
                else XTENSOR_THROW(std::invalid_argument, "interp1d: unknown kind '" + kind + "'");

                ExtrapolateMode ext = ExtrapolateMode::Constant;
                if (extrapolate == "constant") ext = ExtrapolateMode::Constant;
                else if (extrapolate == "linear") ext = ExtrapolateMode::Linear;
                else if (extrapolate == "nearest") ext = ExtrapolateMode::Nearest;
                else if (extrapolate == "none") ext = ExtrapolateMode::None;
                else XTENSOR_THROW(std::invalid_argument, "interp1d: unknown extrapolate mode");

                Interpolator1D<value_type> interp(x_vec, y_vec, k, ext);
                return interp(xi);
            }

            // --------------------------------------------------------------------
            // 2D Interpolation (gridded)
            // --------------------------------------------------------------------
            template <class T = double>
            class Interpolator2D
            {
            public:
                using value_type = T;
                using x_type = std::vector<T>;
                using z_type = xarray_container<T>;

                Interpolator2D() = default;

                Interpolator2D(const x_type& x, const x_type& y, const z_type& z,
                               InterpKind kind = InterpKind::Linear,
                               ExtrapolateMode extrap = ExtrapolateMode::Constant)
                    : m_x(x), m_y(y), m_z(z), m_kind(kind), m_extrap(extrap)
                {
                    if (m_z.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "Interpolator2D: z must be 2D");
                    if (m_z.shape()[0] != m_y.size() || m_z.shape()[1] != m_x.size())
                        XTENSOR_THROW(std::invalid_argument, "Interpolator2D: z shape mismatch with x,y sizes");
                    // Ensure sorted
                    if (!std::is_sorted(m_x.begin(), m_x.end()))
                        XTENSOR_THROW(std::invalid_argument, "x coordinates must be sorted");
                    if (!std::is_sorted(m_y.begin(), m_y.end()))
                        XTENSOR_THROW(std::invalid_argument, "y coordinates must be sorted");
                }

                T operator()(T x, T y) const
                {
                    // Find indices
                    size_t ix = detail::find_interval(m_x, x, m_extrap != ExtrapolateMode::None);
                    size_t iy = detail::find_interval(m_y, y, m_extrap != ExtrapolateMode::None);
                    bool x_in_range = (x >= m_x[0] && x <= m_x.back());
                    bool y_in_range = (y >= m_y[0] && y <= m_y.back());
                    if (!x_in_range || !y_in_range)
                    {
                        if (m_extrap == ExtrapolateMode::None)
                            return std::numeric_limits<T>::quiet_NaN();
                        // Simple extrapolation: clamp
                        ix = std::min(ix, m_x.size() - 2);
                        iy = std::min(iy, m_y.size() - 2);
                    }
                    else
                    {
                        ix = std::min(ix, m_x.size() - 2);
                        iy = std::min(iy, m_y.size() - 2);
                    }

                    T x0 = m_x[ix], x1 = m_x[ix+1];
                    T y0 = m_y[iy], y1 = m_y[iy+1];
                    T z00 = m_z(iy, ix);
                    T z01 = m_z(iy, ix+1);
                    T z10 = m_z(iy+1, ix);
                    T z11 = m_z(iy+1, ix+1);

                    if (m_kind == InterpKind::Linear)
                    {
                        // Bilinear interpolation
                        T tx = (x - x0) / (x1 - x0);
                        T ty = (y - y0) / (y1 - y0);
                        T z0 = z00 * (1 - tx) + z01 * tx;
                        T z1 = z10 * (1 - tx) + z11 * tx;
                        return z0 * (1 - ty) + z1 * ty;
                    }
                    else if (m_kind == InterpKind::Nearest)
                    {
                        // Nearest neighbor
                        T dx0 = x - x0, dx1 = x1 - x;
                        T dy0 = y - y0, dy1 = y1 - y;
                        if (dx0 <= dx1 && dy0 <= dy1) return z00;
                        if (dx1 < dx0 && dy0 <= dy1) return z01;
                        if (dx0 <= dx1 && dy1 < dy0) return z10;
                        return z11;
                    }
                    else
                    {
                        // Fallback to linear
                        T tx = (x - x0) / (x1 - x0);
                        T ty = (y - y0) / (y1 - y0);
                        T z0 = z00 * (1 - tx) + z01 * tx;
                        T z1 = z10 * (1 - tx) + z11 * tx;
                        return z0 * (1 - ty) + z1 * ty;
                    }
                }

            private:
                x_type m_x;
                x_type m_y;
                z_type m_z;
                InterpKind m_kind = InterpKind::Linear;
                ExtrapolateMode m_extrap = ExtrapolateMode::Constant;
            };

            // Convenience for 2D gridded interpolation
            template <class E1, class E2, class E3>
            inline auto interp2d(const xexpression<E1>& x, const xexpression<E2>& y,
                                 const xexpression<E3>& z,
                                 const xexpression<E1>& xi, const xexpression<E2>& yi,
                                 const std::string& kind = "linear")
            {
                using value_type = std::common_type_t<typename E1::value_type,
                                                      typename E2::value_type,
                                                      typename E3::value_type>;
                std::vector<value_type> x_vec(x.derived_cast().begin(), x.derived_cast().end());
                std::vector<value_type> y_vec(y.derived_cast().begin(), y.derived_cast().end());
                auto z_arr = eval(z);
                InterpKind k = (kind == "linear") ? InterpKind::Linear : InterpKind::Nearest;
                Interpolator2D<value_type> interp(x_vec, y_vec, z_arr, k);
                const auto& xi_arr = xi.derived_cast();
                const auto& yi_arr = yi.derived_cast();
                if (xi_arr.shape() != yi_arr.shape())
                    XTENSOR_THROW(std::invalid_argument, "interp2d: xi and yi must have same shape");
                xarray_container<value_type> result(xi_arr.shape());
                for (size_t i = 0; i < xi_arr.size(); ++i)
                {
                    result.flat(i) = interp(static_cast<value_type>(xi_arr.flat(i)),
                                            static_cast<value_type>(yi_arr.flat(i)));
                }
                return result;
            }

            // --------------------------------------------------------------------
            // Spline interpolation (1D cubic spline with natural/not-a-knot)
            // --------------------------------------------------------------------
            enum class SplineBoundary
            {
                Natural,        // second derivative = 0 at ends
                NotAKnot,       // third derivative continuous at second and second-to-last points
                Clamped,        // first derivative specified
                Periodic
            };

            template <class T = double>
            class CubicSpline
            {
            public:
                using value_type = T;

                CubicSpline() = default;

                CubicSpline(const std::vector<T>& x, const std::vector<T>& y,
                            SplineBoundary bc = SplineBoundary::Natural,
                            T left_slope = 0.0, T right_slope = 0.0)
                    : m_x(x), m_y(y), m_bc(bc)
                {
                    if (x.size() != y.size() || x.size() < 2)
                        XTENSOR_THROW(std::invalid_argument, "CubicSpline: need at least 2 points with matching sizes");
                    if (!std::is_sorted(m_x.begin(), m_x.end()))
                    {
                        // Sort
                        std::vector<std::pair<T,T>> pairs;
                        pairs.reserve(x.size());
                        for (size_t i=0; i<x.size(); ++i) pairs.emplace_back(x[i], y[i]);
                        std::sort(pairs.begin(), pairs.end());
                        m_x.resize(pairs.size());
                        m_y.resize(pairs.size());
                        for (size_t i=0; i<pairs.size(); ++i)
                        {
                            m_x[i] = pairs[i].first;
                            m_y[i] = pairs[i].second;
                        }
                    }
                    compute_coefficients(left_slope, right_slope);
                }

                T operator()(T x) const
                {
                    size_t n = m_x.size();
                    if (x < m_x[0] || x > m_x[n-1])
                    {
                        // Simple linear extrapolation using boundary slope
                        size_t idx = (x < m_x[0]) ? 0 : n-2;
                        T dx = x - m_x[idx];
                        if (x < m_x[0])
                            return m_y[0] + dx * m_c[idx];
                        else
                            return m_y[n-1] + (x - m_x[n-1]) * m_c[n-1];
                    }
                    size_t idx = detail::find_interval(m_x, x, false);
                    idx = std::min(idx, n-2);
                    T dx = x - m_x[idx];
                    // Evaluate cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
                    T a = m_a[idx];
                    T b = m_b[idx];
                    T c = m_c[idx];
                    T d = m_d[idx];
                    return a + dx * (b + dx * (c + dx * d));
                }

                T derivative(T x) const
                {
                    size_t n = m_x.size();
                    size_t idx = detail::find_interval(m_x, x, false);
                    idx = std::min(idx, n-2);
                    T dx = x - m_x[idx];
                    return m_b[idx] + dx * (2.0 * m_c[idx] + 3.0 * m_d[idx] * dx);
                }

                T second_derivative(T x) const
                {
                    size_t n = m_x.size();
                    size_t idx = detail::find_interval(m_x, x, false);
                    idx = std::min(idx, n-2);
                    T dx = x - m_x[idx];
                    return 2.0 * m_c[idx] + 6.0 * m_d[idx] * dx;
                }

            private:
                std::vector<T> m_x, m_y;
                std::vector<T> m_a, m_b, m_c, m_d; // coefficients for each interval
                SplineBoundary m_bc = SplineBoundary::Natural;

                void compute_coefficients(T left_slope, T right_slope)
                {
                    size_t n = m_x.size();
                    m_a.resize(n-1);
                    m_b.resize(n-1);
                    m_c.resize(n-1);
                    m_d.resize(n-1);
                    for (size_t i=0; i<n-1; ++i)
                        m_a[i] = m_y[i];

                    std::vector<T> h(n-1);
                    for (size_t i=0; i<n-1; ++i)
                        h[i] = m_x[i+1] - m_x[i];

                    // Setup tridiagonal system for second derivatives (m_c coefficients are actually second deriv/2)
                    std::vector<T> alpha(n-1), beta(n-1);
                    for (size_t i=1; i<n-1; ++i)
                    {
                        alpha[i] = 3.0/h[i] * (m_a[i+1] - m_a[i]) - 3.0/h[i-1] * (m_a[i] - m_a[i-1]);
                    }

                    std::vector<T> l(n), mu(n), z(n);
                    if (m_bc == SplineBoundary::Natural)
                    {
                        l[0] = 1.0;
                        mu[0] = 0.0;
                        z[0] = 0.0;
                    }
                    else if (m_bc == SplineBoundary::Clamped)
                    {
                        l[0] = 2.0 * h[0];
                        mu[0] = 0.5;
                        z[0] = 3.0 * ((m_a[1] - m_a[0]) / h[0] - left_slope);
                    }
                    else // Not-a-knot (simplified: use natural)
                    {
                        l[0] = 1.0;
                        mu[0] = 0.0;
                        z[0] = 0.0;
                    }

                    for (size_t i=1; i<n-1; ++i)
                    {
                        l[i] = 2.0 * (m_x[i+1] - m_x[i-1]) - h[i-1] * mu[i-1];
                        mu[i] = h[i] / l[i];
                        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
                    }

                    size_t last = n-1;
                    if (m_bc == SplineBoundary::Natural)
                    {
                        l[last] = 1.0;
                        z[last] = 0.0;
                        m_c[last-1] = z[last];
                    }
                    else if (m_bc == SplineBoundary::Clamped)
                    {
                        l[last] = h[n-2] * (2.0 - mu[n-2]);
                        z[last] = 3.0 * (right_slope - (m_a[n-1] - m_a[n-2]) / h[n-2]) - h[n-2] * z[n-2];
                        m_c[last-1] = z[last] / l[last];
                    }
                    else
                    {
                        l[last] = 1.0;
                        z[last] = 0.0;
                        m_c[last-1] = z[last];
                    }

                    for (size_t i=n-2; i>0; --i)
                    {
                        m_c[i-1] = z[i] - mu[i] * m_c[i];
                    }
                    // Adjust size of m_c? Actually we need c for each interval (n-1)
                    // We have second derivatives at knots (n values). Let's store in a temporary array.
                    std::vector<T> c_knots(n, 0.0);
                    for (size_t i=0; i<n-1; ++i)
                        c_knots[i] = m_c[i];
                    c_knots[0] = (m_bc == SplineBoundary::Natural) ? 0.0 : m_c[0];
                    // Now compute polynomial coefficients for each interval
                    for (size_t i=0; i<n-1; ++i)
                    {
                        T hi = h[i];
                        m_b[i] = (m_y[i+1] - m_y[i]) / hi - hi * (2.0 * c_knots[i] + c_knots[i+1]) / 3.0;
                        m_c[i] = c_knots[i];
                        m_d[i] = (c_knots[i+1] - c_knots[i]) / (3.0 * hi);
                    }
                }
            };

            // Convenience for cubic spline
            template <class E1, class E2>
            inline auto spline(const xexpression<E1>& x, const xexpression<E2>& y,
                               const xexpression<E1>& xi,
                               const std::string& bc = "natural")
            {
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                std::vector<value_type> xv(x.derived_cast().begin(), x.derived_cast().end());
                std::vector<value_type> yv(y.derived_cast().begin(), y.derived_cast().end());
                SplineBoundary b = (bc == "natural") ? SplineBoundary::Natural : SplineBoundary::NotAKnot;
                CubicSpline<value_type> spl(xv, yv, b);
                const auto& xi_arr = xi.derived_cast();
                xarray_container<value_type> result(xi_arr.shape());
                for (size_t i=0; i<xi_arr.size(); ++i)
                    result.flat(i) = spl(static_cast<value_type>(xi_arr.flat(i)));
                return result;
            }

            // --------------------------------------------------------------------
            // Barycentric rational interpolation (Floater-Hormann)
            // --------------------------------------------------------------------
            template <class T = double>
            class BarycentricRationalInterpolator
            {
            public:
                using value_type = T;

                BarycentricRationalInterpolator() = default;

                BarycentricRationalInterpolator(const std::vector<T>& x, const std::vector<T>& y,
                                                size_t d = 3) // blending parameter
                    : m_x(x), m_y(y), m_d(d)
                {
                    if (x.size() != y.size() || x.size() < d+1)
                        XTENSOR_THROW(std::invalid_argument, "Need at least d+1 points");
                    compute_weights();
                }

                T operator()(T x) const
                {
                    // Check if exact match
                    for (size_t i=0; i<m_x.size(); ++i)
                        if (std::abs(x - m_x[i]) < 1e-12)
                            return m_y[i];

                    T num = 0.0, den = 0.0;
                    for (size_t i=0; i<m_x.size(); ++i)
                    {
                        T w = m_weights[i];
                        T diff = x - m_x[i];
                        if (std::abs(diff) < 1e-15) return m_y[i];
                        T term = w / diff;
                        num += term * m_y[i];
                        den += term;
                    }
                    return num / den;
                }

            private:
                std::vector<T> m_x, m_y, m_weights;
                size_t m_d = 3;

                void compute_weights()
                {
                    size_t n = m_x.size();
                    m_weights.resize(n);
                    for (size_t i=0; i<n; ++i)
                    {
                        T w = 0.0;
                        size_t i_min = (i < m_d) ? 0 : i - m_d;
                        size_t i_max = (i + m_d >= n) ? n - m_d - 1 : i;
                        for (size_t k=i_min; k<=i_max; ++k)
                        {
                            T prod = 1.0;
                            for (size_t j=k; j<=k+m_d; ++j)
                            {
                                if (j == i) continue;
                                prod *= 1.0 / (m_x[i] - m_x[j]);
                            }
                            w += (k % 2 == 0 ? 1.0 : -1.0) * prod;
                        }
                        m_weights[i] = w;
                    }
                }
            };

            // --------------------------------------------------------------------
            // Griddata: interpolate scattered data onto a grid
            // --------------------------------------------------------------------
            enum class GridDataMethod
            {
                Nearest,
                Linear,
                Cubic,
                RBF  // Radial Basis Function (thin plate spline)
            };

            template <class T = double>
            inline xarray_container<T> griddata(const std::vector<T>& points_x,
                                                const std::vector<T>& points_y,
                                                const std::vector<T>& values,
                                                const std::vector<T>& grid_x,
                                                const std::vector<T>& grid_y,
                                                GridDataMethod method = GridDataMethod::Linear)
            {
                size_t nx = grid_x.size();
                size_t ny = grid_y.size();
                xarray_container<T> result({ny, nx});
                if (points_x.size() != points_y.size() || points_x.size() != values.size())
                    XTENSOR_THROW(std::invalid_argument, "griddata: input vectors must have same size");

                if (method == GridDataMethod::Nearest)
                {
                    // Nearest neighbor
                    for (size_t j=0; j<ny; ++j)
                    {
                        T y = grid_y[j];
                        for (size_t i=0; i<nx; ++i)
                        {
                            T x = grid_x[i];
                            T min_dist = std::numeric_limits<T>::max();
                            T best_val = 0;
                            for (size_t k=0; k<points_x.size(); ++k)
                            {
                                T dx = points_x[k] - x;
                                T dy = points_y[k] - y;
                                T dist = dx*dx + dy*dy;
                                if (dist < min_dist)
                                {
                                    min_dist = dist;
                                    best_val = values[k];
                                }
                            }
                            result(j, i) = best_val;
                        }
                    }
                }
                else if (method == GridDataMethod::Linear)
                {
                    // Delaunay triangulation + linear interpolation (simplified using inverse distance weighting)
                    for (size_t j=0; j<ny; ++j)
                    {
                        T y = grid_y[j];
                        for (size_t i=0; i<nx; ++i)
                        {
                            T x = grid_x[i];
                            T sum_weights = 0.0;
                            T sum_val = 0.0;
                            for (size_t k=0; k<points_x.size(); ++k)
                            {
                                T dx = points_x[k] - x;
                                T dy = points_y[k] - y;
                                T dist = std::sqrt(dx*dx + dy*dy);
                                if (dist < 1e-10)
                                {
                                    sum_val = values[k];
                                    sum_weights = 1.0;
                                    break;
                                }
                                T w = 1.0 / (dist * dist);
                                sum_weights += w;
                                sum_val += w * values[k];
                            }
                            result(j, i) = (sum_weights > 0) ? sum_val / sum_weights : 0.0;
                        }
                    }
                }
                else if (method == GridDataMethod::Cubic)
                {
                    // Clough-Tocher or similar - fallback to RBF simplified
                    // Use thin plate spline approximation
                    size_t n = points_x.size();
                    // Build matrix for RBF (thin plate: phi(r) = r^2 * log(r))
                    // This is a simplified placeholder
                    for (size_t j=0; j<ny; ++j)
                    {
                        T y = grid_y[j];
                        for (size_t i=0; i<nx; ++i)
                        {
                            T x = grid_x[i];
                            T val = 0.0;
                            T sum_w = 0.0;
                            for (size_t k=0; k<n; ++k)
                            {
                                T dx = points_x[k] - x;
                                T dy = points_y[k] - y;
                                T r2 = dx*dx + dy*dy;
                                T w = (r2 > 1e-12) ? r2 * std::log(std::sqrt(r2) + 1e-12) : 0.0;
                                sum_w += w;
                                val += w * values[k];
                            }
                            result(j, i) = (sum_w != 0) ? val / sum_w : 0.0;
                        }
                    }
                }
                return result;
            }

            // --------------------------------------------------------------------
            // Interpolation with smoothing (spline smoothing)
            // --------------------------------------------------------------------
            template <class T = double>
            inline std::vector<T> smooth_spline(const std::vector<T>& x, const std::vector<T>& y,
                                                T smoothing = 0.5, size_t num_points = 0)
            {
                // Reinsch smoothing spline (simplified)
                if (num_points == 0) num_points = x.size() * 2;
                std::vector<T> xs(num_points);
                T xmin = x.front(), xmax = x.back();
                for (size_t i=0; i<num_points; ++i)
                    xs[i] = xmin + (xmax - xmin) * i / (num_points - 1);
                // Use cubic spline with reduced weights (not fully implemented)
                CubicSpline<T> spl(x, y, SplineBoundary::Natural);
                std::vector<T> ys(num_points);
                for (size_t i=0; i<num_points; ++i)
                    ys[i] = spl(xs[i]);
                return ys;
            }

        } // namespace interp

        // Bring into xt namespace
        using interp::Interpolator1D;
        using interp::Interpolator2D;
        using interp::CubicSpline;
        using interp::BarycentricRationalInterpolator;
        using interp::interp1d;
        using interp::interp2d;
        using interp::spline;
        using interp::griddata;
        using interp::smooth_spline;
        using interp::InterpKind;
        using interp::ExtrapolateMode;
        using interp::SplineBoundary;
        using interp::GridDataMethod;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XINTERP_HPP

// math/xinterp.hpp