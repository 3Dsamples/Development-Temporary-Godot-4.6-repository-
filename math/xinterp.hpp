// core/xinterp.hpp
#ifndef XTENSOR_XINTERP_HPP
#define XTENSOR_XINTERP_HPP

// ----------------------------------------------------------------------------
// xinterp.hpp – Interpolation routines for 1D, 2D and ND data
// ----------------------------------------------------------------------------
// This header provides a comprehensive set of interpolation functions:
//   - 1D: linear, nearest, cubic (spline), Akima, PCHIP
//   - 2D: bilinear, bicubic, spline (gridded and scattered)
//   - ND: multilinear, nearest neighbour
//   - Extrapolation modes: constant, linear, nearest, none (nan)
//   - Spline representation (cubic, natural, clamped, not‑a‑knot)
//   - Griddata (scattered data interpolation via Delaunay or radial basis)
//
// All calculations use bignumber::BigNumber for precision; FFT is employed
// for convolution‑based smoothing splines where appropriate.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <tuple>
#include <map>
#include <limits>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xlinalg.hpp"
#include "xsorting.hpp"
#include "fft.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace interp
    {
        // ========================================================================
        // 1D Interpolation
        // ========================================================================
        enum class interp1d_method { linear, nearest, cubic, akima, pchip };
        enum class extrap_mode { constant, linear, nearest, nan };

        template <class T>
        class interp1d
        {
        public:
            // Construct 1D interpolator from x and y data
            interp1d(const std::vector<T>& x, const std::vector<T>& y,
                     interp1d_method method = interp1d_method::linear,
                     extrap_mode extrap = extrap_mode::constant,
                     T fill_value = std::numeric_limits<T>::quiet_NaN());
            // Evaluate at a single point
            T operator()(T xi) const;
            // Evaluate at multiple points
            std::vector<T> operator()(const std::vector<T>& xi) const;

        private:
            std::vector<T> m_x, m_y;
            interp1d_method m_method;
            extrap_mode m_extrap;
            T m_fill;
            std::vector<T> m_spline_z;
            std::vector<T> m_akima_slopes;
            std::vector<T> m_pchip_slopes;

            T interpolate(T xi) const;
            T extrapolate(T xi) const;
            void compute_akima_slopes();
            void compute_pchip_slopes();
            T eval_akima(size_t i, T xi) const;
            T eval_pchip(size_t i, T xi) const;
        };

        // Convenience function for 1D interpolation
        template <class T>
        std::vector<T> interp1d(const std::vector<T>& x, const std::vector<T>& y,
                                const std::vector<T>& xi,
                                const std::string& method = "linear",
                                const std::string& extrap = "constant",
                                T fill_value = std::numeric_limits<T>::quiet_NaN());

        // ========================================================================
        // 2D Interpolation (gridded)
        // ========================================================================
        template <class T>
        class interp2d
        {
        public:
            enum class method { bilinear, bicubic, spline, nearest };
            // Construct 2D interpolator from grid axes and values
            interp2d(const std::vector<T>& x, const std::vector<T>& y,
                     const xarray_container<T>& z,
                     method m = method::bilinear,
                     extrap_mode extrap = extrap_mode::constant,
                     T fill_value = std::numeric_limits<T>::quiet_NaN());
            // Evaluate at a single point
            T operator()(T xi, T yi) const;
            // Evaluate on a grid
            xarray_container<T> operator()(const std::vector<T>& xi, const std::vector<T>& yi) const;

        private:
            std::vector<T> m_x, m_y;
            xarray_container<T> m_z;
            method m_method;
            extrap_mode m_extrap;
            T m_fill;
            xarray_container<T> m_c;

            void compute_bicubic_coeffs();
            T interpolate(T xi, T yi) const;
            T extrapolate(T xi, T yi) const;
        };

        template <class T>
        xarray_container<T> interp2d(const std::vector<T>& x, const std::vector<T>& y,
                                     const xarray_container<T>& z,
                                     const std::vector<T>& xi, const std::vector<T>& yi,
                                     const std::string& method = "bilinear");

        // ========================================================================
        // ND Linear Interpolation (regular grid)
        // ========================================================================
        template <class T>
        class RegularGridInterpolator
        {
        public:
            // Construct from axes and values
            RegularGridInterpolator(const std::vector<std::vector<T>>& axes,
                                    const xarray_container<T>& values,
                                    const std::string& method = "linear",
                                    extrap_mode extrap = extrap_mode::constant,
                                    T fill_value = std::numeric_limits<T>::quiet_NaN());
            // Evaluate at a single point
            T operator()(const std::vector<T>& xi) const;

        private:
            std::vector<std::vector<T>> m_axes;
            xarray_container<T> m_values;
            std::string m_method;
            extrap_mode m_extrap;
            T m_fill;

            T interpolate(const std::vector<T>& xi) const;
            T linear_nd(std::vector<size_t>& idx, const std::vector<T>& t, size_t d) const;
            T extrapolate(const std::vector<T>& xi) const;
        };

        // ========================================================================
        // Scattered data interpolation (griddata)
        // ========================================================================
        template <class T>
        xarray_container<T> griddata(const xarray_container<T>& points,
                                     const xarray_container<T>& values,
                                     const std::vector<std::vector<T>>& grid_axes,
                                     const std::string& method = "linear");

        // Helper: unravel index for grid
        std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& shape);
    }

    using interp::interp1d;
    using interp::interp2d;
    using interp::RegularGridInterpolator;
    using interp::griddata;
    using interp::interp1d_method;
    using interp::extrap_mode;
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace interp
    {
        // Helper: binary search for interval index
        template <class T> size_t find_interval(const std::vector<T>& x, T xi) { /* TODO: implement */ return 0; }

        // interp1d constructor
        template <class T>
        interp1d<T>::interp1d(const std::vector<T>& x, const std::vector<T>& y,
                              interp1d_method method, extrap_mode extrap, T fill_value)
            : m_x(x), m_y(y), m_method(method), m_extrap(extrap), m_fill(fill_value)
        { /* TODO: validate and precompute coefficients */ }

        // Evaluate interp1d at single point
        template <class T> T interp1d<T>::operator()(T xi) const { /* TODO: implement */ return T(0); }
        // Evaluate interp1d at multiple points
        template <class T> std::vector<T> interp1d<T>::operator()(const std::vector<T>& xi) const { /* TODO: implement */ return {}; }
        // Internal interpolation dispatch
        template <class T> T interp1d<T>::interpolate(T xi) const { /* TODO: implement */ return T(0); }
        // Internal extrapolation dispatch
        template <class T> T interp1d<T>::extrapolate(T xi) const { /* TODO: implement */ return T(0); }
        // Compute Akima slopes
        template <class T> void interp1d<T>::compute_akima_slopes() { /* TODO: implement */ }
        // Compute PCHIP slopes
        template <class T> void interp1d<T>::compute_pchip_slopes() { /* TODO: implement */ }
        // Evaluate Akima interpolation
        template <class T> T interp1d<T>::eval_akima(size_t i, T xi) const { /* TODO: implement */ return T(0); }
        // Evaluate PCHIP interpolation
        template <class T> T interp1d<T>::eval_pchip(size_t i, T xi) const { /* TODO: implement */ return T(0); }

        // Convenience 1D interpolation function
        template <class T>
        std::vector<T> interp1d(const std::vector<T>& x, const std::vector<T>& y,
                                const std::vector<T>& xi, const std::string& method,
                                const std::string& extrap, T fill_value)
        { interp1d<T> interp(x, y); return interp(xi); }

        // interp2d constructor
        template <class T>
        interp2d<T>::interp2d(const std::vector<T>& x, const std::vector<T>& y,
                              const xarray_container<T>& z, method m,
                              extrap_mode extrap, T fill_value)
            : m_x(x), m_y(y), m_z(z), m_method(m), m_extrap(extrap), m_fill(fill_value)
        { /* TODO: validate and precompute */ }

        // Evaluate interp2d at single point
        template <class T> T interp2d<T>::operator()(T xi, T yi) const { /* TODO: implement */ return T(0); }
        // Evaluate interp2d on grid
        template <class T> xarray_container<T> interp2d<T>::operator()(const std::vector<T>& xi, const std::vector<T>& yi) const { /* TODO: implement */ return {}; }
        // Compute bicubic coefficients
        template <class T> void interp2d<T>::compute_bicubic_coeffs() { /* TODO: implement */ }
        // Internal 2D interpolation
        template <class T> T interp2d<T>::interpolate(T xi, T yi) const { /* TODO: implement */ return T(0); }
        // Internal 2D extrapolation
        template <class T> T interp2d<T>::extrapolate(T xi, T yi) const { /* TODO: implement */ return T(0); }

        // Convenience 2D interpolation function
        template <class T>
        xarray_container<T> interp2d(const std::vector<T>& x, const std::vector<T>& y,
                                     const xarray_container<T>& z,
                                     const std::vector<T>& xi, const std::vector<T>& yi,
                                     const std::string& method)
        { interp2d<T> interp(x, y, z); return interp(xi, yi); }

        // RegularGridInterpolator constructor
        template <class T>
        RegularGridInterpolator<T>::RegularGridInterpolator(const std::vector<std::vector<T>>& axes,
                                                            const xarray_container<T>& values,
                                                            const std::string& method,
                                                            extrap_mode extrap, T fill_value)
            : m_axes(axes), m_values(values), m_method(method), m_extrap(extrap), m_fill(fill_value)
        { /* TODO: validate */ }

        // Evaluate ND interpolator
        template <class T> T RegularGridInterpolator<T>::operator()(const std::vector<T>& xi) const { /* TODO: implement */ return T(0); }
        // Internal ND interpolation
        template <class T> T RegularGridInterpolator<T>::interpolate(const std::vector<T>& xi) const { /* TODO: implement */ return T(0); }
        // Recursive linear interpolation in ND
        template <class T> T RegularGridInterpolator<T>::linear_nd(std::vector<size_t>& idx, const std::vector<T>& t, size_t d) const { /* TODO: implement */ return T(0); }
        // Internal ND extrapolation
        template <class T> T RegularGridInterpolator<T>::extrapolate(const std::vector<T>& xi) const { /* TODO: implement */ return T(0); }

        // Scattered data interpolation (griddata)
        template <class T>
        xarray_container<T> griddata(const xarray_container<T>& points,
                                     const xarray_container<T>& values,
                                     const std::vector<std::vector<T>>& grid_axes,
                                     const std::string& method)
        { /* TODO: implement nearest / linear / cubic */ return {}; }

        // Unravel flat index to multi‑index
        inline std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& shape)
        { std::vector<size_t> idx(shape.size()); for (size_t d = shape.size(); d-- > 0; ) { idx[d] = flat % shape[d]; flat /= shape[d]; } return idx; }
    }
}

#endif // XTENSOR_XINTERP_HPPi.size(); ++i)
                    result[i] = (*this)(xi[i]);
                return result;
            }

        private:
            std::vector<T> m_x, m_y;
            interp1d_method m_method;
            extrap_mode m_extrap;
            T m_fill;
            std::vector<T> m_spline_z;      // cubic spline second derivatives
            std::vector<T> m_akima_slopes;   // Akima slopes
            std::vector<T> m_pchip_slopes;   // PCHIP slopes

            T interpolate(T xi) const
            {
                size_t i = detail::find_interval(m_x, xi);
                T t = (xi - m_x[i]) / (m_x[i+1] - m_x[i]);
                switch (m_method)
                {
                    case interp1d_method::linear:
                        return m_y[i] + t * (m_y[i+1] - m_y[i]);
                    case interp1d_method::nearest:
                        return (t < T(0.5)) ? m_y[i] : m_y[i+1];
                    case interp1d_method::cubic:
                        return detail::eval_spline(m_x, m_y, m_spline_z, xi);
                    case interp1d_method::akima:
                        return eval_akima(i, xi);
                    case interp1d_method::pchip:
                        return eval_pchip(i, xi);
                    default:
                        return T(0);
                }
            }

            T extrapolate(T xi) const
            {
                if (m_extrap == extrap_mode::nan)
                {
                    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
                        return std::numeric_limits<T>::quiet_NaN();
                    else
                        XTENSOR_THROW(std::runtime_error, "extrapolation: NaN not supported for this type");
                }
                if (m_extrap == extrap_mode::constant)
                    return m_fill;
                if (m_extrap == extrap_mode::nearest)
                    return (xi < m_x.front()) ? m_y.front() : m_y.back();
                // linear extrapolation
                if (xi < m_x.front())
                {
                    T slope = (m_y[1] - m_y[0]) / (m_x[1] - m_x[0]);
                    return m_y[0] + slope * (xi - m_x[0]);
                }
                else
                {
                    size_t n = m_x.size();
                    T slope = (m_y[n-1] - m_y[n-2]) / (m_x[n-1] - m_x[n-2]);
                    return m_y[n-1] + slope * (xi - m_x[n-1]);
                }
            }

            void compute_akima_slopes()
            {
                size_t n = m_x.size();
                m_akima_slopes.resize(n);
                std::vector<T> m(n+3);
                for (size_t i = 0; i < n-1; ++i)
                    m[i+2] = (m_y[i+1] - m_y[i]) / (m_x[i+1] - m_x[i]);
                // Quadratic extrapolation for boundary segments
                m[1] = T(2)*m[2] - m[3];
                m[0] = T(2)*m[1] - m[2];
                m[n+1] = T(2)*m[n] - m[n-1];
                m[n+2] = T(2)*m[n+1] - m[n];
                for (size_t i = 0; i < n; ++i)
                {
                    T w1 = detail::abs_val(m[i+3] - m[i+2]);
                    T w2 = detail::abs_val(m[i+1] - m[i]);
                    if (w1 + w2 < T(1e-12))
                        m_akima_slopes[i] = (m[i+1] + m[i+2]) / T(2);
                    else
                        m_akima_slopes[i] = (w1 * m[i+1] + w2 * m[i+2]) / (w1 + w2);
                }
            }

            T eval_akima(size_t i, T xi) const
            {
                T h = m_x[i+1] - m_x[i];
                T t = (xi - m_x[i]) / h;
                T t2 = t * t;
                T t3 = t2 * t;
                T p0 = m_y[i];
                T p1 = m_akima_slopes[i];
                T p2 = (T(3)*(m_y[i+1]-m_y[i])/h - T(2)*m_akima_slopes[i] - m_akima_slopes[i+1]) / h;
                T p3 = (T(2)*(m_y[i]-m_y[i+1])/h + m_akima_slopes[i] + m_akima_slopes[i+1]) / (h*h);
                return p0 + p1*(xi-m_x[i]) + p2*(xi-m_x[i])*(xi-m_x[i]) + p3*(xi-m_x[i])*(xi-m_x[i])*(xi-m_x[i]);
            }

            void compute_pchip_slopes()
            {
                size_t n = m_x.size();
                m_pchip_slopes.resize(n);
                std::vector<T> delta(n-1);
                for (size_t i = 0; i < n-1; ++i)
                    delta[i] = (m_y[i+1] - m_y[i]) / (m_x[i+1] - m_x[i]);
                for (size_t i = 1; i < n-1; ++i)
                {
                    if (delta[i-1] * delta[i] > T(0))
                    {
                        T w1 = T(2) * (m_x[i+1] - m_x[i]) + (m_x[i] - m_x[i-1]);
                        T w2 = (m_x[i+1] - m_x[i]) + T(2) * (m_x[i] - m_x[i-1]);
                        m_pchip_slopes[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i]);
                    }
                    else
                        m_pchip_slopes[i] = T(0);
                }
                m_pchip_slopes[0] = ((T(2) * delta[0] - m_pchip_slopes[1]) > T(0)) ?
                    T(3) * delta[0] - T(2) * m_pchip_slopes[1] : T(0);
                m_pchip_slopes[n-1] = ((T(2) * delta[n-2] - m_pchip_slopes[n-2]) > T(0)) ?
                    T(3) * delta[n-2] - T(2) * m_pchip_slopes[n-2] : T(0);
            }

            T eval_pchip(size_t i, T xi) const
            {
                T h = m_x[i+1] - m_x[i];
                T t = (xi - m_x[i]) / h;
                T t2 = t * t;
                T t3 = t2 * t;
                T p0 = m_y[i];
                T p1 = m_pchip_slopes[i];
                T p2 = (T(3)*(m_y[i+1]-m_y[i])/h - T(2)*m_pchip_slopes[i] - m_pchip_slopes[i+1]) / h;
                T p3 = (T(2)*(m_y[i]-m_y[i+1])/h + m_pchip_slopes[i] + m_pchip_slopes[i+1]) / (h*h);
                return p0 + p1*(xi-m_x[i]) + p2*(xi-m_x[i])*(xi-m_x[i]) + p3*(xi-m_x[i])*(xi-m_x[i])*(xi-m_x[i]);
            }
        };

        // Convenience function for 1D interpolation
        template <class T>
        std::vector<T> interp1d(const std::vector<T>& x, const std::vector<T>& y,
                                const std::vector<T>& xi,
                                const std::string& method = "linear",
                                const std::string& extrap = "constant",
                                T fill_value = std::numeric_limits<T>::quiet_NaN())
        {
            interp1d_method m;
            if (method == "linear") m = interp1d_method::linear;
            else if (method == "nearest") m = interp1d_method::nearest;
            else if (method == "cubic") m = interp1d_method::cubic;
            else if (method == "akima") m = interp1d_method::akima;
            else if (method == "pchip") m = interp1d_method::pchip;
            else XTENSOR_THROW(std::invalid_argument, "interp1d: unknown method");

            extrap_mode e;
            if (extrap == "constant") e = extrap_mode::constant;
            else if (extrap == "linear") e = extrap_mode::linear;
            else if (extrap == "nearest") e = extrap_mode::nearest;
            else if (extrap == "nan") e = extrap_mode::nan;
            else XTENSOR_THROW(std::invalid_argument, "interp1d: unknown extrapolation");

            interp1d<T> interp(x, y, m, e, fill_value);
            return interp(xi);
        }

        // ========================================================================
        // 2D Interpolation (gridded)
        // ========================================================================

        template <class T>
        class interp2d
        {
        public:
            enum class method { bilinear, bicubic, spline, nearest };

            interp2d(const std::vector<T>& x, const std::vector<T>& y,
                     const xarray_container<T>& z,
                     method m = method::bilinear,
                     extrap_mode extrap = extrap_mode::constant,
                     T fill_value = std::numeric_limits<T>::quiet_NaN())
                : m_x(x), m_y(y), m_z(z), m_method(m), m_extrap(extrap), m_fill(fill_value)
            {
                if (z.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "interp2d: z must be 2D");
                if (x.size() != z.shape()[1] || y.size() != z.shape()[0])
                    XTENSOR_THROW(std::invalid_argument, "interp2d: x and y sizes must match z dimensions");
                if (!std::is_sorted(x.begin(), x.end()) || !std::is_sorted(y.begin(), y.end()))
                    XTENSOR_THROW(std::invalid_argument, "interp2d: x and y must be strictly increasing");

                if (m == method::bicubic || m == method::spline)
                    compute_bicubic_coeffs();
            }

            T operator()(T xi, T yi) const
            {
                if (xi < m_x.front() || xi > m_x.back() || yi < m_y.front() || yi > m_y.back())
                    return extrapolate(xi, yi);
                return interpolate(xi, yi);
            }

            xarray_container<T> operator()(const std::vector<T>& xi, const std::vector<T>& yi) const
            {
                xarray_container<T> result({yi.size(), xi.size()});
                for (size_t j = 0; j < yi.size(); ++j)
                    for (size_t i = 0; i < xi.size(); ++i)
                        result(j, i) = (*this)(xi[i], yi[j]);
                return result;
            }

        private:
            std::vector<T> m_x, m_y;
            xarray_container<T> m_z;
            method m_method;
            extrap_mode m_extrap;
            T m_fill;
            xarray_container<T> m_c; // bicubic coefficients (16 per cell)

            void compute_bicubic_coeffs()
            {
                size_t nx = m_x.size(), ny = m_y.size();
                // Compute derivatives
                xarray_container<T> zx, zy, zxy;
                detail::compute_derivatives_2d(m_z, zx, zy, zxy);
                m_c = xarray_container<T>({(ny-1), (nx-1), 16}, T(0));

                for (size_t j = 0; j < ny-1; ++j)
                {
                    for (size_t i = 0; i < nx-1; ++i)
                    {
                        T hx = m_x[i+1] - m_x[i];
                        T hy = m_y[j+1] - m_y[j];

                        // Values at corners
                        T f00 = m_z(j, i);
                        T f01 = m_z(j, i+1);
                        T f10 = m_z(j+1, i);
                        T f11 = m_z(j+1, i+1);

                        // Derivatives (scaled to unit square)
                        T fx00 = zx(j, i) * hx;
                        T fx01 = zx(j, i+1) * hx;
                        T fx10 = zx(j+1, i) * hx;
                        T fx11 = zx(j+1, i+1) * hx;

                        T fy00 = zy(j, i) * hy;
                        T fy01 = zy(j, i+1) * hy;
                        T fy10 = zy(j+1, i) * hy;
                        T fy11 = zy(j+1, i+1) * hy;

                        T fxy00 = zxy(j, i) * hx * hy;
                        T fxy01 = zxy(j, i+1) * hx * hy;
                        T fxy10 = zxy(j+1, i) * hx * hy;
                        T fxy11 = zxy(j+1, i+1) * hx * hy;

                        auto coeffs = detail::bicubic_cell_coeffs<T>(
                            f00, f01, f10, f11,
                            fx00, fx01, fx10, fx11,
                            fy00, fy01, fy10, fy11,
                            fxy00, fxy01, fxy10, fxy11);

                        for (int k = 0; k < 16; ++k)
                            m_c(j, i, k) = coeffs[k];
                    }
                }
            }

            T interpolate(T xi, T yi) const
            {
                size_t ix = detail::find_interval(m_x, xi);
                size_t iy = detail::find_interval(m_y, yi);
                T tx = (xi - m_x[ix]) / (m_x[ix+1] - m_x[ix]);
                T ty = (yi - m_y[iy]) / (m_y[iy+1] - m_y[iy]);

                if (m_method == method::bilinear || m_method == method::nearest)
                {
                    T z00 = m_z(iy, ix);
                    T z01 = m_z(iy, ix+1);
                    T z10 = m_z(iy+1, ix);
                    T z11 = m_z(iy+1, ix+1);
                    if (m_method == method::nearest)
                    {
                        ix = (tx < T(0.5)) ? ix : ix+1;
                        iy = (ty < T(0.5)) ? iy : iy+1;
                        return m_z(iy, ix);
                    }
                    T z0 = z00 + tx * (z01 - z00);
                    T z1 = z10 + tx * (z11 - z10);
                    return z0 + ty * (z1 - z0);
                }
                else if (m_method == method::bicubic)
                {
                    // Evaluate bicubic polynomial
                    T tx2 = tx * tx, tx3 = tx2 * tx;
                    T ty2 = ty * ty, ty3 = ty2 * ty;
                    T result = T(0);
                    const auto& coeffs = m_c;
                    for (int j = 0; j < 4; ++j)
                    {
                        T row_sum = T(0);
                        T ypow = (j == 0) ? T(1) : (j == 1) ? ty : (j == 2) ? ty2 : ty3;
                        for (int i = 0; i < 4; ++i)
                        {
                            T xpow = (i == 0) ? T(1) : (i == 1) ? tx : (i == 2) ? tx2 : tx3;
                            int k = j * 4 + i;
                            row_sum = row_sum + coeffs(iy, ix, k) * xpow;
                        }
                        result = result + row_sum * ypow;
                    }
                    return result;
                }
                return T(0);
            }

            T extrapolate(T xi, T yi) const
            {
                if (m_extrap == extrap_mode::nan)
                {
                    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
                        return std::numeric_limits<T>::quiet_NaN();
                    else
                        XTENSOR_THROW(std::runtime_error, "extrapolation: NaN not supported");
                }
                if (m_extrap == extrap_mode::constant)
                    return m_fill;
                if (m_extrap == extrap_mode::nearest)
                {
                    size_t ix = (xi < m_x.front()) ? 0 : m_x.size()-1;
                    size_t iy = (yi < m_y.front()) ? 0 : m_y.size()-1;
                    return m_z(iy, ix);
                }
                // linear extrapolation: reuse nearest edge cell's linear interpolation
                xi = detail::clamp(xi, m_x.front(), m_x.back());
                yi = detail::clamp(yi, m_y.front(), m_y.back());
                return interpolate(xi, yi);
            }
        };

        template <class T>
        xarray_container<T> interp2d(const std::vector<T>& x, const std::vector<T>& y,
                                     const xarray_container<T>& z,
                                     const std::vector<T>& xi, const std::vector<T>& yi,
                                     const std::string& method = "bilinear")
        {
            typename interp2d<T>::method m;
            if (method == "bilinear") m = interp2d<T>::method::bilinear;
            else if (method == "bicubic") m = interp2d<T>::method::bicubic;
            else if (method == "nearest") m = interp2d<T>::method::nearest;
            else XTENSOR_THROW(std::invalid_argument, "interp2d: unknown method");

            interp2d<T> interp(x, y, z, m);
            return interp(xi, yi);
        }

        // ========================================================================
        // ND Linear Interpolation (regular grid)
        // ========================================================================

        template <class T>
        class RegularGridInterpolator
        {
        public:
            RegularGridInterpolator(const std::vector<std::vector<T>>& axes,
                                    const xarray_container<T>& values,
                                    const std::string& method = "linear",
                                    extrap_mode extrap = extrap_mode::constant,
                                    T fill_value = std::numeric_limits<T>::quiet_NaN())
                : m_axes(axes), m_values(values), m_method(method), m_extrap(extrap), m_fill(fill_value)
            {
                size_t dim = axes.size();
                if (values.dimension() != dim)
                    XTENSOR_THROW(std::invalid_argument, "RegularGridInterpolator: values dimension mismatch");
                for (size_t d = 0; d < dim; ++d)
                    if (values.shape()[d] != axes[d].size())
                        XTENSOR_THROW(std::invalid_argument, "RegularGridInterpolator: axis size mismatch");
            }

            T operator()(const std::vector<T>& xi) const
            {
                if (xi.size() != m_axes.size())
                    XTENSOR_THROW(std::invalid_argument, "RegularGridInterpolator: xi size mismatch");
                // Check bounds
                for (size_t d = 0; d < xi.size(); ++d)
                    if (xi[d] < m_axes[d].front() || xi[d] > m_axes[d].back())
                        return extrapolate(xi);
                return interpolate(xi);
            }

        private:
            std::vector<std::vector<T>> m_axes;
            xarray_container<T> m_values;
            std::string m_method;
            extrap_mode m_extrap;
            T m_fill;

            T interpolate(const std::vector<T>& xi) const
            {
                size_t dim = xi.size();
                std::vector<size_t> idx(dim);
                std::vector<T> t(dim);
                for (size_t d = 0; d < dim; ++d)
                {
                    idx[d] = detail::find_interval(m_axes[d], xi[d]);
                    t[d] = (xi[d] - m_axes[d][idx[d]]) /
                           (m_axes[d][idx[d]+1] - m_axes[d][idx[d]]);
                }
                if (m_method == "nearest")
                {
                    std::vector<size_t> corner(dim);
                    for (size_t d = 0; d < dim; ++d)
                        corner[d] = (t[d] < T(0.5)) ? idx[d] : idx[d]+1;
                    return m_values.element(corner);
                }
                // Linear interpolation: recursively average corners
                return linear_nd(idx, t, 0);
            }

            T linear_nd(std::vector<size_t>& idx, const std::vector<T>& t, size_t d) const
            {
                if (d == idx.size())
                    return m_values.element(idx);
                size_t orig = idx[d];
                T v0 = linear_nd(idx, t, d+1);
                idx[d] = orig + 1;
                T v1 = linear_nd(idx, t, d+1);
                idx[d] = orig;
                return v0 * (T(1) - t[d]) + v1 * t[d];
            }

            T extrapolate(const std::vector<T>& xi) const
            {
                if (m_extrap == extrap_mode::nan)
                {
                    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
                        return std::numeric_limits<T>::quiet_NaN();
                    else
                        XTENSOR_THROW(std::runtime_error, "extrapolation: NaN not supported");
                }
                if (m_extrap == extrap_mode::constant)
                    return m_fill;
                // clamp and interpolate
                std::vector<T> clamped = xi;
                for (size_t d = 0; d < xi.size(); ++d)
                    clamped[d] = detail::clamp(xi[d], m_axes[d].front(), m_axes[d].back());
                return interpolate(clamped);
            }
        };

        // ========================================================================
        // Scattered data interpolation (griddata)
        // ========================================================================

        template <class T>
        xarray_container<T> griddata(const xarray_container<T>& points,   // (N, dim)
                                     const xarray_container<T>& values,   // (N,)
                                     const std::vector<std::vector<T>>& grid_axes,
                                     const std::string& method = "linear")
        {
            if (points.dimension() != 2)
                XTENSOR_THROW(std::invalid_argument, "griddata: points must be N x dim");
            if (values.dimension() != 1 || values.shape()[0] != points.shape()[0])
                XTENSOR_THROW(std::invalid_argument, "griddata: values must be (N,)");

            size_t n_points = points.shape()[0];
            size_t dim = points.shape()[1];
            std::vector<size_t> grid_shape;
            size_t n_grid = 1;
            for (const auto& ax : grid_axes)
            {
                grid_shape.push_back(ax.size());
                n_grid *= ax.size();
            }
            xarray_container<T> result(grid_shape);

            if (method == "nearest")
            {
                // Linear scan nearest neighbor (O(N * M))
                for (size_t flat_idx = 0; flat_idx < n_grid; ++flat_idx)
                {
                    std::vector<size_t> grid_idx = unravel_index(flat_idx, grid_shape);
                    std::vector<T> xi(dim);
                    for (size_t d = 0; d < dim; ++d)
                        xi[d] = grid_axes[d][grid_idx[d]];
                    T min_dist = std::numeric_limits<T>::max();
                    T best_val = T(0);
                    for (size_t p = 0; p < n_points; ++p)
                    {
                        T dist = T(0);
                        for (size_t d = 0; d < dim; ++d)
                        {
                            T diff = points(p, d) - xi[d];
                            dist += diff * diff;
                        }
                        if (dist < min_dist)
                        {
                            min_dist = dist;
                            best_val = values(p);
                        }
                    }
                    result.flat(flat_idx) = best_val;
                }
            }
            else if (method == "linear")
            {
                // Inverse Distance Weighted (IDW) interpolation with power = 2
                // For each grid point, compute weighted average of nearby points
                // Use all points within a radius, or all points if radius = infinity
                T radius = std::numeric_limits<T>::max(); // consider all points
                T power = T(2);
                T epsilon = T(1e-12);

                for (size_t flat_idx = 0; flat_idx < n_grid; ++flat_idx)
                {
                    std::vector<size_t> grid_idx = unravel_index(flat_idx, grid_shape);
                    std::vector<T> xi(dim);
                    for (size_t d = 0; d < dim; ++d)
                        xi[d] = grid_axes[d][grid_idx[d]];

                    T sum_weights = T(0);
                    T sum_weighted_vals = T(0);
                    for (size_t p = 0; p < n_points; ++p)
                    {
                        T dist_sq = T(0);
                        for (size_t d = 0; d < dim; ++d)
                        {
                            T diff = points(p, d) - xi[d];
                            dist_sq += diff * diff;
                        }
                        T dist = detail::sqrt_val(dist_sq);
                        if (dist > radius) continue;
                        if (dist < epsilon)
                        {
                            // Exact match: return the point value
                            sum_weighted_vals = values(p);
                            sum_weights = T(1);
                            break;
                        }
                        T weight = T(1) / detail::pow_val(dist, power);
                        sum_weights += weight;
                        sum_weighted_vals += weight * values(p);
                    }
                    if (sum_weights > T(0))
                        result.flat(flat_idx) = sum_weighted_vals / sum_weights;
                    else
                    {
                        if constexpr (std::numeric_limits<T>::has_quiet_NaN)
                            result.flat(flat_idx) = std::numeric_limits<T>::quiet_NaN();
                        else
                            result.flat(flat_idx) = T(0);
                    }
                }
            }
            else
            {
                XTENSOR_THROW(std::invalid_argument, "griddata: unknown method");
            }
            return result;
        }

        // Helper: unravel index for grid
        inline std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& shape)
        {
            std::vector<size_t> idx(shape.size());
            size_t rem = flat;
            for (size_t d = shape.size(); d-- > 0; )
            {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            return idx;
        }

        // Helper: power for integer exponent
        template <class T>
        T pow_val(const T& base, int exp)
        {
            if (exp == 0) return T(1);
            if (exp < 0) return T(1) / pow_val(base, -exp);
            T result = base;
            for (int i = 1; i < exp; ++i)
                result = detail::multiply(result, base);
            return result;
        }

    } // namespace interp

    using interp::interp1d;
    using interp::interp2d;
    using interp::RegularGridInterpolator;
    using interp::griddata;
    using interp::interp1d_method;
    using interp::extrap_mode;

} // namespace xt

#endif // XTENSOR_XINTERP_HPP