//File 0027 : core/xinterpolate.hpp
//Interpolation: linear, cubic, spline, nearest for 1D/2D/3D/4D gridded data with SIMD batch evaluation.
#ifndef XTENSOR_XINTERPOLATE_HPP
#define XTENSOR_XINTERPOLATE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
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
#include "xsort.hpp"
#include "xlinalg.hpp"

namespace xt {
namespace interpolate {

    // Enum for method selection
    enum class method { nearest, linear, cubic, spline };

    namespace detail {

        // Locate the interval index for a scalar x in ascending array xp.
        // Returns index i such that xp[i] <= x < xp[i+1] (or clamped to ends).
        // Assumes xp size >= 2 and xp is strictly increasing.
        template <class T>
        std::ptrdiff_t find_interval(const T* xp, std::size_t n, T x) {
            if (x <= xp[0]) return 0;
            if (x >= xp[n-1]) return static_cast<std::ptrdiff_t>(n - 2);
            auto it = std::upper_bound(xp, xp + n, x);
            return static_cast<std::ptrdiff_t>(it - xp - 1);
        }

        // Clamp index to [0, n-1]
        inline std::ptrdiff_t clamp_index(std::ptrdiff_t i, std::size_t n) {
            if (i < 0) return 0;
            if (static_cast<std::size_t>(i) >= n) return static_cast<std::ptrdiff_t>(n) - 1;
            return i;
        }

        // 1D cubic convolution kernel (Mitchell-Netravali or Catmull-Rom). We'll use Catmull-Rom.
        template <class T>
        T cubic_kernel(T s) {
            s = std::abs(s);
            if (s <= T(1))
                return (T(1.5) * s - T(2.5)) * s * s + T(1);
            else if (s <= T(2))
                return ((-T(0.5) * s + T(2.5)) * s - T(4.0)) * s + T(2.0);
            else
                return T(0);
        }

        // Natural cubic spline coefficients computation via Thomas algorithm.
        template <class T>
        std::vector<T> spline_coeff(const std::vector<T>& x, const std::vector<T>& y) {
            std::size_t n = x.size();
            if (n < 2) return y; // no spline possible
            std::vector<T> h(n-1), alpha(n, 0), l(n, 0), mu(n, 0), z(n, 0), c(n, 0), b(n-1), d(n-1);
            for (std::size_t i = 0; i < n-1; ++i) h[i] = x[i+1] - x[i];
            for (std::size_t i = 1; i < n-1; ++i) {
                alpha[i] = (T(3)/h[i]) * (y[i+1]-y[i]) - (T(3)/h[i-1]) * (y[i]-y[i-1]);
            }
            l[0] = 1; mu[0] = 0; z[0] = 0;
            for (std::size_t i = 1; i < n-1; ++i) {
                l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1];
                mu[i] = h[i] / l[i];
                z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i];
            }
            l[n-1] = 1; z[n-1] = 0; c[n-1] = 0;
            for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(n)-2; j >= 0; --j) {
                c[j] = z[j] - mu[j]*c[j+1];
                b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/T(3);
                d[j] = (c[j+1]-c[j]) / (T(3)*h[j]);
            }
            std::vector<T> coeffs(4*(n-1)); // store a,b,c,d for each segment i
            for (std::size_t i = 0; i < n-1; ++i) {
                coeffs[4*i+0] = y[i];        // a
                coeffs[4*i+1] = b[i];        // b
                coeffs[4*i+2] = c[i];        // c
                coeffs[4*i+3] = d[i];        // d
            }
            return coeffs;
        }

        // Evaluate cubic spline at x given coefficients and breakpoints
        template <class T>
        T eval_spline(const std::vector<T>& x_breaks, const std::vector<T>& coeffs, T x) {
            std::size_t n = x_breaks.size();
            if (n < 2) return coeffs.empty() ? T(0) : coeffs[0];
            std::ptrdiff_t i = find_interval(x_breaks.data(), n, x);
            if (i < 0) i = 0;
            if (static_cast<std::size_t>(i) >= n-1) i = static_cast<std::ptrdiff_t>(n-2);
            T dx = x - x_breaks[i];
            const T* seg = &coeffs[4*i];
            return seg[0] + dx*(seg[1] + dx*(seg[2] + dx*seg[3]));
        }

    } // namespace detail

    /*********************************************
     * 1D Interpolation
     *********************************************/

    /**
     * 1D linear interpolation. xp and fp must be 1D arrays of same size, x query points.
     */
    template <class T>
    inline auto interp1d_linear(const xarray_container<uvector<T>>& xp,
                                const xarray_container<uvector<T>>& fp,
                                const xarray_container<uvector<T>>& xq) {
        if (xp.dimension() != 1 || fp.dimension() != 1 || xp.size() != fp.size())
            throw std::runtime_error("interp1d_linear: xp/fp must be 1D and same length.");
        auto result = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>(xq.shape());
        const T* xp_data = xp.data();
        const T* fp_data = fp.data();
        std::size_t n = xp.size();
        std::size_t nq = xq.size();
        for (std::size_t j = 0; j < nq; ++j) {
            T x = xq[j];
            std::ptrdiff_t i = detail::find_interval(xp_data, n, x);
            i = detail::clamp_index(i, n);
            std::ptrdiff_t i2 = std::min(static_cast<std::ptrdiff_t>(i+1), static_cast<std::ptrdiff_t>(n-1));
            T x0 = xp_data[i], x1 = xp_data[i2];
            T y0 = fp_data[i], y1 = fp_data[i2];
            T t = (x - x0) / (x1 - x0);
            result[j] = y0 + t * (y1 - y0);
        }
        return result;
    }

    /**
     * 1D cubic convolution interpolation (Catmull-Rom spline for evenly spaced data assumed).
     */
    template <class T>
    inline auto interp1d_cubic(const xarray_container<uvector<T>>& xp,
                               const xarray_container<uvector<T>>& fp,
                               const xarray_container<uvector<T>>& xq) {
        if (xp.dimension() != 1 || fp.dimension() != 1 || xp.size() != fp.size())
            throw std::runtime_error("interp1d_cubic: xp/fp must be 1D and same length.");
        auto result = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>(xq.shape());
        const T* xp_data = xp.data();
        const T* fp_data = fp.data();
        std::size_t n = xp.size();
        std::size_t nq = xq.size();
        for (std::size_t j = 0; j < nq; ++j) {
            T x = xq[j];
            std::ptrdiff_t idx = detail::find_interval(xp_data, n, x);
            T sum = 0;
            for (std::ptrdiff_t k = -1; k <= 2; ++k) {
                std::ptrdiff_t idxk = detail::clamp_index(idx + k, n);
                T s = (x - xp_data[idxk]) / (xp_data[1] - xp_data[0]); // assuming uniform spacing
                sum += fp_data[idxk] * detail::cubic_kernel(s - k);
            }
            result[j] = sum;
        }
        return result;
    }

    /**
     * 1D natural cubic spline interpolation.
     */
    template <class T>
    inline auto interp1d_spline(const xarray_container<uvector<T>>& xp,
                                const xarray_container<uvector<T>>& fp,
                                const xarray_container<uvector<T>>& xq) {
        if (xp.dimension() != 1 || fp.dimension() != 1 || xp.size() != fp.size())
            throw std::runtime_error("interp1d_spline: xp/fp must be 1D and same length.");
        std::size_t n = xp.size();
        if (n < 2) throw std::runtime_error("Need at least 2 points for spline.");
        std::vector<T> xv(xp.data(), xp.data()+n);
        std::vector<T> fv(fp.data(), fp.data()+n);
        auto coeffs = detail::spline_coeff(xv, fv);
        auto result = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>(xq.shape());
        for (std::size_t j = 0; j < xq.size(); ++j) {
            result[j] = detail::eval_spline(xv, coeffs, xq[j]);
        }
        return result;
    }

    /**
     * 1D nearest-neighbor interpolation.
     */
    template <class T>
    inline auto interp1d_nearest(const xarray_container<uvector<T>>& xp,
                                 const xarray_container<uvector<T>>& fp,
                                 const xarray_container<uvector<T>>& xq) {
        if (xp.dimension() != 1 || fp.dimension() != 1 || xp.size() != fp.size())
            throw std::runtime_error("interp1d_nearest: xp/fp must be 1D and same length.");
        auto result = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>(xq.shape());
        const T* xp_data = xp.data();
        const T* fp_data = fp.data();
        std::size_t n = xp.size();
        for (std::size_t j = 0; j < xq.size(); ++j) {
            T x = xq[j];
            std::ptrdiff_t i = detail::find_interval(xp_data, n, x);
            if (i < 0) i = 0;
            if (static_cast<std::size_t>(i) >= n-1)
                result[j] = fp_data[n-1];
            else {
                T dleft = x - xp_data[i];
                T dright = xp_data[i+1] - x;
                result[j] = (dleft <= dright) ? fp_data[i] : fp_data[i+1];
            }
        }
        return result;
    }

    /**
     * Generic 1D interpolation dispatch.
     */
    template <class T>
    inline auto interp1d(const xarray_container<uvector<T>>& xp,
                         const xarray_container<uvector<T>>& fp,
                         const xarray_container<uvector<T>>& xq,
                         method met = method::linear) {
        switch (met) {
            case method::nearest: return interp1d_nearest(xp, fp, xq);
            case method::linear: return interp1d_linear(xp, fp, xq);
            case method::cubic: return interp1d_cubic(xp, fp, xq);
            case method::spline: return interp1d_spline(xp, fp, xq);
            default: throw std::runtime_error("Unknown interpolation method.");
        }
    }

    /*********************************************
     * 2D Interpolation on regular grid
     *********************************************/

    /**
     * 2D bilinear interpolation. Z is a 2D array on grid (x,y). xq, yq are 1D or scalar.
     */
    template <class T>
    inline auto interp2d_linear(const xarray_container<uvector<T>>& x,
                                const xarray_container<uvector<T>>& y,
                                const xarray_container<uvector<T>>& Z,
                                const xarray_container<uvector<T>>& xq,
                                const xarray_container<uvector<T>>& yq) {
        if (x.dimension()!=1 || y.dimension()!=1 || Z.dimension()!=2)
            throw std::runtime_error("interp2d_linear: x,y 1D; Z 2D.");
        std::size_t nx = x.size(), ny = y.size();
        if (Z.shape()[0]!=ny || Z.shape()[1]!=nx)
            throw std::runtime_error("Z shape must be (ny, nx).");
        auto result = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>(xq.shape());
        const T* xd = x.data();
        const T* yd = y.data();
        for (std::size_t k = 0; k < xq.size(); ++k) {
            T xv = xq[k], yv = yq[k];
            std::ptrdiff_t ix = detail::find_interval(xd, nx, xv);
            std::ptrdiff_t iy = detail::find_interval(yd, ny, yv);
            ix = detail::clamp_index(ix, nx); iy = detail::clamp_index(iy, ny);
            std::ptrdiff_t ix2 = std::min(ix+1, static_cast<std::ptrdiff_t>(nx-1));
            std::ptrdiff_t iy2 = std::min(iy+1, static_cast<std::ptrdiff_t>(ny-1));
            T x1 = xd[ix], x2 = xd[ix2];
            T y1 = yd[iy], y2 = yd[iy2];
            T fx = (xv - x1) / (x2 - x1);
            T fy = (yv - y1) / (y2 - y1);
            T q00 = Z(iy, ix);
            T q10 = Z(iy, ix2);
            T q01 = Z(iy2, ix);
            T q11 = Z(iy2, ix2);
            result[k] = (1-fx)*(1-fy)*q00 + fx*(1-fy)*q10 + (1-fx)*fy*q01 + fx*fy*q11;
        }
        return result;
    }

    /**
     * 2D bicubic interpolation (Catmull-Rom spline based).
     */
    template <class T>
    inline auto interp2d_cubic(const xarray_container<uvector<T>>& x,
                               const xarray_container<uvector<T>>& y,
                               const xarray_container<uvector<T>>& Z,
                               const xarray_container<uvector<T>>& xq,
                               const xarray_container<uvector<T>>& yq) {
        if (x.dimension()!=1 || y.dimension()!=1 || Z.dimension()!=2)
            throw std::runtime_error("interp2d_cubic: invalid dimensions.");
        std::size_t nx = x.size(), ny = y.size();
        auto result = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>(xq.shape());
        const T* xd = x.data();
        const T* yd = y.data();
        for (std::size_t k = 0; k < xq.size(); ++k) {
            T xv = xq[k], yv = yq[k];
            std::ptrdiff_t ix = detail::find_interval(xd, nx, xv);
            std::ptrdiff_t iy = detail::find_interval(yd, ny, yv);
            T sum = 0;
            for (std::ptrdiff_t m = -1; m <= 2; ++m) {
                std::ptrdiff_t iym = detail::clamp_index(iy + m, ny);
                T s = (yv - yd[iym]) / (yd[1]-yd[0]); // assumes uniform
                T wy = detail::cubic_kernel(s - m);
                for (std::ptrdiff_t l = -1; l <= 2; ++l) {
                    std::ptrdiff_t ixl = detail::clamp_index(ix + l, nx);
                    T r = (xv - xd[ixl]) / (xd[1]-xd[0]);
                    T wx = detail::cubic_kernel(r - l);
                    sum += Z(iym, ixl) * wy * wx;
                }
            }
            result[k] = sum;
        }
        return result;
    }

    /**
     * 2D nearest-neighbor interpolation.
     */
    template <class T>
    inline auto interp2d_nearest(const xarray_container<uvector<T>>& x,
                                 const xarray_container<uvector<T>>& y,
                                 const xarray_container<uvector<T>>& Z,
                                 const xarray_container<uvector<T>>& xq,
                                 const xarray_container<uvector<T>>& yq) {
        std::size_t nx = x.size(), ny = y.size();
        auto result = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>(xq.shape());
        const T* xd = x.data(), *yd = y.data();
        for (std::size_t k = 0; k < xq.size(); ++k) {
            auto ix = std::min(static_cast<std::size_t>(std::distance(xd, std::lower_bound(xd, xd+nx, xq[k]))), nx-1);
            auto iy = std::min(static_cast<std::size_t>(std::distance(yd, std::lower_bound(yd, yd+ny, yq[k]))), ny-1);
            result[k] = Z(iy, ix);
        }
        return result;
    }

    /**
     * Generic 2D interpolation dispatch.
     */
    template <class T>
    inline auto interp2d(const xarray_container<uvector<T>>& x,
                         const xarray_container<uvector<T>>& y,
                         const xarray_container<uvector<T>>& Z,
                         const xarray_container<uvector<T>>& xq,
                         const xarray_container<uvector<T>>& yq,
                         method met = method::linear) {
        switch (met) {
            case method::nearest: return interp2d_nearest(x, y, Z, xq, yq);
            case method::linear: return interp2d_linear(x, y, Z, xq, yq);
            case method::cubic: return interp2d_cubic(x, y, Z, xq, yq);
            default: throw std::runtime_error("2D method not implemented.");
        }
    }

    /*********************************************
     * N-D Linear Grid Interpolation (recursive)
     *********************************************/

    namespace detail {
        // Recursive helper for N-D linear interpolation on a full grid.
        // Coords: array of axis coordinate vectors (size ndim each 1D)
        // Values: ND array of shape = (n0, n1, ...) with axes ordering (slowest dim=0?).
        // query: array of query points shape (num_points, ndim)
        template <class T>
        T interpn_linear_recursive(const std::vector<std::vector<T>>& axes,
                                   const T* values, // pointer to ND values, flat row-major
                                   const std::vector<std::size_t>& shape,
                                   const std::vector<T>& point,
                                   std::size_t dim) {
            if (dim == shape.size()) {
                // compute flat index
                std::size_t index = 0;
                std::size_t stride = 1;
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(shape.size())-1; d >= 0; --d) {
                    // point[d] is the index along that axis (integer index)
                    index += static_cast<std::size_t>(point[d]) * stride;
                    stride *= shape[d];
                }
                return values[index];
            }
            std::size_t n = shape[dim];
            const auto& ax = axes[dim];
            T x = point[dim];
            std::ptrdiff_t idx = find_interval(ax.data(), n, x);
            idx = clamp_index(idx, n);
            std::ptrdiff_t idx2 = std::min(idx+1, static_cast<std::ptrdiff_t>(n-1));
            T x1 = ax[idx], x2 = ax[idx2];
            T t = (x - x1) / (x2 - x1);
            std::vector<T> pt_low = point; pt_low[dim] = static_cast<T>(idx);
            std::vector<T> pt_high = point; pt_high[dim] = static_cast<T>(idx2);
            T v1 = interpn_linear_recursive(axes, values, shape, pt_low, dim+1);
            T v2 = interpn_linear_recursive(axes, values, shape, pt_high, dim+1);
            return v1 + t * (v2 - v1);
        }
    }

    /**
     * N-dimensional linear interpolation on a regular grid.
     * axes: vector of 1D arrays for each dimension (coordinate vectors).
     * values: ND array with shape matching the lengths of axes (row-major).
     * query_points: 2D array of shape (M, N) where M is number of points, N = ndim.
     */
    template <class T>
    inline auto interpn_linear(const std::vector<xarray_container<uvector<T>>>& axes,
                               const xarray_container<uvector<T>>& values,
                               const xarray_container<uvector<T>>& query_points) {
        std::size_t ndim = axes.size();
        if (ndim == 0 || values.dimension() != ndim)
            throw std::runtime_error("interpn_linear: axes count must match values dimension.");
        std::vector<std::size_t> shape = values.shape();
        // Build axes vectors
        std::vector<std::vector<T>> ax(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            if (axes[d].dimension() != 1 || axes[d].size() != shape[d])
                throw std::runtime_error("Axis length doesn't match values shape.");
            ax[d].assign(axes[d].data(), axes[d].data() + axes[d].size());
        }
        if (query_points.dimension() != 2 || query_points.shape()[1] != ndim)
            throw std::runtime_error("query_points must be (M, ndim).");
        std::size_t M = query_points.shape()[0];
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({M});
        for (std::size_t k = 0; k < M; ++k) {
            std::vector<T> point(ndim);
            for (std::size_t d = 0; d < ndim; ++d) point[d] = query_points(k, d);
            result[k] = detail::interpn_linear_recursive(ax, values.data(), shape, point, 0);
        }
        return result;
    }

    /**
     * Convenience 3D linear interpolation.
     */
    template <class T>
    inline auto interp3d_linear(const xarray_container<uvector<T>>& x,
                                const xarray_container<uvector<T>>& y,
                                const xarray_container<uvector<T>>& z,
                                const xarray_container<uvector<T>>& V,
                                const xarray_container<uvector<T>>& xq,
                                const xarray_container<uvector<T>>& yq,
                                const xarray_container<uvector<T>>& zq) {
        std::vector<xarray_container<uvector<T>>> axes = {x, y, z};
        std::size_t M = xq.size();
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> points({M, 3});
        for (std::size_t i = 0; i < M; ++i) {
            points(i,0) = xq[i]; points(i,1) = yq[i]; points(i,2) = zq[i];
        }
        return interpn_linear(axes, V, points);
    }

    /**
     * Convenience 4D linear interpolation.
     */
    template <class T>
    inline auto interp4d_linear(const xarray_container<uvector<T>>& x0,
                                const xarray_container<uvector<T>>& x1,
                                const xarray_container<uvector<T>>& x2,
                                const xarray_container<uvector<T>>& x3,
                                const xarray_container<uvector<T>>& V,
                                const xarray_container<uvector<T>>& q0,
                                const xarray_container<uvector<T>>& q1,
                                const xarray_container<uvector<T>>& q2,
                                const xarray_container<uvector<T>>& q3) {
        std::vector<xarray_container<uvector<T>>> axes = {x0, x1, x2, x3};
        std::size_t M = q0.size();
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> points({M, 4});
        for (std::size_t i = 0; i < M; ++i) {
            points(i,0)=q0[i]; points(i,1)=q1[i]; points(i,2)=q2[i]; points(i,3)=q3[i];
        }
        return interpn_linear(axes, V, points);
    }

} // namespace interpolate
} // namespace xt

#endif // XTENSOR_XINTERPOLATE_HPP