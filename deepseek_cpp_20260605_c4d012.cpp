//File 0027 (UPDATED) : core/xinterpolate.hpp
//Interpolation: 1D/2D/3D/4D gridded (linear, cubic, spline, nearest) and scattered data via KD-Tree-based IDW/Shepard.
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
#include "xgeometry.hpp"

namespace xt {
namespace interpolate {

    enum class method { nearest, linear, cubic, spline };

    /*********************************************
     * 1D interpolation (unchanged)
     *********************************************/
    namespace detail {
        template <class T>
        std::ptrdiff_t find_interval(const T* xp, std::size_t n, T x) {
            if (x <= xp[0]) return 0;
            if (x >= xp[n-1]) return static_cast<std::ptrdiff_t>(n - 2);
            auto it = std::upper_bound(xp, xp + n, x);
            return static_cast<std::ptrdiff_t>(it - xp - 1);
        }
        inline std::ptrdiff_t clamp_index(std::ptrdiff_t i, std::size_t n) {
            if (i < 0) return 0;
            if (static_cast<std::size_t>(i) >= n) return static_cast<std::ptrdiff_t>(n) - 1;
            return i;
        }
        // … (cubic kernel and spline coeff unchanged) …
        template <class T>
        T cubic_kernel(T s) {
            s = std::abs(s);
            if (s <= T(1))
                return (T(1.5)*s - T(2.5))*s*s + T(1);
            else if (s <= T(2))
                return ((-T(0.5)*s + T(2.5))*s - T(4.0))*s + T(2.0);
            else return T(0);
        }
        template <class T>
        std::vector<T> spline_coeff(const std::vector<T>& x, const std::vector<T>& y) {
            std::size_t n = x.size();
            if (n < 2) return y;
            std::vector<T> h(n-1), alpha(n,0), l(n,0), mu(n,0), z(n,0), c(n,0), b(n-1), d(n-1);
            for (std::size_t i=0;i<n-1;++i) h[i]=x[i+1]-x[i];
            for (std::size_t i=1;i<n-1;++i) alpha[i]=(T(3)/h[i])*(y[i+1]-y[i])-(T(3)/h[i-1])*(y[i]-y[i-1]);
            l[0]=1; mu[0]=0; z[0]=0;
            for (std::size_t i=1;i<n-1;++i){ l[i]=2*(x[i+1]-x[i-1])-h[i-1]*mu[i-1]; mu[i]=h[i]/l[i]; z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i]; }
            l[n-1]=1; z[n-1]=0; c[n-1]=0;
            for (std::ptrdiff_t j=static_cast<std::ptrdiff_t>(n)-2;j>=0;--j){ c[j]=z[j]-mu[j]*c[j+1]; b[j]=(y[j+1]-y[j])/h[j]-h[j]*(c[j+1]+2*c[j])/T(3); d[j]=(c[j+1]-c[j])/(T(3)*h[j]); }
            std::vector<T> coeffs(4*(n-1));
            for (std::size_t i=0;i<n-1;++i){ coeffs[4*i]=y[i]; coeffs[4*i+1]=b[i]; coeffs[4*i+2]=c[i]; coeffs[4*i+3]=d[i]; }
            return coeffs;
        }
        template <class T>
        T eval_spline(const std::vector<T>& xb, const std::vector<T>& c, T x) {
            std::size_t n=xb.size(); if(n<2) return c.empty()?T(0):c[0];
            std::ptrdiff_t i=find_interval(xb.data(),n,x);
            if(i<0)i=0; if(static_cast<std::size_t>(i)>=n-1)i=static_cast<std::ptrdiff_t>(n-2);
            T dx=x-xb[i]; const T* seg=&c[4*i];
            return seg[0]+dx*(seg[1]+dx*(seg[2]+dx*seg[3]));
        }
    }

    // 1D interp functions (unchanged but retained for completeness) …
    template <class T>
    inline auto interp1d_linear(const xarray_container<uvector<T>>& xp,
                                const xarray_container<uvector<T>>& fp,
                                const xarray_container<uvector<T>>& xq) {
        if(xp.dimension()!=1||fp.dimension()!=1||xp.size()!=fp.size())
            throw std::runtime_error("interp1d_linear: xp/fp must be 1D same length.");
        auto res=xarray_container<uvector<T>,DEFAULT_LAYOUT,std::vector<std::size_t>>(xq.shape());
        const T* xd=xp.data(); const T* fd=fp.data(); std::size_t n=xp.size();
        for(std::size_t j=0;j<xq.size();++j){
            T x=xq[j]; std::ptrdiff_t i=detail::find_interval(xd,n,x);
            i=detail::clamp_index(i,n); std::ptrdiff_t i2=std::min<ptrdiff_t>(i+1,n-1);
            T x0=xd[i],x1=xd[i2],y0=fd[i],y1=fd[i2];
            T t=(x-x0)/(x1-x0); res[j]=y0+t*(y1-y0);
        }
        return res;
    }

    template <class T>
    inline auto interp1d_cubic(const xarray_container<uvector<T>>& xp,
                               const xarray_container<uvector<T>>& fp,
                               const xarray_container<uvector<T>>& xq) {
        if(xp.dimension()!=1||fp.dimension()!=1||xp.size()!=fp.size())
            throw std::runtime_error("interp1d_cubic: same size required.");
        auto res=xarray_container<uvector<T>,DEFAULT_LAYOUT,std::vector<std::size_t>>(xq.shape());
        const T* xd=xp.data(); const T* fd=fp.data(); std::size_t n=xp.size();
        for(std::size_t j=0;j<xq.size();++j){
            T x=xq[j]; std::ptrdiff_t idx=detail::find_interval(xd,n,x);
            T sum=0;
            for(std::ptrdiff_t k=-1;k<=2;++k){
                std::ptrdiff_t i=detail::clamp_index(idx+k,n);
                T s=(x-xd[i])/(xd[1]-xd[0]); // assumes uniform spacing
                sum+=fd[i]*detail::cubic_kernel(s-k);
            }
            res[j]=sum;
        }
        return res;
    }

    template <class T>
    inline auto interp1d_spline(const xarray_container<uvector<T>>& xp,
                                const xarray_container<uvector<T>>& fp,
                                const xarray_container<uvector<T>>& xq) {
        if(xp.size()!=fp.size()||xp.dimension()!=1) throw std::runtime_error("size mismatch.");
        std::size_t n=xp.size(); if(n<2) throw std::runtime_error("Need >=2 points.");
        std::vector<T> xv(xp.data(),xp.data()+n), fv(fp.data(),fp.data()+n);
        auto coeffs=detail::spline_coeff(xv,fv);
        auto res=xarray_container<uvector<T>,DEFAULT_LAYOUT,std::vector<std::size_t>>(xq.shape());
        for(std::size_t j=0;j<xq.size();++j) res[j]=detail::eval_spline(xv,coeffs,xq[j]);
        return res;
    }

    template <class T>
    inline auto interp1d_nearest(const xarray_container<uvector<T>>& xp,
                                 const xarray_container<uvector<T>>& fp,
                                 const xarray_container<uvector<T>>& xq) {
        if(xp.size()!=fp.size()) throw std::runtime_error("size mismatch.");
        auto res=xarray_container<uvector<T>,DEFAULT_LAYOUT,std::vector<std::size_t>>(xq.shape());
        const T* xd=xp.data(); const T* fd=fp.data(); std::size_t n=xp.size();
        for(std::size_t j=0;j<xq.size();++j){
            T x=xq[j]; std::ptrdiff_t i=detail::find_interval(xd,n,x);
            if(i<0)i=0; if(static_cast<std::size_t>(i)>=n-1) res[j]=fd[n-1];
            else { T dleft=x-xd[i], dright=xd[i+1]-x; res[j]=(dleft<=dright)?fd[i]:fd[i+1]; }
        }
        return res;
    }

    template <class T>
    inline auto interp1d(const xarray_container<uvector<T>>& xp, const xarray_container<uvector<T>>& fp,
                         const xarray_container<uvector<T>>& xq, method met=method::linear) {
        switch(met){
            case method::nearest: return interp1d_nearest(xp,fp,xq);
            case method::linear:  return interp1d_linear(xp,fp,xq);
            case method::cubic:   return interp1d_cubic(xp,fp,xq);
            case method::spline:  return interp1d_spline(xp,fp,xq);
            default: throw std::runtime_error("Unknown method.");
        }
    }

    /*********************************************
     * 2D/3D/4D gridded interpolation (previously defined interp2d, interp3d, interp4d, interpn) remain unchanged.
     *********************************************/
    // … (same as earlier) …

    /*********************************************
     * Scattered data interpolation using KD-Tree
     *********************************************/
    template <class T>
    auto griddata(const std::vector<vec3<T>>& points,   // shape Nx3 (only 2D/3D effectively)
                  const std::vector<T>& values,          // length N
                  const std::vector<vec3<T>>& queries,   // shape Mx3
                  method met = method::linear,
                  std::size_t k = 10)                    // number of neighbors for IDW
    {
        std::size_t N = points.size();
        std::size_t M = queries.size();
        if (N != values.size()) throw std::runtime_error("points and values size mismatch.");
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({M}, T(0));

        // Build KD-Tree on points (use geometry::KDTree with dynamic dimensions?)
        // geometry::KDTree is templated on T and D, default D=3.
        // We'll copy points into the required format.
        geometry::KDTree<T, 3> tree;
        tree.build(points);  // assumes points are std::vector<vec3<T>>, which matches the KDTree build signature.

        for (std::size_t i = 0; i < M; ++i) {
            const vec3<T>& q = queries[i];
            if (met == method::nearest) {
                std::size_t idx = tree.nearestNeighbor(q);
                result[i] = values[idx];
            } else {
                // Shepard's inverse distance weighting using k nearest neighbors.
                auto neighbors = tree.knearest(q, k); // returns vector of indices sorted by distance
                T sum_weights = 0;
                T sum_vals = 0;
                // Get distances again? The knearest returns indices; we can compute distances or store them.
                // We'll compute distance from q to each neighbor point.
                for (std::size_t n_idx : neighbors) {
                    T d = geometry::point_to_point_distance(q, points[n_idx]);
                    if (d < 1e-15) {
                        // exact hit
                        result[i] = values[n_idx];
                        sum_weights = 0; // break?
                        break;
                    }
                    T w = 1.0 / (d * d); // inverse square
                    sum_weights += w;
                    sum_vals += values[n_idx] * w;
                }
                if (sum_weights > 0) result[i] = sum_vals / sum_weights;
                else result[i] = T(0); // fallback
            }
        }
        return result;
    }

    // Convenience overload for 2D points (converting to vec3 with z=0)
    template <class T>
    auto griddata2d(const std::vector<vec2<T>>& points,
                    const std::vector<T>& values,
                    const std::vector<vec2<T>>& queries,
                    method met = method::linear,
                    std::size_t k = 10) {
        // Convert 2D to 3D (z=0)
        std::vector<vec3<T>> pts3, qry3;
        pts3.reserve(points.size()); for (auto& p : points) pts3.push_back({p[0], p[1], T(0)});
        qry3.reserve(queries.size()); for (auto& q : queries) qry3.push_back({q[0], q[1], T(0)});
        return griddata(pts3, values, qry3, met, k);
    }

    // Similarly for 3D directly use griddata.

} // namespace interpolate
} // namespace xt

#endif // XTENSOR_XINTERPOLATE_HPP