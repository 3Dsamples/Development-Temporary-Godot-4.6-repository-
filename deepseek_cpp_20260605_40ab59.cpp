//File 0030 : core/xintegrate.hpp
//Numerical integration: trapezoidal, Simpson, Romberg, Gauss-Legendre, adaptive quadrature, Monte Carlo for 1D and ND with SIMD evaluation.
#ifndef XTENSOR_XINTEGRATE_HPP
#define XTENSOR_XINTEGRATE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
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
#include "xreducer.hpp"
#include "xrandom.hpp"
#include "xsort.hpp"
#include "xstatistics.hpp"

namespace xt {
namespace integrate {

    using namespace std::complex_literals;

    /*********************************************
     * 1D Trapezoidal Rule
     *********************************************/
    /**
     * Composite trapezoidal rule on a uniform grid given y values and step dx.
     */
    template <class T>
    inline T trapz_uniform(const T* y, std::size_t n, T dx) {
        if (n < 2) return T(0);
        T sum = T(0.5) * (y[0] + y[n-1]);
        // SIMD accumulation for interior points
        if constexpr (is_simd_enabled_v<T>) {
            using batch = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = batch::size;
            std::size_t interior = n - 2;
            std::size_t vec_count = interior / simd_size;
            batch vsum(0);
            const T* start = y + 1;
            for (std::size_t i = 0; i < vec_count; ++i) {
                batch v = batch::load_unaligned(start + i * simd_size);
                vsum = vsum + v;
            }
            T tmp[simd_size];
            vsum.store_unaligned(tmp);
            for (std::size_t k = 0; k < simd_size; ++k) sum += tmp[k];
            for (std::size_t i = vec_count * simd_size + 1; i < n - 1; ++i) sum += y[i];
        } else {
            for (std::size_t i = 1; i < n - 1; ++i) sum += y[i];
        }
        return sum * dx;
    }

    /**
     * Composite trapezoidal rule for non-uniform grid (x, y arrays).
     */
    template <class T>
    inline T trapz_nonuniform(const T* x, const T* y, std::size_t n) {
        if (n < 2) return T(0);
        T sum = 0;
        for (std::size_t i = 1; i < n; ++i) {
            T h = x[i] - x[i-1];
            sum += h * (y[i-1] + y[i]);
        }
        return T(0.5) * sum;
    }

    /**
     * Trapezoidal rule for an expression evaluated on a uniform grid from a to b.
     */
    template <class Func>
    inline auto trapz(Func f, double a, double b, std::size_t n = 1000) {
        using T = decltype(f(a));
        T dx = static_cast<T>(b - a) / n;
        std::vector<T> y(n + 1);
        for (std::size_t i = 0; i <= n; ++i) {
            T x = a + i * dx;
            y[i] = f(x);
        }
        return trapz_uniform(y.data(), y.size(), dx);
    }

    /*********************************************
     * 1D Simpson's Rule
     *********************************************/
    /**
     * Composite Simpson's 1/3 rule on uniform grid (n must be even).
     */
    template <class T>
    inline T simpson_uniform(const T* y, std::size_t n, T dx) {
        if (n < 2 || n % 2 != 0)
            throw std::runtime_error("Simpson rule requires even number of intervals (odd number of points).");
        T sum = y[0] + y[n-1];
        for (std::size_t i = 1; i < n - 1; i += 2)
            sum += T(4) * y[i];
        for (std::size_t i = 2; i < n - 2; i += 2)
            sum += T(2) * y[i];
        return sum * dx / T(3);
    }

    /**
     * Simpson's rule for a function f from a to b with n intervals.
     */
    template <class Func>
    inline auto simpson(Func f, double a, double b, std::size_t n = 1000) {
        if (n % 2 != 0) n += 1;
        using T = decltype(f(a));
        T dx = static_cast<T>(b - a) / n;
        std::vector<T> y(n + 1);
        for (std::size_t i = 0; i <= n; ++i) {
            T x = a + i * dx;
            y[i] = f(x);
        }
        return simpson_uniform(y.data(), y.size(), dx);
    }

    /*********************************************
     * Romberg Integration (extrapolated trapezoidal)
     *********************************************/
    /**
     * Romberg integration with extrapolation to accelerate trapezoidal rule.
     */
    template <class Func>
    inline auto romberg(Func f, double a, double b, std::size_t max_order = 10, double tol = 1e-12) {
        using T = decltype(f(a));
        std::vector<std::vector<T>> R(max_order + 1);
        T h = static_cast<T>(b - a);
        R[0].push_back(T(0.5) * h * (f(a) + f(b)));

        for (std::size_t k = 1; k <= max_order; ++k) {
            // Trapezoidal with 2^k intervals
            std::size_t n = static_cast<std::size_t>(1) << k;
            T sum = 0;
            T dh = h / n;
            for (std::size_t i = 1; i < n; i += 2) {
                T x = a + i * dh;
                sum += f(x);
            }
            T trap = T(0.5) * R[k-1][0] + dh * sum;
            R[k].push_back(trap);

            // Richardson extrapolation
            for (std::size_t j = 1; j <= k; ++j) {
                T factor = std::pow(T(4), static_cast<T>(j));
                T extrap = (factor * R[k][j-1] - R[k-1][j-1]) / (factor - T(1));
                R[k].push_back(extrap);
            }
            if (k >= 1 && std::abs(R[k][k] - R[k-1][k-1]) < tol)
                return R[k][k];
        }
        return R[max_order][max_order];
    }

    /*********************************************
     * Gauss-Legendre Quadrature
     *********************************************/
    namespace detail {
        // Nodes and weights for Gauss-Legendre quadrature of order n.
        // Generated via Golub-Welsch or precomputed for common orders.
        template <class T>
        void gauss_legendre_nodes_weights(std::size_t n, std::vector<T>& nodes, std::vector<T>& weights) {
            // Use Golub-Welsch algorithm to compute nodes and weights from recurrence.
            nodes.resize(n);
            weights.resize(n);
            // Jacobi matrix eigenvalues give nodes; eigenvectors give weights.
            // For simplicity, we'll precompute for typical orders; for general n use iterative method.
            // We implement a Newton method for Legendre polynomial roots.
            for (std::size_t i = 0; i < n; ++i) {
                // Initial guess for i-th root
                T x = std::cos(T(xt::numeric_constants<double>::PI * (i + 0.75) / (n + 0.5)));
                T x_old;
                do {
                    x_old = x;
                    // Legendre polynomial P_n(x) and derivative using recurrence
                    T p0 = 1, p1 = x;
                    for (std::size_t k = 2; k <= n; ++k) {
                        T pk = ((2*k - 1) * x * p1 - (k - 1) * p0) / k;
                        p0 = p1;
                        p1 = pk;
                    }
                    T pn = p1;
                    T pn_deriv = n * (x * pn - p0) / (x*x - 1);
                    x = x_old - pn / pn_deriv;
                } while (std::abs(x - x_old) > 1e-15);
                nodes[i] = x;
                // Weight
                T p0_w = 1, p1_w = x;
                for (std::size_t k = 2; k <= n - 1; ++k) {
                    T pk = ((2*k - 1) * x * p1_w - (k - 1) * p0_w) / k;
                    p0_w = p1_w;
                    p1_w = pk;
                }
                T pn_minus1 = p1_w;
                weights[i] = 2 / ((1 - x*x) * (n * pn_minus1) * (n * pn_minus1));
            }
        }
    }

    /**
     * Gauss-Legendre quadrature of order n on [a, b].
     */
    template <class Func>
    inline auto gauss_legendre(Func f, double a, double b, std::size_t n = 10) {
        using T = decltype(f(a));
        std::vector<T> nodes, weights;
        detail::gauss_legendre_nodes_weights(n, nodes, weights);
        T mid = (b + a) / 2;
        T half = (b - a) / 2;
        T sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            T x = mid + half * nodes[i];
            sum += weights[i] * f(x);
        }
        return half * sum;
    }

    /*********************************************
     * Adaptive Quadrature (Gauss-Kronrod)
     *********************************************/
    /**
     * Adaptive Gauss-Kronrod (G7-K15) quadrature with error estimation.
     */
    template <class Func>
    inline auto adaptive_gauss_kronrod(Func f, double a, double b, double tol = 1e-10, std::size_t max_depth = 20) {
        using T = decltype(f(a));
        std::function<T(double, double, std::size_t)> recursive;
        recursive = [&](double left, double right, std::size_t depth) -> T {
            T mid = (left + right) / 2;
            T half = (right - left) / 2;
            // G7 nodes and weights on [-1,1]
            constexpr double g7_nodes[7] = {0.0, 0.4058451513773972, -0.4058451513773972,
                                            0.7415311855993945, -0.7415311855993945,
                                            0.9491079123427585, -0.9491079123427585};
            constexpr double g7_weights[7] = {0.4179591836734694, 0.3818300505051189, 0.3818300505051189,
                                              0.2797053914892766, 0.2797053914892766,
                                              0.1294849661688697, 0.1294849661688697};
            // K15 nodes (includes G7) and weights
            constexpr double k15_nodes[15] = {0.0, 0.2077849550078985, -0.2077849550078985,
                                              0.4058451513773972, -0.4058451513773972,
                                              0.5860872354676911, -0.5860872354676911,
                                              0.7415311855993945, -0.7415311855993945,
                                              0.8648644233597691, -0.8648644233597691,
                                              0.9491079123427585, -0.9491079123427585,
                                              0.9914553711208126, -0.9914553711208126};
            constexpr double k15_weights[15] = {0.2094821410847278, 0.2044329400752989, 0.2044329400752989,
                                                0.1903505780647854, 0.1903505780647854,
                                                0.1690047266392679, 0.1690047266392679,
                                                0.1406532597155259, 0.1406532597155259,
                                                0.1047900103222502, 0.1047900103222502,
                                                0.0630920926299786, 0.0630920926299786,
                                                0.0229353220105292, 0.0229353220105292};
            T g7 = 0, k15 = 0;
            for (int i = 0; i < 7; ++i) {
                T x = mid + half * g7_nodes[i];
                g7 += g7_weights[i] * f(x);
            }
            for (int i = 0; i < 15; ++i) {
                T x = mid + half * k15_nodes[i];
                k15 += k15_weights[i] * f(x);
            }
            T est_g7 = half * g7;
            T est_k15 = half * k15;
            T error = std::abs(est_k15 - est_g7);
            if (error < tol || depth >= max_depth)
                return est_k15;
            T left_val = recursive(left, mid, depth + 1);
            T right_val = recursive(mid, right, depth + 1);
            return left_val + right_val;
        };
        return recursive(a, b, 0);
    }

    /*********************************************
     * Double Integral (2D)
     *********************************************/
    /**
     * Double integral over rectangular domain [ax,bx] x [ay,by] using iterated 1D integration.
     */
    template <class Func, class Integrator1D>
    inline auto double_integral(Func f, double ax, double bx, double ay, double by,
                                std::size_t nx, std::size_t ny, Integrator1D integrator) {
        using T = decltype(f(ax, ay));
        // Integrate over y for each fixed x using the outer integrator
        auto integrand_over_y = [&](double x) {
            auto fy = [&](double y) { return f(x, y); };
            return integrator(fy, ay, by, ny);
        };
        auto integrand_x = [&](double x) { return integrand_over_y(x); };
        return integrator(integrand_x, ax, bx, nx);
    }

    /**
     * Convenience 2D Simpson-Simpson integration.
     */
    template <class Func>
    inline auto simpson2d(Func f, double ax, double bx, double ay, double by,
                          std::size_t nx = 100, std::size_t ny = 100) {
        auto simpson1d = [](auto func, double a, double b, std::size_t n) {
            return simpson(func, a, b, n);
        };
        return double_integral(f, ax, bx, ay, by, nx, ny, simpson1d);
    }

    /*********************************************
     * Triple Integral (3D)
     *********************************************/
    /**
     * Triple integral over box [ax,bx] x [ay,by] x [az,bz] using iterated integration.
     */
    template <class Func, class Integrator1D>
    inline auto triple_integral(Func f, double ax, double bx, double ay, double by, double az, double bz,
                                std::size_t nx, std::size_t ny, std::size_t nz, Integrator1D integrator) {
        using T = decltype(f(ax, ay, az));
        auto integrand_over_yz = [&](double x) {
            auto fyz = [&](double y, double z) { return f(x, y, z); };
            return double_integral(fyz, ay, by, az, bz, ny, nz, integrator);
        };
        auto integrand_x = [&](double x) { return integrand_over_yz(x); };
        return integrator(integrand_x, ax, bx, nx);
    }

    /**
     * Convenience 3D Simpson integration.
     */
    template <class Func>
    inline auto simpson3d(Func f, double ax, double bx, double ay, double by, double az, double bz,
                          std::size_t nx = 50, std::size_t ny = 50, std::size_t nz = 50) {
        auto simpson1d = [](auto func, double a, double b, std::size_t n) {
            return simpson(func, a, b, n);
        };
        return triple_integral(f, ax, bx, ay, by, az, bz, nx, ny, nz, simpson1d);
    }

    /*********************************************
     * Monte Carlo Integration (N-D)
     *********************************************/
    /**
     * Monte Carlo integration over a hyper-rectangle defined by lower/upper bounds.
     */
    template <class Func>
    inline auto monte_carlo(Func f, const std::vector<double>& lower, const std::vector<double>& upper,
                            std::size_t samples = 100000) {
        std::size_t ndim = lower.size();
        if (ndim != upper.size()) throw std::runtime_error("Bounds size mismatch.");
        double volume = 1.0;
        for (std::size_t d = 0; d < ndim; ++d) volume *= (upper[d] - lower[d]);
        double sum = 0;
        double sum_sq = 0;
        auto& eng = random::detail::get_global_engine();
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> point(ndim);
        for (std::size_t k = 0; k < samples; ++k) {
            for (std::size_t d = 0; d < ndim; ++d)
                point[d] = lower[d] + dist(eng) * (upper[d] - lower[d]);
            double val = f(point);
            sum += val;
            sum_sq += val * val;
        }
        double mean = sum / samples;
        double variance = (sum_sq / samples) - (mean * mean);
        double error = std::sqrt(variance / samples);
        // Return estimate and error as pair
        return std::make_pair(volume * mean, volume * error);
    }

    /**
     * Monte Carlo integration over unit hypercube [0,1]^n.
     */
    template <class Func>
    inline auto monte_carlo_unit(Func f, std::size_t ndim, std::size_t samples = 100000) {
        std::vector<double> lower(ndim, 0.0), upper(ndim, 1.0);
        return monte_carlo(f, lower, upper, samples);
    }

    /*********************************************
     * Cubature: integration on a simplex (2D/3D)
     *********************************************/
    /**
     * Integrate over a triangle using symmetric quadrature rules.
     */
    template <class Func>
    inline auto triangle_quadrature(Func f, double x1, double y1, double x2, double y2,
                                    double x3, double y3, std::size_t order = 3) {
        using T = decltype(f(x1, y1));
        // 7-point quadrature rule for triangles (order 5)
        constexpr double nodes_bary[7][3] = {
            {1.0/3.0, 1.0/3.0, 1.0/3.0},
            {0.797426985353087, 0.101286507323456, 0.101286507323456},
            {0.101286507323456, 0.797426985353087, 0.101286507323456},
            {0.101286507323456, 0.101286507323456, 0.797426985353087},
            {0.059715871789770, 0.470142064105115, 0.470142064105115},
            {0.470142064105115, 0.059715871789770, 0.470142064105115},
            {0.470142064105115, 0.470142064105115, 0.059715871789770}
        };
        constexpr double weights[7] = {0.225, 0.125939180544827, 0.125939180544827,
                                       0.125939180544827, 0.132394152788506,
                                       0.132394152788506, 0.132394152788506};
        double area = std::abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2.0;
        T sum = 0;
        for (int i = 0; i < 7; ++i) {
            double bx = nodes_bary[i][0], by = nodes_bary[i][1], bz = nodes_bary[i][2];
            double px = bx * x1 + by * x2 + bz * x3;
            double py = bx * y1 + by * y2 + bz * y3;
            sum += static_cast<T>(weights[i]) * f(px, py);
        }
        return area * sum;
    }

    /*********************************************
     * Cumulative integration
     *********************************************/
    /**
     * Cumulative trapezoidal integration returning an array of same length as input.
     */
    template <class T>
    inline auto cumtrapz(const T* x, const T* y, std::size_t n) {
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n}, T(0));
        for (std::size_t i = 1; i < n; ++i) {
            T h = x[i] - x[i-1];
            result[i] = result[i-1] + T(0.5) * h * (y[i-1] + y[i]);
        }
        return result;
    }

    /**
     * Cumulative trapezoidal integration with uniform spacing dx.
     */
    template <class E>
    inline auto cumtrapz(const E& y, double dx = 1.0) {
        auto arr = xt::eval(y);
        std::size_t n = arr.size();
        using T = typename std::decay_t<E>::value_type;
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n}, T(0));
        for (std::size_t i = 1; i < n; ++i) {
            result[i] = result[i-1] + T(0.5) * dx * (arr[i-1] + arr[i]);
        }
        return result;
    }

} // namespace integrate
} // namespace xt

#endif // XTENSOR_XINTEGRATE_HPP