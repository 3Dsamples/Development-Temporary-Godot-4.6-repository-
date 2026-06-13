//File 0029 : core/xoptimize.hpp
//Numerical optimization: gradient descent, conjugate gradient, Newton, L-BFGS, least squares solvers with SIMD-accelerated line search.
#ifndef XTENSOR_XOPTIMIZE_HPP
#define XTENSOR_XOPTIMIZE_HPP

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
#include "xreducer.hpp"
#include "xlinalg.hpp"
#include "xsort.hpp"
#include "xstatistics.hpp"

namespace xt {
namespace optimize {

    using namespace std::complex_literals;

    /*********************************************
     * Scalar function wrapper (for objective and gradient)
     *********************************************/
    template <class T>
    using scalar_function = std::function<T(const xarray_container<uvector<T>>&)>;
    template <class T>
    using gradient_function = std::function<xarray_container<uvector<T>>(const xarray_container<uvector<T>>&)>;

    /*********************************************
     * Line search (backtracking Armijo)
     *********************************************/
    namespace detail {
        template <class T>
        T armijo_backtrack(const xarray_container<uvector<T>>& x,
                           const xarray_container<uvector<T>>& dir,
                           scalar_function<T> f, gradient_function<T> grad,
                           T alpha_init = 1.0, T rho = 0.5, T c = 1e-4) {
            T f0 = f(x);
            auto g0 = grad(x);
            T slope = 0;
            for (std::size_t i = 0; i < g0.size(); ++i) slope += g0[i] * dir[i];
            T alpha = alpha_init;
            while (alpha > 1e-16) {
                auto x_new = x + alpha * dir;
                T f_new = f(x_new);
                if (f_new <= f0 + c * alpha * slope) break;
                alpha *= rho;
            }
            return alpha;
        }

        template <class T>
        T wolfe_line_search(const xarray_container<uvector<T>>& x,
                            const xarray_container<uvector<T>>& dir,
                            scalar_function<T> f, gradient_function<T> grad,
                            T alpha_init = 1.0, T c1 = 1e-4, T c2 = 0.9, std::size_t max_iter = 20) {
            T f0 = f(x);
            auto g0 = grad(x);
            T slope = 0;
            for (std::size_t i = 0; i < g0.size(); ++i) slope += g0[i] * dir[i];
            T alpha = alpha_init;
            T alpha_prev = 0;
            T f_prev = f0;
            for (std::size_t iter = 0; iter < max_iter; ++iter) {
                auto x_new = x + alpha * dir;
                T f_new = f(x_new);
                if (f_new > f0 + c1 * alpha * slope || (iter > 0 && f_new >= f_prev)) {
                    return alpha_prev;
                }
                auto g_new = grad(x_new);
                T new_slope = 0;
                for (std::size_t i = 0; i < g_new.size(); ++i) new_slope += g_new[i] * dir[i];
                if (std::abs(new_slope) <= -c2 * slope) {
                    return alpha;
                }
                if (new_slope >= 0) {
                    return alpha;
                }
                alpha_prev = alpha;
                f_prev = f_new;
                alpha *= 2;
            }
            return alpha;
        }
    }

    /*********************************************
     * Gradient Descent
     *********************************************/
    /**
     * Gradient descent optimizer with backtracking line search.
     */
    template <class T>
    inline auto gradient_descent(scalar_function<T> f, gradient_function<T> grad,
                                 xarray_container<uvector<T>> x0,
                                 T tol = 1e-6, std::size_t max_iter = 1000) {
        xarray_container<uvector<T>> x = x0;
        for (std::size_t k = 0; k < max_iter; ++k) {
            auto g = grad(x);
            T norm_g = xt::norm::norm_l2(g);
            if (norm_g < tol) break;
            auto dir = -g;
            T alpha = detail::armijo_backtrack(x, dir, f, grad);
            x = x + alpha * dir;
        }
        return x;
    }

    /*********************************************
     * Conjugate Gradient (Fletcher-Reeves)
     *********************************************/
    /**
     * Nonlinear conjugate gradient (Fletcher-Reeves) optimizer.
     */
    template <class T>
    inline auto conjugate_gradient(scalar_function<T> f, gradient_function<T> grad,
                                   xarray_container<uvector<T>> x0,
                                   T tol = 1e-6, std::size_t max_iter = 1000) {
        xarray_container<uvector<T>> x = x0;
        auto g = grad(x);
        auto d = -g;
        for (std::size_t k = 0; k < max_iter; ++k) {
            T norm_g = xt::norm::norm_l2(g);
            if (norm_g < tol) break;
            T alpha = detail::wolfe_line_search(x, d, f, grad);
            auto x_new = x + alpha * d;
            auto g_new = grad(x_new);
            T beta = (xt::norm::norm_l2(g_new) * xt::norm::norm_l2(g_new)) / (norm_g * norm_g);
            d = -g_new + beta * d;
            x = x_new;
            g = g_new;
        }
        return x;
    }

    /*********************************************
     * Newton's method (with Hessian)
     *********************************************/
    /**
     * Newton's method using Hessian inverse. Hessian provided as function.
     */
    template <class T>
    inline auto newton_method(scalar_function<T> f, gradient_function<T> grad,
                              std::function<xarray_container<uvector<T>>(const xarray_container<uvector<T>>&)> hess,
                              xarray_container<uvector<T>> x0,
                              T tol = 1e-6, std::size_t max_iter = 50) {
        xarray_container<uvector<T>> x = x0;
        for (std::size_t k = 0; k < max_iter; ++k) {
            auto g = grad(x);
            T norm_g = xt::norm::norm_l2(g);
            if (norm_g < tol) break;
            auto H = hess(x);
            auto p = xt::linalg::solve(H, -g);
            T alpha = detail::armijo_backtrack(x, p, f, grad);
            x = x + alpha * p;
        }
        return x;
    }

    /*********************************************
     * L-BFGS (Limited-memory BFGS)
     *********************************************/
    /**
     * L-BFGS optimizer with memory m.
     */
    template <class T>
    inline auto lbfgs(scalar_function<T> f, gradient_function<T> grad,
                      xarray_container<uvector<T>> x0, std::size_t m = 5,
                      T tol = 1e-6, std::size_t max_iter = 1000) {
        std::size_t n = x0.size();
        xarray_container<uvector<T>> x = x0;
        auto g = grad(x);
        std::vector<xarray_container<uvector<T>>> s_list, y_list;
        std::vector<T> rho_list;
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            T norm_g = xt::norm::norm_l2(g);
            if (norm_g < tol) break;

            // Two-loop recursion to compute search direction p = -H * g
            auto q = g;
            std::vector<T> alpha_list;
            // First loop
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(s_list.size())-1; i >= 0; --i) {
                T rho = rho_list[i];
                T alpha = 0;
                for (std::size_t j = 0; j < n; ++j) alpha += s_list[i][j] * q[j];
                alpha *= rho;
                alpha_list.push_back(alpha);
                q = q - alpha * y_list[i];
            }
            // Initial Hessian approximation: scaled identity using most recent y*s / y*y
            auto r = q;
            if (!s_list.empty()) {
                T gamma = 0;
                auto& sy = s_list.back();
                auto& yy = y_list.back();
                T dot_yy = 0, dot_sy = 0;
                for (std::size_t j = 0; j < n; ++j) { dot_yy += yy[j]*yy[j]; dot_sy += sy[j]*yy[j]; }
                if (dot_yy != 0) gamma = dot_sy / dot_yy;
                for (std::size_t j = 0; j < n; ++j) r[j] = gamma * r[j];
            }
            // Second loop
            for (std::size_t i = 0; i < s_list.size(); ++i) {
                T rho = rho_list[i];
                T beta = 0;
                auto& si = s_list[i];
                auto& yi = y_list[i];
                for (std::size_t j = 0; j < n; ++j) beta += yi[j] * r[j];
                beta *= rho;
                r = r + (alpha_list[alpha_list.size()-1-i] - beta) * si;
            }
            auto p = -r;

            T alpha_step = detail::wolfe_line_search(x, p, f, grad);
            auto s = alpha_step * p;
            auto x_new = x + s;
            auto g_new = grad(x_new);
            auto y = g_new - g;

            // Store s and y with limited memory
            if (s_list.size() >= m) {
                s_list.erase(s_list.begin());
                y_list.erase(y_list.begin());
                rho_list.erase(rho_list.begin());
            }
            s_list.push_back(s);
            y_list.push_back(y);
            T dot_sy = 0;
            for (std::size_t j = 0; j < n; ++j) dot_sy += s[j] * y[j];
            rho_list.push_back(dot_sy != 0 ? 1.0 / dot_sy : 0.0);

            x = x_new;
            g = g_new;
        }
        return x;
    }

    /*********************************************
     * Least Squares Solvers
     *********************************************/
    /**
     * Solve linear least squares min ||Ax - b||₂ using normal equations (A^T A x = A^T b).
     */
    template <class T>
    inline auto least_squares_normal(const xarray_container<uvector<T>>& A,
                                     const xarray_container<uvector<T>>& b) {
        auto At = xt::transpose(A);
        auto AtA = xt::linalg::matmul(At, A);
        auto Atb = xt::linalg::matmul(At, b);
        return xt::linalg::solve(AtA, Atb);
    }

    /**
     * Solve linear least squares using QR decomposition (Gram-Schmidt).
     */
    template <class T>
    inline auto least_squares_qr(const xarray_container<uvector<T>>& A,
                                 const xarray_container<uvector<T>>& b) {
        auto sh = A.shape();
        std::size_t m = sh[0], n = sh[1];
        xarray_container<uvector<T>> Q = A; // will be modified to Q
        xarray_container<uvector<T>> R({n, n}, T(0));
        for (std::size_t k = 0; k < n; ++k) {
            // Norm of column k of Q
            T norm2 = 0;
            for (std::size_t i = 0; i < m; ++i) norm2 += Q(i,k) * Q(i,k);
            R(k,k) = std::sqrt(norm2);
            for (std::size_t i = 0; i < m; ++i) Q(i,k) /= R(k,k);
            for (std::size_t j = k+1; j < n; ++j) {
                R(k,j) = 0;
                for (std::size_t i = 0; i < m; ++i) R(k,j) += Q(i,k) * Q(i,j);
                for (std::size_t i = 0; i < m; ++i) Q(i,j) -= R(k,j) * Q(i,k);
            }
        }
        // Q^T b
        auto Qtb = xt::linalg::matmul(xt::transpose(Q), b);
        // Back substitution R x = Qtb
        xarray_container<uvector<T>> x({n});
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n)-1; i >= 0; --i) {
            T sum = Qtb[i];
            for (std::size_t j = i+1; j < n; ++j) sum -= R(i,j) * x[j];
            x[i] = sum / R(i,i);
        }
        return x;
    }

    /*********************************************
     * Root finding (scalar and vector)
     *********************************************/
    /**
     * Newton-Raphson for scalar function f and derivative df.
     */
    template <class Func, class Deriv>
    inline auto newton_raphson(Func f, Deriv df, double x0, double tol = 1e-12, std::size_t max_iter = 100) {
        double x = x0;
        for (std::size_t i = 0; i < max_iter; ++i) {
            double fx = f(x);
            double dfx = df(x);
            if (std::abs(dfx) < 1e-15) break;
            double dx = fx / dfx;
            x -= dx;
            if (std::abs(dx) < tol) break;
        }
        return x;
    }

    /**
     * Bisection method for root finding in interval [a,b].
     */
    template <class Func>
    inline auto bisect(Func f, double a, double b, double tol = 1e-10) {
        double fa = f(a), fb = f(b);
        if (fa * fb >= 0) throw std::runtime_error("Bisection: f(a) and f(b) must have opposite signs.");
        while ((b - a) > tol) {
            double c = (a + b) / 2.0;
            double fc = f(c);
            if (fa * fc < 0) {
                b = c; fb = fc;
            } else {
                a = c; fa = fc;
            }
        }
        return (a + b) / 2.0;
    }

    /*********************************************
     * Minimum and Maximum of a function (golden-section)
     *********************************************/
    /**
     * Golden-section search for minimum of a scalar function in [a,b].
     */
    template <class Func>
    inline auto golden_section_min(Func f, double a, double b, double tol = 1e-8) {
        const double invphi = (std::sqrt(5.0) - 1.0) / 2.0;
        const double invphi2 = (3.0 - std::sqrt(5.0)) / 2.0;
        double x1 = a + invphi2 * (b - a);
        double x2 = a + invphi * (b - a);
        double f1 = f(x1), f2 = f(x2);
        while ((b - a) > tol) {
            if (f1 < f2) {
                b = x2; x2 = x1; f2 = f1;
                x1 = a + invphi2 * (b - a);
                f1 = f(x1);
            } else {
                a = x1; x1 = x2; f1 = f2;
                x2 = a + invphi * (b - a);
                f2 = f(x2);
            }
        }
        return (a + b) / 2.0;
    }

} // namespace optimize
} // namespace xt

#endif // XTENSOR_XOPTIMIZE_HPP