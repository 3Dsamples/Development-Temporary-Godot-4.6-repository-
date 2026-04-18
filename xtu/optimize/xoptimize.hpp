// include/xtu/optimize/xoptimize.hpp
// xtensor-unified - Numerical optimization and root finding algorithms
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_OPTIMIZE_XOPTIMIZE_HPP
#define XTU_OPTIMIZE_XOPTIMIZE_HPP

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
#include "xtu/math/xlinalg.hpp"
#include "xtu/math/xblas.hpp"

XTU_NAMESPACE_BEGIN
namespace optimize {

// #############################################################################
// Line search methods
// #############################################################################

/// Backtracking line search (Armijo condition)
template <class Func, class GradFunc>
double backtracking_line_search(const Func& f, const GradFunc& grad_f,
                                const std::vector<double>& x,
                                const std::vector<double>& direction,
                                double alpha = 1.0, double rho = 0.5, double c = 1e-4,
                                size_t max_iter = 100) {
    double fx = f(x);
    double grad_dot_dir = 0.0;
    std::vector<double> grad = grad_f(x);
    for (size_t i = 0; i < x.size(); ++i) {
        grad_dot_dir += grad[i] * direction[i];
    }
    if (grad_dot_dir >= 0) {
        XTU_THROW(std::invalid_argument, "Direction is not a descent direction");
    }
    double step = alpha;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        std::vector<double> x_new(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] + step * direction[i];
        }
        double f_new = f(x_new);
        if (f_new <= fx + c * step * grad_dot_dir) {
            return step;
        }
        step *= rho;
    }
    return step;
}

/// Wolfe conditions line search (Armijo + curvature)
template <class Func, class GradFunc>
double wolfe_line_search(const Func& f, const GradFunc& grad_f,
                         const std::vector<double>& x,
                         const std::vector<double>& direction,
                         double alpha = 1.0, double c1 = 1e-4, double c2 = 0.9,
                         size_t max_iter = 100) {
    double fx = f(x);
    std::vector<double> grad = grad_f(x);
    double grad_dot_dir = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        grad_dot_dir += grad[i] * direction[i];
    }
    if (grad_dot_dir >= 0) {
        XTU_THROW(std::invalid_argument, "Direction is not a descent direction");
    }
    double step = alpha;
    double step_low = 0.0;
    double step_high = std::numeric_limits<double>::infinity();
    for (size_t iter = 0; iter < max_iter; ++iter) {
        std::vector<double> x_new(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] + step * direction[i];
        }
        double f_new = f(x_new);
        // Armijo condition
        if (f_new > fx + c1 * step * grad_dot_dir) {
            step_high = step;
            step = (step_low + step_high) / 2.0;
            continue;
        }
        std::vector<double> grad_new = grad_f(x_new);
        double grad_new_dot_dir = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            grad_new_dot_dir += grad_new[i] * direction[i];
        }
        // Curvature condition
        if (grad_new_dot_dir < c2 * grad_dot_dir) {
            step_low = step;
            step = std::isinf(step_high) ? 2.0 * step : (step_low + step_high) / 2.0;
            continue;
        }
        return step;
    }
    return step;
}

// #############################################################################
// Gradient Descent
// #############################################################################
template <class Func, class GradFunc>
std::vector<double> gradient_descent(const Func& f, const GradFunc& grad_f,
                                      std::vector<double> x0,
                                      double learning_rate = 0.01,
                                      size_t max_iter = 1000,
                                      double tol = 1e-6,
                                      bool use_line_search = false) {
    size_t n = x0.size();
    std::vector<double> x = std::move(x0);
    std::vector<double> grad(n);
    for (size_t iter = 0; iter < max_iter; ++iter) {
        grad = grad_f(x);
        double grad_norm = 0.0;
        for (size_t i = 0; i < n; ++i) grad_norm += grad[i] * grad[i];
        grad_norm = std::sqrt(grad_norm);
        if (grad_norm < tol) break;
        double step = learning_rate;
        if (use_line_search) {
            // direction is -grad
            std::vector<double> direction(n);
            for (size_t i = 0; i < n; ++i) direction[i] = -grad[i];
            step = backtracking_line_search(f, grad_f, x, direction, 1.0);
        }
        for (size_t i = 0; i < n; ++i) {
            x[i] -= step * grad[i];
        }
    }
    return x;
}

// #############################################################################
// Newton's Method (for unconstrained optimization)
// #############################################################################
template <class Func, class GradFunc, class HessFunc>
std::vector<double> newton_method(const Func& f, const GradFunc& grad_f, const HessFunc& hess_f,
                                   std::vector<double> x0,
                                   size_t max_iter = 100,
                                   double tol = 1e-6,
                                   bool use_line_search = true) {
    size_t n = x0.size();
    std::vector<double> x = std::move(x0);
    std::vector<double> grad(n);
    for (size_t iter = 0; iter < max_iter; ++iter) {
        grad = grad_f(x);
        double grad_norm = 0.0;
        for (size_t i = 0; i < n; ++i) grad_norm += grad[i] * grad[i];
        grad_norm = std::sqrt(grad_norm);
        if (grad_norm < tol) break;
        
        // Compute Hessian
        auto hess = hess_f(x);
        XTU_ASSERT_MSG(hess.dimension() == 2 && hess.shape()[0] == n && hess.shape()[1] == n,
                       "Hessian must be n x n matrix");
        // Solve H * p = -grad for direction p
        std::vector<double> rhs(n);
        for (size_t i = 0; i < n; ++i) rhs[i] = -grad[i];
        // Use linear solver (LU)
        xarray_container<double> H_xt({n, n});
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                H_xt(i, j) = hess(i, j);
        xarray_container<double> rhs_arr({n});
        for (size_t i = 0; i < n; ++i) rhs_arr[i] = rhs[i];
        xarray_container<double> p_arr({n});
        math::solve(H_xt, rhs_arr, p_arr);
        std::vector<double> p(n);
        for (size_t i = 0; i < n; ++i) p[i] = p_arr[i];
        
        double step = 1.0;
        if (use_line_search) {
            step = backtracking_line_search(f, grad_f, x, p, 1.0);
        }
        for (size_t i = 0; i < n; ++i) {
            x[i] += step * p[i];
        }
    }
    return x;
}

// #############################################################################
// BFGS (Quasi-Newton) Method
// #############################################################################
template <class Func, class GradFunc>
std::vector<double> bfgs(const Func& f, const GradFunc& grad_f,
                         std::vector<double> x0,
                         size_t max_iter = 200,
                         double tol = 1e-6,
                         bool use_line_search = true) {
    size_t n = x0.size();
    std::vector<double> x = x0;
    std::vector<double> grad = grad_f(x);
    double grad_norm = 0.0;
    for (size_t i = 0; i < n; ++i) grad_norm += grad[i] * grad[i];
    grad_norm = std::sqrt(grad_norm);
    if (grad_norm < tol) return x;
    
    // Initialize inverse Hessian approximation to identity
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) H[i][i] = 1.0;
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Compute search direction p = -H * grad
        std::vector<double> p(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                p[i] -= H[i][j] * grad[j];
            }
        }
        
        double step = 1.0;
        if (use_line_search) {
            step = wolfe_line_search(f, grad_f, x, p, 1.0);
        }
        
        // Compute s = step * p
        std::vector<double> s(n);
        for (size_t i = 0; i < n; ++i) s[i] = step * p[i];
        // Update x
        std::vector<double> x_new(n);
        for (size_t i = 0; i < n; ++i) x_new[i] = x[i] + s[i];
        
        std::vector<double> grad_new = grad_f(x_new);
        double grad_new_norm = 0.0;
        for (size_t i = 0; i < n; ++i) grad_new_norm += grad_new[i] * grad_new[i];
        grad_new_norm = std::sqrt(grad_new_norm);
        if (grad_new_norm < tol) {
            return x_new;
        }
        
        // Compute y = grad_new - grad
        std::vector<double> y(n);
        for (size_t i = 0; i < n; ++i) y[i] = grad_new[i] - grad[i];
        
        // Compute rho = 1 / (y^T s)
        double ys = 0.0;
        for (size_t i = 0; i < n; ++i) ys += y[i] * s[i];
        if (ys <= 0) {
            // Skip update if curvature condition not satisfied
            x = x_new;
            grad = grad_new;
            continue;
        }
        double rho = 1.0 / ys;
        
        // Update H using BFGS formula: H_new = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
        // Compute Hy
        std::vector<double> Hy(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                Hy[i] += H[i][j] * y[j];
            }
        }
        // Compute y^T H y
        double yHy = 0.0;
        for (size_t i = 0; i < n; ++i) yHy += y[i] * Hy[i];
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                H[i][j] = H[i][j] - rho * (Hy[i] * s[j] + s[i] * Hy[j]) 
                          + rho * rho * (yHy + 1.0 / rho) * s[i] * s[j];
            }
        }
        
        x = x_new;
        grad = grad_new;
    }
    return x;
}

// #############################################################################
// Root finding: Bisection method
// #############################################################################
template <class Func>
double bisection(const Func& f, double a, double b, double tol = 1e-6, size_t max_iter = 100) {
    double fa = f(a);
    double fb = f(b);
    if (fa * fb > 0) {
        XTU_THROW(std::invalid_argument, "Function must have opposite signs at endpoints");
    }
    if (fa == 0) return a;
    if (fb == 0) return b;
    double c = a;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        c = (a + b) / 2.0;
        double fc = f(c);
        if (fc == 0 || (b - a) / 2.0 < tol) {
            return c;
        }
        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    return c;
}

// #############################################################################
// Root finding: Newton's method
// #############################################################################
template <class Func, class DerivFunc>
double newton_root(const Func& f, const DerivFunc& fprime, double x0,
                   double tol = 1e-6, size_t max_iter = 100) {
    double x = x0;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        double fx = f(x);
        if (std::abs(fx) < tol) return x;
        double dfx = fprime(x);
        if (dfx == 0) {
            XTU_THROW(std::runtime_error, "Derivative zero; Newton method fails");
        }
        double x_new = x - fx / dfx;
        if (std::abs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    return x;
}

// #############################################################################
// Root finding: Secant method
// #############################################################################
template <class Func>
double secant(const Func& f, double x0, double x1, double tol = 1e-6, size_t max_iter = 100) {
    double f0 = f(x0);
    double f1 = f(x1);
    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (std::abs(f1) < tol) return x1;
        if (f1 == f0) {
            XTU_THROW(std::runtime_error, "Secant method: division by zero");
        }
        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
        if (std::abs(x1 - x0) < tol) return x1;
    }
    return x1;
}

// #############################################################################
// Scalar minimization: Golden-section search
// #############################################################################
template <class Func>
double golden_section_search(const Func& f, double a, double b, double tol = 1e-6, size_t max_iter = 100) {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double inv_phi = 1.0 / phi;
    double x1 = b - (b - a) * inv_phi;
    double x2 = a + (b - a) * inv_phi;
    double f1 = f(x1);
    double f2 = f(x2);
    for (size_t iter = 0; iter < max_iter; ++iter) {
        if ((b - a) < tol) {
            return (a + b) / 2.0;
        }
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - (b - a) * inv_phi;
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + (b - a) * inv_phi;
            f2 = f(x2);
        }
    }
    return (a + b) / 2.0;
}

// #############################################################################
// Scalar minimization: Brent's method (combines golden-section and parabolic)
// #############################################################################
template <class Func>
double brent_minimize(const Func& f, double a, double b, double tol = 1e-6, size_t max_iter = 100) {
    const double cgold = (3.0 - std::sqrt(5.0)) / 2.0; // 0.381966...
    double x = a + cgold * (b - a);
    double w = x;
    double v = x;
    double fx = f(x);
    double fw = fx;
    double fv = fx;
    double d = 0.0;
    double e = 0.0;
    double m = 0.0;
    double tol_act = 0.0;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        m = (a + b) / 2.0;
        tol_act = tol * std::abs(x) + 1e-10;
        if (std::abs(x - m) <= 2.0 * tol_act - (b - a) / 2.0) {
            return x;
        }
        // Try parabolic fit
        if (std::abs(e) > tol_act) {
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::abs(q);
            double etemp = e;
            e = d;
            if (std::abs(p) >= std::abs(q * etemp / 2.0) || p <= q * (a - x) || p >= q * (b - x)) {
                // Golden section
                e = (x >= m) ? a - x : b - x;
                d = cgold * e;
            } else {
                d = p / q;
                double u = x + d;
                if (u - a < 2.0 * tol_act || b - u < 2.0 * tol_act) {
                    d = (x < m) ? tol_act : -tol_act;
                }
            }
        } else {
            e = (x >= m) ? a - x : b - x;
            d = cgold * e;
        }
        double u = (std::abs(d) >= tol_act) ? x + d : x + ((d > 0) ? tol_act : -tol_act);
        double fu = f(u);
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    return x;
}

} // namespace optimize

// Bring into main namespace for convenience
using optimize::backtracking_line_search;
using optimize::wolfe_line_search;
using optimize::gradient_descent;
using optimize::newton_method;
using optimize::bfgs;
using optimize::bisection;
using optimize::newton_root;
using optimize::secant;
using optimize::golden_section_search;
using optimize::brent_minimize;

XTU_NAMESPACE_END

#endif // XTU_OPTIMIZE_XOPTIMIZE_HPP