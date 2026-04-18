// math/xoptimize.hpp

#ifndef XTENSOR_XOPTIMIZE_HPP
#define XTENSOR_XOPTIMIZE_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xrandom.hpp"
#include "xinterp.hpp"

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
#include <memory>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace optimize
        {
            // --------------------------------------------------------------------
            // Utility types and constants
            // --------------------------------------------------------------------
            using Vector = xarray_container<double>;
            using Matrix = xarray_container<double>;

            // Optimization result structure
            struct OptimizeResult
            {
                Vector x;                    // optimal parameters
                double fun;                  // function value at optimum
                Vector grad;                 // gradient at optimum
                Matrix hessian;              // Hessian approximation (if available)
                size_t nfev = 0;             // number of function evaluations
                size_t ngev = 0;             // number of gradient evaluations
                size_t nhev = 0;             // number of Hessian evaluations
                size_t nit = 0;              // number of iterations
                bool success = false;        // convergence flag
                std::string message;         // termination reason
                std::vector<double> history_fun; // function value history
            };

            // Convergence criteria
            struct ConvergenceCriteria
            {
                double ftol = 1e-8;          // function value tolerance
                double gtol = 1e-6;          // gradient norm tolerance
                double xtol = 1e-8;          // parameter change tolerance
                size_t maxiter = 1000;       // maximum iterations
                size_t maxfev = 5000;        // maximum function evaluations
            };

            // Line search result
            struct LineSearchResult
            {
                double alpha = 0.0;
                double f_alpha = 0.0;
                size_t nfev = 0;
                bool success = false;
            };

            // --------------------------------------------------------------------
            // Line search methods
            // --------------------------------------------------------------------
            namespace linesearch
            {
                // Backtracking line search with Armijo condition
                template <class Func, class GradFunc>
                inline LineSearchResult backtracking(
                    const Vector& x, const Vector& p,
                    Func&& func, GradFunc&& grad_func,
                    double f0, const Vector& g0,
                    double alpha_init = 1.0,
                    double rho = 0.5, double c1 = 1e-4,
                    size_t max_iter = 50)
                {
                    LineSearchResult result;
                    result.alpha = alpha_init;
                    Vector x_new = x + result.alpha * p;
                    result.f_alpha = func(x_new);
                    result.nfev = 1;

                    double slope = xt::sum(g0 * p)();
                    size_t iter = 0;
                    while (result.f_alpha > f0 + c1 * result.alpha * slope && iter < max_iter)
                    {
                        result.alpha *= rho;
                        x_new = x + result.alpha * p;
                        result.f_alpha = func(x_new);
                        result.nfev++;
                        iter++;
                    }
                    result.success = (iter < max_iter);
                    return result;
                }

                // Strong Wolfe conditions line search
                template <class Func, class GradFunc>
                inline LineSearchResult strong_wolfe(
                    const Vector& x, const Vector& p,
                    Func&& func, GradFunc&& grad_func,
                    double f0, const Vector& g0,
                    double alpha_init = 1.0,
                    double c1 = 1e-4, double c2 = 0.9,
                    size_t max_iter = 50)
                {
                    LineSearchResult result;
                    double alpha_prev = 0.0;
                    double f_prev = f0;
                    double dphi0 = xt::sum(g0 * p)();
                    if (dphi0 >= 0)
                    {
                        result.success = false;
                        return result;
                    }

                    result.alpha = alpha_init;
                    size_t iter = 0;
                    bool bracketed = false;
                    double alpha_lo = 0.0, alpha_hi = std::numeric_limits<double>::max();

                    while (iter < max_iter)
                    {
                        Vector x_new = x + result.alpha * p;
                        double f_new = func(x_new);
                        result.nfev++;

                        if (f_new > f0 + c1 * result.alpha * dphi0 ||
                            (f_new >= f_prev && iter > 0))
                        {
                            alpha_hi = result.alpha;
                            bracketed = true;
                        }
                        else
                        {
                            Vector g_new = grad_func(x_new);
                            double dphi_new = xt::sum(g_new * p)();
                            if (std::abs(dphi_new) <= -c2 * dphi0)
                            {
                                result.f_alpha = f_new;
                                result.success = true;
                                return result;
                            }
                            if (dphi_new >= 0)
                            {
                                alpha_hi = result.alpha;
                                bracketed = true;
                            }
                            else
                            {
                                alpha_lo = result.alpha;
                            }
                        }

                        if (bracketed)
                        {
                            // Interpolation to refine alpha
                            result.alpha = (alpha_lo + alpha_hi) / 2.0;
                        }
                        else
                        {
                            result.alpha *= 2.0;
                        }

                        alpha_prev = result.alpha;
                        f_prev = f_new;
                        iter++;
                    }

                    result.f_alpha = f_prev;
                    result.success = (iter < max_iter);
                    return result;
                }

                // More-Thuente line search (robust cubic interpolation)
                template <class Func, class GradFunc>
                inline LineSearchResult more_thuente(
                    const Vector& x, const Vector& p,
                    Func&& func, GradFunc&& grad_func,
                    double f0, const Vector& g0,
                    double alpha_init = 1.0,
                    double c1 = 1e-4, double c2 = 0.9,
                    size_t max_iter = 20)
                {
                    // Implementation based on More-Thuente algorithm
                    LineSearchResult result;
                    double dphi0 = xt::sum(g0 * p)();
                    if (dphi0 >= 0)
                    {
                        result.success = false;
                        return result;
                    }

                    double alpha_min = 0.0;
                    double alpha_max = std::numeric_limits<double>::max();
                    result.alpha = alpha_init;
                    double f_prev = f0;
                    double dphi_prev = dphi0;

                    for (size_t iter = 0; iter < max_iter; ++iter)
                    {
                        Vector x_new = x + result.alpha * p;
                        double f_new = func(x_new);
                        result.nfev++;

                        // Check Armijo condition
                        if (f_new > f0 + c1 * result.alpha * dphi0)
                        {
                            alpha_max = result.alpha;
                        }
                        else
                        {
                            Vector g_new = grad_func(x_new);
                            double dphi_new = xt::sum(g_new * p)();
                            // Check Wolfe curvature condition
                            if (std::abs(dphi_new) <= -c2 * dphi0)
                            {
                                result.f_alpha = f_new;
                                result.success = true;
                                return result;
                            }
                            if (dphi_new >= 0)
                            {
                                alpha_max = result.alpha;
                            }
                            else
                            {
                                alpha_min = result.alpha;
                            }
                        }

                        // Update alpha using cubic interpolation
                        if (alpha_max < std::numeric_limits<double>::max())
                        {
                            result.alpha = (alpha_min + alpha_max) / 2.0;
                        }
                        else
                        {
                            result.alpha = 2.0 * result.alpha;
                        }
                        if (result.alpha < alpha_min || result.alpha > alpha_max)
                            result.alpha = (alpha_min + alpha_max) / 2.0;

                        if (alpha_max - alpha_min < 1e-12 * alpha_max)
                            break;
                    }
                    result.f_alpha = func(x + result.alpha * p);
                    result.success = false;
                    return result;
                }
            }

            // --------------------------------------------------------------------
            // Base optimizer class
            // --------------------------------------------------------------------
            template <class Derived>
            class OptimizerBase
            {
            public:
                using VectorType = Vector;
                using MatrixType = Matrix;

                OptimizerBase() = default;
                virtual ~OptimizerBase() = default;

                // Set options
                void set_ftol(double val) { m_options.ftol = val; }
                void set_gtol(double val) { m_options.gtol = val; }
                void set_xtol(double val) { m_options.xtol = val; }
                void set_maxiter(size_t val) { m_options.maxiter = val; }
                void set_maxfev(size_t val) { m_options.maxfev = val; }
                void set_verbose(bool val) { m_verbose = val; }

                // Get result
                const OptimizeResult& result() const { return m_result; }

            protected:
                ConvergenceCriteria m_options;
                bool m_verbose = false;
                OptimizeResult m_result;

                void reset_result(const Vector& x0)
                {
                    m_result = OptimizeResult{};
                    m_result.x = x0;
                    m_result.history_fun.clear();
                }

                bool check_convergence(double f, double f_prev, const Vector& g, const Vector& dx)
                {
                    double f_diff = std::abs(f - f_prev);
                    double g_norm = xt::norm_l2(g)();
                    double dx_norm = xt::norm_l2(dx)();

                    bool converged = false;
                    if (f_diff < m_options.ftol * (1.0 + std::abs(f)))
                    {
                        m_result.message = "Function value tolerance reached";
                        converged = true;
                    }
                    else if (g_norm < m_options.gtol)
                    {
                        m_result.message = "Gradient norm tolerance reached";
                        converged = true;
                    }
                    else if (dx_norm < m_options.xtol)
                    {
                        m_result.message = "Parameter change tolerance reached";
                        converged = true;
                    }
                    else if (m_result.nit >= m_options.maxiter)
                    {
                        m_result.message = "Maximum iterations reached";
                        converged = true;
                    }
                    else if (m_result.nfev >= m_options.maxfev)
                    {
                        m_result.message = "Maximum function evaluations reached";
                        converged = true;
                    }
                    return converged;
                }
            };

            // --------------------------------------------------------------------
            // Gradient Descent variants
            // --------------------------------------------------------------------
            class GradientDescent : public OptimizerBase<GradientDescent>
            {
            public:
                enum Method
                {
                    Vanilla,
                    Momentum,
                    Nesterov
                };

                GradientDescent(double learning_rate = 0.01, Method method = Vanilla,
                                double momentum = 0.9)
                    : m_lr(learning_rate), m_method(method), m_momentum(momentum) {}

                template <class Func, class GradFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector v = xt::zeros_like(x); // velocity for momentum
                    double f_prev = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f_prev);

                    if (m_verbose)
                    {
                        std::cout << "GradientDescent: iter=0, f=" << f_prev << std::endl;
                    }

                    while (true)
                    {
                        Vector g = grad_func(x);
                        m_result.ngev++;
                        Vector dx;

                        if (m_method == Vanilla)
                        {
                            dx = -m_lr * g;
                        }
                        else if (m_method == Momentum)
                        {
                            v = m_momentum * v - m_lr * g;
                            dx = v;
                        }
                        else // Nesterov
                        {
                            Vector v_prev = v;
                            v = m_momentum * v - m_lr * g;
                            dx = m_momentum * v_prev - (1.0 + m_momentum) * m_lr * g;
                        }

                        Vector x_new = x + dx;
                        double f_new = func(x_new);
                        m_result.nfev++;
                        m_result.history_fun.push_back(f_new);
                        m_result.nit++;

                        if (m_verbose && m_result.nit % 10 == 0)
                        {
                            std::cout << "GradientDescent: iter=" << m_result.nit
                                      << ", f=" << f_new
                                      << ", |g|=" << xt::norm_l2(g)() << std::endl;
                        }

                        if (check_convergence(f_new, f_prev, g, dx))
                        {
                            m_result.x = x_new;
                            m_result.fun = f_new;
                            m_result.grad = g;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }

                        x = std::move(x_new);
                        f_prev = f_new;
                    }
                    return m_result;
                }

                void set_learning_rate(double lr) { m_lr = lr; }
                void set_momentum(double mom) { m_momentum = mom; }

            private:
                double m_lr;
                Method m_method;
                double m_momentum;
            };

            // --------------------------------------------------------------------
            // Adaptive Gradient Methods (AdaGrad, RMSProp, Adam, AdaDelta)
            // --------------------------------------------------------------------
            class Adam : public OptimizerBase<Adam>
            {
            public:
                Adam(double learning_rate = 0.001,
                     double beta1 = 0.9, double beta2 = 0.999,
                     double epsilon = 1e-8)
                    : m_lr(learning_rate), m_beta1(beta1), m_beta2(beta2), m_eps(epsilon) {}

                template <class Func, class GradFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector m = xt::zeros_like(x);
                    Vector v = xt::zeros_like(x);
                    double f_prev = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f_prev);

                    size_t t = 0;
                    while (true)
                    {
                        t++;
                        Vector g = grad_func(x);
                        m_result.ngev++;

                        m = m_beta1 * m + (1.0 - m_beta1) * g;
                        v = m_beta2 * v + (1.0 - m_beta2) * (g * g);

                        Vector m_hat = m / (1.0 - std::pow(m_beta1, t));
                        Vector v_hat = v / (1.0 - std::pow(m_beta2, t));

                        Vector dx = -m_lr * m_hat / (xt::sqrt(v_hat) + m_eps);
                        Vector x_new = x + dx;

                        double f_new = func(x_new);
                        m_result.nfev++;
                        m_result.history_fun.push_back(f_new);
                        m_result.nit++;

                        if (m_verbose && m_result.nit % 10 == 0)
                        {
                            std::cout << "Adam: iter=" << m_result.nit
                                      << ", f=" << f_new
                                      << ", |g|=" << xt::norm_l2(g)() << std::endl;
                        }

                        if (check_convergence(f_new, f_prev, g, dx))
                        {
                            m_result.x = x_new;
                            m_result.fun = f_new;
                            m_result.grad = g;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }

                        x = std::move(x_new);
                        f_prev = f_new;
                    }
                    return m_result;
                }

            private:
                double m_lr;
                double m_beta1, m_beta2;
                double m_eps;
            };

            class RMSProp : public OptimizerBase<RMSProp>
            {
            public:
                RMSProp(double learning_rate = 0.001, double decay = 0.9, double epsilon = 1e-8)
                    : m_lr(learning_rate), m_decay(decay), m_eps(epsilon) {}

                template <class Func, class GradFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector cache = xt::zeros_like(x);
                    double f_prev = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f_prev);

                    while (true)
                    {
                        Vector g = grad_func(x);
                        m_result.ngev++;

                        cache = m_decay * cache + (1.0 - m_decay) * (g * g);
                        Vector dx = -m_lr * g / (xt::sqrt(cache) + m_eps);
                        Vector x_new = x + dx;

                        double f_new = func(x_new);
                        m_result.nfev++;
                        m_result.history_fun.push_back(f_new);
                        m_result.nit++;

                        if (check_convergence(f_new, f_prev, g, dx))
                        {
                            m_result.x = x_new;
                            m_result.fun = f_new;
                            m_result.grad = g;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }

                        x = std::move(x_new);
                        f_prev = f_new;
                    }
                    return m_result;
                }

            private:
                double m_lr;
                double m_decay;
                double m_eps;
            };

            // --------------------------------------------------------------------
            // Quasi-Newton Methods (BFGS, L-BFGS)
            // --------------------------------------------------------------------
            class BFGS : public OptimizerBase<BFGS>
            {
            public:
                BFGS(bool use_linesearch = true) : m_use_linesearch(use_linesearch) {}

                template <class Func, class GradFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, const Vector& x0)
                {
                    reset_result(x0);
                    size_t n = x0.size();
                    Vector x = x0;
                    Vector g = grad_func(x);
                    m_result.ngev = 1;
                    double f_prev = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f_prev);

                    // Initialize Hessian approximation to identity
                    Matrix H = xt::eye<double>(n);
                    Vector p = -xt::linalg::dot(H, g);

                    while (true)
                    {
                        double alpha = 1.0;
                        if (m_use_linesearch)
                        {
                            auto ls = linesearch::strong_wolfe(x, p, func, grad_func, f_prev, g);
                            alpha = ls.alpha;
                            m_result.nfev += ls.nfev;
                        }

                        Vector s = alpha * p;
                        Vector x_new = x + s;
                        double f_new = func(x_new);
                        m_result.nfev++;
                        m_result.history_fun.push_back(f_new);

                        Vector g_new = grad_func(x_new);
                        m_result.ngev++;
                        Vector y = g_new - g;

                        // BFGS update
                        double sy = xt::sum(s * y)();
                        if (sy > 1e-12)
                        {
                            Vector Hy = xt::linalg::dot(H, y);
                            double yHy = xt::sum(y * Hy)();
                            H = H + (sy + yHy) / (sy * sy) * xt::linalg::outer(s, s)
                                - (xt::linalg::outer(Hy, s) + xt::linalg::outer(s, Hy)) / sy;
                        }

                        m_result.nit++;
                        if (m_verbose && m_result.nit % 10 == 0)
                        {
                            std::cout << "BFGS: iter=" << m_result.nit
                                      << ", f=" << f_new
                                      << ", |g|=" << xt::norm_l2(g_new)() << std::endl;
                        }

                        if (check_convergence(f_new, f_prev, g_new, s))
                        {
                            m_result.x = x_new;
                            m_result.fun = f_new;
                            m_result.grad = g_new;
                            m_result.hessian = H;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }

                        p = -xt::linalg::dot(H, g_new);
                        x = std::move(x_new);
                        g = std::move(g_new);
                        f_prev = f_new;
                    }
                    return m_result;
                }

            private:
                bool m_use_linesearch;
            };

            // Limited-memory BFGS
            class LBFGS : public OptimizerBase<LBFGS>
            {
            public:
                LBFGS(size_t memory = 10, bool use_linesearch = true)
                    : m_memory(memory), m_use_linesearch(use_linesearch) {}

                template <class Func, class GradFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector g = grad_func(x);
                    m_result.ngev = 1;
                    double f_prev = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f_prev);

                    std::deque<Vector> s_list, y_list;
                    std::deque<double> rho_list;
                    Vector q = g;

                    while (true)
                    {
                        // Two-loop recursion for L-BFGS direction
                        Vector q = g;
                        std::vector<double> alpha_list(s_list.size());

                        for (int i = static_cast<int>(s_list.size()) - 1; i >= 0; --i)
                        {
                            alpha_list[static_cast<size_t>(i)] = rho_list[static_cast<size_t>(i)] * xt::sum(s_list[static_cast<size_t>(i)] * q)();
                            q = q - alpha_list[static_cast<size_t>(i)] * y_list[static_cast<size_t>(i)];
                        }

                        // Initial Hessian scaling
                        Vector r = q;
                        if (!s_list.empty())
                        {
                            double gamma = xt::sum(s_list.back() * y_list.back())() / xt::sum(y_list.back() * y_list.back())();
                            r = gamma * r;
                        }

                        for (size_t i = 0; i < s_list.size(); ++i)
                        {
                            double beta = rho_list[i] * xt::sum(y_list[i] * r)();
                            r = r + s_list[i] * (alpha_list[i] - beta);
                        }

                        Vector p = -r;

                        double alpha = 1.0;
                        if (m_use_linesearch)
                        {
                            auto ls = linesearch::strong_wolfe(x, p, func, grad_func, f_prev, g);
                            alpha = ls.alpha;
                            m_result.nfev += ls.nfev;
                        }

                        Vector s = alpha * p;
                        Vector x_new = x + s;
                        double f_new = func(x_new);
                        m_result.nfev++;
                        m_result.history_fun.push_back(f_new);

                        Vector g_new = grad_func(x_new);
                        m_result.ngev++;
                        Vector y = g_new - g;
                        double sy = xt::sum(s * y)();

                        if (sy > 1e-12)
                        {
                            s_list.push_back(s);
                            y_list.push_back(y);
                            rho_list.push_back(1.0 / sy);
                            if (s_list.size() > m_memory)
                            {
                                s_list.pop_front();
                                y_list.pop_front();
                                rho_list.pop_front();
                            }
                        }

                        m_result.nit++;
                        if (check_convergence(f_new, f_prev, g_new, s))
                        {
                            m_result.x = x_new;
                            m_result.fun = f_new;
                            m_result.grad = g_new;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }

                        x = std::move(x_new);
                        g = std::move(g_new);
                        f_prev = f_new;
                    }
                    return m_result;
                }

            private:
                size_t m_memory;
                bool m_use_linesearch;
            };

            // --------------------------------------------------------------------
            // Conjugate Gradient methods
            // --------------------------------------------------------------------
            class ConjugateGradient : public OptimizerBase<ConjugateGradient>
            {
            public:
                enum Formula
                {
                    FletcherReeves,
                    PolakRibiere,
                    HestenesStiefel,
                    DaiYuan
                };

                ConjugateGradient(Formula formula = PolakRibiere)
                    : m_formula(formula) {}

                template <class Func, class GradFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector g = grad_func(x);
                    m_result.ngev = 1;
                    double f_prev = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f_prev);

                    Vector p = -g;
                    Vector g_prev = g;

                    while (true)
                    {
                        auto ls = linesearch::strong_wolfe(x, p, func, grad_func, f_prev, g);
                        m_result.nfev += ls.nfev;
                        if (!ls.success) break;

                        Vector x_new = x + ls.alpha * p;
                        double f_new = ls.f_alpha;
                        m_result.history_fun.push_back(f_new);

                        Vector g_new = grad_func(x_new);
                        m_result.ngev++;

                        // Compute beta using selected formula
                        double beta = 0.0;
                        if (m_formula == FletcherReeves)
                        {
                            beta = xt::sum(g_new * g_new)() / xt::sum(g_prev * g_prev)();
                        }
                        else if (m_formula == PolakRibiere)
                        {
                            beta = xt::sum(g_new * (g_new - g_prev))() / xt::sum(g_prev * g_prev)();
                            beta = std::max(0.0, beta);
                        }
                        else if (m_formula == HestenesStiefel)
                        {
                            Vector y = g_new - g_prev;
                            beta = xt::sum(g_new * y)() / xt::sum(p * y)();
                        }
                        else // DaiYuan
                        {
                            Vector y = g_new - g_prev;
                            beta = xt::sum(g_new * g_new)() / xt::sum(p * y)();
                        }

                        p = -g_new + beta * p;

                        m_result.nit++;
                        if (check_convergence(f_new, f_prev, g_new, ls.alpha * p))
                        {
                            m_result.x = x_new;
                            m_result.fun = f_new;
                            m_result.grad = g_new;
                            m_result.success = true;
                            break;
                        }

                        x = std::move(x_new);
                        g_prev = std::move(g);
                        g = std::move(g_new);
                        f_prev = f_new;
                    }
                    return m_result;
                }

            private:
                Formula m_formula;
            };

            // --------------------------------------------------------------------
            // Newton's Method with optional line search
            // --------------------------------------------------------------------
            class Newton : public OptimizerBase<Newton>
            {
            public:
                Newton(bool use_linesearch = true) : m_use_linesearch(use_linesearch) {}

                template <class Func, class GradFunc, class HessFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, HessFunc&& hess_func,
                                        const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector g = grad_func(x);
                    m_result.ngev = 1;
                    double f_prev = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f_prev);

                    while (true)
                    {
                        Matrix H = hess_func(x);
                        m_result.nhev++;

                        // Solve H * p = -g
                        Vector p = -xt::linalg::solve(H, g);
                        m_result.hessian = H;

                        double alpha = 1.0;
                        if (m_use_linesearch)
                        {
                            auto ls = linesearch::backtracking(x, p, func, grad_func, f_prev, g);
                            alpha = ls.alpha;
                            m_result.nfev += ls.nfev;
                        }

                        Vector x_new = x + alpha * p;
                        double f_new = func(x_new);
                        m_result.nfev++;
                        m_result.history_fun.push_back(f_new);

                        Vector g_new = grad_func(x_new);
                        m_result.ngev++;

                        m_result.nit++;
                        if (check_convergence(f_new, f_prev, g_new, alpha * p))
                        {
                            m_result.x = x_new;
                            m_result.fun = f_new;
                            m_result.grad = g_new;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }

                        x = std::move(x_new);
                        g = std::move(g_new);
                        f_prev = f_new;
                    }
                    return m_result;
                }

            private:
                bool m_use_linesearch;
            };

            // --------------------------------------------------------------------
            // Trust Region methods (Dogleg)
            // --------------------------------------------------------------------
            class TrustRegion : public OptimizerBase<TrustRegion>
            {
            public:
                TrustRegion(double delta0 = 1.0, double eta = 0.1)
                    : m_delta(delta0), m_delta_max(delta0 * 10.0), m_eta(eta) {}

                template <class Func, class GradFunc, class HessFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func, HessFunc&& hess_func,
                                        const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector g = grad_func(x);
                    m_result.ngev = 1;
                    double f = func(x);
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f);
                    double delta = m_delta;

                    while (true)
                    {
                        Matrix B = hess_func(x);
                        m_result.nhev++;

                        // Solve trust-region subproblem (dogleg method)
                        Vector p_B = -xt::linalg::solve(B, g);
                        double p_B_norm = xt::norm_l2(p_B)();

                        Vector p_U = -(xt::sum(g * g)() / xt::sum(g * xt::linalg::dot(B, g))()) * g;
                        double p_U_norm = xt::norm_l2(p_U)();

                        Vector p;
                        if (p_B_norm <= delta)
                        {
                            p = p_B;
                        }
                        else if (p_U_norm >= delta)
                        {
                            p = (delta / p_U_norm) * p_U;
                        }
                        else
                        {
                            // Dogleg path
                            Vector diff = p_B - p_U;
                            double a = xt::sum(diff * diff)();
                            double b = 2.0 * xt::sum(p_U * diff)();
                            double c = p_U_norm * p_U_norm - delta * delta;
                            double tau = (-b + std::sqrt(b*b - 4.0*a*c)) / (2.0*a);
                            tau = std::clamp(tau, 0.0, 1.0);
                            p = p_U + tau * diff;
                        }

                        Vector x_new = x + p;
                        double f_new = func(x_new);
                        m_result.nfev++;
                        m_result.history_fun.push_back(f_new);

                        // Compute actual reduction vs predicted reduction
                        Vector g_new = grad_func(x_new);
                        m_result.ngev++;
                        double ared = f - f_new;
                        double pred = -(xt::sum(g * p)() + 0.5 * xt::sum(p * xt::linalg::dot(B, p))());

                        double rho = ared / (pred + 1e-12);
                        if (rho > m_eta)
                        {
                            x = x_new;
                            g = g_new;
                            f = f_new;
                        }
                        if (rho > 0.75)
                            delta = std::min(2.0 * delta, m_delta_max);
                        else if (rho < 0.25)
                            delta *= 0.5;

                        m_result.nit++;
                        if (check_convergence(f_new, f, g_new, p))
                        {
                            m_result.x = x;
                            m_result.fun = f;
                            m_result.grad = g;
                            m_result.hessian = B;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }
                    }
                    return m_result;
                }

            private:
                double m_delta;
                double m_delta_max;
                double m_eta;
            };

            // --------------------------------------------------------------------
            // Levenberg-Marquardt for nonlinear least squares
            // --------------------------------------------------------------------
            class LevenbergMarquardt : public OptimizerBase<LevenbergMarquardt>
            {
            public:
                LevenbergMarquardt(double lambda0 = 1e-3, double nu = 2.0)
                    : m_lambda(lambda0), m_nu(nu) {}

                template <class ResidualFunc, class JacobianFunc>
                OptimizeResult minimize(ResidualFunc&& res_func, JacobianFunc&& jac_func,
                                        const Vector& x0, size_t m_residuals)
                {
                    reset_result(x0);
                    Vector x = x0;
                    Vector r = res_func(x);
                    m_result.nfev = 1;
                    double f = 0.5 * xt::sum(r * r)();
                    m_result.history_fun.push_back(f);
                    double lambda = m_lambda;

                    while (true)
                    {
                        Matrix J = jac_func(x);
                        m_result.ngev++;
                        Vector g = xt::linalg::dot(xt::transpose(J), r);
                        Matrix H = xt::linalg::dot(xt::transpose(J), J);

                        // Add damping
                        for (size_t i = 0; i < H.shape()[0]; ++i)
                            H(i, i) += lambda;

                        Vector p = -xt::linalg::solve(H, g);
                        Vector x_new = x + p;
                        Vector r_new = res_func(x_new);
                        m_result.nfev++;
                        double f_new = 0.5 * xt::sum(r_new * r_new)();
                        m_result.history_fun.push_back(f_new);

                        // Compute gain ratio
                        double ared = f - f_new;
                        double pred = -0.5 * xt::sum(p * (lambda * p + g))();
                        double rho = ared / (pred + 1e-12);

                        if (rho > 0.0)
                        {
                            x = x_new;
                            r = r_new;
                            f = f_new;
                            lambda = std::max(lambda / m_nu, 1e-12);
                        }
                        else
                        {
                            lambda = std::min(lambda * m_nu, 1e12);
                        }

                        m_result.nit++;
                        if (check_convergence(f_new, f, g, p))
                        {
                            m_result.x = x;
                            m_result.fun = f;
                            m_result.grad = g;
                            m_result.hessian = H;
                            m_result.success = (m_result.nit < m_options.maxiter);
                            break;
                        }
                    }
                    return m_result;
                }

            private:
                double m_lambda;
                double m_nu;
            };

            // --------------------------------------------------------------------
            // Constrained Optimization: Penalty and Augmented Lagrangian
            // --------------------------------------------------------------------
            class PenaltyMethod : public OptimizerBase<PenaltyMethod>
            {
            public:
                PenaltyMethod(double mu0 = 1.0, double beta = 10.0)
                    : m_mu(mu0), m_beta(beta) {}

                template <class Func, class GradFunc, class ConstraintFunc>
                OptimizeResult minimize(Func&& func, GradFunc&& grad_func,
                                        ConstraintFunc&& constr_func, // vector of equality constraints
                                        const Vector& x0)
                {
                    reset_result(x0);
                    Vector x = x0;
                    double mu = m_mu;

                    // Inner optimizer (L-BFGS)
                    LBFGS inner_opt(10, true);
                    inner_opt.set_ftol(m_options.ftol * 0.1);
                    inner_opt.set_gtol(m_options.gtol * 0.1);
                    inner_opt.set_maxiter(100);

                    for (size_t outer = 0; outer < 50; ++outer)
                    {
                        // Define penalty function
                        auto penalty_func = [&](const Vector& xv) {
                            double f = func(xv);
                            Vector c = constr_func(xv);
                            double penalty = 0.5 * mu * xt::sum(c * c)();
                            return f + penalty;
                        };

                        // Gradient of penalty function
                        auto penalty_grad = [&](const Vector& xv) {
                            Vector gf = grad_func(xv);
                            // Numerical gradient for constraints (simplified)
                            Vector gc = xt::zeros_like(xv);
                            Vector c = constr_func(xv);
                            double eps = 1e-6;
                            for (size_t i = 0; i < xv.size(); ++i)
                            {
                                Vector x_pert = xv;
                                x_pert(i) += eps;
                                Vector c_pert = constr_func(x_pert);
                                for (size_t j = 0; j < c.size(); ++j)
                                    gc(i) += mu * c(j) * (c_pert(j) - c(j)) / eps;
                            }
                            return gf + gc;
                        };

                        auto inner_result = inner_opt.minimize(penalty_func, penalty_grad, x);
                        x = inner_result.x;
                        m_result.nfev += inner_result.nfev;
                        m_result.ngev += inner_result.ngev;

                        Vector c = constr_func(x);
                        double constraint_violation = xt::norm_l2(c)();

                        if (constraint_violation < m_options.xtol)
                        {
                            m_result.success = true;
                            m_result.message = "Constraints satisfied";
                            break;
                        }

                        mu *= m_beta;
                        m_result.nit++;
                    }

                    m_result.x = x;
                    m_result.fun = func(x);
                    m_result.grad = grad_func(x);
                    return m_result;
                }

            private:
                double m_mu;
                double m_beta;
            };

            // --------------------------------------------------------------------
            // Global Optimization: Simulated Annealing
            // --------------------------------------------------------------------
            class SimulatedAnnealing : public OptimizerBase<SimulatedAnnealing>
            {
            public:
                SimulatedAnnealing(double T0 = 1000.0, double T_min = 1e-8, double alpha = 0.95)
                    : m_T0(T0), m_T_min(T_min), m_alpha(alpha) {}

                template <class Func>
                OptimizeResult minimize(Func&& func, const Vector& lower, const Vector& upper,
                                        const Vector& x0 = {})
                {
                    reset_result(x0.empty() ? 0.5 * (lower + upper) : x0);
                    Vector x = m_result.x;
                    Vector best_x = x;
                    double f = func(x);
                    double best_f = f;
                    m_result.nfev = 1;
                    m_result.history_fun.push_back(f);

                    std::mt19937 rng(std::random_device{}());
                    std::uniform_real_distribution<double> dist01(0.0, 1.0);

                    double T = m_T0;
                    size_t iter_per_T = 100 * x.size();

                    while (T > m_T_min && m_result.nfev < m_options.maxfev)
                    {
                        for (size_t i = 0; i < iter_per_T; ++i)
                        {
                            // Generate neighbor
                            Vector x_new = x;
                            for (size_t d = 0; d < x.size(); ++d)
                            {
                                double range = upper(d) - lower(d);
                                double step = range * 0.1 * (2.0 * dist01(rng) - 1.0);
                                x_new(d) = std::clamp(x(d) + step, lower(d), upper(d));
                            }

                            double f_new = func(x_new);
                            m_result.nfev++;

                            double delta = f_new - f;
                            if (delta < 0.0 || dist01(rng) < std::exp(-delta / T))
                            {
                                x = x_new;
                                f = f_new;
                                if (f < best_f)
                                {
                                    best_x = x;
                                    best_f = f;
                                }
                            }
                            m_result.history_fun.push_back(f);
                        }
                        T *= m_alpha;
                        m_result.nit++;
                    }

                    m_result.x = best_x;
                    m_result.fun = best_f;
                    m_result.success = true;
                    return m_result;
                }

            private:
                double m_T0;
                double m_T_min;
                double m_alpha;
            };

            // --------------------------------------------------------------------
            // Particle Swarm Optimization
            // --------------------------------------------------------------------
            class PSO : public OptimizerBase<PSO>
            {
            public:
                PSO(size_t n_particles = 30, double w = 0.5, double c1 = 1.5, double c2 = 1.5)
                    : m_n_particles(n_particles), m_w(w), m_c1(c1), m_c2(c2) {}

                template <class Func>
                OptimizeResult minimize(Func&& func, const Vector& lower, const Vector& upper)
                {
                    size_t dim = lower.size();
                    std::mt19937 rng(std::random_device{}());
                    std::uniform_real_distribution<double> dist01(0.0, 1.0);

                    // Initialize particles
                    std::vector<Vector> positions(m_n_particles);
                    std::vector<Vector> velocities(m_n_particles);
                    std::vector<double> fitness(m_n_particles);
                    std::vector<Vector> best_positions(m_n_particles);
                    std::vector<double> best_fitness(m_n_particles);

                    Vector global_best_position(dim);
                    double global_best_fitness = std::numeric_limits<double>::max();

                    for (size_t i = 0; i < m_n_particles; ++i)
                    {
                        positions[i] = Vector({dim});
                        velocities[i] = Vector({dim});
                        for (size_t d = 0; d < dim; ++d)
                        {
                            positions[i](d) = lower(d) + dist01(rng) * (upper(d) - lower(d));
                            velocities[i](d) = 0.1 * (upper(d) - lower(d)) * (2.0 * dist01(rng) - 1.0);
                        }
                        fitness[i] = func(positions[i]);
                        m_result.nfev++;
                        best_positions[i] = positions[i];
                        best_fitness[i] = fitness[i];
                        if (fitness[i] < global_best_fitness)
                        {
                            global_best_fitness = fitness[i];
                            global_best_position = positions[i];
                        }
                    }

                    while (m_result.nit < m_options.maxiter && m_result.nfev < m_options.maxfev)
                    {
                        for (size_t i = 0; i < m_n_particles; ++i)
                        {
                            for (size_t d = 0; d < dim; ++d)
                            {
                                double r1 = dist01(rng), r2 = dist01(rng);
                                velocities[i](d) = m_w * velocities[i](d)
                                    + m_c1 * r1 * (best_positions[i](d) - positions[i](d))
                                    + m_c2 * r2 * (global_best_position(d) - positions[i](d));
                            }
                            positions[i] = positions[i] + velocities[i];
                            // Clamp to bounds
                            for (size_t d = 0; d < dim; ++d)
                                positions[i](d) = std::clamp(positions[i](d), lower(d), upper(d));

                            fitness[i] = func(positions[i]);
                            m_result.nfev++;
                            if (fitness[i] < best_fitness[i])
                            {
                                best_fitness[i] = fitness[i];
                                best_positions[i] = positions[i];
                                if (fitness[i] < global_best_fitness)
                                {
                                    global_best_fitness = fitness[i];
                                    global_best_position = positions[i];
                                }
                            }
                        }
                        m_result.history_fun.push_back(global_best_fitness);
                        m_result.nit++;

                        if (m_verbose && m_result.nit % 10 == 0)
                        {
                            std::cout << "PSO: iter=" << m_result.nit
                                      << ", best_f=" << global_best_fitness << std::endl;
                        }

                        // Simple convergence: if global best hasn't improved much
                        if (m_result.nit > 20 &&
                            std::abs(m_result.history_fun[m_result.nit-1] - m_result.history_fun[m_result.nit-11]) < m_options.ftol)
                        {
                            break;
                        }
                    }

                    m_result.x = global_best_position;
                    m_result.fun = global_best_fitness;
                    m_result.success = true;
                    return m_result;
                }

            private:
                size_t m_n_particles;
                double m_w;
                double m_c1;
                double m_c2;
            };

            // --------------------------------------------------------------------
            // Convenience wrapper functions
            // --------------------------------------------------------------------
            template <class Func, class GradFunc>
            inline OptimizeResult minimize_gradient_descent(Func&& f, GradFunc&& g, const Vector& x0,
                                                            double lr = 0.01, size_t maxiter = 1000)
            {
                GradientDescent opt(lr);
                opt.set_maxiter(maxiter);
                return opt.minimize(std::forward<Func>(f), std::forward<GradFunc>(g), x0);
            }

            template <class Func, class GradFunc>
            inline OptimizeResult minimize_adam(Func&& f, GradFunc&& g, const Vector& x0,
                                                double lr = 0.001, size_t maxiter = 1000)
            {
                Adam opt(lr);
                opt.set_maxiter(maxiter);
                return opt.minimize(std::forward<Func>(f), std::forward<GradFunc>(g), x0);
            }

            template <class Func, class GradFunc>
            inline OptimizeResult minimize_bfgs(Func&& f, GradFunc&& g, const Vector& x0,
                                                size_t maxiter = 200)
            {
                BFGS opt(true);
                opt.set_maxiter(maxiter);
                return opt.minimize(std::forward<Func>(f), std::forward<GradFunc>(g), x0);
            }

            template <class Func, class GradFunc>
            inline OptimizeResult minimize_lbfgs(Func&& f, GradFunc&& g, const Vector& x0,
                                                 size_t maxiter = 500)
            {
                LBFGS opt(10, true);
                opt.set_maxiter(maxiter);
                return opt.minimize(std::forward<Func>(f), std::forward<GradFunc>(g), x0);
            }

            template <class Func>
            inline OptimizeResult minimize_simulated_annealing(Func&& f, const Vector& lb, const Vector& ub,
                                                               const Vector& x0 = {},
                                                               double T0 = 1000.0, size_t maxfev = 10000)
            {
                SimulatedAnnealing opt(T0);
                opt.set_maxfev(maxfev);
                return opt.minimize(std::forward<Func>(f), lb, ub, x0);
            }

            template <class Func>
            inline OptimizeResult minimize_pso(Func&& f, const Vector& lb, const Vector& ub,
                                               size_t n_particles = 30, size_t maxiter = 200)
            {
                PSO opt(n_particles);
                opt.set_maxiter(maxiter);
                return opt.minimize(std::forward<Func>(f), lb, ub);
            }

        } // namespace optimize

        // Bring into xt namespace
        using optimize::OptimizeResult;
        using optimize::ConvergenceCriteria;
        using optimize::GradientDescent;
        using optimize::Adam;
        using optimize::RMSProp;
        using optimize::BFGS;
        using optimize::LBFGS;
        using optimize::ConjugateGradient;
        using optimize::Newton;
        using optimize::TrustRegion;
        using optimize::LevenbergMarquardt;
        using optimize::PenaltyMethod;
        using optimize::SimulatedAnnealing;
        using optimize::PSO;
        using optimize::minimize_gradient_descent;
        using optimize::minimize_adam;
        using optimize::minimize_bfgs;
        using optimize::minimize_lbfgs;
        using optimize::minimize_simulated_annealing;
        using optimize::minimize_pso;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XOPTIMIZE_HPP

// math/xoptimize.hpp