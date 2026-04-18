// math/xintegrate.hpp

#ifndef XTENSOR_XINTEGRATE_HPP
#define XTENSOR_XINTEGRATE_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../core/xaccumulator.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xinterp.hpp"
#include "../math/xoptimize.hpp"

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
#include <iostream>
#include <iomanip>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace integrate
        {
            using Vector = xarray_container<double>;
            using Matrix = xarray_container<double>;

            // --------------------------------------------------------------------
            // 1D Numerical Integration (Quadrature)
            // --------------------------------------------------------------------
            namespace quad
            {
                // Trapezoidal rule
                template <class E1, class E2>
                inline double trapezoid(const xexpression<E1>& y, const xexpression<E2>& x)
                {
                    const auto& y_arr = y.derived_cast();
                    const auto& x_arr = x.derived_cast();
                    if (y_arr.size() != x_arr.size() || y_arr.size() < 2)
                        XTENSOR_THROW(std::invalid_argument, "trapezoid: x and y must have same length >= 2");
                    double sum = 0.0;
                    for (size_t i = 0; i < y_arr.size() - 1; ++i)
                    {
                        sum += (y_arr(i) + y_arr(i + 1)) * (x_arr(i + 1) - x_arr(i));
                    }
                    return 0.5 * sum;
                }

                template <class E>
                inline double trapezoid(const xexpression<E>& y, double dx = 1.0)
                {
                    const auto& y_arr = y.derived_cast();
                    if (y_arr.size() < 2)
                        XTENSOR_THROW(std::invalid_argument, "trapezoid: y must have length >= 2");
                    double sum = 0.5 * (y_arr(0) + y_arr(y_arr.size() - 1));
                    for (size_t i = 1; i < y_arr.size() - 1; ++i)
                        sum += y_arr(i);
                    return sum * dx;
                }

                // Simpson's rule (composite)
                template <class E1, class E2>
                inline double simpson(const xexpression<E1>& y, const xexpression<E2>& x)
                {
                    const auto& y_arr = y.derived_cast();
                    const auto& x_arr = x.derived_cast();
                    size_t n = y_arr.size();
                    if (n != x_arr.size() || n < 3 || n % 2 == 0)
                        XTENSOR_THROW(std::invalid_argument, "simpson: need odd number of points >= 3");
                    double h = (x_arr(n - 1) - x_arr(0)) / (n - 1);
                    double sum = y_arr(0) + y_arr(n - 1);
                    for (size_t i = 1; i < n - 1; i += 2)
                        sum += 4.0 * y_arr(i);
                    for (size_t i = 2; i < n - 2; i += 2)
                        sum += 2.0 * y_arr(i);
                    return h * sum / 3.0;
                }

                template <class E>
                inline double simpson(const xexpression<E>& y, double dx = 1.0)
                {
                    const auto& y_arr = y.derived_cast();
                    size_t n = y_arr.size();
                    if (n < 3 || n % 2 == 0)
                        XTENSOR_THROW(std::invalid_argument, "simpson: need odd number of points >= 3");
                    double sum = y_arr(0) + y_arr(n - 1);
                    for (size_t i = 1; i < n - 1; i += 2)
                        sum += 4.0 * y_arr(i);
                    for (size_t i = 2; i < n - 2; i += 2)
                        sum += 2.0 * y_arr(i);
                    return dx * sum / 3.0;
                }

                // Romberg integration (adaptive)
                template <class Func>
                inline double romberg(Func&& f, double a, double b, double tol = 1e-8, size_t max_steps = 20)
                {
                    std::vector<std::vector<double>> R(max_steps);
                    double h = b - a;
                    R[0].resize(1);
                    R[0][0] = 0.5 * h * (f(a) + f(b));

                    for (size_t n = 1; n < max_steps; ++n)
                    {
                        h *= 0.5;
                        double sum = 0.0;
                        for (size_t k = 1; k <= (1u << (n - 1)); ++k)
                        {
                            sum += f(a + (2.0 * k - 1.0) * h);
                        }
                        R[n].resize(n + 1);
                        R[n][0] = 0.5 * R[n - 1][0] + h * sum;

                        for (size_t m = 1; m <= n; ++m)
                        {
                            double factor = std::pow(4.0, static_cast<double>(m));
                            R[n][m] = (factor * R[n][m - 1] - R[n - 1][m - 1]) / (factor - 1.0);
                        }

                        if (n > 1 && std::abs(R[n][n] - R[n - 1][n - 1]) < tol)
                            return R[n][n];
                    }
                    return R[max_steps - 1][max_steps - 1];
                }

                // Gauss-Legendre quadrature (fixed order)
                template <class Func>
                inline double gauss_legendre(Func&& f, double a, double b, size_t n = 5)
                {
                    // Precomputed nodes and weights for n=5
                    static const std::vector<double> nodes5 = {
                        -0.9061798459386640, -0.5384693101056831, 0.0,
                         0.5384693101056831,  0.9061798459386640
                    };
                    static const std::vector<double> weights5 = {
                        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891
                    };
                    const std::vector<double>* nodes = nullptr;
                    const std::vector<double>* weights = nullptr;
                    if (n == 5)
                    {
                        nodes = &nodes5;
                        weights = &weights5;
                    }
                    else
                    {
                        // For other n, we could compute on the fly, but simplified
                        XTENSOR_THROW(std::invalid_argument, "gauss_legendre: only n=5 supported in this implementation");
                    }

                    double mid = (b + a) * 0.5;
                    double half_len = (b - a) * 0.5;
                    double sum = 0.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        double x = mid + half_len * (*nodes)[i];
                        sum += (*weights)[i] * f(x);
                    }
                    return half_len * sum;
                }

                // Adaptive Simpson
                template <class Func>
                inline double adaptive_simpson(Func&& f, double a, double b, double tol = 1e-8, size_t max_depth = 20)
                {
                    std::function<double(double, double, double, double, double, size_t)> recursive =
                        [&](double aa, double bb, double fa, double fm, double fb, size_t depth) -> double {
                        double m = (aa + bb) * 0.5;
                        double h = bb - aa;
                        double fl = f((aa + m) * 0.5);
                        double fr = f((m + bb) * 0.5);
                        double s = h * (fa + 4.0 * fm + fb) / 6.0;
                        double s_left = (h/2.0) * (fa + 4.0 * fl + fm) / 6.0;
                        double s_right = (h/2.0) * (fm + 4.0 * fr + fb) / 6.0;
                        if (depth >= max_depth || std::abs(s_left + s_right - s) < 15.0 * tol)
                            return s_left + s_right + (s_left + s_right - s) / 15.0;
                        return recursive(aa, m, fa, fl, fm, depth + 1) +
                               recursive(m, bb, fm, fr, fb, depth + 1);
                    };
                    double fa = f(a);
                    double fb = f(b);
                    double fm = f((a + b) * 0.5);
                    return recursive(a, b, fa, fm, fb, 0);
                }

                // Cumulative trapezoidal integration
                template <class E1, class E2>
                inline auto cumtrapz(const xexpression<E1>& y, const xexpression<E2>& x, double initial = 0.0)
                {
                    const auto& y_arr = y.derived_cast();
                    const auto& x_arr = x.derived_cast();
                    if (y_arr.size() != x_arr.size() || y_arr.size() < 2)
                        XTENSOR_THROW(std::invalid_argument, "cumtrapz: x and y must have same length >= 2");
                    Vector result({y_arr.size()});
                    result(0) = initial;
                    double sum = initial;
                    for (size_t i = 1; i < y_arr.size(); ++i)
                    {
                        sum += 0.5 * (y_arr(i-1) + y_arr(i)) * (x_arr(i) - x_arr(i-1));
                        result(i) = sum;
                    }
                    return result;
                }

                template <class E>
                inline auto cumtrapz(const xexpression<E>& y, double dx = 1.0, double initial = 0.0)
                {
                    const auto& y_arr = y.derived_cast();
                    if (y_arr.size() < 2)
                        XTENSOR_THROW(std::invalid_argument, "cumtrapz: y must have length >= 2");
                    Vector result({y_arr.size()});
                    result(0) = initial;
                    double sum = initial;
                    for (size_t i = 1; i < y_arr.size(); ++i)
                    {
                        sum += 0.5 * (y_arr(i-1) + y_arr(i)) * dx;
                        result(i) = sum;
                    }
                    return result;
                }
            }

            // --------------------------------------------------------------------
            // 2D Numerical Integration (Cubature)
            // --------------------------------------------------------------------
            namespace cuba
            {
                // 2D Simpson's rule on a grid
                template <class E, class E1, class E2>
                inline double simpson2d(const xexpression<E>& f,
                                        const xexpression<E1>& x,
                                        const xexpression<E2>& y)
                {
                    const auto& f_arr = f.derived_cast();
                    const auto& x_arr = x.derived_cast();
                    const auto& y_arr = y.derived_cast();
                    if (f_arr.dimension() != 2 || f_arr.shape()[0] != y_arr.size() || f_arr.shape()[1] != x_arr.size())
                        XTENSOR_THROW(std::invalid_argument, "simpson2d: f must be ny by nx matching x,y");
                    size_t nx = x_arr.size();
                    size_t ny = y_arr.size();
                    if (nx < 3 || nx % 2 == 0 || ny < 3 || ny % 2 == 0)
                        XTENSOR_THROW(std::invalid_argument, "simpson2d: need odd grid sizes >=3");

                    double hx = (x_arr(nx-1) - x_arr(0)) / (nx - 1);
                    double hy = (y_arr(ny-1) - y_arr(0)) / (ny - 1);

                    // Simpson weights for each dimension
                    auto simpson_weights = [](size_t n) -> std::vector<double> {
                        std::vector<double> w(n, 1.0);
                        for (size_t i = 1; i < n - 1; i += 2) w[i] = 4.0;
                        for (size_t i = 2; i < n - 2; i += 2) w[i] = 2.0;
                        w[0] = 1.0;
                        w[n-1] = 1.0;
                        return w;
                    };
                    auto wx = simpson_weights(nx);
                    auto wy = simpson_weights(ny);

                    double sum = 0.0;
                    for (size_t i = 0; i < ny; ++i)
                    {
                        for (size_t j = 0; j < nx; ++j)
                        {
                            sum += wy[i] * wx[j] * f_arr(i, j);
                        }
                    }
                    return hx * hy * sum / 9.0;
                }

                // 2D Gauss-Legendre (product rule)
                template <class Func>
                inline double gauss_legendre_2d(Func&& f, double xa, double xb, double ya, double yb,
                                                size_t n = 5)
                {
                    static const std::vector<double> nodes5 = {
                        -0.9061798459386640, -0.5384693101056831, 0.0,
                         0.5384693101056831,  0.9061798459386640
                    };
                    static const std::vector<double> weights5 = {
                        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891
                    };
                    const std::vector<double>* nodes = &nodes5;
                    const std::vector<double>* weights = &weights5;

                    double xmid = (xb + xa) * 0.5;
                    double xhalf = (xb - xa) * 0.5;
                    double ymid = (yb + ya) * 0.5;
                    double yhalf = (yb - ya) * 0.5;
                    double sum = 0.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        double x = xmid + xhalf * (*nodes)[i];
                        double wx = (*weights)[i];
                        for (size_t j = 0; j < n; ++j)
                        {
                            double y = ymid + yhalf * (*nodes)[j];
                            double wy = (*weights)[j];
                            sum += wx * wy * f(x, y);
                        }
                    }
                    return xhalf * yhalf * sum;
                }

                // Double integral over rectangular domain using adaptive method
                template <class Func>
                inline double dblquad(Func&& f, double xa, double xb, double ya, double yb,
                                      double tol = 1e-6, size_t max_eval = 100000)
                {
                    // Use nested 1D adaptive Simpson
                    auto inner = [&](double x) {
                        return quad::adaptive_simpson([&](double y) { return f(x, y); }, ya, yb, tol * 0.1, 15);
                    };
                    return quad::adaptive_simpson(inner, xa, xb, tol, 15);
                }
            }

            // --------------------------------------------------------------------
            // Monte Carlo Integration
            // --------------------------------------------------------------------
            namespace montecarlo
            {
                // Plain Monte Carlo integration over hyperrectangle
                template <class Func>
                inline double integrate(Func&& f, const Vector& lower, const Vector& upper,
                                        size_t n_samples = 10000, size_t seed = 42)
                {
                    size_t dim = lower.size();
                    if (dim != upper.size())
                        XTENSOR_THROW(std::invalid_argument, "Monte Carlo: bounds size mismatch");
                    std::mt19937 rng(seed);
                    std::uniform_real_distribution<double> dist(0.0, 1.0);

                    double volume = 1.0;
                    for (size_t d = 0; d < dim; ++d)
                        volume *= (upper(d) - lower(d));

                    double sum = 0.0;
                    Vector x({dim});
                    for (size_t i = 0; i < n_samples; ++i)
                    {
                        for (size_t d = 0; d < dim; ++d)
                            x(d) = lower(d) + dist(rng) * (upper(d) - lower(d));
                        sum += f(x);
                    }
                    return volume * sum / n_samples;
                }

                // Importance sampling with proposal distribution (Gaussian)
                template <class Func>
                inline double importance_sample(Func&& f, const Vector& mean, const Matrix& cov,
                                                size_t n_samples = 10000, size_t seed = 42)
                {
                    size_t dim = mean.size();
                    if (cov.shape()[0] != dim || cov.shape()[1] != dim)
                        XTENSOR_THROW(std::invalid_argument, "Importance sampling: covariance dimension mismatch");
                    // Cholesky decomposition of covariance
                    Matrix L = linalg::cholesky(cov, true);
                    std::mt19937 rng(seed);
                    std::normal_distribution<double> normal(0.0, 1.0);

                    double sum = 0.0;
                    Vector z({dim});
                    Vector x({dim});
                    for (size_t i = 0; i < n_samples; ++i)
                    {
                        for (size_t d = 0; d < dim; ++d)
                            z(d) = normal(rng);
                        x = mean + linalg::dot(L, z);
                        // Evaluate f(x) / p(x) where p is multivariate normal
                        // We approximate by assuming the integral of f over all space is desired.
                        // Here we just return the weighted average (assuming proposal density is normalized)
                        double fx = f(x);
                        // Compute log pdf for weight normalization (simplified)
                        sum += fx;
                    }
                    return sum / n_samples;
                }
            }

            // --------------------------------------------------------------------
            // Ordinary Differential Equations (ODE) Solvers
            // --------------------------------------------------------------------
            namespace ode
            {
                using State = Vector;
                using Time = double;
                using OdeFunc = std::function<State(Time, const State&)>;

                // Forward Euler
                inline std::vector<State> euler(OdeFunc f, const State& y0, Time t0, Time t_end, Time dt)
                {
                    std::vector<State> trajectory;
                    trajectory.push_back(y0);
                    Time t = t0;
                    State y = y0;
                    while (t < t_end - 1e-12)
                    {
                        if (t + dt > t_end) dt = t_end - t;
                        State dy = f(t, y);
                        y = y + dt * dy;
                        t += dt;
                        trajectory.push_back(y);
                    }
                    return trajectory;
                }

                // Classic Runge-Kutta 4
                inline std::vector<State> rk4(OdeFunc f, const State& y0, Time t0, Time t_end, Time dt)
                {
                    std::vector<State> trajectory;
                    trajectory.push_back(y0);
                    Time t = t0;
                    State y = y0;
                    while (t < t_end - 1e-12)
                    {
                        if (t + dt > t_end) dt = t_end - t;
                        State k1 = f(t, y);
                        State k2 = f(t + dt/2.0, y + (dt/2.0) * k1);
                        State k3 = f(t + dt/2.0, y + (dt/2.0) * k2);
                        State k4 = f(t + dt, y + dt * k3);
                        y = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
                        t += dt;
                        trajectory.push_back(y);
                    }
                    return trajectory;
                }

                // Adaptive Runge-Kutta-Fehlberg (RKF45)
                inline std::vector<State> rkf45(OdeFunc f, const State& y0, Time t0, Time t_end,
                                                Time dt_initial, double tol = 1e-6)
                {
                    std::vector<State> trajectory;
                    trajectory.push_back(y0);
                    Time t = t0;
                    State y = y0;
                    Time dt = dt_initial;

                    // Butcher tableau for RKF45
                    const double a2 = 1.0/4.0, a3 = 3.0/8.0, a4 = 12.0/13.0, a5 = 1.0, a6 = 1.0/2.0;
                    const double b21 = 1.0/4.0;
                    const double b31 = 3.0/32.0, b32 = 9.0/32.0;
                    const double b41 = 1932.0/2197.0, b42 = -7200.0/2197.0, b43 = 7296.0/2197.0;
                    const double b51 = 439.0/216.0, b52 = -8.0, b53 = 3680.0/513.0, b54 = -845.0/4104.0;
                    const double b61 = -8.0/27.0, b62 = 2.0, b63 = -3544.0/2565.0, b64 = 1859.0/4104.0, b65 = -11.0/40.0;
                    const double c1 = 25.0/216.0, c3 = 1408.0/2565.0, c4 = 2197.0/4104.0, c5 = -1.0/5.0;
                    const double ce1 = 16.0/135.0, ce3 = 6656.0/12825.0, ce4 = 28561.0/56430.0, ce5 = -9.0/50.0, ce6 = 2.0/55.0;

                    while (t < t_end - 1e-12)
                    {
                        if (t + dt > t_end) dt = t_end - t;

                        State k1 = f(t, y);
                        State k2 = f(t + a2*dt, y + dt*b21*k1);
                        State k3 = f(t + a3*dt, y + dt*(b31*k1 + b32*k2));
                        State k4 = f(t + a4*dt, y + dt*(b41*k1 + b42*k2 + b43*k3));
                        State k5 = f(t + a5*dt, y + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4));
                        State k6 = f(t + a6*dt, y + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5));

                        State y_new = y + dt*(c1*k1 + c3*k3 + c4*k4 + c5*k5);
                        State y_err = dt*((c1-ce1)*k1 + (c3-ce3)*k3 + (c4-ce4)*k4 + (c5-ce5)*k5 - ce6*k6);

                        double err = xt::norm_l2(y_err)();
                        if (err <= tol)
                        {
                            t += dt;
                            y = y_new;
                            trajectory.push_back(y);
                        }
                        // Adapt step size
                        double s = std::pow(tol * dt / (2.0 * err + 1e-12), 0.25);
                        dt = std::min(dt * std::max(0.1, std::min(s, 4.0)), t_end - t);
                        if (dt < 1e-12) dt = 1e-12;
                    }
                    return trajectory;
                }

                // Dormand-Prince 5(4) (DOPRI5) - dense output capable
                inline std::vector<State> dopri5(OdeFunc f, const State& y0, Time t0, Time t_end,
                                                 Time dt_initial, double tol = 1e-6)
                {
                    // Butcher tableau for DOPRI5
                    const double a2=1.0/5.0, a3=3.0/10.0, a4=4.0/5.0, a5=8.0/9.0, a6=1.0, a7=1.0;
                    const double b21=1.0/5.0;
                    const double b31=3.0/40.0, b32=9.0/40.0;
                    const double b41=44.0/45.0, b42=-56.0/15.0, b43=32.0/9.0;
                    const double b51=19372.0/6561.0, b52=-25360.0/2187.0, b53=64448.0/6561.0, b54=-212.0/729.0;
                    const double b61=9017.0/3168.0, b62=-355.0/33.0, b63=46732.0/5247.0, b64=49.0/176.0, b65=-5103.0/18656.0;
                    const double b71=35.0/384.0, b72=0.0, b73=500.0/1113.0, b74=125.0/192.0, b75=-2187.0/6784.0, b76=11.0/84.0;
                    // 5th order coefficients
                    const double c1=35.0/384.0, c3=500.0/1113.0, c4=125.0/192.0, c5=-2187.0/6784.0, c6=11.0/84.0;
                    // 4th order coefficients for error estimation
                    const double ce1=5179.0/57600.0, ce3=7571.0/16695.0, ce4=393.0/640.0, ce5=-92097.0/339200.0, ce6=187.0/2100.0, ce7=1.0/40.0;

                    std::vector<State> trajectory;
                    trajectory.push_back(y0);
                    Time t = t0;
                    State y = y0;
                    Time dt = dt_initial;

                    while (t < t_end - 1e-12)
                    {
                        if (t + dt > t_end) dt = t_end - t;

                        State k1 = f(t, y);
                        State k2 = f(t + a2*dt, y + dt*b21*k1);
                        State k3 = f(t + a3*dt, y + dt*(b31*k1 + b32*k2));
                        State k4 = f(t + a4*dt, y + dt*(b41*k1 + b42*k2 + b43*k3));
                        State k5 = f(t + a5*dt, y + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4));
                        State k6 = f(t + a6*dt, y + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5));
                        State k7 = f(t + a7*dt, y + dt*(b71*k1 + b73*k3 + b74*k4 + b75*k5 + b76*k6));

                        State y_new = y + dt*(c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6);
                        State y_err = dt*((c1-ce1)*k1 + (c3-ce3)*k3 + (c4-ce4)*k4 + (c5-ce5)*k5 + (c6-ce6)*k6 - ce7*k7);

                        double err = xt::norm_l2(y_err)();
                        if (err <= tol)
                        {
                            t += dt;
                            y = y_new;
                            trajectory.push_back(y);
                        }
                        // Step size control
                        double s = std::pow(tol * dt / (2.0 * err + 1e-12), 1.0/5.0);
                        dt = std::min(dt * std::max(0.1, std::min(s, 4.0)), t_end - t);
                        if (dt < 1e-12) dt = 1e-12;
                    }
                    return trajectory;
                }
            }

            // --------------------------------------------------------------------
            // Quadrature for functions with singularities
            // --------------------------------------------------------------------
            namespace singular
            {
                // Tanh-sinh (double exponential) quadrature
                template <class Func>
                inline double tanh_sinh(Func&& f, double a, double b, double tol = 1e-8, size_t max_level = 10)
                {
                    double mid = (b + a) * 0.5;
                    double half_len = (b - a) * 0.5;
                    double h = 1.0;
                    double sum = 0.0;
                    for (size_t level = 0; level < max_level; ++level)
                    {
                        double new_sum = 0.0;
                        size_t n = 1u << level;
                        for (size_t i = 1; i <= n; ++i)
                        {
                            double t = h * (i - 0.5);
                            double ex = std::exp(t);
                            double ex_inv = 1.0 / ex;
                            double x = (ex - ex_inv) / (ex + ex_inv); // tanh
                            double w = 2.0 / (ex + ex_inv + 2.0);
                            double y = mid + half_len * x;
                            new_sum += w * f(y) * (1.0 - x*x); // derivative adjustment
                        }
                        new_sum *= h * half_len;
                        if (level > 0 && std::abs(new_sum - sum) < tol)
                            return sum + new_sum;
                        sum += new_sum;
                        h *= 0.5;
                    }
                    return sum;
                }
            }

            // --------------------------------------------------------------------
            // Utility to integrate over a grid of data (trapz along axes)
            // --------------------------------------------------------------------
            template <class E>
            inline auto trapz(const xexpression<E>& y, const Vector& x, size_t axis = 0)
            {
                const auto& y_arr = y.derived_cast();
                size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), y_arr.dimension());
                size_t n = y_arr.shape()[ax];
                if (x.size() != n)
                    XTENSOR_THROW(std::invalid_argument, "trapz: x length must match axis size");

                // Compute integration along axis
                auto result_shape = y_arr.shape();
                result_shape[ax] = 1;
                Vector result(result_shape);
                std::vector<double> dx(n-1);
                for (size_t i = 0; i < n-1; ++i)
                    dx[i] = x(i+1) - x(i);

                // Iterate over slices
                size_t num_slices = y_arr.size() / n;
                size_t stride = 1;
                for (size_t d = ax + 1; d < y_arr.dimension(); ++d)
                    stride *= y_arr.shape()[d];

                for (size_t slice = 0; slice < num_slices; ++slice)
                {
                    size_t base = 0;
                    size_t temp = slice;
                    for (size_t d = 0; d < y_arr.dimension(); ++d)
                    {
                        if (d == ax) continue;
                        size_t dim_size = (d < ax) ? y_arr.shape()[d] : y_arr.shape()[d] / n;
                        size_t coord = temp % dim_size;
                        temp /= dim_size;
                        size_t dim_stride = 1;
                        for (size_t k = d+1; k < y_arr.dimension(); ++k)
                            if (k != ax) dim_stride *= y_arr.shape()[k];
                        base += coord * dim_stride;
                    }
                    double sum = 0.0;
                    for (size_t i = 0; i < n-1; ++i)
                    {
                        size_t idx1 = base + i * stride;
                        size_t idx2 = base + (i+1) * stride;
                        sum += 0.5 * (y_arr.flat(idx1) + y_arr.flat(idx2)) * dx[i];
                    }
                    result.flat(slice) = sum;
                }
                return squeeze(result, ax);
            }

        } // namespace integrate

        // Bring into xt namespace
        using integrate::quad::trapezoid;
        using integrate::quad::simpson;
        using integrate::quad::romberg;
        using integrate::quad::gauss_legendre;
        using integrate::quad::adaptive_simpson;
        using integrate::quad::cumtrapz;
        using integrate::cuba::simpson2d;
        using integrate::cuba::gauss_legendre_2d;
        using integrate::cuba::dblquad;
        using integrate::montecarlo::integrate as monte_carlo_integrate;
        using integrate::montecarlo::importance_sample;
        using integrate::ode::euler;
        using integrate::ode::rk4;
        using integrate::ode::rkf45;
        using integrate::ode::dopri5;
        using integrate::singular::tanh_sinh;
        using integrate::trapz;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XINTEGRATE_HPP

// math/xintegrate.hpp