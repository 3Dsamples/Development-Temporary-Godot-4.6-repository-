// system name : onetbb-warp
// File 0023 : core/math/quadrature.h
// Description : Numerical integration (quadrature) methods for 1D, 2D, 3D, and Monte Carlo.

#ifndef __TBB_WARP_CORE_MATH_QUADRATURE_H
#define __TBB_WARP_CORE_MATH_QUADRATURE_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/random.h"
#include <cmath>
#include <functional>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Midpoint rule
// ============================================================

template<typename Func>
double midpoint_rule(Func f, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = a + h * (i + 0.5);
        sum += f(x);
    }
    return sum * h;
}

// ============================================================
// Trapezoidal rule
// ============================================================

template<typename Func>
double trapezoidal_rule(Func f, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; ++i) {
        sum += f(a + i * h);
    }
    return sum * h;
}

// ============================================================
// Simpson's rule (n must be even)
// ============================================================

template<typename Func>
double simpson_rule(Func f, double a, double b, int n) {
    if (n % 2 != 0) n++;
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    for (int i = 1; i < n; i += 2) {
        sum += 4.0 * f(a + i * h);
    }
    for (int i = 2; i < n - 1; i += 2) {
        sum += 2.0 * f(a + i * h);
    }
    return sum * h / 3.0;
}

// ============================================================
// Romberg integration (Richardson extrapolation)
// ============================================================

template<typename Func>
double romberg_integration(Func f, double a, double b, int max_order = 10, double tol = 1e-12) {
    std::vector<std::vector<double>> R(max_order + 1, std::vector<double>(max_order + 1, 0.0));
    double h = b - a;
    R[0][0] = 0.5 * h * (f(a) + f(b));
    for (int i = 1; i <= max_order; ++i) {
        h *= 0.5;
        double sum = 0.0;
        int n = 1 << (i - 1);
        for (int k = 1; k <= n; ++k) {
            sum += f(a + (2.0 * k - 1.0) * h);
        }
        R[i][0] = 0.5 * R[i-1][0] + sum * h;
        for (int j = 1; j <= i; ++j) {
            double four_j = std::pow(4.0, j);
            R[i][j] = (four_j * R[i][j-1] - R[i-1][j-1]) / (four_j - 1.0);
        }
        if (i >= 2 && std::abs(R[i][i] - R[i-1][i-1]) < tol * std::abs(R[i][i])) {
            return R[i][i];
        }
    }
    return R[max_order][max_order];
}

// ============================================================
// Adaptive Simpson's rule
// ============================================================

template<typename Func>
double adaptive_simpson_impl(Func f, double a, double b, double eps, int max_depth,
                             double fa, double fb, double fm, double whole) {
    double h = (b - a) * 0.5;
    double c = a + h;
    double fl = f(a + h * 0.5);
    double fr = f(c + h * 0.5);
    double left  = (fa + 4.0*fl + fm) * h / 6.0;
    double right = (fm + 4.0*fr + fb) * h / 6.0;
    double total = left + right;
    if (max_depth <= 0 || std::abs(total - whole) <= 15.0 * eps) {
        return total + (total - whole) / 15.0;
    }
    return adaptive_simpson_impl(f, a, c, eps/2.0, max_depth-1, fa, fm, fl, left) +
           adaptive_simpson_impl(f, c, b, eps/2.0, max_depth-1, fm, fb, fr, right);
}

template<typename Func>
double adaptive_simpson(Func f, double a, double b, double eps = 1e-8, int max_depth = 30) {
    double fa = f(a), fb = f(b), fm = f((a+b)*0.5);
    double whole = (b - a) * (fa + 4.0*fm + fb) / 6.0;
    return adaptive_simpson_impl(f, a, b, eps, max_depth, fa, fb, fm, whole);
}

// ============================================================
// Gauss‑Legendre quadrature (pre‑computed nodes and weights)
// ============================================================

inline std::pair<std::vector<double>, std::vector<double>> gauss_legendre_nodes_weights(int n) {
    std::vector<double> nodes(n), weights(n);
    double eps = 1e-14;
    int m = (n + 1) / 2;
    for (int i = 0; i < m; ++i) {
        double z = std::cos(PI_D * (i + 0.75) / (n + 0.5));
        double z1;
        do {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < n; ++j) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
            }
            double pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (std::abs(z - z1) > eps);
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        double w = 2.0 / ((1.0 - z * z) * pp * pp);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    return {nodes, weights};
}

template<typename Func>
double gauss_legendre_integrate(Func f, double a, double b, int n) {
    auto [nodes, weights] = gauss_legendre_nodes_weights(n);
    double half = (b - a) * 0.5;
    double mid = (a + b) * 0.5;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += weights[i] * f(mid + half * nodes[i]);
    }
    return sum * half;
}

// ============================================================
// Double integral (cubature) over rectangle [ax,bx]×[ay,by]
// ============================================================

template<typename Func>
double double_integral_tensor(Func f, double ax, double bx, double ay, double by,
                              int nx, int ny) {
    auto [nodes_x, weights_x] = gauss_legendre_nodes_weights(nx);
    auto [nodes_y, weights_y] = gauss_legendre_nodes_weights(ny);
    double hx = (bx - ax) * 0.5, hy = (by - ay) * 0.5;
    double cx = (ax + bx) * 0.5, cy = (ay + by) * 0.5;
    double sum = 0.0;
    for (int i = 0; i < nx; ++i) {
        double x = cx + hx * nodes_x[i];
        double wx = weights_x[i];
        for (int j = 0; j < ny; ++j) {
            double y = cy + hy * nodes_y[j];
            double wy = weights_y[j];
            sum += wx * wy * f(x, y);
        }
    }
    return sum * hx * hy;
}

// ============================================================
// Triple integral over box
// ============================================================

template<typename Func>
double triple_integral_tensor(Func f, double ax, double bx, double ay, double by,
                              double az, double bz, int nx, int ny, int nz) {
    auto [nodes_x, weights_x] = gauss_legendre_nodes_weights(nx);
    auto [nodes_y, weights_y] = gauss_legendre_nodes_weights(ny);
    auto [nodes_z, weights_z] = gauss_legendre_nodes_weights(nz);
    double hx = (bx-ax)*0.5, hy = (by-ay)*0.5, hz = (bz-az)*0.5;
    double cx=(ax+bx)*0.5, cy=(ay+by)*0.5, cz=(az+bz)*0.5;
    double sum = 0.0;
    for (int i=0; i<nx; ++i) {
        double x=cx+hx*nodes_x[i]; double wx=weights_x[i];
        for (int j=0; j<ny; ++j) {
            double y=cy+hy*nodes_y[j]; double wy=weights_y[j];
            for (int k=0; k<nz; ++k) {
                double z=cz+hz*nodes_z[k]; double wz=weights_z[k];
                sum += wx*wy*wz * f(x,y,z);
            }
        }
    }
    return sum * hx*hy*hz;
}

// ============================================================
// Monte Carlo integration (1D, 2D, 3D)
// ============================================================

template<typename Func>
double monte_carlo_1d(Func f, double a, double b, int samples, xorshift64& rng) {
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        double x = a + (b - a) * uniform_double(rng);
        sum += f(x);
    }
    return (b - a) * sum / samples;
}

template<typename Func>
double monte_carlo_2d(Func f, double ax, double bx, double ay, double by, int samples, xorshift64& rng) {
    double area = (bx - ax) * (by - ay);
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        double x = ax + (bx - ax) * uniform_double(rng);
        double y = ay + (by - ay) * uniform_double(rng);
        sum += f(x, y);
    }
    return area * sum / samples;
}

template<typename Func>
double monte_carlo_3d(Func f, double ax, double bx, double ay, double by,
                      double az, double bz, int samples, xorshift64& rng) {
    double volume = (bx - ax) * (by - ay) * (bz - az);
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        double x = ax + (bx - ax) * uniform_double(rng);
        double y = ay + (by - ay) * uniform_double(rng);
        double z = az + (bz - az) * uniform_double(rng);
        sum += f(x, y, z);
    }
    return volume * sum / samples;
}

// ============================================================
// Importance sampling Monte Carlo (using provided PDF)
// ============================================================

template<typename Func, typename PDF, typename SampleFunc>
double importance_sampling_1d(Func f, PDF pdf, SampleFunc sample, double a, double b, int samples, xorshift64& rng) {
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        double x = sample(rng);
        double p = pdf(x);
        if (p > 1e-12) sum += f(x) / p;
    }
    return sum / samples;
}

// ============================================================
// Monte Carlo integration over sphere surface (uniform)
// ============================================================

template<typename Func>
double monte_carlo_sphere_surface(Func f, double radius, int samples, xorshift64& rng) {
    double area = 4.0 * PI_D * radius * radius;
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        vector3<double> dir = random_unit_sphere(rng);
        sum += f(dir * radius);
    }
    return area * sum / samples;
}

// ============================================================
// Monte Carlo integration over hemisphere (cosine‑weighted)
// ============================================================

template<typename Func>
double monte_carlo_hemisphere_cosine(Func f, const vector3<double>& normal, int samples, xorshift64& rng) {
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        vector3<double> dir = random_unit_hemisphere(rng, normal);
        double cos_theta = dot(dir, normal);
        sum += f(dir) * cos_theta;
    }
    return (2.0 * PI_D) * sum / samples;
}

// ============================================================
// Newton‑Cotes formulas (closed)
// ============================================================

template<typename Func>
double newton_cotes_closed(Func f, double a, double b, int n) {
    // Composite Newton‑Cotes: for small n we can use Simpson/Trapezoidal, but here we provide generic.
    // We'll implement composite Boole's rule (n must be multiple of 4) as an example.
    // If n not multiple of 4, adjust.
    int m = (n / 4) * 4;
    if (m < 4) m = 4;
    double h = (b - a) / m;
    double sum = 0.0;
    for (int i = 0; i < m; i += 4) {
        double x0 = a + i * h;
        double x1 = x0 + h, x2 = x0 + 2*h, x3 = x0 + 3*h, x4 = x0 + 4*h;
        sum += (2.0*h/45.0) * (7.0*f(x0) + 32.0*f(x1) + 12.0*f(x2) + 32.0*f(x3) + 7.0*f(x4));
    }
    return sum;
}

// ============================================================
// Cubature over triangle (using symmetric Gauss quadrature)
// ============================================================

template<typename Func>
double triangle_integral_gauss(Func f, const vector3<double>& v0, const vector3<double>& v1,
                               const vector3<double>& v2, int order = 3) {
    double area = triangle_area(v0, v1, v2);
    if (order == 1) {
        vector3<double> centroid = (v0+v1+v2)/3.0;
        return area * f(centroid);
    }
    // order 3: 4 points
    static const double s[4][2] = {{1.0/3.0, 1.0/3.0}, {0.6, 0.2}, {0.2, 0.6}, {0.2, 0.2}};
    static const double w[4] = {-27.0/48.0, 25.0/48.0, 25.0/48.0, 25.0/48.0};
    double sum = 0.0;
    for (int i=0; i<4; ++i) {
        vector3<double> p = v0*s[i][0] + v1*s[i][1] + v2*(1.0-s[i][0]-s[i][1]);
        sum += w[i] * f(p);
    }
    return sum * area;
}

// ============================================================
// Helper: triangle area
// ============================================================

inline double triangle_area(const vector3<double>& a, const vector3<double>& b, const vector3<double>& c) {
    return 0.5 * length(cross(b - a, c - a));
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_QUADRATURE_H