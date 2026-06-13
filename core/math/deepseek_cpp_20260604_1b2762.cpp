// system name : onetbb-warp
// File 0020 : core/math/special_functions.h
// Description : Special mathematical functions: Gamma, Beta, Bessel, error, Legendre.

#ifndef __TBB_WARP_CORE_MATH_SPECIAL_FUNCTIONS_H
#define __TBB_WARP_CORE_MATH_SPECIAL_FUNCTIONS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <array>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Sinc function
// ============================================================

inline double sinc(double x) noexcept {
    if (std::abs(x) < 1e-12) return 1.0;
    return std::sin(PI_D * x) / (PI_D * x);
}

inline float sincf(float x) noexcept {
    if (std::abs(x) < 1e-6f) return 1.0f;
    return std::sin(PI_F * x) / (PI_F * x);
}

// ============================================================
// Factorial and double factorial
// ============================================================

constexpr std::uint64_t factorial(int n) noexcept {
    std::uint64_t r = 1;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

constexpr std::uint64_t double_factorial(int n) noexcept {
    if (n <= 1) return 1;
    std::uint64_t r = 1;
    for (int i = n; i > 1; i -= 2) r *= i;
    return r;
}

// ============================================================
// Gamma function – Lanczos approximation
// ============================================================

constexpr double LANCMOS_G = 7.0;
constexpr std::array<double, 9> LANCZOS_COEFF = {{
    0.99999999999980993,
    676.5203681218851,
   -1259.1392167224028,
    771.32342877765313,
   -176.61502916214059,
    12.507343278686905,
   -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
}};

inline double lanczos_gamma(double z) noexcept {
    if (z < 0.5) {
        return PI_D / (std::sin(PI_D * z) * lanczos_gamma(1.0 - z));
    }
    z -= 1.0;
    double x = LANCZOS_COEFF[0];
    for (std::size_t i = 1; i < LANCZOS_COEFF.size(); ++i)
        x += LANCZOS_COEFF[i] / (z + static_cast<double>(i));
    double t = z + LANCMOS_G + 0.5;
    return std::sqrt(TAU_D) * std::pow(t, z + 0.5) * std::exp(-t) * x;
}

inline float gamma_f(float x) noexcept { return static_cast<float>(lanczos_gamma(static_cast<double>(x))); }

// ============================================================
// Log‑Gamma (for numerical stability)
// ============================================================

inline double log_gamma(double x) noexcept {
    double r = std::log(lanczos_gamma(x));
    return r;
}

// ============================================================
// Digamma function (psi) – asymptotic expansion
// ============================================================

inline double digamma(double x) noexcept {
    if (x < 0.0) return digamma(1.0 - x) - PI_D / std::tan(PI_D * x);
    double result = 0.0;
    while (x < 10.0) { result -= 1.0 / x; x += 1.0; }
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    result += std::log(x) - 0.5 * inv_x - inv_x2 * (1.0/12.0 - inv_x2*(1.0/120.0 - inv_x2*(1.0/252.0)));
    return result;
}

// ============================================================
// Beta function
// ============================================================

inline double beta_function(double a, double b) noexcept {
    return std::exp(log_gamma(a) + log_gamma(b) - log_gamma(a + b));
}

inline float beta_f(float a, float b) noexcept {
    return static_cast<float>(beta_function(a, b));
}

// ============================================================
// Binomial coefficient (real arguments using Gamma)
// ============================================================

inline double binomial_general(double n, double k) noexcept {
    if (k < 0.0 || k > n) return 0.0;
    if (k == 0.0 || k == n) return 1.0;
    return std::exp(log_gamma(n + 1.0) - log_gamma(k + 1.0) - log_gamma(n - k + 1.0));
}

// ============================================================
// Incomplete Gamma (lower) – series expansion
// ============================================================

inline double incomplete_gamma_lower(double a, double x) noexcept {
    if (x < 0.0) return 0.0;
    if (x == 0.0) return 0.0;
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int i = 0; i < 200; ++i) {
        ++ap;
        del *= x / ap;
        sum += del;
        if (std::abs(del) < std::abs(sum) * 1e-15) break;
    }
    return sum * std::exp(-x + a * std::log(x) - log_gamma(a + 1.0)) * a;
}

// ============================================================
// Incomplete Gamma (upper) – via lower
// ============================================================

inline double incomplete_gamma_upper(double a, double x) noexcept {
    return lanczos_gamma(a) - incomplete_gamma_lower(a, x);
}

// ============================================================
// Regularized incomplete Gamma (lower) P(a,x)
// ============================================================

inline double gamma_p(double a, double x) noexcept {
    if (x < 0.0 || a <= 0.0) return 0.0;
    if (x < a + 1.0) return incomplete_gamma_lower(a, x) / lanczos_gamma(a);
    // Use continued fraction for large x
    double b = x + 1.0 - a;
    double c = 1.0 / 1e-30;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= 200; ++i) {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = b + an / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-15) break;
    }
    return 1.0 - std::exp(-x + a * std::log(x) - log_gamma(a)) * h;
}

// ============================================================
// Error function (complementary) – already in C++17, but provide extended
// ============================================================

inline float erfc_fast(float x) noexcept {
    // Approximation from Williams
    float abs_x = std::abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * abs_x);
    float p = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
    float y = t * (p + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    float result = y * std::exp(-abs_x * abs_x);
    return (x >= 0.0f) ? result : 2.0f - result;
}

// ============================================================
// Inverse error function (approximation)
// ============================================================

inline float inverse_erf(float x) noexcept {
    if (std::abs(x) >= 1.0f) return (x >= 1.0f) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    float a = 0.140012f;
    float ln1mx2 = std::log(1.0f - x * x);
    float term1 = 2.0f / (PI_F * a) + ln1mx2 / 2.0f;
    float term2 = ln1mx2 / a;
    float sqrt_inner = std::sqrt(term1 * term1 - term2);
    float result = std::sqrt(sqrt_inner - term1);
    return (x >= 0.0f) ? result : -result;
}

// ============================================================
// Bessel functions of the first kind – J0, J1, Jn (integer order)
// ============================================================

inline double bessel_j0(double x) noexcept {
    if (std::abs(x) < 8.0) {
        double y = x * x;
        double ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        double ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        return ans1 / ans2;
    } else {
        double z = 8.0 / x;
        double y = z * z;
        double xx = x - 0.785398164;
        double ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        double ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        return std::sqrt(0.636619772 / x) * (std::cos(xx) * ans1 - z * std::sin(xx) * ans2);
    }
}

inline double bessel_j1(double x) noexcept {
    if (std::abs(x) < 8.0) {
        double y = x * x;
        double ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        double ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        return ans1 / ans2;
    } else {
        double z = 8.0 / x;
        double y = z * z;
        double xx = x - 2.356194491;
        double ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        double ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        return std::sqrt(0.636619772 / x) * (std::cos(xx) * ans1 - z * std::sin(xx) * ans2);
    }
}

inline double bessel_jn(int n, double x) noexcept {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    if (x == 0.0) return 0.0;
    double tox = 2.0 / x;
    double bjm = bessel_j0(x);
    double bj = bessel_j1(x);
    for (int j = 1; j < n; ++j) {
        double bjp = j * tox * bj - bjm;
        bjm = bj;
        bj = bjp;
    }
    return bj;
}

// ============================================================
// Bessel functions of the second kind – Y0, Y1
// ============================================================

inline double bessel_y0(double x) noexcept {
    if (x < 8.0) {
        double y = x * x;
        double ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
        double ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0))));
        return ans1 / ans2 + 0.636619772 * bessel_j0(x) * std::log(x);
    } else {
        double z = 8.0 / x;
        double y = z * z;
        double xx = x - 0.785398164;
        double ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        double ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        return std::sqrt(0.636619772 / x) * (std::sin(xx) * ans1 + z * std::cos(xx) * ans2);
    }
}

inline double bessel_y1(double x) noexcept {
    if (x < 8.0) {
        double y = x * x;
        double ans1 = x * (-0.4900604943e13 + y * (0.1275274390e13 + y * (-0.5153438139e11 + y * (0.7349264551e9 + y * (-0.4237922726e7 + y * 0.8511937935e4)))));
        double ans2 = 0.2499580570e14 + y * (0.4244419664e12 + y * (0.3733650367e10 + y * (0.2245904002e8 + y * (0.1020426050e6 + y * (0.3549632885e3 + y * 1.0)))));
        return ans1 / ans2 + 0.636619772 * (bessel_j1(x) * std::log(x) - 1.0 / x);
    } else {
        double z = 8.0 / x;
        double y = z * z;
        double xx = x - 2.356194491;
        double ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        double ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        return std::sqrt(0.636619772 / x) * (std::sin(xx) * ans1 + z * std::cos(xx) * ans2);
    }
}

// ============================================================
// Modified Bessel functions – I0, I1 (used in Kaiser window)
// ============================================================

inline double bessel_i0(double x) noexcept {
    double ax = std::abs(x);
    if (ax < 3.75) {
        double y = (x / 3.75) * (x / 3.75);
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    } else {
        double y = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
    }
}

inline double bessel_i1(double x) noexcept {
    double ax = std::abs(x);
    double result;
    if (ax < 3.75) {
        double y = (x / 3.75) * (x / 3.75);
        result = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
    } else {
        double y = 3.75 / ax;
        result = (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + y * (-0.03988024 + y * (-0.00362018 + y * (0.00163801 + y * (-0.01031555 + y * (0.02282967 + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059))))))));
    }
    return (x < 0.0) ? -result : result;
}

// ============================================================
// Modified Bessel K0, K1
// ============================================================

inline double bessel_k0(double x) noexcept {
    if (x <= 2.0) {
        double y = x * x / 4.0;
        return (-std::log(x/2.0) * bessel_i0(x)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))))));
    } else {
        double y = 2.0 / x;
        return (std::exp(-x) / std::sqrt(x)) * (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))));
    }
}

inline double bessel_k1(double x) noexcept {
    if (x <= 2.0) {
        double y = x * x / 4.0;
        return (std::log(x/2.0) * bessel_i1(x)) + (1.0 / x) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 - y * 0.00004686))))));
    } else {
        double y = 2.0 / x;
        return (std::exp(-x) / std::sqrt(x)) * (1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (0.01504268 + y * (-0.00780353 + y * (0.00325614 - y * 0.00068245))))));
    }
}

// ============================================================
// Legendre polynomials P_n(x) (Bonnet recurrence)
// ============================================================

inline double legendre_p(int n, double x) noexcept {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double p0 = 1.0, p1 = x, p2;
    for (int i = 2; i <= n; ++i) {
        p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / i;
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

// ============================================================
// Associated Legendre polynomials P_l^m(x) (including Condon‑Shortley phase)
// ============================================================

inline double legendre_plm(int l, int m, double x) noexcept {
    if (m < 0) {
        m = -m;
        double sign = (m % 2 == 1) ? -1.0 : 1.0;
        return sign * gamma_f(l - m + 1) / gamma_f(l + m + 1) * legendre_plm(l, m, x);
    }
    double pmm = 1.0;
    if (m > 0) {
        double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if (l == m) return pmm;
    double pmmp1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) return pmmp1;
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

// ============================================================
// Spherical harmonics Y_l^m (real form) – uses associated Legendre
// ============================================================

inline double spherical_harmonic_real(int l, int m, double theta, double phi) {
    if (m < 0) return std::sqrt(2.0) * spherical_harmonic_K(l, m) * legendre_plm(l, -m, std::cos(theta)) * std::sin(-m * phi);
    if (m == 0) return spherical_harmonic_K(l, 0) * legendre_p(l, std::cos(theta));
    return std::sqrt(2.0) * spherical_harmonic_K(l, m) * legendre_plm(l, m, std::cos(theta)) * std::cos(m * phi);
}

inline double spherical_harmonic_K(int l, int m) {
    double temp = (2.0 * l + 1.0) / (4.0 * PI_D);
    for (int i = l - m + 1; i <= l + m; ++i) temp /= i;
    for (int i = 1; i <= l - m; ++i) temp *= i;
    return std::sqrt(temp);
}

// ============================================================
// Fresnel integrals (approximation)
// ============================================================

inline std::pair<double, double> fresnel(double x) noexcept {
    double ax = std::abs(x);
    if (ax < 1.0) {
        double sum_s = 0.0, sum_c = 0.0;
        double term_s = x, term_c = 1.0;
        for (int n = 0; n < 20; ++n) {
            sum_c += term_c * std::pow(x, 4*n) / factorial(4*n);
            sum_s += term_s * std::pow(x, 4*n+1) / factorial(4*n+1);
        }
        return {sum_c, sum_s};
    }
    double x2 = x * x;
    double C = 0.5 + std::sin(PI_D/2.0 * x2) / (PI_D * x) - std::cos(PI_D/2.0 * x2) / (PI_D * PI_D * x2 * x);
    double S = 0.5 - std::cos(PI_D/2.0 * x2) / (PI_D * x) - std::sin(PI_D/2.0 * x2) / (PI_D * PI_D * x2 * x);
    return {C, S};
}

// ============================================================
// Dawson integral (F(x) = exp(-x^2) * integral_0^x exp(t^2) dt)
// ============================================================

inline double dawson(double x) noexcept {
    double ax = std::abs(x);
    if (ax < 0.2) {
        return x * (1.0 - 2.0/3.0 * x*x * (1.0 - 2.0/5.0 * x*x));
    }
    if (ax < 5.0) {
        double n = 0;
        double d = x;
        for (int i = 1; i < 30; ++i) {
            double a = i;
            double term = a * x * x / ((2.0 * a - 1.0) * (2.0 * a + 1.0) - term);
            double nd = n + d;
            n = d;
            d = nd;
        }
        return x / (1.0 - x*x * n / d);
    }
    return 0.5 / x + 0.25 / (x*x*x);
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_SPECIAL_FUNCTIONS_H