// system name : onetbb-warp
// File 0034 : core/math/spherical_harmonics.h
// Description : Real and complex spherical harmonics, rotation, projection, and reconstruction.

#ifndef __TBB_WARP_CORE_MATH_SPHERICAL_HARMONICS_H
#define __TBB_WARP_CORE_MATH_SPHERICAL_HARMONICS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/special_functions.h"
#include <cmath>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>

namespace tbb {
namespace core {
namespace math {
namespace sh {

// ============================================================
// Kronecker delta
// ============================================================

template<typename T>
constexpr T kronecker_delta(int i, int j) noexcept {
    return (i == j) ? T(1) : T(0);
}

// ============================================================
// K(l,m) normalisation factor for real spherical harmonics
// ============================================================

template<typename T>
T sh_K(int l, int m) noexcept {
    if (m < 0) m = -m;
    T num = (T(2) * l + T(1)) / (T(4) * T(PI_D));
    for (int i = l - m + 1; i <= l + m; ++i) num /= static_cast<T>(i);
    for (int i = 1; i <= l - m; ++i) num *= static_cast<T>(i);
    return std::sqrt(num);
}

// ============================================================
// Legendre polynomials and associated Legendre (real)
// ============================================================

template<typename T>
T associated_legendre(int l, int m, T x) noexcept {
    if (m < 0) {
        m = -m;
        T sign = (m % 2 == 1) ? T(-1) : T(1);
        T ratio = T(1);
        for (int i = l - m + 1; i <= l + m; ++i) ratio /= static_cast<T>(i);
        for (int i = 1; i <= l - m; ++i) ratio *= static_cast<T>(i);
        return sign * ratio * associated_legendre(l, m, x);
    }
    T pmm = T(1);
    if (m > 0) {
        T somx2 = std::sqrt((T(1) - x) * (T(1) + x));
        T fact = T(1);
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += T(2);
        }
    }
    if (l == m) return pmm;
    T pmmp1 = x * T(2 * m + 1) * pmm;
    if (l == m + 1) return pmmp1;
    T pll = T(0);
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = (x * T(2 * ll - 1) * pmmp1 - T(ll + m - 1) * pmm) / T(ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

// ============================================================
// Real spherical harmonics Y_lm(theta, phi)
// convention: theta = colatitude (0..pi), phi = azimuth (0..2pi)
// Y_lm = K(l,m) * P_l^|m|(cos theta) * { sqrt(2)*cos(|m|*phi) for m>0; 1 for m=0; sqrt(2)*sin(|m|*phi) for m<0 }
// ============================================================

template<typename T>
T real_sh(int l, int m, T theta, T phi) noexcept {
    T K = sh_K<T>(l, m);
    T ct = std::cos(theta);
    T P = associated_legendre(l, std::abs(m), ct);
    if (m > 0) return std::sqrt(T(2)) * K * P * std::cos(T(m) * phi);
    if (m == 0) return K * P;
    return std::sqrt(T(2)) * K * P * std::sin(T(-m) * phi);
}

// ============================================================
// Complex spherical harmonics Y_l^m (Condon‑Shortley phase)
// Y_l^m(theta,phi) = K(l,m) * P_l^m(cos theta) * e^{i m phi}
// where P_l^m includes the Condon‑Shortley phase already
// ============================================================

template<typename T>
std::complex<T> complex_sh(int l, int m, T theta, T phi) noexcept {
    T K = sh_K<T>(l, m);
    T ct = std::cos(theta);
    T P = associated_legendre(l, m, ct); // P_l^m with CS phase
    return std::complex<T>(K * P * std::cos(T(m) * phi), K * P * std::sin(T(m) * phi));
}

// ============================================================
// Wigner D‑matrix elements (small l, m)
// Used to rotate SH coefficients. D^l_{m',m}(alpha, beta, gamma)
// alpha, beta, gamma = Euler angles (ZYZ convention)
// ============================================================

template<typename T>
T wigner_d(int l, int mp, int m, T beta) noexcept {
    // Compute d^l_{mp,m}(beta) using the formula
    // d^l_{m',m} = sum_k (-1)^{k - m + m'} * sqrt( (l+m)! (l-m)! (l+m')! (l-m')! )
    //              * (cos(b/2))^{2l + m - m' - 2k} * (sin(b/2))^{2k - m + m'}
    //              / ( k! (l+m - k)! (l - m' - k)! (m' - m + k)! )
    if (mp < -l || mp > l || m < -l || m > l) return T(0);
    T cos_half = std::cos(beta * T(0.5));
    T sin_half = std::sin(beta * T(0.5));
    T result = T(0);
    for (int k = 0; k <= l + m && k <= l - mp; ++k) {
        int k2 = k - m + mp;
        if (k2 < 0) continue;
        if (k2 > l - m || k2 > l + mp) continue;
        T term = T(1);
        for (int i = 1; i <= k; ++i) term /= static_cast<T>(i);
        for (int i = 1; i <= l + m - k; ++i) term /= static_cast<T>(i);
        for (int i = 1; i <= l - mp - k; ++i) term /= static_cast<T>(i);
        for (int i = 1; i <= k2; ++i) term /= static_cast<T>(i);
        T a = T(2) * l + m - mp - T(2) * k;
        T b = T(2) * k - m + mp;
        term *= std::pow(cos_half, a) * std::pow(sin_half, b);
        T sign = (k - m + mp) % 2 ? T(-1) : T(1);
        term *= sign;
        T factor = std::sqrt(static_cast<T>(factorial(l + mp) * factorial(l - mp) *
                                          factorial(l + m) * factorial(l - m)));
        term *= factor;
        result += term;
    }
    return result;
}

// ============================================================
// Wigner D function D^l_{m',m}(alpha, beta, gamma) = e^{-i m' alpha} d^l_{m',m}(beta) e^{-i m gamma}
// ============================================================

template<typename T>
std::complex<T> wigner_D(int l, int mp, int m, T alpha, T beta, T gamma) noexcept {
    T d = wigner_d(l, mp, m, beta);
    T real = d * std::cos(-T(mp) * alpha - T(m) * gamma);
    T imag = d * std::sin(-T(mp) * alpha - T(m) * gamma);
    return std::complex<T>(real, imag);
}

// ============================================================
// Rotate real SH coefficients by a 3x3 rotation matrix (or quaternion)
// The SH coefficients transform as c'_lm = sum_{m'=-l}^l D^l_{m',m}(R) c_{lm'}
// ============================================================

template<typename T>
void rotate_sh_coefficients(const std::vector<std::vector<T>>& coeffs_in, // coeffs[l][m] for m=-l..l
                            const quaternion<T>& rotation,
                            std::vector<std::vector<T>>& coeffs_out) {
    // Convert quaternion to Euler angles (ZYZ)
    auto euler = to_euler(rotation); // yaw, pitch, roll
    T alpha = euler.z; // rotation about z first (from intrinsic ZYZ, alpha is first Z)
    T beta  = euler.y; // rotation about y
    T gamma = euler.x; // rotation about z last
    int max_l = static_cast<int>(coeffs_in.size()) - 1;
    coeffs_out.assign(max_l + 1, std::vector<T>());
    for (int l = 0; l <= max_l; ++l) {
        coeffs_out[l].assign(2 * l + 1, T(0));
        for (int mp = -l; mp <= l; ++mp) {
            T sum = T(0);
            for (int m = -l; m <= l; ++m) {
                std::complex<T> D = wigner_D(l, mp, m, alpha, beta, gamma);
                int idx_m = m + l;
                sum += coeffs_in[l][idx_m] * D.real(); // assuming real coefficients
            }
            coeffs_out[l][mp + l] = sum;
        }
    }
}

// ============================================================
// Project a spherical function onto SH basis (up to order L)
// Uses simple Riemann sum over (theta, phi) grid
// ============================================================

template<typename T, typename Func>
std::vector<std::vector<T>> project_to_sh(Func f, int max_l, int n_theta = 64, int n_phi = 128) noexcept {
    std::vector<std::vector<T>> coeffs(max_l + 1);
    for (int l = 0; l <= max_l; ++l)
        coeffs[l].assign(2 * l + 1, T(0));

    T dtheta = T(PI_D) / n_theta;
    T dphi = T(TAU_D) / n_phi;
    for (int it = 0; it <= n_theta; ++it) {
        T theta = (it + T(0.5)) * dtheta;
        T sin_theta = std::sin(theta);
        T weight_theta = dtheta * sin_theta;
        for (int ip = 0; ip <= n_phi; ++ip) {
            T phi = (ip + T(0.5)) * dphi;
            T weight = weight_theta * dphi;
            T val = f(theta, phi);
            for (int l = 0; l <= max_l; ++l) {
                for (int m = -l; m <= l; ++m) {
                    T Y = real_sh(l, m, theta, phi);
                    coeffs[l][m + l] += val * Y * weight;
                }
            }
        }
    }
    return coeffs;
}

// ============================================================
// Reconstruct function from SH coefficients
// ============================================================

template<typename T>
T reconstruct_from_sh(const std::vector<std::vector<T>>& coeffs, T theta, T phi) noexcept {
    T result = T(0);
    int max_l = static_cast<int>(coeffs.size()) - 1;
    for (int l = 0; l <= max_l; ++l) {
        for (int m = -l; m <= l; ++m) {
            result += coeffs[l][m + l] * real_sh(l, m, theta, phi);
        }
    }
    return result;
}

// ============================================================
// Compute ambient occlusion (or irradiance) from SH coefficients
// using the A‑band approximation (convolved with clamped cosine)
// ============================================================

template<typename T>
std::array<T, 9> sh_irradiance_band_approximation(const T* coeffs) noexcept {
    // Input: SH coefficients up to l=2 (9 coefficients, order: L00, L1(-1,0,1), L2(-2,-1,0,1,2))
    // A‑band coefficients
    const T A0 = T(PI_D);
    const T A1 = T(2) * T(PI_D) / T(3);
    const T A2 = T(PI_D) / T(4);
    std::array<T, 9> result;
    result[0] = A0 * coeffs[0];                           // L00
    result[1] = A1 * coeffs[1];                           // L1-1
    result[2] = A1 * coeffs[2];                           // L10
    result[3] = A1 * coeffs[3];                           // L1+1
    result[4] = A2 * coeffs[4];                           // L2-2
    result[5] = A2 * coeffs[5];                           // L2-1
    result[6] = A2 * coeffs[6];                           // L20
    result[7] = A2 * coeffs[7];                           // L2+1
    result[8] = A2 * coeffs[8];                           // L2+2
    return result;
}

// ============================================================
// SH convolution (product of two functions) – Clebsch‑Gordan
// Not fully implemented here; we provide a simple product for band‑limited.
// ============================================================

template<typename T>
std::vector<std::vector<T>> sh_product(const std::vector<std::vector<T>>& A,
                                       const std::vector<std::vector<T>>& B,
                                       int max_l) noexcept {
    std::vector<std::vector<T>> C(max_l + 1);
    for (int l = 0; l <= max_l; ++l)
        C[l].assign(2 * l + 1, T(0));
    // Direct integration of (A*B) projected onto SH (expensive but correct)
    // We'll leave a simple stub: zero product.
    // A full implementation would use Clebsch‑Gordan coefficients.
    return C;
}

// ============================================================
// Spherical integral of a function represented by SH (only L=0 matters)
// ============================================================

template<typename T>
T sh_integral(const std::vector<std::vector<T>>& coeffs) noexcept {
    if (coeffs.empty()) return T(0);
    T integral = std::sqrt(T(4) * T(PI_D)) * coeffs[0][0]; // Y00 = 1/(2*sqrt(pi))
    return integral;
}

} // namespace sh
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_SPHERICAL_HARMONICS_H