//File 0065 : core/math/complex_math.h
//Extended complex number operations: arithmetic, magnitude, phase, exp, log, pow, sqrt, roots of unity, and batch‑processing with SIMD‑friendly loops.
#ifndef CORE_MATH_COMPLEX_MATH_H
#define CORE_MATH_COMPLEX_MATH_H

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace SimulationMath {
namespace complex_math {

using Complex = std::complex<float>;
using ComplexD = std::complex<double>;

// -----------------------------------------------------------------------------
// 1. Magnitude and phase
// -----------------------------------------------------------------------------
inline float magnitude(const Complex& z) noexcept { return std::abs(z); }
inline float phase(const Complex& z) noexcept { return std::atan2(z.imag(), z.real()); }
inline Complex polar(float r, float theta) noexcept { return Complex(r * std::cos(theta), r * std::sin(theta)); }

// -----------------------------------------------------------------------------
// 2. Conjugate
// -----------------------------------------------------------------------------
inline Complex conjugate(const Complex& z) noexcept { return std::conj(z); }

// -----------------------------------------------------------------------------
// 3. Complex square root (principal branch)
// -----------------------------------------------------------------------------
inline Complex sqrt_complex(const Complex& z) noexcept {
    float r = std::abs(z);
    float a = std::sqrt((r + z.real()) * 0.5f);
    float b = (z.imag() >= 0.0f) ? std::sqrt((r - z.real()) * 0.5f) : -std::sqrt((r - z.real()) * 0.5f);
    return Complex(a, b);
}

// -----------------------------------------------------------------------------
// 4. Complex exponential
// -----------------------------------------------------------------------------
inline Complex exp_complex(const Complex& z) noexcept {
    float e = std::exp(z.real());
    return Complex(e * std::cos(z.imag()), e * std::sin(z.imag()));
}

// -----------------------------------------------------------------------------
// 5. Complex logarithm (principal value)
// -----------------------------------------------------------------------------
inline Complex log_complex(const Complex& z) noexcept {
    return Complex(std::log(std::abs(z)), std::atan2(z.imag(), z.real()));
}

// -----------------------------------------------------------------------------
// 6. Complex power (z^w)
// -----------------------------------------------------------------------------
inline Complex pow_complex(const Complex& z, const Complex& w) noexcept {
    return exp_complex(w * log_complex(z));
}
inline Complex pow_complex(const Complex& z, float real_exp) noexcept {
    return pow_complex(z, Complex(real_exp, 0.0f));
}

// -----------------------------------------------------------------------------
// 7. Roots of unity (n‑th roots)
// -----------------------------------------------------------------------------
inline std::vector<Complex> roots_of_unity(size_t n) noexcept {
    std::vector<Complex> roots(n);
    float angle = 2.0f * 3.14159265358979f / n;
    for (size_t k = 0; k < n; ++k)
        roots[k] = Complex(std::cos(angle * k), std::sin(angle * k));
    return roots;
}

// -----------------------------------------------------------------------------
// 8. Batch operations on vectors of complex numbers
// -----------------------------------------------------------------------------
inline void batch_add(std::vector<Complex>& result, const std::vector<Complex>& a, const std::vector<Complex>& b) noexcept {
    size_t n = std::min(a.size(), b.size());
    result.resize(n);
    for (size_t i = 0; i < n; ++i) result[i] = a[i] + b[i];
}
inline void batch_mul(std::vector<Complex>& result, const std::vector<Complex>& a, const std::vector<Complex>& b) noexcept {
    size_t n = std::min(a.size(), b.size());
    result.resize(n);
    for (size_t i = 0; i < n; ++i) result[i] = a[i] * b[i];
}
inline void batch_scale(std::vector<Complex>& data, float factor) noexcept {
    for (auto& z : data) z *= factor;
}
inline void batch_conjugate(std::vector<Complex>& data) noexcept {
    for (auto& z : data) z = std::conj(z);
}
inline void batch_exp(std::vector<Complex>& data) noexcept {
    for (auto& z : data) z = exp_complex(z);
}
inline void batch_log(std::vector<Complex>& data) noexcept {
    for (auto& z : data) z = log_complex(z);
}
inline void batch_magnitude(std::vector<Complex>& data, std::vector<float>& mags) noexcept {
    mags.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) mags[i] = std::abs(data[i]);
}

// -----------------------------------------------------------------------------
// 9. Cartesian ↔ polar batch conversion
// -----------------------------------------------------------------------------
inline void to_polar(const std::vector<Complex>& cart, std::vector<float>& r, std::vector<float>& theta) noexcept {
    r.resize(cart.size());
    theta.resize(cart.size());
    for (size_t i = 0; i < cart.size(); ++i) {
        r[i] = std::abs(cart[i]);
        theta[i] = std::atan2(cart[i].imag(), cart[i].real());
    }
}
inline void from_polar(const std::vector<float>& r, const std::vector<float>& theta, std::vector<Complex>& cart) noexcept {
    size_t n = std::min(r.size(), theta.size());
    cart.resize(n);
    for (size_t i = 0; i < n; ++i)
        cart[i] = Complex(r[i] * std::cos(theta[i]), r[i] * std::sin(theta[i]));
}

// -----------------------------------------------------------------------------
// 10. Complex dot product (inner product) of two vectors
// -----------------------------------------------------------------------------
inline Complex dot_product(const std::vector<Complex>& a, const std::vector<Complex>& b) noexcept {
    Complex sum(0.0f, 0.0f);
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) sum += std::conj(a[i]) * b[i];
    return sum;
}

} // namespace complex_math
} // namespace SimulationMath

#endif // CORE_MATH_COMPLEX_MATH_H