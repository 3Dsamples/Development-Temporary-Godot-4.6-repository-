// system name : onetbb-warp
// File 0030 : core/math/polynomial.h
// Description : Polynomial arithmetic, evaluation, root finding, special polynomials, fitting.

#ifndef __TBB_WARP_CORE_MATH_POLYNOMIAL_H
#define __TBB_WARP_CORE_MATH_POLYNOMIAL_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/complex.h"
#include "core/math/vector2.h"
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Polynomial class (real coefficients)
// ============================================================

template<typename T>
class polynomial {
public:
    std::vector<T> coeffs;  // coeffs[i] for x^i

    // ---- Constructors ----
    polynomial() = default;
    explicit polynomial(const std::vector<T>& c) : coeffs(c) { trim(); }
    explicit polynomial(std::vector<T>&& c) noexcept : coeffs(std::move(c)) { trim(); }
    explicit polynomial(std::size_t degree) : coeffs(degree+1, T(0)) { coeffs.back() = T(1); }
    polynomial(std::size_t degree, const T& val) : coeffs(degree+1, T(0)) { coeffs[degree] = val; }

    // ---- Access ----
    std::size_t degree() const noexcept { return coeffs.empty() ? 0 : coeffs.size() - 1; }
    const T& operator[](std::size_t i) const noexcept { return coeffs[i]; }
    T& operator[](std::size_t i) noexcept { return coeffs[i]; }

    // ---- Evaluation ----
    T evaluate(T x) const noexcept {
        if (coeffs.empty()) return T(0);
        T result = coeffs.back();
        for (std::size_t i = coeffs.size() - 1; i-- > 0;)
            result = result * x + coeffs[i];
        return result;
    }

    complex<T> evaluate(const complex<T>& x) const noexcept {
        if (coeffs.empty()) return complex<T>(T(0));
        complex<T> result(coeffs.back());
        for (std::size_t i = coeffs.size() - 1; i-- > 0;)
            result = result * x + coeffs[i];
        return result;
    }

    // ---- Derivative ----
    polynomial derivative() const noexcept {
        if (coeffs.size() <= 1) return polynomial();
        std::vector<T> d(coeffs.size() - 1);
        for (std::size_t i = 1; i < coeffs.size(); ++i)
            d[i-1] = coeffs[i] * static_cast<T>(i);
        return polynomial(std::move(d));
    }

    // ---- Arithmetic ----
    polynomial operator+(const polynomial& o) const {
        std::size_t max_n = std::max(coeffs.size(), o.coeffs.size());
        std::vector<T> r(max_n, T(0));
        for (std::size_t i = 0; i < coeffs.size(); ++i) r[i] += coeffs[i];
        for (std::size_t i = 0; i < o.coeffs.size(); ++i) r[i] += o.coeffs[i];
        return polynomial(std::move(r));
    }
    polynomial operator-(const polynomial& o) const {
        std::size_t max_n = std::max(coeffs.size(), o.coeffs.size());
        std::vector<T> r(max_n, T(0));
        for (std::size_t i = 0; i < coeffs.size(); ++i) r[i] += coeffs[i];
        for (std::size_t i = 0; i < o.coeffs.size(); ++i) r[i] -= o.coeffs[i];
        return polynomial(std::move(r));
    }
    polynomial operator*(const polynomial& o) const {
        if (coeffs.empty() || o.coeffs.empty()) return polynomial();
        std::size_t n = coeffs.size() + o.coeffs.size() - 1;
        std::vector<T> r(n, T(0));
        for (std::size_t i = 0; i < coeffs.size(); ++i)
            for (std::size_t j = 0; j < o.coeffs.size(); ++j)
                r[i+j] += coeffs[i] * o.coeffs[j];
        return polynomial(std::move(r));
    }
    polynomial operator*(T s) const {
        std::vector<T> r(coeffs);
        for (auto& c : r) c *= s;
        return polynomial(std::move(r));
    }
    polynomial operator/(T s) const {
        std::vector<T> r(coeffs);
        T inv = T(1) / s;
        for (auto& c : r) c *= inv;
        return polynomial(std::move(r));
    }

    std::pair<polynomial, polynomial> divide(const polynomial& divisor) const {
        if (divisor.coeffs.empty()) throw std::runtime_error("division by zero polynomial");
        std::size_t m = coeffs.size();
        std::size_t n = divisor.coeffs.size();
        if (m < n) return {polynomial(), *this};
        std::size_t k = m - n + 1;
        std::vector<T> q(k, T(0));
        std::vector<T> r = coeffs;
        T leading = divisor.coeffs.back();
        for (std::size_t i = 0; i < k; ++i) {
            std::size_t idx = k - 1 - i;
            T factor = r[m - 1 - i] / leading;
            q[idx] = factor;
            for (std::size_t j = 0; j < n; ++j) {
                r[m - 1 - i - j] -= factor * divisor.coeffs[n - 1 - j];
            }
        }
        r.resize(n - 1);
        return {polynomial(std::move(q)), polynomial(std::move(r))};
    }

    polynomial integral(T constant = T(0)) const {
        if (coeffs.empty()) return polynomial(std::vector<T>{constant});
        std::vector<T> integ(coeffs.size() + 1, T(0));
        integ[0] = constant;
        for (std::size_t i = 0; i < coeffs.size(); ++i)
            integ[i+1] = coeffs[i] / static_cast<T>(i + 1);
        return polynomial(std::move(integ));
    }

private:
    void trim() {
        while (!coeffs.empty() && coeffs.back() == T(0))
            coeffs.pop_back();
    }
};

// ============================================================
// Polynomial utilities
// ============================================================

template<typename T>
polynomial<T> operator*(T s, const polynomial<T>& p) { return p * s; }

// ============================================================
// Root finding: Durand‑Kerner (simultaneous complex iteration)
// ============================================================

template<typename T>
std::vector<complex<T>> durand_kerner_roots(const polynomial<T>& p, int max_iter = 200, T tol = T(1e-10)) {
    std::size_t n = p.degree();
    if (n < 1) return {};
    if (n == 1) return {complex<T>(-p[0] / p[1])};

    std::vector<complex<T>> roots(n);
    complex<T> one(1);
    T radius = T(1) + T(1) / std::abs(p[p.degree()]);
    for (std::size_t i = 0; i < n; ++i) {
        T angle = T(TAU_D) * i / n + T(0.4);
        roots[i] = from_polar(radius, angle);
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        T max_diff = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            complex<T> num = p.evaluate(roots[i]);
            complex<T> den(1, 0);
            for (std::size_t j = 0; j < n; ++j) {
                if (i != j) den = den * (roots[i] - roots[j]);
            }
            if (norm(den) < T(1e-30)) continue;
            complex<T> delta = num / den;
            roots[i] = roots[i] - delta;
            max_diff = std::max(max_diff, norm(delta));
        }
        if (max_diff < tol) break;
    }
    return roots;
}

// ============================================================
// Root finding: Bairstow's method (extracts quadratic factors)
// ============================================================

template<typename T>
std::pair<complex<T>, complex<T>> bairstow_quadratic(const polynomial<T>& p, T u0, T v0,
                                                     int max_iter = 100, T tol = T(1e-12)) {
    std::size_t n = p.degree();
    if (n < 2) return {complex<T>(-p[0]/p[1]), complex<T>(0)};
    std::vector<T> b(n + 1), c(n + 1);
    T u = u0, v = v0;
    for (int iter = 0; iter < max_iter; ++iter) {
        b[n] = p[n]; b[n-1] = p[n-1] + u * b[n];
        c[n] = T(0); c[n-1] = b[n];
        for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
            b[i] = p[i] + u * b[i+1] + v * b[i+2];
            c[i] = b[i+1] + u * c[i+1] + v * c[i+2];
        }
        T det = c[2]*c[2] - c[3]*c[1];
        if (std::abs(det) < T(1e-12)) break;
        T du = (-b[1]*c[2] + b[0]*c[3]) / det;
        T dv = (-b[0]*c[2] + b[1]*c[1]) / det;
        u += du; v += dv;
        if (std::abs(du) + std::abs(dv) < tol) break;
    }
    T disc = u*u - T(4)*v;
    complex<T> r1, r2;
    if (disc >= T(0)) {
        T sq = std::sqrt(disc);
        r1 = complex<T>((-u + sq) * T(0.5));
        r2 = complex<T>((-u - sq) * T(0.5));
    } else {
        T sq = std::sqrt(-disc);
        r1 = complex<T>(-u * T(0.5), sq * T(0.5));
        r2 = complex<T>(-u * T(0.5), -sq * T(0.5));
    }
    return {r1, r2};
}

// ============================================================
// Newton's method (real roots) with deflation
// ============================================================

template<typename T>
std::vector<T> newton_real_roots(const polynomial<T>& p, int max_iter = 100, T tol = T(1e-12)) {
    polynomial<T> poly = p;
    std::vector<T> roots;
    while (poly.degree() >= 1) {
        if (poly.degree() == 1) {
            roots.push_back(-poly[0] / poly[1]);
            break;
        }
        polynomial<T> deriv = poly.derivative();
        T x = T(0);
        bool found = false;
        for (int trial = 0; trial < 20 && !found; ++trial) {
            x = T(trial - 10) * T(0.5);
            for (int i = 0; i < max_iter; ++i) {
                T fx = poly.evaluate(x);
                T dfx = deriv.evaluate(x);
                if (std::abs(dfx) < T(1e-30)) break;
                T dx = fx / dfx;
                x -= dx;
                if (std::abs(dx) < tol) {
                    found = true;
                    break;
                }
            }
        }
        if (!found) break;
        roots.push_back(x);
        // deflate
        polynomial<T> divisor(std::vector<T>{-x, T(1)});
        auto qr = poly.divide(divisor);
        poly = qr.first;
    }
    return roots;
}

// ============================================================
// Laguerre's method (global convergence for complex polynomials)
// ============================================================

template<typename T>
complex<T> laguerre_root(const polynomial<T>& p, complex<T> z0, int max_iter = 100, T tol = T(1e-14)) {
    std::size_t n = p.degree();
    if (n < 1) return z0;
    complex<T> z = z0;
    polynomial<T> deriv = p.derivative();
    polynomial<T> deriv2 = deriv.derivative();
    for (int iter = 0; iter < max_iter; ++iter) {
        complex<T> fz = p.evaluate(z);
        complex<T> f1z = deriv.evaluate(z);
        complex<T> f2z = deriv2.evaluate(z);
        if (norm(fz) < tol) break;
        complex<T> G = f1z / fz;
        complex<T> H = G*G - f2z / fz;
        T nT = static_cast<T>(n);
        complex<T> den1 = G + sqrt((nT - T(1)) * (nT * H - G*G));
        complex<T> den2 = G - sqrt((nT - T(1)) * (nT * H - G*G));
        complex<T> den = (norm(den1) > norm(den2)) ? den1 : den2;
        complex<T> a = nT / den;
        z = z - a;
        if (norm(a) < tol) break;
    }
    return z;
}

// ============================================================
// Special polynomials
// ============================================================

template<typename T>
polynomial<T> legendre_polynomial(int n) {
    if (n == 0) return polynomial<T>(std::vector<T>{T(1)});
    if (n == 1) return polynomial<T>(std::vector<T>{T(0), T(1)});
    polynomial<T> p0({T(1)});
    polynomial<T> p1({T(0), T(1)});
    for (int i = 2; i <= n; ++i) {
        T a = T(2*i - 1) / T(i);
        T b = T(i - 1) / T(i);
        polynomial<T> term1 = p1 * a;
        term1 = polynomial<T>(std::vector<T>{T(0)}).operator+(term1); // shift degree? Actually shift right.
        // Correct recursion: P_n = ((2n-1)/n) x P_{n-1} - ((n-1)/n) P_{n-2}
        // To multiply by x: prepend a zero coefficient.
        std::vector<T> shifted_p1(p1.coeffs.size() + 1, T(0));
        for (std::size_t i = 0; i < p1.coeffs.size(); ++i) shifted_p1[i+1] = p1.coeffs[i] * a;
        polynomial<T> shifted(shifted_p1);
        polynomial<T> reduced = p0 * b;
        p0 = p1;
        p1 = shifted - reduced;
    }
    return p1;
}

template<typename T>
polynomial<T> chebyshev_polynomial(int n) {
    if (n == 0) return polynomial<T>(std::vector<T>{T(1)});
    if (n == 1) return polynomial<T>(std::vector<T>{T(0), T(1)});
    polynomial<T> p0({T(1)});
    polynomial<T> p1({T(0), T(1)});
    for (int i = 2; i <= n; ++i) {
        std::vector<T> shifted(p1.coeffs.size() + 1, T(0));
        for (std::size_t j = 0; j < p1.coeffs.size(); ++j) shifted[j+1] = p1.coeffs[j] * T(2);
        polynomial<T> two_x_p1(shifted);
        p0 = two_x_p1 - p0;
        std::swap(p0, p1);
    }
    return p1;
}

template<typename T>
polynomial<T> hermite_polynomial(int n) {
    if (n == 0) return polynomial<T>(std::vector<T>{T(1)});
    if (n == 1) return polynomial<T>(std::vector<T>{T(0), T(2)});
    polynomial<T> p0({T(1)});
    polynomial<T> p1({T(0), T(2)});
    for (int i = 2; i <= n; ++i) {
        std::vector<T> shifted(p1.coeffs.size() + 1, T(0));
        for (std::size_t j = 0; j < p1.coeffs.size(); ++j) shifted[j+1] = p1.coeffs[j] * T(2);
        polynomial<T> two_x_p1(shifted);
        polynomial<T> reduced = p0 * T(2 * (i - 1));
        p0 = p1;
        p1 = two_x_p1 - reduced;
    }
    return p1;
}

template<typename T>
polynomial<T> laguerre_polynomial(int n) {
    if (n == 0) return polynomial<T>(std::vector<T>{T(1)});
    if (n == 1) return polynomial<T>(std::vector<T>{T(1), T(-1)});
    polynomial<T> p0({T(1)});
    polynomial<T> p1({T(1), T(-1)});
    for (int i = 2; i <= n; ++i) {
        T a = T(2*i - 1) / T(i);
        T b = T(i - 1) / T(i);
        std::vector<T> shifted_p1(p1.coeffs.size() + 1, T(0));
        for (std::size_t j = 0; j < p1.coeffs.size(); ++j) shifted_p1[j+1] = p1.coeffs[j];
        polynomial<T> shifted(shifted_p1);
        polynomial<T> term1 = p1 * a - shifted;
        polynomial<T> term2 = p0 * b;
        p0 = p1;
        p1 = term1 - term2;
    }
    return p1;
}

// ============================================================
// Polynomial least‑squares fitting
// ============================================================

template<typename T>
polynomial<T> least_squares_fit(const std::vector<T>& x, const std::vector<T>& y, int degree) {
    std::size_t n = x.size();
    if (n != y.size() || n == 0 || degree < 0) return polynomial<T>();
    // Build normal equations A^T A c = A^T y, where A_{i,j} = x_i^j
    std::size_t m = degree + 1;
    std::vector<T> ATA(m * m, T(0));
    std::vector<T> ATy(m, T(0));
    std::vector<T> pow_cache(n, T(1));
    for (std::size_t j = 0; j < m; ++j) {
        if (j > 0) for (std::size_t i = 0; i < n; ++i) pow_cache[i] *= x[i];
        for (std::size_t i = 0; i < n; ++i) ATy[j] += pow_cache[i] * y[i];
        for (std::size_t k = 0; k < m; ++k) {
            T sum = T(0);
            if (j == 0 && k == 0) sum = T(n);
            else if (j == 0) { for (std::size_t i = 0; i < n; ++i) sum += std::pow(x[i], T(k)); }
            else if (k == 0) { for (std::size_t i = 0; i < n; ++i) sum += std::pow(x[i], T(j)); }
            else { for (std::size_t i = 0; i < n; ++i) sum += std::pow(x[i], T(j+k)); }
            ATA[j*m + k] = sum;
        }
    }
    // Solve linear system using Gauss elimination with partial pivoting
    std::vector<T> coeffs = ATy;
    std::vector<T> mat = ATA;
    for (std::size_t col = 0; col < m; ++col) {
        std::size_t max_row = col;
        T max_val = std::abs(mat[col*m + col]);
        for (std::size_t row = col+1; row < m; ++row) {
            T val = std::abs(mat[row*m + col]);
            if (val > max_val) { max_val = val; max_row = row; }
        }
        if (max_val < T(1e-12)) continue;
        if (max_row != col) {
            for (std::size_t j = 0; j < m; ++j) std::swap(mat[col*m + j], mat[max_row*m + j]);
            std::swap(coeffs[col], coeffs[max_row]);
        }
        T pivot = mat[col*m + col];
        for (std::size_t row = col+1; row < m; ++row) {
            T factor = mat[row*m + col] / pivot;
            for (std::size_t j = col; j < m; ++j) mat[row*m + j] -= factor * mat[col*m + j];
            coeffs[row] -= factor * coeffs[col];
        }
    }
    for (std::size_t col = m; col-- > 0;) {
        T sum = coeffs[col];
        for (std::size_t j = col+1; j < m; ++j) sum -= mat[col*m + j] * coeffs[j];
        coeffs[col] = sum / mat[col*m + col];
    }
    return polynomial<T>(std::move(coeffs));
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_POLYNOMIAL_H