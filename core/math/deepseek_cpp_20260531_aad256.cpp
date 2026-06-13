//File 0066 : core/math/dual_numbers.h
//Forward‑mode automatic differentiation using dual numbers; all elementary functions, digamma‑based lgamma/tgamma, atan2, batch gradient computation.
#ifndef CORE_MATH_DUAL_NUMBERS_H
#define CORE_MATH_DUAL_NUMBERS_H

#include <cmath>
#include <type_traits>
#include <vector>
#include <limits>
#include <cstdint>

namespace SimulationMath {
namespace dual {

// -----------------------------------------------------------------------------
// 1. Dual number template (value + derivative)
// -----------------------------------------------------------------------------
template <typename T>
struct Dual {
    T real;
    T dual;

    Dual() noexcept : real(0), dual(0) {}
    Dual(T r, T d) noexcept : real(r), dual(d) {}

    template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    Dual(U value) noexcept : real(static_cast<T>(value)), dual(0) {}

    operator T() const noexcept { return real; }
};

// -----------------------------------------------------------------------------
// 2. Arithmetic
// -----------------------------------------------------------------------------
template <typename T>
Dual<T> operator+(const Dual<T>& a, const Dual<T>& b) noexcept { return {a.real + b.real, a.dual + b.dual}; }
template <typename T>
Dual<T> operator-(const Dual<T>& a, const Dual<T>& b) noexcept { return {a.real - b.real, a.dual - b.dual}; }
template <typename T>
Dual<T> operator*(const Dual<T>& a, const Dual<T>& b) noexcept { return {a.real * b.real, a.real * b.dual + a.dual * b.real}; }
template <typename T>
Dual<T> operator/(const Dual<T>& a, const Dual<T>& b) noexcept {
    T inv = T(1) / b.real;
    return {a.real * inv, (a.dual - a.real * b.dual * inv) * inv};
}
template <typename T>
Dual<T> operator-(const Dual<T>& a) noexcept { return {-a.real, -a.dual}; }

template <typename T> Dual<T>& operator+=(Dual<T>& a, const Dual<T>& b) noexcept { a = a + b; return a; }
template <typename T> Dual<T>& operator-=(Dual<T>& a, const Dual<T>& b) noexcept { a = a - b; return a; }
template <typename T> Dual<T>& operator*=(Dual<T>& a, const Dual<T>& b) noexcept { a = a * b; return a; }
template <typename T> Dual<T>& operator/=(Dual<T>& a, const Dual<T>& b) noexcept { a = a / b; return a; }

// Mixed scalar operators
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator*(U s, const Dual<T>& d) noexcept { return {static_cast<T>(s) * d.real, static_cast<T>(s) * d.dual}; }
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator*(const Dual<T>& d, U s) noexcept { return s * d; }
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator/(U s, const Dual<T>& d) noexcept {
    T inv = T(1) / d.real;
    return {static_cast<T>(s) * inv, -static_cast<T>(s) * d.dual * inv * inv};
}
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator/(const Dual<T>& d, U s) noexcept {
    T inv = T(1) / static_cast<T>(s);
    return {d.real * inv, d.dual * inv};
}
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator+(U s, const Dual<T>& d) noexcept { return {static_cast<T>(s) + d.real, d.dual}; }
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator+(const Dual<T>& d, U s) noexcept { return s + d; }
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator-(U s, const Dual<T>& d) noexcept { return {static_cast<T>(s) - d.real, -d.dual}; }
template <typename T, typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
Dual<T> operator-(const Dual<T>& d, U s) noexcept { return {d.real - static_cast<T>(s), d.dual}; }

// -----------------------------------------------------------------------------
// 3. Elementary functions
// -----------------------------------------------------------------------------
template <typename T> Dual<T> sin(const Dual<T>& d) noexcept { return {std::sin(d.real), d.dual * std::cos(d.real)}; }
template <typename T> Dual<T> cos(const Dual<T>& d) noexcept { return {std::cos(d.real), -d.dual * std::sin(d.real)}; }
template <typename T> Dual<T> tan(const Dual<T>& d) noexcept {
    T c = std::cos(d.real);
    T inv = T(1) / (c * c);
    return {std::tan(d.real), d.dual * inv};
}
template <typename T> Dual<T> asin(const Dual<T>& d) noexcept {
    T denom = std::sqrt(T(1) - d.real * d.real);
    return {std::asin(d.real), d.dual / denom};
}
template <typename T> Dual<T> acos(const Dual<T>& d) noexcept {
    T denom = std::sqrt(T(1) - d.real * d.real);
    return {std::acos(d.real), -d.dual / denom};
}
template <typename T> Dual<T> atan(const Dual<T>& d) noexcept { return {std::atan(d.real), d.dual / (T(1) + d.real * d.real)}; }
template <typename T> Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) noexcept {
    T r = y.real / x.real;
    return {std::atan2(y.real, x.real), (x.real * y.dual - y.real * x.dual) / (x.real * x.real + y.real * y.real)};
}
template <typename T> Dual<T> exp(const Dual<T>& d) noexcept { T e = std::exp(d.real); return {e, d.dual * e}; }
template <typename T> Dual<T> log(const Dual<T>& d) noexcept { return {std::log(d.real), d.dual / d.real}; }
template <typename T> Dual<T> sqrt(const Dual<T>& d) noexcept {
    T s = std::sqrt(d.real);
    return {s, d.dual / (T(2) * s)};
}
template <typename T> Dual<T> cbrt(const Dual<T>& d) noexcept {
    T c = std::cbrt(d.real);
    return {c, d.dual / (T(3) * c * c)};
}
template <typename T> Dual<T> fabs(const Dual<T>& d) noexcept { return {std::fabs(d.real), (d.real >= 0 ? d.dual : -d.dual)}; }
template <typename T> Dual<T> pow(const Dual<T>& base, const Dual<T>& exp) noexcept {
    T val = std::pow(base.real, exp.real);
    T deriv = val * (exp.dual * std::log(base.real) + (exp.real * base.dual / base.real));
    return {val, deriv};
}
template <typename T> Dual<T> pow(const Dual<T>& base, T real_exp) noexcept {
    T val = std::pow(base.real, real_exp);
    return {val, real_exp * base.dual * std::pow(base.real, real_exp - T(1))};
}
template <typename T> Dual<T> pow(T real_base, const Dual<T>& exp) noexcept {
    T val = std::pow(real_base, exp.real);
    return {val, val * std::log(real_base) * exp.dual};
}
template <typename T> Dual<T> sinh(const Dual<T>& d) noexcept { return {std::sinh(d.real), d.dual * std::cosh(d.real)}; }
template <typename T> Dual<T> cosh(const Dual<T>& d) noexcept { return {std::cosh(d.real), d.dual * std::sinh(d.real)}; }
template <typename T> Dual<T> tanh(const Dual<T>& d) noexcept {
    T c = std::cosh(d.real);
    T inv = T(1) / (c * c);
    return {std::tanh(d.real), d.dual * inv};
}

// -----------------------------------------------------------------------------
// 4. Special functions (erf, erfc, lgamma, tgamma) with full derivative using digamma
// -----------------------------------------------------------------------------
template <typename T>
Dual<T> erf(const Dual<T>& d) noexcept {
    T val = std::erf(d.real);
    T deriv = (T(2) / std::sqrt(T(3.14159265358979323846))) * std::exp(-d.real * d.real) * d.dual;
    return {val, deriv};
}
template <typename T>
Dual<T> erfc(const Dual<T>& d) noexcept {
    T val = std::erfc(d.real);
    T deriv = -(T(2) / std::sqrt(T(3.14159265358979323846))) * std::exp(-d.real * d.real) * d.dual;
    return {val, deriv};
}

// Accurate digamma function (ψ) for positive real x
namespace detail {
    template <typename T>
    T digamma_impl(T x) noexcept {
        // Shift x up to >= 10 using recurrence: ψ(x+1) = ψ(x) + 1/x
        T shift = 0;
        while (x < T(10)) {
            shift -= T(1) / x;
            x += T(1);
        }
        // Asymptotic expansion for large x: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
        T inv = T(1) / x;
        T inv2 = inv * inv;
        T result = std::log(x) - T(0.5) * inv - T(1.0/12.0) * inv2
                   + T(1.0/120.0) * inv2 * inv2 - T(1.0/252.0) * inv2 * inv2 * inv2;
        return result + shift;
    }
} // namespace detail

template <typename T>
Dual<T> lgamma(const Dual<T>& d) noexcept {
    T val = std::lgamma(d.real);
    T psi_val = detail::digamma_impl(d.real);
    return {val, d.dual * psi_val};
}
template <typename T>
Dual<T> tgamma(const Dual<T>& d) noexcept {
    T val = std::tgamma(d.real);
    T psi_val = detail::digamma_impl(d.real);
    return {val, d.dual * val * psi_val};
}

// -----------------------------------------------------------------------------
// 5. Gradient computation for vector‑valued function (scalar output)
// -----------------------------------------------------------------------------
template <typename T, typename Func>
std::vector<T> gradient_dual(const Func& f, const std::vector<T>& x) noexcept {
    std::vector<T> grad(x.size(), T(0));
    for (size_t i = 0; i < x.size(); ++i) {
        std::vector<Dual<T>> xd(x.size(), Dual<T>(x[i], T(0)));
        xd[i] = Dual<T>(x[i], T(1));
        // f must return Dual<T> and accept std::vector<Dual<T>>
        Dual<T> result = f(xd);
        grad[i] = result.dual;
    }
    return grad;
}

// For a scalar function of a single variable
template <typename T, typename Func>
T derivative(const Func& f, T x) noexcept {
    return f(Dual<T>(x, T(1))).dual;
}

} // namespace dual
} // namespace SimulationMath

#endif // CORE_MATH_DUAL_NUMBERS_H