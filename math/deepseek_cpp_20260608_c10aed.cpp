//File group name : OrthoTree Math
//File 0067 : core/math/interval/affine.h
//Affine arithmetic: representing quantities as central value + linear combination of noise symbols. Provides affine forms, conversion from interval, arithmetic operations (+, -, *), and forward error propagation.

#ifndef ORTHOTREE_CORE_MATH_INTERVAL_AFFINE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_INTERVAL_AFFINE_H_INCLUDED

#include "../../build_config.h"
#include "interval.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Interval {

// ============================================================================
//  AffineForm: x = x0 + Σ x_i * ε_i, where ε_i ∈ [-1, 1] are noise symbols.
//  Provides tighter error bounds than interval arithmetic.
// ============================================================================
template<typename T = float>
class AffineForm {
public:
    using value_type = T;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    AffineForm() noexcept : m_central(T(0)) {}
    explicit AffineForm(T value) noexcept : m_central(value) {}
    AffineForm(T central, const std::vector<T>& errors) noexcept
        : m_central(central), m_errors(errors) {}
    explicit AffineForm(const Interval<T>& iv) {
        m_central = iv.center();
        m_errors = {iv.radius()}; // one noise symbol
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    T central() const noexcept { return m_central; }
    const std::vector<T>& errors() const noexcept { return m_errors; }
    std::vector<T>& errors() noexcept { return m_errors; }
    size_type numSymbols() const noexcept { return m_errors.size(); }

    // ------------------------------------------------------------------------
    //  Convert to interval (conservative bound)
    // ------------------------------------------------------------------------
    Interval<T> toInterval() const noexcept {
        T radius = T(0);
        for (T e : m_errors) radius += std::abs(e);
        return Interval<T>(m_central - radius, m_central + radius);
    }

    // ------------------------------------------------------------------------
    //  Arithmetic operations
    //  Addition: central + central', errors concatenated
    // ------------------------------------------------------------------------
    AffineForm operator+(const AffineForm& other) const noexcept {
        AffineForm result;
        result.m_central = m_central + other.m_central;
        result.m_errors = m_errors;
        result.m_errors.insert(result.m_errors.end(), other.m_errors.begin(), other.m_errors.end());
        return result;
    }
    AffineForm operator-(const AffineForm& other) const noexcept {
        AffineForm result;
        result.m_central = m_central - other.m_central;
        result.m_errors = m_errors;
        for (T e : other.m_errors) result.m_errors.push_back(-e);
        return result;
    }
    AffineForm& operator+=(const AffineForm& other) noexcept { *this = *this + other; return *this; }
    AffineForm& operator-=(const AffineForm& other) noexcept { *this = *this - other; return *this; }

    // ------------------------------------------------------------------------
    //  Multiplication: (x0 + Σ x_i ε_i) * (y0 + Σ y_j ε_j)
    //  = x0*y0 + Σ (x0*y_j + y0*x_i) ε + Σ Σ x_i*y_j ε_i ε_j
    //  The second‑order terms are bounded and added as a new noise symbol.
    // ------------------------------------------------------------------------
    AffineForm operator*(const AffineForm& other) const noexcept {
        // Compute central term
        T central = m_central * other.m_central;
        // Linear terms: combine contributions
        std::vector<T> newErrors;
        // From this * other's central
        for (T e : m_errors) newErrors.push_back(e * other.m_central);
        // From central * other's errors
        for (T e : other.m_errors) newErrors.push_back(m_central * e);
        // Second‑order term: Σ Σ x_i*y_j * ε_i * ε_j -> bounded by sum of absolute values
        T secondOrderBound = T(0);
        for (T ei : m_errors) {
            for (T ej : other.m_errors) {
                secondOrderBound += std::abs(ei * ej);
            }
        }
        // Add as new noise symbol (or merge into existing)
        if (secondOrderBound > T(0)) {
            newErrors.push_back(secondOrderBound);
        }
        AffineForm result;
        result.m_central = central;
        result.m_errors = std::move(newErrors);
        return result;
    }

    // Multiplication by scalar
    AffineForm operator*(T scalar) const noexcept {
        AffineForm result;
        result.m_central = m_central * scalar;
        result.m_errors.reserve(m_errors.size());
        for (T e : m_errors) result.m_errors.push_back(e * scalar);
        return result;
    }
    friend AffineForm operator*(T scalar, const AffineForm& a) noexcept { return a * scalar; }

    // ------------------------------------------------------------------------
    //  Inverse (1/x) – not trivial; simplified using interval conversion
    //  For full affine division, one would use Chebyshev approximation.
    //  Here we implement a simple interval‑based fallback.
    // ------------------------------------------------------------------------
    AffineForm inverse() const {
        Interval<T> iv = toInterval();
        if (iv.contains(T(0))) return AffineForm(Interval<T>(-1e12, 1e12)); // large bound
        T lo = T(1) / iv.high();
        T hi = T(1) / iv.low();
        if (lo > hi) std::swap(lo, hi);
        return AffineForm(Interval<T>(lo, hi));
    }

    // ------------------------------------------------------------------------
    //  Square (x^2) – optimised: central^2 + 2*central*Σ + (Σ)^2
    //  The (Σ)^2 term is bounded as new noise symbol.
    // ------------------------------------------------------------------------
    AffineForm square() const {
        T centralSq = m_central * m_central;
        std::vector<T> newErrors;
        // linear term: 2 * central * each error
        for (T e : m_errors) newErrors.push_back(T(2) * m_central * e);
        // second‑order: sum of absolute pairwise products
        T bound = T(0);
        for (size_t i = 0; i < m_errors.size(); ++i) {
            for (size_t j = 0; j < m_errors.size(); ++j) {
                bound += std::abs(m_errors[i] * m_errors[j]);
            }
        }
        if (bound > T(0)) newErrors.push_back(bound);
        AffineForm result;
        result.m_central = centralSq;
        result.m_errors = std::move(newErrors);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Reduce the number of noise symbols by merging small contributions
    // ------------------------------------------------------------------------
    void compress(T tolerance = T(1e-12)) {
        if (m_errors.empty()) return;
        // Sum all absolute errors to estimate total radius
        T totalRadius = T(0);
        for (T e : m_errors) totalRadius += std::abs(e);
        // Merge small terms into a single new symbol
        T merged = T(0);
        std::vector<T> kept;
        for (T e : m_errors) {
            if (std::abs(e) < tolerance * totalRadius) {
                merged += e;
            } else {
                kept.push_back(e);
            }
        }
        if (std::abs(merged) > T(0)) kept.push_back(merged);
        m_errors.swap(kept);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const AffineForm& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const {
        return std::abs(m_central - other.m_central) < eps && m_errors.size() == other.m_errors.size() &&
               std::equal(m_errors.begin(), m_errors.end(), other.m_errors.begin(),
                          [eps](T a, T b) { return std::abs(a - b) < eps; });
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: add 4 affine forms pairwise (central + errors combined)
    //  Not fully SIMD due to variable errors sizes; we provide a scalar loop.
    // ------------------------------------------------------------------------
    static void batchAdd(const AffineForm* a, const AffineForm* b, AffineForm* out, size_t count) {
        for (size_t i = 0; i < count; ++i) out[i] = a[i] + b[i];
    }

private:
    T m_central;
    std::vector<T> m_errors;
};

// ----------------------------------------------------------------------------
//  Convenience: convert interval to affine form with one noise symbol
// ----------------------------------------------------------------------------
template<typename T>
AffineForm<T> makeAffine(const Interval<T>& iv) {
    return AffineForm<T>(iv);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class AffineEnvironment {
public:
    static AffineEnvironment& instance() {
        static AffineEnvironment env;
        return env;
    }
    void setCompressTolerance(T tol) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_compressTol = tol;
    }
    T compressTolerance() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_compressTol;
    }
private:
    AffineEnvironment() : m_compressTol(T(1e-12)) {}
    mutable std::mutex m_mutex;
    T m_compressTol;
};

} // namespace Interval
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_INTERVAL_AFFINE_H_INCLUDED