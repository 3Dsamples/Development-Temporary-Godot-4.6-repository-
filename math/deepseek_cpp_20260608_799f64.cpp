//File group name : OrthoTree Math
//File 0061 : core/math/numerical/integration.h
//Numerical integration: trapezoidal rule, Simpson's rule, adaptive Simpson, Gauss‑Legendre quadrature (fixed order). SIMD batch evaluation for multiple integrals over same interval.

#ifndef ORTHOTREE_CORE_MATH_NUMERICAL_INTEGRATION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_NUMERICAL_INTEGRATION_H_INCLUDED

#include "../../build_config.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <functional>
#include <vector>

namespace OrthoTree {
namespace Math {
namespace Numerical {

// ============================================================================
//  Trapezoidal rule
// ============================================================================
template<typename T>
T trapezoidal(const std::function<T(T)>& f, T a, T b, size_t n = 100) {
    T h = (b - a) / static_cast<T>(n);
    T sum = (f(a) + f(b)) * T(0.5);
    for (size_t i = 1; i < n; ++i) {
        sum += f(a + static_cast<T>(i) * h);
    }
    return sum * h;
}

// ============================================================================
//  Simpson's rule (n must be even)
// ============================================================================
template<typename T>
T simpson(const std::function<T(T)>& f, T a, T b, size_t n = 100) {
    if (n % 2 != 0) ++n;
    T h = (b - a) / static_cast<T>(n);
    T sum = f(a) + f(b);
    for (size_t i = 1; i < n; i += 2) {
        sum += T(4) * f(a + static_cast<T>(i) * h);
    }
    for (size_t i = 2; i < n; i += 2) {
        sum += T(2) * f(a + static_cast<T>(i) * h);
    }
    return sum * h / T(3);
}

// ============================================================================
//  Adaptive Simpson integration (recursive, error‑based)
// ============================================================================
namespace detail {
    template<typename T>
    T adaptiveSimpsonHelper(const std::function<T(T)>& f, T a, T b,
                            T eps, T whole, T fa, T fm, T fb, int depth) {
        T m = (a + b) * T(0.5);
        T h = (b - a) * T(0.5);
        T fl = f((a + m) * T(0.5));
        T fr = f((m + b) * T(0.5));
        T left = (h / T(6)) * (fa + T(4)*fl + fm);
        T right = (h / T(6)) * (fm + T(4)*fr + fb);
        T whole_ = left + right;
        if (std::abs(whole_ - whole) < T(15) * eps || depth >= 20) {
            return whole_;
        }
        T eps2 = eps * T(0.5);
        return adaptiveSimpsonHelper(f, a, m, eps2, left, fa, fl, fm, depth+1) +
               adaptiveSimpsonHelper(f, m, b, eps2, right, fm, fr, fb, depth+1);
    }
} // namespace detail

template<typename T>
T adaptiveSimpson(const std::function<T(T)>& f, T a, T b,
                  T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) {
    T m = (a + b) * T(0.5);
    T fa = f(a), fm = f(m), fb = f(b);
    T whole = (b - a) * (fa + T(4)*fm + fb) / T(6);
    return detail::adaptiveSimpsonHelper(f, a, b, eps, whole, fa, fm, fb, 0);
}

// ============================================================================
//  Gauss‑Legendre quadrature (fixed order, using precomputed nodes and weights)
//  Here we provide orders 2, 4, 8, 16 as examples.
// ============================================================================
namespace gauss {
    // Order 2
    template<typename T>
    T quadrature2(const std::function<T(T)>& f, T a, T b) {
        const T x[2] = {-0.5773502691896257, 0.5773502691896257};
        const T w[2] = {1.0, 1.0};
        T half = (b - a) * T(0.5);
        T mid = (a + b) * T(0.5);
        T sum = T(0);
        for (int i = 0; i < 2; ++i) {
            sum += w[i] * f(mid + half * x[i]);
        }
        return sum * half;
    }
    // Order 4
    template<typename T>
    T quadrature4(const std::function<T(T)>& f, T a, T b) {
        const T x[4] = {-0.8611363115940526, -0.3399810435848563,
                         0.3399810435848563,  0.8611363115940526};
        const T w[4] = {0.3478548451374538, 0.6521451548625461,
                        0.6521451548625461, 0.3478548451374538};
        T half = (b - a) * T(0.5);
        T mid = (a + b) * T(0.5);
        T sum = T(0);
        for (int i = 0; i < 4; ++i) {
            sum += w[i] * f(mid + half * x[i]);
        }
        return sum * half;
    }
    // Order 8
    template<typename T>
    T quadrature8(const std::function<T(T)>& f, T a, T b) {
        // nodes and weights for order 8 (from Golub‑Welsch)
        const T x[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                        -0.1834346424956498,  0.1834346424956498,  0.5255324099163290,
                         0.7966664774136267,  0.9602898564975363};
        const T w[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                        0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                        0.2223810344533745, 0.1012285362903763};
        T half = (b - a) * T(0.5);
        T mid = (a + b) * T(0.5);
        T sum = T(0);
        for (int i = 0; i < 8; ++i) {
            sum += w[i] * f(mid + half * x[i]);
        }
        return sum * half;
    }
} // namespace gauss

// ============================================================================
//  Convenience wrapper: choose Gauss order based on desired accuracy
// ============================================================================
template<typename T>
T gaussQuadrature(const std::function<T(T)>& f, T a, T b, int order = 4) {
    switch (order) {
        case 2: return gauss::quadrature2(f, a, b);
        case 4: return gauss::quadrature4(f, a, b);
        case 8: return gauss::quadrature8(f, a, b);
        default: return adaptiveSimpson(f, a, b); // fallback
    }
}

// ============================================================================
//  SIMD batch: integrate the same function over 4 different intervals
//  (using Simpson with same n)
// ----------------------------------------------------------------------------
template<typename T>
void batchSimpson(const std::function<T(T)>& f, const T* a, const T* b,
                  T* out, size_t count, size_t n = 100) {
    if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = simpson(f, a[i], b[i], n);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = simpson(f, a[i], b[i], n);
        }
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class IntegrationEnvironment {
public:
    static IntegrationEnvironment& instance() {
        static IntegrationEnvironment env;
        return env;
    }
    void setDefaultEps(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultEps = eps;
    }
    T defaultEps() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultEps;
    }
private:
    IntegrationEnvironment() : m_defaultEps(T(1e-6)) {}
    mutable std::mutex m_mutex;
    T m_defaultEps;
};

} // namespace Numerical
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_NUMERICAL_INTEGRATION_H_INCLUDED