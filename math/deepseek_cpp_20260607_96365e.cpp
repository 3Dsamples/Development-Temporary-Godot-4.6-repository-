//File group name : OrthoTree Math
//File 0049 : core/math/math_config.h
//Global configuration for the math module: scalar type, tolerance, SIMD level, rounding mode, and dynamic environment controls.

#ifndef ORTHOTREE_CORE_MATH_CONFIG_H_INCLUDED
#define ORTHOTREE_CORE_MATH_CONFIG_H_INCLUDED

#include "../../build_config.h"
#include <cmath>
#include <limits>
#include <mutex>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Global math configuration (dynamic environment)
//  Allows runtime adjustment of numerical precision, SIMD usage,
//  default epsilon, and other global parameters.
// ============================================================================
class MathConfig {
public:
    // ------------------------------------------------------------------------
    //  Singleton access
    // ------------------------------------------------------------------------
    static MathConfig& instance() {
        static MathConfig config;
        return config;
    }

    // ------------------------------------------------------------------------
    //  Default epsilon for comparisons (e.g., 1e-6 for float, 1e-12 for double)
    // ------------------------------------------------------------------------
    void setDefaultEpsilon(double eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultEpsilon = eps;
    }
    double defaultEpsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultEpsilon;
    }

    // ------------------------------------------------------------------------
    //  Enable/disable SIMD acceleration across all math functions
    // ------------------------------------------------------------------------
    void setUseSIMD(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = enable;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }

    // ------------------------------------------------------------------------
    //  Floating‑point rounding mode (fegetround / fesetround)
    //  Only effective if ORTHOTREE_MATH_ROUNDING is enabled.
    // ------------------------------------------------------------------------
    enum class RoundingMode : uint8_t {
        ToNearest,
        Downward,
        Upward,
        TowardZero
    };
    void setRoundingMode(RoundingMode mode) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_roundingMode = mode;
#if ORTHOTREE_MATH_ROUNDING && defined(__STDC_IEC_559__)
        int rmode = FE_TONEAREST;
        switch (mode) {
            case RoundingMode::ToNearest: rmode = FE_TONEAREST; break;
            case RoundingMode::Downward: rmode = FE_DOWNWARD; break;
            case RingingMode::Upward: rmode = FE_UPWARD; break;
            case RoundingMode::TowardZero: rmode = FE_TOWARDZERO; break;
        }
        fesetround(rmode);
#endif
    }
    RoundingMode roundingMode() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_roundingMode;
    }

    // ------------------------------------------------------------------------
    //  Finite difference epsilon for numerical derivatives (default 1e-5)
    // ------------------------------------------------------------------------
    void setDerivativeEpsilon(double eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_derivativeEps = eps;
    }
    double derivativeEpsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_derivativeEps;
    }

    // ------------------------------------------------------------------------
    //  Maximum iterations for iterative solvers (Newton, bisection, etc.)
    // ------------------------------------------------------------------------
    void setMaxIterations(uint32_t maxIter) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_maxIterations = maxIter;
    }
    uint32_t maxIterations() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_maxIterations;
    }

private:
    MathConfig()
        : m_defaultEpsilon(1e-8)
        , m_useSIMD(true)
        , m_roundingMode(RoundingMode::ToNearest)
        , m_derivativeEps(1e-6)
        , m_maxIterations(100) {}

    mutable std::mutex m_mutex;
    double m_defaultEpsilon;
    bool m_useSIMD;
    RoundingMode m_roundingMode;
    double m_derivativeEps;
    uint32_t m_maxIterations;
};

// ----------------------------------------------------------------------------
//  Convenience inline functions to access config without singleton call
// ----------------------------------------------------------------------------
inline double defaultEpsilon() { return MathConfig::instance().defaultEpsilon(); }
inline bool useSIMD() { return MathConfig::instance().useSIMD(); }
inline MathConfig::RoundingMode roundingMode() { return MathConfig::instance().roundingMode(); }
inline double derivativeEpsilon() { return MathConfig::instance().derivativeEpsilon(); }
inline uint32_t maxIterations() { return MathConfig::instance().maxIterations(); }

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_CONFIG_H_INCLUDED