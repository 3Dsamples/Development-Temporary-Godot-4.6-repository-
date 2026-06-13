/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_MATH_EXTENDED_MICROSCOPIC_UNITS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_EXTENDED_MICROSCOPIC_UNITS_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "../vector_math.h"
#include "../numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <array>
#include <algorithm>

namespace OrthoTree {
namespace Math {
namespace Extended {

// ============================================================================
//  Fundamental physical constants for microscopic scale (SI units)
// ============================================================================
template<typename T>
struct MicroscopicConstants {
    // Length
    static constexpr T METER_TO_NANOMETER  = T(1e9);
    static constexpr T METER_TO_PICOMETER  = T(1e12);
    static constexpr T METER_TO_FEMTOMETER = T(1e15);
    static constexpr T METER_TO_ANGSTROM   = T(1e10);
    static constexpr T ANGSTROM_TO_METER   = T(1e-10);
    static constexpr T NANOMETER_TO_METER  = T(1e-9);
    static constexpr T PICOMETER_TO_METER  = T(1e-12);

    // Time
    static constexpr T SECOND_TO_PICOSECOND  = T(1e12);
    static constexpr T SECOND_TO_FEMTOSECOND = T(1e15);
    static constexpr T PICOSECOND_TO_SECOND  = T(1e-12);
    static constexpr T FEMTOSECOND_TO_SECOND = T(1e-15);

    // Energy
    static constexpr T JOULE_TO_EV           = T(6.241509074e18);
    static constexpr T EV_TO_JOULE           = T(1.602176634e-19);
    static constexpr T JOULE_TO_KJ_PER_MOL   = T(6.02214076e20); // per mol
    static constexpr T BOLTZMANN             = T(1.380649e-23);   // J/K
    static constexpr T AVOGADRO              = T(6.02214076e23);
    static constexpr T GAS_CONSTANT          = T(8.314462618);    // J/(mol·K)

    // Mass
    static constexpr T KILOGRAM_TO_ATOMIC_MASS = T(6.02214076e26); // kg -> u
    static constexpr T ATOMIC_MASS_TO_KG       = T(1.66053906660e-27);
    static constexpr T ELECTRON_MASS_KG        = T(9.1093837015e-31);
    static constexpr T PROTON_MASS_KG          = T(1.67262192369e-27);
    static constexpr T NEUTRON_MASS_KG         = T(1.67492749804e-27);
};

// ============================================================================
//  Quantities with units (length, time, energy, mass) – compile‑time unit safety
// ============================================================================
template<typename T, int M, int L, int TExp, int I, int Theta, int N, int J>
class Quantity {
public:
    using value_type = T;
    constexpr Quantity() noexcept : m_value(0) {}
    explicit constexpr Quantity(T value) noexcept : m_value(value) {}

    T value() const noexcept { return m_value; }
    Quantity& operator+=(const Quantity& other) noexcept { m_value += other.m_value; return *this; }
    Quantity& operator-=(const Quantity& other) noexcept { m_value -= other.m_value; return *this; }
    Quantity operator+(const Quantity& other) const noexcept { return Quantity(m_value + other.m_value); }
    Quantity operator-(const Quantity& other) const noexcept { return Quantity(m_value - other.m_value); }

    template<int M2, int L2, int T2, int I2, int Theta2, int N2, int J2>
    Quantity< T, M+M2, L+L2, TExp+T2, I+I2, Theta+Theta2, N+N2, J+J2 >
    operator*(const Quantity<T, M2, L2, T2, I2, Theta2, N2, J2>& other) const noexcept {
        return Quantity< T, M+M2, L+L2, TExp+T2, I+I2, Theta+Theta2, N+N2, J+J2 >(m_value * other.value());
    }

    template<int M2, int L2, int T2, int I2, int Theta2, int N2, int J2>
    Quantity< T, M-M2, L-L2, TExp-T2, I-I2, Theta-Theta2, N-N2, J-J2 >
    operator/(const Quantity<T, M2, L2, T2, I2, Theta2, N2, J2>& other) const noexcept {
        return Quantity< T, M-M2, L-L2, TExp-T2, I-I2, Theta-Theta2, N-N2, J-J2 >(m_value / other.value());
    }

    Quantity operator*(T scalar) const noexcept { return Quantity(m_value * scalar); }
    Quantity operator/(T scalar) const noexcept { return Quantity(m_value / scalar); }

private:
    T m_value;
};

// Convenient aliases for common microscopic quantities
template<typename T> using Length   = Quantity<T, 0, 1, 0, 0, 0, 0, 0>;
template<typename T> using Time     = Quantity<T, 0, 0, 1, 0, 0, 0, 0>;
template<typename T> using Mass     = Quantity<T, 1, 0, 0, 0, 0, 0, 0>;
template<typename T> using Energy   = Quantity<T, 1, 2,-2, 0, 0, 0, 0>;
template<typename T> using Velocity = Quantity<T, 0, 1,-1, 0, 0, 0, 0>;
template<typename T> using Force    = Quantity<T, 1, 1,-2, 0, 0, 0, 0>;

// ----------------------------------------------------------------------------
//  Unit conversion helpers (SIMD batch)
// ----------------------------------------------------------------------------
template<typename T, bool UseSIMD = true>
class UnitConverter {
public:
    static T nmToMeters(T nm) noexcept { return nm * T(1e-9); }
    static T metersToNm(T m) noexcept { return m * T(1e9); }
    static T pmToMeters(T pm) noexcept { return pm * T(1e-12); }
    static T metersToPm(T m) noexcept { return m * T(1e12); }
    static T angstromToMeters(T a) noexcept { return a * T(1e-10); }
    static T metersToAngstrom(T m) noexcept { return m * T(1e10); }
    static T evToJoules(T ev) noexcept { return ev * T(1.602176634e-19); }
    static T joulesToEv(T j) noexcept { return j * T(6.241509074e18); }
    static T psToSeconds(T ps) noexcept { return ps * T(1e-12); }
    static T secondsToPs(T s) noexcept { return s * T(1e12); }
    static T fsToSeconds(T fs) noexcept { return fs * T(1e-15); }
    static T secondsToFs(T s) noexcept { return s * T(1e15); }

    // Batch conversions (SIMD friendly)
    static void batchNmToMeters(const T* src, T* dst, std::size_t count) noexcept {
        if constexpr (UseSIMD && ORTHOTREE_SIMD_LEVEL >= 128) {
            // SIMD loop (pseudo - actual implementation would use aligned loads and mulps)
            for (std::size_t i = 0; i < count; ++i) dst[i] = src[i] * T(1e-9);
        } else {
            for (std::size_t i = 0; i < count; ++i) dst[i] = src[i] * T(1e-9);
        }
    }
};

// ============================================================================
//  Adaptive precision for microscopic simulations (sub‑Angstrom)
// ============================================================================
template<typename T>
class MicroscopicPrecision {
public:
    enum class Level : uint8_t {
        Angstrom,     // 1e-10 m
        Picometer,    // 1e-12 m
        Femtometer,   // 1e-15 m
        Attometer     // 1e-18 m
    };

    MicroscopicPrecision() noexcept : m_level(Level::Angstrom) {}

    void setLevel(Level lvl) noexcept { m_level = lvl; }
    Level level() const noexcept { return m_level; }

    T tolerance() const noexcept {
        switch (m_level) {
            case Level::Angstrom:   return T(1e-10);
            case Level::Picometer:  return T(1e-12);
            case Level::Femtometer: return T(1e-15);
            case Level::Attometer:  return T(1e-18);
            default: return T(1e-10);
        }
    }

    // Compare two positions with current tolerance
    bool equal(const Math::Vector<T, 3>& a, const Math::Vector<T, 3>& b) const noexcept {
        return (a - b).length() <= tolerance();
    }

    // Adaptive quantization for molecular coordinates
    uint64_t quantize(T coord) const noexcept {
        T scaled = coord / tolerance();
        return static_cast<uint64_t>(std::abs(scaled));
    }

private:
    Level m_level;
};

// ============================================================================
//  Dynamic environment controller for microscopic simulations
// ============================================================================
template<typename T>
class MicroscopicEnvironment {
public:
    using vec3 = Math::Vector<T, 3>;

    MicroscopicEnvironment() noexcept
        : m_temperature(300.0)      // 300 K
        , m_pressure(101325.0)      // 1 atm in Pa
        , m_permittivity(1.0)       // relative permittivity (vacuum)
        , m_viscosity(0.0)
        , m_usePeriodic(true)
        , m_boxSize(vec3(T(10e-9))) // 10 nm cube
    {}

    void setTemperature(T kelvin) noexcept { m_temperature = kelvin; }
    void setPressure(T pascal) noexcept { m_pressure = pascal; }
    void setPermittivity(T eps) noexcept { m_permittivity = eps; }
    void setViscosity(T eta) noexcept { m_viscosity = eta; }
    void setPeriodicBoundary(bool enable) noexcept { m_usePeriodic = enable; }
    void setBoxSize(const vec3& size) noexcept { m_boxSize = size; }

    T temperature() const noexcept { return m_temperature; }
    T pressure() const noexcept { return m_pressure; }
    T permittivity() const noexcept { return m_permittivity; }
    T viscosity() const noexcept { return m_viscosity; }

    // Thermal energy kT in Joules
    T thermalEnergy() const noexcept {
        return MicroscopicConstants<T>::BOLTZMANN * m_temperature;
    }

    // Mean free path (ideal gas approximation)
    T meanFreePath(T particleDiameter) const noexcept {
        T n = pressure() / (MicroscopicConstants<T>::BOLTZMANN * m_temperature);
        T crossSection = Math::pi<T>() * particleDiameter * particleDiameter;
        return T(1) / (std::sqrt(T(2)) * n * crossSection);
    }

    // Apply periodic boundary conditions
    vec3 applyPeriodic(const vec3& pos) const noexcept {
        if (!m_usePeriodic) return pos;
        vec3 result;
        for (int i = 0; i < 3; ++i) {
            T half = m_boxSize[i] * T(0.5);
            T p = pos[i];
            if (p > half) p -= m_boxSize[i];
            else if (p < -half) p += m_boxSize[i];
            result[i] = p;
        }
        return result;
    }

    // Minimum image distance (for periodic systems)
    T minImageDistance(const vec3& a, const vec3& b) const noexcept {
        if (!m_usePeriodic) return (a - b).length();
        vec3 delta = a - b;
        for (int i = 0; i < 3; ++i) {
            if (delta[i] > m_boxSize[i] * T(0.5)) delta[i] -= m_boxSize[i];
            else if (delta[i] < -m_boxSize[i] * T(0.5)) delta[i] += m_boxSize[i];
        }
        return delta.length();
    }

private:
    T m_temperature;
    T m_pressure;
    T m_permittivity;
    T m_viscosity;
    bool m_usePeriodic;
    vec3 m_boxSize;
};

// ============================================================================
//  Lennard‑Jones potential (microscopic interaction)
// ============================================================================
template<typename T>
class LennardJones {
public:
    LennardJones(T epsilon = T(1.0), T sigma = T(1.0)) noexcept
        : m_epsilon(epsilon), m_sigma(sigma) {}

    T potential(T r) const noexcept {
        T sr = m_sigma / r;
        T sr6 = sr * sr * sr;
        sr6 = sr6 * sr6;
        T sr12 = sr6 * sr6;
        return T(4) * m_epsilon * (sr12 - sr6);
    }

    T force(T r) const noexcept {
        T sr = m_sigma / r;
        T sr6 = sr * sr * sr;
        sr6 = sr6 * sr6;
        T sr12 = sr6 * sr6;
        return T(24) * m_epsilon * (T(2) * sr12 - sr6) / r;
    }

    // SIMD batch evaluation (4 distances)
    static void batchPotential(const T* r, T* out, T epsilon, T sigma, std::size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
            for (std::size_t i = 0; i < count; ++i) {
                T sr = sigma / r[i];
                T sr6 = sr * sr * sr;
                sr6 = sr6 * sr6;
                T sr12 = sr6 * sr6;
                out[i] = T(4) * epsilon * (sr12 - sr6);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                T sr = sigma / r[i];
                T sr6 = sr * sr * sr;
                sr6 = sr6 * sr6;
                T sr12 = sr6 * sr6;
                out[i] = T(4) * epsilon * (sr12 - sr6);
            }
        }
    }

private:
    T m_epsilon;
    T m_sigma;
};

// ============================================================================
//  Molecular dynamics integrator (velocity Verlet) – microscopic time stepping
// ============================================================================
template<typename T>
class VelocityVerlet {
public:
    using vec3 = Math::Vector<T, 3>;

    VelocityVerlet(T dt) noexcept : m_dt(dt), m_halfDt(dt * T(0.5)) {}

    void setTimestep(T dt) noexcept { m_dt = dt; m_halfDt = dt * T(0.5); }

    // Step 1: update positions half‑step
    void step1(vec3& pos, const vec3& vel, const vec3& acc) const noexcept {
        pos = pos + vel * m_dt + acc * (m_halfDt * m_dt);
    }

    // Step 2: update velocities half‑step
    void step2(vec3& vel, const vec3& acc, const vec3& accNew) const noexcept {
        vel = vel + (acc + accNew) * m_halfDt;
    }

    // Full step: integrate positions and velocities given initial and new accelerations
    void integrate(vec3& pos, vec3& vel, const vec3& acc, const vec3& accNew) const noexcept {
        pos = pos + vel * m_dt + acc * (m_halfDt * m_dt);
        vel = vel + (acc + accNew) * m_halfDt;
    }

private:
    T m_dt;
    T m_halfDt;
};

// ============================================================================
//  Simulated annealing temperature scheduler (for dynamic environment)
// ============================================================================
template<typename T>
class AnnealingScheduler {
public:
    AnnealingScheduler(T startTemp, T endTemp, uint64_t steps) noexcept
        : m_start(startTemp), m_end(endTemp), m_maxSteps(steps), m_step(0) {}

    T currentTemperature() const noexcept {
        if (m_step >= m_maxSteps) return m_end;
        T t = static_cast<T>(m_step) / static_cast<T>(m_maxSteps);
        // Exponential cooling
        return m_start * std::pow(m_end / m_start, t);
    }

    void step() noexcept { ++m_step; }
    void reset() noexcept { m_step = 0; }
    bool isComplete() const noexcept { return m_step >= m_maxSteps; }

private:
    T m_start, m_end;
    uint64_t m_maxSteps, m_step;
};

} // namespace Extended
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_EXTENDED_MICROSCOPIC_UNITS_H_INCLUDED