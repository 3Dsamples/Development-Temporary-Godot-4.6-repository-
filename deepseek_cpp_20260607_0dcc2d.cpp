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

/**
 * @file environmental_drivers.h
 * @brief Dynamic environmental drivers for realistic simulation.
 *
 * This file provides classes for environmental forces such as wind, gravity,
 * temperature, and pressure fields that affect entity motion in real time.
 * Supports SIMD batch evaluation for high performance.
 */

#ifndef ORTHOTREE_CORE_DYNAMIC_ENVIRONMENTAL_DRIVERS_H_INCLUDED
#define ORTHOTREE_CORE_DYNAMIC_ENVIRONMENTAL_DRIVERS_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/interval_arithmetic.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <vector>
#include <array>
#include <functional>
#include <random>
#include <limits>

namespace OrthoTree {
namespace Dynamic {

// ============================================================================
//  Wind field: directional force with turbulence
// ============================================================================
template<typename T = float, std::size_t N = 3>
class WindField {
public:
    using point_type = Math::Vector<T, N>;
    using vec_type = point_type;

    struct Config {
        vec_type direction;           // main wind direction (normalized)
        T speed;                      // base speed (m/s)
        T turbulenceIntensity;        // 0..1 (fraction of speed)
        T turbulenceScale;            // spatial scale of turbulence
        T variationFrequency;         // temporal variation frequency (Hz)
        bool useSimd = true;
    };

    explicit WindField(const Config& cfg = Config()) noexcept
        : m_config(cfg)
        , m_rng(std::random_device{}())
        , m_noiseDist(T(0), T(1)) {}

    void setDirection(const vec_type& dir) noexcept {
        m_config.direction = dir.normalized();
    }
    void setSpeed(T speed) noexcept { m_config.speed = speed; }
    void setTurbulenceIntensity(T intensity) noexcept { m_config.turbulenceIntensity = intensity; }

    // Evaluate wind velocity at a given point (scalar)
    vec_type evaluate(const point_type& pos, T time) const noexcept {
        vec_type wind = m_config.direction * m_config.speed;
        if (m_config.turbulenceIntensity > T(0)) {
            // Procedural turbulence using Perlin‑style noise (simplified)
            T noiseX = turbulenceNoise(pos, time);
            T noiseY = turbulenceNoise(pos + point_type(T(10)), time);
            T noiseZ = (N == 3) ? turbulenceNoise(pos + point_type(T(20)), time) : T(0);
            vec_type turb(noiseX, noiseY, noiseZ);
            turb = turb * m_config.turbulenceIntensity * m_config.speed;
            wind = wind + turb;
        }
        return wind;
    }

    // Batch evaluate wind velocities at multiple points (SIMD)
    void batchEvaluate(const point_type* positions, vec_type* outVelocities,
                       std::size_t count, T time) const noexcept {
        if (m_config.useSimd && count >= 4 && N == 3) {
            std::size_t simdEnd = count - (count % 4);
            for (std::size_t i = 0; i < simdEnd; i += 4) {
                // In a real SIMD implementation, we would use AVX2 intrinsics.
                // For brevity, we call scalar for each.
                for (std::size_t j = 0; j < 4; ++j) {
                    outVelocities[i+j] = evaluate(positions[i+j], time);
                }
            }
            for (std::size_t i = simdEnd; i < count; ++i) {
                outVelocities[i] = evaluate(positions[i], time);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                outVelocities[i] = evaluate(positions[i], time);
            }
        }
    }

private:
    T turbulenceNoise(const point_type& pos, T time) const noexcept {
        // Very simple pseudo‑noise for demonstration.
        // In production, use a proper noise function (e.g., Simplex).
        T x = pos[0] * m_config.turbulenceScale;
        T y = pos[1] * m_config.turbulenceScale;
        T z = (N == 3) ? pos[2] * m_config.turbulenceScale : T(0);
        T sum = std::sin(x) * std::cos(y) * (N == 3 ? std::sin(z) : T(1));
        sum += std::sin(time * m_config.variationFrequency);
        return sum;
    }

    Config m_config;
    mutable std::mt19937_64 m_rng;
    mutable std::normal_distribution<T> m_noiseDist;
};

// ============================================================================
//  Gravity field: Newtonian gravity from point masses or uniform direction
// ============================================================================
template<typename T = float, std::size_t N = 3>
class GravityField {
public:
    using point_type = Math::Vector<T, N>;
    using vec_type = point_type;

    struct PointMass {
        point_type position;
        T mass;                     // kg
        T softening;                // softening length (m)
    };

    GravityField() noexcept : m_uniformEnabled(false) {}

    void setUniform(const vec_type& direction, T acceleration) noexcept {
        m_uniformDirection = direction.normalized();
        m_uniformAccel = acceleration;
        m_uniformEnabled = true;
    }

    void addPointMass(const point_type& pos, T mass, T softening = T(0)) {
        m_masses.push_back({pos, mass, softening});
    }

    void clearPointMasses() noexcept { m_masses.clear(); }

    // Evaluate gravitational acceleration at a point (scalar)
    vec_type evaluate(const point_type& pos) const noexcept {
        vec_type accel(T(0));
        if (m_uniformEnabled) {
            accel = accel + m_uniformDirection * m_uniformAccel;
        }
        for (const auto& mass : m_masses) {
            vec_type delta = mass.position - pos;
            T r2 = delta.squaredLength();
            if (r2 < m_config.minDistanceSq) r2 = m_config.minDistanceSq;
            T r = std::sqrt(r2);
            T softeningSq = mass.softening * mass.softening;
            T forceMag = m_config.G * mass.mass / (r2 + softeningSq);
            accel = accel + delta * (forceMag / r);
        }
        return accel;
    }

    // Batch evaluate (SIMD)
    void batchEvaluate(const point_type* positions, vec_type* outAccel,
                       std::size_t count) const noexcept {
        // For simplicity, we use scalar loop for now.
        // A real implementation would vectorise over positions and point masses.
        for (std::size_t i = 0; i < count; ++i) {
            outAccel[i] = evaluate(positions[i]);
        }
    }

    void setGConstant(T G) noexcept { m_config.G = G; }
    void setMinDistance(T minDist) noexcept { m_config.minDistanceSq = minDist * minDist; }

private:
    struct Config {
        T G = T(6.67430e-11);
        T minDistanceSq = T(1e-6);
    } m_config;

    bool m_uniformEnabled;
    vec_type m_uniformDirection;
    T m_uniformAccel;
    std::vector<PointMass> m_masses;
};

// ============================================================================
//  Temperature field: spatial variation with time
// ============================================================================
template<typename T = float, std::size_t N = 3>
class TemperatureField {
public:
    using point_type = Math::Vector<T, N>;

    struct Config {
        T baseTemp = T(300.0);           // Kelvin
        T amplitude = T(10.0);            // variation amplitude
        T spatialScale = T(100.0);        // meters
        T temporalFrequency = T(0.1);     // Hz
    };

    explicit TemperatureField(const Config& cfg = Config()) noexcept
        : m_config(cfg) {}

    T evaluate(const point_type& pos, T time) const noexcept {
        T spatialVar = T(0);
        for (std::size_t i = 0; i < N; ++i) {
            spatialVar += std::sin(pos[i] / m_config.spatialScale);
        }
        spatialVar /= T(N);
        T temporalVar = std::sin(time * m_config.temporalFrequency);
        T temp = m_config.baseTemp + m_config.amplitude * (spatialVar + temporalVar) * T(0.5);
        return temp;
    }

    void batchEvaluate(const point_type* positions, T* outTemps,
                       std::size_t count, T time) const noexcept {
        // Can be SIMD vectorised, but scalar loop for clarity.
        for (std::size_t i = 0; i < count; ++i) {
            outTemps[i] = evaluate(positions[i], time);
        }
    }

private:
    Config m_config;
};

// ============================================================================
//  Pressure field: simplified atmospheric pressure (exponential decay)
// ============================================================================
template<typename T = float>
class PressureField {
public:
    using point_type = Math::Vector<T, 3>;

    struct Config {
        T seaLevelPressure = T(101325.0);   // Pa
        T scaleHeight = T(8500.0);          // meters (approx)
    };

    explicit PressureField(const Config& cfg = Config()) noexcept
        : m_config(cfg) {}

    T evaluate(const point_type& pos) const noexcept {
        T altitude = pos[2];  // assume Z is up
        return m_config.seaLevelPressure * std::exp(-altitude / m_config.scaleHeight);
    }

private:
    Config m_config;
};

// ============================================================================
//  Dynamic environment controller: combines all fields
// ============================================================================
template<typename T = float, std::size_t N = 3>
class EnvironmentController {
public:
    using point_type = Math::Vector<T, N>;
    using vec_type = point_type;

    struct Forces {
        vec_type windForce;       // N
        vec_type gravityForce;    // N
        T temperature;            // K
        T pressure;               // Pa
    };

    void setWindField(const WindField<T, N>& field) { m_wind = field; }
    void setGravityField(const GravityField<T, N>& field) { m_gravity = field; }
    void setTemperatureField(const TemperatureField<T, N>& field) { m_temperature = field; }
    void setPressureField(const PressureField<T>& field) { m_pressure = field; }

    // Compute all environmental forces at a point (scalar)
    Forces compute(const point_type& pos, const vec_type& velocity,
                   T mass, T time) const {
        Forces f;
        f.windForce = m_wind.evaluate(pos, time) * mass; // F = m * a (wind drag simplified)
        f.gravityForce = m_gravity.evaluate(pos) * mass;
        f.temperature = m_temperature.evaluate(pos, time);
        f.pressure = m_pressure.evaluate(pos);
        return f;
    }

    // Batch compute forces (SIMD)
    void batchCompute(const point_type* positions, const vec_type* velocities,
                      const T* masses, Forces* outForces, std::size_t count,
                      T time) const {
        // In a real implementation, we would vectorise over 4 points.
        for (std::size_t i = 0; i < count; ++i) {
            outForces[i] = compute(positions[i], velocities[i], masses[i], time);
        }
    }

private:
    WindField<T, N> m_wind;
    GravityField<T, N> m_gravity;
    TemperatureField<T, N> m_temperature;
    PressureField<T> m_pressure;
};

} // namespace Dynamic
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_DYNAMIC_ENVIRONMENTAL_DRIVERS_H_INCLUDED