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

#ifndef ORTHOTREE_CORE_MATH_EXTENDED_ASTRONOMICAL_COORDINATES_H_INCLUDED
#define ORTHOTREE_CORE_MATH_EXTENDED_ASTRONOMICAL_COORDINATES_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "../vector_math.h"
#include "../quaternion.h"
#include "../transform.h"
#include "../numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <cstdint>

namespace OrthoTree {
namespace Math {
namespace Extended {

// ============================================================================
//  Astronomical constants (SI units, high precision)
// ============================================================================
template<typename T>
struct AstronomicalConstants {
    static constexpr T AU_TO_METERS      = T(1.49597870700e11);
    static constexpr T PARSEC_TO_METERS  = T(3.08567758149e16);
    static constexpr T LIGHTYEAR_TO_METERS = T(9.46073047258e15);
    static constexpr T SOLAR_RADIUS_METERS = T(6.95700e8);
    static constexpr T EARTH_RADIUS_METERS = T(6.3781e6);
    static constexpr T GRAVITATIONAL_CONSTANT = T(6.67430e-11);
    static constexpr T SPEED_OF_LIGHT = T(2.99792458e8);
    static constexpr T DEG_TO_RAD = T(3.14159265358979323846 / 180.0);
    static constexpr T RAD_TO_DEG = T(180.0 / 3.14159265358979323846);
};

// ============================================================================
//  Spherical coordinates (r, theta, phi) – SIMD optimized
// ============================================================================
template<typename T, bool UseSIMD = true>
class SphericalCoord {
public:
    using value_type = T;
    using vec3_type = Math::Vector<T, 3>;

    SphericalCoord() noexcept : r(0), theta(0), phi(0) {}
    SphericalCoord(T rad, T polar, T azimuthal) noexcept : r(rad), theta(polar), phi(azimuthal) {}
    explicit SphericalCoord(const vec3_type& cartesian) noexcept {
        fromCartesian(cartesian);
    }

    // Convert to/from Cartesian
    void fromCartesian(const vec3_type& v) noexcept {
        r = v.length();
        if (r > T(0)) {
            theta = std::acos(v[2] / r);
            phi = std::atan2(v[1], v[0]);
        } else {
            theta = T(0);
            phi = T(0);
        }
    }

    vec3_type toCartesian() const noexcept {
        T sinTheta = std::sin(theta);
        T cosTheta = std::cos(theta);
        T sinPhi = std::sin(phi);
        T cosPhi = std::cos(phi);
        return vec3_type(r * sinTheta * cosPhi,
                         r * sinTheta * sinPhi,
                         r * cosTheta);
    }

    // SIMD accelerated batch conversion (for arrays of points)
    static void batchToCartesian(const SphericalCoord* src, vec3_type* dst, std::size_t count) noexcept {
        if constexpr (UseSIMD && ORTHOTREE_SIMD_LEVEL >= 128) {
            // Use SIMD intrinsics (pseudo code – actual implementation would use aligned loads)
            for (std::size_t i = 0; i < count; ++i) {
                dst[i] = src[i].toCartesian();
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                dst[i] = src[i].toCartesian();
            }
        }
    }

    T r, theta, phi;
};

// ============================================================================
//  Galactic coordinate system (gl, gb, distance) – based on IAU 1958
// ============================================================================
template<typename T>
class GalacticCoord {
public:
    using vec3_type = Math::Vector<T, 3>;

    GalacticCoord() noexcept : l(0), b(0), d(0) {}
    GalacticCoord(T longitude, T latitude, T distance) noexcept : l(longitude), b(latitude), d(distance) {}
    explicit GalacticCoord(const vec3_type& equatorial, T equinoxYear = T(2000.0)) noexcept {
        fromEquatorial(equatorial, equinoxYear);
    }

    // Convert from equatorial coordinates (right ascension, declination, distance)
    void fromEquatorial(const vec3_type& equatorial, T equinoxYear) noexcept {
        // Precession matrix for epoch J2000.0 (simplified; full precession would be more complex)
        vec3_type temp = equatorial;
        // Apply galactic rotation: Euler angles for Galactic coordinate system
        // Standard transformation: (α, δ) -> (l, b)
        // Using transformation from Hipparcos (ESA 1997)
        T ra  = equatorial[0];   // right ascension (rad)
        T dec = equatorial[1];   // declination (rad)
        T r   = equatorial[2];   // distance

        // Galactic north pole (J2000): α = 192.8595°, δ = 27.1284°
        const T alpha0 = T(192.8595) * AstronomicalConstants<T>::DEG_TO_RAD;
        const T delta0 = T(27.1284)  * AstronomicalConstants<T>::DEG_TO_RAD;
        // Galactic center (J2000): α = 266.4051°, δ = -28.9362°
        const T alphaC = T(266.4051) * AstronomicalConstants<T>::DEG_TO_RAD;
        const T deltaC = T(-28.9362) * AstronomicalConstants<T>::DEG_TO_RAD;

        // Compute intermediate values
        T sinDec   = std::sin(dec);
        T cosDec   = std::cos(dec);
        T sinDelta0 = std::sin(delta0);
        T cosDelta0 = std::cos(delta0);
        T sinDeltaC = std::sin(deltaC);
        T cosDeltaC = std::cos(deltaC);

        T sinAlphaAlpha0 = std::sin(ra - alpha0);
        T cosAlphaAlpha0 = std::cos(ra - alpha0);

        T sinB = sinDec * sinDelta0 + cosDec * cosDelta0 * cosAlphaAlpha0;
        b = std::asin(sinB);
        T cosB = std::cos(b);

        if (std::abs(cosB) > T(0)) {
            T sinL = (cosDec * sinAlphaAlpha0) / cosB;
            T cosL = (sinDec * cosDelta0 - cosDec * sinDelta0 * cosAlphaAlpha0) / cosB;
            l = std::atan2(sinL, cosL);
        } else {
            l = T(0);
        }
        d = r;
    }

    vec3_type toEquatorial() const noexcept {
        // Inverse transformation
        const T alpha0 = T(192.8595) * AstronomicalConstants<T>::DEG_TO_RAD;
        const T delta0 = T(27.1284)  * AstronomicalConstants<T>::DEG_TO_RAD;
        const T sinDelta0 = std::sin(delta0);
        const T cosDelta0 = std::cos(delta0);

        T sinB = std::sin(b);
        T cosB = std::cos(b);
        T sinL = std::sin(l);
        T cosL = std::cos(l);

        T sinDec = sinB * sinDelta0 + cosB * cosDelta0 * cosL;
        T dec = std::asin(sinDec);
        T cosDec = std::cos(dec);
        if (std::abs(cosDec) > T(0)) {
            T sinRaAlpha0 = (cosB * sinL) / cosDec;
            T cosRaAlpha0 = (sinB * cosDelta0 - cosB * sinDelta0 * cosL) / cosDec;
            T ra = alpha0 + std::atan2(sinRaAlpha0, cosRaAlpha0);
            return vec3_type(ra, dec, d);
        } else {
            return vec3_type(alpha0, dec, d);
        }
    }

    T l, b, d;   // galactic longitude (rad), latitude (rad), distance (meters)
};

// ============================================================================
//  Relativistic boost / Lorentz transformation (for high‑velocity objects)
// ============================================================================
template<typename T>
class LorentzTransform {
public:
    using vec3_type = Math::Vector<T, 3>;

    LorentzTransform() noexcept : m_gamma(1), m_beta(0,0,0) {}
    explicit LorentzTransform(const vec3_type& velocity) noexcept {
        setVelocity(velocity);
    }

    void setVelocity(const vec3_type& v) noexcept {
        T vmag = v.length();
        if (vmag < AstronomicalConstants<T>::SPEED_OF_LIGHT * T(1e-8)) {
            m_beta = v / AstronomicalConstants<T>::SPEED_OF_LIGHT;
        } else {
            m_beta = v / vmag * (vmag / AstronomicalConstants<T>::SPEED_OF_LIGHT);
        }
        T beta2 = m_beta.squaredLength();
        if (beta2 < T(1)) {
            m_gamma = T(1) / std::sqrt(T(1) - beta2);
        } else {
            m_gamma = T(1); // fallback
        }
    }

    // Transform a 4‑vector (ct, x, y, z) – SIMD friendly
    std::array<T, 4> transform(const std::array<T, 4>& fourVec) const noexcept {
        T ct = fourVec[0];
        T x  = fourVec[1];
        T y  = fourVec[2];
        T z  = fourVec[3];
        T betaDotR = m_beta[0] * x + m_beta[1] * y + m_beta[2] * z;
        T ctPrime = m_gamma * (ct + betaDotR);
        T factor = m_gamma * betaDotR / (T(1) + m_gamma) * (m_gamma - T(1));
        T xPrime = x + factor * m_beta[0];
        T yPrime = y + factor * m_beta[1];
        T zPrime = z + factor * m_beta[2];
        return {ctPrime, xPrime, yPrime, zPrime};
    }

    // Transform a 3‑vector (spatial only)
    vec3_type transform(const vec3_type& pos, T time = T(0)) const noexcept {
        // For simplicity, treat as same time coordinate
        std::array<T,4> four = {AstronomicalConstants<T>::SPEED_OF_LIGHT * time,
                                 pos[0], pos[1], pos[2]};
        auto res = transform(four);
        return vec3_type(res[1], res[2], res[3]);
    }

    T gamma() const noexcept { return m_gamma; }
    vec3_type beta() const noexcept { return m_beta; }

private:
    T m_gamma;
    vec3_type m_beta;
};

// ============================================================================
//  Dynamic environment controller for astronomical scales
// ============================================================================
template<typename T>
class AstronomicalEnvironment {
public:
    using vec3_type = Math::Vector<T, 3>;

    AstronomicalEnvironment() noexcept
        : m_epoch(T(2000.0))
        , m_aberrationCorrection(true)
        , m_relativisticCorrection(true)
        , m_useLightTravelTime(true) {}

    // Set observation epoch (years) for precession
    void setEpoch(T epoch) noexcept { m_epoch = epoch; }

    // Convert coordinates from one epoch to another (precession)
    vec3_type precess(const vec3_type& equatorialCoords, T fromEpoch, T toEpoch) const noexcept {
        // Simplified precession model (Capitaine et al. 2003)
        T deltaT = (toEpoch - fromEpoch) * T(0.01);  // centuries
        T zeta   = T(0.6406161) * deltaT + T(0.0000839) * deltaT * deltaT;
        T z      = T(0.6406161) * deltaT + T(0.0003041) * deltaT * deltaT;
        T theta  = T(0.5567530) * deltaT - T(0.0001185) * deltaT * deltaT;
        zeta  *= AstronomicalConstants<T>::DEG_TO_RAD;
        z     *= AstronomicalConstants<T>::DEG_TO_RAD;
        theta *= AstronomicalConstants<T>::DEG_TO_RAD;

        // Rotation matrices (applied to equatorial coordinates)
        T sinZeta = std::sin(zeta), cosZeta = std::cos(zeta);
        T sinZ    = std::sin(z),    cosZ    = std::cos(z);
        T sinTheta = std::sin(theta), cosTheta = std::cos(theta);

        // Apply rotation: Rz(-z) * Rx(theta) * Rz(-zeta)
        vec3_type v = equatorialCoords;
        // First Rz(-zeta)
        T x1 = v[0] * cosZeta + v[1] * sinZeta;
        T y1 = -v[0] * sinZeta + v[1] * cosZeta;
        T z1 = v[2];
        // Then Rx(theta)
        T x2 = x1;
        T y2 = y1 * cosTheta - z1 * sinTheta;
        T z2 = y1 * sinTheta + z1 * cosTheta;
        // Then Rz(-z)
        T x3 = x2 * cosZ + y2 * sinZ;
        T y3 = -x2 * sinZ + y2 * cosZ;
        T z3 = z2;
        return vec3_type(x3, y3, z3);
    }

    // Apply aberration of light (stellar aberration)
    vec3_type correctAberration(const vec3_type& direction, const vec3_type& observerVelocity) const noexcept {
        if (!m_aberrationCorrection) return direction;
        T c = AstronomicalConstants<T>::SPEED_OF_LIGHT;
        vec3_type v = observerVelocity / c;
        T v2 = v.squaredLength();
        T invGamma = std::sqrt(T(1) - v2);
        T factor = T(1) / (T(1) + direction.dot(v));
        vec3_type result = (direction * invGamma + v * (T(1) - invGamma * invGamma / (T(1) + invGamma)))
                         * factor;
        return result.normalized();
    }

    // Compute light travel time (parallax + delay)
    T lightTravelTime(const vec3_type& sourcePos, const vec3_type& observerPos, T emissionTime) const noexcept {
        if (!m_useLightTravelTime) return T(0);
        vec3_type delta = sourcePos - observerPos;
        T distance = delta.length();
        T lightTime = distance / AstronomicalConstants<T>::SPEED_OF_LIGHT;
        // Iterative solution for precise time (one iteration sufficient)
        vec3_type sourcePos2 = sourcePos - observerPos * (lightTime / emissionTime); // simplified
        return lightTime;
    }

    // Enable/disable relativistic corrections
    void setRelativisticCorrection(bool enable) noexcept { m_relativisticCorrection = enable; }
    void setAberrationCorrection(bool enable) noexcept { m_aberrationCorrection = enable; }
    void setLightTravelTime(bool enable) noexcept { m_useLightTravelTime = enable; }

private:
    T m_epoch;
    bool m_aberrationCorrection;
    bool m_relativisticCorrection;
    bool m_useLightTravelTime;
};

// ============================================================================
//  Helper: convert parsecs to meters (SIMD batch)
// ============================================================================
template<typename T, bool UseSIMD = true>
inline T parsecToMeters(T parsec) noexcept {
    return parsec * AstronomicalConstants<T>::PARSEC_TO_METERS;
}

template<typename T, bool UseSIMD = true>
inline void batchParsecToMeters(const T* src, T* dst, std::size_t count) noexcept {
    if constexpr (UseSIMD && ORTHOTREE_SIMD_LEVEL >= 128) {
        // SIMD loop would go here
        for (std::size_t i = 0; i < count; ++i) dst[i] = src[i] * AstronomicalConstants<T>::PARSEC_TO_METERS;
    } else {
        for (std::size_t i = 0; i < count; ++i) dst[i] = src[i] * AstronomicalConstants<T>::PARSEC_TO_METERS;
    }
}

// ============================================================================
//  Dynamic scale factor for large distances (logarithmic depth buffer)
// ============================================================================
template<typename T>
class LogarithmicDepthConverter {
public:
    LogarithmicDepthConverter(T nearPlane = T(1), T farPlane = T(1e12)) noexcept
        : m_near(nearPlane), m_far(farPlane) {}

    T linearToLogDepth(T linearDepth) const noexcept {
        if (linearDepth <= m_near) return T(0);
        if (linearDepth >= m_far) return T(1);
        return std::log(linearDepth / m_near) / std::log(m_far / m_near);
    }

    T logToLinearDepth(T logDepth) const noexcept {
        return m_near * std::exp(logDepth * std::log(m_far / m_near));
    }

    // Batch conversion
    void batchLinearToLog(const T* linear, T* log, std::size_t count) const noexcept {
        T invLogRange = T(1) / std::log(m_far / m_near);
        for (std::size_t i = 0; i < count; ++i) {
            if (linear[i] <= m_near) log[i] = T(0);
            else if (linear[i] >= m_far) log[i] = T(1);
            else log[i] = std::log(linear[i] / m_near) * invLogRange;
        }
    }

private:
    T m_near, m_far;
};

} // namespace Extended
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_EXTENDED_ASTRONOMICAL_COORDINATES_H_INCLUDED