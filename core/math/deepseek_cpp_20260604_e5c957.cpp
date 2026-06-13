// system name : onetbb-warp
// File 0001 : core/math/constants.h
// Description : Fundamental mathematical and physical constants.

#ifndef __TBB_WARP_CORE_MATH_CONSTANTS_H
#define __TBB_WARP_CORE_MATH_CONSTANTS_H

#include <cstdint>
#include <limits>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Mathematical constants
// ============================================================

constexpr double PI_D      = 3.14159265358979323846;
constexpr double TAU_D     = 6.28318530717958647692;
constexpr double HALF_PI_D = 1.57079632679489661923;
constexpr double SQRT2_D   = 1.41421356237309504880;
constexpr double SQRT3_D   = 1.73205080756887729352;
constexpr double E_D       = 2.71828182845904523536;
constexpr double DEG2RAD_D = PI_D / 180.0;
constexpr double RAD2DEG_D = 180.0 / PI_D;

constexpr float PI_F      = static_cast<float>(PI_D);
constexpr float TAU_F     = static_cast<float>(TAU_D);
constexpr float HALF_PI_F = static_cast<float>(HALF_PI_D);
constexpr float SQRT2_F   = static_cast<float>(SQRT2_D);
constexpr float SQRT3_F   = static_cast<float>(SQRT3_D);
constexpr float E_F       = static_cast<float>(E_D);
constexpr float DEG2RAD_F = static_cast<float>(DEG2RAD_D);
constexpr float RAD2DEG_F = static_cast<float>(RAD2DEG_D);

// ============================================================
// Floating‑point limits
// ============================================================

constexpr float FLOAT_EPSILON     = 1.192092896e-07f;
constexpr double DOUBLE_EPSILON   = 2.2204460492503131e-16;
constexpr float FLOAT_INF         = std::numeric_limits<float>::infinity();
constexpr float FLOAT_NAN         = std::numeric_limits<float>::quiet_NaN();

// ============================================================
// Physical constants (SI)
// ============================================================

constexpr double GRAVITATIONAL_CONSTANT = 6.67430e-11;       // m^3 kg^-1 s^-2
constexpr double SPEED_OF_LIGHT         = 2.99792458e8;      // m/s
constexpr double PLANCK_CONSTANT        = 6.62607015e-34;    // J·s
constexpr double BOLTZMANN_CONSTANT     = 1.380649e-23;      // J/K
constexpr double AVOGADRO_NUMBER        = 6.02214076e23;     // mol^-1
constexpr double STANDARD_GRAVITY       = 9.80665;           // m/s^2
constexpr double EARTH_RADIUS_MEAN      = 6371000.0;         // m
constexpr double SOLAR_CONSTANT_IRRAD   = 1361.0;            // W/m^2
constexpr double STEFAN_BOLTZMANN       = 5.670367e-8;       // W m^-2 K^-4
constexpr double GAS_CONSTANT           = 8.314462618;        // J mol^-1 K^-1

// ============================================================
// Unit conversions
// ============================================================

constexpr double METERS_PER_AU       = 149597870700.0;
constexpr double SECONDS_PER_DAY     = 86400.0;
constexpr double SECONDS_PER_YEAR    = 31557600.0;
constexpr float  INCH_TO_METER       = 0.0254f;
constexpr float  FOOT_TO_METER       = 0.3048f;
constexpr float  MILE_TO_METER       = 1609.344f;
constexpr float  KILOGRAM_TO_POUND   = 2.20462262185f;
constexpr float  POUND_TO_KILOGRAM   = 0.45359237f;

// ============================================================
// Compile‑time helper to compute derived constants
// ============================================================

template<typename T>
constexpr T sqrt2() noexcept { return static_cast<T>(SQRT2_D); }
template<typename T>
constexpr T pi() noexcept { return static_cast<T>(PI_D); }

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_CONSTANTS_H