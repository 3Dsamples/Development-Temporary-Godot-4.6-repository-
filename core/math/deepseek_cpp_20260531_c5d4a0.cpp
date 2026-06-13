//File 0048 : core/math/math_constants.h
//High‑precision mathematical and physical constants (CODATA 2018), float/double versions, unit conversion factors, and common angle tables for real‑time and scientific simulation.
#ifndef CORE_MATH_MATH_CONSTANTS_H
#define CORE_MATH_MATH_CONSTANTS_H

#include <cmath>
#include <cstdint>

namespace SimulationMath {
namespace constants {

// -----------------------------------------------------------------------------
// 1. Pure mathematical constants (double and float)
// -----------------------------------------------------------------------------
inline constexpr double PI         = 3.14159265358979323846;
inline constexpr double TWO_PI     = 6.28318530717958647692;
inline constexpr double HALF_PI    = 1.57079632679489661923;
inline constexpr double QUARTER_PI = 0.78539816339744830962;
inline constexpr double INV_PI     = 0.31830988618379067154;
inline constexpr double SQRT2      = 1.41421356237309504880;
inline constexpr double SQRT3      = 1.73205080756887729352;
inline constexpr double INV_SQRT2  = 0.70710678118654752440;
inline constexpr double E          = 2.71828182845904523536;
inline constexpr double LN2        = 0.69314718055994530942;
inline constexpr double LN10       = 2.30258509299404568402;
inline constexpr double DEG_TO_RAD = PI / 180.0;
inline constexpr double RAD_TO_DEG = 180.0 / PI;
inline constexpr double GOLDEN_RATIO = 1.61803398874989484820;

inline constexpr float PIf         = 3.14159265358979323846f;
inline constexpr float TWO_PIf     = 6.28318530717958647692f;
inline constexpr float HALF_PIf    = 1.57079632679489661923f;
inline constexpr float QUARTER_PIf = 0.78539816339744830962f;
inline constexpr float INV_PIf     = 0.31830988618379067154f;
inline constexpr float SQRT2f      = 1.41421356237309504880f;
inline constexpr float SQRT3f      = 1.73205080756887729352f;
inline constexpr float INV_SQRT2f  = 0.70710678118654752440f;
inline constexpr float Ef          = 2.71828182845904523536f;
inline constexpr float LN2f        = 0.69314718055994530942f;
inline constexpr float LN10f       = 2.30258509299404568402f;
inline constexpr float DEG_TO_RADf = PIf / 180.0f;
inline constexpr float RAD_TO_DEGf = 180.0f / PIf;
inline constexpr float GOLDEN_RATIOf = 1.61803398874989484820f;

// -----------------------------------------------------------------------------
// 2. Physical constants (SI, CODATA 2018)
// -----------------------------------------------------------------------------
inline constexpr double SPEED_OF_LIGHT    = 299792458.0;                     // m/s (exact)
inline constexpr double GRAVITATIONAL_G   = 6.67430e-11;                     // m^3/(kg s^2)
inline constexpr double PLANCK_H          = 6.62607015e-34;                  // J s (exact)
inline constexpr double HBAR              = PLANCK_H / (2.0 * PI);           // J s
inline constexpr double BOLTZMANN_K       = 1.380649e-23;                    // J/K (exact)
inline constexpr double AVOGADRO_NA       = 6.02214076e23;                   // mol^-1 (exact)
inline constexpr double STEFAN_BOLTZMANN  = 5.670374419e-8;                 // W/(m^2 K^4)
inline constexpr double WIEN_DISPLACEMENT = 2.897771955e-3;                  // m K
inline constexpr double FINE_STRUCTURE    = 7.2973525693e-3;                 // dimensionless
inline constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;                 // C (exact)
inline constexpr double ELECTRON_MASS     = 9.1093837015e-31;                // kg
inline constexpr double PROTON_MASS       = 1.67262192369e-27;               // kg
inline constexpr double NEUTRON_MASS      = 1.67492749804e-27;               // kg
inline constexpr double VACUUM_PERMITTIVITY = 8.8541878128e-12;              // F/m
inline constexpr double VACUUM_PERMEABILITY = 1.25663706212e-6;              // H/m
inline constexpr double COULOMB_CONSTANT    = 8.9875517873681764e9;          // N m^2 / C^2
inline constexpr double ATOMIC_MASS_UNIT    = 1.66053906660e-27;             // kg
inline constexpr double STANDARD_GRAVITY    = 9.80665;                        // m/s^2

inline constexpr float SPEED_OF_LIGHTf    = 299792458.0f;
inline constexpr float GRAVITATIONAL_Gf   = 6.67430e-11f;
inline constexpr float PLANCK_Hf          = 6.62607015e-34f;
inline constexpr float HBARf              = PLANCK_Hf / (2.0f * PIf);
inline constexpr float BOLTZMANN_Kf       = 1.380649e-23f;
inline constexpr float AVOGADRO_NAf       = 6.02214076e23f;
inline constexpr float STEFAN_BOLTZMANNf  = 5.670374419e-8f;
inline constexpr float ELEMENTARY_CHARGEf = 1.602176634e-19f;
inline constexpr float COULOMB_CONSTANTf  = 8.9875517873681764e9f;

// -----------------------------------------------------------------------------
// 3. Angle tables (precomputed sine/cosine for fast lookup, e.g., 1024 samples)
// -----------------------------------------------------------------------------
namespace detail {
    template <size_t N>
    struct TrigTable {
        float sin[N];
        float cos[N];
        constexpr TrigTable() : sin{}, cos{} {
            for (size_t i = 0; i < N; ++i) {
                double angle = (double(i) + 0.5) * 2.0 * PI / double(N);
                sin[i] = static_cast<float>(std::sin(angle));
                cos[i] = static_cast<float>(std::cos(angle));
            }
        }
    };
    inline constexpr size_t TRIG_TABLE_SIZE = 1024;
    inline constexpr TrigTable<TRIG_TABLE_SIZE> trigTable{};
}

inline float fast_sin_table(float angle) noexcept {
    angle = std::fmod(angle, 2.0f * PIf);
    if (angle < 0.0f) angle += 2.0f * PIf;
    float idx = angle * (detail::TRIG_TABLE_SIZE / (2.0f * PIf));
    size_t i = static_cast<size_t>(idx) % detail::TRIG_TABLE_SIZE;
    return detail::trigTable.sin[i];
}
inline float fast_cos_table(float angle) noexcept {
    angle = std::fmod(angle, 2.0f * PIf);
    if (angle < 0.0f) angle += 2.0f * PIf;
    float idx = angle * (detail::TRIG_TABLE_SIZE / (2.0f * PIf));
    size_t i = static_cast<size_t>(idx) % detail::TRIG_TABLE_SIZE;
    return detail::trigTable.cos[i];
}

} // namespace constants
} // namespace SimulationMath

#endif // CORE_MATH_MATH_CONSTANTS_H