// core/core_constants.h
#ifndef CORE_CONSTANTS_H
#define CORE_CONSTANTS_H

#include "../big_number.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

// ----------------------------------------------------------------------------
// CoreConstants: Provides immutable mathematical and physical constants
// as BigNumber tensors. These constants are initialized with high-precision
// hex-limb patterns to ensure deterministic behavior across all platforms.
// ----------------------------------------------------------------------------

class CoreConstants : public Object {
    GDCLASS(CoreConstants, Object);

    static CoreConstants* singleton;

protected:
    static void _bind_methods();

public:
    static CoreConstants* get_singleton() { return singleton; }
    
    CoreConstants();
    ~CoreConstants();

    // ------------------------------------------------------------------------
    // Mathematical Constants
    // ------------------------------------------------------------------------
    static uep::BigNumber PI();
    static uep::BigNumber TAU();
    static uep::BigNumber E();
    static uep::BigNumber LN2();
    static uep::BigNumber LN10();
    static uep::BigNumber SQRT2();
    static uep::BigNumber SQRT3();
    static uep::BigNumber EULER_GAMMA();
    static uep::BigNumber GOLDEN_RATIO();

    // ------------------------------------------------------------------------
    // Physical Constants (SI units, scaled appropriately)
    // ------------------------------------------------------------------------
    static uep::BigNumber SPEED_OF_LIGHT();        // c in m/s
    static uep::BigNumber PLANCK_CONSTANT();       // h in J·s
    static uep::BigNumber REDUCED_PLANCK();        // ħ = h/(2π)
    static uep::BigNumber GRAVITATIONAL_CONSTANT(); // G in m^3·kg^-1·s^-2
    static uep::BigNumber AVOGADRO_NUMBER();       // N_A
    static uep::BigNumber BOLTZMANN_CONSTANT();    // k_B in J/K
    static uep::BigNumber ELEMENTARY_CHARGE();     // e in C
    static uep::BigNumber VACUUM_PERMITTIVITY();   // ε0 in F/m
    static uep::BigNumber VACUUM_PERMEABILITY();   // μ0 in N/A^2
    static uep::BigNumber FINE_STRUCTURE();        // α ≈ 1/137
    static uep::BigNumber STEFAN_BOLTZMANN();      // σ in W·m^-2·K^-4
    static uep::BigNumber RYDBERG_CONSTANT();      // R∞ in m^-1
    static uep::BigNumber ASTRONOMICAL_UNIT();     // AU in meters
    static uep::BigNumber PARSEC();                // pc in meters
    static uep::BigNumber LIGHT_YEAR();            // ly in meters
    static uep::BigNumber SOLAR_MASS();            // M☉ in kg
    static uep::BigNumber EARTH_MASS();            // M🜨 in kg
    static uep::BigNumber EARTH_RADIUS();          // R🜨 in meters
    static uep::BigNumber STANDARD_GRAVITY();      // g0 in m/s^2

    // ------------------------------------------------------------------------
    // Tensor representation: return as xtensor if enabled
    // ------------------------------------------------------------------------
#ifdef UEP_USE_XTENSOR
    static xt::xarray<double> PI_TENSOR();
    static xt::xarray<double> SPEED_OF_LIGHT_TENSOR();
    // Additional tensor constants can be added.
#endif

private:
    // Initialize all constants once (called from constructor)
    void initialize_constants();
    
    // Storage for the constants (as BigNumber)
    struct ConstantsStorage {
        uep::BigNumber pi;
        uep::BigNumber tau;
        uep::BigNumber e;
        uep::BigNumber ln2;
        uep::BigNumber ln10;
        uep::BigNumber sqrt2;
        uep::BigNumber sqrt3;
        uep::BigNumber euler_gamma;
        uep::BigNumber golden_ratio;
        
        uep::BigNumber c;
        uep::BigNumber h;
        uep::BigNumber hbar;
        uep::BigNumber G;
        uep::BigNumber N_A;
        uep::BigNumber k_B;
        uep::BigNumber e_charge;
        uep::BigNumber epsilon0;
        uep::BigNumber mu0;
        uep::BigNumber alpha;
        uep::BigNumber sigma;
        uep::BigNumber R_inf;
        uep::BigNumber AU;
        uep::BigNumber pc;
        uep::BigNumber ly;
        uep::BigNumber M_sun;
        uep::BigNumber M_earth;
        uep::BigNumber R_earth;
        uep::BigNumber g0;
    };
    
    static ConstantsStorage constants;
    static bool initialized;
};

// ----------------------------------------------------------------------------
// Inline accessors for CoreConstants
// These are defined in the header for fast access without function call overhead.
// The actual constant values are initialized in core_constants.cpp using high-
// precision hex limb patterns.
// ----------------------------------------------------------------------------
inline uep::BigNumber CoreConstants::PI() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.pi;
}
inline uep::BigNumber CoreConstants::TAU() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.tau;
}
inline uep::BigNumber CoreConstants::E() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.e;
}
inline uep::BigNumber CoreConstants::LN2() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.ln2;
}
inline uep::BigNumber CoreConstants::LN10() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.ln10;
}
inline uep::BigNumber CoreConstants::SQRT2() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.sqrt2;
}
inline uep::BigNumber CoreConstants::SQRT3() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.sqrt3;
}
inline uep::BigNumber CoreConstants::EULER_GAMMA() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.euler_gamma;
}
inline uep::BigNumber CoreConstants::GOLDEN_RATIO() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.golden_ratio;
}

inline uep::BigNumber CoreConstants::SPEED_OF_LIGHT() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.c;
}
inline uep::BigNumber CoreConstants::PLANCK_CONSTANT() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.h;
}
inline uep::BigNumber CoreConstants::REDUCED_PLANCK() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.hbar;
}
inline uep::BigNumber CoreConstants::GRAVITATIONAL_CONSTANT() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.G;
}
inline uep::BigNumber CoreConstants::AVOGADRO_NUMBER() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.N_A;
}
inline uep::BigNumber CoreConstants::BOLTZMANN_CONSTANT() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.k_B;
}
inline uep::BigNumber CoreConstants::ELEMENTARY_CHARGE() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.e_charge;
}
inline uep::BigNumber CoreConstants::VACUUM_PERMITTIVITY() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.epsilon0;
}
inline uep::BigNumber CoreConstants::VACUUM_PERMEABILITY() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.mu0;
}
inline uep::BigNumber CoreConstants::FINE_STRUCTURE() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.alpha;
}
inline uep::BigNumber CoreConstants::STEFAN_BOLTZMANN() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.sigma;
}
inline uep::BigNumber CoreConstants::RYDBERG_CONSTANT() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.R_inf;
}
inline uep::BigNumber CoreConstants::ASTRONOMICAL_UNIT() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.AU;
}
inline uep::BigNumber CoreConstants::PARSEC() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.pc;
}
inline uep::BigNumber CoreConstants::LIGHT_YEAR() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.ly;
}
inline uep::BigNumber CoreConstants::SOLAR_MASS() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.M_sun;
}
inline uep::BigNumber CoreConstants::EARTH_MASS() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.M_earth;
}
inline uep::BigNumber CoreConstants::EARTH_RADIUS() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.R_earth;
}
inline uep::BigNumber CoreConstants::STANDARD_GRAVITY() {
    if (!initialized) { const_cast<CoreConstants*>(singleton)->initialize_constants(); }
    return constants.g0;
}

#ifdef UEP_USE_XTENSOR
inline xt::xarray<double> CoreConstants::PI_TENSOR() {
    return xt::xarray<double>({PI().to_double()});
}
inline xt::xarray<double> CoreConstants::SPEED_OF_LIGHT_TENSOR() {
    return xt::xarray<double>({SPEED_OF_LIGHT().to_double()});
}
#endif

// ----------------------------------------------------------------------------
// Global constant accessors (convenience functions)
// These are defined to mirror Godot's Math::PI style.
// ----------------------------------------------------------------------------
namespace uep {
    inline BigNumber pi() { return CoreConstants::PI(); }
    inline BigNumber tau() { return CoreConstants::TAU(); }
    inline BigNumber e() { return CoreConstants::E(); }
    inline BigNumber ln2() { return CoreConstants::LN2(); }
    inline BigNumber ln10() { return CoreConstants::LN10(); }
    inline BigNumber sqrt2() { return CoreConstants::SQRT2(); }
    inline BigNumber sqrt3() { return CoreConstants::SQRT3(); }
    inline BigNumber speed_of_light() { return CoreConstants::SPEED_OF_LIGHT(); }
    inline BigNumber planck_constant() { return CoreConstants::PLANCK_CONSTANT(); }
    inline BigNumber gravitational_constant() { return CoreConstants::GRAVITATIONAL_CONSTANT(); }
}

#endif // CORE_CONSTANTS_H
// Ending of File 11 of 15 (core/core_constants.h)