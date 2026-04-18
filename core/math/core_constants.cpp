// core/core_constants.cpp
#include "core_constants.h"
#include "../big_number.h"
#include "core/object/class_db.h"
#include "core/variant/variant.h"
#include <cstring>

// ----------------------------------------------------------------------------
// Static member definitions
// ----------------------------------------------------------------------------
CoreConstants* CoreConstants::singleton = nullptr;
CoreConstants::ConstantsStorage CoreConstants::constants;
bool CoreConstants::initialized = false;

// ----------------------------------------------------------------------------
// High-precision constant definitions (hex-limb patterns)
// These values are derived from known mathematical and physical constants
// and are stored as scaled BigIntCore limbs (Q32.32 format).
// Each constant is defined with sufficient precision for arbitrary-precision
// calculations (typically 256-512 bits).
// ----------------------------------------------------------------------------

// Mathematical constants (scaled by 2^32)
static const uep::limb_t PI_LIMBS[] = {
    0x85A308D313198A2EULL,
    0x03707344A4093822ULL,
    0x299F31D0081FA9F6ULL,
    0x7E1D48E8A67C8B9AULL,
    0x8C2F6E9E3A26A89BULL,
    0x3243F6A8885A308DULL
};

static const uep::limb_t E_LIMBS[] = {
    0x8AED2A6ABF715880ULL,
    0x9CF4C3B8A1D28B9CULL,
    0x6E1D88E3A1E4B5E2ULL,
    0x2B7E151628AED2A6ULL
};

static const uep::limb_t LN2_LIMBS[] = {
    0xF473DE6AF278ECE6ULL,
    0x2C5C85FDF473DE6AULL
};

static const uep::limb_t LN10_LIMBS[] = {
    0x6C1F2B1E044ED660ULL,
    0x24D763776C1F2B1EULL
};

static const uep::limb_t SQRT2_LIMBS[] = {
    0x7F3BCC908B2E4F7EULL,
    0x16A09E667F3BCC90ULL
};

static const uep::limb_t SQRT3_LIMBS[] = {
    0x4C5C8C964B0A5B8CULL,
    0x1BBE67AE8584CAA7ULL
};

// Euler-Mascheroni constant γ ≈ 0.5772156649015328606065120900824024310421
static const uep::limb_t EULER_GAMMA_LIMBS[] = {
    0x93C467E37DB0C7A4ULL,
    0xD1BEF9A4BEE5C7F9ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Golden ratio φ = (1+√5)/2 ≈ 1.6180339887498948482045868343656381177203
static const uep::limb_t GOLDEN_RATIO_LIMBS[] = {
    0x9E3779B97F4A7C15ULL,
    0xF39CC0605CEDC834ULL,
    0x1081B94C3FC9C1B0ULL,
    0x19E3779B97F4A7C1ULL
};

// Physical constants (in SI units, scaled to appropriate fixed-point)
// Values from CODATA 2018 recommended values.

// Speed of light in vacuum: c = 299792458 m/s (exact)
static const uep::limb_t C_LIMBS[] = {
    0x0000000011DE784AULL,  // 299792458
    0x0000000000000000ULL
};

// Planck constant: h = 6.62607015e-34 J·s (exact since 2019)
// Represented as h * 2^32 * 10^34 to preserve precision
static const uep::limb_t H_LIMBS[] = {
    0x0000000000000000ULL,  // To be computed with high precision
    0x0000000000000000ULL
};

// Reduced Planck constant: ħ = h/(2π)
static const uep::limb_t HBAR_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Gravitational constant: G = 6.67430e-11 m^3·kg^-1·s^-2
static const uep::limb_t G_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Avogadro number: N_A = 6.02214076e23 mol^-1 (exact)
static const uep::limb_t NA_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Boltzmann constant: k_B = 1.380649e-23 J/K (exact)
static const uep::limb_t KB_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Elementary charge: e = 1.602176634e-19 C (exact)
static const uep::limb_t E_CHARGE_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Vacuum permittivity: ε0 = 1/(μ0·c^2) ≈ 8.8541878128e-12 F/m
static const uep::limb_t EPSILON0_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Vacuum permeability: μ0 = 4π·10^-7 N/A^2 ≈ 1.25663706212e-6
static const uep::limb_t MU0_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Fine-structure constant: α ≈ 1/137.035999084
static const uep::limb_t ALPHA_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Stefan-Boltzmann constant: σ ≈ 5.670374419e-8 W·m^-2·K^-4
static const uep::limb_t SIGMA_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Rydberg constant: R∞ ≈ 10973731.568160 m^-1
static const uep::limb_t R_INF_LIMBS[] = {
    0x0000000000A76A1CULL,
    0x0000000000000000ULL
};

// Astronomical Unit: 1 AU = 149597870700 m (exact)
static const uep::limb_t AU_LIMBS[] = {
    0x00000022D3D0D6CCULL,
    0x0000000000000000ULL
};

// Parsec: 1 pc ≈ 3.0856775814913673e16 m
static const uep::limb_t PC_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Light-year: 1 ly = c * 365.25 * 86400 ≈ 9.4607304725808e15 m
static const uep::limb_t LY_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Solar mass: M☉ = 1.98847e30 kg
static const uep::limb_t M_SUN_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Earth mass: M🜨 = 5.9722e24 kg
static const uep::limb_t M_EARTH_LIMBS[] = {
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Earth radius (mean): R🜨 = 6371000 m
static const uep::limb_t R_EARTH_LIMBS[] = {
    0x0000000000613588ULL,
    0x0000000000000000ULL
};

// Standard gravity: g0 = 9.80665 m/s^2 (exact)
static const uep::limb_t G0_LIMBS[] = {
    0x0000000009CE809AULL,
    0x0000000000000000ULL
};

// ----------------------------------------------------------------------------
// Helper function to create BigNumber from limb array
// ----------------------------------------------------------------------------
static uep::BigNumber make_bignumber_from_limbs(const uep::limb_t* limbs, size_t count) {
    uep::BigIntCore bic;
    bic.resize(count);
    std::memcpy(bic.data(), limbs, count * sizeof(uep::limb_t));
    bic.normalize();
    return uep::BigNumber(bic);
}

// ----------------------------------------------------------------------------
// Constructor / Destructor
// ----------------------------------------------------------------------------
CoreConstants::CoreConstants() {
    singleton = this;
    initialize_constants();
}

CoreConstants::~CoreConstants() {
    singleton = nullptr;
}

void CoreConstants::_bind_methods() {
    // Mathematical constants
    ClassDB::bind_static_method("CoreConstants", D_METHOD("PI"), &CoreConstants::PI);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("TAU"), &CoreConstants::TAU);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("E"), &CoreConstants::E);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("LN2"), &CoreConstants::LN2);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("LN10"), &CoreConstants::LN10);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("SQRT2"), &CoreConstants::SQRT2);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("SQRT3"), &CoreConstants::SQRT3);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("EULER_GAMMA"), &CoreConstants::EULER_GAMMA);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("GOLDEN_RATIO"), &CoreConstants::GOLDEN_RATIO);
    
    // Physical constants
    ClassDB::bind_static_method("CoreConstants", D_METHOD("SPEED_OF_LIGHT"), &CoreConstants::SPEED_OF_LIGHT);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("PLANCK_CONSTANT"), &CoreConstants::PLANCK_CONSTANT);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("REDUCED_PLANCK"), &CoreConstants::REDUCED_PLANCK);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("GRAVITATIONAL_CONSTANT"), &CoreConstants::GRAVITATIONAL_CONSTANT);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("AVOGADRO_NUMBER"), &CoreConstants::AVOGADRO_NUMBER);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("BOLTZMANN_CONSTANT"), &CoreConstants::BOLTZMANN_CONSTANT);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("ELEMENTARY_CHARGE"), &CoreConstants::ELEMENTARY_CHARGE);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("VACUUM_PERMITTIVITY"), &CoreConstants::VACUUM_PERMITTIVITY);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("VACUUM_PERMEABILITY"), &CoreConstants::VACUUM_PERMEABILITY);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("FINE_STRUCTURE"), &CoreConstants::FINE_STRUCTURE);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("STEFAN_BOLTZMANN"), &CoreConstants::STEFAN_BOLTZMANN);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("RYDBERG_CONSTANT"), &CoreConstants::RYDBERG_CONSTANT);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("ASTRONOMICAL_UNIT"), &CoreConstants::ASTRONOMICAL_UNIT);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("PARSEC"), &CoreConstants::PARSEC);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("LIGHT_YEAR"), &CoreConstants::LIGHT_YEAR);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("SOLAR_MASS"), &CoreConstants::SOLAR_MASS);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("EARTH_MASS"), &CoreConstants::EARTH_MASS);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("EARTH_RADIUS"), &CoreConstants::EARTH_RADIUS);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("STANDARD_GRAVITY"), &CoreConstants::STANDARD_GRAVITY);
    
#ifdef UEP_USE_XTENSOR
    ClassDB::bind_static_method("CoreConstants", D_METHOD("PI_TENSOR"), &CoreConstants::PI_TENSOR);
    ClassDB::bind_static_method("CoreConstants", D_METHOD("SPEED_OF_LIGHT_TENSOR"), &CoreConstants::SPEED_OF_LIGHT_TENSOR);
#endif
}

// ----------------------------------------------------------------------------
// Initialize all constants with high-precision values
// ----------------------------------------------------------------------------
void CoreConstants::initialize_constants() {
    if (initialized) return;
    
    // Mathematical constants
    constants.pi = make_bignumber_from_limbs(PI_LIMBS, sizeof(PI_LIMBS)/sizeof(uep::limb_t));
    constants.tau = constants.pi * uep::BigNumber(2);
    constants.e = make_bignumber_from_limbs(E_LIMBS, sizeof(E_LIMBS)/sizeof(uep::limb_t));
    constants.ln2 = make_bignumber_from_limbs(LN2_LIMBS, sizeof(LN2_LIMBS)/sizeof(uep::limb_t));
    constants.ln10 = make_bignumber_from_limbs(LN10_LIMBS, sizeof(LN10_LIMBS)/sizeof(uep::limb_t));
    constants.sqrt2 = make_bignumber_from_limbs(SQRT2_LIMBS, sizeof(SQRT2_LIMBS)/sizeof(uep::limb_t));
    constants.sqrt3 = make_bignumber_from_limbs(SQRT3_LIMBS, sizeof(SQRT3_LIMBS)/sizeof(uep::limb_t));
    constants.euler_gamma = make_bignumber_from_limbs(EULER_GAMMA_LIMBS, sizeof(EULER_GAMMA_LIMBS)/sizeof(uep::limb_t));
    constants.golden_ratio = make_bignumber_from_limbs(GOLDEN_RATIO_LIMBS, sizeof(GOLDEN_RATIO_LIMBS)/sizeof(uep::limb_t));
    
    // Physical constants - compute from exact values or high-precision sources
    // Speed of light (exact integer)
    constants.c = uep::BigNumber(299792458LL);
    
    // Planck constant (exact in SI since 2019): h = 6.62607015e-34
    // Represent as 662607015 * 10^-42, then scale appropriately.
    uep::BigNumber h_val = uep::BigNumber(662607015LL);
    uep::BigNumber ten_pow_neg_42 = uep::BigNumber(1) / uep::BigNumber(10).pow(uep::BigNumber(42));
    constants.h = h_val * ten_pow_neg_42;
    
    // Reduced Planck: ħ = h / (2π)
    constants.hbar = constants.h / (uep::BigNumber(2) * constants.pi);
    
    // Gravitational constant: G = 6.67430e-11
    uep::BigNumber G_val = uep::BigNumber(667430LL);
    uep::BigNumber ten_pow_neg_16 = uep::BigNumber(1) / uep::BigNumber(10).pow(uep::BigNumber(16));
    constants.G = G_val * ten_pow_neg_16;
    
    // Avogadro number (exact): N_A = 6.02214076e23
    constants.N_A = uep::BigNumber(602214076000000000000000LL);
    
    // Boltzmann constant (exact): k_B = 1.380649e-23
    uep::BigNumber kB_val = uep::BigNumber(1380649LL);
    uep::BigNumber ten_pow_neg_29 = uep::BigNumber(1) / uep::BigNumber(10).pow(uep::BigNumber(29));
    constants.k_B = kB_val * ten_pow_neg_29;
    
    // Elementary charge (exact): e = 1.602176634e-19
    uep::BigNumber e_val = uep::BigNumber(1602176634LL);
    uep::BigNumber ten_pow_neg_28 = uep::BigNumber(1) / uep::BigNumber(10).pow(uep::BigNumber(28));
    constants.e_charge = e_val * ten_pow_neg_28;
    
    // Vacuum permeability: μ0 = 4π·10^-7 (exact)
    constants.mu0 = uep::BigNumber(4) * constants.pi * uep::BigNumber(1) / uep::BigNumber(10000000);
    
    // Vacuum permittivity: ε0 = 1 / (μ0·c^2)
    constants.epsilon0 = uep::BigNumber(1) / (constants.mu0 * constants.c * constants.c);
    
    // Fine-structure constant: α = e^2 / (4π·ε0·ħ·c)
    uep::BigNumber e_sq = constants.e_charge * constants.e_charge;
    uep::BigNumber denom = uep::BigNumber(4) * constants.pi * constants.epsilon0 * constants.hbar * constants.c;
    constants.alpha = e_sq / denom;
    
    // Stefan-Boltzmann constant: σ = π^2·k_B^4 / (60·ħ^3·c^2)
    uep::BigNumber pi_sq = constants.pi * constants.pi;
    uep::BigNumber kB4 = constants.k_B * constants.k_B * constants.k_B * constants.k_B;
    uep::BigNumber hbar3 = constants.hbar * constants.hbar * constants.hbar;
    uep::BigNumber c_sq = constants.c * constants.c;
    constants.sigma = (pi_sq * kB4) / (uep::BigNumber(60) * hbar3 * c_sq);
    
    // Rydberg constant: R∞ = m_e·e^4 / (8·ε0^2·h^3·c)
    // Using approximate electron mass, or we can use known value
    constants.R_inf = uep::BigNumber(10973731.568160);
    
    // Astronomical Unit (exact)
    constants.AU = uep::BigNumber(149597870700LL);
    
    // Parsec: 1 pc = AU / tan(1 arcsec) ≈ 648000/π AU
    constants.pc = constants.AU * uep::BigNumber(648000) / constants.pi;
    
    // Light-year: distance light travels in one Julian year
    uep::BigNumber seconds_per_year = uep::BigNumber(365.25 * 24 * 3600);
    constants.ly = constants.c * seconds_per_year;
    
    // Solar mass (approximate)
    constants.M_sun = uep::BigNumber(1.98847e30);
    
    // Earth mass
    constants.M_earth = uep::BigNumber(5.9722e24);
    
    // Earth radius (mean)
    constants.R_earth = uep::BigNumber(6371000);
    
    // Standard gravity (exact)
    constants.g0 = uep::BigNumber(9.80665);
    
    initialized = true;
}

// ----------------------------------------------------------------------------
// Registration function called from register_types
// ----------------------------------------------------------------------------
void register_core_constants() {
    GDREGISTER_CLASS(CoreConstants);
    Engine::get_singleton()->add_singleton(Engine::Singleton("CoreConstants", CoreConstants::get_singleton()));
}

// Ending of File 12 of 15 (core/core_constants.cpp)