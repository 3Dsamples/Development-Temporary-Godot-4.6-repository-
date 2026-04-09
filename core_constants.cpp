//core/core_constants.cpp

#include "core/core_constants.h"

// Mathematical Constants initialized from high-precision strings to avoid double-precision loss
const BigNumber CoreConstants::PI = BigNumber("3.141592653589793238462643383279502884197169399375105820974944592307816406286");
const BigNumber CoreConstants::TAU = BigNumber("6.283185307179586476925286766559005768394338798750211641949889184615632812572");
const BigNumber CoreConstants::PI_HALF = BigNumber("1.570796326794896619231321691639751442098584699687552910487472296153908203143");
const BigNumber CoreConstants::E = BigNumber("2.718281828459045235360287471352662497757247093699959574966967627724076630353");
const BigNumber CoreConstants::SQRT2 = BigNumber("1.414213562373095048801688724209698078569671875376948073176679737990732478462");
const BigNumber CoreConstants::LN2 = BigNumber("0.693147180559945309417232121458176568075500134360255254120680009493393621969");

// Precision and Epsilon Constants
const BigNumber CoreConstants::CMP_EPSILON = BigNumber("0.00001");
const BigNumber CoreConstants::UNIT_EPSILON = BigNumber("0.0000000001");

// Universal Physics Constants
const BigNumber CoreConstants::SPEED_OF_LIGHT = BigNumber("299792458"); // Exact m/s
const BigNumber CoreConstants::GRAVITATIONAL_CONSTANT = BigNumber("0.0000000000667430"); // G in m^3 kg^-1 s^-2
const BigNumber CoreConstants::PLANCK_CONSTANT = BigNumber("0.000000000000000000000000000000000662607015"); // h in J*s
const BigNumber CoreConstants::BOLTZMANN_CONSTANT = BigNumber("0.00000000000000000000001380649"); // k in J/K

// Limits for Galactic scale (Exceeding 64-bit float range)
const BigNumber CoreConstants::BIG_INF = BigNumber("99999999999999999999999999999999999999999999999999.9");
const BigNumber CoreConstants::BIG_NEG_INF = BigNumber("-99999999999999999999999999999999999999999999999999.9");
const BigNumber CoreConstants::BIG_NAN = BigNumber("0.0"); // In BigNumber logic, we use a specific state flag if needed, here represented as zero for base logic.

// Scale Multipliers
const BigNumber CoreConstants::MICROSCOPIC_SCALE = BigNumber("0.000000001");
const BigNumber CoreConstants::GALACTIC_SCALE = BigNumber("1000000000000");
const BigNumber CoreConstants::UNIVERSAL_SCALE = BigNumber("1000000000000000000000000");

// 120FPS timing: 1 / 120 = 0.00833333333...
const BigNumber CoreConstants::FRAME_DELTA_120 = BigNumber("0.0083333333333333333333333333333333333333333");