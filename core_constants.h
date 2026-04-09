//core/core_constants.h

#ifndef CORE_CONSTANTS_H
#define CORE_CONSTANTS_H

#include "src/big_number.h"

/**
 * @class CoreConstants
 * @brief Container for global engine constants implemented with BigNumber.
 * 
 * These constants are used to ensure that all calculations across the engine,
 * from microscopic collisions to galactic orbital mechanics, remain deterministic
 * and free from floating-point drift or 32/64-bit overflow issues.
 */

class CoreConstants {
public:
	// Mathematical Constants
	static const BigNumber PI;
	static const BigNumber TAU;
	static const BigNumber PI_HALF;
	static const BigNumber E;
	static const BigNumber SQRT2;
	static const BigNumber LN2;
	
	// Precision Constants
	static const BigNumber CMP_EPSILON;
	static const BigNumber UNIT_EPSILON;
	
	// Physics Constants (Galactic Scale)
	static const BigNumber SPEED_OF_LIGHT; // m/s
	static const BigNumber GRAVITATIONAL_CONSTANT; // G
	static const BigNumber PLANCK_CONSTANT;
	static const BigNumber BOLTZMANN_CONSTANT;
	
	// Limits
	static const BigNumber BIG_INF;
	static const BigNumber BIG_NEG_INF;
	static const BigNumber BIG_NAN;

	// Scale Factors
	static const BigNumber MICROSCOPIC_SCALE; // 1e-9
	static const BigNumber GALACTIC_SCALE;    // 1e12
	static const BigNumber UNIVERSAL_SCALE;   // 1e24

	// Time Constants (120fps precision)
	static const BigNumber FRAME_DELTA_120; // 1/120 seconds
};

#endif // CORE_CONSTANTS_H