// File 39: modules/gaia/src/types/constants.h

#ifndef GAIA_TYPES_CONSTANTS_H
#define GAIA_TYPES_CONSTANTS_H

#include "core/math/math_defs.h"

// ---------------------------------------------------------------------------
// Common physics and math constants used by Gaia algorithms.
// ---------------------------------------------------------------------------
namespace gaia::types {

	// Small epsilon for floating-point comparisons
	constexpr real_t EPSILON = CMP_EPSILON;

	// Larger epsilon for squared distances, etc.
	constexpr real_t EPSILON_SQ = CMP_EPSILON * CMP_EPSILON;

	// Machine epsilon (double precision variant, but we'll use real_t)
	constexpr real_t MACHINE_EPSILON = CMP_EPSILON;

	// Pi / 180
	constexpr real_t DEG_TO_RAD = Math_PI / 180.0;

	// 180 / Pi
	constexpr real_t RAD_TO_DEG = 180.0 / Math_PI;

	// Very large number
	constexpr real_t INF = INFINITY;

	// Maximum finite real
	constexpr real_t MAX_REAL = FLT_MAX;

	// Minimum finite real
	constexpr real_t MIN_REAL = FLT_MIN;

} // namespace gaia::types

#endif // GAIA_TYPES_CONSTANTS_H