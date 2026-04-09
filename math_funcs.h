//core/math/math_funcs.h

#ifndef MATH_FUNCS_H
#define MATH_FUNCS_H

#include "core/math/math_defs.h"
#include "core/core_constants.h"

/**
 * @class Math
 * @brief Static utility class for all mathematical operations in ETengine.
 * 
 * Every function is implemented to use BigNumber, ensuring that transcendental
 * operations are deterministic across all platforms. This prevents desync in 
 * multi-threaded galactic simulations.
 */
class Math {
public:
	static real_t sin(real_t p_x);
	static real_t cos(real_t p_x);
	static real_t tan(real_t p_x);
	static real_t asin(real_t p_x);
	static real_t acos(real_t p_x);
	static real_t atan(real_t p_x);
	static real_t atan2(real_t p_y, real_t p_x);

	static real_t sqrt(real_t p_x);
	static real_t pow(real_t p_x, real_t p_y);
	static real_t log(real_t p_x);
	static real_t exp(real_t p_x);

	static real_t floor(real_t p_x);
	static real_t ceil(real_t p_x);
	static real_t round(real_t p_x);
	static real_t abs(real_t p_x);
	static real_t sign(real_t p_x);
	static real_t fmod(real_t p_x, real_t p_y);
	static real_t fposmod(real_t p_x, real_t p_y);

	static real_t lerp(real_t p_from, real_t p_to, real_t p_weight);
	static real_t lerp_angle(real_t p_from, real_t p_to, real_t p_weight);
	static real_t cubic_interpolate(real_t p_from, real_t p_to, real_t p_pre, real_t p_post, real_t p_weight);
	static real_t bezier_interpolate(real_t p_start, real_t p_control_1, real_t p_control_2, real_t p_end, real_t p_t);
	
	static real_t deg_to_rad(real_t p_y);
	static real_t rad_to_deg(real_t p_y);
	
	static real_t clamp(real_t p_val, real_t p_min, real_t p_max);
	static real_t smoothstep(real_t p_from, real_t p_to, real_t p_weight);
	static real_t move_toward(real_t p_from, real_t p_to, real_t p_delta);

	static bool is_equal_approx(real_t p_a, real_t p_b);
	static bool is_zero_approx(real_t p_x);

	static uint32_t stepify(real_t p_value, real_t p_step);
};

#endif // MATH_FUNCS_H