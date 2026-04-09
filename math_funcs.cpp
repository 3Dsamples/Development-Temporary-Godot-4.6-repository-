//core/math/math_funcs.cpp

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "core/core_constants.h"

real_t Math::sin(real_t p_x) {
	// Range reduction: sin(x) = sin(x mod 2PI)
	real_t x = fmod(p_x, CoreConstants::TAU);
	if (x < BigNumber(0)) {
		x += CoreConstants::TAU;
	}
	// Extract fixed part for Taylor/CORDIC logic in FixedMathCore
	FixedMathCore::fixed_t fx = FixedMathCore::from_double(x.to_double());
	FixedMathCore::fixed_t res = FixedMathCore::sin(fx);
	return BigNumber(BigIntCore(0), res);
}

real_t Math::cos(real_t p_x) {
	real_t x = fmod(p_x, CoreConstants::TAU);
	if (x < BigNumber(0)) {
		x += CoreConstants::TAU;
	}
	FixedMathCore::fixed_t fx = FixedMathCore::from_double(x.to_double());
	FixedMathCore::fixed_t res = FixedMathCore::cos(fx);
	return BigNumber(BigIntCore(0), res);
}

real_t Math::tan(real_t p_x) {
	real_t s = sin(p_x);
	real_t c = cos(p_x);
	if (is_zero_approx(c)) {
		return CoreConstants::BIG_INF;
	}
	return s / c;
}

real_t Math::atan2(real_t p_y, real_t p_x) {
	FixedMathCore::fixed_t fy = FixedMathCore::from_double(p_y.to_double());
	FixedMathCore::fixed_t fx = FixedMathCore::from_double(p_x.to_double());
	FixedMathCore::fixed_t res = FixedMathCore::atan2(fy, fx);
	return BigNumber(BigIntCore(0), res);
}

real_t Math::sqrt(real_t p_x) {
	if (p_x < BigNumber(0)) return BigNumber(0);
	if (p_x == BigNumber(0)) return BigNumber(0);

	// For large numbers, we utilize the property sqrt(a * 2^2n) = sqrt(a) * 2^n
	// But since BigNumber is sector-based, we use FixedMathCore's internal 128-bit sqrt
	// or iterative refinement for the BigInt part.
	real_t res = p_x / BigNumber(2); // Initial guess
	for (int i = 0; i < 10; i++) {
		res = (res + p_x / res) / BigNumber(2);
	}
	return res;
}

real_t Math::fmod(real_t p_x, real_t p_y) {
	if (p_y == BigNumber(0)) return BigNumber(0);
	BigNumber quot = p_x / p_y;
	return p_x - floor(quot) * p_y;
}

real_t Math::floor(real_t p_x) {
	return BigNumber::floor(p_x);
}

real_t Math::ceil(real_t p_x) {
	return BigNumber::ceil(p_x);
}

real_t Math::round(real_t p_x) {
	return BigNumber::round(p_x);
}

real_t Math::abs(real_t p_x) {
	return BigNumber::abs(p_x);
}

real_t Math::lerp(real_t p_from, real_t p_to, real_t p_weight) {
	return p_from + (p_to - p_from) * p_weight;
}

real_t Math::clamp(real_t p_val, real_t p_min, real_t p_max) {
	if (p_val < p_min) return p_min;
	if (p_val > p_max) return p_max;
	return p_val;
}

real_t Math::deg_to_rad(real_t p_y) {
	return p_y * CoreConstants::PI / BigNumber(180);
}

real_t Math::rad_to_deg(real_t p_y) {
	return p_y * BigNumber(180) / CoreConstants::PI;
}

bool Math::is_equal_approx(real_t p_a, real_t p_b) {
	return abs(p_a - p_b) < CoreConstants::CMP_EPSILON;
}

bool Math::is_zero_approx(real_t p_x) {
	return abs(p_x) < CoreConstants::CMP_EPSILON;
}

real_t Math::exp(real_t p_x) {
	FixedMathCore::fixed_t fx = FixedMathCore::from_double(p_x.to_double());
	FixedMathCore::fixed_t res = FixedMathCore::exp(fx);
	return BigNumber(BigIntCore(0), res);
}

real_t Math::log(real_t p_x) {
	FixedMathCore::fixed_t fx = FixedMathCore::from_double(p_x.to_double());
	FixedMathCore::fixed_t res = FixedMathCore::log(fx);
	return BigNumber(BigIntCore(0), res);
}

real_t Math::pow(real_t p_x, real_t p_y) {
	// x^y = exp(y * ln(x))
	if (p_x == BigNumber(0)) return BigNumber(0);
	if (p_y == BigNumber(0)) return BigNumber(1);
	return exp(p_y * log(p_x));
}

real_t Math::smoothstep(real_t p_from, real_t p_to, real_t p_weight) {
	if (is_equal_approx(p_from, p_to)) return p_from;
	real_t t = clamp((p_weight - p_from) / (p_to - p_from), BigNumber(0), BigNumber(1));
	return t * t * (BigNumber(3) - BigNumber(2) * t);
}

real_t Math::move_toward(real_t p_from, real_t p_to, real_t p_delta) {
	if (abs(p_to - p_from) <= p_delta) return p_to;
	return p_from + sign(p_to - p_from) * p_delta;
}

real_t Math::sign(real_t p_x) {
	if (p_x < BigNumber(0)) return BigNumber(-1);
	if (p_x > BigNumber(0)) return BigNumber(1);
	return BigNumber(0);
}