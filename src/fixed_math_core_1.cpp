--- START OF FILE src/fixed_math_core.cpp ---

#include "src/fixed_math_core.h"
#include "core/math/math_funcs.h"
#include <sstream>
#include <iomanip>

/**
 * FixedMathCore String Constructor
 * 
 * Deterministically parses a string decimal into Q32.32 bits.
 * Bypasses all standard C library float functions to ensure bit-perfection.
 */
FixedMathCore::FixedMathCore(const std::string &p_val) {
	if (p_val.empty()) {
		raw_value = 0;
		return;
	}

	size_t dot_pos = p_val.find('.');
	bool is_neg = (p_val[0] == '-');
	std::string abs_val = is_neg ? p_val.substr(1) : p_val;
	dot_pos = abs_val.find('.');

	int64_t integer_part = 0;
	int64_t fractional_part = 0;

	if (dot_pos == std::string::npos) {
		integer_part = std::stoll(abs_val);
	} else {
		std::string s_int = abs_val.substr(0, dot_pos);
		std::string s_frac = abs_val.substr(dot_pos + 1);
		
		if (!s_int.empty()) integer_part = std::stoll(s_int);
		
		// Scale fractional part to 10^10 to maintain precision before bit-shifting
		size_t frac_len = s_frac.length();
		if (frac_len > 10) s_frac = s_frac.substr(0, 10);
		else while (s_frac.length() < 10) s_frac += '0';
		
		int64_t frac_val = std::stoll(s_frac);
		fractional_part = (frac_val << 32) / 10000000000LL;
	}

	raw_value = (integer_part << 32) + (fractional_part & 0xFFFFFFFFLL);
	if (is_neg) raw_value = -raw_value;
}

// ============================================================================
// Transcendental Implementations (Taylor Series & Bitwise Approximations)
// ============================================================================

/**
 * sin()
 * 7th Order Taylor Series with bit-perfect range reduction.
 * 120 FPS Optimized.
 */
FixedMathCore FixedMathCore::sin() const {
	int64_t x = raw_value % TWO_PI_RAW;
	if (x > PI_RAW) x -= TWO_PI_RAW;
	if (x < -PI_RAW) x += TWO_PI_RAW;

	FixedMathCore fx(x, true);
	FixedMathCore x2 = fx * fx;
	FixedMathCore x3 = x2 * fx;
	FixedMathCore x5 = x3 * x2;
	FixedMathCore x7 = x5 * x2;

	// Coefficients: 1/3! (0.1666...), 1/5! (0.00833...), 1/7! (0.000198...)
	FixedMathCore c3(715827882LL, true);
	FixedMathCore c5(35791394LL, true);
	FixedMathCore c7(852176LL, true);

	return fx - (x3 * c3) + (x5 * c5) - (x7 * c7);
}

FixedMathCore FixedMathCore::cos() const {
	// cos(x) = sin(x + pi/2)
	return FixedMathCore(raw_value + HALF_PI_RAW, true).sin();
}

FixedMathCore FixedMathCore::tan() const {
	FixedMathCore s = sin();
	FixedMathCore c = cos();
	CRASH_COND_MSG(c.raw_value == 0, "FixedMathCore: Tangent undefined (Cos is 0).");
	return s / c;
}

/**
 * square_root()
 * Bit-by-bit integer square root algorithm scaled for Q32.32.
 */
FixedMathCore FixedMathCore::square_root() const {
	CRASH_COND_MSG(raw_value < 0, "FixedMathCore: Square root of negative value.");
	if (raw_value == 0) return FixedMathCore(0LL, true);

	uint64_t op = static_cast<uint64_t>(raw_value);
	uint64_t res = 0;
	uint64_t one = 1ULL << 62; // Second-to-top bit

	while (one > op) one >>= 2;

	while (one != 0) {
		if (op >= res + one) {
			op -= res + one;
			res = (res >> 1) + one;
		} else {
			res >>= 1;
		}
		one >>= 2;
	}
	// Scale result: sqrt(x * 2^32) = sqrt(x) * 2^16.
	// To return to Q32.32, we shift left by 16.
	return FixedMathCore(static_cast<int64_t>(res << 16), true);
}

/**
 * exp()
 * Deterministic e^x approximation. 
 * Essential for atmospheric density, SPH kernels, and physical damping.
 */
FixedMathCore FixedMathCore::exp() const {
	FixedMathCore x(*this);
	FixedMathCore res = MathConstants<FixedMathCore>::one();
	FixedMathCore term = MathConstants<FixedMathCore>::one();

	// 8-term Taylor expansion for high accuracy in physics
	for (int i = 1; i < 9; i++) {
		term = (term * x) / FixedMathCore(static_cast<int64_t>(i));
		res += term;
	}
	return res;
}

/**
 * log()
 * Natural logarithm approximation using Newton's method.
 * Essential for Shannon Entropy and WFC observation stability.
 */
FixedMathCore FixedMathCore::log() const {
	CRASH_COND_MSG(raw_value <= 0, "FixedMathCore: Log of non-positive value.");
	
	FixedMathCore x(*this);
	FixedMathCore y(0LL); // Initial guess
	FixedMathCore e = MathConstants<FixedMathCore>::e();

	// Iterative refinement for 120 FPS performance budget
	for (int i = 0; i < 6; i++) {
		FixedMathCore ey = y.exp();
		y = y + (x - ey) / ey;
	}
	return y;
}

FixedMathCore FixedMathCore::power(int32_t p_exp) const {
	if (p_exp == 0) return MathConstants<FixedMathCore>::one();
	if (p_exp < 0) return MathConstants<FixedMathCore>::one() / power(-p_exp);
	
	FixedMathCore res = MathConstants<FixedMathCore>::one();
	FixedMathCore base = *this;
	uint32_t e = static_cast<uint32_t>(p_exp);

	while (e > 0) {
		if (e & 1) res *= base;
		base *= base;
		e >>= 1;
	}
	return res;
}

/**
 * atan2()
 * Polynomial approximation with range reduction for high-speed robotic sensors.
 */
FixedMathCore FixedMathCore::atan2(const FixedMathCore &p_x) const {
	FixedMathCore abs_y = absolute();
	FixedMathCore abs_x = p_x.absolute();
	bool swap = abs_y > abs_x;
	
	FixedMathCore a = swap ? abs_x / abs_y : abs_y / abs_x;
	FixedMathCore s = a * a;
	
	// Deterministic polynomial: r = (((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a)
	FixedMathCore r = (FixedMathCore("-0.0464964749") * s + FixedMathCore("0.15931422")) * s;
	r = (r - FixedMathCore("0.327622764")) * s * a + a;

	if (swap) r = FixedMathCore(HALF_PI_RAW, true) - r;
	if (p_x.raw_value < 0) r = FixedMathCore(PI_RAW, true) - r;
	if (raw_value < 0) r = -r;

	return r;
}

std::string FixedMathCore::to_string() const {
	int64_t integer_part = to_int();
	uint64_t fractional_part = static_cast<uint64_t>(raw_value < 0 ? -raw_value : raw_value) & 0xFFFFFFFFULL;
	
	// Convert fraction to base 10
	uint64_t base10_frac = (fractional_part * 10000000000LL) >> 32;
	
	std::ostringstream oss;
	if (raw_value < 0 && integer_part == 0) oss << "-";
	oss << integer_part << "." << std::setfill('0') << std::setw(10) << base10_frac;
	return oss.str();
}

--- END OF FILE src/fixed_math_core.cpp ---
