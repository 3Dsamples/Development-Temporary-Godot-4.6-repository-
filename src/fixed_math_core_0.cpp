--- START OF FILE src/fixed_math_core.cpp ---

#include "fixed_math_core.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>

// ============================================================================
// Deterministic String Parsing
// ============================================================================

FixedMathCore::FixedMathCore(const std::string& p_value) {
    size_t dot_pos = p_value.find('.');
    if (dot_pos == std::string::npos) {
        raw_value = std::stoll(p_value) << FRACTIONAL_BITS;
    } else {
        std::string int_part = p_value.substr(0, dot_pos);
        std::string frac_part = p_value.substr(dot_pos + 1);
        
        int64_t integer_val = int_part.empty() ? 0 : std::stoll(int_part);
        bool is_negative = (!int_part.empty() && int_part[0] == '-');
        
        raw_value = std::abs(integer_val) << FRACTIONAL_BITS;
        
        // Parse fractional part safely using exact bitwise math
        int64_t fraction = 0;
        int64_t multiplier = 1000000000000000ULL; // High precision base 10 scale
        int64_t current_mult = multiplier;
        
        for (char c : frac_part) {
            current_mult /= 10;
            if (current_mult == 0) break; // Limit precision to available bits
            fraction += (c - '0') * current_mult;
        }
        
        // Map Base-10 fraction to Q32.32 Base-2 fraction
        uint64_t binary_fraction = (static_cast<uint64_t>(fraction) * static_cast<uint64_t>(ONE)) / multiplier;
        raw_value += binary_fraction;
        
        if (is_negative) {
            raw_value = -raw_value;
        }
    }
}

// ============================================================================
// Advanced Deterministic Math (No FPU used)
// ============================================================================

FixedMathCore FixedMathCore::square_root() const {
    if (raw_value < 0) {
        throw std::runtime_error("UniversalSolver Error: Square root of negative number in FixedMathCore");
    }
    if (raw_value == 0) return FixedMathCore(0LL, true);

    uint64_t res = 0;
    uint64_t bit = 1ULL << 62; 

    // "bit" starts at the highest power of four <= 2^62.
    uint64_t val = static_cast<uint64_t>(raw_value);

    while (bit > val) {
        bit >>= 2;
    }

    while (bit != 0) {
        if (val >= res + bit) {
            val -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }

    // Since we took the square root of a Q32.32 number, we effectively took
    // sqrt(X * 2^32). The result is sqrt(X) * 2^16. To convert back to Q32.32,
    // we need to shift the result left by 16 bits.
    return FixedMathCore(static_cast<int64_t>(res << 16), true);
}

FixedMathCore FixedMathCore::power(int32_t exponent) const {
    if (exponent == 0) return FixedMathCore(ONE, true);
    if (exponent < 0) return FixedMathCore(ONE, true) / power(-exponent);

    FixedMathCore result(ONE, true);
    FixedMathCore base = *this;
    int32_t exp = exponent;

    // Exponentiation by squaring for ultra-fast scaling
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = result * base;
        }
        base = base * base;
        exp /= 2;
    }
    return result;
}

// ============================================================================
// Deterministic Trigonometry (Taylor Series)
// ============================================================================

FixedMathCore FixedMathCore::sin() const {
    int64_t normalized = raw_value % TWO_PI_RAW;
    if (normalized > PI_RAW) normalized -= TWO_PI_RAW;
    if (normalized < -PI_RAW) normalized += TWO_PI_RAW;

    FixedMathCore x(normalized, true);
    FixedMathCore x2 = x * x;
    FixedMathCore x3 = x2 * x;
    FixedMathCore x5 = x3 * x2;
    FixedMathCore x7 = x5 * x2;
    FixedMathCore x9 = x7 * x2;

    // Taylor series: x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
    FixedMathCore term3 = x3 / FixedMathCore(6LL << FRACTIONAL_BITS, true);
    FixedMathCore term5 = x5 / FixedMathCore(120LL << FRACTIONAL_BITS, true);
    FixedMathCore term7 = x7 / FixedMathCore(5040LL << FRACTIONAL_BITS, true);
    FixedMathCore term9 = x9 / FixedMathCore(362880LL << FRACTIONAL_BITS, true);

    return x - term3 + term5 - term7 + term9;
}

FixedMathCore FixedMathCore::cos() const {
    FixedMathCore shifted = *this + FixedMathCore(HALF_PI_RAW, true);
    return shifted.sin();
}

FixedMathCore FixedMathCore::tan() const {
    FixedMathCore cosine = this->cos();
    if (cosine.raw_value == 0) {
        throw std::runtime_error("UniversalSolver Error: Tangent undefined (division by zero)");
    }
    return this->sin() / cosine;
}

// ============================================================================
// Deterministic Atan2 (Rational Polynomial Approx replacing std::atan2)
// ============================================================================

FixedMathCore FixedMathCore::atan2(const FixedMathCore& x) const {
    if (x.raw_value == 0 && raw_value == 0) return FixedMathCore(0LL, true);

    FixedMathCore abs_y = this->absolute();
    FixedMathCore abs_x = x.absolute();
    
    bool invert = abs_y > abs_x;
    FixedMathCore z = invert ? (abs_x / abs_y) : (abs_y / abs_x);

    // Fast Polynomial Approximation for atan(z) where z in [0, 1]
    // approx = (pi/4)*z - z*(|z| - 1)*(0.2447 + 0.0663*|z|)
    FixedMathCore pi_over_4(3373259426LL, true); // (PI/4) * 2^32
    FixedMathCore const_A(1050965647LL, true);   // 0.2447 * 2^32
    FixedMathCore const_B(284760539LL, true);    // 0.0663 * 2^32
    FixedMathCore one(ONE, true);

    FixedMathCore term1 = const_B * z;
    FixedMathCore term2 = const_A + term1;
    FixedMathCore term3 = z - one;
    FixedMathCore term4 = z * term3;
    FixedMathCore approx = (pi_over_4 * z) - (term4 * term2);

    if (invert) {
        approx = FixedMathCore(HALF_PI_RAW, true) - approx;
    }

    if (x.raw_value < 0) {
        approx = FixedMathCore(PI_RAW, true) - approx;
    }
    
    if (raw_value < 0) {
        approx = -approx;
    }

    return approx;
}

// ============================================================================
// Zero-Copy String Output
// ============================================================================

std::string FixedMathCore::to_string() const {
    // Pure integer math to convert Q32.32 to a precise decimal string without FPU
    int64_t int_part = to_int();
    uint64_t frac_part = absolute().raw_value & (ONE - 1);
    
    // Scale fraction up to 6 decimal places (10^6)
    uint64_t frac_base10 = (frac_part * 1000000ULL) >> FRACTIONAL_BITS;
    
    std::ostringstream oss;
    if (raw_value < 0 && int_part == 0) oss << "-"; // Handle edge case like -0.5
    oss << int_part << "." << std::setfill('0') << std::setw(6) << frac_base10;
    return oss.str();
}

--- END OF FILE src/fixed_math_core.cpp ---
