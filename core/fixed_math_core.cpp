// src/fixed_math_core.cpp
#include "fixed_math_core.h"
#include "big_int_core.h"
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifdef __AVX512F__
#include <immintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace uep {

// ----------------------------------------------------------------------------
// Precomputed lookup tables for trigonometric functions (bit-perfect)
// ----------------------------------------------------------------------------

// Factorials for Taylor series (precomputed as fixed-point)
static const FixedMathCore FACTORIALS[] = {
    FixedMathCore(1),                    // 0!
    FixedMathCore(1),                    // 1!
    FixedMathCore(2),                    // 2!
    FixedMathCore(6),                    // 3!
    FixedMathCore(24),                   // 4!
    FixedMathCore(120),                  // 5!
    FixedMathCore(720),                  // 6!
    FixedMathCore(5040),                 // 7!
    FixedMathCore(40320),                // 8!
    FixedMathCore(362880),               // 9!
    FixedMathCore(3628800),              // 10!
    FixedMathCore(39916800),             // 11!
    FixedMathCore(479001600),            // 12!
    FixedMathCore(6227020800LL),         // 13!
};

// Precomputed arctan table for CORDIC (angles in radians, Q32.32)
static const FixedMathCore CORDIC_ANGLES[] = {
    FixedMathCore::from_raw(0x3243F6A8885A308DLL >> 2), // atan(2^0) = π/4
    FixedMathCore::from_raw(0x1DAC670561BB4F68LL),      // atan(2^-1)
    FixedMathCore::from_raw(0x0FADBAFC9649B1B0LL),      // atan(2^-2)
    FixedMathCore::from_raw(0x07F56EA6A0D3F6C8LL),      // atan(2^-3)
    FixedMathCore::from_raw(0x03FEAB76E8A4C244LL),      // atan(2^-4)
    FixedMathCore::from_raw(0x01FFD55BBCB2E8C8LL),      // atan(2^-5)
    FixedMathCore::from_raw(0x00FFFAAADDDB9D50LL),      // atan(2^-6)
    FixedMathCore::from_raw(0x007FFF5556EEF430LL),      // atan(2^-7)
    FixedMathCore::from_raw(0x003FFFAAAAB776C8LL),      // atan(2^-8)
    FixedMathCore::from_raw(0x001FFFF5556EEF40LL),      // atan(2^-9)
    FixedMathCore::from_raw(0x000FFFFEAAAADD80LL),      // atan(2^-10)
    FixedMathCore::from_raw(0x0007FFFF5556EEF0LL),      // atan(2^-11)
    FixedMathCore::from_raw(0x0003FFFFAAAAADD8LL),      // atan(2^-12)
    FixedMathCore::from_raw(0x0001FFFFD55556E0LL),      // atan(2^-13)
    FixedMathCore::from_raw(0x0000FFFFEAAAAADCLL),      // atan(2^-14)
    FixedMathCore::from_raw(0x00007FFFF55556E0LL),      // atan(2^-15)
    FixedMathCore::from_raw(0x00003FFFFEAAAAD8LL),      // atan(2^-16)
    FixedMathCore::from_raw(0x00001FFFFD55556ELL),      // atan(2^-17)
    FixedMathCore::from_raw(0x00000FFFFEAAAAAALL),      // atan(2^-18)
    FixedMathCore::from_raw(0x000007FFFF555554LL),      // atan(2^-19)
    FixedMathCore::from_raw(0x000003FFFFEAAAAALL),      // atan(2^-20)
    FixedMathCore::from_raw(0x000001FFFFD55555LL),      // atan(2^-21)
    FixedMathCore::from_raw(0x000000FFFFEAAAAALL),      // atan(2^-22)
    FixedMathCore::from_raw(0x0000007FFFF55555LL),      // atan(2^-23)
    FixedMathCore::from_raw(0x0000003FFFFEAAAALL),      // atan(2^-24)
    FixedMathCore::from_raw(0x0000001FFFFD5555LL),      // atan(2^-25)
    FixedMathCore::from_raw(0x0000000FFFFEAAAALL),      // atan(2^-26)
    FixedMathCore::from_raw(0x00000007FFFF5555LL),      // atan(2^-27)
    FixedMathCore::from_raw(0x00000003FFFFEAAALL),      // atan(2^-28)
    FixedMathCore::from_raw(0x00000001FFFFD555LL),      // atan(2^-29)
    FixedMathCore::from_raw(0x00000000FFFFEAAALL),      // atan(2^-30)
    FixedMathCore::from_raw(0x000000007FFFF555LL),      // atan(2^-31)
};

// Precomputed CORDIC gain: K = ∏ cos(atan(2^-i)) ≈ 0.607252935
static const FixedMathCore CORDIC_GAIN = FixedMathCore::from_raw(0x26DD3B6A10D7F6B8LL);

// ----------------------------------------------------------------------------
// Taylor Series for sin(x) with range reduction to [-π, π]
// 13th-degree polynomial: sin(x) ≈ x - x^3/3! + x^5/5! - ... + x^13/13!
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::sin(FixedMathCore x) {
    // Range reduction: bring x into [-π, π]
    FixedMathCore two_pi = pi() * FixedMathCore(2);
    FixedMathCore pi_val = pi();
    
    // x = x mod 2π
    int64_t k = (x / two_pi).to_int();
    x = x - two_pi * FixedMathCore(k);
    
    // If x > π, map to [0, π] with sign flip
    bool negate = false;
    if (x > pi_val) {
        x = two_pi - x;
        negate = true;
    } else if (x < FixedMathCore(0)) {
        x = -x;
        negate = true;
    }
    if (x > pi_val) {
        x = two_pi - x;
        // negate remains true if original was > π
    }
    
    // Use Taylor series around 0: sin(x) = x - x^3/3! + x^5/5! - ...
    FixedMathCore x2 = x * x;
    FixedMathCore x3 = x2 * x;
    FixedMathCore term = x;
    FixedMathCore result = term;
    
    // Precomputed factorials for odd powers
    // 3! = 6, 5! = 120, 7! = 5040, 9! = 362880, 11! = 39916800, 13! = 6227020800
    FixedMathCore fact3  = FixedMathCore(6);
    FixedMathCore fact5  = FixedMathCore(120);
    FixedMathCore fact7  = FixedMathCore(5040);
    FixedMathCore fact9  = FixedMathCore(362880LL);
    FixedMathCore fact11 = FixedMathCore(39916800LL);
    FixedMathCore fact13 = FixedMathCore(6227020800LL);
    
    // Compute terms iteratively
    FixedMathCore current_power = x3;
    FixedMathCore current_fact = fact3;
    bool add = false;
    
    // term = -x^3/3!
    term = current_power / current_fact;
    result = result - term;
    
    current_power = current_power * x2; // x^5
    current_fact = fact5;
    term = current_power / current_fact;
    result = result + term;
    
    current_power = current_power * x2; // x^7
    current_fact = fact7;
    term = current_power / current_fact;
    result = result - term;
    
    current_power = current_power * x2; // x^9
    current_fact = fact9;
    term = current_power / current_fact;
    result = result + term;
    
    current_power = current_power * x2; // x^11
    current_fact = fact11;
    term = current_power / current_fact;
    result = result - term;
    
    current_power = current_power * x2; // x^13
    current_fact = fact13;
    term = current_power / current_fact;
    result = result + term;
    
    if (negate) result = -result;
    return result;
}

// ----------------------------------------------------------------------------
// cos(x) = sin(x + π/2)
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::cos(FixedMathCore x) {
    return sin(x + pi() / FixedMathCore(2));
}

// ----------------------------------------------------------------------------
// tan(x) = sin(x) / cos(x)
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::tan(FixedMathCore x) {
    FixedMathCore c = cos(x);
    if (c == FixedMathCore(0)) {
        // Undefined, return large value with sign
        return FixedMathCore::from_raw((sin(x) > FixedMathCore(0) ? 1 : -1) * 0x7FFFFFFFFFFFFFFFLL);
    }
    return sin(x) / c;
}

// ----------------------------------------------------------------------------
// asin(x) using Taylor series: asin(x) = x + (1/2)x^3/3 + (1*3/2*4)x^5/5 + ...
// Range [-1,1]
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::asin(FixedMathCore x) {
    if (x < FixedMathCore(-1) || x > FixedMathCore(1)) {
        return FixedMathCore(0); // error, return 0
    }
    bool negate = false;
    if (x < FixedMathCore(0)) {
        x = -x;
        negate = true;
    }
    
    FixedMathCore x2 = x * x;
    FixedMathCore term = x;
    FixedMathCore result = term;
    
    // Coefficients for series up to degree 13
    FixedMathCore coeff1 = FixedMathCore::from_raw(0x1555555555555555LL); // 1/6 * 2^32? Actually 1/6 ≈ 0x2AAAAAAA...
    // We'll use rational combinations for accuracy
    // term = x^3 * (1/6)
    FixedMathCore x3 = x2 * x;
    term = x3 / FixedMathCore(6);
    result = result + term;
    
    FixedMathCore x5 = x3 * x2;
    term = (x5 * FixedMathCore(3)) / FixedMathCore(40); // 3/40 ≈ 0.075
    result = result + term;
    
    FixedMathCore x7 = x5 * x2;
    term = (x7 * FixedMathCore(5)) / FixedMathCore(112); // 5/112 ≈ 0.0446428
    result = result + term;
    
    FixedMathCore x9 = x7 * x2;
    term = (x9 * FixedMathCore(35)) / FixedMathCore(1152); // 35/1152 ≈ 0.03038
    result = result + term;
    
    FixedMathCore x11 = x9 * x2;
    term = (x11 * FixedMathCore(63)) / FixedMathCore(2816); // 63/2816 ≈ 0.02237
    result = result + term;
    
    FixedMathCore x13 = x11 * x2;
    term = (x13 * FixedMathCore(231)) / FixedMathCore(13312); // 231/13312 ≈ 0.01735
    result = result + term;
    
    if (negate) result = -result;
    return result;
}

// ----------------------------------------------------------------------------
// acos(x) = π/2 - asin(x)
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::acos(FixedMathCore x) {
    return pi() / FixedMathCore(2) - asin(x);
}

// ----------------------------------------------------------------------------
// atan(x) using CORDIC algorithm for vectoring mode
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::atan(FixedMathCore x) {
    return atan2(x, FixedMathCore(1));
}

// ----------------------------------------------------------------------------
// atan2(y, x) using CORDIC vectoring mode
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::atan2(FixedMathCore y, FixedMathCore x) {
    if (x == FixedMathCore(0) && y == FixedMathCore(0)) {
        return FixedMathCore(0);
    }
    
    // CORDIC gain factor (precomputed)
    // We'll use 32 iterations for full precision
    const int ITERATIONS = 32;
    
    FixedMathCore x_val = x;
    FixedMathCore y_val = y;
    FixedMathCore angle = FixedMathCore(0);
    
    // Handle quadrant adjustments
    if (x_val < FixedMathCore(0)) {
        if (y_val >= FixedMathCore(0)) {
            // Second quadrant: rotate by π
            angle = pi();
            x_val = -x_val;
            y_val = -y_val;
        } else {
            // Third quadrant: rotate by -π
            angle = -pi();
            x_val = -x_val;
            y_val = -y_val;
        }
    }
    
    // CORDIC iterations
    for (int i = 0; i < ITERATIONS; ++i) {
        FixedMathCore d;
        if (y_val >= FixedMathCore(0)) {
            d = FixedMathCore(1);
        } else {
            d = FixedMathCore(-1);
        }
        
        // Rotation: x' = x - d * y * 2^-i
        //          y' = y + d * x * 2^-i
        //          z' = z + d * atan(2^-i)
        FixedMathCore shift = FixedMathCore(1) >> i; // 2^-i
        FixedMathCore x_new = x_val - d * y_val * shift;
        FixedMathCore y_new = y_val + d * x_val * shift;
        x_val = x_new;
        y_val = y_new;
        angle = angle + d * CORDIC_ANGLES[i];
    }
    
    return angle;
}

// ----------------------------------------------------------------------------
// Hyperbolic functions using exponential
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::sinh(FixedMathCore x) {
    FixedMathCore exp_x = exp(x);
    FixedMathCore exp_neg_x = FixedMathCore(1) / exp_x;
    return (exp_x - exp_neg_x) / FixedMathCore(2);
}

FixedMathCore FixedMathCore::cosh(FixedMathCore x) {
    FixedMathCore exp_x = exp(x);
    FixedMathCore exp_neg_x = FixedMathCore(1) / exp_x;
    return (exp_x + exp_neg_x) / FixedMathCore(2);
}

FixedMathCore FixedMathCore::tanh(FixedMathCore x) {
    return sinh(x) / cosh(x);
}

// Ending of Part 1 of 4 (fixed_math_core.cpp)

// src/fixed_math_core.cpp (continued from Part 1)

// ----------------------------------------------------------------------------
// Exponential function exp(x) using Taylor series with range reduction
// exp(x) = 1 + x + x^2/2! + x^3/3! + ... (converges for all x)
// For efficiency, reduce x to [-ln2/2, ln2/2] via: exp(x) = 2^(k) * exp(r)
// where k = round(x / ln2) and r = x - k*ln2.
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::exp(FixedMathCore x) {
    if (x == FixedMathCore(0)) return FixedMathCore(1);
    
    // ln2 constant in Q32.32
    static const FixedMathCore LN2 = ln2();
    static const FixedMathCore INV_LN2 = FixedMathCore(1) / LN2;
    
    // Range reduction: k = round(x / ln2)
    int64_t k_raw = (x * INV_LN2).round().to_int();
    FixedMathCore k = FixedMathCore(k_raw);
    FixedMathCore r = x - k * LN2;
    
    // Now compute exp(r) using Taylor series up to degree 12
    // exp(r) = 1 + r + r^2/2! + r^3/3! + ... + r^12/12!
    FixedMathCore term = FixedMathCore(1);
    FixedMathCore sum = term;  // term 0 = 1
    
    FixedMathCore r_power = r;
    FixedMathCore fact = FixedMathCore(1);
    
    // Precomputed factorials: 1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600
    const int MAX_ITER = 12;
    for (int i = 1; i <= MAX_ITER; ++i) {
        fact = fact * FixedMathCore(i);
        term = r_power / fact;
        sum = sum + term;
        r_power = r_power * r;
    }
    
    // Multiply by 2^k: result = sum * 2^k
    // Since k is integer, we can shift left if k >= 0 else right
    if (k_raw >= 0) {
        return sum * (FixedMathCore(1) << static_cast<size_t>(k_raw));
    } else {
        return sum / (FixedMathCore(1) << static_cast<size_t>(-k_raw));
    }
}

// ----------------------------------------------------------------------------
// Natural logarithm log(x) for x > 0 using Newton's method on exp(y)=x
// or series expansion after range reduction.
// We'll use: log(x) = log(m * 2^e) = log(m) + e*ln2
// where m is in [0.5, 1) and e is integer exponent.
// log(m) computed via polynomial approximation.
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::log(FixedMathCore x) {
    if (x <= FixedMathCore(0)) {
        // undefined, return -inf? return 0 for safety
        return FixedMathCore(0);
    }
    if (x == FixedMathCore(1)) return FixedMathCore(0);
    
    // Extract exponent: count leading zeros in the raw value
    // We want to normalize x to range [0.5, 1) by multiplying/dividing by powers of 2.
    int64_t raw = x.raw();
    int exponent = 0;
    
    // Find the position of highest set bit (assume raw > 0)
    // Use builtin clz for performance
    if (raw > 0) {
        int leading_zeros = __builtin_clzll(raw);
        exponent = 63 - leading_zeros - 32; // because ONE is at bit 32
    } else {
        // raw is negative? shouldn't happen since x>0
        return FixedMathCore(0);
    }
    
    // Normalize m to range [0.5, 1) in fixed-point terms: m in [ONE/2, ONE)
    // m = x / 2^(exponent+1) * 2? Let's adjust.
    // Actually, we want m = x / 2^exponent where exponent chosen so that m in [0.5, 1).
    // Since ONE = 2^32, we can compute exponent relative to 32.
    int shift = exponent; // if exponent positive, right shift reduces to < ONE
    FixedMathCore m;
    if (shift >= 0) {
        m = FixedMathCore::from_raw(raw >> shift);
    } else {
        m = FixedMathCore::from_raw(raw << (-shift));
    }
    // Adjust exponent to match the normalization: we want m in [0.5, 1)
    while (m >= FixedMathCore(1)) {
        m = m / FixedMathCore(2);
        exponent++;
    }
    while (m < FixedMathCore(1) / FixedMathCore(2)) {
        m = m * FixedMathCore(2);
        exponent--;
    }
    
    // Now log(x) = log(m) + exponent * ln2
    // Compute log(m) using polynomial: log(1+y) ≈ y - y^2/2 + y^3/3 - y^4/4 ...
    // where y = m - 1 (range [-0.5, 0))
    FixedMathCore y = m - FixedMathCore(1);
    FixedMathCore y_power = y;
    FixedMathCore sum = y;
    
    // Taylor series for log(1+y) converges faster for y near 0.
    // Use up to degree 8 for sufficient precision.
    for (int i = 2; i <= 8; ++i) {
        y_power = y_power * y;
        FixedMathCore term = y_power / FixedMathCore(i);
        if (i % 2 == 0) {
            sum = sum - term;
        } else {
            sum = sum + term;
        }
    }
    
    // Add exponent * ln2
    FixedMathCore result = sum + FixedMathCore(exponent) * ln2();
    return result;
}

// ----------------------------------------------------------------------------
// log10(x) = log(x) / ln(10)
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::log10(FixedMathCore x) {
    static const FixedMathCore LN10 = FixedMathCore::from_raw(0x24D763776C1F2B1ELL); // ln(10) * 2^32
    return log(x) / LN10;
}

// ----------------------------------------------------------------------------
// Power function: x^y = exp(y * log(x))
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::pow(FixedMathCore base, FixedMathCore exponent) {
    if (base == FixedMathCore(0)) {
        if (exponent <= FixedMathCore(0)) return FixedMathCore(0); // undefined
        return FixedMathCore(0);
    }
    if (base < FixedMathCore(0)) {
        // For negative base, only integer exponents produce real results.
        // Check if exponent is integer.
        if (exponent.frac() == FixedMathCore(0)) {
            int64_t exp_int = exponent.to_int();
            FixedMathCore result = FixedMathCore(1);
            FixedMathCore b = base;
            uint64_t e = (exp_int >= 0) ? exp_int : -exp_int;
            while (e > 0) {
                if (e & 1) result = result * b;
                b = b * b;
                e >>= 1;
            }
            if (exp_int < 0) result = FixedMathCore(1) / result;
            if (exp_int % 2 != 0 && base < FixedMathCore(0)) result = -result;
            return result;
        }
        return FixedMathCore(0); // complex result, return 0
    }
    // base > 0
    return exp(exponent * log(base));
}

// ----------------------------------------------------------------------------
// Square root using Newton-Raphson iteration
// sqrt(x) = y such that y^2 ≈ x
// Iteration: y_{n+1} = (y_n + x / y_n) / 2
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::sqrt(FixedMathCore x) {
    if (x <= FixedMathCore(0)) return FixedMathCore(0);
    if (x == FixedMathCore(1)) return FixedMathCore(1);
    
    // Initial guess using bit shift approximation
    int64_t raw = x.raw();
    int shift = (64 - __builtin_clzll(raw)) / 2;
    FixedMathCore y = FixedMathCore::from_raw(1LL << (shift + 16)); // rough guess
    
    // Newton iterations (5 iterations sufficient for 32-bit fraction)
    for (int i = 0; i < 5; ++i) {
        FixedMathCore y_next = (y + x / y) / FixedMathCore(2);
        if (y_next == y) break;
        y = y_next;
    }
    return y;
}

// ----------------------------------------------------------------------------
// String conversion for FixedMathCore
// ----------------------------------------------------------------------------
std::string FixedMathCore::to_string() const {
    // Convert to decimal with fixed precision
    int64_t int_part = value_ >> FRACTIONAL_BITS;
    uint64_t frac_part_raw = value_ & (ONE - 1);
    
    // Convert fractional part to decimal digits
    // Each multiplication by 10 shifts out a decimal digit.
    std::string result = std::to_string(int_part);
    if (frac_part_raw != 0) {
        result += '.';
        uint64_t frac = frac_part_raw;
        // Up to 10 decimal digits (more than enough for ~1e-10 precision)
        for (int i = 0; i < 10 && frac != 0; ++i) {
            frac *= 10;
            uint64_t digit = frac >> FRACTIONAL_BITS;
            result += char('0' + digit);
            frac &= (ONE - 1);
        }
    }
    return result;
}

FixedMathCore FixedMathCore::from_string(const std::string& str) {
    if (str.empty()) return FixedMathCore(0);
    
    size_t pos = 0;
    bool negative = false;
    if (str[0] == '-') {
        negative = true;
        ++pos;
    } else if (str[0] == '+') {
        ++pos;
    }
    
    int64_t int_part = 0;
    while (pos < str.size() && std::isdigit(str[pos])) {
        int_part = int_part * 10 + (str[pos] - '0');
        ++pos;
    }
    
    FixedMathCore result(int_part);
    if (pos < str.size() && str[pos] == '.') {
        ++pos;
        uint64_t frac_part = 0;
        uint64_t frac_div = 1;
        while (pos < str.size() && std::isdigit(str[pos])) {
            frac_part = frac_part * 10 + (str[pos] - '0');
            frac_div *= 10;
            ++pos;
        }
        // Convert fraction to fixed-point: frac_part / frac_div
        if (frac_part > 0) {
            // Compute frac_part * 2^32 / frac_div using 128-bit arithmetic
            __int128 frac_scaled = (__int128(frac_part) << FRACTIONAL_BITS) / frac_div;
            result = result + FixedMathCore::from_raw(int64_t(frac_scaled));
        }
    }
    
    if (negative) result = -result;
    return result;
}

// ----------------------------------------------------------------------------
// Conversion to/from BigIntCore (defined here after BigIntCore is complete)
// ----------------------------------------------------------------------------
#include "big_int_core.h"

FixedMathCore FixedMathCore::from_bigint(const BigIntCore& bi) {
    // Convert BigIntCore to fixed-point by interpreting as Q32.32
    // This assumes the BigIntCore represents a raw 64-bit value shifted appropriately.
    // For simplicity, we take the least significant 64 bits and treat as raw.
    if (bi.is_zero()) return FixedMathCore(0);
    uint64_t low = bi.data()[0];
    int64_t raw = static_cast<int64_t>(low);
    if (bi.is_negative()) raw = -raw;
    return from_raw(raw);
}

BigIntCore FixedMathCore::to_bigint() const {
    // Represent raw value as BigIntCore (signed)
    BigIntCore result;
    if (value_ < 0) {
        result = BigIntCore(static_cast<uint64_t>(-value_));
        result = -result;
    } else {
        result = BigIntCore(static_cast<uint64_t>(value_));
    }
    return result;
}

// ----------------------------------------------------------------------------
// Stream operators
// ----------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const FixedMathCore& num) {
    os << num.to_string();
    return os;
}

std::istream& operator>>(std::istream& is, FixedMathCore& num) {
    std::string s;
    is >> s;
    num = FixedMathCore::from_string(s);
    return is;
}

} // namespace uep

// Ending of Part 2 of 4 (fixed_math_core.cpp)

// src/fixed_math_core.cpp (continued from Part 1)

// ----------------------------------------------------------------------------
// Exponential function exp(x) using Taylor series with range reduction
// exp(x) = 1 + x + x^2/2! + x^3/3! + ... (converges for all x)
// For efficiency, reduce x to [-ln2/2, ln2/2] via: exp(x) = 2^(k) * exp(r)
// where k = round(x / ln2) and r = x - k*ln2.
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::exp(FixedMathCore x) {
    if (x == FixedMathCore(0)) return FixedMathCore(1);
    
    // ln2 constant in Q32.32
    static const FixedMathCore LN2 = ln2();
    static const FixedMathCore INV_LN2 = FixedMathCore(1) / LN2;
    
    // Range reduction: k = round(x / ln2)
    int64_t k_raw = (x * INV_LN2).round().to_int();
    FixedMathCore k = FixedMathCore(k_raw);
    FixedMathCore r = x - k * LN2;
    
    // Now compute exp(r) using Taylor series up to degree 12
    // exp(r) = 1 + r + r^2/2! + r^3/3! + ... + r^12/12!
    FixedMathCore term = FixedMathCore(1);
    FixedMathCore sum = term;  // term 0 = 1
    
    FixedMathCore r_power = r;
    FixedMathCore fact = FixedMathCore(1);
    
    // Precomputed factorials: 1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600
    const int MAX_ITER = 12;
    for (int i = 1; i <= MAX_ITER; ++i) {
        fact = fact * FixedMathCore(i);
        term = r_power / fact;
        sum = sum + term;
        r_power = r_power * r;
    }
    
    // Multiply by 2^k: result = sum * 2^k
    // Since k is integer, we can shift left if k >= 0 else right
    if (k_raw >= 0) {
        return sum * (FixedMathCore(1) << static_cast<size_t>(k_raw));
    } else {
        return sum / (FixedMathCore(1) << static_cast<size_t>(-k_raw));
    }
}

// ----------------------------------------------------------------------------
// Natural logarithm log(x) for x > 0 using Newton's method on exp(y)=x
// or series expansion after range reduction.
// We'll use: log(x) = log(m * 2^e) = log(m) + e*ln2
// where m is in [0.5, 1) and e is integer exponent.
// log(m) computed via polynomial approximation.
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::log(FixedMathCore x) {
    if (x <= FixedMathCore(0)) {
        // undefined, return -inf? return 0 for safety
        return FixedMathCore(0);
    }
    if (x == FixedMathCore(1)) return FixedMathCore(0);
    
    // Extract exponent: count leading zeros in the raw value
    // We want to normalize x to range [0.5, 1) by multiplying/dividing by powers of 2.
    int64_t raw = x.raw();
    int exponent = 0;
    
    // Find the position of highest set bit (assume raw > 0)
    // Use builtin clz for performance
    if (raw > 0) {
        int leading_zeros = __builtin_clzll(raw);
        exponent = 63 - leading_zeros - 32; // because ONE is at bit 32
    } else {
        // raw is negative? shouldn't happen since x>0
        return FixedMathCore(0);
    }
    
    // Normalize m to range [0.5, 1) in fixed-point terms: m in [ONE/2, ONE)
    // m = x / 2^(exponent+1) * 2? Let's adjust.
    // Actually, we want m = x / 2^exponent where exponent chosen so that m in [0.5, 1).
    // Since ONE = 2^32, we can compute exponent relative to 32.
    int shift = exponent; // if exponent positive, right shift reduces to < ONE
    FixedMathCore m;
    if (shift >= 0) {
        m = FixedMathCore::from_raw(raw >> shift);
    } else {
        m = FixedMathCore::from_raw(raw << (-shift));
    }
    // Adjust exponent to match the normalization: we want m in [0.5, 1)
    while (m >= FixedMathCore(1)) {
        m = m / FixedMathCore(2);
        exponent++;
    }
    while (m < FixedMathCore(1) / FixedMathCore(2)) {
        m = m * FixedMathCore(2);
        exponent--;
    }
    
    // Now log(x) = log(m) + exponent * ln2
    // Compute log(m) using polynomial: log(1+y) ≈ y - y^2/2 + y^3/3 - y^4/4 ...
    // where y = m - 1 (range [-0.5, 0))
    FixedMathCore y = m - FixedMathCore(1);
    FixedMathCore y_power = y;
    FixedMathCore sum = y;
    
    // Taylor series for log(1+y) converges faster for y near 0.
    // Use up to degree 8 for sufficient precision.
    for (int i = 2; i <= 8; ++i) {
        y_power = y_power * y;
        FixedMathCore term = y_power / FixedMathCore(i);
        if (i % 2 == 0) {
            sum = sum - term;
        } else {
            sum = sum + term;
        }
    }
    
    // Add exponent * ln2
    FixedMathCore result = sum + FixedMathCore(exponent) * ln2();
    return result;
}

// ----------------------------------------------------------------------------
// log10(x) = log(x) / ln(10)
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::log10(FixedMathCore x) {
    static const FixedMathCore LN10 = FixedMathCore::from_raw(0x24D763776C1F2B1ELL); // ln(10) * 2^32
    return log(x) / LN10;
}

// ----------------------------------------------------------------------------
// Power function: x^y = exp(y * log(x))
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::pow(FixedMathCore base, FixedMathCore exponent) {
    if (base == FixedMathCore(0)) {
        if (exponent <= FixedMathCore(0)) return FixedMathCore(0); // undefined
        return FixedMathCore(0);
    }
    if (base < FixedMathCore(0)) {
        // For negative base, only integer exponents produce real results.
        // Check if exponent is integer.
        if (exponent.frac() == FixedMathCore(0)) {
            int64_t exp_int = exponent.to_int();
            FixedMathCore result = FixedMathCore(1);
            FixedMathCore b = base;
            uint64_t e = (exp_int >= 0) ? exp_int : -exp_int;
            while (e > 0) {
                if (e & 1) result = result * b;
                b = b * b;
                e >>= 1;
            }
            if (exp_int < 0) result = FixedMathCore(1) / result;
            if (exp_int % 2 != 0 && base < FixedMathCore(0)) result = -result;
            return result;
        }
        return FixedMathCore(0); // complex result, return 0
    }
    // base > 0
    return exp(exponent * log(base));
}

// ----------------------------------------------------------------------------
// Square root using Newton-Raphson iteration
// sqrt(x) = y such that y^2 ≈ x
// Iteration: y_{n+1} = (y_n + x / y_n) / 2
// ----------------------------------------------------------------------------
FixedMathCore FixedMathCore::sqrt(FixedMathCore x) {
    if (x <= FixedMathCore(0)) return FixedMathCore(0);
    if (x == FixedMathCore(1)) return FixedMathCore(1);
    
    // Initial guess using bit shift approximation
    int64_t raw = x.raw();
    int shift = (64 - __builtin_clzll(raw)) / 2;
    FixedMathCore y = FixedMathCore::from_raw(1LL << (shift + 16)); // rough guess
    
    // Newton iterations (5 iterations sufficient for 32-bit fraction)
    for (int i = 0; i < 5; ++i) {
        FixedMathCore y_next = (y + x / y) / FixedMathCore(2);
        if (y_next == y) break;
        y = y_next;
    }
    return y;
}

// ----------------------------------------------------------------------------
// String conversion for FixedMathCore
// ----------------------------------------------------------------------------
std::string FixedMathCore::to_string() const {
    // Convert to decimal with fixed precision
    int64_t int_part = value_ >> FRACTIONAL_BITS;
    uint64_t frac_part_raw = value_ & (ONE - 1);
    
    // Convert fractional part to decimal digits
    // Each multiplication by 10 shifts out a decimal digit.
    std::string result = std::to_string(int_part);
    if (frac_part_raw != 0) {
        result += '.';
        uint64_t frac = frac_part_raw;
        // Up to 10 decimal digits (more than enough for ~1e-10 precision)
        for (int i = 0; i < 10 && frac != 0; ++i) {
            frac *= 10;
            uint64_t digit = frac >> FRACTIONAL_BITS;
            result += char('0' + digit);
            frac &= (ONE - 1);
        }
    }
    return result;
}

FixedMathCore FixedMathCore::from_string(const std::string& str) {
    if (str.empty()) return FixedMathCore(0);
    
    size_t pos = 0;
    bool negative = false;
    if (str[0] == '-') {
        negative = true;
        ++pos;
    } else if (str[0] == '+') {
        ++pos;
    }
    
    int64_t int_part = 0;
    while (pos < str.size() && std::isdigit(str[pos])) {
        int_part = int_part * 10 + (str[pos] - '0');
        ++pos;
    }
    
    FixedMathCore result(int_part);
    if (pos < str.size() && str[pos] == '.') {
        ++pos;
        uint64_t frac_part = 0;
        uint64_t frac_div = 1;
        while (pos < str.size() && std::isdigit(str[pos])) {
            frac_part = frac_part * 10 + (str[pos] - '0');
            frac_div *= 10;
            ++pos;
        }
        // Convert fraction to fixed-point: frac_part / frac_div
        if (frac_part > 0) {
            // Compute frac_part * 2^32 / frac_div using 128-bit arithmetic
            __int128 frac_scaled = (__int128(frac_part) << FRACTIONAL_BITS) / frac_div;
            result = result + FixedMathCore::from_raw(int64_t(frac_scaled));
        }
    }
    
    if (negative) result = -result;
    return result;
}

// ----------------------------------------------------------------------------
// Conversion to/from BigIntCore with full precision and proper scaling
// ----------------------------------------------------------------------------
#include "big_int_core.h"

FixedMathCore FixedMathCore::from_bigint(const BigIntCore& bi) {
    // FixedMathCore is Q32.32: value = integer_part + fractional_part / 2^32.
    // The BigIntCore 'bi' represents the raw integer value that should be
    // interpreted as the fixed-point value multiplied by 2^32.
    // In other words: bi corresponds to (FixedMathCore * 2^32) as an integer.
    // We need to extract the integer and fractional parts by dividing by 2^32.
    
    if (bi.is_zero()) return FixedMathCore(0);
    
    // Create divisor = 2^32 as BigIntCore
    static const BigIntCore DIVISOR = BigIntCore(1) << 32;
    
    // Divide bi by DIVISOR to get integer part and remainder (fractional part * 2^32)
    BigIntCore int_part_bi, frac_part_bi;
    bi.divmod(DIVISOR, int_part_bi, frac_part_bi);
    
    // The remainder is the fractional part scaled by 2^32 (i.e., the raw fractional bits)
    uint64_t frac_raw = frac_part_bi.to_uint64(); // fractional part fits in 64 bits because divisor is 2^32
    
    // Integer part must fit within 31 bits (since FixedMathCore uses signed 32-bit integer part)
    // We check for overflow and saturate if necessary.
    int64_t int_part_signed;
    bool overflow = false;
    
    // Convert integer part BigIntCore to int64_t, clamping to [INT32_MIN, INT32_MAX]
    // because FixedMathCore integer part is 32-bit signed.
    static const BigIntCore INT32_MAX_BI = BigIntCore(0x7FFFFFFF);
    static const BigIntCore INT32_MIN_BI = -BigIntCore(0x80000000);
    
    if (int_part_bi > INT32_MAX_BI) {
        int_part_signed = 0x7FFFFFFF;
        overflow = true;
    } else if (int_part_bi < INT32_MIN_BI) {
        int_part_signed = -0x80000000LL;
        overflow = true;
    } else {
        int_part_signed = int_part_bi.to_int64(); // safe because within 32-bit range
    }
    
    // Combine integer and fractional parts into raw fixed-point value
    int64_t raw = (int_part_signed << FRACTIONAL_BITS) | (frac_raw & (ONE - 1));
    
    // Apply sign from original BigIntCore (divmod preserved sign? Actually divmod quotient sign is xor of signs,
    // remainder sign matches dividend. We need to ensure the raw value has correct sign.)
    // Since we already used signed int_part_signed, the raw value sign is correct.
    
    // If overflow occurred, we could optionally saturate to max/min raw value.
    // Here we just return the clamped raw value.
    FixedMathCore result = from_raw(raw);
    
    // If overflow, clamp to max/min representable value
    if (overflow) {
        if (bi.is_negative())
            result = FixedMathCore::from_raw(0x8000000000000000LL); // min negative
        else
            result = FixedMathCore::from_raw(0x7FFFFFFFFFFFFFFFLL); // max positive
    }
    
    return result;
}

BigIntCore FixedMathCore::to_bigint() const {
    // Convert fixed-point value to BigIntCore by multiplying by 2^32.
    // That is: value * 2^32 = (integer_part * 2^32) + fractional_part.
    // Since BigIntCore can hold arbitrarily large integers, we can represent exactly.
    
    // Extract integer and fractional parts
    int64_t int_part = value_ >> FRACTIONAL_BITS;
    uint64_t frac_part = value_ & (ONE - 1);
    
    // Create BigIntCore for integer part scaled by 2^32
    BigIntCore result = BigIntCore(int_part) * (BigIntCore(1) << 32);
    
    // Add fractional part (which is already the correct magnitude because it's the raw fractional bits)
    result += BigIntCore(frac_part);
    
    // The result is the exact representation of the fixed-point value multiplied by 2^32.
    // This preserves full precision for further arbitrary-precision operations.
    return result;
}

// ----------------------------------------------------------------------------
// Stream operators
// ----------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const FixedMathCore& num) {
    os << num.to_string();
    return os;
}

std::istream& operator>>(std::istream& is, FixedMathCore& num) {
    std::string s;
    is >> s;
    num = FixedMathCore::from_string(s);
    return is;
}

} // namespace uep

// Ending of Part 2 of 4 (fixed_math_core.cpp)