// src/big_int_core.cpp
#include "big_int_core.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cctype>
#include <cassert>

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
// Karatsuba Multiplication - Recursive Algorithm with SIMD Base Case
// ----------------------------------------------------------------------------
static void karatsuba_recursive(const limb_t* a, size_t size_a,
                                 const limb_t* b, size_t size_b,
                                 limb_t* result) {
    // Base case: if either number is small, use naive multiplication
    const size_t KARATSUBA_THRESHOLD = 32; // limbs threshold

    if (size_a <= KARATSUBA_THRESHOLD || size_b <= KARATSUBA_THRESHOLD) {
        // Naive O(n*m) multiplication with SIMD where possible
        std::memset(result, 0, (size_a + size_b) * sizeof(limb_t));
        for (size_t i = 0; i < size_a; ++i) {
            limb_t carry = 0;
            dlimb_t ai = a[i];
            size_t j = 0;

#ifdef BIGINT_SIMD_AVX512
            // AVX-512: process 8 multipliers at once
            __m512i ai_vec = _mm512_set1_epi64(ai);
            for (; j + 7 < size_b; j += 8) {
                __m512i b_vec = _mm512_loadu_si512(b + j);
                __m512i prod_low = _mm512_mullo_epi64(ai_vec, b_vec); // low 64 bits
                __m512i prod_high = _mm512_mulhi_epu64(ai_vec, b_vec); // high 64 bits

                // Load current result accumulators
                __m512i res_low = _mm512_loadu_si512(result + i + j);
                __m512i res_high = _mm512_loadu_si512(result + i + j + 1);

                // Add low product
                __m512i sum_low = _mm512_add_epi64(res_low, prod_low);
                __mmask8 carry_mask = _mm512_cmplt_epu64_mask(sum_low, res_low);
                __m512i carry_add = _mm512_maskz_set1_epi64(carry_mask, 1);

                // Add high product plus carry
                __m512i sum_high = _mm512_add_epi64(res_high, _mm512_add_epi64(prod_high, carry_add));

                // Store back
                _mm512_storeu_si512(result + i + j, sum_low);
                _mm512_storeu_si512(result + i + j + 1, sum_high);
            }
#endif

#ifdef BIGINT_SIMD_AVX2
            __m256i ai_vec = _mm256_set1_epi64x(ai);
            for (; j + 3 < size_b; j += 4) {
                __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
                // AVX2 lacks 64-bit multiply; use 32-bit mul and combine, or scalar
                // For correctness, we'll use scalar for AVX2 in this implementation
                break;
            }
#endif

#ifdef BIGINT_SIMD_NEON
            uint64x2_t ai_vec = vdupq_n_u64(ai);
            for (; j + 1 < size_b; j += 2) {
                uint64x2_t b_vec = vld1q_u64(b + j);
                // NEON: use vmull_u64 for 64x64 -> 128, but it returns two 64-bit parts
                uint64x2_t prod_low = vreinterpretq_u64_u128(vmull_u64(vget_low_u64(ai_vec), vget_low_u64(b_vec)));
                // Need to handle high parts; use scalar for simplicity
                break;
            }
#endif

            // Scalar loop for remaining limbs
            for (; j < size_b; ++j) {
                dlimb_t prod = ai * b[j] + result[i + j] + carry;
                result[i + j] = limb_t(prod);
                carry = limb_t(prod >> 64);
            }
            result[i + size_b] = carry;
        }
        return;
    }

    // Split into halves for Karatsuba
    size_t m = std::max(size_a, size_b) / 2;
    size_t a_low_size = std::min(m, size_a);
    size_t a_high_size = size_a - a_low_size;
    size_t b_low_size = std::min(m, size_b);
    size_t b_high_size = size_b - b_low_size;

    const limb_t* a_low = a;
    const limb_t* a_high = a + a_low_size;
    const limb_t* b_low = b;
    const limb_t* b_high = b + b_low_size;

    // Allocate temporary buffers (aligned)
    size_t z0_size = a_low_size + b_low_size;
    size_t z2_size = a_high_size + b_high_size;
    limb_t* z0 = AlignedLimbAllocator::allocate(z0_size);
    limb_t* z2 = AlignedLimbAllocator::allocate(z2_size);
    std::memset(z0, 0, z0_size * sizeof(limb_t));
    std::memset(z2, 0, z2_size * sizeof(limb_t));

    // Compute z0 = a_low * b_low
    karatsuba_recursive(a_low, a_low_size, b_low, b_low_size, z0);
    // Compute z2 = a_high * b_high
    karatsuba_recursive(a_high, a_high_size, b_high, b_high_size, z2);

    // Compute sums: a_sum = a_low + a_high, b_sum = b_low + b_high
    size_t a_sum_size = std::max(a_low_size, a_high_size) + 1;
    size_t b_sum_size = std::max(b_low_size, b_high_size) + 1;
    limb_t* a_sum = AlignedLimbAllocator::allocate(a_sum_size);
    limb_t* b_sum = AlignedLimbAllocator::allocate(b_sum_size);
    std::memset(a_sum, 0, a_sum_size * sizeof(limb_t));
    std::memset(b_sum, 0, b_sum_size * sizeof(limb_t));

    limb_t carry_a = BigIntCore::add_with_carry(a_sum, a_low, a_high, std::max(a_low_size, a_high_size));
    if (carry_a) a_sum[std::max(a_low_size, a_high_size)] = carry_a;
    size_t actual_a_sum = (carry_a ? a_sum_size : std::max(a_low_size, a_high_size));

    limb_t carry_b = BigIntCore::add_with_carry(b_sum, b_low, b_high, std::max(b_low_size, b_high_size));
    if (carry_b) b_sum[std::max(b_low_size, b_high_size)] = carry_b;
    size_t actual_b_sum = (carry_b ? b_sum_size : std::max(b_low_size, b_high_size));

    // Compute z1 = a_sum * b_sum
    size_t z1_size = actual_a_sum + actual_b_sum;
    limb_t* z1 = AlignedLimbAllocator::allocate(z1_size);
    std::memset(z1, 0, z1_size * sizeof(limb_t));
    karatsuba_recursive(a_sum, actual_a_sum, b_sum, actual_b_sum, z1);

    // z1 = z1 - z0 - z2
    BigIntCore::sub_with_borrow(z1, z1, z0, z0_size);
    BigIntCore::sub_with_borrow(z1, z1, z2, z2_size);

    // Combine: result = z0 + (z1 << (m*64)) + (z2 << (2*m*64))
    // Copy z0 to result
    std::memcpy(result, z0, z0_size * sizeof(limb_t));
    // Add z1 shifted by m limbs
    limb_t carry = BigIntCore::add_with_carry(result + m, result + m, z1, z1_size);
    // Add z2 shifted by 2*m limbs
    carry = BigIntCore::add_with_carry(result + 2*m, result + 2*m, z2, z2_size);
    // Propagate carry if any
    if (carry) {
        size_t idx = 2*m + z2_size;
        while (carry && idx < size_a + size_b) {
            dlimb_t sum = dlimb_t(result[idx]) + carry;
            result[idx] = limb_t(sum);
            carry = limb_t(sum >> 64);
            ++idx;
        }
    }

    // Cleanup
    AlignedLimbAllocator::deallocate(z0);
    AlignedLimbAllocator::deallocate(z2);
    AlignedLimbAllocator::deallocate(z1);
    AlignedLimbAllocator::deallocate(a_sum);
    AlignedLimbAllocator::deallocate(b_sum);
}

void BigIntCore::karatsuba_multiply(const limb_t* a, size_t size_a,
                                     const limb_t* b, size_t size_b,
                                     limb_t* result) {
    karatsuba_recursive(a, size_a, b, size_b, result);
}

// ----------------------------------------------------------------------------
// Knuth's Algorithm D - Arbitrary-Precision Division
// ----------------------------------------------------------------------------
void BigIntCore::knuth_divide(const limb_t* num, size_t num_size,
                               const limb_t* den, size_t den_size,
                               limb_t* quot, size_t& quot_size,
                               limb_t* rem, size_t& rem_size) {
    // Algorithm D from Knuth TAOCP Vol 2, Section 4.3.1
    // Assumes den_size >= 1, num_size >= den_size

    if (den_size == 0) {
        // Division by zero undefined; return zero
        quot_size = 0;
        rem_size = 0;
        return;
    }

    if (num_size < den_size) {
        // Quotient zero, remainder = numerator
        std::memcpy(rem, num, num_size * sizeof(limb_t));
        rem_size = num_size;
        quot_size = 0;
        return;
    }

    // Normalize: shift left so that high bit of divisor is 1
    size_t shift = __builtin_clzll(den[den_size - 1]); // count leading zeros
    size_t norm_num_size = num_size + (shift ? 1 : 0);
    limb_t* norm_num = AlignedLimbAllocator::allocate(norm_num_size);
    limb_t* norm_den = AlignedLimbAllocator::allocate(den_size);

    // Shift numerator left by 'shift' bits
    if (shift == 0) {
        std::memcpy(norm_num, num, num_size * sizeof(limb_t));
        std::memcpy(norm_den, den, den_size * sizeof(limb_t));
    } else {
        limb_t carry = 0;
        for (size_t i = 0; i < num_size; ++i) {
            norm_num[i] = (num[i] << shift) | carry;
            carry = num[i] >> (64 - shift);
        }
        norm_num[num_size] = carry;
        if (carry == 0) norm_num_size = num_size;

        carry = 0;
        for (size_t i = 0; i < den_size; ++i) {
            norm_den[i] = (den[i] << shift) | carry;
            carry = den[i] >> (64 - shift);
        }
    }

    limb_t v_n1 = norm_den[den_size - 1];
    limb_t v_n2 = (den_size >= 2) ? norm_den[den_size - 2] : 0;

    quot_size = num_size - den_size + 1;
    std::memset(quot, 0, quot_size * sizeof(limb_t));

    for (size_t j = num_size - den_size; j != size_t(-1); --j) {
        // Calculate q_hat
        dlimb_t u_j = dlimb_t(norm_num[j + den_size]) << 64 | norm_num[j + den_size - 1];
        limb_t q_hat = limb_t(u_j / v_n1);
        limb_t r_hat = limb_t(u_j % v_n1);

        // Adjust q_hat if too large
        while (q_hat == (dlimb_t(1) << 64) || 
               (dlimb_t(q_hat) * v_n2 > (dlimb_t(r_hat) << 64) + norm_num[j + den_size - 2])) {
            --q_hat;
            r_hat += v_n1;
            if (r_hat >= (dlimb_t(1) << 64)) break;
        }

        // Multiply and subtract
        limb_t borrow = 0;
        for (size_t i = 0; i < den_size; ++i) {
            dlimb_t prod = dlimb_t(q_hat) * norm_den[i] + borrow;
            borrow = limb_t(prod >> 64);
            dlimb_t diff = dlimb_t(norm_num[j + i]) - limb_t(prod) - borrow;
            norm_num[j + i] = limb_t(diff);
            borrow = (diff >> 64) ? 1 : 0;
        }

        // Handle borrow from the high limb
        if (borrow) {
            dlimb_t diff = dlimb_t(norm_num[j + den_size]) - borrow;
            norm_num[j + den_size] = limb_t(diff);
        }

        quot[j] = q_hat;

        // If subtraction went negative, add back divisor
        if (norm_num[j + den_size] != 0 || borrow) {
            --quot[j];
            limb_t carry = 0;
            for (size_t i = 0; i < den_size; ++i) {
                dlimb_t sum = dlimb_t(norm_num[j + i]) + norm_den[i] + carry;
                norm_num[j + i] = limb_t(sum);
                carry = limb_t(sum >> 64);
            }
            norm_num[j + den_size] += carry;
        }
    }

    // Unnormalize remainder
    rem_size = den_size;
    if (shift == 0) {
        std::memcpy(rem, norm_num, rem_size * sizeof(limb_t));
    } else {
        limb_t carry = 0;
        for (size_t i = rem_size; i-- > 0; ) {
            limb_t val = norm_num[i];
            rem[i] = (val >> shift) | carry;
            carry = val << (64 - shift);
        }
    }

    // Trim quotient and remainder
    while (quot_size > 0 && quot[quot_size - 1] == 0) --quot_size;
    while (rem_size > 0 && rem[rem_size - 1] == 0) --rem_size;

    AlignedLimbAllocator::deallocate(norm_num);
    AlignedLimbAllocator::deallocate(norm_den);
}

// ----------------------------------------------------------------------------
// Public operators implementation
// ----------------------------------------------------------------------------
BigIntCore& BigIntCore::operator*=(const BigIntCore& other) {
    if (is_zero() || other.is_zero()) {
        *this = BigIntCore();
        return *this;
    }

    size_t result_size = size_ + other.size_;
    limb_t* result = AlignedLimbAllocator::allocate(result_size);
    std::memset(result, 0, result_size * sizeof(limb_t));

    karatsuba_multiply(limbs_, size_, other.limbs_, other.size_, result);

    // Move result to this
    AlignedLimbAllocator::deallocate(limbs_);
    limbs_ = result;
    capacity_ = result_size;
    size_ = result_size;
    sign_ = sign_ ^ other.sign_;
    normalize();
    return *this;
}

BigIntCore BigIntCore::operator*(const BigIntCore& other) const {
    BigIntCore result(*this);
    result *= other;
    return result;
}

void BigIntCore::divmod(const BigIntCore& divisor, BigIntCore& quotient, BigIntCore& remainder) const {
    if (divisor.is_zero()) {
        // Division by zero
        quotient = BigIntCore();
        remainder = BigIntCore();
        return;
    }

    if (size_ < divisor.size_) {
        quotient = BigIntCore();
        remainder = *this;
        return;
    }

    size_t quot_size = size_ - divisor.size_ + 1;
    size_t rem_size = divisor.size_;
    limb_t* quot_limbs = AlignedLimbAllocator::allocate(quot_size);
    limb_t* rem_limbs = AlignedLimbAllocator::allocate(rem_size);

    knuth_divide(limbs_, size_, divisor.limbs_, divisor.size_, quot_limbs, quot_size, rem_limbs, rem_size);

    quotient = BigIntCore();
    quotient.resize(quot_size);
    std::memcpy(quotient.limbs_, quot_limbs, quot_size * sizeof(limb_t));
    quotient.sign_ = sign_ ^ divisor.sign_;
    quotient.normalize();

    remainder = BigIntCore();
    remainder.resize(rem_size);
    std::memcpy(remainder.limbs_, rem_limbs, rem_size * sizeof(limb_t));
    remainder.sign_ = sign_;
    remainder.normalize();

    AlignedLimbAllocator::deallocate(quot_limbs);
    AlignedLimbAllocator::deallocate(rem_limbs);
}

BigIntCore& BigIntCore::operator/=(const BigIntCore& other) {
    BigIntCore q, r;
    divmod(other, q, r);
    *this = std::move(q);
    return *this;
}

BigIntCore BigIntCore::operator/(const BigIntCore& other) const {
    BigIntCore result(*this);
    result /= other;
    return result;
}

BigIntCore& BigIntCore::operator%=(const BigIntCore& other) {
    BigIntCore q, r;
    divmod(other, q, r);
    *this = std::move(r);
    return *this;
}

BigIntCore BigIntCore::operator%(const BigIntCore& other) const {
    BigIntCore result(*this);
    result %= other;
    return result;
}

// ----------------------------------------------------------------------------
// String conversion
// ----------------------------------------------------------------------------
std::string BigIntCore::to_string() const {
    if (is_zero()) return "0";

    static const char digits[] = "0123456789abcdef";
    const int base = 10; // decimal output

    BigIntCore temp(*this);
    temp.sign_ = false; // work with absolute value

    std::string result;
    while (!temp.is_zero()) {
        BigIntCore q, r;
        temp.divmod(BigIntCore(base), q, r);
        uint64_t digit = r.to_uint64();
        result.push_back(digits[digit]);
        temp = std::move(q);
    }

    if (sign_) result.push_back('-');
    std::reverse(result.begin(), result.end());
    return result;
}

BigIntCore BigIntCore::from_string(const std::string& str) {
    if (str.empty()) return BigIntCore();

    size_t pos = 0;
    bool negative = false;
    if (str[0] == '-') {
        negative = true;
        ++pos;
    } else if (str[0] == '+') {
        ++pos;
    }

    BigIntCore result;
    const int base = 10;
    for (; pos < str.size(); ++pos) {
        char c = str[pos];
        if (!std::isdigit(c)) break;
        int digit = c - '0';
        result = result * base + BigIntCore(digit);
    }

    result.sign_ = negative;
    return result;
}

// ----------------------------------------------------------------------------
// Additional SIMD utility functions (if not inlined)
// ----------------------------------------------------------------------------
limb_t BigIntCore::add_with_carry(limb_t* dst, const limb_t* a, const limb_t* b, size_t n) {
    limb_t carry = 0;
    size_t i = 0;
#ifdef BIGINT_SIMD_AVX512
    for (; i + 7 < n; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        __m512i sum = _mm512_add_epi64(va, vb);
        __mmask8 carry_mask = _mm512_cmplt_epu64_mask(sum, va);
        __m512i carry_vec = _mm512_maskz_set1_epi64(carry_mask, 1);
        // Add incoming carry (simplified - full carry propagation is complex)
        // For correctness, we'll use scalar fallback for carry chain
        break;
    }
#endif
    for (; i < n; ++i) {
        dlimb_t s = dlimb_t(a[i]) + b[i] + carry;
        dst[i] = limb_t(s);
        carry = limb_t(s >> 64);
    }
    return carry;
}

limb_t BigIntCore::sub_with_borrow(limb_t* dst, const limb_t* a, const limb_t* b, size_t n) {
    limb_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
        dlimb_t diff = dlimb_t(a[i]) - b[i] - borrow;
        dst[i] = limb_t(diff);
        borrow = (diff >> 64) ? 1 : 0;
    }
    return borrow;
}

} // namespace uep

// ----------------------------------------------------------------------------
// Full SIMD-optimized addition with carry propagation (non-inline heavy path)
// ----------------------------------------------------------------------------
limb_t add_limbs_heavy(limb_t* dst, const limb_t* a, const limb_t* b, size_t count) {
    limb_t carry = 0;
    size_t i = 0;

#ifdef BIGINT_SIMD_AVX512
    // AVX-512: Process 8 limbs (512 bits) at a time with full carry chain
    // We maintain a 64-bit carry that propagates across blocks.
    __m512i zero = _mm512_setzero_si512();
    __m512i carry_vec = zero;
    
    for (; i + 7 < count; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        
        // First addition without incoming carry
        __m512i sum1 = _mm512_add_epi64(va, vb);
        // Carry out from this addition: lanes where sum1 < va
        __mmask8 carry_mask1 = _mm512_cmplt_epu64_mask(sum1, va);
        
        // Add the incoming carry (which is in the lowest lane of carry_vec)
        __m512i inc = _mm512_set1_epi64(carry);
        __m512i sum2 = _mm512_add_epi64(sum1, inc);
        // Carry out from adding the incoming carry: lanes where sum2 < sum1
        __mmask8 carry_mask2 = _mm512_cmplt_epu64_mask(sum2, sum1);
        
        // The total carry out for each lane is the OR of both masks.
        // We need to propagate these carries to higher lanes within the vector.
        // Strategy: shift the mask left by 1 and add to the next lane.
        // For AVX-512 we can use mask shift and addition with carry mask.
        
        // Create a vector of carries (1 where either mask indicates overflow)
        __mmask8 combined_mask = _kor_mask8(carry_mask1, carry_mask2);
        
        // We need to ripple the carries: if lane j overflows, add 1 to lane j+1.
        // Efficient method: use _mm512_maskz_expand_epi64 to shift carries.
        // However, expand works with mask, we'll use a loop over mask bits.
        // Since count is typically small, we can use a scalar loop for the ripple
        // but vectorized across the 8 lanes.
        
        // Store the sum first
        _mm512_storeu_si512(dst + i, sum2);
        
        // Determine outgoing carry for the block (carry from the highest lane)
        // The highest lane (index 7) will produce a carry if its combined_mask bit is set.
        carry = ((combined_mask >> 7) & 1) ? 1 : 0;
        
        // Also, if any lower lane overflowed, we need to add that carry to the next lane.
        // We can do this by shifting the combined_mask left and adding to the result.
        // This is done in a second pass. For simplicity and correctness, we'll use a
        // scalar loop within the block to propagate carries.
        alignas(64) limb_t temp[8];
        _mm512_store_si512(temp, sum2);
        alignas(64) limb_t carries[8] = {0};
        for (int k = 0; k < 8; ++k) {
            if (combined_mask & (1 << k)) {
                carries[k] = 1;
            }
        }
        // Ripple carry through the 8 limbs
        limb_t local_carry = 0;
        for (int k = 0; k < 8; ++k) {
            dlimb_t s = dlimb_t(temp[k]) + carries[k] + local_carry;
            temp[k] = limb_t(s);
            local_carry = limb_t(s >> 64);
        }
        // Update the carry for the next block: local_carry is the carry out of lane 7
        carry = local_carry;
        _mm512_store_si512(dst + i, _mm512_load_si512(temp));
    }
#endif

#ifdef BIGINT_SIMD_AVX2
    // AVX2: Process 4 limbs (256 bits) at a time with carry chain
    for (; i + 3 < count; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        
        // Add without carry-in
        __m256i sum1 = _mm256_add_epi64(va, vb);
        // Detect carry: if sum1 < va then overflow occurred
        // For unsigned comparison, we can use _mm256_cmpgt_epi64(va, sum1) but that's signed.
        // Better: compute (va > sum1) using _mm256_xor_si256 with sign bit trick.
        // We'll use a scalar fallback for AVX2 for clarity, but here's a full implementation:
        // Use _mm256_sub_epi64(va, sum1) and check if result > 0 (unsigned).
        // Since that's messy, we'll break out to scalar for AVX2 carry propagation.
        break;
    }
#endif

#ifdef BIGINT_SIMD_NEON
    // NEON: Process 2 limbs (128 bits) at a time with full carry chain
    for (; i + 1 < count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        
        uint64x2_t sum1 = vaddq_u64(va, vb);
        uint64x2_t carry_mask1 = vcltq_u64(sum1, va);
        
        uint64x2_t inc = vsetq_lane_u64(carry, vdupq_n_u64(0), 0);
        uint64x2_t sum2 = vaddq_u64(sum1, inc);
        uint64x2_t carry_mask2 = vcltq_u64(sum2, sum1);
        
        uint64x2_t total_mask = vorrq_u64(carry_mask1, carry_mask2);
        
        vst1q_u64(dst + i, sum2);
        
        // Extract flags
        uint64_t carry_low = vgetq_lane_u64(total_mask, 0) ? 1 : 0;
        uint64_t carry_high = vgetq_lane_u64(total_mask, 1) ? 1 : 0;
        
        // Propagate low carry to high lane
        if (carry_low) {
            uint64_t high = vgetq_lane_u64(sum2, 1);
            uint64_t new_high = high + 1;
            if (new_high < high) {
                carry_high = 1;
            }
            sum2 = vsetq_lane_u64(new_high, sum2, 1);
            vst1q_u64(dst + i, sum2);
        }
        carry = carry_high;
    }
#endif

    // Scalar fallback for remaining limbs or when SIMD not available
    for (; i < count; ++i) {
        dlimb_t sum = dlimb_t(a[i]) + b[i] + carry;
        dst[i] = limb_t(sum);
        carry = limb_t(sum >> 64);
    }
    return carry;
}

// ----------------------------------------------------------------------------
// Full SIMD-optimized subtraction with borrow propagation
// ----------------------------------------------------------------------------
limb_t sub_limbs_heavy(limb_t* dst, const limb_t* a, const limb_t* b, size_t count) {
    limb_t borrow = 0;
    size_t i = 0;

#ifdef BIGINT_SIMD_NEON
    for (; i + 1 < count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        
        uint64x2_t diff1 = vsubq_u64(va, vb);
        uint64x2_t borrow_mask1 = vcgtq_u64(vb, va);
        
        uint64x2_t dec = vsetq_lane_u64(borrow, vdupq_n_u64(0), 0);
        uint64x2_t diff2 = vsubq_u64(diff1, dec);
        uint64x2_t borrow_mask2 = vcgtq_u64(dec, diff1);
        
        uint64x2_t total_mask = vorrq_u64(borrow_mask1, borrow_mask2);
        
        vst1q_u64(dst + i, diff2);
        
        uint64_t borrow_low = vgetq_lane_u64(total_mask, 0) ? 1 : 0;
        uint64_t borrow_high = vgetq_lane_u64(total_mask, 1) ? 1 : 0;
        
        if (borrow_low) {
            uint64_t high = vgetq_lane_u64(diff2, 1);
            uint64_t new_high = high - 1;
            if (new_high > high) { // underflow
                borrow_high = 1;
            }
            diff2 = vsetq_lane_u64(new_high, diff2, 1);
            vst1q_u64(dst + i, diff2);
        }
        borrow = borrow_high;
    }
#endif

#ifdef BIGINT_SIMD_AVX512
    // AVX-512 subtraction with borrow chain (similar to addition)
    for (; i + 7 < count; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        
        __m512i diff1 = _mm512_sub_epi64(va, vb);
        // Borrow out: where vb > va
        __mmask8 borrow_mask1 = _mm512_cmpgt_epu64_mask(vb, va);
        
        __m512i dec = _mm512_set1_epi64(borrow);
        __m512i diff2 = _mm512_sub_epi64(diff1, dec);
        __mmask8 borrow_mask2 = _mm512_cmpgt_epu64_mask(dec, diff1);
        
        __mmask8 combined_mask = _kor_mask8(borrow_mask1, borrow_mask2);
        _mm512_storeu_si512(dst + i, diff2);
        
        // Ripple borrows across lanes
        alignas(64) limb_t temp[8];
        _mm512_store_si512(temp, diff2);
        alignas(64) limb_t borrows[8] = {0};
        for (int k = 0; k < 8; ++k) {
            if (combined_mask & (1 << k)) {
                borrows[k] = 1;
            }
        }
        limb_t local_borrow = 0;
        for (int k = 0; k < 8; ++k) {
            dlimb_t s = dlimb_t(temp[k]) - borrows[k] - local_borrow;
            temp[k] = limb_t(s);
            local_borrow = (s >> 64) ? 1 : 0;
        }
        borrow = local_borrow;
        _mm512_store_si512(dst + i, _mm512_load_si512(temp));
    }
#endif

#ifdef BIGINT_SIMD_AVX2
    // AVX2 subtraction: break to scalar for correctness
    // (Full AVX2 borrow chain would be similar to AVX-512 but with 4 lanes)
#endif

    // Scalar fallback
    for (; i < count; ++i) {
        dlimb_t diff = dlimb_t(a[i]) - b[i] - borrow;
        dst[i] = limb_t(diff);
        borrow = (diff >> 64) ? 1 : 0;
    }
    return borrow;
}

// ----------------------------------------------------------------------------
// Public static wrappers that call the heavy implementations
// These are used by the inlined functions if they need the full optimized path.
// For simplicity, the inlined functions may call these non-inline versions
// when they detect a block size large enough to warrant the overhead.
// ----------------------------------------------------------------------------
limb_t BigIntCore::add_limbs(limb_t* dst, const limb_t* a, const limb_t* b, size_t count) {
    // Dispatch to heavy implementation for better optimization
    return add_limbs_heavy(dst, a, b, count);
}

limb_t BigIntCore::sub_limbs(limb_t* dst, const limb_t* a, const limb_t* b, size_t count) {
    return sub_limbs_heavy(dst, a, b, count);
}

// ----------------------------------------------------------------------------
// Additional bitwise operations that may have been declared but not defined
// (Most were inlined, but we provide non-inline fallbacks if needed)
// ----------------------------------------------------------------------------
BigIntCore BigIntCore::operator~() const {
    // Already inlined, but provide out-of-line version if needed
    BigIntCore result(*this);
    if (result.size_ == 0) {
        result.resize(1);
        result.limbs_[0] = ~limb_t(0);
        result.sign_ = true;
        return result;
    }
    for (size_t i = 0; i < result.size_; ++i) {
        result.limbs_[i] = ~result.limbs_[i];
    }
    result.normalize();
    result.sign_ = !result.sign_;
    return result;
}

// ----------------------------------------------------------------------------
// String conversion: additional bases (hex, binary) if needed
// ----------------------------------------------------------------------------
std::string BigIntCore::to_hex_string() const {
    if (is_zero()) return "0";
    static const char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(size_ * 16); // each limb up to 16 hex digits
    for (size_t i = size_; i-- > 0; ) {
        limb_t val = limbs_[i];
        for (int j = 60; j >= 0; j -= 4) {
            result.push_back(digits[(val >> j) & 0xF]);
        }
    }
    // Remove leading zeros
    size_t pos = result.find_first_not_of('0');
    if (pos != std::string::npos) {
        result = result.substr(pos);
    }
    if (sign_) result.insert(0, "-");
    return result;
}

BigIntCore BigIntCore::from_hex_string(const std::string& str) {
    if (str.empty()) return BigIntCore();
    size_t pos = 0;
    bool negative = false;
    if (str[0] == '-') {
        negative = true;
        ++pos;
    }
    BigIntCore result;
    for (; pos < str.size(); ++pos) {
        char c = str[pos];
        int digit;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else break;
        result = (result << 4) + BigIntCore(digit);
    }
    result.sign_ = negative;
    return result;
}

} // namespace uep

// ... continuing from Part 2 ...

// ----------------------------------------------------------------------------
// Additional helper: conversion to/from binary string
// ----------------------------------------------------------------------------
std::string BigIntCore::to_binary_string() const {
    if (is_zero()) return "0";
    std::string result;
    result.reserve(size_ * 64);
    for (size_t i = size_; i-- > 0; ) {
        limb_t val = limbs_[i];
        for (int j = 63; j >= 0; --j) {
            result.push_back(((val >> j) & 1) ? '1' : '0');
        }
    }
    size_t pos = result.find_first_not_of('0');
    if (pos != std::string::npos) {
        result = result.substr(pos);
    }
    if (sign_) result.insert(0, "-");
    return result;
}

BigIntCore BigIntCore::from_binary_string(const std::string& str) {
    if (str.empty()) return BigIntCore();
    size_t pos = 0;
    bool negative = false;
    if (str[0] == '-') {
        negative = true;
        ++pos;
    }
    BigIntCore result;
    for (; pos < str.size(); ++pos) {
        char c = str[pos];
        if (c != '0' && c != '1') break;
        result = (result << 1) + BigIntCore(c == '1' ? 1 : 0);
    }
    result.sign_ = negative;
    return result;
}

// ----------------------------------------------------------------------------
// Power function (exponentiation by squaring)
// ----------------------------------------------------------------------------
BigIntCore pow(const BigIntCore& base, uint64_t exponent) {
    if (exponent == 0) return BigIntCore(1);
    BigIntCore result(1);
    BigIntCore current = base;
    uint64_t e = exponent;
    while (e > 0) {
        if (e & 1) {
            result *= current;
        }
        current *= current;
        e >>= 1;
    }
    return result;
}

// ----------------------------------------------------------------------------
// Greatest Common Divisor (Euclidean algorithm)
// ----------------------------------------------------------------------------
BigIntCore gcd(const BigIntCore& a, const BigIntCore& b) {
    if (b.is_zero()) return a;
    BigIntCore x = a;
    BigIntCore y = b;
    // Make positive
    x.sign_ = false;
    y.sign_ = false;
    while (!y.is_zero()) {
        BigIntCore temp = y;
        y = x % y;
        x = std::move(temp);
    }
    return x;
}

// ----------------------------------------------------------------------------
// Least Common Multiple
// ----------------------------------------------------------------------------
BigIntCore lcm(const BigIntCore& a, const BigIntCore& b) {
    if (a.is_zero() || b.is_zero()) return BigIntCore();
    BigIntCore g = gcd(a, b);
    return (a / g) * b;
}

// ----------------------------------------------------------------------------
// Modular exponentiation (used in cryptography)
// ----------------------------------------------------------------------------
BigIntCore mod_pow(const BigIntCore& base, const BigIntCore& exponent, const BigIntCore& modulus) {
    if (modulus.is_zero()) return BigIntCore();
    BigIntCore result(1);
    BigIntCore b = base % modulus;
    BigIntCore e = exponent;
    while (!e.is_zero()) {
        if ((e.limbs_[0] & 1) && !e.is_zero()) { // check if odd
            result = (result * b) % modulus;
        }
        b = (b * b) % modulus;
        e >>= 1;
    }
    return result;
}

// ----------------------------------------------------------------------------
// Square root (integer floor)
// ----------------------------------------------------------------------------
BigIntCore sqrt(const BigIntCore& n) {
    if (n.is_zero()) return BigIntCore(0);
    if (n < BigIntCore(0)) return BigIntCore(); // undefined for negative
    BigIntCore x = n >> 1;
    if (x.is_zero()) x = BigIntCore(1);
    BigIntCore y = (x + n / x) >> 1;
    while (y < x) {
        x = y;
        y = (x + n / x) >> 1;
    }
    return x;
}

// ----------------------------------------------------------------------------
// Factorial
// ----------------------------------------------------------------------------
BigIntCore factorial(uint64_t n) {
    BigIntCore result(1);
    for (uint64_t i = 2; i <= n; ++i) {
        result *= BigIntCore(i);
    }
    return result;
}

// ----------------------------------------------------------------------------
// Random number generation (platform-specific seed)
// ----------------------------------------------------------------------------
#include <random>
BigIntCore random_bigint(size_t bits) {
    if (bits == 0) return BigIntCore(0);
    size_t limbs_needed = (bits + 63) / 64;
    BigIntCore result;
    result.resize(limbs_needed);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < limbs_needed; ++i) {
        result.limbs_[i] = dist(gen);
    }
    // Mask the top limb to exact bit length
    if (bits % 64 != 0) {
        limb_t mask = (limb_t(1) << (bits % 64)) - 1;
        result.limbs_[limbs_needed - 1] &= mask;
    }
    result.normalize();
    return result;
}

// ----------------------------------------------------------------------------
// Miller-Rabin primality test
// ----------------------------------------------------------------------------
bool is_probable_prime(const BigIntCore& n, int rounds = 40) {
    if (n < BigIntCore(2)) return false;
    if (n == BigIntCore(2) || n == BigIntCore(3)) return true;
    if ((n.limbs_[0] & 1) == 0) return false; // even

    // Write n-1 = d * 2^s
    BigIntCore d = n - BigIntCore(1);
    size_t s = 0;
    while ((d.limbs_[0] & 1) == 0) {
        d >>= 1;
        ++s;
    }

    for (int round = 0; round < rounds; ++round) {
        BigIntCore a = random_bigint(n.size() * 64) % (n - BigIntCore(2)) + BigIntCore(2);
        BigIntCore x = mod_pow(a, d, n);
        if (x == BigIntCore(1) || x == n - BigIntCore(1)) continue;
        bool composite = true;
        for (size_t r = 0; r < s - 1; ++r) {
            x = (x * x) % n;
            if (x == n - BigIntCore(1)) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Stream output operator
// ----------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const BigIntCore& num) {
    os << num.to_string();
    return os;
}

// ----------------------------------------------------------------------------
// Stream input operator
// ----------------------------------------------------------------------------
std::istream& operator>>(std::istream& is, BigIntCore& num) {
    std::string s;
    is >> s;
    num = BigIntCore::from_string(s);
    return is;
}

} // namespace uep

// Ending of Part 3 of 3 (big_int_core.cpp complete)