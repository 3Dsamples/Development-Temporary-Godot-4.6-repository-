// src/fft_utils.hpp
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace fastfft {
namespace detail {

// ----------------------------------------------------------------------------
// Prime number utilities
// ----------------------------------------------------------------------------
inline bool is_prime(size_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (size_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Prime factorization (returns pairs of {prime, exponent})
// ----------------------------------------------------------------------------
inline std::vector<std::pair<size_t, size_t>> factorize(size_t n) {
    std::vector<std::pair<size_t, size_t>> factors;
    if (n <= 1) return factors;

    // Factor out 2
    size_t count = 0;
    while (n % 2 == 0) {
        n /= 2;
        ++count;
    }
    if (count > 0) factors.emplace_back(2, count);

    // Factor out odd primes
    size_t p = 3;
    while (p * p <= n) {
        count = 0;
        while (n % p == 0) {
            n /= p;
            ++count;
        }
        if (count > 0) factors.emplace_back(p, count);
        p += 2;
    }
    if (n > 1) factors.emplace_back(n, 1);
    return factors;
}

// ----------------------------------------------------------------------------
// Find the next "good" size for FFT (highly composite, small prime factors)
// ----------------------------------------------------------------------------
inline size_t next_good_size(size_t n) {
    if (n <= 6) return n;

    // List of small primes considered "good" for FFT
    const size_t good_primes[] = {2, 3, 5, 7, 11, 13};

    while (true) {
        size_t m = n;
        for (size_t p : good_primes) {
            while (m % p == 0) m /= p;
        }
        if (m == 1) return n;
        ++n;
    }
}

// ----------------------------------------------------------------------------
// Bit-reversal permutation (in-place, arbitrary length)
// ----------------------------------------------------------------------------
template<typename T>
inline void bit_reverse_permute(T* data, size_t n) {
    if (n <= 1) return;
    size_t log2_n = 0;
    size_t temp = n;
    while (temp >>= 1) ++log2_n;
    
    size_t rev = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i < rev) {
            std::swap(data[i], data[rev]);
        }
        size_t bit = n >> 1;
        while (rev & bit) {
            rev ^= bit;
            bit >>= 1;
        }
        rev ^= bit;
    }
}

// Specialized for complex data (interleaved real/imag)
template<typename T>
inline void bit_reverse_permute_complex(T* data, size_t n) {
    if (n <= 1) return;
    size_t log2_n = 0;
    size_t temp = n;
    while (temp >>= 1) ++log2_n;
    
    size_t rev = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i < rev) {
            std::swap(data[2*i], data[2*rev]);
            std::swap(data[2*i+1], data[2*rev+1]);
        }
        size_t bit = n >> 1;
        while (rev & bit) {
            rev ^= bit;
            bit >>= 1;
        }
        rev ^= bit;
    }
}

// ----------------------------------------------------------------------------
// Sine/cosine table generation for twiddle factors
// ----------------------------------------------------------------------------
template<typename T>
inline void fill_trig_table(T* table, size_t length, bool cosine_only = false) {
    const T two_pi = T(2.0 * 3.14159265358979323846);
    for (size_t i = 0; i < length; ++i) {
        T angle = -two_pi * T(i) / T(length);
        table[2*i] = std::cos(angle);
        if (!cosine_only) {
            table[2*i+1] = std::sin(angle);
        }
    }
}

// For half-complex storage (R2C output)
template<typename T>
inline void fill_half_trig_table(T* table, size_t n) {
    const T two_pi = T(2.0 * 3.14159265358979323846);
    for (size_t i = 0; i < n/2 + 1; ++i) {
        T angle = -two_pi * T(i) / T(n);
        table[2*i] = std::cos(angle);
        table[2*i+1] = std::sin(angle);
    }
}

// ----------------------------------------------------------------------------
// Alignment helpers
// ----------------------------------------------------------------------------
inline void* aligned_malloc(size_t size, size_t align) {
#ifdef _MSC_VER
    return _aligned_malloc(size, align);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, size) != 0) return nullptr;
    return ptr;
#endif
}

inline void aligned_free(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ----------------------------------------------------------------------------
// Product of all elements in an array
// ----------------------------------------------------------------------------
inline size_t prod(const std::vector<size_t>& v) {
    size_t res = 1;
    for (size_t x : v) res *= x;
    return res;
}

// ----------------------------------------------------------------------------
// Next power of two
// ----------------------------------------------------------------------------
inline size_t next_power_of_two(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// ----------------------------------------------------------------------------
// Check if a number is a power of two
// ----------------------------------------------------------------------------
inline bool is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

} // namespace detail
} // namespace fastfft