/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file utils.h
 * @brief Assorted utility functions: hashing, bit twiddling, string conversion,
 *        time measurement, and miscellaneous helpers for OrthoTree.
 *
 * This file provides low‑level utilities that do not fit elsewhere:
 * - Hash combiners for custom types (e.g., for unordered_map keys)
 * - Fast integer rounding / power‑of‑two checks
 * - Compile‑time string hashing (for debug names)
 * - High‑resolution timing (std::chrono wrapper)
 * - Simple command line parser (for test executables)
 * - Pointer alignment helpers
 * - Type name demangling (for debug output)
 *
 * All functions are inline or constexpr where possible, and they are designed
 * to have zero overhead for release builds.
 */

#ifndef ORTHOTREE_DETAIL_UTILS_H_INCLUDED
#define ORTHOTREE_DETAIL_UTILS_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "common.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

namespace OrthoTree {
namespace detail {

// ============================================================================
//  Hash utilities (for custom keys in unordered maps)
// ============================================================================

/**
 * @brief Combine two hash values into one (boost::hash_combine style).
 * @param seed Reference to current hash value.
 * @param value Value to combine (will be hashed via std::hash).
 */
template <typename T>
inline void hashCombine(std::size_t& seed, const T& value) noexcept {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/**
 * @brief Hash a contiguous range of bytes.
 * @param data Pointer to bytes.
 * @param len Length in bytes.
 * @return Hash value (simple FNV‑1a).
 */
inline std::size_t hashBytes(const void* data, std::size_t len) noexcept {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    std::size_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < len; ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    }
    return hash;
}

// ============================================================================
//  Integer bit utilities
// ============================================================================

/**
 * @brief Check if an integer is a power of two (non‑zero).
 */
template <typename Int>
constexpr bool isPowerOfTwo(Int x) noexcept {
    static_assert(std::is_integral_v<Int>);
    return (x > 0) && ((x & (x - 1)) == 0);
}

/**
 * @brief Round up to the next power of two (returns 1 for x==0).
 */
template <typename Int>
constexpr Int nextPowerOfTwo(Int x) noexcept {
    static_assert(std::is_integral_v<Int>);
    if (x <= 1) return 1;
    --x;
    for (std::size_t i = 1; i < sizeof(Int) * 8; i <<= 1) {
        x |= x >> i;
    }
    return ++x;
}

/**
 * @brief Log base 2 (floor) for integers.
 */
template <typename Int>
constexpr Int log2Floor(Int x) noexcept {
    static_assert(std::is_integral_v<Int>);
    Int result = 0;
    while (x >>= 1) ++result;
    return result;
}

// ============================================================================
//  Random number generation (fast, deterministic)
// ============================================================================

/**
 * @brief Fast pseudo‑random generator (xorshift) for non‑crypto use.
 */
class FastRandom {
public:
    using result_type = uint32_t;

    explicit FastRandom(uint32_t seed = 123456789U) noexcept : m_state(seed) {}

    uint32_t operator()() noexcept {
        uint32_t x = m_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        m_state = x;
        return x;
    }

    static constexpr uint32_t min() noexcept { return 0U; }
    static constexpr uint32_t max() noexcept { return ~0U; }

private:
    uint32_t m_state;
};

/**
 * @brief Generate random float in [0,1).
 */
inline float randomFloat(FastRandom& rng) noexcept {
    return static_cast<float>(rng()) / static_cast<float>(FastRandom::max());
}

// ============================================================================
//  String utilities (debug, logging)
// ============================================================================

/**
 * @brief Demangle C++ type name (GCC/Clang).
 * @return Demangled string, or original name if demangling fails.
 */
inline std::string demangleTypeName(const char* mangled) {
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        std::free(demangled);
        return result;
    }
#endif
    return std::string(mangled);
}

/**
 * @brief Convert an integer to a hexadecimal string (for debug).
 */
template <typename Int>
std::string toHex(Int value) {
    char buffer[sizeof(Int) * 2 + 3];
    const char* digits = "0123456789ABCDEF";
    char* ptr = buffer + sizeof(buffer) - 1;
    *ptr = '\0';
    do {
        *--ptr = digits[value & 0xF];
        value >>= 4;
    } while (value != 0);
    *--ptr = 'x';
    *--ptr = '0';
    return std::string(ptr);
}

// ============================================================================
//  Time measurement (high‑resolution clock)
// ============================================================================

/**
 * @brief Simple timer for profiling.
 */
class Timer {
public:
    Timer() : m_start(clock::now()) {}

    void reset() { m_start = clock::now(); }

    double elapsedSeconds() const {
        auto end = clock::now();
        return std::chrono::duration<double>(end - m_start).count();
    }

    double elapsedMilliseconds() const {
        return elapsedSeconds() * 1000.0;
    }

    double elapsedMicroseconds() const {
        return elapsedSeconds() * 1e6;
    }

private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point m_start;
};

// ============================================================================
//  Simple command line argument parser (for test apps)
// ============================================================================

/**
 * @brief Parse command line arguments for key‑value pairs (e.g., --key value).
 */
class ArgParser {
public:
    ArgParser(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
                std::string key = arg.substr(2);
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    m_args[key] = argv[i+1];
                    ++i;
                } else {
                    m_args[key] = "";
                }
            } else {
                m_positional.push_back(arg);
            }
        }
    }

    bool has(const std::string& key) const {
        return m_args.find(key) != m_args.end();
    }

    std::string get(const std::string& key, const std::string& defaultValue = "") const {
        auto it = m_args.find(key);
        return (it != m_args.end()) ? it->second : defaultValue;
    }

    const std::vector<std::string>& positional() const { return m_positional; }

private:
    std::unordered_map<std::string, std::string> m_args;
    std::vector<std::string> m_positional;
};

// ============================================================================
//  Pointer utilities (alignment)
// ============================================================================

/**
 * @brief Align a pointer to a given alignment (must be power of two).
 */
template <typename T>
inline T* alignPointer(T* ptr, std::size_t alignment) noexcept {
    static_assert(std::is_pointer_v<T*>);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<T*>(aligned);
}

/**
 * @brief Check if pointer is aligned.
 */
template <typename T>
inline bool isAligned(const T* ptr, std::size_t alignment) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ============================================================================
//  Compile‑time string hashing (for switch on string literals)
// ============================================================================

/**
 * @brief Constexpr hash for string literals (FNV‑1a 64‑bit).
 */
constexpr uint64_t constexprHash(const char* str, uint64_t hash = 14695981039346656037ULL) {
    return (*str == '\0') ? hash : constexprHash(str + 1, (hash ^ static_cast<uint64_t>(*str)) * 1099511628211ULL);
}

// ============================================================================
//  RAII helper for temporary setting a value
// ============================================================================

/**
 * @brief Temporarily set a variable and restore on destruction.
 */
template <typename T>
class ScopedSetter {
public:
    ScopedSetter(T& var, const T& newValue) : m_var(var), m_old(var) {
        m_var = newValue;
    }
    ~ScopedSetter() { m_var = m_old; }
private:
    T& m_var;
    T m_old;
};

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_UTILS_H_INCLUDED