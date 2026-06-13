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

#ifndef ORTHOTREE_SERIALIZATION_STL_ARRAY_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_ARRAY_H_INCLUDED

#include "../../../core/build_config.h"
#include "../../../core/types.h"
#include "../../../core/math/numerical_methods.h"
#include "../../../detail/common.h"
#include "../../../detail/simd_utils.h"
#include "../nvp.h"
#include "../traits.h"
#include "../binary_archive.h"
#include "../msgpack_archive.h"

#include <array>
#include <cstddef>
#include <type_traits>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::array<T, N>.
//  Supports SIMD batch read/write for trivially serializable types.
//  Dynamic environment controls: batch threshold, alignment, endianness.
// ============================================================================

// ----------------------------------------------------------------------------
//  Environment controller for array serialization (tuning, alignment)
// ----------------------------------------------------------------------------
class ArraySerializationEnvironment {
public:
    static ArraySerializationEnvironment& instance() {
        static ArraySerializationEnvironment env;
        return env;
    }

    // Minimum array size to use SIMD batch operations (default: 4)
    void setBatchThreshold(size_t threshold) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_batchThreshold = threshold;
    }
    size_t batchThreshold() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_batchThreshold;
    }

    // Force SIMD alignment (if true, arrays will be aligned to SIMD boundary)
    void setForceSIMDAlignment(bool force) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_forceSIMDAlignment = force;
    }
    bool forceSIMDAlignment() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_forceSIMDAlignment;
    }

    // Enable compression for large arrays (future)
    void setEnableCompression(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableCompression = enable;
    }
    bool enableCompression() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableCompression;
    }

private:
    ArraySerializationEnvironment() : m_batchThreshold(4), m_forceSIMDAlignment(false), m_enableCompression(false) {}
    mutable std::mutex m_mutex;
    size_t m_batchThreshold;
    bool m_forceSIMDAlignment;
    bool m_enableCompression;
};

// ----------------------------------------------------------------------------
//  Helper: check if an array can be serialized as a batch (trivial + archive supports batch)
// ----------------------------------------------------------------------------
template<typename Archive, typename T, size_t N>
constexpr bool can_batch_serialize_array() {
    return is_trivially_serializable_v<T> &&
           (is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>) &&
           (N >= ArraySerializationEnvironment::instance().batchThreshold());
}

// ----------------------------------------------------------------------------
//  Main serialize function for std::array
// ----------------------------------------------------------------------------
template<typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, std::array<T, N>& arr, const unsigned int /*version*/) {
    // If the array is empty, nothing to do
    if constexpr (N == 0) return;

    // Use batch operations for large trivially serializable arrays
    if constexpr (can_batch_serialize_array<Archive, T, N>()) {
        if constexpr (is_output_archive_v<Archive>) {
            // Write the entire array in one batch (SIMD friendly)
            ar.write_batch(arr.data(), N);
        } else {
            // Read the entire array in one batch
            ar.read_batch(arr.data(), N);
        }
    } else {
        // Fallback: element‑wise serialization with names for debugging
        for (std::size_t i = 0; i < N; ++i) {
            // Use indexed name for each element (to support schema evolution)
            std::string elem_name = "elem_" + std::to_string(i);
            ar & make_nvp(elem_name.c_str(), arr[i]);
        }
    }
}

// ----------------------------------------------------------------------------
//  SIMD‑aware batch serialization of multiple arrays (for strided data)
//  Useful for arrays of structures (AoS) to structure of arrays (SoA) conversion.
// ----------------------------------------------------------------------------
template<typename Archive, typename T, std::size_t N, std::size_t Count>
void serialize_array_batch(Archive& ar, std::array<T, N> (&arrs)[Count], const unsigned int version) {
    // Treat as a contiguous block of Count * N elements
    static_assert(is_trivially_serializable_v<T>, "Batch array serialization requires trivially serializable type");
    T* data = reinterpret_cast<T*>(arrs);
    size_t total = Count * N;
    if constexpr (is_output_archive_v<Archive>) {
        ar.write_batch(data, total);
    } else {
        ar.read_batch(data, total);
    }
    (void)version;
}

// ----------------------------------------------------------------------------
//  Versioned array wrapper (for adding version tag to array serialization)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
struct versioned_array {
    std::array<T, N> data;
    unsigned int version;
};

template<typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, versioned_array<T, N>& va, const unsigned int /*version*/) {
    ar & make_nvp("version", va.version);
    ar & make_nvp("data", va.data);
}

// ============================================================================
//  Dynamic environment integration: adjust batch threshold at runtime
// ============================================================================
inline void set_array_batch_threshold(size_t threshold) {
    ArraySerializationEnvironment::instance().setBatchThreshold(threshold);
}

inline size_t get_array_batch_threshold() {
    return ArraySerializationEnvironment::instance().batchThreshold();
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_ARRAY_H_INCLUDED