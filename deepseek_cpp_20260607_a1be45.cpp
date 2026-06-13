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

#ifndef ORTHOTREE_SERIALIZATION_STL_VECTOR_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_VECTOR_H_INCLUDED

#include "../../../core/build_config.h"
#include "../../../core/types.h"
#include "../../../core/math/numerical_methods.h"
#include "../../../detail/common.h"
#include "../../../detail/simd_utils.h"
#include "../nvp.h"
#include "../traits.h"
#include "../binary_archive.h"
#include "../msgpack_archive.h"
#include "common.h"

#include <vector>
#include <type_traits>
#include <cstddef>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::vector.
//  Supports SIMD batch read/write for trivially serializable element types.
//  Dynamic environment controls: batch threshold, reserve policy, compression.
// ============================================================================

// ----------------------------------------------------------------------------
//  Helper: batch serialization for vectors of trivially serializable types
// ----------------------------------------------------------------------------
template<typename Archive, typename T, typename Allocator>
bool try_batch_serialize_vector(Archive& ar, std::vector<T, Allocator>& vec) {
    constexpr bool trivial = is_trivially_serializable_v<T>;
    constexpr bool batch_archive = is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>;
    if constexpr (trivial && batch_archive) {
        size_t size = vec.size();
        ar & make_nvp("size", size);
        if constexpr (is_output_archive_v<Archive>) {
            ar.write_batch(vec.data(), size);
        } else {
            vec.resize(size);
            ar.read_batch(vec.data(), size);
        }
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
//  Main serialization function for std::vector
// ----------------------------------------------------------------------------
template<typename Archive, typename T, typename Allocator>
void serialize(Archive& ar, std::vector<T, Allocator>& vec, const unsigned int /*version*/) {
    // Try batch optimisation first
    if (try_batch_serialize_vector(ar, vec)) {
        return;
    }

    // Fallback: element‑wise serialization
    size_t size = vec.size();
    ar & make_nvp("size", size);
    if constexpr (is_output_archive_v<Archive>) {
        for (size_t i = 0; i < size; ++i) {
            ar & make_nvp(("elem" + std::to_string(i)).c_str(), vec[i]);
        }
    } else {
        vec.resize(size);
        for (size_t i = 0; i < size; ++i) {
            ar & make_nvp(("elem" + std::to_string(i)).c_str(), vec[i]);
        }
    }
}

// ============================================================================
//  Dynamic environment controller for vector serialization
//  Allows runtime tuning of batch thresholds and memory allocation strategies.
// ============================================================================
class VectorSerializationEnvironment {
public:
    static VectorSerializationEnvironment& instance() {
        static VectorSerializationEnvironment env;
        return env;
    }

    // Minimum size to use batch serialization (default: 64)
    void setBatchThreshold(size_t threshold) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_batchThreshold = threshold;
    }
    size_t batchThreshold() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_batchThreshold;
    }

    // Enable/disable reserve on deserialization (to avoid repeated allocations)
    void setEnableReserve(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableReserve = enable;
    }
    bool enableReserve() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableReserve;
    }

    // Reserve factor (extra capacity) when reading: capacity = size * factor
    void setReserveFactor(double factor) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_reserveFactor = factor;
    }
    double reserveFactor() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_reserveFactor;
    }

    // Enable compression for large vectors (future)
    void setEnableCompression(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableCompression = enable;
    }
    bool enableCompression() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableCompression;
    }

private:
    VectorSerializationEnvironment()
        : m_batchThreshold(64)
        , m_enableReserve(true)
        , m_reserveFactor(1.2)
        , m_enableCompression(false) {}
    mutable std::mutex m_mutex;
    size_t m_batchThreshold;
    bool m_enableReserve;
    double m_reserveFactor;
    bool m_enableCompression;
};

// ----------------------------------------------------------------------------
//  Helper: conditionally reserve capacity for vectors on deserialization
// ----------------------------------------------------------------------------
template<typename T, typename Allocator>
void vector_maybe_reserve(std::vector<T, Allocator>& vec, size_t new_size) {
    if (VectorSerializationEnvironment::instance().enableReserve()) {
        double factor = VectorSerializationEnvironment::instance().reserveFactor();
        size_t cap = static_cast<size_t>(static_cast<double>(new_size) * factor);
        if (cap > vec.capacity()) {
            vec.reserve(cap);
        }
    }
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_VECTOR_H_INCLUDED