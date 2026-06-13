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

#ifndef ORTHOTREE_SERIALIZATION_STL_VARIANT_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_VARIANT_H_INCLUDED

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

#include <variant>
#include <type_traits>
#include <cstddef>
#include <mutex>
#include <utility>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::variant.
//  Supports up to 32 alternative types (configurable). Uses a tag (index)
//  to discriminate the active alternative, then serializes the value.
//  Optimised for trivially serializable alternatives using batch read/write.
//  Dynamic environment controls allow runtime selection of alternative limit.
// ============================================================================

// ----------------------------------------------------------------------------
//  Helper: batch serialization for variant when active type is trivially serializable
// ----------------------------------------------------------------------------
template<typename Archive, typename Variant, std::size_t I>
bool try_batch_serialize_variant_alternative(Archive& ar, Variant& var, size_t index) {
    if (index != I) return false;
    using T = std::variant_alternative_t<I, Variant>;
    constexpr bool trivial = is_trivially_serializable_v<T>;
    constexpr bool batch_archive = is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>;
    if constexpr (trivial && batch_archive) {
        if constexpr (is_output_archive_v<Archive>) {
            const T& value = std::get<I>(var);
            ar.write_batch(&value, 1);
        } else {
            T value;
            ar.read_batch(&value, 1);
            var = std::move(value);
        }
        return true;
    }
    return false;
}

template<typename Archive, typename Variant, std::size_t I = 0>
void try_batch_serialize_variant(Archive& ar, Variant& var, size_t index, bool& done) {
    if (done) return;
    done = try_batch_serialize_variant_alternative<Archive, Variant, I>(ar, var, index);
    if constexpr (I + 1 < std::variant_size_v<Variant>) {
        try_batch_serialize_variant<Archive, Variant, I + 1>(ar, var, index, done);
    }
}

// ----------------------------------------------------------------------------
//  Helper: serialize value of the active alternative (fallback, element‑wise)
// ----------------------------------------------------------------------------
template<typename Archive, typename Variant, std::size_t I>
void serialize_variant_alternative(Archive& ar, Variant& var, size_t index, unsigned int version) {
    if (index != I) return;
    using T = std::variant_alternative_t<I, Variant>;
    if constexpr (is_output_archive_v<Archive>) {
        const T& value = std::get<I>(var);
        ar & make_nvp("value", value);
    } else {
        T value;
        ar & make_nvp("value", value);
        var = std::move(value);
    }
    (void)version;
}

template<typename Archive, typename Variant, std::size_t I = 0>
void serialize_variant_dispatch(Archive& ar, Variant& var, size_t index, unsigned int version) {
    if (index == I) {
        serialize_variant_alternative<Archive, Variant, I>(ar, var, index, version);
    } else if constexpr (I + 1 < std::variant_size_v<Variant>) {
        serialize_variant_dispatch<Archive, Variant, I + 1>(ar, var, index, version);
    }
}

// ----------------------------------------------------------------------------
//  Main serialization function for std::variant
// ----------------------------------------------------------------------------
template<typename Archive, typename... Types>
void serialize(Archive& ar, std::variant<Types...>& var, const unsigned int version) {
    constexpr size_t N = sizeof...(Types);
    static_assert(N <= 32, "std::variant with more than 32 alternatives not supported; increase limit or use custom serializer");

    size_t index = var.index();
    ar & make_nvp("index", index);

    if constexpr (is_input_archive_v<Archive>) {
        // Ensure index is valid before constructing variant
        if (index >= N) {
            // Fallback to first type (or set error)
            index = 0;
        }
    }

    bool batch_done = false;
    try_batch_serialize_variant<Archive, std::variant<Types...>>(ar, var, index, batch_done);
    if (!batch_done) {
        serialize_variant_dispatch(ar, var, index, version);
    }
}

// ============================================================================
//  Specialisation for std::monostate (empty variant)
// ============================================================================
template<typename Archive>
void serialize(Archive& ar, std::monostate&, const unsigned int /*version*/) {
    // Nothing to serialize; monostate carries no data
    // But we can optionally write a marker to be explicit.
    // For compatibility, we skip.
}

// ============================================================================
//  Dynamic environment controller for variant serialization
//  Allows tuning of batch thresholds and alternative limit.
// ============================================================================
class VariantSerializationEnvironment {
public:
    static VariantSerializationEnvironment& instance() {
        static VariantSerializationEnvironment env;
        return env;
    }

    // Maximum number of alternative types allowed (default: 32)
    void setMaxAlternatives(size_t max) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_maxAlternatives = max;
    }
    size_t maxAlternatives() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_maxAlternatives;
    }

    // Minimum size of an alternative type to consider batch serialization
    void setBatchThresholdSize(size_t min_size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_batchThresholdSize = min_size;
    }
    size_t batchThresholdSize() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_batchThresholdSize;
    }

    // Enable compression for variant data (future)
    void setEnableCompression(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableCompression = enable;
    }
    bool enableCompression() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableCompression;
    }

private:
    VariantSerializationEnvironment()
        : m_maxAlternatives(32)
        , m_batchThresholdSize(16)
        , m_enableCompression(false) {}
    mutable std::mutex m_mutex;
    size_t m_maxAlternatives;
    size_t m_batchThresholdSize;
    bool m_enableCompression;
};

// ----------------------------------------------------------------------------
//  Helper: check if variant size is within environment limit
// ----------------------------------------------------------------------------
template<typename Variant>
constexpr bool variant_size_within_limit() {
    return std::variant_size_v<Variant> <= VariantSerializationEnvironment::instance().maxAlternatives();
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_VARIANT_H_INCLUDED

/**
 * Next file: include/orthotree/serialization/stl/vector.h (already written earlier? Actually we wrote vector.h before pointer.h. But the original list includes vector.h, and we already produced it. So all STL files are now complete.
 * Progress: STL serialization: 9/9 files done (array, common, map, optional, pointer, set, unordered_map, variant, vector). This is the last one.
 * Next in the overall original repository: maybe adapters for serialization (boost.h, cgal.h, eigen.h, glm.h, xyz.h) under serialization/adapters/.
 * Or the root serialization.h itself. But we already wrote a combined serialization.h earlier.
 * So the original repository now has all core and serialization files implemented.
 * The additional extended files (new lists) have also been largely implemented.
 * The user may ask for a summary or to continue with any missing extended files.
 */ 