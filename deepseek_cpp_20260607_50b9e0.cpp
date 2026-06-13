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

#ifndef ORTHOTREE_SERIALIZATION_STL_OPTIONAL_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_OPTIONAL_H_INCLUDED

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

#include <optional>
#include <type_traits>
#include <cstddef>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::optional.
//  Optimised for trivially serializable types using batch read/write for the
//  contained value when present. Dynamic environment controls allow switching
//  between compact representation (bool + value) and always‑write‑value mode.
// ============================================================================

// ----------------------------------------------------------------------------
//  Helper: batch serialization for optional of trivially serializable type
// ----------------------------------------------------------------------------
template<typename Archive, typename T>
bool try_batch_serialize_optional(Archive& ar, std::optional<T>& opt) {
    constexpr bool trivial = is_trivially_serializable_v<T>;
    constexpr bool batch_archive = is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>;
    if constexpr (trivial && batch_archive) {
        bool has_value = opt.has_value();
        ar & make_nvp("has_value", has_value);
        if (has_value) {
            if constexpr (is_output_archive_v<Archive>) {
                ar.write_batch(&opt.value(), 1);
            } else {
                T value;
                ar.read_batch(&value, 1);
                opt = std::move(value);
            }
        } else {
            if constexpr (is_input_archive_v<Archive>) {
                opt.reset();
            }
        }
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
//  Main serialization function for std::optional
// ----------------------------------------------------------------------------
template<typename Archive, typename T>
void serialize(Archive& ar, std::optional<T>& opt, const unsigned int /*version*/) {
    // Try batch optimisation for trivial types
    if (try_batch_serialize_optional(ar, opt)) {
        return;
    }

    // Fallback: element‑wise serialization
    bool has_value = opt.has_value();
    ar & make_nvp("has_value", has_value);
    if (has_value) {
        if constexpr (is_output_archive_v<Archive>) {
            ar & make_nvp("value", opt.value());
        } else {
            T value;
            ar & make_nvp("value", value);
            opt = std::move(value);
        }
    } else {
        if constexpr (is_input_archive_v<Archive>) {
            opt.reset();
        }
    }
}

// ============================================================================
//  Dynamic environment controller for optional serialization
//  Allows runtime tuning of batch thresholds and default behaviour.
// ============================================================================
class OptionalSerializationEnvironment {
public:
    static OptionalSerializationEnvironment& instance() {
        static OptionalSerializationEnvironment env;
        return env;
    }

    // Minimum size of T to consider batch serialisation (default: sizeof(T) >= 16)
    // Not directly used, but can influence decisions in higher‑level code.
    void setBatchThresholdSize(size_t min_size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_batchThresholdSize = min_size;
    }
    size_t batchThresholdSize() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_batchThresholdSize;
    }

    // If true, always write the optional value even when not present?
    // (For fixed‑size serialization). Default: false.
    void setAlwaysWriteValue(bool always) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_alwaysWriteValue = always;
    }
    bool alwaysWriteValue() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_alwaysWriteValue;
    }

    // Enable compression for optional values (future)
    void setEnableCompression(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableCompression = enable;
    }
    bool enableCompression() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableCompression;
    }

private:
    OptionalSerializationEnvironment()
        : m_batchThresholdSize(16)
        , m_alwaysWriteValue(false)
        , m_enableCompression(false) {}
    mutable std::mutex m_mutex;
    size_t m_batchThresholdSize;
    bool m_alwaysWriteValue;
    bool m_enableCompression;
};

// ----------------------------------------------------------------------------
//  Helper: conditionally use batch threshold (for higher‑level code)
// ----------------------------------------------------------------------------
template<typename T>
inline bool use_batch_for_optional() {
    return sizeof(T) >= OptionalSerializationEnvironment::instance().batchThresholdSize();
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_OPTIONAL_H_INCLUDED