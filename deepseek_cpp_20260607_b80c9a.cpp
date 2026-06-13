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

#ifndef ORTHOTREE_SERIALIZATION_STL_UNORDERED_MAP_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_UNORDERED_MAP_H_INCLUDED

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

#include <unordered_map>
#include <type_traits>
#include <cstddef>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::unordered_map (hash map)
//  Supports batch key/value serialization, reserve control, and SIMD
//  for trivially serializable keys/values (if both are trivial).
// ============================================================================

// ----------------------------------------------------------------------------
//  Helper: check if a type is a std::unordered_map (for SFINAE)
// ----------------------------------------------------------------------------
template<typename T>
struct is_unordered_map : std::false_type {};

template<typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
struct is_unordered_map<std::unordered_map<Key, T, Hash, KeyEqual, Allocator>> : std::true_type {};

// ----------------------------------------------------------------------------
//  Batch optimisation for unordered_map when keys and values are trivial
//  We read/write the size, then a contiguous block of key‑value pairs.
//  This is only possible if the archive supports batch reads/writes.
// ----------------------------------------------------------------------------
template<typename Archive, typename Map>
bool try_batch_serialize_unordered_map(Archive& ar, Map& m) {
    using key_type = typename Map::key_type;
    using mapped_type = typename Map::mapped_type;
    using value_type = std::pair<key_type, mapped_type>;

    constexpr bool key_trivial = is_trivially_serializable_v<key_type>;
    constexpr bool value_trivial = is_trivially_serializable_v<mapped_type>;

    if constexpr (key_trivial && value_trivial &&
                  (is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>)) {
        size_t size = m.size();
        ar & make_nvp("size", size);
        if constexpr (is_output_archive_v<Archive>) {
            // Write as a single batch of key‑value pairs (contiguous in memory not guaranteed)
            // Instead, we write keys and values separately as two arrays.
            std::vector<key_type> keys;
            std::vector<mapped_type> values;
            keys.reserve(size);
            values.reserve(size);
            for (const auto& pair : m) {
                keys.push_back(pair.first);
                values.push_back(pair.second);
            }
            ar & make_nvp("keys", keys);
            ar & make_nvp("values", values);
        } else {
            std::vector<key_type> keys;
            std::vector<mapped_type> values;
            ar & make_nvp("keys", keys);
            ar & make_nvp("values", values);
            m.clear();
            // Reserve space
            if (UnorderedMapSerializationEnvironment::instance().enableReserve()) {
                double factor = UnorderedMapSerializationEnvironment::instance().reserveFactor();
                m.reserve(static_cast<size_t>(static_cast<double>(keys.size()) * factor));
            }
            for (size_t i = 0; i < keys.size(); ++i) {
                m.emplace(std::move(keys[i]), std::move(values[i]));
            }
        }
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
//  Main serialization function for std::unordered_map
// ----------------------------------------------------------------------------
template<typename Archive, typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
void serialize(Archive& ar, std::unordered_map<Key, T, Hash, KeyEqual, Allocator>& m, const unsigned int /*version*/) {
    // Try batch optimisation
    if (try_batch_serialize_unordered_map(ar, m)) {
        return;
    }

    // Fallback: element‑wise serialization
    using size_type = typename std::unordered_map<Key, T, Hash, KeyEqual, Allocator>::size_type;
    size_type size = m.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        for (auto& pair : m) {
            ar & make_nvp("key", const_cast<Key&>(pair.first));
            ar & make_nvp("value", pair.second);
        }
    } else {
        m.clear();
        // Reserve capacity to avoid rehashing
        if (UnorderedMapSerializationEnvironment::instance().enableReserve()) {
            double factor = UnorderedMapSerializationEnvironment::instance().reserveFactor();
            m.reserve(static_cast<size_type>(static_cast<double>(size) * factor));
        }
        for (size_type i = 0; i < size; ++i) {
            Key key;
            T value;
            ar & make_nvp("key", key);
            ar & make_nvp("value", value);
            m.emplace(std::move(key), std::move(value));
        }
    }
}

// ============================================================================
//  Serialization for std::unordered_multimap (hash multimap)
// ============================================================================
template<typename Archive, typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
void serialize(Archive& ar, std::unordered_multimap<Key, T, Hash, KeyEqual, Allocator>& umm, const unsigned int /*version*/) {
    using size_type = typename std::unordered_multimap<Key, T, Hash, KeyEqual, Allocator>::size_type;
    size_type size = umm.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        for (auto& pair : umm) {
            ar & make_nvp("key", const_cast<Key&>(pair.first));
            ar & make_nvp("value", pair.second);
        }
    } else {
        umm.clear();
        if (UnorderedMapSerializationEnvironment::instance().enableReserve()) {
            double factor = UnorderedMapSerializationEnvironment::instance().reserveFactor();
            umm.reserve(static_cast<size_type>(static_cast<double>(size) * factor));
        }
        for (size_type i = 0; i < size; ++i) {
            Key key;
            T value;
            ar & make_nvp("key", key);
            ar & make_nvp("value", value);
            umm.emplace(std::move(key), std::move(value));
        }
    }
}

// ============================================================================
//  Dynamic environment controller for unordered_map serialization
//  Controls reserve factor, batch threshold, and SIMD usage.
// ============================================================================
class UnorderedMapSerializationEnvironment {
public:
    static UnorderedMapSerializationEnvironment& instance() {
        static UnorderedMapSerializationEnvironment env;
        return env;
    }

    void setEnableReserve(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableReserve = enable;
    }
    bool enableReserve() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableReserve;
    }

    void setReserveFactor(double factor) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_reserveFactor = factor;
    }
    double reserveFactor() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_reserveFactor;
    }

    // Minimum size to use batch optimization (default: 128)
    void setBatchThreshold(size_t threshold) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_batchThreshold = threshold;
    }
    size_t batchThreshold() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_batchThreshold;
    }

    // Enable SIMD batch for key‑value pairs (if supported)
    void setEnableSIMDBatch(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableSIMDBatch = enable;
    }
    bool enableSIMDBatch() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableSIMDBatch;
    }

private:
    UnorderedMapSerializationEnvironment()
        : m_enableReserve(true)
        , m_reserveFactor(1.2)
        , m_batchThreshold(128)
        , m_enableSIMDBatch(true) {}
    mutable std::mutex m_mutex;
    bool m_enableReserve;
    double m_reserveFactor;
    size_t m_batchThreshold;
    bool m_enableSIMDBatch;
};

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_UNORDERED_MAP_H_INCLUDED