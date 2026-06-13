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

#ifndef ORTHOTREE_SERIALIZATION_STL_MAP_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_MAP_H_INCLUDED

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

#include <map>
#include <unordered_map>
#include <type_traits>
#include <cstddef>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::map (ordered map)
// ============================================================================
template<typename Archive, typename Key, typename T, typename Compare, typename Allocator>
void serialize(Archive& ar, std::map<Key, T, Compare, Allocator>& m, const unsigned int /*version*/) {
    using size_type = typename std::map<Key, T, Compare, Allocator>::size_type;
    size_type size = m.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        // Write key‑value pairs sequentially
        for (auto& pair : m) {
            // Use const_cast to avoid copying – safe for output
            ar & make_nvp("key", const_cast<Key&>(pair.first));
            ar & make_nvp("value", pair.second);
        }
    } else {
        m.clear();
        // Reserve space if possible (only if allocator supports)
        if constexpr (has_reserve_method_v<decltype(m)>) {
            m.reserve(static_cast<size_type>(size));
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
//  Serialization for std::multimap (ordered multimap)
// ============================================================================
template<typename Archive, typename Key, typename T, typename Compare, typename Allocator>
void serialize(Archive& ar, std::multimap<Key, T, Compare, Allocator>& mm, const unsigned int /*version*/) {
    using size_type = typename std::multimap<Key, T, Compare, Allocator>::size_type;
    size_type size = mm.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        for (auto& pair : mm) {
            ar & make_nvp("key", const_cast<Key&>(pair.first));
            ar & make_nvp("value", pair.second);
        }
    } else {
        mm.clear();
        for (size_type i = 0; i < size; ++i) {
            Key key;
            T value;
            ar & make_nvp("key", key);
            ar & make_nvp("value", value);
            mm.emplace(std::move(key), std::move(value));
        }
    }
}

// ============================================================================
//  Serialization for std::unordered_map (hash map)
// ============================================================================
template<typename Archive, typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
void serialize(Archive& ar, std::unordered_map<Key, T, Hash, KeyEqual, Allocator>& um, const unsigned int /*version*/) {
    using size_type = typename std::unordered_map<Key, T, Hash, KeyEqual, Allocator>::size_type;
    size_type size = um.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        for (auto& pair : um) {
            ar & make_nvp("key", const_cast<Key&>(pair.first));
            ar & make_nvp("value", pair.second);
        }
    } else {
        um.clear();
        // Reserve to avoid rehashing
        um.reserve(static_cast<size_type>(size * 1.2)); // 20% extra
        for (size_type i = 0; i < size; ++i) {
            Key key;
            T value;
            ar & make_nvp("key", key);
            ar & make_nvp("value", value);
            um.emplace(std::move(key), std::move(value));
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
        umm.reserve(static_cast<size_type>(size * 1.2));
        for (size_type i = 0; i < size; ++i) {
            Key key;
            T value;
            ar & make_nvp("key", key);
            ar & make_nvp("value", value);
            umm.emplace(std::move(key), std::move(value));
        }
    }
}

// ----------------------------------------------------------------------------
//  Helper trait to detect `reserve` method (for unordered containers)
// ----------------------------------------------------------------------------
template<typename T, typename = void>
struct has_reserve_method : std::false_type {};

template<typename T>
struct has_reserve_method<T, std::void_t<decltype(std::declval<T>().reserve(std::declval<size_t>()))>>
    : std::true_type {};

// ============================================================================
//  Dynamic environment controller for map serialization
// ============================================================================
class MapSerializationEnvironment {
public:
    static MapSerializationEnvironment& instance() {
        static MapSerializationEnvironment env;
        return env;
    }

    // Enable/disable reserve for unordered maps (to avoid rehashing)
    void setEnableReserve(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableReserve = enable;
    }
    bool enableReserve() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableReserve;
    }

    // Load factor for reserve (default 1.2)
    void setReserveFactor(double factor) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_reserveFactor = factor;
    }
    double reserveFactor() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_reserveFactor;
    }

private:
    MapSerializationEnvironment() : m_enableReserve(true), m_reserveFactor(1.2) {}
    mutable std::mutex m_mutex;
    bool m_enableReserve;
    double m_reserveFactor;
};

// ----------------------------------------------------------------------------
//  Helper: conditionally reserve capacity for unordered maps
// ----------------------------------------------------------------------------
template<typename Map>
void reserve_unordered_map(Map& m, size_t size) {
    if constexpr (has_reserve_method_v<Map>) {
        if (MapSerializationEnvironment::instance().enableReserve()) {
            double factor = MapSerializationEnvironment::instance().reserveFactor();
            m.reserve(static_cast<size_t>(static_cast<double>(size) * factor));
        }
    }
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_MAP_H_INCLUDED