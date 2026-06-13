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

#ifndef ORTHOTREE_SERIALIZATION_STL_SET_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_SET_H_INCLUDED

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

#include <set>
#include <unordered_set>
#include <type_traits>
#include <cstddef>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::set (ordered set)
// ============================================================================
template<typename Archive, typename Key, typename Compare, typename Allocator>
void serialize(Archive& ar, std::set<Key, Compare, Allocator>& s, const unsigned int /*version*/) {
    using size_type = typename std::set<Key, Compare, Allocator>::size_type;
    size_type size = s.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        // Write keys sequentially
        for (const Key& key : s) {
            ar & make_nvp("key", const_cast<Key&>(key)); // output only, safe cast
        }
    } else {
        s.clear();
        for (size_type i = 0; i < size; ++i) {
            Key key;
            ar & make_nvp("key", key);
            s.insert(std::move(key));
        }
    }
}

// ============================================================================
//  Serialization for std::multiset (ordered multiset)
// ============================================================================
template<typename Archive, typename Key, typename Compare, typename Allocator>
void serialize(Archive& ar, std::multiset<Key, Compare, Allocator>& ms, const unsigned int /*version*/) {
    using size_type = typename std::multiset<Key, Compare, Allocator>::size_type;
    size_type size = ms.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        for (const Key& key : ms) {
            ar & make_nvp("key", const_cast<Key&>(key));
        }
    } else {
        ms.clear();
        for (size_type i = 0; i < size; ++i) {
            Key key;
            ar & make_nvp("key", key);
            ms.insert(std::move(key));
        }
    }
}

// ============================================================================
//  Serialization for std::unordered_set (hash set)
// ============================================================================
template<typename Archive, typename Key, typename Hash, typename KeyEqual, typename Allocator>
void serialize(Archive& ar, std::unordered_set<Key, Hash, KeyEqual, Allocator>& us, const unsigned int /*version*/) {
    using size_type = typename std::unordered_set<Key, Hash, KeyEqual, Allocator>::size_type;
    size_type size = us.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        for (const Key& key : us) {
            ar & make_nvp("key", const_cast<Key&>(key));
        }
    } else {
        us.clear();
        // Reserve space to avoid rehashing
        reserve_unordered_set(us, size);
        for (size_type i = 0; i < size; ++i) {
            Key key;
            ar & make_nvp("key", key);
            us.insert(std::move(key));
        }
    }
}

// ============================================================================
//  Serialization for std::unordered_multiset (hash multiset)
// ============================================================================
template<typename Archive, typename Key, typename Hash, typename KeyEqual, typename Allocator>
void serialize(Archive& ar, std::unordered_multiset<Key, Hash, KeyEqual, Allocator>& ums, const unsigned int /*version*/) {
    using size_type = typename std::unordered_multiset<Key, Hash, KeyEqual, Allocator>::size_type;
    size_type size = ums.size();
    ar & make_nvp("size", size);

    if constexpr (is_output_archive_v<Archive>) {
        for (const Key& key : ums) {
            ar & make_nvp("key", const_cast<Key&>(key));
        }
    } else {
        ums.clear();
        reserve_unordered_set(ums, size);
        for (size_type i = 0; i < size; ++i) {
            Key key;
            ar & make_nvp("key", key);
            ums.insert(std::move(key));
        }
    }
}

// ----------------------------------------------------------------------------
//  Helper trait to detect `reserve` method for unordered sets
// ----------------------------------------------------------------------------
template<typename T, typename = void>
struct has_reserve_method_set : std::false_type {};

template<typename T>
struct has_reserve_method_set<T, std::void_t<decltype(std::declval<T>().reserve(std::declval<size_t>()))>>
    : std::true_type {};

template<typename Set>
void reserve_unordered_set(Set& s, size_t size) {
    if constexpr (has_reserve_method_set<Set>::value) {
        if (SetSerializationEnvironment::instance().enableReserve()) {
            double factor = SetSerializationEnvironment::instance().reserveFactor();
            s.reserve(static_cast<size_t>(static_cast<double>(size) * factor));
        }
    }
}

// ============================================================================
//  Dynamic environment controller for set serialization
// ============================================================================
class SetSerializationEnvironment {
public:
    static SetSerializationEnvironment& instance() {
        static SetSerializationEnvironment env;
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

private:
    SetSerializationEnvironment() : m_enableReserve(true), m_reserveFactor(1.2) {}
    mutable std::mutex m_mutex;
    bool m_enableReserve;
    double m_reserveFactor;
};

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_SET_H_INCLUDED