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

#ifndef ORTHOTREE_SERIALIZATION_STL_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/numerical_methods.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"
#include "nvp.h"
#include "traits.h"
#include "binary_archive.h"
#include "msgpack_archive.h"

#include <vector>
#include <list>
#include <deque>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <string>
#include <optional>
#include <variant>
#include <tuple>
#include <utility>
#include <memory>
#include <type_traits>
#include <cstddef>
#include <algorithm>
#include <iterator>
#include <functional>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for std::pair
// ============================================================================
template<typename Archive, typename T1, typename T2>
void serialize(Archive& ar, std::pair<T1, T2>& p, const unsigned int version) {
    ar & make_nvp("first", p.first);
    ar & make_nvp("second", p.second);
    (void)version; // unused
}

// ============================================================================
//  Serialization for std::tuple (up to 10 elements, can be extended)
// ============================================================================
namespace detail {
    template<typename Archive, typename Tuple, std::size_t... I>
    void serialize_tuple_impl(Archive& ar, Tuple& t, std::index_sequence<I...>, unsigned int) {
        ( (ar & make_nvp(("v" + std::to_string(I)).c_str(), std::get<I>(t))), ... );
    }
}

template<typename Archive, typename... Args>
void serialize(Archive& ar, std::tuple<Args...>& t, const unsigned int version) {
    detail::serialize_tuple_impl(ar, t, std::index_sequence_for<Args...>{}, version);
}

// ============================================================================
//  Serialization for std::vector (with SIMD batch optimization)
// ============================================================================
template<typename Archive, typename T, typename Allocator>
void serialize(Archive& ar, std::vector<T, Allocator>& vec, const unsigned int /*version*/) {
    size_t size = vec.size();
    ar & make_nvp("size", size);
    if constexpr (is_input_archive_v<Archive>) {
        vec.resize(size);
    }
    if constexpr (is_trivially_serializable_v<T> && (is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>)) {
        // Use batch read/write for arithmetic types
        if constexpr (is_output_archive_v<Archive>) {
            ar.write_batch(vec.data(), size);
        } else {
            ar.read_batch(vec.data(), size);
        }
    } else {
        // Sequential serialization
        for (size_t i = 0; i < size; ++i) {
            ar & make_nvp(("elem" + std::to_string(i)).c_str(), vec[i]);
        }
    }
}

// ============================================================================
//  Serialization for std::deque (no SIMD batch, but element-wise)
// ============================================================================
template<typename Archive, typename T, typename Allocator>
void serialize(Archive& ar, std::deque<T, Allocator>& dq, const unsigned int /*version*/) {
    size_t size = dq.size();
    ar & make_nvp("size", size);
    if constexpr (is_input_archive_v<Archive>) {
        dq.resize(size);
    }
    auto it = dq.begin();
    for (size_t i = 0; i < size; ++i, ++it) {
        ar & make_nvp(("elem" + std::to_string(i)).c_str(), *it);
    }
}

// ============================================================================
//  Serialization for std::list (sequential, no random access)
// ============================================================================
template<typename Archive, typename T, typename Allocator>
void serialize(Archive& ar, std::list<T, Allocator>& lst, const unsigned int /*version*/) {
    size_t size = lst.size();
    ar & make_nvp("size", size);
    if constexpr (is_input_archive_v<Archive>) {
        lst.clear();
        for (size_t i = 0; i < size; ++i) {
            T value;
            ar & make_nvp(("elem" + std::to_string(i)).c_str(), value);
            lst.push_back(std::move(value));
        }
    } else {
        auto it = lst.begin();
        for (size_t i = 0; i < size; ++i, ++it) {
            ar & make_nvp(("elem" + std::to_string(i)).c_str(), *it);
        }
    }
}

// ============================================================================
//  Serialization for associative containers (std::set, std::map, etc.)
// ============================================================================
template<typename Archive, typename Key, typename Compare, typename Allocator>
void serialize(Archive& ar, std::set<Key, Compare, Allocator>& s, const unsigned int /*version*/) {
    size_t size = s.size();
    ar & make_nvp("size", size);
    if constexpr (is_input_archive_v<Archive>) {
        s.clear();
        for (size_t i = 0; i < size; ++i) {
            Key k;
            ar & make_nvp(("key" + std::to_string(i)).c_str(), k);
            s.insert(std::move(k));
        }
    } else {
        auto it = s.begin();
        for (size_t i = 0; i < size; ++i, ++it) {
            ar & make_nvp(("key" + std::to_string(i)).c_str(), const_cast<Key&>(*it));
        }
    }
}

template<typename Archive, typename Key, typename T, typename Compare, typename Allocator>
void serialize(Archive& ar, std::map<Key, T, Compare, Allocator>& m, const unsigned int /*version*/) {
    size_t size = m.size();
    ar & make_nvp("size", size);
    if constexpr (is_input_archive_v<Archive>) {
        m.clear();
        for (size_t i = 0; i < size; ++i) {
            Key k;
            T v;
            ar & make_nvp(("key" + std::to_string(i)).c_str(), k);
            ar & make_nvp(("value" + std::to_string(i)).c_str(), v);
            m.emplace(std::move(k), std::move(v));
        }
    } else {
        auto it = m.begin();
        for (size_t i = 0; i < size; ++i, ++it) {
            ar & make_nvp(("key" + std::to_string(i)).c_str(), const_cast<Key&>(it->first));
            ar & make_nvp(("value" + std::to_string(i)).c_str(), it->second);
        }
    }
}

// Unordered set / map (similar to ordered, but no comparison)
template<typename Archive, typename Key, typename Hash, typename KeyEqual, typename Allocator>
void serialize(Archive& ar, std::unordered_set<Key, Hash, KeyEqual, Allocator>& us, const unsigned int /*version*/) {
    size_t size = us.size();
    ar & make_nvp("size", size);
    if constexpr (is_input_archive_v<Archive>) {
        us.clear();
        us.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            Key k;
            ar & make_nvp(("key" + std::to_string(i)).c_str(), k);
            us.insert(std::move(k));
        }
    } else {
        auto it = us.begin();
        for (size_t i = 0; i < size; ++i, ++it) {
            ar & make_nvp(("key" + std::to_string(i)).c_str(), const_cast<Key&>(*it));
        }
    }
}

template<typename Archive, typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
void serialize(Archive& ar, std::unordered_map<Key, T, Hash, KeyEqual, Allocator>& um, const unsigned int /*version*/) {
    size_t size = um.size();
    ar & make_nvp("size", size);
    if constexpr (is_input_archive_v<Archive>) {
        um.clear();
        um.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            Key k;
            T v;
            ar & make_nvp(("key" + std::to_string(i)).c_str(), k);
            ar & make_nvp(("value" + std::to_string(i)).c_str(), v);
            um.emplace(std::move(k), std::move(v));
        }
    } else {
        auto it = um.begin();
        for (size_t i = 0; i < size; ++i, ++it) {
            ar & make_nvp(("key" + std::to_string(i)).c_str(), const_cast<Key&>(it->first));
            ar & make_nvp(("value" + std::to_string(i)).c_str(), it->second);
        }
    }
}

// ============================================================================
//  Serialization for std::array (fixed size)
// ============================================================================
template<typename Archive, typename T, std::size_t N>
void serialize(Archive& ar, std::array<T, N>& arr, const unsigned int /*version*/) {
    if constexpr (is_trivially_serializable_v<T> && (is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>)) {
        if constexpr (is_output_archive_v<Archive>) {
            ar.write_batch(arr.data(), N);
        } else {
            ar.read_batch(arr.data(), N);
        }
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            ar & make_nvp(("elem" + std::to_string(i)).c_str(), arr[i]);
        }
    }
}

// ============================================================================
//  Serialization for std::string (specialised for efficiency)
// ============================================================================
template<typename Archive>
void serialize(Archive& ar, std::string& str, const unsigned int /*version*/) {
    size_t size = str.size();
    ar & make_nvp("size", size);
    if constexpr (is_output_archive_v<Archive>) {
        ar.write_bytes(reinterpret_cast<const uint8_t*>(str.data()), size);
    } else {
        str.resize(size);
        ar.read_bytes(reinterpret_cast<uint8_t*>(&str[0]), size);
    }
}

// ============================================================================
//  Serialization for std::optional
// ============================================================================
template<typename Archive, typename T>
void serialize(Archive& ar, std::optional<T>& opt, const unsigned int /*version*/) {
    bool has_value = opt.has_value();
    ar & make_nvp("has_value", has_value);
    if (has_value) {
        if constexpr (is_output_archive_v<Archive>) {
            T& val = opt.value();
            ar & make_nvp("value", val);
        } else {
            T val;
            ar & make_nvp("value", val);
            opt = std::move(val);
        }
    } else {
        if constexpr (is_input_archive_v<Archive>) {
            opt.reset();
        }
    }
}

// ============================================================================
//  Serialization for std::variant (using index and value)
// ============================================================================
namespace detail {
    template<typename Archive, typename Variant, std::size_t I>
    void serialize_variant_impl(Archive& ar, Variant& var, size_t index, unsigned int version) {
        if (index == I) {
            using T = std::variant_alternative_t<I, Variant>;
            T value;
            if constexpr (is_output_archive_v<Archive>) {
                value = std::get<I>(var);
                ar & make_nvp("value", value);
            } else {
                ar & make_nvp("value", value);
                var = std::move(value);
            }
        } else {
            serialize_variant_impl<Archive, Variant, I+1>(ar, var, index, version);
        }
    }
    template<typename Archive, typename Variant>
    void serialize_variant_impl(Archive&, Variant&, size_t, unsigned int) {}
}

template<typename Archive, typename... Types>
void serialize(Archive& ar, std::variant<Types...>& var, const unsigned int version) {
    size_t index = var.index();
    ar & make_nvp("index", index);
    detail::serialize_variant_impl<Archive, std::variant<Types...>, 0>(ar, var, index, version);
}

// ============================================================================
//  Serialization for smart pointers (unique_ptr, shared_ptr)
//  Note: unique_ptr requires move semantics; we treat as raw pointer + ownership flag.
// ============================================================================
template<typename Archive, typename T, typename Deleter>
void serialize(Archive& ar, std::unique_ptr<T, Deleter>& ptr, const unsigned int /*version*/) {
    bool is_null = (ptr == nullptr);
    ar & make_nvp("is_null", is_null);
    if (!is_null) {
        if constexpr (is_output_archive_v<Archive>) {
            ar & make_nvp("value", *ptr);
        } else {
            T value;
            ar & make_nvp("value", value);
            ptr = std::make_unique<T>(std::move(value));
        }
    } else {
        if constexpr (is_input_archive_v<Archive>) {
            ptr.reset();
        }
    }
}

template<typename Archive, typename T>
void serialize(Archive& ar, std::shared_ptr<T>& ptr, const unsigned int /*version*/) {
    bool is_null = (ptr == nullptr);
    ar & make_nvp("is_null", is_null);
    if (!is_null) {
        if constexpr (is_output_archive_v<Archive>) {
            ar & make_nvp("value", *ptr);
        } else {
            T value;
            ar & make_nvp("value", value);
            ptr = std::make_shared<T>(std::move(value));
        }
    } else {
        if constexpr (is_input_archive_v<Archive>) {
            ptr.reset();
        }
    }
}

// ============================================================================
//  Dynamic environment controller for STL serialization (e.g., batch threshold)
// ============================================================================
class STLSerializationEnvironment {
public:
    static STLSerializationEnvironment& instance() {
        static STLSerializationEnvironment env;
        return env;
    }

    // Set minimum size for using batch serialization (for vectors of arithmetic types)
    void setBatchThreshold(size_t threshold) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_batchThreshold = threshold;
    }
    size_t batchThreshold() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_batchThreshold;
    }

    // Enable/disable compression for string and vector data (future)
    void setEnableCompression(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableCompression = enable;
    }
    bool enableCompression() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableCompression;
    }

private:
    STLSerializationEnvironment() : m_batchThreshold(64), m_enableCompression(false) {}
    mutable std::mutex m_mutex;
    size_t m_batchThreshold;
    bool m_enableCompression;
};

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_H_INCLUDED