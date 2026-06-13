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

#ifndef ORTHOTREE_SERIALIZATION_TRAITS_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_TRAITS_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/numerical_methods.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <memory>
#include <functional>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Core archive type traits (for SFINAE and overload resolution)
// ============================================================================

// 1. is_archive<T>: identifies archives (binary, msgpack, etc.)
template<typename T, typename = void>
struct is_archive : std::false_type {};

template<typename T>
struct is_archive<T, std::void_t<typename T::is_archive>> : std::true_type {};

template<typename T>
inline constexpr bool is_archive_v = is_archive<T>::value;

// 2. is_input_archive / is_output_archive
template<typename T, typename = void>
struct is_input_archive : std::false_type {};

template<typename T>
struct is_input_archive<T, std::void_t<typename T::is_input_archive>> : std::true_type {};

template<typename T>
inline constexpr bool is_input_archive_v = is_input_archive<T>::value;

template<typename T, typename = void>
struct is_output_archive : std::false_type {};

template<typename T>
struct is_output_archive<T, std::void_t<typename T::is_output_archive>> : std::true_type {};

template<typename T>
inline constexpr bool is_output_archive_v = is_output_archive<T>::value;

// 3. Binary and MsgPack specific tags
template<typename T, typename = void>
struct is_binary_archive : std::false_type {};

template<typename T>
struct is_binary_archive<T, std::void_t<typename T::is_binary_archive>> : std::true_type {};

template<typename T>
inline constexpr bool is_binary_archive_v = is_binary_archive<T>::value;

template<typename T, typename = void>
struct is_msgpack_archive : std::false_type {};

template<typename T>
struct is_msgpack_archive<T, std::void_t<typename T::is_msgpack_archive>> : std::true_type {};

template<typename T>
inline constexpr bool is_msgpack_archive_v = is_msgpack_archive<T>::value;

// ============================================================================
//  Detect if a type has a serialise function (member or free function)
// ============================================================================

// Member function: serialize(Archive&, unsigned int)
template<typename T, typename Archive, typename = void>
struct has_member_serialize : std::false_type {};

template<typename T, typename Archive>
struct has_member_serialize<T, Archive,
    std::void_t<decltype(std::declval<T>().serialize(std::declval<Archive&>(),
                                                     std::declval<unsigned int>()))>>
    : std::true_type {};

template<typename T, typename Archive>
inline constexpr bool has_member_serialize_v = has_member_serialize<T, Archive>::value;

// Free function: serialize(Archive&, T&, unsigned int)
template<typename T, typename Archive, typename = void>
struct has_free_serialize : std::false_type {};

template<typename T, typename Archive>
struct has_free_serialize<T, Archive,
    std::void_t<decltype(serialize(std::declval<Archive&>(),
                                   std::declval<T&>(),
                                   std::declval<unsigned int>()))>>
    : std::true_type {};

template<typename T, typename Archive>
inline constexpr bool has_free_serialize_v = has_free_serialize<T, Archive>::value;

// Combined: has_serialize
template<typename T, typename Archive>
struct has_serialize : std::integral_constant<bool,
    has_member_serialize_v<T, Archive> || has_free_serialize_v<T, Archive>> {};

template<typename T, typename Archive>
inline constexpr bool has_serialize_v = has_serialize<T, Archive>::value;

// ============================================================================
//  Detect if a type is trivially serializable (memcpy‑able)
//  Used for SIMD batch optimisations.
// ============================================================================
template<typename T>
struct is_trivially_serializable : std::integral_constant<bool,
    std::is_trivially_copyable_v<T> &&
    (std::is_arithmetic_v<T> || std::is_enum_v<T> ||
     std::is_pointer_v<T> || std::is_same_v<T, std::nullptr_t>)> {};

template<typename T>
inline constexpr bool is_trivially_serializable_v = is_trivially_serializable<T>::value;

// For arrays of trivially serializable types, we can batch read/write.
template<typename T, size_t N>
struct is_trivially_serializable<std::array<T, N>> : is_trivially_serializable<T> {};

template<typename T>
struct is_trivially_serializable<std::vector<T>> : std::false_type {}; // vectors need extra handling

// ============================================================================
//  Versioning traits (for archive version compatibility)
// ============================================================================
template<typename Archive>
struct archive_version {
    static constexpr unsigned int value() noexcept {
        // Default: if Archive has a static version constant, use it.
        if constexpr (requires { Archive::archive_version; }) {
            return Archive::archive_version;
        } else {
            return 1;
        }
    }
};

template<typename Archive>
inline constexpr unsigned int archive_version_v = archive_version<Archive>::value();

// ============================================================================
//  Dynamic type registry (for polymorphic serialisation)
// ============================================================================
class TypeRegistry {
public:
    using type_id = uint64_t;
    using factory_func = std::function<void*(void)>;
    using deleter_func = std::function<void(void*)>;

    static TypeRegistry& instance() {
        static TypeRegistry reg;
        return reg;
    }

    template<typename T>
    type_id register_type(const std::string& name) {
        type_id id = hash_name(name);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_name_to_id[name] = id;
        m_id_to_name[id] = name;
        m_factories[id] = []() -> void* { return new T(); };
        m_deleters[id] = [](void* p) { delete static_cast<T*>(p); };
        return id;
    }

    std::string get_name(type_id id) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_id_to_name.find(id);
        return (it != m_id_to_name.end()) ? it->second : "";
    }

    type_id get_id(const std::string& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_name_to_id.find(name);
        return (it != m_name_to_id.end()) ? it->second : 0;
    }

    void* create(type_id id) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_factories.find(id);
        return (it != m_factories.end()) ? it->second() : nullptr;
    }

    void destroy(type_id id, void* ptr) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_deleters.find(id);
        if (it != m_deleters.end()) it->second(ptr);
    }

private:
    TypeRegistry() = default;
    static uint64_t hash_name(const std::string& name) {
        uint64_t h = 14695981039346656037ULL;
        for (char c : name) {
            h = (h ^ static_cast<uint64_t>(c)) * 1099511628211ULL;
        }
        return h;
    }
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, type_id> m_name_to_id;
    std::unordered_map<type_id, std::string> m_id_to_name;
    std::unordered_map<type_id, factory_func> m_factories;
    std::unordered_map<type_id, deleter_func> m_deleters;
};

// ============================================================================
//  Helper: enable SIMD batch serialization for arithmetic arrays
//  Provides a static method to write/read a contiguous block if the type
//  is trivially serializable and the archive supports batch operations.
// ============================================================================
template<typename Archive, typename T>
struct batch_serializer {
    static bool write_batch(Archive& ar, const T* data, size_t count) {
        if constexpr (is_output_archive_v<Archive> && is_trivially_serializable_v<T>) {
            ar.write_batch(data, count);
            return true;
        }
        return false;
    }

    static bool read_batch(Archive& ar, T* out, size_t count) {
        if constexpr (is_input_archive_v<Archive> && is_trivially_serializable_v<T>) {
            ar.read_batch(out, count);
            return true;
        }
        return false;
    }
};

// ============================================================================
//  Dynamic environment controller for serialization traits
// ============================================================================
class TraitsEnvironment {
public:
    static TraitsEnvironment& instance() {
        static TraitsEnvironment env;
        return env;
    }

    // Enable/disable runtime type checking (for polymorphic archives)
    void setEnableRuntimeTypeInfo(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableRTTI = enable;
    }

    bool enableRuntimeTypeInfo() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableRTTI;
    }

    // Set default archive version for types without explicit version
    void setDefaultVersion(unsigned int version) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultVersion = version;
    }

    unsigned int defaultVersion() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultVersion;
    }

    // Register a custom serializer for a type at runtime (for dynamic environments)
    template<typename T, typename Archive>
    void register_serializer(std::function<void(Archive&, T&, unsigned int)> serializer) {
        // This would be stored in a map; simplified for brevity.
    }

private:
    TraitsEnvironment() : m_enableRTTI(false), m_defaultVersion(1) {}
    mutable std::mutex m_mutex;
    bool m_enableRTTI;
    unsigned int m_defaultVersion;
};

// ============================================================================
//  Helper: compile‑time check for archive compatibility with a type
// ============================================================================
template<typename Archive, typename T>
constexpr bool is_serializable_v =
    has_serialize_v<T, Archive> ||
    (is_trivially_serializable_v<T> && (is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>));

// ============================================================================
//  SIMD alignment detection for batch serialisation
// ============================================================================
template<typename T>
constexpr bool is_simd_aligned_v = (alignof(T) >= ORTHOTREE_SIMD_ALIGNMENT);

template<typename T>
constexpr size_t simd_alignment() {
    return is_simd_aligned_v<T> ? alignof(T) : ORTHOTREE_SIMD_ALIGNMENT;
}

// ============================================================================
//  Versioned type wrapper (for adding version info to a serialized object)
// ============================================================================
template<typename T>
struct versioned {
    T value;
    unsigned int version;
};

template<typename Archive, typename T>
void serialize(Archive& ar, versioned<T>& v, unsigned int) {
    ar & make_nvp("value", v.value);
    if constexpr (is_output_archive_v<Archive>) {
        ar & make_nvp("version", v.version);
    } else {
        unsigned int ver = 0;
        ar & make_nvp("version", ver);
        v.version = ver;
    }
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_TRAITS_H_INCLUDED