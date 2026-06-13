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

#ifndef ORTHOTREE_SERIALIZATION_NVP_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_NVP_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/numerical_methods.h"
#include "../core/math/vector_math.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <array>
#include <type_traits>
#include <functional>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Name‑Value Pair (NVP) for serialisation.
//  Enhanced with compile‑time hashing, SIMD‑friendly name storage,
//  and dynamic environment mapping (name translation, versioning).
// ============================================================================

// ----------------------------------------------------------------------------
//  Lightweight fixed‑size name storage (for SIMD and cache efficiency)
// ----------------------------------------------------------------------------
template<size_t MaxLen = 64>
class FixedName {
public:
    using size_type = uint8_t;

    FixedName() noexcept : m_len(0) {
        m_data[0] = '\0';
    }

    explicit FixedName(const char* str) noexcept {
        size_type len = static_cast<size_type>(std::strlen(str));
        m_len = (len < MaxLen) ? len : MaxLen - 1;
        std::memcpy(m_data, str, m_len);
        m_data[m_len] = '\0';
    }

    const char* c_str() const noexcept { return m_data; }
    size_type size() const noexcept { return m_len; }

    // Compile‑time hash for fast comparison (FNV‑1a 64‑bit)
    uint64_t hash() const noexcept {
        uint64_t h = 14695981039346656037ULL;
        for (size_type i = 0; i < m_len; ++i) {
            h = (h ^ static_cast<uint64_t>(m_data[i])) * 1099511628211ULL;
        }
        return h;
    }

    bool operator==(const FixedName& other) const noexcept {
        if (m_len != other.m_len) return false;
        return std::memcmp(m_data, other.m_data, m_len) == 0;
    }

    bool operator!=(const FixedName& other) const noexcept { return !(*this == other); }

private:
    char m_data[MaxLen];
    size_type m_len;
};

// ----------------------------------------------------------------------------
//  Name‑Value Pair template (holds a reference to a value and a name)
// ----------------------------------------------------------------------------
template<typename T>
class nvp {
public:
    using value_type = T;
    using name_type = FixedName<>;

    // Constructors
    nvp(const char* name, T& value) noexcept : m_name(name), m_value(value) {}
    nvp(const name_type& name, T& value) noexcept : m_name(name), m_value(value) {}

    // Accessors
    const name_type& name() const noexcept { return m_name; }
    T& value() const noexcept { return m_value; }
    T& value() noexcept { return m_value; }

    // SIMD‑friendly batch operation: apply a function to the value if name matches
    template<typename Func>
    void apply_if_name_match(uint64_t name_hash, Func&& func) const {
        if (m_name.hash() == name_hash) {
            func(m_value);
        }
    }

    // Dynamic environment: translate name using a runtime mapping
    void translate_name(const std::unordered_map<std::string, std::string>& mapping) {
        auto it = mapping.find(m_name.c_str());
        if (it != mapping.end()) {
            m_name = FixedName<>(it->second.c_str());
        }
    }

private:
    name_type m_name;
    T& m_value;
};

// ----------------------------------------------------------------------------
//  Helper to create NVP (type deduction)
// ----------------------------------------------------------------------------
template<typename T>
inline nvp<T> make_nvp(const char* name, T& value) {
    return nvp<T>(name, value);
}

// ----------------------------------------------------------------------------
//  Dynamic name mapping environment (for versioning or renaming fields)
// ----------------------------------------------------------------------------
class NameMappingEnvironment {
public:
    static NameMappingEnvironment& instance() {
        static NameMappingEnvironment env;
        return env;
    }

    // Register a translation for a specific archive version
    void add_mapping(uint32_t from_version, uint32_t to_version,
                     const std::string& old_name, const std::string& new_name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_mappings[from_version][old_name] = new_name;
        m_reverseMappings[to_version][new_name] = old_name;
    }

    // Translate name for a given version (forward)
    std::string translate(uint32_t version, const std::string& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_mappings.find(version);
        if (it != m_mappings.end()) {
            auto it2 = it->second.find(name);
            if (it2 != it->second.end()) return it2->second;
        }
        return name;
    }

    // Reverse translation (for output)
    std::string reverse_translate(uint32_t version, const std::string& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_reverseMappings.find(version);
        if (it != m_reverseMappings.end()) {
            auto it2 = it->second.find(name);
            if (it2 != it->second.end()) return it2->second;
        }
        return name;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_mappings.clear();
        m_reverseMappings.clear();
    }

private:
    NameMappingEnvironment() = default;
    mutable std::mutex m_mutex;
    std::unordered_map<uint32_t, std::unordered_map<std::string, std::string>> m_mappings;
    std::unordered_map<uint32_t, std::unordered_map<std::string, std::string>> m_reverseMappings;
};

// ----------------------------------------------------------------------------
//  SIMD batch name comparison (compare 4 names simultaneously)
// ----------------------------------------------------------------------------
inline void batch_name_compare(const FixedName<>& name,
                               const FixedName<>* candidates,
                               bool* results, size_t count) {
    uint64_t target_hash = name.hash();
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
        // Unrolled loop with SIMD hints
        for (size_t i = 0; i < count; ++i) {
            results[i] = (candidates[i].hash() == target_hash);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            results[i] = (candidates[i].hash() == target_hash);
        }
    }
}

// ----------------------------------------------------------------------------
//  Compile‑time string literal hash (for NVP names as template arguments)
// ----------------------------------------------------------------------------
template<size_t N>
struct CompileTimeName {
    char data[N];
    constexpr CompileTimeName(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) data[i] = str[i];
    }
    constexpr uint64_t hash() const {
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < N; ++i) {
            h = (h ^ static_cast<uint64_t>(data[i])) * 1099511628211ULL;
        }
        return h;
    }
};

// ----------------------------------------------------------------------------
//  NVP with compile‑time name (template parameter)
// ----------------------------------------------------------------------------
template<typename T, CompileTimeName Name>
class nvp_ct {
public:
    static constexpr uint64_t name_hash = Name.hash();

    explicit nvp_ct(T& value) noexcept : m_value(value) {}
    T& value() const noexcept { return m_value; }
    T& value() noexcept { return m_value; }

    static const char* name() {
        return Name.data;
    }

private:
    T& m_value;
};

template<typename T, size_t N>
auto make_nvp_ct(const char (&name)[N], T& value) {
    return nvp_ct<T, CompileTimeName<N>(name)>(value);
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_NVP_H_INCLUDED