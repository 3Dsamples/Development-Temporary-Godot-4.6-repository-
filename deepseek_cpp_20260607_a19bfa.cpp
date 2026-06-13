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

#ifndef ORTHOTREE_SERIALIZATION_STL_POINTER_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_POINTER_H_INCLUDED

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

#include <memory>
#include <type_traits>
#include <cstddef>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Serialization for raw pointers (C-style).
//  WARNING: Raw pointers are dangerous to serialize without context.
//  This implementation stores the pointee value as if it were a regular object.
//  On deserialization, a new object is allocated (requires copy/move).
//  Only use when you are certain about memory ownership.
// ============================================================================

template<typename Archive, typename T>
void serialize(Archive& ar, T*& ptr, const unsigned int /*version*/) {
    static_assert(!std::is_void_v<T>, "Cannot serialize pointer to void");
    bool is_null = (ptr == nullptr);
    ar & make_nvp("is_null", is_null);
    if (!is_null) {
        if constexpr (is_output_archive_v<Archive>) {
            ar & make_nvp("value", *ptr);
        } else {
            // Allocate a new object of type T (default constructed)
            // Note: this may leak if not properly managed; prefer smart pointers.
            T* new_ptr = new T();
            ar & make_nvp("value", *new_ptr);
            ptr = new_ptr;
        }
    } else {
        if constexpr (is_input_archive_v<Archive>) {
            ptr = nullptr;
        }
    }
}

// ============================================================================
//  Serialization for std::unique_ptr (move‑only)
//  On output, writes a boolean indicating nullity and then the pointee.
//  On input, creates a new unique_ptr with a copy of the deserialized value.
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

// ============================================================================
//  Serialization for std::shared_ptr
//  Similar to unique_ptr, but uses make_shared on input.
//  Note: This does not preserve aliasing or weak references; it's a shallow
//  serialisation of the pointee.
// ============================================================================
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
//  Serialization for std::weak_ptr (not directly supported because weak_ptr
//  cannot be dereferenced without locking. Instead, we serialize the shared_ptr
//  version; on input, a new shared_ptr is created and weak_ptr is obtained.
// ============================================================================
template<typename Archive, typename T>
void serialize(Archive& ar, std::weak_ptr<T>& wptr, const unsigned int version) {
    // Convert weak to shared to get the pointee (if still alive)
    // On output, we need to lock; if expired, treat as null.
    std::shared_ptr<T> sptr = wptr.lock();
    ar & make_nvp("shared_ptr", sptr);
    if constexpr (is_input_archive_v<Archive>) {
        wptr = sptr;
    }
    (void)version;
}

// ============================================================================
//  Batch optimisation for arrays of pointers? Not supported directly.
//  We provide a helper to serialise an array of pointers (raw or smart)
//  by iterating.
// ============================================================================
template<typename Archive, typename PtrType>
void serialize_pointer_array(Archive& ar, PtrType* ptrs, size_t count, const unsigned int version) {
    for (size_t i = 0; i < count; ++i) {
        ar & make_nvp(("ptr" + std::to_string(i)).c_str(), ptrs[i]);
    }
}

// ============================================================================
//  Dynamic environment controller for pointer serialization
//  Allows tuning of allocation strategies and null handling.
// ============================================================================
class PointerSerializationEnvironment {
public:
    static PointerSerializationEnvironment& instance() {
        static PointerSerializationEnvironment env;
        return env;
    }

    // If true, always allocate a new object on deserialization even if
    // the pointer was null (useful for fixed‑size arrays). Default: false.
    void setAlwaysAllocate(bool always) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_alwaysAllocate = always;
    }
    bool alwaysAllocate() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_alwaysAllocate;
    }

    // For std::unique_ptr, choose between make_unique and new (for custom deleters)
    void setUseMakeUnique(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useMakeUnique = use;
    }
    bool useMakeUnique() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useMakeUnique;
    }

    // Enable tracking of pointer aliasing (future)
    void setEnableTracking(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableTracking = enable;
    }
    bool enableTracking() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableTracking;
    }

private:
    PointerSerializationEnvironment()
        : m_alwaysAllocate(false)
        , m_useMakeUnique(true)
        , m_enableTracking(false) {}
    mutable std::mutex m_mutex;
    bool m_alwaysAllocate;
    bool m_useMakeUnique;
    bool m_enableTracking;
};

// ----------------------------------------------------------------------------
//  Helper: conditional allocation for raw pointers
// ----------------------------------------------------------------------------
template<typename T>
T* allocate_pointer() {
    if (PointerSerializationEnvironment::instance().alwaysAllocate()) {
        return new T();
    }
    return nullptr;
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_POINTER_H_INCLUDED

/**
 * Next file: include/orthotree/serialization/stl/variant.h
 * Remaining: 2 files in serialization/stl (variant.h and then done)
 * Total STL files: 9, completed: 7 (array, common, map, optional, set, unordered_map, vector, pointer -> 8? Let's recount)
 * Actually we have: array, common, map, optional, set, unordered_map, vector, pointer = 8. Only variant left.
 * So after this, 1 file left (variant.h). Then the original repository STL serialization is complete.
 */ 