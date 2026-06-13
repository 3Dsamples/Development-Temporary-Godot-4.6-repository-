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

/**
 * @file memory_resource.h
 * @brief Polymorphic memory resource wrappers and custom allocators.
 *
 * This file provides:
 * - A wrapper for std::pmr::polymorphic_allocator when available.
 * - Fallback to std::allocator if PMR not present.
 * - A monotonic buffer resource (arena allocator) for fast, low‑fragmentation
 *   allocations within a fixed block.
 * - Aligned allocator for SIMD types.
 *
 * These memory resources are critical for real‑time systems: they eliminate
 * per‑allocation overhead and provide deterministic performance.
 */

#ifndef ORTHOTREE_DETAIL_MEMORY_RESOURCE_H_INCLUDED
#define ORTHOTREE_DETAIL_MEMORY_RESOURCE_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "common.h"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

#if ORTHOTREE_HAS_PMR
    #include <memory_resource>
#endif

namespace OrthoTree {
namespace detail {

// ----------------------------------------------------------------------------
//  PMR allocator type (polymorphic allocator if available, else std::allocator)
// ----------------------------------------------------------------------------

#if ORTHOTREE_HAS_PMR

template <typename T>
using PMRAllocator = std::pmr::polymorphic_allocator<T>;

#else

// Fallback: a simple wrapper that mimics polymorphic_allocator interface
class MemoryResource {
public:
    virtual ~MemoryResource() = default;
    virtual void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) = 0;
    virtual void deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) = 0;
    virtual bool is_equal(const MemoryResource& other) const noexcept = 0;
};

template <typename T>
class PMRAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    PMRAllocator() noexcept : m_resource(nullptr) {}
    explicit PMRAllocator(MemoryResource* resource) noexcept : m_resource(resource) {}

    template <typename U>
    PMRAllocator(const PMRAllocator<U>& other) noexcept : m_resource(other.resource()) {}

    pointer allocate(size_type n) {
        if (n > max_size()) throw std::bad_alloc();
        void* p = m_resource ? m_resource->allocate(n * sizeof(T), alignof(T))
                             : ::operator new(n * sizeof(T), std::nothrow);
        if (!p) throw std::bad_alloc();
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type n) noexcept {
        if (m_resource)
            m_resource->deallocate(p, n * sizeof(T), alignof(T));
        else
            ::operator delete(p);
    }

    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    MemoryResource* resource() const noexcept { return m_resource; }

    bool operator==(const PMRAllocator& other) const noexcept {
        return m_resource == other.m_resource;
    }
    bool operator!=(const PMRAllocator& other) const noexcept {
        return !(*this == other);
    }

private:
    MemoryResource* m_resource;
};

#endif // ORTHOTREE_HAS_PMR

// ----------------------------------------------------------------------------
//  Monotonic buffer resource (arena allocator)
// ----------------------------------------------------------------------------

/**
 * @brief A memory resource that allocates from a fixed‑size buffer linearly.
 *        Fast, no fragmentation, but cannot deallocate individual objects.
 *        Reset clears the whole arena.
 */
class MonotonicBufferResource : public MemoryResource {
public:
    explicit MonotonicBufferResource(std::byte* buffer, std::size_t size) noexcept
        : m_buffer(buffer), m_size(size), m_offset(0) {}

    ~MonotonicBufferResource() override = default;

    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override {
        std::size_t alignedOffset = alignUp(m_offset, alignment);
        if (alignedOffset + bytes > m_size) {
            // Out of memory – could fallback to heap, but we choose to throw.
            throw std::bad_alloc();
        }
        void* p = m_buffer + alignedOffset;
        m_offset = alignedOffset + bytes;
        return p;
    }

    void deallocate(void* /*p*/, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
        // No‑op: monotonic buffer does not support individual deallocation.
    }

    bool is_equal(const MemoryResource& other) const noexcept override {
        return this == &other;
    }

    void reset() noexcept {
        m_offset = 0;
    }

    std::size_t used() const noexcept { return m_offset; }
    std::size_t capacity() const noexcept { return m_size; }

private:
    std::byte* m_buffer;
    std::size_t m_size;
    std::size_t m_offset;
};

// ----------------------------------------------------------------------------
//  Global new/delete resource (default system allocator)
// ----------------------------------------------------------------------------

class NewDeleteResource : public MemoryResource {
public:
    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override {
        // Use aligned_alloc if alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__
        if (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
            return ::operator new(bytes, std::nothrow);
        } else {
            return ::aligned_alloc(alignment, bytes);
        }
    }

    void deallocate(void* p, std::size_t /*bytes*/, std::size_t alignment) override {
        if (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
            ::operator delete(p);
        } else {
            ::free(p);
        }
    }

    bool is_equal(const MemoryResource& other) const noexcept override {
        return this == &other;
    }

    static NewDeleteResource* instance() {
        static NewDeleteResource res;
        return &res;
    }
};

// ----------------------------------------------------------------------------
//  Aligned allocator wrapper (for SIMD types)
// ----------------------------------------------------------------------------

template <typename T, std::size_t Alignment = 32>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n > max_size()) throw std::bad_alloc();
        void* p = ::aligned_alloc(Alignment, n * sizeof(T));
        if (!p) throw std::bad_alloc();
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) noexcept {
        ::free(p);
    }

    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

// ----------------------------------------------------------------------------
//  Scoped memory resource guard (temporarily sets a resource for a scope)
// ----------------------------------------------------------------------------

#if ORTHOTREE_HAS_PMR

class ScopedMemoryResource {
public:
    explicit ScopedMemoryResource(std::pmr::memory_resource* newResource)
        : m_oldResource(std::pmr::get_default_resource()) {
        std::pmr::set_default_resource(newResource);
    }
    ~ScopedMemoryResource() {
        std::pmr::set_default_resource(m_oldResource);
    }
private:
    std::pmr::memory_resource* m_oldResource;
};

#else

class ScopedMemoryResource {
public:
    explicit ScopedMemoryResource(MemoryResource* newResource)
        : m_oldResource(defaultMemoryResource()), m_newResource(newResource) {
        setDefaultMemoryResource(m_newResource);
    }
    ~ScopedMemoryResource() {
        setDefaultMemoryResource(m_oldResource);
    }
private:
    MemoryResource* m_oldResource;
    MemoryResource* m_newResource;
};

#endif // ORTHOTREE_HAS_PMR

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_MEMORY_RESOURCE_H_INCLUDED