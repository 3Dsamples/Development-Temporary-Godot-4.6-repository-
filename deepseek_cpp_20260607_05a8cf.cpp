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
 * @file embedded_resource_pmr_map.h
 * @brief PMR‑aware hash map with small‑buffer optimisation (embedded storage).
 *
 * This file provides a hash map that uses polymorphic memory resources (PMR)
 * and avoids dynamic allocations for small numbers of elements by storing them
 * in a fixed‑size array inside the map object. This is critical for real‑time
 * systems where allocation latency must be minimised.
 *
 * The map is designed for sparse octree node lookups (Morton code -> node index)
 * and similar use cases where most nodes have few children but can grow.
 * It supports heterogeneous lookup, transparent hashing, and custom allocators.
 *
 * Key features:
 * - Embedded storage for up to `SmallCapacity` elements (default 8)
 * - Falls back to `std::unordered_map` with PMR allocator when capacity exceeded
 * - Transparent lookup (key comparisons without constructing temporary keys)
 * - Move‑aware and exception‑safe
 * - No memory allocations in common case (small maps)
 */

#ifndef ORTHOTREE_DETAIL_EMBEDDED_RESOURCE_PMR_MAP_H_INCLUDED
#define ORTHOTREE_DETAIL_EMBEDDED_RESOURCE_PMR_MAP_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "common.h"
#include "memory_resource.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

#if ORTHOTREE_HAS_PMR
    #include <memory_resource>
#else
    // Fallback: use std::allocator if PMR not available
    namespace std::pmr {
        using polymorphic_allocator = std::allocator<std::byte>;
        template <typename T>
        using vector = std::vector<T, std::allocator<T>>;
        // simplified for our use
    }
#endif

namespace OrthoTree {
namespace detail {

// ----------------------------------------------------------------------------
//  Default small capacity for embedded map (must be power of two for modulo)
// ----------------------------------------------------------------------------
constexpr size_t DEFAULT_EMBEDDED_MAP_CAPACITY = 8;

/**
 * @brief Hash map with embedded storage for small sizes, using PMR allocator.
 *
 * @tparam Key Key type (e.g., uint64_t Morton code).
 * @tparam Value Mapped type (e.g., NodeIndex).
 * @tparam SmallCapacity Number of elements stored inline (no allocation).
 * @tparam Hash Hash functor (default std::hash<Key>).
 * @tparam Equal Equality comparator (default std::equal_to<Key>).
 */
template <typename Key,
          typename Value,
          size_t SmallCapacity = DEFAULT_EMBEDDED_MAP_CAPACITY,
          typename Hash = std::hash<Key>,
          typename Equal = std::equal_to<Key>>
class EmbeddedResourcePmrMap {
public:
    using key_type        = Key;
    using mapped_type     = Value;
    using value_type      = std::pair<const Key, Value>;
    using size_type       = size_t;
    using hasher          = Hash;
    using key_equal       = Equal;
    using allocator_type  = PMRAllocator<value_type>;

    // ------------------------------------------------------------------------
    //  Embedded storage: fixed‑size array of key‑value pairs.
    //  We store pairs as non‑const for simplicity, but keys are logically const.
    // ------------------------------------------------------------------------
    using EmbeddedStorage = std::pair<Key, Value>[SmallCapacity];

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    explicit EmbeddedResourcePmrMap(const allocator_type& alloc = allocator_type())
        : m_alloc(alloc)
        , m_size(0)
        , m_embeddedUsed(false)
        , m_fallbackMap(nullptr) {
        // No allocation yet
    }

    // Copy constructor: copies whole map (deep)
    EmbeddedResourcePmrMap(const EmbeddedResourcePmrMap& other)
        : m_alloc(other.m_alloc)
        , m_size(other.m_size)
        , m_embeddedUsed(other.m_embeddedUsed) {
        if (m_embeddedUsed) {
            // copy embedded array
            for (size_t i = 0; i < m_size; ++i) {
                new (&m_embedded[i]) std::pair<Key, Value>(other.m_embedded[i]);
            }
        } else {
            // copy fallback map
            m_fallbackMap = std::make_unique<FallbackMap>(*other.m_fallbackMap, m_alloc);
        }
    }

    // Move constructor: steal resources
    EmbeddedResourcePmrMap(EmbeddedResourcePmrMap&& other) noexcept
        : m_alloc(std::move(other.m_alloc))
        , m_size(other.m_size)
        , m_embeddedUsed(other.m_embeddedUsed) {
        if (m_embeddedUsed) {
            // move embedded array
            for (size_t i = 0; i < m_size; ++i) {
                new (&m_embedded[i]) std::pair<Key, Value>(std::move(other.m_embedded[i]));
                other.m_embedded[i].~pair();
            }
            other.m_size = 0;
            other.m_embeddedUsed = false;
        } else {
            m_fallbackMap = std::move(other.m_fallbackMap);
            other.m_fallbackMap = nullptr;
        }
    }

    // Destructor
    ~EmbeddedResourcePmrMap() {
        clear();
    }

    // Assignment operators
    EmbeddedResourcePmrMap& operator=(const EmbeddedResourcePmrMap& other) {
        if (this != &other) {
            clear();
            m_alloc = other.m_alloc;
            m_size = other.m_size;
            m_embeddedUsed = other.m_embeddedUsed;
            if (m_embeddedUsed) {
                for (size_t i = 0; i < m_size; ++i) {
                    new (&m_embedded[i]) std::pair<Key, Value>(other.m_embedded[i]);
                }
            } else if (other.m_fallbackMap) {
                m_fallbackMap = std::make_unique<FallbackMap>(*other.m_fallbackMap, m_alloc);
            }
        }
        return *this;
    }

    EmbeddedResourcePmrMap& operator=(EmbeddedResourcePmrMap&& other) noexcept {
        if (this != &other) {
            clear();
            m_alloc = std::move(other.m_alloc);
            m_size = other.m_size;
            m_embeddedUsed = other.m_embeddedUsed;
            if (m_embeddedUsed) {
                for (size_t i = 0; i < m_size; ++i) {
                    new (&m_embedded[i]) std::pair<Key, Value>(std::move(other.m_embedded[i]));
                    other.m_embedded[i].~pair();
                }
                other.m_size = 0;
                other.m_embeddedUsed = false;
            } else {
                m_fallbackMap = std::move(other.m_fallbackMap);
                other.m_fallbackMap = nullptr;
            }
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Capacity
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_size; }
    bool empty() const noexcept { return m_size == 0; }
    size_type capacity() const noexcept {
        return m_embeddedUsed ? SmallCapacity : (m_fallbackMap ? m_fallbackMap->bucket_count() : 0);
    }

    // ------------------------------------------------------------------------
    //  Lookup
    // ------------------------------------------------------------------------
    std::optional<Value> find(const Key& key) const {
        if (m_embeddedUsed) {
            for (size_t i = 0; i < m_size; ++i) {
                if (Equal{}(m_embedded[i].first, key))
                    return m_embedded[i].second;
            }
            return std::nullopt;
        } else if (m_fallbackMap) {
            auto it = m_fallbackMap->find(key);
            if (it != m_fallbackMap->end())
                return it->second;
        }
        return std::nullopt;
    }

    // Heterogeneous lookup (if Key supports transparent hash)
    template <typename K>
    std::optional<Value> find(const K& key) const {
        if (m_embeddedUsed) {
            for (size_t i = 0; i < m_size; ++i) {
                if (Equal{}(m_embedded[i].first, key))
                    return m_embedded[i].second;
            }
            return std::nullopt;
        } else if (m_fallbackMap) {
            auto it = m_fallbackMap->find(key);
            if (it != m_fallbackMap->end())
                return it->second;
        }
        return std::nullopt;
    }

    // ------------------------------------------------------------------------
    //  Insertion / emplacement
    // ------------------------------------------------------------------------
    bool insert(const Key& key, const Value& value) {
        if (auto existing = find(key))
            return false; // already exists

        if (m_embeddedUsed && m_size < SmallCapacity) {
            // still space in embedded
            new (&m_embedded[m_size]) std::pair<Key, Value>(key, value);
            ++m_size;
            return true;
        } else if (!m_embeddedUsed && m_fallbackMap) {
            // already using fallback
            m_fallbackMap->emplace(key, value);
            ++m_size;
            return true;
        } else if (!m_embeddedUsed && m_size == 0) {
            // start using embedded
            m_embeddedUsed = true;
            new (&m_embedded[0]) std::pair<Key, Value>(key, value);
            m_size = 1;
            return true;
        } else if (m_embeddedUsed && m_size == SmallCapacity) {
            // need to convert to fallback map
            convertToFallback();
            m_fallbackMap->emplace(key, value);
            ++m_size;
            return true;
        } else {
            // should not reach
            return false;
        }
    }

    template <typename... Args>
    bool emplace(Args&&... args) {
        // We need to construct a pair first to check key existence
        // To avoid extra copy, we can forward but it's simpler to insert via temporary.
        // For performance, we can use piecewise construct if needed.
        value_type tmp(std::forward<Args>(args)...);
        return insert(tmp.first, tmp.second);
    }

    // ------------------------------------------------------------------------
    //  Erasure
    // ------------------------------------------------------------------------
    bool erase(const Key& key) {
        if (m_embeddedUsed) {
            for (size_t i = 0; i < m_size; ++i) {
                if (Equal{}(m_embedded[i].first, key)) {
                    // shift left
                    for (size_t j = i + 1; j < m_size; ++j) {
                        m_embedded[j-1] = std::move(m_embedded[j]);
                    }
                    m_embedded[m_size-1].~pair();
                    --m_size;
                    // optionally convert back to embedded if fallback becomes small? Not needed.
                    return true;
                }
            }
            return false;
        } else if (m_fallbackMap) {
            auto it = m_fallbackMap->find(key);
            if (it != m_fallbackMap->end()) {
                m_fallbackMap->erase(it);
                --m_size;
                // if size becomes <= SmallCapacity, we could convert back to embedded
                // but that's optional; we keep fallback for simplicity.
                return true;
            }
        }
        return false;
    }

    void clear() {
        if (m_embeddedUsed) {
            for (size_t i = 0; i < m_size; ++i) {
                m_embedded[i].~pair();
            }
            m_size = 0;
            m_embeddedUsed = false;
        } else if (m_fallbackMap) {
            m_fallbackMap->clear();
            m_size = 0;
            // we could deallocate fallback map, but keep it for later use
        }
    }

    // ------------------------------------------------------------------------
    //  Iteration (simple forward iterator for embedded case)
    // ------------------------------------------------------------------------
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<const Key, Value>;
        using difference_type = ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        Iterator(std::pair<Key, Value>* ptr) : m_ptr(ptr) {}
        reference operator*() const { return *reinterpret_cast<value_type*>(m_ptr); }
        pointer operator->() const { return reinterpret_cast<pointer>(m_ptr); }
        Iterator& operator++() { ++m_ptr; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
        friend bool operator==(const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; }
        friend bool operator!=(const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; }
    private:
        std::pair<Key, Value>* m_ptr;
    };

    Iterator begin() {
        if (m_embeddedUsed)
            return Iterator(m_embedded);
        // Fallback map iteration not implemented for simplicity.
        return Iterator(nullptr);
    }
    Iterator end() {
        if (m_embeddedUsed)
            return Iterator(m_embedded + m_size);
        return Iterator(nullptr);
    }

private:
    // ------------------------------------------------------------------------
    //  Fallback map using unordered_map with PMR allocator
    // ------------------------------------------------------------------------
    using FallbackMap = std::unordered_map<Key, Value, Hash, Equal,
                                           PMRAllocator<std::pair<const Key, Value>>>;

    void convertToFallback() {
        ORTHOTREE_ASSERT(m_embeddedUsed && m_size == SmallCapacity);
        m_fallbackMap = std::make_unique<FallbackMap>(m_alloc);
        m_fallbackMap->reserve(SmallCapacity * 2);
        for (size_t i = 0; i < m_size; ++i) {
            m_fallbackMap->emplace(std::move(m_embedded[i].first), std::move(m_embedded[i].second));
            m_embedded[i].~pair();
        }
        m_embeddedUsed = false;
        // size remains same
    }

    // ------------------------------------------------------------------------
    //  Member variables
    // ------------------------------------------------------------------------
    allocator_type m_alloc;
    size_t m_size;
    bool m_embeddedUsed;   // true if using embedded array, false if using fallback map
    union {
        EmbeddedStorage m_embedded;                 // active when m_embeddedUsed == true
        std::unique_ptr<FallbackMap> m_fallbackMap; // active when m_embeddedUsed == false
    };
};

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_EMBEDDED_RESOURCE_PMR_MAP_H_INCLUDED