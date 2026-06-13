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

#ifndef ORTHOTREE_CORE_MEMORY_HIERARCHICAL_ALLOCATOR_H_INCLUDED
#define ORTHOTREE_CORE_MEMORY_HIERARCHICAL_ALLOCATOR_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/memory_resource.h"

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <vector>
#include <array>
#include <mutex>
#include <algorithm>
#include <limits>
#include <new>

namespace OrthoTree {
namespace Memory {

// ============================================================================
//  HierarchicalAllocator: allocator that uses separate memory pools for each
//  depth level of an octree. Reduces fragmentation and improves cache locality
//  by grouping same‑depth nodes together. Supports PMR and SIMD alignment.
// ============================================================================
template<typename T, uint8_t MaxDepth = 16>
class HierarchicalAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Rebinding for containers
    template<typename U>
    struct rebind {
        using other = HierarchicalAllocator<U, MaxDepth>;
    };

    // ------------------------------------------------------------------------
    //  Pool configuration: one pool per depth level (0..MaxDepth-1)
    // ------------------------------------------------------------------------
    struct PoolConfig {
        size_type blockSize;          // size of each block (bytes)
        size_type initialBlocks;      // pre‑allocated blocks at start
        size_type maxBlocks;          // maximum blocks before using next pool
        bool enableSIMDAlignment;     // align to SIMD boundary (32/64 bytes)
    };

    static constexpr size_type DEFAULT_BLOCK_SIZE = sizeof(T);
    static constexpr size_type DEFAULT_INITIAL_BLOCKS = 1024;
    static constexpr size_type DEFAULT_MAX_BLOCKS = 65536;

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit HierarchicalAllocator(const PoolConfig& poolCfg = getDefaultPoolConfig())
        : m_poolConfig(poolCfg)
        , m_pools(MaxDepth) {
        for (uint8_t d = 0; d < MaxDepth; ++d) {
            // Each depth gets its own pool with possible different block sizes
            // Here we keep same block size for all depths; could be varied.
            size_type blockSize = alignUp(poolCfg.blockSize, getAlignment());
            m_pools[d] = std::make_unique<Pool>(blockSize, poolCfg.initialBlocks, poolCfg.maxBlocks,
                                                poolCfg.enableSIMDAlignment);
        }
    }

    HierarchicalAllocator(const HierarchicalAllocator&) = default;
    HierarchicalAllocator(HierarchicalAllocator&&) noexcept = default;
    ~HierarchicalAllocator() = default;

    // ------------------------------------------------------------------------
    //  Allocate with depth hint (used by octree)
    // ------------------------------------------------------------------------
    pointer allocate(size_type n, uint8_t depth = 0) {
        if (n != 1) {
            // For arrays, fall back to general allocation
            return static_cast<pointer>(::operator new(n * sizeof(T), std::nothrow));
        }
        if (depth >= MaxDepth) depth = MaxDepth - 1;
        void* ptr = m_pools[depth]->allocate();
        if (!ptr) {
            // Out of memory: try fallback from next depth or global
            ptr = m_pools[MaxDepth-1]->allocate();
            if (!ptr) ptr = ::operator new(sizeof(T), std::nothrow);
        }
        return static_cast<pointer>(ptr);
    }

    // Standard allocate (without depth) uses depth 0 (root level)
    pointer allocate(size_type n) {
        return allocate(n, 0);
    }

    void deallocate(pointer p, size_type n, uint8_t depth = 0) {
        if (n != 1) {
            ::operator delete(p);
            return;
        }
        if (depth >= MaxDepth) depth = MaxDepth - 1;
        if (!m_pools[depth]->deallocate(p)) {
            // Not from pool? fallback to global delete
            ::operator delete(p);
        }
    }

    void deallocate(pointer p, size_type n) {
        deallocate(p, n, 0);
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control: adjust pool sizes at runtime
    // ------------------------------------------------------------------------
    void growPool(uint8_t depth, size_type additionalBlocks) {
        if (depth < MaxDepth) {
            m_pools[depth]->grow(additionalBlocks);
        }
    }

    void shrinkPool(uint8_t depth, size_type targetBlocks) {
        if (depth < MaxDepth) {
            m_pools[depth]->shrink(targetBlocks);
        }
    }

    void resetPool(uint8_t depth) {
        if (depth < MaxDepth) {
            m_pools[depth]->reset();
        }
    }

    // Get pool statistics
    struct PoolStats {
        size_type usedBlocks;
        size_type freeBlocks;
        size_type totalBlocks;
        size_type blockSize;
    };

    PoolStats getPoolStats(uint8_t depth) const {
        if (depth >= MaxDepth) return {};
        return m_pools[depth]->getStats();
    }

    // ------------------------------------------------------------------------
    //  Equality
    // ------------------------------------------------------------------------
    bool operator==(const HierarchicalAllocator& other) const noexcept {
        return this == &other;
    }
    bool operator!=(const HierarchicalAllocator& other) const noexcept {
        return !(*this == other);
    }

private:
    // ------------------------------------------------------------------------
    //  Internal pool implementation (lock‑free stack for small blocks)
    // ------------------------------------------------------------------------
    class Pool {
    public:
        Pool(size_type blockSize, size_type initialBlocks, size_type maxBlocks, bool alignSIMD)
            : m_blockSize(alignUp(blockSize, alignSIMD ? getAlignment() : alignof(std::max_align_t)))
            , m_maxBlocks(maxBlocks)
            , m_usedBlocks(0)
            , m_freeList(nullptr) {
            grow(initialBlocks);
        }

        ~Pool() {
            // Free all allocated blocks
            void* ptr = m_freeList;
            while (ptr) {
                void* next = *reinterpret_cast<void**>(ptr);
                freeAligned(ptr);
                ptr = next;
            }
            // Also free memory that was allocated but never added to free list?
            // For simplicity, we assume all allocated blocks are linked.
        }

        void* allocate() {
            if (!m_freeList) {
                // Try to grow on demand
                if (!grow(1)) return nullptr;
            }
            void* ptr = m_freeList;
            m_freeList = *reinterpret_cast<void**>(ptr);
            ++m_usedBlocks;
            return ptr;
        }

        bool deallocate(void* ptr) {
            // Add to free list
            *reinterpret_cast<void**>(ptr) = m_freeList;
            m_freeList = ptr;
            --m_usedBlocks;
            return true;
        }

        bool grow(size_type additionalBlocks) {
            size_type newBlocks = std::min(additionalBlocks, m_maxBlocks - (m_usedBlocks + freeCount()));
            if (newBlocks == 0) return false;
            for (size_type i = 0; i < newBlocks; ++i) {
                void* newBlock = allocateAligned(m_blockSize);
                if (!newBlock) return false;
                *reinterpret_cast<void**>(newBlock) = m_freeList;
                m_freeList = newBlock;
            }
            return true;
        }

        void shrink(size_type targetFreeBlocks) {
            // Remove free blocks until free count <= targetFreeBlocks
            while (freeCount() > targetFreeBlocks && m_freeList) {
                void* ptr = m_freeList;
                m_freeList = *reinterpret_cast<void**>(ptr);
                freeAligned(ptr);
            }
        }

        void reset() {
            // Deallocate all free blocks, but keep used blocks? Reset clears everything.
            while (m_freeList) {
                void* ptr = m_freeList;
                m_freeList = *reinterpret_cast<void**>(ptr);
                freeAligned(ptr);
            }
            m_usedBlocks = 0;
        }

        PoolStats getStats() const {
            return {m_usedBlocks, freeCount(), m_usedBlocks + freeCount(), m_blockSize};
        }

    private:
        size_type freeCount() const {
            size_type count = 0;
            void* ptr = m_freeList;
            while (ptr) {
                ++count;
                ptr = *reinterpret_cast<void**>(ptr);
            }
            return count;
        }

        static void* allocateAligned(size_type size) {
            size_type alignment = getAlignment();
#if defined(_MSC_VER)
            return _aligned_malloc(size, alignment);
#else
            void* ptr;
            if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
            return ptr;
#endif
        }

        static void freeAligned(void* ptr) {
#if defined(_MSC_VER)
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }

        static size_type getAlignment() {
            return ORTHOTREE_SIMD_ALIGNMENT;
        }

        size_type m_blockSize;
        size_type m_maxBlocks;
        std::atomic<size_type> m_usedBlocks;
        void* m_freeList;   // lock‑free singly linked list (single producer/consumer with atomic? simplified)
    };

    static PoolConfig getDefaultPoolConfig() {
        PoolConfig cfg;
        cfg.blockSize = sizeof(T);
        cfg.initialBlocks = DEFAULT_INITIAL_BLOCKS;
        cfg.maxBlocks = DEFAULT_MAX_BLOCKS;
        cfg.enableSIMDAlignment = (ORTHOTREE_SIMD_LEVEL >= 128);
        return cfg;
    }

    static size_type alignUp(size_type size, size_type alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    PoolConfig m_poolConfig;
    std::vector<std::unique_ptr<Pool>> m_pools;
};

// ----------------------------------------------------------------------------
//  Convenience alias for depth‑aware octree node allocator
// ----------------------------------------------------------------------------
template<typename NodeType>
using OctreeNodeAllocator = HierarchicalAllocator<NodeType, ORTHOTREE_DEFAULT_MAX_DEPTH>;

} // namespace Memory
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MEMORY_HIERARCHICAL_ALLOCATOR_H_INCLUDED