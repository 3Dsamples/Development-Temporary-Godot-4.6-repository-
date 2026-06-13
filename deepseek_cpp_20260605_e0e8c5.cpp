//File 0051 : core/xstorage.hpp
//Storage utilities with SIMD-aligned allocators, small‑buffer optimization, memory pool, and type‑aware buffer management for xtensor containers.
#ifndef XTENSOR_XSTORAGE_HPP
#define XTENSOR_XSTORAGE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /*********************************************
     * aligned_allocator – SIMD‑aligned allocations
     *********************************************/
    template <class T, std::size_t Alignment = SIMD_ALIGNMENT>
    class aligned_allocator
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using propagate_on_container_move_assignment = std::true_type;

        template <class U>
        struct rebind
        {
            using other = aligned_allocator<U, Alignment>;
        };

        aligned_allocator() noexcept = default;

        template <class U>
        aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

        T* allocate(std::size_t n)
        {
            if (n == 0) return nullptr;
            if (n > max_size()) throw std::bad_alloc();
            void* ptr = nullptr;
            #if defined(_MSC_VER) || defined(__MINGW32__)
                ptr = _aligned_malloc(n * sizeof(T), Alignment);
            #else
                if (::posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
                    ptr = nullptr;
            #endif
            if (!ptr) throw std::bad_alloc();
            return static_cast<T*>(ptr);
        }

        void deallocate(T* p, std::size_t) noexcept
        {
            #if defined(_MSC_VER) || defined(__MINGW32__)
                _aligned_free(p);
            #else
                std::free(p);
            #endif
        }

        size_type max_size() const noexcept
        {
            return std::numeric_limits<std::size_t>::max() / sizeof(T);
        }
    };

    template <class T, std::size_t A1, class U, std::size_t A2>
    bool operator==(const aligned_allocator<T, A1>&, const aligned_allocator<U, A2>&) noexcept
    {
        return A1 == A2;
    }

    template <class T, std::size_t A1, class U, std::size_t A2>
    bool operator!=(const aligned_allocator<T, A1>&, const aligned_allocator<U, A2>&) noexcept
    {
        return A1 != A2;
    }

    /*********************************************
     * small_buffer_storage – inline storage for small shapes
     *********************************************/
    namespace detail
    {
        template <class T, std::size_t N>
        class small_buffer_storage
        {
        public:
            using value_type = T;
            using size_type = std::size_t;
            using pointer = T*;
            using const_pointer = const T*;
            using iterator = T*;
            using const_iterator = const T*;

            small_buffer_storage() noexcept : m_size(0), m_using_small(true) {}

            small_buffer_storage(std::initializer_list<T> init) : small_buffer_storage()
            {
                resize(init.size());
                std::copy(init.begin(), init.end(), data());
            }

            small_buffer_storage(const small_buffer_storage& rhs)
                : m_size(rhs.m_size), m_using_small(rhs.m_using_small)
            {
                if (m_using_small)
                    std::copy(rhs.m_small, rhs.m_small + m_size, m_small);
                else
                {
                    m_heap = std::make_unique<T[]>(m_size);
                    std::copy(rhs.m_heap.get(), rhs.m_heap.get() + m_size, m_heap.get());
                }
            }

            small_buffer_storage& operator=(const small_buffer_storage& rhs)
            {
                if (this != &rhs)
                {
                    small_buffer_storage tmp(rhs);
                    swap(tmp);
                }
                return *this;
            }

            small_buffer_storage(small_buffer_storage&& rhs) noexcept
                : m_size(rhs.m_size), m_using_small(rhs.m_using_small)
            {
                if (m_using_small)
                    std::move(rhs.m_small, rhs.m_small + m_size, m_small);
                else
                    m_heap = std::move(rhs.m_heap);
                rhs.m_size = 0;
                rhs.m_using_small = true;
            }

            small_buffer_storage& operator=(small_buffer_storage&& rhs) noexcept
            {
                if (this != &rhs) swap(rhs);
                return *this;
            }

            ~small_buffer_storage() = default;

            T& operator[](size_type i) { return data()[i]; }
            const T& operator[](size_type i) const { return data()[i]; }

            pointer data() noexcept { return m_using_small ? m_small : m_heap.get(); }
            const_pointer data() const noexcept { return m_using_small ? m_small : m_heap.get(); }

            iterator begin() noexcept { return data(); }
            iterator end() noexcept { return data() + m_size; }
            const_iterator begin() const noexcept { return data(); }
            const_iterator end() const noexcept { return data() + m_size; }

            size_type size() const noexcept { return m_size; }
            bool empty() const noexcept { return m_size == 0; }

            void resize(size_type n)
            {
                if (n <= N && m_using_small)
                {
                    m_size = n;
                    return;
                }
                if (n <= N && !m_using_small)
                {
                    auto tmp = std::make_unique<T[]>(n);
                    std::copy(m_heap.get(), m_heap.get() + std::min(m_size, n), tmp.get());
                    m_heap.reset();
                    std::copy(tmp.get(), tmp.get() + n, m_small);
                    m_using_small = true;
                    m_size = n;
                    return;
                }
                // Need heap storage
                auto new_heap = std::make_unique<T[]>(n);
                std::copy(data(), data() + std::min(m_size, n), new_heap.get());
                m_heap = std::move(new_heap);
                m_using_small = false;
                m_size = n;
            }

            void clear() noexcept { m_size = 0; }

            void swap(small_buffer_storage& other) noexcept
            {
                if (m_using_small && other.m_using_small)
                {
                    std::swap_ranges(m_small, m_small + std::max(m_size, other.m_size),
                                     other.m_small);
                    std::swap(m_size, other.m_size);
                }
                else if (!m_using_small && !other.m_using_small)
                {
                    m_heap.swap(other.m_heap);
                    std::swap(m_size, other.m_size);
                }
                else
                {
                    small_buffer_storage tmp = std::move(*this);
                    *this = std::move(other);
                    other = std::move(tmp);
                }
                std::swap(m_using_small, other.m_using_small);
            }

        private:
            size_type m_size;
            bool m_using_small;
            union
            {
                T m_small[N];
            };
            std::unique_ptr<T[]> m_heap;
        };
    }

    /*********************************************
     * memory_pool – thread‑safe pool for reuse
     *********************************************/
    template <class T, std::size_t BlockSize = 4096>
    class memory_pool
    {
    public:
        using value_type = T;
        using size_type = std::size_t;

        static memory_pool& instance()
        {
            static memory_pool pool;
            return pool;
        }

        T* allocate(std::size_t n)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Find a block in the free list that's large enough
            for (auto it = m_free_blocks.begin(); it != m_free_blocks.end(); ++it)
            {
                if (it->size() >= n)
                {
                    auto block = std::move(*it);
                    m_free_blocks.erase(it);
                    return block.get();
                }
            }
            // Allocate new block
            m_allocated.emplace_back(std::make_unique<T[]>(n));
            return m_allocated.back().get();
        }

        void deallocate(T* ptr, std::size_t n) noexcept
        {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = std::find_if(m_allocated.begin(), m_allocated.end(),
                [ptr](const auto& u) { return u.get() == ptr; });
            if (it != m_allocated.end())
            {
                m_free_blocks.push_back(std::move(*it));
                m_allocated.erase(it);
            }
        }

    private:
        std::mutex m_mutex;
        std::vector<std::unique_ptr<T[]>> m_allocated;
        std::vector<std::unique_ptr<T[]>> m_free_blocks;
    };

    /*********************************************
     * uvector – default vector with aligned allocator
     *********************************************/
    template <class T>
    using uvector = std::vector<T, aligned_allocator<T, SIMD_ALIGNMENT>>;

    /*********************************************
     * fixed_vector – stack‑allocated small vector
     *********************************************/
    template <class T, std::size_t N>
    using fixed_vector = detail::small_buffer_storage<T, N>;

    /*********************************************
     * Helper: aligned allocation / free
     *********************************************/
    inline void* aligned_alloc(std::size_t size, std::size_t alignment = SIMD_ALIGNMENT)
    {
        void* ptr = nullptr;
        #if defined(_MSC_VER) || defined(__MINGW32__)
            ptr = _aligned_malloc(size, alignment);
        #else
            if (::posix_memalign(&ptr, alignment, size) != 0)
                ptr = nullptr;
        #endif
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    inline void aligned_free(void* ptr) noexcept
    {
        #if defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(ptr);
        #else
            std::free(ptr);
        #endif
    }

    /*********************************************
     * storage_type_for – deduce storage type for T
     *********************************************/
    template <class T>
    using default_storage_type = uvector<T>;

    template <class T, std::size_t N>
    using default_fixed_storage_type = std::array<T, N>;

} // namespace xt

#endif // XTENSOR_XSTORAGE_HPP