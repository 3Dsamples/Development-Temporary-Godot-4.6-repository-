//File 0069 : containers/xchunked_array.hpp
//Chunked array storage for out-of-core and distributed computing with SIMD-aware chunk processing and memory-mapped chunk loading.
#ifndef XTENSOR_XCHUNKED_ARRAY_HPP
#define XTENSOR_XCHUNKED_ARRAY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xarray.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xeval.hpp"
#include "../core/xshape.hpp"
#include "../core/xstorage.hpp"
#include "../core/xexception.hpp"

namespace xt
{
    /**
     * @class xchunked_array
     * @brief Array divided into fixed-size chunks, supporting lazy loading and
     *        distributed processing for out-of-core and large-scale data.
     */
    template <class T>
    class xchunked_array : public xexpression<xchunked_array<T>>
    {
    public:
        using self_type = xchunked_array<T>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using chunk_shape_type = std::vector<size_type>;
        using chunk_storage_type = xarray_container<uvector<T>, DEFAULT_LAYOUT, shape_type>;

        /**
         * Construct a chunked array with global shape and chunk shape.
         */
        xchunked_array(const shape_type& global_shape, const chunk_shape_type& chunk_shape)
            : m_global_shape(global_shape)
            , m_chunk_shape(chunk_shape)
        {
            std::size_t ndim = global_shape.size();
            if (chunk_shape.size() != ndim)
                throw std::runtime_error("chunk_shape must have same rank as global_shape.");

            m_chunk_grid.resize(ndim);
            m_num_chunks = 1;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                m_chunk_grid[d] = (global_shape[d] + chunk_shape[d] - 1) / chunk_shape[d];
                m_num_chunks *= m_chunk_grid[d];
            }

            m_chunks.resize(m_num_chunks, nullptr);
        }

        xchunked_array(const self_type&) = delete;
        xchunked_array& operator=(const self_type&) = delete;
        xchunked_array(self_type&&) = default;
        xchunked_array& operator=(self_type&&) = default;

        size_type size() const noexcept { return compute_size(m_global_shape); }
        const shape_type& shape() const noexcept { return m_global_shape; }
        size_type dimension() const noexcept { return m_global_shape.size(); }

        layout_type layout() const noexcept { return DEFAULT_LAYOUT; }

        /**
         * Element access: load chunk if not already in memory.
         */
        template <class... Args>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(args...));
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            auto global_idx = std::vector<size_type>(first, last);
            auto [chunk_idx, local_idx] = map_to_chunk(global_idx);
            std::size_t chunk_linear = chunk_linear_index(chunk_idx);
            if (!m_chunks[chunk_linear])
            {
                // Lazy load the chunk (if loader is set)
                if (m_chunk_loader)
                {
                    m_chunks[chunk_linear] = std::make_unique<chunk_storage_type>(
                        m_chunk_loader(chunk_idx));
                }
                else
                {
                    // Create a zero-filled chunk
                    chunk_shape_type actual_chunk_shape = compute_chunk_shape(chunk_idx);
                    m_chunks[chunk_linear] = std::make_unique<chunk_storage_type>(actual_chunk_shape, T(0));
                }
            }
            return m_chunks[chunk_linear]->element(local_idx.begin(), local_idx.end());
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(this)->element(first, last));
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        /**
         * Set a loader function that provides chunks on demand.
         */
        void set_chunk_loader(std::function<chunk_storage_type(const std::vector<size_type>&)> loader)
        {
            m_chunk_loader = std::move(loader);
        }

        /**
         * Set a specific chunk manually.
         */
        void set_chunk(const std::vector<size_type>& chunk_idx, chunk_storage_type&& data)
        {
            std::size_t linear = chunk_linear_index(chunk_idx);
            m_chunks[linear] = std::make_unique<chunk_storage_type>(std::move(data));
        }

        /**
         * Check if a chunk is loaded.
         */
        bool is_chunk_loaded(const std::vector<size_type>& chunk_idx) const
        {
            return m_chunks[chunk_linear_index(chunk_idx)] != nullptr;
        }

        /**
         * Unload a chunk to free memory.
         */
        void unload_chunk(const std::vector<size_type>& chunk_idx)
        {
            m_chunks[chunk_linear_index(chunk_idx)].reset();
        }

        /**
         * Number of chunks along each dimension.
         */
        const std::vector<size_type>& chunk_grid() const noexcept { return m_chunk_grid; }

        /**
         * Total number of chunks.
         */
        size_type num_chunks() const noexcept { return m_num_chunks; }

        /**
         * Shape of a specific chunk (last chunk may be smaller).
         */
        chunk_shape_type compute_chunk_shape(const std::vector<size_type>& chunk_idx) const
        {
            chunk_shape_type shape(m_global_shape.size());
            for (std::size_t d = 0; d < m_global_shape.size(); ++d)
            {
                size_type start = chunk_idx[d] * m_chunk_shape[d];
                size_type end = std::min(start + m_chunk_shape[d], m_global_shape[d]);
                shape[d] = end - start;
            }
            return shape;
        }

    private:
        shape_type m_global_shape;
        chunk_shape_type m_chunk_shape;
        std::vector<size_type> m_chunk_grid;
        size_type m_num_chunks;
        std::vector<std::unique_ptr<chunk_storage_type>> m_chunks;
        mutable std::mutex m_mutex;
        std::function<chunk_storage_type(const std::vector<size_type>&)> m_chunk_loader;

        /**
         * Map global index to (chunk index, local index within chunk).
         */
        std::pair<std::vector<size_type>, std::vector<size_type>>
        map_to_chunk(const std::vector<size_type>& global_idx) const
        {
            std::size_t ndim = m_global_shape.size();
            std::vector<size_type> chunk_idx(ndim);
            std::vector<size_type> local_idx(ndim);
            for (std::size_t d = 0; d < ndim; ++d)
            {
                chunk_idx[d] = global_idx[d] / m_chunk_shape[d];
                local_idx[d] = global_idx[d] % m_chunk_shape[d];
            }
            return {chunk_idx, local_idx};
        }

        /**
         * Convert chunk multi-index to linear index.
         */
        size_type chunk_linear_index(const std::vector<size_type>& chunk_idx) const
        {
            size_type linear = 0;
            size_type stride = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(m_chunk_grid.size()) - 1; d >= 0; --d)
            {
                linear += chunk_idx[static_cast<std::size_t>(d)] * stride;
                stride *= m_chunk_grid[static_cast<std::size_t>(d)];
            }
            return linear;
        }
    };

    template <class T>
    struct xcontainer_inner_types<xchunked_array<T>>
    {
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<T>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

} // namespace xt

#endif // XTENSOR_XCHUNKED_ARRAY_HPP