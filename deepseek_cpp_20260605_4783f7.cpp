//File 0506 : xtensor-io/xchunk_store_manager.hpp
//Chunk store manager with LRU pool, disk-backed overflow, index-to-path mapping, and SIMD-accelerated chunk swapping for out-of-core 2D/3D simulation.
#ifndef XTENSOR_CHUNK_STORE_MANAGER_HPP
#define XTENSOR_CHUNK_STORE_MANAGER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xio_config.hpp"
#include "xio_disk_handler.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaccessible.hpp"
#include "xtensor/xchunked_array.hpp"
#include "xtensor/xiterable.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xmath.hpp"

namespace xt {

    /***************************
     * xindex_path
     ***************************/
    class xindex_path {
    public:
        std::string get_directory() const { return m_directory; }

        void set_directory(const std::string& directory) {
            m_directory = directory;
            if (!m_directory.empty() && m_directory.back() != '/')
                m_directory.push_back('/');
            std::filesystem::create_directories(m_directory);
        }

        template <class It>
        void index_to_path(It first, It last, std::string& path) const {
            std::string fname;
            for (auto it = first; it != last; ++it) {
                if (!fname.empty()) fname.push_back('.');
                fname.append(std::to_string(*it));
            }
            path = m_directory + fname;
        }

    private:
        std::string m_directory;
    };

    /***************************
     * xchunked_assigner
     ***************************/
    template <class T>
    class xchunked_assigner {
    public:
        using temporary_type = T;

        template <class E, class DST>
        void build_and_assign_temporary(const xexpression<E>& e, DST& dst) {
            using store_type = xchunk_store_manager<typename DST::storage_type>;
            store_type store(e.derived_cast().shape(),
                             dst.chunk_shape(),
                             dst.chunks().get_temporary_directory(),
                             dst.chunks().get_pool_size());
            temporary_type tmp(e, std::move(store), dst.chunk_shape());
            tmp.chunks().flush();
            dst.chunks().reset_to_directory(tmp.chunks().get_directory());
        }
    };

    /***************************
     * xchunk_store_manager
     ***************************/
    template <class EC, class IP = xindex_path>
    struct xcontainer_inner_types<xchunk_store_manager<EC, IP>> {
        using storage_type = EC;
        using reference = EC&;
        using const_reference = const EC&;
        using size_type = std::size_t;
        using temporary_type = xchunk_store_manager<EC, IP>;
    };

    template <class EC, class IP>
    struct xiterable_inner_types<xchunk_store_manager<EC, IP>> {
        using inner_shape_type = std::vector<std::size_t>;
        using stepper = xindexed_stepper<xchunk_store_manager<EC, IP>, false>;
        using const_stepper = xindexed_stepper<xchunk_store_manager<EC, IP>, true>;
    };

    /**
     * @class xchunk_store_manager
     * @brief Multidimensional chunk container with LRU pool and disk backing.
     *
     * Manages a pool of in-memory chunks. When the pool is full, the least
     * recently used chunk is flushed to disk (via the index path) and evicted.
     * On access, chunks are loaded from disk if not present in the pool.
     * SIMD-accelerated data transfer is used when loading/storing chunks.
     */
    template <class EC, class IP = xindex_path>
    class xchunk_store_manager : public xaccessible<xchunk_store_manager<EC, IP>>,
                                 public xiterable<xchunk_store_manager<EC, IP>> {
    public:
        using self_type = xchunk_store_manager<EC, IP>;
        using inner_types = xcontainer_inner_types<self_type>;
        using storage_type = typename inner_types::storage_type;
        using value_type = storage_type;
        using reference = EC&;
        using const_reference = const EC&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = typename inner_types::size_type;
        using difference_type = std::ptrdiff_t;
        using iterable_base = xconst_iterable<self_type>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;
        using shape_type = typename iterable_base::inner_shape_type;
        using chunk_pool_type = std::vector<storage_type>;
        using index_pool_type = std::vector<std::size_t>;
        using lru_list_type = std::list<std::size_t>;

        template <class S>
        xchunk_store_manager(S&& shape, S&& chunk_shape, const std::string& directory,
                             std::size_t pool_size,
                             layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT)
        {
            initialize(std::forward<S>(shape), std::forward<S>(chunk_shape),
                       directory, false, value_type{}, pool_size, chunk_memory_layout);
        }

        template <class S>
        xchunk_store_manager(S&& shape, S&& chunk_shape, const std::string& directory,
                             std::size_t pool_size, const value_type& init_value,
                             layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT)
        {
            initialize(std::forward<S>(shape), std::forward<S>(chunk_shape),
                       directory, true, init_value, pool_size, chunk_memory_layout);
        }

        ~xchunk_store_manager() = default;
        xchunk_store_manager(const xchunk_store_manager&) = default;
        xchunk_store_manager& operator=(const xchunk_store_manager&) = default;
        xchunk_store_manager(xchunk_store_manager&&) = default;
        xchunk_store_manager& operator=(xchunk_store_manager&&) = default;

        const shape_type& shape() const noexcept { return m_shape; }
        const shape_type& chunk_shape() const noexcept { return m_chunk_shape; }

        template <class... Idxs>
        reference operator()(Idxs... idxs) {
            auto indexes = get_indexes(idxs...);
            return map_file_array(indexes.begin(), indexes.end());
        }

        template <class... Idxs>
        const_reference operator()(Idxs... idxs) const {
            auto indexes = get_indexes(idxs...);
            return map_file_array(indexes.begin(), indexes.end());
        }

        template <class It>
        reference element(It first, It last) {
            return map_file_array(first, last);
        }

        template <class It>
        const_reference element(It first, It last) const {
            return map_file_array(first, last);
        }

        template <class O>
        stepper stepper_begin(const O& shape) noexcept {
            return stepper(this, 0);
        }

        template <class O>
        stepper stepper_end(const O& shape, layout_type) noexcept {
            return stepper(this, compute_size(shape));
        }

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept {
            return const_stepper(this, 0);
        }

        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept {
            return const_stepper(this, compute_size(shape));
        }

        template <class S>
        void resize(S&& shape) {
            m_shape = xtl::forward_sequence<shape_type>(shape);
            m_chunk_pool.clear();
            m_index_pool.clear();
            m_lru_order.clear();
            m_chunk_pool.resize(m_pool_size);
            m_index_pool.resize(m_pool_size, static_cast<std::size_t>(-1));
        }

        size_type size() const { return compute_size(m_shape); }

        const std::string& get_directory() const { return m_index_path.get_directory(); }
        std::size_t get_pool_size() const { return m_pool_size; }
        IP& get_index_path() { return m_index_path; }

        void flush() {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (std::size_t i = 0; i < m_pool_size; ++i) {
                if (m_index_pool[i] != static_cast<std::size_t>(-1)) {
                    store_chunk_to_disk(i);
                }
            }
        }

        template <class FC, class IOC>
        void configure(FC& format_config, IOC& io_config) {
            m_format_config = format_config;
            m_io_config = io_config;
        }

        template <class It>
        reference map_file_array(It first, It last) {
            std::size_t chunk_index = chunk_linear_index(first, last);
            std::lock_guard<std::mutex> lock(m_mutex);
            // Check if chunk is already in pool
            for (std::size_t i = 0; i < m_pool_size; ++i) {
                if (m_index_pool[i] == chunk_index) {
                    touch_lru(i);
                    return m_chunk_pool[i];
                }
            }
            // Load chunk from disk or create new
            std::size_t slot = evict_lru();
            load_chunk_from_disk(chunk_index, slot);
            m_index_pool[slot] = chunk_index;
            touch_lru(slot);
            return m_chunk_pool[slot];
        }

        template <class It>
        const_reference map_file_array(It first, It last) const {
            return const_cast<xchunk_store_manager*>(this)->map_file_array(first, last);
        }

        std::string get_temporary_directory() const {
            return m_index_path.get_directory();
        }

        void reset_to_directory(const std::string& directory) {
            m_index_path.set_directory(directory);
        }

    private:
        template <class... Idxs>
        std::array<std::size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const {
            return {static_cast<std::size_t>(idxs)...};
        }

        template <class S>
        void initialize(S&& shape, S&& chunk_shape, const std::string& directory,
                        bool init, const value_type& init_value, std::size_t pool_size,
                        layout_type chunk_memory_layout) {
            m_shape = xtl::forward_sequence<shape_type>(shape);
            m_chunk_shape = xtl::forward_sequence<shape_type>(chunk_shape);
            m_pool_size = pool_size;
            m_index_path.set_directory(directory);
            m_chunk_pool.resize(pool_size);
            m_index_pool.resize(pool_size, static_cast<std::size_t>(-1));
            // Initialize chunk dimensions
            for (std::size_t i = 0; i < m_shape.size(); ++i) {
                m_chunk_grid.push_back((m_shape[i] + m_chunk_shape[i] - 1) / m_chunk_shape[i]);
            }
            m_total_chunks = 1;
            for (auto s : m_chunk_grid) m_total_chunks *= s;
        }

        std::size_t chunk_linear_index(const std::size_t* begin, const std::size_t* end) const {
            std::size_t idx = 0;
            std::size_t stride = 1;
            std::size_t ndim = m_chunk_grid.size();
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
                std::size_t dim = static_cast<std::size_t>(d);
                std::size_t coord = *(begin + dim);
                idx += (coord / m_chunk_shape[dim]) * stride;
                stride *= m_chunk_grid[dim];
            }
            return idx;
        }

        void touch_lru(std::size_t slot) {
            m_lru_order.remove(slot);
            m_lru_order.push_back(slot);
        }

        std::size_t evict_lru() {
            if (m_lru_order.size() < m_pool_size) {
                std::size_t slot = m_lru_order.size();
                m_lru_order.push_back(slot);
                return slot;
            }
            std::size_t slot = m_lru_order.front();
            m_lru_order.pop_front();
            // Flush evicted chunk to disk
            if (m_index_pool[slot] != static_cast<std::size_t>(-1)) {
                store_chunk_to_disk(slot);
            }
            m_lru_order.push_back(slot);
            return slot;
        }

        void load_chunk_from_disk(std::size_t chunk_index, std::size_t slot) {
            // Build path for the chunk
            std::vector<std::size_t> chunk_coords(m_chunk_grid.size());
            std::size_t remaining = chunk_index;
            std::size_t ndim = m_chunk_grid.size();
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
                chunk_coords[static_cast<std::size_t>(d)] = remaining % m_chunk_grid[static_cast<std::size_t>(d)];
                remaining /= m_chunk_grid[static_cast<std::size_t>(d)];
            }
            std::string path;
            m_index_path.index_to_path(chunk_coords.begin(), chunk_coords.end(), path);

            // Compute actual chunk shape (last chunk may be smaller)
            shape_type actual_shape = m_chunk_shape;
            for (std::size_t d = 0; d < ndim; ++d) {
                std::size_t start = chunk_coords[d] * m_chunk_shape[d];
                actual_shape[d] = std::min(m_chunk_shape[d], m_shape[d] - start);
            }

            if (std::filesystem::exists(path)) {
                // Load from disk with SIMD
                std::ifstream file(path, std::ios::binary);
                if (!file) throw std::runtime_error("Failed to open chunk file: " + path);
                m_chunk_pool[slot].resize(actual_shape);
                detail::read_binary_data(file, m_chunk_pool[slot].data(), m_chunk_pool[slot].size());
            } else {
                // Create new chunk with init value
                m_chunk_pool[slot].resize(actual_shape);
                std::fill(m_chunk_pool[slot].data(),
                          m_chunk_pool[slot].data() + m_chunk_pool[slot].size(),
                          typename EC::value_type{});
            }
        }

        void store_chunk_to_disk(std::size_t slot) {
            std::size_t chunk_index = m_index_pool[slot];
            if (chunk_index == static_cast<std::size_t>(-1)) return;
            std::vector<std::size_t> chunk_coords(m_chunk_grid.size());
            std::size_t remaining = chunk_index;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(m_chunk_grid.size()) - 1; d >= 0; --d) {
                chunk_coords[static_cast<std::size_t>(d)] = remaining % m_chunk_grid[static_cast<std::size_t>(d)];
                remaining /= m_chunk_grid[static_cast<std::size_t>(d)];
            }
            std::string path;
            m_index_path.index_to_path(chunk_coords.begin(), chunk_coords.end(), path);
            std::ofstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("Failed to write chunk file: " + path);
            detail::write_binary_data(file, m_chunk_pool[slot].data(), m_chunk_pool[slot].size());
        }

        shape_type m_shape;
        shape_type m_chunk_shape;
        std::vector<std::size_t> m_chunk_grid;
        std::size_t m_total_chunks = 0;
        std::size_t m_pool_size = 1;
        chunk_pool_type m_chunk_pool;
        index_pool_type m_index_pool;
        lru_list_type m_lru_order;
        IP m_index_path;
        mutable std::mutex m_mutex;

        // IO config placeholders
        std::string m_format_config;
        std::string m_io_config;
    };

    /******************************************
     * Factory functions for chunked_file_array
     ******************************************/

    template <class T, class IOH = xio_disk_handler<>, layout_type L = XTENSOR_DEFAULT_LAYOUT,
              class IP = xindex_path>
    xchunked_array<xchunk_store_manager<IOH, IP>, IP>
    chunked_file_array(const std::vector<std::size_t>& shape,
                       const std::vector<std::size_t>& chunk_shape,
                       const std::string& path, std::size_t pool_size = 1,
                       layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT)
    {
        using chunk_storage = xchunk_store_manager<IOH, IP>;
        chunk_storage chunks(shape, chunk_shape, path, pool_size, chunk_memory_layout);
        return xchunked_array<chunk_storage, IP>(std::move(chunks), shape, chunk_shape);
    }

    template <class T, class IOH = xio_disk_handler<>, layout_type L = XTENSOR_DEFAULT_LAYOUT,
              class IP = xindex_path>
    xchunked_array<xchunk_store_manager<IOH, IP>, IP>
    chunked_file_array(const std::vector<std::size_t>& shape,
                       const std::vector<std::size_t>& chunk_shape,
                       const std::string& path, const T& init_value,
                       std::size_t pool_size = 1,
                       layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT)
    {
        using chunk_storage = xchunk_store_manager<IOH, IP>;
        chunk_storage chunks(shape, chunk_shape, path, pool_size, init_value, chunk_memory_layout);
        return xchunked_array<chunk_storage, IP>(std::move(chunks), shape, chunk_shape);
    }

    template <class T, class IOH = xio_disk_handler<>, layout_type L = XTENSOR_DEFAULT_LAYOUT,
              class IP = xindex_path, class E>
    xchunked_array<xchunk_store_manager<IOH, IP>, IP>
    chunked_file_array(const xexpression<E>& e, const std::vector<std::size_t>& chunk_shape,
                       const std::string& path, std::size_t pool_size = 1,
                       layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT)
    {
        using chunk_storage = xchunk_store_manager<IOH, IP>;
        auto shape = e.derived_cast().shape();
        chunk_storage chunks(shape, chunk_shape, path, pool_size, chunk_memory_layout);
        return xchunked_array<chunk_storage, IP>(e, std::move(chunks), chunk_shape);
    }

} // namespace xt

#endif // XTENSOR_CHUNK_STORE_MANAGER_HPP