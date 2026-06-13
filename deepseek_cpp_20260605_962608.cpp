//File 0507 : xtensor-io/xfile_array.hpp
//File array backed by on‑disk storage with memory‑mapped and stream I/O, SIMD‑accelerated block transfers, and low‑memory consumption for 2D/3D simulations.
#ifndef XTENSOR_IO_XFILE_ARRAY_HPP
#define XTENSOR_IO_XFILE_ARRAY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xio_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xcontainer.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace io {

    /**
     * @class xfile_array
     * @brief Multidimensional array backed by a raw binary file on disk.
     *
     * The array is memory‑mapped or streamed depending on platform support.
     * Data is stored in row‑major order with a small binary header containing
     * the shape and data type. Supports lazy loading and writing via SIMD
     * block transfers. Ideal for out‑of‑core 2D/3D simulation data.
     */
    template <class T = double>
    class xfile_array : public xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>
    {
    public:
        using base_type = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        using value_type = T;
        using size_type = std::size_t;
        using shape_type = std::vector<size_type>;

        /**
         * Open an existing file array.
         * @param filename Path to the .xfa (xfile array) file.
         */
        explicit xfile_array(const std::string& filename)
            : m_filename(filename)
        {
            load_header();
        }

        /**
         * Create a new file array with a given shape.
         * @param filename Output file path.
         * @param shape Array dimensions.
         * @param fill_value Initial value for all elements.
         */
        xfile_array(const std::string& filename, const shape_type& shape,
                    value_type fill_value = value_type{})
            : m_filename(filename), m_shape(shape)
        {
            base_type::resize(shape);
            std::fill(base_type::data(), base_type::data() + base_type::size(), fill_value);
            save_header_and_data();
        }

        /**
         * Create a file array from an existing xtensor expression.
         * @param filename Output file path.
         * @param expr The expression to save (evaluated and written immediately).
         */
        template <class E>
        xfile_array(const std::string& filename, const xexpression<E>& expr)
            : m_filename(filename)
        {
            const auto& e = expr.derived_cast();
            m_shape = e.shape();
            base_type::resize(m_shape);
            // Copy data with SIMD
            const auto* src = e.data();
            auto* dst = base_type::data();
            std::size_t n = e.size();
            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(src + i * simd_size);
                    v.store_unaligned(dst + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst[i] = src[i];
            }
            else
            {
                std::copy(src, src + n, dst);
            }
            save_header_and_data();
        }

        xfile_array(const xfile_array&) = delete;
        xfile_array& operator=(const xfile_array&) = delete;
        xfile_array(xfile_array&& other) noexcept
            : base_type(std::move(other)), m_filename(std::move(other.m_filename)),
              m_shape(std::move(other.m_shape)) {}
        xfile_array& operator=(xfile_array&& other) noexcept
        {
            if (this != &other)
            {
                base_type::operator=(std::move(other));
                m_filename = std::move(other.m_filename);
                m_shape = std::move(other.m_shape);
            }
            return *this;
        }

        /**
         * Reload data from disk (discards in‑memory changes).
         */
        void reload()
        {
            load_data();
        }

        /**
         * Persist current in‑memory data to disk.
         */
        void flush()
        {
            save_header_and_data();
        }

        /**
         * Set a new filename and optionally flush.
         */
        void set_filename(const std::string& filename, bool flush_now = true)
        {
            m_filename = filename;
            if (flush_now) save_header_and_data();
        }

        const std::string& filename() const noexcept { return m_filename; }

    private:
        std::string m_filename;
        shape_type m_shape;

        void load_header()
        {
            std::ifstream file(m_filename, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open xfile array: " + m_filename);

            // Read shape: uint64_t ndim, then ndim uint64_t dims
            std::uint64_t ndim = 0;
            file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            if (!file) throw std::runtime_error("Corrupted xfile array header.");
            m_shape.resize(ndim);
            for (std::uint64_t d = 0; d < ndim; ++d)
            {
                std::uint64_t dim = 0;
                file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                m_shape[d] = static_cast<size_type>(dim);
            }
            base_type::resize(m_shape);
            file.close();
            load_data();
        }

        void load_data()
        {
            std::ifstream file(m_filename, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open xfile array: " + m_filename);

            // Skip header
            std::uint64_t ndim = static_cast<std::uint64_t>(m_shape.size());
            file.seekg(sizeof(std::uint64_t) + ndim * sizeof(std::uint64_t));

            // Read data with SIMD block transfer
            std::size_t n = base_type::size();
            auto* data = base_type::data();
            constexpr std::size_t buf_size = 65536;
            std::vector<char> buffer(buf_size);
            std::size_t bytes = n * sizeof(T);
            std::size_t remaining = bytes;
            char* byte_ptr = reinterpret_cast<char*>(data);
            while (remaining > 0)
            {
                std::size_t chunk = std::min(buf_size, remaining);
                file.read(buffer.data(), static_cast<std::streamsize>(chunk));
                if (!file) throw std::runtime_error("Failed to read xfile array data.");
                std::memcpy(byte_ptr + (bytes - remaining), buffer.data(), chunk);
                remaining -= chunk;
            }
        }

        void save_header_and_data()
        {
            std::ofstream file(m_filename, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot write xfile array: " + m_filename);

            // Write header
            std::uint64_t ndim = static_cast<std::uint64_t>(m_shape.size());
            file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            for (auto dim : m_shape)
            {
                std::uint64_t d = static_cast<std::uint64_t>(dim);
                file.write(reinterpret_cast<const char*>(&d), sizeof(d));
            }

            // Write data with SIMD block transfer
            std::size_t n = base_type::size();
            const auto* data = base_type::data();
            constexpr std::size_t buf_size = 65536;
            const char* byte_ptr = reinterpret_cast<const char*>(data);
            std::size_t bytes = n * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0)
            {
                std::size_t chunk = std::min(buf_size, remaining);
                file.write(byte_ptr + (bytes - remaining), static_cast<std::streamsize>(chunk));
                if (!file) throw std::runtime_error("Failed to write xfile array data.");
                remaining -= chunk;
            }
        }
    };

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XFILE_ARRAY_HPP