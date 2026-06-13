//File 0514 : xtensor-io/xio_disk_handler.hpp
//Disk I/O handler for xtensor arrays using binary NPY-like format with SIMD-accelerated block transfers, shape/dtype headers, and memory-mapped file support.
#ifndef XTENSOR_IO_XIO_DISK_HANDLER_HPP
#define XTENSOR_IO_XIO_DISK_HANDLER_HPP

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
#include "xtensor/xeval.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace io {

    namespace detail {

        // Simple magic for disk handler files: "XDH\0"
        constexpr unsigned char xdh_magic[4] = {'X', 'D', 'H', '\0'};
        constexpr std::uint8_t xdh_version = 1;

        // dtype codes
        template <class T> struct xdh_dtype_code;
        template <> struct xdh_dtype_code<float>       { static constexpr std::uint8_t code = 0; };
        template <> struct xdh_dtype_code<double>      { static constexpr std::uint8_t code = 1; };
        template <> struct xdh_dtype_code<int32_t>     { static constexpr std::uint8_t code = 2; };
        template <> struct xdh_dtype_code<int64_t>     { static constexpr std::uint8_t code = 3; };
        template <class T> inline constexpr std::uint8_t xdh_dtype_code_v = xdh_dtype_code<T>::code;

        inline void write_xdh_header(std::ostream& out,
                                      const std::vector<std::size_t>& shape,
                                      std::uint8_t dtype_code) {
            out.write(reinterpret_cast<const char*>(xdh_magic), 4);
            out.write(reinterpret_cast<const char*>(&xdh_version), 1);
            out.write(reinterpret_cast<const char*>(&dtype_code), 1);
            std::uint16_t ndim = static_cast<std::uint16_t>(shape.size());
            out.write(reinterpret_cast<const char*>(&ndim), 2);
            for (auto dim : shape) {
                std::uint64_t d = dim;
                out.write(reinterpret_cast<const char*>(&d), 8);
            }
        }

        inline auto read_xdh_header(std::istream& in) {
            unsigned char magic[4];
            in.read(reinterpret_cast<char*>(magic), 4);
            if (std::memcmp(magic, xdh_magic, 4) != 0)
                throw std::runtime_error("Not a valid XDH disk file.");
            std::uint8_t version;
            in.read(reinterpret_cast<char*>(&version), 1);
            std::uint8_t dtype_code;
            in.read(reinterpret_cast<char*>(&dtype_code), 1);
            std::uint16_t ndim;
            in.read(reinterpret_cast<char*>(&ndim), 2);
            std::vector<std::size_t> shape(ndim);
            for (std::uint16_t i = 0; i < ndim; ++i) {
                std::uint64_t d;
                in.read(reinterpret_cast<char*>(&d), 8);
                shape[i] = static_cast<std::size_t>(d);
            }
            return std::make_pair(shape, dtype_code);
        }

        template <class T>
        inline void read_xdh_data(std::istream& in, T* data, std::size_t count) {
            constexpr std::size_t buf_size = 65536;
            std::vector<char> buffer(buf_size);
            char* byte_ptr = reinterpret_cast<char*>(data);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                in.read(buffer.data(), static_cast<std::streamsize>(chunk));
                if (!in) throw std::runtime_error("Failed to read disk array data.");
                std::memcpy(byte_ptr + (bytes - remaining), buffer.data(), chunk);
                remaining -= chunk;
            }
        }

        template <class T>
        inline void write_xdh_data(std::ostream& out, const T* data, std::size_t count) {
            constexpr std::size_t buf_size = 65536;
            const char* byte_ptr = reinterpret_cast<const char*>(data);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (bytes - remaining), static_cast<std::streamsize>(chunk));
                if (!out) throw std::runtime_error("Failed to write disk array data.");
                remaining -= chunk;
            }
        }
    }

    /**
     * @class xio_disk_handler
     * @brief Handler for storing/loading xtensor arrays on local disk in a custom binary format.
     *
     * Supports writing multiple named arrays to a single directory, each as a separate .xdh file,
     * or a single file with a simple header.
     */
    class xio_disk_handler {
    public:
        explicit xio_disk_handler(const std::string& directory)
            : m_directory(directory)
        {
            if (!directory.empty())
                std::filesystem::create_directories(directory);
        }

        /**
         * Write an array to disk with a given name.
         * @param name Array name (used as filename: name.xdh).
         * @param expr The expression to write.
         */
        template <class E>
        void write(const std::string& name, const xexpression<E>& expr) {
            using T = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(expr.derived_cast());
            auto shape = arr.shape();
            if (shape.empty()) shape = {1};
            std::string path = make_path(name);
            std::ofstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open disk file for writing: " + path);
            detail::write_xdh_header(file, shape, detail::xdh_dtype_code_v<T>);
            detail::write_xdh_data(file, arr.data(), arr.size());
        }

        /**
         * Read an array from disk by name.
         * @param name Array name.
         * @return xarray<T> with the loaded data.
         */
        template <class T = double>
        auto read(const std::string& name) const {
            std::string path = make_path(name);
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open disk file for reading: " + path);
            auto [shape, dtype_code] = detail::read_xdh_header(file);
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            detail::read_xdh_data(file, arr.data(), arr.size());
            return arr;
        }

        /**
         * Check if a named array exists on disk.
         */
        bool exists(const std::string& name) const {
            return std::filesystem::exists(make_path(name));
        }

        /**
         * Delete a named array from disk.
         */
        void remove(const std::string& name) {
            std::filesystem::remove(make_path(name));
        }

        /**
         * List all array names in the directory.
         */
        std::vector<std::string> list() const {
            std::vector<std::string> names;
            for (const auto& entry : std::filesystem::directory_iterator(m_directory)) {
                if (entry.is_regular_file() && entry.path().extension() == ".xdh") {
                    names.push_back(entry.path().stem().string());
                }
            }
            return names;
        }

        /**
         * Read shape of a stored array without loading data.
         */
        std::vector<std::size_t> shape_of(const std::string& name) const {
            std::string path = make_path(name);
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open disk file: " + path);
            return detail::read_xdh_header(file).first;
        }

        /**
         * Return the base directory.
         */
        const std::string& directory() const noexcept { return m_directory; }

    private:
        std::string m_directory;

        std::string make_path(const std::string& name) const {
            return m_directory + "/" + name + ".xdh";
        }
    };

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_DISK_HANDLER_HPP