//File 0515 : xtensor-io/xio_file_wrapper.hpp
//File wrapper for xtensor arrays: provides a uniform interface for local file I/O with SIMD-accelerated data transfer, automatic header handling, and support for raw, NPY, and XTB formats.
#ifndef XTENSOR_IO_XIO_FILE_WRAPPER_HPP
#define XTENSOR_IO_XIO_FILE_WRAPPER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

    /**
     * @class xio_file_wrapper
     * @brief Unified file wrapper for reading/writing xtensor arrays in multiple formats.
     *
     * Detects file format from extension or magic bytes and dispatches to the
     * appropriate reader/writer. Supports .npy (NumPy), .xtb (custom binary),
     * .csv, .json, and raw binary. The wrapper handles shape/dtype metadata
     * and uses SIMD‑accelerated block transfers for performance.
     */
    class xio_file_wrapper {
    public:
        /**
         * @enum file_format
         * @brief Supported file formats.
         */
        enum class file_format : uint8_t {
            auto_detect,
            npy,
            xtb,    // custom xtensor binary
            csv,
            json,
            raw_binary
        };

        xio_file_wrapper() noexcept = default;

        /**
         * Set the file path and optionally the format.
         */
        explicit xio_file_wrapper(const std::string& filename,
                                   file_format fmt = file_format::auto_detect)
            : m_filename(filename), m_format(fmt)
        {
            if (fmt == file_format::auto_detect)
                m_format = detect_format(filename);
        }

        /**
         * Write an xtensor expression to the file.
         * @param expr The expression to save.
         */
        template <class E>
        void write(const xexpression<E>& expr) {
            switch (m_format) {
                case file_format::npy:
                    save_npy(m_filename, expr);
                    break;
                case file_format::xtb:
                    save_binary(m_filename, expr);
                    break;
                case file_format::csv:
                    save_csv(m_filename, expr);
                    break;
                case file_format::json:
                    save_json(m_filename, expr);
                    break;
                case file_format::raw_binary:
                    dump_raw(m_filename, expr);
                    break;
                default:
                    throw std::runtime_error("xio_file_wrapper: unknown format for writing.");
            }
        }

        /**
         * Read the file into an xtensor array.
         * @return xarray<T> with the data.
         */
        template <class T = double>
        auto read() const {
            switch (m_format) {
                case file_format::npy:
                    return load_npy<T>(m_filename);
                case file_format::xtb:
                    return load_binary<T>(m_filename);
                case file_format::csv:
                    return load_csv<T>(m_filename);
                case file_format::json:
                    return load_json<T>(m_filename);
                case file_format::raw_binary:
                    return load_raw<T>(m_filename, {});
                default:
                    throw std::runtime_error("xio_file_wrapper: unknown format for reading.");
            }
        }

        /**
         * Read shape of the stored array without loading the full data.
         * Only supported for NPY and XTB formats.
         */
        std::vector<std::size_t> shape() const {
            if (m_format == file_format::npy) {
                return npy_shape(m_filename);
            } else if (m_format == file_format::xtb) {
                return xtb_shape(m_filename);
            }
            throw std::runtime_error("Shape query not supported for this format.");
        }

        /**
         * Detect file format from extension.
         */
        static file_format detect_format(const std::string& filename) {
            auto pos = filename.rfind('.');
            if (pos == std::string::npos) return file_format::xtb;
            std::string ext = filename.substr(pos);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".npy")  return file_format::npy;
            if (ext == ".xtb")  return file_format::xtb;
            if (ext == ".csv")  return file_format::csv;
            if (ext == ".json") return file_format::json;
            if (ext == ".bin" || ext == ".dat" || ext == ".raw")
                return file_format::raw_binary;
            return file_format::xtb;
        }

        const std::string& filename() const noexcept { return m_filename; }
        file_format format() const noexcept { return m_format; }
        void set_format(file_format fmt) noexcept { m_format = fmt; }

    private:
        std::string m_filename;
        file_format m_format = file_format::auto_detect;

        // Helper to read NPY shape only
        static std::vector<std::size_t> npy_shape(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open NPY file: " + filename);
            char magic[6];
            file.read(magic, 6);
            std::uint16_t hlen;
            file.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
            std::string header(hlen, '\0');
            file.read(&header[0], hlen);
            std::regex shape_re("'shape'\\s*:\\s*\\(([^)]*)\\)");
            std::smatch m;
            std::vector<std::size_t> shape;
            if (std::regex_search(header, m, shape_re)) {
                std::string shape_str = m[1].str();
                std::regex num_re("\\d+");
                auto begin = std::sregex_iterator(shape_str.begin(), shape_str.end(), num_re);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it)
                    shape.push_back(std::stoull(it->str()));
            }
            if (shape.empty()) shape.push_back(1);
            return shape;
        }

        // Helper to read XTB shape only
        static std::vector<std::size_t> xtb_shape(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open XTB file: " + filename);
            unsigned char magic[4];
            file.read(reinterpret_cast<char*>(magic), 4);
            std::uint8_t version;
            file.read(reinterpret_cast<char*>(&version), 1);
            std::uint8_t dtype_code;
            file.read(reinterpret_cast<char*>(&dtype_code), 1);
            std::uint16_t ndim;
            file.read(reinterpret_cast<char*>(&ndim), 2);
            std::vector<std::size_t> shape(ndim);
            for (std::uint16_t i = 0; i < ndim; ++i) {
                std::uint64_t dim;
                file.read(reinterpret_cast<char*>(&dim), 8);
                shape[i] = static_cast<std::size_t>(dim);
            }
            return shape;
        }
    };

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_FILE_WRAPPER_HPP