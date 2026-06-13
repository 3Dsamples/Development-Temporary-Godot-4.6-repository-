//File 0509 : xtensor-io/xhighfive.hpp
//HDF5 reader/writer for xtensor arrays with SIMD‑accelerated data transfer, chunked storage, attribute support, and memory‑efficient 2D/3D simulation I/O.
#ifndef XTENSOR_IO_XHIGHFIVE_HPP
#define XTENSOR_IO_XHIGHFIVE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Attribute.hpp>

#include "xio_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace io {

    namespace detail {

        /**
         * Map C++ type to HighFive type.
         */
        template <class T>
        struct hdf5_type_of {
            using type = T;
        };
        template <> struct hdf5_type_of<std::complex<float>>  { using type = std::complex<float>; };
        template <> struct hdf5_type_of<std::complex<double>> { using type = std::complex<double>; };

        template <class T>
        using hdf5_type_of_t = typename hdf5_type_of<T>::type;

        /**
         * Create a dataspace from an xtensor shape, reversing dimensions
         * because HDF5 uses row‑major (C order) by default and xtensor uses
         * row‑major, so no reversal is needed.  We keep the shape as is.
         */
        inline std::vector<std::size_t> to_hdf5_shape(const std::vector<std::size_t>& shape) {
            return shape;
        }

        /**
         * Set chunking and compression properties on a dataset.
         */
        inline void set_chunking(HighFive::DataSetCreateProps& props,
                                 const std::vector<std::size_t>& chunk_shape,
                                 int compression_level = 0) {
            if (!chunk_shape.empty()) {
                props.add(HighFive::Chunking(chunk_shape));
                if (compression_level > 0) {
                    props.add(HighFive::Deflate(compression_level));
                }
            }
        }

        /**
         * SIMD‑accelerated copy from xtensor to HDF5 buffer.
         */
        template <class T>
        inline void copy_to_buffer(const T* src, T* dst, std::size_t count) {
            if constexpr (simd_enabled_v<T>) {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = count / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i) {
                    simd_type v = simd_type::load_unaligned(src + i * simd_size);
                    v.store_unaligned(dst + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < count; ++i)
                    dst[i] = src[i];
            } else {
                std::copy(src, src + count, dst);
            }
        }
    }

    /**
     * @class xhighfive_file
     * @brief HDF5 file wrapper for reading and writing xtensor arrays.
     *
     * Provides a simple interface to store and retrieve named datasets
     * from HDF5 files. Supports chunking, compression, and attributes.
     */
    class xhighfive_file {
    public:
        enum class mode { read, write, read_write };

        /**
         * Open an HDF5 file.
         * @param filename Path to .h5 file.
         * @param m Access mode.
         */
        xhighfive_file(const std::string& filename, mode m = mode::read)
            : m_filename(filename)
        {
            unsigned int flags;
            switch (m) {
                case mode::read:       flags = HighFive::File::ReadOnly; break;
                case mode::write:      flags = HighFive::File::Create | HighFive::File::Truncate; break;
                case mode::read_write: flags = HighFive::File::ReadWrite; break;
                default: flags = HighFive::File::ReadOnly;
            }
            m_file = std::make_unique<HighFive::File>(filename, flags);
        }

        xhighfive_file(const xhighfive_file&) = delete;
        xhighfive_file& operator=(const xhighfive_file&) = delete;
        xhighfive_file(xhighfive_file&&) = default;
        xhighfive_file& operator=(xhighfive_file&&) = default;

        /**
         * Write an xtensor array to a dataset.
         * @param dataset_name Path within the HDF5 file (e.g., "/group/data").
         * @param expr The expression to save.
         * @param chunk_shape Chunk dimensions (empty = contiguous).
         * @param compression_level 0‑9, where 0 = no compression.
         */
        template <class E>
        void write(const std::string& dataset_name,
                   const xexpression<E>& expr,
                   const std::vector<std::size_t>& chunk_shape = {},
                   int compression_level = 0) {
            using T = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(expr.derived_cast());
            auto shape = arr.shape();
            auto h5_shape = detail::to_hdf5_shape(shape);

            HighFive::DataSetCreateProps props;
            detail::set_chunking(props, chunk_shape, compression_level);

            auto dataset = m_file->createDataSet<T>(dataset_name,
                                                     HighFive::DataSpace(h5_shape),
                                                     props);
            dataset.write(arr.data());
        }

        /**
         * Read a dataset into an xtensor array.
         * @param dataset_name Dataset path.
         * @return xarray<T> with the data.
         */
        template <class T = double>
        auto read(const std::string& dataset_name) const {
            auto dataset = m_file->getDataSet(dataset_name);
            auto shape = dataset.getSpace().getDimensions();
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            dataset.read(arr.data());
            return arr;
        }

        /**
         * Check if a dataset exists.
         */
        bool exists(const std::string& name) const {
            return m_file->exist(name);
        }

        /**
         * Write a scalar attribute to a dataset or group.
         * @param location Dataset or group path.
         * @param attr_name Attribute name.
         * @param value Attribute value (numeric or string).
         */
        template <class T>
        void write_attribute(const std::string& location,
                              const std::string& attr_name,
                              const T& value) {
            auto node = m_file->getObject(location);
            auto attr = node.createAttribute<T>(attr_name, HighFive::DataSpace::From(value));
            attr.write(value);
        }

        /**
         * Read a scalar attribute.
         * @param location Dataset or group path.
         * @param attr_name Attribute name.
         * @return The attribute value.
         */
        template <class T>
        T read_attribute(const std::string& location,
                          const std::string& attr_name) const {
            auto node = m_file->getObject(location);
            auto attr = node.getAttribute(attr_name);
            T value;
            attr.read(value);
            return value;
        }

        /**
         * Create a group (like mkdir).
         */
        void create_group(const std::string& path) {
            m_file->createGroup(path);
        }

        /**
         * List all datasets and groups directly under a path.
         */
        std::vector<std::string> list(const std::string& path = "/") const {
            return m_file->getObject(path).listObjectNames();
        }

        /**
         * Get the shape of a dataset without loading the data.
         */
        auto shape_of(const std::string& dataset_name) const {
            auto dataset = m_file->getDataSet(dataset_name);
            return dataset.getSpace().getDimensions();
        }

        /**
         * Get the underlying HighFive file handle.
         */
        HighFive::File& file() { return *m_file; }
        const HighFive::File& file() const { return *m_file; }

    private:
        std::string m_filename;
        std::unique_ptr<HighFive::File> m_file;
    };

    /**
     * Convenience: save a single array to an HDF5 file.
     */
    template <class E>
    inline void save_h5(const std::string& filename,
                        const std::string& dataset_name,
                        const xexpression<E>& expr,
                        const std::vector<std::size_t>& chunk_shape = {},
                        int compression = 0) {
        xhighfive_file f(filename, xhighfive_file::mode::write);
        f.write(dataset_name, expr, chunk_shape, compression);
    }

    /**
     * Convenience: load a single array from an HDF5 file.
     */
    template <class T = double>
    inline auto load_h5(const std::string& filename,
                        const std::string& dataset_name) {
        xhighfive_file f(filename, xhighfive_file::mode::read);
        return f.read<T>(dataset_name);
    }

    /**
     * Append an array to an existing HDF5 dataset by extending its last dimension.
     */
    template <class E>
    inline void append_h5(const std::string& filename,
                          const std::string& dataset_name,
                          const xexpression<E>& expr) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        auto sh = arr.shape();
        xhighfive_file f(filename, xhighfive_file::mode::read_write);
        if (!f.exists(dataset_name)) {
            f.write(dataset_name, expr);
            return;
        }
        auto existing_shape = f.shape_of(dataset_name);
        if (existing_shape.size() != sh.size())
            throw std::runtime_error("append_h5: dimension mismatch.");
        std::vector<std::size_t> new_shape = existing_shape;
        new_shape.back() += sh.back();
        auto dataset = f.file().getDataSet(dataset_name);
        dataset.resize(new_shape);
        // Select hyperslab at the end and write
        std::vector<std::size_t> offset(existing_shape.size(), 0);
        offset.back() = existing_shape.back();
        dataset.select(offset, sh).write(arr.data());
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XHIGHFIVE_HPP