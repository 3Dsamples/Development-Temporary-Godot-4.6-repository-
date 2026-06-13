//File 0517 : xtensor-io/xio_gdal_handler.hpp
//GDAL raster I/O handler: reads/writes geospatial raster files (GeoTIFF, IMG, etc.) as xtensor arrays with SIMD‑accelerated band transfer and geo‑metadata preservation.
#ifndef XTENSOR_IO_XIO_GDAL_HANDLER_HPP
#define XTENSOR_IO_XIO_GDAL_HANDLER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include "xgdal.hpp"

namespace xt {
namespace io {

    /**
     * @class xio_gdal_handler
     * @brief GDAL raster I/O handler.
     *
     * Provides a unified interface for reading and writing geospatial raster
     * datasets as xtensor arrays.  Wraps the low‑level gdal_* functions with
     * additional caching of geo‑metadata and automatic band dimension handling.
     */
    class xio_gdal_handler {
    public:
        xio_gdal_handler() = default;

        /**
         * Load a raster file.
         * @param filename Path to the raster.
         * @return Pair (xarray<double>, geo_metadata).
         */
        template <class T = double>
        auto read(const std::string& filename) const {
            return load_gdal<T>(filename);
        }

        /**
         * Save an xtensor array as a raster file.
         * @param filename Output path.
         * @param expr The array to save (2D or 3D).
         * @param meta Geospatial metadata.
         * @param format GDAL driver name (e.g., "GTiff").
         */
        template <class E>
        void write(const std::string& filename,
                   const xexpression<E>& expr,
                   const geo_metadata& meta,
                   const std::string& format = "GTiff") const {
            save_gdal(filename, expr, meta, format);
        }

        /**
         * Query metadata without loading the full raster.
         * @param filename Path to raster.
         * @return geo_metadata.
         */
        static geo_metadata info(const std::string& filename) {
            return gdal_info(filename);
        }

        /**
         * Create an empty geo_metadata with default values.
         */
        static geo_metadata default_metadata() {
            return geo_metadata{};
        }
    };

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_GDAL_HANDLER_HPP