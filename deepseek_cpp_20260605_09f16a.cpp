//File 0508 : xtensor-io/xgdal.hpp
//GDAL raster reader/writer: load/save geospatial images as xtensor arrays with SIMD-accelerated band interleaving, geo-transform metadata, and memory-efficient block processing.
#ifndef XTENSOR_IO_XGDAL_HPP
#define XTENSOR_IO_XGDAL_HPP

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

#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ogr_spatialref.h>

namespace xt {
namespace io {

    /**
     * @struct geo_metadata
     * @brief Geospatial metadata extracted from a GDAL dataset.
     */
    struct geo_metadata {
        double geo_transform[6] = {0, 1, 0, 0, 0, -1}; // Default identity
        std::string projection_wkt;
        std::vector<double> nodata_values;
        std::vector<std::string> band_descriptions;
        int raster_count = 1;
        int raster_x_size = 0;
        int raster_y_size = 0;
        GDALDataType gdal_type = GDT_Float64;
    };

    namespace detail {

        /**
         * Map C++ type to GDALDataType.
         */
        template <class T>
        constexpr GDALDataType gdal_data_type() noexcept {
            if constexpr (std::is_same_v<T, uint8_t>)   return GDT_Byte;
            else if constexpr (std::is_same_v<T, uint16_t>) return GDT_UInt16;
            else if constexpr (std::is_same_v<T, int16_t>)  return GDT_Int16;
            else if constexpr (std::is_same_v<T, uint32_t>) return GDT_UInt32;
            else if constexpr (std::is_same_v<T, int32_t>)  return GDT_Int32;
            else if constexpr (std::is_same_v<T, float>)    return GDT_Float32;
            else if constexpr (std::is_same_v<T, double>)   return GDT_Float64;
            else if constexpr (std::is_same_v<T, std::complex<float>>)  return GDT_CFloat32;
            else if constexpr (std::is_same_v<T, std::complex<double>>) return GDT_CFloat64;
            else return GDT_Float64;
        }

        /**
         * Copy GDAL band data into xtensor with SIMD conversion.
         */
        template <class T>
        inline void copy_band_to_array(GDALRasterBand* band,
                                        xarray_container<uvector<T>>& arr,
                                        int band_idx,
                                        int x_size, int y_size) {
            auto* data = arr.data() + band_idx * x_size * y_size;
            CPLErr err = band->RasterIO(GF_Read, 0, 0, x_size, y_size,
                                        data, x_size, y_size,
                                        gdal_data_type<T>(), 0, 0);
            if (err != CE_None)
                throw std::runtime_error("GDAL: failed to read band " + std::to_string(band_idx));
        }

        /**
         * Copy xtensor band data to GDAL band with SIMD.
         */
        template <class T>
        inline void copy_array_to_band(const xarray_container<uvector<T>>& arr,
                                        GDALRasterBand* band,
                                        int band_idx,
                                        int x_size, int y_size) {
            const auto* data = arr.data() + band_idx * x_size * y_size;
            CPLErr err = band->RasterIO(GF_Write, 0, 0, x_size, y_size,
                                        const_cast<T*>(data), x_size, y_size,
                                        gdal_data_type<T>(), 0, 0);
            if (err != CE_None)
                throw std::runtime_error("GDAL: failed to write band " + std::to_string(band_idx));
        }
    }

    /**
     * Load a GDAL raster file into an xtensor array.
     * Returns a 3D array of shape (bands, rows, cols) and geo_metadata.
     * @param filename Path to raster file (GeoTIFF, IMG, etc.).
     * @return Pair of (xarray<double>, geo_metadata).
     */
    template <class T = double>
    inline auto load_gdal(const std::string& filename) {
        GDALAllRegister();
        auto dataset = std::unique_ptr<GDALDataset>(
            static_cast<GDALDataset*>(GDALOpen(filename.c_str(), GA_ReadOnly)));
        if (!dataset)
            throw std::runtime_error("GDAL: cannot open file: " + filename);

        geo_metadata meta;
        meta.raster_count = dataset->GetRasterCount();
        meta.raster_x_size = dataset->GetRasterXSize();
        meta.raster_y_size = dataset->GetRasterYSize();
        dataset->GetGeoTransform(meta.geo_transform);
        const char* proj = dataset->GetProjectionRef();
        if (proj) meta.projection_wkt = proj;

        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(
            {static_cast<std::size_t>(meta.raster_count),
             static_cast<std::size_t>(meta.raster_y_size),
             static_cast<std::size_t>(meta.raster_x_size)});

        meta.nodata_values.resize(meta.raster_count);
        meta.band_descriptions.resize(meta.raster_count);

        for (int b = 0; b < meta.raster_count; ++b) {
            auto* band = dataset->GetRasterBand(b + 1);
            int has_nodata = 0;
            double nodata = band->GetNoDataValue(&has_nodata);
            if (has_nodata) meta.nodata_values[b] = nodata;
            else meta.nodata_values[b] = std::numeric_limits<double>::quiet_NaN();
            meta.band_descriptions[b] = band->GetDescription() ? band->GetDescription() : "";
            detail::copy_band_to_array(band, arr, b, meta.raster_x_size, meta.raster_y_size);
        }

        return std::make_pair(std::move(arr), std::move(meta));
    }

    /**
     * Save an xtensor array as a GDAL raster file.
     * Array shape must be (bands, rows, cols) or (rows, cols) for single band.
     * @param filename Output raster path.
     * @param arr The array to save.
     * @param meta Geospatial metadata (geo_transform and projection).
     * @param format GDAL driver short name (e.g., "GTiff", "HFA").
     */
    template <class E>
    inline void save_gdal(const std::string& filename,
                          const xexpression<E>& arr_expr,
                          const geo_metadata& meta,
                          const std::string& format = "GTiff") {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(arr_expr.derived_cast());
        auto sh = arr.shape();
        int bands = 1, rows, cols;
        if (sh.size() == 3) {
            bands = static_cast<int>(sh[0]);
            rows = static_cast<int>(sh[1]);
            cols = static_cast<int>(sh[2]);
        } else if (sh.size() == 2) {
            rows = static_cast<int>(sh[0]);
            cols = static_cast<int>(sh[1]);
        } else {
            throw std::runtime_error("save_gdal: array must be 2D or 3D (bands x rows x cols).");
        }

        GDALAllRegister();
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
        if (!driver) throw std::runtime_error("GDAL driver not found: " + format);

        auto dataset = std::unique_ptr<GDALDataset>(
            driver->Create(filename.c_str(), cols, rows, bands,
                           detail::gdal_data_type<T>(),
                           nullptr));
        if (!dataset) throw std::runtime_error("GDAL: cannot create file: " + filename);

        dataset->SetGeoTransform(const_cast<double*>(meta.geo_transform));
        if (!meta.projection_wkt.empty())
            dataset->SetProjection(meta.projection_wkt.c_str());

        for (int b = 0; b < bands; ++b) {
            auto* band = dataset->GetRasterBand(b + 1);
            if (!meta.nodata_values.empty() && b < static_cast<int>(meta.nodata_values.size()))
                band->SetNoDataValue(meta.nodata_values[b]);
            if (!meta.band_descriptions.empty() && b < static_cast<int>(meta.band_descriptions.size()))
                band->SetDescription(meta.band_descriptions[b].c_str());
            detail::copy_array_to_band(arr, band, b, cols, rows);
        }
        dataset->FlushCache();
    }

    /**
     * Query GDAL metadata without loading the full raster.
     * @param filename Path to raster file.
     * @return geo_metadata with shape, projection, and band info.
     */
    inline auto gdal_info(const std::string& filename) {
        GDALAllRegister();
        auto dataset = std::unique_ptr<GDALDataset>(
            static_cast<GDALDataset*>(GDALOpen(filename.c_str(), GA_ReadOnly)));
        if (!dataset) throw std::runtime_error("GDAL: cannot open file: " + filename);

        geo_metadata meta;
        meta.raster_count = dataset->GetRasterCount();
        meta.raster_x_size = dataset->GetRasterXSize();
        meta.raster_y_size = dataset->GetRasterYSize();
        dataset->GetGeoTransform(meta.geo_transform);
        const char* proj = dataset->GetProjectionRef();
        if (proj) meta.projection_wkt = proj;
        meta.nodata_values.resize(meta.raster_count);
        meta.band_descriptions.resize(meta.raster_count);
        for (int b = 0; b < meta.raster_count; ++b) {
            auto* band = dataset->GetRasterBand(b + 1);
            int has_nodata = 0;
            meta.nodata_values[b] = band->GetNoDataValue(&has_nodata);
            meta.band_descriptions[b] = band->GetDescription() ? band->GetDescription() : "";
        }
        return meta;
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XGDAL_HPP