// io/ximage.hpp

#ifndef XTENSOR_XIMAGE_HPP
#define XTENSOR_XIMAGE_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <memory>
#include <map>
#include <functional>

// Optionally include external libraries if available
#if __has_include(<png.h>)
    #define XTENSOR_HAS_PNG 1
    #include <png.h>
#else
    #define XTENSOR_HAS_PNG 0
#endif

#if __has_include(<jpeglib.h>)
    #define XTENSOR_HAS_JPEG 1
    #include <jpeglib.h>
#else
    #define XTENSOR_HAS_JPEG 0
#endif

#if __has_include(<tiffio.h>)
    #define XTENSOR_HAS_TIFF 1
    #include <tiffio.h>
#else
    #define XTENSOR_HAS_TIFF 0
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace io
        {
            // --------------------------------------------------------------------
            // Pixel type and image class
            // --------------------------------------------------------------------
            enum class ColorSpace
            {
                GRAYSCALE,
                RGB,
                RGBA,
                HSV,
                LAB,
                YUV,
                CMYK,
                UNKNOWN
            };

            enum class DataType
            {
                UINT8,
                UINT16,
                FLOAT32,
                FLOAT64
            };

            // Type traits for data type
            template <DataType DT> struct data_type_traits;
            template <> struct data_type_traits<DataType::UINT8> { using type = uint8_t; static constexpr size_t size = 1; };
            template <> struct data_type_traits<DataType::UINT16> { using type = uint16_t; static constexpr size_t size = 2; };
            template <> struct data_type_traits<DataType::FLOAT32> { using type = float; static constexpr size_t size = 4; };
            template <> struct data_type_traits<DataType::FLOAT64> { using type = double; static constexpr size_t size = 8; };

            // Base image class using type erasure
            class Image
            {
            public:
                using size_type = std::size_t;

                virtual ~Image() = default;

                virtual size_type width() const = 0;
                virtual size_type height() const = 0;
                virtual size_type channels() const = 0;
                virtual ColorSpace color_space() const = 0;
                virtual DataType data_type() const = 0;
                virtual size_type element_size() const = 0;

                // Access to raw data
                virtual const void* data() const = 0;
                virtual void* data() = 0;
                virtual size_type data_size() const = 0;

                // Convert to xarray of specific type
                virtual xarray_container<uint8_t> to_uint8() const = 0;
                virtual xarray_container<float> to_float() const = 0;
                virtual xarray_container<double> to_double() const = 0;

                // Clone
                virtual std::unique_ptr<Image> clone() const = 0;

                // Basic operations
                virtual void resize(size_type new_width, size_type new_height) = 0;
                virtual void convert_color_space(ColorSpace target) = 0;
                virtual void convert_data_type(DataType target) = 0;
            };

            // Templated concrete image class
            template <typename T, ColorSpace CS>
            class TypedImage : public Image
            {
            public:
                using value_type = T;
                using container_type = xarray_container<T>;
                static constexpr ColorSpace color_space_v = CS;

                TypedImage() = default;
                
                TypedImage(size_type width, size_type height, size_type channels_override = 0)
                    : m_width(width), m_height(height)
                {
                    size_type ch = channels_override ? channels_override : default_channels();
                    m_data = container_type({height, width, ch});
                }

                explicit TypedImage(const container_type& data, ColorSpace cs = CS)
                    : m_data(data), m_color_space(cs)
                {
                    if (data.dimension() != 3)
                        XTENSOR_THROW(std::invalid_argument, "TypedImage: data must be 3D (HxWxC)");
                    m_height = data.shape()[0];
                    m_width = data.shape()[1];
                    m_channels = data.shape()[2];
                }

                size_type width() const override { return m_width; }
                size_type height() const override { return m_height; }
                size_type channels() const override { return m_channels; }
                ColorSpace color_space() const override { return m_color_space; }
                DataType data_type() const override;
                size_type element_size() const override { return sizeof(T); }

                const void* data() const override { return m_data.data(); }
                void* data() override { return m_data.data(); }
                size_type data_size() const override { return m_data.size() * sizeof(T); }

                const container_type& array() const { return m_data; }
                container_type& array() { return m_data; }

                // Access operators
                T& operator()(size_type y, size_type x, size_type c) { return m_data(y, x, c); }
                const T& operator()(size_type y, size_type x, size_type c) const { return m_data(y, x, c); }

                xarray_container<uint8_t> to_uint8() const override;
                xarray_container<float> to_float() const override;
                xarray_container<double> to_double() const override;

                std::unique_ptr<Image> clone() const override
                {
                    return std::make_unique<TypedImage<T, CS>>(m_data, m_color_space);
                }

                void resize(size_type new_width, size_type new_height) override;
                void convert_color_space(ColorSpace target) override;
                void convert_data_type(DataType target) override;

                static size_type default_channels()
                {
                    if constexpr (CS == ColorSpace::GRAYSCALE) return 1;
                    else if constexpr (CS == ColorSpace::RGB) return 3;
                    else if constexpr (CS == ColorSpace::RGBA) return 4;
                    else return 3;
                }

            private:
                size_type m_width = 0;
                size_type m_height = 0;
                size_type m_channels = default_channels();
                ColorSpace m_color_space = CS;
                container_type m_data;
            };

            // Data type deduction
            template <typename T, ColorSpace CS>
            DataType TypedImage<T, CS>::data_type() const
            {
                if constexpr (std::is_same_v<T, uint8_t>) return DataType::UINT8;
                else if constexpr (std::is_same_v<T, uint16_t>) return DataType::UINT16;
                else if constexpr (std::is_same_v<T, float>) return DataType::FLOAT32;
                else if constexpr (std::is_same_v<T, double>) return DataType::FLOAT64;
                else return DataType::FLOAT32;
            }

            // Conversion implementations
            template <typename T, ColorSpace CS>
            xarray_container<uint8_t> TypedImage<T, CS>::to_uint8() const
            {
                xarray_container<uint8_t> result({m_height, m_width, m_channels});
                if constexpr (std::is_same_v<T, uint8_t>)
                {
                    std::copy(m_data.begin(), m_data.end(), result.begin());
                }
                else
                {
                    double min_val = xt::amin(m_data)();
                    double max_val = xt::amax(m_data)();
                    double scale = (max_val > min_val) ? 255.0 / (max_val - min_val) : 1.0;
                    for (size_type i = 0; i < m_data.size(); ++i)
                    {
                        double val = static_cast<double>(m_data.flat(i));
                        val = (val - min_val) * scale;
                        result.flat(i) = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
                    }
                }
                return result;
            }

            template <typename T, ColorSpace CS>
            xarray_container<float> TypedImage<T, CS>::to_float() const
            {
                xarray_container<float> result({m_height, m_width, m_channels});
                if constexpr (std::is_same_v<T, float>)
                {
                    std::copy(m_data.begin(), m_data.end(), result.begin());
                }
                else
                {
                    double scale = 1.0;
                    if constexpr (std::is_integral_v<T>)
                        scale = 1.0 / 255.0;
                    for (size_type i = 0; i < m_data.size(); ++i)
                        result.flat(i) = static_cast<float>(static_cast<double>(m_data.flat(i)) * scale);
                }
                return result;
            }

            template <typename T, ColorSpace CS>
            xarray_container<double> TypedImage<T, CS>::to_double() const
            {
                xarray_container<double> result({m_height, m_width, m_channels});
                double scale = 1.0;
                if constexpr (std::is_integral_v<T>)
                    scale = 1.0 / 255.0;
                for (size_type i = 0; i < m_data.size(); ++i)
                    result.flat(i) = static_cast<double>(m_data.flat(i)) * scale;
                return result;
            }

            template <typename T, ColorSpace CS>
            void TypedImage<T, CS>::resize(size_type new_width, size_type new_height)
            {
                if (new_width == m_width && new_height == m_height) return;
                container_type new_data({new_height, new_width, m_channels});
                // Bilinear interpolation
                double scale_x = static_cast<double>(m_width) / new_width;
                double scale_y = static_cast<double>(m_height) / new_height;
                for (size_type y = 0; y < new_height; ++y)
                {
                    for (size_type x = 0; x < new_width; ++x)
                    {
                        double src_x = (x + 0.5) * scale_x - 0.5;
                        double src_y = (y + 0.5) * scale_y - 0.5;
                        int x0 = std::max(0, static_cast<int>(std::floor(src_x)));
                        int y0 = std::max(0, static_cast<int>(std::floor(src_y)));
                        int x1 = std::min(static_cast<int>(m_width) - 1, x0 + 1);
                        int y1 = std::min(static_cast<int>(m_height) - 1, y0 + 1);
                        double wx = src_x - x0;
                        double wy = src_y - y0;
                        for (size_type c = 0; c < m_channels; ++c)
                        {
                            double v00 = m_data(y0, x0, c);
                            double v01 = m_data(y0, x1, c);
                            double v10 = m_data(y1, x0, c);
                            double v11 = m_data(y1, x1, c);
                            double v0 = v00 * (1 - wx) + v01 * wx;
                            double v1 = v10 * (1 - wx) + v11 * wx;
                            double val = v0 * (1 - wy) + v1 * wy;
                            new_data(y, x, c) = static_cast<T>(val);
                        }
                    }
                }
                m_data = std::move(new_data);
                m_width = new_width;
                m_height = new_height;
            }

            template <typename T, ColorSpace CS>
            void TypedImage<T, CS>::convert_color_space(ColorSpace target)
            {
                if (target == m_color_space) return;
                // Implement common conversions
                if (m_color_space == ColorSpace::RGB && target == ColorSpace::GRAYSCALE)
                {
                    container_type new_data({m_height, m_width, 1});
                    for (size_type y = 0; y < m_height; ++y)
                        for (size_type x = 0; x < m_width; ++x)
                            new_data(y, x, 0) = static_cast<T>(0.2989 * m_data(y,x,0) + 0.5870 * m_data(y,x,1) + 0.1140 * m_data(y,x,2));
                    m_data = std::move(new_data);
                    m_channels = 1;
                    m_color_space = target;
                }
                else if (m_color_space == ColorSpace::GRAYSCALE && target == ColorSpace::RGB)
                {
                    container_type new_data({m_height, m_width, 3});
                    for (size_type y = 0; y < m_height; ++y)
                        for (size_type x = 0; x < m_width; ++x)
                            for (size_type c = 0; c < 3; ++c)
                                new_data(y, x, c) = m_data(y, x, 0);
                    m_data = std::move(new_data);
                    m_channels = 3;
                    m_color_space = target;
                }
                else if (m_color_space == ColorSpace::RGB && target == ColorSpace::RGBA)
                {
                    container_type new_data({m_height, m_width, 4});
                    for (size_type y = 0; y < m_height; ++y)
                        for (size_type x = 0; x < m_width; ++x)
                        {
                            for (size_type c = 0; c < 3; ++c)
                                new_data(y, x, c) = m_data(y, x, c);
                            new_data(y, x, 3) = static_cast<T>(1);
                        }
                    m_data = std::move(new_data);
                    m_channels = 4;
                    m_color_space = target;
                }
                else if (m_color_space == ColorSpace::RGBA && target == ColorSpace::RGB)
                {
                    container_type new_data({m_height, m_width, 3});
                    for (size_type y = 0; y < m_height; ++y)
                        for (size_type x = 0; x < m_width; ++x)
                            for (size_type c = 0; c < 3; ++c)
                                new_data(y, x, c) = m_data(y, x, c);
                    m_data = std::move(new_data);
                    m_channels = 3;
                    m_color_space = target;
                }
                else
                {
                    XTENSOR_THROW(std::runtime_error, "Color space conversion not implemented for this pair");
                }
            }

            template <typename T, ColorSpace CS>
            void TypedImage<T, CS>::convert_data_type(DataType target)
            {
                DataType current = data_type();
                if (current == target) return;
                // This would require changing the template type, so we'd need to create a new image.
                // Not implemented in-place; user should use to_xxx and create new image.
                XTENSOR_THROW(std::runtime_error, "convert_data_type not supported in-place; use to_uint8/to_float etc.");
            }

            // Aliases
            using ImageGrayU8 = TypedImage<uint8_t, ColorSpace::GRAYSCALE>;
            using ImageGrayF32 = TypedImage<float, ColorSpace::GRAYSCALE>;
            using ImageRGBU8 = TypedImage<uint8_t, ColorSpace::RGB>;
            using ImageRGBF32 = TypedImage<float, ColorSpace::RGB>;
            using ImageRGBAU8 = TypedImage<uint8_t, ColorSpace::RGBA>;
            using ImageRGBAF32 = TypedImage<float, ColorSpace::RGBA>;

            // --------------------------------------------------------------------
            // Image I/O functions
            // --------------------------------------------------------------------
            namespace detail
            {
                // Write PPM/PGM (binary)
                template <typename T>
                void write_ppm(const std::string& filename, const TypedImage<T, ColorSpace::RGB>& img)
                {
                    auto u8_img = img.to_uint8();
                    std::ofstream file(filename, std::ios::binary);
                    if (!file) XTENSOR_THROW(std::runtime_error, "Cannot open file for writing: " + filename);
                    file << "P6\n" << img.width() << " " << img.height() << "\n255\n";
                    file.write(reinterpret_cast<const char*>(u8_img.data()), u8_img.size());
                }

                template <typename T>
                void write_pgm(const std::string& filename, const TypedImage<T, ColorSpace::GRAYSCALE>& img)
                {
                    auto u8_img = img.to_uint8();
                    std::ofstream file(filename, std::ios::binary);
                    if (!file) XTENSOR_THROW(std::runtime_error, "Cannot open file for writing: " + filename);
                    file << "P5\n" << img.width() << " " << img.height() << "\n255\n";
                    file.write(reinterpret_cast<const char*>(u8_img.data()), u8_img.size());
                }

                // Read PPM/PGM
                inline std::unique_ptr<Image> read_ppm(const std::string& filename)
                {
                    std::ifstream file(filename, std::ios::binary);
                    if (!file) XTENSOR_THROW(std::runtime_error, "Cannot open file: " + filename);
                    std::string magic;
                    file >> magic;
                    if (magic != "P6" && magic != "P5")
                        XTENSOR_THROW(std::runtime_error, "Unsupported PPM/PGM format: " + magic);
                    size_t w, h, maxval;
                    file >> w >> h >> maxval;
                    file.get(); // consume newline
                    if (magic == "P6")
                    {
                        auto img = std::make_unique<ImageRGBU8>(w, h);
                        file.read(reinterpret_cast<char*>(img->data()), img->data_size());
                        return img;
                    }
                    else // P5
                    {
                        auto img = std::make_unique<ImageGrayU8>(w, h);
                        file.read(reinterpret_cast<char*>(img->data()), img->data_size());
                        return img;
                    }
                }

#if XTENSOR_HAS_PNG
                // PNG writing using libpng
                inline void write_png(const std::string& filename, const Image& img)
                {
                    auto u8 = img.to_uint8();
                    size_t w = img.width(), h = img.height();
                    int color_type;
                    if (img.color_space() == ColorSpace::GRAYSCALE)
                        color_type = PNG_COLOR_TYPE_GRAY;
                    else if (img.color_space() == ColorSpace::RGB)
                        color_type = PNG_COLOR_TYPE_RGB;
                    else if (img.color_space() == ColorSpace::RGBA)
                        color_type = PNG_COLOR_TYPE_RGBA;
                    else
                        XTENSOR_THROW(std::runtime_error, "Unsupported color space for PNG");

                    FILE* fp = fopen(filename.c_str(), "wb");
                    if (!fp) XTENSOR_THROW(std::runtime_error, "Cannot open PNG file for writing");
                    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
                    png_infop info = png_create_info_struct(png);
                    if (setjmp(png_jmpbuf(png)))
                    {
                        png_destroy_write_struct(&png, &info);
                        fclose(fp);
                        XTENSOR_THROW(std::runtime_error, "PNG write error");
                    }
                    png_init_io(png, fp);
                    png_set_IHDR(png, info, w, h, 8, color_type, PNG_INTERLACE_NONE,
                                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
                    png_write_info(png, info);
                    std::vector<png_bytep> rows(h);
                    for (size_t y = 0; y < h; ++y)
                        rows[y] = const_cast<png_bytep>(u8.data() + y * w * img.channels());
                    png_write_image(png, rows.data());
                    png_write_end(png, nullptr);
                    png_destroy_write_struct(&png, &info);
                    fclose(fp);
                }

                inline std::unique_ptr<Image> read_png(const std::string& filename)
                {
                    FILE* fp = fopen(filename.c_str(), "rb");
                    if (!fp) XTENSOR_THROW(std::runtime_error, "Cannot open PNG file");
                    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
                    png_infop info = png_create_info_struct(png);
                    if (setjmp(png_jmpbuf(png)))
                    {
                        png_destroy_read_struct(&png, &info, nullptr);
                        fclose(fp);
                        XTENSOR_THROW(std::runtime_error, "PNG read error");
                    }
                    png_init_io(png, fp);
                    png_read_info(png, info);
                    size_t w = png_get_image_width(png, info);
                    size_t h = png_get_image_height(png, info);
                    int color_type = png_get_color_type(png, info);
                    int bit_depth = png_get_bit_depth(png, info);
                    if (bit_depth == 16) png_set_strip_16(png);
                    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
                    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
                    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
                    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
                        png_set_gray_to_rgb(png);
                    png_read_update_info(png, info);
                    size_t channels = png_get_channels(png, info);
                    std::vector<uint8_t> buffer(w * h * channels);
                    std::vector<png_bytep> rows(h);
                    for (size_t y = 0; y < h; ++y)
                        rows[y] = buffer.data() + y * w * channels;
                    png_read_image(png, rows.data());
                    png_destroy_read_struct(&png, &info, nullptr);
                    fclose(fp);
                    if (channels == 1)
                        return std::make_unique<ImageGrayU8>(w, h, 1);
                    else if (channels == 3)
                        return std::make_unique<ImageRGBU8>(w, h, 3);
                    else if (channels == 4)
                        return std::make_unique<ImageRGBAU8>(w, h, 4);
                    else
                        XTENSOR_THROW(std::runtime_error, "Unsupported PNG channel count");
                }
#endif // XTENSOR_HAS_PNG

            } // namespace detail

            // Public I/O functions
            inline std::unique_ptr<Image> imread(const std::string& filename)
            {
                // Detect format by extension
                std::string ext;
                size_t dot = filename.find_last_of('.');
                if (dot != std::string::npos) ext = filename.substr(dot);
                if (ext == ".ppm" || ext == ".pgm")
                    return detail::read_ppm(filename);
#if XTENSOR_HAS_PNG
                else if (ext == ".png")
                    return detail::read_png(filename);
#endif
                else
                    XTENSOR_THROW(std::runtime_error, "Unsupported image format: " + ext);
            }

            inline void imwrite(const std::string& filename, const Image& img)
            {
                std::string ext;
                size_t dot = filename.find_last_of('.');
                if (dot != std::string::npos) ext = filename.substr(dot);
                if (ext == ".ppm")
                {
                    if (img.color_space() != ColorSpace::RGB)
                        XTENSOR_THROW(std::runtime_error, "PPM requires RGB image");
                    const auto* rgb = dynamic_cast<const ImageRGBU8*>(&img);
                    if (rgb) detail::write_ppm(filename, *rgb);
                    else XTENSOR_THROW(std::runtime_error, "Image type mismatch for PPM");
                }
                else if (ext == ".pgm")
                {
                    if (img.color_space() != ColorSpace::GRAYSCALE)
                        XTENSOR_THROW(std::runtime_error, "PGM requires grayscale image");
                    const auto* gray = dynamic_cast<const ImageGrayU8*>(&img);
                    if (gray) detail::write_pgm(filename, *gray);
                    else XTENSOR_THROW(std::runtime_error, "Image type mismatch for PGM");
                }
#if XTENSOR_HAS_PNG
                else if (ext == ".png")
                    detail::write_png(filename, img);
#endif
                else
                    XTENSOR_THROW(std::runtime_error, "Unsupported output format: " + ext);
            }

            // Convenience factory functions
            inline std::unique_ptr<Image> zeros(size_t width, size_t height, ColorSpace cs = ColorSpace::RGB, DataType dt = DataType::UINT8)
            {
                if (dt == DataType::UINT8)
                {
                    if (cs == ColorSpace::GRAYSCALE) return std::make_unique<ImageGrayU8>(width, height);
                    else if (cs == ColorSpace::RGB) return std::make_unique<ImageRGBU8>(width, height);
                    else if (cs == ColorSpace::RGBA) return std::make_unique<ImageRGBAU8>(width, height);
                }
                else if (dt == DataType::FLOAT32)
                {
                    if (cs == ColorSpace::GRAYSCALE) return std::make_unique<ImageGrayF32>(width, height);
                    else if (cs == ColorSpace::RGB) return std::make_unique<ImageRGBF32>(width, height);
                    else if (cs == ColorSpace::RGBA) return std::make_unique<ImageRGBAF32>(width, height);
                }
                XTENSOR_THROW(std::invalid_argument, "Unsupported image type");
            }

            inline std::unique_ptr<Image> from_array(const xarray_container<uint8_t>& arr, ColorSpace cs)
            {
                if (cs == ColorSpace::GRAYSCALE && arr.shape()[2] == 1)
                    return std::make_unique<ImageGrayU8>(arr, cs);
                else if (cs == ColorSpace::RGB && arr.shape()[2] == 3)
                    return std::make_unique<ImageRGBU8>(arr, cs);
                else if (cs == ColorSpace::RGBA && arr.shape()[2] == 4)
                    return std::make_unique<ImageRGBAU8>(arr, cs);
                else
                    XTENSOR_THROW(std::invalid_argument, "from_array: array shape doesn't match color space");
            }

            inline std::unique_ptr<Image> from_array(const xarray_container<float>& arr, ColorSpace cs)
            {
                if (cs == ColorSpace::GRAYSCALE && arr.shape()[2] == 1)
                    return std::make_unique<ImageGrayF32>(arr, cs);
                else if (cs == ColorSpace::RGB && arr.shape()[2] == 3)
                    return std::make_unique<ImageRGBF32>(arr, cs);
                else if (cs == ColorSpace::RGBA && arr.shape()[2] == 4)
                    return std::make_unique<ImageRGBAF32>(arr, cs);
                else
                    XTENSOR_THROW(std::invalid_argument, "from_array: array shape doesn't match color space");
            }

        } // namespace io

        // Bring image classes into xt namespace
        using io::Image;
        using io::TypedImage;
        using io::ImageGrayU8;
        using io::ImageGrayF32;
        using io::ImageRGBU8;
        using io::ImageRGBF32;
        using io::ImageRGBAU8;
        using io::ImageRGBAF32;
        using io::ColorSpace;
        using io::DataType;
        using io::imread;
        using io::imwrite;
        using io::zeros;
        using io::from_array;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIMAGE_HPP

// io/ximage.hpp