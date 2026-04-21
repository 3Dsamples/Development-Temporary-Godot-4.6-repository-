// core/ximage.hpp
#ifndef XTENSOR_XIMAGE_HPP
#define XTENSOR_XIMAGE_HPP

// ----------------------------------------------------------------------------
// ximage.hpp – Image container and processing for xtensor
// ----------------------------------------------------------------------------
// This header provides a comprehensive image class with:
//   - Multi‑channel image representation (grayscale, RGB, RGBA, etc.)
//   - Pixel access and manipulation (individual, row, region)
//   - Color space conversions (RGB ↔ HSV, YUV, LAB, Grayscale, etc.)
//   - Image I/O: BMP, PNG (via stb_image), JPEG, TIFF, PPM, HDR
//   - Basic operations: resize, crop, rotate, flip, pad, border
//   - Filtering: Gaussian, median, bilateral, Sobel, Laplacian, custom kernel
//   - Morphological operations: erode, dilate, open, close, gradient, tophat
//   - Histogram: compute, equalize, match, back‑project
//   - Feature detection: Harris corners, FAST, ORB descriptors
//   - Transformations: affine, perspective, polar, log‑polar
//   - Drawing primitives: line, rectangle, circle, ellipse, text, fill
//   - FFT‑accelerated frequency‑domain filtering and convolution
//
// All pixel types are supported, including bignumber::BigNumber. FFT is used
// for frequency‑domain filtering and fast convolution.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstring>
#include <limits>
#include <functional>
#include <array>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsorting.hpp"
#include "xstats.hpp"
#include "xinterp.hpp"
#include "fft.hpp"
#include "xgraphics.hpp"
#include "io/xstb_image.hpp"
#include "io/xstb_image_write.hpp"
#include "io/xstb_image_resize.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace image
    {
        // ========================================================================
        // Pixel types and traits
        // ========================================================================
        enum class color_space
        {
            grayscale,
            rgb,
            rgba,
            bgr,
            bgra,
            hsv,
            yuv,
            lab,
            xyz,
            cmyk
        };

        template <class T, size_t Channels>
        struct pixel_traits
        {
            static constexpr size_t channels = Channels;
            using value_type = T;
        };

        // ========================================================================
        // Image class
        // ========================================================================
        template <class T, size_t Channels = 3>
        class ximage
        {
        public:
            using value_type = T;
            using pixel_type = std::array<T, Channels>;
            using reference = pixel_type&;
            using const_reference = const pixel_type&;
            using size_type = std::size_t;
            using shape_type = std::vector<size_type>;

            static constexpr size_t channels = Channels;

            // --------------------------------------------------------------------
            // Constructors
            // --------------------------------------------------------------------
            ximage();
            ximage(size_type width, size_type height, color_space cs = color_space::rgb);
            ximage(size_type width, size_type height, const T& fill_value, color_space cs = color_space::rgb);
            ximage(const xarray_container<T>& data, color_space cs = color_space::rgb);
            explicit ximage(const std::string& filename);

            // Copy and move
            ximage(const ximage&) = default;
            ximage(ximage&&) = default;
            ximage& operator=(const ximage&) = default;
            ximage& operator=(ximage&&) = default;

            // --------------------------------------------------------------------
            // Basic properties
            // --------------------------------------------------------------------
            size_type width() const noexcept;
            size_type height() const noexcept;
            size_type size() const noexcept;
            shape_type shape() const;
            bool empty() const noexcept;
            color_space colorspace() const noexcept;
            void set_colorspace(color_space cs) noexcept;

            // --------------------------------------------------------------------
            // Data access
            // --------------------------------------------------------------------
            const xarray_container<T>& data() const noexcept;
            xarray_container<T>& data() noexcept;

            // --------------------------------------------------------------------
            // Pixel access
            // --------------------------------------------------------------------
            const T* pixel(size_type x, size_type y) const;
            T* pixel(size_type x, size_type y);
            T& operator()(size_type x, size_type y, size_type c);
            const T& operator()(size_type x, size_type y, size_type c) const;
            void set_pixel(size_type x, size_type y, const std::array<T, Channels>& val);
            std::array<T, Channels> get_pixel(size_type x, size_type y) const;

            // --------------------------------------------------------------------
            // Row access
            // --------------------------------------------------------------------
            T* row(size_type y);
            const T* row(size_type y) const;

            // --------------------------------------------------------------------
            // Region of interest
            // --------------------------------------------------------------------
            ximage roi(size_type x, size_type y, size_type w, size_type h) const;
            void paste(const ximage& src, size_type dst_x, size_type dst_y);

            // --------------------------------------------------------------------
            // Clear / fill
            // --------------------------------------------------------------------
            void fill(const T& value);
            void fill(const std::array<T, Channels>& value);

            // --------------------------------------------------------------------
            // Clone
            // --------------------------------------------------------------------
            ximage clone() const;

            // --------------------------------------------------------------------
            // Color space conversions
            // --------------------------------------------------------------------
            ximage<T, 1> to_grayscale() const;
            template <size_t C = Channels, typename std::enable_if<C == 1, int>::type = 0>
            ximage<T, 3> to_rgb() const;
            template <size_t C = Channels, typename std::enable_if<C == 4, int>::type = 0>
            ximage<T, 3> to_rgb() const;
            ximage<T, 3> to_hsv() const;
            ximage<T, 3> hsv_to_rgb() const;
            ximage<T, 3> to_yuv() const;
            ximage<T, 3> yuv_to_rgb() const;
            ximage<T, 3> to_lab() const;
            ximage<T, 3> lab_to_rgb() const;
            ximage<T, 4> to_cmyk() const;
            ximage<T, 4> cmyk_to_rgb() const;

            // --------------------------------------------------------------------
            // Resize and geometric transforms
            // --------------------------------------------------------------------
            ximage resize(size_type new_width, size_type new_height,
                          const std::string& method = "bilinear") const;
            ximage crop(size_type x, size_type y, size_type w, size_type h) const;
            ximage rotate90(int times = 1) const;
            ximage rotate(T angle_deg, const std::string& method = "bilinear") const;
            ximage flip(bool horizontal, bool vertical) const;
            ximage pad(size_type top, size_type bottom, size_type left, size_type right,
                       const std::string& mode = "constant", T constant = T(0)) const;
            ximage affine_transform(const xarray_container<T>& matrix,
                                    const std::string& method = "bilinear") const;
            ximage perspective_transform(const xarray_container<T>& src_pts,
                                         const xarray_container<T>& dst_pts,
                                         const std::string& method = "bilinear") const;
            ximage polar_transform(bool inverse = false) const;
            ximage log_polar_transform(bool inverse = false) const;

            // --------------------------------------------------------------------
            // Filtering
            // --------------------------------------------------------------------
            ximage convolve(const xarray_container<T>& kernel,
                            const std::string& border = "reflect") const;
            ximage gaussian_blur(T sigma, int kernel_size = 0) const;
            ximage median_filter(int radius = 1) const;
            ximage bilateral_filter(T sigma_spatial, T sigma_range, int radius = 0) const;
            ximage box_filter(int radius) const;
            ximage sobel() const;
            ximage laplacian() const;
            ximage prewitt() const;
            ximage roberts() const;
            ximage scharr() const;
            ximage unsharp_mask(T sigma = T(1), T amount = T(1)) const;
            ximage wiener_filter(const ximage& noise_power_spectrum) const;

            // --------------------------------------------------------------------
            // Morphological operations
            // --------------------------------------------------------------------
            ximage erode(int kernel_size = 3) const;
            ximage dilate(int kernel_size = 3) const;
            ximage opening(int kernel_size = 3) const;
            ximage closing(int kernel_size = 3) const;
            ximage morph_gradient(int kernel_size = 3) const;
            ximage tophat(int kernel_size = 3) const;
            ximage blackhat(int kernel_size = 3) const;
            ximage skeletonize() const;
            ximage distance_transform() const;

            // --------------------------------------------------------------------
            // Edge detection
            // --------------------------------------------------------------------
            ximage canny(T low_thresh, T high_thresh, T sigma = T(1)) const;
            ximage edge_laplacian() const;
            ximage edge_zerocross() const;

            // --------------------------------------------------------------------
            // Histogram
            // --------------------------------------------------------------------
            std::vector<size_t> histogram(size_type bins = 256, T min_val = T(0), T max_val = T(255)) const;
            ximage equalize_histogram() const;
            ximage match_histogram(const ximage& reference) const;
            ximage clahe(T clip_limit = T(2.0), size_type tile_size = 8) const;

            // --------------------------------------------------------------------
            // Feature detection
            // --------------------------------------------------------------------
            std::vector<std::pair<size_t, size_t>> harris_corners(T k = T(0.04), T threshold = T(0.01)) const;
            std::vector<std::pair<size_t, size_t>> fast_corners(T threshold = T(20), bool nonmax_suppression = true) const;
            xarray_container<T> orb_descriptors(std::vector<std::pair<size_t, size_t>>& keypoints) const;

            // --------------------------------------------------------------------
            // Thresholding
            // --------------------------------------------------------------------
            ximage threshold(T thresh, T max_val = T(255)) const;
            ximage adaptive_threshold(T max_val, const std::string& method = "mean",
                                      int block_size = 11, T C = T(2)) const;
            ximage otsu_threshold() const;

            // --------------------------------------------------------------------
            // I/O
            // --------------------------------------------------------------------
            void load(const std::string& filename);
            void save(const std::string& filename, int quality = 80) const;
            void save_ppm(const std::string& filename) const;
            void save_bmp(const std::string& filename) const;
            void save_png(const std::string& filename) const;
            void save_jpeg(const std::string& filename, int quality = 80) const;
            void save_tga(const std::string& filename) const;
            void save_hdr(const std::string& filename) const;

            // --------------------------------------------------------------------
            // FFT‑based frequency‑domain processing
            // --------------------------------------------------------------------
            ximage fft_lowpass(T cutoff_ratio) const;
            ximage fft_highpass(T cutoff_ratio) const;
            ximage fft_bandpass(T low_cutoff, T high_cutoff) const;
            ximage fft_wiener_filter(const ximage& noise_power) const;
            ximage fft_deconvolve(const ximage& kernel) const;

            // --------------------------------------------------------------------
            // Drawing primitives
            // --------------------------------------------------------------------
            void draw_line(int x0, int y0, int x1, int y1,
                           const std::array<T, Channels>& color, int thickness = 1);
            void draw_rectangle(int x, int y, int w, int h,
                                const std::array<T, Channels>& color,
                                bool fill = false, int thickness = 1);
            void draw_circle(int cx, int cy, int radius,
                             const std::array<T, Channels>& color,
                             bool fill = false, int thickness = 1);
            void draw_ellipse(int cx, int cy, int rx, int ry,
                              const std::array<T, Channels>& color,
                              bool fill = false, int thickness = 1);
            void draw_polygon(const std::vector<std::pair<int, int>>& points,
                              const std::array<T, Channels>& color,
                              bool fill = false, int thickness = 1);
            void draw_text(const std::string& text, int x, int y,
                           const std::array<T, Channels>& color,
                           const std::string& font = "default", int font_size = 12);

        private:
            size_type m_width;
            size_type m_height;
            color_space m_color_space;
            xarray_container<T> m_data;
        };

        // ========================================================================
        // Convenience aliases
        // ========================================================================
        using image_gray = ximage<uint8_t, 1>;
        using image_rgb = ximage<uint8_t, 3>;
        using image_rgba = ximage<uint8_t, 4>;
        using image_gray_bn = ximage<bignumber::BigNumber, 1>;
        using image_rgb_bn = ximage<bignumber::BigNumber, 3>;

        // ------------------------------------------------------------------------
        // Factory functions
        // ------------------------------------------------------------------------
        template <class T = uint8_t, size_t C = 3>
        ximage<T, C> imread(const std::string& filename);
        template <class T = uint8_t, size_t C = 3>
        void imwrite(const std::string& filename, const ximage<T, C>& img, int quality = 80);

        // ------------------------------------------------------------------------
        // Drawing functions (free functions operating on image)
        // ------------------------------------------------------------------------
        template <class T, size_t C>
        void draw_line(ximage<T, C>& img, int x0, int y0, int x1, int y1,
                       const std::array<T, C>& color, int thickness = 1);
        template <class T, size_t C>
        void draw_rectangle(ximage<T, C>& img, int x, int y, int w, int h,
                            const std::array<T, C>& color, bool fill = false, int thickness = 1);
        template <class T, size_t C>
        void draw_circle(ximage<T, C>& img, int cx, int cy, int radius,
                         const std::array<T, C>& color, bool fill = false, int thickness = 1);
    }

    using image::ximage;
    using image::image_gray;
    using image::image_rgb;
    using image::image_rgba;
    using image::imread;
    using image::imwrite;
    using image::draw_line;
    using image::draw_rectangle;
    using image::draw_circle;
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace image
    {
        // Default constructor
        template <class T, size_t C> ximage<T, C>::ximage() : m_width(0), m_height(0), m_color_space(color_space::rgb) {}
        // Construct with dimensions
        template <class T, size_t C> ximage<T, C>::ximage(size_type width, size_type height, color_space cs)
            : m_width(width), m_height(height), m_color_space(cs), m_data({height, width, C}, T(0)) {}
        // Construct with fill value
        template <class T, size_t C> ximage<T, C>::ximage(size_type width, size_type height, const T& fill_value, color_space cs)
            : m_width(width), m_height(height), m_color_space(cs), m_data({height, width, C}, fill_value) {}
        // Construct from xtensor array
        template <class T, size_t C> ximage<T, C>::ximage(const xarray_container<T>& data, color_space cs)
            : m_color_space(cs), m_data(data) { m_height = data.shape()[0]; m_width = data.shape()[1]; }
        // Construct by loading from file
        template <class T, size_t C> ximage<T, C>::ximage(const std::string& filename) { load(filename); }

        // Get image width
        template <class T, size_t C> auto ximage<T, C>::width() const noexcept -> size_type { return m_width; }
        // Get image height
        template <class T, size_t C> auto ximage<T, C>::height() const noexcept -> size_type { return m_height; }
        // Get total number of pixels
        template <class T, size_t C> auto ximage<T, C>::size() const noexcept -> size_type { return m_width * m_height; }
        // Get shape as vector
        template <class T, size_t C> auto ximage<T, C>::shape() const -> shape_type { return {m_height, m_width, C}; }
        // Check if image is empty
        template <class T, size_t C> bool ximage<T, C>::empty() const noexcept { return m_width == 0 || m_height == 0; }
        // Get color space
        template <class T, size_t C> color_space ximage<T, C>::colorspace() const noexcept { return m_color_space; }
        // Set color space
        template <class T, size_t C> void ximage<T, C>::set_colorspace(color_space cs) noexcept { m_color_space = cs; }

        // Const access to underlying data
        template <class T, size_t C> const xarray_container<T>& ximage<T, C>::data() const noexcept { return m_data; }
        // Mutable access to underlying data
        template <class T, size_t C> xarray_container<T>& ximage<T, C>::data() noexcept { return m_data; }

        // Get const pointer to pixel
        template <class T, size_t C> const T* ximage<T, C>::pixel(size_type x, size_type y) const { return &m_data(y, x, 0); }
        // Get mutable pointer to pixel
        template <class T, size_t C> T* ximage<T, C>::pixel(size_type x, size_type y) { return &m_data(y, x, 0); }
        // Access pixel channel (mutable)
        template <class T, size_t C> T& ximage<T, C>::operator()(size_type x, size_type y, size_type c) { return m_data(y, x, c); }
        // Access pixel channel (const)
        template <class T, size_t C> const T& ximage<T, C>::operator()(size_type x, size_type y, size_type c) const { return m_data(y, x, c); }
        // Set pixel from array
        template <class T, size_t C> void ximage<T, C>::set_pixel(size_type x, size_type y, const std::array<T, C>& val)
        { for (size_type c = 0; c < C; ++c) m_data(y, x, c) = val[c]; }
        // Get pixel as array
        template <class T, size_t C> std::array<T, C> ximage<T, C>::get_pixel(size_type x, size_type y) const
        { std::array<T, C> val; for (size_type c = 0; c < C; ++c) val[c] = m_data(y, x, c); return val; }

        // Get mutable pointer to row
        template <class T, size_t C> T* ximage<T, C>::row(size_type y) { return &m_data(y, 0, 0); }
        // Get const pointer to row
        template <class T, size_t C> const T* ximage<T, C>::row(size_type y) const { return &m_data(y, 0, 0); }

        // Extract region of interest
        template <class T, size_t C> ximage<T, C> ximage<T, C>::roi(size_type x, size_type y, size_type w, size_type h) const
        { /* TODO: return sub‑image view or copy */ return {}; }
        // Paste another image into this one
        template <class T, size_t C> void ximage<T, C>::paste(const ximage& src, size_type dst_x, size_type dst_y)
        { /* TODO: copy pixel data */ }

        // Fill with scalar value
        template <class T, size_t C> void ximage<T, C>::fill(const T& value) { m_data.fill(value); }
        // Fill with pixel value
        template <class T, size_t C> void ximage<T, C>::fill(const std::array<T, C>& value)
        { for (size_type y = 0; y < m_height; ++y) for (size_type x = 0; x < m_width; ++x) set_pixel(x, y, value); }

        // Create a deep copy
        template <class T, size_t C> ximage<T, C> ximage<T, C>::clone() const { return ximage(m_data, m_color_space); }

        // Convert to grayscale (luminance)
        template <class T, size_t C> ximage<T, 1> ximage<T, C>::to_grayscale() const
        { /* TODO: luminance weighted sum */ return {}; }
        // Convert 1‑channel to RGB (replicate)
        template <class T, size_t C> template <size_t C2, typename std::enable_if<C2 == 1, int>::type>
        ximage<T, 3> ximage<T, C>::to_rgb() const { /* TODO: replicate channel */ return {}; }
        // Convert RGBA to RGB (drop alpha)
        template <class T, size_t C> template <size_t C2, typename std::enable_if<C2 == 4, int>::type>
        ximage<T, 3> ximage<T, C>::to_rgb() const { /* TODO: drop alpha */ return {}; }
        // Convert RGB to HSV
        template <class T, size_t C> ximage<T, 3> ximage<T, C>::to_hsv() const { /* TODO: RGB→HSV */ return {}; }
        // Convert HSV to RGB
        template <class T, size_t C> ximage<T, 3> ximage<T, C>::hsv_to_rgb() const { /* TODO: HSV→RGB */ return {}; }
        // Convert RGB to YUV
        template <class T, size_t C> ximage<T, 3> ximage<T, C>::to_yuv() const { /* TODO: RGB→YUV */ return {}; }
        // Convert YUV to RGB
        template <class T, size_t C> ximage<T, 3> ximage<T, C>::yuv_to_rgb() const { /* TODO: YUV→RGB */ return {}; }
        // Convert RGB to CIE L*a*b*
        template <class T, size_t C> ximage<T, 3> ximage<T, C>::to_lab() const { /* TODO: RGB→XYZ→LAB */ return {}; }
        // Convert LAB to RGB
        template <class T, size_t C> ximage<T, 3> ximage<T, C>::lab_to_rgb() const { /* TODO: LAB→XYZ→RGB */ return {}; }
        // Convert RGB to CMYK
        template <class T, size_t C> ximage<T, 4> ximage<T, C>::to_cmyk() const { /* TODO: RGB→CMYK */ return {}; }
        // Convert CMYK to RGB
        template <class T, size_t C> ximage<T, 4> ximage<T, C>::cmyk_to_rgb() const { /* TODO: CMYK→RGB */ return {}; }

        // Resize image
        template <class T, size_t C> ximage<T, C> ximage<T, C>::resize(size_type new_width, size_type new_height, const std::string& method) const
        { /* TODO: use stbir or custom interpolation */ return {}; }
        // Crop image
        template <class T, size_t C> ximage<T, C> ximage<T, C>::crop(size_type x, size_type y, size_type w, size_type h) const
        { /* TODO: extract subregion */ return {}; }
        // Rotate by multiples of 90°
        template <class T, size_t C> ximage<T, C> ximage<T, C>::rotate90(int times) const
        { /* TODO: transpose and flip */ return {}; }
        // Rotate by arbitrary angle
        template <class T, size_t C> ximage<T, C> ximage<T, C>::rotate(T angle_deg, const std::string& method) const
        { /* TODO: affine rotation with interpolation */ return {}; }
        // Flip horizontally/vertically
        template <class T, size_t C> ximage<T, C> ximage<T, C>::flip(bool horizontal, bool vertical) const
        { /* TODO: mirror pixels */ return {}; }
        // Pad image
        template <class T, size_t C> ximage<T, C> ximage<T, C>::pad(size_type top, size_type bottom, size_type left, size_type right, const std::string& mode, T constant) const
        { /* TODO: add border */ return {}; }
        // Affine transformation
        template <class T, size_t C> ximage<T, C> ximage<T, C>::affine_transform(const xarray_container<T>& matrix, const std::string& method) const
        { /* TODO: warp with 2x3 or 3x3 matrix */ return {}; }
        // Perspective transformation
        template <class T, size_t C> ximage<T, C> ximage<T, C>::perspective_transform(const xarray_container<T>& src_pts, const xarray_container<T>& dst_pts, const std::string& method) const
        { /* TODO: compute homography and warp */ return {}; }
        // Polar transform
        template <class T, size_t C> ximage<T, C> ximage<T, C>::polar_transform(bool inverse) const
        { /* TODO: Cartesian↔Polar mapping */ return {}; }
        // Log‑polar transform
        template <class T, size_t C> ximage<T, C> ximage<T, C>::log_polar_transform(bool inverse) const
        { /* TODO: log‑polar mapping */ return {}; }

        // Convolution with arbitrary kernel
        template <class T, size_t C> ximage<T, C> ximage<T, C>::convolve(const xarray_container<T>& kernel, const std::string& border) const
        { /* TODO: 2D convolution with border handling */ return {}; }
        // Gaussian blur
        template <class T, size_t C> ximage<T, C> ximage<T, C>::gaussian_blur(T sigma, int kernel_size) const
        { /* TODO: separable Gaussian filter */ return {}; }
        // Median filter
        template <class T, size_t C> ximage<T, C> ximage<T, C>::median_filter(int radius) const
        { /* TODO: sliding window median */ return {}; }
        // Bilateral filter
        template <class T, size_t C> ximage<T, C> ximage<T, C>::bilateral_filter(T sigma_spatial, T sigma_range, int radius) const
        { /* TODO: edge‑preserving smoothing */ return {}; }
        // Box filter (mean)
        template <class T, size_t C> ximage<T, C> ximage<T, C>::box_filter(int radius) const
        { /* TODO: uniform kernel convolution */ return {}; }
        // Sobel edge detector
        template <class T, size_t C> ximage<T, C> ximage<T, C>::sobel() const
        { /* TODO: Sobel magnitude */ return {}; }
        // Laplacian
        template <class T, size_t C> ximage<T, C> ximage<T, C>::laplacian() const
        { /* TODO: Laplacian kernel */ return {}; }
        // Prewitt
        template <class T, size_t C> ximage<T, C> ximage<T, C>::prewitt() const
        { /* TODO: Prewitt operator */ return {}; }
        // Roberts cross
        template <class T, size_t C> ximage<T, C> ximage<T, C>::roberts() const
        { /* TODO: Roberts operator */ return {}; }
        // Scharr
        template <class T, size_t C> ximage<T, C> ximage<T, C>::scharr() const
        { /* TODO: Scharr operator */ return {}; }
        // Unsharp mask
        template <class T, size_t C> ximage<T, C> ximage<T, C>::unsharp_mask(T sigma, T amount) const
        { /* TODO: original + amount * (original - blurred) */ return {}; }
        // Wiener filter
        template <class T, size_t C> ximage<T, C> ximage<T, C>::wiener_filter(const ximage& noise_power_spectrum) const
        { /* TODO: frequency‑domain Wiener deconvolution */ return {}; }

        // Erosion
        template <class T, size_t C> ximage<T, C> ximage<T, C>::erode(int kernel_size) const
        { /* TODO: morphological erosion */ return {}; }
        // Dilation
        template <class T, size_t C> ximage<T, C> ximage<T, C>::dilate(int kernel_size) const
        { /* TODO: morphological dilation */ return {}; }
        // Opening
        template <class T, size_t C> ximage<T, C> ximage<T, C>::opening(int kernel_size) const
        { return erode(kernel_size).dilate(kernel_size); }
        // Closing
        template <class T, size_t C> ximage<T, C> ximage<T, C>::closing(int kernel_size) const
        { return dilate(kernel_size).erode(kernel_size); }
        // Morphological gradient
        template <class T, size_t C> ximage<T, C> ximage<T, C>::morph_gradient(int kernel_size) const
        { /* TODO: dilate - erode */ return {}; }
        // Top‑hat
        template <class T, size_t C> ximage<T, C> ximage<T, C>::tophat(int kernel_size) const
        { /* TODO: original - opening */ return {}; }
        // Black‑hat
        template <class T, size_t C> ximage<T, C> ximage<T, C>::blackhat(int kernel_size) const
        { /* TODO: closing - original */ return {}; }
        // Skeletonize
        template <class T, size_t C> ximage<T, C> ximage<T, C>::skeletonize() const
        { /* TODO: morphological thinning */ return {}; }
        // Distance transform
        template <class T, size_t C> ximage<T, C> ximage<T, C>::distance_transform() const
        { /* TODO: Euclidean distance transform */ return {}; }

        // Canny edge detector
        template <class T, size_t C> ximage<T, C> ximage<T, C>::canny(T low_thresh, T high_thresh, T sigma) const
        { /* TODO: multi‑stage Canny */ return {}; }
        // Laplacian edge
        template <class T, size_t C> ximage<T, C> ximage<T, C>::edge_laplacian() const
        { return laplacian(); }
        // Zero‑crossing edge
        template <class T, size_t C> ximage<T, C> ximage<T, C>::edge_zerocross() const
        { /* TODO: find Laplacian zero‑crossings */ return {}; }

        // Histogram
        template <class T, size_t C> std::vector<size_t> ximage<T, C>::histogram(size_type bins, T min_val, T max_val) const
        { /* TODO: compute per‑channel histogram */ return {}; }
        // Histogram equalization
        template <class T, size_t C> ximage<T, C> ximage<T, C>::equalize_histogram() const
        { /* TODO: contrast enhancement */ return {}; }
        // Histogram matching
        template <class T, size_t C> ximage<T, C> ximage<T, C>::match_histogram(const ximage& reference) const
        { /* TODO: transform to match reference distribution */ return {}; }
        // CLAHE
        template <class T, size_t C> ximage<T, C> ximage<T, C>::clahe(T clip_limit, size_type tile_size) const
        { /* TODO: Contrast Limited Adaptive Histogram Equalization */ return {}; }

        // Harris corners
        template <class T, size_t C> std::vector<std::pair<size_t, size_t>> ximage<T, C>::harris_corners(T k, T threshold) const
        { /* TODO: Harris corner response */ return {}; }
        // FAST corners
        template <class T, size_t C> std::vector<std::pair<size_t, size_t>> ximage<T, C>::fast_corners(T threshold, bool nonmax_suppression) const
        { /* TODO: FAST feature detector */ return {}; }
        // ORB descriptors
        template <class T, size_t C> xarray_container<T> ximage<T, C>::orb_descriptors(std::vector<std::pair<size_t, size_t>>& keypoints) const
        { /* TODO: compute ORB binary descriptors */ return {}; }

        // Binary threshold
        template <class T, size_t C> ximage<T, C> ximage<T, C>::threshold(T thresh, T max_val) const
        { /* TODO: pixel‑wise threshold */ return {}; }
        // Adaptive threshold
        template <class T, size_t C> ximage<T, C> ximage<T, C>::adaptive_threshold(T max_val, const std::string& method, int block_size, T C) const
        { /* TODO: local threshold based on neighborhood */ return {}; }
        // Otsu's threshold
        template <class T, size_t C> ximage<T, C> ximage<T, C>::otsu_threshold() const
        { /* TODO: automatic global threshold */ return {}; }

        // Load from file (auto‑detect format)
        template <class T, size_t C> void ximage<T, C>::load(const std::string& filename)
        { /* TODO: use stbi_load or custom decoders */ }
        // Save to file (auto‑detect format by extension)
        template <class T, size_t C> void ximage<T, C>::save(const std::string& filename, int quality) const
        { /* TODO: dispatch to appropriate writer */ }
        // Save as PPM
        template <class T, size_t C> void ximage<T, C>::save_ppm(const std::string& filename) const
        { /* TODO: write P6 format */ }
        // Save as BMP
        template <class T, size_t C> void ximage<T, C>::save_bmp(const std::string& filename) const
        { /* TODO: write BMP header and pixel data */ }
        // Save as PNG
        template <class T, size_t C> void ximage<T, C>::save_png(const std::string& filename) const
        { /* TODO: use stbi_write_png */ }
        // Save as JPEG
        template <class T, size_t C> void ximage<T, C>::save_jpeg(const std::string& filename, int quality) const
        { /* TODO: use stbi_write_jpg */ }
        // Save as TGA
        template <class T, size_t C> void ximage<T, C>::save_tga(const std::string& filename) const
        { /* TODO: write TGA format */ }
        // Save as HDR
        template <class T, size_t C> void ximage<T, C>::save_hdr(const std::string& filename) const
        { /* TODO: write Radiance HDR format */ }

        // FFT low‑pass filter
        template <class T, size_t C> ximage<T, C> ximage<T, C>::fft_lowpass(T cutoff_ratio) const
        { /* TODO: FFT→mask→IFFT */ return {}; }
        // FFT high‑pass filter
        template <class T, size_t C> ximage<T, C> ximage<T, C>::fft_highpass(T cutoff_ratio) const
        { /* TODO: FFT→inverse mask→IFFT */ return {}; }
        // FFT band‑pass filter
        template <class T, size_t C> ximage<T, C> ximage<T, C>::fft_bandpass(T low_cutoff, T high_cutoff) const
        { /* TODO: FFT→bandpass mask→IFFT */ return {}; }
        // FFT Wiener filter
        template <class T, size_t C> ximage<T, C> ximage<T, C>::fft_wiener_filter(const ximage& noise_power) const
        { /* TODO: frequency‑domain Wiener deconvolution */ return {}; }
        // FFT deconvolution
        template <class T, size_t C> ximage<T, C> ximage<T, C>::fft_deconvolve(const ximage& kernel) const
        { /* TODO: divide in frequency domain */ return {}; }

        // Draw line (Bresenham)
        template <class T, size_t C> void ximage<T, C>::draw_line(int x0, int y0, int x1, int y1, const std::array<T, C>& color, int thickness)
        { /* TODO: Bresenham with thickness */ }
        // Draw rectangle
        template <class T, size_t C> void ximage<T, C>::draw_rectangle(int x, int y, int w, int h, const std::array<T, C>& color, bool fill, int thickness)
        { /* TODO: draw/fill rectangle */ }
        // Draw circle (Midpoint)
        template <class T, size_t C> void ximage<T, C>::draw_circle(int cx, int cy, int radius, const std::array<T, C>& color, bool fill, int thickness)
        { /* TODO: draw/fill circle */ }
        // Draw ellipse
        template <class T, size_t C> void ximage<T, C>::draw_ellipse(int cx, int cy, int rx, int ry, const std::array<T, C>& color, bool fill, int thickness)
        { /* TODO: draw/fill ellipse */ }
        // Draw polygon
        template <class T, size_t C> void ximage<T, C>::draw_polygon(const std::vector<std::pair<int, int>>& points, const std::array<T, C>& color, bool fill, int thickness)
        { /* TODO: draw/fill polygon */ }
        // Draw text (requires font)
        template <class T, size_t C> void ximage<T, C>::draw_text(const std::string& text, int x, int y, const std::array<T, C>& color, const std::string& font, int font_size)
        { /* TODO: render text using stb_truetype or easy_font */ }

        // Factory: imread
        template <class T, size_t C> ximage<T, C> imread(const std::string& filename)
        { return ximage<T, C>(filename); }
        // Factory: imwrite
        template <class T, size_t C> void imwrite(const std::string& filename, const ximage<T, C>& img, int quality)
        { img.save(filename, quality); }

        // Free function: draw_line
        template <class T, size_t C>
        void draw_line(ximage<T, C>& img, int x0, int y0, int x1, int y1, const std::array<T, C>& color, int thickness)
        { img.draw_line(x0, y0, x1, y1, color, thickness); }
        // Free function: draw_rectangle
        template <class T, size_t C>
        void draw_rectangle(ximage<T, C>& img, int x, int y, int w, int h, const std::array<T, C>& color, bool fill, int thickness)
        { img.draw_rectangle(x, y, w, h, color, fill, thickness); }
        // Free function: draw_circle
        template <class T, size_t C>
        void draw_circle(ximage<T, C>& img, int cx, int cy, int radius, const std::array<T, C>& color, bool fill, int thickness)
        { img.draw_circle(cx, cy, radius, color, fill, thickness); }
    }
}

#endif // XTENSOR_XIMAGE_HPP