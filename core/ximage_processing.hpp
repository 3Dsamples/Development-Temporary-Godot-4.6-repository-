// core/ximage_processing.hpp
#ifndef XTENSOR_XIMAGE_PROCESSING_HPP
#define XTENSOR_XIMAGE_PROCESSING_HPP

// ----------------------------------------------------------------------------
// ximage_processing.hpp – Image processing algorithms for xtensor
// ----------------------------------------------------------------------------
// This header provides a comprehensive suite of image processing operations:
//   - Color space conversions (RGB ↔ HSV, YUV, LAB, Grayscale)
//   - Filtering: Gaussian blur, median, bilateral, Sobel, Laplacian, etc.
//   - Morphological operations: erosion, dilation, opening, closing
//   - Edge detection: Canny, Prewitt, Roberts cross
//   - Geometric transformations: resize, rotate, affine, perspective warp
//   - Histogram operations: equalization, matching, back projection
//   - Feature detection: Harris corner, Hough line transform
//   - Image arithmetic and blending
//
// All operations support bignumber::BigNumber pixel types and use FFT
// acceleration for convolution-based filters where beneficial.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <tuple>
#include <array>
#include <queue>
#include <limits>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsorting.hpp"
#include "xstats.hpp"
#include "fft.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace image
    {
        // ========================================================================
        // Color space conversions (RGB assumed in [0,1] range)
        // ========================================================================
        // Convert RGB image to grayscale
        template <class T> xarray_container<T> rgb_to_grayscale(const xarray_container<T>& rgb);
        // Convert RGB to HSV
        template <class T> xarray_container<T> rgb_to_hsv(const xarray_container<T>& rgb);
        // Convert HSV to RGB
        template <class T> xarray_container<T> hsv_to_rgb(const xarray_container<T>& hsv);
        // Convert RGB to YUV
        template <class T> xarray_container<T> rgb_to_yuv(const xarray_container<T>& rgb);
        // Convert YUV to RGB
        template <class T> xarray_container<T> yuv_to_rgb(const xarray_container<T>& yuv);
        // Convert RGB to LAB (CIE L*a*b*)
        template <class T> xarray_container<T> rgb_to_lab(const xarray_container<T>& rgb);
        // Convert LAB to RGB
        template <class T> xarray_container<T> lab_to_rgb(const xarray_container<T>& lab);

        // ========================================================================
        // Filtering operations
        // ========================================================================
        // Gaussian blur with given sigma
        template <class T> xarray_container<T> gaussian_blur(const xarray_container<T>& image, T sigma, int kernel_size = 0);
        // Median filter with given radius
        template <class T> xarray_container<T> median_filter(const xarray_container<T>& image, int radius = 1);
        // Bilateral filter (edge‑preserving smoothing)
        template <class T> xarray_container<T> bilateral_filter(const xarray_container<T>& image, T sigma_spatial, T sigma_range, int radius = 0);
        // Sobel edge detector (returns magnitude)
        template <class T> xarray_container<T> sobel(const xarray_container<T>& image, bool compute_magnitude = true);
        // Laplacian edge detector
        template <class T> xarray_container<T> laplacian(const xarray_container<T>& image);
        // Unsharp mask sharpening
        template <class T> xarray_container<T> unsharp_mask(const xarray_container<T>& image, T sigma = T(1), T amount = T(1));
        // Prewitt edge detector
        template <class T> xarray_container<T> prewitt(const xarray_container<T>& image);
        // Roberts cross edge detector
        template <class T> xarray_container<T> roberts(const xarray_container<T>& image);
        // Custom 2D convolution with border handling
        template <class T> xarray_container<T> convolve2d(const xarray_container<T>& image, const xarray_container<T>& kernel, const std::string& border = "reflect");

        // ========================================================================
        // Morphological operations (binary and grayscale)
        // ========================================================================
        // Erosion
        template <class T> xarray_container<T> erode(const xarray_container<T>& image, int kernel_size = 3);
        // Dilation
        template <class T> xarray_container<T> dilate(const xarray_container<T>& image, int kernel_size = 3);
        // Opening (erode then dilate)
        template <class T> xarray_container<T> opening(const xarray_container<T>& image, int kernel_size = 3);
        // Closing (dilate then erode)
        template <class T> xarray_container<T> closing(const xarray_container<T>& image, int kernel_size = 3);
        // Morphological gradient (dilate - erode)
        template <class T> xarray_container<T> morph_gradient(const xarray_container<T>& image, int kernel_size = 3);
        // Top‑hat transform (image - opening)
        template <class T> xarray_container<T> tophat(const xarray_container<T>& image, int kernel_size = 3);
        // Black‑hat transform (closing - image)
        template <class T> xarray_container<T> blackhat(const xarray_container<T>& image, int kernel_size = 3);

        // ========================================================================
        // Edge detection
        // ========================================================================
        // Canny edge detector
        template <class T> xarray_container<T> canny(const xarray_container<T>& image, T low_thresh, T high_thresh, T sigma = T(1));

        // ========================================================================
        // Geometric transformations
        // ========================================================================
        // Resize image to new dimensions
        template <class T> xarray_container<T> resize(const xarray_container<T>& image, size_t new_h, size_t new_w, const std::string& interpolation = "bilinear");
        // Rotate image by given angle (degrees)
        template <class T> xarray_container<T> rotate(const xarray_container<T>& image, T angle_deg, const std::string& interpolation = "bilinear");
        // Affine transformation
        template <class T> xarray_container<T> affine_transform(const xarray_container<T>& image, const xarray_container<T>& matrix, const std::string& interpolation = "bilinear");
        // Perspective warp
        template <class T> xarray_container<T> perspective_warp(const xarray_container<T>& image, const xarray_container<T>& src_pts, const xarray_container<T>& dst_pts, const std::string& interpolation = "bilinear");
        // Flip horizontally / vertically
        template <class T> xarray_container<T> flip(const xarray_container<T>& image, bool horizontal, bool vertical);
        // Crop region of interest
        template <class T> xarray_container<T> crop(const xarray_container<T>& image, size_t x, size_t y, size_t w, size_t h);

        // ========================================================================
        // Histogram operations
        // ========================================================================
        // Compute histogram
        template <class T> std::pair<std::vector<T>, std::vector<size_t>> histogram(const xarray_container<T>& image, size_t bins = 256, T min_val = T(0), T max_val = T(255));
        // Histogram equalization
        template <class T> xarray_container<T> equalize_histogram(const xarray_container<T>& image);
        // Histogram matching
        template <class T> xarray_container<T> match_histogram(const xarray_container<T>& src, const xarray_container<T>& ref);
        // Back projection
        template <class T> xarray_container<T> back_project(const xarray_container<T>& image, const std::vector<size_t>& hist, T min_val = T(0), T max_val = T(255));

        // ========================================================================
        // Feature detection
        // ========================================================================
        // Harris corner detector
        template <class T> std::vector<std::pair<size_t, size_t>> harris_corners(const xarray_container<T>& image, T k = T(0.04), T threshold_ratio = T(0.01));
        // Hough line transform
        template <class T> std::vector<std::tuple<T, T>> hough_lines(const xarray_container<T>& edges, T angle_step = T(1), T rho_step = T(1));
        // Hough circle transform
        template <class T> std::vector<std::tuple<size_t, size_t, size_t>> hough_circles(const xarray_container<T>& edges, size_t min_radius, size_t max_radius, T threshold = T(0.5));

        // ========================================================================
        // Image arithmetic and blending
        // ========================================================================
        // Add two images (with optional scaling)
        template <class T> xarray_container<T> add_weighted(const xarray_container<T>& img1, T alpha, const xarray_container<T>& img2, T beta, T gamma = T(0));
        // Linear blending (alpha * img1 + (1-alpha) * img2)
        template <class T> xarray_container<T> blend(const xarray_container<T>& img1, const xarray_container<T>& img2, T alpha);
        // Subtract two images
        template <class T> xarray_container<T> subtract(const xarray_container<T>& img1, const xarray_container<T>& img2);
        // Absolute difference
        template <class T> xarray_container<T> absdiff(const xarray_container<T>& img1, const xarray_container<T>& img2);
    }

    // Bring image processing functions into xt namespace
    using image::rgb_to_grayscale;
    using image::rgb_to_hsv;
    using image::hsv_to_rgb;
    using image::rgb_to_yuv;
    using image::yuv_to_rgb;
    using image::rgb_to_lab;
    using image::lab_to_rgb;
    using image::gaussian_blur;
    using image::median_filter;
    using image::bilateral_filter;
    using image::sobel;
    using image::laplacian;
    using image::unsharp_mask;
    using image::prewitt;
    using image::roberts;
    using image::convolve2d;
    using image::erode;
    using image::dilate;
    using image::opening;
    using image::closing;
    using image::morph_gradient;
    using image::tophat;
    using image::blackhat;
    using image::canny;
    using image::resize;
    using image::rotate;
    using image::affine_transform;
    using image::perspective_warp;
    using image::flip;
    using image::crop;
    using image::histogram;
    using image::equalize_histogram;
    using image::match_histogram;
    using image::back_project;
    using image::harris_corners;
    using image::hough_lines;
    using image::hough_circles;
    using image::add_weighted;
    using image::blend;
    using image::subtract;
    using image::absdiff;
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace image
    {
        // Convert RGB to grayscale using luminance weights
        template <class T> xarray_container<T> rgb_to_grayscale(const xarray_container<T>& rgb)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Convert RGB to HSV color space
        template <class T> xarray_container<T> rgb_to_hsv(const xarray_container<T>& rgb)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Convert HSV to RGB color space
        template <class T> xarray_container<T> hsv_to_rgb(const xarray_container<T>& hsv)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Convert RGB to YUV color space
        template <class T> xarray_container<T> rgb_to_yuv(const xarray_container<T>& rgb)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Convert YUV to RGB color space
        template <class T> xarray_container<T> yuv_to_rgb(const xarray_container<T>& yuv)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Convert RGB to CIE L*a*b* color space
        template <class T> xarray_container<T> rgb_to_lab(const xarray_container<T>& rgb)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Convert LAB to RGB color space
        template <class T> xarray_container<T> lab_to_rgb(const xarray_container<T>& lab)
        { /* TODO: implement */ return xarray_container<T>(); }

        // Apply Gaussian blur with given sigma
        template <class T> xarray_container<T> gaussian_blur(const xarray_container<T>& image, T sigma, int kernel_size)
        { /* TODO: implement separable Gaussian filter */ return image; }
        // Apply median filter with given radius
        template <class T> xarray_container<T> median_filter(const xarray_container<T>& image, int radius)
        { /* TODO: implement sliding window median */ return image; }
        // Apply bilateral filter (edge‑preserving)
        template <class T> xarray_container<T> bilateral_filter(const xarray_container<T>& image, T sigma_spatial, T sigma_range, int radius)
        { /* TODO: implement bilateral filtering */ return image; }
        // Sobel edge detector returning magnitude
        template <class T> xarray_container<T> sobel(const xarray_container<T>& image, bool compute_magnitude)
        { /* TODO: implement Sobel operator */ return image; }
        // Laplacian edge detector
        template <class T> xarray_container<T> laplacian(const xarray_container<T>& image)
        { /* TODO: implement Laplacian kernel convolution */ return image; }
        // Unsharp mask sharpening
        template <class T> xarray_container<T> unsharp_mask(const xarray_container<T>& image, T sigma, T amount)
        { /* TODO: implement unsharp masking */ return image; }
        // Prewitt edge detector
        template <class T> xarray_container<T> prewitt(const xarray_container<T>& image)
        { /* TODO: implement Prewitt operator */ return image; }
        // Roberts cross edge detector
        template <class T> xarray_container<T> roberts(const xarray_container<T>& image)
        { /* TODO: implement Roberts cross operator */ return image; }
        // Custom 2D convolution with border handling
        template <class T> xarray_container<T> convolve2d(const xarray_container<T>& image, const xarray_container<T>& kernel, const std::string& border)
        { /* TODO: implement generic 2D convolution */ return image; }

        // Erosion morphological operation
        template <class T> xarray_container<T> erode(const xarray_container<T>& image, int kernel_size)
        { /* TODO: implement erosion */ return image; }
        // Dilation morphological operation
        template <class T> xarray_container<T> dilate(const xarray_container<T>& image, int kernel_size)
        { /* TODO: implement dilation */ return image; }
        // Opening morphological operation
        template <class T> xarray_container<T> opening(const xarray_container<T>& image, int kernel_size)
        { return dilate(erode(image, kernel_size), kernel_size); }
        // Closing morphological operation
        template <class T> xarray_container<T> closing(const xarray_container<T>& image, int kernel_size)
        { return erode(dilate(image, kernel_size), kernel_size); }
        // Morphological gradient
        template <class T> xarray_container<T> morph_gradient(const xarray_container<T>& image, int kernel_size)
        { /* TODO: implement */ return image; }
        // Top‑hat transform
        template <class T> xarray_container<T> tophat(const xarray_container<T>& image, int kernel_size)
        { /* TODO: implement */ return image; }
        // Black‑hat transform
        template <class T> xarray_container<T> blackhat(const xarray_container<T>& image, int kernel_size)
        { /* TODO: implement */ return image; }

        // Canny edge detector
        template <class T> xarray_container<T> canny(const xarray_container<T>& image, T low_thresh, T high_thresh, T sigma)
        { /* TODO: implement multi‑stage Canny */ return image; }

        // Resize image to new dimensions
        template <class T> xarray_container<T> resize(const xarray_container<T>& image, size_t new_h, size_t new_w, const std::string& interpolation)
        { /* TODO: implement image resizing */ return xarray_container<T>(); }
        // Rotate image by given angle in degrees
        template <class T> xarray_container<T> rotate(const xarray_container<T>& image, T angle_deg, const std::string& interpolation)
        { /* TODO: implement image rotation */ return image; }
        // Apply affine transformation matrix
        template <class T> xarray_container<T> affine_transform(const xarray_container<T>& image, const xarray_container<T>& matrix, const std::string& interpolation)
        { /* TODO: implement affine warping */ return image; }
        // Apply perspective warp
        template <class T> xarray_container<T> perspective_warp(const xarray_container<T>& image, const xarray_container<T>& src_pts, const xarray_container<T>& dst_pts, const std::string& interpolation)
        { /* TODO: implement perspective transformation */ return image; }
        // Flip horizontally and/or vertically
        template <class T> xarray_container<T> flip(const xarray_container<T>& image, bool horizontal, bool vertical)
        { /* TODO: implement flipping */ return image; }
        // Crop region of interest
        template <class T> xarray_container<T> crop(const xarray_container<T>& image, size_t x, size_t y, size_t w, size_t h)
        { /* TODO: implement cropping */ return xarray_container<T>(); }

        // Compute histogram of image
        template <class T> std::pair<std::vector<T>, std::vector<size_t>> histogram(const xarray_container<T>& image, size_t bins, T min_val, T max_val)
        { /* TODO: implement histogram */ return {}; }
        // Equalize image histogram
        template <class T> xarray_container<T> equalize_histogram(const xarray_container<T>& image)
        { /* TODO: implement histogram equalization */ return image; }
        // Match histogram of src to reference
        template <class T> xarray_container<T> match_histogram(const xarray_container<T>& src, const xarray_container<T>& ref)
        { /* TODO: implement histogram matching */ return src; }
        // Back project histogram onto image
        template <class T> xarray_container<T> back_project(const xarray_container<T>& image, const std::vector<size_t>& hist, T min_val, T max_val)
        { /* TODO: implement back projection */ return image; }

        // Detect Harris corners
        template <class T> std::vector<std::pair<size_t, size_t>> harris_corners(const xarray_container<T>& image, T k, T threshold_ratio)
        { /* TODO: implement Harris corner detector */ return {}; }
        // Detect lines using Hough transform
        template <class T> std::vector<std::tuple<T, T>> hough_lines(const xarray_container<T>& edges, T angle_step, T rho_step)
        { /* TODO: implement Hough line transform */ return {}; }
        // Detect circles using Hough transform
        template <class T> std::vector<std::tuple<size_t, size_t, size_t>> hough_circles(const xarray_container<T>& edges, size_t min_radius, size_t max_radius, T threshold)
        { /* TODO: implement Hough circle transform */ return {}; }

        // Add two images with weights and optional gamma
        template <class T> xarray_container<T> add_weighted(const xarray_container<T>& img1, T alpha, const xarray_container<T>& img2, T beta, T gamma)
        { /* TODO: implement weighted addition */ return img1; }
        // Linear blend of two images
        template <class T> xarray_container<T> blend(const xarray_container<T>& img1, const xarray_container<T>& img2, T alpha)
        { return add_weighted(img1, alpha, img2, T(1)-alpha, T(0)); }
        // Subtract two images
        template <class T> xarray_container<T> subtract(const xarray_container<T>& img1, const xarray_container<T>& img2)
        { /* TODO: implement subtraction */ return img1; }
        // Absolute difference between two images
        template <class T> xarray_container<T> absdiff(const xarray_container<T>& img1, const xarray_container<T>& img2)
        { /* TODO: implement absolute difference */ return img1; }
    }
}

#endif // XTENSOR_XIMAGE_PROCESSING_HPP image, int kernel_size = 3)
        {
            return erode(dilate(image, kernel_size), kernel_size);
        }

        // ========================================================================
        // Edge detection
        // ========================================================================

        template <class T>
        xarray_container<T> canny(const xarray_container<T>& image, T low_thresh, T high_thresh, T sigma = T(1))
        {
            auto blurred = gaussian_blur(image, sigma);
            auto Gx_kernel = xarray_container<T>({3,3}, {T(-1), T(0), T(1), T(-2), T(0), T(2), T(-1), T(0), T(1)});
            auto Gy_kernel = xarray_container<T>({3,3}, {T(-1), T(-2), T(-1), T(0), T(0), T(0), T(1), T(2), T(1)});
            auto Gx = detail::convolve2d_generic(blurred, Gx_kernel, "reflect");
            auto Gy = detail::convolve2d_generic(blurred, Gy_kernel, "reflect");
            size_t h = image.shape()[0], w = image.shape()[1];
            xarray_container<T> magnitude({h, w});
            xarray_container<T> direction({h, w});
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    magnitude(y, x) = detail::sqrt_val(Gx(y,x)*Gx(y,x) + Gy(y,x)*Gy(y,x));
                    T angle = std::atan2(Gy(y,x), Gx(y,x)) * T(180) / detail::pi<T>();
                    direction(y, x) = angle < T(0) ? angle + T(180) : angle;
                }
            }
            // Non‑maximum suppression
            xarray_container<T> nms({h, w}, T(0));
            for (size_t y = 1; y < h-1; ++y)
            {
                for (size_t x = 1; x < w-1; ++x)
                {
                    T ang = direction(y, x);
                    T mag = magnitude(y, x);
                    T q = T(255), r = T(255);
                    if ((ang >= T(0) && ang < T(22.5)) || (ang >= T(157.5) && ang <= T(180)))
                    {
                        q = magnitude(y, x+1); r = magnitude(y, x-1);
                    }
                    else if (ang >= T(22.5) && ang < T(67.5))
                    {
                        q = magnitude(y+1, x+1); r = magnitude(y-1, x-1);
                    }
                    else if (ang >= T(67.5) && ang < T(112.5))
                    {
                        q = magnitude(y+1, x); r = magnitude(y-1, x);
                    }
                    else if (ang >= T(112.5) && ang < T(157.5))
                    {
                        q = magnitude(y-1, x+1); r = magnitude(y+1, x-1);
                    }
                    if (mag >= q && mag >= r)
                        nms(y, x) = mag;
                }
            }
            // Hysteresis thresholding
            xarray_container<T> edges({h, w}, T(0));
            std::queue<std::pair<size_t,size_t>> strong_edges;
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    if (nms(y, x) >= high_thresh)
                    {
                        edges(y, x) = T(255);
                        strong_edges.push({y, x});
                    }
                }
            }
            while (!strong_edges.empty())
            {
                auto [y, x] = strong_edges.front(); strong_edges.pop();
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int ny = (int)y + dy, nx = (int)x + dx;
                        if (ny < 0 || ny >= (int)h || nx < 0 || nx >= (int)w) continue;
                        if (nms(ny, nx) >= low_thresh && edges(ny, nx) == T(0))
                        {
                            edges(ny, nx) = T(255);
                            strong_edges.push({ny, nx});
                        }
                    }
                }
            }
            return edges;
        }

        // ========================================================================
        // Geometric transformations
        // ========================================================================

        template <class T>
        xarray_container<T> resize(const xarray_container<T>& image, size_t new_h, size_t new_w,
                                   const std::string& interpolation = "bilinear")
        {
            size_t h = image.shape()[0], w = image.shape()[1];
            size_t channels = (image.dimension() == 3) ? image.shape()[2] : 1;
            xarray_container<T> result;
            if (channels == 1)
                result = xarray_container<T>({new_h, new_w});
            else
                result = xarray_container<T>({new_h, new_w, channels});

            T scale_y = T(h) / T(new_h);
            T scale_x = T(w) / T(new_w);
            for (size_t y = 0; y < new_h; ++y)
            {
                T src_y = (T(y) + T(0.5)) * scale_y - T(0.5);
                size_t y0 = (size_t)std::floor(src_y);
                size_t y1 = std::min(y0 + 1, h - 1);
                T wy = src_y - T(y0);
                for (size_t x = 0; x < new_w; ++x)
                {
                    T src_x = (T(x) + T(0.5)) * scale_x - T(0.5);
                    size_t x0 = (size_t)std::floor(src_x);
                    size_t x1 = std::min(x0 + 1, w - 1);
                    T wx = src_x - T(x0);
                    if (channels == 1)
                    {
                        T v00 = image(y0, x0);
                        T v01 = image(y0, x1);
                        T v10 = image(y1, x0);
                        T v11 = image(y1, x1);
                        T v0 = v00 * (T(1) - wx) + v01 * wx;
                        T v1 = v10 * (T(1) - wx) + v11 * wx;
                        result(y, x) = v0 * (T(1) - wy) + v1 * wy;
                    }
                    else
                    {
                        for (size_t c = 0; c < channels; ++c)
                        {
                            T v00 = image(y0, x0, c);
                            T v01 = image(y0, x1, c);
                            T v10 = image(y1, x0, c);
                            T v11 = image(y1, x1, c);
                            T v0 = v00 * (T(1) - wx) + v01 * wx;
                            T v1 = v10 * (T(1) - wx) + v11 * wx;
                            result(y, x, c) = v0 * (T(1) - wy) + v1 * wy;
                        }
                    }
                }
            }
            return result;
        }

        template <class T>
        xarray_container<T> rotate(const xarray_container<T>& image, T angle_deg,
                                   const std::string& interpolation = "bilinear")
        {
            size_t h = image.shape()[0], w = image.shape()[1];
            T angle_rad = angle_deg * detail::pi<T>() / T(180);
            T cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);
            // Compute new bounding box
            T corners_x[4] = {0, T(w-1), T(w-1), 0};
            T corners_y[4] = {0, 0, T(h-1), T(h-1)};
            T min_x = std::numeric_limits<T>::max(), max_x = std::numeric_limits<T>::lowest();
            T min_y = std::numeric_limits<T>::max(), max_y = std::numeric_limits<T>::lowest();
            for (int i = 0; i < 4; ++i)
            {
                T x = corners_x[i], y = corners_y[i];
                T nx = cos_a * x - sin_a * y;
                T ny = sin_a * x + cos_a * y;
                min_x = detail::min_val(min_x, nx); max_x = detail::max_val(max_x, nx);
                min_y = detail::min_val(min_y, ny); max_y = detail::max_val(max_y, ny);
            }
            size_t new_w = (size_t)std::ceil(max_x - min_x);
            size_t new_h = (size_t)std::ceil(max_y - min_y);
            xarray_container<T> result;
            size_t channels = (image.dimension() == 3) ? image.shape()[2] : 1;
            if (channels == 1)
                result = xarray_container<T>({new_h, new_w}, T(0));
            else
                result = xarray_container<T>({new_h, new_w, channels}, T(0));
            T cx = T(w-1)/T(2), cy = T(h-1)/T(2);
            T ncx = T(new_w-1)/T(2), ncy = T(new_h-1)/T(2);
            for (size_t y = 0; y < new_h; ++y)
            {
                for (size_t x = 0; x < new_w; ++x)
                {
                    T nx = T(x) - ncx, ny = T(y) - ncy;
                    T src_x =  cos_a * nx + sin_a * ny + cx;
                    T src_y = -sin_a * nx + cos_a * ny + cy;
                    if (src_x >= 0 && src_x < T(w-1) && src_y >= 0 && src_y < T(h-1))
                    {
                        size_t x0 = (size_t)std::floor(src_x);
                        size_t y0 = (size_t)std::floor(src_y);
                        size_t x1 = std::min(x0+1, w-1);
                        size_t y1 = std::min(y0+1, h-1);
                        T wx = src_x - T(x0);
                        T wy = src_y - T(y0);
                        if (channels == 1)
                        {
                            T v00 = image(y0, x0), v01 = image(y0, x1);
                            T v10 = image(y1, x0), v11 = image(y1, x1);
                            T v0 = v00*(T(1)-wx) + v01*wx;
                            T v1 = v10*(T(1)-wx) + v11*wx;
                            result(y, x) = v0*(T(1)-wy) + v1*wy;
                        }
                        else
                        {
                            for (size_t c = 0; c < channels; ++c)
                            {
                                T v00 = image(y0, x0, c), v01 = image(y0, x1, c);
                                T v10 = image(y1, x0, c), v11 = image(y1, x1, c);
                                T v0 = v00*(T(1)-wx) + v01*wx;
                                T v1 = v10*(T(1)-wx) + v11*wx;
                                result(y, x, c) = v0*(T(1)-wy) + v1*wy;
                            }
                        }
                    }
                }
            }
            return result;
        }

        // ========================================================================
        // Histogram operations
        // ========================================================================

        template <class T>
        std::pair<std::vector<T>, std::vector<size_t>> histogram(const xarray_container<T>& image,
                                                                  size_t bins = 256, T min_val = T(0), T max_val = T(255))
        {
            std::vector<size_t> hist(bins, 0);
            T bin_width = (max_val - min_val) / T(bins);
            for (size_t i = 0; i < image.size(); ++i)
            {
                T v = image.flat(i);
                if (v < min_val || v > max_val) continue;
                size_t bin = (size_t)((v - min_val) / bin_width);
                if (bin >= bins) bin = bins - 1;
                ++hist[bin];
            }
            std::vector<T> bin_edges(bins + 1);
            for (size_t i = 0; i <= bins; ++i)
                bin_edges[i] = min_val + T(i) * bin_width;
            return {bin_edges, hist};
        }

        template <class T>
        xarray_container<T> equalize_histogram(const xarray_container<T>& image)
        {
            auto [edges, hist] = histogram(image, 256, T(0), T(255));
            std::vector<T> cdf(256);
            T sum = 0;
            for (size_t i = 0; i < 256; ++i)
            {
                sum += T(hist[i]);
                cdf[i] = sum / T(image.size());
            }
            size_t h = image.shape()[0], w = image.shape()[1];
            xarray_container<T> result({h, w});
            T min_cdf = cdf[0];
            T denom = T(255) / (T(1) - min_cdf);
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    T v = image(y, x);
                    size_t bin = (size_t)detail::clamp(v, T(0), T(255));
                    result(y, x) = (cdf[bin] - min_cdf) * denom;
                }
            }
            return result;
        }

        // ========================================================================
        // Feature detection
        // ========================================================================

        template <class T>
        std::vector<std::pair<size_t, size_t>> harris_corners(const xarray_container<T>& image,
                                                               T k = T(0.04), T threshold_ratio = T(0.01))
        {
            auto Ix_kernel = xarray_container<T>({3,3}, {T(-1), T(0), T(1), T(-2), T(0), T(2), T(-1), T(0), T(1)});
            auto Iy_kernel = xarray_container<T>({3,3}, {T(-1), T(-2), T(-1), T(0), T(0), T(0), T(1), T(2), T(1)});
            auto Ix = detail::convolve2d_generic(image, Ix_kernel, "reflect");
            auto Iy = detail::convolve2d_generic(image, Iy_kernel, "reflect");
            size_t h = image.shape()[0], w = image.shape()[1];
            xarray_container<T> R({h, w});
            T max_R = std::numeric_limits<T>::lowest();
            for (size_t y = 1; y < h-1; ++y)
            {
                for (size_t x = 1; x < w-1; ++x)
                {
                    T a = 0, b = 0, c = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            T ix = Ix(y+dy, x+dx), iy = Iy(y+dy, x+dx);
                            a += ix * ix;
                            b += ix * iy;
                            c += iy * iy;
                        }
                    }
                    T det = a*c - b*b;
                    T trace = a + c;
                    R(y, x) = det - k * trace * trace;
                    if (R(y, x) > max_R) max_R = R(y, x);
                }
            }
            T threshold = max_R * threshold_ratio;
            std::vector<std::pair<size_t, size_t>> corners;
            for (size_t y = 1; y < h-1; ++y)
            {
                for (size_t x = 1; x < w-1; ++x)
                {
                    if (R(y, x) > threshold)
                    {
                        bool is_max = true;
                        for (int dy = -1; dy <= 1 && is_max; ++dy)
                            for (int dx = -1; dx <= 1; ++dx)
                                if (R(y+dy, x+dx) > R(y, x))
                                    { is_max = false; break; }
                        if (is_max) corners.emplace_back(y, x);
                    }
                }
            }
            return corners;
        }

        // ========================================================================
        // Hough line transform
        // ========================================================================

        template <class T>
        std::vector<std::tuple<T, T>> hough_lines(const xarray_container<T>& edges,
                                                   T angle_step = T(1), T rho_step = T(1))
        {
            size_t h = edges.shape()[0], w = edges.shape()[1];
            T max_rho = detail::sqrt_val(T(h*h + w*w));
            int num_angles = (int)(T(180) / angle_step);
            int num_rhos = (int)(T(2) * max_rho / rho_step);
            xarray_container<size_t> accumulator({(size_t)num_rhos, (size_t)num_angles}, size_t(0));
            T theta_rad, rho;
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    if (edges(y, x) > T(0))
                    {
                        for (int t = 0; t < num_angles; ++t)
                        {
                            theta_rad = T(t) * angle_step * detail::pi<T>() / T(180);
                            rho = T(x) * std::cos(theta_rad) + T(y) * std::sin(theta_rad);
                            int rho_idx = (int)((rho + max_rho) / rho_step);
                            if (rho_idx >= 0 && rho_idx < num_rhos)
                                ++accumulator(rho_idx, t);
                        }
                    }
                }
            }
            size_t max_votes = 0;
            for (auto v : accumulator) max_votes = std::max(max_votes, v);
            size_t threshold = max_votes / 2;
            std::vector<std::tuple<T, T>> lines;
            for (int r = 0; r < num_rhos; ++r)
            {
                for (int t = 0; t < num_angles; ++t)
                {
                    if (accumulator(r, t) > threshold)
                    {
                        T rho_val = T(r) * rho_step - max_rho;
                        T theta_val = T(t) * angle_step;
                        lines.emplace_back(rho_val, theta_val);
                    }
                }
            }
            return lines;
        }

    } // namespace image

    using image::rgb_to_grayscale;
    using image::rgb_to_hsv;
    using image::hsv_to_rgb;
    using image::gaussian_blur;
    using image::median_filter;
    using image::bilateral_filter;
    using image::sobel;
    using image::laplacian;
    using image::unsharp_mask;
    using image::erode;
    using image::dilate;
    using image::opening;
    using image::closing;
    using image::canny;
    using image::resize;
    using image::rotate;
    using image::histogram;
    using image::equalize_histogram;
    using image::harris_corners;
    using image::hough_lines;

} // namespace xt

#endif // XTENSOR_XIMAGE_PROCESSING_HPP