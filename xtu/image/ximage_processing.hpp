// image/ximage_processing.hpp

#ifndef XTENSOR_XIMAGE_PROCESSING_HPP
#define XTENSOR_XIMAGE_PROCESSING_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xsorting.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xmaterial.hpp"
#include "../signal/xwindows.hpp"
#include "../signal/lfilter.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <complex>
#include <map>
#include <queue>
#include <set>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace image
        {
            using Image = xarray_container<double>; // typically H x W x C, or H x W for grayscale
            using ImageU8 = xarray_container<uint8_t>;

            // --------------------------------------------------------------------
            // Color space conversions
            // --------------------------------------------------------------------
            namespace color
            {
                // RGB to Grayscale (luminance)
                template <class E>
                inline auto rgb2gray(const xexpression<E>& img)
                {
                    const auto& src = img.derived_cast();
                    if (src.dimension() != 3 || src.shape()[2] != 3)
                        XTENSOR_THROW(std::invalid_argument, "rgb2gray: expected HxWx3 image");
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    xarray_container<double> gray({h, w});
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            double r = src(y, x, 0);
                            double g = src(y, x, 1);
                            double b = src(y, x, 2);
                            gray(y, x) = 0.2989 * r + 0.5870 * g + 0.1140 * b;
                        }
                    }
                    return gray;
                }

                // RGB to HSV
                template <class E>
                inline auto rgb2hsv(const xexpression<E>& img)
                {
                    const auto& src = img.derived_cast();
                    if (src.dimension() != 3 || src.shape()[2] != 3)
                        XTENSOR_THROW(std::invalid_argument, "rgb2hsv: expected HxWx3 image");
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    xarray_container<double> hsv({h, w, 3});
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            double r = src(y, x, 0);
                            double g = src(y, x, 1);
                            double b = src(y, x, 2);
                            double maxc = std::max({r, g, b});
                            double minc = std::min({r, g, b});
                            double delta = maxc - minc;
                            double hue = 0.0;
                            if (delta > 1e-8)
                            {
                                if (maxc == r)
                                    hue = 60.0 * std::fmod((g - b) / delta, 6.0);
                                else if (maxc == g)
                                    hue = 60.0 * ((b - r) / delta + 2.0);
                                else
                                    hue = 60.0 * ((r - g) / delta + 4.0);
                                if (hue < 0) hue += 360.0;
                            }
                            double sat = (maxc < 1e-8) ? 0.0 : delta / maxc;
                            double val = maxc;
                            hsv(y, x, 0) = hue / 360.0;
                            hsv(y, x, 1) = sat;
                            hsv(y, x, 2) = val;
                        }
                    }
                    return hsv;
                }

                // HSV to RGB
                template <class E>
                inline auto hsv2rgb(const xexpression<E>& img)
                {
                    const auto& src = img.derived_cast();
                    if (src.dimension() != 3 || src.shape()[2] != 3)
                        XTENSOR_THROW(std::invalid_argument, "hsv2rgb: expected HxWx3 image");
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    xarray_container<double> rgb({h, w, 3});
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            double hue = src(y, x, 0) * 360.0;
                            double sat = src(y, x, 1);
                            double val = src(y, x, 2);
                            double c = val * sat;
                            double hp = hue / 60.0;
                            double xc = c * (1.0 - std::abs(std::fmod(hp, 2.0) - 1.0));
                            double r, g, b;
                            if (hp < 1)      { r = c; g = xc; b = 0; }
                            else if (hp < 2) { r = xc; g = c; b = 0; }
                            else if (hp < 3) { r = 0; g = c; b = xc; }
                            else if (hp < 4) { r = 0; g = xc; b = c; }
                            else if (hp < 5) { r = xc; g = 0; b = c; }
                            else             { r = c; g = 0; b = xc; }
                            double m = val - c;
                            rgb(y, x, 0) = r + m;
                            rgb(y, x, 1) = g + m;
                            rgb(y, x, 2) = b + m;
                        }
                    }
                    return rgb;
                }

                // RGB to LAB
                template <class E>
                inline auto rgb2lab(const xexpression<E>& img)
                {
                    const auto& src = img.derived_cast();
                    if (src.dimension() != 3 || src.shape()[2] != 3)
                        XTENSOR_THROW(std::invalid_argument, "rgb2lab: expected HxWx3 image");
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    xarray_container<double> lab({h, w, 3});
                    // RGB to XYZ matrix (sRGB D65)
                    const double mat[3][3] = {
                        {0.4124564, 0.3575761, 0.1804375},
                        {0.2126729, 0.7151522, 0.0721750},
                        {0.0193339, 0.1191920, 0.9503041}
                    };
                    auto f = [](double t) -> double {
                        if (t > 0.008856) return std::cbrt(t);
                        return (7.787 * t) + (16.0 / 116.0);
                    };
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            double r = src(y, x, 0);
                            double g = src(y, x, 1);
                            double b = src(y, x, 2);
                            // linearize
                            auto linear = [](double c) {
                                if (c <= 0.04045) return c / 12.92;
                                return std::pow((c + 0.055) / 1.055, 2.4);
                            };
                            double rl = linear(r);
                            double gl = linear(g);
                            double bl = linear(b);
                            double X = mat[0][0]*rl + mat[0][1]*gl + mat[0][2]*bl;
                            double Y = mat[1][0]*rl + mat[1][1]*gl + mat[1][2]*bl;
                            double Z = mat[2][0]*rl + mat[2][1]*gl + mat[2][2]*bl;
                            // normalize by white point D65
                            X /= 0.95047;
                            Z /= 1.08883;
                            double fx = f(X);
                            double fy = f(Y);
                            double fz = f(Z);
                            double L = 116.0 * fy - 16.0;
                            double A = 500.0 * (fx - fy);
                            double B = 200.0 * (fy - fz);
                            lab(y, x, 0) = L / 100.0;
                            lab(y, x, 1) = (A + 128.0) / 255.0;
                            lab(y, x, 2) = (B + 128.0) / 255.0;
                        }
                    }
                    return lab;
                }
            } // namespace color

            // --------------------------------------------------------------------
            // Filtering operations (convolution-based)
            // --------------------------------------------------------------------
            namespace filters
            {
                // 2D convolution with padding
                template <class E, class K>
                inline auto convolve2d(const xexpression<E>& img, const xexpression<K>& kernel,
                                       const std::string& mode = "same", const std::string& boundary = "reflect")
                {
                    const auto& src = img.derived_cast();
                    const auto& k = kernel.derived_cast();
                    if (src.dimension() == 2)
                    {
                        size_t h = src.shape()[0];
                        size_t w = src.shape()[1];
                        size_t kh = k.shape()[0];
                        size_t kw = k.shape()[1];
                        int pad_h = static_cast<int>(kh / 2);
                        int pad_w = static_cast<int>(kw / 2);
                        xarray_container<double> result({h, w}, 0.0);
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                double sum = 0.0;
                                for (size_t ky = 0; ky < kh; ++ky)
                                {
                                    for (size_t kx = 0; kx < kw; ++kx)
                                    {
                                        int sy = static_cast<int>(y) + static_cast<int>(ky) - pad_h;
                                        int sx = static_cast<int>(x) + static_cast<int>(kx) - pad_w;
                                        double val = 0.0;
                                        if (boundary == "reflect")
                                        {
                                            if (sy < 0) sy = -sy - 1;
                                            if (sy >= static_cast<int>(h)) sy = 2 * static_cast<int>(h) - sy - 1;
                                            if (sx < 0) sx = -sx - 1;
                                            if (sx >= static_cast<int>(w)) sx = 2 * static_cast<int>(w) - sx - 1;
                                            sy = std::clamp(sy, 0, static_cast<int>(h) - 1);
                                            sx = std::clamp(sx, 0, static_cast<int>(w) - 1);
                                            val = src(sy, sx);
                                        }
                                        else if (boundary == "constant")
                                        {
                                            if (sy >= 0 && sy < static_cast<int>(h) && sx >= 0 && sx < static_cast<int>(w))
                                                val = src(sy, sx);
                                        }
                                        else // nearest
                                        {
                                            sy = std::clamp(sy, 0, static_cast<int>(h) - 1);
                                            sx = std::clamp(sx, 0, static_cast<int>(w) - 1);
                                            val = src(sy, sx);
                                        }
                                        sum += val * k(ky, kx);
                                    }
                                }
                                result(y, x) = sum;
                            }
                        }
                        return result;
                    }
                    else if (src.dimension() == 3)
                    {
                        size_t h = src.shape()[0];
                        size_t w = src.shape()[1];
                        size_t c = src.shape()[2];
                        size_t kh = k.shape()[0];
                        size_t kw = k.shape()[1];
                        int pad_h = static_cast<int>(kh / 2);
                        int pad_w = static_cast<int>(kw / 2);
                        xarray_container<double> result({h, w, c}, 0.0);
                        for (size_t ch = 0; ch < c; ++ch)
                        {
                            for (size_t y = 0; y < h; ++y)
                            {
                                for (size_t x = 0; x < w; ++x)
                                {
                                    double sum = 0.0;
                                    for (size_t ky = 0; ky < kh; ++ky)
                                    {
                                        for (size_t kx = 0; kx < kw; ++kx)
                                        {
                                            int sy = static_cast<int>(y) + static_cast<int>(ky) - pad_h;
                                            int sx = static_cast<int>(x) + static_cast<int>(kx) - pad_w;
                                            sy = std::clamp(sy, 0, static_cast<int>(h) - 1);
                                            sx = std::clamp(sx, 0, static_cast<int>(w) - 1);
                                            sum += src(sy, sx, ch) * k(ky, kx);
                                        }
                                    }
                                    result(y, x, ch) = sum;
                                }
                            }
                        }
                        return result;
                    }
                    XTENSOR_THROW(std::invalid_argument, "convolve2d: input must be 2D or 3D");
                    return xarray_container<double>();
                }

                // Gaussian blur
                template <class E>
                inline auto gaussian_blur(const xexpression<E>& img, double sigma, int kernel_size = 0)
                {
                    if (kernel_size == 0)
                        kernel_size = static_cast<int>(std::ceil(6.0 * sigma)) | 1; // odd
                    int half = kernel_size / 2;
                    xarray_container<double> kernel({static_cast<size_t>(kernel_size), static_cast<size_t>(kernel_size)});
                    double sum = 0.0;
                    for (int y = -half; y <= half; ++y)
                    {
                        for (int x = -half; x <= half; ++x)
                        {
                            double val = std::exp(-(x*x + y*y) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
                            kernel(y+half, x+half) = val;
                            sum += val;
                        }
                    }
                    kernel = kernel / sum;
                    return convolve2d(img, kernel, "same", "reflect");
                }

                // Median filter
                template <class E>
                inline auto median_filter(const xexpression<E>& img, int kernel_size = 3)
                {
                    const auto& src = img.derived_cast();
                    if (kernel_size % 2 == 0) kernel_size++;
                    int half = kernel_size / 2;
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    if (src.dimension() == 2)
                    {
                        xarray_container<double> result({h, w});
                        std::vector<double> window(kernel_size * kernel_size);
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                size_t cnt = 0;
                                for (int dy = -half; dy <= half; ++dy)
                                {
                                    for (int dx = -half; dx <= half; ++dx)
                                    {
                                        int sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(h) - 1);
                                        int sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(w) - 1);
                                        window[cnt++] = src(sy, sx);
                                    }
                                }
                                std::sort(window.begin(), window.begin() + cnt);
                                result(y, x) = window[cnt / 2];
                            }
                        }
                        return result;
                    }
                    else if (src.dimension() == 3)
                    {
                        size_t c = src.shape()[2];
                        xarray_container<double> result({h, w, c});
                        std::vector<double> window(kernel_size * kernel_size);
                        for (size_t ch = 0; ch < c; ++ch)
                        {
                            for (size_t y = 0; y < h; ++y)
                            {
                                for (size_t x = 0; x < w; ++x)
                                {
                                    size_t cnt = 0;
                                    for (int dy = -half; dy <= half; ++dy)
                                    {
                                        for (int dx = -half; dx <= half; ++dx)
                                        {
                                            int sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(h) - 1);
                                            int sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(w) - 1);
                                            window[cnt++] = src(sy, sx, ch);
                                        }
                                    }
                                    std::sort(window.begin(), window.begin() + cnt);
                                    result(y, x, ch) = window[cnt / 2];
                                }
                            }
                        }
                        return result;
                    }
                    XTENSOR_THROW(std::invalid_argument, "median_filter: input must be 2D or 3D");
                    return xarray_container<double>();
                }

                // Bilateral filter
                template <class E>
                inline auto bilateral_filter(const xexpression<E>& img, double sigma_space, double sigma_color, int kernel_size = 0)
                {
                    const auto& src = img.derived_cast();
                    if (kernel_size == 0)
                        kernel_size = static_cast<int>(std::ceil(3.0 * sigma_space)) | 1;
                    int half = kernel_size / 2;
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    if (src.dimension() == 2)
                    {
                        xarray_container<double> result({h, w});
                        double inv_sigma_color2 = 1.0 / (2.0 * sigma_color * sigma_color);
                        double inv_sigma_space2 = 1.0 / (2.0 * sigma_space * sigma_space);
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                double center_val = src(y, x);
                                double sum_weights = 0.0;
                                double sum_values = 0.0;
                                for (int dy = -half; dy <= half; ++dy)
                                {
                                    for (int dx = -half; dx <= half; ++dx)
                                    {
                                        int sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(h) - 1);
                                        int sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(w) - 1);
                                        double spatial_dist2 = dy*dy + dx*dx;
                                        double color_diff = src(sy, sx) - center_val;
                                        double weight = std::exp(-spatial_dist2 * inv_sigma_space2) *
                                                        std::exp(-color_diff * color_diff * inv_sigma_color2);
                                        sum_weights += weight;
                                        sum_values += weight * src(sy, sx);
                                    }
                                }
                                result(y, x) = sum_values / sum_weights;
                            }
                        }
                        return result;
                    }
                    else if (src.dimension() == 3)
                    {
                        size_t c = src.shape()[2];
                        xarray_container<double> result({h, w, c});
                        double inv_sigma_space2 = 1.0 / (2.0 * sigma_space * sigma_space);
                        double inv_sigma_color2 = 1.0 / (2.0 * sigma_color * sigma_color);
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                for (size_t ch = 0; ch < c; ++ch)
                                {
                                    double center_val = src(y, x, ch);
                                    double sum_weights = 0.0;
                                    double sum_values = 0.0;
                                    for (int dy = -half; dy <= half; ++dy)
                                    {
                                        for (int dx = -half; dx <= half; ++dx)
                                        {
                                            int sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(h) - 1);
                                            int sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(w) - 1);
                                            double spatial_dist2 = dy*dy + dx*dx;
                                            double color_diff = src(sy, sx, ch) - center_val;
                                            double weight = std::exp(-spatial_dist2 * inv_sigma_space2) *
                                                            std::exp(-color_diff * color_diff * inv_sigma_color2);
                                            sum_weights += weight;
                                            sum_values += weight * src(sy, sx, ch);
                                        }
                                    }
                                    result(y, x, ch) = sum_values / sum_weights;
                                }
                            }
                        }
                        return result;
                    }
                    XTENSOR_THROW(std::invalid_argument, "bilateral_filter: input must be 2D or 3D");
                    return xarray_container<double>();
                }

                // Sobel edge detection
                template <class E>
                inline auto sobel(const xexpression<E>& img)
                {
                    const auto& src = img.derived_cast();
                    xarray_container<double> kernel_x = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
                    xarray_container<double> kernel_y = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
                    if (src.dimension() == 2)
                    {
                        auto gx = convolve2d(src, kernel_x, "same", "constant");
                        auto gy = convolve2d(src, kernel_y, "same", "constant");
                        size_t h = src.shape()[0];
                        size_t w = src.shape()[1];
                        xarray_container<double> magnitude({h, w});
                        xarray_container<double> direction({h, w});
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                double gx_val = gx(y, x);
                                double gy_val = gy(y, x);
                                magnitude(y, x) = std::sqrt(gx_val*gx_val + gy_val*gy_val);
                                direction(y, x) = std::atan2(gy_val, gx_val);
                            }
                        }
                        return std::make_pair(magnitude, direction);
                    }
                    else
                    {
                        // Convert to grayscale first
                        auto gray = color::rgb2gray(src);
                        return sobel(gray);
                    }
                }

                // Laplacian of Gaussian (LoG)
                template <class E>
                inline auto laplacian_of_gaussian(const xexpression<E>& img, double sigma)
                {
                    int size = static_cast<int>(std::ceil(6.0 * sigma)) | 1;
                    int half = size / 2;
                    xarray_container<double> kernel({static_cast<size_t>(size), static_cast<size_t>(size)});
                    double sigma2 = sigma * sigma;
                    double sigma4 = sigma2 * sigma2;
                    for (int y = -half; y <= half; ++y)
                    {
                        for (int x = -half; x <= half; ++x)
                        {
                            double r2 = x*x + y*y;
                            kernel(y+half, x+half) = -1.0 / (M_PI * sigma4) * (1.0 - r2 / (2.0 * sigma2)) * std::exp(-r2 / (2.0 * sigma2));
                        }
                    }
                    return convolve2d(img, kernel, "same", "reflect");
                }

                // Canny edge detector
                template <class E>
                inline auto canny(const xexpression<E>& img, double low_thresh, double high_thresh, double sigma = 1.4)
                {
                    auto gray = (img.derived_cast().dimension() == 3) ? color::rgb2gray(img) : eval(img);
                    auto blurred = gaussian_blur(gray, sigma);
                    auto [mag, dir] = sobel(blurred);
                    size_t h = mag.shape()[0];
                    size_t w = mag.shape()[1];
                    // Non-maximum suppression
                    xarray_container<double> nms({h, w}, 0.0);
                    for (size_t y = 1; y < h-1; ++y)
                    {
                        for (size_t x = 1; x < w-1; ++x)
                        {
                            double angle = dir(y, x) * 180.0 / M_PI;
                            angle = (angle < 0) ? angle + 180.0 : angle;
                            double m = mag(y, x);
                            double n1 = 0.0, n2 = 0.0;
                            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
                            {
                                n1 = mag(y, x-1);
                                n2 = mag(y, x+1);
                            }
                            else if (angle >= 22.5 && angle < 67.5)
                            {
                                n1 = mag(y-1, x+1);
                                n2 = mag(y+1, x-1);
                            }
                            else if (angle >= 67.5 && angle < 112.5)
                            {
                                n1 = mag(y-1, x);
                                n2 = mag(y+1, x);
                            }
                            else
                            {
                                n1 = mag(y-1, x-1);
                                n2 = mag(y+1, x+1);
                            }
                            if (m >= n1 && m >= n2)
                                nms(y, x) = m;
                        }
                    }
                    // Hysteresis thresholding
                    xarray_container<uint8_t> edges({h, w}, 0);
                    std::queue<std::pair<size_t, size_t>> q;
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            if (nms(y, x) >= high_thresh)
                            {
                                edges(y, x) = 255;
                                q.push({y, x});
                            }
                        }
                    }
                    while (!q.empty())
                    {
                        auto [y, x] = q.front(); q.pop();
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                if (dy == 0 && dx == 0) continue;
                                int ny = static_cast<int>(y) + dy;
                                int nx = static_cast<int>(x) + dx;
                                if (ny >= 0 && ny < static_cast<int>(h) && nx >= 0 && nx < static_cast<int>(w))
                                {
                                    if (edges(ny, nx) == 0 && nms(ny, nx) >= low_thresh)
                                    {
                                        edges(ny, nx) = 255;
                                        q.push({static_cast<size_t>(ny), static_cast<size_t>(nx)});
                                    }
                                }
                            }
                        }
                    }
                    return edges;
                }

                // Prewitt operator
                inline auto prewitt_kx() { return xarray_container<double>({{-1,0,1},{-1,0,1},{-1,0,1}}); }
                inline auto prewitt_ky() { return xarray_container<double>({{-1,-1,-1},{0,0,0},{1,1,1}}); }

                // Scharr operator (better rotational symmetry)
                inline auto scharr_kx() { return xarray_container<double>({{-3,0,3},{-10,0,10},{-3,0,3}}); }
                inline auto scharr_ky() { return xarray_container<double>({{-3,-10,-3},{0,0,0},{3,10,3}}); }

                // Unsharp mask (sharpening)
                template <class E>
                inline auto unsharp_mask(const xexpression<E>& img, double sigma = 1.0, double amount = 1.0)
                {
                    auto blurred = gaussian_blur(img, sigma);
                    const auto& src = img.derived_cast();
                    return src + amount * (src - blurred);
                }
            } // namespace filters

            // --------------------------------------------------------------------
            // Morphological operations (binary and grayscale)
            // --------------------------------------------------------------------
            namespace morphology
            {
                // Structuring element: disk of radius r
                inline xarray_container<uint8_t> disk(int radius)
                {
                    int size = 2 * radius + 1;
                    xarray_container<uint8_t> se({static_cast<size_t>(size), static_cast<size_t>(size)});
                    int r2 = radius * radius;
                    for (int y = -radius; y <= radius; ++y)
                        for (int x = -radius; x <= radius; ++x)
                            se(y+radius, x+radius) = (x*x + y*y <= r2) ? 1 : 0;
                    return se;
                }

                inline xarray_container<uint8_t> rectangle(int h, int w)
                {
                    return xt::ones<uint8_t>({static_cast<size_t>(h), static_cast<size_t>(w)});
                }

                inline xarray_container<uint8_t> cross(int size)
                {
                    xarray_container<uint8_t> se = xt::zeros<uint8_t>({static_cast<size_t>(size), static_cast<size_t>(size)});
                    int mid = size / 2;
                    for (int i = 0; i < size; ++i)
                    {
                        se(mid, i) = 1;
                        se(i, mid) = 1;
                    }
                    return se;
                }

                // Erosion (binary)
                template <class E, class SE>
                inline auto erode(const xexpression<E>& img, const xexpression<SE>& se)
                {
                    const auto& src = img.derived_cast();
                    const auto& strel = se.derived_cast();
                    if (src.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "erode: binary image must be 2D");
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    int se_h = static_cast<int>(strel.shape()[0]);
                    int se_w = static_cast<int>(strel.shape()[1]);
                    int pad_h = se_h / 2;
                    int pad_w = se_w / 2;
                    xarray_container<uint8_t> result({h, w}, 0);
                    for (size_t y = pad_h; y < h - pad_h; ++y)
                    {
                        for (size_t x = pad_w; x < w - pad_w; ++x)
                        {
                            bool fit = true;
                            for (int ky = 0; ky < se_h && fit; ++ky)
                            {
                                for (int kx = 0; kx < se_w; ++kx)
                                {
                                    if (strel(ky, kx) && !src(y + ky - pad_h, x + kx - pad_w))
                                    {
                                        fit = false;
                                        break;
                                    }
                                }
                            }
                            result(y, x) = fit ? 255 : 0;
                        }
                    }
                    return result;
                }

                // Dilation (binary)
                template <class E, class SE>
                inline auto dilate(const xexpression<E>& img, const xexpression<SE>& se)
                {
                    const auto& src = img.derived_cast();
                    const auto& strel = se.derived_cast();
                    if (src.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "dilate: binary image must be 2D");
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    int se_h = static_cast<int>(strel.shape()[0]);
                    int se_w = static_cast<int>(strel.shape()[1]);
                    int pad_h = se_h / 2;
                    int pad_w = se_w / 2;
                    xarray_container<uint8_t> result({h, w}, 0);
                    for (size_t y = pad_h; y < h - pad_h; ++y)
                    {
                        for (size_t x = pad_w; x < w - pad_w; ++x)
                        {
                            if (src(y, x))
                            {
                                for (int ky = 0; ky < se_h; ++ky)
                                {
                                    for (int kx = 0; kx < se_w; ++kx)
                                    {
                                        if (strel(ky, kx))
                                        {
                                            size_t ty = y + ky - pad_h;
                                            size_t tx = x + kx - pad_w;
                                            if (ty < h && tx < w)
                                                result(ty, tx) = 255;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return result;
                }

                // Opening = erode then dilate
                template <class E, class SE>
                inline auto opening(const xexpression<E>& img, const xexpression<SE>& se)
                {
                    return dilate(erode(img, se), se);
                }

                // Closing = dilate then erode
                template <class E, class SE>
                inline auto closing(const xexpression<E>& img, const xexpression<SE>& se)
                {
                    return erode(dilate(img, se), se);
                }

                // Morphological gradient
                template <class E, class SE>
                inline auto morph_gradient(const xexpression<E>& img, const xexpression<SE>& se)
                {
                    auto d = dilate(img, se);
                    auto e = erode(img, se);
                    return d - e;
                }

                // Top-hat (white top-hat)
                template <class E, class SE>
                inline auto tophat(const xexpression<E>& img, const xexpression<SE>& se)
                {
                    return img - opening(img, se);
                }

                // Black top-hat
                template <class E, class SE>
                inline auto blackhat(const xexpression<E>& img, const xexpression<SE>& se)
                {
                    return closing(img, se) - img;
                }

                // Skeletonization (medial axis)
                template <class E>
                inline auto skeletonize(const xexpression<E>& img)
                {
                    auto bin = (img.derived_cast().dimension() == 2) ? eval(img) : color::rgb2gray(img);
                    // Threshold to binary
                    for (auto& v : bin) v = v > 0.5 ? 1 : 0;
                    xarray_container<uint8_t> skel = xt::zeros<uint8_t>(bin.shape());
                    auto se = cross(3);
                    xarray_container<uint8_t> temp = bin;
                    while (true)
                    {
                        auto eroded = erode(temp, se);
                        auto opened = opening(eroded, se);
                        auto diff = eroded - opened;
                        skel = skel | diff;
                        temp = eroded;
                        if (xt::sum(temp)() == 0) break;
                    }
                    return skel * 255;
                }
            } // namespace morphology

            // --------------------------------------------------------------------
            // Geometric transformations
            // --------------------------------------------------------------------
            namespace transform
            {
                // Resize image using bilinear interpolation
                template <class E>
                inline auto resize(const xexpression<E>& img, size_t new_h, size_t new_w)
                {
                    const auto& src = img.derived_cast();
                    size_t src_h = src.shape()[0];
                    size_t src_w = src.shape()[1];
                    double scale_y = static_cast<double>(src_h) / new_h;
                    double scale_x = static_cast<double>(src_w) / new_w;
                    if (src.dimension() == 2)
                    {
                        xarray_container<double> result({new_h, new_w});
                        for (size_t y = 0; y < new_h; ++y)
                        {
                            for (size_t x = 0; x < new_w; ++x)
                            {
                                double src_y = (y + 0.5) * scale_y - 0.5;
                                double src_x = (x + 0.5) * scale_x - 0.5;
                                int y0 = static_cast<int>(std::floor(src_y));
                                int x0 = static_cast<int>(std::floor(src_x));
                                int y1 = std::min(y0 + 1, static_cast<int>(src_h) - 1);
                                int x1 = std::min(x0 + 1, static_cast<int>(src_w) - 1);
                                y0 = std::max(0, y0);
                                x0 = std::max(0, x0);
                                double wy = src_y - y0;
                                double wx = src_x - x0;
                                double v00 = src(y0, x0);
                                double v01 = src(y0, x1);
                                double v10 = src(y1, x0);
                                double v11 = src(y1, x1);
                                double v0 = v00 * (1 - wx) + v01 * wx;
                                double v1 = v10 * (1 - wx) + v11 * wx;
                                result(y, x) = v0 * (1 - wy) + v1 * wy;
                            }
                        }
                        return result;
                    }
                    else if (src.dimension() == 3)
                    {
                        size_t c = src.shape()[2];
                        xarray_container<double> result({new_h, new_w, c});
                        for (size_t ch = 0; ch < c; ++ch)
                        {
                            for (size_t y = 0; y < new_h; ++y)
                            {
                                for (size_t x = 0; x < new_w; ++x)
                                {
                                    double src_y = (y + 0.5) * scale_y - 0.5;
                                    double src_x = (x + 0.5) * scale_x - 0.5;
                                    int y0 = std::max(0, static_cast<int>(std::floor(src_y)));
                                    int x0 = std::max(0, static_cast<int>(std::floor(src_x)));
                                    int y1 = std::min(y0 + 1, static_cast<int>(src_h) - 1);
                                    int x1 = std::min(x0 + 1, static_cast<int>(src_w) - 1);
                                    double wy = src_y - y0;
                                    double wx = src_x - x0;
                                    double v00 = src(y0, x0, ch);
                                    double v01 = src(y0, x1, ch);
                                    double v10 = src(y1, x0, ch);
                                    double v11 = src(y1, x1, ch);
                                    double v0 = v00 * (1 - wx) + v01 * wx;
                                    double v1 = v10 * (1 - wx) + v11 * wx;
                                    result(y, x, ch) = v0 * (1 - wy) + v1 * wy;
                                }
                            }
                        }
                        return result;
                    }
                    XTENSOR_THROW(std::invalid_argument, "resize: input must be 2D or 3D");
                    return xarray_container<double>();
                }

                // Rotate image by angle (degrees) around center
                template <class E>
                inline auto rotate(const xexpression<E>& img, double angle_deg)
                {
                    const auto& src = img.derived_cast();
                    double angle = angle_deg * M_PI / 180.0;
                    double c = std::cos(angle);
                    double s = std::sin(angle);
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    double cx = w / 2.0;
                    double cy = h / 2.0;
                    // Compute new bounds
                    double corners[4][2] = {{0,0}, {w-1.0,0}, {0,h-1.0}, {w-1.0,h-1.0}};
                    double min_x = std::numeric_limits<double>::max();
                    double max_x = std::numeric_limits<double>::lowest();
                    double min_y = min_x, max_y = max_x;
                    for (auto& corner : corners)
                    {
                        double x = corner[0] - cx;
                        double y = corner[1] - cy;
                        double rx = c * x - s * y + cx;
                        double ry = s * x + c * y + cy;
                        min_x = std::min(min_x, rx);
                        max_x = std::max(max_x, rx);
                        min_y = std::min(min_y, ry);
                        max_y = std::max(max_y, ry);
                    }
                    size_t new_w = static_cast<size_t>(std::ceil(max_x - min_x));
                    size_t new_h = static_cast<size_t>(std::ceil(max_y - min_y));
                    double offset_x = min_x;
                    double offset_y = min_y;
                    if (src.dimension() == 2)
                    {
                        xarray_container<double> result({new_h, new_w}, 0.0);
                        for (size_t y = 0; y < new_h; ++y)
                        {
                            for (size_t x = 0; x < new_w; ++x)
                            {
                                double rx = x + offset_x - cx;
                                double ry = y + offset_y - cy;
                                double src_x = c * rx + s * ry + cx;
                                double src_y = -s * rx + c * ry + cy;
                                if (src_x >= 0 && src_x < w-1 && src_y >= 0 && src_y < h-1)
                                {
                                    int x0 = static_cast<int>(std::floor(src_x));
                                    int y0 = static_cast<int>(std::floor(src_y));
                                    int x1 = x0 + 1;
                                    int y1 = y0 + 1;
                                    double wx = src_x - x0;
                                    double wy = src_y - y0;
                                    double v00 = src(y0, x0);
                                    double v01 = src(y0, x1);
                                    double v10 = src(y1, x0);
                                    double v11 = src(y1, x1);
                                    double v0 = v00 * (1 - wx) + v01 * wx;
                                    double v1 = v10 * (1 - wx) + v11 * wx;
                                    result(y, x) = v0 * (1 - wy) + v1 * wy;
                                }
                            }
                        }
                        return result;
                    }
                    else if (src.dimension() == 3)
                    {
                        size_t c = src.shape()[2];
                        xarray_container<double> result({new_h, new_w, c}, 0.0);
                        for (size_t ch = 0; ch < c; ++ch)
                        {
                            for (size_t y = 0; y < new_h; ++y)
                            {
                                for (size_t x = 0; x < new_w; ++x)
                                {
                                    double rx = x + offset_x - cx;
                                    double ry = y + offset_y - cy;
                                    double src_x = c * rx + s * ry + cx;
                                    double src_y = -s * rx + c * ry + cy;
                                    if (src_x >= 0 && src_x < w-1 && src_y >= 0 && src_y < h-1)
                                    {
                                        int x0 = static_cast<int>(std::floor(src_x));
                                        int y0 = static_cast<int>(std::floor(src_y));
                                        int x1 = x0 + 1;
                                        int y1 = y0 + 1;
                                        double wx = src_x - x0;
                                        double wy = src_y - y0;
                                        double v00 = src(y0, x0, ch);
                                        double v01 = src(y0, x1, ch);
                                        double v10 = src(y1, x0, ch);
                                        double v11 = src(y1, x1, ch);
                                        double v0 = v00 * (1 - wx) + v01 * wx;
                                        double v1 = v10 * (1 - wx) + v11 * wx;
                                        result(y, x, ch) = v0 * (1 - wy) + v1 * wy;
                                    }
                                }
                            }
                        }
                        return result;
                    }
                    XTENSOR_THROW(std::invalid_argument, "rotate: input must be 2D or 3D");
                    return xarray_container<double>();
                }

                // Flip horizontally/vertically
                template <class E>
                inline auto flip_horizontal(const xexpression<E>& img)
                {
                    auto result = eval(img);
                    size_t w = result.shape()[1];
                    if (result.dimension() == 2)
                    {
                        for (size_t y = 0; y < result.shape()[0]; ++y)
                            for (size_t x = 0; x < w/2; ++x)
                                std::swap(result(y, x), result(y, w-1-x));
                    }
                    else if (result.dimension() == 3)
                    {
                        for (size_t y = 0; y < result.shape()[0]; ++y)
                            for (size_t x = 0; x < w/2; ++x)
                                for (size_t c = 0; c < result.shape()[2]; ++c)
                                    std::swap(result(y, x, c), result(y, w-1-x, c));
                    }
                    return result;
                }

                template <class E>
                inline auto flip_vertical(const xexpression<E>& img)
                {
                    auto result = eval(img);
                    size_t h = result.shape()[0];
                    if (result.dimension() == 2)
                    {
                        for (size_t y = 0; y < h/2; ++y)
                            for (size_t x = 0; x < result.shape()[1]; ++x)
                                std::swap(result(y, x), result(h-1-y, x));
                    }
                    else if (result.dimension() == 3)
                    {
                        for (size_t y = 0; y < h/2; ++y)
                            for (size_t x = 0; x < result.shape()[1]; ++x)
                                for (size_t c = 0; c < result.shape()[2]; ++c)
                                    std::swap(result(y, x, c), result(h-1-y, x, c));
                    }
                    return result;
                }

                // Affine transformation (using 2x3 matrix)
                template <class E>
                inline auto warp_affine(const xexpression<E>& img, const std::array<double, 6>& M, size_t out_h, size_t out_w)
                {
                    const auto& src = img.derived_cast();
                    double a = M[0], b = M[1], tx = M[2];
                    double c = M[3], d = M[4], ty = M[5];
                    double det = a * d - b * c;
                    if (std::abs(det) < 1e-8)
                        XTENSOR_THROW(std::invalid_argument, "warp_affine: singular matrix");
                    double inv_a = d / det, inv_b = -b / det;
                    double inv_c = -c / det, inv_d = a / det;
                    double inv_tx = (b * ty - d * tx) / det;
                    double inv_ty = (c * tx - a * ty) / det;
                    size_t h = src.shape()[0];
                    size_t w = src.shape()[1];
                    if (src.dimension() == 2)
                    {
                        xarray_container<double> result({out_h, out_w}, 0.0);
                        for (size_t y = 0; y < out_h; ++y)
                        {
                            for (size_t x = 0; x < out_w; ++x)
                            {
                                double src_x = inv_a * x + inv_b * y + inv_tx;
                                double src_y = inv_c * x + inv_d * y + inv_ty;
                                if (src_x >= 0 && src_x < w-1 && src_y >= 0 && src_y < h-1)
                                {
                                    int x0 = static_cast<int>(std::floor(src_x));
                                    int y0 = static_cast<int>(std::floor(src_y));
                                    int x1 = x0 + 1;
                                    int y1 = y0 + 1;
                                    double wx = src_x - x0;
                                    double wy = src_y - y0;
                                    double v00 = src(y0, x0);
                                    double v01 = src(y0, x1);
                                    double v10 = src(y1, x0);
                                    double v11 = src(y1, x1);
                                    double v0 = v00 * (1 - wx) + v01 * wx;
                                    double v1 = v10 * (1 - wx) + v11 * wx;
                                    result(y, x) = v0 * (1 - wy) + v1 * wy;
                                }
                            }
                        }
                        return result;
                    }
                    else
                    {
                        // For multi-channel, process per channel
                        size_t c = src.shape()[2];
                        xarray_container<double> result({out_h, out_w, c}, 0.0);
                        for (size_t ch = 0; ch < c; ++ch)
                        {
                            for (size_t y = 0; y < out_h; ++y)
                            {
                                for (size_t x = 0; x < out_w; ++x)
                                {
                                    double src_x = inv_a * x + inv_b * y + inv_tx;
                                    double src_y = inv_c * x + inv_d * y + inv_ty;
                                    if (src_x >= 0 && src_x < w-1 && src_y >= 0 && src_y < h-1)
                                    {
                                        int x0 = static_cast<int>(std::floor(src_x));
                                        int y0 = static_cast<int>(std::floor(src_y));
                                        int x1 = x0 + 1;
                                        int y1 = y0 + 1;
                                        double wx = src_x - x0;
                                        double wy = src_y - y0;
                                        double v00 = src(y0, x0, ch);
                                        double v01 = src(y0, x1, ch);
                                        double v10 = src(y1, x0, ch);
                                        double v11 = src(y1, x1, ch);
                                        double v0 = v00 * (1 - wx) + v01 * wx;
                                        double v1 = v10 * (1 - wx) + v11 * wx;
                                        result(y, x, ch) = v0 * (1 - wy) + v1 * wy;
                                    }
                                }
                            }
                        }
                        return result;
                    }
                }
            } // namespace transform

            // --------------------------------------------------------------------
            // Feature detection
            // --------------------------------------------------------------------
            namespace features
            {
                // Harris corner detector
                template <class E>
                inline auto harris_corners(const xexpression<E>& img, double k = 0.04, double threshold = 0.01, int window_size = 3)
                {
                    auto gray = (img.derived_cast().dimension() == 3) ? color::rgb2gray(img) : eval(img);
                    auto Ix = filters::convolve2d(gray, filters::prewitt_kx(), "same", "reflect");
                    auto Iy = filters::convolve2d(gray, filters::prewitt_ky(), "same", "reflect");
                    auto Ixx = Ix * Ix;
                    auto Iyy = Iy * Iy;
                    auto Ixy = Ix * Iy;
                    // Sum over window
                    auto gauss = filters::gaussian_blur(xarray_container<double>(), 1.0); // create kernel
                    auto Sxx = filters::gaussian_blur(Ixx, 1.0);
                    auto Syy = filters::gaussian_blur(Iyy, 1.0);
                    auto Sxy = filters::gaussian_blur(Ixy, 1.0);
                    size_t h = gray.shape()[0];
                    size_t w = gray.shape()[1];
                    xarray_container<double> R({h, w});
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            double det = Sxx(y, x) * Syy(y, x) - Sxy(y, x) * Sxy(y, x);
                            double trace = Sxx(y, x) + Syy(y, x);
                            R(y, x) = det - k * trace * trace;
                        }
                    }
                    // Non-maximum suppression
                    xarray_container<uint8_t> corners({h, w}, 0);
                    double maxR = xt::amax(R)();
                    for (size_t y = 1; y < h-1; ++y)
                    {
                        for (size_t x = 1; x < w-1; ++x)
                        {
                            if (R(y, x) > threshold * maxR)
                            {
                                bool is_max = true;
                                for (int dy = -1; dy <= 1 && is_max; ++dy)
                                    for (int dx = -1; dx <= 1; ++dx)
                                        if (R(y+dy, x+dx) > R(y, x))
                                            is_max = false;
                                if (is_max) corners(y, x) = 255;
                            }
                        }
                    }
                    return corners;
                }

                // FAST corner detector
                template <class E>
                inline auto fast_corners(const xexpression<E>& img, int threshold = 50, bool nonmax_suppression = true)
                {
                    auto gray = (img.derived_cast().dimension() == 3) ? color::rgb2gray(img) : eval(img);
                    size_t h = gray.shape()[0];
                    size_t w = gray.shape()[1];
                    xarray_container<uint8_t> corners({h, w}, 0);
                    // Circle of 16 pixels
                    const int circle[16][2] = {{0,3},{1,3},{2,2},{3,1},{3,0},{3,-1},{2,-2},{1,-3},{0,-3},{-1,-3},{-2,-2},{-3,-1},{-3,0},{-3,1},{-2,2},{-1,3}};
                    for (size_t y = 3; y < h-3; ++y)
                    {
                        for (size_t x = 3; x < w-3; ++x)
                        {
                            double center = gray(y, x);
                            int brighter = 0, darker = 0;
                            for (int i = 0; i < 16; ++i)
                            {
                                double val = gray(y + circle[i][1], x + circle[i][0]);
                                if (val > center + threshold) brighter++;
                                if (val < center - threshold) darker++;
                            }
                            if (brighter >= 12 || darker >= 12)
                                corners(y, x) = 255;
                        }
                    }
                    // Non-maximum suppression (simple)
                    if (nonmax_suppression)
                    {
                        auto suppressed = xt::zeros<uint8_t>(corners.shape());
                        for (size_t y = 1; y < h-1; ++y)
                        {
                            for (size_t x = 1; x < w-1; ++x)
                            {
                                if (corners(y, x))
                                {
                                    bool is_max = true;
                                    for (int dy = -1; dy <= 1; ++dy)
                                        for (int dx = -1; dx <= 1; ++dx)
                                            if (corners(y+dy, x+dx) && (gray(y+dy, x+dx) > gray(y, x)))
                                                is_max = false;
                                    if (is_max) suppressed(y, x) = 255;
                                }
                            }
                        }
                        return suppressed;
                    }
                    return corners;
                }

                // ORB feature descriptor (simplified)
                template <class E>
                inline auto orb_descriptors(const xexpression<E>& img, const std::vector<std::pair<int,int>>& keypoints)
                {
                    auto gray = (img.derived_cast().dimension() == 3) ? color::rgb2gray(img) : eval(img);
                    size_t n = keypoints.size();
                    xarray_container<uint8_t> descriptors({n, 32}); // 256 bits
                    // For each keypoint, compute BRIEF descriptor (simplified)
                    const int pattern[256][4] = {{0}}; // pre-defined pairs, simplified
                    for (size_t i = 0; i < n; ++i)
                    {
                        int cx = keypoints[i].first;
                        int cy = keypoints[i].second;
                        for (int byte = 0; byte < 32; ++byte)
                        {
                            uint8_t val = 0;
                            for (int bit = 0; bit < 8; ++bit)
                            {
                                int idx = byte * 8 + bit;
                                // Random pair (simplified)
                                int x1 = cx + ((idx * 13) % 31) - 15;
                                int y1 = cy + ((idx * 17) % 31) - 15;
                                int x2 = cx + ((idx * 23) % 31) - 15;
                                int y2 = cy + ((idx * 29) % 31) - 15;
                                x1 = std::clamp(x1, 0, static_cast<int>(gray.shape()[1])-1);
                                y1 = std::clamp(y1, 0, static_cast<int>(gray.shape()[0])-1);
                                x2 = std::clamp(x2, 0, static_cast<int>(gray.shape()[1])-1);
                                y2 = std::clamp(y2, 0, static_cast<int>(gray.shape()[0])-1);
                                if (gray(y1, x1) < gray(y2, x2))
                                    val |= (1 << bit);
                            }
                            descriptors(i, byte) = val;
                        }
                    }
                    return descriptors;
                }

                // Hough line transform
                template <class E>
                inline auto hough_lines(const xexpression<E>& edges, double rho_res = 1.0, double theta_res = M_PI/180.0, int threshold = 100)
                {
                    const auto& edge_img = edges.derived_cast();
                    size_t h = edge_img.shape()[0];
                    size_t w = edge_img.shape()[1];
                    double max_rho = std::sqrt(h*h + w*w);
                    size_t rho_bins = static_cast<size_t>(std::ceil(2.0 * max_rho / rho_res));
                    size_t theta_bins = static_cast<size_t>(std::ceil(M_PI / theta_res));
                    xarray_container<size_t> accum({rho_bins, theta_bins}, 0);
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            if (edge_img(y, x) > 128)
                            {
                                for (size_t t = 0; t < theta_bins; ++t)
                                {
                                    double theta = t * theta_res;
                                    double rho = x * std::cos(theta) + y * std::sin(theta);
                                    int rho_idx = static_cast<int>((rho + max_rho) / rho_res);
                                    if (rho_idx >= 0 && static_cast<size_t>(rho_idx) < rho_bins)
                                        accum(rho_idx, t)++;
                                }
                            }
                        }
                    }
                    // Find peaks
                    std::vector<std::tuple<double, double, size_t>> lines;
                    for (size_t r = 1; r < rho_bins-1; ++r)
                    {
                        for (size_t t = 1; t < theta_bins-1; ++t)
                        {
                            if (accum(r, t) > static_cast<size_t>(threshold))
                            {
                                bool is_max = true;
                                for (int dr = -1; dr <= 1; ++dr)
                                    for (int dt = -1; dt <= 1; ++dt)
                                        if (accum(r+dr, t+dt) > accum(r, t))
                                            is_max = false;
                                if (is_max)
                                {
                                    double rho = r * rho_res - max_rho;
                                    double theta = t * theta_res;
                                    lines.emplace_back(rho, theta, accum(r, t));
                                }
                            }
                        }
                    }
                    return lines;
                }

                // Hough circle transform
                template <class E>
                inline auto hough_circles(const xexpression<E>& edges, int min_radius, int max_radius, double dp = 1.0, int min_dist = 20, int threshold = 100)
                {
                    const auto& edge_img = edges.derived_cast();
                    size_t h = edge_img.shape()[0];
                    size_t w = edge_img.shape()[1];
                    std::vector<std::tuple<int, int, int>> circles;
                    // 3D accumulator (x, y, r)
                    size_t r_bins = static_cast<size_t>(max_radius - min_radius + 1);
                    xarray_container<size_t> accum({h, w, r_bins}, 0);
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            if (edge_img(y, x) > 128)
                            {
                                for (int r = min_radius; r <= max_radius; ++r)
                                {
                                    // For each radius, increment along gradient direction
                                    // Simplified: increment all possible centers in a circle (expensive)
                                    // For performance, we use gradient direction (not fully implemented)
                                    // Placeholder: just mark around
                                    for (int theta = 0; theta < 360; theta += 10)
                                    {
                                        double rad = theta * M_PI / 180.0;
                                        int cx = static_cast<int>(x - r * std::cos(rad));
                                        int cy = static_cast<int>(y - r * std::sin(rad));
                                        if (cx >= 0 && cx < static_cast<int>(w) && cy >= 0 && cy < static_cast<int>(h))
                                            accum(cy, cx, r - min_radius)++;
                                    }
                                }
                            }
                        }
                    }
                    // Find peaks
                    for (size_t r_idx = 0; r_idx < r_bins; ++r_idx)
                    {
                        for (size_t y = 1; y < h-1; ++y)
                        {
                            for (size_t x = 1; x < w-1; ++x)
                            {
                                if (accum(y, x, r_idx) > static_cast<size_t>(threshold))
                                {
                                    bool is_max = true;
                                    for (int dy = -1; dy <= 1; ++dy)
                                        for (int dx = -1; dx <= 1; ++dx)
                                            if (accum(y+dy, x+dx, r_idx) > accum(y, x, r_idx))
                                                is_max = false;
                                    if (is_max)
                                    {
                                        int r = static_cast<int>(r_idx) + min_radius;
                                        // Check min distance to existing circles
                                        bool too_close = false;
                                        for (const auto& c : circles)
                                        {
                                            int cx = std::get<0>(c);
                                            int cy = std::get<1>(c);
                                            if (std::abs(static_cast<int>(x) - cx) < min_dist && std::abs(static_cast<int>(y) - cy) < min_dist)
                                            {
                                                too_close = true;
                                                break;
                                            }
                                        }
                                        if (!too_close)
                                            circles.emplace_back(static_cast<int>(x), static_cast<int>(y), r);
                                    }
                                }
                            }
                        }
                    }
                    return circles;
                }
            } // namespace features

            // --------------------------------------------------------------------
            // Histogram and thresholding
            // --------------------------------------------------------------------
            namespace hist
            {
                template <class E>
                inline auto histogram(const xexpression<E>& img, size_t bins = 256)
                {
                    const auto& src = img.derived_cast();
                    xarray_container<size_t> hist({bins}, 0);
                    double min_val = xt::amin(src)();
                    double max_val = xt::amax(src)();
                    double scale = (bins - 1) / (max_val - min_val + 1e-10);
                    for (size_t i = 0; i < src.size(); ++i)
                    {
                        double val = src.flat(i);
                        size_t bin = static_cast<size_t>((val - min_val) * scale);
                        bin = std::min(bin, bins - 1);
                        hist(bin)++;
                    }
                    return hist;
                }

                template <class E>
                inline double otsu_threshold(const xexpression<E>& img)
                {
                    auto h = histogram(img, 256);
                    size_t total = xt::sum(h)();
                    double sum = 0;
                    for (size_t i = 0; i < 256; ++i)
                        sum += i * h(i);
                    double sumB = 0;
                    size_t wB = 0;
                    double max_var = 0;
                    double threshold = 0;
                    for (size_t t = 0; t < 256; ++t)
                    {
                        wB += h(t);
                        if (wB == 0) continue;
                        size_t wF = total - wB;
                        if (wF == 0) break;
                        sumB += t * h(t);
                        double mB = sumB / wB;
                        double mF = (sum - sumB) / wF;
                        double var_between = wB * wF * (mB - mF) * (mB - mF);
                        if (var_between > max_var)
                        {
                            max_var = var_between;
                            threshold = t;
                        }
                    }
                    return threshold / 255.0;
                }

                template <class E>
                inline auto threshold_binary(const xexpression<E>& img, double thresh)
                {
                    auto result = eval(img);
                    for (auto& v : result)
                        v = (v > thresh) ? 1.0 : 0.0;
                    return result;
                }

                template <class E>
                inline auto adaptive_threshold(const xexpression<E>& img, int block_size = 11, double C = 2.0)
                {
                    auto gray = (img.derived_cast().dimension() == 3) ? color::rgb2gray(img) : eval(img);
                    size_t h = gray.shape()[0];
                    size_t w = gray.shape()[1];
                    int half = block_size / 2;
                    xarray_container<uint8_t> result({h, w});
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            int y0 = std::max(0, static_cast<int>(y) - half);
                            int y1 = std::min(static_cast<int>(h)-1, static_cast<int>(y) + half);
                            int x0 = std::max(0, static_cast<int>(x) - half);
                            int x1 = std::min(static_cast<int>(w)-1, static_cast<int>(x) + half);
                            double sum = 0;
                            size_t cnt = 0;
                            for (int yy = y0; yy <= y1; ++yy)
                                for (int xx = x0; xx <= x1; ++xx)
                                    { sum += gray(yy, xx); cnt++; }
                            double mean = sum / cnt;
                            result(y, x) = (gray(y, x) > mean - C) ? 255 : 0;
                        }
                    }
                    return result;
                }
            } // namespace hist

            // --------------------------------------------------------------------
            // Connected components labeling
            // --------------------------------------------------------------------
            inline auto connected_components(const xarray_container<uint8_t>& binary, int connectivity = 4)
            {
                size_t h = binary.shape()[0];
                size_t w = binary.shape()[1];
                xarray_container<int> labels({h, w}, 0);
                int current_label = 1;
                std::vector<int> parent = {0};
                // First pass
                for (size_t y = 0; y < h; ++y)
                {
                    for (size_t x = 0; x < w; ++x)
                    {
                        if (binary(y, x) == 0) continue;
                        std::set<int> neighbor_labels;
                        if (x > 0 && labels(y, x-1) > 0) neighbor_labels.insert(labels(y, x-1));
                        if (y > 0 && labels(y-1, x) > 0) neighbor_labels.insert(labels(y-1, x));
                        if (connectivity == 8)
                        {
                            if (x > 0 && y > 0 && labels(y-1, x-1) > 0) neighbor_labels.insert(labels(y-1, x-1));
                            if (x < w-1 && y > 0 && labels(y-1, x+1) > 0) neighbor_labels.insert(labels(y-1, x+1));
                        }
                        if (neighbor_labels.empty())
                        {
                            labels(y, x) = current_label;
                            parent.push_back(current_label);
                            current_label++;
                        }
                        else
                        {
                            int min_label = *std::min_element(neighbor_labels.begin(), neighbor_labels.end());
                            labels(y, x) = min_label;
                            for (int nl : neighbor_labels)
                            {
                                if (nl != min_label)
                                {
                                    // Union
                                    int root1 = min_label;
                                    while (parent[root1] != root1) root1 = parent[root1];
                                    int root2 = nl;
                                    while (parent[root2] != root2) root2 = parent[root2];
                                    if (root1 != root2)
                                        parent[root2] = root1;
                                }
                            }
                        }
                    }
                }
                // Second pass: flatten labels
                for (size_t y = 0; y < h; ++y)
                    for (size_t x = 0; x < w; ++x)
                        if (labels(y, x) > 0)
                        {
                            int root = labels(y, x);
                            while (parent[root] != root) root = parent[root];
                            labels(y, x) = root;
                        }
                // Renumber sequentially
                std::map<int, int> new_labels;
                int next_label = 1;
                for (size_t y = 0; y < h; ++y)
                    for (size_t x = 0; x < w; ++x)
                        if (labels(y, x) > 0)
                        {
                            int old = labels(y, x);
                            if (new_labels.find(old) == new_labels.end())
                                new_labels[old] = next_label++;
                            labels(y, x) = new_labels[old];
                        }
                return std::make_pair(labels, next_label - 1);
            }

            // --------------------------------------------------------------------
            // Image utilities
            // --------------------------------------------------------------------
            inline auto imread(const std::string& filename)
            {
                // Simplified PPM reader
                std::ifstream file(filename, std::ios::binary);
                if (!file) XTENSOR_THROW(std::runtime_error, "Cannot open file");
                std::string magic;
                file >> magic;
                if (magic != "P6") XTENSOR_THROW(std::runtime_error, "Only PPM P6 format supported");
                size_t w, h, maxval;
                file >> w >> h >> maxval;
                file.get(); // newline
                xarray_container<uint8_t> img({h, w, 3});
                for (size_t y = 0; y < h; ++y)
                    for (size_t x = 0; x < w; ++x)
                        for (size_t c = 0; c < 3; ++c)
                            img(y, x, c) = static_cast<uint8_t>(file.get());
                return img;
            }

            inline void imwrite(const std::string& filename, const xarray_container<uint8_t>& img)
            {
                std::ofstream file(filename, std::ios::binary);
                size_t h = img.shape()[0];
                size_t w = img.shape()[1];
                size_t c = img.shape()[2];
                if (c == 3)
                    file << "P6\n" << w << " " << h << "\n255\n";
                else if (c == 1)
                    file << "P5\n" << w << " " << h << "\n255\n";
                else
                    XTENSOR_THROW(std::invalid_argument, "Only 1 or 3 channels supported");
                for (size_t y = 0; y < h; ++y)
                    for (size_t x = 0; x < w; ++x)
                        for (size_t ch = 0; ch < c; ++ch)
                            file.put(static_cast<char>(img(y, x, ch)));
            }

        } // namespace image

        // Bring image functions into xt namespace
        using image::color::rgb2gray;
        using image::color::rgb2hsv;
        using image::color::hsv2rgb;
        using image::color::rgb2lab;
        using image::filters::convolve2d;
        using image::filters::gaussian_blur;
        using image::filters::median_filter;
        using image::filters::bilateral_filter;
        using image::filters::sobel;
        using image::filters::laplacian_of_gaussian;
        using image::filters::canny;
        using image::filters::unsharp_mask;
        using image::morphology::erode;
        using image::morphology::dilate;
        using image::morphology::opening;
        using image::morphology::closing;
        using image::morphology::skeletonize;
        using image::transform::resize;
        using image::transform::rotate;
        using image::transform::flip_horizontal;
        using image::transform::flip_vertical;
        using image::transform::warp_affine;
        using image::features::harris_corners;
        using image::features::fast_corners;
        using image::features::hough_lines;
        using image::features::hough_circles;
        using image::hist::otsu_threshold;
        using image::hist::adaptive_threshold;
        using image::connected_components;
        using image::imread;
        using image::imwrite;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIMAGE_PROCESSING_HPP

// image/ximage_processing.hpp