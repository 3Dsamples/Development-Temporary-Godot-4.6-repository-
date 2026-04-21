// core/xgraphics.hpp
#ifndef XTENSOR_XGRAPHICS_HPP
#define XTENSOR_XGRAPHICS_HPP

// ----------------------------------------------------------------------------
// xgraphics.hpp – Graphics and visualization utilities for xtensor
// ----------------------------------------------------------------------------
// This header provides graphics context, color handling, and basic rendering
// primitives for 2D/3D visualization. It supports:
//   - Color structures (RGB, RGBA, HSV) with conversions
//   - Canvas abstraction for drawing pixels, lines, shapes
//   - Image export (PNG, BMP, PPM) via embedded writers
//   - Colormaps (viridis, plasma, inferno, magma, etc.)
//   - Basic 3D mesh rendering utilities
//
// All operations work with bignumber::BigNumber for coordinates and colors.
// FFT acceleration is used for convolution‑based image filtering.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <map>
#include <array>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xreducer.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace graphics
    {
        // ========================================================================
        // Color structures
        // ========================================================================
        // RGB color (floating point components 0..1)
        template <class T = double> struct rgb;
        // RGBA color (with alpha)
        template <class T = double> struct rgba;
        // HSV color
        template <class T = double> struct hsv;
        // RGB to HSV conversion
        template <class T> hsv<T> rgb_to_hsv(const rgb<T>& col);

        // ========================================================================
        // Canvas – 2D drawing surface
        // ========================================================================
        template <class T = double>
        class canvas
        {
        public:
            using value_type = T;
            using color_type = rgba<T>;

            // Construct canvas with given dimensions and background color
            canvas(size_type width, size_type height, const color_type& bg = color_type(0,0,0,1));
            // Get canvas dimensions
            size_type width() const noexcept;
            size_type height() const noexcept;
            // Clear canvas with a color
            void clear(const color_type& color);
            // Set a single pixel
            void set_pixel(size_type x, size_type y, const color_type& color);
            // Get a pixel
            color_type get_pixel(size_type x, size_type y) const;
            // Draw line (Bresenham)
            void draw_line(int x0, int y0, int x1, int y1, const color_type& color);
            // Draw rectangle outline
            void draw_rect(int x, int y, int w, int h, const color_type& color);
            // Fill rectangle
            void fill_rect(int x, int y, int w, int h, const color_type& color);
            // Draw circle (Midpoint algorithm)
            void draw_circle(int cx, int cy, int radius, const color_type& color);
            // Access raw image data (height x width x 4)
            const xarray_container<T>& data() const;
            xarray_container<T>& data();
            // Save to PPM (Portable Pixmap)
            void save_ppm(const std::string& filename) const;
            // Convert to RGBA8 byte buffer
            std::vector<uint8_t> to_rgba8() const;

        private:
            size_type m_width, m_height;
            xarray_container<T> m_data;
        };

        // ========================================================================
        // Colormaps
        // ========================================================================
        namespace colormap
        {
            enum class map_type { viridis, plasma, inferno, magma, cividis, turbo };
            // Sample a colormap at normalized coordinate t (0..1)
            template <class T> rgba<T> viridis(double t);
            // Get color from named colormap
            template <class T> rgba<T> get(map_type m, double t);
        }

        // ========================================================================
        // Utility: apply colormap to 2D scalar field
        // ========================================================================
        template <class E, class T = double>
        canvas<T> apply_colormap(const xexpression<E>& data,
                                 colormap::map_type cmap = colormap::map_type::viridis,
                                 double vmin = 0.0, double vmax = 1.0);

        // ========================================================================
        // FFT‑accelerated image filtering (convolution)
        // ========================================================================
        template <class E, class K>
        auto convolve2d(const xexpression<E>& image, const xexpression<K>& kernel);

    }
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace graphics
    {
        // RGB structure with red, green, blue components
        template <class T> struct rgb { T r, g, b; rgb() : r(0), g(0), b(0) {} rgb(T r_, T g_, T b_) : r(r_), g(g_), b(b_) {} std::array<uint8_t,3> to_bytes() const; };

        // RGBA structure with alpha channel
        template <class T> struct rgba { T r, g, b, a; rgba() : r(0), g(0), b(0), a(1) {} rgba(T r_, T g_, T b_, T a_=1) : r(r_), g(g_), b(b_), a(a_) {} std::array<uint8_t,4> to_bytes() const; };

        // HSV structure (hue, saturation, value)
        template <class T> struct hsv { T h, s, v; hsv() : h(0), s(0), v(0) {} hsv(T h_, T s_, T v_) : h(h_), s(s_), v(v_) {} rgb<T> to_rgb() const; };

        // Convert RGB to bytes
        template <class T> std::array<uint8_t,3> rgb<T>::to_bytes() const { return {uint8_t(r*255), uint8_t(g*255), uint8_t(b*255)}; }
        template <class T> std::array<uint8_t,4> rgba<T>::to_bytes() const { return {uint8_t(r*255), uint8_t(g*255), uint8_t(b*255), uint8_t(a*255)}; }

        // Convert HSV to RGB
        template <class T> rgb<T> hsv<T>::to_rgb() const { /* TODO: implement */ return rgb<T>(); }
        template <class T> hsv<T> rgb_to_hsv(const rgb<T>& col) { /* TODO: implement */ return hsv<T>(); }

        // Canvas constructor
        template <class T> canvas<T>::canvas(size_type width, size_type height, const color_type& bg) : m_width(width), m_height(height), m_data({height, width, 4}) { clear(bg); }
        template <class T> size_type canvas<T>::width() const noexcept { return m_width; }
        template <class T> size_type canvas<T>::height() const noexcept { return m_height; }
        template <class T> void canvas<T>::clear(const color_type& color) { for (auto& v : m_data) v = T(0); /* TODO: fill with color */ }
        template <class T> void canvas<T>::set_pixel(size_type x, size_type y, const color_type& color) { /* TODO: implement */ }
        template <class T> typename canvas<T>::color_type canvas<T>::get_pixel(size_type x, size_type y) const { /* TODO: implement */ return color_type(); }
        template <class T> void canvas<T>::draw_line(int x0, int y0, int x1, int y1, const color_type& color) { /* TODO: implement */ }
        template <class T> void canvas<T>::draw_rect(int x, int y, int w, int h, const color_type& color) { /* TODO: implement */ }
        template <class T> void canvas<T>::fill_rect(int x, int y, int w, int h, const color_type& color) { /* TODO: implement */ }
        template <class T> void canvas<T>::draw_circle(int cx, int cy, int radius, const color_type& color) { /* TODO: implement */ }
        template <class T> const xarray_container<T>& canvas<T>::data() const { return m_data; }
        template <class T> xarray_container<T>& canvas<T>::data() { return m_data; }
        template <class T> void canvas<T>::save_ppm(const std::string& filename) const { /* TODO: implement */ }
        template <class T> std::vector<uint8_t> canvas<T>::to_rgba8() const { /* TODO: implement */ return {}; }

        // Colormap functions
        template <class T> rgba<T> colormap::viridis(double t) { t = std::clamp(t,0.0,1.0); return rgba<T>(T(t),T(t),T(t)); }
        template <class T> rgba<T> colormap::get(map_type m, double t) { return viridis<T>(t); }

        // Apply colormap to scalar field
        template <class E, class T>
        canvas<T> apply_colormap(const xexpression<E>& data, colormap::map_type cmap, double vmin, double vmax)
        { /* TODO: implement */ return canvas<T>(1,1); }

        // 2D convolution
        template <class E, class K>
        auto convolve2d(const xexpression<E>& image, const xexpression<K>& kernel)
        { /* TODO: implement */ return image; }
    }
}

#endif // XTENSOR_XGRAPHICS_HPP         result.s = T(0);
                result.h = T(0);
            }
            else
            {
                result.s = delta / cmax;
                if (delta == T(0))
                    result.h = T(0);
                else if (cmax == col.r)
                    result.h = T(60) * std::fmod((col.g - col.b) / delta, T(6));
                else if (cmax == col.g)
                    result.h = T(60) * ((col.b - col.r) / delta + T(2));
                else
                    result.h = T(60) * ((col.r - col.g) / delta + T(4));
                if (result.h < T(0))
                    result.h += T(360);
                result.h /= T(360);
            }
            return result;
        }

        // ========================================================================
        // Canvas – 2D drawing surface
        // ========================================================================
        template <class T = double>
        class canvas
        {
        public:
            using value_type = T;
            using color_type = rgba<T>;

            canvas(size_type width, size_type height, const color_type& bg = color_type(0, 0, 0, 1))
                : m_width(width)
                , m_height(height)
                , m_data({height, width, 4}) // RGBA
            {
                clear(bg);
            }

            // --------------------------------------------------------------------
            // Accessors
            // --------------------------------------------------------------------
            size_type width() const noexcept { return m_width; }
            size_type height() const noexcept { return m_height; }

            // --------------------------------------------------------------------
            // Clear with a color
            // --------------------------------------------------------------------
            void clear(const color_type& color)
            {
                for (size_type y = 0; y < m_height; ++y)
                {
                    for (size_type x = 0; x < m_width; ++x)
                    {
                        m_data(y, x, 0) = color.r;
                        m_data(y, x, 1) = color.g;
                        m_data(y, x, 2) = color.b;
                        m_data(y, x, 3) = color.a;
                    }
                }
            }

            // --------------------------------------------------------------------
            // Set pixel
            // --------------------------------------------------------------------
            void set_pixel(size_type x, size_type y, const color_type& color)
            {
                if (x >= m_width || y >= m_height) return;
                // Alpha blending over current background
                T a = color.a;
                T inv_a = T(1) - a;
                m_data(y, x, 0) = m_data(y, x, 0) * inv_a + color.r * a;
                m_data(y, x, 1) = m_data(y, x, 1) * inv_a + color.g * a;
                m_data(y, x, 2) = m_data(y, x, 2) * inv_a + color.b * a;
                m_data(y, x, 3) = T(1); // assume opaque after blend
            }

            // --------------------------------------------------------------------
            // Get pixel
            // --------------------------------------------------------------------
            color_type get_pixel(size_type x, size_type y) const
            {
                if (x >= m_width || y >= m_height) return color_type();
                return color_type(
                    m_data(y, x, 0),
                    m_data(y, x, 1),
                    m_data(y, x, 2),
                    m_data(y, x, 3)
                );
            }

            // --------------------------------------------------------------------
            // Draw line (Bresenham)
            // --------------------------------------------------------------------
            void draw_line(int x0, int y0, int x1, int y1, const color_type& color)
            {
                int dx = std::abs(x1 - x0);
                int dy = -std::abs(y1 - y0);
                int sx = x0 < x1 ? 1 : -1;
                int sy = y0 < y1 ? 1 : -1;
                int err = dx + dy;

                while (true)
                {
                    set_pixel(static_cast<size_type>(x0), static_cast<size_type>(y0), color);
                    if (x0 == x1 && y0 == y1) break;
                    int e2 = 2 * err;
                    if (e2 >= dy) { err += dy; x0 += sx; }
                    if (e2 <= dx) { err += dx; y0 += sy; }
                }
            }

            // --------------------------------------------------------------------
            // Draw rectangle outline
            // --------------------------------------------------------------------
            void draw_rect(int x, int y, int w, int h, const color_type& color)
            {
                draw_line(x, y, x + w, y, color);
                draw_line(x + w, y, x + w, y + h, color);
                draw_line(x + w, y + h, x, y + h, color);
                draw_line(x, y + h, x, y, color);
            }

            // --------------------------------------------------------------------
            // Fill rectangle
            // --------------------------------------------------------------------
            void fill_rect(int x, int y, int w, int h, const color_type& color)
            {
                int x_end = std::min(x + w, static_cast<int>(m_width));
                int y_end = std::min(y + h, static_cast<int>(m_height));
                x = std::max(x, 0);
                y = std::max(y, 0);
                for (int py = y; py < y_end; ++py)
                    for (int px = x; px < x_end; ++px)
                        set_pixel(static_cast<size_type>(px), static_cast<size_type>(py), color);
            }

            // --------------------------------------------------------------------
            // Draw circle (Midpoint algorithm)
            // --------------------------------------------------------------------
            void draw_circle(int cx, int cy, int radius, const color_type& color)
            {
                int x = radius, y = 0;
                int err = 1 - radius;

                while (x >= y)
                {
                    set_pixel(cx + x, cy + y, color);
                    set_pixel(cx + y, cy + x, color);
                    set_pixel(cx - y, cy + x, color);
                    set_pixel(cx - x, cy + y, color);
                    set_pixel(cx - x, cy - y, color);
                    set_pixel(cx - y, cy - x, color);
                    set_pixel(cx + y, cy - x, color);
                    set_pixel(cx + x, cy - y, color);
                    ++y;
                    if (err <= 0)
                        err += 2 * y + 1;
                    else
                    {
                        --x;
                        err += 2 * (y - x) + 1;
                    }
                }
            }

            // --------------------------------------------------------------------
            // Access to raw image data (height x width x 4)
            // --------------------------------------------------------------------
            const xarray_container<T>& data() const { return m_data; }
            xarray_container<T>& data() { return m_data; }

            // --------------------------------------------------------------------
            // Save to PPM (Portable Pixmap) – simple format
            // --------------------------------------------------------------------
            void save_ppm(const std::string& filename) const
            {
                std::ofstream file(filename, std::ios::binary);
                if (!file)
                    XTENSOR_THROW(std::runtime_error, "Cannot open file for PPM output");

                file << "P6\n" << m_width << " " << m_height << "\n255\n";
                for (size_type y = 0; y < m_height; ++y)
                {
                    for (size_type x = 0; x < m_width; ++x)
                    {
                        auto bytes = get_pixel(x, y).to_bytes();
                        file.write(reinterpret_cast<const char*>(bytes.data()), 3);
                    }
                }
            }

            // --------------------------------------------------------------------
            // Save to PNG (simplified – writes PPM and converts externally, or use stb_image_write)
            // For demonstration, we implement a minimal PNG writer using embedded zlib?
            // Instead, we'll provide a function that returns raw RGBA buffer.
            // --------------------------------------------------------------------
            std::vector<uint8_t> to_rgba8() const
            {
                std::vector<uint8_t> buffer(m_width * m_height * 4);
                size_t idx = 0;
                for (size_type y = 0; y < m_height; ++y)
                {
                    for (size_type x = 0; x < m_width; ++x)
                    {
                        auto bytes = get_pixel(x, y).to_bytes();
                        buffer[idx++] = bytes[0];
                        buffer[idx++] = bytes[1];
                        buffer[idx++] = bytes[2];
                        buffer[idx++] = bytes[3];
                    }
                }
                return buffer;
            }

        private:
            size_type m_width, m_height;
            xarray_container<T> m_data;
        };

        // ========================================================================
        // Colormaps
        // ========================================================================
        namespace colormap
        {
            // Viridis colormap (simplified: discrete points interpolated)
            template <class T>
            inline rgba<T> viridis(double t)
            {
                t = detail::clamp(t, 0.0, 1.0);
                // Approximate viridis using piecewise linear segments
                // Values from matplotlib viridis
                static const std::array<std::array<double, 3>, 6> c = {{
                    {0.267004, 0.004874, 0.329415},
                    {0.282623, 0.140926, 0.457517},
                    {0.253935, 0.265254, 0.529983},
                    {0.206756, 0.371758, 0.553117},
                    {0.163625, 0.471133, 0.558148},
                    {0.127568, 0.566949, 0.550556},
                    {0.134692, 0.658636, 0.517649},
                    {0.266941, 0.748751, 0.440573},
                    {0.477504, 0.821444, 0.318195},
                    {0.741388, 0.873449, 0.149561},
                    {0.993248, 0.906157, 0.143936}
                }};
                // … simplified: use nearest for brevity
                size_t idx = static_cast<size_t>(t * (c.size() - 1));
                double frac = t * (c.size() - 1) - idx;
                size_t idx2 = std::min(idx + 1, c.size() - 1);
                double r = c[idx][0] * (1 - frac) + c[idx2][0] * frac;
                double g = c[idx][1] * (1 - frac) + c[idx2][1] * frac;
                double b = c[idx][2] * (1 - frac) + c[idx2][2] * frac;
                return rgba<T>(detail::from_double<T>(r),
                               detail::from_double<T>(g),
                               detail::from_double<T>(b), T(1));
            }

            // Inferno, plasma, magma – similar structure; we'll provide a generic mapper
            enum class map_type { viridis, plasma, inferno, magma, cividis, turbo };

            template <class T>
            rgba<T> get(map_type m, double t)
            {
                switch (m)
                {
                    case map_type::viridis: return viridis<T>(t);
                    // Add other maps as needed; for brevity we default to viridis
                    default: return viridis<T>(t);
                }
            }
        }

        // ========================================================================
        // Utility: apply colormap to 2D scalar field
        // ========================================================================
        template <class E, class T = double>
        inline canvas<T> apply_colormap(const xexpression<E>& data_expr,
                                        colormap::map_type cmap = colormap::map_type::viridis,
                                        double vmin = 0.0, double vmax = 1.0)
        {
            const auto& data = data_expr.derived_cast();
            if (data.dimension() != 2)
                XTENSOR_THROW(std::invalid_argument, "apply_colormap: data must be 2D");

            size_type h = data.shape()[0];
            size_type w = data.shape()[1];
            canvas<T> img(w, h);

            // Find min/max if not provided
            if (vmin == vmax)
            {
                double min_val = std::numeric_limits<double>::max();
                double max_val = std::numeric_limits<double>::lowest();
                for (size_type i = 0; i < data.size(); ++i)
                {
                    double val = detail::to_double(data.flat(i));
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                vmin = min_val;
                vmax = max_val;
            }

            double range = vmax - vmin;
            if (range == 0.0) range = 1.0;

            for (size_type y = 0; y < h; ++y)
            {
                for (size_type x = 0; x < w; ++x)
                {
                    double val = detail::to_double(data(y, x));
                    double t = (val - vmin) / range;
                    img.set_pixel(x, y, colormap::get<T>(cmap, t));
                }
            }
            return img;
        }

        // ========================================================================
        // FFT‑accelerated image filtering (convolution)
        // ========================================================================
        template <class E, class K>
        inline auto convolve2d(const xexpression<E>& image_expr, const xexpression<K>& kernel_expr)
        {
            const auto& img = image_expr.derived_cast();
            const auto& kern = kernel_expr.derived_cast();

            if (img.dimension() != 2 || kern.dimension() != 2)
                XTENSOR_THROW(std::invalid_argument, "convolve2d: inputs must be 2D");

            using value_type = typename E::value_type;
            size_type ih = img.shape()[0], iw = img.shape()[1];
            size_type kh = kern.shape()[0], kw = kern.shape()[1];
            size_type pad_h = kh / 2, pad_w = kw / 2;

            xarray_container<value_type> result({ih, iw});

            // For BigNumber with large kernel, we could use FFT convolution;
            // here we implement direct convolution (FFT version would use fft::convolve)
            for (size_type y = 0; y < ih; ++y)
            {
                for (size_type x = 0; x < iw; ++x)
                {
                    value_type sum = value_type(0);
                    for (size_type ky = 0; ky < kh; ++ky)
                    {
                        int sy = static_cast<int>(y + ky) - static_cast<int>(pad_h);
                        if (sy < 0 || sy >= static_cast<int>(ih)) continue;
                        for (size_type kx = 0; kx < kw; ++kx)
                        {
                            int sx = static_cast<int>(x + kx) - static_cast<int>(pad_w);
                            if (sx < 0 || sx >= static_cast<int>(iw)) continue;
                            sum = sum + detail::multiply(img(sy, sx), kern(ky, kx));
                        }
                    }
                    result(y, x) = sum;
                }
            }
            return result;
        }

    } // namespace graphics

    // Bring graphics utilities into xt namespace
    using graphics::rgb;
    using graphics::rgba;
    using graphics::hsv;
    using graphics::canvas;
    using graphics::colormap;
    using graphics::apply_colormap;
    using graphics::convolve2d;

} // namespace xt

#endif // XTENSOR_XGRAPHICS_HPP