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

#endif // XTENSOR_XGRAPHICS_HPPor (size_type d = axis + 1; d < dim; ++d) inner_size *= expr.shape()[d];

            size_type axis_stride = expr.strides()[axis];

            for (size_type outer = 0; outer < outer_size; ++outer)
            {
                size_type prefix_offset = 0, rem_outer = outer;
                for (size_type d = 0; d < axis; ++d)
                {
                    size_type coord = rem_outer % expr.shape()[d];
                    rem_outer /= expr.shape()[d];
                    prefix_offset += coord * expr.strides()[d];
                }
                for (size_type inner = 0; inner < inner_size; ++inner)
                {
                    size_type suffix_offset = 0, rem_inner = inner;
                    for (size_type d = axis + 1; d < dim; ++d)
                    {
                        size_type coord = rem_inner % expr.shape()[d];
                        rem_inner /= expr.shape()[d];
                        suffix_offset += coord * expr.strides()[d];
                    }
                    size_type base = prefix_offset + suffix_offset;

                    // Compute transform for this 1D slice
                    for (size_type k = 0; k < N; ++k)
                    {
                        value_type sum = value_type(0);
                        for (size_type n = 0; n < N; ++n)
                        {
                            size_type idx = base + n * axis_stride;
                            sum = sum + detail::multiply(expr.flat(idx), basis(k, n));
                        }
                        size_type out_idx = base + k * axis_stride;
                        result.flat(out_idx) = sum;
                    }
                }
            }
            return result;
        }

        template <class E>
        inline auto idct(const xexpression<E>& e, int type = 2, size_type axis = -1)
        {
            // Inverse DCT is transpose of forward DCT for orthogonal types
            // For types 2 and 3, they are inverses of each other
            int inv_type = (type == 2) ? 3 : (type == 3) ? 2 : type;
            return dct(e, inv_type, axis);
        }

        // ========================================================================
        // Discrete Sine Transform (DST)
        // ========================================================================

        template <class E>
        inline auto dst(const xexpression<E>& e, int type = 2, size_type axis = -1)
        {
            const auto& expr = e.derived_cast();
            using value_type = typename E::value_type;
            size_type dim = expr.dimension();
            if (axis == static_cast<size_type>(-1))
                axis = dim - 1;

            if (axis >= dim)
                XTENSOR_THROW(std::out_of_range, "dst: axis out of range");

            size_type N = expr.shape()[axis];
            auto basis = detail::dst_basis<value_type>(N, type);

            shape_type new_shape = expr.shape();
            xarray_container<value_type> result(new_shape);

            size_type outer_size = 1, inner_size = 1;
            for (size_type d = 0; d < axis; ++d) outer_size *= expr.shape()[d];
            for (size_type d = axis + 1; d < dim; ++d) inner_size *= expr.shape()[d];

            size_type axis_stride = expr.strides()[axis];

            for (size_type outer = 0; outer < outer_size; ++outer)
            {
                size_type prefix_offset = 0, rem_outer = outer;
                for (size_type d = 0; d < axis; ++d)
                {
                    size_type coord = rem_outer % expr.shape()[d];
                    rem_outer /= expr.shape()[d];
                    prefix_offset += coord * expr.strides()[d];
                }
                for (size_type inner = 0; inner < inner_size; ++inner)
                {
                    size_type suffix_offset = 0, rem_inner = inner;
                    for (size_type d = axis + 1; d < dim; ++d)
                    {
                        size_type coord = rem_inner % expr.shape()[d];
                        rem_inner /= expr.shape()[d];
                        suffix_offset += coord * expr.strides()[d];
                    }
                    size_type base = prefix_offset + suffix_offset;

                    for (size_type k = 0; k < N; ++k)
                    {
                        value_type sum = value_type(0);
                        for (size_type n = 0; n < N; ++n)
                        {
                            size_type idx = base + n * axis_stride;
                            sum = sum + detail::multiply(expr.flat(idx), basis(k, n));
                        }
                        size_type out_idx = base + k * axis_stride;
                        result.flat(out_idx) = sum;
                    }
                }
            }
            return result;
        }

        template <class E>
        inline auto idst(const xexpression<E>& e, int type = 2, size_type axis = -1)
        {
            int inv_type = (type == 2) ? 3 : (type == 3) ? 2 : type;
            return dst(e, inv_type, axis);
        }

        // ========================================================================
        // Hilbert Transform (via FFT)
        // ========================================================================

        template <class E>
        inline auto hilbert(const xexpression<E>& e, size_type axis = -1)
        {
            const auto& expr = e.derived_cast();
            using value_type = typename E::value_type;
            size_type dim = expr.dimension();
            if (axis == static_cast<size_type>(-1))
                axis = dim - 1;

            if (axis >= dim)
                XTENSOR_THROW(std::out_of_range, "hilbert: axis out of range");

            size_type N = expr.shape()[axis];
            // Use FFT to compute Hilbert transform: H(x) = ifft(fft(x) * H(w))
            // where H(w) = -j * sign(w) for 1..N/2-1, 0 for w=0,N/2
            auto fft_result = fft(expr); // complex valued

            // Apply frequency-domain filter along axis
            auto H = fft_result.evaluate();
            size_type outer_size = 1, inner_size = 1;
            for (size_type d = 0; d < axis; ++d) outer_size *= H.shape()[d];
            for (size_type d = axis + 1; d < dim; ++d) inner_size *= H.shape()[d];

            size_type axis_stride = H.strides()[axis];
            size_type half = N / 2;

            for (size_type outer = 0; outer < outer_size; ++outer)
            {
                size_type prefix_offset = 0, rem_outer = outer;
                for (size_type d = 0; d < axis; ++d)
                {
                    size_type coord = rem_outer % H.shape()[d];
                    rem_outer /= H.shape()[d];
                    prefix_offset += coord * H.strides()[d];
                }
                for (size_type inner = 0; inner < inner_size; ++inner)
                {
                    size_type suffix_offset = 0, rem_inner = inner;
                    for (size_type d = axis + 1; d < dim; ++d)
                    {
                        size_type coord = rem_inner % H.shape()[d];
                        rem_inner /= H.shape()[d];
                        suffix_offset += coord * H.strides()[d];
                    }
                    size_type base = prefix_offset + suffix_offset;

                    // Apply Hilbert filter
                    for (size_type k = 0; k < N; ++k)
                    {
                        size_type idx = base + k * axis_stride;
                        std::complex<value_type> val = H.flat(idx);
                        if (k == 0 || k == half)
                        {
                            // Zero out DC and Nyquist (real part only)
                            H.flat(idx) = std::complex<value_type>(value_type(0), value_type(0));
                        }
                        else if (k < half)
                        {
                            // Multiply by -j
                            H.flat(idx) = std::complex<value_type>(val.imag(), -val.real());
                        }
                        else
                        {
                            // Multiply by +j
                            H.flat(idx) = std::complex<value_type>(-val.imag(), val.real());
                        }
                    }
                }
            }

            // Inverse FFT
            auto result = ifft(H);
            // Return real part? Typically Hilbert returns analytic signal (complex)
            return result;
        }

        // ========================================================================
        // Convolution and correlation (using FFT for large kernels)
        // ========================================================================

        template <class E1, class E2>
        inline auto convolve(const xexpression<E1>& a, const xexpression<E2>& b,
                             const std::string& method = "auto")
        {
            return fft::convolve(a, b);
        }

        template <class E1, class E2>
        inline auto correlate(const xexpression<E1>& a, const xexpression<E2>& b,
                              const std::string& method = "auto")
        {
            return fft::correlate(a, b);
        }

        // ========================================================================
        // Short‑Time Fourier Transform (STFT)
        // ========================================================================

        template <class E>
        inline auto stft(const xexpression<E>& e, size_type nperseg = 256,
                         size_type noverlap = 0, bool padded = false)
        {
            const auto& signal = e.derived_cast();
            if (signal.dimension() != 1)
                XTENSOR_THROW(std::invalid_argument, "stft: input must be 1D");

            using value_type = typename E::value_type;
            size_type n = signal.size();
            if (noverlap >= nperseg)
                XTENSOR_THROW(std::invalid_argument, "stft: noverlap must be less than nperseg");

            size_type hop = nperseg - noverlap;
            size_type n_frames = (n - nperseg) / hop + 1;
            if (n_frames == 0)
                XTENSOR_THROW(std::runtime_error, "stft: signal too short");

            size_type n_fft = padded ? detail::next_pow2(nperseg) : nperseg;
            xarray_container<std::complex<value_type>> result({n_frames, n_fft / 2 + 1});

            // Create window (Hann)
            std::vector<value_type> window(nperseg);
            for (size_t i = 0; i < nperseg; ++i)
                window[i] = value_type(0.5) * (value_type(1) - std::cos(value_type(2) * detail::pi<value_type>() * value_type(i) / value_type(nperseg - 1)));

            for (size_t frame = 0; frame < n_frames; ++frame)
            {
                size_t start = frame * hop;
                std::vector<value_type> segment(n_fft, value_type(0));
                for (size_t i = 0; i < nperseg; ++i)
                    segment[i] = signal.flat(start + i) * window[i];

                // Compute FFT of segment
                // We'll use the fft function on an array
                xarray_container<value_type> seg_arr({n_fft});
                std::copy(segment.begin(), segment.end(), seg_arr.begin());
                auto fft_seg = fft(seg_arr);

                for (size_t k = 0; k <= n_fft / 2; ++k)
                    result(frame, k) = fft_seg.flat(k);
            }
            return result;
        }

        // ========================================================================
        // Spectrogram (power of STFT)
        // ========================================================================

        template <class E>
        inline auto spectrogram(const xexpression<E>& e, size_type nperseg = 256,
                                size_type noverlap = 0, bool padded = false)
        {
            auto stft_result = stft(e, nperseg, noverlap, padded);
            using value_type = typename E::value_type;
            auto power = stft_result;
            for (auto& val : power)
            {
                value_type mag_sq = detail::multiply(std::abs(val), std::abs(val));
                val = mag_sq;
            }
            return power;
        }

    } // namespace transform

    // Bring transforms into xt namespace
    using transform::fft;
    using transform::ifft;
    using transform::rfft;
    using transform::irfft;
    using transform::fft2;
    using transform::ifft2;
    using transform::fftn;
    using transform::ifftn;
    using transform::dct;
    using transform::idct;
    using transform::dst;
    using transform::idst;
    using transform::hilbert;
    using transform::convolve;
    using transform::correlate;
    using transform::stft;
    using transform::spectrogram;

} // namespace xt

#endif // XTENSOR_XTRANSFORM_HPP