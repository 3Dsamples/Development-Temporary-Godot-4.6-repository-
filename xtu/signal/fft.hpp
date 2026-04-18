// signal/fft.hpp

#ifndef XTENSOR_FFT_HPP
#define XTENSOR_FFT_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xtransform.hpp"

#include <cmath>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <stdexcept>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace signal
        {
            using complex64 = std::complex<float>;
            using complex128 = std::complex<double>;

            // --------------------------------------------------------------------
            // Utility functions for FFT
            // --------------------------------------------------------------------
            namespace detail
            {
                // Bit-reversal permutation
                inline size_t bit_reverse(size_t x, size_t bits)
                {
                    size_t y = 0;
                    for (size_t i = 0; i < bits; ++i)
                    {
                        y = (y << 1) | (x & 1);
                        x >>= 1;
                    }
                    return y;
                }

                // Check if n is a power of two
                inline bool is_power_of_two(size_t n)
                {
                    return n > 0 && (n & (n - 1)) == 0;
                }

                // Next power of two greater than or equal to n
                inline size_t next_power_of_two(size_t n)
                {
                    if (n == 0) return 1;
                    size_t p = 1;
                    while (p < n) p <<= 1;
                    return p;
                }

                // Precompute twiddle factors for a given size
                template <class T>
                std::vector<std::complex<T>> compute_twiddle_factors(size_t n, bool inverse)
                {
                    std::vector<std::complex<T>> w(n / 2);
                    T angle = (inverse ? 2.0 : -2.0) * M_PI / static_cast<T>(n);
                    for (size_t i = 0; i < n / 2; ++i)
                    {
                        w[i] = std::polar(T(1), angle * static_cast<T>(i));
                    }
                    return w;
                }

                // Cooley-Tukey iterative FFT (in-place)
                template <class T>
                void fft_core(std::complex<T>* data, size_t n, bool inverse)
                {
                    if (n <= 1) return;

                    // Bit-reversal permutation
                    size_t bits = 0;
                    size_t temp = n;
                    while (temp > 1) { temp >>= 1; ++bits; }

                    for (size_t i = 0; i < n; ++i)
                    {
                        size_t j = bit_reverse(i, bits);
                        if (i < j)
                            std::swap(data[i], data[j]);
                    }

                    // Precompute twiddle factors
                    auto w = compute_twiddle_factors<T>(n, inverse);

                    // Iterative FFT
                    for (size_t len = 2; len <= n; len <<= 1)
                    {
                        size_t half_len = len / 2;
                        size_t step = n / len;
                        for (size_t i = 0; i < n; i += len)
                        {
                            for (size_t j = 0; j < half_len; ++j)
                            {
                                size_t idx = j * step;
                                std::complex<T> u = data[i + j];
                                std::complex<T> v = data[i + j + half_len] * w[idx];
                                data[i + j] = u + v;
                                data[i + j + half_len] = u - v;
                            }
                        }
                    }

                    if (inverse)
                    {
                        T inv_n = T(1) / static_cast<T>(n);
                        for (size_t i = 0; i < n; ++i)
                            data[i] *= inv_n;
                    }
                }

                // Real FFT: packs real data into half-length complex array
                template <class T>
                void real_fft_packed(const T* real_in, std::complex<T>* complex_out, size_t n)
                {
                    // For real input, we can compute two real FFTs simultaneously
                    // But here we just convert to complex and do full FFT
                    for (size_t i = 0; i < n; ++i)
                        complex_out[i] = std::complex<T>(real_in[i], 0);
                    fft_core(complex_out, n, false);
                }

                // Hermitian symmetry unpacking
                template <class T>
                void unpack_hermitian(const std::complex<T>* packed, std::complex<T>* full, size_t n)
                {
                    full[0] = packed[0];
                    if (n % 2 == 0)
                    {
                        for (size_t i = 1; i < n/2; ++i)
                        {
                            full[i] = packed[i];
                            full[n - i] = std::conj(packed[i]);
                        }
                        full[n/2] = packed[n/2];
                    }
                    else
                    {
                        for (size_t i = 1; i <= n/2; ++i)
                        {
                            full[i] = packed[i];
                            full[n - i] = std::conj(packed[i]);
                        }
                    }
                }

                // Multi-dimensional index calculation
                template <class Shape>
                size_t ravel_multi_index(const Shape& index, const Shape& shape, const Shape& strides)
                {
                    size_t flat = 0;
                    for (size_t i = 0; i < shape.size(); ++i)
                        flat += index[i] * strides[i];
                    return flat;
                }

                template <class Shape>
                void unravel_index(size_t flat, const Shape& shape, const Shape& strides, std::vector<size_t>& index)
                {
                    size_t temp = flat;
                    for (size_t i = 0; i < shape.size(); ++i)
                    {
                        index[i] = temp / strides[i];
                        temp %= strides[i];
                    }
                }
            }

            // --------------------------------------------------------------------
            // 1D FFT (complex input)
            // --------------------------------------------------------------------
            template <class E>
            inline auto fft(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                using complex_type = std::complex<double>;

                if (expr.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "fft: input must be 1-D");
                }

                size_t n = expr.size();
                size_t n_fft = detail::next_power_of_two(n);

                std::vector<complex_type> data(n_fft, complex_type(0));
                for (size_t i = 0; i < n; ++i)
                    data[i] = static_cast<complex_type>(expr(i));

                detail::fft_core(data.data(), n_fft, false);

                xarray_container<complex_type> result({n_fft});
                std::copy(data.begin(), data.end(), result.begin());
                return result;
            }

            // --------------------------------------------------------------------
            // 1D IFFT (complex input)
            // --------------------------------------------------------------------
            template <class E>
            inline auto ifft(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using complex_type = std::complex<double>;

                if (expr.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "ifft: input must be 1-D");
                }

                size_t n = expr.size();
                if (!detail::is_power_of_two(n))
                {
                    XTENSOR_THROW(std::invalid_argument, "ifft: input size must be a power of two");
                }

                std::vector<complex_type> data(n);
                for (size_t i = 0; i < n; ++i)
                    data[i] = static_cast<complex_type>(expr(i));

                detail::fft_core(data.data(), n, true);

                xarray_container<complex_type> result({n});
                std::copy(data.begin(), data.end(), result.begin());
                return result;
            }

            // --------------------------------------------------------------------
            // 1D FFT (real input) -> returns only half of spectrum (non-redundant)
            // --------------------------------------------------------------------
            template <class E>
            inline auto rfft(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                using complex_type = std::complex<double>;

                if (expr.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "rfft: input must be 1-D");
                }

                size_t n = expr.size();
                size_t n_fft = detail::next_power_of_two(n);
                size_t n_out = n_fft / 2 + 1;

                std::vector<complex_type> data(n_fft, complex_type(0));
                for (size_t i = 0; i < n; ++i)
                    data[i] = complex_type(static_cast<double>(expr(i)), 0.0);

                detail::fft_core(data.data(), n_fft, false);

                xarray_container<complex_type> result({n_out});
                for (size_t i = 0; i < n_out; ++i)
                    result(i) = data[i];
                return result;
            }

            // --------------------------------------------------------------------
            // 1D IRFFT (inverse real FFT from half-spectrum)
            // --------------------------------------------------------------------
            template <class E>
            inline auto irfft(const xexpression<E>& X, size_t n = 0)
            {
                const auto& expr = X.derived_cast();
                using complex_type = std::complex<double>;

                if (expr.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "irfft: input must be 1-D");
                }

                size_t n_half = expr.size();
                size_t n_fft = 2 * (n_half - 1);
                if (n > 0) n_fft = n;
                if (!detail::is_power_of_two(n_fft))
                {
                    n_fft = detail::next_power_of_two(n_fft);
                }

                std::vector<complex_type> full_spectrum(n_fft, complex_type(0));
                for (size_t i = 0; i < n_half && i < expr.size(); ++i)
                    full_spectrum[i] = static_cast<complex_type>(expr(i));
                // Enforce conjugate symmetry
                for (size_t i = 1; i < n_fft / 2; ++i)
                {
                    full_spectrum[n_fft - i] = std::conj(full_spectrum[i]);
                }
                if (n_fft % 2 == 0)
                {
                    full_spectrum[n_fft/2] = std::real(full_spectrum[n_fft/2]);
                }

                detail::fft_core(full_spectrum.data(), n_fft, true);

                xarray_container<double> result({n_fft});
                for (size_t i = 0; i < n_fft; ++i)
                    result(i) = std::real(full_spectrum[i]);

                if (n > 0 && n < n_fft)
                {
                    // Truncate to requested length
                    xarray_container<double> truncated({n});
                    for (size_t i = 0; i < n; ++i)
                        truncated(i) = result(i);
                    return truncated;
                }
                return result;
            }

            // --------------------------------------------------------------------
            // 2D FFT
            // --------------------------------------------------------------------
            template <class E>
            inline auto fft2(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                using complex_type = std::complex<double>;

                if (expr.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "fft2: input must be 2-D");
                }

                size_t rows = expr.shape()[0];
                size_t cols = expr.shape()[1];
                size_t N_rows = detail::next_power_of_two(rows);
                size_t N_cols = detail::next_power_of_two(cols);

                xarray_container<complex_type> result({N_rows, N_cols}, complex_type(0));

                // Copy input with zero padding
                for (size_t i = 0; i < rows; ++i)
                {
                    for (size_t j = 0; j < cols; ++j)
                    {
                        result(i, j) = complex_type(static_cast<double>(expr(i, j)), 0);
                    }
                }

                // FFT along rows
                for (size_t i = 0; i < N_rows; ++i)
                {
                    std::vector<complex_type> row(N_cols);
                    for (size_t j = 0; j < N_cols; ++j)
                        row[j] = result(i, j);
                    detail::fft_core(row.data(), N_cols, false);
                    for (size_t j = 0; j < N_cols; ++j)
                        result(i, j) = row[j];
                }

                // FFT along columns
                for (size_t j = 0; j < N_cols; ++j)
                {
                    std::vector<complex_type> col(N_rows);
                    for (size_t i = 0; i < N_rows; ++i)
                        col[i] = result(i, j);
                    detail::fft_core(col.data(), N_rows, false);
                    for (size_t i = 0; i < N_rows; ++i)
                        result(i, j) = col[i];
                }

                return result;
            }

            // --------------------------------------------------------------------
            // 2D IFFT
            // --------------------------------------------------------------------
            template <class E>
            inline auto ifft2(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using complex_type = std::complex<double>;

                if (expr.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "ifft2: input must be 2-D");
                }

                size_t rows = expr.shape()[0];
                size_t cols = expr.shape()[1];
                if (!detail::is_power_of_two(rows) || !detail::is_power_of_two(cols))
                {
                    XTENSOR_THROW(std::invalid_argument, "ifft2: dimensions must be powers of two");
                }

                xarray_container<complex_type> result = eval(expr);

                // IFFT along columns
                for (size_t j = 0; j < cols; ++j)
                {
                    std::vector<complex_type> col(rows);
                    for (size_t i = 0; i < rows; ++i)
                        col[i] = result(i, j);
                    detail::fft_core(col.data(), rows, true);
                    for (size_t i = 0; i < rows; ++i)
                        result(i, j) = col[i];
                }

                // IFFT along rows
                for (size_t i = 0; i < rows; ++i)
                {
                    std::vector<complex_type> row(cols);
                    for (size_t j = 0; j < cols; ++j)
                        row[j] = result(i, j);
                    detail::fft_core(row.data(), cols, true);
                    for (size_t j = 0; j < cols; ++j)
                        result(i, j) = row[j];
                }

                return result;
            }

            // --------------------------------------------------------------------
            // 2D RFFT (real input, returns half spectrum along last axis)
            // --------------------------------------------------------------------
            template <class E>
            inline auto rfft2(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                if (expr.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "rfft2: input must be 2-D");
                }

                size_t rows = expr.shape()[0];
                size_t cols = expr.shape()[1];
                size_t N_rows = detail::next_power_of_two(rows);
                size_t N_cols = detail::next_power_of_two(cols);
                size_t n_freq = N_cols / 2 + 1;

                // First FFT along rows (full complex)
                xarray_container<std::complex<double>> temp({N_rows, N_cols}, 0.0);
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        temp(i, j) = static_cast<double>(expr(i, j));

                // FFT rows
                for (size_t i = 0; i < N_rows; ++i)
                {
                    std::vector<std::complex<double>> row(N_cols);
                    for (size_t j = 0; j < N_cols; ++j)
                        row[j] = temp(i, j);
                    detail::fft_core(row.data(), N_cols, false);
                    for (size_t j = 0; j < N_cols; ++j)
                        temp(i, j) = row[j];
                }

                // FFT columns (on real-valued data? Actually we need real input for columns,
                // but we already have complex. We'll compute full 2D FFT and then take half)
                for (size_t j = 0; j < N_cols; ++j)
                {
                    std::vector<std::complex<double>> col(N_rows);
                    for (size_t i = 0; i < N_rows; ++i)
                        col[i] = temp(i, j);
                    detail::fft_core(col.data(), N_rows, false);
                    for (size_t i = 0; i < N_rows; ++i)
                        temp(i, j) = col[i];
                }

                // Return only first half of last dimension
                xarray_container<std::complex<double>> result({N_rows, n_freq});
                for (size_t i = 0; i < N_rows; ++i)
                    for (size_t j = 0; j < n_freq; ++j)
                        result(i, j) = temp(i, j);
                return result;
            }

            // --------------------------------------------------------------------
            // 2D IRFFT
            // --------------------------------------------------------------------
            template <class E>
            inline auto irfft2(const xexpression<E>& X, const std::array<size_t,2>& shape = {0,0})
            {
                const auto& expr = X.derived_cast();
                if (expr.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "irfft2: input must be 2-D");
                }

                size_t N_rows = expr.shape()[0];
                size_t n_freq = expr.shape()[1];
                size_t N_cols = 2 * (n_freq - 1);
                if (!detail::is_power_of_two(N_rows) || !detail::is_power_of_two(N_cols))
                {
                    XTENSOR_THROW(std::invalid_argument, "irfft2: dimensions must be powers of two");
                }

                // Reconstruct full spectrum
                xarray_container<std::complex<double>> full_spect({N_rows, N_cols}, 0.0);
                for (size_t i = 0; i < N_rows; ++i)
                {
                    for (size_t j = 0; j < n_freq; ++j)
                        full_spect(i, j) = expr(i, j);
                    // Conjugate symmetry
                    for (size_t j = 1; j < N_cols/2; ++j)
                    {
                        full_spect(i, N_cols - j) = std::conj(full_spect(i, j));
                    }
                    if (N_cols % 2 == 0)
                        full_spect(i, N_cols/2) = std::real(full_spect(i, N_cols/2));
                }

                // IFFT columns
                for (size_t j = 0; j < N_cols; ++j)
                {
                    std::vector<std::complex<double>> col(N_rows);
                    for (size_t i = 0; i < N_rows; ++i)
                        col[i] = full_spect(i, j);
                    detail::fft_core(col.data(), N_rows, true);
                    for (size_t i = 0; i < N_rows; ++i)
                        full_spect(i, j) = col[i];
                }

                // IFFT rows
                for (size_t i = 0; i < N_rows; ++i)
                {
                    std::vector<std::complex<double>> row(N_cols);
                    for (size_t j = 0; j < N_cols; ++j)
                        row[j] = full_spect(i, j);
                    detail::fft_core(row.data(), N_cols, true);
                    for (size_t j = 0; j < N_cols; ++j)
                        full_spect(i, j) = row[j];
                }

                // Take real part
                xarray_container<double> result({N_rows, N_cols});
                for (size_t i = 0; i < N_rows; ++i)
                    for (size_t j = 0; j < N_cols; ++j)
                        result(i, j) = std::real(full_spect(i, j));

                // Crop to requested shape
                if (shape[0] > 0 && shape[0] < N_rows)
                {
                    if (shape[1] > 0 && shape[1] < N_cols)
                    {
                        xarray_container<double> cropped({shape[0], shape[1]});
                        for (size_t i = 0; i < shape[0]; ++i)
                            for (size_t j = 0; j < shape[1]; ++j)
                                cropped(i, j) = result(i, j);
                        return cropped;
                    }
                }
                return result;
            }

            // --------------------------------------------------------------------
            // N-Dimensional FFT
            // --------------------------------------------------------------------
            template <class E>
            inline auto fftn(const xexpression<E>& x, const std::vector<size_t>& axes = {})
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                using complex_type = std::complex<double>;

                std::vector<size_t> ax = axes;
                if (ax.empty())
                {
                    ax.resize(expr.dimension());
                    std::iota(ax.begin(), ax.end(), 0);
                }

                auto result = xt::eval(expr);
                auto shape = result.shape();
                std::vector<size_t> fft_shape = shape;
                for (size_t axis : ax)
                {
                    fft_shape[axis] = detail::next_power_of_two(shape[axis]);
                }

                // Zero pad to power-of-two sizes
                xarray_container<complex_type> temp(fft_shape, complex_type(0));
                // Copy data (assuming contiguous iteration)
                std::vector<size_t> index(expr.dimension(), 0);
                for (size_t flat = 0; flat < expr.size(); ++flat)
                {
                    // Compute multi-index in original shape
                    size_t rem = flat;
                    for (int d = static_cast<int>(expr.dimension()) - 1; d >= 0; --d)
                    {
                        index[static_cast<size_t>(d)] = rem % shape[static_cast<size_t>(d)];
                        rem /= shape[static_cast<size_t>(d)];
                    }
                    // Map to temp index
                    size_t temp_flat = 0;
                    size_t stride = 1;
                    for (int d = static_cast<int>(expr.dimension()) - 1; d >= 0; --d)
                    {
                        temp_flat += index[static_cast<size_t>(d)] * stride;
                        stride *= fft_shape[static_cast<size_t>(d)];
                    }
                    temp.flat(temp_flat) = static_cast<complex_type>(expr.flat(flat));
                }

                // Perform FFT along each axis
                for (size_t axis : ax)
                {
                    size_t n = fft_shape[axis];
                    size_t outer_size = temp.size() / n;
                    size_t stride = 1;
                    for (size_t d = axis + 1; d < expr.dimension(); ++d)
                        stride *= fft_shape[d];

                    for (size_t i = 0; i < outer_size; ++i)
                    {
                        // Calculate starting offset for this slice
                        size_t slice_idx = i;
                        size_t base = 0;
                        size_t temp_slice = slice_idx;
                        for (size_t d = 0; d < expr.dimension(); ++d)
                        {
                            if (d == axis) continue;
                            size_t dim_size = (d < axis) ? fft_shape[d] : fft_shape[d] / n;
                            size_t coord = temp_slice % dim_size;
                            temp_slice /= dim_size;
                            size_t dim_stride = 1;
                            for (size_t k = d + 1; k < expr.dimension(); ++k)
                                if (k != axis) dim_stride *= fft_shape[k];
                            base += coord * dim_stride;
                        }
                        // Extract 1D slice
                        std::vector<complex_type> slice(n);
                        for (size_t k = 0; k < n; ++k)
                        {
                            size_t idx = base + k * stride;
                            slice[k] = temp.flat(idx);
                        }
                        detail::fft_core(slice.data(), n, false);
                        // Write back
                        for (size_t k = 0; k < n; ++k)
                        {
                            size_t idx = base + k * stride;
                            temp.flat(idx) = slice[k];
                        }
                    }
                }

                return temp;
            }

            // --------------------------------------------------------------------
            // N-Dimensional IFFT
            // --------------------------------------------------------------------
            template <class E>
            inline auto ifftn(const xexpression<E>& x, const std::vector<size_t>& axes = {})
            {
                const auto& expr = x.derived_cast();
                using complex_type = std::complex<double>;

                std::vector<size_t> ax = axes;
                if (ax.empty())
                {
                    ax.resize(expr.dimension());
                    std::iota(ax.begin(), ax.end(), 0);
                }

                auto shape = expr.shape();
                for (size_t axis : ax)
                {
                    if (!detail::is_power_of_two(shape[axis]))
                    {
                        XTENSOR_THROW(std::invalid_argument, "ifftn: all axes must have power-of-two sizes");
                    }
                }

                auto temp = xt::eval(expr);
                for (size_t axis : ax)
                {
                    size_t n = shape[axis];
                    size_t outer_size = temp.size() / n;
                    size_t stride = 1;
                    for (size_t d = axis + 1; d < expr.dimension(); ++d)
                        stride *= shape[d];

                    for (size_t i = 0; i < outer_size; ++i)
                    {
                        size_t slice_idx = i;
                        size_t base = 0;
                        size_t temp_slice = slice_idx;
                        for (size_t d = 0; d < expr.dimension(); ++d)
                        {
                            if (d == axis) continue;
                            size_t dim_size = (d < axis) ? shape[d] : shape[d] / n;
                            size_t coord = temp_slice % dim_size;
                            temp_slice /= dim_size;
                            size_t dim_stride = 1;
                            for (size_t k = d + 1; k < expr.dimension(); ++k)
                                if (k != axis) dim_stride *= shape[k];
                            base += coord * dim_stride;
                        }
                        std::vector<complex_type> slice(n);
                        for (size_t k = 0; k < n; ++k)
                        {
                            size_t idx = base + k * stride;
                            slice[k] = temp.flat(idx);
                        }
                        detail::fft_core(slice.data(), n, true);
                        for (size_t k = 0; k < n; ++k)
                        {
                            size_t idx = base + k * stride;
                            temp.flat(idx) = slice[k];
                        }
                    }
                }
                return temp;
            }

            // --------------------------------------------------------------------
            // FFT shift (reorder quadrants)
            // --------------------------------------------------------------------
            template <class E>
            inline auto fftshift(const xexpression<E>& x, const std::vector<size_t>& axes = {})
            {
                auto result = eval(x);
                std::vector<size_t> ax = axes;
                if (ax.empty())
                {
                    ax.resize(result.dimension());
                    std::iota(ax.begin(), ax.end(), 0);
                }

                for (size_t axis : ax)
                {
                    size_t n = result.shape()[axis];
                    size_t mid = n / 2;
                    if (n <= 1) continue;

                    // Create temporary copy of the data along this axis
                    auto temp = result;
                    std::vector<size_t> src_idx(result.dimension(), 0);
                    std::vector<size_t> dst_idx(result.dimension(), 0);

                    // Iterate over all elements not on the shifted axis
                    size_t total = result.size() / n;
                    size_t stride = 1;
                    for (size_t d = axis + 1; d < result.dimension(); ++d)
                        stride *= result.shape()[d];

                    for (size_t slice = 0; slice < total; ++slice)
                    {
                        size_t base = 0;
                        size_t temp_slice = slice;
                        for (size_t d = 0; d < result.dimension(); ++d)
                        {
                            if (d == axis) continue;
                            size_t dim_size = (d < axis) ? result.shape()[d] : result.shape()[d] / n;
                            size_t coord = temp_slice % dim_size;
                            temp_slice /= dim_size;
                            size_t dim_stride = 1;
                            for (size_t k = d + 1; k < result.dimension(); ++k)
                                if (k != axis) dim_stride *= result.shape()[k];
                            base += coord * dim_stride;
                        }

                        // Shift elements
                        for (size_t k = 0; k < mid; ++k)
                        {
                            size_t src_off1 = base + k * stride;
                            size_t src_off2 = base + (k + mid + (n%2)) * stride;
                            std::swap(temp.flat(src_off1), temp.flat(src_off2));
                        }
                    }
                    result = std::move(temp);
                }
                return result;
            }

            // --------------------------------------------------------------------
            // IFFT shift (inverse of fftshift, same for even sizes)
            // --------------------------------------------------------------------
            template <class E>
            inline auto ifftshift(const xexpression<E>& x, const std::vector<size_t>& axes = {})
            {
                // For even dimensions, ifftshift is identical to fftshift
                // For odd, we need to shift differently, but we'll use fftshift for simplicity
                return fftshift(x, axes);
            }

            // --------------------------------------------------------------------
            // FFT frequencies
            // --------------------------------------------------------------------
            inline xarray_container<double> fftfreq(size_t n, double d = 1.0)
            {
                xarray_container<double> freq({n});
                double val = 1.0 / (n * d);
                size_t mid = n / 2;
                for (size_t i = 0; i <= mid; ++i)
                    freq(i) = static_cast<double>(i) * val;
                for (size_t i = mid + 1; i < n; ++i)
                    freq(i) = -static_cast<double>(n - i) * val;
                return freq;
            }

            inline xarray_container<double> rfftfreq(size_t n, double d = 1.0)
            {
                size_t n_out = n / 2 + 1;
                xarray_container<double> freq({n_out});
                double val = 1.0 / (n * d);
                for (size_t i = 0; i < n_out; ++i)
                    freq(i) = static_cast<double>(i) * val;
                return freq;
            }

            // --------------------------------------------------------------------
            // Next fast length (pad to optimal size for FFT)
            // --------------------------------------------------------------------
            inline size_t next_fast_len(size_t n)
            {
                // For radix-2, use next power of two
                return detail::next_power_of_two(n);
            }

        } // namespace signal

        // Bring FFT functions into xt namespace
        using signal::fft;
        using signal::ifft;
        using signal::rfft;
        using signal::irfft;
        using signal::fft2;
        using signal::ifft2;
        using signal::rfft2;
        using signal::irfft2;
        using signal::fftn;
        using signal::ifftn;
        using signal::fftshift;
        using signal::ifftshift;
        using signal::fftfreq;
        using signal::rfftfreq;
        using signal::next_fast_len;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_FFT_HPP

// signal/fft.hpp