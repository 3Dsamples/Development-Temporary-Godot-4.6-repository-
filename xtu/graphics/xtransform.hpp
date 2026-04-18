// math/xtransform.hpp

#ifndef XTENSOR_XTRANSFORM_HPP
#define XTENSOR_XTRANSFORM_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xfunction.hpp"
#include "../core/xview.hpp"
#include "xsorting.hpp"
#include "xlinalg.hpp"
#include "xstats.hpp"
#include "xmissing.hpp"

#include <cmath>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstring>
#include <cassert>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace transform
        {
            // --------------------------------------------------------------------
            // Complex number utilities
            // --------------------------------------------------------------------
            using complex64 = std::complex<float>;
            using complex128 = std::complex<double>;
            using complex256 = std::complex<long double>;
            
            template <class T>
            using complex_t = std::complex<T>;
            
            template <class T>
            inline T real(const std::complex<T>& z) { return std::real(z); }
            
            template <class T>
            inline T imag(const std::complex<T>& z) { return std::imag(z); }
            
            template <class T>
            inline std::complex<T> conj(const std::complex<T>& z) { return std::conj(z); }
            
            // --------------------------------------------------------------------
            // Discrete Fourier Transform (DFT) - naive O(N^2) implementation
            // --------------------------------------------------------------------
            template <class E>
            inline auto dft(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                using complex_type = std::complex<double>;
                
                if (expr.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "dft: input must be 1-D");
                }
                
                size_t N = expr.size();
                xarray_container<complex_type> result({N});
                
                for (size_t k = 0; k < N; ++k)
                {
                    complex_type sum = 0;
                    for (size_t n = 0; n < N; ++n)
                    {
                        double angle = -2.0 * M_PI * k * n / N;
                        complex_type w(std::cos(angle), std::sin(angle));
                        sum += static_cast<complex_type>(expr(n)) * w;
                    }
                    result(k) = sum;
                }
                return result;
            }
            
            // Inverse DFT
            template <class E>
            inline auto idft(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                using complex_type = std::complex<double>;
                
                if (expr.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "idft: input must be 1-D");
                }
                
                size_t N = expr.size();
                xarray_container<complex_type> result({N});
                
                for (size_t k = 0; k < N; ++k)
                {
                    complex_type sum = 0;
                    for (size_t n = 0; n < N; ++n)
                    {
                        double angle = 2.0 * M_PI * k * n / N;
                        complex_type w(std::cos(angle), std::sin(angle));
                        sum += static_cast<complex_type>(expr(n)) * w;
                    }
                    result(k) = sum / static_cast<double>(N);
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Fast Fourier Transform (FFT) - Cooley-Tukey iterative radix-2
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
                
                // Check if N is power of two
                inline bool is_power_of_two(size_t n)
                {
                    return n > 0 && (n & (n - 1)) == 0;
                }
                
                // Next power of two
                inline size_t next_power_of_two(size_t n)
                {
                    size_t p = 1;
                    while (p < n) p <<= 1;
                    return p;
                }
                
                // FFT core (in-place, complex)
                template <class T>
                void fft_core(std::complex<T>* data, size_t N, bool inverse)
                {
                    if (!is_power_of_two(N))
                    {
                        throw std::runtime_error("fft_core: size must be power of two");
                    }
                    
                    // Bit-reversal permutation
                    size_t bits = 0;
                    size_t temp = N;
                    while (temp > 1) { temp >>= 1; ++bits; }
                    
                    for (size_t i = 0; i < N; ++i)
                    {
                        size_t j = bit_reverse(i, bits);
                        if (i < j)
                            std::swap(data[i], data[j]);
                    }
                    
                    // Cooley-Tukey iterative FFT
                    for (size_t len = 2; len <= N; len <<= 1)
                    {
                        double angle = 2.0 * M_PI / len;
                        if (inverse) angle = -angle;
                        std::complex<T> wlen(std::cos(angle), std::sin(angle));
                        
                        for (size_t i = 0; i < N; i += len)
                        {
                            std::complex<T> w(1);
                            for (size_t j = 0; j < len/2; ++j)
                            {
                                std::complex<T> u = data[i + j];
                                std::complex<T> v = data[i + j + len/2] * w;
                                data[i + j] = u + v;
                                data[i + j + len/2] = u - v;
                                w *= wlen;
                            }
                        }
                    }
                    
                    if (inverse)
                    {
                        for (size_t i = 0; i < N; ++i)
                            data[i] /= static_cast<T>(N);
                    }
                }
            }
            
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
                
                size_t N = expr.size();
                size_t N2 = detail::next_power_of_two(N);
                
                std::vector<complex_type> data(N2, complex_type(0));
                for (size_t i = 0; i < N; ++i)
                    data[i] = static_cast<complex_type>(expr(i));
                
                detail::fft_core(data.data(), N2, false);
                
                xarray_container<complex_type> result({N2});
                for (size_t i = 0; i < N2; ++i)
                    result(i) = data[i];
                return result;
            }
            
            template <class E>
            inline auto ifft(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using complex_type = std::complex<double>;
                
                if (expr.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "ifft: input must be 1-D");
                }
                
                size_t N = expr.size();
                if (!detail::is_power_of_two(N))
                {
                    XTENSOR_THROW(std::invalid_argument, "ifft: size must be power of two for inverse FFT (use idft otherwise)");
                }
                
                std::vector<complex_type> data(N);
                for (size_t i = 0; i < N; ++i)
                    data[i] = static_cast<complex_type>(expr(i));
                
                detail::fft_core(data.data(), N, true);
                
                xarray_container<complex_type> result({N});
                for (size_t i = 0; i < N; ++i)
                    result(i) = data[i];
                return result;
            }
            
            // Real FFT (returns only non-redundant half)
            template <class E>
            inline auto rfft(const xexpression<E>& x)
            {
                auto full = fft(x);
                size_t N = full.size();
                size_t half = N / 2 + 1;
                xarray_container<std::complex<double>> result({half});
                for (size_t i = 0; i < half; ++i)
                    result(i) = full(i);
                return result;
            }
            
            // Inverse real FFT
            template <class E>
            inline auto irfft(const xexpression<E>& X, size_t n = 0)
            {
                const auto& expr = X.derived_cast();
                size_t half = expr.size();
                size_t N = 2 * (half - 1);
                if (n > 0) N = n;
                if (!detail::is_power_of_two(N))
                {
                    XTENSOR_THROW(std::invalid_argument, "irfft: size must be power of two");
                }
                
                xarray_container<std::complex<double>> full({N}, 0.0);
                for (size_t i = 0; i < half; ++i)
                    full(i) = expr(i);
                // Conjugate symmetry for the rest
                for (size_t i = 1; i < N/2; ++i)
                    full(N - i) = std::conj(full(i));
                
                auto result_complex = ifft(full);
                // Take real part
                xarray_container<double> result({N});
                for (size_t i = 0; i < N; ++i)
                    result(i) = std::real(result_complex(i));
                return result;
            }
            
            // --------------------------------------------------------------------
            // 2D FFT
            // --------------------------------------------------------------------
            template <class E>
            inline auto fft2(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                if (expr.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "fft2: input must be 2-D");
                
                size_t rows = expr.shape()[0];
                size_t cols = expr.shape()[1];
                size_t N_rows = detail::next_power_of_two(rows);
                size_t N_cols = detail::next_power_of_two(cols);
                
                xarray_container<std::complex<double>> result({N_rows, N_cols}, 0.0);
                
                // FFT along rows
                for (size_t i = 0; i < N_rows; ++i)
                {
                    std::vector<std::complex<double>> row(N_cols, 0.0);
                    for (size_t j = 0; j < cols && i < rows; ++j)
                        row[j] = static_cast<std::complex<double>>(expr(i, j));
                    detail::fft_core(row.data(), N_cols, false);
                    for (size_t j = 0; j < N_cols; ++j)
                        result(i, j) = row[j];
                }
                
                // FFT along columns
                for (size_t j = 0; j < N_cols; ++j)
                {
                    std::vector<std::complex<double>> col(N_rows, 0.0);
                    for (size_t i = 0; i < N_rows; ++i)
                        col[i] = result(i, j);
                    detail::fft_core(col.data(), N_rows, false);
                    for (size_t i = 0; i < N_rows; ++i)
                        result(i, j) = col[i];
                }
                
                return result;
            }
            
            template <class E>
            inline auto ifft2(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                if (expr.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "ifft2: input must be 2-D");
                
                size_t rows = expr.shape()[0];
                size_t cols = expr.shape()[1];
                if (!detail::is_power_of_two(rows) || !detail::is_power_of_two(cols))
                    XTENSOR_THROW(std::invalid_argument, "ifft2: dimensions must be powers of two");
                
                xarray_container<std::complex<double>> result = eval(expr);
                
                // IFFT along columns
                for (size_t j = 0; j < cols; ++j)
                {
                    std::vector<std::complex<double>> col(rows);
                    for (size_t i = 0; i < rows; ++i)
                        col[i] = result(i, j);
                    detail::fft_core(col.data(), rows, true);
                    for (size_t i = 0; i < rows; ++i)
                        result(i, j) = col[i];
                }
                
                // IFFT along rows
                for (size_t i = 0; i < rows; ++i)
                {
                    std::vector<std::complex<double>> row(cols);
                    for (size_t j = 0; j < cols; ++j)
                        row[j] = result(i, j);
                    detail::fft_core(row.data(), cols, true);
                    for (size_t j = 0; j < cols; ++j)
                        result(i, j) = row[j];
                }
                
                return result;
            }
            
            // --------------------------------------------------------------------
            // FFT shift (swap quadrants)
            // --------------------------------------------------------------------
            template <class E>
            inline auto fftshift(const xexpression<E>& x)
            {
                auto result = eval(x);
                size_t dim = result.dimension();
                if (dim == 1)
                {
                    size_t N = result.size();
                    size_t mid = N / 2;
                    for (size_t i = 0; i < mid; ++i)
                        std::swap(result(i), result(i + mid + (N%2)));
                }
                else if (dim == 2)
                {
                    size_t rows = result.shape()[0];
                    size_t cols = result.shape()[1];
                    size_t rmid = rows / 2;
                    size_t cmid = cols / 2;
                    // Swap quadrants
                    for (size_t i = 0; i < rmid; ++i)
                    {
                        for (size_t j = 0; j < cmid; ++j)
                        {
                            std::swap(result(i, j), result(i + rmid + (rows%2), j + cmid + (cols%2)));
                            std::swap(result(i, j + cmid + (cols%2)), result(i + rmid + (rows%2), j));
                        }
                    }
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "fftshift: only 1D and 2D supported");
                }
                return result;
            }
            
            template <class E>
            inline auto ifftshift(const xexpression<E>& x)
            {
                // Same as fftshift for even dimensions
                return fftshift(x);
            }
            
            // --------------------------------------------------------------------
            // Discrete Cosine Transform (DCT) Type II (The most common)
            // --------------------------------------------------------------------
            template <class E>
            inline auto dct(const xexpression<E>& x, int type = 2)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                size_t N = expr.size();
                
                if (type == 2) // DCT-II (used in JPEG)
                {
                    xarray_container<double> result({N});
                    double factor = std::sqrt(2.0 / N);
                    for (size_t k = 0; k < N; ++k)
                    {
                        double sum = 0.0;
                        for (size_t n = 0; n < N; ++n)
                        {
                            sum += static_cast<double>(expr(n)) * 
                                   std::cos(M_PI * k * (2*n + 1) / (2.0 * N));
                        }
                        result(k) = factor * sum;
                        if (k == 0) result(k) /= std::sqrt(2.0);
                    }
                    return result;
                }
                else if (type == 3) // DCT-III (inverse of DCT-II)
                {
                    xarray_container<double> result({N});
                    double factor = std::sqrt(2.0 / N);
                    for (size_t k = 0; k < N; ++k)
                    {
                        double sum = static_cast<double>(expr(0)) / std::sqrt(2.0);
                        for (size_t n = 1; n < N; ++n)
                        {
                            sum += static_cast<double>(expr(n)) * 
                                   std::cos(M_PI * n * (2*k + 1) / (2.0 * N));
                        }
                        result(k) = factor * sum;
                    }
                    return result;
                }
                else if (type == 1) // DCT-I
                {
                    xarray_container<double> result({N});
                    double factor = std::sqrt(2.0 / (N - 1));
                    for (size_t k = 0; k < N; ++k)
                    {
                        double sum = 0.5 * (static_cast<double>(expr(0)) + 
                                            static_cast<double>(expr(N-1)) * ((k%2==0)?1:-1));
                        for (size_t n = 1; n < N-1; ++n)
                            sum += static_cast<double>(expr(n)) * std::cos(M_PI * k * n / (N - 1));
                        result(k) = factor * sum;
                    }
                    return result;
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "dct: unsupported type (use 1,2,3)");
                }
                return xarray_container<double>();
            }
            
            template <class E>
            inline auto idct(const xexpression<E>& x, int type = 2)
            {
                // Inverse DCT-II is DCT-III, etc.
                if (type == 2) return dct(x, 3);
                else if (type == 3) return dct(x, 2);
                else return dct(x, type); // DCT-I is self-inverse up to scale
            }
            
            // --------------------------------------------------------------------
            // Discrete Sine Transform (DST) Type II
            // --------------------------------------------------------------------
            template <class E>
            inline auto dst(const xexpression<E>& x, int type = 2)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                size_t N = expr.size();
                
                if (type == 2)
                {
                    xarray_container<double> result({N});
                    double factor = std::sqrt(2.0 / N);
                    for (size_t k = 0; k < N; ++k)
                    {
                        double sum = 0.0;
                        for (size_t n = 0; n < N; ++n)
                        {
                            sum += static_cast<double>(expr(n)) * 
                                   std::sin(M_PI * (k+1) * (2*n + 1) / (2.0 * N));
                        }
                        result(k) = factor * sum;
                    }
                    return result;
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "dst: unsupported type (use 2)");
                }
                return xarray_container<double>();
            }
            
            template <class E>
            inline auto idst(const xexpression<E>& x, int type = 2)
            {
                // DST is self-inverse up to scale factor
                return dst(x, type);
            }
            
            // --------------------------------------------------------------------
            // Hilbert Transform (via FFT)
            // --------------------------------------------------------------------
            template <class E>
            inline auto hilbert(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                size_t N = expr.size();
                size_t Nfft = detail::next_power_of_two(N);
                
                // FFT of real signal
                xarray_container<std::complex<double>> X({Nfft}, 0.0);
                for (size_t i = 0; i < N; ++i)
                    X(i) = static_cast<complex128>(expr(i));
                auto Xf = fft(X);
                
                // Multiply by -j * sign(w)
                for (size_t i = 0; i < Nfft; ++i)
                {
                    if (i == 0 || i == Nfft/2)
                        Xf(i) = 0;
                    else if (i < Nfft/2)
                        Xf(i) *= complex128(0, -1);
                    else
                        Xf(i) *= complex128(0, 1);
                }
                
                // Inverse FFT, take imaginary part
                auto xh = ifft(Xf);
                xarray_container<double> result({N});
                for (size_t i = 0; i < N; ++i)
                    result(i) = std::imag(xh(i));
                return result;
            }
            
            // Analytic signal
            template <class E>
            inline auto analytic_signal(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                size_t N = expr.size();
                size_t Nfft = detail::next_power_of_two(N);
                
                xarray_container<std::complex<double>> X({Nfft}, 0.0);
                for (size_t i = 0; i < N; ++i)
                    X(i) = static_cast<complex128>(expr(i));
                auto Xf = fft(X);
                
                // Zero out negative frequencies
                for (size_t i = Nfft/2 + 1; i < Nfft; ++i)
                    Xf(i) = 0;
                // Double positive frequencies except DC and Nyquist
                for (size_t i = 1; i < Nfft/2; ++i)
                    Xf(i) *= 2.0;
                
                auto xa = ifft(Xf);
                xarray_container<std::complex<double>> result({N});
                for (size_t i = 0; i < N; ++i)
                    result(i) = xa(i);
                return result;
            }
            
            // --------------------------------------------------------------------
            // Wavelet Transform (Discrete Haar Wavelet)
            // --------------------------------------------------------------------
            template <class E>
            inline auto haar_wavelet(const xexpression<E>& x, int level = 1)
            {
                const auto& expr = x.derived_cast();
                using value_type = typename E::value_type;
                size_t N = expr.size();
                if (!detail::is_power_of_two(N))
                    XTENSOR_THROW(std::invalid_argument, "haar_wavelet: size must be power of two");
                
                auto coeffs = eval(expr);
                size_t current_len = N;
                
                for (int l = 0; l < level && current_len > 1; ++l)
                {
                    std::vector<value_type> temp(current_len);
                    for (size_t i = 0; i < current_len/2; ++i)
                    {
                        value_type avg = (coeffs(2*i) + coeffs(2*i+1)) / std::sqrt(2.0);
                        value_type diff = (coeffs(2*i) - coeffs(2*i+1)) / std::sqrt(2.0);
                        temp[i] = avg;
                        temp[current_len/2 + i] = diff;
                    }
                    for (size_t i = 0; i < current_len; ++i)
                        coeffs(i) = temp[i];
                    current_len /= 2;
                }
                return coeffs;
            }
            
            template <class E>
            inline auto inverse_haar_wavelet(const xexpression<E>& coeffs, int level = 1)
            {
                const auto& expr = coeffs.derived_cast();
                using value_type = typename E::value_type;
                size_t N = expr.size();
                auto result = eval(expr);
                size_t current_len = N >> level;
                
                for (int l = 0; l < level; ++l)
                {
                    std::vector<value_type> temp(current_len * 2);
                    for (size_t i = 0; i < current_len; ++i)
                    {
                        value_type avg = result(i);
                        value_type diff = result(current_len + i);
                        temp[2*i] = (avg + diff) / std::sqrt(2.0);
                        temp[2*i+1] = (avg - diff) / std::sqrt(2.0);
                    }
                    for (size_t i = 0; i < 2*current_len; ++i)
                        result(i) = temp[i];
                    current_len *= 2;
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Convolution using FFT (overlap-add method)
            // --------------------------------------------------------------------
            template <class E1, class E2>
            inline auto fftconvolve(const xexpression<E1>& a, const xexpression<E2>& b, const std::string& mode = "full")
            {
                const auto& A = a.derived_cast();
                const auto& B = b.derived_cast();
                if (A.dimension() != 1 || B.dimension() != 1)
                    XTENSOR_THROW(std::invalid_argument, "fftconvolve: inputs must be 1-D");
                
                size_t Na = A.size();
                size_t Nb = B.size();
                size_t N = Na + Nb - 1;
                size_t Nfft = detail::next_power_of_two(N);
                
                std::vector<complex128> Af(Nfft, 0.0), Bf(Nfft, 0.0);
                for (size_t i = 0; i < Na; ++i) Af[i] = static_cast<complex128>(A(i));
                for (size_t i = 0; i < Nb; ++i) Bf[i] = static_cast<complex128>(B(i));
                
                detail::fft_core(Af.data(), Nfft, false);
                detail::fft_core(Bf.data(), Nfft, false);
                
                for (size_t i = 0; i < Nfft; ++i)
                    Af[i] *= Bf[i];
                
                detail::fft_core(Af.data(), Nfft, true);
                
                xarray_container<double> result_full({N});
                for (size_t i = 0; i < N; ++i)
                    result_full(i) = std::real(Af[i]);
                
                if (mode == "full")
                    return result_full;
                else if (mode == "same")
                {
                    size_t start = Nb / 2;
                    xarray_container<double> result_same({Na});
                    for (size_t i = 0; i < Na; ++i)
                        result_same(i) = (i + start < N) ? result_full(i + start) : 0.0;
                    return result_same;
                }
                else if (mode == "valid")
                {
                    if (Na < Nb) return xarray_container<double>();
                    size_t valid_len = Na - Nb + 1;
                    xarray_container<double> result_valid({valid_len});
                    for (size_t i = 0; i < valid_len; ++i)
                        result_valid(i) = result_full(i + Nb - 1);
                    return result_valid;
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "fftconvolve: unknown mode (full, same, valid)");
                }
                return result_full;
            }
            
            // --------------------------------------------------------------------
            // Short-Time Fourier Transform (STFT)
            // --------------------------------------------------------------------
            template <class E>
            inline auto stft(const xexpression<E>& x, size_t nperseg = 256, size_t noverlap = 0,
                             const std::string& window_type = "hann")
            {
                const auto& signal = x.derived_cast();
                if (signal.dimension() != 1)
                    XTENSOR_THROW(std::invalid_argument, "stft: input must be 1-D");
                
                if (noverlap >= nperseg)
                    XTENSOR_THROW(std::invalid_argument, "stft: noverlap must be less than nperseg");
                
                size_t N = signal.size();
                size_t hop = nperseg - noverlap;
                size_t n_frames = (N - nperseg) / hop + 1;
                if (n_frames == 0) n_frames = 1;
                
                // Create window
                xarray_container<double> window({nperseg});
                if (window_type == "hann")
                {
                    for (size_t i = 0; i < nperseg; ++i)
                        window(i) = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (nperseg - 1)));
                }
                else if (window_type == "hamming")
                {
                    for (size_t i = 0; i < nperseg; ++i)
                        window(i) = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (nperseg - 1));
                }
                else if (window_type == "blackman")
                {
                    for (size_t i = 0; i < nperseg; ++i)
                        window(i) = 0.42 - 0.5 * std::cos(2.0 * M_PI * i / (nperseg - 1)) + 
                                    0.08 * std::cos(4.0 * M_PI * i / (nperseg - 1));
                }
                else // rectangular
                {
                    for (size_t i = 0; i < nperseg; ++i)
                        window(i) = 1.0;
                }
                
                size_t n_fft = detail::next_power_of_two(nperseg);
                size_t n_freq = n_fft / 2 + 1;
                
                xarray_container<complex128> result({n_freq, n_frames});
                
                for (size_t frame = 0; frame < n_frames; ++frame)
                {
                    size_t start = frame * hop;
                    std::vector<complex128> segment(n_fft, 0.0);
                    for (size_t i = 0; i < nperseg && start + i < N; ++i)
                        segment[i] = static_cast<complex128>(signal(start + i)) * window(i);
                    
                    detail::fft_core(segment.data(), n_fft, false);
                    for (size_t f = 0; f < n_freq; ++f)
                        result(f, frame) = segment[f];
                }
                
                return result;
            }
            
            // Inverse STFT (overlap-add)
            template <class E>
            inline auto istft(const xexpression<E>& Zxx, size_t nperseg = 256, size_t noverlap = 0,
                              const std::string& window_type = "hann")
            {
                const auto& stft_mat = Zxx.derived_cast();
                if (stft_mat.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "istft: input must be 2-D (freq x time)");
                
                size_t n_freq = stft_mat.shape()[0];
                size_t n_frames = stft_mat.shape()[1];
                size_t n_fft = 2 * (n_freq - 1);
                size_t hop = nperseg - noverlap;
                size_t expected_len = (n_frames - 1) * hop + nperseg;
                
                // Create window
                xarray_container<double> window({nperseg});
                if (window_type == "hann")
                {
                    for (size_t i = 0; i < nperseg; ++i)
                        window(i) = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (nperseg - 1)));
                }
                else
                {
                    for (size_t i = 0; i < nperseg; ++i) window(i) = 1.0;
                }
                
                xarray_container<double> result({expected_len}, 0.0);
                xarray_container<double> norm({expected_len}, 0.0);
                
                for (size_t frame = 0; frame < n_frames; ++frame)
                {
                    std::vector<complex128> segment(n_fft, 0.0);
                    for (size_t f = 0; f < n_freq; ++f)
                        segment[f] = stft_mat(f, frame);
                    // Conjugate symmetry for real output
                    for (size_t f = 1; f < n_fft/2; ++f)
                        segment[n_fft - f] = std::conj(segment[f]);
                    
                    detail::fft_core(segment.data(), n_fft, true);
                    
                    size_t start = frame * hop;
                    for (size_t i = 0; i < nperseg; ++i)
                    {
                        if (start + i < expected_len)
                        {
                            result(start + i) += std::real(segment[i]) * window(i);
                            norm(start + i) += window(i) * window(i);
                        }
                    }
                }
                
                for (size_t i = 0; i < expected_len; ++i)
                    if (norm(i) > 1e-10) result(i) /= norm(i);
                
                return result;
            }
            
            // --------------------------------------------------------------------
            // Power Spectrum / Spectrogram
            // --------------------------------------------------------------------
            template <class E>
            inline auto spectrogram(const xexpression<E>& x, size_t nperseg = 256, size_t noverlap = 0,
                                    const std::string& window = "hann", const std::string& scaling = "density")
            {
                auto Zxx = stft(x, nperseg, noverlap, window);
                size_t n_freq = Zxx.shape()[0];
                size_t n_frames = Zxx.shape()[1];
                
                xarray_container<double> Pxx({n_freq, n_frames});
                double scale = 1.0;
                if (scaling == "density")
                {
                    // Compute window sum of squares
                    double win_sum_sq = 0.0;
                    if (window == "hann")
                    {
                        for (size_t i = 0; i < nperseg; ++i)
                        {
                            double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (nperseg - 1)));
                            win_sum_sq += w * w;
                        }
                    }
                    else
                    {
                        win_sum_sq = static_cast<double>(nperseg);
                    }
                    scale = 1.0 / (win_sum_sq * static_cast<double>(nperseg));
                }
                else // spectrum
                {
                    scale = 1.0 / (nperseg * nperseg);
                }
                
                for (size_t i = 0; i < n_freq; ++i)
                    for (size_t j = 0; j < n_frames; ++j)
                        Pxx(i, j) = scale * std::norm(Zxx(i, j));
                
                return Pxx;
            }
            
            // --------------------------------------------------------------------
            // Cepstrum (real and complex)
            // --------------------------------------------------------------------
            template <class E>
            inline auto rceps(const xexpression<E>& x)
            {
                // Real cepstrum: ifft(log(|fft(x)|))
                auto spectrum = fft(x);
                for (auto& v : spectrum)
                    v = std::log(std::abs(v) + 1e-10);
                auto ceps = ifft(spectrum);
                xarray_container<double> result({ceps.size()});
                for (size_t i = 0; i < ceps.size(); ++i)
                    result(i) = std::real(ceps(i));
                return result;
            }
            
            template <class E>
            inline auto cceps(const xexpression<E>& x)
            {
                // Complex cepstrum: ifft(log(fft(x)))
                auto spectrum = fft(x);
                for (auto& v : spectrum)
                    v = std::log(v + 1e-10);
                return ifft(spectrum);
            }
            
            // --------------------------------------------------------------------
            // Goertzel algorithm for single frequency detection
            // --------------------------------------------------------------------
            template <class E>
            inline auto goertzel(const xexpression<E>& x, double target_freq, double sample_rate)
            {
                const auto& expr = x.derived_cast();
                size_t N = expr.size();
                double k = 0.5 + N * target_freq / sample_rate;
                double omega = 2.0 * M_PI * k / N;
                double coeff = 2.0 * std::cos(omega);
                
                double s0 = 0.0, s1 = 0.0, s2 = 0.0;
                for (size_t i = 0; i < N; ++i)
                {
                    s0 = static_cast<double>(expr(i)) + coeff * s1 - s2;
                    s2 = s1;
                    s1 = s0;
                }
                std::complex<double> result(s1 - s2 * std::cos(omega), s2 * std::sin(omega));
                result *= std::exp(std::complex<double>(0, -omega * (N - 1)));
                return result;
            }
            
            // --------------------------------------------------------------------
            // Chirp Z-Transform (CZT)
            // --------------------------------------------------------------------
            template <class E>
            inline auto czt(const xexpression<E>& x, size_t M = 0, double w = 1.0, double a = 1.0)
            {
                const auto& expr = x.derived_cast();
                size_t N = expr.size();
                if (M == 0) M = N;
                
                // Create complex A and W
                std::complex<double> A = std::polar(1.0, -2.0 * M_PI * a);
                std::complex<double> W = std::polar(1.0, -2.0 * M_PI * w);
                
                // Use Bluestein's algorithm via convolution (simplified)
                size_t L = detail::next_power_of_two(N + M - 1);
                std::vector<std::complex<double>> h(L, 0.0), y(L, 0.0);
                
                for (size_t n = 0; n < N; ++n)
                {
                    std::complex<double> An = std::pow(A, static_cast<double>(n));
                    std::complex<double> Wn2 = std::pow(W, -static_cast<double>(n*n)/2.0);
                    y[n] = static_cast<std::complex<double>>(expr(n)) * An * Wn2;
                }
                for (size_t n = 0; n < M; ++n)
                {
                    h[n] = std::pow(W, static_cast<double>(n*n)/2.0);
                }
                for (size_t n = 1; n < N; ++n)
                {
                    h[L - n] = std::pow(W, -static_cast<double>(n*n)/2.0);
                }
                
                // Convolution via FFT
                std::vector<std::complex<double>> Yf(L), Hf(L);
                std::copy(y.begin(), y.end(), Yf.begin());
                std::copy(h.begin(), h.end(), Hf.begin());
                detail::fft_core(Yf.data(), L, false);
                detail::fft_core(Hf.data(), L, false);
                for (size_t i = 0; i < L; ++i) Yf[i] *= Hf[i];
                detail::fft_core(Yf.data(), L, true);
                
                xarray_container<std::complex<double>> result({M});
                for (size_t k = 0; k < M; ++k)
                {
                    std::complex<double> Wk2 = std::pow(W, static_cast<double>(k*k)/2.0);
                    result(k) = Yf[k] * Wk2;
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Walsh-Hadamard Transform
            // --------------------------------------------------------------------
            template <class E>
            inline auto fwht(const xexpression<E>& x)
            {
                auto result = eval(x);
                size_t N = result.size();
                if (!detail::is_power_of_two(N))
                    XTENSOR_THROW(std::invalid_argument, "fwht: size must be power of two");
                
                for (size_t len = 1; len < N; len <<= 1)
                {
                    for (size_t i = 0; i < N; i += 2*len)
                    {
                        for (size_t j = 0; j < len; ++j)
                        {
                            auto u = result(i + j);
                            auto v = result(i + j + len);
                            result(i + j) = u + v;
                            result(i + j + len) = u - v;
                        }
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Discrete Hartley Transform
            // --------------------------------------------------------------------
            template <class E>
            inline auto dht(const xexpression<E>& x)
            {
                const auto& expr = x.derived_cast();
                size_t N = expr.size();
                xarray_container<double> result({N});
                for (size_t k = 0; k < N; ++k)
                {
                    double sum = 0.0;
                    for (size_t n = 0; n < N; ++n)
                    {
                        double angle = 2.0 * M_PI * k * n / N;
                        sum += static_cast<double>(expr(n)) * (std::cos(angle) + std::sin(angle));
                    }
                    result(k) = sum;
                }
                return result;
            }
            
            template <class E>
            inline auto idht(const xexpression<E>& x)
            {
                auto result = dht(x);
                size_t N = result.size();
                for (auto& v : result) v /= static_cast<double>(N);
                return result;
            }
            
        } // namespace transform
        
        // Bring transforms into xt namespace
        using transform::dft;
        using transform::idft;
        using transform::fft;
        using transform::ifft;
        using transform::rfft;
        using transform::irfft;
        using transform::fft2;
        using transform::ifft2;
        using transform::fftshift;
        using transform::ifftshift;
        using transform::dct;
        using transform::idct;
        using transform::dst;
        using transform::idst;
        using transform::hilbert;
        using transform::analytic_signal;
        using transform::haar_wavelet;
        using transform::inverse_haar_wavelet;
        using transform::fftconvolve;
        using transform::stft;
        using transform::istft;
        using transform::spectrogram;
        using transform::rceps;
        using transform::cceps;
        using transform::goertzel;
        using transform::czt;
        using transform::fwht;
        using transform::dht;
        using transform::idht;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XTRANSFORM_HPP

// math/xtransform.hpp