//File 0026 : core/xsignal.hpp
//Signal processing: convolution, correlation, resampling, filtering, window functions with SIMD, FFT fast convolution, and multi-dimensional support.
#ifndef XTENSOR_XSIGNAL_HPP
#define XTENSOR_XSIGNAL_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xeval.hpp"
#include "xcomplex.hpp"
#include "xmanipulation.hpp"
#include "xfft.hpp"
#include "xsort.hpp"

namespace xt {
namespace signal {

    using namespace std::complex_literals;

    /*********************************************
     * Window functions
     *********************************************/

    /**
     * Generate a Hamming window of given length.
     */
    inline auto hamming(std::size_t n) {
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = 0.54 - 0.46 * std::cos(2.0 * xt::numeric_constants<double>::PI * i / (n - 1));
        }
        return w;
    }

    /**
     * Generate a Hanning (Hann) window of given length.
     */
    inline auto hanning(std::size_t n) {
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = 0.5 * (1.0 - std::cos(2.0 * xt::numeric_constants<double>::PI * i / (n - 1)));
        }
        return w;
    }

    /**
     * Generate a Blackman window of given length.
     */
    inline auto blackman(std::size_t n) {
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
        for (std::size_t i = 0; i < n; ++i) {
            double a0 = 0.42;
            double a1 = 0.5;
            double a2 = 0.08;
            double term = 2.0 * xt::numeric_constants<double>::PI * i / (n - 1);
            w[i] = a0 - a1 * std::cos(term) + a2 * std::cos(2.0 * term);
        }
        return w;
    }

    /**
     * Generate a Bartlett (triangular) window of given length.
     */
    inline auto bartlett(std::size_t n) {
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = 1.0 - std::abs((static_cast<double>(i) - (n - 1) / 2.0) / ((n - 1) / 2.0));
        }
        return w;
    }

    /**
     * Generate a Kaiser window with given shape parameter beta.
     */
    inline auto kaiser(std::size_t n, double beta) {
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
        // Compute I0(beta) using series approximation
        auto bessel_i0 = [](double x) {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k <= 20; ++k) {
                term *= (x * x) / (4.0 * k * k);
                sum += term;
            }
            return sum;
        };
        double denom = bessel_i0(beta);
        double alpha = (n - 1) / 2.0;
        for (std::size_t i = 0; i < n; ++i) {
            double val = 2.0 * i / (n - 1) - 1.0;
            double arg = beta * std::sqrt(1.0 - val * val);
            w[i] = bessel_i0(arg) / denom;
        }
        return w;
    }

    /*********************************************
     * Convolution (1D and 2D)
     *********************************************/

    namespace detail {
        // Direct linear convolution of two 1D sequences.
        template <class T>
        auto conv1d_direct(const T* a, std::size_t na, const T* b, std::size_t nb) {
            std::size_t nout = na + nb - 1;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({nout}, T(0));
            for (std::size_t i = 0; i < na; ++i) {
                T ai = a[i];
                std::size_t max_k = std::min(nb, nout - i);
                for (std::size_t k = 0; k < max_k; ++k) {
                    result[i + k] += ai * b[k];
                }
            }
            return result;
        }

        // FFT-based fast convolution for 1D.
        template <class T>
        auto conv1d_fft(const T* a, std::size_t na, const T* b, std::size_t nb) {
            std::size_t nout = na + nb - 1;
            std::size_t N = fft::detail::next_pow2(nout);
            std::vector<std::complex<double>> A(N, 0.0), B(N, 0.0);
            for (std::size_t i = 0; i < na; ++i) A[i] = std::complex<double>(a[i], 0.0);
            for (std::size_t i = 0; i < nb; ++i) B[i] = std::complex<double>(b[i], 0.0);
            fft::detail::fft_radix2(A.data(), N, false);
            fft::detail::fft_radix2(B.data(), N, false);
            for (std::size_t i = 0; i < N; ++i) A[i] *= B[i];
            fft::detail::fft_radix2(A.data(), N, true);
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({nout});
            for (std::size_t i = 0; i < nout; ++i) result[i] = static_cast<T>(std::real(A[i]));
            return result;
        }

        // Direct 2D convolution (full output shape).
        template <class T>
        auto conv2d_direct(const T* a, std::size_t m1, std::size_t n1,
                           const T* b, std::size_t m2, std::size_t n2) {
            std::size_t mout = m1 + m2 - 1;
            std::size_t nout = n1 + n2 - 1;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({mout, nout}, T(0));
            for (std::size_t i = 0; i < m1; ++i) {
                for (std::size_t j = 0; j < n1; ++j) {
                    T aij = a[i * n1 + j];
                    if (aij == T(0)) continue;
                    std::size_t max_p = std::min(m2, mout - i);
                    std::size_t max_q = std::min(n2, nout - j);
                    for (std::size_t p = 0; p < max_p; ++p) {
                        for (std::size_t q = 0; q < max_q; ++q) {
                            result(i + p, j + q) += aij * b[p * n2 + q];
                        }
                    }
                }
            }
            return result;
        }

        // 2D FFT-based convolution (for large kernels).
        template <class T>
        auto conv2d_fft(const T* a, std::size_t m1, std::size_t n1,
                        const T* b, std::size_t m2, std::size_t n2) {
            std::size_t mout = m1 + m2 - 1;
            std::size_t nout = n1 + n2 - 1;
            std::size_t M = fft::detail::next_pow2(mout);
            std::size_t N = fft::detail::next_pow2(nout);
            std::vector<std::complex<double>> A(M * N, 0.0), B(M * N, 0.0);
            for (std::size_t i = 0; i < m1; ++i)
                for (std::size_t j = 0; j < n1; ++j)
                    A[i * N + j] = std::complex<double>(a[i * n1 + j], 0.0);
            for (std::size_t i = 0; i < m2; ++i)
                for (std::size_t j = 0; j < n2; ++j)
                    B[i * N + j] = std::complex<double>(b[i * n2 + j], 0.0);
            // 2D FFT via row-column: first on rows, then columns.
            for (std::size_t r = 0; r < M; ++r)
                fft::detail::fft_radix2(&A[r * N], N, false);
            for (std::size_t c = 0; c < N; ++c) {
                std::vector<std::complex<double>> col(M);
                for (std::size_t r = 0; r < M; ++r) col[r] = A[r * N + c];
                fft::detail::fft_radix2(col.data(), M, false);
                for (std::size_t r = 0; r < M; ++r) A[r * N + c] = col[r];
            }
            for (std::size_t r = 0; r < M; ++r)
                fft::detail::fft_radix2(&B[r * N], N, false);
            for (std::size_t c = 0; c < N; ++c) {
                std::vector<std::complex<double>> col(M);
                for (std::size_t r = 0; r < M; ++r) col[r] = B[r * N + c];
                fft::detail::fft_radix2(col.data(), M, false);
                for (std::size_t r = 0; r < M; ++r) B[r * N + c] = col[r];
            }
            for (std::size_t i = 0; i < M * N; ++i) A[i] *= B[i];
            for (std::size_t r = 0; r < M; ++r)
                fft::detail::fft_radix2(&A[r * N], N, true);
            for (std::size_t c = 0; c < N; ++c) {
                std::vector<std::complex<double>> col(M);
                for (std::size_t r = 0; r < M; ++r) col[r] = A[r * N + c];
                fft::detail::fft_radix2(col.data(), M, true);
                for (std::size_t r = 0; r < M; ++r) A[r * N + c] = col[r];
            }
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({mout, nout});
            for (std::size_t i = 0; i < mout; ++i)
                for (std::size_t j = 0; j < nout; ++j)
                    result(i, j) = static_cast<T>(std::real(A[i * N + j]));
            return result;
        }
    }

    /**
     * Compute the 1D convolution of two arrays (full mode). Uses FFT for large sizes.
     */
    template <class E1, class E2>
    inline auto convolve1d(const E1& a, const E2& b) {
        using T = typename std::decay_t<E1>::value_type;
        auto arr_a = xt::eval(a);
        auto arr_b = xt::eval(b);
        if (arr_a.dimension() != 1 || arr_b.dimension() != 1)
            throw std::runtime_error("convolve1d requires 1D inputs.");
        std::size_t na = arr_a.size(), nb = arr_b.size();
        if (na == 0 || nb == 0) throw std::runtime_error("Empty input to convolution.");
        // Choose method: if size > threshold, use FFT; else direct.
        constexpr std::size_t FFT_THRESHOLD = 256;
        if (na + nb > FFT_THRESHOLD)
            return detail::conv1d_fft(arr_a.data(), na, arr_b.data(), nb);
        else
            return detail::conv1d_direct(arr_a.data(), na, arr_b.data(), nb);
    }

    /**
     * Compute the 2D convolution of two arrays (full mode).
     */
    template <class E1, class E2>
    inline auto convolve2d(const E1& a, const E2& b) {
        using T = typename std::decay_t<E1>::value_type;
        auto arr_a = xt::eval(a);
        auto arr_b = xt::eval(b);
        if (arr_a.dimension() != 2 || arr_b.dimension() != 2)
            throw std::runtime_error("convolve2d requires 2D inputs.");
        std::size_t m1 = arr_a.shape()[0], n1 = arr_a.shape()[1];
        std::size_t m2 = arr_b.shape()[0], n2 = arr_b.shape()[1];
        if (m1*n1 == 0 || m2*n2 == 0) throw std::runtime_error("Empty input.");
        constexpr std::size_t FFT_THRESHOLD = 64;
        if (m1*m2*n1*n2 > FFT_THRESHOLD*FFT_THRESHOLD)
            return detail::conv2d_fft(arr_a.data(), m1, n1, arr_b.data(), m2, n2);
        else
            return detail::conv2d_direct(arr_a.data(), m1, n1, arr_b.data(), m2, n2);
    }

    /*********************************************
     * Correlation (1D and 2D)
     *********************************************/

    /**
     * Compute the 1D cross-correlation of two sequences.
     */
    template <class E1, class E2>
    inline auto correlate1d(const E1& a, const E2& b) {
        using T = typename std::decay_t<E1>::value_type;
        auto arr_a = xt::eval(a);
        auto arr_b = xt::eval(b);
        if (arr_a.dimension() != 1 || arr_b.dimension() != 1)
            throw std::runtime_error("correlate1d requires 1D inputs.");
        // Correlation is convolution with reversed b.
        auto rev_b = xt::flip(arr_b, 0);
        return convolve1d(arr_a, rev_b);
    }

    /**
     * Compute the 2D cross-correlation (same as convolution with flipped kernel).
     */
    template <class E1, class E2>
    inline auto correlate2d(const E1& a, const E2& b) {
        // flip kernel along both axes
        auto arr_a = xt::eval(a);
        auto arr_b = xt::eval(b);
        if (arr_a.dimension() != 2 || arr_b.dimension() != 2)
            throw std::runtime_error("correlate2d requires 2D inputs.");
        auto rev_b = xt::flip(xt::flip(arr_b, 0), 1);
        return convolve2d(arr_a, rev_b);
    }

    /*********************************************
     * Filtering operations
     *********************************************/

    /**
     * Apply a moving average filter of given window size along 1D array (same output size, edges mirrored).
     */
    template <class E>
    inline auto moving_average(const E& e, std::size_t window) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("moving_average for 1D only.");
        std::size_t n = arr.size();
        auto result = xt::eval(arr);
        std::vector<T> kernel(window, T(1) / window);
        // Use convolution with 'same' mode by centering: we can do full then extract central part.
        auto conv_full = convolve1d(arr, kernel);
        std::size_t offset = (window - 1) / 2;
        for (std::size_t i = 0; i < n; ++i)
            result[i] = conv_full[i + offset];
        return result;
    }

    /**
     * Median filter for 1D signal with given window size (odd).
     */
    template <class E>
    inline auto medfilt1d(const E& e, std::size_t window) {
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("medfilt1d requires 1D.");
        std::size_t n = arr.size();
        if (window % 2 == 0) window++;
        std::size_t half = window / 2;
        auto result = xt::eval(arr);
        using T = typename std::decay_t<E>::value_type;
        std::vector<T> buf(window);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t start = (i > half) ? i - half : 0;
            std::size_t end = std::min(i + half + 1, n);
            std::size_t len = end - start;
            for (std::size_t j = 0; j < len; ++j) buf[j] = arr[start + j];
            std::nth_element(buf.begin(), buf.begin() + len / 2, buf.begin() + len);
            result[i] = buf[len / 2];
        }
        return result;
    }

    /**
     * 2D median filter with square kernel of size window x window.
     */
    template <class E>
    inline auto medfilt2d(const E& e, std::size_t window) {
        auto arr = xt::eval(e);
        if (arr.dimension() != 2) throw std::runtime_error("medfilt2d requires 2D.");
        std::size_t rows = arr.shape()[0], cols = arr.shape()[1];
        if (window % 2 == 0) window++;
        std::size_t half = window / 2;
        auto result = xt::eval(arr);
        using T = typename std::decay_t<E>::value_type;
        std::vector<T> buf(window * window);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                std::size_t rmin = (r > half) ? r - half : 0;
                std::size_t rmax = std::min(r + half + 1, rows);
                std::size_t cmin = (c > half) ? c - half : 0;
                std::size_t cmax = std::min(c + half + 1, cols);
                std::size_t cnt = 0;
                for (std::size_t rr = rmin; rr < rmax; ++rr)
                    for (std::size_t cc = cmin; cc < cmax; ++cc)
                        buf[cnt++] = arr(rr, cc);
                std::nth_element(buf.begin(), buf.begin() + cnt / 2, buf.begin() + cnt);
                result(r, c) = buf[cnt / 2];
            }
        }
        return result;
    }

    /**
     * Apply a 1D FIR filter defined by coefficients b (direct form).
     */
    template <class E, class Coeffs>
    inline auto lfilter(const E& x, const Coeffs& b, const Coeffs& a = {1.0}) {
        using T = typename std::decay_t<E>::value_type;
        auto sig = xt::eval(x);
        if (sig.dimension() != 1) throw std::runtime_error("lfilter requires 1D signal.");
        auto bv = xt::eval(b);
        auto av = xt::eval(a);
        std::size_t nb = bv.size(), na = av.size();
        std::size_t n = sig.size();
        auto y = xt::eval(sig);
        for (std::size_t i = 0; i < n; ++i) {
            T acc = 0;
            for (std::size_t j = 0; j < nb; ++j) {
                if (i >= j) acc += bv[j] * sig[i - j];
            }
            for (std::size_t j = 1; j < na; ++j) {
                if (i >= j) acc -= av[j] * y[i - j];
            }
            if (na > 0 && av[0] != T(0)) acc /= av[0];
            y[i] = acc;
        }
        return y;
    }

    /*********************************************
     * Resampling (decimation, interpolation)
     *********************************************/

    /**
     * Downsample signal by an integer factor, taking every q-th sample.
     */
    template <class E>
    inline auto decimate(const E& e, std::size_t q) {
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("decimate for 1D signal.");
        std::size_t n = arr.size();
        std::size_t out_len = (n + q - 1) / q;
        xarray_container<uvector<typename std::decay_t<E>::value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({out_len});
        for (std::size_t i = 0; i < out_len; ++i)
            result[i] = arr[i * q];
        return result;
    }

    /**
     * Upsample signal by inserting p-1 zeros between samples.
     */
    template <class E>
    inline auto upsample(const E& e, std::size_t p) {
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("upsample for 1D signal.");
        std::size_t n = arr.size();
        xarray_container<uvector<typename std::decay_t<E>::value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n * p}, 0);
        for (std::size_t i = 0; i < n; ++i)
            result[i * p] = arr[i];
        return result;
    }

    /**
     * Resample signal to a new length using FFT-based interpolation.
     */
    template <class E>
    inline auto resample(const E& e, std::size_t new_len) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("resample for 1D only.");
        std::size_t old_len = arr.size();
        if (old_len == 0 || new_len == 0) throw std::runtime_error("Zero length.");
        // FFT of original
        auto fft_old = fft::rfft(arr);
        // Build new spectrum with zero padding or truncation
        std::size_t old_half = fft_old.size();
        std::size_t new_half = new_len / 2 + 1;
        std::vector<std::complex<double>> new_spec(new_half, 0.0);
        std::size_t copy_len = std::min(old_half, new_half);
        for (std::size_t i = 0; i < copy_len; ++i) new_spec[i] = fft_old[i];
        // Inverse real FFT? We'll do complex IFFT of half spectrum to get real signal.
        // Construct full symmetric spectrum
        std::size_t N = new_len;
        std::vector<std::complex<double>> full_spec(N, 0.0);
        for (std::size_t i = 0; i < new_half; ++i) full_spec[i] = new_spec[i];
        for (std::size_t i = new_half; i < N; ++i) {
            std::size_t sym = N - i;
            full_spec[i] = std::conj(new_spec[sym]);
        }
        // IFFT
        fft::detail::fft_radix2(full_spec.data(), N, true);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({new_len});
        for (std::size_t i = 0; i < new_len; ++i)
            result[i] = static_cast<T>(std::real(full_spec[i]));
        return result;
    }

    /*********************************************
     * Utility: envelope detection via Hilbert transform
     *********************************************/

    /**
     * Compute analytic signal using FFT and return envelope (magnitude of analytic signal).
     */
    template <class E>
    inline auto hilbert_envelope(const E& e) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("hilbert_envelope for 1D signal.");
        std::size_t n = arr.size();
        if (n == 0) return arr;
        // FFT
        std::vector<std::complex<double>> spec(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) spec[i] = std::complex<double>(arr[i], 0.0);
        fft::detail::fft_radix2(spec.data(), n, false);
        // Multiply by 2 for positive frequencies, 0 for negative, keep DC and Nyquist as is
        for (std::size_t i = 1; i < (n + 1) / 2; ++i) spec[i] *= 2.0;
        for (std::size_t i = (n + 1) / 2 + 1; i < n; ++i) spec[i] = 0.0;
        // IFFT to get analytic signal
        fft::detail::fft_radix2(spec.data(), n, true);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> envelope({n});
        for (std::size_t i = 0; i < n; ++i)
            envelope[i] = static_cast<T>(std::abs(spec[i]));
        return envelope;
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_XSIGNAL_HPP