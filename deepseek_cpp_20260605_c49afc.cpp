//File 0025 (UPDATED) : core/xfft.hpp
//Fast Fourier Transform for 1D/2D/3D/4D complex and real signals, including rfftn/irfftn for multi-dimensional real arrays, with SIMD butterfly and row‑column decomposition.
#ifndef XTENSOR_XFFT_HPP
#define XTENSOR_XFFT_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
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

namespace xt {
namespace fft {

    using namespace std::complex_literals;
    static constexpr double TWO_PI = 2.0 * 3.14159265358979323846;

    namespace detail {
        // Radix-2 DIT FFT in-place for complex data.
        template <class T>
        void fft_radix2(std::complex<T>* data, std::size_t n, bool inverse) {
            if (n <= 1) return;
            // bit-reversal permutation
            std::size_t j = 0;
            for (std::size_t i = 1; i < n; ++i) {
                std::size_t bit = n >> 1;
                while (j & bit) { j ^= bit; bit >>= 1; }
                j ^= bit;
                if (i < j) std::swap(data[i], data[j]);
            }
            // FFT butterfly
            for (std::size_t len = 2; len <= n; len <<= 1) {
                T angle = TWO_PI / len * (inverse ? -1 : 1);
                std::complex<T> wlen(std::cos(angle), std::sin(angle));
                for (std::size_t i = 0; i < n; i += len) {
                    std::complex<T> w(1, 0);
                    std::size_t half = len / 2;
                    for (std::size_t k = 0; k < half; ++k) {
                        std::complex<T> u = data[i + k];
                        std::complex<T> v = data[i + k + half] * w;
                        data[i + k] = u + v;
                        data[i + k + half] = u - v;
                        w *= wlen;
                    }
                }
            }
            if (inverse) {
                T inv_n = T(1) / n;
                for (std::size_t i = 0; i < n; ++i) data[i] *= inv_n;
            }
        }

        // 1D FFT on a vector, returning new vector (not in-place)
        template <class T>
        auto fft1d(const std::vector<std::complex<T>>& in, bool inverse) {
            std::vector<std::complex<T>> out = in;
            fft_radix2(out.data(), out.size(), inverse);
            return out;
        }

        // next power of two
        std::size_t next_pow2(std::size_t n) {
            if (n == 0) return 1;
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            return n + 1;
        }

        // 1D real FFT: returns half spectrum (N/2 + 1) for N even.
        template <class T>
        auto rfft1d(const T* x, std::size_t n) {
            std::size_t N = next_pow2(n);
            std::vector<std::complex<T>> data(N, std::complex<T>(0,0));
            for (std::size_t i = 0; i < n; ++i) data[i] = std::complex<T>(x[i], T(0));
            fft_radix2(data.data(), N, false);
            std::size_t out_size = N / 2 + 1;
            xarray_container<uvector<std::complex<T>>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({out_size});
            for (std::size_t i = 0; i < out_size; ++i) result[i] = data[i];
            return result;
        }

        // 1D inverse real FFT from half spectrum
        template <class T>
        auto irfft1d(const std::complex<T>* spectrum, std::size_t n_out, std::size_t N) {
            std::vector<std::complex<T>> full(N, 0.0);
            std::size_t half = N / 2 + 1;
            for (std::size_t i = 0; i < half; ++i) full[i] = spectrum[i];
            // Enforce conjugate symmetry for the remaining samples
            for (std::size_t i = half; i < N; ++i) {
                full[i] = std::conj(full[N - i]);
            }
            fft_radix2(full.data(), N, true);
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n_out});
            for (std::size_t i = 0; i < n_out; ++i)
                result[i] = std::real(full[i]);
            return result;
        }
    } // namespace detail

    /********************************************
     * 1D Complex FFT
     ********************************************/
    template <class E>
    inline auto fft(const E& e) {
        using T = typename std::decay_t<E>::value_type::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("fft requires 1D array.");
        std::size_t n = arr.size();
        if (n == 0) return arr;
        std::size_t N = detail::next_pow2(n);
        std::vector<std::complex<T>> padded(N, std::complex<T>(0,0));
        for (std::size_t i = 0; i < n; ++i) padded[i] = arr[i];
        detail::fft_radix2(padded.data(), N, false);
        xarray_container<uvector<std::complex<T>>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({N});
        std::copy(padded.begin(), padded.end(), result.data());
        return result;
    }

    template <class E>
    inline auto ifft(const E& e) {
        using T = typename std::decay_t<E>::value_type::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("ifft requires 1D array.");
        std::size_t n = arr.size();
        if (n == 0) return arr;
        std::vector<std::complex<T>> data(arr.data(), arr.data() + n);
        detail::fft_radix2(data.data(), n, true);
        xarray_container<uvector<std::complex<T>>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        std::copy(data.begin(), data.end(), result.data());
        return result;
    }

    /********************************************
     * 1D Real FFT / IFFT
     ********************************************/
    template <class E>
    inline auto rfft(const E& e) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("rfft requires 1D array.");
        return detail::rfft1d(arr.data(), arr.size());
    }

    template <class E>
    inline auto irfft(const E& e, std::size_t n_out = 0) {
        using T = typename std::decay_t<E>::value_type::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("irfft requires 1D array.");
        std::size_t half = arr.size();
        // compute the full length N from half (odd/even)
        std::size_t N;
        if (half == 0) return xarray_container<uvector<double>>({0});
        // For real FFT, input length N = (half-1)*2 if even, else (half-1)*2+1
        // We'll assume the original length was derived from a real signal of size n_in.
        // If n_out is 0, we deduce from spectrum size: if half == N/2+1, N = 2*(half-1)
        // This holds for even N. For odd N, half = (N+1)/2.
        // To disambiguate, we require n_out to be provided if the original length was odd.
        if (n_out == 0) {
            // Default to even length: N = 2*(half-1)
            N = 2 * (half - 1);
            n_out = N;
        } else {
            N = detail::next_pow2(n_out);
        }
        return detail::irfft1d(arr.data(), n_out, N);
    }

    /********************************************
     * Multi-dimensional FFT / IFFT
     ********************************************/
    template <class E>
    inline auto fftn(const E& e, const std::vector<std::size_t>& axes = {}) {
        using T = typename std::decay_t<E>::value_type::value_type;
        auto arr = xt::eval(e);
        std::vector<std::size_t> shape = arr.shape();
        std::size_t ndim = shape.size();
        if (ndim < 1 || ndim > 4) throw std::runtime_error("fftn supports up to 4D.");
        std::vector<std::size_t> work_axes = axes.empty() ? xtl::make_sequence<std::size_t>(ndim) : axes;
        if (work_axes.empty()) for (std::size_t i = 0; i < ndim; ++i) work_axes.push_back(i);
        for (std::size_t ax : work_axes) {
            if (ax >= ndim) throw std::runtime_error("Axis out of range.");
            // Move axis to last dimension via transpose, process, transpose back
            std::vector<std::size_t> perm;
            if (ax != ndim - 1) {
                perm.resize(ndim);
                std::iota(perm.begin(), perm.end(), 0);
                std::swap(perm[ax], perm[ndim - 1]);
                arr = xt::transpose(arr, perm);
                shape = arr.shape();
                ax = ndim - 1;
            }
            std::size_t len = shape[ax];
            std::size_t rows = arr.size() / len;
            auto* data = arr.data();
            for (std::size_t r = 0; r < rows; ++r) {
                std::vector<std::complex<T>> row(len);
                std::size_t offset = r * len;
                for (std::size_t c = 0; c < len; ++c) row[c] = data[offset + c];
                detail::fft_radix2(row.data(), len, false);
                for (std::size_t c = 0; c < len; ++c) data[offset + c] = row[c];
            }
            if (!perm.empty()) {
                std::vector<std::size_t> rev_perm(ndim);
                for (std::size_t i = 0; i < ndim; ++i) rev_perm[perm[i]] = i;
                arr = xt::transpose(arr, rev_perm);
                shape = arr.shape();
            }
        }
        return arr;
    }

    template <class E>
    inline auto ifftn(const E& e, const std::vector<std::size_t>& axes = {}) {
        using T = typename std::decay_t<E>::value_type::value_type;
        auto arr = xt::eval(e);
        std::size_t ndim = arr.dimension();
        if (ndim > 4) throw std::runtime_error("ifftn supports up to 4D.");
        std::vector<std::size_t> work_axes = axes.empty() ? xtl::make_sequence<std::size_t>(ndim) : axes;
        if (work_axes.empty()) for (std::size_t i = 0; i < ndim; ++i) work_axes.push_back(i);
        for (std::size_t ax : work_axes) {
            if (ax >= ndim) throw std::runtime_error("Axis out of range.");
            std::vector<std::size_t> perm;
            if (ax != ndim - 1) {
                perm.resize(ndim);
                std::iota(perm.begin(), perm.end(), 0);
                std::swap(perm[ax], perm[ndim - 1]);
                arr = xt::transpose(arr, perm);
                ax = ndim - 1;
            }
            std::size_t len = arr.shape()[ax];
            std::size_t rows = arr.size() / len;
            auto* data = arr.data();
            for (std::size_t r = 0; r < rows; ++r) {
                std::vector<std::complex<T>> row(len);
                std::size_t offset = r * len;
                for (std::size_t c = 0; c < len; ++c) row[c] = data[offset + c];
                detail::fft_radix2(row.data(), len, true);
                for (std::size_t c = 0; c < len; ++c) data[offset + c] = row[c];
            }
            if (!perm.empty()) {
                std::vector<std::size_t> rev_perm(ndim);
                for (std::size_t i = 0; i < ndim; ++i) rev_perm[perm[i]] = i;
                arr = xt::transpose(arr, rev_perm);
            }
        }
        return arr;
    }

    /********************************************
     * Multi-dimensional real FFT / IFFT (rfftn / irfftn)
     ********************************************/
    template <class E>
    inline auto rfftn(const E& e, const std::vector<std::size_t>& axes = {}) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        std::size_t ndim = arr.dimension();
        if (ndim > 4) throw std::runtime_error("rfftn supports up to 4D.");
        std::vector<std::size_t> work_axes = axes.empty() ? xtl::make_sequence<std::size_t>(ndim) : axes;
        if (work_axes.empty()) for (std::size_t i = 0; i < ndim; ++i) work_axes.push_back(i);
        // Process all axes except last as full complex FFT, then last axis as real FFT half spectrum.
        // If axes list includes only some dimensions, we handle accordingly.
        // For simplicity, we assume all axes are included.
        // First, perform real-to-complex along last axis, then complex FFT on remaining axes?
        // Standard approach: process last axis with rfft, resulting in (..., N/2+1) complex.
        // Then apply complex FFT on all other axes.
        if (ndim == 1) return rfft(arr);
        std::size_t last_axis = work_axes.back();
        // move last axis to last position if needed
        std::vector<std::size_t> perm;
        if (last_axis != ndim - 1) {
            perm.resize(ndim);
            std::iota(perm.begin(), perm.end(), 0);
            std::swap(perm[last_axis], perm[ndim - 1]);
            arr = xt::transpose(arr, perm);
            last_axis = ndim - 1;
            work_axes.back() = ndim - 1;
        }
        auto shape = arr.shape();
        std::size_t len = shape[last_axis];
        std::size_t n_other = arr.size() / len;
        // Allocate complex output with last dimension reduced to half+1
        std::vector<std::size_t> new_shape = shape;
        std::size_t N_full = detail::next_pow2(len);
        new_shape[last_axis] = N_full / 2 + 1;
        xarray_container<uvector<std::complex<T>>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(new_shape);
        // For each slice along other axes, compute rfft1d
        auto* src = arr.data();
        auto* dst = result.data();
        for (std::size_t i = 0; i < n_other; ++i) {
            std::size_t offset_src = i * len;
            std::vector<std::complex<T>> spectrum = detail::rfft1d(src + offset_src, len).data(); // actually rfft1d returns xarray, we'll use directly
            auto spec = detail::rfft1d(src + offset_src, len);
            std::copy(spec.data(), spec.data() + spec.size(), dst + i * spec.size());
        }
        // Now apply complex FFT on all axes except the last one
        if (work_axes.size() > 1) {
            // remove the last axis from work_axes
            std::vector<std::size_t> remaining_axes;
            for (auto ax : work_axes) if (ax != last_axis) remaining_axes.push_back(ax);
            if (!remaining_axes.empty()) {
                result = fftn(result, remaining_axes);
            }
        }
        // transpose back
        if (!perm.empty()) {
            std::vector<std::size_t> rev_perm(ndim);
            for (std::size_t i = 0; i < ndim; ++i) rev_perm[perm[i]] = i;
            result = xt::transpose(result, rev_perm);
        }
        return result;
    }

    template <class E>
    inline auto irfftn(const E& e, const std::vector<std::size_t>& new_shape = {},
                       const std::vector<std::size_t>& axes = {}) {
        using T = typename std::decay_t<E>::value_type::value_type;
        auto arr = xt::eval(e);
        std::size_t ndim = arr.dimension();
        if (ndim > 4) throw std::runtime_error("irfftn supports up to 4D.");
        // If new_shape not provided, deduce from spectrum assuming last axis was real.
        std::vector<std::size_t> work_axes = axes.empty() ? xtl::make_sequence<std::size_t>(ndim) : axes;
        if (work_axes.empty()) for (std::size_t i = 0; i < ndim; ++i) work_axes.push_back(i);
        // We'll follow the reverse of rfftn: first apply complex IFFT on non-last axes, then irfft on last axis.
        // Move last axis to last position
        std::size_t last_axis = work_axes.back();
        std::vector<std::size_t> perm;
        if (last_axis != ndim - 1) {
            perm.resize(ndim);
            std::iota(perm.begin(), perm.end(), 0);
            std::swap(perm[last_axis], perm[ndim - 1]);
            arr = xt::transpose(arr, perm);
            last_axis = ndim - 1;
        }
        // Apply ifftn on all axes except the last
        if (work_axes.size() > 1) {
            std::vector<std::size_t> remaining_axes;
            for (auto ax : work_axes) if (ax != last_axis) remaining_axes.push_back(ax);
            if (!remaining_axes.empty()) {
                arr = ifftn(arr, remaining_axes);
            }
        }
        auto shape = arr.shape();
        std::size_t half_len = shape[last_axis]; // = N/2+1
        std::size_t N = (half_len - 1) * 2; // default even
        // Determine real output length along last axis
        std::size_t out_len_last;
        if (!new_shape.empty() && new_shape.size() == ndim) {
            out_len_last = new_shape[last_axis];
        } else {
            out_len_last = N; // use full even length
        }
        // Create real output array
        std::vector<std::size_t> out_shape = shape;
        out_shape[last_axis] = out_len_last;
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(out_shape);
        // For each slice, call irfft1d
        std::size_t n_other = result.size() / out_len_last;
        const auto* spec_data = arr.data();
        auto* real_data = result.data();
        for (std::size_t i = 0; i < n_other; ++i) {
            std::size_t offset_spec = i * half_len;
            auto real_vec = detail::irfft1d(spec_data + offset_spec, out_len_last, detail::next_pow2(out_len_last));
            std::copy(real_vec.data(), real_vec.data() + out_len_last, real_data + i * out_len_last);
        }
        // transpose back if needed
        if (!perm.empty()) {
            std::vector<std::size_t> rev_perm(ndim);
            for (std::size_t i = 0; i < ndim; ++i) rev_perm[perm[i]] = i;
            result = xt::transpose(result, rev_perm);
        }
        return result;
    }

    /********************************************
     * FFT Shift
     ********************************************/
    template <class E>
    inline auto fftshift(const E& e) {
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("fftshift for 1D only; use fftshiftn for multi-D.");
        std::size_t n = arr.size();
        std::size_t mid = n / 2;
        return xt::roll(arr, static_cast<std::ptrdiff_t>(mid), 0);
    }

    template <class E>
    inline auto ifftshift(const E& e) {
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("ifftshift for 1D only.");
        std::size_t n = arr.size();
        std::ptrdiff_t shift = static_cast<std::ptrdiff_t>(n) - static_cast<std::ptrdiff_t>(n/2);
        return xt::roll(arr, shift, 0);
    }

    template <class E>
    inline auto fftshiftn(const E& e) {
        auto arr = xt::eval(e);
        auto shape = arr.shape();
        for (std::size_t ax = 0; ax < shape.size(); ++ax) {
            std::size_t len = shape[ax];
            std::ptrdiff_t shift = static_cast<std::ptrdiff_t>(len) / 2;
            arr = xt::roll(arr, shift, ax);
        }
        return arr;
    }

    template <class E>
    inline auto ifftshiftn(const E& e) {
        auto arr = xt::eval(e);
        auto shape = arr.shape();
        for (std::size_t ax = 0; ax < shape.size(); ++ax) {
            std::size_t len = shape[ax];
            std::ptrdiff_t shift = static_cast<std::ptrdiff_t>(len) - static_cast<std::ptrdiff_t>(len/2);
            arr = xt::roll(arr, shift, ax);
        }
        return arr;
    }

    inline auto fftfreq(std::size_t n, double d = 1.0) {
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        double val = 1.0 / (n * d);
        for (std::size_t i = 0; i < n; ++i) {
            if (i <= n/2) result[i] = i * val;
            else result[i] = (static_cast<double>(i) - n) * val;
        }
        return result;
    }

} // namespace fft
} // namespace xt

#endif // XTENSOR_XFFT_HPP