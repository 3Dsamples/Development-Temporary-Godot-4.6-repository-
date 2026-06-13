//File 0406 : xtensor-fftw/xtensor_fftw_util.hpp
//FFT utilities: fftshift, ifftshift, fftfreq, rfftfreq, spectral normalization, and frequency axis generation for real/complex transforms.
#ifndef XTENSOR_FFTW_UTIL_HPP
#define XTENSOR_FFTW_UTIL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "xtensor_fftw_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xmath.hpp"

namespace xt {
namespace fftw {

    /**
     * Shift the zero-frequency component to the center of the spectrum.
     * Works for 1D and multi-dimensional arrays. For even-length dimensions,
     * the shift is by N/2; for odd, by (N-1)/2.
     * @param e The input array (complex or real).
     * @return A new array with zero-frequency centered.
     */
    template <class E>
    inline auto fftshift(const xexpression<E>& e)
    {
        auto arr = xt::eval(e.derived_cast());
        auto shape = arr.shape();
        std::size_t ndim = shape.size();
        for (std::size_t d = 0; d < ndim; ++d)
        {
            std::size_t len = shape[d];
            std::ptrdiff_t shift = static_cast<std::ptrdiff_t>(len) / 2;
            arr = xt::roll(arr, shift, d);
        }
        return arr;
    }

    /**
     * Inverse of fftshift: undo the zero-frequency centering.
     * @param e The input array with zero-frequency centered.
     * @return A new array with the original ordering.
     */
    template <class E>
    inline auto ifftshift(const xexpression<E>& e)
    {
        auto arr = xt::eval(e.derived_cast());
        auto shape = arr.shape();
        std::size_t ndim = shape.size();
        for (std::size_t d = 0; d < ndim; ++d)
        {
            std::size_t len = shape[d];
            std::ptrdiff_t shift = static_cast<std::ptrdiff_t>(len) - static_cast<std::ptrdiff_t>(len) / 2;
            arr = xt::roll(arr, shift, d);
        }
        return arr;
    }

    /**
     * Generate the frequency sample points for a complex FFT.
     * For N points with spacing d, the frequencies are:
     *   f = [0, 1, ..., (N-1)//2, -N//2, ..., -1] / (N * d)
     * @param n Number of samples.
     * @param d Sample spacing (default 1.0).
     * @return 1D array of frequencies.
     */
    inline auto fftfreq(std::size_t n, double d = 1.0)
    {
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        double factor = 1.0 / (n * d);
        if (n % 2 == 0)
        {
            for (std::size_t i = 0; i <= n / 2; ++i)
                result[i] = static_cast<double>(i) * factor;
            for (std::size_t i = n / 2 + 1; i < n; ++i)
                result[i] = (static_cast<double>(i) - static_cast<double>(n)) * factor;
        }
        else
        {
            for (std::size_t i = 0; i <= (n - 1) / 2; ++i)
                result[i] = static_cast<double>(i) * factor;
            for (std::size_t i = (n - 1) / 2 + 1; i < n; ++i)
                result[i] = (static_cast<double>(i) - static_cast<double>(n)) * factor;
        }
        return result;
    }

    /**
     * Generate the frequency sample points for a real FFT (non-negative frequencies).
     * Returns the first n//2 + 1 frequencies (including zero and Nyquist).
     * @param n Number of samples in the original real signal.
     * @param d Sample spacing (default 1.0).
     * @return 1D array of length n/2 + 1.
     */
    inline auto rfftfreq(std::size_t n, double d = 1.0)
    {
        std::size_t n_out = n / 2 + 1;
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n_out});
        double factor = 1.0 / (n * d);
        for (std::size_t i = 0; i < n_out; ++i)
            result[i] = static_cast<double>(i) * factor;
        return result;
    }

    /**
     * Generate the radial frequency grid for a 2D FFT.
     * Returns a 2D array where each element is sqrt(kx^2 + ky^2),
     * with zero frequency at the corner (use fftshift to center).
     * @param nx Number of samples in x-direction.
     * @param ny Number of samples in y-direction.
     * @return 2D array of radial frequencies.
     */
    inline auto fftfreq2d(std::size_t nx, std::size_t ny)
    {
        auto kx = fftfreq(nx);
        auto ky = fftfreq(ny);
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({nx, ny});
        for (std::size_t i = 0; i < nx; ++i)
            for (std::size_t j = 0; j < ny; ++j)
                result(i, j) = std::sqrt(kx[i] * kx[i] + ky[j] * ky[j]);
        return result;
    }

    /**
     * Normalize a complex spectrum to form a power spectral density (PSD).
     * PSD = |X|^2 / (fs * N), where fs is the sample rate (1/d).
     * For real signals, only the positive frequencies are returned.
     * @param spectrum The complex FFT output.
     * @param d Sample spacing (default 1.0).
     * @return Power spectral density array.
     */
    template <class E>
    inline auto psd(const xexpression<E>& spectrum, double d = 1.0)
    {
        const auto& sp = spectrum.derived_cast();
        using T = typename std::decay_t<E>::value_type;
        using real_type = typename T::value_type;
        auto power = xt::real(sp * xt::conj(sp));
        real_type factor = static_cast<real_type>(1.0) / (static_cast<real_type>(sp.size()) * real_type(d));
        return power * factor;
    }

    /**
     * Compute the magnitude spectrum (absolute value).
     * @param spectrum The complex FFT output.
     * @return Real array of magnitudes.
     */
    template <class E>
    inline auto magnitude_spectrum(const xexpression<E>& spectrum)
    {
        return xt::abs(spectrum.derived_cast());
    }

    /**
     * Compute the phase spectrum (angle).
     * @param spectrum The complex FFT output.
     * @return Real array of phase angles in radians.
     */
    template <class E>
    inline auto phase_spectrum(const xexpression<E>& spectrum)
    {
        return xt::arg(spectrum.derived_cast());
    }

} // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_UTIL_HPP