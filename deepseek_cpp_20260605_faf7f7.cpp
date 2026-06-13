//File 0402 : xtensor-fftw/xtensor_fftw.hpp
//Top‑level include for xtensor-fftw: aggregates all FFT modules for real and complex transforms, providing SIMD-accelerated frequency analysis for 2D/3D simulation.
#ifndef XTENSOR_FFTW_HPP
#define XTENSOR_FFTW_HPP

#include "xtensor_fftw_config.hpp"
#include "xtensor_fftw_common.hpp"
#include "xtensor_fftw_complex.hpp"
#include "xtensor_fftw_real.hpp"
#include "xtensor_fftw_multidim.hpp"
#include "xtensor_fftw_util.hpp"

namespace xt {
namespace fftw {

    // Re‑export common types to user namespace
    using detail::fftw_ptr;
    using detail::plan_holder;

    // Forward transforms (real → complex)
    template <class E>
    auto rfft(const xexpression<E>& e);

    template <class E>
    auto rfft2(const xexpression<E>& e);

    template <class E>
    auto rfftn(const xexpression<E>& e);

    // Inverse transforms (complex → real)
    template <class E>
    auto irfft(const xexpression<E>& e, std::size_t n_out = 0);

    template <class E>
    auto irfft2(const xexpression<E>& e);

    template <class E>
    auto irfftn(const xexpression<E>& e);

    // Complex transforms
    template <class E>
    auto fft(const xexpression<E>& e);

    template <class E>
    auto ifft(const xexpression<E>& e);

    template <class E>
    auto fft2(const xexpression<E>& e);

    template <class E>
    auto ifft2(const xexpression<E>& e);

    template <class E>
    auto fftn(const xexpression<E>& e);

    template <class E>
    auto ifftn(const xexpression<E>& e);

    // Utility: shift zero-frequency to center
    template <class E>
    auto fftshift(const xexpression<E>& e);

    template <class E>
    auto ifftshift(const xexpression<E>& e);

    template <class E>
    auto fftfreq(std::size_t n, double d = 1.0);

} // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_HPP