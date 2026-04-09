// src/fft_real.cpp

#include "fft_plan.hpp"
#include "fft_simd.hpp"
#include <cassert>
#include <cstring>
#include <algorithm>

namespace fastfft {
namespace detail {

using simd::vfloat;
using simd::vdouble;
using simd::simd_traits;

// ----------------------------------------------------------------------------
// Real-to-complex post-processing (after forward FFT of real data)
// Input:  n complex values from real input (tmp[0..n-1])
// Output: n/2+1 complex values in half-complex format
// ----------------------------------------------------------------------------
template<typename T>
static void pack_real_forward(const std::complex<T>* tmp, std::complex<T>* out, size_t n) {
    const size_t nh = n / 2;

    // DC component (real)
    out[0] = std::complex<T>(tmp[0].real(), T(0));

    // Nyquist if n is even
    if (n % 2 == 0) {
        out[nh] = std::complex<T>(tmp[nh].real(), T(0));
    } else {
        out[nh] = tmp[nh];
    }

    // Pack symmetric pairs: out[k] = (tmp[k] + conj(tmp[n-k])) / 2
    const T half = T(0.5);
    for (size_t k = 1; k < nh; ++k) {
        const auto& a = tmp[k];
        const auto& b = tmp[n - k];
        out[k] = std::complex<T>(
            (a.real() + b.real()) * half,
            (a.imag() - b.imag()) * half
        );
    }
}

// ----------------------------------------------------------------------------
// Complex-to-real pre-processing (before inverse FFT)
// Input:  n/2+1 complex values in half-complex format
// Output: n complex values placed in tmp (which will then be inverse FFT'd)
// ----------------------------------------------------------------------------
template<typename T>
static void unpack_real_inverse(const std::complex<T>* in, std::complex<T>* tmp, size_t n) {
    const size_t nh = n / 2;

    tmp[0] = std::complex<T>(in[0].real(), T(0));
    if (n % 2 == 0) {
        tmp[nh] = std::complex<T>(in[nh].real(), T(0));
    } else {
        tmp[nh] = in[nh];
    }

    for (size_t k = 1; k < nh; ++k) {
        const auto& val = in[k];
        tmp[k] = std::complex<T>(val.real(), val.imag());
        tmp[n - k] = std::complex<T>(val.real(), -val.imag());
    }
}

// ----------------------------------------------------------------------------
// Optimized R2C execution using a single complex FFT + packing
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::rfft_impl(const T* in, Complex* out) const {
    const size_t n = length;
    if (work.size() < n) {
        const_cast<std::vector<Complex>&>(work).resize(n);
    }

    // Copy real input to complex work array
    for (size_t i = 0; i < n; ++i) {
        work[i] = Complex(in[i], T(0));
    }

    // Forward complex FFT
    execute_c2c(work.data());

    // Pack into half-complex format
    pack_real_forward(work.data(), out, n);
}

// ----------------------------------------------------------------------------
// Optimized C2R execution using unpack + inverse FFT
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::irfft_impl(const Complex* in, T* out) const {
    const size_t n = length;
    if (work.size() < n) {
        const_cast<std::vector<Complex>&>(work).resize(n);
    }

    // Unpack half-complex to full complex array
    unpack_real_inverse(in, work.data(), n);

    // Inverse complex FFT (with scaling)
    execute_c2c(work.data());

    // Extract real part
    for (size_t i = 0; i < n; ++i) {
        out[i] = work[i].real();
    }
}

// Explicit instantiations for both precisions
template void PlanImpl<float>::rfft_impl(const float*, Complex*) const;
template void PlanImpl<double>::rfft_impl(const double*, Complex*) const;
template void PlanImpl<float>::irfft_impl(const Complex*, float*) const;
template void PlanImpl<double>::irfft_impl(const Complex*, double*) const;

} // namespace detail
} // namespace fastfft