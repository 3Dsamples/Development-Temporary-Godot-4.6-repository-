// src/fft_plan.cpp

#include "fft_plan.hpp"
#include "fft_kernels.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fastfft {
namespace detail {

using kernels::small_fft_dispatch;

// ----------------------------------------------------------------------------
// Plan building: complex-to-complex
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::build_c2c(size_t n, int dir) {
    length = n;
    sign = dir;
    transform_type = TransformType::C2C;
    factors = factorize(n);

    // If the length is a large prime, use Bluestein's algorithm
    if (factors.size() == 1 && factors[0].second == 1 && n > 13) {
        // Prepare Bluestein convolution
        bluestein = std::make_unique<BluesteinData>();
        bluestein->n2 = 1;
        while (bluestein->n2 < 2 * n - 1) {
            bluestein->n2 *= 2;
        }

        bluestein->chirp.resize(n);
        bluestein->w.resize(bluestein->n2);
        bluestein->tmp.resize(bluestein->n2);

        const T two_pi = T(2.0 * M_PI) * T(sign);
        for (size_t i = 0; i < n; ++i) {
            T phase = two_pi * T((i * i) % (2 * n)) / T(2 * n);
            bluestein->chirp[i] = Complex(std::cos(phase), std::sin(phase));
        }

        // Build sub-plan for power-of-two FFT (used in convolution)
        bluestein->subplan = new PlanImpl<T>();
        bluestein->subplan->build_c2c(bluestein->n2, -1); // forward for convolution
        return;
    }

    // Otherwise, generate twiddle factors for mixed-radix decomposition
    generate_twiddles();
    work.resize(n);
}

template<typename T>
void PlanImpl<T>::build_r2c(size_t n) {
    build_c2c(n, -1); // forward transform
    transform_type = TransformType::R2C;
}

template<typename T>
void PlanImpl<T>::build_c2r(size_t n) {
    build_c2c(n, 1);  // backward transform
    transform_type = TransformType::C2R;
}

// ----------------------------------------------------------------------------
// Twiddle factor generation (recursive product of factors)
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::generate_twiddles() {
    // Compute total size needed: product of (factor - 1) over all factors
    size_t tw_size = 0;
    size_t prod = 1;
    for (const auto& factor_pair : factors) {
        size_t p = factor_pair.first;
        size_t e = factor_pair.second;
        for (size_t i = 0; i < e; ++i) {
            prod *= p;
            tw_size += (p - 1) * (length / prod);
        }
    }
    twiddle.resize(2 * tw_size); // interleaved cos/sin

    // Fill recursively (similar to PocketFFT's fill_twiddle)
    T* tw = twiddle.data();
    size_t n = length;
    for (const auto& factor_pair : factors) {
        size_t p = factor_pair.first;
        size_t e = factor_pair.second;
        for (size_t i = 0; i < e; ++i) {
            size_t m = n / p;
            for (size_t j = 0; j < p - 1; ++j) {
                size_t idx = j + 1;
                T angle = T(-2.0 * M_PI) * T(idx) / T(p);
                T c = std::cos(angle);
                T s = std::sin(angle);
                for (size_t k = 0; k < m; ++k) {
                    tw[0] = c;
                    tw[1] = s;
                    tw += 2;
                }
            }
            n = m;
        }
    }
}

// ----------------------------------------------------------------------------
// Mixed-radix FFT execution (recursive factorization)
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::factor_step(Complex* data, size_t stride, size_t n, const T* tw, size_t tw_stride) const {
    if (n == 1) return;

    // Find the largest prime factor <= n (for best locality)
    size_t p = 2;
    for (size_t cand : {2, 3, 4, 5, 7, 8, 11}) {
        if (n % cand == 0) {
            p = cand;
            break;
        }
    }
    if (p == 2 && n % 2 != 0) {
        // Fallback to smallest factor
        for (size_t cand = 3; cand * cand <= n; cand += 2) {
            if (n % cand == 0) {
                p = cand;
                break;
            }
        }
        if (p == 2) p = n; // n is prime (should have been caught earlier)
    }

    size_t m = n / p;

    // Reorder data using transposition (if p is small, we can do direct loops)
    if (p <= 11) {
        // Use optimized codelet for small radices
        for (size_t i = 0; i < m; ++i) {
            // Apply twiddle factors to the sub-transforms
            Complex* block = data + i * p * stride;
            if (i > 0) {
                const T* tw_i = tw + 2 * (i - 1) * tw_stride;
                for (size_t j = 1; j < p; ++j) {
                    Complex& val = block[j * stride];
                    T wr = tw_i[0];
                    T wi = tw_i[1];
                    val = Complex(val.real() * wr - val.imag() * wi,
                                  val.real() * wi + val.imag() * wr);
                    tw_i += 2;
                }
            }
            // Perform radix-p DFT on this block
            small_fft_dispatch<T>(block, stride, p, p * stride);
        }
    } else {
        // Generic radix with explicit loops
        std::vector<Complex> tmp(p);
        for (size_t i = 0; i < m; ++i) {
            Complex* block = data + i * p * stride;
            // Gather with twiddle multiplication
            for (size_t j = 0; j < p; ++j) {
                if (i == 0 || j == 0) {
                    tmp[j] = block[j * stride];
                } else {
                    const T* tw_ij = tw + 2 * ((i * (p - 1)) + (j - 1)) * tw_stride;
                    T wr = tw_ij[0];
                    T wi = tw_ij[1];
                    const Complex& val = block[j * stride];
                    tmp[j] = Complex(val.real() * wr - val.imag() * wi,
                                     val.real() * wi + val.imag() * wr);
                }
            }
            // DFT (scalar for generic radix)
            for (size_t j = 0; j < p; ++j) {
                Complex sum = 0;
                T angle = T(-2.0 * M_PI) * T(j) / T(p);
                T wkr = std::cos(angle);
                T wki = std::sin(angle);
                T wr = 1.0;
                T wi = 0.0;
                for (size_t k = 0; k < p; ++k) {
                    sum += tmp[k] * Complex(wr, wi);
                    T nwr = wr * wkr - wi * wki;
                    T nwi = wr * wki + wi * wkr;
                    wr = nwr;
                    wi = nwi;
                }
                block[j * stride] = sum;
            }
        }
    }

    // Recurse on the m sub-transforms
    for (size_t j = 0; j < p; ++j) {
        factor_step(data + j * stride, p * stride, m, tw + (p - 1) * tw_stride, tw_stride);
    }
}

template<typename T>
void PlanImpl<T>::fft_mixed_radix(Complex* data, size_t stride, size_t n, const T* tw) const {
    factor_step(data, stride, n, tw, 1);
}

// ----------------------------------------------------------------------------
// Bluestein's algorithm for prime lengths
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::fft_bluestein(const Complex* in, Complex* out) const {
    if (!bluestein) return;
    const size_t n = length;
    const size_t n2 = bluestein->n2;
    auto& chirp = bluestein->chirp;
    auto& w = bluestein->w;
    auto& tmp = bluestein->tmp;
    auto* subplan = bluestein->subplan;

    // Step 1: multiply input by chirp and pad to n2
    for (size_t i = 0; i < n2; ++i) {
        w[i] = Complex(0, 0);
    }
    for (size_t i = 0; i < n; ++i) {
        w[i] = in[i] * std::conj(chirp[i]);
    }

    // Step 2: forward FFT of size n2
    subplan->execute_c2c(w.data(), tmp.data());

    // Step 3: multiply by chirp spectrum
    std::vector<Complex> chirp_ext(n2, Complex(0, 0));
    for (size_t i = 0; i < n; ++i) {
        chirp_ext[i] = chirp[i];
    }
    for (size_t i = 1; i < n; ++i) {
        chirp_ext[n2 - i] = std::conj(chirp[i]);
    }
    subplan->execute_c2c(chirp_ext.data(), w.data()); // reuse w

    // Step 4: multiply tmp and w pointwise
    for (size_t i = 0; i < n2; ++i) {
        tmp[i] = tmp[i] * w[i];
    }

    // Step 5: inverse FFT
    subplan->sign = 1;
    subplan->execute_c2c(tmp.data(), w.data());
    subplan->sign = -1; // restore

    // Step 6: multiply by conj(chirp) and scale
    T scale = T(1) / T(n2);
    for (size_t i = 0; i < n; ++i) {
        out[i] = w[i] * std::conj(chirp[i]) * scale;
    }
}

// ----------------------------------------------------------------------------
// Execution entry points
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::execute_c2c(const Complex* in, Complex* out) const {
    if (bluestein) {
        fft_bluestein(in, out);
        return;
    }

    // Copy to work buffer
    for (size_t i = 0; i < length; ++i) {
        work[i] = in[i];
    }
    execute_c2c(work.data());
    for (size_t i = 0; i < length; ++i) {
        out[i] = work[i];
    }
}

template<typename T>
void PlanImpl<T>::execute_c2c(Complex* data) const {
    if (bluestein) {
        // In-place Bluestein requires temporary, use work
        if (work.size() < length) {
            const_cast<std::vector<Complex>&>(work).resize(length);
        }
        for (size_t i = 0; i < length; ++i) {
            work[i] = data[i];
        }
        fft_bluestein(work.data(), data);
        return;
    }

    // Mixed-radix FFT
    fft_mixed_radix(data, 1, length, twiddle.data());

    // If backward transform, apply scaling
    if (sign == 1) {
        T scale = T(1) / T(length);
        for (size_t i = 0; i < length; ++i) {
            data[i] = data[i] * scale;
        }
    }
}

// ----------------------------------------------------------------------------
// Real transform implementations (calls rfft_impl and irfft_impl)
// ----------------------------------------------------------------------------
template<typename T>
void PlanImpl<T>::execute_r2c(const T* in, Complex* out) const {
    rfft_impl(in, out);
}

template<typename T>
void PlanImpl<T>::execute_c2r(const Complex* in, T* out) const {
    irfft_impl(in, out);
}

// ----------------------------------------------------------------------------
// Explicit template instantiations
// ----------------------------------------------------------------------------
template class PlanImpl<float>;
template class PlanImpl<double>;

} // namespace detail
} // namespace fastfft