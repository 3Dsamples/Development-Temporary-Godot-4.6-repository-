// src/fft_plan.hpp
#pragma once

#include "fft_simd.hpp"
#include "fft_utils.hpp"
#include "../include/fastfft/fastfft.hpp"
#include <cstddef>
#include <vector>
#include <complex>
#include <memory>

namespace fastfft {
namespace detail {

// ----------------------------------------------------------------------------
// Plan implementation for a specific precision (float or double)
// ----------------------------------------------------------------------------
template<typename T>
class PlanImpl {
public:
    using Complex = std::complex<T>;

    // Transform direction (forward = -1, backward = +1)
    int sign;

    // Total transform length
    size_t length;

    // Type of transform (C2C, R2C, C2R)
    TransformType transform_type;

    // Factorization of length (pairs of {prime, exponent})
    std::vector<std::pair<size_t, size_t>> factors;

    // Twiddle factors storage (interleaved real/imag)
    std::vector<T> twiddle;

    // Work buffer for in-place transforms (size = length)
    std::vector<Complex> work;

    // Bluestein-specific data (for large prime lengths)
    struct BluesteinData {
        size_t n2;                     // Next power of two >= 2*length - 1
        std::vector<Complex> chirp;    // Chirp sequence
        std::vector<Complex> w;        // Twiddle factors for convolution
        std::vector<Complex> tmp;      // Temporary buffer
        PlanImpl<T>* subplan;          // Sub-plan for power-of-two FFT
        ~BluesteinData() { delete subplan; }
    };
    std::unique_ptr<BluesteinData> bluestein;

    // Constructor (empty plan)
    PlanImpl() : sign(1), length(0), transform_type(TransformType::C2C) {}

    // Build a complex-to-complex plan
    void build_c2c(size_t n, int dir);

    // Build a real-to-complex plan
    void build_r2c(size_t n);

    // Build a complex-to-real plan
    void build_c2r(size_t n);

    // Execute complex-to-complex transform (in-place or out-of-place)
    void execute_c2c(const Complex* in, Complex* out) const;
    void execute_c2c(Complex* data) const;

    // Execute real-to-complex transform
    void execute_r2c(const T* in, Complex* out) const;

    // Execute complex-to-real transform
    void execute_c2r(const Complex* in, T* out) const;

    // Internal: mixed-radix FFT using precomputed factors and twiddles
    void fft_mixed_radix(Complex* data, size_t stride, size_t n, const T* tw) const;

    // Internal: Bluestein's algorithm for prime lengths
    void fft_bluestein(const Complex* in, Complex* out) const;

    // Internal: real FFT via half-complex packing
    void rfft_impl(const T* in, Complex* out) const;
    void irfft_impl(const Complex* in, T* out) const;

private:
    // Generate twiddle factors for a given factor list
    void generate_twiddles();

    // Recursively apply factor steps
    void factor_step(Complex* data, size_t stride, size_t n, const T* tw, size_t tw_stride) const;
};

// ----------------------------------------------------------------------------
// Opaque plan type exposed in public API
// ----------------------------------------------------------------------------
struct Plan::Impl {
    std::unique_ptr<PlanImpl<float>>  f32_plan;
    std::unique_ptr<PlanImpl<double>> f64_plan;
    TransformType type;
    Direction dir;
    size_t length;

    Impl() : type(TransformType::C2C), dir(Direction::Forward), length(0) {}
};

} // namespace detail
} // namespace fastfft