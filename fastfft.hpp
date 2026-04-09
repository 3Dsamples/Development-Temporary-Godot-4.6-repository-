// include/fastfft/fastfft.hpp
#pragma once

#include <cstddef>
#include <complex>
#include <vector>
#include <memory>

#ifdef __has_include
#  if __has_include(<span>)
#    include <span>
#    define FASTFFT_HAS_SPAN 1
#  endif
#endif

namespace fastfft {

// Forward declarations
class Plan;
enum class Direction : int { Forward = -1, Backward = 1 };
enum class TransformType : int { C2C, R2C, C2R };

// ----------------------------------------------------------------------
// Error handling (lightweight, exception-free option available)
// ----------------------------------------------------------------------
enum class ErrorCode : int {
    Success = 0,
    InvalidSize,
    InvalidAlignment,
    OutOfMemory,
    NotSupported,
    InternalError
};

// Optional: use exceptions if desired; otherwise return ErrorCode.
#ifndef FASTFFT_NO_EXCEPTIONS
#include <stdexcept>
class Exception : public std::runtime_error {
public:
    Exception(ErrorCode code, const char* msg) : std::runtime_error(msg), code_(code) {}
    ErrorCode code() const noexcept { return code_; }
private:
    ErrorCode code_;
};
#endif

// ----------------------------------------------------------------------
// Main FFT plan class (opaque handle)
// ----------------------------------------------------------------------
class Plan {
public:
    // Complex-to-complex plan
    static std::unique_ptr<Plan> create_c2c(size_t length, Direction dir, unsigned flags = 0);

    // Real-to-complex / complex-to-real plan
    static std::unique_ptr<Plan> create_r2c(size_t length, unsigned flags = 0);
    static std::unique_ptr<Plan> create_c2r(size_t length, unsigned flags = 0);

    // Destructor
    ~Plan();

    // Move-only semantics
    Plan(Plan&&) noexcept;
    Plan& operator=(Plan&&) noexcept;

    // No copy
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    // Execute the transform
    // Complex input/output (C2C)
    void execute(const std::complex<float>* in, std::complex<float>* out) const;
    void execute(const std::complex<double>* in, std::complex<double>* out) const;

    // Real input, complex output (R2C)
    void execute(const float* in, std::complex<float>* out) const;
    void execute(const double* in, std::complex<double>* out) const;

    // Complex input, real output (C2R)
    void execute(const std::complex<float>* in, float* out) const;
    void execute(const std::complex<double>* in, double* out) const;

    // In-place variants (C2C only)
    void execute(std::complex<float>* data) const;
    void execute(std::complex<double>* data) const;

#ifdef FASTFFT_HAS_SPAN
    // C++20 span support
    void execute(std::span<const std::complex<float>> in, std::span<std::complex<float>> out) const;
    void execute(std::span<const std::complex<double>> in, std::span<std::complex<double>> out) const;
    void execute(std::span<const float> in, std::span<std::complex<float>> out) const;
    void execute(std::span<const double> in, std::span<std::complex<double>> out) const;
    void execute(std::span<const std::complex<float>> in, std::span<float> out) const;
    void execute(std::span<const std::complex<double>> in, std::span<double> out) const;
    void execute(std::span<std::complex<float>> data) const;
    void execute(std::span<std::complex<double>> data) const;
#endif

    // Query properties
    size_t size() const noexcept;
    TransformType type() const noexcept;
    Direction direction() const noexcept;

    // Advanced: get required alignment for SIMD (usually 16 or 32 bytes)
    static size_t required_alignment() noexcept;

private:
    Plan() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ----------------------------------------------------------------------
// Simple one-shot interface (no plan reuse)
// ----------------------------------------------------------------------
void fft(const std::complex<float>* in, std::complex<float>* out, size_t length, Direction dir = Direction::Forward);
void fft(const std::complex<double>* in, std::complex<double>* out, size_t length, Direction dir = Direction::Forward);

void rfft(const float* in, std::complex<float>* out, size_t length);
void rfft(const double* in, std::complex<double>* out, size_t length);

void irfft(const std::complex<float>* in, float* out, size_t length);
void irfft(const std::complex<double>* in, double* out, size_t length);

// ----------------------------------------------------------------------
// Utility functions
// ----------------------------------------------------------------------
// Compute the next good size for optimal performance (highly composite)
size_t next_good_size(size_t n);

// Get library version
const char* version() noexcept;

} // namespace fastfft