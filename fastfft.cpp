// src/fastfft.cpp

#include "fastfft/fastfft.hpp"
#include "fft_plan.hpp"
#include <cstring>
#include <new>

namespace fastfft {

using detail::PlanImpl;

// ----------------------------------------------------------------------------
// Plan::Impl definition (opaque) - already declared in fft_plan.hpp
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Plan creation helpers
// ----------------------------------------------------------------------------
static std::unique_ptr<Plan::Impl> make_plan_impl(size_t length, TransformType type, Direction dir) {
    auto impl = std::make_unique<Plan::Impl>();
    impl->length = length;
    impl->type = type;
    impl->dir = dir;

    // Build the float plan immediately (double will be lazy)
    impl->f32_plan = std::make_unique<PlanImpl<float>>();
    switch (type) {
        case TransformType::C2C:
            impl->f32_plan->build_c2c(length, static_cast<int>(dir));
            break;
        case TransformType::R2C:
            impl->f32_plan->build_r2c(length);
            break;
        case TransformType::C2R:
            impl->f32_plan->build_c2r(length);
            break;
    }
    return impl;
}

std::unique_ptr<Plan> Plan::create_c2c(size_t length, Direction dir, unsigned /*flags*/) {
    if (length == 0) {
#ifndef FASTFFT_NO_EXCEPTIONS
        throw Exception(ErrorCode::InvalidSize, "FFT length must be positive");
#else
        return nullptr;
#endif
    }
    auto plan = std::unique_ptr<Plan>(new Plan());
    plan->impl_ = make_plan_impl(length, TransformType::C2C, dir);
    return plan;
}

std::unique_ptr<Plan> Plan::create_r2c(size_t length, unsigned /*flags*/) {
    if (length == 0) {
#ifndef FASTFFT_NO_EXCEPTIONS
        throw Exception(ErrorCode::InvalidSize, "FFT length must be positive");
#else
        return nullptr;
#endif
    }
    auto plan = std::unique_ptr<Plan>(new Plan());
    plan->impl_ = make_plan_impl(length, TransformType::R2C, Direction::Forward);
    return plan;
}

std::unique_ptr<Plan> Plan::create_c2r(size_t length, unsigned /*flags*/) {
    if (length == 0) {
#ifndef FASTFFT_NO_EXCEPTIONS
        throw Exception(ErrorCode::InvalidSize, "FFT length must be positive");
#else
        return nullptr;
#endif
    }
    auto plan = std::unique_ptr<Plan>(new Plan());
    plan->impl_ = make_plan_impl(length, TransformType::C2R, Direction::Backward);
    return plan;
}

// ----------------------------------------------------------------------------
// Plan destructor and move operations
// ----------------------------------------------------------------------------
Plan::~Plan() = default;

Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;

// ----------------------------------------------------------------------------
// Lazy double-precision plan initialization
// ----------------------------------------------------------------------------
static void ensure_f64_plan(Plan::Impl* impl) {
    if (impl->f64_plan) return;
    impl->f64_plan = std::make_unique<PlanImpl<double>>();
    switch (impl->type) {
        case TransformType::C2C:
            impl->f64_plan->build_c2c(impl->length, static_cast<int>(impl->dir));
            break;
        case TransformType::R2C:
            impl->f64_plan->build_r2c(impl->length);
            break;
        case TransformType::C2R:
            impl->f64_plan->build_c2r(impl->length);
            break;
    }
}

// ----------------------------------------------------------------------------
// Execution methods (float)
// ----------------------------------------------------------------------------
void Plan::execute(const std::complex<float>* in, std::complex<float>* out) const {
    if (!impl_ || !impl_->f32_plan) return;
    impl_->f32_plan->execute_c2c(in, out);
}

void Plan::execute(const float* in, std::complex<float>* out) const {
    if (!impl_ || !impl_->f32_plan) return;
    impl_->f32_plan->execute_r2c(in, out);
}

void Plan::execute(const std::complex<float>* in, float* out) const {
    if (!impl_ || !impl_->f32_plan) return;
    impl_->f32_plan->execute_c2r(in, out);
}

void Plan::execute(std::complex<float>* data) const {
    if (!impl_ || !impl_->f32_plan) return;
    impl_->f32_plan->execute_c2c(data);
}

// ----------------------------------------------------------------------------
// Execution methods (double)
// ----------------------------------------------------------------------------
void Plan::execute(const std::complex<double>* in, std::complex<double>* out) const {
    if (!impl_) return;
    ensure_f64_plan(impl_.get());
    impl_->f64_plan->execute_c2c(in, out);
}

void Plan::execute(const double* in, std::complex<double>* out) const {
    if (!impl_) return;
    ensure_f64_plan(impl_.get());
    impl_->f64_plan->execute_r2c(in, out);
}

void Plan::execute(const std::complex<double>* in, double* out) const {
    if (!impl_) return;
    ensure_f64_plan(impl_.get());
    impl_->f64_plan->execute_c2r(in, out);
}

void Plan::execute(std::complex<double>* data) const {
    if (!impl_) return;
    ensure_f64_plan(impl_.get());
    impl_->f64_plan->execute_c2c(data);
}

// ----------------------------------------------------------------------------
// C++20 span support (if available)
// ----------------------------------------------------------------------------
#ifdef FASTFFT_HAS_SPAN
void Plan::execute(std::span<const std::complex<float>> in, std::span<std::complex<float>> out) const {
    if (in.size() != size() || out.size() != size()) return;
    execute(in.data(), out.data());
}
void Plan::execute(std::span<const std::complex<double>> in, std::span<std::complex<double>> out) const {
    if (in.size() != size() || out.size() != size()) return;
    execute(in.data(), out.data());
}
void Plan::execute(std::span<const float> in, std::span<std::complex<float>> out) const {
    if (in.size() != size() || out.size() != size()/2 + 1) return;
    execute(in.data(), out.data());
}
void Plan::execute(std::span<const double> in, std::span<std::complex<double>> out) const {
    if (in.size() != size() || out.size() != size()/2 + 1) return;
    execute(in.data(), out.data());
}
void Plan::execute(std::span<const std::complex<float>> in, std::span<float> out) const {
    if (in.size() != size()/2 + 1 || out.size() != size()) return;
    execute(in.data(), out.data());
}
void Plan::execute(std::span<const std::complex<double>> in, std::span<double> out) const {
    if (in.size() != size()/2 + 1 || out.size() != size()) return;
    execute(in.data(), out.data());
}
void Plan::execute(std::span<std::complex<float>> data) const {
    if (data.size() != size()) return;
    execute(data.data());
}
void Plan::execute(std::span<std::complex<double>> data) const {
    if (data.size() != size()) return;
    execute(data.data());
}
#endif

// ----------------------------------------------------------------------------
// Query methods
// ----------------------------------------------------------------------------
size_t Plan::size() const noexcept {
    return impl_ ? impl_->length : 0;
}

TransformType Plan::type() const noexcept {
    return impl_ ? impl_->type : TransformType::C2C;
}

Direction Plan::direction() const noexcept {
    return impl_ ? impl_->dir : Direction::Forward;
}

size_t Plan::required_alignment() noexcept {
    return simd::alignment;
}

// ----------------------------------------------------------------------------
// One-shot convenience functions
// ----------------------------------------------------------------------------
void fft(const std::complex<float>* in, std::complex<float>* out, size_t length, Direction dir) {
    auto plan = Plan::create_c2c(length, dir);
    if (plan) plan->execute(in, out);
}

void fft(const std::complex<double>* in, std::complex<double>* out, size_t length, Direction dir) {
    auto plan = Plan::create_c2c(length, dir);
    if (plan) plan->execute(in, out);
}

void rfft(const float* in, std::complex<float>* out, size_t length) {
    auto plan = Plan::create_r2c(length);
    if (plan) plan->execute(in, out);
}

void rfft(const double* in, std::complex<double>* out, size_t length) {
    auto plan = Plan::create_r2c(length);
    if (plan) plan->execute(in, out);
}

void irfft(const std::complex<float>* in, float* out, size_t length) {
    auto plan = Plan::create_c2r(length);
    if (plan) plan->execute(in, out);
}

void irfft(const std::complex<double>* in, double* out, size_t length) {
    auto plan = Plan::create_c2r(length);
    if (plan) plan->execute(in, out);
}

// ----------------------------------------------------------------------------
// Utility functions
// ----------------------------------------------------------------------------
size_t next_good_size(size_t n) {
    return detail::next_good_size(n);
}

const char* version() noexcept {
    return "FastFFT 1.0.0";
}

} // namespace fastfft