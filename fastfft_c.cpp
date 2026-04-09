// src/fastfft_c.cpp

#include "fastfft/fastfft.h"
#include "fastfft/fastfft.hpp"
#include <complex>
#include <cstddef>

using namespace fastfft;

extern "C" {

// ----------------------------------------------------------------------------
// Helper to cast opaque handle to C++ Plan*
// ----------------------------------------------------------------------------
static inline Plan* to_plan(fastfft_plan* p) {
    return reinterpret_cast<Plan*>(p);
}

static inline const Plan* to_plan_const(const fastfft_plan* p) {
    return reinterpret_cast<const Plan*>(p);
}

// ----------------------------------------------------------------------------
// Plan creation
// ----------------------------------------------------------------------------
fastfft_plan* fastfft_plan_create_c2c(size_t length, fastfft_direction dir) {
    auto plan = Plan::create_c2c(length, static_cast<Direction>(dir));
    if (!plan) return nullptr;
    return reinterpret_cast<fastfft_plan*>(plan.release());
}

fastfft_plan* fastfft_plan_create_r2c(size_t length) {
    auto plan = Plan::create_r2c(length);
    if (!plan) return nullptr;
    return reinterpret_cast<fastfft_plan*>(plan.release());
}

fastfft_plan* fastfft_plan_create_c2r(size_t length) {
    auto plan = Plan::create_c2r(length);
    if (!plan) return nullptr;
    return reinterpret_cast<fastfft_plan*>(plan.release());
}

void fastfft_plan_destroy(fastfft_plan* plan) {
    delete to_plan(plan);
}

// ----------------------------------------------------------------------------
// Execution (single precision)
// ----------------------------------------------------------------------------
void fastfft_execute_c2c_f32(const fastfft_plan* plan, const float _Complex* in, float _Complex* out) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(reinterpret_cast<const std::complex<float>*>(in),
               reinterpret_cast<std::complex<float>*>(out));
}

void fastfft_execute_c2c_inplace_f32(const fastfft_plan* plan, float _Complex* data) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(reinterpret_cast<std::complex<float>*>(data));
}

void fastfft_execute_r2c_f32(const fastfft_plan* plan, const float* in, float _Complex* out) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(in, reinterpret_cast<std::complex<float>*>(out));
}

void fastfft_execute_c2r_f32(const fastfft_plan* plan, const float _Complex* in, float* out) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(reinterpret_cast<const std::complex<float>*>(in), out);
}

// ----------------------------------------------------------------------------
// Execution (double precision)
// ----------------------------------------------------------------------------
void fastfft_execute_c2c_f64(const fastfft_plan* plan, const double _Complex* in, double _Complex* out) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(reinterpret_cast<const std::complex<double>*>(in),
               reinterpret_cast<std::complex<double>*>(out));
}

void fastfft_execute_c2c_inplace_f64(const fastfft_plan* plan, double _Complex* data) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(reinterpret_cast<std::complex<double>*>(data));
}

void fastfft_execute_r2c_f64(const fastfft_plan* plan, const double* in, double _Complex* out) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(in, reinterpret_cast<std::complex<double>*>(out));
}

void fastfft_execute_c2r_f64(const fastfft_plan* plan, const double _Complex* in, double* out) {
    const Plan* p = to_plan_const(plan);
    if (!p) return;
    p->execute(reinterpret_cast<const std::complex<double>*>(in), out);
}

// ----------------------------------------------------------------------------
// One-shot functions (single precision)
// ----------------------------------------------------------------------------
void fastfft_fft_f32(const float _Complex* in, float _Complex* out, size_t length, fastfft_direction dir) {
    fft(reinterpret_cast<const std::complex<float>*>(in),
        reinterpret_cast<std::complex<float>*>(out),
        length, static_cast<Direction>(dir));
}

void fastfft_fft_f64(const double _Complex* in, double _Complex* out, size_t length, fastfft_direction dir) {
    fft(reinterpret_cast<const std::complex<double>*>(in),
        reinterpret_cast<std::complex<double>*>(out),
        length, static_cast<Direction>(dir));
}

void fastfft_rfft_f32(const float* in, float _Complex* out, size_t length) {
    rfft(in, reinterpret_cast<std::complex<float>*>(out), length);
}

void fastfft_rfft_f64(const double* in, double _Complex* out, size_t length) {
    rfft(in, reinterpret_cast<std::complex<double>*>(out), length);
}

void fastfft_irfft_f32(const float _Complex* in, float* out, size_t length) {
    irfft(reinterpret_cast<const std::complex<float>*>(in), out, length);
}

void fastfft_irfft_f64(const double _Complex* in, double* out, size_t length) {
    irfft(reinterpret_cast<const std::complex<double>*>(in), out, length);
}

// ----------------------------------------------------------------------------
// Query functions
// ----------------------------------------------------------------------------
size_t fastfft_plan_get_size(const fastfft_plan* plan) {
    const Plan* p = to_plan_const(plan);
    return p ? p->size() : 0;
}

fastfft_transform_type fastfft_plan_get_type(const fastfft_plan* plan) {
    const Plan* p = to_plan_const(plan);
    if (!p) return FASTFFT_C2C;
    return static_cast<fastfft_transform_type>(p->type());
}

fastfft_direction fastfft_plan_get_direction(const fastfft_plan* plan) {
    const Plan* p = to_plan_const(plan);
    if (!p) return FASTFFT_FORWARD;
    return static_cast<fastfft_direction>(p->direction());
}

size_t fastfft_alignment(void) {
    return Plan::required_alignment();
}

size_t fastfft_next_good_size(size_t n) {
    return next_good_size(n);
}

const char* fastfft_version(void) {
    return version();
}

} // extern "C"