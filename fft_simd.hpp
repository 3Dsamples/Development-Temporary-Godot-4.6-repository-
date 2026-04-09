// src/fft_simd.hpp
#pragma once

// Unified SIMD abstraction layer for FastFFT
// Combines the best of PFFFT's SIMD wrappers and PocketFFT's vectorization

#include <cstdint>
#include <cstddef>
#include <cmath>

#if defined(__AVX2__) || defined(FASTFFT_USE_AVX2)
  #define FASTFFT_SIMD_AVX2 1
  #include <immintrin.h>
#elif defined(__AVX__) || defined(FASTFFT_USE_AVX)
  #define FASTFFT_SIMD_AVX 1
  #include <immintrin.h>
#elif defined(__SSE__) || defined(__SSE2__) || defined(FASTFFT_USE_SSE)
  #define FASTFFT_SIMD_SSE 1
  #include <xmmintrin.h>
  #include <emmintrin.h>
#else
  #define FASTFFT_SIMD_NONE 1
#endif

namespace fastfft {
namespace simd {

// ============================================================================
// Common type definitions and constants
// ============================================================================

// Alignment requirements
#ifdef FASTFFT_SIMD_AVX2
  constexpr size_t alignment = 32;
#elif defined(FASTFFT_SIMD_AVX)
  constexpr size_t alignment = 32;
#elif defined(FASTFFT_SIMD_SSE)
  constexpr size_t alignment = 16;
#else
  constexpr size_t alignment = sizeof(void*);
#endif

// ============================================================================
// Single-precision (float) SIMD vector
// ============================================================================

#ifdef FASTFFT_SIMD_AVX2
  // AVX2: 8 floats per vector
  struct vfloat {
      __m256 v;
      vfloat() : v(_mm256_setzero_ps()) {}
      vfloat(__m256 x) : v(x) {}
      operator __m256() const { return v; }
      static vfloat zero() { return _mm256_setzero_ps(); }
      static vfloat load(const float* p) { return _mm256_loadu_ps(p); }
      static vfloat load_aligned(const float* p) { return _mm256_load_ps(p); }
      static vfloat set1(float x) { return _mm256_set1_ps(x); }
      void store(float* p) const { _mm256_storeu_ps(p, v); }
      void store_aligned(float* p) const { _mm256_store_ps(p, v); }
  };
#elif defined(FASTFFT_SIMD_AVX)
  // AVX: 8 floats per vector
  struct vfloat {
      __m256 v;
      vfloat() : v(_mm256_setzero_ps()) {}
      vfloat(__m256 x) : v(x) {}
      operator __m256() const { return v; }
      static vfloat zero() { return _mm256_setzero_ps(); }
      static vfloat load(const float* p) { return _mm256_loadu_ps(p); }
      static vfloat load_aligned(const float* p) { return _mm256_load_ps(p); }
      static vfloat set1(float x) { return _mm256_set1_ps(x); }
      void store(float* p) const { _mm256_storeu_ps(p, v); }
      void store_aligned(float* p) const { _mm256_store_ps(p, v); }
  };
#elif defined(FASTFFT_SIMD_SSE)
  // SSE/SSE2: 4 floats per vector
  struct vfloat {
      __m128 v;
      vfloat() : v(_mm_setzero_ps()) {}
      vfloat(__m128 x) : v(x) {}
      operator __m128() const { return v; }
      static vfloat zero() { return _mm_setzero_ps(); }
      static vfloat load(const float* p) { return _mm_loadu_ps(p); }
      static vfloat load_aligned(const float* p) { return _mm_load_ps(p); }
      static vfloat set1(float x) { return _mm_set1_ps(x); }
      void store(float* p) const { _mm_storeu_ps(p, v); }
      void store_aligned(float* p) const { _mm_store_ps(p, v); }
  };
#else
  // Scalar fallback
  struct vfloat {
      float v;
      vfloat() : v(0.0f) {}
      vfloat(float x) : v(x) {}
      operator float() const { return v; }
      static vfloat zero() { return 0.0f; }
      static vfloat load(const float* p) { return *p; }
      static vfloat load_aligned(const float* p) { return *p; }
      static vfloat set1(float x) { return x; }
      void store(float* p) const { *p = v; }
      void store_aligned(float* p) const { *p = v; }
  };
#endif

// ----------------------------------------------------------------------------
// Arithmetic operators for vfloat
// ----------------------------------------------------------------------------
inline vfloat operator+(vfloat a, vfloat b) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_add_ps(a.v, b.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_add_ps(a.v, b.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_add_ps(a.v, b.v);
#else
    return a.v + b.v;
#endif
}

inline vfloat operator-(vfloat a, vfloat b) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_sub_ps(a.v, b.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_sub_ps(a.v, b.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_sub_ps(a.v, b.v);
#else
    return a.v - b.v;
#endif
}

inline vfloat operator*(vfloat a, vfloat b) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_mul_ps(a.v, b.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_mul_ps(a.v, b.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_mul_ps(a.v, b.v);
#else
    return a.v * b.v;
#endif
}

// Fused multiply-add: a * b + c
inline vfloat fmadd(vfloat a, vfloat b, vfloat c) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_fmadd_ps(a.v, b.v, c.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_add_ps(_mm256_mul_ps(a.v, b.v), c.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v);
#else
    return a.v * b.v + c.v;
#endif
}

// Fused multiply-subtract: a * b - c
inline vfloat fmsub(vfloat a, vfloat b, vfloat c) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_fmsub_ps(a.v, b.v, c.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_sub_ps(_mm256_mul_ps(a.v, b.v), c.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_sub_ps(_mm_mul_ps(a.v, b.v), c.v);
#else
    return a.v * b.v - c.v;
#endif
}

// Horizontal sum of all elements in a vector
inline float hadd(vfloat x) {
#ifdef FASTFFT_SIMD_AVX2
    __m256 t1 = _mm256_hadd_ps(x.v, x.v);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 lo = _mm256_castps256_ps128(t2);
    __m128 hi = _mm256_extractf128_ps(t2, 1);
    __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
#elif defined(FASTFFT_SIMD_AVX)
    __m256 t1 = _mm256_hadd_ps(x.v, x.v);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 lo = _mm256_castps256_ps128(t2);
    __m128 hi = _mm256_extractf128_ps(t2, 1);
    __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
#elif defined(FASTFFT_SIMD_SSE)
    __m128 t1 = _mm_hadd_ps(x.v, x.v);
    __m128 t2 = _mm_hadd_ps(t1, t1);
    return _mm_cvtss_f32(t2);
#else
    return x.v;
#endif
}

// Complex multiplication: (ar,ai) * (br,bi) -> (ar*br - ai*bi, ar*bi + ai*br)
// For SIMD, we interleave operations on packed real and imaginary parts
inline void cmul(vfloat ar, vfloat ai, vfloat br, vfloat bi, vfloat& out_r, vfloat& out_i) {
    out_r = fmsub(ar, br, ai * bi);
    out_i = fmadd(ar, bi, ai * br);
}

// ============================================================================
// Double-precision (double) SIMD vector
// ============================================================================

#ifdef FASTFFT_SIMD_AVX2
  // AVX2: 4 doubles per vector
  struct vdouble {
      __m256d v;
      vdouble() : v(_mm256_setzero_pd()) {}
      vdouble(__m256d x) : v(x) {}
      operator __m256d() const { return v; }
      static vdouble zero() { return _mm256_setzero_pd(); }
      static vdouble load(const double* p) { return _mm256_loadu_pd(p); }
      static vdouble load_aligned(const double* p) { return _mm256_load_pd(p); }
      static vdouble set1(double x) { return _mm256_set1_pd(x); }
      void store(double* p) const { _mm256_storeu_pd(p, v); }
      void store_aligned(double* p) const { _mm256_store_pd(p, v); }
  };
#elif defined(FASTFFT_SIMD_AVX)
  // AVX: 4 doubles per vector
  struct vdouble {
      __m256d v;
      vdouble() : v(_mm256_setzero_pd()) {}
      vdouble(__m256d x) : v(x) {}
      operator __m256d() const { return v; }
      static vdouble zero() { return _mm256_setzero_pd(); }
      static vdouble load(const double* p) { return _mm256_loadu_pd(p); }
      static vdouble load_aligned(const double* p) { return _mm256_load_pd(p); }
      static vdouble set1(double x) { return _mm256_set1_pd(x); }
      void store(double* p) const { _mm256_storeu_pd(p, v); }
      void store_aligned(double* p) const { _mm256_store_pd(p, v); }
  };
#elif defined(FASTFFT_SIMD_SSE)
  // SSE2: 2 doubles per vector
  struct vdouble {
      __m128d v;
      vdouble() : v(_mm_setzero_pd()) {}
      vdouble(__m128d x) : v(x) {}
      operator __m128d() const { return v; }
      static vdouble zero() { return _mm_setzero_pd(); }
      static vdouble load(const double* p) { return _mm_loadu_pd(p); }
      static vdouble load_aligned(const double* p) { return _mm_load_pd(p); }
      static vdouble set1(double x) { return _mm_set1_pd(x); }
      void store(double* p) const { _mm_storeu_pd(p, v); }
      void store_aligned(double* p) const { _mm_store_pd(p, v); }
  };
#else
  // Scalar fallback
  struct vdouble {
      double v;
      vdouble() : v(0.0) {}
      vdouble(double x) : v(x) {}
      operator double() const { return v; }
      static vdouble zero() { return 0.0; }
      static vdouble load(const double* p) { return *p; }
      static vdouble load_aligned(const double* p) { return *p; }
      static vdouble set1(double x) { return x; }
      void store(double* p) const { *p = v; }
      void store_aligned(double* p) const { *p = v; }
  };
#endif

// ----------------------------------------------------------------------------
// Arithmetic operators for vdouble
// ----------------------------------------------------------------------------
inline vdouble operator+(vdouble a, vdouble b) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_add_pd(a.v, b.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_add_pd(a.v, b.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_add_pd(a.v, b.v);
#else
    return a.v + b.v;
#endif
}

inline vdouble operator-(vdouble a, vdouble b) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_sub_pd(a.v, b.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_sub_pd(a.v, b.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_sub_pd(a.v, b.v);
#else
    return a.v - b.v;
#endif
}

inline vdouble operator*(vdouble a, vdouble b) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_mul_pd(a.v, b.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_mul_pd(a.v, b.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_mul_pd(a.v, b.v);
#else
    return a.v * b.v;
#endif
}

inline vdouble fmadd(vdouble a, vdouble b, vdouble c) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_fmadd_pd(a.v, b.v, c.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_add_pd(_mm256_mul_pd(a.v, b.v), c.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_add_pd(_mm_mul_pd(a.v, b.v), c.v);
#else
    return a.v * b.v + c.v;
#endif
}

inline vdouble fmsub(vdouble a, vdouble b, vdouble c) {
#ifdef FASTFFT_SIMD_AVX2
    return _mm256_fmsub_pd(a.v, b.v, c.v);
#elif defined(FASTFFT_SIMD_AVX)
    return _mm256_sub_pd(_mm256_mul_pd(a.v, b.v), c.v);
#elif defined(FASTFFT_SIMD_SSE)
    return _mm_sub_pd(_mm_mul_pd(a.v, b.v), c.v);
#else
    return a.v * b.v - c.v;
#endif
}

inline double hadd(vdouble x) {
#ifdef FASTFFT_SIMD_AVX2
    __m256d t1 = _mm256_hadd_pd(x.v, x.v);
    __m128d lo = _mm256_castpd256_pd128(t1);
    __m128d hi = _mm256_extractf128_pd(t1, 1);
    __m128d sum = _mm_add_sd(lo, hi);
    return _mm_cvtsd_f64(sum);
#elif defined(FASTFFT_SIMD_AVX)
    __m256d t1 = _mm256_hadd_pd(x.v, x.v);
    __m128d lo = _mm256_castpd256_pd128(t1);
    __m128d hi = _mm256_extractf128_pd(t1, 1);
    __m128d sum = _mm_add_sd(lo, hi);
    return _mm_cvtsd_f64(sum);
#elif defined(FASTFFT_SIMD_SSE)
    __m128d t1 = _mm_hadd_pd(x.v, x.v);
    return _mm_cvtsd_f64(t1);
#else
    return x.v;
#endif
}

inline void cmul(vdouble ar, vdouble ai, vdouble br, vdouble bi, vdouble& out_r, vdouble& out_i) {
    out_r = fmsub(ar, br, ai * bi);
    out_i = fmadd(ar, bi, ai * br);
}

// ============================================================================
// Vector size and alignment utilities
// ============================================================================

template<typename T> struct simd_traits;

template<> struct simd_traits<float> {
    using vtype = vfloat;
#ifdef FASTFFT_SIMD_AVX2
    static constexpr size_t width = 8;
#elif defined(FASTFFT_SIMD_AVX)
    static constexpr size_t width = 8;
#elif defined(FASTFFT_SIMD_SSE)
    static constexpr size_t width = 4;
#else
    static constexpr size_t width = 1;
#endif
};

template<> struct simd_traits<double> {
    using vtype = vdouble;
#ifdef FASTFFT_SIMD_AVX2
    static constexpr size_t width = 4;
#elif defined(FASTFFT_SIMD_AVX)
    static constexpr size_t width = 4;
#elif defined(FASTFFT_SIMD_SSE)
    static constexpr size_t width = 2;
#else
    static constexpr size_t width = 1;
#endif
};

// ----------------------------------------------------------------------------
// Utility to check if a pointer is aligned
// ----------------------------------------------------------------------------
inline bool is_aligned(const void* ptr, size_t align = alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (align - 1)) == 0;
}

} // namespace simd
} // namespace fastfft