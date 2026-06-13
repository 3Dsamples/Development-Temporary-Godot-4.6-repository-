// system name : onetbb-warp
// File 0049 : core/math/simd_wrapper.h
// Description : Cross‑platform SIMD abstraction for SSE, AVX, NEON with 4‑wide float and double operations.

#ifndef __TBB_WARP_CORE_MATH_SIMD_WRAPPER_H
#define __TBB_WARP_CORE_MATH_SIMD_WRAPPER_H

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

#if defined(__AVX512F__)
    #include <immintrin.h>
    #define TBB_SIMD_AVX512 1
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define TBB_SIMD_AVX2 1
#elif defined(__AVX__)
    #include <immintrin.h>
    #define TBB_SIMD_AVX 1
#elif defined(__SSE2__) || defined(__SSE3__) || defined(__SSE4_1__)
    #include <xmmintrin.h>
    #include <emmintrin.h>
    #include <pmmintrin.h>
    #include <tmmintrin.h>
    #include <smmintrin.h>
    #define TBB_SIMD_SSE 1
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define TBB_SIMD_NEON 1
#else
    #define TBB_SIMD_SCALAR 1
#endif

namespace tbb {
namespace core {
namespace math {
namespace simd {

// ============================================================
// 4‑wide float vector
// ============================================================

struct alignas(16) float4 {
#if TBB_SIMD_AVX512
    __m128 v;
    float4() : v(_mm_setzero_ps()) {}
    explicit float4(__m128 vec) : v(vec) {}
    explicit float4(float s) : v(_mm_set1_ps(s)) {}
    float4(float a, float b, float c, float d) : v(_mm_set_ps(d, c, b, a)) {}
    float operator[](int i) const { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(i,i,i,i))); }
    // Actually element access via union or _mm_extract_ps
#elif TBB_SIMD_SSE
    __m128 v;
    float4() noexcept : v(_mm_setzero_ps()) {}
    explicit float4(__m128 vec) noexcept : v(vec) {}
    explicit float4(float s) noexcept : v(_mm_set1_ps(s)) {}
    float4(float x, float y, float z, float w) noexcept : v(_mm_set_ps(w, z, y, x)) {}
    float operator[](int i) const noexcept {
        float arr[4]; _mm_store_ps(arr, v); return arr[i];
    }
#elif TBB_SIMD_NEON
    float32x4_t v;
    float4() noexcept : v(vdupq_n_f32(0.0f)) {}
    explicit float4(float32x4_t vec) noexcept : v(vec) {}
    explicit float4(float s) noexcept : v(vdupq_n_f32(s)) {}
    float4(float x, float y, float z, float w) noexcept {
        float arr[4] = {x, y, z, w};
        v = vld1q_f32(arr);
    }
    float operator[](int i) const noexcept { return vgetq_lane_f32(v, i); }
#else
    float data[4];
    float4() noexcept : data{0,0,0,0} {}
    explicit float4(float s) noexcept : data{s,s,s,s} {}
    float4(float x, float y, float z, float w) noexcept : data{x,y,z,w} {}
    float operator[](int i) const noexcept { return data[i]; }
    float& operator[](int i) noexcept { return data[i]; }
#endif
};

// ============================================================
// Float4 arithmetic operators (SSE/AVX fallback)
// ============================================================

inline float4 operator+(const float4& a, const float4& b) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_add_ps(a.v, b.v));
#elif TBB_SIMD_NEON
    return float4(vaddq_f32(a.v, b.v));
#else
    return float4(a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]);
#endif
}

inline float4 operator-(const float4& a, const float4& b) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_sub_ps(a.v, b.v));
#elif TBB_SIMD_NEON
    return float4(vsubq_f32(a.v, b.v));
#else
    return float4(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]);
#endif
}

inline float4 operator*(const float4& a, const float4& b) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_mul_ps(a.v, b.v));
#elif TBB_SIMD_NEON
    return float4(vmulq_f32(a.v, b.v));
#else
    return float4(a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]);
#endif
}

inline float4 operator/(const float4& a, const float4& b) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_div_ps(a.v, b.v));
#elif TBB_SIMD_NEON
    return float4(vdivq_f32(a.v, b.v));
#else
    return float4(a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3]);
#endif
}

inline float4 operator*(float s, const float4& v) noexcept { return float4(s) * v; }
inline float4 operator*(const float4& v, float s) noexcept { return float4(s) * v; }

// ============================================================
// Fused multiply‑add: a*b + c
// ============================================================

inline float4 fma(const float4& a, const float4& b, const float4& c) noexcept {
#if TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_fmadd_ps(a.v, b.v, c.v));
#elif TBB_SIMD_AVX
    return float4(_mm_add_ps(_mm_mul_ps(a.v, b.v), c.v));
#elif TBB_SIMD_NEON
    return float4(vfmaq_f32(c.v, a.v, b.v));
#else
    return a * b + c;
#endif
}

// ============================================================
// Comparison: min, max, abs, clamp
// ============================================================

inline float4 min(const float4& a, const float4& b) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_min_ps(a.v, b.v));
#elif TBB_SIMD_NEON
    return float4(vminq_f32(a.v, b.v));
#else
    return float4(std::fmin(a[0],b[0]), std::fmin(a[1],b[1]), std::fmin(a[2],b[2]), std::fmin(a[3],b[3]));
#endif
}

inline float4 max(const float4& a, const float4& b) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_max_ps(a.v, b.v));
#elif TBB_SIMD_NEON
    return float4(vmaxq_f32(a.v, b.v));
#else
    return float4(std::fmax(a[0],b[0]), std::fmax(a[1],b[1]), std::fmax(a[2],b[2]), std::fmax(a[3],b[3]));
#endif
}

inline float4 abs(const float4& v) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_andnot_ps(_mm_set1_ps(-0.0f), v.v)); // clear sign bit
#elif TBB_SIMD_NEON
    return float4(vabsq_f32(v.v));
#else
    return float4(std::fabs(v[0]), std::fabs(v[1]), std::fabs(v[2]), std::fabs(v[3]));
#endif
}

inline float4 clamp(const float4& v, float lo, float hi) noexcept {
    return min(max(v, float4(lo)), float4(hi));
}

// ============================================================
// Square root and reciprocal
// ============================================================

inline float4 sqrt(const float4& v) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_sqrt_ps(v.v));
#elif TBB_SIMD_NEON
    return float4(vsqrtq_f32(v.v));
#else
    return float4(std::sqrt(v[0]), std::sqrt(v[1]), std::sqrt(v[2]), std::sqrt(v[3]));
#endif
}

inline float4 rcp(const float4& v) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_rcp_ps(v.v));
#elif TBB_SIMD_NEON
    float32x4_t est = vrecpeq_f32(v.v);
    return float4(vmulq_f32(vrecpsq_f32(v.v, est), est));
#else
    return float4(1.0f/v[0], 1.0f/v[1], 1.0f/v[2], 1.0f/v[3]);
#endif
}

inline float4 rsqrt(const float4& v) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_rsqrt_ps(v.v));
#elif TBB_SIMD_NEON
    float32x4_t est = vrsqrteq_f32(v.v);
    return float4(vmulq_f32(vrsqrtsq_f32(vmulq_f32(v.v, est), est), est));
#else
    return float4(1.0f/std::sqrt(v[0]), 1.0f/std::sqrt(v[1]), 1.0f/std::sqrt(v[2]), 1.0f/std::sqrt(v[3]));
#endif
}

// ============================================================
// Horizontal operations
// ============================================================

inline float horizontal_add(const float4& v) noexcept {
#if TBB_SIMD_SSE3 || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    __m128 t = _mm_hadd_ps(v.v, v.v);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
#elif TBB_SIMD_NEON
    float32x2_t t = vadd_f32(vget_low_f32(v.v), vget_high_f32(v.v));
    return vget_lane_f32(vpadd_f32(t, t), 0);
#else
    return v[0] + v[1] + v[2] + v[3];
#endif
}

inline float horizontal_max(const float4& v) noexcept {
    return std::max({v[0], v[1], v[2], v[3]});
}

inline float horizontal_min(const float4& v) noexcept {
    return std::min({v[0], v[1], v[2], v[3]});
}

// ============================================================
// Dot product
// ============================================================

inline float dot(const float4& a, const float4& b) noexcept {
#if TBB_SIMD_SSE4_1 || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return _mm_cvtss_f32(_mm_dp_ps(a.v, b.v, 0xF1));
#else
    return horizontal_add(a * b);
#endif
}

// ============================================================
// Shuffle / swizzle
// ============================================================

inline float4 shuffle(const float4& v, int x, int y, int z, int w) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(w, z, y, x)));
#elif TBB_SIMD_NEON
    float arr[4];
    vst1q_f32(arr, v.v);
    return float4(arr[x], arr[y], arr[z], arr[w]);
#else
    return float4(v[x], v[y], v[z], v[w]);
#endif
}

// ============================================================
// 8‑wide float vector (AVX) – fallback for SSE
// ============================================================

#if TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
struct alignas(32) float8 {
    __m256 v;
    float8() noexcept : v(_mm256_setzero_ps()) {}
    explicit float8(__m256 vec) noexcept : v(vec) {}
    explicit float8(float s) noexcept : v(_mm256_set1_ps(s)) {}
    float8(float a, float b, float c, float d, float e, float f, float g, float h) noexcept
        : v(_mm256_set_ps(h, g, f, e, d, c, b, a)) {}
    float operator[](int i) const noexcept {
        float arr[8]; _mm256_store_ps(arr, v); return arr[i];
    }
};

inline float8 operator+(const float8& a, const float8& b) noexcept {
    return float8(_mm256_add_ps(a.v, b.v));
}
inline float8 operator*(const float8& a, const float8& b) noexcept {
    return float8(_mm256_mul_ps(a.v, b.v));
}
inline float8 fma(const float8& a, const float8& b, const float8& c) noexcept {
    return float8(_mm256_fmadd_ps(a.v, b.v, c.v));
}
#endif

// ============================================================
// 4‑wide double vector (SSE2 / AVX)
// ============================================================

#if TBB_SIMD_SSE2 || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
struct alignas(16) double2 {
    __m128d v;
    double2() noexcept : v(_mm_setzero_pd()) {}
    explicit double2(__m128d vec) noexcept : v(vec) {}
    explicit double2(double s) noexcept : v(_mm_set1_pd(s)) {}
    double2(double x, double y) noexcept : v(_mm_set_pd(y, x)) {}
    double operator[](int i) const noexcept {
        double arr[2]; _mm_store_pd(arr, v); return arr[i];
    }
};
inline double2 operator+(const double2& a, const double2& b) noexcept { return double2(_mm_add_pd(a.v, b.v)); }
inline double2 operator*(const double2& a, const double2& b) noexcept { return double2(_mm_mul_pd(a.v, b.v)); }
inline double2 fma(const double2& a, const double2& b, const double2& c) noexcept {
    return double2(_mm_fmadd_pd(a.v, b.v, c.v));
}
#endif

#if TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
struct alignas(32) double4 {
    __m256d v;
    double4() noexcept : v(_mm256_setzero_pd()) {}
    explicit double4(__m256d vec) noexcept : v(vec) {}
    explicit double4(double s) noexcept : v(_mm256_set1_pd(s)) {}
    double4(double a, double b, double c, double d) noexcept : v(_mm256_set_pd(d, c, b, a)) {}
    double operator[](int i) const noexcept { double arr[4]; _mm256_store_pd(arr, v); return arr[i]; }
};
inline double4 operator+(const double4& a, const double4& b) noexcept { return double4(_mm256_add_pd(a.v, b.v)); }
inline double4 operator*(const double4& a, const double4& b) noexcept { return double4(_mm256_mul_pd(a.v, b.v)); }
inline double4 fma(const double4& a, const double4& b, const double4& c) noexcept {
    return double4(_mm256_fmadd_pd(a.v, b.v, c.v));
}
#endif

// ============================================================
// Load / Store aligned
// ============================================================

inline float4 load_aligned(const float* p) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    return float4(_mm_load_ps(p));
#elif TBB_SIMD_NEON
    return float4(vld1q_f32(p));
#else
    return float4(p[0], p[1], p[2], p[3]);
#endif
}

inline void store_aligned(float* p, const float4& v) noexcept {
#if TBB_SIMD_SSE || TBB_SIMD_AVX || TBB_SIMD_AVX2 || TBB_SIMD_AVX512
    _mm_store_ps(p, v.v);
#elif TBB_SIMD_NEON
    vst1q_f32(p, v.v);
#else
    p[0]=v[0]; p[1]=v[1]; p[2]=v[2]; p[3]=v[3];
#endif
}

} // namespace simd
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_SIMD_WRAPPER_H