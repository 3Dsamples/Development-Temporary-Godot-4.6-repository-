// core/math/math_defs.h
#ifndef MATH_DEFS_H
#define MATH_DEFS_H

#include "../big_number.h"

// ----------------------------------------------------------------------------
// SIMD Width Detection and Configuration
// The UEP (Unified Engine Physics) system dynamically selects the optimal
// SIMD width based on the target platform. PC defaults to AVX-512 (512-bit),
// mobile defaults to NEON (128-bit). These constants are used throughout
// the engine to align memory and vectorize operations.
// ----------------------------------------------------------------------------

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
    #define UEP_SIMD_AVX512 1
    #define UEP_SIMD_WIDTH_BITS 512
    #define UEP_SIMD_WIDTH_BYTES 64
    #define UEP_SIMD_LIMBS_PER_VEC 8   // 512 bits / 64 bits per limb
    #define UEP_SIMD_ALIGNMENT 64
#elif defined(__AVX2__)
    #define UEP_SIMD_AVX2 1
    #define UEP_SIMD_WIDTH_BITS 256
    #define UEP_SIMD_WIDTH_BYTES 32
    #define UEP_SIMD_LIMBS_PER_VEC 4    // 256 bits / 64 bits per limb
    #define UEP_SIMD_ALIGNMENT 32
#elif defined(__ARM_NEON)
    #define UEP_SIMD_NEON 1
    #define UEP_SIMD_WIDTH_BITS 128
    #define UEP_SIMD_WIDTH_BYTES 16
    #define UEP_SIMD_LIMBS_PER_VEC 2    // 128 bits / 64 bits per limb
    #define UEP_SIMD_ALIGNMENT 16
#else
    #define UEP_SIMD_WIDTH_BITS 64
    #define UEP_SIMD_WIDTH_BYTES 8
    #define UEP_SIMD_LIMBS_PER_VEC 1
    #define UEP_SIMD_ALIGNMENT 8
#endif

// ----------------------------------------------------------------------------
// Force-Inline Macro (Compiler Agnostic)
// ----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
    #define _FORCE_INLINE_ __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    #define _FORCE_INLINE_ __forceinline
#else
    #define _FORCE_INLINE_ inline
#endif

// ----------------------------------------------------------------------------
// Redefinition of Godot's real_t to UEP's BigNumber
// This is the core change that replaces all floating-point math in the engine
// with arbitrary-precision deterministic arithmetic.
// ----------------------------------------------------------------------------
namespace uep {
    using real_t = BigNumber;
} // namespace uep

// For compatibility with existing Godot code that expects 'real_t' in global scope
using uep::real_t;

// ----------------------------------------------------------------------------
// Math Constants (as inline functions returning BigNumber)
// These are defined as functions to ensure they are initialized after the
// BigNumber subsystem is ready.
// ----------------------------------------------------------------------------
namespace Math {
    _FORCE_INLINE_ real_t pi() { return real_t::pi(); }
    _FORCE_INLINE_ real_t tau() { return real_t::pi() * real_t(2); }
    _FORCE_INLINE_ real_t e() { return real_t::e(); }
    _FORCE_INLINE_ real_t ln2() { return real_t::ln2(); }
    _FORCE_INLINE_ real_t sqrt2() { return real_t::sqrt2(); }
    
    // Common constants as real_t
    _FORCE_INLINE_ real_t POS_INF() { 
        // BigNumber doesn't have infinity; return max representable value
        return real_t(INT64_MAX); 
    }
    _FORCE_INLINE_ real_t NEG_INF() { 
        return real_t(INT64_MIN); 
    }
    _FORCE_INLINE_ real_t NaN() { 
        // Not-a-Number not supported; return zero
        return real_t(0); 
    }
}

// For backward compatibility, define global constants
#define Math_PI (Math::pi())
#define Math_TAU (Math::tau())
#define Math_E (Math::e())
#define Math_SQRT2 (Math::sqrt2())

// ----------------------------------------------------------------------------
// Inline Math Functions (dispatched to BigNumber static methods)
// These replace Godot's existing math_funcs.h functions when real_t is BigNumber.
// ----------------------------------------------------------------------------
namespace Math {

    // Trigonometric
    _FORCE_INLINE_ real_t sin(real_t x) { return real_t::sin(x); }
    _FORCE_INLINE_ real_t cos(real_t x) { return real_t::cos(x); }
    _FORCE_INLINE_ real_t tan(real_t x) { return real_t::tan(x); }
    _FORCE_INLINE_ real_t asin(real_t x) { return real_t::asin(x); }
    _FORCE_INLINE_ real_t acos(real_t x) { return real_t::acos(x); }
    _FORCE_INLINE_ real_t atan(real_t x) { return real_t::atan(x); }
    _FORCE_INLINE_ real_t atan2(real_t y, real_t x) { return real_t::atan2(y, x); }
    
    // Hyperbolic
    _FORCE_INLINE_ real_t sinh(real_t x) { return real_t::sinh(x); }
    _FORCE_INLINE_ real_t cosh(real_t x) { return real_t::cosh(x); }
    _FORCE_INLINE_ real_t tanh(real_t x) { return real_t::tanh(x); }
    
    // Exponential and Logarithmic
    _FORCE_INLINE_ real_t exp(real_t x) { return real_t::exp(x); }
    _FORCE_INLINE_ real_t log(real_t x) { return real_t::log(x); }
    _FORCE_INLINE_ real_t log10(real_t x) { return real_t::log10(x); }
    _FORCE_INLINE_ real_t pow(real_t base, real_t exp) { return real_t::pow(base, exp); }
    _FORCE_INLINE_ real_t sqrt(real_t x) { return real_t::sqrt(x); }
    
    // Rounding and Remainder
    _FORCE_INLINE_ real_t abs(real_t x) { return x.abs(); }
    _FORCE_INLINE_ real_t floor(real_t x) { return x.floor(); }
    _FORCE_INLINE_ real_t ceil(real_t x) { return x.ceil(); }
    _FORCE_INLINE_ real_t round(real_t x) { return x.round(); }
    _FORCE_INLINE_ real_t fmod(real_t x, real_t y) { return x % y; }
    _FORCE_INLINE_ real_t fposmod(real_t x, real_t y) { 
        real_t r = x % y;
        if (r < real_t(0)) r += y.abs();
        return r;
    }
    _FORCE_INLINE_ real_t fract(real_t x) { return x.frac(); }
    _FORCE_INLINE_ real_t sign(real_t x) { return (x > real_t(0)) ? real_t(1) : ((x < real_t(0)) ? real_t(-1) : real_t(0)); }
    
    // Interpolation
    _FORCE_INLINE_ real_t lerp(real_t a, real_t b, real_t t) { 
        return a + (b - a) * t; 
    }
    _FORCE_INLINE_ real_t inverse_lerp(real_t a, real_t b, real_t v) { 
        return (v - a) / (b - a); 
    }
    _FORCE_INLINE_ real_t remap(real_t i_min, real_t i_max, real_t o_min, real_t o_max, real_t v) {
        return lerp(o_min, o_max, inverse_lerp(i_min, i_max, v));
    }
    _FORCE_INLINE_ real_t smoothstep(real_t edge0, real_t edge1, real_t x) {
        real_t t = clamp((x - edge0) / (edge1 - edge0), real_t(0), real_t(1));
        return t * t * (real_t(3) - real_t(2) * t);
    }
    
    // Clamping
    _FORCE_INLINE_ real_t clamp(real_t value, real_t min_val, real_t max_val) {
        if (value < min_val) return min_val;
        if (value > max_val) return max_val;
        return value;
    }
    _FORCE_INLINE_ real_t min(real_t a, real_t b) { return a < b ? a : b; }
    _FORCE_INLINE_ real_t max(real_t a, real_t b) { return a > b ? a : b; }
    
    // Conversion to/from native types (for compatibility)
    _FORCE_INLINE_ double to_double(real_t x) { return x.to_double(); }
    _FORCE_INLINE_ float to_float(real_t x) { return x.to_float(); }
    _FORCE_INLINE_ int64_t to_int64(real_t x) { return x.to_int64(); }
    _FORCE_INLINE_ real_t from_double(double d) { return real_t(d); }
    _FORCE_INLINE_ real_t from_float(float f) { return real_t(f); }
    _FORCE_INLINE_ real_t from_int64(int64_t i) { return real_t(i); }
    
    // Additional utility
    _FORCE_INLINE_ bool is_zero_approx(real_t x) { return x.abs() < CMP_EPSILON; }
    _FORCE_INLINE_ bool is_equal_approx(real_t a, real_t b) { return (a - b).abs() < CMP_EPSILON; }
    _FORCE_INLINE_ bool is_nan(real_t x) { return false; } // BigNumber never NaN
    _FORCE_INLINE_ bool is_inf(real_t x) { return false; } // BigNumber finite
    _FORCE_INLINE_ real_t move_toward(real_t from, real_t to, real_t delta) {
        real_t diff = to - from;
        real_t dist = diff.abs();
        if (dist <= delta || dist <= CMP_EPSILON) return to;
        return from + (diff / dist) * delta;
    }
    _FORCE_INLINE_ real_t deg2rad(real_t deg) { return deg * Math::pi() / real_t(180); }
    _FORCE_INLINE_ real_t rad2deg(real_t rad) { return rad * real_t(180) / Math::pi(); }
    
} // namespace Math

// ----------------------------------------------------------------------------
// Compatibility Macros for Existing Godot Code
// These macros allow code written for float/double to compile with BigNumber.
// ----------------------------------------------------------------------------
#define CMP_EPSILON (real_t(1) >> 20) // Approximately 1e-6 in fixed-point
#define Math_SQRT12 real_t(0.707106781186547524400844362104849039284) // sqrt(1/2)
#define Math_LN2 Math::ln2()
#define Math_PI Math::pi()
#define Math_TAU Math::tau()
#define Math_E Math::e()

// Rounding modes (Godot compatibility)
#define FLOOR(x) Math::floor(x)
#define CEIL(x) Math::ceil(x)
#define ROUND(x) Math::round(x)
#define ABS(x) Math::abs(x)
#define SIGN(x) Math::sign(x)
#define MIN(a,b) Math::min(a,b)
#define MAX(a,b) Math::max(a,b)
#define CLAMP(v,lo,hi) Math::clamp(v,lo,hi)

// Trigonometric macros
#define SIN(x) Math::sin(x)
#define COS(x) Math::cos(x)
#define TAN(x) Math::tan(x)
#define ASIN(x) Math::asin(x)
#define ACOS(x) Math::acos(x)
#define ATAN(x) Math::atan(x)
#define ATAN2(y,x) Math::atan2(y,x)

// Exponential/log
#define EXP(x) Math::exp(x)
#define LOG(x) Math::log(x)
#define POW(base,exp) Math::pow(base,exp)
#define SQRT(x) Math::sqrt(x)

// Interpolation
#define LERP(a,b,t) Math::lerp(a,b,t)
#define INVERSE_LERP(a,b,v) Math::inverse_lerp(a,b,v)
#define REMAP(i_min,i_max,o_min,o_max,v) Math::remap(i_min,i_max,o_min,o_max,v)
#define SMOOTHSTEP(edge0,edge1,x) Math::smoothstep(edge0,edge1,x)

// Conversion
#define DEG2RAD(d) Math::deg2rad(d)
#define RAD2DEG(r) Math::rad2deg(r)

// ----------------------------------------------------------------------------
// Type Aliases for Godot's Math Types
// Replace float and double with real_t (BigNumber) for unified precision.
// ----------------------------------------------------------------------------
typedef uep::real_t real_t;

// For vector types, we rely on Godot's existing Vector2/3/4 templates,
// which will now use real_t = BigNumber.
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/math/plane.h"
#include "core/math/aabb.h"
#include "core/math/rect2.h"
#include "core/math/color.h"

// Ensure the vector classes are instantiated with real_t
// (Godot's vector templates are already parameterized by real_t)

// ----------------------------------------------------------------------------
// SIMD-accelerated batch operations (inline wrappers)
// These functions leverage the underlying BigIntCore SIMD operations
// to perform element-wise arithmetic on arrays of real_t.
// ----------------------------------------------------------------------------
namespace Math {
    // Batch addition: dst[i] = a[i] + b[i] for i=0..count-1
    _FORCE_INLINE_ void batch_add(real_t* dst, const real_t* a, const real_t* b, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = a[i] + b[i];
        }
        // The BigIntCore inside each real_t will auto-vectorize the limb operations.
        // For full SIMD across multiple real_t, see core_bind.cpp batch operations.
    }
    
    // Batch multiplication: dst[i] = a[i] * b[i]
    _FORCE_INLINE_ void batch_mul(real_t* dst, const real_t* a, const real_t* b, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = a[i] * b[i];
        }
    }
    
    // Batch linear interpolation: dst[i] = a[i] + (b[i] - a[i]) * t[i]
    _FORCE_INLINE_ void batch_lerp(real_t* dst, const real_t* a, const real_t* b, const real_t* t, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = Math::lerp(a[i], b[i], t[i]);
        }
    }
}

// ----------------------------------------------------------------------------
// End of math_defs.h
// ----------------------------------------------------------------------------
#endif // MATH_DEFS_H
// Ending of File 13 of 15 (core/math/math_defs.h)