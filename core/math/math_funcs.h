// core/math/math_funcs.h
#ifndef MATH_FUNCS_H
#define MATH_FUNCS_H

#include "math_defs.h"
#include "../big_number.h"
#include "core/typedefs.h"

// ----------------------------------------------------------------------------
// MathFuncs: A static wrapper providing _FORCE_INLINE_ access to deterministic
// kernels. This enables "Instruction Fusion" for complex formulas by allowing
// the compiler to merge multiple BigNumber operations into a single fused
// sequence at the limb level.
// ----------------------------------------------------------------------------

class MathFuncs {
public:
    // ------------------------------------------------------------------------
    // Basic Arithmetic Fusions
    // ------------------------------------------------------------------------
    
    // Fused multiply-add: a * b + c
    static _FORCE_INLINE_ real_t fma(real_t a, real_t b, real_t c) {
        return a * b + c;
    }
    
    // Fused multiply-sub: a * b - c
    static _FORCE_INLINE_ real_t fms(real_t a, real_t b, real_t c) {
        return a * b - c;
    }
    
    // Fused add-multiply: (a + b) * c
    static _FORCE_INLINE_ real_t fam(real_t a, real_t b, real_t c) {
        return (a + b) * c;
    }
    
    // Fused sub-multiply: (a - b) * c
    static _FORCE_INLINE_ real_t fsm(real_t a, real_t b, real_t c) {
        return (a - b) * c;
    }
    
    // ------------------------------------------------------------------------
    // Trigonometric Instruction Fusion
    // ------------------------------------------------------------------------
    
    // sin(x)^2 + cos(x)^2 (optimized to 1)
    static _FORCE_INLINE_ real_t sin_cos_hypot(real_t x) {
        // Since sin^2 + cos^2 = 1 exactly, we can return 1 without computation
        // This fusion avoids expensive transcendental evaluations.
        return real_t(1);
    }
    
    // sin(a) * cos(b) + cos(a) * sin(b) = sin(a+b)
    static _FORCE_INLINE_ real_t sin_add(real_t a, real_t b) {
        return Math::sin(a + b);
    }
    
    // cos(a) * cos(b) - sin(a) * sin(b) = cos(a+b)
    static _FORCE_INLINE_ real_t cos_add(real_t a, real_t b) {
        return Math::cos(a + b);
    }
    
    // tan(x) = sin(x) / cos(x) with division safety
    static _FORCE_INLINE_ real_t tan_safe(real_t x) {
        real_t c = Math::cos(x);
        if (c.abs() < CMP_EPSILON) {
            return (Math::sin(x) > real_t(0)) ? real_t(INT64_MAX) : real_t(INT64_MIN);
        }
        return Math::sin(x) / c;
    }
    
    // ------------------------------------------------------------------------
    // Exponential/Logarithmic Fusions
    // ------------------------------------------------------------------------
    
    // exp(a) * exp(b) = exp(a+b) (optimized)
    static _FORCE_INLINE_ real_t exp_mul(real_t a, real_t b) {
        return Math::exp(a + b);
    }
    
    // log(a) + log(b) = log(a*b) (optimized)
    static _FORCE_INLINE_ real_t log_add(real_t a, real_t b) {
        return Math::log(a * b);
    }
    
    // pow(base, exp) with integer exponent fast path
    static _FORCE_INLINE_ real_t pow_fast(real_t base, real_t exp) {
        if (exp.frac().is_zero()) {
            // Integer exponent - use fast exponentiation by squaring
            int64_t e = exp.to_int64();
            if (e == 0) return real_t(1);
            if (e == 1) return base;
            if (e == 2) return base * base;
            // Fallback to general pow for large exponents
        }
        return Math::pow(base, exp);
    }
    
    // ------------------------------------------------------------------------
    // Vector Operations (2D/3D) with Fused Kernels
    // ------------------------------------------------------------------------
    
    // Dot product: a.x*b.x + a.y*b.y
    static _FORCE_INLINE_ real_t dot2(const Vector2& a, const Vector2& b) {
        return fma(a.x, b.x, a.y * b.y);
    }
    
    // Dot product 3D: a.x*b.x + a.y*b.y + a.z*b.z
    static _FORCE_INLINE_ real_t dot3(const Vector3& a, const Vector3& b) {
        return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
    }
    
    // Cross product (2D scalar): a.x*b.y - a.y*b.x
    static _FORCE_INLINE_ real_t cross2(const Vector2& a, const Vector2& b) {
        return fms(a.x, b.y, a.y * b.x);
    }
    
    // Length squared (avoids sqrt)
    static _FORCE_INLINE_ real_t length_squared2(const Vector2& v) {
        return fma(v.x, v.x, v.y * v.y);
    }
    
    static _FORCE_INLINE_ real_t length_squared3(const Vector3& v) {
        return fma(v.x, v.x, fma(v.y, v.y, v.z * v.z));
    }
    
    // Length (with sqrt)
    static _FORCE_INLINE_ real_t length2(const Vector2& v) {
        return Math::sqrt(length_squared2(v));
    }
    
    static _FORCE_INLINE_ real_t length3(const Vector3& v) {
        return Math::sqrt(length_squared3(v));
    }
    
    // Normalize with safety
    static _FORCE_INLINE_ Vector2 normalize2(const Vector2& v) {
        real_t len = length2(v);
        if (len > CMP_EPSILON) {
            return v / len;
        }
        return Vector2();
    }
    
    static _FORCE_INLINE_ Vector3 normalize3(const Vector3& v) {
        real_t len = length3(v);
        if (len > CMP_EPSILON) {
            return v / len;
        }
        return Vector3();
    }
    
    // Distance between points
    static _FORCE_INLINE_ real_t distance2(const Vector2& a, const Vector2& b) {
        return length2(a - b);
    }
    
    static _FORCE_INLINE_ real_t distance3(const Vector3& a, const Vector3& b) {
        return length3(a - b);
    }
    
    // ------------------------------------------------------------------------
    // Interpolation Fusions
    // ------------------------------------------------------------------------
    
    // Spherical linear interpolation (quaternion)
    static Quaternion slerp(const Quaternion& from, const Quaternion& to, real_t t);
    
    // Cubic interpolation (Hermite)
    static _FORCE_INLINE_ real_t cubic_interpolate(real_t from, real_t to, real_t pre, real_t post, real_t t) {
        real_t t2 = t * t;
        real_t t3 = t2 * t;
        return real_t(0.5) * (
            (from * real_t(2)) +
            (-pre + to) * t +
            (real_t(2) * pre - real_t(5) * from + real_t(4) * to - post) * t2 +
            (-pre + real_t(3) * from - real_t(3) * to + post) * t3
        );
    }
    
    // Bezier interpolation
    static _FORCE_INLINE_ real_t bezier(real_t p0, real_t p1, real_t p2, real_t p3, real_t t) {
        real_t mt = real_t(1) - t;
        real_t mt2 = mt * mt;
        real_t mt3 = mt2 * mt;
        real_t t2 = t * t;
        real_t t3 = t2 * t;
        return p0 * mt3 + p1 * real_t(3) * mt2 * t + p2 * real_t(3) * mt * t2 + p3 * t3;
    }
    
    // ------------------------------------------------------------------------
    // Angle Utilities
    // ------------------------------------------------------------------------
    
    // Normalize angle to [-PI, PI]
    static _FORCE_INLINE_ real_t angle_difference(real_t from, real_t to) {
        real_t diff = (to - from) % Math_TAU;
        if (diff > Math_PI) diff -= Math_TAU;
        else if (diff < -Math_PI) diff += Math_TAU;
        return diff;
    }
    
    // Linear interpolation of angles (shortest path)
    static _FORCE_INLINE_ real_t lerp_angle(real_t from, real_t to, real_t t) {
        real_t diff = angle_difference(from, to);
        return from + diff * t;
    }
    
    // ------------------------------------------------------------------------
    // Geometric Utilities
    // ------------------------------------------------------------------------
    
    // Area of triangle (2D)
    static _FORCE_INLINE_ real_t triangle_area2(const Vector2& a, const Vector2& b, const Vector2& c) {
        return Math::abs(cross2(b - a, c - a)) * real_t(0.5);
    }
    
    // Check if point is inside triangle (barycentric)
    static bool point_in_triangle2(const Vector2& p, const Vector2& a, const Vector2& b, const Vector2& c);
    
    // Line segment intersection
    static bool segment_intersects2(const Vector2& a1, const Vector2& a2, const Vector2& b1, const Vector2& b2, Vector2* result = nullptr);
    
    // ------------------------------------------------------------------------
    // Random Number Generation (deterministic, seedable)
    // ------------------------------------------------------------------------
    
    // PCG32 random generator (deterministic, high-quality)
    class PCG32 {
        uint64_t state;
        uint64_t inc;
    public:
        PCG32(uint64_t seed = 0x853c49e6748fea9bULL, uint64_t seq = 0xda3e39cb94b95bdbULL);
        uint32_t rand();
        real_t randf(); // [0, 1)
        real_t randf_range(real_t min, real_t max);
        void seed(uint64_t s, uint64_t seq = 0);
    };
    
    // Global random instance (thread-local or static)
    static PCG32& get_random();
    
    // ------------------------------------------------------------------------
    // Noise Functions (Deterministic)
    // ------------------------------------------------------------------------
    
    // Perlin noise (2D)
    static real_t perlin_noise2(real_t x, real_t y, uint32_t seed = 0);
    
    // Simplex noise (3D)
    static real_t simplex_noise3(real_t x, real_t y, real_t z, uint32_t seed = 0);
    
    // ------------------------------------------------------------------------
    // Matrix/Transform Utilities (inline fusions)
    // ------------------------------------------------------------------------
    
    // Transform a 3D point by a Transform3D
    static _FORCE_INLINE_ Vector3 xform(const Transform3D& t, const Vector3& v) {
        return Vector3(
            fma(t.basis[0].x, v.x, fma(t.basis[1].x, v.y, fma(t.basis[2].x, v.z, t.origin.x))),
            fma(t.basis[0].y, v.x, fma(t.basis[1].y, v.y, fma(t.basis[2].y, v.z, t.origin.y))),
            fma(t.basis[0].z, v.x, fma(t.basis[1].z, v.y, fma(t.basis[2].z, v.z, t.origin.z)))
        );
    }
    
    // Inverse transform a 3D point
    static _FORCE_INLINE_ Vector3 xform_inv(const Transform3D& t, const Vector3& v) {
        Vector3 v_origin = v - t.origin;
        return Vector3(
            fma(t.basis[0].x, v_origin.x, fma(t.basis[0].y, v_origin.y, t.basis[0].z * v_origin.z)),
            fma(t.basis[1].x, v_origin.x, fma(t.basis[1].y, v_origin.y, t.basis[1].z * v_origin.z)),
            fma(t.basis[2].x, v_origin.x, fma(t.basis[2].y, v_origin.y, t.basis[2].z * v_origin.z))
        );
    }
    
    // ------------------------------------------------------------------------
    // Heavy Math Dispatchers (delegated to .cpp for TBB parallel execution)
    // ------------------------------------------------------------------------
    
    // Compute multiple sin values in parallel (uses oneTBB task arena)
    static void sin_batch(const real_t* src, real_t* dst, size_t count);
    
    // Compute multiple exp values in parallel
    static void exp_batch(const real_t* src, real_t* dst, size_t count);
    
    // Matrix multiplication batch (for many small matrices)
    static void mat_mul_batch(const Transform3D* a, const Transform3D* b, Transform3D* dst, size_t count);
};

// ----------------------------------------------------------------------------
// Additional inline global math functions (compatibility with Godot's Math)
// ----------------------------------------------------------------------------
namespace Math {
    // Delegate to MathFuncs for fused operations
    _FORCE_INLINE_ real_t fma(real_t a, real_t b, real_t c) { return MathFuncs::fma(a, b, c); }
    _FORCE_INLINE_ real_t fms(real_t a, real_t b, real_t c) { return MathFuncs::fms(a, b, c); }
    _FORCE_INLINE_ Vector2 normalize(const Vector2& v) { return MathFuncs::normalize2(v); }
    _FORCE_INLINE_ Vector3 normalize(const Vector3& v) { return MathFuncs::normalize3(v); }
    _FORCE_INLINE_ real_t dot(const Vector2& a, const Vector2& b) { return MathFuncs::dot2(a, b); }
    _FORCE_INLINE_ real_t dot(const Vector3& a, const Vector3& b) { return MathFuncs::dot3(a, b); }
    _FORCE_INLINE_ real_t cross(const Vector2& a, const Vector2& b) { return MathFuncs::cross2(a, b); }
    _FORCE_INLINE_ Vector3 cross(const Vector3& a, const Vector3& b) {
        return Vector3(
            MathFuncs::fms(a.y, b.z, a.z * b.y),
            MathFuncs::fms(a.z, b.x, a.x * b.z),
            MathFuncs::fms(a.x, b.y, a.y * b.x)
        );
    }
    _FORCE_INLINE_ real_t length(const Vector2& v) { return MathFuncs::length2(v); }
    _FORCE_INLINE_ real_t length(const Vector3& v) { return MathFuncs::length3(v); }
    _FORCE_INLINE_ real_t length_squared(const Vector2& v) { return MathFuncs::length_squared2(v); }
    _FORCE_INLINE_ real_t length_squared(const Vector3& v) { return MathFuncs::length_squared3(v); }
    _FORCE_INLINE_ real_t distance_to(const Vector2& a, const Vector2& b) { return MathFuncs::distance2(a, b); }
    _FORCE_INLINE_ real_t distance_to(const Vector3& a, const Vector3& b) { return MathFuncs::distance3(a, b); }
}

#endif // MATH_FUNCS_H
// Ending of File 14 of 15 (core/math/math_funcs.h)