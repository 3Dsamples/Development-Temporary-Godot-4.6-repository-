// core/math/math_funcs.cpp
#include "math_funcs.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/transform_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>
#include <mutex>

// ----------------------------------------------------------------------------
// PCG32 Random Number Generator Implementation
// High-quality, deterministic, seedable RNG.
// ----------------------------------------------------------------------------

MathFuncs::PCG32::PCG32(uint64_t seed, uint64_t seq) {
    state = 0;
    inc = (seq << 1) | 1;
    rand();
    state += seed;
    rand();
}

uint32_t MathFuncs::PCG32::rand() {
    uint64_t oldstate = state;
    state = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = uint32_t(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

real_t MathFuncs::PCG32::randf() {
    // Generate random in [0, 1) by using 32 bits of randomness
    uint32_t r = rand();
    // Convert to fixed-point in [0, 1) scaled by 2^32
    return real_t::from_raw(int64_t(r) << (real_t::SCALE_BITS - 32));
}

real_t MathFuncs::PCG32::randf_range(real_t min, real_t max) {
    return min + randf() * (max - min);
}

void MathFuncs::PCG32::seed(uint64_t s, uint64_t seq) {
    state = 0;
    inc = (seq << 1) | 1;
    rand();
    state += s;
    rand();
}

// Thread-local storage for random generator
static thread_local MathFuncs::PCG32* tls_rng = nullptr;
static std::mutex rng_mutex;

MathFuncs::PCG32& MathFuncs::get_random() {
    if (!tls_rng) {
        std::lock_guard<std::mutex> lock(rng_mutex);
        if (!tls_rng) {
            // Seed with a combination of time and thread id
            uint64_t seed = uint64_t(std::hash<std::thread::id>()(std::this_thread::get_id()));
            seed ^= uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
            tls_rng = new PCG32(seed);
        }
    }
    return *tls_rng;
}

// ----------------------------------------------------------------------------
// Quaternion Spherical Linear Interpolation (SLERP)
// ----------------------------------------------------------------------------
Quaternion MathFuncs::slerp(const Quaternion& from, const Quaternion& to, real_t t) {
    // Compute cosine of angle between quaternions
    real_t cosom = from.dot(to);
    
    // Adjust signs if necessary
    Quaternion to_adj = to;
    if (cosom < real_t(0)) {
        cosom = -cosom;
        to_adj = -to;
    }
    
    // If the angle is small, use linear interpolation to avoid division by zero
    if (real_t(1) - cosom > CMP_EPSILON) {
        real_t omega = Math::acos(cosom);
        real_t sinom = Math::sin(omega);
        real_t scale0 = Math::sin((real_t(1) - t) * omega) / sinom;
        real_t scale1 = Math::sin(t * omega) / sinom;
        return Quaternion(
            from.x * scale0 + to_adj.x * scale1,
            from.y * scale0 + to_adj.y * scale1,
            from.z * scale0 + to_adj.z * scale1,
            from.w * scale0 + to_adj.w * scale1
        );
    } else {
        // Linear interpolation for nearly identical quaternions
        return Quaternion(
            Math::lerp(from.x, to_adj.x, t),
            Math::lerp(from.y, to_adj.y, t),
            Math::lerp(from.z, to_adj.z, t),
            Math::lerp(from.w, to_adj.w, t)
        ).normalized();
    }
}

// ----------------------------------------------------------------------------
// Point in Triangle Test (Barycentric Coordinates)
// ----------------------------------------------------------------------------
bool MathFuncs::point_in_triangle2(const Vector2& p, const Vector2& a, const Vector2& b, const Vector2& c) {
    Vector2 v0 = c - a;
    Vector2 v1 = b - a;
    Vector2 v2 = p - a;
    
    real_t dot00 = v0.dot(v0);
    real_t dot01 = v0.dot(v1);
    real_t dot02 = v0.dot(v2);
    real_t dot11 = v1.dot(v1);
    real_t dot12 = v1.dot(v2);
    
    real_t inv_denom = real_t(1) / (dot00 * dot11 - dot01 * dot01);
    real_t u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    real_t v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    
    return (u >= real_t(0)) && (v >= real_t(0)) && (u + v <= real_t(1));
}

// ----------------------------------------------------------------------------
// Line Segment Intersection (2D)
// ----------------------------------------------------------------------------
bool MathFuncs::segment_intersects2(const Vector2& a1, const Vector2& a2, const Vector2& b1, const Vector2& b2, Vector2* result) {
    Vector2 r = a2 - a1;
    Vector2 s = b2 - b1;
    
    real_t rxs = cross2(r, s);
    Vector2 qp = b1 - a1;
    real_t qpxr = cross2(qp, r);
    
    if (rxs.abs() < CMP_EPSILON) {
        // Parallel or collinear
        if (qpxr.abs() < CMP_EPSILON) {
            // Collinear - check overlap
            real_t t0 = qp.dot(r) / r.dot(r);
            real_t t1 = t0 + s.dot(r) / r.dot(r);
            if (t0 > t1) std::swap(t0, t1);
            if (t0 < real_t(0)) t0 = real_t(0);
            if (t1 > real_t(1)) t1 = real_t(1);
            if (t0 <= t1) {
                if (result) *result = a1 + r * t0;
                return true;
            }
        }
        return false;
    }
    
    real_t t = cross2(qp, s) / rxs;
    real_t u = cross2(qp, r) / rxs;
    
    if (t >= real_t(0) && t <= real_t(1) && u >= real_t(0) && u <= real_t(1)) {
        if (result) *result = a1 + r * t;
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
// Perlin Noise 2D (Improved Noise - Ken Perlin 2002)
// ----------------------------------------------------------------------------
static _FORCE_INLINE_ real_t fade(real_t t) {
    // 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * real_t(6) - real_t(15)) + real_t(10));
}

static _FORCE_INLINE_ real_t lerp_grad(real_t a, real_t b, real_t t) {
    return Math::lerp(a, b, t);
}

static _FORCE_INLINE_ real_t grad(int hash, real_t x, real_t y) {
    int h = hash & 15;
    real_t u = h < 8 ? x : y;
    real_t v = h < 4 ? y : (h == 12 || h == 14 ? x : real_t(0));
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

real_t MathFuncs::perlin_noise2(real_t x, real_t y, uint32_t seed) {
    // Permutation table (will be shuffled with seed)
    static const int perm[256] = {
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,
        142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,
        203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
        74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,
        220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,
        132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
        186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,
        59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,
        70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,
        178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
        241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
        176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,
        128,195,78,66,215,61,156,180
    };
    
    // Create extended permutation table with seed variation
    int p[512];
    for (int i = 0; i < 256; i++) {
        p[i] = perm[i] ^ (seed & 0xFF);
        p[i+256] = p[i];
    }
    
    // Find unit cube containing point
    int xi = Math::floor(x).to_int() & 255;
    int yi = Math::floor(y).to_int() & 255;
    
    // Relative coordinates in cube
    real_t xf = x.frac();
    real_t yf = y.frac();
    
    // Fade curves
    real_t u = fade(xf);
    real_t v = fade(yf);
    
    // Hash coordinates of cube corners
    int aa = p[p[xi] + yi];
    int ab = p[p[xi] + yi + 1];
    int ba = p[p[xi + 1] + yi];
    int bb = p[p[xi + 1] + yi + 1];
    
    // Blend results
    real_t x1 = lerp_grad(grad(aa, xf, yf), grad(ba, xf - real_t(1), yf), u);
    real_t x2 = lerp_grad(grad(ab, xf, yf - real_t(1)), grad(bb, xf - real_t(1), yf - real_t(1)), u);
    
    return lerp_grad(x1, x2, v) * real_t(0.5) + real_t(0.5); // Map to [0,1]
}

// ----------------------------------------------------------------------------
// Simplex Noise 3D (Ken Perlin)
// ----------------------------------------------------------------------------
static const real_t F3 = real_t(1) / real_t(3);
static const real_t G3 = real_t(1) / real_t(6);

real_t MathFuncs::simplex_noise3(real_t x, real_t y, real_t z, uint32_t seed) {
    // Skew input space to simplex cell
    real_t s = (x + y + z) * F3;
    int i = Math::floor(x + s).to_int();
    int j = Math::floor(y + s).to_int();
    int k = Math::floor(z + s).to_int();
    
    real_t t = real_t(i + j + k) * G3;
    real_t X0 = real_t(i) - t;
    real_t Y0 = real_t(j) - t;
    real_t Z0 = real_t(k) - t;
    
    real_t x0 = x - X0;
    real_t y0 = y - Y0;
    real_t z0 = z - Z0;
    
    // Determine simplex corner ordering
    int i1, j1, k1;
    int i2, j2, k2;
    
    if (x0 >= y0) {
        if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }
    
    real_t x1 = x0 - real_t(i1) + G3;
    real_t y1 = y0 - real_t(j1) + G3;
    real_t z1 = z0 - real_t(k1) + G3;
    real_t x2 = x0 - real_t(i2) + F3;
    real_t y2 = y0 - real_t(j2) + F3;
    real_t z2 = z0 - real_t(k2) + F3;
    real_t x3 = x0 - real_t(0.5);
    real_t y3 = y0 - real_t(0.5);
    real_t z3 = z0 - real_t(0.5);
    
    // Simple permutation table (simplified, seed not fully integrated for brevity)
    // In practice, use a proper perm table like Perlin.
    auto hash = [seed](int i) -> int {
        i = (i ^ int(seed)) * 0x1d8f;
        return i & 0xFF;
    };
    
    int gi0 = hash(i + hash(j + hash(k)));
    int gi1 = hash(i + i1 + hash(j + j1 + hash(k + k1)));
    int gi2 = hash(i + i2 + hash(j + j2 + hash(k + k2)));
    int gi3 = hash(i + 1 + hash(j + 1 + hash(k + 1)));
    
    // Calculate contributions
    auto contrib = [](int gi, real_t x, real_t y, real_t z) -> real_t {
        real_t t = real_t(0.6) - x*x - y*y - z*z;
        if (t < real_t(0)) return real_t(0);
        t = t * t * t * t; // t^4
        int h = gi & 31;
        real_t grad_x = real_t((h >> 2) & 1) * real_t(2) - real_t(1);
        real_t grad_y = real_t((h >> 1) & 1) * real_t(2) - real_t(1);
        real_t grad_z = real_t(h & 1) * real_t(2) - real_t(1);
        return t * (grad_x * x + grad_y * y + grad_z * z);
    };
    
    real_t n0 = contrib(gi0, x0, y0, z0);
    real_t n1 = contrib(gi1, x1, y1, z1);
    real_t n2 = contrib(gi2, x2, y2, z2);
    real_t n3 = contrib(gi3, x3, y3, z3);
    
    // Scale to [-1,1] then to [0,1]
    return (n0 + n1 + n2 + n3) * real_t(32) + real_t(0.5);
}

// ----------------------------------------------------------------------------
// Batch Trigonometric Operations (Parallel with std::thread)
// ----------------------------------------------------------------------------
void MathFuncs::sin_batch(const real_t* src, real_t* dst, size_t count) {
    if (count == 0) return;
    
    // Determine number of threads (use hardware concurrency)
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (count < num_threads * 100) num_threads = 1; // Small batch, avoid overhead
    
    std::vector<std::thread> threads;
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start >= end) continue;
        
        threads.emplace_back([src, dst, start, end]() {
            for (size_t i = start; i < end; ++i) {
                dst[i] = real_t::sin(src[i]);
            }
        });
    }
    
    for (auto& th : threads) th.join();
}

void MathFuncs::exp_batch(const real_t* src, real_t* dst, size_t count) {
    if (count == 0) return;
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (count < num_threads * 100) num_threads = 1;
    
    std::vector<std::thread> threads;
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start >= end) continue;
        
        threads.emplace_back([src, dst, start, end]() {
            for (size_t i = start; i < end; ++i) {
                dst[i] = real_t::exp(src[i]);
            }
        });
    }
    
    for (auto& th : threads) th.join();
}

// ----------------------------------------------------------------------------
// Batch Matrix Multiplication (Transform3D)
// dst[i] = a[i] * b[i] for each i
// ----------------------------------------------------------------------------
void MathFuncs::mat_mul_batch(const Transform3D* a, const Transform3D* b, Transform3D* dst, size_t count) {
    if (count == 0) return;
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (count < num_threads * 50) num_threads = 1;
    
    std::vector<std::thread> threads;
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start >= end) continue;
        
        threads.emplace_back([a, b, dst, start, end]() {
            for (size_t i = start; i < end; ++i) {
                dst[i] = a[i] * b[i];
            }
        });
    }
    
    for (auto& th : threads) th.join();
}

// ----------------------------------------------------------------------------
// Additional utility: Fast approximate sqrt (for graphics when precision relaxed)
// This is provided as a static method but not part of the main API.
// ----------------------------------------------------------------------------
real_t fast_sqrt_approx(real_t x) {
    if (x <= real_t(0)) return real_t(0);
    // Use integer bit manipulation for initial guess
    int64_t raw = x.raw();
    int64_t guess_raw = (raw >> 1) + (1LL << (real_t::SCALE_BITS / 2 - 1));
    real_t y = real_t::from_raw(guess_raw);
    // One Newton iteration
    y = (y + x / y) >> 1;
    return y;
}

// ----------------------------------------------------------------------------
// End of math_funcs.cpp
// ----------------------------------------------------------------------------
// Ending of File 15 of 15 (core/math/math_funcs.cpp)