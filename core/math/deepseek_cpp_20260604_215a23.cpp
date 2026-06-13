// system name : onetbb-warp
// File 0004 : core/math/vector3.h
// Description : 3D vector type with full arithmetic, geometric, and dynamics operations.

#ifndef __TBB_WARP_CORE_MATH_VECTOR3_H
#define __TBB_WARP_CORE_MATH_VECTOR3_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Vector3 class template
// ============================================================

template<typename T>
struct vector3 {
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;

    T x, y, z;

    // ---- Constructors ----
    constexpr vector3() noexcept : x(T(0)), y(T(0)), z(T(0)) {}
    constexpr vector3(T v) noexcept : x(v), y(v), z(v) {}
    constexpr vector3(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}
    template<typename U>
    constexpr explicit vector3(const vector3<U>& v) noexcept : x(static_cast<T>(v.x)), y(static_cast<T>(v.y)), z(static_cast<T>(v.z)) {}
    constexpr vector3(const std::array<T,3>& arr) noexcept : x(arr[0]), y(arr[1]), z(arr[2]) {}
    constexpr vector3(const vector2<T>& v, T z_ = T(0)) noexcept : x(v.x), y(v.y), z(z_) {}

    // ---- Access ----
    constexpr T& operator[](std::size_t i) noexcept { return (&x)[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return (&x)[i]; }

    // ---- Compound assignment ----
    constexpr vector3& operator+=(const vector3& v) noexcept { x+=v.x; y+=v.y; z+=v.z; return *this; }
    constexpr vector3& operator-=(const vector3& v) noexcept { x-=v.x; y-=v.y; z-=v.z; return *this; }
    constexpr vector3& operator*=(T s) noexcept { x*=s; y*=s; z*=s; return *this; }
    constexpr vector3& operator/=(T s) noexcept { x/=s; y/=s; z/=s; return *this; }

    // ---- Unary ----
    constexpr vector3 operator+() const noexcept { return *this; }
    constexpr vector3 operator-() const noexcept { return vector3(-x, -y, -z); }

    // ---- Conversion ----
    constexpr operator std::array<T,3>() const noexcept { return {x, y, z}; }
    explicit constexpr operator bool() const noexcept { return x!=T(0) || y!=T(0) || z!=T(0); }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr vector3<T> operator+(const vector3<T>& a, const vector3<T>& b) noexcept { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
template<typename T> constexpr vector3<T> operator-(const vector3<T>& a, const vector3<T>& b) noexcept { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
template<typename T> constexpr vector3<T> operator*(const vector3<T>& v, T s) noexcept { return {v.x*s, v.y*s, v.z*s}; }
template<typename T> constexpr vector3<T> operator*(T s, const vector3<T>& v) noexcept { return {v.x*s, v.y*s, v.z*s}; }
template<typename T> constexpr vector3<T> operator/(const vector3<T>& v, T s) noexcept { return {v.x/s, v.y/s, v.z/s}; }
template<typename T> constexpr bool operator==(const vector3<T>& a, const vector3<T>& b) noexcept { return a.x==b.x && a.y==b.y && a.z==b.z; }
template<typename T> constexpr bool operator!=(const vector3<T>& a, const vector3<T>& b) noexcept { return !(a==b); }

// ============================================================
// Geometric operations
// ============================================================

template<typename T>
constexpr T dot(const vector3<T>& a, const vector3<T>& b) noexcept {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

template<typename T>
constexpr vector3<T> cross(const vector3<T>& a, const vector3<T>& b) noexcept {
    return vector3<T>(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

template<typename T>
constexpr T length_sq(const vector3<T>& v) noexcept {
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

template<typename T>
T length(const vector3<T>& v) noexcept {
    return std::sqrt(length_sq(v));
}

template<typename T>
vector3<T> normalize(const vector3<T>& v) noexcept {
    T len = length(v);
    if (len < T(FLOAT_EPSILON)) return vector3<T>(T(0));
    return v / len;
}

template<typename T>
vector3<T> safe_normalize(const vector3<T>& v, const vector3<T>& fallback = vector3<T>(T(1),T(0),T(0))) noexcept {
    T len = length(v);
    if (len < T(FLOAT_EPSILON)) return fallback;
    return v / len;
}

template<typename T>
T distance(const vector3<T>& a, const vector3<T>& b) noexcept {
    return length(b - a);
}

template<typename T>
constexpr T distance_sq(const vector3<T>& a, const vector3<T>& b) noexcept {
    return length_sq(b - a);
}

template<typename T>
T manhattan_distance(const vector3<T>& a, const vector3<T>& b) noexcept {
    return std::abs(a.x-b.x) + std::abs(a.y-b.y) + std::abs(a.z-b.z);
}

template<typename T>
T chebyshev_distance(const vector3<T>& a, const vector3<T>& b) noexcept {
    return std::max({std::abs(a.x-b.x), std::abs(a.y-b.y), std::abs(a.z-b.z)});
}

template<typename T>
constexpr vector3<T> lerp(const vector3<T>& a, const vector3<T>& b, T t) noexcept {
    return a + (b - a) * t;
}

template<typename T>
constexpr vector3<T> reflect(const vector3<T>& incident, const vector3<T>& normal) noexcept {
    return incident - normal * T(2) * dot(incident, normal);
}

template<typename T>
vector3<T> refract(const vector3<T>& incident, const vector3<T>& normal, T eta) noexcept {
    T ndoti = dot(incident, normal);
    T k = T(1) - eta * eta * (T(1) - ndoti * ndoti);
    if (k < T(0)) return vector3<T>(T(0));
    return incident * eta - normal * (eta * ndoti + std::sqrt(k));
}

template<typename T>
constexpr vector3<T> project(const vector3<T>& a, const vector3<T>& b) noexcept {
    T b2 = dot(b,b);
    if (b2 < T(FLOAT_EPSILON)) return vector3<T>(T(0));
    return b * (dot(a,b) / b2);
}

template<typename T>
constexpr vector3<T> project_on_plane(const vector3<T>& v, const vector3<T>& plane_normal) noexcept {
    return v - project(v, plane_normal);
}

template<typename T>
constexpr T triple_product(const vector3<T>& a, const vector3<T>& b, const vector3<T>& c) noexcept {
    return dot(a, cross(b, c));
}

template<typename T>
constexpr T angle(const vector3<T>& a, const vector3<T>& b) noexcept {
    return std::acos(clamp(dot(a,b) / (length(a)*length(b) + T(FLOAT_EPSILON)), T(-1), T(1)));
}

template<typename T>
vector3<T> rotate_around_axis(const vector3<T>& v, const vector3<T>& axis, T angle_rad) noexcept {
    vector3<T> ax = normalize(axis);
    T c = std::cos(angle_rad), s = std::sin(angle_rad);
    T omc = T(1) - c;
    return vector3<T>(
        v.x*(c + ax.x*ax.x*omc) + v.y*(ax.x*ax.y*omc - ax.z*s) + v.z*(ax.x*ax.z*omc + ax.y*s),
        v.x*(ax.y*ax.x*omc + ax.z*s) + v.y*(c + ax.y*ax.y*omc) + v.z*(ax.y*ax.z*omc - ax.x*s),
        v.x*(ax.z*ax.x*omc - ax.y*s) + v.y*(ax.z*ax.y*omc + ax.x*s) + v.z*(c + ax.z*ax.z*omc)
    );
}

// ============================================================
// Orthonormal basis generation
// ============================================================

template<typename T>
void orthonormal_basis(const vector3<T>& normal, vector3<T>& tangent, vector3<T>& bitangent) noexcept {
    tangent = (std::abs(normal.x) > T(0.99))
        ? normalize(cross(vector3<T>(T(0), T(1), T(0)), normal))
        : normalize(cross(vector3<T>(T(1), T(0), T(0)), normal));
    bitangent = cross(normal, tangent);
}

template<typename T>
vector3<T> spherical_to_cartesian(T azimuth, T elevation, T radius = T(1)) noexcept {
    T sin_elev = std::sin(elevation);
    return vector3<T>(
        radius * std::cos(azimuth) * sin_elev,
        radius * std::cos(elevation),
        radius * std::sin(azimuth) * sin_elev
    );
}

template<typename T>
vector3<T> cartesian_to_spherical(const vector3<T>& v, T& azimuth, T& elevation, T& radius) noexcept {
    radius = length(v);
    elevation = std::acos(clamp(v.y / radius, T(-1), T(1)));
    azimuth = std::atan2(v.z, v.x);
}

// ============================================================
// Component‑wise min / max / abs / clamp
// ============================================================

template<typename T> constexpr vector3<T> abs(const vector3<T>& v) noexcept { return vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z)); }
template<typename T> constexpr vector3<T> min(const vector3<T>& a, const vector3<T>& b) noexcept { return vector3<T>(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
template<typename T> constexpr vector3<T> max(const vector3<T>& a, const vector3<T>& b) noexcept { return vector3<T>(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
template<typename T> constexpr vector3<T> clamp(const vector3<T>& v, const vector3<T>& lo, const vector3<T>& hi) noexcept {
    return vector3<T>(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z));
}
template<typename T> constexpr T sum(const vector3<T>& v) noexcept { return v.x + v.y + v.z; }
template<typename T> constexpr T product(const vector3<T>& v) noexcept { return v.x * v.y * v.z; }
template<typename T> constexpr vector3<T> floor(const vector3<T>& v) noexcept { return vector3<T>(std::floor(v.x), std::floor(v.y), std::floor(v.z)); }
template<typename T> constexpr vector3<T> ceil(const vector3<T>& v) noexcept { return vector3<T>(std::ceil(v.x), std::ceil(v.y), std::ceil(v.z)); }
template<typename T> constexpr vector3<T> round(const vector3<T>& v) noexcept { return vector3<T>(std::round(v.x), std::round(v.y), std::round(v.z)); }
template<typename T> constexpr vector3<T> fract(const vector3<T>& v) noexcept { return v - floor(v); }

// ============================================================
// Comparison
// ============================================================

template<typename T> constexpr bool is_zero(const vector3<T>& v) noexcept { return v.x==T(0) && v.y==T(0) && v.z==T(0); }
template<typename T> constexpr bool is_normalized(const vector3<T>& v, T epsilon = T(FLOAT_EPSILON)) noexcept { return std::abs(length_sq(v) - T(1)) < epsilon; }
template<typename T> bool is_finite(const vector3<T>& v) noexcept { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
template<typename T> bool is_nan(const vector3<T>& v) noexcept { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z); }

// ============================================================
// Conversion between types
// ============================================================

template<typename U, typename T>
constexpr vector3<U> vector_cast(const vector3<T>& v) noexcept {
    return vector3<U>(static_cast<U>(v.x), static_cast<U>(v.y), static_cast<U>(v.z));
}

// ============================================================
// Smooth dynamics (damping, critical, spring)
// ============================================================

template<typename T>
vector3<T> smooth_damp(const vector3<T>& current, const vector3<T>& target,
                       vector3<T>& velocity, T smooth_time, T max_speed, T dt) {
    smooth_time = std::max(T(0.0001), smooth_time);
    T omega = T(2) / smooth_time;
    T x = omega * dt;
    T exp = T(1) / (T(1) + x + T(0.48)*x*x + T(0.235)*x*x*x);
    vector3<T> change = current - target;
    vector3<T> original_to = target;
    T max_change = max_speed * smooth_time;
    T change_len = length(change);
    if (change_len > max_change) change = change * (max_change / change_len);
    vector3<T> temp = (velocity + change * omega) * dt;
    velocity = (velocity - temp * omega) * exp;
    vector3<T> output = (current - change) + (change + temp) * exp;
    if (dot(original_to - current, output - original_to) > T(0)) {
        output = original_to;
        velocity = vector3<T>(T(0));
    }
    return output;
}

template<typename T>
vector3<T> critical_damp(const vector3<T>& current, const vector3<T>& target,
                         vector3<T>& velocity, T frequency, T dt) {
    T w = T(TAU_D) * frequency;
    T d = T(1) + T(2) * dt * w;
    T w2 = w * w;
    T inv = T(1) / (T(1) + T(2) * dt * w + dt * dt * w2);
    vector3<T> output = (current * T(1) + velocity * dt + target * (dt * dt * w2)) * inv;
    velocity = (velocity + (target - output) * (dt * w2)) * inv;
    return output;
}

template<typename T>
vector3<T> spring(const vector3<T>& current, const vector3<T>& target,
                  vector3<T>& velocity, T stiffness, T damping, T dt) {
    vector3<T> force = (target - current) * stiffness;
    T damp_force = -damping * length(velocity);
    vector3<T> acceleration;
    if (length(velocity) > T(0)) acceleration = force + velocity * (damp_force / length(velocity));
    else acceleration = force;
    velocity += acceleration * dt;
    return current + velocity * dt;
}

// ============================================================
// Linear regression (least squares) for 3D points
// ============================================================

template<typename T>
vector3<T> linear_regression(const std::vector<vector3<T>>& points, T& slope, T& intercept) noexcept {
    std::size_t n = points.size();
    if (n < 2) return vector3<T>(T(0));
    T sum_x = T(0), sum_y = T(0), sum_z = T(0);
    T sum_xx = T(0), sum_xy = T(0), sum_xz = T(0);
    T min_x = points[0].x;
    for (const auto& p : points) {
        T x = p.x - min_x;
        sum_x += x;
        sum_y += p.y;
        sum_z += p.z;
        sum_xx += x * x;
        sum_xy += x * p.y;
        sum_xz += x * p.z;
    }
    T denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < T(FLOAT_EPSILON)) return vector3<T>(T(0));
    T slope_y = (n * sum_xy - sum_x * sum_y) / denom;
    T slope_z = (n * sum_xz - sum_x * sum_z) / denom;
    T intercept_y = (sum_y - slope_y * sum_x) / n;
    T intercept_z = (sum_z - slope_z * sum_x) / n;
    slope = slope_y;
    intercept = intercept_y;
    return vector3<T>(T(0), slope_y, slope_z);
}

// ============================================================
// Type aliases
// ============================================================

using vector3f = vector3<float>;
using vector3d = vector3<double>;
using vector3i = vector3<std::int32_t>;
using vector3u = vector3<std::uint32_t>;

// ============================================================
// Fold expression helpers
// ============================================================

template<typename T, typename... Rest>
constexpr vector3<T> sum_vectors(const vector3<T>& first, const Rest&... rest) noexcept {
    return (first + ... + rest);
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_VECTOR3_H