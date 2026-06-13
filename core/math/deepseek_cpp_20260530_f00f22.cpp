// File 0009 : core/math/transform3.h
// 3D affine transformation with translation, rotation (quat), and scale (vec3), supporting composition, inverse, and point/vector application.

#pragma once

#include "vec3.h"
#include "quat.h"
#include "mat4.h"
#include <cmath>
#include <type_traits>

namespace wp {

template <typename T>
struct transform3 {
    vec3<T> translation;
    quat<T> rotation;
    vec3<T> scale;

    constexpr transform3() noexcept : translation(T(0)), rotation(T(0),T(0),T(0),T(1)), scale(T(1)) {}
    constexpr transform3(const vec3<T>& t, const quat<T>& r, const vec3<T>& s) noexcept : translation(t), rotation(r), scale(s) {}
    explicit constexpr transform3(const mat4<T>& m) noexcept {
        translation = vec3<T>(m.m03, m.m13, m.m23);
        rotation = from_matrix4(m);
        scale = vec3<T>(length(m.col(0).xyz()), length(m.col(1).xyz()), length(m.col(2).xyz()));
    }
    template <typename U> constexpr explicit transform3(const transform3<U>& o) noexcept
        : translation(o.translation), rotation(o.rotation), scale(o.scale) {}

    constexpr mat4<T> to_matrix4() const noexcept {
        mat3<T> r = to_matrix3(rotation);
        return mat4<T>(r.m00 * scale.x, r.m01 * scale.y, r.m02 * scale.z, translation.x,
                       r.m10 * scale.x, r.m11 * scale.y, r.m12 * scale.z, translation.y,
                       r.m20 * scale.x, r.m21 * scale.y, r.m22 * scale.z, translation.z,
                       T(0), T(0), T(0), T(1));
    }

    constexpr vec3<T> transform_point(const vec3<T>& p) const noexcept {
        return translation + rotate(rotation, scale * p);
    }
    constexpr vec3<T> transform_vector(const vec3<T>& v) const noexcept {
        return rotate(rotation, scale * v);
    }

    constexpr transform3 operator*(const transform3& other) const noexcept {
        transform3 result;
        result.rotation = mul(rotation, other.rotation);
        result.scale = scale * other.scale;
        result.translation = translation + rotate(rotation, scale * other.translation);
        return result;
    }
    constexpr transform3 inverse() const noexcept {
        vec3<T> inv_scale(T(1) / scale.x, T(1) / scale.y, T(1) / scale.z);
        quat<T> inv_rot = conjugate(rotation);
        return transform3(rotate(inv_rot, -translation * inv_scale), inv_rot, inv_scale);
    }

    constexpr bool operator==(const transform3& o) const noexcept {
        return translation == o.translation && rotation == o.rotation && scale == o.scale;
    }
    constexpr bool operator!=(const transform3& o) const noexcept { return !(*this == o); }
};

template <typename T> constexpr transform3<T> identity_transform3() noexcept {
    return transform3<T>(vec3<T>(T(0)), quat<T>(T(0),T(0),T(0),T(1)), vec3<T>(T(1)));
}
template <typename T> constexpr transform3<T> translation_transform3(const vec3<T>& t) noexcept {
    return transform3<T>(t, quat<T>(), vec3<T>(T(1)));
}
template <typename T> transform3<T> rotation_transform3(const quat<T>& r) noexcept {
    return transform3<T>(vec3<T>(T(0)), r, vec3<T>(T(1)));
}
template <typename T> constexpr transform3<T> scaling_transform3(const vec3<T>& s) noexcept {
    return transform3<T>(vec3<T>(T(0)), quat<T>(), s);
}
template <typename T> transform3<T> look_at_transform3(const vec3<T>& eye, const vec3<T>& center, const vec3<T>& up) noexcept {
    mat3<T> rot;
    vec3<T> f = normalize(center - eye);
    vec3<T> s = normalize(cross(f, up));
    vec3<T> u = cross(s, f);
    rot = mat3<T>(s, u, -f);
    return transform3<T>(eye, from_matrix3(rot), vec3<T>(T(1)));
}

template <typename T> constexpr transform3<T> lerp(const transform3<T>& a, const transform3<T>& b, T t) noexcept {
    return transform3<T>(
        lerp(a.translation, b.translation, t),
        slerp(a.rotation, b.rotation, t),
        lerp(a.scale, b.scale, t)
    );
}

using transform3f = transform3<float>;
using transform3d = transform3<double>;

} // namespace wp