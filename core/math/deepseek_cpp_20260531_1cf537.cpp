// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
#pragma once

#include "quat.hpp"

namespace wp {

// ── Spatial vector (twist / wrench) ──
template <typename Type>
using spatial_vector_t = vec_t<6, Type>;

template <typename T> vec_t<3,T>& w_vec(spatial_vector_t<T>& sv) noexcept {
    return *reinterpret_cast<vec_t<3,T>*>(&sv[0]);
}
template <typename T> vec_t<3,T>& v_vec(spatial_vector_t<T>& sv) noexcept {
    return *reinterpret_cast<vec_t<3,T>*>(&sv[3]);
}
template <typename T> const vec_t<3,T>& w_vec(const spatial_vector_t<T>& sv) noexcept {
    return *reinterpret_cast<const vec_t<3,T>*>(&sv[0]);
}
template <typename T> const vec_t<3,T>& v_vec(const spatial_vector_t<T>& sv) noexcept {
    return *reinterpret_cast<const vec_t<3,T>*>(&sv[3]);
}

// ── Spatial dot product ──
template <typename T> T spatial_dot(const spatial_vector_t<T>& a, const spatial_vector_t<T>& b) noexcept {
    return dot(a, b);
}

// ── Spatial cross products (motion–motion and motion–force) ──
template <typename T>
spatial_vector_t<T> spatial_cross(const spatial_vector_t<T>& a, const spatial_vector_t<T>& b) noexcept {
    vec_t<3,T> w = cross(w_vec(a), w_vec(b));
    vec_t<3,T> v = cross(v_vec(a), w_vec(b)) + cross(w_vec(a), v_vec(b));
    return {w[0], w[1], w[2], v[0], v[1], v[2]};
}

template <typename T>
spatial_vector_t<T> spatial_cross_dual(const spatial_vector_t<T>& a, const spatial_vector_t<T>& b) noexcept {
    vec_t<3,T> w = cross(w_vec(a), w_vec(b)) + cross(v_vec(a), v_vec(b));
    vec_t<3,T> v = cross(w_vec(a), v_vec(b));
    return {w[0], w[1], w[2], v[0], v[1], v[2]};
}

// ── Spatial transform (6×6) ──
template <typename T>
struct spatial_transform_t {
    mat_t<3,3,T> R;       // rotation matrix
    vec_t<3,T>    p;      // translation vector

    spatial_vector_t<T> apply(const spatial_vector_t<T>& sv) const noexcept {
        vec_t<3,T> w = mul(R, w_vec(sv));
        vec_t<3,T> v = mul(R, v_vec(sv)) + cross(p, w);
        return {w[0], w[1], w[2], v[0], v[1], v[2]};
    }

    spatial_vector_t<T> apply_dual(const spatial_vector_t<T>& sv) const noexcept {
        vec_t<3,T> w = mul(R, w_vec(sv)) + cross(p, mul(R, v_vec(sv)));
        vec_t<3,T> v = mul(R, v_vec(sv));
        return {w[0], w[1], w[2], v[0], v[1], v[2]};
    }
};

// ── Plücker transform from position + quaternion ──
template <typename T>
spatial_transform_t<T> spatial_transform_from_pose(const vec_t<3,T>& pos, const quat_t<T>& rot) noexcept {
    return { to_matrix(rot), pos };
}

// ── Composite rigid inertia ──
template <typename T>
struct spatial_inertia_t {
    T mass;
    mat_t<3,3,T> I;   // inertia tensor at COM

    spatial_vector_t<T> mul(const spatial_vector_t<T>& acc) const noexcept {
        vec_t<3,T> f_ang = mul(I, w_vec(acc));
        vec_t<3,T> f_lin = { mass * v_vec(acc)[0], mass * v_vec(acc)[1], mass * v_vec(acc)[2] };
        return {f_ang[0], f_ang[1], f_ang[2], f_lin[0], f_lin[1], f_lin[2]};
    }
};

} // namespace wp