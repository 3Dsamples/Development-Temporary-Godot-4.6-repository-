// File 0043 : core/math/quat_swing_twist.h
// Swing‑twist decomposition of unit quaternions around a given axis, and vector‑to‑vector rotation.

#pragma once

#include "quat.h"
#include "vec3.h"
#include "constants.h"
#include <cmath>

namespace wp {

// ── Decompose a unit quaternion into swing and twist around a world‑space axis ──
// Returns twist (rotation around the axis) and swing (rotation that aligns the axis to the final direction).
template <typename T>
void swing_twist_decompose(const quat<T>& q,
                           const vec3<T>& axis,
                           quat<T>& swing,
                           quat<T>& twist) {
    vec3<T> v(q.x, q.y, q.z);
    // Project v onto axis
    T v_dot_a = dot(v, axis);
    vec3<T> v_parallel = axis * v_dot_a;

    // Twist quaternion: component of q that lies in the subspace (1, a)
    twist = quat<T>(v_parallel.x, v_parallel.y, v_parallel.z, q.w);
    if (norm_sq(twist) < epsilon<T>) {
        // Degenerate: no twist component
        twist = quat<T>(T(0), T(0), T(0), T(1));
    } else {
        twist = normalize(twist);
    }

    // Swing = q * conjugate(twist)
    swing = mul(q, conjugate(twist));
    swing = normalize(swing);
}

// ── Compose a quaternion from swing and twist ───────────────────────
template <typename T>
quat<T> swing_twist_compose(const quat<T>& swing,
                            const quat<T>& twist) {
    return mul(swing, twist);
}

// ── Rotation quaternion that maps vector 'from' to 'to' ─────────────
template <typename T>
quat<T> rotation_between_vectors(const vec3<T>& from, const vec3<T>& to) {
    vec3<T> f = normalize(from);
    vec3<T> t = normalize(to);
    T d = dot(f, t);
    if (d > T(1) - epsilon<T>) {
        // Parallel, return identity
        return quat<T>();
    } else if (d < T(-1) + epsilon<T>) {
        // Anti‑parallel, rotate 180° around an orthogonal axis
        vec3<T> axis = perpendicular(f);
        return from_axis_angle(axis, pi<T>);
    }
    vec3<T> axis = cross(f, t);
    T s = std::sqrt((T(1) + d) * T(2));
    T inv_s = T(1) / s;
    return quat<T>(axis.x * inv_s, axis.y * inv_s, axis.z * inv_s, s * T(0.5));
}

// ── Swing and twist angles extraction ───────────────────────────────
template <typename T>
void swing_twist_angles(const quat<T>& q,
                        const vec3<T>& axis,
                        T& swing_angle,
                        T& twist_angle) {
    quat<T> swing, twist;
    swing_twist_decompose(q, axis, swing, twist);
    vec3<T> twist_axis;
    to_axis_angle(twist, twist_axis, twist_angle);
    // Ensure twist angle sign matches axis direction
    if (dot(twist_axis, axis) < T(0)) twist_angle = -twist_angle;
    vec3<T> swing_axis;
    to_axis_angle(swing, swing_axis, swing_angle);
    // Swing angle is the total rotation of the swing part, which is around the axis perpendicular to 'axis'.
}

} // namespace wp