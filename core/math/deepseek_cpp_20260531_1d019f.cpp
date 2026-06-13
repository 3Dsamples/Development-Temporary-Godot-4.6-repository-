//File 0026 : core/math/vector_math.h
//High‑performance 3D/4D SIMD vector operations: arithmetic, dot/cross, length, normalize, lerp, swizzle, comparison, and min/max — backed by DirectXMath and integrated with Godot/Eigen/GLM via sim_math_unified_conversions.h.
#ifndef CORE_MATH_VECTOR_MATH_H
#define CORE_MATH_VECTOR_MATH_H

#include "sim_math_unified_conversions.h"   // for type conversions, DirectXMath is already included there
#include <type_traits>
#include <cmath>

namespace SimulationMath {
namespace vector_math {

// -----------------------------------------------------------------------------
// 1. Basic constructors
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR zero() noexcept { return DirectX::XMVectorZero(); }
inline DirectX::XMVECTOR replicate(float f) noexcept { return DirectX::XMVectorReplicate(f); }
inline DirectX::XMVECTOR load(float x, float y, float z, float w) noexcept { return DirectX::XMVectorSet(x, y, z, w); }
inline DirectX::XMVECTOR load3(float x, float y, float z) noexcept { return DirectX::XMVectorSet(x, y, z, 0.0f); }
inline DirectX::XMVECTOR load4(const float* p) noexcept { return DirectX::XMVectorLoadFloat4(reinterpret_cast<const DirectX::XMFLOAT4*>(p)); }
inline DirectX::XMVECTOR load3(const float* p) noexcept {
    DirectX::XMFLOAT3 f3;
    f3.x = p[0]; f3.y = p[1]; f3.z = p[2];
    return DirectX::XMLoadFloat3(&f3);
}
inline void store3(float* p, DirectX::FXMVECTOR v) noexcept {
    DirectX::XMStoreFloat3(reinterpret_cast<DirectX::XMFLOAT3*>(p), v);
}
inline void store4(float* p, DirectX::FXMVECTOR v) noexcept {
    DirectX::XMStoreFloat4(reinterpret_cast<DirectX::XMFLOAT4*>(p), v);
}

// -----------------------------------------------------------------------------
// 2. Arithmetic
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR add(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorAdd(a, b); }
inline DirectX::XMVECTOR sub(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorSubtract(a, b); }
inline DirectX::XMVECTOR mul(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorMultiply(a, b); }
inline DirectX::XMVECTOR div(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorDivide(a, b); }
inline DirectX::XMVECTOR scale(DirectX::FXMVECTOR v, float s) noexcept { return DirectX::XMVectorScale(v, s); }
inline DirectX::XMVECTOR neg(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorNegate(v); }
inline DirectX::XMVECTOR abs(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorAbs(v); }
inline DirectX::XMVECTOR recip(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorReciprocalEst(v); }
inline DirectX::XMVECTOR sqrt(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorSqrt(v); }
inline DirectX::XMVECTOR rsqrt(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorReciprocalSqrtEst(v); }
inline DirectX::XMVECTOR mad(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) noexcept { return DirectX::XMVectorMultiplyAdd(a, b, c); }

// -----------------------------------------------------------------------------
// 3. Dot product (returns scalar replicated to all components)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR dot3(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVector3Dot(a, b); }
inline DirectX::XMVECTOR dot4(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVector4Dot(a, b); }
inline float dot3_scalar(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept {
    return DirectX::XMVectorGetX(DirectX::XMVector3Dot(a, b));
}
inline float dot4_scalar(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept {
    return DirectX::XMVectorGetX(DirectX::XMVector4Dot(a, b));
}

// -----------------------------------------------------------------------------
// 4. Cross product (3D)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR cross3(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVector3Cross(a, b); }

// -----------------------------------------------------------------------------
// 5. Length
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR length3(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector3Length(v); }
inline DirectX::XMVECTOR length4(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector4Length(v); }
inline float length3_scalar(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetX(DirectX::XMVector3Length(v)); }
inline float length4_scalar(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetX(DirectX::XMVector4Length(v)); }
inline DirectX::XMVECTOR length_sq3(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector3LengthSq(v); }
inline DirectX::XMVECTOR length_sq4(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector4LengthSq(v); }
inline float length_sq3_scalar(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetX(DirectX::XMVector3LengthSq(v)); }
inline float length_sq4_scalar(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetX(DirectX::XMVector4LengthSq(v)); }

// -----------------------------------------------------------------------------
// 6. Normalize
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR normalize3(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector3Normalize(v); }
inline DirectX::XMVECTOR normalize4(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector4Normalize(v); }
inline DirectX::XMVECTOR normalize3_est(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector3NormalizeEst(v); }
inline DirectX::XMVECTOR normalize4_est(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVector4NormalizeEst(v); }
inline DirectX::XMVECTOR safe_normalize3(DirectX::FXMVECTOR v) noexcept {
    DirectX::XMVECTOR len = DirectX::XMVector3Length(v);
    DirectX::XMVECTOR zero = DirectX::XMVectorZero();
    DirectX::XMVECTOR mask = DirectX::XMVectorGreater(len, DirectX::XMVectorReplicate(1e-12f));
    DirectX::XMVECTOR inv = DirectX::XMVectorDivide(DirectX::XMVectorReplicate(1.0f), len);
    return DirectX::XMVectorSelect(zero, DirectX::XMVectorMultiply(v, inv), mask);
}

// -----------------------------------------------------------------------------
// 7. Lerp and Slerp
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR lerp(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, float t) noexcept {
    return DirectX::XMVectorLerp(a, b, t);
}
inline DirectX::XMVECTOR lerpV(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR t) noexcept {
    return DirectX::XMVectorLerpV(a, b, t);
}
inline DirectX::XMVECTOR slerp3(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, float t) noexcept {
    return DirectX::XMQuaternionSlerp(a, b, t); // works for normalized vectors treated as quaternions
}

// -----------------------------------------------------------------------------
// 8. Min, max, clamp
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR min(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorMin(a, b); }
inline DirectX::XMVECTOR max(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorMax(a, b); }
inline DirectX::XMVECTOR clamp(DirectX::FXMVECTOR v, DirectX::FXMVECTOR minV, DirectX::FXMVECTOR maxV) noexcept {
    return DirectX::XMVectorClamp(v, minV, maxV);
}
inline DirectX::XMVECTOR saturate(DirectX::FXMVECTOR v) noexcept {
    return DirectX::XMVectorSaturate(v);
}

// -----------------------------------------------------------------------------
// 9. Comparison (returns mask vector: all bits 1 for true, 0 for false)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR equal(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorEqual(a, b); }
inline DirectX::XMVECTOR not_equal(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorNotEqual(a, b); }
inline DirectX::XMVECTOR greater(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorGreater(a, b); }
inline DirectX::XMVECTOR greater_or_equal(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorGreaterOrEqual(a, b); }
inline DirectX::XMVECTOR less(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorLess(a, b); }
inline DirectX::XMVECTOR less_or_equal(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept { return DirectX::XMVectorLessOrEqual(a, b); }
inline DirectX::XMVECTOR select(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR mask) noexcept {
    return DirectX::XMVectorSelect(b, a, mask); // mask bits 1 -> choose a, else b
}

// -----------------------------------------------------------------------------
// 10. Swizzle (common patterns)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR swizzle_xxxx(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorSwizzle<0,0,0,0>(v); }
inline DirectX::XMVECTOR swizzle_yyyy(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorSwizzle<1,1,1,1>(v); }
inline DirectX::XMVECTOR swizzle_zzzz(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorSwizzle<2,2,2,2>(v); }
inline DirectX::XMVECTOR swizzle_xyzw(DirectX::FXMVECTOR v) noexcept { return v; }

// -----------------------------------------------------------------------------
// 11. Extract component
// -----------------------------------------------------------------------------
inline float get_x(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetX(v); }
inline float get_y(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetY(v); }
inline float get_z(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetZ(v); }
inline float get_w(DirectX::FXMVECTOR v) noexcept { return DirectX::XMVectorGetW(v); }
inline DirectX::XMVECTOR set_x(DirectX::FXMVECTOR v, float x) noexcept { return DirectX::XMVectorSetX(v, x); }
inline DirectX::XMVECTOR set_y(DirectX::FXMVECTOR v, float y) noexcept { return DirectX::XMVectorSetY(v, y); }
inline DirectX::XMVECTOR set_z(DirectX::FXMVECTOR v, float z) noexcept { return DirectX::XMVectorSetZ(v, z); }
inline DirectX::XMVECTOR set_w(DirectX::FXMVECTOR v, float w) noexcept { return DirectX::XMVectorSetW(v, w); }

// -----------------------------------------------------------------------------
// 12. Horizontal operations (sum, avg, min, max across components)
// -----------------------------------------------------------------------------
inline float horizontal_sum3(DirectX::FXMVECTOR v) noexcept {
    DirectX::XMVECTOR tmp = DirectX::XMVectorAdd(v, DirectX::XMVectorSwizzle<1,0,2,3>(v));
    return get_x(tmp) + get_z(v);
}
inline float horizontal_sum4(DirectX::FXMVECTOR v) noexcept {
    DirectX::XMVECTOR t0 = DirectX::XMVectorAdd(v, DirectX::XMVectorSwizzle<2,3,0,1>(v));
    DirectX::XMVECTOR t1 = DirectX::XMVectorAdd(t0, DirectX::XMVectorSwizzle<1,0,3,2>(t0));
    return get_x(t1);
}
inline float horizontal_min3(DirectX::FXMVECTOR v) noexcept {
    DirectX::XMVECTOR tmp = DirectX::XMVectorMin(v, DirectX::XMVectorSwizzle<1,0,2,3>(v));
    return std::min(get_x(tmp), get_z(v));
}
inline float horizontal_max3(DirectX::FXMVECTOR v) noexcept {
    DirectX::XMVECTOR tmp = DirectX::XMVectorMax(v, DirectX::XMVectorSwizzle<1,0,2,3>(v));
    return std::max(get_x(tmp), get_z(v));
}

// -----------------------------------------------------------------------------
// 13. Convert between Godot::Vector3 and DirectX (reuse sim_math_unified_conversions)
// -----------------------------------------------------------------------------
inline Godot::Vector3 to_godot(DirectX::FXMVECTOR v) noexcept {
    return Godot::Vector3(get_x(v), get_y(v), get_z(v));
}
inline DirectX::XMVECTOR to_directx(const Godot::Vector3& v) noexcept {
    return DirectX::XMLoadFloat3(&DirectX::XMFLOAT3(v.x, v.y, v.z));
}
inline Godot::Vector4 to_godot4(DirectX::FXMVECTOR v) noexcept {
    return Godot::Vector4(get_x(v), get_y(v), get_z(v), get_w(v));
}
inline DirectX::XMVECTOR to_directx4(const Godot::Vector4& v) noexcept {
    return DirectX::XMLoadFloat4(&DirectX::XMFLOAT4(v.x, v.y, v.z, v.w));
}

// -----------------------------------------------------------------------------
// 14. Wrap-around for other engines (Eigen, glm) via the unified conversions
// -----------------------------------------------------------------------------
inline Eigen::Vector3f to_eigen(DirectX::FXMVECTOR v) noexcept {
    return Eigen::Vector3f(get_x(v), get_y(v), get_z(v));
}
inline DirectX::XMVECTOR to_directx(const Eigen::Vector3f& v) noexcept {
    return DirectX::XMVectorSet(v.x(), v.y(), v.z(), 0.0f);
}
inline glm::vec3 to_glm(DirectX::FXMVECTOR v) noexcept {
    return glm::vec3(get_x(v), get_y(v), get_z(v));
}
inline DirectX::XMVECTOR to_directx(const glm::vec3& v) noexcept {
    return DirectX::XMVectorSet(v.x, v.y, v.z, 0.0f);
}

} // namespace vector_math
} // namespace SimulationMath

#endif // CORE_MATH_VECTOR_MATH_H