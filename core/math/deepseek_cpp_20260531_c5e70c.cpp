//File 0047 : core/math/point_adapter.h
//Generic point adapters for 3D vector types (Godot, Eigen, GLM, std::array, Unreal, raw float array) used by spatial partitioning and geometry modules; extracts coordinates and sets them.
#ifndef CORE_MATH_POINT_ADAPTER_H
#define CORE_MATH_POINT_ADAPTER_H

#include "vector_math.h"           // for DirectX::XMVECTOR conversions, etc.
#include <Eigen/Core>
#include <glm/glm.hpp>
#include <array>

// Forward declare Godot types (actual headers included elsewhere if needed, but we only need the type name for specializations)
namespace Godot {
    class Vector3;
}

// Forward declare Unreal type (if used)
struct UnrealFVector { float X, Y, Z; };

namespace SimulationMath {

// Primary template – must be specialized for each type
template <typename T>
struct PointAdapter3D {
    static float x(const T& v) noexcept;
    static float y(const T& v) noexcept;
    static float z(const T& v) noexcept;
    static void set(T& v, float x_, float y_, float z_) noexcept;
};

// Specialization for Godot::Vector3
template <>
struct PointAdapter3D<Godot::Vector3> {
    static float x(const Godot::Vector3& v) noexcept { return v.x; }
    static float y(const Godot::Vector3& v) noexcept { return v.y; }
    static float z(const Godot::Vector3& v) noexcept { return v.z; }
    static void set(Godot::Vector3& v, float x_, float y_, float z_) noexcept { v.x = x_; v.y = y_; v.z = z_; }
};

// Specialization for Eigen::Vector3f
template <>
struct PointAdapter3D<Eigen::Vector3f> {
    static float x(const Eigen::Vector3f& v) noexcept { return v.x(); }
    static float y(const Eigen::Vector3f& v) noexcept { return v.y(); }
    static float z(const Eigen::Vector3f& v) noexcept { return v.z(); }
    static void set(Eigen::Vector3f& v, float x_, float y_, float z_) noexcept { v.x() = x_; v.y() = y_; v.z() = z_; }
};

// Specialization for glm::vec3
template <>
struct PointAdapter3D<glm::vec3> {
    static float x(const glm::vec3& v) noexcept { return v.x; }
    static float y(const glm::vec3& v) noexcept { return v.y; }
    static float z(const glm::vec3& v) noexcept { return v.z; }
    static void set(glm::vec3& v, float x_, float y_, float z_) noexcept { v.x = x_; v.y = y_; v.z = z_; }
};

// Specialization for std::array<float,3>
template <>
struct PointAdapter3D<std::array<float, 3>> {
    static float x(const std::array<float, 3>& v) noexcept { return v[0]; }
    static float y(const std::array<float, 3>& v) noexcept { return v[1]; }
    static float z(const std::array<float, 3>& v) noexcept { return v[2]; }
    static void set(std::array<float, 3>& v, float x_, float y_, float z_) noexcept { v[0] = x_; v[1] = y_; v[2] = z_; }
};

// Specialization for UnrealFVector
template <>
struct PointAdapter3D<UnrealFVector> {
    static float x(const UnrealFVector& v) noexcept { return v.X; }
    static float y(const UnrealFVector& v) noexcept { return v.Y; }
    static float z(const UnrealFVector& v) noexcept { return v.Z; }
    static void set(UnrealFVector& v, float x_, float y_, float z_) noexcept { v.X = x_; v.Y = y_; v.Z = z_; }
};

// Specialization for raw float[3] (Vec3Raw)
using Vec3Raw = float[3];
template <>
struct PointAdapter3D<Vec3Raw> {
    static float x(const Vec3Raw& v) noexcept { return v[0]; }
    static float y(const Vec3Raw& v) noexcept { return v[1]; }
    static float z(const Vec3Raw& v) noexcept { return v[2]; }
    static void set(Vec3Raw& v, float x_, float y_, float z_) noexcept { v[0] = x_; v[1] = y_; v[2] = z_; }
};

// Default convenience function to get centroid as XMVECTOR using any adapter
template <typename T>
DirectX::XMVECTOR get_centroid(const T& obj) {
    return DirectX::XMVectorSet(PointAdapter3D<T>::x(obj),
                                PointAdapter3D<T>::y(obj),
                                PointAdapter3D<T>::z(obj), 0.0f);
}

// Function to set coordinates from XMVECTOR
template <typename T>
void set_from_centroid(T& obj, DirectX::FXMVECTOR vec) {
    PointAdapter3D<T>::set(obj,
        DirectX::XMVectorGetX(vec),
        DirectX::XMVectorGetY(vec),
        DirectX::XMVectorGetZ(vec));
}

} // namespace SimulationMath

#endif // CORE_MATH_POINT_ADAPTER_H