// genesis/datatypes.h

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <memory>
#include <iostream>
#include <algorithm>
#include <initializer_list>
#include <utility>

namespace genesis {
namespace datatypes {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------
template<typename T, size_t N> class Vector;
template<typename T> class Quaternion;
template<typename T> class Matrix3;
template<typename T> class Matrix4;
template<typename T> class Transform;
class BoundingBox;
class Ray;
class Plane;
class Sphere;
class AABB;
class OBB;

//------------------------------------------------------------------------------
// Basic type aliases
//------------------------------------------------------------------------------
using float32 = float;
using float64 = double;
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using byte = uint8_t;
using index_t = int64_t;

//------------------------------------------------------------------------------
// Precision selector
//------------------------------------------------------------------------------
#ifdef GENESIS_USE_DOUBLE_PRECISION
using real = float64;
#else
using real = float32;
#endif

//------------------------------------------------------------------------------
// N-dimensional Vector class
//------------------------------------------------------------------------------
template<typename T, size_t N>
class Vector {
    static_assert(N > 0, "Vector dimension must be positive");
public:
    std::array<T, N> data;

    // Constructors
    Vector() : data{} {}
    explicit Vector(T val) { data.fill(val); }
    Vector(std::initializer_list<T> init) {
        if (init.size() != N) {
            throw std::invalid_argument("Initializer list size mismatch");
        }
        std::copy(init.begin(), init.end(), data.begin());
    }
    template<typename... Args, typename = std::enable_if_t<sizeof...(Args) == N>>
    Vector(Args... args) : data{static_cast<T>(args)...} {}

    // Copy / move
    Vector(const Vector&) = default;
    Vector& operator=(const Vector&) = default;
    Vector(Vector&&) = default;
    Vector& operator=(Vector&&) = default;

    // Access
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
    size_t size() const { return N; }

    // Arithmetic operators
    Vector operator+(const Vector& other) const {
        Vector result;
        for (size_t i = 0; i < N; ++i) result[i] = data[i] + other[i];
        return result;
    }
    Vector operator-(const Vector& other) const {
        Vector result;
        for (size_t i = 0; i < N; ++i) result[i] = data[i] - other[i];
        return result;
    }
    Vector operator*(T scalar) const {
        Vector result;
        for (size_t i = 0; i < N; ++i) result[i] = data[i] * scalar;
        return result;
    }
    Vector operator/(T scalar) const {
        Vector result;
        for (size_t i = 0; i < N; ++i) result[i] = data[i] / scalar;
        return result;
    }
    Vector& operator+=(const Vector& other) {
        for (size_t i = 0; i < N; ++i) data[i] += other[i];
        return *this;
    }
    Vector& operator-=(const Vector& other) {
        for (size_t i = 0; i < N; ++i) data[i] -= other[i];
        return *this;
    }
    Vector& operator*=(T scalar) {
        for (size_t i = 0; i < N; ++i) data[i] *= scalar;
        return *this;
    }
    Vector& operator/=(T scalar) {
        for (size_t i = 0; i < N; ++i) data[i] /= scalar;
        return *this;
    }
    Vector operator-() const {
        Vector result;
        for (size_t i = 0; i < N; ++i) result[i] = -data[i];
        return result;
    }

    // Vector operations
    T dot(const Vector& other) const {
        T sum = 0;
        for (size_t i = 0; i < N; ++i) sum += data[i] * other[i];
        return sum;
    }
    T squaredNorm() const { return dot(*this); }
    T norm() const { return std::sqrt(squaredNorm()); }
    Vector normalized() const {
        T n = norm();
        if (n > 0) return (*this) / n;
        return *this;
    }
    void normalize() { *this = normalized(); }

    // Comparison
    bool operator==(const Vector& other) const {
        for (size_t i = 0; i < N; ++i) if (data[i] != other[i]) return false;
        return true;
    }
    bool operator!=(const Vector& other) const { return !(*this == other); }

    // Utilities
    Vector cross(const Vector& other) const {
        static_assert(N == 3, "Cross product only defined for 3D vectors");
        return Vector(data[1]*other[2] - data[2]*other[1],
                      data[2]*other[0] - data[0]*other[2],
                      data[0]*other[1] - data[1]*other[0]);
    }
    Vector<T, 3> cross3(const Vector<T, 3>& other) const {
        static_assert(N == 3, "Cross product only defined for 3D vectors");
        return cross(other);
    }
};

// Common aliases
using Vector2f = Vector<float, 2>;
using Vector3f = Vector<float, 3>;
using Vector4f = Vector<float, 4>;
using Vector2d = Vector<double, 2>;
using Vector3d = Vector<double, 3>;
using Vector4d = Vector<double, 4>;
using Vector2 = Vector<real, 2>;
using Vector3 = Vector<real, 3>;
using Vector4 = Vector<real, 4>;

//------------------------------------------------------------------------------
// Quaternion class (Hamilton convention)
//------------------------------------------------------------------------------
template<typename T>
class Quaternion {
public:
    T w, x, y, z;

    Quaternion() : w(1), x(0), y(0), z(0) {}
    Quaternion(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}
    Quaternion(const Vector<T, 3>& axis, T angle) {
        T half = angle * 0.5;
        T s = std::sin(half);
        w = std::cos(half);
        x = axis[0] * s;
        y = axis[1] * s;
        z = axis[2] * s;
    }

    // Access
    T& operator[](size_t i) {
        switch (i) {
            case 0: return w;
            case 1: return x;
            case 2: return y;
            case 3: return z;
            default: throw std::out_of_range("Quaternion index out of range");
        }
    }
    const T& operator[](size_t i) const {
        switch (i) {
            case 0: return w;
            case 1: return x;
            case 2: return y;
            case 3: return z;
            default: throw std::out_of_range("Quaternion index out of range");
        }
    }

    // Operations
    Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }
    T squaredNorm() const { return w*w + x*x + y*y + z*z; }
    T norm() const { return std::sqrt(squaredNorm()); }
    Quaternion normalized() const {
        T n = norm();
        if (n > 0) return Quaternion(w/n, x/n, y/n, z/n);
        return *this;
    }
    Quaternion inverse() const { return conjugate() / squaredNorm(); }

    Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w*other.w - x*other.x - y*other.y - z*other.z,
            w*other.x + x*other.w + y*other.z - z*other.y,
            w*other.y - x*other.z + y*other.w + z*other.x,
            w*other.z + x*other.y - y*other.x + z*other.w
        );
    }
    Quaternion operator*(T scalar) const { return Quaternion(w*scalar, x*scalar, y*scalar, z*scalar); }
    Quaternion operator/(T scalar) const { return Quaternion(w/scalar, x/scalar, y/scalar, z/scalar); }
    Quaternion operator+(const Quaternion& other) const {
        return Quaternion(w+other.w, x+other.x, y+other.y, z+other.z);
    }
    Quaternion operator-() const { return Quaternion(-w, -x, -y, -z); }

    // Rotate a vector
    Vector<T, 3> rotate(const Vector<T, 3>& v) const {
        Quaternion qv(0, v[0], v[1], v[2]);
        Quaternion result = (*this) * qv * conjugate();
        return Vector<T, 3>(result.x, result.y, result.z);
    }

    // Convert to rotation matrix
    Matrix3<T> toRotationMatrix() const;

    // Spherical linear interpolation
    static Quaternion slerp(const Quaternion& a, const Quaternion& b, T t) {
        T cosTheta = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
        Quaternion b2 = b;
        if (cosTheta < 0) {
            b2 = -b2;
            cosTheta = -cosTheta;
        }
        if (cosTheta > 0.9995) {
            Quaternion result = a * (1-t) + b2 * t;
            return result.normalized();
        }
        T theta = std::acos(cosTheta);
        T sinTheta = std::sin(theta);
        T w1 = std::sin((1-t)*theta) / sinTheta;
        T w2 = std::sin(t*theta) / sinTheta;
        return (a * w1 + b2 * w2).normalized();
    }
};

using Quatf = Quaternion<float>;
using Quatd = Quaternion<double>;
using Quat = Quaternion<real>;

//------------------------------------------------------------------------------
// 3x3 Matrix class
//------------------------------------------------------------------------------
template<typename T>
class Matrix3 {
public:
    std::array<T, 9> data; // column-major storage

    Matrix3() : data{1,0,0, 0,1,0, 0,0,1} {} // identity
    Matrix3(T diag) : data{diag,0,0, 0,diag,0, 0,0,diag} {}
    Matrix3(std::initializer_list<T> init) {
        if (init.size() != 9) throw std::invalid_argument("Matrix3 requires 9 elements");
        std::copy(init.begin(), init.end(), data.begin());
    }

    // Access (row, col)
    T& operator()(size_t row, size_t col) { return data[col*3 + row]; }
    const T& operator()(size_t row, size_t col) const { return data[col*3 + row]; }

    // Column access
    Vector<T, 3> col(size_t c) const {
        return Vector<T, 3>(data[c*3], data[c*3+1], data[c*3+2]);
    }
    void setCol(size_t c, const Vector<T, 3>& v) {
        data[c*3] = v[0];
        data[c*3+1] = v[1];
        data[c*3+2] = v[2];
    }

    // Row access
    Vector<T, 3> row(size_t r) const {
        return Vector<T, 3>(data[r], data[r+3], data[r+6]);
    }
    void setRow(size_t r, const Vector<T, 3>& v) {
        data[r] = v[0];
        data[r+3] = v[1];
        data[r+6] = v[2];
    }

    // Matrix operations
    Matrix3 operator*(const Matrix3& other) const {
        Matrix3 result(0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T sum = 0;
                for (int k = 0; k < 3; ++k) sum += (*this)(i,k) * other(k,j);
                result(i,j) = sum;
            }
        }
        return result;
    }
    Vector<T, 3> operator*(const Vector<T, 3>& v) const {
        return Vector<T, 3>(
            (*this)(0,0)*v[0] + (*this)(0,1)*v[1] + (*this)(0,2)*v[2],
            (*this)(1,0)*v[0] + (*this)(1,1)*v[1] + (*this)(1,2)*v[2],
            (*this)(2,0)*v[0] + (*this)(2,1)*v[1] + (*this)(2,2)*v[2]
        );
    }
    Matrix3 operator*(T scalar) const {
        Matrix3 result;
        for (size_t i = 0; i < 9; ++i) result.data[i] = data[i] * scalar;
        return result;
    }
    Matrix3 operator+(const Matrix3& other) const {
        Matrix3 result;
        for (size_t i = 0; i < 9; ++i) result.data[i] = data[i] + other.data[i];
        return result;
    }
    Matrix3 operator-(const Matrix3& other) const {
        Matrix3 result;
        for (size_t i = 0; i < 9; ++i) result.data[i] = data[i] - other.data[i];
        return result;
    }

    Matrix3 transpose() const {
        Matrix3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i,j) = (*this)(j,i);
        return result;
    }
    T determinant() const {
        return (*this)(0,0)*((*this)(1,1)*(*this)(2,2) - (*this)(1,2)*(*this)(2,1))
             - (*this)(0,1)*((*this)(1,0)*(*this)(2,2) - (*this)(1,2)*(*this)(2,0))
             + (*this)(0,2)*((*this)(1,0)*(*this)(2,1) - (*this)(1,1)*(*this)(2,0));
    }
    Matrix3 inverse() const {
        T det = determinant();
        if (std::abs(det) < 1e-12) throw std::runtime_error("Matrix is singular");
        T invDet = 1 / det;
        Matrix3 result;
        result(0,0) = ((*this)(1,1)*(*this)(2,2) - (*this)(1,2)*(*this)(2,1)) * invDet;
        result(0,1) = ((*this)(0,2)*(*this)(2,1) - (*this)(0,1)*(*this)(2,2)) * invDet;
        result(0,2) = ((*this)(0,1)*(*this)(1,2) - (*this)(0,2)*(*this)(1,1)) * invDet;
        result(1,0) = ((*this)(1,2)*(*this)(2,0) - (*this)(1,0)*(*this)(2,2)) * invDet;
        result(1,1) = ((*this)(0,0)*(*this)(2,2) - (*this)(0,2)*(*this)(2,0)) * invDet;
        result(1,2) = ((*this)(0,2)*(*this)(1,0) - (*this)(0,0)*(*this)(1,2)) * invDet;
        result(2,0) = ((*this)(1,0)*(*this)(2,1) - (*this)(1,1)*(*this)(2,0)) * invDet;
        result(2,1) = ((*this)(0,1)*(*this)(2,0) - (*this)(0,0)*(*this)(2,1)) * invDet;
        result(2,2) = ((*this)(0,0)*(*this)(1,1) - (*this)(0,1)*(*this)(1,0)) * invDet;
        return result;
    }

    static Matrix3 fromQuaternion(const Quaternion<T>& q) {
        T xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
        T xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
        T wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
        Matrix3 result;
        result(0,0) = 1 - 2*(yy + zz);
        result(0,1) = 2*(xy - wz);
        result(0,2) = 2*(xz + wy);
        result(1,0) = 2*(xy + wz);
        result(1,1) = 1 - 2*(xx + zz);
        result(1,2) = 2*(yz - wx);
        result(2,0) = 2*(xz - wy);
        result(2,1) = 2*(yz + wx);
        result(2,2) = 1 - 2*(xx + yy);
        return result;
    }
};

template<typename T>
Matrix3<T> Quaternion<T>::toRotationMatrix() const {
    return Matrix3<T>::fromQuaternion(*this);
}

using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;
using Matrix3r = Matrix3<real>;

//------------------------------------------------------------------------------
// 4x4 Matrix class (for homogeneous transformations)
//------------------------------------------------------------------------------
template<typename T>
class Matrix4 {
public:
    std::array<T, 16> data; // column-major

    Matrix4() : data{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1} {}
    Matrix4(T diag) : data{diag,0,0,0, 0,diag,0,0, 0,0,diag,0, 0,0,0,diag} {}
    Matrix4(std::initializer_list<T> init) {
        if (init.size() != 16) throw std::invalid_argument("Matrix4 requires 16 elements");
        std::copy(init.begin(), init.end(), data.begin());
    }

    T& operator()(size_t row, size_t col) { return data[col*4 + row]; }
    const T& operator()(size_t row, size_t col) const { return data[col*4 + row]; }

    Vector<T, 4> col(size_t c) const {
        return Vector<T, 4>(data[c*4], data[c*4+1], data[c*4+2], data[c*4+3]);
    }
    void setCol(size_t c, const Vector<T, 4>& v) {
        data[c*4] = v[0]; data[c*4+1] = v[1]; data[c*4+2] = v[2]; data[c*4+3] = v[3];
    }

    Matrix4 operator*(const Matrix4& other) const {
        Matrix4 result(0);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    result(i,j) += (*this)(i,k) * other(k,j);
        return result;
    }
    Vector<T, 3> transformPoint(const Vector<T, 3>& v) const {
        return Vector<T, 3>(
            (*this)(0,0)*v[0] + (*this)(0,1)*v[1] + (*this)(0,2)*v[2] + (*this)(0,3),
            (*this)(1,0)*v[0] + (*this)(1,1)*v[1] + (*this)(1,2)*v[2] + (*this)(1,3),
            (*this)(2,0)*v[0] + (*this)(2,1)*v[1] + (*this)(2,2)*v[2] + (*this)(2,3)
        );
    }
    Vector<T, 3> transformVector(const Vector<T, 3>& v) const {
        return Vector<T, 3>(
            (*this)(0,0)*v[0] + (*this)(0,1)*v[1] + (*this)(0,2)*v[2],
            (*this)(1,0)*v[0] + (*this)(1,1)*v[1] + (*this)(1,2)*v[2],
            (*this)(2,0)*v[0] + (*this)(2,1)*v[1] + (*this)(2,2)*v[2]
        );
    }

    Matrix4 inverse() const {
        // Full 4x4 inversion using adjugate/determinant (for general transform)
        T inv[16], det;
        const T* m = data.data();
        inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
        inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
        inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
        inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
        inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
        inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
        inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
        inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
        inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
        inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
        inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
        inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
        inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
        inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
        inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
        inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
        det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
        if (std::abs(det) < 1e-12) throw std::runtime_error("Matrix4 is singular");
        det = 1.0 / det;
        Matrix4 result;
        for (int i = 0; i < 16; ++i) result.data[i] = inv[i] * det;
        return result;
    }
};

using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;
using Matrix4r = Matrix4<real>;

//------------------------------------------------------------------------------
// Transform class (position + rotation)
//------------------------------------------------------------------------------
template<typename T>
class Transform {
public:
    Vector<T, 3> translation;
    Quaternion<T> rotation;

    Transform() : translation(0), rotation() {}
    Transform(const Vector<T, 3>& t, const Quaternion<T>& r) : translation(t), rotation(r) {}

    Matrix4<T> matrix() const {
        Matrix3<T> rot = rotation.toRotationMatrix();
        Matrix4<T> mat;
        mat(0,0)=rot(0,0); mat(0,1)=rot(0,1); mat(0,2)=rot(0,2); mat(0,3)=translation[0];
        mat(1,0)=rot(1,0); mat(1,1)=rot(1,1); mat(1,2)=rot(1,2); mat(1,3)=translation[1];
        mat(2,0)=rot(2,0); mat(2,1)=rot(2,1); mat(2,2)=rot(2,2); mat(2,3)=translation[2];
        return mat;
    }

    Transform inverse() const {
        Quaternion<T> invRot = rotation.conjugate();
        return Transform(invRot.rotate(-translation), invRot);
    }

    Transform operator*(const Transform& other) const {
        return Transform(
            translation + rotation.rotate(other.translation),
            rotation * other.rotation
        );
    }

    Vector<T, 3> transformPoint(const Vector<T, 3>& p) const {
        return translation + rotation.rotate(p);
    }
    Vector<T, 3> transformVector(const Vector<T, 3>& v) const {
        return rotation.rotate(v);
    }
};

using Transformf = Transform<float>;
using Transformd = Transform<double>;
using Transformr = Transform<real>;

//------------------------------------------------------------------------------
// Bounding volume types
//------------------------------------------------------------------------------
class BoundingBox {
public:
    Vector3 min, max;

    BoundingBox() : min(std::numeric_limits<real>::max()), max(std::numeric_limits<real>::lowest()) {}
    BoundingBox(const Vector3& min_, const Vector3& max_) : min(min_), max(max_) {}

    void expand(const Vector3& point) {
        for (int i = 0; i < 3; ++i) {
            min[i] = std::min(min[i], point[i]);
            max[i] = std::max(max[i], point[i]);
        }
    }
    void expand(const BoundingBox& other) {
        expand(other.min);
        expand(other.max);
    }
    Vector3 center() const { return (min + max) * 0.5; }
    Vector3 extents() const { return (max - min) * 0.5; }
    bool contains(const Vector3& point) const {
        return point[0] >= min[0] && point[0] <= max[0] &&
               point[1] >= min[1] && point[1] <= max[1] &&
               point[2] >= min[2] && point[2] <= max[2];
    }
    bool intersects(const BoundingBox& other) const {
        return (min[0] <= other.max[0] && max[0] >= other.min[0]) &&
               (min[1] <= other.max[1] && max[1] >= other.min[1]) &&
               (min[2] <= other.max[2] && max[2] >= other.min[2]);
    }
    real volume() const {
        Vector3 e = max - min;
        return e[0] * e[1] * e[2];
    }
};

// Axis-aligned bounding box
using AABB = BoundingBox;

// Oriented bounding box
class OBB {
public:
    Vector3 center;
    Vector3 extents;
    Quat rotation;

    OBB() : center(0), extents(0), rotation() {}
    OBB(const Vector3& c, const Vector3& e, const Quat& r) : center(c), extents(e), rotation(r) {}

    AABB toAABB() const {
        Matrix3r rot = rotation.toRotationMatrix();
        Vector3 ext = extents;
        Vector3 aabb_extents = Vector3(
            std::abs(rot(0,0))*ext[0] + std::abs(rot(0,1))*ext[1] + std::abs(rot(0,2))*ext[2],
            std::abs(rot(1,0))*ext[0] + std::abs(rot(1,1))*ext[1] + std::abs(rot(1,2))*ext[2],
            std::abs(rot(2,0))*ext[0] + std::abs(rot(2,1))*ext[1] + std::abs(rot(2,2))*ext[2]
        );
        return AABB(center - aabb_extents, center + aabb_extents);
    }
};

// Sphere
class Sphere {
public:
    Vector3 center;
    real radius;

    Sphere() : center(0), radius(0) {}
    Sphere(const Vector3& c, real r) : center(c), radius(r) {}

    bool contains(const Vector3& point) const {
        return (point - center).squaredNorm() <= radius * radius;
    }
    bool intersects(const Sphere& other) const {
        return (center - other.center).squaredNorm() <= (radius + other.radius) * (radius + other.radius);
    }
    bool intersects(const AABB& aabb) const {
        Vector3 closest = center;
        for (int i = 0; i < 3; ++i) {
            closest[i] = std::max(aabb.min[i], std::min(closest[i], aabb.max[i]));
        }
        return (closest - center).squaredNorm() <= radius * radius;
    }
};

// Ray
class Ray {
public:
    Vector3 origin;
    Vector3 direction;

    Ray() : origin(0), direction(0,0,1) {}
    Ray(const Vector3& o, const Vector3& d) : origin(o), direction(d.normalized()) {}

    Vector3 pointAt(real t) const { return origin + direction * t; }
    bool intersectsAABB(const AABB& aabb, real& tmin, real& tmax) const {
        tmin = std::numeric_limits<real>::lowest();
        tmax = std::numeric_limits<real>::max();
        for (int i = 0; i < 3; ++i) {
            if (std::abs(direction[i]) < 1e-12) {
                if (origin[i] < aabb.min[i] || origin[i] > aabb.max[i]) return false;
            } else {
                real ood = 1.0 / direction[i];
                real t1 = (aabb.min[i] - origin[i]) * ood;
                real t2 = (aabb.max[i] - origin[i]) * ood;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return false;
            }
        }
        return true;
    }
};

// Plane
class Plane {
public:
    Vector3 normal;
    real d; // distance from origin

    Plane() : normal(0,0,1), d(0) {}
    Plane(const Vector3& n, real dist) : normal(n.normalized()), d(dist) {}
    Plane(const Vector3& n, const Vector3& point) : normal(n.normalized()), d(-normal.dot(point)) {}

    real signedDistance(const Vector3& point) const {
        return normal.dot(point) + d;
    }
    bool pointSide(const Vector3& point) const {
        return signedDistance(point) >= 0;
    }
};

//------------------------------------------------------------------------------
// Particle data structures (used in MPM, SPH, etc.)
//------------------------------------------------------------------------------
struct Particle {
    Vector3 position;
    Vector3 velocity;
    real mass;
    real density;
    real pressure;
    uint32_t material_id;
    uint32_t flags;

    Particle() : position(0), velocity(0), mass(1), density(0), pressure(0), material_id(0), flags(0) {}
};

struct MPMParticle : public Particle {
    Matrix3r deformation_gradient;  // F
    Matrix3r affine_velocity;       // C matrix for APIC
    real volume;
    uint32_t cell_hash;

    MPMParticle() : deformation_gradient(1), affine_velocity(0), volume(1), cell_hash(0) {}
};

struct SPHParticle : public Particle {
    Vector3 acceleration;
    real rest_density;
    real viscosity;
    real surface_tension;
    std::array<uint32_t, 64> neighbors;
    uint32_t neighbor_count;

    SPHParticle() : acceleration(0), rest_density(1000), viscosity(0.01), surface_tension(0), neighbor_count(0) {}
};

//------------------------------------------------------------------------------
// Mesh data structures
//------------------------------------------------------------------------------
struct Triangle {
    std::array<uint32_t, 3> indices;
    Vector3 normal;
};

struct Mesh {
    std::vector<Vector3> vertices;
    std::vector<Vector3> normals;
    std::vector<Vector2> texcoords;
    std::vector<uint32_t> indices;  // triangle list
    std::string name;

    void computeNormals() {
        normals.resize(vertices.size(), Vector3(0));
        for (size_t i = 0; i < indices.size(); i += 3) {
            Vector3 v0 = vertices[indices[i]];
            Vector3 v1 = vertices[indices[i+1]];
            Vector3 v2 = vertices[indices[i+2]];
            Vector3 n = (v1 - v0).cross(v2 - v0).normalized();
            normals[indices[i]] += n;
            normals[indices[i+1]] += n;
            normals[indices[i+2]] += n;
        }
        for (auto& n : normals) n.normalize();
    }

    AABB computeAABB() const {
        AABB box;
        for (const auto& v : vertices) box.expand(v);
        return box;
    }
};

//------------------------------------------------------------------------------
// Joint / constraint data
//------------------------------------------------------------------------------
struct JointInfo {
    std::string name;
    std::string type; // "revolute", "prismatic", "fixed", etc.
    std::string parent_link;
    std::string child_link;
    Transformr origin;
    Vector3 axis;
    real lower_limit;
    real upper_limit;
    real effort_limit;
    real velocity_limit;
};

//------------------------------------------------------------------------------
// Link / rigid body data
//------------------------------------------------------------------------------
struct LinkInfo {
    std::string name;
    Transformr inertial_origin;
    real mass;
    Vector3 inertia;
    Mesh visual_mesh;
    Mesh collision_mesh;
};

} // namespace datatypes
} // namespace genesis