// genesis/datatypes.cpp

#include "genesis/datatypes.h"
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <algorithm>

namespace genesis {
namespace datatypes {

//------------------------------------------------------------------------------
// Stream output operators for debugging
//------------------------------------------------------------------------------

template<typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const Vector<T, N>& v) {
    os << "[";
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) os << ", ";
        os << v[i];
    }
    os << "]";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Quaternion<T>& q) {
    os << "Quat(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix3<T>& m) {
    os << "Matrix3(\n";
    for (int r = 0; r < 3; ++r) {
        os << "  ";
        for (int c = 0; c < 3; ++c) {
            os << std::setw(12) << m(r, c) << " ";
        }
        os << "\n";
    }
    os << ")";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix4<T>& m) {
    os << "Matrix4(\n";
    for (int r = 0; r < 4; ++r) {
        os << "  ";
        for (int c = 0; c < 4; ++c) {
            os << std::setw(12) << m(r, c) << " ";
        }
        os << "\n";
    }
    os << ")";
    return os;
}

//------------------------------------------------------------------------------
// String conversion utilities
//------------------------------------------------------------------------------

template<typename T, size_t N>
std::string to_string(const Vector<T, N>& v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

template<typename T>
std::string to_string(const Quaternion<T>& q) {
    std::ostringstream oss;
    oss << q;
    return oss.str();
}

template<typename T>
std::string to_string(const Matrix3<T>& m) {
    std::ostringstream oss;
    oss << m;
    return oss.str();
}

template<typename T>
std::string to_string(const Matrix4<T>& m) {
    std::ostringstream oss;
    oss << m;
    return oss.str();
}

//------------------------------------------------------------------------------
// Additional vector operations not defined inline
//------------------------------------------------------------------------------

template<typename T, size_t N>
Vector<T, N> operator*(T scalar, const Vector<T, N>& v) {
    return v * scalar;
}

template<typename T>
Vector<T, 3> orthogonal(const Vector<T, 3>& v) {
    // Find a non-parallel vector
    Vector<T, 3> other;
    if (std::abs(v[0]) < 0.9) {
        other = Vector<T, 3>(1, 0, 0);
    } else if (std::abs(v[1]) < 0.9) {
        other = Vector<T, 3>(0, 1, 0);
    } else {
        other = Vector<T, 3>(0, 0, 1);
    }
    Vector<T, 3> result = v.cross(other).normalized();
    return result;
}

template<typename T>
std::pair<Vector<T, 3>, Vector<T, 3>> basis(const Vector<T, 3>& n) {
    Vector<T, 3> u = orthogonal(n);
    Vector<T, 3> v = n.cross(u).normalized();
    return {u, v};
}

//------------------------------------------------------------------------------
// Quaternion utilities
//------------------------------------------------------------------------------

template<typename T>
Quaternion<T> Quaternion<T>::fromEulerAngles(T roll, T pitch, T yaw) {
    T cr = std::cos(roll * 0.5);
    T sr = std::sin(roll * 0.5);
    T cp = std::cos(pitch * 0.5);
    T sp = std::sin(pitch * 0.5);
    T cy = std::cos(yaw * 0.5);
    T sy = std::sin(yaw * 0.5);
    return Quaternion<T>(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    );
}

template<typename T>
void Quaternion<T>::toEulerAngles(T& roll, T& pitch, T& yaw) const {
    // roll (x-axis rotation)
    T sinr_cosp = 2 * (w * x + y * z);
    T cosr_cosp = 1 - 2 * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    T sinp = 2 * (w * y - z * x);
    if (std::abs(sinp) >= 1)
        pitch = std::copysign(3.14159265358979323846 / 2, sinp);
    else
        pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    T siny_cosp = 2 * (w * z + x * y);
    T cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

template<typename T>
Quaternion<T> Quaternion<T>::lookAt(const Vector<T, 3>& direction, const Vector<T, 3>& up) {
    Vector<T, 3> forward = direction.normalized();
    Vector<T, 3> right = up.cross(forward).normalized();
    Vector<T, 3> newUp = forward.cross(right);
    
    // Build rotation matrix
    Matrix3<T> rot;
    rot(0,0) = right[0]; rot(0,1) = right[1]; rot(0,2) = right[2];
    rot(1,0) = newUp[0]; rot(1,1) = newUp[1]; rot(1,2) = newUp[2];
    rot(2,0) = forward[0]; rot(2,1) = forward[1]; rot(2,2) = forward[2];
    
    // Convert to quaternion
    T trace = rot(0,0) + rot(1,1) + rot(2,2);
    T w, x, y, z;
    if (trace > 0) {
        T s = 0.5 / std::sqrt(trace + 1.0);
        w = 0.25 / s;
        x = (rot(2,1) - rot(1,2)) * s;
        y = (rot(0,2) - rot(2,0)) * s;
        z = (rot(1,0) - rot(0,1)) * s;
    } else {
        if (rot(0,0) > rot(1,1) && rot(0,0) > rot(2,2)) {
            T s = 2.0 * std::sqrt(1.0 + rot(0,0) - rot(1,1) - rot(2,2));
            w = (rot(2,1) - rot(1,2)) / s;
            x = 0.25 * s;
            y = (rot(0,1) + rot(1,0)) / s;
            z = (rot(0,2) + rot(2,0)) / s;
        } else if (rot(1,1) > rot(2,2)) {
            T s = 2.0 * std::sqrt(1.0 + rot(1,1) - rot(0,0) - rot(2,2));
            w = (rot(0,2) - rot(2,0)) / s;
            x = (rot(0,1) + rot(1,0)) / s;
            y = 0.25 * s;
            z = (rot(1,2) + rot(2,1)) / s;
        } else {
            T s = 2.0 * std::sqrt(1.0 + rot(2,2) - rot(0,0) - rot(1,1));
            w = (rot(1,0) - rot(0,1)) / s;
            x = (rot(0,2) + rot(2,0)) / s;
            y = (rot(1,2) + rot(2,1)) / s;
            z = 0.25 * s;
        }
    }
    return Quaternion<T>(w, x, y, z).normalized();
}

//------------------------------------------------------------------------------
// Additional Matrix operations
//------------------------------------------------------------------------------

template<typename T>
Matrix3<T> Matrix3<T>::fromAxes(const Vector<T, 3>& xAxis, const Vector<T, 3>& yAxis, const Vector<T, 3>& zAxis) {
    Matrix3<T> m;
    m(0,0) = xAxis[0]; m(0,1) = yAxis[0]; m(0,2) = zAxis[0];
    m(1,0) = xAxis[1]; m(1,1) = yAxis[1]; m(1,2) = zAxis[1];
    m(2,0) = xAxis[2]; m(2,1) = yAxis[2]; m(2,2) = zAxis[2];
    return m;
}

template<typename T>
Matrix3<T> Matrix3<T>::rotationX(T angle) {
    T c = std::cos(angle);
    T s = std::sin(angle);
    return Matrix3<T>{
        1, 0, 0,
        0, c,-s,
        0, s, c
    };
}

template<typename T>
Matrix3<T> Matrix3<T>::rotationY(T angle) {
    T c = std::cos(angle);
    T s = std::sin(angle);
    return Matrix3<T>{
        c, 0, s,
        0, 1, 0,
       -s, 0, c
    };
}

template<typename T>
Matrix3<T> Matrix3<T>::rotationZ(T angle) {
    T c = std::cos(angle);
    T s = std::sin(angle);
    return Matrix3<T>{
        c,-s, 0,
        s, c, 0,
        0, 0, 1
    };
}

template<typename T>
Matrix3<T> Matrix3<T>::scale(const Vector<T, 3>& s) {
    return Matrix3<T>{
        s[0], 0,    0,
        0,    s[1], 0,
        0,    0,    s[2]
    };
}

template<typename T>
Matrix3<T> Matrix3<T>::skewSymmetric(const Vector<T, 3>& v) {
    return Matrix3<T>{
        0,    -v[2],  v[1],
        v[2],  0,    -v[0],
       -v[1],  v[0],  0
    };
}

template<typename T>
Matrix4<T> Matrix4<T>::translation(const Vector<T, 3>& t) {
    Matrix4<T> m;
    m(0,3) = t[0];
    m(1,3) = t[1];
    m(2,3) = t[2];
    return m;
}

template<typename T>
Matrix4<T> Matrix4<T>::rotation(const Matrix3<T>& r) {
    Matrix4<T> m;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            m(i,j) = r(i,j);
    return m;
}

template<typename T>
Matrix4<T> Matrix4<T>::scale(const Vector<T, 3>& s) {
    return Matrix4<T>{
        s[0], 0,    0,    0,
        0,    s[1], 0,    0,
        0,    0,    s[2], 0,
        0,    0,    0,    1
    };
}

template<typename T>
Matrix4<T> Matrix4<T>::perspective(T fovY, T aspect, T near, T far) {
    T tanHalfFov = std::tan(fovY * 0.5);
    Matrix4<T> m(0);
    m(0,0) = 1.0 / (aspect * tanHalfFov);
    m(1,1) = 1.0 / tanHalfFov;
    m(2,2) = -(far + near) / (far - near);
    m(2,3) = -(2.0 * far * near) / (far - near);
    m(3,2) = -1.0;
    return m;
}

template<typename T>
Matrix4<T> Matrix4<T>::orthographic(T left, T right, T bottom, T top, T near, T far) {
    Matrix4<T> m;
    m(0,0) = 2.0 / (right - left);
    m(1,1) = 2.0 / (top - bottom);
    m(2,2) = -2.0 / (far - near);
    m(0,3) = -(right + left) / (right - left);
    m(1,3) = -(top + bottom) / (top - bottom);
    m(2,3) = -(far + near) / (far - near);
    return m;
}

template<typename T>
Matrix4<T> Matrix4<T>::lookAt(const Vector<T, 3>& eye, const Vector<T, 3>& center, const Vector<T, 3>& up) {
    Vector<T, 3> f = (center - eye).normalized();
    Vector<T, 3> s = f.cross(up).normalized();
    Vector<T, 3> u = s.cross(f);
    
    Matrix4<T> m;
    m(0,0) = s[0]; m(0,1) = s[1]; m(0,2) = s[2]; m(0,3) = -s.dot(eye);
    m(1,0) = u[0]; m(1,1) = u[1]; m(1,2) = u[2]; m(1,3) = -u.dot(eye);
    m(2,0) = -f[0]; m(2,1) = -f[1]; m(2,2) = -f[2]; m(2,3) = f.dot(eye);
    m(3,3) = 1.0;
    return m;
}

//------------------------------------------------------------------------------
// Bounding volume additional methods
//------------------------------------------------------------------------------

real AABB::surfaceArea() const {
    Vector3 e = max - min;
    return 2.0 * (e[0]*e[1] + e[1]*e[2] + e[2]*e[0]);
}

Vector3 AABB::closestPoint(const Vector3& point) const {
    Vector3 result;
    for (int i = 0; i < 3; ++i) {
        result[i] = std::max(min[i], std::min(point[i], max[i]));
    }
    return result;
}

real AABB::distanceToPoint(const Vector3& point) const {
    return (point - closestPoint(point)).norm();
}

bool AABB::rayIntersect(const Ray& ray, real& tmin, real& tmax) const {
    return ray.intersectsAABB(*this, tmin, tmax);
}

void AABB::transform(const Matrix4r& m) {
    // Transform the 8 corners and recompute AABB
    Vector3 corners[8] = {
        Vector3(min[0], min[1], min[2]),
        Vector3(max[0], min[1], min[2]),
        Vector3(min[0], max[1], min[2]),
        Vector3(max[0], max[1], min[2]),
        Vector3(min[0], min[1], max[2]),
        Vector3(max[0], min[1], max[2]),
        Vector3(min[0], max[1], max[2]),
        Vector3(max[0], max[1], max[2])
    };
    *this = AABB();
    for (const auto& corner : corners) {
        expand(m.transformPoint(corner));
    }
}

//------------------------------------------------------------------------------
// Explicit template instantiations for common types
//------------------------------------------------------------------------------

// Vector instantiations
template class Vector<float, 2>;
template class Vector<float, 3>;
template class Vector<float, 4>;
template class Vector<double, 2>;
template class Vector<double, 3>;
template class Vector<double, 4>;

template std::ostream& operator<<(std::ostream&, const Vector<float, 2>&);
template std::ostream& operator<<(std::ostream&, const Vector<float, 3>&);
template std::ostream& operator<<(std::ostream&, const Vector<float, 4>&);
template std::ostream& operator<<(std::ostream&, const Vector<double, 2>&);
template std::ostream& operator<<(std::ostream&, const Vector<double, 3>&);
template std::ostream& operator<<(std::ostream&, const Vector<double, 4>&);

template Vector<float, 2> operator*<float, 2>(float, const Vector<float, 2>&);
template Vector<float, 3> operator*<float, 3>(float, const Vector<float, 3>&);
template Vector<float, 4> operator*<float, 4>(float, const Vector<float, 4>&);
template Vector<double, 2> operator*<double, 2>(double, const Vector<double, 2>&);
template Vector<double, 3> operator*<double, 3>(double, const Vector<double, 3>&);
template Vector<double, 4> operator*<double, 4>(double, const Vector<double, 4>&);

template Vector<float, 3> orthogonal<float>(const Vector<float, 3>&);
template Vector<double, 3> orthogonal<double>(const Vector<double, 3>&);
template std::pair<Vector<float, 3>, Vector<float, 3>> basis<float>(const Vector<float, 3>&);
template std::pair<Vector<double, 3>, Vector<double, 3>> basis<double>(const Vector<double, 3>&);

template std::string to_string(const Vector<float, 2>&);
template std::string to_string(const Vector<float, 3>&);
template std::string to_string(const Vector<float, 4>&);
template std::string to_string(const Vector<double, 2>&);
template std::string to_string(const Vector<double, 3>&);
template std::string to_string(const Vector<double, 4>&);

// Quaternion instantiations
template class Quaternion<float>;
template class Quaternion<double>;

template std::ostream& operator<<(std::ostream&, const Quaternion<float>&);
template std::ostream& operator<<(std::ostream&, const Quaternion<double>&);

template Quaternion<float> Quaternion<float>::fromEulerAngles(float, float, float);
template Quaternion<double> Quaternion<double>::fromEulerAngles(double, double, double);
template void Quaternion<float>::toEulerAngles(float&, float&, float&) const;
template void Quaternion<double>::toEulerAngles(double&, double&, double&) const;
template Quaternion<float> Quaternion<float>::lookAt(const Vector<float, 3>&, const Vector<float, 3>&);
template Quaternion<double> Quaternion<double>::lookAt(const Vector<double, 3>&, const Vector<double, 3>&);

template std::string to_string(const Quaternion<float>&);
template std::string to_string(const Quaternion<double>&);

// Matrix3 instantiations
template class Matrix3<float>;
template class Matrix3<double>;

template std::ostream& operator<<(std::ostream&, const Matrix3<float>&);
template std::ostream& operator<<(std::ostream&, const Matrix3<double>&);

template Matrix3<float> Matrix3<float>::fromAxes(const Vector<float, 3>&, const Vector<float, 3>&, const Vector<float, 3>&);
template Matrix3<double> Matrix3<double>::fromAxes(const Vector<double, 3>&, const Vector<double, 3>&, const Vector<double, 3>&);
template Matrix3<float> Matrix3<float>::rotationX(float);
template Matrix3<double> Matrix3<double>::rotationX(double);
template Matrix3<float> Matrix3<float>::rotationY(float);
template Matrix3<double> Matrix3<double>::rotationY(double);
template Matrix3<float> Matrix3<float>::rotationZ(float);
template Matrix3<double> Matrix3<double>::rotationZ(double);
template Matrix3<float> Matrix3<float>::scale(const Vector<float, 3>&);
template Matrix3<double> Matrix3<double>::scale(const Vector<double, 3>&);
template Matrix3<float> Matrix3<float>::skewSymmetric(const Vector<float, 3>&);
template Matrix3<double> Matrix3<double>::skewSymmetric(const Vector<double, 3>&);

template std::string to_string(const Matrix3<float>&);
template std::string to_string(const Matrix3<double>&);

// Matrix4 instantiations
template class Matrix4<float>;
template class Matrix4<double>;

template std::ostream& operator<<(std::ostream&, const Matrix4<float>&);
template std::ostream& operator<<(std::ostream&, const Matrix4<double>&);

template Matrix4<float> Matrix4<float>::translation(const Vector<float, 3>&);
template Matrix4<double> Matrix4<double>::translation(const Vector<double, 3>&);
template Matrix4<float> Matrix4<float>::rotation(const Matrix3<float>&);
template Matrix4<double> Matrix4<double>::rotation(const Matrix3<double>&);
template Matrix4<float> Matrix4<float>::scale(const Vector<float, 3>&);
template Matrix4<double> Matrix4<double>::scale(const Vector<double, 3>&);
template Matrix4<float> Matrix4<float>::perspective(float, float, float, float);
template Matrix4<double> Matrix4<double>::perspective(double, double, double, double);
template Matrix4<float> Matrix4<float>::orthographic(float, float, float, float, float, float);
template Matrix4<double> Matrix4<double>::orthographic(double, double, double, double, double, double);
template Matrix4<float> Matrix4<float>::lookAt(const Vector<float, 3>&, const Vector<float, 3>&, const Vector<float, 3>&);
template Matrix4<double> Matrix4<double>::lookAt(const Vector<double, 3>&, const Vector<double, 3>&, const Vector<double, 3>&);

template std::string to_string(const Matrix4<float>&);
template std::string to_string(const Matrix4<double>&);

// Transform instantiations
template class Transform<float>;
template class Transform<double>;

} // namespace datatypes
} // namespace genesis