// system name : onetbb-warp
// File 0007 : core/math/matrix3.h
// Description : 3x3 matrix for linear transformations, eigenvalues, and SVD.

#ifndef __TBB_WARP_CORE_MATH_MATRIX3_H
#define __TBB_WARP_CORE_MATH_MATRIX3_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>
#include <algorithm>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Matrix3 class template (column‑major storage)
// ============================================================

template<typename T>
struct matrix3 {
    using value_type = T;
    using col_type = vector3<T>;
    using row_type = std::array<T,3>;

    // column‑major: columns[0], columns[1], columns[2]
    col_type col[3];

    // ---- Constructors ----
    constexpr matrix3() noexcept : col{col_type(T(1),T(0),T(0)), col_type(T(0),T(1),T(0)), col_type(T(0),T(0),T(1))} {}
    constexpr matrix3(const col_type& c0, const col_type& c1, const col_type& c2) noexcept : col{c0, c1, c2} {}
    constexpr matrix3(
        T m00, T m01, T m02,
        T m10, T m11, T m12,
        T m20, T m21, T m22) noexcept
        : col{col_type(m00,m10,m20), col_type(m01,m11,m21), col_type(m02,m12,m22)} {}
    explicit constexpr matrix3(const quaternion<T>& q) noexcept {
        T xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
        T xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
        T wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
        col[0] = col_type(T(1)-T(2)*(yy+zz), T(2)*(xy+wz), T(2)*(xz-wy));
        col[1] = col_type(T(2)*(xy-wz), T(1)-T(2)*(xx+zz), T(2)*(yz+wx));
        col[2] = col_type(T(2)*(xz+wy), T(2)*(yz-wx), T(1)-T(2)*(xx+yy));
    }
    constexpr matrix3(T diag) noexcept : col{col_type(diag,T(0),T(0)), col_type(T(0),diag,T(0)), col_type(T(0),T(0),diag)} {}
    template<typename U> constexpr explicit matrix3(const matrix3<U>& m) noexcept : col{col_type(m.col[0]), col_type(m.col[1]), col_type(m.col[2])} {}

    // ---- Access ----
    constexpr col_type& operator[](std::size_t i) noexcept { return col[i]; }
    constexpr const col_type& operator[](std::size_t i) const noexcept { return col[i]; }
    constexpr T& operator()(std::size_t row, std::size_t col_idx) noexcept { return col[col_idx][row]; }
    constexpr const T& operator()(std::size_t row, std::size_t col_idx) const noexcept { return col[col_idx][row]; }

    // ---- Compound assignment ----
    constexpr matrix3& operator+=(const matrix3& m) noexcept { col[0]+=m.col[0]; col[1]+=m.col[1]; col[2]+=m.col[2]; return *this; }
    constexpr matrix3& operator-=(const matrix3& m) noexcept { col[0]-=m.col[0]; col[1]-=m.col[1]; col[2]-=m.col[2]; return *this; }
    constexpr matrix3& operator*=(T s) noexcept { col[0]*=s; col[1]*=s; col[2]*=s; return *this; }
    constexpr matrix3& operator/=(T s) noexcept { col[0]/=s; col[1]/=s; col[2]/=s; return *this; }

    // ---- Unary ----
    constexpr matrix3 operator+() const noexcept { return *this; }
    constexpr matrix3 operator-() const noexcept { return matrix3(-col[0], -col[1], -col[2]); }

    // ---- Conversion ----
    constexpr operator std::array<std::array<T,3>,3>() const noexcept {
        return {{ {col[0].x,col[1].x,col[2].x}, {col[0].y,col[1].y,col[2].y}, {col[0].z,col[1].z,col[2].z} }};
    }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr matrix3<T> operator+(const matrix3<T>& a, const matrix3<T>& b) noexcept { return matrix3<T>(a.col[0]+b.col[0], a.col[1]+b.col[1], a.col[2]+b.col[2]); }
template<typename T> constexpr matrix3<T> operator-(const matrix3<T>& a, const matrix3<T>& b) noexcept { return matrix3<T>(a.col[0]-b.col[0], a.col[1]-b.col[1], a.col[2]-b.col[2]); }
template<typename T> constexpr matrix3<T> operator*(const matrix3<T>& m, T s) noexcept { return matrix3<T>(m.col[0]*s, m.col[1]*s, m.col[2]*s); }
template<typename T> constexpr matrix3<T> operator*(T s, const matrix3<T>& m) noexcept { return m*s; }
template<typename T> constexpr bool operator==(const matrix3<T>& a, const matrix3<T>& b) noexcept { return a.col[0]==b.col[0] && a.col[1]==b.col[1] && a.col[2]==b.col[2]; }
template<typename T> constexpr bool operator!=(const matrix3<T>& a, const matrix3<T>& b) noexcept { return !(a==b); }

// ============================================================
// Matrix‑vector multiplication
// ============================================================

template<typename T>
constexpr vector3<T> operator*(const matrix3<T>& m, const vector3<T>& v) noexcept {
    return m.col[0]*v.x + m.col[1]*v.y + m.col[2]*v.z;
}

// ============================================================
// Matrix‑matrix multiplication
// ============================================================

template<typename T>
constexpr matrix3<T> operator*(const matrix3<T>& a, const matrix3<T>& b) noexcept {
    return matrix3<T>(
        a*b.col[0],
        a*b.col[1],
        a*b.col[2]
    );
}

// ============================================================
// Transpose
// ============================================================

template<typename T>
constexpr matrix3<T> transpose(const matrix3<T>& m) noexcept {
    return matrix3<T>(
        m(0,0), m(1,0), m(2,0),
        m(0,1), m(1,1), m(2,1),
        m(0,2), m(1,2), m(2,2)
    );
}

// ============================================================
// Trace
// ============================================================

template<typename T>
constexpr T trace(const matrix3<T>& m) noexcept {
    return m(0,0) + m(1,1) + m(2,2);
}

// ============================================================
// Determinant
// ============================================================

template<typename T>
constexpr T determinant(const matrix3<T>& m) noexcept {
    return m(0,0)*(m(1,1)*m(2,2) - m(1,2)*m(2,1))
         - m(0,1)*(m(1,0)*m(2,2) - m(1,2)*m(2,0))
         + m(0,2)*(m(1,0)*m(2,1) - m(1,1)*m(2,0));
}

// ============================================================
// Inverse (via cofactors)
// ============================================================

template<typename T>
matrix3<T> inverse(const matrix3<T>& m) noexcept {
    T det = determinant(m);
    if (std::abs(det) < T(FLOAT_EPSILON)) return matrix3<T>(T(1));
    T inv_det = T(1) / det;
    T a=m(0,0), b=m(0,1), c=m(0,2);
    T d=m(1,0), e=m(1,1), f=m(1,2);
    T g=m(2,0), h=m(2,1), i=m(2,2);
    return matrix3<T>(
        (e*i - f*h) * inv_det,
        (c*h - b*i) * inv_det,
        (b*f - c*e) * inv_det,
        (f*g - d*i) * inv_det,
        (a*i - c*g) * inv_det,
        (c*d - a*f) * inv_det,
        (d*h - e*g) * inv_det,
        (b*g - a*h) * inv_det,
        (a*e - b*d) * inv_det
    );
}

// ============================================================
// Adjugate
// ============================================================

template<typename T>
constexpr matrix3<T> adjugate(const matrix3<T>& m) noexcept {
    return matrix3<T>(
        m(1,1)*m(2,2)-m(1,2)*m(2,1), m(0,2)*m(2,1)-m(0,1)*m(2,2), m(0,1)*m(1,2)-m(0,2)*m(1,1),
        m(1,2)*m(2,0)-m(1,0)*m(2,2), m(0,0)*m(2,2)-m(0,2)*m(2,0), m(0,2)*m(1,0)-m(0,0)*m(1,2),
        m(1,0)*m(2,1)-m(1,1)*m(2,0), m(0,1)*m(2,0)-m(0,0)*m(2,1), m(0,0)*m(1,1)-m(0,1)*m(1,0)
    );
}

// ============================================================
// Frobenius norm
// ============================================================

template<typename T>
T norm(const matrix3<T>& m) noexcept {
    T sum = T(0);
    for (int i=0; i<3; ++i) sum += length_sq(m.col[i]);
    return std::sqrt(sum);
}

// ============================================================
// Scaling, rotation matrices
// ============================================================

template<typename T>
constexpr matrix3<T> scaling(const vector3<T>& s) noexcept {
    return matrix3<T>(s.x,T(0),T(0), T(0),s.y,T(0), T(0),T(0),s.z);
}

template<typename T>
matrix3<T> rotation_x(T angle_rad) noexcept {
    T c=std::cos(angle_rad), s=std::sin(angle_rad);
    return matrix3<T>(T(1),T(0),T(0), T(0),c,-s, T(0),s,c);
}

template<typename T>
matrix3<T> rotation_y(T angle_rad) noexcept {
    T c=std::cos(angle_rad), s=std::sin(angle_rad);
    return matrix3<T>(c,T(0),s, T(0),T(1),T(0), -s,T(0),c);
}

template<typename T>
matrix3<T> rotation_z(T angle_rad) noexcept {
    T c=std::cos(angle_rad), s=std::sin(angle_rad);
    return matrix3<T>(c,-s,T(0), s,c,T(0), T(0),T(0),T(1));
}

template<typename T>
constexpr matrix3<T> rotation_from_axis_angle(const vector3<T>& axis, T angle_rad) noexcept {
    return matrix3<T>(quaternion<T>(axis, angle_rad));
}

// ============================================================
// Cross‑product matrix
// ============================================================

template<typename T>
constexpr matrix3<T> cross_matrix(const vector3<T>& v) noexcept {
    return matrix3<T>(T(0), -v.z, v.y, v.z, T(0), -v.x, -v.y, v.x, T(0));
}

// ============================================================
// Outer product
// ============================================================

template<typename T>
constexpr matrix3<T> outer_product(const vector3<T>& a, const vector3<T>& b) noexcept {
    return matrix3<T>(a.x*b, a.y*b, a.z*b);
}

// ============================================================
// Diagonal matrix from vector
// ============================================================

template<typename T>
constexpr matrix3<T> diagonal(const vector3<T>& d) noexcept {
    return matrix3<T>(d.x,T(0),T(0), T(0),d.y,T(0), T(0),T(0),d.z);
}

// ============================================================
// Component‑wise operations
// ============================================================

template<typename T> constexpr matrix3<T> min(const matrix3<T>& a, const matrix3<T>& b) noexcept {
    return matrix3<T>(min(a.col[0],b.col[0]), min(a.col[1],b.col[1]), min(a.col[2],b.col[2]));
}
template<typename T> constexpr matrix3<T> max(const matrix3<T>& a, const matrix3<T>& b) noexcept {
    return matrix3<T>(max(a.col[0],b.col[0]), max(a.col[1],b.col[1]), max(a.col[2],b.col[2]));
}
template<typename T> constexpr matrix3<T> abs(const matrix3<T>& m) noexcept {
    return matrix3<T>(abs(m.col[0]), abs(m.col[1]), abs(m.col[2]));
}

// ============================================================
// Symmetric eigen decomposition (Jacobi method for 3x3)
// ============================================================

template<typename T>
void symmetric_eigen(const matrix3<T>& m, vector3<T>& eigenvalues, matrix3<T>& eigenvectors) noexcept {
    eigenvectors = matrix3<T>(T(1));
    matrix3<T> a = m;
    std::array<T,3> b = {a(0,0), a(1,1), a(2,2)};
    std::array<T,3> z = {T(0), T(0), T(0)};
    const int max_iter = 50;
    for (int iter=0; iter<max_iter; ++iter) {
        T sm = std::abs(a(0,1)) + std::abs(a(0,2)) + std::abs(a(1,2));
        if (sm < T(FLOAT_EPSILON)) break;
        T thresh = (iter<4) ? T(0.2)*sm/(T(9)) : T(0);
        for (int p=0; p<2; ++p) {
            for (int q=p+1; q<3; ++q) {
                T g = T(100) * std::abs(a(p,q));
                if (iter>4 && std::abs(b[p])+g==std::abs(b[p]) && std::abs(b[q])+g==std::abs(b[q]))
                    a(p,q) = T(0);
                else if (std::abs(a(p,q))>thresh) {
                    T h = b[q] - b[p];
                    T t;
                    if (std::abs(h)+g == std::abs(h))
                        t = a(p,q)/h;
                    else {
                        T theta = T(0.5)*h/a(p,q);
                        t = T(1)/(std::abs(theta)+std::sqrt(T(1)+theta*theta));
                        if (theta<T(0)) t = -t;
                    }
                    T c = T(1)/std::sqrt(T(1)+t*t);
                    T s = t*c;
                    T tau = s/(T(1)+c);
                    h = t*a(p,q);
                    z[p] -= h; z[q] += h;
                    b[p] -= h; b[q] += h;
                    a(p,q) = T(0);
                    for (int j=0; j<p; ++j) { T g_=a(j,p); T h_=a(j,q); a(j,p)=g_-s*(h_+g_*tau); a(j,q)=h_+s*(g_-h_*tau); }
                    for (int j=p+1; j<q; ++j) { T g_=a(p,j); T h_=a(j,q); a(p,j)=g_-s*(h_+g_*tau); a(j,q)=h_+s*(g_-h_*tau); }
                    for (int j=q+1; j<3; ++j) { T g_=a(p,j); T h_=a(q,j); a(p,j)=g_-s*(h_+g_*tau); a(q,j)=h_+s*(g_-h_*tau); }
                    for (int j=0; j<3; ++j) { T g_=eigenvectors(j,p); T h_=eigenvectors(j,q); eigenvectors(j,p)=g_-s*(h_+g_*tau); eigenvectors(j,q)=h_+s*(g_-h_*tau); }
                }
            }
        }
        b[0]+=z[0]; b[1]+=z[1]; b[2]+=z[2];
        z={T(0),T(0),T(0)};
    }
    eigenvalues = vector3<T>(b[0], b[1], b[2]);
}

// ============================================================
// Singular Value Decomposition (SVD) for 3x3 (Jacobi)
// ============================================================

template<typename T>
void svd(const matrix3<T>& m, matrix3<T>& U, vector3<T>& sigma, matrix3<T>& V) noexcept {
    matrix3<T> A = m;
    U = matrix3<T>(T(1));
    V = matrix3<T>(T(1));
    const int max_iter = 50;
    for (int iter=0; iter<max_iter; ++iter) {
        T off = T(0);
        for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) if (i!=j) off+=A(i,j)*A(i,j);
        if (off < T(FLOAT_EPSILON)) break;
        for (int p=0; p<2; ++p) {
            for (int q=p+1; q<3; ++q) {
                T apq = A(p,q);
                if (std::abs(apq) < T(FLOAT_EPSILON)) continue;
                T app = A(p,p), aqq = A(q,q);
                T theta = (aqq - app) / (T(2)*apq);
                T t = T(1)/(std::abs(theta)+std::sqrt(T(1)+theta*theta));
                if (theta<T(0)) t=-t;
                T c=T(1)/std::sqrt(T(1)+t*t), s=t*c;
                T tau = s/(T(1)+c);
                for (int k=0; k<3; ++k) {
                    T akp = A(k,p), akq = A(k,q);
                    A(k,p) = akp - s*(akq + akp*tau);
                    A(k,q) = akq + s*(akp - akq*tau);
                }
                for (int k=0; k<3; ++k) {
                    T apk = A(p,k), aqk = A(q,k);
                    A(p,k) = apk - s*(aqk + apk*tau);
                    A(q,k) = aqk + s*(apk - aqk*tau);
                }
                for (int k=0; k<3; ++k) {
                    T ukp = U(k,p), ukq = U(k,q);
                    U(k,p) = ukp - s*(ukq + ukp*tau);
                    U(k,q) = ukq + s*(ukp - ukq*tau);
                }
                for (int k=0; k<3; ++k) {
                    T vkp = V(k,p), vkq = V(k,q);
                    V(k,p) = vkp - s*(vkq + vkp*tau);
                    V(k,q) = vkq + s*(vkp - vkq*tau);
                }
            }
        }
    }
    sigma = vector3<T>(A(0,0), A(1,1), A(2,2));
    for (int i=0; i<3; ++i) if (sigma[i]<T(0)) { sigma[i]=-sigma[i]; U.col[i]=-U.col[i]; }
}

// ============================================================
// Polar decomposition
// ============================================================

template<typename T>
void polar_decompose(const matrix3<T>& m, matrix3<T>& R, matrix3<T>& S) noexcept {
    const int max_iter = 20;
    R = m;
    for (int iter=0; iter<max_iter; ++iter) {
        matrix3<T> R_inv_t = transpose(inverse(R));
        matrix3<T> next = (R + R_inv_t) * T(0.5);
        if (norm(next - R) < T(FLOAT_EPSILON)) break;
        R = next;
    }
    S = transpose(R) * m;
}

// ============================================================
// Interpolation (element‑wise lerp)
// ============================================================

template<typename T>
constexpr matrix3<T> lerp(const matrix3<T>& a, const matrix3<T>& b, T t) noexcept {
    return a + (b - a) * t;
}

// ============================================================
// Conversion between types
// ============================================================

template<typename U, typename T>
constexpr matrix3<U> matrix_cast(const matrix3<T>& m) noexcept {
    return matrix3<U>(vector_cast<U>(m.col[0]), vector_cast<U>(m.col[1]), vector_cast<U>(m.col[2]));
}

// ============================================================
// Type aliases
// ============================================================

using matrix3f = matrix3<float>;
using matrix3d = matrix3<double>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_MATRIX3_H