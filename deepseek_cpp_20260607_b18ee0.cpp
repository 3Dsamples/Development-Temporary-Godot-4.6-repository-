/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once
#ifndef ORTHOTREE_CORE_MATH_TRANSFORM_H_INCLUDED
#define ORTHOTREE_CORE_MATH_TRANSFORM_H_INCLUDED

#include "vector_math.h"
#include "quaternion.h"
#include <array>
#include <cstddef>

namespace OrthoTree::Math {

template<typename T, std::size_t N>
class Matrix {
public:
    using value_type = T;
    using size_type = std::size_t;
    
    static constexpr size_type rows() noexcept { return N; }
    static constexpr size_type cols() noexcept { return N; }
    
    constexpr Matrix() noexcept : m_data{} {
        for (size_type i = 0; i < N; ++i) m_data[i][i] = T{1};
    }
    
    constexpr Matrix(std::initializer_list<std::initializer_list<T>> init) noexcept {
        size_type i = 0;
        for (auto rowIt = init.begin(); rowIt != init.end() && i < N; ++rowIt, ++i) {
            size_type j = 0;
            for (auto colIt = rowIt->begin(); colIt != rowIt->end() && j < N; ++colIt, ++j) {
                m_data[i][j] = *colIt;
            }
            for (; j < N; ++j) m_data[i][j] = T{0};
        }
        for (; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) m_data[i][j] = T{0};
        }
    }
    
    constexpr T& operator()(size_type row, size_type col) noexcept {
        return m_data[row][col];
    }
    
    constexpr const T& operator()(size_type row, size_type col) const noexcept {
        return m_data[row][col];
    }
    
    constexpr T* data() noexcept { return &m_data[0][0]; }
    constexpr const T* data() const noexcept { return &m_data[0][0]; }
    
    constexpr Matrix operator+(const Matrix& other) const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                result(i, j) = m_data[i][j] + other(i, j);
            }
        }
        return result;
    }
    
    constexpr Matrix operator-(const Matrix& other) const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                result(i, j) = m_data[i][j] - other(i, j);
            }
        }
        return result;
    }
    
    constexpr Matrix operator*(T scalar) const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                result(i, j) = m_data[i][j] * scalar;
            }
        }
        return result;
    }
    
    constexpr Matrix operator*(const Matrix& other) const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                T sum = T{0};
                for (size_type k = 0; k < N; ++k) {
                    sum += m_data[i][k] * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    constexpr Vector<T, N> operator*(const Vector<T, N>& vec) const noexcept {
        Vector<T, N> result;
        for (size_type i = 0; i < N; ++i) {
            T sum = T{0};
            for (size_type j = 0; j < N; ++j) {
                sum += m_data[i][j] * vec[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    constexpr Matrix& operator+=(const Matrix& other) noexcept {
        *this = *this + other;
        return *this;
    }
    
    constexpr Matrix& operator-=(const Matrix& other) noexcept {
        *this = *this - other;
        return *this;
    }
    
    constexpr Matrix& operator*=(T scalar) noexcept {
        *this = *this * scalar;
        return *this;
    }
    
    constexpr Matrix& operator*=(const Matrix& other) noexcept {
        *this = *this * other;
        return *this;
    }
    
    constexpr Matrix transpose() const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                result(i, j) = m_data[j][i];
            }
        }
        return result;
    }
    
    constexpr T determinant() const noexcept {
        static_assert(N <= 4, "Determinant only implemented for N <= 4");
        if constexpr (N == 1) return m_data[0][0];
        if constexpr (N == 2) return m_data[0][0] * m_data[1][1] - m_data[0][1] * m_data[1][0];
        if constexpr (N == 3) {
            return m_data[0][0] * (m_data[1][1] * m_data[2][2] - m_data[1][2] * m_data[2][1])
                 - m_data[0][1] * (m_data[1][0] * m_data[2][2] - m_data[1][2] * m_data[2][0])
                 + m_data[0][2] * (m_data[1][0] * m_data[2][1] - m_data[1][1] * m_data[2][0]);
        }
        if constexpr (N == 4) {
            T det = T{0};
            for (size_type i = 0; i < 4; ++i) {
                Matrix<T, 3> sub;
                size_type subRow = 0;
                for (size_type r = 1; r < 4; ++r) {
                    size_type subCol = 0;
                    for (size_type c = 0; c < 4; ++c) {
                        if (c != i) {
                            sub(subRow, subCol) = m_data[r][c];
                            ++subCol;
                        }
                    }
                    ++subRow;
                }
                T sign = (i % 2 == 0) ? T{1} : T{-1};
                det += sign * m_data[0][i] * sub.determinant();
            }
            return det;
        }
        return T{0};
    }
    
    constexpr Matrix adjugate() const noexcept {
        static_assert(N <= 4, "Adjugate only implemented for N <= 4");
        if constexpr (N == 1) return Matrix({ { T{1} } });
        if constexpr (N == 2) {
            return Matrix({
                {  m_data[1][1], -m_data[0][1] },
                { -m_data[1][0],  m_data[0][0] }
            });
        }
        Matrix result;
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                Matrix<T, N - 1> sub;
                size_type subRow = 0;
                for (size_type r = 0; r < N; ++r) {
                    if (r == i) continue;
                    size_type subCol = 0;
                    for (size_type c = 0; c < N; ++c) {
                        if (c == j) continue;
                        sub(subRow, subCol) = m_data[r][c];
                        ++subCol;
                    }
                    ++subRow;
                }
                T sign = ((i + j) % 2 == 0) ? T{1} : T{-1};
                result(j, i) = sign * sub.determinant();
            }
        }
        return result;
    }
    
    constexpr Matrix inverse() const noexcept {
        T det = determinant();
        if (det == T{0}) return Matrix();
        return adjugate() * (T{1} / det);
    }
    
    static constexpr Matrix identity() noexcept {
        return Matrix();
    }
    
    static constexpr Matrix translation(const Vector<T, N - 1>& offset) noexcept {
        static_assert(N >= 2, "Translation requires N >= 2");
        Matrix result;
        for (size_type i = 0; i < N - 1; ++i) {
            result(i, N - 1) = offset[i];
        }
        return result;
    }
    
    static Matrix rotation(const Vector<T, 3>& axis, T angle) noexcept {
        static_assert(N == 4, "Rotation requires 4x4 matrix");
        Quaternion<T> q = Quaternion<T>::fromAxisAngle(axis, angle);
        Matrix<T, 4> result;
        result(0, 0) = T{1} - T{2} * (q.y() * q.y() + q.z() * q.z());
        result(0, 1) = T{2} * (q.x() * q.y() - q.z() * q.w());
        result(0, 2) = T{2} * (q.x() * q.z() + q.y() * q.w());
        result(1, 0) = T{2} * (q.x() * q.y() + q.z() * q.w());
        result(1, 1) = T{1} - T{2} * (q.x() * q.x() + q.z() * q.z());
        result(1, 2) = T{2} * (q.y() * q.z() - q.x() * q.w());
        result(2, 0) = T{2} * (q.x() * q.z() - q.y() * q.w());
        result(2, 1) = T{2} * (q.y() * q.z() + q.x() * q.w());
        result(2, 2) = T{1} - T{2} * (q.x() * q.x() + q.y() * q.y());
        return result;
    }
    
    static Matrix scaling(const Vector<T, N - 1>& scales) noexcept {
        static_assert(N >= 2, "Scaling requires N >= 2");
        Matrix result;
        for (size_type i = 0; i < N - 1; ++i) {
            result(i, i) = scales[i];
        }
        return result;
    }
    
private:
    std::array<std::array<T, N>, N> m_data;
};

template<typename T, std::size_t N>
class AffineTransform {
public:
    using value_type = T;
    using size_type = std::size_t;
    
    constexpr AffineTransform() noexcept : m_matrix(), m_translation() {}
    constexpr AffineTransform(const Matrix<T, N>& matrix, const Vector<T, N>& translation) noexcept
        : m_matrix(matrix), m_translation(translation) {}
    
    constexpr const Matrix<T, N>& matrix() const noexcept { return m_matrix; }
    constexpr const Vector<T, N>& translation() const noexcept { return m_translation; }
    constexpr void setMatrix(const Matrix<T, N>& matrix) noexcept { m_matrix = matrix; }
    constexpr void setTranslation(const Vector<T, N>& translation) noexcept { m_translation = translation; }
    
    constexpr Vector<T, N> transform(const Vector<T, N>& point) const noexcept {
        return m_matrix * point + m_translation;
    }
    
    constexpr Vector<T, N> transformDirection(const Vector<T, N>& direction) const noexcept {
        return m_matrix * direction;
    }
    
    constexpr Vector<T, N> transformNormal(const Vector<T, N>& normal) const noexcept {
        return m_matrix.transpose().inverse() * normal;
    }
    
    constexpr AffineTransform inverse() const noexcept {
        Matrix<T, N> invMatrix = m_matrix.inverse();
        return AffineTransform(invMatrix, invMatrix * (-m_translation));
    }
    
    constexpr AffineTransform operator*(const AffineTransform& other) const noexcept {
        return AffineTransform(m_matrix * other.m_matrix, m_matrix * other.m_translation + m_translation);
    }
    
    constexpr AffineTransform& operator*=(const AffineTransform& other) noexcept {
        *this = *this * other;
        return *this;
    }
    
    static constexpr AffineTransform identity() noexcept {
        return AffineTransform();
    }
    
    static AffineTransform translation(const Vector<T, N>& offset) noexcept {
        return AffineTransform(Matrix<T, N>::identity(), offset);
    }
    
    static AffineTransform rotation(const Quaternion<T>& q) noexcept {
        if constexpr (N == 3) {
            Matrix<T, 3> rot;
            rot(0, 0) = T{1} - T{2} * (q.y() * q.y() + q.z() * q.z());
            rot(0, 1) = T{2} * (q.x() * q.y() - q.z() * q.w());
            rot(0, 2) = T{2} * (q.x() * q.z() + q.y() * q.w());
            rot(1, 0) = T{2} * (q.x() * q.y() + q.z() * q.w());
            rot(1, 1) = T{1} - T{2} * (q.x() * q.x() + q.z() * q.z());
            rot(1, 2) = T{2} * (q.y() * q.z() - q.x() * q.w());
            rot(2, 0) = T{2} * (q.x() * q.z() - q.y() * q.w());
            rot(2, 1) = T{2} * (q.y() * q.z() + q.x() * q.w());
            rot(2, 2) = T{1} - T{2} * (q.x() * q.x() + q.y() * q.y());
            return AffineTransform(rot, Vector<T, 3>());
        }
        return AffineTransform::identity();
    }
    
    static AffineTransform scaling(const Vector<T, N>& scales) noexcept {
        Matrix<T, N> mat;
        for (size_type i = 0; i < N; ++i) mat(i, i) = scales[i];
        return AffineTransform(mat, Vector<T, N>());
    }
    
private:
    Matrix<T, N> m_matrix;
    Vector<T, N> m_translation;
};

template<typename T>
using Transform2 = AffineTransform<T, 2>;
template<typename T>
using Transform3 = AffineTransform<T, 3>;

using Transform2f = AffineTransform<float, 2>;
using Transform3f = AffineTransform<float, 3>;
using Transform2d = AffineTransform<double, 2>;
using Transform3d = AffineTransform<double, 3>;

using Matrix3f = Matrix<float, 3>;
using Matrix4f = Matrix<float, 4>;
using Matrix3d = Matrix<double, 3>;
using Matrix4d = Matrix<double, 4>;

} // namespace OrthoTree::Math

#endif // ORTHOTREE_CORE_MATH_TRANSFORM_H_INCLUDED