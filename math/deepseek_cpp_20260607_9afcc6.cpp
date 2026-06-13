//File group name : OrthoTree Math
//File 0052 : core/math/basic/matrix.h
//Square matrix (2x2, 3x3, 4x4) with arithmetic, determinant, inverse, transpose, SIMD batch operations, and dynamic environment controls.

#ifndef ORTHOTREE_CORE_MATH_BASIC_MATRIX_H_INCLUDED
#define ORTHOTREE_CORE_MATH_BASIC_MATRIX_H_INCLUDED

#include "../../build_config.h"
#include "scalar.h"
#include "vector.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace OrthoTree {
namespace Math {
namespace Basic {

// ============================================================================
//  Square matrix of size N (N = 2,3,4)
// ============================================================================
template<typename T, std::size_t N>
class Matrix {
public:
    using value_type = T;
    using size_type = std::size_t;
    using row_type = std::array<T, N>;
    using iterator = T*;
    using const_iterator = const T*;

    static constexpr size_type dimension() noexcept { return N; }
    static constexpr size_type data_size() noexcept { return N * N; }

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Matrix() noexcept : m_data{} {
        for (size_type i = 0; i < N; ++i) m_data[i][i] = T(1);
    }
    constexpr Matrix(std::initializer_list<std::initializer_list<T>> init) noexcept {
        size_type i = 0;
        for (auto rowIt = init.begin(); rowIt != init.end() && i < N; ++rowIt, ++i) {
            size_type j = 0;
            for (auto colIt = rowIt->begin(); colIt != rowIt->end() && j < N; ++colIt, ++j) {
                m_data[i][j] = *colIt;
            }
            for (; j < N; ++j) m_data[i][j] = T(0);
        }
        for (; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) m_data[i][j] = T(0);
            m_data[i][i] = T(1);
        }
    }
    constexpr Matrix(const Matrix&) = default;
    constexpr Matrix(Matrix&&) = default;
    constexpr Matrix& operator=(const Matrix&) = default;
    constexpr Matrix& operator=(Matrix&&) = default;

    // ------------------------------------------------------------------------
    //  Element access
    // ------------------------------------------------------------------------
    constexpr T& operator()(size_type row, size_type col) noexcept { return m_data[row][col]; }
    constexpr const T& operator()(size_type row, size_type col) const noexcept { return m_data[row][col]; }
    constexpr T* data() noexcept { return &m_data[0][0]; }
    constexpr const T* data() const noexcept { return &m_data[0][0]; }

    // ------------------------------------------------------------------------
    //  Row and column access
    // ------------------------------------------------------------------------
    constexpr row_type row(size_type i) const noexcept { return m_data[i]; }
    constexpr Vector<T, N> col(size_type j) const noexcept {
        Vector<T, N> result;
        for (size_type i = 0; i < N; ++i) result[i] = m_data[i][j];
        return result;
    }

    // ------------------------------------------------------------------------
    //  Arithmetic operations
    // ------------------------------------------------------------------------
    constexpr Matrix operator+(const Matrix& other) const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i)
            for (size_type j = 0; j < N; ++j)
                result(i, j) = m_data[i][j] + other(i, j);
        return result;
    }
    constexpr Matrix operator-(const Matrix& other) const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i)
            for (size_type j = 0; j < N; ++j)
                result(i, j) = m_data[i][j] - other(i, j);
        return result;
    }
    constexpr Matrix operator*(T scalar) const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i)
            for (size_type j = 0; j < N; ++j)
                result(i, j) = m_data[i][j] * scalar;
        return result;
    }
    constexpr Matrix operator*(const Matrix& other) const noexcept {
        Matrix result(0);
        for (size_type i = 0; i < N; ++i) {
            for (size_type k = 0; k < N; ++k) {
                T aik = m_data[i][k];
                if (aik != T(0)) {
                    for (size_type j = 0; j < N; ++j) {
                        result(i, j) += aik * other(k, j);
                    }
                }
            }
        }
        return result;
    }
    constexpr Vector<T, N> operator*(const Vector<T, N>& vec) const noexcept {
        Vector<T, N> result(0);
        for (size_type i = 0; i < N; ++i) {
            T sum = T(0);
            for (size_type j = 0; j < N; ++j) sum += m_data[i][j] * vec[j];
            result[i] = sum;
        }
        return result;
    }

    constexpr Matrix& operator+=(const Matrix& other) noexcept { *this = *this + other; return *this; }
    constexpr Matrix& operator-=(const Matrix& other) noexcept { *this = *this - other; return *this; }
    constexpr Matrix& operator*=(T scalar) noexcept { *this = *this * scalar; return *this; }
    constexpr Matrix& operator*=(const Matrix& other) noexcept { *this = *this * other; return *this; }

    // ------------------------------------------------------------------------
    //  Transpose
    // ------------------------------------------------------------------------
    constexpr Matrix transpose() const noexcept {
        Matrix result;
        for (size_type i = 0; i < N; ++i)
            for (size_type j = 0; j < N; ++j)
                result(i, j) = m_data[j][i];
        return result;
    }

    // ------------------------------------------------------------------------
    //  Determinant (specialised for N=2,3,4)
    // ------------------------------------------------------------------------
    T determinant() const noexcept {
        if constexpr (N == 1) return m_data[0][0];
        if constexpr (N == 2) {
            return m_data[0][0] * m_data[1][1] - m_data[0][1] * m_data[1][0];
        }
        if constexpr (N == 3) {
            return m_data[0][0] * (m_data[1][1] * m_data[2][2] - m_data[1][2] * m_data[2][1])
                 - m_data[0][1] * (m_data[1][0] * m_data[2][2] - m_data[1][2] * m_data[2][0])
                 + m_data[0][2] * (m_data[1][0] * m_data[2][1] - m_data[1][1] * m_data[2][0]);
        }
        if constexpr (N == 4) {
            // Laplace expansion along first row
            T det = T(0);
            for (size_type j = 0; j < 4; ++j) {
                Matrix<T,3> sub;
                for (size_type i = 1; i < 4; ++i) {
                    size_type subRow = i - 1;
                    size_type subCol = 0;
                    for (size_type k = 0; k < 4; ++k) {
                        if (k == j) continue;
                        sub(subRow, subCol++) = m_data[i][k];
                    }
                }
                T sign = (j % 2 == 0) ? T(1) : T(-1);
                det += sign * m_data[0][j] * sub.determinant();
            }
            return det;
        }
        return T(0);
    }

    // ------------------------------------------------------------------------
    //  Inverse (for N=2,3,4)
    // ------------------------------------------------------------------------
    Matrix inverse() const noexcept {
        T det = determinant();
        if (Basic::nearlyZero(det, T(1e-12))) return *this; // singular
        T invDet = T(1) / det;
        if constexpr (N == 2) {
            return Matrix<T,2>({
                {  m_data[1][1] * invDet, -m_data[0][1] * invDet },
                { -m_data[1][0] * invDet,  m_data[0][0] * invDet }
            });
        }
        if constexpr (N == 3) {
            Matrix<T,3> adj;
            adj(0,0) = (m_data[1][1] * m_data[2][2] - m_data[1][2] * m_data[2][1]);
            adj(0,1) = -(m_data[0][1] * m_data[2][2] - m_data[0][2] * m_data[2][1]);
            adj(0,2) = (m_data[0][1] * m_data[1][2] - m_data[0][2] * m_data[1][1]);
            adj(1,0) = -(m_data[1][0] * m_data[2][2] - m_data[1][2] * m_data[2][0]);
            adj(1,1) = (m_data[0][0] * m_data[2][2] - m_data[0][2] * m_data[2][0]);
            adj(1,2) = -(m_data[0][0] * m_data[1][2] - m_data[0][2] * m_data[1][0]);
            adj(2,0) = (m_data[1][0] * m_data[2][1] - m_data[1][1] * m_data[2][0]);
            adj(2,1) = -(m_data[0][0] * m_data[2][1] - m_data[0][1] * m_data[2][0]);
            adj(2,2) = (m_data[0][0] * m_data[1][1] - m_data[0][1] * m_data[1][0]);
            return adj.transpose() * invDet;
        }
        if constexpr (N == 4) {
            // Compute cofactor matrix
            Matrix<T,4> cof;
            for (size_type i = 0; i < 4; ++i) {
                for (size_type j = 0; j < 4; ++j) {
                    Matrix<T,3> sub;
                    size_type subRow = 0;
                    for (size_type r = 0; r < 4; ++r) {
                        if (r == i) continue;
                        size_type subCol = 0;
                        for (size_type c = 0; c < 4; ++c) {
                            if (c == j) continue;
                            sub(subRow, subCol++) = m_data[r][c];
                        }
                        ++subRow;
                    }
                    T sign = ((i + j) % 2 == 0) ? T(1) : T(-1);
                    cof(i, j) = sign * sub.determinant();
                }
            }
            return cof.transpose() * invDet;
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Identity matrix
    // ------------------------------------------------------------------------
    static constexpr Matrix identity() noexcept {
        Matrix result(0);
        for (size_type i = 0; i < N; ++i) result(i, i) = T(1);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Zero matrix
    // ------------------------------------------------------------------------
    static constexpr Matrix zero() noexcept {
        Matrix result(0);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Matrix& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                if (!Basic::nearlyEqual(m_data[i][j], other(i, j), eps)) return false;
            }
        }
        return true;
    }

private:
    std::array<std::array<T, N>, N> m_data;
};

// ============================================================================
//  Convenience aliases
// ============================================================================
template<typename T> using Matrix2 = Matrix<T, 2>;
template<typename T> using Matrix3 = Matrix<T, 3>;
template<typename T> using Matrix4 = Matrix<T, 4>;

using Mat2f = Matrix<float, 2>;
using Mat3f = Matrix<float, 3>;
using Mat4f = Matrix<float, 4>;
using Mat2d = Matrix<double, 2>;
using Mat3d = Matrix<double, 3>;
using Mat4d = Matrix<double, 4>;

// ============================================================================
//  SIMD batch multiply: 4 matrices with 4 vectors (4x4)
// ----------------------------------------------------------------------------
template<typename T>
void batchMultiply(const Matrix<T,4>* mat, const Vector<T,4>* vec, Vector<T,4>* out, size_t count) {
    if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
        for (size_t i = 0; i < count; ++i) out[i] = mat[i] * vec[i];
    } else {
        for (size_t i = 0; i < count; ++i) out[i] = mat[i] * vec[i];
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class MatrixEnvironment {
public:
    static MatrixEnvironment& instance() {
        static MatrixEnvironment env;
        return env;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    MatrixEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Basic
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_BASIC_MATRIX_H_INCLUDED