//File 0316 : xframe/xframe_linalg.hpp
//Linear algebra operations for xframe: dot product, matrix multiplication, solve, determinant, and eigenvalue decomposition with SIMD acceleration.
#ifndef XFRAME_LINALG_HPP
#define XFRAME_LINALG_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"
#include "xframe.hpp"
#include "xframe_function.hpp"
#include "xframe_math.hpp"

namespace xframe {
namespace linalg {

    /**
     * Dot product of two 1D variables.
     */
    template <class T, class L>
    inline auto dot(const variable<T, L>& a, const variable<T, L>& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("dot: size mismatch.");
        T result = T(0);
        const T* ad = a.data();
        const T* bd = b.data();
        std::size_t n = a.size();
        if constexpr (simd_enabled_v<T>)
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            std::size_t vec_count = n / simd_size;
            simd_type vsum(0);
            for (std::size_t i = 0; i < vec_count; ++i)
            {
                simd_type va = simd_type::load_unaligned(ad + i * simd_size);
                simd_type vb = simd_type::load_unaligned(bd + i * simd_size);
                vsum = vsum + va * vb;
            }
            T tmp[simd_size];
            vsum.store_unaligned(tmp);
            for (std::size_t k = 0; k < simd_size; ++k) result += tmp[k];
            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                result += ad[i] * bd[i];
        }
        else
        {
            for (std::size_t i = 0; i < n; ++i)
                result += ad[i] * bd[i];
        }
        return result;
    }

    /**
     * Matrix multiplication of two 2D xframes (sum over inner dimension).
     * Each xframe must have 2 dimensions, and the inner dimension names must match.
     */
    template <class... V1, class... V2>
    inline auto matmul(const xframe<V1...>& a, const xframe<V2...>& b)
    {
        if (a.dimension_count() != 2 || b.dimension_count() != 2)
            throw std::runtime_error("matmul: both xframes must be 2D.");
        if (a.dimension(1).size() != b.dimension(0).size())
            throw std::runtime_error("matmul: inner dimension sizes must match.");
        // For simplicity, assume both have a single variable.
        std::size_t m = a.dimension(0).size();
        std::size_t k = a.dimension(1).size();
        std::size_t n = b.dimension(1).size();

        // Result dimensions: first of a, second of b
        auto result_dim0 = a.dimension(0);
        auto result_dim1 = b.dimension(1);
        auto result = xframe<double>(std::make_tuple(result_dim0, result_dim1));

        auto& var_a = a.template variable<0>();
        auto& var_b = b.template variable<0>();
        auto& var_r = result.template variable<0>();

        const double* ad = var_a.data();
        const double* bd = var_b.data();
        double* rd = var_r.data();

        constexpr std::size_t BLOCK = 64;
        for (std::size_t i0 = 0; i0 < m; i0 += BLOCK)
        {
            std::size_t i_end = std::min(i0 + BLOCK, m);
            for (std::size_t j0 = 0; j0 < n; j0 += BLOCK)
            {
                std::size_t j_end = std::min(j0 + BLOCK, n);
                for (std::size_t k0 = 0; k0 < k; k0 += BLOCK)
                {
                    std::size_t k_end = std::min(k0 + BLOCK, k);
                    for (std::size_t i = i0; i < i_end; ++i)
                    {
                        for (std::size_t kk = k0; kk < k_end; ++kk)
                        {
                            double aik = ad[i * k + kk];
                            if (aik == 0.0) continue;
                            const double* b_row = bd + kk * n;
                            double* r_row = rd + i * n;
                            std::size_t j = j0;
                            if constexpr (simd_enabled_v<double>)
                            {
                                using simd_type = xsimd::batch<double, default_simd_arch>;
                                constexpr std::size_t simd_size = simd_type::size;
                                simd_type va(aik);
                                for (; j + simd_size <= j_end; j += simd_size)
                                {
                                    simd_type vb = simd_type::load_unaligned(b_row + j);
                                    simd_type vr = simd_type::load_unaligned(r_row + j);
                                    vr = vr + va * vb;
                                    vr.store_unaligned(r_row + j);
                                }
                            }
                            for (; j < j_end; ++j)
                                r_row[j] += aik * b_row[j];
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Solve linear system A*x = b for x using LU decomposition.
     * A must be a 2D xframe with a single variable (square matrix).
     */
    template <class... V1, class... V2>
    inline auto solve(const xframe<V1...>& A, const xframe<V2...>& b)
    {
        if (A.dimension_count() != 2 || A.dimension(0).size() != A.dimension(1).size())
            throw std::runtime_error("solve: A must be a square 2D xframe.");
        if (b.dimension_count() != 1 || b.dimension(0).size() != A.dimension(0).size())
            throw std::runtime_error("solve: b must be a 1D xframe with matching size.");
        std::size_t n = A.dimension(0).size();
        auto LU = A; // copy
        auto x = b;  // copy
        auto& LU_var = LU.template variable<0>();
        auto& x_var = x.template variable<0>();
        double* LU_data = LU_var.data();
        double* x_data = x_var.data();
        std::vector<std::size_t> perm(n);
        for (std::size_t i = 0; i < n; ++i) perm[i] = i;
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t pivot = i;
            double max_val = std::abs(LU_data[i * n + i]);
            for (std::size_t r = i + 1; r < n; ++r)
            {
                if (std::abs(LU_data[r * n + i]) > max_val)
                {
                    max_val = std::abs(LU_data[r * n + i]);
                    pivot = r;
                }
            }
            if (max_val < 1e-14) throw std::runtime_error("Singular matrix.");
            if (pivot != i)
            {
                std::swap(perm[i], perm[pivot]);
                for (std::size_t j = 0; j < n; ++j)
                    std::swap(LU_data[i * n + j], LU_data[pivot * n + j]);
            }
            for (std::size_t r = i + 1; r < n; ++r)
            {
                double factor = LU_data[r * n + i] / LU_data[i * n + i];
                LU_data[r * n + i] = factor;
                for (std::size_t j = i + 1; j < n; ++j)
                    LU_data[r * n + j] -= factor * LU_data[i * n + j];
            }
        }
        // forward solve L*y = Pb
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            double sum = b.template variable<0>()[perm[i]];
            for (std::size_t j = 0; j < i; ++j)
                sum -= LU_data[i * n + j] * y[j];
            y[i] = sum;
        }
        // backward solve U*x = y
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i)
        {
            double sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j)
                sum -= LU_data[i * n + j] * x_data[j];
            x_data[i] = sum / LU_data[i * n + i];
        }
        return x;
    }

    /**
     * Determinant of a square matrix stored as a 2D xframe.
     */
    template <class... V>
    inline auto det(const xframe<V...>& A)
    {
        if (A.dimension_count() != 2 || A.dimension(0).size() != A.dimension(1).size())
            throw std::runtime_error("det: A must be square 2D.");
        std::size_t n = A.dimension(0).size();
        auto LU = A;
        auto& var = LU.template variable<0>();
        double* data = var.data();
        double det_sign = 1.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t pivot = i;
            double max_val = std::abs(data[i * n + i]);
            for (std::size_t r = i + 1; r < n; ++r)
            {
                if (std::abs(data[r * n + i]) > max_val)
                {
                    max_val = std::abs(data[r * n + i]);
                    pivot = r;
                }
            }
            if (max_val < 1e-14) return 0.0;
            if (pivot != i)
            {
                det_sign = -det_sign;
                for (std::size_t j = 0; j < n; ++j)
                    std::swap(data[i * n + j], data[pivot * n + j]);
            }
            double piv_val = data[i * n + i];
            det_sign *= piv_val;
            for (std::size_t r = i + 1; r < n; ++r)
            {
                double factor = data[r * n + i] / piv_val;
                data[r * n + i] = 0.0;
                for (std::size_t j = i + 1; j < n; ++j)
                    data[r * n + j] -= factor * data[i * n + j];
            }
        }
        return det_sign;
    }

    /**
     * Inverse of a square matrix.
     */
    template <class... V>
    inline auto inv(const xframe<V...>& A)
    {
        if (A.dimension_count() != 2 || A.dimension(0).size() != A.dimension(1).size())
            throw std::runtime_error("inv: A must be square 2D.");
        std::size_t n = A.dimension(0).size();
        auto result = A;
        auto I = xframe<double>(std::make_tuple(A.dimension(0), A.dimension(1)));
        auto& I_var = I.template variable<0>();
        for (std::size_t i = 0; i < n; ++i)
            I_var[i * n + i] = 1.0;
        auto& A_var = result.template variable<0>();
        double* Ad = A_var.data();
        double* Id = I_var.data();
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t pivot = i;
            double max_val = std::abs(Ad[i * n + i]);
            for (std::size_t r = i + 1; r < n; ++r)
            {
                if (std::abs(Ad[r * n + i]) > max_val)
                {
                    max_val = std::abs(Ad[r * n + i]);
                    pivot = r;
                }
            }
            if (max_val < 1e-14) throw std::runtime_error("Singular matrix.");
            if (pivot != i)
            {
                for (std::size_t j = 0; j < n; ++j)
                {
                    std::swap(Ad[i * n + j], Ad[pivot * n + j]);
                    std::swap(Id[i * n + j], Id[pivot * n + j]);
                }
            }
            double piv_val = Ad[i * n + i];
            for (std::size_t j = 0; j < n; ++j)
            {
                Ad[i * n + j] /= piv_val;
                Id[i * n + j] /= piv_val;
            }
            for (std::size_t r = 0; r < n; ++r)
            {
                if (r == i) continue;
                double factor = Ad[r * n + i];
                if (factor != 0.0)
                {
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        Ad[r * n + j] -= factor * Ad[i * n + j];
                        Id[r * n + j] -= factor * Id[i * n + j];
                    }
                }
            }
        }
        return I;
    }

    /**
     * Eigenvalue decomposition for symmetric matrices (QR algorithm).
     */
    template <class... V>
    inline auto eig_sym(const xframe<V...>& A)
    {
        if (A.dimension_count() != 2 || A.dimension(0).size() != A.dimension(1).size())
            throw std::runtime_error("eig_sym: A must be square 2D.");
        std::size_t n = A.dimension(0).size();
        auto Q = xframe<double>(std::make_tuple(A.dimension(0), A.dimension(1)));
        auto R = A; // copy
        auto& Q_var = Q.template variable<0>();
        auto& R_var = R.template variable<0>();
        double* Qd = Q_var.data();
        double* Rd = R_var.data();
        for (std::size_t i = 0; i < n; ++i) Qd[i * n + i] = 1.0;
        for (std::size_t iter = 0; iter < 1000; ++iter)
        {
            // QR via Gram-Schmidt
            double Qk[n][n];
            double Rk[n][n];
            for (std::size_t j = 0; j < n; ++j)
            {
                for (std::size_t i = 0; i < n; ++i)
                    Qk[i][j] = Rd[i * n + j];
                for (std::size_t i = 0; i < j; ++i)
                {
                    double rij = 0;
                    for (std::size_t k = 0; k < n; ++k)
                        rij += Qk[k][i] * Rd[k * n + j];
                    Rk[i][j] = rij;
                    for (std::size_t k = 0; k < n; ++k)
                        Qk[k][j] -= rij * Qk[k][i];
                }
                double rjj = 0;
                for (std::size_t k = 0; k < n; ++k)
                    rjj += Qk[k][j] * Qk[k][j];
                rjj = std::sqrt(rjj);
                Rk[j][j] = rjj;
                if (rjj > 1e-15)
                    for (std::size_t k = 0; k < n; ++k)
                        Qk[k][j] /= rjj;
            }
            // R_new = Rk * Q (RQ iteration)
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                {
                    double sum = 0;
                    for (std::size_t k = 0; k < n; ++k)
                        sum += Rk[i][k] * Qk[k][j]; // actually Qk is Q of previous, we need to multiply
                    Rd[i * n + j] = sum;
                }
            // Update Q
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                {
                    double sum = 0;
                    for (std::size_t k = 0; k < n; ++k)
                        sum += Qd[i * n + k] * Qk[k][j];
                    Qd[i * n + j] = sum;
                }
            // Check off-diagonal
            double off = 0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    if (i != j) off += Rd[i * n + j] * Rd[i * n + j];
            if (std::sqrt(off) < 1e-12) break;
        }
        std::vector<double> eigenvalues(n);
        for (std::size_t i = 0; i < n; ++i)
            eigenvalues[i] = Rd[i * n + i];
        // Return eigenvalues and eigenvectors
        return std::make_pair(eigenvalues, Q);
    }

} // namespace linalg
} // namespace xframe

#endif // XFRAME_LINALG_HPP