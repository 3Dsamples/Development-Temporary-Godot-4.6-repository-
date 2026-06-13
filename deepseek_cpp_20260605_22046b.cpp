//File 0110 : numdot/linalg.h
//Linear algebra operations: dot, matmul, outer, trace, determinant, inverse, linear solve, eigenvalues with SIMD blocking and partial pivoting.
#ifndef NUMDOT_LINALG_H
#define NUMDOT_LINALG_H

#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <numeric>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"
#include "array.h"
#include "elementwise.h"
#include "math.h"
#include "reductions.h"

namespace numdot
{
namespace linalg
{

    /**
     * Dot product of two 1D vectors.
     */
    template <class E1, class E2>
    inline auto dot(const expression<E1>& a, const expression<E2>& b)
    {
        using T = common_value_type_t<E1, E2>;
        const auto& lhs = a.derived();
        const auto& rhs = b.derived();
        if (lhs.shape().size() != 1 || rhs.shape().size() != 1 || lhs.size() != rhs.size())
            throw std::runtime_error("dot: both arguments must be 1D vectors of same length.");
        T result = T(0);
        const T* adata = lhs.data();
        const T* bdata = rhs.data();
        std::size_t n = lhs.size();
        if constexpr (simd_enabled_v<T>)
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            std::size_t vec_count = n / simd_size;
            simd_type vsum(0);
            for (std::size_t i = 0; i < vec_count; ++i)
            {
                simd_type va = simd_type::load_unaligned(adata + i * simd_size);
                simd_type vb = simd_type::load_unaligned(bdata + i * simd_size);
                vsum = vsum + va * vb;
            }
            T tmp[simd_size];
            vsum.store_unaligned(tmp);
            for (std::size_t k = 0; k < simd_size; ++k) result += tmp[k];
            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                result += adata[i] * bdata[i];
        }
        else
        {
            for (std::size_t i = 0; i < n; ++i) result += adata[i] * bdata[i];
        }
        return result;
    }

    /**
     * Outer product of two 1D vectors.
     */
    template <class E1, class E2>
    inline auto outer(const expression<E1>& a, const expression<E2>& b)
    {
        using T = common_value_type_t<E1, E2>;
        const auto& lhs = a.derived();
        const auto& rhs = b.derived();
        auto m = lhs.size();
        auto n = rhs.size();
        array<T> result({m, n});
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                result(i, j) = lhs[i] * rhs[j];
        return result;
    }

    /**
     * Trace of a 2D square matrix.
     */
    template <class E>
    inline auto trace(const expression<E>& mat)
    {
        using T = typename E::value_type;
        const auto& m = mat.derived();
        auto sh = m.shape();
        if (sh.size() != 2) throw std::runtime_error("trace: input must be 2D.");
        std::size_t n = std::min(sh[0], sh[1]);
        T tr = T(0);
        for (std::size_t i = 0; i < n; ++i) tr += m(i, i);
        return tr;
    }

    /**
     * Matrix multiplication with SIMD blocking (C = A * B).
     */
    template <class E1, class E2>
    inline auto matmul(const expression<E1>& a, const expression<E2>& b)
    {
        using T = common_value_type_t<E1, E2>;
        const auto& lhs = a.derived();
        const auto& rhs = b.derived();
        auto sh_a = lhs.shape();
        auto sh_b = rhs.shape();
        if (sh_a.size() != 2 || sh_b.size() != 2)
            throw std::runtime_error("matmul: arguments must be 2D.");
        std::size_t m = sh_a[0], k = sh_a[1], n = sh_b[1];
        if (sh_b[0] != k) throw std::runtime_error("matmul: inner dimension mismatch.");
        array<T> result({m, n}, T(0));
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
                            T aik = lhs(i, kk);
                            if (aik == T(0)) continue;
                            const T* b_row = &rhs(kk, 0);
                            T* res_row = &result(i, 0);
                            std::size_t j = j0;
                            if constexpr (simd_enabled_v<T>)
                            {
                                using simd_type = xsimd::batch<T, default_simd_arch>;
                                constexpr std::size_t simd_size = simd_type::size;
                                simd_type vaik(aik);
                                for (; j + simd_size <= j_end; j += simd_size)
                                {
                                    simd_type vres = simd_type::load_unaligned(res_row + j);
                                    simd_type vb = simd_type::load_unaligned(b_row + j);
                                    vres = vres + vaik * vb;
                                    vres.store_unaligned(res_row + j);
                                }
                            }
                            for (; j < j_end; ++j)
                                res_row[j] += aik * b_row[j];
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Determinant of a square matrix via LU decomposition.
     */
    template <class E>
    inline auto det(const expression<E>& mat)
    {
        using T = typename E::value_type;
        auto sh = mat.derived().shape();
        if (sh.size() != 2 || sh[0] != sh[1]) throw std::runtime_error("det: square matrix required.");
        std::size_t n = sh[0];
        array<T> LU = mat.derived(); // copy
        T det_sign = T(1);
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t pivot = i;
            T max_val = std::abs(LU(i, i));
            for (std::size_t r = i + 1; r < n; ++r)
            {
                if (std::abs(LU(r, i)) > max_val)
                {
                    max_val = std::abs(LU(r, i));
                    pivot = r;
                }
            }
            if (max_val < 1e-14) return T(0);
            if (pivot != i)
            {
                det_sign = -det_sign;
                for (std::size_t j = 0; j < n; ++j) std::swap(LU(i, j), LU(pivot, j));
            }
            T piv_val = LU(i, i);
            det_sign *= piv_val;
            for (std::size_t r = i + 1; r < n; ++r)
            {
                T factor = LU(r, i) / piv_val;
                LU(r, i) = 0;
                for (std::size_t c = i + 1; c < n; ++c) LU(r, c) -= factor * LU(i, c);
            }
        }
        return det_sign;
    }

    /**
     * Inverse of a square matrix using Gauss-Jordan elimination.
     */
    template <class E>
    inline auto inv(const expression<E>& mat)
    {
        using T = typename E::value_type;
        auto sh = mat.derived().shape();
        if (sh.size() != 2 || sh[0] != sh[1]) throw std::runtime_error("inv: square matrix required.");
        std::size_t n = sh[0];
        array<T> A = mat.derived();
        array<T> I({n, n}, T(0));
        for (std::size_t i = 0; i < n; ++i) I(i, i) = T(1);
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t pivot = i;
            T max_val = std::abs(A(i, i));
            for (std::size_t r = i + 1; r < n; ++r)
            {
                if (std::abs(A(r, i)) > max_val) { max_val = std::abs(A(r, i)); pivot = r; }
            }
            if (max_val < 1e-14) throw std::runtime_error("inv: singular matrix.");
            if (pivot != i)
            {
                for (std::size_t j = 0; j < n; ++j) { std::swap(A(i, j), A(pivot, j)); std::swap(I(i, j), I(pivot, j)); }
            }
            T piv_val = A(i, i);
            for (std::size_t j = 0; j < n; ++j) { A(i, j) /= piv_val; I(i, j) /= piv_val; }
            for (std::size_t r = 0; r < n; ++r)
            {
                if (r == i) continue;
                T factor = A(r, i);
                if (factor != T(0))
                    for (std::size_t j = 0; j < n; ++j) { A(r, j) -= factor * A(i, j); I(r, j) -= factor * I(i, j); }
            }
        }
        return I;
    }

    /**
     * Solve linear system Ax = b using LU decomposition with partial pivoting.
     */
    template <class E1, class E2>
    inline auto solve(const expression<E1>& A, const expression<E2>& b)
    {
        using T = typename E1::value_type;
        auto shA = A.derived().shape(), shb = b.derived().shape();
        if (shA.size() != 2 || shb.size() != 1 || shA[0] != shb[0])
            throw std::runtime_error("solve: A must be square, b 1D with matching size.");
        std::size_t n = shA[0];
        array<T> LU = A.derived();
        array<T> x = b.derived();
        std::vector<std::size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t pivot = i;
            T max_val = std::abs(LU(i, i));
            for (std::size_t r = i + 1; r < n; ++r)
            {
                if (std::abs(LU(r, i)) > max_val) { max_val = std::abs(LU(r, i)); pivot = r; }
            }
            if (max_val < 1e-14) throw std::runtime_error("solve: singular matrix.");
            if (pivot != i)
            {
                std::swap(perm[i], perm[pivot]);
                for (std::size_t j = 0; j < n; ++j) std::swap(LU(i, j), LU(pivot, j));
            }
            for (std::size_t r = i + 1; r < n; ++r)
            {
                T factor = LU(r, i) / LU(i, i);
                LU(r, i) = factor;
                for (std::size_t j = i + 1; j < n; ++j) LU(r, j) -= factor * LU(i, j);
            }
        }
        array<T> y({n});
        for (std::size_t i = 0; i < n; ++i)
        {
            T sum = b.derived()[perm[i]];
            for (std::size_t j = 0; j < i; ++j) sum -= LU(i, j) * y[j];
            y[i] = sum;
        }
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i)
        {
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j) sum -= LU(i, j) * x[j];
            x[i] = sum / LU(i, i);
        }
        return x;
    }

    /**
     * Power iteration to compute the dominant eigenvalue and eigenvector.
     */
    template <class E>
    inline auto eig_power(const expression<E>& A, std::size_t max_iter = 1000, double tol = 1e-12)
    {
        using T = typename E::value_type;
        auto sh = A.derived().shape();
        if (sh.size() != 2 || sh[0] != sh[1]) throw std::runtime_error("eig_power: square matrix required.");
        std::size_t n = sh[0];
        array<T> v({n});
        for (std::size_t i = 0; i < n; ++i) v[i] = T(1) / std::sqrt(T(n));
        T lambda = T(0);
        for (std::size_t iter = 0; iter < max_iter; ++iter)
        {
            auto Av = matmul(A, v);
            T lambda_new = dot(Av, v);
            if (std::abs(lambda_new - lambda) < tol) return std::make_pair(lambda_new, v);
            lambda = lambda_new;
            T norm = std::sqrt(dot(Av, Av));
            if (norm > 0) for (std::size_t i = 0; i < n; ++i) v[i] = Av[i] / norm;
        }
        return std::make_pair(lambda, v);
    }

    /**
     * SVD placeholder (not implemented) – returns identity-like decomposition.
     */
    template <class E>
    inline auto svd(const expression<E>& A)
    {
        auto sh = A.derived().shape();
        if (sh.size() != 2) throw std::runtime_error("svd: 2D matrix required.");
        std::size_t m = sh[0], n = sh[1];
        array<double> U({m, m}, 0.0), Vt({n, n}, 0.0);
        for (std::size_t i = 0; i < m; ++i) U(i, i) = 1.0;
        for (std::size_t i = 0; i < n; ++i) Vt(i, i) = 1.0;
        array<double> S({std::min(m, n)}, 0.0);
        throw std::runtime_error("SVD not yet implemented.");
    }

} // namespace linalg
} // namespace numdot

#endif // NUMDOT_LINALG_H