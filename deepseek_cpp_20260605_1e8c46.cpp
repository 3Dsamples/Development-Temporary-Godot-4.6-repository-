//File 0023 : core/xlinalg.hpp
//Linear algebra: matrix multiplication, determinant, inverse, linear system solve with SIMD blocking and pivoting.
#ifndef XTENSOR_XLINALG_HPP
#define XTENSOR_XLINALG_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xreducer.hpp"
#include "xeval.hpp"
#include "xnorm.hpp"
#include "xbroadcast.hpp"
#include "xmanipulation.hpp"

namespace xt {
namespace linalg {

    using value_type_default = double;

    // Forward declaration for internal use
    template <class T>
    using xarray_t = xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;

    /********************************************
     * trace - sum of diagonal elements
     ********************************************/
    /**
     * Compute the trace of a 2D matrix (sum of diagonal elements).
     */
    template <class E>
    inline auto trace(const E& mat) {
        auto sh = mat.shape();
        if (sh.size() != 2)
            throw std::runtime_error("trace requires a 2D matrix.");
        std::size_t n = std::min(sh[0], sh[1]);
        typename std::decay_t<E>::value_type tr = 0;
        for (std::size_t i = 0; i < n; ++i)
            tr += mat(i, i);
        return tr;
    }

    /********************************************
     * dot / inner product (1D vectors)
     ********************************************/
    /**
     * Dot product of two 1D vectors.
     */
    template <class E1, class E2>
    inline auto dot(const E1& a, const E2& b) {
        auto sh_a = a.shape(), sh_b = b.shape();
        if (sh_a.size() != 1 || sh_b.size() != 1 || sh_a[0] != sh_b[0])
            throw std::runtime_error("dot: incompatible 1D shapes.");
        using T = typename std::decay_t<E1>::value_type;
        T result = 0;
        if constexpr (is_simd_enabled_v<T>) {
            using batch = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = batch::size;
            std::size_t n = sh_a[0];
            std::size_t vec_count = n / simd_size;
            batch vsum(0);
            for (std::size_t i = 0; i < vec_count; ++i) {
                batch va = batch::load_unaligned(&a[i * simd_size]);
                batch vb = batch::load_unaligned(&b[i * simd_size]);
                vsum = vsum + va * vb;
            }
            T tmp[simd_size];
            vsum.store_unaligned(tmp);
            for (std::size_t k = 0; k < simd_size; ++k) result += tmp[k];
            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                result += a[i] * b[i];
        } else {
            for (std::size_t i = 0; i < sh_a[0]; ++i)
                result += a[i] * b[i];
        }
        return result;
    }

    /********************************************
     * outer product (1D vectors -> 2D matrix)
     ********************************************/
    /**
     * Outer product of two 1D vectors.
     */
    template <class E1, class E2>
    inline auto outer(const E1& a, const E2& b) {
        using T = typename std::decay_t<E1>::value_type;
        auto m = a.shape()[0], n = b.shape()[0];
        xarray_t<T> result({m, n});
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                result(i, j) = a[i] * b[j];
        return result;
    }

    /********************************************
     * matmul / matrix multiplication (2D)
     ********************************************/
    /**
     * Matrix multiplication C = A * B with SIMD and blocking.
     */
    template <class E1, class E2>
    inline auto matmul(const E1& a, const E2& b) {
        using T = typename std::decay_t<E1>::value_type;
        auto sh_a = a.shape(), sh_b = b.shape();
        if (sh_a.size() != 2 || sh_b.size() != 2)
            throw std::runtime_error("matmul requires 2D matrices.");
        std::size_t m = sh_a[0], k = sh_a[1], n = sh_b[0];
        if (sh_b[1] != k)
            throw std::runtime_error("matmul inner dimensions mismatch.");
        // b is expected k x n? Shape is (k, n) if row-major? We'll assume both row-major: a(m x k), b(k x n)
        // So b shape[0] should equal k. We'll adjust: if user passes b with shape (k,n), good.
        if (sh_b[0] != k)
            throw std::runtime_error("matmul: a.columns != b.rows");
        xarray_t<T> result({m, sh_b[1]}, T(0));
        constexpr std::size_t BLOCK = 64; // cache blocking
        for (std::size_t i0 = 0; i0 < m; i0 += BLOCK) {
            std::size_t i_end = std::min(i0 + BLOCK, m);
            for (std::size_t j0 = 0; j0 < sh_b[1]; j0 += BLOCK) {
                std::size_t j_end = std::min(j0 + BLOCK, sh_b[1]);
                for (std::size_t k0 = 0; k0 < k; k0 += BLOCK) {
                    std::size_t k_end = std::min(k0 + BLOCK, k);
                    for (std::size_t i = i0; i < i_end; ++i) {
                        for (std::size_t kk = k0; kk < k_end; ++kk) {
                            T aik = a(i, kk);
                            if (aik == T(0)) continue;
                            auto* res_row = &result(i, 0);
                            const auto* b_row = &b(kk, 0);
                            // Use SIMD for inner j loop
                            std::size_t j = j0;
                            if constexpr (is_simd_enabled_v<T>) {
                                using batch = xsimd::batch<T, default_simd_arch>;
                                constexpr std::size_t simd_size = batch::size;
                                batch vaik(aik);
                                for (; j + simd_size <= j_end; j += simd_size) {
                                    batch vres = batch::load_unaligned(&res_row[j]);
                                    batch vb = batch::load_unaligned(&b_row[j]);
                                    vres = vres + vaik * vb;
                                    vres.store_unaligned(&res_row[j]);
                                }
                            }
                            for (; j < j_end; ++j) {
                                res_row[j] += aik * b_row[j];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * General matrix multiplication dispatching: if 1D treat as dot, else matmul.
     */
    template <class E1, class E2>
    inline auto matrix_dot(const E1& a, const E2& b) {
        if (a.shape().size() == 1 && b.shape().size() == 1)
            return dot(a, b);
        else if (a.shape().size() == 2 && b.shape().size() == 2)
            return matmul(a, b);
        else
            throw std::runtime_error("matrix_dot only supports 1D or 2D.");
    }

    /********************************************
     * Determinant via LU decomposition
     ********************************************/
    /**
     * Compute determinant of a square matrix using LU with partial pivoting.
     */
    template <class E>
    inline auto det(const E& mat) {
        auto sh = mat.shape();
        if (sh.size() != 2 || sh[0] != sh[1])
            throw std::runtime_error("det requires a square 2D matrix.");
        using T = typename std::decay_t<E>::value_type;
        std::size_t n = sh[0];
        auto A = xt::eval(mat); // copy
        T det_sign = 1.0;
        for (std::size_t i = 0; i < n; ++i) {
            // Partial pivot
            std::size_t pivot_row = i;
            T max_val = std::abs(A(i, i));
            for (std::size_t r = i + 1; r < n; ++r) {
                if (std::abs(A(r, i)) > max_val) {
                    max_val = std::abs(A(r, i));
                    pivot_row = r;
                }
            }
            if (max_val < std::numeric_limits<T>::epsilon() * 100)
                return T(0); // singular
            if (pivot_row != i) {
                det_sign = -det_sign;
                // swap rows i and pivot_row
                for (std::size_t j = 0; j < n; ++j)
                    std::swap(A(i, j), A(pivot_row, j));
            }
            T pivot = A(i, i);
            det_sign *= pivot;
            // eliminate below
            for (std::size_t r = i + 1; r < n; ++r) {
                T factor = A(r, i) / pivot;
                A(r, i) = 0;
                for (std::size_t c = i + 1; c < n; ++c)
                    A(r, c) -= factor * A(i, c);
            }
        }
        return det_sign;
    }

    /********************************************
     * Inverse via Gauss-Jordan
     ********************************************/
    /**
     * Compute inverse of a square matrix using Gauss-Jordan elimination with partial pivoting.
     */
    template <class E>
    inline auto inv(const E& mat) {
        auto sh = mat.shape();
        if (sh.size() != 2 || sh[0] != sh[1])
            throw std::runtime_error("inv requires square matrix.");
        using T = typename std::decay_t<E>::value_type;
        std::size_t n = sh[0];
        xarray_t<T> A = xt::eval(mat);
        xarray_t<T> I({n, n}, T(0));
        for (std::size_t i = 0; i < n; ++i) I(i, i) = T(1);
        for (std::size_t i = 0; i < n; ++i) {
            // partial pivot
            std::size_t pivot_row = i;
            T max_val = std::abs(A(i, i));
            for (std::size_t r = i + 1; r < n; ++r) {
                if (std::abs(A(r, i)) > max_val) {
                    max_val = std::abs(A(r, i));
                    pivot_row = r;
                }
            }
            if (max_val < 1e-14)
                throw std::runtime_error("Matrix is singular.");
            if (pivot_row != i) {
                for (std::size_t j = 0; j < n; ++j) {
                    std::swap(A(i, j), A(pivot_row, j));
                    std::swap(I(i, j), I(pivot_row, j));
                }
            }
            T pivot = A(i, i);
            // normalize row i
            for (std::size_t j = 0; j < n; ++j) {
                A(i, j) /= pivot;
                I(i, j) /= pivot;
            }
            // eliminate other rows
            for (std::size_t r = 0; r < n; ++r) {
                if (r == i) continue;
                T factor = A(r, i);
                if (factor != T(0)) {
                    for (std::size_t j = 0; j < n; ++j) {
                        A(r, j) -= factor * A(i, j);
                        I(r, j) -= factor * I(i, j);
                    }
                }
            }
        }
        return I;
    }

    /********************************************
     * Solve linear system Ax = b via LU
     ********************************************/
    /**
     * Solve Ax = b for x using LU decomposition with partial pivoting.
     */
    template <class E1, class E2>
    inline auto solve(const E1& A, const E2& b) {
        auto shA = A.shape(), shb = b.shape();
        if (shA.size() != 2 || shA[0] != shA[1] || shb.size() != 1 || shb[0] != shA[0])
            throw std::runtime_error("solve: A must be square, b 1D with matching size.");
        using T = typename std::decay_t<E1>::value_type;
        std::size_t n = shA[0];
        auto LU = xt::eval(A);
        auto x = xt::eval(b);
        std::vector<std::size_t> perm(n);
        for (std::size_t i = 0; i < n; ++i) perm[i] = i;
        // LU with partial pivot
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t pivot_row = i;
            T max_val = std::abs(LU(i, i));
            for (std::size_t r = i + 1; r < n; ++r) {
                if (std::abs(LU(r, i)) > max_val) {
                    max_val = std::abs(LU(r, i));
                    pivot_row = r;
                }
            }
            if (max_val < 1e-14) throw std::runtime_error("Singular matrix.");
            if (pivot_row != i) {
                std::swap(perm[i], perm[pivot_row]);
                for (std::size_t j = 0; j < n; ++j)
                    std::swap(LU(i, j), LU(pivot_row, j));
            }
            for (std::size_t r = i + 1; r < n; ++r) {
                T factor = LU(r, i) / LU(i, i);
                LU(r, i) = factor;
                for (std::size_t j = i + 1; j < n; ++j)
                    LU(r, j) -= factor * LU(i, j);
            }
        }
        // forward solve Ly = Pb
        xarray_t<T> y({n});
        for (std::size_t i = 0; i < n; ++i) {
            T sum = b[perm[i]];
            for (std::size_t j = 0; j < i; ++j)
                sum -= LU(i, j) * y[j];
            y[i] = sum;
        }
        // backward solve Ux = y
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j)
                sum -= LU(i, j) * x[j];
            x[i] = sum / LU(i, i);
        }
        return x;
    }

    /********************************************
     * Eigenvalues and eigenvectors via power iteration (placeholder)
     ********************************************/
    // Not fully implemented; can be added later.

    /********************************************
     * Norms (re-export from xnorm with linalg alias)
     ********************************************/
    template <class E>
    inline auto norm(const E& e, const std::string& type = "l2") {
        return xt::norm::norm(e, type);
    }

} // namespace linalg
} // namespace xt

#endif // XTENSOR_XLINALG_HPP