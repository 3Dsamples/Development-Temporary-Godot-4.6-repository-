//File 0023 (UPDATED) : core/xlinalg.hpp
//Linear algebra: matrix multiplication, determinant, inverse, solve, eigenvalues (power iteration, symmetric QR), SVD placeholder, with SIMD blocking.
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

    template <class T>
    using xarray_t = xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;

    /********************************************
     * trace
     ********************************************/
    template <class E>
    inline auto trace(const E& mat) {
        auto sh = mat.shape();
        if (sh.size() != 2) throw std::runtime_error("trace requires a 2D matrix.");
        std::size_t n = std::min(sh[0], sh[1]);
        typename std::decay_t<E>::value_type tr = 0;
        for (std::size_t i = 0; i < n; ++i) tr += mat(i, i);
        return tr;
    }

    /********************************************
     * dot / inner product
     ********************************************/
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
            for (std::size_t i = vec_count * simd_size; i < n; ++i) result += a[i] * b[i];
        } else {
            for (std::size_t i = 0; i < sh_a[0]; ++i) result += a[i] * b[i];
        }
        return result;
    }

    /********************************************
     * outer product
     ********************************************/
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
     * matmul
     ********************************************/
    template <class E1, class E2>
    inline auto matmul(const E1& a, const E2& b) {
        using T = typename std::decay_t<E1>::value_type;
        auto sh_a = a.shape(), sh_b = b.shape();
        if (sh_a.size() != 2 || sh_b.size() != 2)
            throw std::runtime_error("matmul requires 2D matrices.");
        std::size_t m = sh_a[0], k = sh_a[1], n = sh_b[1];
        if (sh_b[0] != k) throw std::runtime_error("matmul inner dimensions mismatch.");
        xarray_t<T> result({m, n}, T(0));
        constexpr std::size_t BLOCK = 64;
        for (std::size_t i0 = 0; i0 < m; i0 += BLOCK) {
            std::size_t i_end = std::min(i0 + BLOCK, m);
            for (std::size_t j0 = 0; j0 < n; j0 += BLOCK) {
                std::size_t j_end = std::min(j0 + BLOCK, n);
                for (std::size_t k0 = 0; k0 < k; k0 += BLOCK) {
                    std::size_t k_end = std::min(k0 + BLOCK, k);
                    for (std::size_t i = i0; i < i_end; ++i) {
                        for (std::size_t kk = k0; kk < k_end; ++kk) {
                            T aik = a(i, kk);
                            if (aik == T(0)) continue;
                            auto* res_row = &result(i, 0);
                            const auto* b_row = &b(kk, 0);
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

    template <class E1, class E2>
    inline auto matrix_dot(const E1& a, const E2& b) {
        if (a.shape().size() == 1 && b.shape().size() == 1) return dot(a, b);
        else if (a.shape().size() == 2 && b.shape().size() == 2) return matmul(a, b);
        else throw std::runtime_error("matrix_dot only supports 1D or 2D.");
    }

    /********************************************
     * Determinant
     ********************************************/
    template <class E>
    inline auto det(const E& mat) {
        auto sh = mat.shape();
        if (sh.size() != 2 || sh[0] != sh[1])
            throw std::runtime_error("det requires a square 2D matrix.");
        using T = typename std::decay_t<E>::value_type;
        std::size_t n = sh[0];
        auto A = xt::eval(mat);
        T det_sign = 1.0;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t pivot_row = i;
            T max_val = std::abs(A(i, i));
            for (std::size_t r = i + 1; r < n; ++r) {
                if (std::abs(A(r, i)) > max_val) {
                    max_val = std::abs(A(r, i));
                    pivot_row = r;
                }
            }
            if (max_val < 1e-14) return T(0);
            if (pivot_row != i) {
                det_sign = -det_sign;
                for (std::size_t j = 0; j < n; ++j) std::swap(A(i, j), A(pivot_row, j));
            }
            T pivot = A(i, i);
            det_sign *= pivot;
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
     * Inverse
     ********************************************/
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
            std::size_t pivot_row = i;
            T max_val = std::abs(A(i, i));
            for (std::size_t r = i + 1; r < n; ++r) {
                if (std::abs(A(r, i)) > max_val) {
                    max_val = std::abs(A(r, i));
                    pivot_row = r;
                }
            }
            if (max_val < 1e-14) throw std::runtime_error("Matrix is singular.");
            if (pivot_row != i) {
                for (std::size_t j = 0; j < n; ++j) {
                    std::swap(A(i, j), A(pivot_row, j));
                    std::swap(I(i, j), I(pivot_row, j));
                }
            }
            T pivot = A(i, i);
            for (std::size_t j = 0; j < n; ++j) {
                A(i, j) /= pivot;
                I(i, j) /= pivot;
            }
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
     * Solve Ax = b
     ********************************************/
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
        xarray_t<T> y({n});
        for (std::size_t i = 0; i < n; ++i) {
            T sum = b[perm[i]];
            for (std::size_t j = 0; j < i; ++j) sum -= LU(i, j) * y[j];
            y[i] = sum;
        }
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j) sum -= LU(i, j) * x[j];
            x[i] = sum / LU(i, i);
        }
        return x;
    }

    /********************************************
     * Eigenvalues: power iteration for dominant
     ********************************************/
    template <class E>
    inline auto eig_power(const E& A, std::size_t max_iter = 1000, double tol = 1e-12) {
        auto sh = A.shape();
        if (sh.size() != 2 || sh[0] != sh[1]) throw std::runtime_error("eig_power requires square matrix.");
        using T = typename std::decay_t<E>::value_type;
        std::size_t n = sh[0];
        // Random initial vector
        xarray_t<T> v({n});
        auto& eng = random::detail::get_global_engine();
        std::uniform_real_distribution<double> dist(-1,1);
        for (std::size_t i = 0; i < n; ++i) v[i] = dist(eng);
        v = v / xt::norm::norm_l2(v)();
        T lambda = 0;
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            auto Av = matmul(A, v);
            T lambda_new = dot(v, Av)();
            if (std::abs(lambda_new - lambda) < tol) {
                return std::make_pair(lambda_new, v);
            }
            lambda = lambda_new;
            v = Av / xt::norm::norm_l2(Av)();
        }
        return std::make_pair(lambda, v);
    }

    /********************************************
     * Symmetric QR algorithm for full spectrum
     ********************************************/
    template <class E>
    inline auto eig_sym(const E& A, std::size_t max_iter = 1000, double tol = 1e-12) {
        auto sh = A.shape();
        if (sh.size() != 2 || sh[0] != sh[1]) throw std::runtime_error("eig_sym requires square symmetric matrix.");
        using T = typename std::decay_t<E>::value_type;
        std::size_t n = sh[0];
        xarray_t<T> Q = xt::eye({n, n});
        xarray_t<T> R = xt::eval(A); // will hold the tridiagonal? Actually basic QR without tridiagonalization
        // Simple QR iteration (not efficient for large n, but correct)
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            // QR decomposition via Gram-Schmidt on columns
            xarray_t<T> Qk({n, n});
            xarray_t<T> Rk({n, n}, T(0));
            // Copy R into A
            xarray_t<T> Ak = R;
            for (std::size_t j = 0; j < n; ++j) {
                // Column j of Ak
                xarray_t<T> col({n});
                for (std::size_t i = 0; i < n; ++i) col[i] = Ak(i,j);
                for (std::size_t i = 0; i < j; ++i) {
                    T rij = 0;
                    for (std::size_t k = 0; k < n; ++k) rij += Qk(k,i) * Ak(k,j);
                    Rk(i,j) = rij;
                    for (std::size_t k = 0; k < n; ++k) col[k] -= rij * Qk(k,i);
                }
                T rjj = xt::norm::norm_l2(col)();
                Rk(j,j) = rjj;
                if (rjj > 1e-15) {
                    for (std::size_t k = 0; k < n; ++k) Qk(k,j) = col[k] / rjj;
                } else {
                    // column zero, just set to zero
                    for (std::size_t k = 0; k < n; ++k) Qk(k,j) = 0;
                }
            }
            R = matmul(Rk, Qk); // R_{k+1} = R_k * Q_k? Actually R_{new} = Rk * Qk (RQ iteration)
            Q = matmul(Q, Qk);
            // check convergence of off-diagonal elements
            T off = 0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    if (i != j) off += R(i,j)*R(i,j);
            if (std::sqrt(off) < tol) break;
        }
        // eigenvalues are on diagonal of R
        xarray_t<T> eigenvalues({n});
        for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = R(i,i);
        // eigenvectors are columns of Q
        return std::make_pair(eigenvalues, Q);
    }

    /********************************************
     * SVD placeholder (returns U, S, Vt)
     ********************************************/
    template <class E>
    inline auto svd(const E& A) {
        // For a full SVD we need a bidiagonalization, which is complex.
        // Placeholder: return identity-like decomposition.
        auto sh = A.shape();
        if (sh.size() != 2) throw std::runtime_error("svd requires 2D matrix.");
        using T = typename std::decay_t<E>::value_type;
        std::size_t m = sh[0], n = sh[1];
        auto U = xt::eye({m, m});
        auto Vt = xt::eye({n, n});
        xarray_t<T> S({std::min(m,n)});
        // Use power method on A^T A? Not now.
        throw std::runtime_error("SVD not implemented yet.");
    }

    /********************************************
     * Norms (re-export from xnorm)
     ********************************************/
    template <class E>
    inline auto norm(const E& e, const std::string& type = "l2") {
        return xt::norm::norm(e, type);
    }

    // Helper: identity matrix
    template <class T = double>
    inline auto eye(std::initializer_list<std::size_t> shape) {
        auto sh = std::vector<std::size_t>(shape);
        if (sh.size() != 2) throw std::runtime_error("eye requires 2D shape.");
        xarray_t<T> result(sh, T(0));
        for (std::size_t i = 0; i < std::min(sh[0], sh[1]); ++i) result(i,i) = T(1);
        return result;
    }

} // namespace linalg
} // namespace xt

#endif // XTENSOR_XLINALG_HPP