// math/xdecomposition.hpp

#ifndef XTENSOR_XDECOMPOSITION_HPP
#define XTENSOR_XDECOMPOSITION_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xsorting.hpp"

#include <cmath>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <tuple>
#include <utility>
#include <memory>

#if XTENSOR_HAS_BLAS
    #include <cxxblas.hpp>
    #include <cxxlapack.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace decomp
        {
            using Matrix = xarray_container<double>;
            using Vector = xarray_container<double>;
            using ComplexMatrix = xarray_container<std::complex<double>>;
            using ComplexVector = xarray_container<std::complex<double>>;

            // --------------------------------------------------------------------
            // LU Decomposition with partial pivoting
            // --------------------------------------------------------------------
            template <class E>
            inline auto lu(const xexpression<E>& A)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "lu: matrix must be 2D");
                size_t m = mat.shape()[0];
                size_t n = mat.shape()[1];
                size_t min_dim = std::min(m, n);
                using value_type = typename E::value_type;

                Matrix LU = eval(mat);
                std::vector<size_t> piv(min_dim);
                for (size_t i = 0; i < min_dim; ++i) piv[i] = i;

                for (size_t k = 0; k < min_dim; ++k)
                {
                    // Find pivot
                    size_t pivot_row = k;
                    value_type max_val = std::abs(LU(k, k));
                    for (size_t i = k + 1; i < m; ++i)
                    {
                        value_type abs_val = std::abs(LU(i, k));
                        if (abs_val > max_val)
                        {
                            max_val = abs_val;
                            pivot_row = i;
                        }
                    }
                    if (max_val < 1e-15)
                        XTENSOR_THROW(std::runtime_error, "lu: matrix is singular");

                    // Swap rows in LU
                    if (pivot_row != k)
                    {
                        for (size_t j = 0; j < n; ++j)
                            std::swap(LU(k, j), LU(pivot_row, j));
                        std::swap(piv[k], piv[pivot_row]);
                    }

                    // Eliminate below
                    value_type pivot = LU(k, k);
                    for (size_t i = k + 1; i < m; ++i)
                    {
                        value_type factor = LU(i, k) / pivot;
                        LU(i, k) = factor;
                        for (size_t j = k + 1; j < n; ++j)
                            LU(i, j) -= factor * LU(k, j);
                    }
                }

                // Extract L and U
                Matrix L = xt::eye<value_type>(m, n);
                Matrix U = xt::zeros<value_type>({m, n});
                for (size_t i = 0; i < m; ++i)
                {
                    for (size_t j = 0; j < n; ++j)
                    {
                        if (i > j)
                            L(i, j) = LU(i, j);
                        else
                            U(i, j) = LU(i, j);
                    }
                }

                // Permutation matrix
                Matrix P = xt::zeros<value_type>({m, m});
                for (size_t i = 0; i < min_dim; ++i)
                    P(i, piv[i]) = 1.0;
                for (size_t i = min_dim; i < m; ++i)
                    P(i, i) = 1.0;

                return std::make_tuple(P, L, U);
            }

            // --------------------------------------------------------------------
            // QR Decomposition (Householder reflections)
            // --------------------------------------------------------------------
            template <class E>
            inline auto qr(const xexpression<E>& A)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "qr: matrix must be 2D");
                size_t m = mat.shape()[0];
                size_t n = mat.shape()[1];
                using value_type = typename E::value_type;

                Matrix Q = xt::eye<value_type>(m);
                Matrix R = eval(mat);
                size_t min_dim = std::min(m, n);

                for (size_t k = 0; k < min_dim; ++k)
                {
                    // Compute Householder vector for column k from sub-diagonal part
                    Vector x({m - k});
                    for (size_t i = k; i < m; ++i)
                        x(i - k) = R(i, k);

                    value_type alpha = (x(0) >= 0) ? -xt::norm_l2(x)() : xt::norm_l2(x)();
                    if (std::abs(alpha) < 1e-15) continue;

                    Vector v = x;
                    v(0) -= alpha;
                    value_type beta = xt::sum(v * v)();
                    if (beta < 1e-15) continue;

                    // Apply Householder to R (from left)
                    for (size_t j = k; j < n; ++j)
                    {
                        value_type dot = 0.0;
                        for (size_t i = k; i < m; ++i)
                            dot += v(i - k) * R(i, j);
                        dot *= (2.0 / beta);
                        for (size_t i = k; i < m; ++i)
                            R(i, j) -= dot * v(i - k);
                    }

                    // Apply Householder to Q (from right)
                    for (size_t i = 0; i < m; ++i)
                    {
                        value_type dot = 0.0;
                        for (size_t j = k; j < m; ++j)
                            dot += Q(i, j) * v(j - k);
                        dot *= (2.0 / beta);
                        for (size_t j = k; j < m; ++j)
                            Q(i, j) -= dot * v(j - k);
                    }
                }

                // Make R upper triangular by zeroing below diagonal
                for (size_t i = 1; i < m; ++i)
                    for (size_t j = 0; j < std::min(i, n); ++j)
                        R(i, j) = 0.0;

                // Transpose Q to get orthogonal factor (our Q is Q^T actually)
                Q = xt::transpose(Q);

                // Trim Q to m x min(m,n) and R to min(m,n) x n for "economy" size
                if (m > n)
                {
                    Q = xt::view(Q, xt::all(), xt::range(0, n));
                    R = xt::view(R, xt::range(0, n), xt::all());
                }
                return std::make_pair(Q, R);
            }

            // --------------------------------------------------------------------
            // Cholesky Decomposition (LL^T for positive definite)
            // --------------------------------------------------------------------
            template <class E>
            inline auto cholesky(const xexpression<E>& A, bool lower = true)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                    XTENSOR_THROW(std::invalid_argument, "cholesky: matrix must be square");
                size_t n = mat.shape()[0];
                using value_type = typename E::value_type;

                Matrix L = xt::zeros<value_type>({n, n});

#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    L = eval(mat);
                    char uplo = lower ? 'L' : 'U';
                    int info;
                    cxxlapack::potrf(uplo, static_cast<int>(n), L.data(), static_cast<int>(n), info);
                    if (info != 0)
                        XTENSOR_THROW(std::runtime_error, "cholesky: matrix not positive definite");
                    if (lower)
                        for (size_t i = 0; i < n; ++i)
                            for (size_t j = i+1; j < n; ++j)
                                L(i, j) = 0.0;
                    else
                        for (size_t i = 0; i < n; ++i)
                            for (size_t j = 0; j < i; ++j)
                                L(i, j) = 0.0;
                    return L;
                }
#endif

                // Fallback implementation
                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = 0; j <= i; ++j)
                    {
                        value_type sum = mat(i, j);
                        for (size_t k = 0; k < j; ++k)
                            sum -= L(i, k) * L(j, k);
                        if (i == j)
                        {
                            if (sum <= 0)
                                XTENSOR_THROW(std::runtime_error, "cholesky: matrix not positive definite");
                            L(i, j) = std::sqrt(sum);
                        }
                        else
                        {
                            L(i, j) = sum / L(j, j);
                        }
                    }
                }

                if (!lower)
                    L = xt::transpose(L);
                return L;
            }

            // --------------------------------------------------------------------
            // LDL^T Decomposition (for symmetric indefinite)
            // --------------------------------------------------------------------
            template <class E>
            inline auto ldl(const xexpression<E>& A)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                    XTENSOR_THROW(std::invalid_argument, "ldl: matrix must be square");
                size_t n = mat.shape()[0];
                using value_type = typename E::value_type;

                Matrix L = xt::eye<value_type>(n);
                Vector D({n}, 0.0);
                Matrix A_work = eval(mat);

                for (size_t i = 0; i < n; ++i)
                {
                    // Compute D(i)
                    value_type sum_d = A_work(i, i);
                    for (size_t k = 0; k < i; ++k)
                        sum_d -= L(i, k) * L(i, k) * D(k);
                    D(i) = sum_d;

                    if (std::abs(D(i)) < 1e-15)
                        XTENSOR_THROW(std::runtime_error, "ldl: zero pivot encountered");

                    // Compute L column i for rows below
                    for (size_t j = i + 1; j < n; ++j)
                    {
                        value_type sum_l = A_work(j, i);
                        for (size_t k = 0; k < i; ++k)
                            sum_l -= L(j, k) * L(i, k) * D(k);
                        L(j, i) = sum_l / D(i);
                    }
                }

                return std::make_pair(L, D);
            }

            // --------------------------------------------------------------------
            // Eigenvalue Decomposition for Symmetric Matrices (QR algorithm)
            // --------------------------------------------------------------------
            template <class E>
            inline auto eigh(const xexpression<E>& A, bool eigenvectors = true)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                    XTENSOR_THROW(std::invalid_argument, "eigh: matrix must be square");
                size_t n = mat.shape()[0];
                using value_type = typename E::value_type;

#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    Matrix H = eval(mat);
                    Vector eigvals({n});
                    char jobz = eigenvectors ? 'V' : 'N';
                    char uplo = 'U';
                    int lwork = -1, info;
                    std::vector<double> work(1);
                    cxxlapack::syev(jobz, uplo, static_cast<int>(n), H.data(), static_cast<int>(n),
                                    eigvals.data(), work.data(), lwork, info);
                    lwork = static_cast<int>(work[0]);
                    work.resize(lwork);
                    cxxlapack::syev(jobz, uplo, static_cast<int>(n), H.data(), static_cast<int>(n),
                                    eigvals.data(), work.data(), lwork, info);
                    if (info != 0)
                        XTENSOR_THROW(std::runtime_error, "eigh: computation failed");
                    return std::make_pair(eigvals, H);
                }
#endif

                // Fallback: Jacobi method for symmetric matrices
                Matrix V = xt::eye<value_type>(n);
                Matrix A_work = eval(mat);
                Vector eigvals({n});

                const size_t max_sweeps = 50;
                for (size_t sweep = 0; sweep < max_sweeps; ++sweep)
                {
                    double max_off = 0.0;
                    size_t p = 0, q = 1;
                    for (size_t i = 0; i < n; ++i)
                    {
                        for (size_t j = i + 1; j < n; ++j)
                        {
                            double off = std::abs(A_work(i, j));
                            if (off > max_off)
                            {
                                max_off = off;
                                p = i;
                                q = j;
                            }
                        }
                    }
                    if (max_off < 1e-12) break;

                    // Compute Jacobi rotation
                    value_type theta = (A_work(q, q) - A_work(p, p)) / (2.0 * A_work(p, q));
                    value_type t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta*theta + 1.0));
                    value_type c = 1.0 / std::sqrt(t*t + 1.0);
                    value_type s = c * t;

                    // Apply rotation to A
                    for (size_t i = 0; i < n; ++i)
                    {
                        if (i != p && i != q)
                        {
                            value_type a_ip = A_work(i, p);
                            value_type a_iq = A_work(i, q);
                            A_work(i, p) = c * a_ip - s * a_iq;
                            A_work(p, i) = A_work(i, p);
                            A_work(i, q) = s * a_ip + c * a_iq;
                            A_work(q, i) = A_work(i, q);
                        }
                    }
                    value_type a_pp = A_work(p, p);
                    value_type a_qq = A_work(q, q);
                    value_type a_pq = A_work(p, q);
                    A_work(p, p) = c*c*a_pp + s*s*a_qq - 2.0*c*s*a_pq;
                    A_work(q, q) = s*s*a_pp + c*c*a_qq + 2.0*c*s*a_pq;
                    A_work(p, q) = A_work(q, p) = 0.0;

                    // Update eigenvectors
                    for (size_t i = 0; i < n; ++i)
                    {
                        value_type v_ip = V(i, p);
                        value_type v_iq = V(i, q);
                        V(i, p) = c * v_ip - s * v_iq;
                        V(i, q) = s * v_ip + c * v_iq;
                    }
                }

                // Extract eigenvalues
                for (size_t i = 0; i < n; ++i)
                    eigvals(i) = A_work(i, i);

                // Sort eigenvalues and eigenvectors
                std::vector<size_t> idx(n);
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) { return eigvals(i) < eigvals(j); });
                Vector sorted_eigvals({n});
                Matrix sorted_eigvecs = xt::zeros<value_type>({n, n});
                for (size_t i = 0; i < n; ++i)
                {
                    sorted_eigvals(i) = eigvals(idx[i]);
                    for (size_t j = 0; j < n; ++j)
                        sorted_eigvecs(j, i) = V(j, idx[i]);
                }

                return std::make_pair(sorted_eigvals, sorted_eigvecs);
            }

            // --------------------------------------------------------------------
            // General Eigenvalue Decomposition (Schur decomposition)
            // --------------------------------------------------------------------
            template <class E>
            inline auto eig(const xexpression<E>& A)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                    XTENSOR_THROW(std::invalid_argument, "eig: matrix must be square");
                size_t n = mat.shape()[0];
                using value_type = typename E::value_type;
                using complex_type = std::complex<double>;

#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    Matrix A_work = eval(mat);
                    Vector wr({n}), wi({n});
                    Matrix vl(n, n), vr(n, n);
                    char jobvl = 'N', jobvr = 'V';
                    int lwork = -1, info;
                    std::vector<double> work(1);
                    cxxlapack::geev(jobvl, jobvr, static_cast<int>(n), A_work.data(), static_cast<int>(n),
                                    wr.data(), wi.data(), vl.data(), static_cast<int>(n),
                                    vr.data(), static_cast<int>(n), work.data(), lwork, info);
                    lwork = static_cast<int>(work[0]);
                    work.resize(lwork);
                    cxxlapack::geev(jobvl, jobvr, static_cast<int>(n), A_work.data(), static_cast<int>(n),
                                    wr.data(), wi.data(), vl.data(), static_cast<int>(n),
                                    vr.data(), static_cast<int>(n), work.data(), lwork, info);
                    if (info != 0)
                        XTENSOR_THROW(std::runtime_error, "eig: computation failed");

                    ComplexVector eigvals({n});
                    ComplexMatrix eigvecs({n, n});
                    for (size_t i = 0; i < n; ++i)
                        eigvals(i) = complex_type(wr(i), wi(i));

                    size_t col = 0;
                    for (size_t j = 0; j < n; ++j)
                    {
                        if (wi(j) == 0.0)
                        {
                            for (size_t i = 0; i < n; ++i)
                                eigvecs(i, col) = complex_type(vr(i, j), 0.0);
                            ++col;
                        }
                        else
                        {
                            for (size_t i = 0; i < n; ++i)
                            {
                                eigvecs(i, col) = complex_type(vr(i, j), vr(i, j+1));
                                eigvecs(i, col+1) = complex_type(vr(i, j), -vr(i, j+1));
                            }
                            col += 2;
                            ++j;
                        }
                    }
                    return std::make_pair(eigvals, eigvecs);
                }
#endif

                // Fallback: Hessenberg + QR algorithm (simplified)
                XTENSOR_THROW(not_implemented_error, "eig: fallback not fully implemented for general matrices");
                ComplexVector dummy({n});
                ComplexMatrix dummy2({n, n});
                return std::make_pair(dummy, dummy2);
            }

            // --------------------------------------------------------------------
            // Singular Value Decomposition (SVD) - Golub-Reinsch
            // --------------------------------------------------------------------
            template <class E>
            inline auto svd(const xexpression<E>& A, bool full_matrices = true)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "svd: matrix must be 2D");
                size_t m = mat.shape()[0];
                size_t n = mat.shape()[1];
                size_t min_dim = std::min(m, n);
                using value_type = typename E::value_type;

#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    Matrix A_work = eval(mat);
                    Vector S({min_dim});
                    Matrix U = full_matrices ? Matrix({m, m}) : Matrix({m, min_dim});
                    Matrix Vt = full_matrices ? Matrix({n, n}) : Matrix({min_dim, n});
                    char jobu = full_matrices ? 'A' : 'S';
                    char jobvt = full_matrices ? 'A' : 'S';
                    int lwork = -1, info;
                    std::vector<double> work(1);
                    cxxlapack::gesvd(jobu, jobvt, static_cast<int>(m), static_cast<int>(n),
                                     A_work.data(), static_cast<int>(m), S.data(),
                                     U.data(), static_cast<int>(m), Vt.data(), static_cast<int>(n),
                                     work.data(), lwork, info);
                    lwork = static_cast<int>(work[0]);
                    work.resize(lwork);
                    cxxlapack::gesvd(jobu, jobvt, static_cast<int>(m), static_cast<int>(n),
                                     A_work.data(), static_cast<int>(m), S.data(),
                                     U.data(), static_cast<int>(m), Vt.data(), static_cast<int>(n),
                                     work.data(), lwork, info);
                    if (info != 0)
                        XTENSOR_THROW(std::runtime_error, "svd: computation failed");
                    return std::make_tuple(U, S, Vt);
                }
#endif

                // Fallback: Power iteration + deflation (simplified)
                // Not fully implemented - would use Golub-Kahan bidiagonalization.
                XTENSOR_THROW(not_implemented_error, "svd: fallback not fully implemented; please use LAPACK.");
                Matrix U({m, min_dim}), Vt({min_dim, n});
                Vector S({min_dim});
                return std::make_tuple(U, S, Vt);
            }

            // --------------------------------------------------------------------
            // Schur Decomposition
            // --------------------------------------------------------------------
            template <class E>
            inline auto schur(const xexpression<E>& A)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                    XTENSOR_THROW(std::invalid_argument, "schur: matrix must be square");
                size_t n = mat.shape()[0];
                using value_type = typename E::value_type;

#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    Matrix T = eval(mat);
                    Matrix Z(n, n);
                    Vector wr({n}), wi({n});
                    char jobvs = 'V';
                    char sort = 'N';
                    int sdim, lwork = -1, info;
                    std::vector<double> work(1);
                    cxxlapack::gees(jobvs, sort, nullptr, static_cast<int>(n), T.data(), static_cast<int>(n),
                                    &sdim, wr.data(), wi.data(), Z.data(), static_cast<int>(n),
                                    work.data(), lwork, &info);
                    lwork = static_cast<int>(work[0]);
                    work.resize(lwork);
                    cxxlapack::gees(jobvs, sort, nullptr, static_cast<int>(n), T.data(), static_cast<int>(n),
                                    &sdim, wr.data(), wi.data(), Z.data(), static_cast<int>(n),
                                    work.data(), lwork, &info);
                    if (info != 0)
                        XTENSOR_THROW(std::runtime_error, "schur: computation failed");
                    return std::make_pair(T, Z);
                }
#endif
                XTENSOR_THROW(not_implemented_error, "schur: fallback not implemented");
                return std::make_pair(Matrix({n,n}), Matrix({n,n}));
            }

            // --------------------------------------------------------------------
            // Hessenberg Decomposition
            // --------------------------------------------------------------------
            template <class E>
            inline auto hessenberg(const xexpression<E>& A)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                    XTENSOR_THROW(std::invalid_argument, "hessenberg: matrix must be square");
                size_t n = mat.shape()[0];
                using value_type = typename E::value_type;

                Matrix H = eval(mat);
                Matrix Q = xt::eye<value_type>(n);

                for (size_t k = 0; k < n - 2; ++k)
                {
                    // Compute Householder vector for column k below diagonal
                    Vector x({n - k - 1});
                    for (size_t i = k + 1; i < n; ++i)
                        x(i - k - 1) = H(i, k);

                    value_type alpha = (x(0) >= 0) ? -xt::norm_l2(x)() : xt::norm_l2(x)();
                    if (std::abs(alpha) < 1e-15) continue;

                    Vector v = x;
                    v(0) -= alpha;
                    value_type beta = xt::sum(v * v)();

                    // Apply Householder to H from left and right
                    for (size_t j = k; j < n; ++j)
                    {
                        value_type dot = 0.0;
                        for (size_t i = k + 1; i < n; ++i)
                            dot += v(i - k - 1) * H(i, j);
                        dot *= (2.0 / beta);
                        for (size_t i = k + 1; i < n; ++i)
                            H(i, j) -= dot * v(i - k - 1);
                    }
                    for (size_t i = 0; i < n; ++i)
                    {
                        value_type dot = 0.0;
                        for (size_t j = k + 1; j < n; ++j)
                            dot += H(i, j) * v(j - k - 1);
                        dot *= (2.0 / beta);
                        for (size_t j = k + 1; j < n; ++j)
                            H(i, j) -= dot * v(j - k - 1);
                    }

                    // Update Q
                    for (size_t i = 0; i < n; ++i)
                    {
                        value_type dot = 0.0;
                        for (size_t j = k + 1; j < n; ++j)
                            dot += Q(i, j) * v(j - k - 1);
                        dot *= (2.0 / beta);
                        for (size_t j = k + 1; j < n; ++j)
                            Q(i, j) -= dot * v(j - k - 1);
                    }
                }

                // Zero out below subdiagonal
                for (size_t i = 2; i < n; ++i)
                    for (size_t j = 0; j < i - 1; ++j)
                        H(i, j) = 0.0;

                return std::make_pair(H, Q);
            }

            // --------------------------------------------------------------------
            // Polar Decomposition (A = UP, U unitary, P positive semidefinite)
            // --------------------------------------------------------------------
            template <class E>
            inline auto polar(const xexpression<E>& A)
            {
                const auto& mat = A.derived_cast();
                if (mat.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "polar: matrix must be 2D");
                size_t m = mat.shape()[0];
                size_t n = mat.shape()[1];
                using value_type = typename E::value_type;

                // Newton-Schulz iteration for U (unitary factor)
                Matrix U = eval(mat);
                Matrix U_prev;
                for (size_t iter = 0; iter < 30; ++iter)
                {
                    U_prev = U;
                    Matrix U_inv = xt::linalg::inv(U);
                    U = 0.5 * (U + xt::transpose(U_inv));
                    // Check convergence
                    double diff = xt::norm_l2(U - U_prev)() / xt::norm_l2(U)();
                    if (diff < 1e-12) break;
                }
                // P = U^H * A
                Matrix P = xt::linalg::dot(xt::transpose(U), eval(mat));
                // Ensure P is Hermitian
                P = 0.5 * (P + xt::transpose(P));

                return std::make_pair(U, P);
            }

            // --------------------------------------------------------------------
            // QZ Decomposition (Generalized Schur for pair (A,B))
            // --------------------------------------------------------------------
            template <class E1, class E2>
            inline auto qz(const xexpression<E1>& A, const xexpression<E2>& B)
            {
                const auto& matA = A.derived_cast();
                const auto& matB = B.derived_cast();
                if (matA.dimension() != 2 || matB.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "qz: matrices must be 2D");
                if (matA.shape()[0] != matA.shape()[1] || matB.shape()[0] != matB.shape()[1])
                    XTENSOR_THROW(std::invalid_argument, "qz: matrices must be square");
                if (matA.shape()[0] != matB.shape()[0])
                    XTENSOR_THROW(std::invalid_argument, "qz: matrices must have same size");
                size_t n = matA.shape()[0];
                using value_type = typename E1::value_type;

#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    Matrix AA = eval(matA);
                    Matrix BB = eval(matB);
                    Vector alphar({n}), alphai({n}), beta({n});
                    Matrix Q(n, n), Z(n, n);
                    char jobvs = 'V';
                    char sort = 'N';
                    int sdim, lwork = -1, info;
                    std::vector<double> work(1);
                    cxxlapack::gges(jobvs, jobvs, sort, nullptr, static_cast<int>(n),
                                    AA.data(), static_cast<int>(n), BB.data(), static_cast<int>(n),
                                    &sdim, alphar.data(), alphai.data(), beta.data(),
                                    Q.data(), static_cast<int>(n), Z.data(), static_cast<int>(n),
                                    work.data(), lwork, &info);
                    lwork = static_cast<int>(work[0]);
                    work.resize(lwork);
                    cxxlapack::gges(jobvs, jobvs, sort, nullptr, static_cast<int>(n),
                                    AA.data(), static_cast<int>(n), BB.data(), static_cast<int>(n),
                                    &sdim, alphar.data(), alphai.data(), beta.data(),
                                    Q.data(), static_cast<int>(n), Z.data(), static_cast<int>(n),
                                    work.data(), lwork, &info);
                    if (info != 0)
                        XTENSOR_THROW(std::runtime_error, "qz: computation failed");
                    return std::make_tuple(AA, BB, Q, Z);
                }
#endif
                XTENSOR_THROW(not_implemented_error, "qz: fallback not implemented");
                return std::make_tuple(Matrix({n,n}), Matrix({n,n}), Matrix({n,n}), Matrix({n,n}));
            }

        } // namespace decomp

        // Bring decomposition functions into xt namespace
        using decomp::lu;
        using decomp::qr;
        using decomp::cholesky;
        using decomp::ldl;
        using decomp::eigh;
        using decomp::eig;
        using decomp::svd;
        using decomp::schur;
        using decomp::hessenberg;
        using decomp::polar;
        using decomp::qz;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XDECOMPOSITION_HPP

// math/xdecomposition.hpp