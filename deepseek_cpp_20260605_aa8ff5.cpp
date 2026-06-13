//File 0223 : sparse/xsparse_symmetry.hpp
//Symmetry-aware sparse operations: symmetric SpMV, triangular solve, and extraction of upper/lower triangular parts with SIMD acceleration.
#ifndef XTENSOR_XSPARSE_SYMMETRY_HPP
#define XTENSOR_XSPARSE_SYMMETRY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xarray.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../core/xeval.hpp"
#include "../core/xnorm.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_reducer.hpp"

namespace xt {
namespace sparse {

    /**
     * Extract the upper triangular part (including diagonal) of a CSR matrix.
     * Assumes the matrix may have entries in lower triangle; those are removed.
     * Returns a new CSR with only nonzeros where col >= row.
     */
    template <class T>
    inline auto upper_triangle(const xcsr_matrix<T>& A)
    {
        std::size_t n = A.rows();
        if (n != A.cols()) throw std::runtime_error("upper_triangle: matrix must be square.");
        xcoo_matrix<T> coo(n, n);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t j = A.row_ptr()[r]; j < A.row_ptr()[r + 1]; ++j)
            {
                std::size_t c = A.col_idx()[j];
                if (c >= r)
                    coo.append(r, c, A.values()[j]);
            }
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Extract the lower triangular part (including diagonal) of a CSR matrix.
     */
    template <class T>
    inline auto lower_triangle(const xcsr_matrix<T>& A)
    {
        std::size_t n = A.rows();
        if (n != A.cols()) throw std::runtime_error("lower_triangle: matrix must be square.");
        xcoo_matrix<T> coo(n, n);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t j = A.row_ptr()[r]; j < A.row_ptr()[r + 1]; ++j)
            {
                std::size_t c = A.col_idx()[j];
                if (c <= r)
                    coo.append(r, c, A.values()[j]);
            }
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Symmetric matrix-vector multiplication: y = A * x.
     * Assumes A is stored as upper triangular only (col >= row) and is symmetric.
     * This halves the number of operations compared to full SpMV.
     * For each entry (i,j) with i<=j, computes y[i] += A(i,j)*x[j] and y[j] += A(i,j)*x[i].
     */
    template <class T>
    inline auto symmetric_spmv(const xcsr_matrix<T>& A_upper, const xarray_container<uvector<T>>& x)
    {
        std::size_t n = A_upper.rows();
        if (n != A_upper.cols() || x.size() != n)
            throw std::runtime_error("symmetric_spmv: dimension mismatch.");
        xarray_container<uvector<T>> y({n}, T(0));
        T* y_data = y.data();
        const T* x_data = x.data();
        const std::size_t* row_ptr = A_upper.row_ptr().data();
        const std::size_t* col_idx = A_upper.col_idx().data();
        const T* values = A_upper.values().data();

        for (std::size_t r = 0; r < n; ++r)
        {
            T sum_diag = 0;
            for (std::size_t j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
            {
                std::size_t c = col_idx[j];
                T a = values[j];
                if (c == r)
                {
                    sum_diag += a * x_data[r];
                }
                else if (c > r)
                {
                    // y[r] += a * x[c]; y[c] += a * x[r];
                    T axc = a * x_data[c];
                    T axr = a * x_data[r];
                    y_data[r] += axc;
                    y_data[c] += axr;
                }
                // c < r should not appear in upper storage
            }
            y_data[r] += sum_diag;
        }
        return y;
    }

    /**
     * Symmetric triangular solve (unit lower triangular): L * x = b.
     * Assumes L is stored as lower triangular (including diagonal, which is 1).
     * Only the lower part (col <= row) is accessed.
     */
    template <class T>
    inline auto symmetric_lower_solve(const xcsr_matrix<T>& L, const xarray_container<uvector<T>>& b)
    {
        std::size_t n = L.rows();
        if (n != L.cols() || b.size() != n)
            throw std::runtime_error("symmetric_lower_solve: dimension mismatch.");
        xarray_container<uvector<T>> x({n}, T(0));
        T* x_data = x.data();
        const T* b_data = b.data();
        const std::size_t* row_ptr = L.row_ptr().data();
        const std::size_t* col_idx = L.col_idx().data();
        const T* values = L.values().data();

        for (std::size_t i = 0; i < n; ++i)
        {
            T sum = b_data[i];
            for (std::size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            {
                std::size_t col = col_idx[j];
                if (col >= i) break; // only lower part (col < i) contributes; diagonal (col=i) is 1.
                sum -= values[j] * x_data[col];
            }
            x_data[i] = sum; // diagonal = 1
        }
        return x;
    }

    /**
     * Symmetric upper triangular solve: U * x = b.
     * Assumes U is stored as upper triangular (col >= row). Back substitution.
     */
    template <class T>
    inline auto symmetric_upper_solve(const xcsr_matrix<T>& U, const xarray_container<uvector<T>>& b)
    {
        std::size_t n = U.rows();
        if (n != U.cols() || b.size() != n)
            throw std::runtime_error("symmetric_upper_solve: dimension mismatch.");
        xarray_container<uvector<T>> x({n}, T(0));
        T* x_data = x.data();
        const T* b_data = b.data();
        const std::size_t* row_ptr = U.row_ptr().data();
        const std::size_t* col_idx = U.col_idx().data();
        const T* values = U.values().data();

        // Backward substitution: for i = n-1 down to 0
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i)
        {
            T sum = b_data[i];
            std::size_t diag_pos = std::numeric_limits<std::size_t>::max();
            for (std::size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            {
                std::size_t col = col_idx[j];
                if (col == static_cast<std::size_t>(i))
                {
                    diag_pos = j;
                    continue;
                }
                if (col > static_cast<std::size_t>(i))
                    sum -= values[j] * x_data[col];
            }
            if (diag_pos == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error("symmetric_upper_solve: zero diagonal.");
            x_data[i] = sum / values[diag_pos];
        }
        return x;
    }

    /**
     * Check if a CSR matrix is symmetric (within tolerance) and return its upper triangular storage.
     * If symmetric, the upper triangle can be used for storage-efficient operations.
     */
    template <class T>
    inline std::pair<bool, xcsr_matrix<T>> is_symmetric_and_upper(const xcsr_matrix<T>& A, T tol = T(1e-10))
    {
        if (A.rows() != A.cols()) return {false, A};
        std::size_t n = A.rows();
        // Check symmetry A(i,j) == A(j,i)
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t j = A.row_ptr()[r]; j < A.row_ptr()[r + 1]; ++j)
            {
                std::size_t c = A.col_idx()[j];
                T aij = A.values()[j];
                if (c > r) // only check upper half once
                {
                    T aji = A(c, r);
                    if (std::abs(aij - aji) > tol)
                        return {false, A};
                }
            }
        }
        // Is symmetric, now extract upper triangle
        auto upper = upper_triangle(A);
        return {true, upper};
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_SYMMETRY_HPP