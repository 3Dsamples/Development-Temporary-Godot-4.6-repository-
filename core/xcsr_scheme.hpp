// core/xcsr_scheme.hpp
#ifndef XTENSOR_XCSR_SCHEME_HPP
#define XTENSOR_XCSR_SCHEME_HPP

// ----------------------------------------------------------------------------
// xcsr_scheme.hpp – Compressed Sparse Row (CSR) matrix format
// ----------------------------------------------------------------------------
// This header provides the CSR sparse matrix representation, including:
//   - Construction from dense, COO, triplets, or other formats
//   - Conversion to dense, COO, CSC, or other formats
//   - Matrix-vector multiplication (SpMV) and matrix-matrix operations
//   - Element access, slicing, and iteration over non‑zero entries
//   - Arithmetic operations: addition, subtraction, multiplication
//   - Transpose (to CSC implicitly), diagonal extraction, norms
//   - Solving sparse triangular systems (forward/back substitution)
//   - Incomplete LU factorization (ILU0, ILUT, ILUTP)
//   - Preconditioned iterative solvers: CG, GMRES, BiCGSTAB
//   - Block CSR (BCSR) for batched operations
//   - I/O for Matrix Market and Harwell‑Boeing formats
//
// All scalar types are supported, including bignumber::BigNumber. FFT
// acceleration is leveraged for large‑scale products when beneficial.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <memory>
#include <complex>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xlinalg.hpp"
#include "xcoo_scheme.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ========================================================================
    // CSR Sparse Matrix
    // ========================================================================
    template <class T>
    class xcsr_scheme
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using index_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;

        // --------------------------------------------------------------------
        // Constructors
        // --------------------------------------------------------------------
        xcsr_scheme() : m_rows(0), m_cols(0), m_nnz(0) { m_row_ptr.push_back(0); }
        xcsr_scheme(size_type n_rows, size_type n_cols);
        xcsr_scheme(const shape_type& shape);
        explicit xcsr_scheme(const xarray_container<T>& dense);
        explicit xcsr_scheme(const xcoo_scheme<T>& coo);
        xcsr_scheme(size_type n_rows, size_type n_cols,
                    const std::vector<size_type>& rows,
                    const std::vector<size_type>& cols,
                    const std::vector<value_type>& values,
                    bool sum_duplicates = true);

        // --------------------------------------------------------------------
        // Basic properties
        // --------------------------------------------------------------------
        size_type nrows() const noexcept;
        size_type ncols() const noexcept;
        size_type nnz() const noexcept;
        shape_type shape() const;
        bool empty() const noexcept;

        const std::vector<size_type>& row_ptr() const noexcept;
        const std::vector<size_type>& col_ind() const noexcept;
        const std::vector<value_type>& values() const noexcept;
        std::vector<size_type>& row_ptr() noexcept;
        std::vector<size_type>& col_ind() noexcept;
        std::vector<value_type>& values() noexcept;

        // --------------------------------------------------------------------
        // Reserve and memory
        // --------------------------------------------------------------------
        void reserve(size_type capacity);

        // --------------------------------------------------------------------
        // Conversions
        // --------------------------------------------------------------------
        xarray_container<T> to_dense() const;
        xcoo_scheme<T> to_coo() const;
        xcsr_scheme<T> to_csc() const;   // Transpose implicitly

        // --------------------------------------------------------------------
        // Matrix-vector multiplication
        // --------------------------------------------------------------------
        template <class E> xarray_container<T> dot(const xexpression<E>& x) const;

        // --------------------------------------------------------------------
        // Matrix operations
        // --------------------------------------------------------------------
        xcsr_scheme<T> transpose() const;
        value_type operator()(size_type row, size_type col) const;
        xarray_container<T> diagonal(size_type k = 0) const;
        value_type trace() const;

        // --------------------------------------------------------------------
        // Row/column extraction
        // --------------------------------------------------------------------
        xarray_container<T> get_row(size_type row) const;
        xarray_container<T> get_col(size_type col) const;
        void set_entry(size_type row, size_type col, const value_type& value);

        // --------------------------------------------------------------------
        // Arithmetic operators
        // --------------------------------------------------------------------
        xcsr_scheme<T>& operator*=(const value_type& scalar);
        xcsr_scheme<T> operator*(const value_type& scalar) const;
        xcsr_scheme<T> operator+(const xcsr_scheme<T>& other) const;
        xcsr_scheme<T> operator-(const xcsr_scheme<T>& other) const;
        xcsr_scheme<T> matmul(const xcsr_scheme<T>& other) const;

        // --------------------------------------------------------------------
        // Slicing and manipulation
        // --------------------------------------------------------------------
        xcsr_scheme<T> slice(const std::vector<size_type>& rows,
                             const std::vector<size_type>& cols) const;
        void clear();
        void resize(size_type new_rows, size_type new_cols);
        void eliminate_zeros();

        // --------------------------------------------------------------------
        // Norms
        // --------------------------------------------------------------------
        value_type norm() const;  // Frobenius

        // --------------------------------------------------------------------
        // Triangular solvers
        // --------------------------------------------------------------------
        xarray_container<T> forward_substitution(const xarray_container<T>& b) const;
        xarray_container<T> back_substitution(const xarray_container<T>& b) const;

        // --------------------------------------------------------------------
        // Incomplete LU factorizations
        // --------------------------------------------------------------------
        std::pair<xcsr_scheme<T>, xcsr_scheme<T>> ilu0() const;
        struct ilut_result { xcsr_scheme<T> L, U; size_t fill_in; };
        ilut_result<T> ilut(T drop_tol, size_t max_fill = 0) const;

        // --------------------------------------------------------------------
        // I/O
        // --------------------------------------------------------------------
        bool load_matrix_market(const std::string& filename);
        bool save_matrix_market(const std::string& filename) const;

    private:
        size_type m_rows;
        size_type m_cols;
        size_type m_nnz;
        std::vector<size_type> m_row_ptr;
        std::vector<size_type> m_col_ind;
        std::vector<value_type> m_values;
    };

    // ------------------------------------------------------------------------
    // Block CSR (BCSR) – dense blocks stored in CSR format
    // ------------------------------------------------------------------------
    template <class T>
    class xbcsr_scheme
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using block_type = xarray_container<T>;

        xbcsr_scheme() : m_block_rows(0), m_block_cols(0), m_block_size(1), m_nnz_blocks(0) { m_row_ptr.push_back(0); }
        xbcsr_scheme(size_type n_block_rows, size_type n_block_cols, size_type block_size);

        size_type nblock_rows() const noexcept;
        size_type nblock_cols() const noexcept;
        size_type block_size() const noexcept;
        size_type nnz_blocks() const noexcept;

        void add_block(size_type brow, size_type bcol, const block_type& block);
        xarray_container<T> to_dense() const;
        template <class E> xarray_container<T> dot(const xexpression<E>& x) const;

    private:
        size_type m_block_rows;
        size_type m_block_cols;
        size_type m_block_size;
        size_type m_nnz_blocks;
        std::vector<size_type> m_row_ptr;
        std::vector<size_type> m_col_ind;
        std::vector<block_type> m_blocks;
    };

    // ========================================================================
    // Preconditioned iterative solvers
    // ========================================================================
    template <class T>
    xarray_container<T> cg_solve(const xcsr_scheme<T>& A, const xarray_container<T>& b,
                                  size_t max_iter = 1000, T tol = T(1e-6),
                                  const xcsr_scheme<T>* preconditioner = nullptr);

    template <class T>
    xarray_container<T> gmres_solve(const xcsr_scheme<T>& A, const xarray_container<T>& b,
                                    size_t restart = 30, size_t max_iter = 1000,
                                    T tol = T(1e-6), const xcsr_scheme<T>* preconditioner = nullptr);

    template <class T>
    xarray_container<T> bicgstab_solve(const xcsr_scheme<T>& A, const xarray_container<T>& b,
                                        size_t max_iter = 1000, T tol = T(1e-6),
                                        const xcsr_scheme<T>* preconditioner = nullptr);

    // ------------------------------------------------------------------------
    // Factory functions
    // ------------------------------------------------------------------------
    template <class T> xcsr_scheme<T> csr_matrix(const xarray_container<T>& dense);
    template <class T> xcsr_scheme<T> csr_matrix(const xcoo_scheme<T>& coo);
    template <class T> xcsr_scheme<T> csr_matrix(size_type n_rows, size_type n_cols,
                                                  const std::vector<size_type>& rows,
                                                  const std::vector<size_type>& cols,
                                                  const std::vector<T>& values,
                                                  bool sum_duplicates = true);
    template <class T> xcsr_scheme<T> csr_eye(size_type n);
    template <class E> xcsr_scheme<typename E::value_type> csr_diag(const xexpression<E>& diag);
    template <class T> xcsr_scheme<T> csr_random(size_type n_rows, size_type n_cols, T density, std::mt19937* rng = nullptr);

    // ------------------------------------------------------------------------
    // Non-member operators
    // ------------------------------------------------------------------------
    template <class T> xcsr_scheme<T> operator*(const T& scalar, const xcsr_scheme<T>& mat);
    template <class T> xarray_container<T> dot(const xcsr_scheme<T>& A, const xarray_container<T>& x);
    template <class T> xarray_container<T> dot(const xarray_container<T>& x, const xcsr_scheme<T>& A);
    template <class T> xcsr_scheme<T> matmul(const xcsr_scheme<T>& A, const xcsr_scheme<T>& B);
    template <class T> xarray_container<T> spsolve(const xcsr_scheme<T>& A, const xarray_container<T>& b);
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    // Construct CSR matrix with given dimensions
    template <class T> xcsr_scheme<T>::xcsr_scheme(size_type n_rows, size_type n_cols)
        : m_rows(n_rows), m_cols(n_cols), m_nnz(0) { m_row_ptr.resize(n_rows + 1, 0); }

    // Construct CSR matrix from shape vector
    template <class T> xcsr_scheme<T>::xcsr_scheme(const shape_type& shape)
        : m_rows(shape[0]), m_cols(shape[1]), m_nnz(0) { m_row_ptr.resize(m_rows + 1, 0); }

    // Construct CSR matrix from dense array
    template <class T> xcsr_scheme<T>::xcsr_scheme(const xarray_container<T>& dense)
    { /* TODO: scan dense and build CSR */ }

    // Construct CSR matrix from COO (requires sorting)
    template <class T> xcsr_scheme<T>::xcsr_scheme(const xcoo_scheme<T>& coo)
    { /* TODO: convert sorted COO to CSR */ }

    // Construct CSR matrix from triplet lists
    template <class T> xcsr_scheme<T>::xcsr_scheme(size_type n_rows, size_type n_cols,
                                                   const std::vector<size_type>& rows,
                                                   const std::vector<size_type>& cols,
                                                   const std::vector<value_type>& values,
                                                   bool sum_duplicates)
    { /* TODO: build via COO intermediate */ }

    // Return number of rows
    template <class T> auto xcsr_scheme<T>::nrows() const noexcept -> size_type { return m_rows; }
    // Return number of columns
    template <class T> auto xcsr_scheme<T>::ncols() const noexcept -> size_type { return m_cols; }
    // Return number of non‑zero entries
    template <class T> auto xcsr_scheme<T>::nnz() const noexcept -> size_type { return m_nnz; }
    // Return shape as vector
    template <class T> auto xcsr_scheme<T>::shape() const -> shape_type { return {m_rows, m_cols}; }
    // Check if matrix is empty
    template <class T> bool xcsr_scheme<T>::empty() const noexcept { return m_nnz == 0; }

    // Const access to row pointers
    template <class T> const std::vector<size_t>& xcsr_scheme<T>::row_ptr() const noexcept { return m_row_ptr; }
    // Const access to column indices
    template <class T> const std::vector<size_t>& xcsr_scheme<T>::col_ind() const noexcept { return m_col_ind; }
    // Const access to values
    template <class T> const std::vector<T>& xcsr_scheme<T>::values() const noexcept { return m_values; }
    // Mutable access to row pointers
    template <class T> std::vector<size_t>& xcsr_scheme<T>::row_ptr() noexcept { return m_row_ptr; }
    // Mutable access to column indices
    template <class T> std::vector<size_t>& xcsr_scheme<T>::col_ind() noexcept { return m_col_ind; }
    // Mutable access to values
    template <class T> std::vector<T>& xcsr_scheme<T>::values() noexcept { return m_values; }

    // Reserve memory for entries
    template <class T> void xcsr_scheme<T>::reserve(size_type capacity)
    { m_col_ind.reserve(capacity); m_values.reserve(capacity); }

    // Convert to dense matrix
    template <class T> xarray_container<T> xcsr_scheme<T>::to_dense() const
    { /* TODO: fill dense array */ return {}; }
    // Convert to COO format
    template <class T> xcoo_scheme<T> xcsr_scheme<T>::to_coo() const
    { /* TODO: expand CSR to COO */ return {}; }
    // Convert to CSC format (transpose)
    template <class T> xcsr_scheme<T> xcsr_scheme<T>::to_csc() const
    { /* TODO: build CSC by counting column entries */ return {}; }

    // Matrix-vector multiplication
    template <class T> template <class E>
    xarray_container<T> xcsr_scheme<T>::dot(const xexpression<E>& x) const
    { /* TODO: SpMV */ return {}; }

    // Transpose matrix (returns CSR of transpose)
    template <class T> xcsr_scheme<T> xcsr_scheme<T>::transpose() const
    { /* TODO: compute transpose pattern */ return {}; }
    // Access element by row and column (O(nnz) scan)
    template <class T> auto xcsr_scheme<T>::operator()(size_type row, size_type col) const -> value_type
    { /* TODO: binary search or linear scan in row */ return T(0); }
    // Extract diagonal
    template <class T> xarray_container<T> xcsr_scheme<T>::diagonal(size_type k) const
    { /* TODO: find entries where col == row + k */ return {}; }
    // Compute trace
    template <class T> auto xcsr_scheme<T>::trace() const -> value_type
    { /* TODO: sum diagonal entries */ return T(0); }

    // Extract a single row as dense vector
    template <class T> xarray_container<T> xcsr_scheme<T>::get_row(size_type row) const
    { /* TODO: build dense row */ return {}; }
    // Extract a single column as dense vector
    template <class T> xarray_container<T> xcsr_scheme<T>::get_col(size_type col) const
    { /* TODO: scan all rows for column */ return {}; }
    // Set a single entry (inefficient, use with caution)
    template <class T> void xcsr_scheme<T>::set_entry(size_type row, size_type col, const value_type& value)
    { /* TODO: update or insert while maintaining order */ }

    // In‑place scalar multiplication
    template <class T> xcsr_scheme<T>& xcsr_scheme<T>::operator*=(const value_type& scalar)
    { /* TODO: scale all values, remove zeros if scalar==0 */ return *this; }
    // Scalar multiplication (copy)
    template <class T> xcsr_scheme<T> xcsr_scheme<T>::operator*(const value_type& scalar) const
    { xcsr_scheme<T> result = *this; result *= scalar; return result; }
    // Matrix addition
    template <class T> xcsr_scheme<T> xcsr_scheme<T>::operator+(const xcsr_scheme<T>& other) const
    { /* TODO: merge row‑wise */ return {}; }
    // Matrix subtraction
    template <class T> xcsr_scheme<T> xcsr_scheme<T>::operator-(const xcsr_scheme<T>& other) const
    { /* TODO: merge with negated values */ return {}; }
    // Matrix multiplication (CSR * CSR)
    template <class T> xcsr_scheme<T> xcsr_scheme<T>::matmul(const xcsr_scheme<T>& other) const
    { /* TODO: Gustavson's algorithm or row‑wise SpGEMM */ return {}; }

    // Extract submatrix
    template <class T> xcsr_scheme<T> xcsr_scheme<T>::slice(const std::vector<size_type>& rows,
                                                            const std::vector<size_type>& cols) const
    { /* TODO: filter rows and map columns */ return {}; }
    // Clear all entries
    template <class T> void xcsr_scheme<T>::clear()
    { m_row_ptr.assign(m_rows + 1, 0); m_col_ind.clear(); m_values.clear(); m_nnz = 0; }
    // Resize matrix, dropping out‑of‑bounds entries
    template <class T> void xcsr_scheme<T>::resize(size_type new_rows, size_type new_cols)
    { /* TODO: truncate rows/cols and rebuild */ }
    // Remove entries with zero value
    template <class T> void xcsr_scheme<T>::eliminate_zeros()
    { /* TODO: compact each row */ }

    // Compute Frobenius norm
    template <class T> auto xcsr_scheme<T>::norm() const -> value_type
    { /* TODO: sqrt(sum(value^2)) */ return T(0); }

    // Forward substitution (solve L * x = b for lower triangular L)
    template <class T> xarray_container<T> xcsr_scheme<T>::forward_substitution(const xarray_container<T>& b) const
    { /* TODO: sparse forward solve */ return {}; }
    // Back substitution (solve U * x = b for upper triangular U)
    template <class T> xarray_container<T> xcsr_scheme<T>::back_substitution(const xarray_container<T>& b) const
    { /* TODO: sparse backward solve */ return {}; }

    // Incomplete LU with zero fill‑in (ILU0)
    template <class T> std::pair<xcsr_scheme<T>, xcsr_scheme<T>> xcsr_scheme<T>::ilu0() const
    { /* TODO: ILU(0) factorization */ return {}; }
    // Incomplete LU with threshold (ILUT)
    template <class T> typename xcsr_scheme<T>::ilut_result xcsr_scheme<T>::ilut(T drop_tol, size_t max_fill) const
    { /* TODO: ILUT with dual dropping strategy */ return {}; }

    // Load from Matrix Market file
    template <class T> bool xcsr_scheme<T>::load_matrix_market(const std::string& filename)
    { /* TODO: parse .mtx and build CSR */ return false; }
    // Save to Matrix Market file (coordinate format)
    template <class T> bool xcsr_scheme<T>::save_matrix_market(const std::string& filename) const
    { /* TODO: write coordinate format */ return false; }

    // Block CSR constructor
    template <class T> xbcsr_scheme<T>::xbcsr_scheme(size_type n_block_rows, size_type n_block_cols, size_type block_size)
        : m_block_rows(n_block_rows), m_block_cols(n_block_cols), m_block_size(block_size), m_nnz_blocks(0)
    { m_row_ptr.resize(n_block_rows + 1, 0); }
    template <class T> auto xbcsr_scheme<T>::nblock_rows() const noexcept -> size_type { return m_block_rows; }
    template <class T> auto xbcsr_scheme<T>::nblock_cols() const noexcept -> size_type { return m_block_cols; }
    template <class T> auto xbcsr_scheme<T>::block_size() const noexcept -> size_type { return m_block_size; }
    template <class T> auto xbcsr_scheme<T>::nnz_blocks() const noexcept -> size_type { return m_nnz_blocks; }
    template <class T> void xbcsr_scheme<T>::add_block(size_type brow, size_type bcol, const block_type& block)
    { /* TODO: insert block into CSR structure */ }
    template <class T> xarray_container<T> xbcsr_scheme<T>::to_dense() const
    { /* TODO: assemble dense from blocks */ return {}; }
    template <class T> template <class E>
    xarray_container<T> xbcsr_scheme<T>::dot(const xexpression<E>& x) const
    { /* TODO: block SpMV */ return {}; }

    // Conjugate Gradient solver (with optional preconditioner)
    template <class T>
    xarray_container<T> cg_solve(const xcsr_scheme<T>& A, const xarray_container<T>& b,
                                  size_t max_iter, T tol, const xcsr_scheme<T>* preconditioner)
    { /* TODO: implement PCG */ return {}; }
    // GMRES solver (restarted)
    template <class T>
    xarray_container<T> gmres_solve(const xcsr_scheme<T>& A, const xarray_container<T>& b,
                                    size_t restart, size_t max_iter, T tol,
                                    const xcsr_scheme<T>* preconditioner)
    { /* TODO: implement restarted GMRES */ return {}; }
    // BiCGSTAB solver
    template <class T>
    xarray_container<T> bicgstab_solve(const xcsr_scheme<T>& A, const xarray_container<T>& b,
                                        size_t max_iter, T tol, const xcsr_scheme<T>* preconditioner)
    { /* TODO: implement BiCGSTAB */ return {}; }

    // Factory: CSR from dense
    template <class T> xcsr_scheme<T> csr_matrix(const xarray_container<T>& dense)
    { return xcsr_scheme<T>(dense); }
    // Factory: CSR from COO
    template <class T> xcsr_scheme<T> csr_matrix(const xcoo_scheme<T>& coo)
    { return xcsr_scheme<T>(coo); }
    // Factory: CSR from triplets
    template <class T> xcsr_scheme<T> csr_matrix(size_type n_rows, size_type n_cols,
                                                  const std::vector<size_type>& rows,
                                                  const std::vector<size_type>& cols,
                                                  const std::vector<T>& values,
                                                  bool sum_duplicates)
    { return xcsr_scheme<T>(n_rows, n_cols, rows, cols, values, sum_duplicates); }
    // Factory: identity matrix in CSR format
    template <class T> xcsr_scheme<T> csr_eye(size_type n)
    { xcsr_scheme<T> result(n, n); result.row_ptr().resize(n + 1); for (size_t i = 0; i < n; ++i) { result.row_ptr()[i] = i; result.col_ind().push_back(i); result.values().push_back(T(1)); } result.row_ptr()[n] = n; return result; }
    // Factory: diagonal matrix from vector
    template <class E> xcsr_scheme<typename E::value_type> csr_diag(const xexpression<E>& diag)
    { /* TODO: build diagonal CSR */ return {}; }
    // Factory: random sparse matrix
    template <class T> xcsr_scheme<T> csr_random(size_type n_rows, size_type n_cols, T density, std::mt19937* rng)
    { /* TODO: generate random non‑zeros */ return {}; }

    // Scalar multiplication from left
    template <class T> xcsr_scheme<T> operator*(const T& scalar, const xcsr_scheme<T>& mat)
    { return mat * scalar; }
    // Matrix-vector product
    template <class T> xarray_container<T> dot(const xcsr_scheme<T>& A, const xarray_container<T>& x)
    { return A.dot(x); }
    // Vector-matrix product (xᵀ * A)
    template <class T> xarray_container<T> dot(const xarray_container<T>& x, const xcsr_scheme<T>& A)
    { /* TODO: row vector times sparse matrix */ return {}; }
    // Sparse matrix multiplication
    template <class T> xcsr_scheme<T> matmul(const xcsr_scheme<T>& A, const xcsr_scheme<T>& B)
    { return A.matmul(B); }
    // Direct sparse solver (wrapper around dense or iterative)
    template <class T> xarray_container<T> spsolve(const xcsr_scheme<T>& A, const xarray_container<T>& b)
    { /* TODO: choose appropriate solver */ return {}; }
}

#endif // XTENSOR_XCSR_SCHEME_HPP    }

            if (value == T(0)) return;

            // Insert new entry (maintain sorted order)
            size_type insert_pos = m_row_ptr[row];
            while (insert_pos < m_row_ptr[row+1] && m_col_ind[insert_pos] < col)
                ++insert_pos;

            m_col_ind.insert(m_col_ind.begin() + insert_pos, col);
            m_values.insert(m_values.begin() + insert_pos, value);
            for (size_type i = row + 1; i <= m_rows; ++i)
                m_row_ptr[i]++;
            m_nnz++;
        }

        // --------------------------------------------------------------------
        // Forward substitution: solve L * x = b for lower triangular L
        // --------------------------------------------------------------------
        xarray_container<T> forward_substitution(const xarray_container<T>& b) const
        {
            if (b.dimension() != 1 || b.size() != m_rows)
                XTENSOR_THROW(std::invalid_argument, "forward_substitution: b size mismatch");

            xarray_container<T> x({m_rows}, T(0));
            for (size_type i = 0; i < m_rows; ++i)
            {
                T sum = b(i);
                for (size_type k = m_row_ptr[i]; k < m_row_ptr[i+1]; ++k)
                {
                    size_type j = m_col_ind[k];
                    if (j >= i) break; // only lower triangular part
                    sum = sum - detail::multiply(m_values[k], x(j));
                }
                // Diagonal element (assumed to be the first entry with col == i)
                T diag = T(1);
                for (size_type k = m_row_ptr[i]; k < m_row_ptr[i+1]; ++k)
                {
                    if (m_col_ind[k] == i)
                    {
                        diag = m_values[k];
                        break;
                    }
                }
                if (diag == T(0))
                    XTENSOR_THROW(std::runtime_error, "forward_substitution: zero diagonal");
                x(i) = sum / diag;
            }
            return x;
        }

        // --------------------------------------------------------------------
        // Back substitution: solve U * x = b for upper triangular U
        // --------------------------------------------------------------------
        xarray_container<T> back_substitution(const xarray_container<T>& b) const
        {
            if (b.dimension() != 1 || b.size() != m_rows)
                XTENSOR_THROW(std::invalid_argument, "back_substitution: b size mismatch");

            xarray_container<T> x({m_rows}, T(0));
            for (size_type i = m_rows; i-- > 0; )
            {
                T sum = b(i);
                for (size_type k = m_row_ptr[i]; k < m_row_ptr[i+1]; ++k)
                {
                    size_type j = m_col_ind[k];
                    if (j <= i) continue; // only upper triangular part
                    sum = sum - detail::multiply(m_values[k], x(j));
                }
                T diag = T(1);
                for (size_type k = m_row_ptr[i]; k < m_row_ptr[i+1]; ++k)
                {
                    if (m_col_ind[k] == i)
                    {
                        diag = m_values[k];
                        break;
                    }
                }
                if (diag == T(0))
                    XTENSOR_THROW(std::runtime_error, "back_substitution: zero diagonal");
                x(i) = sum / diag;
            }
            return x;
        }

        // --------------------------------------------------------------------
        // Incomplete LU factorization (ILU0) - no fill-in
        // --------------------------------------------------------------------
        std::pair<xcsr_scheme<T>, xcsr_scheme<T>> ilu0() const
        {
            if (m_rows != m_cols)
                XTENSOR_THROW(std::invalid_argument, "ilu0: matrix must be square");

            xcsr_scheme<T> L(m_rows, m_cols);
            xcsr_scheme<T> U(m_rows, m_cols);
            L.m_row_ptr.resize(m_rows + 1, 0);
            U.m_row_ptr.resize(m_rows + 1, 0);

            // Working copy of matrix values (dense for simplicity, could be sparse)
            std::vector<std::unordered_map<size_type, T>> row_vals(m_rows);
            for (size_type i = 0; i < m_rows; ++i)
            {
                for (size_type k = m_row_ptr[i]; k < m_row_ptr[i+1]; ++k)
                {
                    row_vals[i][m_col_ind[k]] = m_values[k];
                }
            }

            for (size_type i = 0; i < m_rows; ++i)
            {
                L.m_row_ptr[i] = L.m_col_ind.size();
                U.m_row_ptr[i] = U.m_col_ind.size();

                // Diagonal of L is 1
                L.m_col_ind.push_back(i);
                L.m_values.push_back(T(1));

                for (size_type k = m_row_ptr[i]; k < m_row_ptr[i+1]; ++k)
                {
                    size_type j = m_col_ind[k];
                    if (j < i)
                    {
                        // L entry
                        T sum = row_vals[i][j];
                        for (size_type p = 0; p < j; ++p)
                        {
                            auto itL = row_vals[i].find(p);
                            auto itU = row_vals[p].find(j);
                            if (itL != row_vals[i].end() && itU != row_vals[p].end())
                                sum = sum - detail::multiply(itL->second, itU->second);
                        }
                        if (row_vals[j].find(j) != row_vals[j].end())
                            sum = sum / row_vals[j][j];
                        if (sum != T(0))
                        {
                            L.m_col_ind.push_back(j);
                            L.m_values.push_back(sum);
                            row_vals[i][j] = sum;
                        }
                    }
                    else if (j >= i)
                    {
                        // U entry
                        T sum = row_vals[i][j];
                        for (size_type p = 0; p < i; ++p)
                        {
                            auto itL = row_vals[i].find(p);
                            auto itU = row_vals[p].find(j);
                            if (itL != row_vals[i].end() && itU != row_vals[p].end())
                                sum = sum - detail::multiply(itL->second, itU->second);
                        }
                        if (sum != T(0) || j == i)
                        {
                            U.m_col_ind.push_back(j);
                            U.m_values.push_back(sum);
                            row_vals[i][j] = sum;
                        }
                    }
                }
            }
            L.m_row_ptr[m_rows] = L.m_col_ind.size();
            U.m_row_ptr[m_rows] = U.m_col_ind.size();
            L.m_nnz = L.m_values.size();
            U.m_nnz = U.m_values.size();

            return {L, U};
        }

    private:
        size_type m_rows;
        size_type m_cols;
        size_type m_nnz;
        std::vector<size_type> m_row_ptr;   // size = m_rows + 1
        std::vector<size_type> m_col_ind;   // size = nnz
        std::vector<value_type> m_values;   // size = nnz

        namespace detail
        {
            template <class U>
            static inline U multiply(const U& a, const U& b)
            {
                if constexpr (std::is_same_v<U, bignumber::BigNumber>)
                {
                    if (config::use_fft_multiply)
                        return bignumber::fft_multiply(a, b);
                }
                return a * b;
            }

            template <class U>
            static inline U sqrt_val(const U& x)
            {
                if constexpr (std::is_same_v<U, bignumber::BigNumber>)
                    return bignumber::sqrt(x);
                else
                    return std::sqrt(x);
            }
        };
    };

    // ------------------------------------------------------------------------
    // Non-member operators and functions
    // ------------------------------------------------------------------------
    template <class T>
    xcsr_scheme<T> operator*(const T& scalar, const xcsr_scheme<T>& mat)
    {
        return mat * scalar;
    }

    template <class T>
    xarray_container<T> dot(const xcsr_scheme<T>& A, const xarray_container<T>& x)
    {
        return A.dot(x);
    }

    template <class T>
    xarray_container<T> dot(const xarray_container<T>& x, const xcsr_scheme<T>& A)
    {
        if (x.dimension() != 1 || x.size() != A.nrows())
            XTENSOR_THROW(std::invalid_argument, "dot(x, CSR): dimension mismatch");

        xarray_container<T> y({A.ncols()}, T(0));
        for (size_type i = 0; i < A.nrows(); ++i)
        {
            T xi = x(i);
            if (xi == T(0)) continue;
            for (size_type k = A.row_ptr()[i]; k < A.row_ptr()[i+1]; ++k)
            {
                size_type j = A.col_ind()[k];
                y(j) = y(j) + detail::multiply(xi, A.values()[k]);
            }
        }
        return y;
    }

    template <class T>
    xcsr_scheme<T> matmul(const xcsr_scheme<T>& A, const xcsr_scheme<T>& B)
    {
        return A.matmul(B);
    }

    // ------------------------------------------------------------------------
    // Factory functions
    // ------------------------------------------------------------------------
    template <class T>
    xcsr_scheme<T> csr_matrix(const xarray_container<T>& dense)
    {
        return xcsr_scheme<T>(dense);
    }

    template <class T>
    xcsr_scheme<T> csr_matrix(const xcoo_scheme<T>& coo)
    {
        return xcsr_scheme<T>(coo);
    }

    template <class T>
    xcsr_scheme<T> csr_matrix(size_type n_rows, size_type n_cols,
                              const std::vector<size_type>& rows,
                              const std::vector<size_type>& cols,
                              const std::vector<T>& values,
                              bool sum_duplicates = true)
    {
        return xcsr_scheme<T>(n_rows, n_cols, rows, cols, values, sum_duplicates);
    }

    template <class T>
    xcsr_scheme<T> csr_eye(size_type n)
    {
        xcsr_scheme<T> result(n, n);
        result.row_ptr().resize(n + 1);
        result.col_ind().reserve(n);
        result.values().reserve(n);
        for (size_type i = 0; i < n; ++i)
        {
            result.row_ptr()[i] = i;
            result.col_ind().push_back(i);
            result.values().push_back(T(1));
        }
        result.row_ptr()[n] = n;
        return result;
    }

    template <class E>
    xcsr_scheme<typename E::value_type> csr_diag(const xexpression<E>& diag_expr)
    {
        const auto& diag = diag_expr.derived_cast();
        if (diag.dimension() != 1)
            XTENSOR_THROW(std::invalid_argument, "csr_diag: input must be 1D");
        size_type n = diag.size();
        xcsr_scheme<typename E::value_type> result(n, n);
        result.row_ptr().resize(n + 1);
        result.col_ind().reserve(n);
        result.values().reserve(n);
        for (size_type i = 0; i < n; ++i)
        {
            result.row_ptr()[i] = i;
            T val = diag(i);
            if (val != T(0))
            {
                result.col_ind().push_back(i);
                result.values().push_back(val);
            }
        }
        result.row_ptr()[n] = result.col_ind().size();
        return result;
    }

    // ------------------------------------------------------------------------
    // Sparse solvers
    // ------------------------------------------------------------------------
    template <class T>
    xarray_container<T> spsolve_triangular(const xcsr_scheme<T>& A,
                                            const xarray_container<T>& b,
                                            bool lower = true,
                                            bool unit_diagonal = false)
    {
        if (lower)
            return A.forward_substitution(b);
        else
            return A.back_substitution(b);
    }

    template <class T>
    xarray_container<T> spsolve(const xcsr_scheme<T>& A,
                                 const xarray_container<T>& b)
    {
        // Use ILU preconditioned GMRES or direct solver depending on size
        // For simplicity, we convert to dense and solve
        auto denseA = A.to_dense();
        return linalg::solve(denseA, b);
    }

} // namespace xt

#endif // XTENSOR_XCSR_SCHEME_HPP