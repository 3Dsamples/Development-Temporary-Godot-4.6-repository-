// core/xcoo_scheme.hpp
#ifndef XTENSOR_XCOO_SCHEME_HPP
#define XTENSOR_XCOO_SCHEME_HPP

// ----------------------------------------------------------------------------
// xcoo_scheme.hpp – Coordinate list (COO) sparse matrix format
// ----------------------------------------------------------------------------
// This header provides the COO sparse matrix representation, including:
//   - Construction from dense, triplets, or other sparse formats
//   - Conversion to CSR, dense, or other formats
//   - Matrix-vector multiplication (SpMV) and matrix-matrix operations
//   - Element access, slicing, and iteration over non‑zero entries
//   - Arithmetic operations with dense and sparse matrices
//   - Sorting and duplicate entry summation
//   - Block COO (BCOO) support for batched operations
//   - I/O for Matrix Market (.mtx) format
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
#include <tuple>
#include <unordered_map>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ========================================================================
    // COO Sparse Matrix
    // ========================================================================
    template <class T>
    class xcoo_scheme
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using index_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;

        struct entry
        {
            size_type row;
            size_type col;
            value_type value;
            entry() : row(0), col(0), value(T(0)) {}
            entry(size_type r, size_type c, const value_type& v) : row(r), col(c), value(v) {}
        };

        // --------------------------------------------------------------------
        // Constructors
        // --------------------------------------------------------------------
        xcoo_scheme() : m_rows(0), m_cols(0), m_nnz(0) {}
        xcoo_scheme(size_type n_rows, size_type n_cols);
        xcoo_scheme(const shape_type& shape);
        explicit xcoo_scheme(const xarray_container<T>& dense);
        xcoo_scheme(size_type n_rows, size_type n_cols,
                    const std::vector<size_type>& rows,
                    const std::vector<size_type>& cols,
                    const std::vector<value_type>& values);

        // --------------------------------------------------------------------
        // Basic properties
        // --------------------------------------------------------------------
        size_type nrows() const noexcept;
        size_type ncols() const noexcept;
        size_type nnz() const noexcept;
        shape_type shape() const;
        bool empty() const noexcept;

        const std::vector<entry>& entries() const;
        std::vector<entry>& entries();

        // --------------------------------------------------------------------
        // Reserve and add entries
        // --------------------------------------------------------------------
        void reserve(size_type capacity);
        void add_entry(size_type row, size_type col, const value_type& value);

        // --------------------------------------------------------------------
        // Duplicate handling
        // --------------------------------------------------------------------
        void sum_duplicates();
        void eliminate_zeros();

        // --------------------------------------------------------------------
        // Conversions
        // --------------------------------------------------------------------
        xarray_container<T> to_dense() const;
        // Convert to CSR (requires forward declaration)
        // xcsr_scheme<T> to_csr() const;

        // --------------------------------------------------------------------
        // Matrix-vector multiplication
        // --------------------------------------------------------------------
        template <class E> xarray_container<T> dot(const xexpression<E>& x) const;

        // --------------------------------------------------------------------
        // Matrix-matrix operations
        // --------------------------------------------------------------------
        xcoo_scheme<T> transpose() const;
        value_type operator()(size_type row, size_type col) const;
        bool contains(size_type row, size_type col) const;
        xarray_container<T> diagonal(size_type k = 0) const;
        value_type trace() const;

        // --------------------------------------------------------------------
        // Arithmetic operators
        // --------------------------------------------------------------------
        xcoo_scheme<T>& operator*=(const value_type& scalar);
        xcoo_scheme<T> operator*(const value_type& scalar) const;
        xcoo_scheme<T> operator+(const xcoo_scheme<T>& other) const;
        xcoo_scheme<T> operator-(const xcoo_scheme<T>& other) const;
        xcoo_scheme<T> matmul(const xcoo_scheme<T>& other) const;
        xcoo_scheme<T> hadamard(const xarray_container<T>& dense) const;

        // --------------------------------------------------------------------
        // Slicing and manipulation
        // --------------------------------------------------------------------
        xcoo_scheme<T> slice(const std::vector<size_type>& rows,
                             const std::vector<size_type>& cols) const;
        void clear();
        void resize(size_type new_rows, size_type new_cols);

        // --------------------------------------------------------------------
        // Norms
        // --------------------------------------------------------------------
        value_type norm() const;  // Frobenius

        // --------------------------------------------------------------------
        // I/O
        // --------------------------------------------------------------------
        bool load_matrix_market(const std::string& filename);
        bool save_matrix_market(const std::string& filename) const;

    private:
        size_type m_rows;
        size_type m_cols;
        size_type m_nnz;
        std::vector<entry> m_entries;
    };

    // ------------------------------------------------------------------------
    // Block COO (BCOO) – dense blocks stored in COO format
    // ------------------------------------------------------------------------
    template <class T>
    class xbcoo_scheme
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using block_type = xarray_container<T>;

        struct block_entry
        {
            size_type block_row;
            size_type block_col;
            block_type values;
        };

        xbcoo_scheme() : m_block_rows(0), m_block_cols(0), m_block_size(1) {}
        xbcoo_scheme(size_type n_block_rows, size_type n_block_cols, size_type block_size);

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
        std::vector<block_entry> m_blocks;
    };

    // ------------------------------------------------------------------------
    // Factory functions
    // ------------------------------------------------------------------------
    template <class T> xcoo_scheme<T> coo_matrix(const xarray_container<T>& dense);
    template <class T> xcoo_scheme<T> coo_matrix(size_type n_rows, size_type n_cols,
                                                  const std::vector<size_type>& rows,
                                                  const std::vector<size_type>& cols,
                                                  const std::vector<T>& values);
    template <class T> xcoo_scheme<T> coo_eye(size_type n);
    template <class E> xcoo_scheme<typename E::value_type> coo_diag(const xexpression<E>& diag);
    template <class T> xcoo_scheme<T> coo_random(size_type n_rows, size_type n_cols, T density, std::mt19937* rng = nullptr);

    // ------------------------------------------------------------------------
    // Non-member operators
    // ------------------------------------------------------------------------
    template <class T> xcoo_scheme<T> operator*(const T& scalar, const xcoo_scheme<T>& mat);
    template <class T> xarray_container<T> dot(const xcoo_scheme<T>& A, const xarray_container<T>& x);
    template <class T> xarray_container<T> dot(const xarray_container<T>& x, const xcoo_scheme<T>& A);
    template <class T> xcoo_scheme<T> matmul(const xcoo_scheme<T>& A, const xcoo_scheme<T>& B);
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    // Construct COO matrix with given dimensions
    template <class T> xcoo_scheme<T>::xcoo_scheme(size_type n_rows, size_type n_cols)
        : m_rows(n_rows), m_cols(n_cols), m_nnz(0) {}

    // Construct COO matrix from shape vector
    template <class T> xcoo_scheme<T>::xcoo_scheme(const shape_type& shape)
        : m_rows(shape[0]), m_cols(shape[1]), m_nnz(0) {}

    // Construct COO matrix from dense array
    template <class T> xcoo_scheme<T>::xcoo_scheme(const xarray_container<T>& dense)
    { /* TODO: extract non‑zeros */ }

    // Construct COO matrix from triplet lists
    template <class T> xcoo_scheme<T>::xcoo_scheme(size_type n_rows, size_type n_cols,
                                                   const std::vector<size_type>& rows,
                                                   const std::vector<size_type>& cols,
                                                   const std::vector<value_type>& values)
    { /* TODO: validate and build entries */ }

    // Return number of rows
    template <class T> auto xcoo_scheme<T>::nrows() const noexcept -> size_type { return m_rows; }
    // Return number of columns
    template <class T> auto xcoo_scheme<T>::ncols() const noexcept -> size_type { return m_cols; }
    // Return number of non‑zero entries
    template <class T> auto xcoo_scheme<T>::nnz() const noexcept -> size_type { return m_nnz; }
    // Return shape as vector
    template <class T> auto xcoo_scheme<T>::shape() const -> shape_type { return {m_rows, m_cols}; }
    // Check if matrix is empty
    template <class T> bool xcoo_scheme<T>::empty() const noexcept { return m_nnz == 0; }

    // Const access to entries
    template <class T> const std::vector<typename xcoo_scheme<T>::entry>& xcoo_scheme<T>::entries() const { return m_entries; }
    // Mutable access to entries
    template <class T> std::vector<typename xcoo_scheme<T>::entry>& xcoo_scheme<T>::entries() { return m_entries; }

    // Reserve memory for entries
    template <class T> void xcoo_scheme<T>::reserve(size_type capacity) { m_entries.reserve(capacity); }
    // Add a single entry
    template <class T> void xcoo_scheme<T>::add_entry(size_type row, size_type col, const value_type& value)
    { /* TODO: add if value != 0 */ }

    // Sum entries with identical row and column indices
    template <class T> void xcoo_scheme<T>::sum_duplicates()
    { /* TODO: sort and merge */ }
    // Remove entries with zero value
    template <class T> void xcoo_scheme<T>::eliminate_zeros()
    { /* TODO: filter zeros */ }

    // Convert to dense matrix
    template <class T> xarray_container<T> xcoo_scheme<T>::to_dense() const
    { /* TODO: build dense array */ return {}; }

    // Matrix-vector multiplication
    template <class T> template <class E>
    xarray_container<T> xcoo_scheme<T>::dot(const xexpression<E>& x) const
    { /* TODO: SpMV */ return {}; }

    // Transpose matrix
    template <class T> xcoo_scheme<T> xcoo_scheme<T>::transpose() const
    { /* TODO: swap rows and cols */ return {}; }
    // Access element by row and column
    template <class T> auto xcoo_scheme<T>::operator()(size_type row, size_type col) const -> value_type
    { /* TODO: linear scan or map lookup */ return T(0); }
    // Check if entry exists
    template <class T> bool xcoo_scheme<T>::contains(size_type row, size_type col) const
    { /* TODO: check presence */ return false; }
    // Extract diagonal
    template <class T> xarray_container<T> xcoo_scheme<T>::diagonal(size_type k) const
    { /* TODO: collect entries where col == row + k */ return {}; }
    // Compute trace
    template <class T> auto xcoo_scheme<T>::trace() const -> value_type
    { /* TODO: sum diagonal entries */ return T(0); }

    // In‑place scalar multiplication
    template <class T> xcoo_scheme<T>& xcoo_scheme<T>::operator*=(const value_type& scalar)
    { /* TODO: scale all values */ return *this; }
    // Scalar multiplication (copy)
    template <class T> xcoo_scheme<T> xcoo_scheme<T>::operator*(const value_type& scalar) const
    { xcoo_scheme<T> result = *this; result *= scalar; return result; }
    // Matrix addition
    template <class T> xcoo_scheme<T> xcoo_scheme<T>::operator+(const xcoo_scheme<T>& other) const
    { /* TODO: merge and sum duplicates */ return {}; }
    // Matrix subtraction
    template <class T> xcoo_scheme<T> xcoo_scheme<T>::operator-(const xcoo_scheme<T>& other) const
    { /* TODO: merge with negated values */ return {}; }
    // Matrix multiplication (COO * COO)
    template <class T> xcoo_scheme<T> xcoo_scheme<T>::matmul(const xcoo_scheme<T>& other) const
    { /* TODO: sparse‑sparse multiplication */ return {}; }
    // Element‑wise multiplication with dense matrix
    template <class T> xcoo_scheme<T> xcoo_scheme<T>::hadamard(const xarray_container<T>& dense) const
    { /* TODO: multiply overlapping entries */ return {}; }

    // Extract submatrix
    template <class T> xcoo_scheme<T> xcoo_scheme<T>::slice(const std::vector<size_type>& rows,
                                                            const std::vector<size_type>& cols) const
    { /* TODO: filter entries */ return {}; }
    // Clear all entries
    template <class T> void xcoo_scheme<T>::clear() { m_entries.clear(); m_nnz = 0; }
    // Resize matrix, dropping out‑of‑bounds entries
    template <class T> void xcoo_scheme<T>::resize(size_type new_rows, size_type new_cols)
    { /* TODO: filter entries */ }

    // Compute Frobenius norm
    template <class T> auto xcoo_scheme<T>::norm() const -> value_type
    { /* TODO: sqrt(sum(value^2)) */ return T(0); }

    // Load from Matrix Market file
    template <class T> bool xcoo_scheme<T>::load_matrix_market(const std::string& filename)
    { /* TODO: parse .mtx file */ return false; }
    // Save to Matrix Market file
    template <class T> bool xcoo_scheme<T>::save_matrix_market(const std::string& filename) const
    { /* TODO: write coordinate format */ return false; }

    // Block COO constructor
    template <class T> xbcoo_scheme<T>::xbcoo_scheme(size_type n_block_rows, size_type n_block_cols, size_type block_size)
        : m_block_rows(n_block_rows), m_block_cols(n_block_cols), m_block_size(block_size) {}
    template <class T> auto xbcoo_scheme<T>::nblock_rows() const noexcept -> size_type { return m_block_rows; }
    template <class T> auto xbcoo_scheme<T>::nblock_cols() const noexcept -> size_type { return m_block_cols; }
    template <class T> auto xbcoo_scheme<T>::block_size() const noexcept -> size_type { return m_block_size; }
    template <class T> auto xbcoo_scheme<T>::nnz_blocks() const noexcept -> size_type { return m_blocks.size(); }
    template <class T> void xbcoo_scheme<T>::add_block(size_type brow, size_type bcol, const block_type& block)
    { /* TODO: store block */ }
    template <class T> xarray_container<T> xbcoo_scheme<T>::to_dense() const
    { /* TODO: assemble dense matrix */ return {}; }
    template <class T> template <class E>
    xarray_container<T> xbcoo_scheme<T>::dot(const xexpression<E>& x) const
    { /* TODO: block SpMV */ return {}; }

    // Factory: COO from dense
    template <class T> xcoo_scheme<T> coo_matrix(const xarray_container<T>& dense)
    { return xcoo_scheme<T>(dense); }
    // Factory: COO from triplets
    template <class T> xcoo_scheme<T> coo_matrix(size_type n_rows, size_type n_cols,
                                                  const std::vector<size_type>& rows,
                                                  const std::vector<size_type>& cols,
                                                  const std::vector<T>& values)
    { return xcoo_scheme<T>(n_rows, n_cols, rows, cols, values); }
    // Factory: identity matrix in COO format
    template <class T> xcoo_scheme<T> coo_eye(size_type n)
    { xcoo_scheme<T> result(n, n); for (size_t i = 0; i < n; ++i) result.add_entry(i, i, T(1)); return result; }
    // Factory: diagonal matrix from vector
    template <class E> xcoo_scheme<typename E::value_type> coo_diag(const xexpression<E>& diag)
    { /* TODO: build diagonal COO */ return {}; }
    // Factory: random sparse matrix
    template <class T> xcoo_scheme<T> coo_random(size_type n_rows, size_type n_cols, T density, std::mt19937* rng)
    { /* TODO: generate random entries */ return {}; }

    // Scalar multiplication from left
    template <class T> xcoo_scheme<T> operator*(const T& scalar, const xcoo_scheme<T>& mat)
    { return mat * scalar; }
    // Matrix-vector product
    template <class T> xarray_container<T> dot(const xcoo_scheme<T>& A, const xarray_container<T>& x)
    { return A.dot(x); }
    // Vector-matrix product (xᵀ * A)
    template <class T> xarray_container<T> dot(const xarray_container<T>& x, const xcoo_scheme<T>& A)
    { /* TODO: row vector times sparse matrix */ return {}; }
    // Sparse matrix multiplication
    template <class T> xcoo_scheme<T> matmul(const xcoo_scheme<T>& A, const xcoo_scheme<T>& B)
    { return A.matmul(B); }
}

#endif // XTENSOR_XCOO_SCHEME_HPPBigNumber>)
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
    // Non-member operators
    // ------------------------------------------------------------------------
    template <class T>
    xcoo_scheme<T> operator*(const T& scalar, const xcoo_scheme<T>& mat)
    {
        return mat * scalar;
    }

    template <class T>
    xarray_container<T> dot(const xcoo_scheme<T>& A, const xarray_container<T>& x)
    {
        return A.dot(x);
    }

    template <class T>
    xarray_container<T> dot(const xarray_container<T>& x, const xcoo_scheme<T>& A)
    {
        // xᵀ * A: result is 1 x A.cols
        if (x.dimension() != 1 || x.size() != A.nrows())
            XTENSOR_THROW(std::invalid_argument, "dot(x, COO): dimension mismatch");

        xarray_container<T> y({A.ncols()}, T(0));
        for (const auto& e : A.entries())
            y(e.col) = y(e.col) + detail::multiply(x(e.row), e.value);
        return y;
    }

    template <class T>
    xcoo_scheme<T> matmul(const xcoo_scheme<T>& A, const xcoo_scheme<T>& B)
    {
        return A.matmul(B);
    }

    // ------------------------------------------------------------------------
    // Factory function: create COO from dense
    // ------------------------------------------------------------------------
    template <class T>
    xcoo_scheme<T> coo_matrix(const xarray_container<T>& dense)
    {
        return xcoo_scheme<T>(dense);
    }

    template <class T>
    xcoo_scheme<T> coo_matrix(size_type n_rows, size_type n_cols,
                              const std::vector<size_type>& rows,
                              const std::vector<size_type>& cols,
                              const std::vector<T>& values)
    {
        return xcoo_scheme<T>(n_rows, n_cols, rows, cols, values);
    }

    // ------------------------------------------------------------------------
    // Identity matrix in COO format
    // ------------------------------------------------------------------------
    template <class T>
    xcoo_scheme<T> coo_eye(size_type n)
    {
        xcoo_scheme<T> result(n, n);
        result.reserve(n);
        for (size_type i = 0; i < n; ++i)
            result.add_entry(i, i, T(1));
        result.sum_duplicates();
        return result;
    }

    // ------------------------------------------------------------------------
    // Diagonal matrix from vector
    // ------------------------------------------------------------------------
    template <class E>
    xcoo_scheme<typename E::value_type> coo_diag(const xexpression<E>& diag_expr)
    {
        const auto& diag = diag_expr.derived_cast();
        if (diag.dimension() != 1)
            XTENSOR_THROW(std::invalid_argument, "coo_diag: input must be 1D");
        size_type n = diag.size();
        xcoo_scheme<typename E::value_type> result(n, n);
        result.reserve(n);
        for (size_type i = 0; i < n; ++i)
        {
            T val = diag(i);
            if (val != T(0))
                result.add_entry(i, i, val);
        }
        return result;
    }

} // namespace xt

#endif // XTENSOR_XCOO_SCHEME_HPP