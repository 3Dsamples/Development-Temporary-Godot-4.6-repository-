//File 0208 : sparse/xcsc.hpp
//Compressed Sparse Column matrix with SIMD-accelerated column-wise operations, fast column slicing, COO/CSR conversion, and low memory storage.
#ifndef XTENSOR_XCSC_HPP
#define XTENSOR_XCSC_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xcsr.hpp"

namespace xt {
namespace sparse {

    /**
     * @class xcsc_matrix
     * @brief Compressed Sparse Column matrix format.
     *
     * Stores non-zero values in column-major order with column pointers and row indices.
     * Provides fast column slicing, matrix-vector multiplication, and conversion to/from CSR.
     * Supports binary search within columns for fast element lookup.
     */
    template <class T>
    class xcsc_matrix
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using index_type = size_type;

        xcsc_matrix() noexcept : m_rows(0), m_cols(0) {}

        /**
         * Construct a CSC matrix from dimensions and pre-built arrays.
         * @param nrows Number of rows.
         * @param ncols Number of columns.
         * @param col_ptr Column pointer array (size ncols+1).
         * @param row_idx Row indices array (size nnz).
         * @param values Non-zero values array (size nnz).
         */
        xcsc_matrix(size_type nrows, size_type ncols,
                    std::vector<size_type> col_ptr,
                    std::vector<size_type> row_idx,
                    std::vector<value_type> values)
            : m_rows(nrows), m_cols(ncols)
            , m_col_ptr(std::move(col_ptr))
            , m_row_idx(std::move(row_idx))
            , m_values(std::move(values))
        {
            if (m_col_ptr.size() != m_cols + 1)
                throw std::runtime_error("xcsc_matrix: col_ptr size must be ncols+1.");
            if (m_col_ptr.back() != m_values.size())
                throw std::runtime_error("xcsc_matrix: last col_ptr must equal nnz.");
        }

        // Accessors
        size_type rows() const noexcept { return m_rows; }
        size_type cols() const noexcept { return m_cols; }
        size_type nnz() const noexcept { return m_values.size(); }
        size_type size() const noexcept { return static_cast<size_type>(m_rows) * m_cols; }

        const std::vector<size_type>& col_ptr() const noexcept { return m_col_ptr; }
        const std::vector<size_type>& row_idx() const noexcept { return m_row_idx; }
        const std::vector<value_type>& values() const noexcept { return m_values; }

        std::vector<size_type>& col_ptr() noexcept { return m_col_ptr; }
        std::vector<size_type>& row_idx() noexcept { return m_row_idx; }
        std::vector<value_type>& values() noexcept { return m_values; }

        /**
         * Element access by row and column (binary search within column).
         */
        value_type operator()(size_type row, size_type col) const
        {
            size_type begin = m_col_ptr[col];
            size_type end = m_col_ptr[col + 1];
            if (end - begin > 16)
            {
                auto it = std::lower_bound(m_row_idx.begin() + begin, m_row_idx.begin() + end, row);
                if (it != m_row_idx.begin() + end && *it == row)
                    return m_values[static_cast<size_type>(it - m_row_idx.begin())];
            }
            else
            {
                for (size_type i = begin; i < end; ++i)
                    if (m_row_idx[i] == row)
                        return m_values[i];
            }
            return value_type(0);
        }

        /**
         * Column slicing: returns a pair of iterators over row indices and values for a given column.
         */
        auto col_slice(size_type col) const
        {
            size_type begin = m_col_ptr[col];
            size_type end = m_col_ptr[col + 1];
            return std::make_pair(
                std::make_pair(m_row_idx.data() + begin, m_row_idx.data() + end),
                std::make_pair(m_values.data() + begin, m_values.data() + end)
            );
        }

        /**
         * Build CSC from a COO matrix.
         */
        static xcsc_matrix from_coo(const xcoo_matrix<T>& coo)
        {
            // CSC is the transpose of CSR: build CSR for transposed data
            size_type nrows = coo.rows();
            size_type ncols = coo.cols();
            size_type nnz = coo.nnz();

            std::vector<size_type> col_ptr(ncols + 1, 0);
            std::vector<size_type> row_idx(nnz);
            std::vector<T> values(nnz);

            // Count non-zeros per column
            for (size_type i = 0; i < nnz; ++i)
                col_ptr[coo.col_indices()[i] + 1]++;

            for (size_type c = 1; c <= ncols; ++c)
                col_ptr[c] += col_ptr[c - 1];

            std::vector<size_type> offset = col_ptr;
            for (size_type i = 0; i < nnz; ++i)
            {
                size_type c = coo.col_indices()[i];
                size_type pos = offset[c]++;
                row_idx[pos] = coo.row_indices()[i];
                values[pos] = coo.values()[i];
            }
            // Sort each column by row index
            for (size_type c = 0; c < ncols; ++c)
            {
                size_type beg = col_ptr[c];
                size_type end = col_ptr[c + 1];
                if (end - beg <= 1) continue;
                std::vector<size_type> perm(end - beg);
                std::iota(perm.begin(), perm.end(), beg);
                std::sort(perm.begin(), perm.end(),
                          [&row_idx](size_type a, size_type b) { return row_idx[a] < row_idx[b]; });
                std::vector<size_type> tmp_row(end - beg);
                std::vector<T> tmp_val(end - beg);
                for (size_type i = 0; i < end - beg; ++i)
                {
                    tmp_row[i] = row_idx[perm[i]];
                    tmp_val[i] = values[perm[i]];
                }
                std::copy(tmp_row.begin(), tmp_row.end(), row_idx.begin() + beg);
                std::copy(tmp_val.begin(), tmp_val.end(), values.begin() + beg);
            }
            return xcsc_matrix(nrows, ncols, std::move(col_ptr), std::move(row_idx), std::move(values));
        }

        /**
         * Build CSC from a CSR matrix (transpose).
         */
        static xcsc_matrix from_csr(const xcsr_matrix<T>& csr)
        {
            size_type nrows = csr.rows();
            size_type ncols = csr.cols();
            size_type nnz = csr.nnz();
            std::vector<size_type> col_ptr(ncols + 1, 0);
            std::vector<size_type> row_idx(nnz);
            std::vector<T> values(nnz);

            // Count non-zeros per column
            for (size_type r = 0; r < nrows; ++r)
                for (size_type i = csr.row_ptr()[r]; i < csr.row_ptr()[r + 1]; ++i)
                    col_ptr[csr.col_idx()[i] + 1]++;

            for (size_type c = 1; c <= ncols; ++c)
                col_ptr[c] += col_ptr[c - 1];

            std::vector<size_type> offset = col_ptr;
            for (size_type r = 0; r < nrows; ++r)
            {
                for (size_type i = csr.row_ptr()[r]; i < csr.row_ptr()[r + 1]; ++i)
                {
                    size_type c = csr.col_idx()[i];
                    size_type pos = offset[c]++;
                    row_idx[pos] = r;
                    values[pos] = csr.values()[i];
                }
            }
            // Already sorted because we process rows in order
            return xcsc_matrix(nrows, ncols, std::move(col_ptr), std::move(row_idx), std::move(values));
        }

        /**
         * Convert to CSR (transpose).
         */
        xcsr_matrix<T> to_csr() const
        {
            return xcsr_matrix<T>::from_coo(this->to_coo()); // or directly transpose
        }

        /**
         * Convert to COO.
         */
        xcoo_matrix<T> to_coo() const
        {
            xcoo_matrix<T> coo(m_rows, m_cols);
            coo.reserve(nnz());
            for (size_type c = 0; c < m_cols; ++c)
                for (size_type i = m_col_ptr[c]; i < m_col_ptr[c + 1]; ++i)
                    coo.append(m_row_idx[i], c, m_values[i]);
            return coo;
        }

        /**
         * Transpose: CSC to CSR (just swap dimensions).
         */
        xcsr_matrix<T> transpose() const
        {
            return xcsr_matrix<T>(m_cols, m_rows, m_col_ptr, m_row_idx, m_values);
        }

        /**
         * Scale all non-zero values by a scalar.
         */
        xcsc_matrix& operator*=(value_type alpha)
        {
            if constexpr (is_simd_enabled_v<value_type>)
            {
                using simd_type = xsimd::batch<value_type, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                size_type n = m_values.size();
                size_type vec_count = n / simd_size;
                simd_type valpha(alpha);
                for (size_type i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(m_values.data() + i * simd_size);
                    v = v * valpha;
                    v.store_unaligned(m_values.data() + i * simd_size);
                }
                for (size_type i = vec_count * simd_size; i < n; ++i)
                    m_values[i] *= alpha;
            }
            else
            {
                for (auto& v : m_values) v *= alpha;
            }
            return *this;
        }

        /**
         * Sparse matrix-vector multiplication: y = A * x (A is CSC).
         * Optimized for column-wise access: y = sum_j x_j * A_col_j.
         */
        template <class E>
        auto dot(const E& x_expr) const
        {
            using vec_value_type = typename std::decay_t<E>::value_type;
            using common_type = std::common_type_t<value_type, vec_value_type>;
            const auto& x = x_expr.derived_cast();
            if (x.dimension() != 1 || x.size() != m_cols)
                throw std::runtime_error("xcsc_matrix::dot: dimension mismatch.");
            xarray_container<uvector<common_type>> y({m_rows}, common_type(0));
            common_type* y_data = y.data();
            const common_type* x_data = x.data();
            for (size_type c = 0; c < m_cols; ++c)
            {
                common_type xc = x_data[c];
                if (xc == common_type(0)) continue;
                for (size_type i = m_col_ptr[c]; i < m_col_ptr[c + 1]; ++i)
                    y_data[m_row_idx[i]] += static_cast<common_type>(m_values[i]) * xc;
            }
            return y;
        }

        /**
         * Sparse matrix-dense matrix multiplication: C = A * B, where A is CSC and B is dense.
         */
        template <class E>
        auto matmul_dense(const E& B_expr) const
        {
            using B_value_type = typename std::decay_t<E>::value_type;
            using common_type = std::common_type_t<value_type, B_value_type>;
            const auto& B = B_expr.derived_cast();
            if (B.dimension() != 2 || B.shape()[0] != m_cols)
                throw std::runtime_error("xcsc_matrix::matmul_dense: dimension mismatch.");
            size_type n = B.shape()[1];
            xarray_container<uvector<common_type>> C({m_rows, n}, common_type(0));
            common_type* C_data = C.data();
            const common_type* B_data = B.data();
            for (size_type c = 0; c < m_cols; ++c)
            {
                const common_type* B_row = B_data + c * n;
                for (size_type i = m_col_ptr[c]; i < m_col_ptr[c + 1]; ++i)
                {
                    common_type a_val = static_cast<common_type>(m_values[i]);
                    size_type r = m_row_idx[i];
                    common_type* C_row = C_data + r * n;
                    size_type j = 0;
                    if constexpr (is_simd_enabled_v<common_type>)
                    {
                        using simd_type = xsimd::batch<common_type, default_simd_arch>;
                        constexpr std::size_t simd_size = simd_type::size;
                        simd_type va(a_val);
                        for (; j + simd_size <= n; j += simd_size)
                        {
                            simd_type vB = simd_type::load_unaligned(B_row + j);
                            simd_type vC = simd_type::load_unaligned(C_row + j);
                            vC = vC + va * vB;
                            vC.store_unaligned(C_row + j);
                        }
                    }
                    for (; j < n; ++j)
                        C_row[j] += a_val * B_row[j];
                }
            }
            return C;
        }

        /**
         * Add two CSC matrices (same dimensions).
         */
        static xcsc_matrix add(const xcsc_matrix& A, const xcsc_matrix& B)
        {
            if (A.rows() != B.rows() || A.cols() != B.cols())
                throw std::runtime_error("xcsc_matrix::add: dimension mismatch.");
            size_type nrows = A.rows(), ncols = A.cols();
            std::vector<size_type> col_ptr(ncols + 1, 0);
            std::vector<size_type> row_idx;
            std::vector<T> values;
            row_idx.reserve(A.nnz() + B.nnz());
            values.reserve(A.nnz() + B.nnz());
            for (size_type c = 0; c < ncols; ++c)
            {
                size_type ia = A.m_col_ptr[c];
                size_type ib = B.m_col_ptr[c];
                size_type ia_end = A.m_col_ptr[c+1];
                size_type ib_end = B.m_col_ptr[c+1];
                while (ia < ia_end || ib < ib_end)
                {
                    size_type row_a = (ia < ia_end) ? A.m_row_idx[ia] : std::numeric_limits<size_type>::max();
                    size_type row_b = (ib < ib_end) ? B.m_row_idx[ib] : std::numeric_limits<size_type>::max();
                    if (row_a < row_b)
                    {
                        row_idx.push_back(row_a);
                        values.push_back(A.m_values[ia]);
                        ++ia;
                    }
                    else if (row_b < row_a)
                    {
                        row_idx.push_back(row_b);
                        values.push_back(B.m_values[ib]);
                        ++ib;
                    }
                    else
                    {
                        T val = A.m_values[ia] + B.m_values[ib];
                        if (val != T(0))
                        {
                            row_idx.push_back(row_a);
                            values.push_back(val);
                        }
                        ++ia; ++ib;
                    }
                }
                col_ptr[c+1] = row_idx.size();
            }
            return xcsc_matrix(nrows, ncols, std::move(col_ptr), std::move(row_idx), std::move(values));
        }

        friend xcsc_matrix operator+(const xcsc_matrix& A, const xcsc_matrix& B)
        {
            return add(A, B);
        }

        friend std::ostream& operator<<(std::ostream& os, const xcsc_matrix& mat)
        {
            os << "CSC matrix " << mat.m_rows << "x" << mat.m_cols << " nnz=" << mat.nnz() << "\n";
            for (size_type c = 0; c < mat.m_cols; ++c)
            {
                os << "col " << c << ": ";
                for (size_type i = mat.m_col_ptr[c]; i < mat.m_col_ptr[c+1]; ++i)
                    os << "(" << mat.m_row_idx[i] << "," << mat.m_values[i] << ") ";
                os << "\n";
            }
            return os;
        }

    private:
        size_type m_rows;
        size_type m_cols;
        std::vector<size_type> m_col_ptr;
        std::vector<size_type> m_row_idx;
        std::vector<value_type> m_values;
    };

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XCSC_HPP