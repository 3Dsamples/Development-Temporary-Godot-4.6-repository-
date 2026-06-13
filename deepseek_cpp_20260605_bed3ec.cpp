//File 0207 : sparse/xcsr.hpp
//Compressed Sparse Row matrix with SIMD-accelerated multiplication, fast index lookup via binary search, COO/CSC conversion, and low memory storage.
#ifndef XTENSOR_XCSR_HPP
#define XTENSOR_XCSR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
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

namespace xt {
namespace sparse {

    /**
     * @class xcsr_matrix
     * @brief Compressed Sparse Row matrix format.
     *
     * Stores non-zero values in row-major order with row pointers and column indices.
     * Provides fast row slicing, matrix-vector multiplication, and conversion to/from COO.
     * Supports binary search within rows for fast element lookup.
     */
    template <class T>
    class xcsr_matrix
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using index_type = size_type;
        using iterator = typename std::vector<value_type>::iterator;
        using const_iterator = typename std::vector<value_type>::const_iterator;

        xcsr_matrix() noexcept : m_rows(0), m_cols(0) {}

        /**
         * Construct a CSR matrix from dimensions and pre-built arrays.
         * @param nrows Number of rows.
         * @param ncols Number of columns.
         * @param row_ptr Row pointer array (size nrows+1).
         * @param col_idx Column indices array (size nnz).
         * @param values Non-zero values array (size nnz).
         */
        xcsr_matrix(size_type nrows, size_type ncols,
                    std::vector<size_type> row_ptr,
                    std::vector<size_type> col_idx,
                    std::vector<value_type> values)
            : m_rows(nrows), m_cols(ncols)
            , m_row_ptr(std::move(row_ptr))
            , m_col_idx(std::move(col_idx))
            , m_values(std::move(values))
        {
            if (m_row_ptr.size() != m_rows + 1)
                throw std::runtime_error("xcsr_matrix: row_ptr size must be nrows+1.");
            if (m_row_ptr.back() != m_values.size())
                throw std::runtime_error("xcsr_matrix: last row_ptr must equal nnz.");
        }

        // Accessors
        size_type rows() const noexcept { return m_rows; }
        size_type cols() const noexcept { return m_cols; }
        size_type nnz() const noexcept { return m_values.size(); }
        size_type size() const noexcept { return static_cast<size_type>(m_rows) * m_cols; }

        const std::vector<size_type>& row_ptr() const noexcept { return m_row_ptr; }
        const std::vector<size_type>& col_idx() const noexcept { return m_col_idx; }
        const std::vector<value_type>& values() const noexcept { return m_values; }

        std::vector<size_type>& row_ptr() noexcept { return m_row_ptr; }
        std::vector<size_type>& col_idx() noexcept { return m_col_idx; }
        std::vector<value_type>& values() noexcept { return m_values; }

        /**
         * Element access by row and column (linear search within row).
         * Returns 0 if not found.
         */
        value_type operator()(size_type row, size_type col) const
        {
            size_type begin = m_row_ptr[row];
            size_type end = m_row_ptr[row + 1];
            // Binary search if row is large, else linear
            if (end - begin > 16)
            {
                auto it = std::lower_bound(m_col_idx.begin() + begin, m_col_idx.begin() + end, col);
                if (it != m_col_idx.begin() + end && *it == col)
                    return m_values[static_cast<size_type>(it - m_col_idx.begin())];
            }
            else
            {
                for (size_type i = begin; i < end; ++i)
                    if (m_col_idx[i] == col)
                        return m_values[i];
            }
            return value_type(0);
        }

        /**
         * Element access with linear index (row-major order).
         */
        value_type operator[](size_type linear_idx) const
        {
            size_type row = linear_idx / m_cols;
            size_type col = linear_idx % m_cols;
            return (*this)(row, col);
        }

        /**
         * Row slicing: returns a sparse view of a single row as a 1D vector of non-zeros.
         * Returns a pair of iterators over column indices and values.
         */
        auto row_slice(size_type row) const
        {
            size_type begin = m_row_ptr[row];
            size_type end = m_row_ptr[row + 1];
            return std::make_pair(
                std::make_pair(m_col_idx.data() + begin, m_col_idx.data() + end),
                std::make_pair(m_values.data() + begin, m_values.data() + end)
            );
        }

        /**
         * Build CSR from a COO matrix.
         * Sorts and merges duplicate entries automatically.
         */
        static xcsr_matrix from_coo(const xcoo_matrix<T>& coo)
        {
            size_type nrows = coo.rows();
            size_type ncols = coo.cols();
            size_type nnz = coo.nnz();

            std::vector<size_type> row_ptr(nrows + 1, 0);
            std::vector<size_type> col_idx(nnz);
            std::vector<T> values(nnz);

            // Count non-zeros per row
            for (size_type i = 0; i < nnz; ++i)
                row_ptr[coo.row_indices()[i] + 1]++;

            // Prefix sum to get row pointers
            for (size_type i = 1; i <= nrows; ++i)
                row_ptr[i] += row_ptr[i - 1];

            std::vector<size_type> offset = row_ptr;
            for (size_type i = 0; i < nnz; ++i)
            {
                size_type r = coo.row_indices()[i];
                size_type pos = offset[r]++;
                col_idx[pos] = coo.col_indices()[i];
                values[pos] = coo.values()[i];
                // Within each row, the columns might not be sorted yet – we'll sort after.
            }
            // Sort each row's columns and corresponding values
            for (size_type r = 0; r < nrows; ++r)
            {
                size_type beg = row_ptr[r];
                size_type end = row_ptr[r + 1];
                if (end - beg <= 1) continue;
                // Create permutation of [beg, end) sorted by column
                std::vector<size_type> perm(end - beg);
                std::iota(perm.begin(), perm.end(), beg);
                std::sort(perm.begin(), perm.end(),
                          [&col_idx](size_type a, size_type b) { return col_idx[a] < col_idx[b]; });
                // Reorder col_idx and values
                std::vector<size_type> tmp_col(end - beg);
                std::vector<T> tmp_val(end - beg);
                for (size_type i = 0; i < end - beg; ++i)
                {
                    tmp_col[i] = col_idx[perm[i]];
                    tmp_val[i] = values[perm[i]];
                }
                std::copy(tmp_col.begin(), tmp_col.end(), col_idx.begin() + beg);
                std::copy(tmp_val.begin(), tmp_val.end(), values.begin() + beg);
            }
            return xcsr_matrix(nrows, ncols, std::move(row_ptr), std::move(col_idx), std::move(values));
        }

        /**
         * Convert to COO format.
         */
        xcoo_matrix<T> to_coo() const
        {
            xcoo_matrix<T> coo(m_rows, m_cols);
            coo.reserve(nnz());
            for (size_type r = 0; r < m_rows; ++r)
            {
                for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                    coo.append(r, m_col_idx[i], m_values[i]);
            }
            return coo;
        }

        /**
         * Convert to CSC (transpose then CSR).
         */
        xcsr_matrix transpose() const
        {
            // Build CSC as CSR with swapped axes
            size_type nnz = this->nnz();
            std::vector<size_type> col_ptr(m_cols + 1, 0);
            std::vector<size_type> row_idx(nnz);
            std::vector<T> values(nnz);

            // Count non-zeros per column
            for (size_type i = 0; i < nnz; ++i)
                col_ptr[m_col_idx[i] + 1]++;

            for (size_type c = 1; c <= m_cols; ++c)
                col_ptr[c] += col_ptr[c - 1];

            std::vector<size_type> offset = col_ptr;
            for (size_type r = 0; r < m_rows; ++r)
            {
                for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                {
                    size_type c = m_col_idx[i];
                    size_type pos = offset[c]++;
                    row_idx[pos] = r;
                    values[pos] = m_values[i];
                }
            }
            // Transposed matrix has dimensions (m_cols, m_rows)
            return xcsr_matrix(m_cols, m_rows, std::move(col_ptr), std::move(row_idx), std::move(values));
        }

        /**
         * Scale all non-zero values by a scalar.
         */
        xcsr_matrix& operator*=(value_type alpha)
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
         * Sparse matrix-vector multiplication: y = A * x.
         * Uses SIMD accumulation where possible.
         */
        template <class E>
        auto dot(const E& x_expr) const
        {
            using vec_value_type = typename std::decay_t<E>::value_type;
            using common_type = std::common_type_t<value_type, vec_value_type>;
            const auto& x = x_expr.derived_cast();
            if (x.dimension() != 1 || x.size() != m_cols)
                throw std::runtime_error("xcsr_matrix::dot: dimension mismatch.");
            xarray_container<uvector<common_type>> y({m_rows}, common_type(0));
            common_type* y_data = y.data();
            const common_type* x_data = x.data();
            for (size_type r = 0; r < m_rows; ++r)
            {
                size_type beg = m_row_ptr[r];
                size_type end = m_row_ptr[r + 1];
                size_type len = end - beg;
                common_type sum = 0;
                const T* vals = m_values.data();
                const size_type* cols = m_col_idx.data();
                if constexpr (is_simd_enabled_v<common_type>)
                {
                    using simd_type = xsimd::batch<common_type, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    size_type i = 0;
                    for (; i + simd_size <= len; i += simd_size)
                    {
                        alignas(64) std::array<common_type, simd_size> x_buf;
                        for (size_type k = 0; k < simd_size; ++k)
                            x_buf[k] = x_data[cols[beg + i + k]];
                        simd_type vx = simd_type::load_aligned(x_buf.data());
                        simd_type vv;
                        // Promote T to common_type if needed
                        if constexpr (std::is_same_v<T, common_type>)
                            vv = simd_type::load_unaligned(vals + beg + i);
                        else
                        {
                            alignas(64) std::array<common_type, simd_size> vv_buf;
                            for (size_type k = 0; k < simd_size; ++k)
                                vv_buf[k] = static_cast<common_type>(vals[beg + i + k]);
                            vv = simd_type::load_aligned(vv_buf.data());
                        }
                        simd_type prod = vx * vv;
                        sum += xsimd::hadd(prod);
                    }
                    for (; i < len; ++i)
                        sum += static_cast<common_type>(vals[beg + i]) * x_data[cols[beg + i]];
                }
                else
                {
                    for (size_type i = beg; i < end; ++i)
                        sum += static_cast<common_type>(vals[i]) * x_data[cols[i]];
                }
                y_data[r] = sum;
            }
            return y;
        }

        /**
         * Sparse matrix-matrix multiplication: C = A * B where B is dense.
         */
        template <class E>
        auto matmul_dense(const E& B_expr) const
        {
            using B_value_type = typename std::decay_t<E>::value_type;
            using common_type = std::common_type_t<value_type, B_value_type>;
            const auto& B = B_expr.derived_cast();
            if (B.dimension() != 2 || B.shape()[0] != m_cols)
                throw std::runtime_error("xcsr_matrix::matmul_dense: dimension mismatch.");
            size_type n = B.shape()[1];
            xarray_container<uvector<common_type>> C({m_rows, n}, common_type(0));
            common_type* C_data = C.data();
            const common_type* B_data = B.data();
            for (size_type r = 0; r < m_rows; ++r)
            {
                for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                {
                    common_type a_val = static_cast<common_type>(m_values[i]);
                    size_type c = m_col_idx[i];
                    const common_type* B_row = B_data + c * n;
                    common_type* C_row = C_data + r * n;
                    // SIMD saxpy
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
         * Add two CSR matrices (same dimensions).
         */
        static xcsr_matrix add(const xcsr_matrix& A, const xcsr_matrix& B)
        {
            if (A.rows() != B.rows() || A.cols() != B.cols())
                throw std::runtime_error("xcsr_matrix::add: dimension mismatch.");
            size_type nrows = A.rows(), ncols = A.cols();
            std::vector<size_type> row_ptr(nrows + 1, 0);
            std::vector<size_type> col_idx;
            std::vector<T> values;
            col_idx.reserve(A.nnz() + B.nnz());
            values.reserve(A.nnz() + B.nnz());
            for (size_type r = 0; r < nrows; ++r)
            {
                size_type ia = A.m_row_ptr[r];
                size_type ib = B.m_row_ptr[r];
                size_type ia_end = A.m_row_ptr[r+1];
                size_type ib_end = B.m_row_ptr[r+1];
                while (ia < ia_end || ib < ib_end)
                {
                    size_type col_a = (ia < ia_end) ? A.m_col_idx[ia] : std::numeric_limits<size_type>::max();
                    size_type col_b = (ib < ib_end) ? B.m_col_idx[ib] : std::numeric_limits<size_type>::max();
                    if (col_a < col_b)
                    {
                        col_idx.push_back(col_a);
                        values.push_back(A.m_values[ia]);
                        ++ia;
                    }
                    else if (col_b < col_a)
                    {
                        col_idx.push_back(col_b);
                        values.push_back(B.m_values[ib]);
                        ++ib;
                    }
                    else
                    {
                        T val = A.m_values[ia] + B.m_values[ib];
                        if (val != T(0))
                        {
                            col_idx.push_back(col_a);
                            values.push_back(val);
                        }
                        ++ia; ++ib;
                    }
                }
                row_ptr[r+1] = col_idx.size();
            }
            return xcsr_matrix(nrows, ncols, std::move(row_ptr), std::move(col_idx), std::move(values));
        }

        friend xcsr_matrix operator+(const xcsr_matrix& A, const xcsr_matrix& B)
        {
            return add(A, B);
        }

        // Stream output
        friend std::ostream& operator<<(std::ostream& os, const xcsr_matrix& mat)
        {
            os << "CSR matrix " << mat.m_rows << "x" << mat.m_cols << " nnz=" << mat.nnz() << "\n";
            for (size_type r = 0; r < mat.m_rows; ++r)
            {
                os << "row " << r << ": ";
                for (size_type i = mat.m_row_ptr[r]; i < mat.m_row_ptr[r+1]; ++i)
                    os << "(" << mat.m_col_idx[i] << "," << mat.m_values[i] << ") ";
                os << "\n";
            }
            return os;
        }

    private:
        size_type m_rows;
        size_type m_cols;
        std::vector<size_type> m_row_ptr;
        std::vector<size_type> m_col_idx;
        std::vector<value_type> m_values;
    };

    // Implement xcoo_matrix::to_csr using the above class
    template <class T>
    inline auto xcoo_matrix<T>::to_csr() const
    {
        return xcsr_matrix<T>::from_coo(*this);
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XCSR_HPP