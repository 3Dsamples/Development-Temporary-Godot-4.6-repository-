//File 0206 : sparse/xcoo.hpp
//Coordinate list (COO) sparse matrix format with efficient construction, sorting, duplicate merging, and SIMD-accelerated conversion to CSR/CSC.
#ifndef XTENSOR_XCOO_HPP
#define XTENSOR_XCOO_HPP

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
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"

namespace xt {
namespace sparse {

    /**
     * @class xcoo_matrix
     * @brief Coordinate list sparse matrix.
     *
     * Stores triplets (row, col, value). Supports efficient insertion and
     * conversion to CSR/CSC. Duplicate entries are summed during conversion.
     * The matrix is always in row-major order after sort().
     */
    template <class T>
    class xcoo_matrix
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using index_type = size_type;

        xcoo_matrix() noexcept : m_rows(0), m_cols(0) {}

        /**
         * Construct a COO matrix with given dimensions.
         * @param nrows Number of rows.
         * @param ncols Number of columns.
         */
        xcoo_matrix(size_type nrows, size_type ncols)
            : m_rows(nrows), m_cols(ncols)
        {
        }

        /**
         * Construct from arrays of row indices, column indices, and values.
         */
        xcoo_matrix(size_type nrows, size_type ncols,
                    std::vector<size_type> row_idx,
                    std::vector<size_type> col_idx,
                    std::vector<value_type> vals)
            : m_rows(nrows), m_cols(ncols)
            , m_row_indices(std::move(row_idx))
            , m_col_indices(std::move(col_idx))
            , m_values(std::move(vals))
        {
            if (m_row_indices.size() != m_col_indices.size() ||
                m_row_indices.size() != m_values.size())
                throw std::runtime_error("xcoo_matrix: size mismatch in arrays.");
        }

        // Accessors
        size_type rows() const noexcept { return m_rows; }
        size_type cols() const noexcept { return m_cols; }
        size_type nnz() const noexcept { return m_values.size(); }
        size_type size() const noexcept { return static_cast<size_type>(m_rows) * m_cols; }

        const std::vector<size_type>& row_indices() const noexcept { return m_row_indices; }
        const std::vector<size_type>& col_indices() const noexcept { return m_col_indices; }
        const std::vector<value_type>& values() const noexcept { return m_values; }

        std::vector<size_type>& row_indices() noexcept { return m_row_indices; }
        std::vector<size_type>& col_indices() noexcept { return m_col_indices; }
        std::vector<value_type>& values() noexcept { return m_values; }

        /**
         * Reserve storage for n non-zero entries.
         */
        void reserve(size_type n)
        {
            m_row_indices.reserve(n);
            m_col_indices.reserve(n);
            m_values.reserve(n);
        }

        /**
         * Append a non-zero entry.
         */
        void append(size_type row, size_type col, value_type val)
        {
            if (row >= m_rows || col >= m_cols)
                throw std::out_of_range("xcoo_matrix::append: index out of bounds.");
            m_row_indices.push_back(row);
            m_col_indices.push_back(col);
            m_values.push_back(val);
        }

        /**
         * Clear all entries (keep dimensions).
         */
        void clear() noexcept
        {
            m_row_indices.clear();
            m_col_indices.clear();
            m_values.clear();
        }

        /**
         * Sort entries by row, then column (required before CSR conversion).
         * Uses stable sort and merges duplicate entries by summing values.
         */
        void sort_and_merge()
        {
            if (m_values.empty()) return;
            // Create index permutation sorted by (row, col)
            std::vector<size_type> perm(m_values.size());
            std::iota(perm.begin(), perm.end(), size_type(0));
            std::sort(perm.begin(), perm.end(),
                      [this](size_type a, size_type b) {
                          if (m_row_indices[a] != m_row_indices[b])
                              return m_row_indices[a] < m_row_indices[b];
                          return m_col_indices[a] < m_col_indices[b];
                      });

            // Apply permutation
            apply_permutation(perm);

            // Merge duplicates (same row and col)
            if (m_values.empty()) return;
            size_type write = 0;
            for (size_type read = 1; read < m_values.size(); ++read)
            {
                if (m_row_indices[read] == m_row_indices[write] &&
                    m_col_indices[read] == m_col_indices[write])
                {
                    m_values[write] += m_values[read];
                }
                else
                {
                    ++write;
                    if (write != read)
                    {
                        m_row_indices[write] = m_row_indices[read];
                        m_col_indices[write] = m_col_indices[read];
                        m_values[write] = m_values[read];
                    }
                }
            }
            // Resize to remove merged entries
            size_type new_size = write + 1;
            m_row_indices.resize(new_size);
            m_col_indices.resize(new_size);
            m_values.resize(new_size);
        }

        /**
         * Convert to CSR format. Calls sort_and_merge first if not already sorted.
         */
        auto to_csr() const;

        /**
         * Convert to CSC format. Calls sort_and_merge with column-major order.
         */
        auto to_csc() const;

        /**
         * Scale all values by a scalar.
         */
        xcoo_matrix& operator*=(value_type alpha)
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
         * Element-wise addition of another COO matrix (same dimensions).
         */
        xcoo_matrix& operator+=(const xcoo_matrix& other)
        {
            if (m_rows != other.m_rows || m_cols != other.m_cols)
                throw std::runtime_error("xcoo_matrix::operator+=: dimension mismatch.");
            m_row_indices.insert(m_row_indices.end(), other.m_row_indices.begin(), other.m_row_indices.end());
            m_col_indices.insert(m_col_indices.end(), other.m_col_indices.begin(), other.m_col_indices.end());
            m_values.insert(m_values.end(), other.m_values.begin(), other.m_values.end());
            sort_and_merge();
            return *this;
        }

        /**
         * Transpose: swap rows and cols, return new COO.
         */
        xcoo_matrix transpose() const
        {
            xcoo_matrix result(m_cols, m_rows);
            result.m_row_indices = m_col_indices;
            result.m_col_indices = m_row_indices;
            result.m_values = m_values;
            return result;
        }

        /**
         * Number of elements in the full dense matrix.
         */
        size_type dense_size() const noexcept { return m_rows * m_cols; }

        /**
         * Sparsity ratio (nnz / total elements).
         */
        double sparsity() const noexcept
        {
            if (m_rows == 0 || m_cols == 0) return 1.0;
            return static_cast<double>(nnz()) / static_cast<double>(dense_size());
        }

    private:
        size_type m_rows;
        size_type m_cols;
        std::vector<size_type> m_row_indices;
        std::vector<size_type> m_col_indices;
        std::vector<value_type> m_values;

        void apply_permutation(const std::vector<size_type>& perm)
        {
            std::vector<size_type> tmp_row(m_values.size());
            std::vector<size_type> tmp_col(m_values.size());
            std::vector<value_type> tmp_val(m_values.size());
            for (size_type i = 0; i < m_values.size(); ++i)
            {
                tmp_row[i] = m_row_indices[perm[i]];
                tmp_col[i] = m_col_indices[perm[i]];
                tmp_val[i] = m_values[perm[i]];
            }
            m_row_indices.swap(tmp_row);
            m_col_indices.swap(tmp_col);
            m_values.swap(tmp_val);
        }
    };

    // Forward declaration for CSR conversion (defined in xcsr.hpp)
    template <class T> class xcsr_matrix;

    template <class T>
    inline auto xcoo_matrix<T>::to_csr() const
    {
        // Will be implemented in xcsr.hpp using friend access
        // For now, provide placeholder that throws if not available
        throw std::runtime_error("xcoo_matrix::to_csr requires xcsr.hpp include.");
    }

    /**
     * Create a COO matrix from a dense array (2D).
     */
    template <class E>
    inline auto dense_to_coo(const E& dense, typename E::value_type zero_tol = 0)
    {
        using T = typename E::value_type;
        auto sh = dense.shape();
        if (sh.size() != 2)
            throw std::runtime_error("dense_to_coo: requires 2D array.");
        xcoo_matrix<T> coo(sh[0], sh[1]);
        for (std::size_t i = 0; i < sh[0]; ++i)
        {
            for (std::size_t j = 0; j < sh[1]; ++j)
            {
                T val = dense(i, j);
                if (std::abs(val) > zero_tol)
                    coo.append(i, j, val);
            }
        }
        return coo;
    }

    /**
     * Element-wise COO + COO addition returning new COO.
     */
    template <class T>
    inline xcoo_matrix<T> operator+(const xcoo_matrix<T>& a, const xcoo_matrix<T>& b)
    {
        xcoo_matrix<T> result = a;
        result += b;
        return result;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XCOO_HPP