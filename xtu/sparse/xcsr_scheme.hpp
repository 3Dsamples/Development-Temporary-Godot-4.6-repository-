// sparse/xcsr_scheme.hpp

#ifndef XTENSOR_XCSR_SCHEME_HPP
#define XTENSOR_XCSR_SCHEME_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xsorting.hpp"
#include "xcoo_scheme.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <memory>
#include <functional>
#include <unordered_map>
#include <map>
#include <set>
#include <cmath>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace sparse
        {
            // --------------------------------------------------------------------
            // CSR Scheme: Compressed Sparse Row format for 2D matrices
            // --------------------------------------------------------------------
            template <class T>
            class xcsr_scheme
            {
            public:
                using value_type = T;
                using size_type = std::size_t;
                using index_type = std::vector<size_type>;
                using indices_type = std::vector<size_type>;
                using values_type = std::vector<T>;
                using shape_type = std::vector<size_type>;
                using storage_type = std::tuple<indices_type, indices_type, values_type>;

                // Construction
                xcsr_scheme() = default;
                
                explicit xcsr_scheme(const shape_type& shape)
                    : m_shape(shape), m_size(compute_size(shape))
                {
                    if (m_shape.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "xcsr_scheme: only 2D matrices supported");
                    m_row_ptr.resize(m_shape[0] + 1, 0);
                }

                xcsr_scheme(const shape_type& shape,
                            const indices_type& row_ptr,
                            const indices_type& col_indices,
                            const values_type& values)
                    : m_shape(shape), m_row_ptr(row_ptr), m_col_indices(col_indices), m_values(values),
                      m_size(compute_size(shape))
                {
                    if (m_shape.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "xcsr_scheme: only 2D matrices supported");
                    if (m_row_ptr.size() != m_shape[0] + 1)
                        XTENSOR_THROW(std::invalid_argument, "xcsr_scheme: row_ptr size must be rows+1");
                    if (m_col_indices.size() != m_values.size())
                        XTENSOR_THROW(std::invalid_argument, "xcsr_scheme: col_indices and values size mismatch");
                    if (!m_row_ptr.empty() && m_row_ptr.back() != m_values.size())
                        XTENSOR_THROW(std::invalid_argument, "xcsr_scheme: row_ptr back must equal nnz");
                }

                // Accessors
                const shape_type& shape() const noexcept { return m_shape; }
                size_type dimension() const noexcept { return m_shape.size(); }
                size_type size() const noexcept { return m_size; }
                size_type rows() const noexcept { return m_shape[0]; }
                size_type cols() const noexcept { return m_shape[1]; }
                size_type nnz() const noexcept { return m_values.size(); }
                
                const indices_type& row_ptr() const noexcept { return m_row_ptr; }
                const indices_type& col_indices() const noexcept { return m_col_indices; }
                const values_type& values() const noexcept { return m_values; }
                
                indices_type& row_ptr() noexcept { return m_row_ptr; }
                indices_type& col_indices() noexcept { return m_col_indices; }
                values_type& values() noexcept { return m_values; }

                // Get value at (row, col)
                T get(size_type row, size_type col) const
                {
                    if (row >= rows() || col >= cols())
                        XTENSOR_THROW(std::out_of_range, "xcsr_scheme::get: index out of bounds");
                    size_type start = m_row_ptr[row];
                    size_type end = m_row_ptr[row + 1];
                    for (size_type i = start; i < end; ++i)
                        if (m_col_indices[i] == col)
                            return m_values[i];
                    return T(0);
                }

                // Set value (updates or inserts)
                void set(size_type row, size_type col, const T& value)
                {
                    if (row >= rows() || col >= cols())
                        XTENSOR_THROW(std::out_of_range, "xcsr_scheme::set: index out of bounds");
                    size_type start = m_row_ptr[row];
                    size_type end = m_row_ptr[row + 1];
                    for (size_type i = start; i < end; ++i)
                    {
                        if (m_col_indices[i] == col)
                        {
                            if (value == T(0))
                            {
                                // Remove element
                                m_col_indices.erase(m_col_indices.begin() + static_cast<std::ptrdiff_t>(i));
                                m_values.erase(m_values.begin() + static_cast<std::ptrdiff_t>(i));
                                for (size_type r = row + 1; r <= rows(); ++r)
                                    m_row_ptr[r]--;
                            }
                            else
                                m_values[i] = value;
                            return;
                        }
                    }
                    if (value != T(0))
                    {
                        // Insert new element at correct position to keep column indices sorted
                        auto col_it = std::lower_bound(m_col_indices.begin() + static_cast<std::ptrdiff_t>(start),
                                                       m_col_indices.begin() + static_cast<std::ptrdiff_t>(end), col);
                        size_type insert_pos = static_cast<size_type>(std::distance(m_col_indices.begin(), col_it));
                        m_col_indices.insert(col_it, col);
                        m_values.insert(m_values.begin() + static_cast<std::ptrdiff_t>(insert_pos), value);
                        for (size_type r = row + 1; r <= rows(); ++r)
                            m_row_ptr[r]++;
                    }
                }

                // Add value (accumulates)
                void add(size_type row, size_type col, const T& value)
                {
                    T cur = get(row, col);
                    set(row, col, cur + value);
                }

                // Build CSR from triplets (COO-like)
                void build_from_triplets(const std::vector<size_type>& rows,
                                         const std::vector<size_type>& cols,
                                         const values_type& vals)
                {
                    if (rows.size() != cols.size() || rows.size() != vals.size())
                        XTENSOR_THROW(std::invalid_argument, "build_from_triplets: size mismatch");
                    
                    m_values.clear();
                    m_col_indices.clear();
                    m_row_ptr.assign(rows() + 1, 0);

                    // Count non-zeros per row
                    for (size_type r : rows)
                    {
                        if (r >= rows())
                            XTENSOR_THROW(std::out_of_range, "build_from_triplets: row index out of bounds");
                        m_row_ptr[r + 1]++;
                    }
                    // Cumulative sum for row pointers
                    for (size_type i = 1; i <= rows(); ++i)
                        m_row_ptr[i] += m_row_ptr[i - 1];

                    // Temporary array to track insert positions
                    std::vector<size_type> insert_pos = m_row_ptr;
                    m_col_indices.resize(m_row_ptr.back());
                    m_values.resize(m_row_ptr.back());

                    for (size_type k = 0; k < rows.size(); ++k)
                    {
                        size_type r = rows[k];
                        size_type c = cols[k];
                        size_type pos = insert_pos[r]++;
                        m_col_indices[pos] = c;
                        m_values[pos] = vals[k];
                    }

                    // Sort each row's columns
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        size_type start = m_row_ptr[r];
                        size_type end = m_row_ptr[r + 1];
                        if (start >= end) continue;
                        std::vector<std::pair<size_type, T>> row_data;
                        for (size_type i = start; i < end; ++i)
                            row_data.emplace_back(m_col_indices[i], m_values[i]);
                        std::sort(row_data.begin(), row_data.end());
                        for (size_type i = 0; i < row_data.size(); ++i)
                        {
                            m_col_indices[start + i] = row_data[i].first;
                            m_values[start + i] = row_data[i].second;
                        }
                    }

                    // Sum duplicates
                    sum_duplicates();
                }

                // Sum duplicate entries (assumes columns sorted)
                void sum_duplicates()
                {
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        size_type start = m_row_ptr[r];
                        size_type end = m_row_ptr[r + 1];
                        if (end - start <= 1) continue;
                        size_type write = start;
                        for (size_type read = start + 1; read < end; ++read)
                        {
                            if (m_col_indices[read] == m_col_indices[write])
                                m_values[write] += m_values[read];
                            else
                            {
                                ++write;
                                m_col_indices[write] = m_col_indices[read];
                                m_values[write] = m_values[read];
                            }
                        }
                        size_type new_end = write + 1;
                        if (new_end < end)
                        {
                            m_col_indices.erase(m_col_indices.begin() + static_cast<std::ptrdiff_t>(new_end),
                                                m_col_indices.begin() + static_cast<std::ptrdiff_t>(end));
                            m_values.erase(m_values.begin() + static_cast<std::ptrdiff_t>(new_end),
                                           m_values.begin() + static_cast<std::ptrdiff_t>(end));
                            size_type diff = end - new_end;
                            for (size_type r2 = r + 1; r2 <= rows(); ++r2)
                                m_row_ptr[r2] -= diff;
                        }
                    }
                }

                // Eliminate zeros
                void eliminate_zeros()
                {
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        size_type start = m_row_ptr[r];
                        size_type end = m_row_ptr[r + 1];
                        size_type write = start;
                        for (size_type read = start; read < end; ++read)
                        {
                            if (m_values[read] != T(0))
                            {
                                if (write != read)
                                {
                                    m_col_indices[write] = m_col_indices[read];
                                    m_values[write] = m_values[read];
                                }
                                ++write;
                            }
                        }
                        if (write < end)
                        {
                            size_type diff = end - write;
                            m_col_indices.erase(m_col_indices.begin() + static_cast<std::ptrdiff_t>(write),
                                                m_col_indices.begin() + static_cast<std::ptrdiff_t>(end));
                            m_values.erase(m_values.begin() + static_cast<std::ptrdiff_t>(write),
                                           m_values.begin() + static_cast<std::ptrdiff_t>(end));
                            for (size_type r2 = r + 1; r2 <= rows(); ++r2)
                                m_row_ptr[r2] -= diff;
                        }
                    }
                }

                // Convert to dense
                xarray_container<T> to_dense() const
                {
                    xarray_container<T> dense(m_shape, T(0));
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                            dense(r, m_col_indices[i]) = m_values[i];
                    }
                    return dense;
                }

                // Create from dense
                static xcsr_scheme from_dense(const xarray_container<T>& dense)
                {
                    if (dense.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "from_dense: only 2D arrays supported");
                    xcsr_scheme result(dense.shape());
                    for (size_type r = 0; r < dense.shape()[0]; ++r)
                        for (size_type c = 0; c < dense.shape()[1]; ++c)
                            if (dense(r, c) != T(0))
                                result.set(r, c, dense(r, c));
                    return result;
                }

                // Convert to COO
                xcoo_scheme<T> to_coo() const
                {
                    xcoo_scheme<T> coo(m_shape);
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                            coo.add({r, m_col_indices[i]}, m_values[i]);
                    }
                    return coo;
                }

                // Create from COO
                static xcsr_scheme from_coo(const xcoo_scheme<T>& coo)
                {
                    if (coo.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "from_coo: only 2D supported");
                    xcsr_scheme result(coo.shape());
                    std::vector<size_type> rows, cols;
                    for (const auto& idx : coo.indices())
                    {
                        rows.push_back(idx[0]);
                        cols.push_back(idx[1]);
                    }
                    result.build_from_triplets(rows, cols, coo.values());
                    return result;
                }

                // Matrix-vector multiplication: y = A * x
                xarray_container<T> matvec(const xarray_container<T>& x) const
                {
                    if (x.dimension() != 1 || x.size() != cols())
                        XTENSOR_THROW(std::invalid_argument, "matvec: dimension mismatch");
                    xarray_container<T> y({rows()}, T(0));
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        T sum = 0;
                        for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                            sum += m_values[i] * x(m_col_indices[i]);
                        y(r) = sum;
                    }
                    return y;
                }

                // Matrix-vector multiplication with transpose: y = A^T * x
                xarray_container<T> matvec_transpose(const xarray_container<T>& x) const
                {
                    if (x.dimension() != 1 || x.size() != rows())
                        XTENSOR_THROW(std::invalid_argument, "matvec_transpose: dimension mismatch");
                    xarray_container<T> y({cols()}, T(0));
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        T x_r = x(r);
                        if (x_r == T(0)) continue;
                        for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                            y(m_col_indices[i]) += m_values[i] * x_r;
                    }
                    return y;
                }

                // Sparse matrix-matrix multiplication: C = A * B
                xcsr_scheme matmul(const xcsr_scheme& other) const
                {
                    if (cols() != other.rows())
                        XTENSOR_THROW(std::invalid_argument, "matmul: inner dimension mismatch");
                    
                    xcsr_scheme result({rows(), other.cols()});
                    std::vector<size_type> result_row_ptr(rows() + 1, 0);
                    
                    // Temporary dense accumulator for each row of result
                    std::vector<T> row_acc(other.cols(), T(0));
                    std::vector<bool> row_mask(other.cols(), false);
                    
                    for (size_type i = 0; i < rows(); ++i)
                    {
                        std::fill(row_acc.begin(), row_acc.end(), T(0));
                        std::fill(row_mask.begin(), row_mask.end(), false);
                        size_type nnz_row = 0;
                        
                        for (size_type jj = m_row_ptr[i]; jj < m_row_ptr[i+1]; ++jj)
                        {
                            size_type k = m_col_indices[jj];
                            T a_ik = m_values[jj];
                            for (size_type j = other.m_row_ptr[k]; j < other.m_row_ptr[k+1]; ++j)
                            {
                                size_type col = other.m_col_indices[j];
                                row_acc[col] += a_ik * other.m_values[j];
                                if (!row_mask[col])
                                {
                                    row_mask[col] = true;
                                    ++nnz_row;
                                }
                            }
                        }
                        
                        // Store non-zeros
                        for (size_type col = 0; col < other.cols(); ++col)
                        {
                            if (row_mask[col] && row_acc[col] != T(0))
                            {
                                result.m_col_indices.push_back(col);
                                result.m_values.push_back(row_acc[col]);
                            }
                        }
                        result_row_ptr[i+1] = result.m_values.size();
                    }
                    
                    result.m_row_ptr = std::move(result_row_ptr);
                    result.m_shape = {rows(), other.cols()};
                    result.m_size = rows() * other.cols();
                    return result;
                }

                // Transpose
                xcsr_scheme transpose() const
                {
                    xcsr_scheme result({cols(), rows()});
                    // Count non-zeros per column
                    std::vector<size_type> col_counts(cols(), 0);
                    for (size_type r = 0; r < rows(); ++r)
                        for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                            col_counts[m_col_indices[i]]++;
                    
                    // Build row pointers for result (which are column pointers of original)
                    result.m_row_ptr.resize(cols() + 1, 0);
                    for (size_type c = 0; c < cols(); ++c)
                        result.m_row_ptr[c + 1] = result.m_row_ptr[c] + col_counts[c];
                    
                    size_type nnz = result.m_row_ptr.back();
                    result.m_col_indices.resize(nnz);
                    result.m_values.resize(nnz);
                    
                    // Temporary array to track insert positions
                    std::vector<size_type> insert_pos = result.m_row_ptr;
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                        {
                            size_type c = m_col_indices[i];
                            size_type pos = insert_pos[c]++;
                            result.m_col_indices[pos] = r;
                            result.m_values[pos] = m_values[i];
                        }
                    }
                    return result;
                }

                // Extract diagonal
                xarray_container<T> diagonal(std::ptrdiff_t offset = 0) const
                {
                    size_type diag_size = 0;
                    if (offset >= 0)
                        diag_size = std::min(cols() - static_cast<size_type>(offset), rows());
                    else
                        diag_size = std::min(rows() - static_cast<size_type>(-offset), cols());
                    
                    xarray_container<T> result({diag_size}, T(0));
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        size_type c = static_cast<size_type>(static_cast<std::ptrdiff_t>(r) + offset);
                        if (c < cols())
                        {
                            // Binary search in row r for column c
                            size_type start = m_row_ptr[r];
                            size_type end = m_row_ptr[r + 1];
                            auto it = std::lower_bound(m_col_indices.begin() + static_cast<std::ptrdiff_t>(start),
                                                       m_col_indices.begin() + static_cast<std::ptrdiff_t>(end), c);
                            if (it != m_col_indices.begin() + static_cast<std::ptrdiff_t>(end) && *it == c)
                            {
                                size_type idx = static_cast<size_type>(std::distance(m_col_indices.begin(), it));
                                size_type diag_idx = offset >= 0 ? r : r - static_cast<size_type>(-offset);
                                result(diag_idx) = m_values[idx];
                            }
                        }
                    }
                    return result;
                }

                // Scale by scalar
                xcsr_scheme operator*(T scalar) const
                {
                    xcsr_scheme result(*this);
                    for (auto& v : result.m_values)
                        v *= scalar;
                    return result;
                }

                // Addition
                xcsr_scheme operator+(const xcsr_scheme& other) const
                {
                    if (m_shape != other.m_shape)
                        XTENSOR_THROW(std::invalid_argument, "operator+: shape mismatch");
                    xcsr_scheme result(m_shape);
                    for (size_type r = 0; r < rows(); ++r)
                    {
                        size_type i = m_row_ptr[r], i_end = m_row_ptr[r+1];
                        size_type j = other.m_row_ptr[r], j_end = other.m_row_ptr[r+1];
                        while (i < i_end || j < j_end)
                        {
                            size_type col_i = (i < i_end) ? m_col_indices[i] : cols();
                            size_type col_j = (j < j_end) ? other.m_col_indices[j] : cols();
                            if (col_i < col_j)
                            {
                                result.m_col_indices.push_back(col_i);
                                result.m_values.push_back(m_values[i]);
                                ++i;
                            }
                            else if (col_j < col_i)
                            {
                                result.m_col_indices.push_back(col_j);
                                result.m_values.push_back(other.m_values[j]);
                                ++j;
                            }
                            else
                            {
                                T sum = m_values[i] + other.m_values[j];
                                if (sum != T(0))
                                {
                                    result.m_col_indices.push_back(col_i);
                                    result.m_values.push_back(sum);
                                }
                                ++i; ++j;
                            }
                        }
                        result.m_row_ptr.push_back(result.m_values.size());
                    }
                    return result;
                }

                // Subtraction
                xcsr_scheme operator-(const xcsr_scheme& other) const
                {
                    return *this + (other * T(-1));
                }

                // Clear
                void clear()
                {
                    m_row_ptr.assign(rows() + 1, 0);
                    m_col_indices.clear();
                    m_values.clear();
                }

                // Reserve
                void reserve(size_type nnz_estimate)
                {
                    m_col_indices.reserve(nnz_estimate);
                    m_values.reserve(nnz_estimate);
                }

            private:
                shape_type m_shape;
                indices_type m_row_ptr;      // size rows+1
                indices_type m_col_indices;  // size nnz
                values_type m_values;        // size nnz
                size_type m_size = 0;
            };

            // --------------------------------------------------------------------
            // CSR Sparse Tensor expression wrapper
            // --------------------------------------------------------------------
            template <class T>
            class xsparse_csr_tensor : public xexpression<xsparse_csr_tensor<T>>
            {
            public:
                using self_type = xsparse_csr_tensor<T>;
                using base_type = xexpression<self_type>;
                using scheme_type = xcsr_scheme<T>;
                using value_type = T;
                using reference = T&;
                using const_reference = const T&;
                using pointer = T*;
                using const_pointer = const T*;
                using size_type = typename scheme_type::size_type;
                using difference_type = std::ptrdiff_t;
                using shape_type = typename scheme_type::shape_type;
                using strides_type = shape_type;
                using expression_tag = xsparse_tag;
                
                static constexpr layout_type layout = layout_type::dynamic;
                static constexpr bool is_const = false;

                xsparse_csr_tensor() = default;
                
                explicit xsparse_csr_tensor(const shape_type& shape)
                    : m_scheme(shape)
                {
                    if (shape.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "xsparse_csr_tensor: only 2D supported");
                }

                xsparse_csr_tensor(const shape_type& shape,
                                   const std::vector<size_type>& row_ptr,
                                   const std::vector<size_type>& col_indices,
                                   const std::vector<T>& values)
                    : m_scheme(shape, row_ptr, col_indices, values)
                {
                }

                explicit xsparse_csr_tensor(const scheme_type& scheme) : m_scheme(scheme) {}
                explicit xsparse_csr_tensor(scheme_type&& scheme) : m_scheme(std::move(scheme)) {}

                size_type size() const noexcept { return m_scheme.size(); }
                size_type dimension() const noexcept { return 2; }
                const shape_type& shape() const noexcept { return m_scheme.shape(); }
                const strides_type& strides() const noexcept { return m_scheme.shape(); }
                const strides_type& backstrides() const noexcept { return m_scheme.shape(); }
                layout_type layout() const noexcept { return layout_type::dynamic; }
                size_type nnz() const noexcept { return m_scheme.nnz(); }

                const scheme_type& scheme() const noexcept { return m_scheme; }
                scheme_type& scheme() noexcept { return m_scheme; }

                value_type operator()(size_type row, size_type col) const
                {
                    return m_scheme.get(row, col);
                }

                template <class S>
                value_type element(const S& index) const
                {
                    if (index.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "element: index must be size 2");
                    return m_scheme.get(index[0], index[1]);
                }

                xarray_container<T> to_dense() const
                {
                    return m_scheme.to_dense();
                }

                template <class E>
                self_type& operator=(const xexpression<E>& e)
                {
                    auto dense = eval(e);
                    *this = xsparse_csr_tensor(scheme_type::from_dense(dense));
                    return *this;
                }

                xsparse_csr_tensor operator+(const xsparse_csr_tensor& other) const
                {
                    return xsparse_csr_tensor(m_scheme + other.m_scheme);
                }

                xsparse_csr_tensor operator-(const xsparse_csr_tensor& other) const
                {
                    return xsparse_csr_tensor(m_scheme - other.m_scheme);
                }

                xsparse_csr_tensor operator*(T scalar) const
                {
                    return xsparse_csr_tensor(m_scheme * scalar);
                }

                xsparse_csr_tensor matmul(const xsparse_csr_tensor& other) const
                {
                    return xsparse_csr_tensor(m_scheme.matmul(other.m_scheme));
                }

                xarray_container<T> matvec(const xarray_container<T>& x) const
                {
                    return m_scheme.matvec(x);
                }

                xsparse_csr_tensor transpose() const
                {
                    return xsparse_csr_tensor(m_scheme.transpose());
                }

                xarray_container<T> diagonal(std::ptrdiff_t offset = 0) const
                {
                    return m_scheme.diagonal(offset);
                }

                // Conversion to/from COO
                coo_tensor<T> to_coo() const
                {
                    return coo_tensor<T>(m_scheme.to_coo());
                }

                static xsparse_csr_tensor from_coo(const coo_tensor<T>& coo)
                {
                    return xsparse_csr_tensor(scheme_type::from_coo(coo.scheme()));
                }

                // Iterator over non-zeros (row-wise)
                class row_iterator
                {
                public:
                    using iterator_category = std::forward_iterator_tag;
                    using value_type = std::tuple<size_type, size_type, T>;
                    
                    row_iterator(const xsparse_csr_tensor* tensor, size_type row, size_type idx)
                        : m_tensor(tensor), m_row(row), m_idx(idx) {}
                    
                    value_type operator*() const
                    {
                        return {m_row, m_tensor->m_scheme.col_indices()[m_idx],
                                m_tensor->m_scheme.values()[m_idx]};
                    }
                    
                    row_iterator& operator++() { ++m_idx; return *this; }
                    row_iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }
                    
                    bool operator==(const row_iterator& other) const
                    {
                        return m_tensor == other.m_tensor && m_idx == other.m_idx;
                    }
                    bool operator!=(const row_iterator& other) const { return !(*this == other); }
                    
                private:
                    const xsparse_csr_tensor* m_tensor = nullptr;
                    size_type m_row = 0;
                    size_type m_idx = 0;
                };

                row_iterator row_begin(size_type row) const
                {
                    return row_iterator(this, row, m_scheme.row_ptr()[row]);
                }
                row_iterator row_end(size_type row) const
                {
                    return row_iterator(this, row, m_scheme.row_ptr()[row + 1]);
                }

            private:
                scheme_type m_scheme;
            };

            // Convenience alias
            template <class T>
            using csr_tensor = xsparse_csr_tensor<T>;

            // Helper functions
            template <class T>
            inline csr_tensor<T> csr_zeros(const std::vector<std::size_t>& shape)
            {
                return csr_tensor<T>(shape);
            }

            template <class T>
            inline csr_tensor<T> csr_eye(std::size_t n)
            {
                csr_tensor<T> result({n, n});
                for (std::size_t i = 0; i < n; ++i)
                    result.scheme().set(i, i, T(1));
                return result;
            }

            template <class T>
            inline csr_tensor<T> csr_diag(const xarray_container<T>& diag)
            {
                size_t n = diag.size();
                csr_tensor<T> result({n, n});
                for (size_t i = 0; i < n; ++i)
                    if (diag(i) != T(0))
                        result.scheme().set(i, i, diag(i));
                return result;
            }

        } // namespace sparse

        using sparse::xcsr_scheme;
        using sparse::xsparse_csr_tensor;
        using sparse::csr_tensor;
        using sparse::csr_zeros;
        using sparse::csr_eye;
        using sparse::csr_diag;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XCSR_SCHEME_HPP

// sparse/xcsr_scheme.hpp