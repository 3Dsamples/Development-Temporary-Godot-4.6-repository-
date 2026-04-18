// sparse/xcoo_scheme.hpp

#ifndef XTENSOR_XCOO_SCHEME_HPP
#define XTENSOR_XCOO_SCHEME_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xsorting.hpp"

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

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace sparse
        {
            // --------------------------------------------------------------------
            // COO Scheme: Coordinate list format for sparse tensors
            // --------------------------------------------------------------------
            template <class T>
            class xcoo_scheme
            {
            public:
                using value_type = T;
                using size_type = std::size_t;
                using index_type = std::vector<size_type>;
                using indices_type = std::vector<index_type>;
                using values_type = std::vector<T>;
                using shape_type = std::vector<size_type>;
                using storage_type = std::pair<indices_type, values_type>;

                // Construction
                xcoo_scheme() = default;
                
                explicit xcoo_scheme(const shape_type& shape)
                    : m_shape(shape), m_size(compute_size(shape))
                {
                }

                xcoo_scheme(const shape_type& shape, const indices_type& indices, const values_type& values)
                    : m_shape(shape), m_indices(indices), m_values(values), m_size(compute_size(shape))
                {
                    if (m_indices.size() != m_values.size())
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme: indices and values must have same size");
                    for (const auto& idx : m_indices)
                        if (idx.size() != m_shape.size())
                            XTENSOR_THROW(std::invalid_argument, "xcoo_scheme: index dimension mismatch");
                }

                // Accessors
                const shape_type& shape() const noexcept { return m_shape; }
                size_type dimension() const noexcept { return m_shape.size(); }
                size_type size() const noexcept { return m_size; }
                size_type nnz() const noexcept { return m_values.size(); }
                const indices_type& indices() const noexcept { return m_indices; }
                const values_type& values() const noexcept { return m_values; }
                
                // Mutable access
                indices_type& indices() noexcept { return m_indices; }
                values_type& values() noexcept { return m_values; }

                // Add a single non-zero element
                void add(const index_type& index, const T& value)
                {
                    if (index.size() != m_shape.size())
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::add: index dimension mismatch");
                    for (size_t d = 0; d < m_shape.size(); ++d)
                        if (index[d] >= m_shape[d])
                            XTENSOR_THROW(std::out_of_range, "xcoo_scheme::add: index out of bounds");
                    m_indices.push_back(index);
                    m_values.push_back(value);
                }

                // Remove element at given position (if exists)
                bool remove(const index_type& index)
                {
                    auto it = std::find(m_indices.begin(), m_indices.end(), index);
                    if (it != m_indices.end())
                    {
                        size_t pos = std::distance(m_indices.begin(), it);
                        m_indices.erase(it);
                        m_values.erase(m_values.begin() + static_cast<std::ptrdiff_t>(pos));
                        return true;
                    }
                    return false;
                }

                // Get value at index, or zero if not present
                T get(const index_type& index) const
                {
                    auto it = std::find(m_indices.begin(), m_indices.end(), index);
                    if (it != m_indices.end())
                        return m_values[static_cast<size_t>(std::distance(m_indices.begin(), it))];
                    return T(0);
                }

                // Set value at index (adds if not present)
                void set(const index_type& index, const T& value)
                {
                    auto it = std::find(m_indices.begin(), m_indices.end(), index);
                    if (it != m_indices.end())
                    {
                        size_t pos = static_cast<size_t>(std::distance(m_indices.begin(), it));
                        m_values[pos] = value;
                    }
                    else
                    {
                        add(index, value);
                    }
                }

                // Sort indices (lexicographically)
                void sort_indices()
                {
                    std::vector<size_t> order(m_indices.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [this](size_t a, size_t b) {
                        return m_indices[a] < m_indices[b];
                    });
                    indices_type new_indices(m_indices.size());
                    values_type new_values(m_values.size());
                    for (size_t i = 0; i < order.size(); ++i)
                    {
                        new_indices[i] = m_indices[order[i]];
                        new_values[i] = m_values[order[i]];
                    }
                    m_indices = std::move(new_indices);
                    m_values = std::move(new_values);
                }

                // Sum duplicates (after sorting)
                void sum_duplicates()
                {
                    if (m_indices.empty()) return;
                    sort_indices();
                    indices_type unique_indices;
                    values_type unique_values;
                    unique_indices.reserve(m_indices.size());
                    unique_values.reserve(m_values.size());
                    
                    unique_indices.push_back(m_indices[0]);
                    unique_values.push_back(m_values[0]);
                    for (size_t i = 1; i < m_indices.size(); ++i)
                    {
                        if (m_indices[i] == unique_indices.back())
                            unique_values.back() += m_values[i];
                        else
                        {
                            unique_indices.push_back(m_indices[i]);
                            unique_values.push_back(m_values[i]);
                        }
                    }
                    m_indices = std::move(unique_indices);
                    m_values = std::move(unique_values);
                }

                // Clear all non-zero elements
                void clear()
                {
                    m_indices.clear();
                    m_values.clear();
                }

                // Reserve memory
                void reserve(size_type capacity)
                {
                    m_indices.reserve(capacity);
                    m_values.reserve(capacity);
                }

                // Convert to dense format
                xarray_container<T> to_dense() const
                {
                    xarray_container<T> dense(m_shape, T(0));
                    for (size_t i = 0; i < m_indices.size(); ++i)
                        dense.element(m_indices[i]) = m_values[i];
                    return dense;
                }

                // Create from dense tensor
                static xcoo_scheme from_dense(const xarray_container<T>& dense)
                {
                    xcoo_scheme result(dense.shape());
                    auto flat_shape = dense.shape();
                    for (size_t flat = 0; flat < dense.size(); ++flat)
                    {
                        T val = dense.flat(flat);
                        if (val != T(0))
                        {
                            // Convert flat index to multi-index
                            index_type idx(flat_shape.size());
                            size_t temp = flat;
                            for (size_t d = flat_shape.size(); d > 0; --d)
                            {
                                idx[d-1] = temp % flat_shape[d-1];
                                temp /= flat_shape[d-1];
                            }
                            result.add(idx, val);
                        }
                    }
                    return result;
                }

                // Element-wise operations with broadcasting (simplified)
                xcoo_scheme operator+(const xcoo_scheme& other) const
                {
                    if (m_shape != other.m_shape)
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::operator+: shape mismatch");
                    xcoo_scheme result(m_shape);
                    // Merge both lists and sum duplicates
                    result.m_indices = m_indices;
                    result.m_values = m_values;
                    result.m_indices.insert(result.m_indices.end(), other.m_indices.begin(), other.m_indices.end());
                    result.m_values.insert(result.m_values.end(), other.m_values.begin(), other.m_values.end());
                    result.sum_duplicates();
                    return result;
                }

                xcoo_scheme operator*(T scalar) const
                {
                    xcoo_scheme result(*this);
                    for (auto& v : result.m_values)
                        v *= scalar;
                    return result;
                }

                xcoo_scheme operator-() const
                {
                    return (*this) * T(-1);
                }

                xcoo_scheme operator-(const xcoo_scheme& other) const
                {
                    return *this + (-other);
                }

                // Sparse matrix-vector multiplication (if 2D)
                xarray_container<T> matvec(const xarray_container<T>& x) const
                {
                    if (m_shape.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::matvec: scheme must be 2D");
                    if (x.dimension() != 1 || x.size() != m_shape[1])
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::matvec: dimension mismatch");
                    xarray_container<T> y({m_shape[0]}, T(0));
                    for (size_t i = 0; i < m_indices.size(); ++i)
                    {
                        size_type row = m_indices[i][0];
                        size_type col = m_indices[i][1];
                        y(row) += m_values[i] * x(col);
                    }
                    return y;
                }

                // Sparse matrix-matrix multiplication (2D)
                xcoo_scheme matmul(const xcoo_scheme& other) const
                {
                    if (m_shape.size() != 2 || other.m_shape.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::matmul: both must be 2D");
                    if (m_shape[1] != other.m_shape[0])
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::matmul: inner dimension mismatch");

                    xcoo_scheme result({m_shape[0], other.m_shape[1]});
                    // Group by column for the second operand
                    std::unordered_map<size_type, std::vector<std::pair<size_type, T>>> col_map;
                    for (size_t i = 0; i < other.m_indices.size(); ++i)
                        col_map[other.m_indices[i][1]].emplace_back(other.m_indices[i][0], other.m_values[i]);

                    // For each non-zero in first operand, multiply with matching in second
                    std::map<std::pair<size_type, size_type>, T> accum;
                    for (size_t i = 0; i < m_indices.size(); ++i)
                    {
                        size_type row = m_indices[i][0];
                        size_type k = m_indices[i][1];
                        T val1 = m_values[i];
                        auto it = col_map.find(k);
                        if (it != col_map.end())
                        {
                            for (const auto& p : it->second)
                            {
                                size_type col = p.first;
                                T val2 = p.second;
                                accum[{row, col}] += val1 * val2;
                            }
                        }
                    }

                    for (const auto& p : accum)
                    {
                        if (p.second != T(0))
                            result.add({p.first.first, p.first.second}, p.second);
                    }
                    return result;
                }

                // Transpose (2D)
                xcoo_scheme transpose() const
                {
                    if (m_shape.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::transpose: only 2D supported");
                    xcoo_scheme result({m_shape[1], m_shape[0]});
                    result.m_indices.reserve(m_indices.size());
                    result.m_values.reserve(m_values.size());
                    for (size_t i = 0; i < m_indices.size(); ++i)
                    {
                        result.m_indices.push_back({m_indices[i][1], m_indices[i][0]});
                        result.m_values.push_back(m_values[i]);
                    }
                    return result;
                }

                // Extract diagonal (2D)
                xarray_container<T> diagonal(std::ptrdiff_t offset = 0) const
                {
                    if (m_shape.size() != 2)
                        XTENSOR_THROW(std::invalid_argument, "xcoo_scheme::diagonal: only 2D supported");
                    size_type diag_size = 0;
                    if (offset >= 0)
                        diag_size = std::min(m_shape[1] - static_cast<size_type>(offset), m_shape[0]);
                    else
                        diag_size = std::min(m_shape[0] - static_cast<size_type>(-offset), m_shape[1]);
                    xarray_container<T> result({diag_size}, T(0));
                    for (size_t i = 0; i < m_indices.size(); ++i)
                    {
                        size_type r = m_indices[i][0];
                        size_type c = m_indices[i][1];
                        if ((offset >= 0 && c == r + static_cast<size_type>(offset)) ||
                            (offset < 0 && r == c + static_cast<size_type>(-offset)))
                        {
                            size_type idx = offset >= 0 ? r : c;
                            if (idx < diag_size)
                                result(idx) += m_values[i];
                        }
                    }
                    return result;
                }

            private:
                shape_type m_shape;
                indices_type m_indices;
                values_type m_values;
                size_type m_size = 0;
            };

            // --------------------------------------------------------------------
            // COO Sparse Tensor (expression wrapper)
            // --------------------------------------------------------------------
            template <class T>
            class xsparse_coo_tensor : public xexpression<xsparse_coo_tensor<T>>
            {
            public:
                using self_type = xsparse_coo_tensor<T>;
                using base_type = xexpression<self_type>;
                using scheme_type = xcoo_scheme<T>;
                using value_type = T;
                using reference = T&;
                using const_reference = const T&;
                using pointer = T*;
                using const_pointer = const T*;
                using size_type = typename scheme_type::size_type;
                using difference_type = std::ptrdiff_t;
                using shape_type = typename scheme_type::shape_type;
                using strides_type = shape_type; // Not meaningful for sparse, but required
                using indices_type = typename scheme_type::indices_type;
                using values_type = typename scheme_type::values_type;
                using expression_tag = xsparse_tag;
                
                static constexpr layout_type layout = layout_type::dynamic;
                static constexpr bool is_const = false;

                // Construction
                xsparse_coo_tensor() = default;
                
                explicit xsparse_coo_tensor(const shape_type& shape)
                    : m_scheme(shape)
                {
                }

                xsparse_coo_tensor(const shape_type& shape, const indices_type& indices, const values_type& values)
                    : m_scheme(shape, indices, values)
                {
                }

                explicit xsparse_coo_tensor(const scheme_type& scheme) : m_scheme(scheme) {}
                explicit xsparse_coo_tensor(scheme_type&& scheme) : m_scheme(std::move(scheme)) {}

                // Accessors
                size_type size() const noexcept { return m_scheme.size(); }
                size_type dimension() const noexcept { return m_scheme.dimension(); }
                const shape_type& shape() const noexcept { return m_scheme.shape(); }
                const strides_type& strides() const noexcept { return m_scheme.shape(); } // dummy
                const strides_type& backstrides() const noexcept { return m_scheme.shape(); } // dummy
                layout_type layout() const noexcept { return layout_type::dynamic; }
                size_type nnz() const noexcept { return m_scheme.nnz(); }

                const scheme_type& scheme() const noexcept { return m_scheme; }
                scheme_type& scheme() noexcept { return m_scheme; }

                // Element access (read-only, expensive)
                template <class... Args>
                value_type operator()(Args... args) const
                {
                    std::array<size_type, sizeof...(Args)> idx = {static_cast<size_type>(args)...};
                    return m_scheme.get(std::vector<size_type>(idx.begin(), idx.end()));
                }

                template <class S>
                value_type element(const S& index) const
                {
                    return m_scheme.get(std::vector<size_type>(index.begin(), index.end()));
                }

                // Not efficient for random flat access
                value_type flat(size_type i) const
                {
                    // Convert flat index to multi-index
                    std::vector<size_type> idx(dimension());
                    size_type temp = i;
                    for (size_t d = dimension(); d > 0; --d)
                    {
                        idx[d-1] = temp % m_scheme.shape()[d-1];
                        temp /= m_scheme.shape()[d-1];
                    }
                    return m_scheme.get(idx);
                }

                // Conversion to dense
                xarray_container<T> to_dense() const
                {
                    return m_scheme.to_dense();
                }

                // Assignment from expression (converts to dense and back)
                template <class E>
                self_type& operator=(const xexpression<E>& e)
                {
                    auto dense = eval(e);
                    *this = xsparse_coo_tensor(scheme_type::from_dense(dense));
                    return *this;
                }

                // Arithmetic with broadcasting (returns dense for simplicity)
                template <class E>
                auto operator+(const xexpression<E>& e) const
                {
                    return to_dense() + e;
                }

                template <class E>
                auto operator-(const xexpression<E>& e) const
                {
                    return to_dense() - e;
                }

                template <class E>
                auto operator*(const xexpression<E>& e) const
                {
                    return to_dense() * e;
                }

                // Sparse-sparse operations (stay sparse if possible)
                xsparse_coo_tensor operator+(const xsparse_coo_tensor& other) const
                {
                    return xsparse_coo_tensor(m_scheme + other.m_scheme);
                }

                xsparse_coo_tensor operator-(const xsparse_coo_tensor& other) const
                {
                    return xsparse_coo_tensor(m_scheme - other.m_scheme);
                }

                xsparse_coo_tensor operator*(T scalar) const
                {
                    return xsparse_coo_tensor(m_scheme * scalar);
                }

                // Matrix multiplication (2D only)
                xsparse_coo_tensor matmul(const xsparse_coo_tensor& other) const
                {
                    return xsparse_coo_tensor(m_scheme.matmul(other.m_scheme));
                }

                // Iterator over non-zero elements
                class nnz_iterator
                {
                public:
                    using iterator_category = std::forward_iterator_tag;
                    using value_type = std::pair<std::vector<size_type>, T>;
                    using reference = const value_type&;
                    using pointer = const value_type*;

                    nnz_iterator() = default;
                    nnz_iterator(const xsparse_coo_tensor* tensor, size_t idx)
                        : m_tensor(tensor), m_idx(idx)
                    {
                        if (m_tensor && m_idx < m_tensor->nnz())
                        {
                            m_current.first = m_tensor->m_scheme.indices()[m_idx];
                            m_current.second = m_tensor->m_scheme.values()[m_idx];
                        }
                    }

                    reference operator*() const { return m_current; }
                    
                    nnz_iterator& operator++()
                    {
                        ++m_idx;
                        if (m_tensor && m_idx < m_tensor->nnz())
                        {
                            m_current.first = m_tensor->m_scheme.indices()[m_idx];
                            m_current.second = m_tensor->m_scheme.values()[m_idx];
                        }
                        return *this;
                    }

                    nnz_iterator operator++(int)
                    {
                        nnz_iterator tmp = *this;
                        ++*this;
                        return tmp;
                    }

                    bool operator==(const nnz_iterator& other) const
                    {
                        return m_tensor == other.m_tensor && m_idx == other.m_idx;
                    }

                    bool operator!=(const nnz_iterator& other) const
                    {
                        return !(*this == other);
                    }

                private:
                    const xsparse_coo_tensor* m_tensor = nullptr;
                    size_t m_idx = 0;
                    value_type m_current;
                };

                nnz_iterator nnz_begin() const { return nnz_iterator(this, 0); }
                nnz_iterator nnz_end() const { return nnz_iterator(this, nnz()); }

            private:
                scheme_type m_scheme;
            };

            // Convenience alias
            template <class T>
            using coo_tensor = xsparse_coo_tensor<T>;

            // Helper functions
            template <class T>
            inline coo_tensor<T> coo_zeros(const std::vector<std::size_t>& shape)
            {
                return coo_tensor<T>(shape);
            }

            template <class T>
            inline coo_tensor<T> coo_eye(std::size_t n)
            {
                coo_tensor<T> result({n, n});
                for (std::size_t i = 0; i < n; ++i)
                    result.scheme().add({i, i}, T(1));
                return result;
            }

            template <class T>
            inline coo_tensor<T> coo_diag(const xarray_container<T>& diag)
            {
                size_t n = diag.size();
                coo_tensor<T> result({n, n});
                for (size_t i = 0; i < n; ++i)
                    if (diag(i) != T(0))
                        result.scheme().add({i, i}, diag(i));
                return result;
            }

        } // namespace sparse

        // Bring sparse types into xt namespace
        using sparse::xcoo_scheme;
        using sparse::xsparse_coo_tensor;
        using sparse::coo_tensor;
        using sparse::coo_zeros;
        using sparse::coo_eye;
        using sparse::coo_diag;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XCOO_SCHEME_HPP

// sparse/xcoo_scheme.hpp