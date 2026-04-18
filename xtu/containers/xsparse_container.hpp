// xtensor-unified - Sparse tensor container (COO-based)
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_SPARSE_XSPARSE_CONTAINER_HPP
#define XTU_SPARSE_XSPARSE_CONTAINER_HPP

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/sparse/xcoo_scheme.hpp"
#include "xtu/containers/xarray.hpp"

XTU_NAMESPACE_BEGIN

// #############################################################################
// xsparse_container - Sparse array with expression support
// #############################################################################
template <class T, class I = index_type, class Tag = xtensor_expression_tag>
class xsparse_container : public xexpression<xsparse_container<T, I, Tag>> {
public:
    using value_type = T;
    using index_type = I;
    using size_type = xtu::size_type;
    using shape_type = std::vector<size_type>;
    using storage_type = xcoo_storage<T, I>;
    using scheme_type = xcoo_scheme<T, I>;
    using expression_tag = Tag;
    using self_type = xsparse_container<T, I, Tag>;

    // Iterator for non-zero elements (sparse iteration)
    class const_nnz_iterator {
    private:
        const self_type* m_container;
        size_type m_pos;
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<std::vector<index_type>, T>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = value_type;

        const_nnz_iterator(const self_type* cont, size_type pos)
            : m_container(cont), m_pos(pos) {}

        bool operator==(const const_nnz_iterator& other) const {
            return m_container == other.m_container && m_pos == other.m_pos;
        }
        bool operator!=(const const_nnz_iterator& other) const { return !(*this == other); }

        value_type operator*() const {
            const auto& storage = m_container->scheme().storage();
            index_type linear = storage.indices()[m_pos];
            // Convert linear index to coordinates
            std::vector<index_type> coords = m_container->unravel_index(static_cast<size_type>(linear));
            return {coords, storage.values()[m_pos]};
        }

        const_nnz_iterator& operator++() { ++m_pos; return *this; }
        const_nnz_iterator operator++(int) { const_nnz_iterator tmp = *this; ++m_pos; return tmp; }
    };

private:
    scheme_type m_scheme;
    bool m_finalized;

    // Helper to convert linear index to coordinates
    std::vector<index_type> unravel_index(size_type linear) const {
        std::vector<index_type> coords(m_scheme.dimension());
        const auto& shp = m_scheme.shape();
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shp.size()) - 1; i >= 0; --i) {
            coords[static_cast<std::size_t>(i)] = static_cast<index_type>(linear % shp[static_cast<std::size_t>(i)]);
            linear /= shp[static_cast<std::size_t>(i)];
        }
        return coords;
    }

public:
    // Constructors
    xsparse_container() : m_finalized(false) {}

    explicit xsparse_container(const shape_type& shape)
        : m_scheme(shape), m_finalized(false) {}

    xsparse_container(const shape_type& shape, const std::vector<value_type>& values,
                      const std::vector<index_type>& indices)
        : m_scheme(shape, values, indices), m_finalized(false) {}

    // From initializer list of triplets: {{i,j,k}, value} format for 3D, etc.
    // For simplicity, we provide a builder pattern.

    // Copy / move
    xsparse_container(const xsparse_container&) = default;
    xsparse_container(xsparse_container&&) noexcept = default;
    xsparse_container& operator=(const xsparse_container&) = default;
    xsparse_container& operator=(xsparse_container&&) noexcept = default;

    // #########################################################################
    // Expression assignment
    // #########################################################################
    template <class E>
    self_type& operator=(const xexpression<E>& e) {
        const E& expr = e.derived_cast();
        // If dense, convert to sparse by iterating over all elements
        if (expr.dimension() != dimension()) {
            XTU_THROW(std::runtime_error, "Dimension mismatch in sparse assignment");
        }
        // Clear current
        m_scheme = scheme_type(expr.shape().begin(), expr.shape().end());
        m_finalized = false;
        // Naive: iterate over dense and insert non-zeros
        // For large arrays, this is inefficient but correct.
        std::vector<index_type> coords(dimension());
        std::function<void(size_type, size_type)> assign_rec = [&](size_type dim, size_type offset) {
            if (dim == dimension()) {
                value_type val = expr.data()[offset];
                if (val != value_type(0)) {
                    m_scheme.insert(coords, val);
                }
                return;
            }
            for (index_type i = 0; i < static_cast<index_type>(expr.shape()[dim]); ++i) {
                coords[dim] = i;
                assign_rec(dim + 1, offset + i * expr.strides()[dim]);
            }
        };
        assign_rec(0, 0);
        finalize();
        return *this;
    }

    // Assignment from another sparse container
    self_type& operator=(const self_type& rhs) {
        if (this != &rhs) {
            m_scheme = rhs.m_scheme;
            m_finalized = rhs.m_finalized;
        }
        return *this;
    }

    // #########################################################################
    // Shape and capacity
    // #########################################################################
    const shape_type& shape() const noexcept { return m_scheme.shape(); }
    size_type dimension() const noexcept { return m_scheme.dimension(); }
    size_type size() const noexcept {
        return std::accumulate(shape().begin(), shape().end(), size_type(1), std::multiplies<size_type>());
    }
    size_type nnz() const noexcept { return m_scheme.nnz(); }
    bool is_finalized() const noexcept { return m_finalized; }

    // #########################################################################
    // Modifiers
    // #########################################################################
    void insert(const std::vector<index_type>& coords, const value_type& value) {
        XTU_ASSERT_MSG(!m_finalized, "Cannot insert after finalize()");
        m_scheme.insert(coords, value);
    }

    void finalize() {
        if (!m_finalized) {
            m_scheme.finalize();
            m_finalized = true;
        }
    }

    // #########################################################################
    // Element access (read-only)
    // #########################################################################
    value_type operator()(const std::vector<index_type>& coords) const {
        return m_scheme(coords);
    }

    template <class... Idx>
    value_type operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) > 0, "At least one index required");
        std::vector<index_type> coords = {static_cast<index_type>(idx)...};
        XTU_ASSERT_MSG(coords.size() == dimension(), "Number of indices must match dimension");
        return (*this)(coords);
    }

    // #########################################################################
    // Conversion to dense
    // #########################################################################
    template <class Container = xarray_container<value_type>>
    Container to_dense() const {
        Container dense(shape());
        m_scheme.to_dense(dense);
        return dense;
    }

    // #########################################################################
    // Iteration over non-zero elements
    // #########################################################################
    const_nnz_iterator nnz_begin() const { return const_nnz_iterator(this, 0); }
    const_nnz_iterator nnz_end() const { return const_nnz_iterator(this, nnz()); }

    // #########################################################################
    // Access to scheme (for advanced use)
    // #########################################################################
    scheme_type& scheme() noexcept { return m_scheme; }
    const scheme_type& scheme() const noexcept { return m_scheme; }

    // #########################################################################
    // Arithmetic operators (expression building)
    // #########################################################################
    // For integration with expression system, we would define xsparse_function etc.
    // For now, we provide in-place operations.
    self_type& operator+=(const self_type& rhs) {
        XTU_ASSERT_MSG(shape() == rhs.shape(), "Shape mismatch");
        finalize();
        auto rhs_copy = rhs;
        rhs_copy.finalize();
        *this = *this + rhs_copy;
        return *this;
    }

    self_type& operator-=(const self_type& rhs) {
        XTU_ASSERT_MSG(shape() == rhs.shape(), "Shape mismatch");
        finalize();
        auto rhs_copy = rhs;
        rhs_copy.finalize();
        *this = *this - rhs_copy;
        return *this;
    }

    self_type& operator*=(const value_type& scalar) {
        finalize();
        *this = scalar * (*this);
        return *this;
    }

    self_type& operator*=(const self_type& rhs) {
        XTU_ASSERT_MSG(shape() == rhs.shape(), "Shape mismatch");
        finalize();
        auto rhs_copy = rhs;
        rhs_copy.finalize();
        *this = (*this) * rhs_copy;
        return *this;
    }
};

// #############################################################################
// Convenience aliases
// #############################################################################
template <class T>
using xcoo_array = xsparse_container<T, index_type, xarray_expression_tag>;

template <class T, std::size_t N>
using xcoo_tensor = xsparse_container<T, index_type, xtensor_expression_tag>; // Fixed dimension not enforced at compile time

// #############################################################################
// Free functions for sparse-dense interactions
// #############################################################################

/// Convert dense to sparse (COO)
template <class E>
auto dense_to_sparse(const xexpression<E>& expr) {
    using value_type = typename E::value_type;
    const auto& e = expr.derived_cast();
    xsparse_container<value_type> sparse(e.shape().begin(), e.shape().end());
    // Iterate over dense and insert non-zeros
    std::vector<index_type> coords(e.dimension());
    std::function<void(size_type, size_type)> visit = [&](size_type dim, size_type offset) {
        if (dim == e.dimension()) {
            value_type val = e.data()[offset];
            if (val != value_type(0)) {
                sparse.insert(coords, val);
            }
            return;
        }
        for (index_type i = 0; i < static_cast<index_type>(e.shape()[dim]); ++i) {
            coords[dim] = i;
            visit(dim + 1, offset + i * e.strides()[dim]);
        }
    };
    visit(0, 0);
    sparse.finalize();
    return sparse;
}

/// Sparse matrix-vector multiplication (SpMV)
template <class T, class I, class E>
auto spmv(const xsparse_container<T, I>& A, const xexpression<E>& x) {
    XTU_ASSERT_MSG(A.dimension() == 2, "Matrix must be 2D");
    const auto& vec = x.derived_cast();
    XTU_ASSERT_MSG(vec.dimension() == 1 && vec.shape()[0] == A.shape()[1],
                   "Vector size must match matrix columns");
    using value_type = typename std::common_type<T, typename E::value_type>::type;
    xarray_container<value_type> y({A.shape()[0]}, value_type(0));
    A.finalize();
    const auto& storage = A.scheme().storage();
    const auto& indices = storage.indices();
    const auto& values = storage.values();
    size_type n_cols = A.shape()[1];
    for (size_type k = 0; k < A.nnz(); ++k) {
        size_type linear = static_cast<size_type>(indices[k]);
        size_type row = linear / n_cols;
        size_type col = linear % n_cols;
        y[row] += values[k] * vec[col];
    }
    return y;
}

/// Sparse matrix-matrix multiplication (SpGEMM) - returns dense or sparse
template <class T, class I, class E>
auto spmm(const xsparse_container<T, I>& A, const xexpression<E>& B) {
    XTU_ASSERT_MSG(A.dimension() == 2, "A must be matrix");
    const auto& matB = B.derived_cast();
    XTU_ASSERT_MSG(matB.dimension() == 2 && A.shape()[1] == matB.shape()[0],
                   "Dimension mismatch for matrix multiplication");
    using value_type = typename std::common_type<T, typename E::value_type>::type;
    xarray_container<value_type> C({A.shape()[0], matB.shape()[1]}, value_type(0));
    A.finalize();
    const auto& storage = A.scheme().storage();
    const auto& indices = storage.indices();
    const auto& values = storage.values();
    size_type n_cols_A = A.shape()[1];
    size_type n_cols_B = matB.shape()[1];
    for (size_type k = 0; k < A.nnz(); ++k) {
        size_type linear = static_cast<size_type>(indices[k]);
        size_type row = linear / n_cols_A;
        size_type col = linear % n_cols_A;
        value_type a_val = values[k];
        for (size_type j = 0; j < n_cols_B; ++j) {
            C(row, j) += a_val * matB(col, j);
        }
    }
    return C;
}

XTU_NAMESPACE_END

#endif // XTU_SPARSE_XSPARSE_CONTAINER_HPP