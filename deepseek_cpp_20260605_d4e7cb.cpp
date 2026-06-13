//File 0321 : xframe/xframe_adaptor.hpp
//Adaptor class that wraps an existing 2D range of data into an xframe without copying, supporting SIMD access and full expression integration.
#ifndef XFRAME_ADAPTOR_HPP
#define XFRAME_ADAPTOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_dimension.hpp"
#include "xframe_variable.hpp"
#include "xframe.hpp"

namespace xframe
{
    /**
     * @class xframe_adaptor
     * @brief Adapts external memory (e.g., a flat C array) into an xframe.
     *
     * The adaptor does not own the data. It provides a non‑owning view
     * with two dimensions and a single variable. The user is responsible
     * for ensuring that the underlying memory remains valid during the
     * lifetime of the adaptor.
     */
    template <class T = double, class L = label_type>
    class xframe_adaptor : public expression<xframe_adaptor<T, L>>
    {
    public:
        using self_type = xframe_adaptor<T, L>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using label_type = L;

        /**
         * Construct an adaptor from a raw pointer, rows, cols, and dimension names.
         * @param data Pointer to the first element (row‑major).
         * @param rows Number of rows.
         * @param cols Number of columns.
         * @param row_dim_name Name of the row dimension.
         * @param col_dim_name Name of the column dimension.
         * @param var_name Name of the variable.
         */
        xframe_adaptor(pointer data, size_type rows, size_type cols,
                       const label_type& row_dim_name = label_type("rows"),
                       const label_type& col_dim_name = label_type("cols"),
                       const label_type& var_name = label_type("value"))
            : m_data(data), m_rows(rows), m_cols(cols)
            , m_row_dim(row_dim_name, rows)
            , m_col_dim(col_dim_name, cols)
            , m_var_name(var_name)
        {
        }

        /**
         * Construct from a vector (which the adaptor may reference as long as
         * the vector is alive). This constructor takes a const reference to
         * a vector, but the adaptor itself is non‑owning – the caller must
         * ensure the vector outlives the adaptor.
         */
        xframe_adaptor(const std::vector<T>& data, size_type rows, size_type cols,
                       const label_type& row_dim_name = label_type("rows"),
                       const label_type& col_dim_name = label_type("cols"),
                       const label_type& var_name = label_type("value"))
            : m_data(const_cast<T*>(data.data())), m_rows(rows), m_cols(cols)
            , m_row_dim(row_dim_name, rows)
            , m_col_dim(col_dim_name, cols)
            , m_var_name(var_name)
        {
            if (data.size() != rows * cols)
                throw std::runtime_error("xframe_adaptor: data size must equal rows*cols.");
        }

        xframe_adaptor(const self_type&) = default;
        xframe_adaptor& operator=(const self_type&) = default;
        xframe_adaptor(self_type&&) = default;
        xframe_adaptor& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return 2; }
        std::size_t size() const noexcept { return m_rows * m_cols; }

        const dimension<label_type>& dimension(std::size_t i) const
        {
            if (i == 0) return m_row_dim;
            if (i == 1) return m_col_dim;
            throw std::out_of_range("xframe_adaptor::dimension: index out of range.");
        }

        reference operator()(size_type row, size_type col) { return m_data[row * m_cols + col]; }
        const_reference operator()(size_type row, size_type col) const { return m_data[row * m_cols + col]; }

        reference operator[](size_type flat) { return m_data[flat]; }
        const_reference operator[](size_type flat) const { return m_data[flat]; }

        template <class... Args>
        reference locate(Args... args)
        {
            std::array<label_type, sizeof...(Args)> labels{args...};
            return (*this)(m_row_dim.index_of(labels[0]), m_col_dim.index_of(labels[1]));
        }

        template <class... Args>
        const_reference locate(Args... args) const
        {
            std::array<label_type, sizeof...(Args)> labels{args...};
            return (*this)(m_row_dim.index_of(labels[0]), m_col_dim.index_of(labels[1]));
        }

        pointer data() noexcept { return m_data; }
        const_pointer data() const noexcept { return m_data; }

        const label_type& variable_name() const noexcept { return m_var_name; }

        // SIMD load
        template <class Align, class U = T>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<U, default_simd_arch>;
            return simd_type::load_unaligned(m_data + i);
        }

    private:
        pointer m_data;
        size_type m_rows;
        size_type m_cols;
        dimension<label_type> m_row_dim;
        dimension<label_type> m_col_dim;
        label_type m_var_name;
    };

    /**
     * Helper function to wrap a raw pointer into an xframe adaptor.
     */
    template <class T = double, class L = label_type>
    inline auto adapt(T* data, std::size_t rows, std::size_t cols,
                      const L& row_name = L("rows"),
                      const L& col_name = L("cols"),
                      const L& var_name = L("value"))
    {
        return xframe_adaptor<T, L>(data, rows, cols, row_name, col_name, var_name);
    }

    template <class T = double, class L = label_type>
    inline auto adapt(const std::vector<T>& data, std::size_t rows, std::size_t cols,
                      const L& row_name = L("rows"),
                      const L& col_name = L("cols"),
                      const L& var_name = L("value"))
    {
        return xframe_adaptor<T, L>(data, rows, cols, row_name, col_name, var_name);
    }

} // namespace xframe

#endif // XFRAME_ADAPTOR_HPP