//File 0050 : core/xlayout.hpp
//Layout utilities with compile-time and runtime dispatch for row-major, column-major, and dynamic layouts, stride computation, and transposition helpers.
#ifndef XTENSOR_XLAYOUT_HPP
#define XTENSOR_XLAYOUT_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /*********************************************
     * Layout enumeration and compile-time traits
     *********************************************/
    template <layout_type L>
    struct layout_trait
    {
        static constexpr layout_type value = L;
    };

    using row_major_t = layout_trait<layout_type::row_major>;
    using column_major_t = layout_trait<layout_type::column_major>;
    using dynamic_t = layout_trait<layout_type::dynamic>;

    template <layout_type L>
    inline constexpr bool is_row_major_v = (L == layout_type::row_major);

    template <layout_type L>
    inline constexpr bool is_column_major_v = (L == layout_type::column_major);

    template <layout_type L>
    inline constexpr bool is_dynamic_v = (L == layout_type::dynamic);

    /*********************************************
     * Compile-time stride computation helpers
     *********************************************/
    namespace detail
    {
        template <layout_type L, class Shape, class Strides>
        inline void compute_strides_impl(const Shape& shape, Strides& strides)
        {
            if constexpr (L == layout_type::row_major)
            {
                if (shape.size() == 0) return;
                strides[shape.size() - 1] = 1;
                for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2; i >= 0; --i)
                {
                    strides[static_cast<std::size_t>(i)] = strides[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
                }
            }
            else if constexpr (L == layout_type::column_major)
            {
                if (shape.size() == 0) return;
                strides[0] = 1;
                for (std::size_t i = 1; i < shape.size(); ++i)
                {
                    strides[i] = strides[i - 1] * shape[i - 1];
                }
            }
            else
            {
                // dynamic layout – do nothing, strides must be provided externally
            }
        }

        template <layout_type L, class Shape>
        inline auto compute_strides_from_shape(const Shape& shape)
        {
            Shape strides = shape; // allocate same shape container
            compute_strides_impl<L>(shape, strides);
            return strides;
        }
    }

    /*********************************************
     * Runtime layout dispatching
     *********************************************/
    template <class Shape, class Strides>
    inline void compute_strides(const Shape& shape, layout_type l, Strides& strides)
    {
        if (l == layout_type::row_major)
        {
            detail::compute_strides_impl<layout_type::row_major>(shape, strides);
        }
        else if (l == layout_type::column_major)
        {
            detail::compute_strides_impl<layout_type::column_major>(shape, strides);
        }
        else
        {
            // dynamic: assume row-major default
            detail::compute_strides_impl<layout_type::row_major>(shape, strides);
        }
    }

    /*********************************************
     * Layout deduction from strides
     *********************************************/
    template <class Shape, class Strides>
    inline layout_type deduce_layout(const Shape& shape, const Strides& strides)
    {
        if (shape.size() <= 1)
        {
            return layout_type::row_major;
        }

        // Check row-major
        bool is_row = true;
        std::size_t expected = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i)
        {
            if (strides[static_cast<std::size_t>(i)] != expected)
            {
                is_row = false;
                break;
            }
            expected *= shape[static_cast<std::size_t>(i)];
        }
        if (is_row) return layout_type::row_major;

        // Check column-major
        bool is_col = true;
        expected = 1;
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (strides[i] != expected)
            {
                is_col = false;
                break;
            }
            expected *= shape[i];
        }
        if (is_col) return layout_type::column_major;

        return layout_type::dynamic;
    }

    /*********************************************
     * Layout-aware element offset computation
     *********************************************/
    template <layout_type L, class Index, class Strides>
    inline std::size_t element_offset(const Index& index, const Strides& strides)
    {
        std::size_t offset = 0;
        if constexpr (L == layout_type::row_major)
        {
            for (std::size_t i = 0; i < index.size(); ++i)
            {
                offset += index[i] * strides[i];
            }
        }
        else
        {
            for (std::size_t i = 0; i < index.size(); ++i)
            {
                offset += index[i] * strides[i];
            }
        }
        return offset;
    }

    /*********************************************
     * Unravel / ravel index with layout
     *********************************************/
    template <layout_type L, class Shape, class SizeType>
    inline Shape unravel_index(SizeType linear_index, const Shape& shape)
    {
        Shape result(shape.size());
        if constexpr (L == layout_type::row_major)
        {
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i)
            {
                result[static_cast<std::size_t>(i)] = linear_index % shape[static_cast<std::size_t>(i)];
                linear_index /= shape[static_cast<std::size_t>(i)];
            }
        }
        else
        {
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                result[i] = linear_index % shape[i];
                linear_index /= shape[i];
            }
        }
        return result;
    }

    template <layout_type L, class Index, class Strides>
    inline typename Index::value_type ravel_index(const Index& index, const Strides& strides)
    {
        using size_type = typename Index::value_type;
        size_type linear = 0;
        for (std::size_t i = 0; i < index.size(); ++i)
        {
            linear += index[i] * strides[i];
        }
        return linear;
    }

    /*********************************************
     * Transpose layout (swap row-major and column-major)
     *********************************************/
    constexpr layout_type transpose_layout(layout_type l) noexcept
    {
        if (l == layout_type::row_major) return layout_type::column_major;
        if (l == layout_type::column_major) return layout_type::row_major;
        return layout_type::dynamic;
    }

    /*********************************************
     * Layout-to-string conversion
     *********************************************/
    inline const char* layout_to_string(layout_type l) noexcept
    {
        switch (l)
        {
            case layout_type::row_major:    return "row_major";
            case layout_type::column_major: return "column_major";
            case layout_type::dynamic:      return "dynamic";
            default: return "unknown";
        }
    }

} // namespace xt

#endif // XTENSOR_XLAYOUT_HPP