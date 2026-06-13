//File 0005 (UPDATED) : core/xstrides.hpp
//Complete stride computation, broadcasting, index conversion, memory alignment, dynamic layout, and adapt_strides with C++17.
#ifndef XTENSOR_XSTRIDES_HPP
#define XTENSOR_XSTRIDES_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "xmath.hpp"
#include "xfunction.hpp"
#include "xsemantic.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /***********************************
     * Stride computation utilities
     ***********************************/

    namespace detail
    {
        template <class shape_type>
        inline auto compute_strides_row_major(const shape_type& shape)
        {
            using size_type = typename shape_type::value_type;
            shape_type strides(shape.size());
            if (shape.empty()) return strides;
            strides[shape.size() - 1] = size_type(1);
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2; i >= 0; --i)
            {
                strides[static_cast<std::size_t>(i)] = strides[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
            }
            return strides;
        }

        template <class shape_type>
        inline auto compute_strides_column_major(const shape_type& shape)
        {
            using size_type = typename shape_type::value_type;
            shape_type strides(shape.size());
            if (shape.empty()) return strides;
            strides[0] = size_type(1);
            for (std::size_t i = 1; i < shape.size(); ++i)
            {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
            return strides;
        }

        template <class strides_type, class shape_type>
        inline auto compute_backstrides(const strides_type& strides, const shape_type& shape)
        {
            using size_type = typename strides_type::value_type;
            strides_type backstrides(shape.size());
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                backstrides[i] = (shape[i] - size_type(1)) * strides[i];
            }
            return backstrides;
        }
    }

    /**
     * Return the total number of elements in a shape.
     */
    template <class S>
    inline auto compute_size(const S& shape)
    {
        using size_type = typename S::value_type;
        return std::accumulate(shape.begin(), shape.end(), size_type(1), std::multiplies<size_type>());
    }

    /**
     * Compute strides for a given shape and layout (compile-time).
     */
    template <layout_type L, class shape_type>
    inline auto compute_strides(const shape_type& shape)
    {
        if constexpr (L == layout_type::row_major)
        {
            return detail::compute_strides_row_major(shape);
        }
        else if constexpr (L == layout_type::column_major)
        {
            return detail::compute_strides_column_major(shape);
        }
        else
        {
            static_assert(L == layout_type::row_major || L == layout_type::column_major, "Unsupported layout");
            return shape;
        }
    }

    /**
     * Compute strides for a shape with a given layout (runtime).
     */
    template <class shape_type>
    inline auto compute_strides(const shape_type& shape, layout_type l)
    {
        if (l == layout_type::row_major)
        {
            return detail::compute_strides_row_major(shape);
        }
        else
        {
            return detail::compute_strides_column_major(shape);
        }
    }

    /**
     * Adapt strides to an existing container size (used for reshaping).
     */
    template <class shape_type, class strides_type>
    inline void adapt_strides(const shape_type& new_shape, strides_type& strides, layout_type l)
    {
        strides = compute_strides(new_shape, l);
    }

    /**
     * Compute strides for a dynamic layout (e.g., from existing strides), reverse-engineer.
     */
    template <class shape_type, class strides_type>
    inline layout_type deduce_layout(const shape_type& shape, const strides_type& strides)
    {
        if (shape.size() <= 1) return layout_type::row_major;
        // Check if row-major: last stride == 1, each stride[i] == shape[i+1]*stride[i+1]
        auto row_strides = detail::compute_strides_row_major(shape);
        if (strides == row_strides) return layout_type::row_major;
        auto col_strides = detail::compute_strides_column_major(shape);
        if (strides == col_strides) return layout_type::column_major;
        return layout_type::dynamic;
    }

    /***********************************
     * Broadcasting utilities
     ***********************************/

    namespace detail
    {
        template <class S>
        inline S broadcast_shape_impl(const S& s1, const S& s2)
        {
            if (s1.empty()) return s2;
            if (s2.empty()) return s1;
            if (s1.size() != s2.size())
                throw std::runtime_error("Broadcast shape mismatch: dimensions differ.");
            S result(s1.size());
            for (std::size_t i = 0; i < s1.size(); ++i)
            {
                if (s1[i] == 1) result[i] = s2[i];
                else if (s2[i] == 1) result[i] = s1[i];
                else if (s1[i] == s2[i]) result[i] = s1[i];
                else throw std::runtime_error("Broadcast shape mismatch: incompatible dimensions.");
            }
            return result;
        }

        template <class S>
        inline S broadcast_shape_variadic(const S& first) { return first; }

        template <class S, class... Shapes>
        inline S broadcast_shape_variadic(const S& first, const S& second, const Shapes&... rest)
        {
            return broadcast_shape_variadic(broadcast_shape_impl(first, second), rest...);
        }
    }

    /**
     * Compute the broadcast shape for any number of shapes.
     */
    template <class S1, class S2, class... S>
    inline S1 broadcast_shape(const S1& s1, const S2& s2, const S&... rest)
    {
        return detail::broadcast_shape_variadic(s1, s2, rest...);
    }

    /**
     * Broadcast strides: compute new strides for a broadcasted shape given original strides.
     */
    template <class shape_type, class strides_type>
    inline strides_type broadcast_strides(const shape_type& old_shape, const strides_type& old_strides,
                                          const shape_type& new_shape)
    {
        strides_type new_strides(new_shape.size());
        std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(new_shape.size()) - static_cast<std::ptrdiff_t>(old_shape.size());
        if (offset < 0) throw std::runtime_error("Cannot broadcast to smaller number of dimensions.");
        // Leading new dimensions get stride 0
        for (std::size_t i = 0; i < static_cast<std::size_t>(offset); ++i)
            new_strides[i] = 0;
        for (std::size_t i = 0; i < old_shape.size(); ++i)
        {
            if (old_shape[i] == 1) new_strides[i + static_cast<std::size_t>(offset)] = 0;
            else new_strides[i + static_cast<std::size_t>(offset)] = old_strides[i];
        }
        return new_strides;
    }

    /***********************************
     * Index conversion utilities
     ***********************************/

    template <class S, class size_type>
    inline S unravel_index(size_type index, const S& shape, layout_type l = DEFAULT_LAYOUT)
    {
        S multi_index(shape.size());
        if (l == layout_type::row_major)
        {
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i)
            {
                multi_index[static_cast<std::size_t>(i)] = index % shape[static_cast<std::size_t>(i)];
                index /= shape[static_cast<std::size_t>(i)];
            }
        }
        else
        {
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                multi_index[i] = index % shape[i];
                index /= shape[i];
            }
        }
        return multi_index;
    }

    template <class S>
    inline typename S::value_type ravel_index(const S& index, const S& strides)
    {
        using size_type = typename S::value_type;
        size_type result = 0;
        for (std::size_t i = 0; i < index.size(); ++i)
            result += index[i] * strides[i];
        return result;
    }

    /***********************************
     * SIMD alignment utilities
     ***********************************/

    inline std::size_t simd_aligned_offset(std::size_t current_size, std::size_t alignment = 64)
    {
        std::size_t remainder = current_size % alignment;
        if (remainder == 0) return 0;
        return alignment - remainder;
    }

    constexpr std::size_t align_to(std::size_t value, std::size_t alignment) noexcept
    {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    template <class container_type>
    inline void enforce_simd_alignment(container_type& data, std::size_t alignment = 64)
    {
        if (data.size() % alignment != 0)
            data.resize(align_to(data.size(), alignment), typename container_type::value_type(0));
    }

    /***********************************
     * xstrided_container_base
     ***********************************/

    template <class D>
    class xstrided_container : public xcontainer_semantic<D>
    {
    public:
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;

        size_type size() const { return compute_size(derived_cast().shape()); }
        size_type dimension() const { return derived_cast().shape().size(); }

        const shape_type& shape() const { return derived_cast().shape(); }
        const strides_type& strides() const { return derived_cast().strides(); }
        const backstrides_type& backstrides() const { return derived_cast().backstrides(); }

        layout_type layout() const noexcept { return inner_types::layout; }

        void set_shape(const shape_type& s) { derived_cast().set_shape(s); }
        void set_strides(const strides_type& st) { derived_cast().set_strides(st); }

        template <class... Args>
        reference operator()(Args... args) { return element(args...); }
        template <class... Args>
        const_reference operator()(Args... args) const { return element(args...); }
        template <class... Args>
        reference at(Args... args) { check_access(derived_cast().shape(), args...); return this->operator()(args...); }
        template <class... Args>
        const_reference at(Args... args) const { check_access(derived_cast().shape(), args...); return this->operator()(args...); }
        template <class... Args>
        reference periodic(Args... args) { adjust_periodic(derived_cast().shape(), args...); return this->operator()(args...); }
        template <class... Args>
        const_reference periodic(Args... args) const { adjust_periodic(derived_cast().shape(), args...); return this->operator()(args...); }

        reference operator[](size_type i) { return derived_cast().data()[i]; }
        const_reference operator[](size_type i) const { return derived_cast().data()[i]; }

        pointer data() noexcept { return derived_cast().data(); }
        const_pointer data() const noexcept { return derived_cast().data(); }

        using iterator = pointer;
        using const_iterator = const_pointer;
        iterator begin() noexcept { return data(); }
        iterator end() noexcept { return data() + size(); }
        const_iterator begin() const noexcept { return data(); }
        const_iterator end() const noexcept { return data() + size(); }
        const_iterator cbegin() const noexcept { return data(); }
        const_iterator cend() const noexcept { return data() + size(); }

    protected:
        xstrided_container() = default;
        ~xstrided_container() = default;
        xstrided_container(const xstrided_container&) = default;
        xstrided_container& operator=(const xstrided_container&) = default;
        xstrided_container(xstrided_container&&) = default;
        xstrided_container& operator=(xstrided_container&&) = default;

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }

        template <class It>
        reference element(It first, It last)
        {
            auto index = std::vector<size_type>(first, last);
            size_type linear_index = ravel_index(index, derived_cast().strides());
            return derived_cast().data()[linear_index];
        }
        template <class It>
        const_reference element(It first, It last) const
        {
            auto index = std::vector<size_type>(first, last);
            size_type linear_index = ravel_index(index, derived_cast().strides());
            return derived_cast().data()[linear_index];
        }

    private:
        template <class S, class... Args>
        static void check_access(const S& shape, Args... args)
        {
            auto index = std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...};
            if (sizeof...(Args) != shape.size()) throw std::out_of_range("Index dimension mismatch.");
            for (std::size_t i = 0; i < shape.size(); ++i)
                if (index[i] >= shape[i]) throw std::out_of_range("Index out of bounds.");
        }
        template <class S, class... Args>
        static void adjust_periodic(S& shape, Args&... args)
        {
            auto index = std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...};
            for (std::size_t i = 0; i < sizeof...(Args); ++i)
                if (shape[i] > 0) index[i] = ((index[i] % shape[i]) + shape[i]) % shape[i];
            std::size_t pos = 0;
            ((args = index[pos++]), ...);
        }
    };

} // namespace xt

#endif // XTENSOR_XSTRIDES_HPP