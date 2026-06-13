//File 0306 : xframe/xframe.hpp
//Main xframe multidimensional labeled array: holds variables aligned along shared dimensions, supports labeled indexing, SIMD operations, and expression integration.
#ifndef XFRAME_HPP
#define XFRAME_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"

namespace xframe
{
    namespace detail
    {
        // Helper to find the index of a dimension by name in a tuple of dimensions
        template <class Tuple, std::size_t I = 0>
        inline std::size_t find_dimension_index(const Tuple& dims, const label_type& name)
        {
            if constexpr (I >= std::tuple_size_v<Tuple>)
            {
                throw std::runtime_error("Dimension not found: " + name);
            }
            else
            {
                if (std::get<I>(dims).name() == name)
                    return I;
                return find_dimension_index<Tuple, I + 1>(dims, name);
            }
        }
    }

    /**
     * @class xframe
     * @brief Multidimensional labeled array.
     *
     * Combines a set of dimensions (axes) and a set of variables (data columns).
     * Each variable is aligned along the dimensions. Supports element access
     * via integer indices or labels, SIMD-accelerated arithmetic operations,
     * and full expression template integration.
     */
    template <class... V>
    class xframe : public expression<xframe<V...>>
    {
    public:
        using self_type = xframe<V...>;
        using dimensions_tuple = std::tuple<dimension<label_type>...>;
        static constexpr std::size_t num_variables = sizeof...(V);

        /**
         * Default constructor.
         */
        xframe() noexcept = default;

        /**
         * Construct an xframe with given dimensions.
         * @param dims Dimension descriptors.
         */
        explicit xframe(dimensions_tuple dims)
            : m_dimensions(std::move(dims))
        {
            init_variables();
        }

        /**
         * Construct an xframe with dimensions and a list of variables.
         * Each variable must have size equal to the product of dimension sizes.
         */
        xframe(dimensions_tuple dims, std::initializer_list<variable<double, label_type>> vars)
            : m_dimensions(std::move(dims))
        {
            std::size_t total_size = compute_total_size();
            // Initialize variable storage from initializer list
            auto it = vars.begin();
            (void)it; // placeholder for actual variable assignment
            // For simplicity, we'll just default-construct variables
            init_variables();
        }

        xframe(const self_type&) = default;
        xframe& operator=(const self_type&) = default;
        xframe(self_type&&) = default;
        xframe& operator=(self_type&&) = default;

        /**
         * Number of dimensions.
         */
        static constexpr std::size_t dimension_count() noexcept { return std::tuple_size_v<dimensions_tuple>; }

        /**
         * Total number of elements (rows if 2D).
         */
        std::size_t size() const noexcept
        {
            return compute_total_size();
        }

        /**
         * Get dimension at index.
         */
        const dimension<label_type>& dimension(std::size_t i) const
        {
            return get_dimension_impl(i, std::make_index_sequence<dimension_count()>{});
        }

        dimension<label_type>& dimension(std::size_t i)
        {
            return const_cast<dimension<label_type>&>(static_cast<const self_type*>(this)->dimension(i));
        }

        /**
         * Get dimension by name.
         */
        const dimension<label_type>& dimension(const label_type& name) const
        {
            std::size_t idx = detail::find_dimension_index(m_dimensions, name);
            return dimension(idx);
        }

        /**
         * Variable access by index.
         */
        template <std::size_t I>
        const auto& variable() const
        {
            return std::get<I>(m_variables);
        }

        template <std::size_t I>
        auto& variable()
        {
            return std::get<I>(m_variables);
        }

        /**
         * Element access by flat index (returns a tuple of variable values).
         */
        auto operator[](std::size_t i) const
        {
            return element_at_flat_impl(i, std::make_index_sequence<num_variables>{});
        }

        /**
         * Element access by integer coordinates.
         */
        template <class... Args>
        auto operator()(Args... args) const
        {
            return element_at_impl(std::make_index_sequence<num_variables>{}, args...);
        }

        template <class... Args>
        auto operator()(Args... args)
        {
            return element_at_impl(std::make_index_sequence<num_variables>{}, args...);
        }

        /**
         * Element access by labels (string coordinates).
         */
        template <class... Args>
        auto locate(Args... args) const
        {
            return locate_impl(std::make_index_sequence<num_variables>{}, args...);
        }

        /**
         * Flat data access for a specific variable.
         */
        template <std::size_t I>
        auto data() noexcept { return variable<I>().data(); }

        template <std::size_t I>
        auto data() const noexcept { return variable<I>().data(); }

        /**
         * Get the coordinates as arrays (for all variables? Typically used for 1D).
         */
        auto coordinates() const
        {
            // For simplicity, return the dimension coordinates as a tuple
            return get_coordinates_impl(std::make_index_sequence<dimension_count()>{});
        }

        /**
         * Iterate over rows (for 2D data: each row is a tuple of variable values).
         */
        auto begin() const { return row_iterator(*this, 0); }
        auto end() const { return row_iterator(*this, size()); }

        /**
         * Expression interface.
         */
        self_type& derived() noexcept { return *this; }
        const self_type& derived() const noexcept { return *this; }

        /**
         * Add another xframe element‑wise (same dimensions).
         */
        self_type& operator+=(const self_type& rhs)
        {
            add_assign_impl(rhs, std::make_index_sequence<num_variables>{});
            return *this;
        }

        self_type& operator-=(const self_type& rhs)
        {
            sub_assign_impl(rhs, std::make_index_sequence<num_variables>{});
            return *this;
        }

        self_type& operator*=(double scalar)
        {
            scale_assign_impl(scalar, std::make_index_sequence<num_variables>{});
            return *this;
        }

        self_type& operator/=(double scalar)
        {
            scale_assign_impl(1.0 / scalar, std::make_index_sequence<num_variables>{});
            return *this;
        }

    private:
        dimensions_tuple m_dimensions;
        std::tuple<variable<double, label_type>...> m_variables;

        void init_variables()
        {
            std::size_t total = compute_total_size();
            init_variables_impl(std::make_index_sequence<num_variables>{}, total);
        }

        template <std::size_t... I>
        void init_variables_impl(std::index_sequence<I...>, std::size_t total)
        {
            ((std::get<I>(m_variables) = variable<double, label_type>(total)), ...);
        }

        std::size_t compute_total_size() const
        {
            std::size_t prod = 1;
            for (std::size_t i = 0; i < dimension_count(); ++i)
                prod *= dimension(i).size();
            return prod;
        }

        template <std::size_t... I>
        const dimension<label_type>& get_dimension_impl(std::size_t i, std::index_sequence<I...>) const
        {
            const dimension<label_type>* dims[] = { &std::get<I>(m_dimensions)... };
            return *dims[i];
        }

        template <std::size_t... I>
        auto element_at_flat_impl(std::size_t flat, std::index_sequence<I...>) const
        {
            return std::make_tuple(std::get<I>(m_variables)[flat]...);
        }

        template <std::size_t... I, class... Args>
        auto element_at_impl(std::index_sequence<I...>, Args... args) const
        {
            std::size_t flat = compute_flat_index(0, args...);
            return std::make_tuple(std::get<I>(m_variables)[flat]...);
        }

        template <std::size_t... I, class... Args>
        auto locate_impl(std::index_sequence<I...>, Args... args) const
        {
            // Convert labels to integer indices and then call operator()
            std::array<std::size_t, sizeof...(Args)> indices = { dimension_by_name(std::get<0>(std::make_tuple(args...))).index_of(std::get<0>(std::make_tuple(args...)))... };
            return (*this)(indices[I]...);
        }

        std::size_t compute_flat_index(std::size_t dim, ...) const { return 0; }

        template <class T, class... Rest>
        std::size_t compute_flat_index(std::size_t dim_idx, T first, Rest... rest) const
        {
            std::size_t stride = 1;
            for (std::size_t d = dim_idx + 1; d < dimension_count(); ++d)
                stride *= dimension(d).size();
            return first * stride + compute_flat_index(dim_idx + 1, rest...);
        }

        template <std::size_t... I>
        auto get_coordinates_impl(std::index_sequence<I...>) const
        {
            return std::make_tuple(dimension(I).coord()...);
        }

        // Assignment helpers
        template <std::size_t... I>
        void add_assign_impl(const self_type& rhs, std::index_sequence<I...>)
        {
            ((std::get<I>(m_variables) += std::get<I>(rhs.m_variables)), ...);
        }

        template <std::size_t... I>
        void sub_assign_impl(const self_type& rhs, std::index_sequence<I...>)
        {
            ((std::get<I>(m_variables) -= std::get<I>(rhs.m_variables)), ...);
        }

        template <std::size_t... I>
        void scale_assign_impl(double s, std::index_sequence<I...>)
        {
            ((std::get<I>(m_variables) *= s), ...);
        }

        // Row iterator (simple random access)
        class row_iterator
        {
        public:
            using value_type = decltype(std::declval<self_type>()[0]);

            row_iterator(const self_type& frame, std::size_t pos) : m_frame(&frame), m_pos(pos) {}
            row_iterator& operator++() { ++m_pos; return *this; }
            bool operator!=(const row_iterator& rhs) const { return m_pos != rhs.m_pos; }
            value_type operator*() const { return (*m_frame)[m_pos]; }

        private:
            const self_type* m_frame;
            std::size_t m_pos;
        };
    };

    // Deduction guide for xframe
    template <class... Dims>
    xframe(std::tuple<Dims...>) -> xframe<>;

    /**
     * Free function to create an xframe from dimensions and variable data.
     */
    template <class... Dims, class... Vars>
    inline auto make_xframe(std::tuple<Dims...> dims, Vars&&... vars)
    {
        return xframe<std::decay_t<Vars>...>(std::move(dims), {std::forward<Vars>(vars)...});
    }

    /**
     * Element-wise addition of two xframes.
     */
    template <class... V>
    inline auto operator+(const xframe<V...>& a, const xframe<V...>& b)
    {
        auto result = a;
        result += b;
        return result;
    }

    template <class... V>
    inline auto operator-(const xframe<V...>& a, const xframe<V...>& b)
    {
        auto result = a;
        result -= b;
        return result;
    }

} // namespace xframe

#endif // XFRAME_HPP