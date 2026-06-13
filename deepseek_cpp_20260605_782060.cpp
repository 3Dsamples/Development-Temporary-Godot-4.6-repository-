//File 0305 : xframe/xframe_expression.hpp
//Base expression CRTP class for all xframe expressions, providing common interface, shape, dimension access, and broadcasting support.
#ifndef XFRAME_EXPRESSION_HPP
#define XFRAME_EXPRESSION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <string>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"

namespace xframe
{
    /**
     * @class expression
     * @brief CRTP base class for all xframe expressions.
     *
     * Provides a uniform interface for xframe arrays, views, and functions.
     * Derived classes must implement derived(), dimension(), size(), and
     * element access methods.
     */
    template <class D>
    class expression
    {
    public:
        using derived_type = D;

        derived_type& derived() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived() const noexcept { return *static_cast<const derived_type*>(this); }

        /**
         * Number of dimensions.
         */
        std::size_t dimension_count() const noexcept { return derived().dimension_count(); }

        /**
         * Total number of elements.
         */
        std::size_t size() const noexcept { return derived().size(); }

        /**
         * Access element by flat index.
         */
        decltype(auto) operator[](std::size_t i) { return derived()[i]; }
        decltype(auto) operator[](std::size_t i) const { return derived()[i]; }

        /**
         * Access element by variadic coordinates.
         */
        template <class... Args>
        decltype(auto) operator()(Args... args) { return derived()(args...); }
        template <class... Args>
        decltype(auto) operator()(Args... args) const { return derived()(args...); }

        /**
         * Access element by label (string).
         */
        template <class... Args>
        decltype(auto) locate(Args... args) { return derived().locate(args...); }
        template <class... Args>
        decltype(auto) locate(Args... args) const { return derived().locate(args...); }

        /**
         * Get the dimension descriptor at index.
         */
        decltype(auto) dimension(std::size_t i) const { return derived().dimension(i); }

        /**
         * Return the shape as a vector of sizes.
         */
        auto shape() const
        {
            std::vector<std::size_t> s(dimension_count());
            for (std::size_t i = 0; i < s.size(); ++i)
                s[i] = dimension(i).size();
            return s;
        }

        /**
         * Check if two expressions have the same dimension structure.
         */
        template <class E>
        bool same_dimensions(const expression<E>& other) const
        {
            if (dimension_count() != other.dimension_count()) return false;
            for (std::size_t i = 0; i < dimension_count(); ++i)
                if (dimension(i).name() != other.dimension(i).name() ||
                    dimension(i).size() != other.dimension(i).size())
                    return false;
            return true;
        }

    protected:
        expression() = default;
        ~expression() = default;
        expression(const expression&) = default;
        expression& operator=(const expression&) = default;
        expression(expression&&) = default;
        expression& operator=(expression&&) = default;
    };

} // namespace xframe

#endif // XFRAME_EXPRESSION_HPP