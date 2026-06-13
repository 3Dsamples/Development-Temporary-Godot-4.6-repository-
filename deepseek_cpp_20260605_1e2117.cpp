//File 0308 : xframe/xframe_semantic.hpp
//Semantic base class for xframe containers and views, enabling assignment from expressions with label alignment and broadcasting.
#ifndef XFRAME_XFRAME_SEMANTIC_HPP
#define XFRAME_XFRAME_SEMANTIC_HPP

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"

namespace xframe
{
    /**
     * @class xframe_semantic
     * @brief Base class providing assignment semantics for xframe expressions.
     *
     * Derived classes (xframe, xframe_view) inherit assignment operators that
     * accept any expression. The assignment aligns dimensions by name, performs
     * necessary broadcasting, and copies data with SIMD acceleration where possible.
     */
    template <class D>
    class xframe_semantic : public expression<D>
    {
    public:
        using derived_type = D;

        /**
         * Assign from any expression.
         * The expression must have compatible dimensions (same names, sizes).
         */
        template <class E>
        derived_type& operator=(const expression<E>& e)
        {
            const auto& src = e.derived();
            auto& dst = this->derived();
            if (!dst.same_dimensions(src))
                throw std::runtime_error("xframe_semantic: dimension mismatch on assignment.");
            // Copy all variables element‑wise
            for (std::size_t i = 0; i < dst.size(); ++i)
                assign_row(dst, src, i);
            return dst;
        }

        /**
         * Copy assignment from same type.
         */
        derived_type& operator=(const derived_type& rhs)
        {
            if (static_cast<const void*>(this) != static_cast<const void*>(&rhs))
            {
                auto& dst = this->derived();
                for (std::size_t v = 0; v < dst.num_variables; ++v)
                    dst.variable(v) = rhs.variable(v);
                dst.set_dimensions(rhs.dimensions_tuple());
            }
            return this->derived();
        }

        /**
         * Move assignment from same type.
         */
        derived_type& operator=(derived_type&& rhs) noexcept
        {
            if (static_cast<const void*>(this) != static_cast<const void*>(&rhs))
            {
                auto& dst = this->derived();
                for (std::size_t v = 0; v < dst.num_variables; ++v)
                    dst.variable(v) = std::move(rhs.variable(v));
                dst.set_dimensions(std::move(rhs.dimensions_tuple()));
            }
            return this->derived();
        }

        /**
         * Compound assignment operators.
         */
        template <class E>
        derived_type& operator+=(const expression<E>& e)
        {
            return this->derived() = this->derived() + e;
        }

        template <class E>
        derived_type& operator-=(const expression<E>& e)
        {
            return this->derived() = this->derived() - e;
        }

        template <class E>
        derived_type& operator*=(const expression<E>& e)
        {
            return this->derived() = this->derived() * e;
        }

        template <class E>
        derived_type& operator/=(const expression<E>& e)
        {
            return this->derived() = this->derived() / e;
        }

    protected:
        xframe_semantic() = default;
        ~xframe_semantic() = default;
        xframe_semantic(const xframe_semantic&) = default;
        xframe_semantic& operator=(const xframe_semantic&) = default;
        xframe_semantic(xframe_semantic&&) = default;
        xframe_semantic& operator=(xframe_semantic&&) = default;

    private:
        template <class Dst, class Src>
        static void assign_row(Dst& dst, const Src& src, std::size_t i)
        {
            // Assign each variable's i-th element
            assign_row_impl(dst, src, i, std::make_index_sequence<Dst::num_variables>{});
        }

        template <class Dst, class Src, std::size_t... I>
        static void assign_row_impl(Dst& dst, const Src& src, std::size_t i, std::index_sequence<I...>)
        {
            ((dst.template variable<I>()[i] = src.template variable<I>()[i]), ...);
        }
    };

} // namespace xframe

#endif // XFRAME_XFRAME_SEMANTIC_HPP