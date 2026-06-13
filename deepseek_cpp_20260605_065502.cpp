//File 0048 : core/xexpression_holder.hpp
//Type-erased holder for any xtensor expression, providing polymorphic storage, cloning, and element access with SIMD-aware forwarding.
#ifndef XTENSOR_XEXPRESSION_HOLDER_HPP
#define XTENSOR_XEXPRESSION_HOLDER_HPP

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xexpression_traits.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xeval.hpp"

namespace xt
{
    namespace detail
    {
        // Abstract base for type erasure
        template <class T>
        class xexpression_holder_base
        {
        public:
            using value_type = T;
            using shape_type = std::vector<std::size_t>;

            virtual ~xexpression_holder_base() = default;

            virtual value_type element(const std::vector<std::size_t>& index) const = 0;
            virtual shape_type shape() const = 0;
            virtual std::size_t size() const = 0;
            virtual std::unique_ptr<xexpression_holder_base> clone() const = 0;
            virtual const value_type* data() const noexcept { return nullptr; }
        };

        // Concrete holder for any expression
        template <class E>
        class xexpression_holder_impl : public xexpression_holder_base<typename std::decay_t<E>::value_type>
        {
        public:
            using expr_type = std::decay_t<E>;
            using value_type = typename expr_type::value_type;
            using base_type = xexpression_holder_base<value_type>;

            explicit xexpression_holder_impl(E&& expr) noexcept
                : m_expression(std::forward<E>(expr))
            {
            }

            value_type element(const std::vector<std::size_t>& index) const override
            {
                return m_expression.element(index.begin(), index.end());
            }

            typename base_type::shape_type shape() const override
            {
                auto s = m_expression.shape();
                return typename base_type::shape_type(s.begin(), s.end());
            }

            std::size_t size() const override
            {
                return m_expression.size();
            }

            std::unique_ptr<base_type> clone() const override
            {
                return std::make_unique<xexpression_holder_impl>(m_expression);
            }

            const value_type* data() const noexcept override
            {
                return m_expression.data();
            }

        private:
            expr_type m_expression;
        };
    }

    /**
     * @class xexpression_holder
     * @brief Type‑erased holder for any xtensor expression.
     *
     * Stores any expression (e.g., xarray, xfunction) polymorphically,
     * enabling virtual dispatch for shape, size, and element access.
     * Supports copy, move, and cloning.
     */
    template <class T>
    class xexpression_holder : public xexpression<xexpression_holder<T>>
    {
    public:
        using self_type = xexpression_holder<T>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = T;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;

        xexpression_holder() = default;

        /**
         * Construct from any expression, erasing its type.
         */
        template <class E, XTL_REQUIRES(is_xexpression_v<E>)>
        xexpression_holder(E&& e)
            : m_holder(std::make_unique<detail::xexpression_holder_impl<E>>(std::forward<E>(e)))
        {
        }

        xexpression_holder(const self_type& rhs)
            : m_holder(rhs.m_holder ? rhs.m_holder->clone() : nullptr)
        {
        }

        xexpression_holder& operator=(const self_type& rhs)
        {
            if (this != &rhs)
            {
                m_holder = rhs.m_holder ? rhs.m_holder->clone() : nullptr;
            }
            return *this;
        }

        xexpression_holder(self_type&&) = default;
        xexpression_holder& operator=(self_type&&) = default;

        size_type size() const noexcept { return m_holder ? m_holder->size() : 0; }
        shape_type shape() const noexcept { return m_holder ? m_holder->shape() : shape_type{}; }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).operator()(args...));
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            if (!m_holder)
                throw std::runtime_error("Accessing empty xexpression_holder.");
            std::vector<size_type> idx(first, last);
            // Cache value? We need to return a reference but holder returns by value.
            // We'll store a mutable cache for such cases.
            m_cached_value = m_holder->element(idx);
            return m_cached_value;
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return m_holder ? const_cast<pointer>(m_holder->data()) : nullptr; }
        const_pointer data() const noexcept { return m_holder ? m_holder->data() : nullptr; }

    private:
        std::unique_ptr<detail::xexpression_holder_base<T>> m_holder;
        mutable T m_cached_value = T{};
    };

    template <class T>
    struct xcontainer_inner_types<xexpression_holder<T>>
    {
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<T>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

} // namespace xt

#endif // XTENSOR_XEXPRESSION_HOLDER_HPP