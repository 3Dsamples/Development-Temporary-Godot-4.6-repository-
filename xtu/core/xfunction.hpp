// core/xfunction.hpp

#ifndef XTENSOR_XFUNCTION_HPP
#define XTENSOR_XFUNCTION_HPP

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"

#include <cstddef>
#include <type_traits>
#include <functional>
#include <utility>
#include <tuple>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stdexcept>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // xfunction_base - Base class for function expressions
        // --------------------------------------------------------------------
        template <class F, class R, class... CT>
        class xfunction_base : public xexpression<xfunction_base<F, R, CT...>>
        {
        public:
            using self_type = xfunction_base<F, R, CT...>;
            using base_type = xexpression<self_type>;
            using functor_type = F;
            
            using value_type = R;
            using reference = value_type;
            using const_reference = value_type;
            using pointer = value_type*;
            using const_pointer = const value_type*;
            using size_type = common_size_type_t<CT...>;
            using difference_type = common_difference_type_t<CT...>;
            using shape_type = std::vector<size_type>;
            using strides_type = std::vector<size_type>;
            
            using expression_tag = xfunction_tag;
            
            static constexpr bool is_const = true;
            
            // Store the operands in a tuple
            using tuple_type = std::tuple<CT...>;
            
            // Construction
            template <class... Args>
            explicit xfunction_base(const F& func, Args&&... args)
                : m_functor(func)
                , m_operands(std::forward<Args>(args)...)
                , m_shape(broadcast_shape())
                , m_strides(compute_strides(m_shape, layout_type::row_major))
                , m_backstrides(compute_backstrides(m_shape, m_strides))
                , m_dimension(m_shape.size())
                , m_size(compute_size(m_shape))
            {
            }
            
            // Access
            size_type dimension() const noexcept { return m_dimension; }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return layout_type::row_major; }
            size_type size() const noexcept { return m_size; }
            
            // Evaluation
            template <class... Args>
            value_type operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<CT...>(),
                                                     static_cast<size_type>(args)...);
                return apply(index);
            }
            
            template <class... Args>
            value_type unchecked(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<CT...>(),
                                                     static_cast<size_type>(args)...);
                return apply(index);
            }
            
            value_type flat(size_type i) const
            {
                return apply(i);
            }
            
            template <class S>
            value_type element(const S& index) const
            {
                return apply(compute_index(index));
            }
            
            // Access to operands
            const tuple_type& operands() const noexcept { return m_operands; }
            const functor_type& functor() const noexcept { return m_functor; }
            
        protected:
            functor_type m_functor;
            tuple_type m_operands;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_dimension;
            size_type m_size;
            
            // Compute broadcasted shape from all operands
            shape_type broadcast_shape() const
            {
                return broadcast_shape_impl(std::index_sequence_for<CT...>{});
            }
            
            template <std::size_t... I>
            shape_type broadcast_shape_impl(std::index_sequence<I...>) const
            {
                std::array<const shape_type*, sizeof...(CT)> shapes = {
                    &std::get<I>(m_operands).shape()...
                };
                shape_type result;
                size_type max_dim = 0;
                for (auto* s : shapes)
                {
                    max_dim = std::max(max_dim, static_cast<size_type>(s->size()));
                }
                result.resize(max_dim, size_type(1));
                for (auto* s : shapes)
                {
                    std::size_t offset = max_dim - s->size();
                    for (std::size_t i = 0; i < s->size(); ++i)
                    {
                        size_type dim_val = (*s)[i];
                        size_type& res_val = result[offset + i];
                        if (dim_val == 1)
                        {
                            // keep current
                        }
                        else if (res_val == 1)
                        {
                            res_val = dim_val;
                        }
                        else if (dim_val != res_val)
                        {
                            XTENSOR_THROW(broadcast_error, "Incompatible shapes for broadcasting");
                        }
                    }
                }
                return result;
            }
            
            // Compute flat index from multi-index and strides
            size_type compute_index(const shape_type& index) const
            {
                size_type idx = 0;
                for (std::size_t i = 0; i < m_dimension; ++i)
                {
                    idx += index[i] * m_strides[i];
                }
                return idx;
            }
            
            template <std::size_t... I, class... Args>
            size_type compute_index_impl(std::index_sequence<I...>, Args... args) const
            {
                std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...};
                size_type idx = 0;
                ((idx += indices[I] * m_strides[I]), ...);
                return idx;
            }
            
            // Apply the functor to the operands at a given flat index
            value_type apply(size_type flat_index) const
            {
                return apply_impl(flat_index, std::index_sequence_for<CT...>{});
            }
            
            template <std::size_t... I>
            value_type apply_impl(size_type flat_index, std::index_sequence<I...>) const
            {
                // For each operand, we need to map flat_index back to its own flat index
                // by taking into account broadcasting.
                return m_functor(access_operand<I>(flat_index)...);
            }
            
            template <std::size_t I>
            auto access_operand(size_type flat_index) const -> typename std::tuple_element<I, tuple_type>::type::value_type
            {
                const auto& operand = std::get<I>(m_operands);
                if (operand.dimension() == 0)
                {
                    return operand();
                }
                // Compute the operand's flat index from the global flat index.
                // The operand's shape is broadcasted to m_shape.
                const auto& op_shape = operand.shape();
                if (operand.dimension() == m_dimension && std::equal(op_shape.begin(), op_shape.end(), m_shape.begin()))
                {
                    // Same shape, just use flat_index directly
                    return operand.flat(flat_index);
                }
                else
                {
                    // Need to unravel the global flat_index, then for dimensions where
                    // operand shape is 1, we index 0; otherwise use the coordinate.
                    size_type temp = flat_index;
                    size_type op_index = 0;
                    size_type op_stride = 1;
                    std::ptrdiff_t dim_diff = static_cast<std::ptrdiff_t>(m_dimension - operand.dimension());
                    
                    for (std::size_t i = 0; i < operand.dimension(); ++i)
                    {
                        size_type global_dim = i + static_cast<std::size_t>(dim_diff);
                        size_type coord = (temp / m_strides[global_dim]) % m_shape[global_dim];
                        if (op_shape[i] == 1)
                        {
                            coord = 0;
                        }
                        op_index += coord * operand.strides()[i];
                    }
                    return operand.flat(op_index);
                }
            }
        };
        
        // --------------------------------------------------------------------
        // xfunction - Concrete function expression
        // --------------------------------------------------------------------
        template <class F, class... CT>
        class xfunction : public xfunction_base<F,
                                                std::invoke_result_t<F,
                                                    typename std::decay_t<CT>::value_type...>,
                                                std::decay_t<CT>...>
        {
        public:
            using base_type = xfunction_base<F,
                                             std::invoke_result_t<F,
                                                 typename std::decay_t<CT>::value_type...>,
                                             std::decay_t<CT>...>;
            using base_type::base_type;
            
            using value_type = typename base_type::value_type;
            using size_type = typename base_type::size_type;
            using shape_type = typename base_type::shape_type;
            
            // Iterator support
            class const_iterator
            {
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = typename base_type::value_type;
                using difference_type = std::ptrdiff_t;
                using pointer = const value_type*;
                using reference = value_type;
                
                const_iterator() = default;
                const_iterator(const xfunction* func, size_type index)
                    : m_func(func), m_index(index)
                {
                }
                
                reference operator*() const { return m_func->flat(m_index); }
                
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                const_iterator& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                
                const_iterator operator+(difference_type n) const { return const_iterator(m_func, m_index + n); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_func, m_index - n); }
                
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                
                bool operator==(const const_iterator& rhs) const { return m_func == rhs.m_func && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const const_iterator& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const const_iterator& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const const_iterator& rhs) const { return m_index >= rhs.m_index; }
                
                reference operator[](difference_type n) const { return m_func->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
                
            private:
                const xfunction* m_func = nullptr;
                size_type m_index = 0;
            };
            
            using iterator = const_iterator;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            using reverse_iterator = const_reverse_iterator;
            
            const_iterator begin() const { return const_iterator(this, 0); }
            const_iterator end() const { return const_iterator(this, this->size()); }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
            
            reverse_iterator rbegin() const { return reverse_iterator(end()); }
            reverse_iterator rend() const { return reverse_iterator(begin()); }
            const_reverse_iterator crbegin() const { return rbegin(); }
            const_reverse_iterator crend() const { return rend(); }
            
            // Broadcasting
            template <class S>
            auto broadcast(const S& new_shape) const
            {
                shape_type shape(new_shape.begin(), new_shape.end());
                return xbroadcast<const self_type, shape_type>(*this, std::move(shape));
            }
            
            // Assignment is not allowed (expression is read-only)
            template <class E>
            disable_xexpression<E, xfunction&> operator=(const E&) = delete;
        };
        
        // --------------------------------------------------------------------
        // xfunction with reference result type (for functions returning lvalue)
        // --------------------------------------------------------------------
        template <class F, class... CT>
        class xfunction_ref : public xfunction_base<F,
                                                    typename std::invoke_result<F,
                                                        typename std::decay_t<CT>::value_type...>::type&,
                                                    std::decay_t<CT>...>
        {
        public:
            using base_type = xfunction_base<F,
                                             typename std::invoke_result<F,
                                                 typename std::decay_t<CT>::value_type...>::type&,
                                             std::decay_t<CT>...>;
            using base_type::base_type;
            using value_type = typename base_type::value_type;
            using reference = value_type;
            using const_reference = value_type;
            
            // Iterator support (similar to xfunction but returns references)
            class const_iterator
            {
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = typename base_type::value_type;
                using difference_type = std::ptrdiff_t;
                using pointer = const value_type*;
                using reference = value_type;
                
                const_iterator() = default;
                const_iterator(const xfunction_ref* func, size_type index)
                    : m_func(func), m_index(index)
                {
                }
                
                reference operator*() const { return m_func->flat(m_index); }
                
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index += n; return *this; }
                const_iterator& operator-=(difference_type n) { m_index -= n; return *this; }
                
                const_iterator operator+(difference_type n) const { return const_iterator(m_func, m_index + n); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_func, m_index - n); }
                
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                
                bool operator==(const const_iterator& rhs) const { return m_func == rhs.m_func && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const const_iterator& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const const_iterator& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const const_iterator& rhs) const { return m_index >= rhs.m_index; }
                
                reference operator[](difference_type n) const { return m_func->flat(m_index + n); }
                
            private:
                const xfunction_ref* m_func = nullptr;
                size_type m_index = 0;
            };
            
            using iterator = const_iterator;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            using reverse_iterator = const_reverse_iterator;
            
            const_iterator begin() const { return const_iterator(this, 0); }
            const_iterator end() const { return const_iterator(this, this->size()); }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
            
            reverse_iterator rbegin() const { return reverse_iterator(end()); }
            reverse_iterator rend() const { return reverse_iterator(begin()); }
            const_reverse_iterator crbegin() const { return rbegin(); }
            const_reverse_iterator crend() const { return rend(); }
            
            template <class E>
            disable_xexpression<E, xfunction_ref&> operator=(const E&) = delete;
        };
        
        // --------------------------------------------------------------------
        // Helper functions to create xfunction expressions
        // --------------------------------------------------------------------
        namespace detail
        {
            template <class F, class... E>
            struct select_xfunction_impl
            {
                using type = xfunction<F, E...>;
            };
            
            template <class F, class... E>
            struct select_xfunction_impl<F, E&...>
            {
                using value_type = std::invoke_result_t<F, typename std::decay_t<E>::value_type...>;
                using type = std::conditional_t<
                    std::is_reference<value_type>::value,
                    xfunction_ref<F, E...>,
                    xfunction<F, E...>
                >;
            };
            
            template <class F, class... E>
            using select_xfunction = typename select_xfunction_impl<F, E...>::type;
        }
        
        template <class F, class... E>
        inline auto make_xfunction(F&& func, E&&... e)
        {
            return detail::select_xfunction<F, E...>(
                std::forward<F>(func),
                std::forward<E>(e)...
            );
        }
        
        // Predefined mathematical functions as function objects
        namespace math
        {
#define XTENSOR_UNARY_FUNCTOR(NAME, EXPR) \
            struct NAME##_fun \
            { \
                template <class T> \
                auto operator()(T&& x) const \
                { \
                    using std::EXPR; \
                    return EXPR(std::forward<T>(x)); \
                } \
            };
            
            XTENSOR_UNARY_FUNCTOR(abs, abs)
            XTENSOR_UNARY_FUNCTOR(fabs, fabs)
            XTENSOR_UNARY_FUNCTOR(sqrt, sqrt)
            XTENSOR_UNARY_FUNCTOR(cbrt, cbrt)
            XTENSOR_UNARY_FUNCTOR(exp, exp)
            XTENSOR_UNARY_FUNCTOR(exp2, exp2)
            XTENSOR_UNARY_FUNCTOR(expm1, expm1)
            XTENSOR_UNARY_FUNCTOR(log, log)
            XTENSOR_UNARY_FUNCTOR(log2, log2)
            XTENSOR_UNARY_FUNCTOR(log10, log10)
            XTENSOR_UNARY_FUNCTOR(log1p, log1p)
            XTENSOR_UNARY_FUNCTOR(sin, sin)
            XTENSOR_UNARY_FUNCTOR(cos, cos)
            XTENSOR_UNARY_FUNCTOR(tan, tan)
            XTENSOR_UNARY_FUNCTOR(asin, asin)
            XTENSOR_UNARY_FUNCTOR(acos, acos)
            XTENSOR_UNARY_FUNCTOR(atan, atan)
            XTENSOR_UNARY_FUNCTOR(sinh, sinh)
            XTENSOR_UNARY_FUNCTOR(cosh, cosh)
            XTENSOR_UNARY_FUNCTOR(tanh, tanh)
            XTENSOR_UNARY_FUNCTOR(asinh, asinh)
            XTENSOR_UNARY_FUNCTOR(acosh, acosh)
            XTENSOR_UNARY_FUNCTOR(atanh, atanh)
            XTENSOR_UNARY_FUNCTOR(erf, erf)
            XTENSOR_UNARY_FUNCTOR(erfc, erfc)
            XTENSOR_UNARY_FUNCTOR(tgamma, tgamma)
            XTENSOR_UNARY_FUNCTOR(lgamma, lgamma)
            XTENSOR_UNARY_FUNCTOR(ceil, ceil)
            XTENSOR_UNARY_FUNCTOR(floor, floor)
            XTENSOR_UNARY_FUNCTOR(trunc, trunc)
            XTENSOR_UNARY_FUNCTOR(round, round)
            XTENSOR_UNARY_FUNCTOR(nearbyint, nearbyint)
            XTENSOR_UNARY_FUNCTOR(rint, rint)
            XTENSOR_UNARY_FUNCTOR(isnan, isnan)
            XTENSOR_UNARY_FUNCTOR(isinf, isinf)
            XTENSOR_UNARY_FUNCTOR(isfinite, isfinite)
            
#undef XTENSOR_UNARY_FUNCTOR
            
#define XTENSOR_BINARY_FUNCTOR(NAME, EXPR) \
            struct NAME##_fun \
            { \
                template <class T1, class T2> \
                auto operator()(T1&& x, T2&& y) const \
                { \
                    using std::EXPR; \
                    return EXPR(std::forward<T1>(x), std::forward<T2>(y)); \
                } \
            };
            
            XTENSOR_BINARY_FUNCTOR(pow, pow)
            XTENSOR_BINARY_FUNCTOR(atan2, atan2)
            XTENSOR_BINARY_FUNCTOR(hypot, hypot)
            XTENSOR_BINARY_FUNCTOR(fmod, fmod)
            XTENSOR_BINARY_FUNCTOR(remainder, remainder)
            XTENSOR_BINARY_FUNCTOR(copysign, copysign)
            XTENSOR_BINARY_FUNCTOR(nextafter, nextafter)
            XTENSOR_BINARY_FUNCTOR(fdim, fdim)
            XTENSOR_BINARY_FUNCTOR(fmax, fmax)
            XTENSOR_BINARY_FUNCTOR(fmin, fmin)
            
#undef XTENSOR_BINARY_FUNCTOR
        }
        
        // Wrapper functions for creating xfunction from math functors
        template <class E>
        inline auto abs(const xexpression<E>& e)
        {
            return make_xfunction(math::abs_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto fabs(const xexpression<E>& e)
        {
            return make_xfunction(math::fabs_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto sqrt(const xexpression<E>& e)
        {
            return make_xfunction(math::sqrt_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto cbrt(const xexpression<E>& e)
        {
            return make_xfunction(math::cbrt_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto exp(const xexpression<E>& e)
        {
            return make_xfunction(math::exp_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto exp2(const xexpression<E>& e)
        {
            return make_xfunction(math::exp2_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto expm1(const xexpression<E>& e)
        {
            return make_xfunction(math::expm1_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto log(const xexpression<E>& e)
        {
            return make_xfunction(math::log_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto log2(const xexpression<E>& e)
        {
            return make_xfunction(math::log2_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto log10(const xexpression<E>& e)
        {
            return make_xfunction(math::log10_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto log1p(const xexpression<E>& e)
        {
            return make_xfunction(math::log1p_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto sin(const xexpression<E>& e)
        {
            return make_xfunction(math::sin_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto cos(const xexpression<E>& e)
        {
            return make_xfunction(math::cos_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto tan(const xexpression<E>& e)
        {
            return make_xfunction(math::tan_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto asin(const xexpression<E>& e)
        {
            return make_xfunction(math::asin_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto acos(const xexpression<E>& e)
        {
            return make_xfunction(math::acos_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto atan(const xexpression<E>& e)
        {
            return make_xfunction(math::atan_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto sinh(const xexpression<E>& e)
        {
            return make_xfunction(math::sinh_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto cosh(const xexpression<E>& e)
        {
            return make_xfunction(math::cosh_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto tanh(const xexpression<E>& e)
        {
            return make_xfunction(math::tanh_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto asinh(const xexpression<E>& e)
        {
            return make_xfunction(math::asinh_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto acosh(const xexpression<E>& e)
        {
            return make_xfunction(math::acosh_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto atanh(const xexpression<E>& e)
        {
            return make_xfunction(math::atanh_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto erf(const xexpression<E>& e)
        {
            return make_xfunction(math::erf_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto erfc(const xexpression<E>& e)
        {
            return make_xfunction(math::erfc_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto tgamma(const xexpression<E>& e)
        {
            return make_xfunction(math::tgamma_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto lgamma(const xexpression<E>& e)
        {
            return make_xfunction(math::lgamma_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto ceil(const xexpression<E>& e)
        {
            return make_xfunction(math::ceil_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto floor(const xexpression<E>& e)
        {
            return make_xfunction(math::floor_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto trunc(const xexpression<E>& e)
        {
            return make_xfunction(math::trunc_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto round(const xexpression<E>& e)
        {
            return make_xfunction(math::round_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto nearbyint(const xexpression<E>& e)
        {
            return make_xfunction(math::nearbyint_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto rint(const xexpression<E>& e)
        {
            return make_xfunction(math::rint_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto isnan(const xexpression<E>& e)
        {
            return make_xfunction(math::isnan_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto isinf(const xexpression<E>& e)
        {
            return make_xfunction(math::isinf_fun{}, e.derived_cast());
        }
        
        template <class E>
        inline auto isfinite(const xexpression<E>& e)
        {
            return make_xfunction(math::isfinite_fun{}, e.derived_cast());
        }
        
        template <class E1, class E2>
        inline auto pow(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            return make_xfunction(math::pow_fun{}, e1.derived_cast(), e2.derived_cast());
        }
        
        template <class E1, class E2>
        inline auto atan2(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            return make_xfunction(math::atan2_fun{}, e1.derived_cast(), e2.derived_cast());
        }
        
        template <class E1, class E2>
        inline auto hypot(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            return make_xfunction(math::hypot_fun{}, e1.derived_cast(), e2.derived_cast());
        }
        
        template <class E1, class E2>
        inline auto fmod(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            return make_xfunction(math::fmod_fun{}, e1.derived_cast(), e2.derived_cast());
        }
        
        template <class E1, class E2>
        inline auto fmax(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            return make_xfunction(math::fmax_fun{}, e1.derived_cast(), e2.derived_cast());
        }
        
        template <class E1, class E2>
        inline auto fmin(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            return make_xfunction(math::fmin_fun{}, e1.derived_cast(), e2.derived_cast());
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XFUNCTION_HPP

// core/xfunction.hpp