// core/xexpression.hpp

#ifndef XTENSOR_XEXPRESSION_HPP
#define XTENSOR_XEXPRESSION_HPP

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <functional>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <initializer_list>
#include <vector>
#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <optional>
#include <cassert>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // xexpression - Base class for all tensor expressions using CRTP
        // --------------------------------------------------------------------
        template <class D>
        class xexpression
        {
        public:
            using derived_type = D;
            
            derived_type& derived_cast() & noexcept
            {
                return *static_cast<derived_type*>(this);
            }
            
            const derived_type& derived_cast() const & noexcept
            {
                return *static_cast<const derived_type*>(this);
            }
            
            derived_type&& derived_cast() && noexcept
            {
                return *static_cast<derived_type*>(this);
            }
            
            const derived_type&& derived_cast() const && noexcept
            {
                return *static_cast<const derived_type*>(this);
            }
            
        protected:
            xexpression() = default;
            ~xexpression() = default;
            xexpression(const xexpression&) = default;
            xexpression& operator=(const xexpression&) = default;
            xexpression(xexpression&&) = default;
            xexpression& operator=(xexpression&&) = default;
        };
        
        // --------------------------------------------------------------------
        // xcontainer_inner_types - Traits for container storage
        // --------------------------------------------------------------------
        template <class C>
        struct xcontainer_inner_types
        {
            using storage_type = C;
            using reference = typename storage_type::reference;
            using const_reference = typename storage_type::const_reference;
            using size_type = typename storage_type::size_type;
            using difference_type = typename storage_type::difference_type;
            using value_type = typename storage_type::value_type;
            using allocator_type = typename storage_type::allocator_type;
            using pointer = typename storage_type::pointer;
            using const_pointer = typename storage_type::const_pointer;
            using temporary_type = C;
        };
        
        template <class T, class A>
        struct xcontainer_inner_types<std::vector<T, A>>
        {
            using storage_type = std::vector<T, A>;
            using reference = T&;
            using const_reference = const T&;
            using size_type = typename std::vector<T, A>::size_type;
            using difference_type = typename std::vector<T, A>::difference_type;
            using value_type = T;
            using allocator_type = A;
            using pointer = T*;
            using const_pointer = const T*;
            using temporary_type = std::vector<T, A>;
        };
        
        template <class T, std::size_t N, class A>
        struct xcontainer_inner_types<std::array<T, N>>
        {
            using storage_type = std::array<T, N>;
            using reference = T&;
            using const_reference = const T&;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using value_type = T;
            using allocator_type = std::allocator<T>;
            using pointer = T*;
            using const_pointer = const T*;
            using temporary_type = std::array<T, N>;
        };
        
        // --------------------------------------------------------------------
        // xexpression_traits - Traits for any expression
        // --------------------------------------------------------------------
        template <class E>
        struct xexpression_traits
        {
            using derived_type = E;
            using value_type = typename E::value_type;
            using reference = typename E::reference;
            using const_reference = typename E::const_reference;
            using pointer = typename E::pointer;
            using const_pointer = typename E::const_pointer;
            using size_type = typename E::size_type;
            using difference_type = typename E::difference_type;
            using shape_type = typename E::shape_type;
            using strides_type = typename E::strides_type;
            using layout_type = typename E::layout_type;
            using expression_tag = typename E::expression_tag;
            
            static constexpr bool is_expression = std::is_base_of<xexpression<E>, E>::value;
            static constexpr bool is_const = E::is_const;
            static constexpr bool is_mutable = !is_const;
            static constexpr std::size_t dimension = E::dimension();
            static constexpr layout_type layout = E::layout;
        };
        
        // --------------------------------------------------------------------
        // SFINAE utilities for expression detection
        // --------------------------------------------------------------------
        template <class E>
        using enable_xexpression = std::enable_if_t<xexpression_traits<E>::is_expression>;
        
        template <class E, class R = void>
        using disable_xexpression = std::enable_if_t<!xexpression_traits<E>::is_expression, R>;
        
        template <class E1, class E2>
        using enable_same_dimension = std::enable_if_t<
            xexpression_traits<E1>::dimension == xexpression_traits<E2>::dimension
        >;
        
        template <class E1, class E2>
        using enable_different_dimension = std::enable_if_t<
            xexpression_traits<E1>::dimension != xexpression_traits<E2>::dimension
        >;
        
        template <class E>
        using enable_fixed_dimension = std::enable_if_t<
            xexpression_traits<E>::dimension != SIZE_MAX
        >;
        
        template <class E>
        using enable_dynamic_dimension = std::enable_if_t<
            xexpression_traits<E>::dimension == SIZE_MAX
        >;
        
        // --------------------------------------------------------------------
        // common_type metafunctions for expressions
        // --------------------------------------------------------------------
        template <class... E>
        struct common_value_type
        {
            using type = std::common_type_t<typename xexpression_traits<E>::value_type...>;
        };
        
        template <class... E>
        using common_value_type_t = typename common_value_type<E...>::type;
        
        template <class... E>
        struct common_size_type
        {
            using type = std::common_type_t<typename xexpression_traits<E>::size_type...>;
        };
        
        template <class... E>
        using common_size_type_t = typename common_size_type<E...>::type;
        
        template <class... E>
        struct common_difference_type
        {
            using type = std::common_type_t<typename xexpression_traits<E>::difference_type...>;
        };
        
        template <class... E>
        using common_difference_type_t = typename common_difference_type<E...>::type;
        
        template <class... E>
        struct common_shape_type
        {
            using type = std::common_type_t<typename xexpression_traits<E>::shape_type...>;
        };
        
        template <class... E>
        using common_shape_type_t = typename common_shape_type<E...>::type;
        
        template <class... E>
        struct common_strides_type
        {
            using type = std::common_type_t<typename xexpression_traits<E>::strides_type...>;
        };
        
        template <class... E>
        using common_strides_type_t = typename common_strides_type<E...>::type;
        
        // --------------------------------------------------------------------
        // get_dimension - Helper to extract dimension from shape-like
        // --------------------------------------------------------------------
        template <class S>
        inline std::size_t get_dimension(const S& shape)
        {
            using size_type = typename S::value_type;
            return static_cast<std::size_t>(std::distance(std::begin(shape), std::end(shape)));
        }
        
        template <class T, std::size_t N>
        inline constexpr std::size_t get_dimension(const std::array<T, N>&)
        {
            return N;
        }
        
        template <class T, std::size_t N>
        inline constexpr std::size_t get_dimension(const T (&)[N])
        {
            return N;
        }
        
        // --------------------------------------------------------------------
        // compute_size - Compute total number of elements from shape
        // --------------------------------------------------------------------
        template <class S>
        inline auto compute_size(const S& shape)
        {
            using size_type = typename S::value_type;
            return std::accumulate(std::begin(shape), std::end(shape),
                                   size_type(1), std::multiplies<size_type>());
        }
        
        // --------------------------------------------------------------------
        // compute_strides - Compute strides from shape and layout
        // --------------------------------------------------------------------
        template <class S, class R>
        inline void compute_strides(const S& shape, layout_type l, R& strides)
        {
            using size_type = typename S::value_type;
            std::size_t dim = get_dimension(shape);
            strides.resize(dim);
            
            if (dim == 0) return;
            
            if (l == layout_type::row_major)
            {
                strides[dim - 1] = 1;
                for (std::size_t i = dim - 1; i > 0; --i)
                {
                    strides[i - 1] = strides[i] * static_cast<size_type>(shape[i]);
                }
            }
            else // column_major
            {
                strides[0] = 1;
                for (std::size_t i = 0; i < dim - 1; ++i)
                {
                    strides[i + 1] = strides[i] * static_cast<size_type>(shape[i]);
                }
            }
        }
        
        template <class S>
        inline auto compute_strides(const S& shape, layout_type l)
        {
            using size_type = typename S::value_type;
            std::vector<size_type> strides(get_dimension(shape));
            compute_strides(shape, l, strides);
            return strides;
        }
        
        template <class S, class St>
        inline void compute_strides(const S& shape, layout_type l, St& strides, St& backstrides)
        {
            compute_strides(shape, l, strides);
            backstrides.resize(strides.size());
            for (std::size_t i = 0; i < strides.size(); ++i)
            {
                backstrides[i] = strides[i] * (static_cast<typename St::value_type>(shape[i]) - 1);
            }
        }
        
        // --------------------------------------------------------------------
        // uninitialized_copy - Copy to uninitialized memory
        // --------------------------------------------------------------------
        template <class InputIt, class ForwardIt>
        inline ForwardIt uninitialized_copy(InputIt first, InputIt last, ForwardIt d_first)
        {
            using T = typename std::iterator_traits<ForwardIt>::value_type;
            ForwardIt current = d_first;
            XTENSOR_TRY
            {
                for (; first != last; ++first, ++current)
                {
                    ::new (static_cast<void*>(std::addressof(*current))) T(*first);
                }
                return current;
            }
            XTENSOR_CATCH_ALL
            {
                for (; d_first != current; ++d_first)
                {
                    d_first->~T();
                }
                XTENSOR_RETHROW;
            }
        }
        
        // --------------------------------------------------------------------
        // uninitialized_fill - Fill uninitialized memory
        // --------------------------------------------------------------------
        template <class ForwardIt, class T>
        inline void uninitialized_fill(ForwardIt first, ForwardIt last, const T& value)
        {
            using V = typename std::iterator_traits<ForwardIt>::value_type;
            ForwardIt current = first;
            XTENSOR_TRY
            {
                for (; current != last; ++current)
                {
                    ::new (static_cast<void*>(std::addressof(*current))) V(value);
                }
            }
            XTENSOR_CATCH_ALL
            {
                for (; first != current; ++first)
                {
                    first->~V();
                }
                XTENSOR_RETHROW;
            }
        }
        
        // --------------------------------------------------------------------
        // broadcast_shape - Broadcast two shapes
        // --------------------------------------------------------------------
        template <class S1, class S2>
        inline auto broadcast_shape(const S1& shape1, const S2& shape2)
        {
            using size_type = std::common_type_t<typename S1::value_type, typename S2::value_type>;
            std::vector<size_type> result;
            
            auto it1 = shape1.rbegin();
            auto it2 = shape2.rbegin();
            auto end1 = shape1.rend();
            auto end2 = shape2.rend();
            
            while (it1 != end1 || it2 != end2)
            {
                size_type dim1 = (it1 != end1) ? *it1 : 1;
                size_type dim2 = (it2 != end2) ? *it2 : 1;
                
                if (dim1 == dim2)
                {
                    result.push_back(dim1);
                }
                else if (dim1 == 1)
                {
                    result.push_back(dim2);
                }
                else if (dim2 == 1)
                {
                    result.push_back(dim1);
                }
                else
                {
                    XTENSOR_THROW(broadcast_error, "Incompatible shapes for broadcasting");
                }
                
                if (it1 != end1) ++it1;
                if (it2 != end2) ++it2;
            }
            
            std::reverse(result.begin(), result.end());
            return result;
        }
        
        template <class S>
        inline bool has_fixed_stride(const S& shape, const S& strides)
        {
            if (shape.empty()) return true;
            
            using size_type = typename S::value_type;
            S expected_strides(shape.size());
            
            // Check row-major
            expected_strides.back() = 1;
            for (std::size_t i = shape.size() - 1; i > 0; --i)
            {
                expected_strides[i - 1] = expected_strides[i] * shape[i];
            }
            if (std::equal(strides.begin(), strides.end(), expected_strides.begin()))
            {
                return true;
            }
            
            // Check column-major
            expected_strides[0] = 1;
            for (std::size_t i = 0; i < shape.size() - 1; ++i)
            {
                expected_strides[i + 1] = expected_strides[i] * shape[i];
            }
            return std::equal(strides.begin(), strides.end(), expected_strides.begin());
        }
        
        // --------------------------------------------------------------------
        // flat_index_from_strides - Compute flat index from multi-index and strides
        // --------------------------------------------------------------------
        template <class S, class I>
        inline auto flat_index_from_strides(const S& strides, const I& index)
        {
            using size_type = typename S::value_type;
            size_type result = 0;
            for (std::size_t i = 0; i < strides.size(); ++i)
            {
                result += static_cast<size_type>(index[i]) * strides[i];
            }
            return result;
        }
        
        template <class S, class I>
        inline auto flat_index_from_strides(const S& strides, const I& index, const S& backstrides)
        {
            using size_type = typename S::value_type;
            size_type result = 0;
            for (std::size_t i = 0; i < strides.size(); ++i)
            {
                size_type idx = static_cast<size_type>(index[i]);
                if (idx == 0)
                {
                    result += 0;
                }
                else
                {
                    result += strides[i] + (idx - 1) * strides[i];
                }
            }
            return result;
        }
        
        // --------------------------------------------------------------------
        // unravel_index - Convert flat index to multi-index given shape
        // --------------------------------------------------------------------
        template <class S, class V>
        inline void unravel_index(typename S::value_type flat_index,
                                  const S& shape,
                                  layout_type l,
                                  V& index)
        {
            using size_type = typename S::value_type;
            std::size_t dim = shape.size();
            index.resize(dim);
            
            if (l == layout_type::row_major)
            {
                for (std::size_t i = dim; i > 0; --i)
                {
                    std::size_t d = i - 1;
                    index[d] = flat_index % shape[d];
                    flat_index /= shape[d];
                }
            }
            else
            {
                for (std::size_t i = 0; i < dim; ++i)
                {
                    index[i] = flat_index % shape[i];
                    flat_index /= shape[i];
                }
            }
        }
        
        template <class S>
        inline auto unravel_index(typename S::value_type flat_index,
                                  const S& shape,
                                  layout_type l)
        {
            std::vector<typename S::value_type> index;
            unravel_index(flat_index, shape, l, index);
            return index;
        }
        
        // --------------------------------------------------------------------
        // ravel_index - Convert multi-index to flat index given shape and layout
        // --------------------------------------------------------------------
        template <class S, class I>
        inline auto ravel_index(const I& index, const S& shape, layout_type l)
        {
            using size_type = typename S::value_type;
            std::size_t dim = shape.size();
            size_type result = 0;
            
            if (l == layout_type::row_major)
            {
                size_type stride = 1;
                for (std::size_t i = dim; i > 0; --i)
                {
                    std::size_t d = i - 1;
                    result += static_cast<size_type>(index[d]) * stride;
                    stride *= shape[d];
                }
            }
            else
            {
                size_type stride = 1;
                for (std::size_t i = 0; i < dim; ++i)
                {
                    result += static_cast<size_type>(index[i]) * stride;
                    stride *= shape[i];
                }
            }
            return result;
        }
        
        // --------------------------------------------------------------------
        // promote_shape - Promote shape to at least N dimensions
        // --------------------------------------------------------------------
        template <class S>
        inline auto promote_shape(const S& shape, std::size_t ndim)
        {
            using size_type = typename S::value_type;
            std::vector<size_type> result(ndim, 1);
            std::size_t offset = ndim - shape.size();
            std::copy(shape.begin(), shape.end(), result.begin() + static_cast<ptrdiff_t>(offset));
            return result;
        }
        
        // --------------------------------------------------------------------
        // normalize_axis - Handle negative axis indexing
        // --------------------------------------------------------------------
        inline std::size_t normalize_axis(std::ptrdiff_t axis, std::size_t ndim)
        {
            if (axis < 0)
            {
                axis += static_cast<std::ptrdiff_t>(ndim);
            }
            if (axis < 0 || static_cast<std::size_t>(axis) >= ndim)
            {
                XTENSOR_THROW(std::out_of_range, "Axis out of range");
            }
            return static_cast<std::size_t>(axis);
        }
        
        template <class Container>
        inline Container normalize_axis(const Container& axes, std::size_t ndim)
        {
            Container result;
            result.reserve(axes.size());
            for (auto axis : axes)
            {
                result.push_back(normalize_axis(static_cast<std::ptrdiff_t>(axis), ndim));
            }
            return result;
        }
        
        // --------------------------------------------------------------------
        // remove_axis - Remove axis from shape/strides
        // --------------------------------------------------------------------
        template <class S>
        inline S remove_axis(const S& shape, std::size_t axis)
        {
            S result;
            result.reserve(shape.size() - 1);
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                if (i != axis)
                {
                    result.push_back(shape[i]);
                }
            }
            return result;
        }
        
        // --------------------------------------------------------------------
        // keep_dims_shape - Shape after reduction with keepdims
        // --------------------------------------------------------------------
        template <class S, class Axes>
        inline S keep_dims_shape(const S& shape, const Axes& axes)
        {
            S result = shape;
            for (auto axis : axes)
            {
                result[axis] = 1;
            }
            return result;
        }
        
        // --------------------------------------------------------------------
        // slice utilities
        // --------------------------------------------------------------------
        struct xall_tag {};
        struct xnewaxis_tag {};
        struct xellipsis_tag {};
        
        XTENSOR_INLINE_VARIABLE xall_tag all = {};
        XTENSOR_INLINE_VARIABLE xnewaxis_tag newaxis = {};
        XTENSOR_INLINE_VARIABLE xellipsis_tag ellipsis = {};
        
        template <class T>
        class xrange
        {
        public:
            using size_type = T;
            
            xrange() = default;
            xrange(size_type start, size_type stop, size_type step = 1)
                : m_start(start), m_stop(stop), m_step(step)
            {
            }
            
            size_type start() const { return m_start; }
            size_type stop() const { return m_stop; }
            size_type step() const { return m_step; }
            size_type size() const
            {
                if (m_step > 0)
                {
                    return m_stop > m_start ? (m_stop - m_start - 1) / m_step + 1 : 0;
                }
                else
                {
                    return m_start > m_stop ? (m_start - m_stop - 1) / (-m_step) + 1 : 0;
                }
            }
            
        private:
            size_type m_start = 0;
            size_type m_stop = 0;
            size_type m_step = 1;
        };
        
        template <class T>
        inline auto range(T stop)
        {
            return xrange<T>(0, stop);
        }
        
        template <class T>
        inline auto range(T start, T stop, T step = 1)
        {
            return xrange<T>(start, stop, step);
        }
        
        template <class T>
        inline auto arange(T start, T stop, T step = 1)
        {
            return range(start, stop, step);
        }
        
        inline auto arange(std::size_t stop)
        {
            return range(stop);
        }
        
        // --------------------------------------------------------------------
        // placeholder for advanced indexing
        // --------------------------------------------------------------------
        template <class E>
        class xplaceholder : public xexpression<xplaceholder<E>>
        {
        public:
            using value_type = typename E::value_type;
            using reference = typename E::reference;
            using const_reference = typename E::const_reference;
            using pointer = typename E::pointer;
            using const_pointer = typename E::const_pointer;
            using size_type = typename E::size_type;
            using difference_type = typename E::difference_type;
            using shape_type = typename E::shape_type;
            using strides_type = typename E::strides_type;
            using layout_type = typename E::layout_type;
            using expression_tag = xscalar_tag;
            
            static constexpr layout_type layout = E::layout;
            static constexpr bool is_const = true;
            
            explicit xplaceholder(const E& expr) : m_expr(expr) {}
            
            size_type dimension() const { return m_expr.dimension(); }
            const shape_type& shape() const { return m_expr.shape(); }
            const strides_type& strides() const { return m_expr.strides(); }
            
            template <class... Args>
            const_reference operator()(Args... args) const
            {
                return m_expr(args...);
            }
            
        private:
            const E& m_expr;
        };
        
        template <class E>
        inline auto placeholder(const xexpression<E>& expr)
        {
            return xplaceholder<E>(expr.derived_cast());
        }
        
        // --------------------------------------------------------------------
        // xscalar - Scalar expression wrapper
        // --------------------------------------------------------------------
        template <class T>
        class xscalar : public xexpression<xscalar<T>>
        {
        public:
            using value_type = T;
            using reference = T&;
            using const_reference = const T&;
            using pointer = T*;
            using const_pointer = const T*;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using shape_type = std::array<size_type, 0>;
            using strides_type = std::array<size_type, 0>;
            using layout_type = layout_type;
            using expression_tag = xscalar_tag;
            
            static constexpr layout_type layout = layout_type::row_major;
            static constexpr bool is_const = true;
            
            xscalar() = default;
            explicit xscalar(const T& value) : m_value(value) {}
            
            size_type dimension() const { return 0; }
            shape_type shape() const { return {}; }
            strides_type strides() const { return {}; }
            
            const_reference operator()() const { return m_value; }
            
            template <class... Args>
            const_reference operator()(Args...) const
            {
                return m_value;
            }
            
            template <class S>
            const_reference element(const S&) const
            {
                return m_value;
            }
            
            const_reference flat(size_type) const { return m_value; }
            
            // Iterator support
            const T* begin() const { return &m_value; }
            const T* end() const { return &m_value + 1; }
            
        private:
            T m_value;
        };
        
        // --------------------------------------------------------------------
        // make_xscalar
        // --------------------------------------------------------------------
        template <class T>
        inline auto make_xscalar(T&& value)
        {
            return xscalar<std::decay_t<T>>(std::forward<T>(value));
        }
        
        // --------------------------------------------------------------------
        // eval - Force evaluation of an expression
        // --------------------------------------------------------------------
        template <class E>
        inline auto eval(E&& expr)
        {
            using expr_type = std::decay_t<E>;
            using value_type = typename expr_type::value_type;
            using shape_type = typename expr_type::shape_type;
            using container_type = xarray_container<value_type,
                                                    expr_type::layout,
                                                    default_allocator<value_type>>;
            
            container_type result(expr.shape());
            result = expr;
            return result;
        }
        
        // --------------------------------------------------------------------
        // eval_expr - Evaluate to specified container type
        // --------------------------------------------------------------------
        template <class C, class E>
        inline auto eval_expr(E&& expr)
        {
            C result(expr.shape());
            result = expr;
            return result;
        }
        
        // --------------------------------------------------------------------
        // operator overloads for expressions
        // --------------------------------------------------------------------
        
        // Unary operators
        template <class E>
        inline auto operator+(const xexpression<E>& e)
        {
            return make_xfunction<detail::identity>(e.derived_cast());
        }
        
        template <class E>
        inline auto operator-(const xexpression<E>& e)
        {
            return make_xfunction<detail::negate>(e.derived_cast());
        }
        
        template <class E>
        inline auto operator~(const xexpression<E>& e)
        {
            return make_xfunction<detail::bitwise_not>(e.derived_cast());
        }
        
        template <class E>
        inline auto operator!(const xexpression<E>& e)
        {
            return make_xfunction<detail::logical_not>(e.derived_cast());
        }
        
        // Binary operators
#define XTENSOR_BINARY_OPERATOR(OP, NAME) \
        template <class E1, class E2> \
        inline auto operator OP (const xexpression<E1>& e1, const xexpression<E2>& e2) \
        { \
            return make_xfunction<detail::NAME>(e1.derived_cast(), e2.derived_cast()); \
        } \
        template <class E, class T> \
        inline auto operator OP (const xexpression<E>& e, const T& t) \
        { \
            return make_xfunction<detail::NAME>(e.derived_cast(), make_xscalar(t)); \
        } \
        template <class T, class E> \
        inline auto operator OP (const T& t, const xexpression<E>& e) \
        { \
            return make_xfunction<detail::NAME>(make_xscalar(t), e.derived_cast()); \
        }
        
        XTENSOR_BINARY_OPERATOR(+, plus)
        XTENSOR_BINARY_OPERATOR(-, minus)
        XTENSOR_BINARY_OPERATOR(*, multiplies)
        XTENSOR_BINARY_OPERATOR(/, divides)
        XTENSOR_BINARY_OPERATOR(%, modulus)
        XTENSOR_BINARY_OPERATOR(&&, logical_and)
        XTENSOR_BINARY_OPERATOR(||, logical_or)
        XTENSOR_BINARY_OPERATOR(&, bitwise_and)
        XTENSOR_BINARY_OPERATOR(|, bitwise_or)
        XTENSOR_BINARY_OPERATOR(^, bitwise_xor)
        XTENSOR_BINARY_OPERATOR(<<, left_shift)
        XTENSOR_BINARY_OPERATOR(>>, right_shift)
        
#undef XTENSOR_BINARY_OPERATOR
        
        // Comparison operators
#define XTENSOR_COMPARISON_OPERATOR(OP, NAME) \
        template <class E1, class E2> \
        inline auto operator OP (const xexpression<E1>& e1, const xexpression<E2>& e2) \
        { \
            return make_xfunction<detail::NAME>(e1.derived_cast(), e2.derived_cast()); \
        } \
        template <class E, class T> \
        inline auto operator OP (const xexpression<E>& e, const T& t) \
        { \
            return make_xfunction<detail::NAME>(e.derived_cast(), make_xscalar(t)); \
        } \
        template <class T, class E> \
        inline auto operator OP (const T& t, const xexpression<E>& e) \
        { \
            return make_xfunction<detail::NAME>(make_xscalar(t), e.derived_cast()); \
        }
        
        XTENSOR_COMPARISON_OPERATOR(==, equal_to)
        XTENSOR_COMPARISON_OPERATOR(!=, not_equal_to)
        XTENSOR_COMPARISON_OPERATOR(<, less)
        XTENSOR_COMPARISON_OPERATOR(<=, less_equal)
        XTENSOR_COMPARISON_OPERATOR(>, greater)
        XTENSOR_COMPARISON_OPERATOR(>=, greater_equal)
        
#undef XTENSOR_COMPARISON_OPERATOR
        
        // Compound assignment
#define XTENSOR_COMPOUND_ASSIGNMENT(OP, NAME) \
        template <class E1, class E2> \
        inline E1& operator OP (xexpression<E1>& e1, const xexpression<E2>& e2) \
        { \
            auto& derived = e1.derived_cast(); \
            derived = derived OP e2.derived_cast(); \
            return derived; \
        } \
        template <class E, class T> \
        inline E& operator OP (xexpression<E>& e, const T& t) \
        { \
            auto& derived = e.derived_cast(); \
            derived = derived OP t; \
            return derived; \
        }
        
        XTENSOR_COMPOUND_ASSIGNMENT(+=, plus)
        XTENSOR_COMPOUND_ASSIGNMENT(-=, minus)
        XTENSOR_COMPOUND_ASSIGNMENT(*=, multiplies)
        XTENSOR_COMPOUND_ASSIGNMENT(/=, divides)
        XTENSOR_COMPOUND_ASSIGNMENT(%=, modulus)
        XTENSOR_COMPOUND_ASSIGNMENT(&=, bitwise_and)
        XTENSOR_COMPOUND_ASSIGNMENT(|=, bitwise_or)
        XTENSOR_COMPOUND_ASSIGNMENT(^=, bitwise_xor)
        XTENSOR_COMPOUND_ASSIGNMENT(<<=, left_shift)
        XTENSOR_COMPOUND_ASSIGNMENT(>>=, right_shift)
        
#undef XTENSOR_COMPOUND_ASSIGNMENT
        
        // --------------------------------------------------------------------
        // Helper functions for expression creation
        // --------------------------------------------------------------------
        template <class E>
        inline auto transpose(const xexpression<E>& e)
        {
            auto& derived = e.derived_cast();
            if (derived.dimension() == 2)
            {
                std::vector<std::size_t> perm = {1, 0};
                return transpose(derived, perm);
            }
            else
            {
                std::vector<std::size_t> perm(derived.dimension());
                std::iota(perm.rbegin(), perm.rend(), 0);
                return transpose(derived, perm);
            }
        }
        
        template <class E, class P>
        inline auto transpose(const xexpression<E>& e, const P& perm)
        {
            using temporary_type = typename xexpression_traits<E>::temporary_type;
            temporary_type result(e.derived_cast().shape());
            // Need proper transpose implementation
            return result;
        }
        
        // --------------------------------------------------------------------
        // Expression evaluator for immediate assignment
        // --------------------------------------------------------------------
        struct immediate_assign_tag {};
        struct lazy_assign_tag {};
        
        XTENSOR_INLINE_VARIABLE immediate_assign_tag immediate_assign = {};
        XTENSOR_INLINE_VARIABLE lazy_assign_tag lazy_assign = {};
        
        template <class E, class Tag = lazy_assign_tag>
        class xexpression_assigner
        {
        public:
            template <class T>
            static void assign(E& expr, const T& value)
            {
                expr.derived_cast() = value;
            }
        };
        
        template <class E>
        class xexpression_assigner<E, immediate_assign_tag>
        {
        public:
            template <class T>
            static void assign(E& expr, const T& value)
            {
                expr.derived_cast().assign_from(value);
            }
        };
        
        // --------------------------------------------------------------------
        // noalias - Prevent aliasing issues
        // --------------------------------------------------------------------
        template <class E>
        class noalias_proxy
        {
        public:
            explicit noalias_proxy(E& expr) : m_expr(expr) {}
            
            template <class T>
            E& operator=(const T& value)
            {
                m_expr.assign_from(value);
                return m_expr;
            }
            
        private:
            E& m_expr;
        };
        
        template <class E>
        inline noalias_proxy<E> noalias(E& expr)
        {
            return noalias_proxy<E>(expr);
        }
        
        // --------------------------------------------------------------------
        // Exception types
        // --------------------------------------------------------------------
        class broadcast_error : public std::runtime_error
        {
        public:
            broadcast_error(const char* msg) : std::runtime_error(msg) {}
            broadcast_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class dimension_mismatch : public std::runtime_error
        {
        public:
            dimension_mismatch(const char* msg) : std::runtime_error(msg) {}
            dimension_mismatch(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class index_error : public std::out_of_range
        {
        public:
            index_error(const char* msg) : std::out_of_range(msg) {}
            index_error(const std::string& msg) : std::out_of_range(msg) {}
        };
        
        class incompatible_shapes : public std::runtime_error
        {
        public:
            incompatible_shapes(const char* msg) : std::runtime_error(msg) {}
            incompatible_shapes(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class reduction_error : public std::runtime_error
        {
        public:
            reduction_error(const char* msg) : std::runtime_error(msg) {}
            reduction_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class accumulation_error : public std::runtime_error
        {
        public:
            accumulation_error(const char* msg) : std::runtime_error(msg) {}
            accumulation_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class iterator_error : public std::runtime_error
        {
        public:
            iterator_error(const char* msg) : std::runtime_error(msg) {}
            iterator_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class memory_allocation_error : public std::bad_alloc
        {
        public:
            memory_allocation_error() noexcept = default;
            const char* what() const noexcept override
            {
                return "xtensor memory allocation failed";
            }
        };
        
        class alignment_error : public std::runtime_error
        {
        public:
            alignment_error(const char* msg) : std::runtime_error(msg) {}
            alignment_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class not_implemented_error : public std::logic_error
        {
        public:
            not_implemented_error(const char* msg) : std::logic_error(msg) {}
            not_implemented_error(const std::string& msg) : std::logic_error(msg) {}
        };
        
        class invalid_layout_error : public std::runtime_error
        {
        public:
            invalid_layout_error(const char* msg) : std::runtime_error(msg) {}
            invalid_layout_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class invalid_slice_error : public std::runtime_error
        {
        public:
            invalid_slice_error(const char* msg) : std::runtime_error(msg) {}
            invalid_slice_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        class invalid_stride_error : public std::runtime_error
        {
        public:
            invalid_stride_error(const char* msg) : std::runtime_error(msg) {}
            invalid_stride_error(const std::string& msg) : std::runtime_error(msg) {}
        };
        
        // --------------------------------------------------------------------
        // Detail namespace for internal implementation
        // --------------------------------------------------------------------
        namespace detail
        {
            // Function objects for operations
            struct identity { template <class T> auto operator()(T&& t) const { return std::forward<T>(t); } };
            struct negate { template <class T> auto operator()(T&& t) const { return -std::forward<T>(t); } };
            struct bitwise_not { template <class T> auto operator()(T&& t) const { return ~std::forward<T>(t); } };
            struct logical_not { template <class T> auto operator()(T&& t) const { return !std::forward<T>(t); } };
            
            struct plus { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) + std::forward<T2>(t2); } };
            struct minus { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) - std::forward<T2>(t2); } };
            struct multiplies { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) * std::forward<T2>(t2); } };
            struct divides { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) / std::forward<T2>(t2); } };
            struct modulus { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) % std::forward<T2>(t2); } };
            struct logical_and { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) && std::forward<T2>(t2); } };
            struct logical_or { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) || std::forward<T2>(t2); } };
            struct bitwise_and { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) & std::forward<T2>(t2); } };
            struct bitwise_or { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) | std::forward<T2>(t2); } };
            struct bitwise_xor { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) ^ std::forward<T2>(t2); } };
            struct left_shift { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) << std::forward<T2>(t2); } };
            struct right_shift { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) >> std::forward<T2>(t2); } };
            
            struct equal_to { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) == std::forward<T2>(t2); } };
            struct not_equal_to { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) != std::forward<T2>(t2); } };
            struct less { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) < std::forward<T2>(t2); } };
            struct less_equal { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) <= std::forward<T2>(t2); } };
            struct greater { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) > std::forward<T2>(t2); } };
            struct greater_equal { template <class T1, class T2> auto operator()(T1&& t1, T2&& t2) const { return std::forward<T1>(t1) >= std::forward<T2>(t2); } };
            
            // make_xfunction helper
            template <class F, class... E>
            inline auto make_xfunction(E&&... e)
            {
                return xfunction<F, std::decay_t<E>...>(F{}, std::forward<E>(e)...);
            }
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XEXPRESSION_HPP