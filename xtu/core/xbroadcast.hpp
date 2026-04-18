// core/xbroadcast.hpp

#ifndef XTENSOR_XBROADCAST_HPP
#define XTENSOR_XBROADCAST_HPP

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iterator>
#include <functional>
#include <cassert>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // xbroadcast - Expression that broadcasts a source to a new shape
        // --------------------------------------------------------------------
        template <class CT, class X>
        class xbroadcast : public xexpression<xbroadcast<CT, X>>
        {
        public:
            using self_type = xbroadcast<CT, X>;
            using base_type = xexpression<self_type>;
            using xexpression_type = std::decay_t<CT>;
            
            using value_type = typename xexpression_type::value_type;
            using reference = typename xexpression_type::reference;
            using const_reference = typename xexpression_type::const_reference;
            using pointer = typename xexpression_type::pointer;
            using const_pointer = typename xexpression_type::const_pointer;
            using size_type = typename xexpression_type::size_type;
            using difference_type = typename xexpression_type::difference_type;
            
            using shape_type = std::vector<size_type>;
            using strides_type = std::vector<size_type>;
            
            using expression_tag = xbroadcast_tag;
            
            static constexpr bool is_const = std::is_const<CT>::value;
            
            // Construction
            template <class CTA, class S>
            xbroadcast(CTA&& e, S&& shape)
                : m_expression(std::forward<CTA>(e))
                , m_shape(std::forward<S>(shape))
            {
                if (m_shape.size() < m_expression.dimension())
                {
                    XTENSOR_THROW(broadcast_error, "Broadcast shape must have at least as many dimensions as source");
                }
                compute_strides_and_backstrides();
                m_size = compute_size(m_shape);
            }
            
            // Size and shape
            size_type size() const noexcept { return m_size; }
            size_type dimension() const noexcept { return m_shape.size(); }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return m_expression.layout(); }
            
            // Access to underlying expression
            const xexpression_type& expression() const noexcept { return m_expression; }
            
            // Element access
            template <class... Args>
            const_reference operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            template <class... Args>
            const_reference unchecked(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            template <class S>
            const_reference element(const S& index) const
            {
                return flat(compute_index(index));
            }
            
            const_reference flat(size_type i) const
            {
                if (m_expression.dimension() == 0)
                {
                    return m_expression();
                }
                
                // Map the flat index in the broadcasted shape back to the flat index in the original expression.
                size_type src_index = 0;
                size_type temp = i;
                
                // Determine the dimension offset
                std::ptrdiff_t dim_diff = static_cast<std::ptrdiff_t>(dimension()) - 
                                          static_cast<std::ptrdiff_t>(m_expression.dimension());
                
                // For each dimension of the broadcasted array, compute the coordinate.
                // For dimensions that are not present in the source (i.e., new dimensions added on the left),
                // the source dimension size is implicitly 1, so the coordinate should be 0.
                // For existing dimensions, if the broadcasted size equals the source size, use the coordinate.
                // If the broadcasted size is larger but source size is 1, the source coordinate is 0.
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    size_type coord = temp / m_strides[d];
                    temp %= m_strides[d];
                    
                    std::ptrdiff_t src_d = static_cast<std::ptrdiff_t>(d) - dim_diff;
                    if (src_d >= 0 && src_d < static_cast<std::ptrdiff_t>(m_expression.dimension()))
                    {
                        size_type src_dim_size = m_expression.shape()[static_cast<std::size_t>(src_d)];
                        if (src_dim_size == 1)
                        {
                            coord = 0;
                        }
                        src_index += coord * m_expression.strides()[static_cast<std::size_t>(src_d)];
                    }
                    // else: new dimension, source coordinate is implicitly 0, nothing to add
                }
                
                return m_expression.flat(src_index);
            }
            
            // Iterator support
            class const_iterator
            {
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = typename self_type::value_type;
                using difference_type = std::ptrdiff_t;
                using pointer = const value_type*;
                using reference = value_type;
                
                const_iterator() = default;
                const_iterator(const self_type* broadcast, size_type index)
                    : m_broadcast(broadcast), m_index(index)
                {
                }
                
                reference operator*() const { return m_broadcast->flat(m_index); }
                
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                const_iterator& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                
                const_iterator operator+(difference_type n) const { return const_iterator(m_broadcast, m_index + static_cast<size_type>(n)); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_broadcast, m_index - static_cast<size_type>(n)); }
                
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                
                bool operator==(const const_iterator& rhs) const { return m_broadcast == rhs.m_broadcast && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const const_iterator& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const const_iterator& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const const_iterator& rhs) const { return m_index >= rhs.m_index; }
                
                reference operator[](difference_type n) const { return m_broadcast->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
                
            private:
                const self_type* m_broadcast = nullptr;
                size_type m_index = 0;
            };
            
            using iterator = const_iterator;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            using reverse_iterator = const_reverse_iterator;
            
            const_iterator begin() const { return const_iterator(this, 0); }
            const_iterator end() const { return const_iterator(this, m_size); }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
            
            reverse_iterator rbegin() const { return reverse_iterator(end()); }
            reverse_iterator rend() const { return reverse_iterator(begin()); }
            const_reverse_iterator crbegin() const { return rbegin(); }
            const_reverse_iterator crend() const { return rend(); }
            
            // Broadcasting a broadcast (no-op or combine)
            template <class S>
            auto broadcast(const S& new_shape) const
            {
                shape_type combined_shape = broadcast_shape(m_shape, new_shape);
                return xbroadcast<const xexpression_type, shape_type>(m_expression, std::move(combined_shape));
            }
            
            // Assignment disabled for broadcast expression (read-only)
            template <class E>
            disable_xexpression<E, self_type&> operator=(const E&) = delete;
            
        private:
            CT m_expression;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            
            void compute_strides_and_backstrides()
            {
                std::size_t dim = m_shape.size();
                m_strides.resize(dim);
                m_backstrides.resize(dim);
                
                if (dim == 0) return;
                
                // Row-major stride calculation (consistent with typical broadcast behavior)
                m_strides[dim - 1] = 1;
                for (std::size_t i = dim - 1; i > 0; --i)
                {
                    m_strides[i - 1] = m_strides[i] * m_shape[i];
                }
                
                for (std::size_t i = 0; i < dim; ++i)
                {
                    m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
                }
            }
            
            template <class S>
            size_type compute_index(const S& index) const
            {
                size_type result = 0;
                for (std::size_t i = 0; i < m_shape.size(); ++i)
                {
                    result += static_cast<size_type>(index[i]) * m_strides[i];
                }
                return result;
            }
            
            template <std::size_t... I, class... Args>
            size_type compute_index_impl(std::index_sequence<I...>, Args... args) const
            {
                std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...};
                size_type result = 0;
                ((result += indices[I] * m_strides[I]), ...);
                return result;
            }
        };
        
        // --------------------------------------------------------------------
        // Helper function to create a broadcast expression
        // --------------------------------------------------------------------
        template <class E, class S>
        inline auto broadcast(E&& e, S&& shape)
        {
            using broadcast_type = xbroadcast<std::decay_t<E>, std::decay_t<S>>;
            return broadcast_type(std::forward<E>(e), std::forward<S>(shape));
        }
        
        // --------------------------------------------------------------------
        // broadcast_shape utility for xbroadcast
        // --------------------------------------------------------------------
        template <class S1, class S2>
        inline auto broadcast_shape(const S1& s1, const S2& s2)
        {
            using size_type = std::common_type_t<typename S1::value_type, typename S2::value_type>;
            std::vector<size_type> result;
            
            auto it1 = s1.rbegin();
            auto it2 = s2.rbegin();
            auto end1 = s1.rend();
            auto end2 = s2.rend();
            
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
        
        // --------------------------------------------------------------------
        // Advanced broadcasting: xbroadcast with explicit stride control
        // --------------------------------------------------------------------
        template <class CT>
        class xbroadcast_strided : public xexpression<xbroadcast_strided<CT>>
        {
        public:
            using self_type = xbroadcast_strided<CT>;
            using xexpression_type = std::decay_t<CT>;
            
            using value_type = typename xexpression_type::value_type;
            using reference = typename xexpression_type::reference;
            using const_reference = typename xexpression_type::const_reference;
            using pointer = typename xexpression_type::pointer;
            using const_pointer = typename xexpression_type::const_pointer;
            using size_type = typename xexpression_type::size_type;
            using difference_type = typename xexpression_type::difference_type;
            
            using shape_type = std::vector<size_type>;
            using strides_type = std::vector<size_type>;
            
            static constexpr bool is_const = std::is_const<CT>::value;
            
            template <class CTA, class S, class ST>
            xbroadcast_strided(CTA&& e, S&& shape, ST&& strides)
                : m_expression(std::forward<CTA>(e))
                , m_shape(std::forward<S>(shape))
                , m_strides(std::forward<ST>(strides))
            {
                m_size = compute_size(m_shape);
                m_backstrides.resize(m_strides.size());
                for (std::size_t i = 0; i < m_strides.size(); ++i)
                {
                    m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
                }
            }
            
            size_type size() const noexcept { return m_size; }
            size_type dimension() const noexcept { return m_shape.size(); }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return layout_type::dynamic; }
            
            const xexpression_type& expression() const noexcept { return m_expression; }
            
            const_reference flat(size_type i) const
            {
                if (m_expression.dimension() == 0)
                {
                    return m_expression();
                }
                
                size_type src_index = 0;
                size_type temp = i;
                std::ptrdiff_t dim_diff = static_cast<std::ptrdiff_t>(dimension()) - 
                                          static_cast<std::ptrdiff_t>(m_expression.dimension());
                
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    size_type coord = temp / m_strides[d];
                    temp %= m_strides[d];
                    
                    std::ptrdiff_t src_d = static_cast<std::ptrdiff_t>(d) - dim_diff;
                    if (src_d >= 0 && src_d < static_cast<std::ptrdiff_t>(m_expression.dimension()))
                    {
                        size_type src_dim_size = m_expression.shape()[static_cast<std::size_t>(src_d)];
                        if (src_dim_size == 1)
                        {
                            coord = 0;
                        }
                        src_index += coord * m_expression.strides()[static_cast<std::size_t>(src_d)];
                    }
                }
                return m_expression.flat(src_index);
            }
            
            template <class... Args>
            const_reference operator()(Args... args) const
            {
                size_type index = 0;
                std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...};
                for (std::size_t i = 0; i < sizeof...(Args); ++i)
                {
                    index += indices[i] * m_strides[i];
                }
                return flat(index);
            }
            
            // Iterator support
            class const_iterator
            {
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = typename self_type::value_type;
                using difference_type = std::ptrdiff_t;
                using pointer = const value_type*;
                using reference = value_type;
                
                const_iterator() = default;
                const_iterator(const self_type* broadcast, size_type index)
                    : m_broadcast(broadcast), m_index(index) {}
                
                reference operator*() const { return m_broadcast->flat(m_index); }
                
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                const_iterator& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                
                const_iterator operator+(difference_type n) const { return const_iterator(m_broadcast, m_index + static_cast<size_type>(n)); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_broadcast, m_index - static_cast<size_type>(n)); }
                
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                
                bool operator==(const const_iterator& rhs) const { return m_broadcast == rhs.m_broadcast && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                
                reference operator[](difference_type n) const { return m_broadcast->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
                
            private:
                const self_type* m_broadcast = nullptr;
                size_type m_index = 0;
            };
            
            using iterator = const_iterator;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            using reverse_iterator = const_reverse_iterator;
            
            const_iterator begin() const { return const_iterator(this, 0); }
            const_iterator end() const { return const_iterator(this, m_size); }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
            
            reverse_iterator rbegin() const { return reverse_iterator(end()); }
            reverse_iterator rend() const { return reverse_iterator(begin()); }
            const_reverse_iterator crbegin() const { return rbegin(); }
            const_reverse_iterator crend() const { return rend(); }
            
        private:
            CT m_expression;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
        };
        
        // --------------------------------------------------------------------
        // broadcast_to - Convenience function to broadcast to a specific shape
        // --------------------------------------------------------------------
        template <class E, class S>
        inline auto broadcast_to(E&& e, const S& shape)
        {
            return broadcast(std::forward<E>(e), shape);
        }
        
        // --------------------------------------------------------------------
        // expand_dims - Insert new axes into an expression
        // --------------------------------------------------------------------
        template <class E>
        inline auto expand_dims(E&& e, std::size_t axis)
        {
            auto& expr = e.derived_cast();
            auto old_shape = expr.shape();
            std::vector<typename std::decay_t<decltype(old_shape)>::value_type> new_shape = old_shape;
            if (axis > new_shape.size())
            {
                axis = new_shape.size();
            }
            new_shape.insert(new_shape.begin() + static_cast<std::ptrdiff_t>(axis), 1);
            return broadcast(std::forward<E>(e), new_shape);
        }
        
        template <class E, class Axes>
        inline auto expand_dims(E&& e, const Axes& axes)
        {
            auto result = e;
            std::vector<std::size_t> sorted_axes(axes.begin(), axes.end());
            std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<std::size_t>());
            for (auto axis : sorted_axes)
            {
                result = expand_dims(result, axis);
            }
            return result;
        }
        
        // --------------------------------------------------------------------
        // squeeze - Remove axes of length 1
        // --------------------------------------------------------------------
        template <class E>
        inline auto squeeze(E&& e)
        {
            auto& expr = e.derived_cast();
            auto old_shape = expr.shape();
            std::vector<std::size_t> new_shape;
            for (auto dim : old_shape)
            {
                if (dim != 1)
                {
                    new_shape.push_back(dim);
                }
            }
            if (new_shape.empty())
            {
                new_shape = {1}; // scalar becomes 1D of size 1
            }
            // Need a view that reshapes, but for simplicity we can return a reshaped view
            // Actually squeeze is a view operation in xtensor.
            // We'll implement it as a reshape_view if possible, but here we can use xstrided_view.
            // For now, we rely on xtensor's own squeeze implementation, but we can provide a placeholder.
            // Proper implementation would use xstrided_view with new shape and same strides for non-1 dims.
            return xt::reshape_view(expr, new_shape);
        }
        
        template <class E>
        inline auto squeeze(E&& e, std::size_t axis)
        {
            auto& expr = e.derived_cast();
            if (expr.shape()[axis] != 1)
            {
                XTENSOR_THROW(std::runtime_error, "Cannot squeeze axis with size != 1");
            }
            auto new_shape = expr.shape();
            new_shape.erase(new_shape.begin() + static_cast<std::ptrdiff_t>(axis));
            return xt::reshape_view(expr, new_shape);
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XBROADCAST_HPP

// core/xbroadcast.hpp