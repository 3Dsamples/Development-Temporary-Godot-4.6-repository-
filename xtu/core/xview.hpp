// core/xview.hpp

#ifndef XTENSOR_XVIEW_HPP
#define XTENSOR_XVIEW_HPP

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xbroadcast.hpp"

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
#include <tuple>
#include <cassert>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // slice utilities
        // --------------------------------------------------------------------
        
        struct xall_tag {};
        struct xnewaxis_tag {};
        struct xellipsis_tag {};
        struct xkeep_dim_tag {};
        struct xdrop_dim_tag {};
        
        XTENSOR_INLINE_VARIABLE xall_tag all = {};
        XTENSOR_INLINE_VARIABLE xnewaxis_tag newaxis = {};
        XTENSOR_INLINE_VARIABLE xellipsis_tag ellipsis = {};
        XTENSOR_INLINE_VARIABLE xkeep_dim_tag keep_dim = {};
        XTENSOR_INLINE_VARIABLE xdrop_dim_tag drop_dim = {};
        
        template <class T>
        class xrange
        {
        public:
            using size_type = T;
            
            xrange() : m_start(0), m_stop(0), m_step(1) {}
            xrange(size_type stop) : m_start(0), m_stop(stop), m_step(1) {}
            xrange(size_type start, size_type stop, size_type step = 1)
                : m_start(start), m_stop(stop), m_step(step)
            {
                if (m_step == 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "xrange step cannot be zero");
                }
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
            
            size_type operator[](size_type i) const
            {
                return m_start + i * m_step;
            }
            
        private:
            size_type m_start;
            size_type m_stop;
            size_type m_step;
        };
        
        template <class T>
        inline auto range(T stop)
        {
            return xrange<T>(stop);
        }
        
        template <class T>
        inline auto range(T start, T stop, T step = 1)
        {
            return xrange<T>(start, stop, step);
        }
        
        // Placeholder for advanced indexing
        template <class E>
        class xplaceholder
        {
        public:
            using value_type = typename std::decay_t<E>::value_type;
            
            explicit xplaceholder(E&& expr) : m_expr(std::forward<E>(expr)) {}
            
            const auto& expression() const { return m_expr; }
            
        private:
            E m_expr;
        };
        
        template <class E>
        inline auto placeholders::xtensor_placeholder(E&& e)
        {
            return xplaceholder<E>(std::forward<E>(e));
        }
        
        // --------------------------------------------------------------------
        // xview - Sliced view of an expression
        // --------------------------------------------------------------------
        template <class CT, class... S>
        class xview : public xexpression<xview<CT, S...>>
        {
        public:
            using self_type = xview<CT, S...>;
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
            using slice_type = std::tuple<S...>;
            
            using expression_tag = xview_tag;
            
            static constexpr bool is_const = std::is_const<CT>::value;
            static constexpr bool is_assignable = !is_const && xexpression_type::is_assignable;
            
            // Construction
            template <class CTA, class... Args>
            explicit xview(CTA&& e, Args&&... args)
                : m_expression(std::forward<CTA>(e))
                , m_slices(std::forward<Args>(args)...)
            {
                compute_shape_and_strides();
                m_size = compute_size(m_shape);
            }
            
            // Copy/move
            xview(const self_type&) = default;
            xview(self_type&&) = default;
            
            // Assignment (only if underlying expression is mutable)
            template <class E>
            std::enable_if_t<is_assignable, self_type&> operator=(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                if (expr.dimension() != dimension() || expr.shape() != m_shape)
                {
                    XTENSOR_THROW(std::runtime_error, "Shape mismatch in view assignment");
                }
                if (expr.layout() == layout_type::row_major)
                {
                    std::copy(expr.begin(), expr.end(), begin());
                }
                else
                {
                    for (size_type i = 0; i < m_size; ++i)
                    {
                        flat(i) = expr.flat(i);
                    }
                }
                return *this;
            }
            
            self_type& operator=(const self_type& rhs)
            {
                if (this != &rhs)
                {
                    if (rhs.shape() != m_shape)
                    {
                        XTENSOR_THROW(std::runtime_error, "Shape mismatch in view assignment");
                    }
                    std::copy(rhs.begin(), rhs.end(), begin());
                }
                return *this;
            }
            
            template <class T>
            std::enable_if_t<is_assignable && !std::is_base_of<xexpression<T>, T>::value, self_type&>
            operator=(const T& value)
            {
                std::fill(begin(), end(), value);
                return *this;
            }
            
            // Size and shape
            size_type size() const noexcept { return m_size; }
            size_type dimension() const noexcept { return m_shape.size(); }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return layout_type::dynamic; }
            
            // Access to underlying expression and slices
            const xexpression_type& expression() const noexcept { return m_expression; }
            const slice_type& slices() const noexcept { return m_slices; }
            
            // Element access
            template <class... Args>
            reference operator()(Args... args)
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            template <class... Args>
            const_reference operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            template <class... Args>
            reference unchecked(Args... args)
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
            reference element(const S& index)
            {
                return flat(compute_index(index));
            }
            
            template <class S>
            const_reference element(const S& index) const
            {
                return flat(compute_index(index));
            }
            
            reference flat(size_type i)
            {
                size_type src_index = compute_src_index(i);
                return m_expression.flat(src_index);
            }
            
            const_reference flat(size_type i) const
            {
                size_type src_index = compute_src_index(i);
                return m_expression.flat(src_index);
            }
            
            // Data access (only if contiguous)
            pointer data() noexcept
            {
                return is_contiguous() ? m_expression.data() + m_data_offset : nullptr;
            }
            
            const_pointer data() const noexcept
            {
                return is_contiguous() ? m_expression.data() + m_data_offset : nullptr;
            }
            
            bool is_contiguous() const noexcept
            {
                if (m_expression.layout() == layout_type::row_major)
                {
                    size_type expected_stride = 1;
                    for (std::size_t i = m_shape.size(); i > 0; --i)
                    {
                        if (m_strides[i-1] != expected_stride) return false;
                        expected_stride *= m_shape[i-1];
                    }
                    return true;
                }
                else if (m_expression.layout() == layout_type::column_major)
                {
                    size_type expected_stride = 1;
                    for (std::size_t i = 0; i < m_shape.size(); ++i)
                    {
                        if (m_strides[i] != expected_stride) return false;
                        expected_stride *= m_shape[i];
                    }
                    return true;
                }
                return false;
            }
            
            // Iterator support
            class iterator_impl
            {
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = typename self_type::value_type;
                using difference_type = std::ptrdiff_t;
                using pointer = value_type*;
                using reference = value_type&;
                
                iterator_impl() = default;
                iterator_impl(self_type* view, size_type index)
                    : m_view(view), m_index(index) {}
                
                reference operator*() const { return m_view->flat(m_index); }
                
                iterator_impl& operator++() { ++m_index; return *this; }
                iterator_impl operator++(int) { iterator_impl tmp = *this; ++*this; return tmp; }
                iterator_impl& operator--() { --m_index; return *this; }
                iterator_impl operator--(int) { iterator_impl tmp = *this; --*this; return tmp; }
                iterator_impl& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                iterator_impl& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                
                iterator_impl operator+(difference_type n) const { return iterator_impl(m_view, m_index + static_cast<size_type>(n)); }
                iterator_impl operator-(difference_type n) const { return iterator_impl(m_view, m_index - static_cast<size_type>(n)); }
                
                difference_type operator-(const iterator_impl& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                
                bool operator==(const iterator_impl& rhs) const { return m_view == rhs.m_view && m_index == rhs.m_index; }
                bool operator!=(const iterator_impl& rhs) const { return !(*this == rhs); }
                bool operator<(const iterator_impl& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const iterator_impl& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const iterator_impl& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const iterator_impl& rhs) const { return m_index >= rhs.m_index; }
                
                reference operator[](difference_type n) const { return m_view->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
                
            private:
                self_type* m_view = nullptr;
                size_type m_index = 0;
            };
            
            using iterator = std::conditional_t<is_const, const iterator_impl, iterator_impl>;
            using const_iterator = iterator_impl; // Actually const version should be separate but we simplify
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            
            iterator begin() { return iterator(this, 0); }
            iterator end() { return iterator(this, m_size); }
            const_iterator begin() const { return const_iterator(const_cast<self_type*>(this), 0); }
            const_iterator end() const { return const_iterator(const_cast<self_type*>(this), m_size); }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
            
            reverse_iterator rbegin() { return reverse_iterator(end()); }
            reverse_iterator rend() { return reverse_iterator(begin()); }
            const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
            const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
            const_reverse_iterator crbegin() const { return rbegin(); }
            const_reverse_iterator crend() const { return rend(); }
            
            // View operations
            template <class... NewS>
            auto view(NewS&&... new_slices) const
            {
                return xview<const self_type, NewS...>(*this, std::forward<NewS>(new_slices)...);
            }
            
            template <class... NewS>
            auto view(NewS&&... new_slices)
            {
                return xview<self_type, NewS...>(*this, std::forward<NewS>(new_slices)...);
            }
            
            template <class S>
            auto broadcast(const S& new_shape) const
            {
                shape_type shape(new_shape.begin(), new_shape.end());
                return xbroadcast<const self_type, shape_type>(*this, std::move(shape));
            }
            
        private:
            CT m_expression;
            slice_type m_slices;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            size_type m_data_offset = 0; // offset in flat data if contiguous
            
            void compute_shape_and_strides()
            {
                std::size_t expr_dim = m_expression.dimension();
                const auto& expr_shape = m_expression.shape();
                const auto& expr_strides = m_expression.strides();
                
                // First pass: count output dimensions and collect slice info
                std::vector<std::ptrdiff_t> slice_indices; // mapping from output dim to slice index
                std::vector<bool> is_range;
                std::vector<size_type> range_sizes;
                std::vector<size_type> range_steps;
                std::vector<size_type> range_starts;
                std::vector<size_type> integer_indices; // for integer slices (reduce dimension)
                
                process_slices(expr_dim, expr_shape, slice_indices, is_range, range_sizes, range_steps, range_starts, integer_indices);
                
                // Build output shape and strides
                std::size_t out_dim = 0;
                for (std::size_t i = 0; i < slice_indices.size(); ++i)
                {
                    std::ptrdiff_t slice_idx = slice_indices[i];
                    if (slice_idx >= 0)
                    {
                        if (is_range[static_cast<std::size_t>(slice_idx)])
                        {
                            m_shape.push_back(range_sizes[static_cast<std::size_t>(slice_idx)]);
                            ++out_dim;
                        }
                        else
                        {
                            // integer slice, dimension dropped
                        }
                    }
                    else if (slice_idx == -1) // newaxis
                    {
                        m_shape.push_back(1);
                        ++out_dim;
                    }
                }
                
                m_strides.resize(out_dim);
                m_backstrides.resize(out_dim);
                
                // Compute strides
                out_dim = 0;
                size_type stride_accum = 1;
                for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(slice_indices.size()) - 1; i >= 0; --i)
                {
                    std::ptrdiff_t slice_idx = slice_indices[static_cast<std::size_t>(i)];
                    if (slice_idx >= 0)
                    {
                        if (is_range[static_cast<std::size_t>(slice_idx)])
                        {
                            size_type step = range_steps[static_cast<std::size_t>(slice_idx)];
                            size_type orig_stride = expr_strides[static_cast<std::size_t>(slice_idx)];
                            m_strides[out_dim] = stride_accum;
                            stride_accum *= m_shape[out_dim];
                            // For efficient element access, we need to map flat index to source index
                            // We'll compute the source offset and stride factor separately.
                            // We'll store the mapping in separate arrays for flat() computation.
                            ++out_dim;
                        }
                    }
                    else if (slice_idx == -1)
                    {
                        m_strides[out_dim] = stride_accum;
                        stride_accum *= 1;
                        ++out_dim;
                    }
                }
                std::reverse(m_strides.begin(), m_strides.end());
                
                // Adjust stride order (we built from inner to outer, but we reversed? Actually we built from last to first and didn't reverse, need to ensure correct)
                // Let's recalc properly:
                out_dim = 0;
                for (std::size_t i = 0; i < slice_indices.size(); ++i)
                {
                    std::ptrdiff_t slice_idx = slice_indices[i];
                    if (slice_idx >= 0)
                    {
                        if (is_range[static_cast<std::size_t>(slice_idx)])
                        {
                            ++out_dim;
                        }
                    }
                    else if (slice_idx == -1)
                    {
                        ++out_dim;
                    }
                }
                // recompute strides from outer to inner
                if (out_dim > 0)
                {
                    m_strides[out_dim - 1] = 1;
                    for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(out_dim) - 2; j >= 0; --j)
                    {
                        m_strides[static_cast<std::size_t>(j)] = m_strides[static_cast<std::size_t>(j) + 1] * m_shape[static_cast<std::size_t>(j) + 1];
                    }
                }
                
                for (std::size_t i = 0; i < out_dim; ++i)
                {
                    m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
                }
                
                // Precompute source index mapping parameters
                compute_src_mapping(expr_dim, expr_strides, slice_indices, is_range, range_steps, range_starts, integer_indices);
            }
            
            void compute_src_mapping(std::size_t expr_dim, const strides_type& expr_strides,
                                     const std::vector<std::ptrdiff_t>& slice_indices,
                                     const std::vector<bool>& is_range,
                                     const std::vector<size_type>& range_steps,
                                     const std::vector<size_type>& range_starts,
                                     const std::vector<size_type>& integer_indices)
            {
                // We'll store the mapping parameters in member variables for flat() to use.
                // Instead of recomputing each time, we can store per-dimension info.
                m_src_strides.resize(dimension());
                m_src_steps.resize(dimension());
                m_src_starts.resize(dimension());
                m_src_dim_map.resize(dimension(), std::numeric_limits<std::size_t>::max());
                
                std::size_t out_dim = 0;
                std::size_t integer_count = 0;
                for (std::size_t i = 0; i < slice_indices.size(); ++i)
                {
                    std::ptrdiff_t slice_idx = slice_indices[i];
                    if (slice_idx >= 0)
                    {
                        if (is_range[static_cast<std::size_t>(slice_idx)])
                        {
                            m_src_dim_map[out_dim] = static_cast<std::size_t>(slice_idx);
                            m_src_strides[out_dim] = expr_strides[static_cast<std::size_t>(slice_idx)];
                            m_src_steps[out_dim] = range_steps[static_cast<std::size_t>(slice_idx)];
                            m_src_starts[out_dim] = range_starts[static_cast<std::size_t>(slice_idx)];
                            ++out_dim;
                        }
                        else
                        {
                            // integer slice, store offset contribution
                            m_integer_offsets.push_back(integer_indices[integer_count] * expr_strides[static_cast<std::size_t>(slice_idx)]);
                            ++integer_count;
                        }
                    }
                    else if (slice_idx == -1)
                    {
                        // newaxis, no source dimension
                        m_src_dim_map[out_dim] = std::numeric_limits<std::size_t>::max();
                        m_src_strides[out_dim] = 0;
                        m_src_steps[out_dim] = 1;
                        m_src_starts[out_dim] = 0;
                        ++out_dim;
                    }
                }
                
                m_base_offset = std::accumulate(m_integer_offsets.begin(), m_integer_offsets.end(), size_type(0));
            }
            
            size_type compute_src_index(size_type flat_index) const
            {
                // Unravel flat_index to coordinates in output view
                size_type src_offset = m_base_offset;
                size_type temp = flat_index;
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    size_type coord = temp / m_strides[d];
                    temp %= m_strides[d];
                    if (m_src_dim_map[d] != std::numeric_limits<std::size_t>::max())
                    {
                        size_type src_coord = m_src_starts[d] + coord * m_src_steps[d];
                        src_offset += src_coord * m_src_strides[d];
                    }
                }
                return src_offset;
            }
            
            std::vector<size_type> m_src_strides;
            std::vector<size_type> m_src_steps;
            std::vector<size_type> m_src_starts;
            std::vector<std::size_t> m_src_dim_map;
            std::vector<size_type> m_integer_offsets;
            size_type m_base_offset = 0;
            
            void process_slices(std::size_t expr_dim,
                                const shape_type& expr_shape,
                                std::vector<std::ptrdiff_t>& slice_indices,
                                std::vector<bool>& is_range,
                                std::vector<size_type>& range_sizes,
                                std::vector<size_type>& range_steps,
                                std::vector<size_type>& range_starts,
                                std::vector<size_type>& integer_indices)
            {
                // We need to handle ellipsis: replace with as many 'all' as needed.
                std::size_t num_slices = sizeof...(S);
                std::vector<std::size_t> slice_order;
                bool has_ellipsis = false;
                std::size_t ellipsis_pos = 0;
                
                // First, determine if there's an ellipsis
                analyze_slice_types(has_ellipsis, ellipsis_pos);
                
                if (has_ellipsis)
                {
                    std::size_t slices_before = ellipsis_pos;
                    std::size_t slices_after = num_slices - ellipsis_pos - 1;
                    std::size_t ellipsis_count = expr_dim > (slices_before + slices_after) ? expr_dim - (slices_before + slices_after) : 0;
                    
                    // Build slice indices: for each dimension, map to slice argument index
                    std::size_t arg_idx = 0;
                    for (std::size_t d = 0; d < expr_dim; ++d)
                    {
                        if (d < slices_before)
                        {
                            slice_indices.push_back(static_cast<std::ptrdiff_t>(d));
                            ++arg_idx;
                        }
                        else if (d >= slices_before && d < slices_before + ellipsis_count)
                        {
                            // ellipsis: treat as 'all' (range(0, expr_shape[d]))
                            if (arg_idx == ellipsis_pos)
                            {
                                // skip the ellipsis token
                                ++arg_idx;
                            }
                            // we'll treat this as a range slice
                            range_sizes.push_back(expr_shape[d]);
                            range_steps.push_back(1);
                            range_starts.push_back(0);
                            is_range.push_back(true);
                            slice_indices.push_back(static_cast<std::ptrdiff_t>(range_sizes.size() - 1));
                        }
                        else
                        {
                            // after ellipsis
                            slice_indices.push_back(static_cast<std::ptrdiff_t>(arg_idx));
                            ++arg_idx;
                        }
                    }
                    // Now process the actual slice arguments to fill the vectors
                    fill_slice_vectors(expr_shape, range_sizes, range_steps, range_starts, is_range, integer_indices);
                }
                else
                {
                    if (num_slices != expr_dim)
                    {
                        XTENSOR_THROW(std::invalid_argument, "Number of slices must match expression dimension when no ellipsis");
                    }
                    for (std::size_t d = 0; d < expr_dim; ++d)
                    {
                        slice_indices.push_back(static_cast<std::ptrdiff_t>(d));
                    }
                    fill_slice_vectors(expr_shape, range_sizes, range_steps, range_starts, is_range, integer_indices);
                }
            }
            
            template <std::size_t... I>
            void analyze_slice_types_impl(std::index_sequence<I...>, bool& has_ellipsis, std::size_t& pos) const
            {
                std::array<bool, sizeof...(I)> is_ellipsis = {std::is_same<std::decay_t<decltype(std::get<I>(m_slices))>, xellipsis_tag>::value...};
                std::size_t count = 0;
                ((is_ellipsis[I] ? (has_ellipsis = true, pos = I, ++count) : void()), ...);
                if (count > 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "Only one ellipsis allowed");
                }
            }
            
            void analyze_slice_types(bool& has_ellipsis, std::size_t& pos) const
            {
                analyze_slice_types_impl(std::index_sequence_for<S...>{}, has_ellipsis, pos);
            }
            
            template <std::size_t... I>
            void fill_slice_vectors_impl(std::index_sequence<I...>,
                                         const shape_type& expr_shape,
                                         std::vector<size_type>& range_sizes,
                                         std::vector<size_type>& range_steps,
                                         std::vector<size_type>& range_starts,
                                         std::vector<bool>& is_range,
                                         std::vector<size_type>& integer_indices) const
            {
                // Process each slice argument
                (void)std::initializer_list<int>{
                    (process_one_slice<I>(std::get<I>(m_slices), expr_shape[I], range_sizes, range_steps, range_starts, is_range, integer_indices), 0)...
                };
            }
            
            template <std::size_t I, class Slice>
            void process_one_slice(const Slice& slice,
                                   size_type dim_size,
                                   std::vector<size_type>& range_sizes,
                                   std::vector<size_type>& range_steps,
                                   std::vector<size_type>& range_starts,
                                   std::vector<bool>& is_range,
                                   std::vector<size_type>& integer_indices) const
            {
                using slice_type = std::decay_t<Slice>;
                if constexpr (std::is_same<slice_type, xall_tag>::value)
                {
                    range_sizes.push_back(dim_size);
                    range_steps.push_back(1);
                    range_starts.push_back(0);
                    is_range.push_back(true);
                }
                else if constexpr (std::is_same<slice_type, xnewaxis_tag>::value)
                {
                    // handled separately in slice_indices
                }
                else if constexpr (std::is_same<slice_type, xellipsis_tag>::value)
                {
                    // skip
                }
                else if constexpr (std::is_integral<slice_type>::value)
                {
                    size_type idx = static_cast<size_type>(slice);
                    if (idx >= dim_size)
                    {
                        XTENSOR_THROW(std::out_of_range, "Index out of bounds in view");
                    }
                    integer_indices.push_back(idx);
                    is_range.push_back(false);
                    range_sizes.push_back(0); // placeholder
                    range_steps.push_back(0);
                    range_starts.push_back(0);
                }
                else if constexpr (std::is_same<slice_type, xrange<size_type>>::value)
                {
                    size_type start = slice.start();
                    size_type stop = slice.stop();
                    size_type step = slice.step();
                    if (start >= dim_size) start = dim_size;
                    if (stop > dim_size) stop = dim_size;
                    size_type size = slice.size();
                    range_sizes.push_back(size);
                    range_steps.push_back(step);
                    range_starts.push_back(start);
                    is_range.push_back(true);
                }
                else
                {
                    // assume it's an integer index container (e.g., vector<int>)
                    // advanced indexing not fully implemented here
                    XTENSOR_THROW(not_implemented_error, "Advanced indexing not implemented");
                }
            }
            
            void fill_slice_vectors(const shape_type& expr_shape,
                                    std::vector<size_type>& range_sizes,
                                    std::vector<size_type>& range_steps,
                                    std::vector<size_type>& range_starts,
                                    std::vector<bool>& is_range,
                                    std::vector<size_type>& integer_indices)
            {
                fill_slice_vectors_impl(std::index_sequence_for<S...>{}, expr_shape, range_sizes, range_steps, range_starts, is_range, integer_indices);
            }
            
            template <class S>
            size_type compute_index(const S& index) const
            {
                size_type result = 0;
                for (std::size_t i = 0; i < dimension(); ++i)
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
        // xstrided_view - View with explicit shape and strides
        // --------------------------------------------------------------------
        template <class CT, class... S>
        class xstrided_view : public xexpression<xstrided_view<CT, S...>>
        {
        public:
            using self_type = xstrided_view<CT, S...>;
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
            
            template <class CTA, class Shape, class Strides>
            xstrided_view(CTA&& e, Shape&& shape, Strides&& strides, size_type offset = 0)
                : m_expression(std::forward<CTA>(e))
                , m_shape(std::forward<Shape>(shape))
                , m_strides(std::forward<Strides>(strides))
                , m_offset(offset)
            {
                m_size = compute_size(m_shape);
                compute_backstrides();
            }
            
            size_type size() const noexcept { return m_size; }
            size_type dimension() const noexcept { return m_shape.size(); }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return layout_type::dynamic; }
            
            const xexpression_type& expression() const noexcept { return m_expression; }
            size_type offset() const noexcept { return m_offset; }
            
            reference flat(size_type i)
            {
                size_type src_index = m_offset + compute_src_index(i);
                return m_expression.flat(src_index);
            }
            
            const_reference flat(size_type i) const
            {
                size_type src_index = m_offset + compute_src_index(i);
                return m_expression.flat(src_index);
            }
            
            template <class... Args>
            reference operator()(Args... args)
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            template <class... Args>
            const_reference operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            // Iterator support similar to xview (omitted for brevity but would be implemented)
            
        private:
            CT m_expression;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            size_type m_offset;
            
            void compute_backstrides()
            {
                m_backstrides.resize(m_strides.size());
                for (std::size_t i = 0; i < m_strides.size(); ++i)
                {
                    m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
                }
            }
            
            size_type compute_src_index(size_type flat_index) const
            {
                size_type src_offset = 0;
                size_type temp = flat_index;
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    size_type coord = temp / m_strides[d];
                    temp %= m_strides[d];
                    src_offset += coord * m_expression.strides()[d]; // Assuming dimension matches; for broadcast use view's strides logic.
                }
                return src_offset;
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
        // Helper functions to create views
        // --------------------------------------------------------------------
        template <class E, class... S>
        inline auto view(E&& e, S&&... slices)
        {
            return xview<std::decay_t<E>, std::decay_t<S>...>(std::forward<E>(e), std::forward<S>(slices)...);
        }
        
        template <class E, class Shape, class Strides>
        inline auto strided_view(E&& e, Shape&& shape, Strides&& strides, std::size_t offset = 0)
        {
            return xstrided_view<std::decay_t<E>, std::decay_t<Shape>, std::decay_t<Strides>>(
                std::forward<E>(e), std::forward<Shape>(shape), std::forward<Strides>(strides), offset);
        }
        
        // reshape_view
        template <class E, class S>
        inline auto reshape_view(E&& e, const S& new_shape)
        {
            auto& expr = e.derived_cast();
            std::vector<typename std::decay_t<decltype(expr.shape())>::value_type> shape(new_shape.begin(), new_shape.end());
            auto new_size = compute_size(shape);
            if (new_size != expr.size())
            {
                XTENSOR_THROW(std::runtime_error, "reshape_view cannot change total size");
            }
            // If contiguous, we can reuse strides; otherwise we need to compute new strides.
            // Simplified: assume contiguous row-major.
            std::vector<std::size_t> new_strides(shape.size());
            if (!shape.empty())
            {
                new_strides.back() = 1;
                for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2; i >= 0; --i)
                {
                    new_strides[static_cast<std::size_t>(i)] = new_strides[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
                }
            }
            return strided_view(std::forward<E>(e), shape, new_strides, 0);
        }
        
        // flatten view (returns 1D view)
        template <class E>
        inline auto flatten_view(E&& e)
        {
            auto& expr = e.derived_cast();
            std::vector<std::size_t> shape = {expr.size()};
            std::vector<std::size_t> strides = {1};
            return strided_view(std::forward<E>(e), shape, strides, 0);
        }
        
        // diagonal view
        template <class E>
        inline auto diagonal_view(E&& e, std::ptrdiff_t offset = 0)
        {
            auto& expr = e.derived_cast();
            if (expr.dimension() != 2)
            {
                XTENSOR_THROW(std::runtime_error, "diagonal_view requires 2D expression");
            }
            std::size_t rows = expr.shape()[0];
            std::size_t cols = expr.shape()[1];
            std::size_t diag_size = 0;
            std::size_t start_row = 0, start_col = 0;
            if (offset >= 0)
            {
                diag_size = std::min(cols - static_cast<std::size_t>(offset), rows);
                start_col = static_cast<std::size_t>(offset);
            }
            else
            {
                diag_size = std::min(rows - static_cast<std::size_t>(-offset), cols);
                start_row = static_cast<std::size_t>(-offset);
            }
            std::vector<std::size_t> shape = {diag_size};
            std::vector<std::size_t> strides = {expr.strides()[0] + expr.strides()[1]};
            size_type offset_flat = start_row * expr.strides()[0] + start_col * expr.strides()[1];
            return strided_view(std::forward<E>(e), shape, strides, offset_flat);
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XVIEW_HPP

// core/xview.hpp