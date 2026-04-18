// core/xaccumulator.hpp

#ifndef XTENSOR_XACCUMULATOR_HPP
#define XTENSOR_XACCUMULATOR_HPP

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"

#include <cstddef>
#include <type_traits>
#include <functional>
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iterator>
#include <tuple>
#include <cassert>
#include <cmath>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // accumulator function objects
        // --------------------------------------------------------------------
        namespace detail
        {
            template <class T>
            struct sum_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 0;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        accum += *it;
                        *out = accum;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 0;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        accum += *it;
                        *rout = accum;
                    }
                }
            };
            
            template <class T>
            struct prod_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 1;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        accum *= *it;
                        *out = accum;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 1;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        accum *= *it;
                        *rout = accum;
                    }
                }
            };
            
            template <class T>
            struct min_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    if (first == last) return;
                    result_type accum = *first;
                    *out = accum;
                    ++first; ++out;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        accum = std::min(accum, *it);
                        *out = accum;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    if (first == last) return;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    result_type accum = *rit;
                    *rout = accum;
                    ++rit; ++rout;
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        accum = std::min(accum, *it);
                        *rout = accum;
                    }
                }
            };
            
            template <class T>
            struct max_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    if (first == last) return;
                    result_type accum = *first;
                    *out = accum;
                    ++first; ++out;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        accum = std::max(accum, *it);
                        *out = accum;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    if (first == last) return;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    result_type accum = *rit;
                    *rout = accum;
                    ++rit; ++rout;
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        accum = std::max(accum, *it);
                        *rout = accum;
                    }
                }
            };
            
            template <class T>
            struct mean_accumulator
            {
                using result_type = double;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type sum = 0;
                    std::size_t count = 0;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        sum += static_cast<result_type>(*it);
                        ++count;
                        *out = sum / static_cast<result_type>(count);
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type sum = 0;
                    std::size_t count = 0;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        sum += static_cast<result_type>(*it);
                        ++count;
                        *rout = sum / static_cast<result_type>(count);
                    }
                }
            };
            
            template <class T>
            struct variance_accumulator
            {
                using result_type = double;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type mean = 0;
                    result_type m2 = 0;
                    std::size_t count = 0;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        ++count;
                        result_type delta = static_cast<result_type>(*it) - mean;
                        mean += delta / static_cast<result_type>(count);
                        result_type delta2 = static_cast<result_type>(*it) - mean;
                        m2 += delta * delta2;
                        *out = (count > 1) ? m2 / static_cast<result_type>(count - 1) : 0;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    // For reverse, we process in reverse order
                    result_type mean = 0;
                    result_type m2 = 0;
                    std::size_t count = 0;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        ++count;
                        result_type delta = static_cast<result_type>(*it) - mean;
                        mean += delta / static_cast<result_type>(count);
                        result_type delta2 = static_cast<result_type>(*it) - mean;
                        m2 += delta * delta2;
                        *rout = (count > 1) ? m2 / static_cast<result_type>(count - 1) : 0;
                    }
                }
            };
            
            // NaN-aware accumulators
            template <class T>
            struct nansum_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 0;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                            accum += *it;
                        *out = accum;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 0;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                            accum += *it;
                        *rout = accum;
                    }
                }
            };
            
            template <class T>
            struct nanprod_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 1;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                            accum *= *it;
                        *out = accum;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = 1;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                            accum *= *it;
                        *rout = accum;
                    }
                }
            };
            
            template <class T>
            struct nanmin_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = std::numeric_limits<result_type>::max();
                    bool found = false;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            if (!found || *it < accum)
                            {
                                accum = *it;
                                found = true;
                            }
                        }
                        *out = found ? accum : result_type(0);
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = std::numeric_limits<result_type>::max();
                    bool found = false;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            if (!found || *it < accum)
                            {
                                accum = *it;
                                found = true;
                            }
                        }
                        *rout = found ? accum : result_type(0);
                    }
                }
            };
            
            template <class T>
            struct nanmax_accumulator
            {
                using result_type = T;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = std::numeric_limits<result_type>::lowest();
                    bool found = false;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            if (!found || *it > accum)
                            {
                                accum = *it;
                                found = true;
                            }
                        }
                        *out = found ? accum : result_type(0);
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type accum = std::numeric_limits<result_type>::lowest();
                    bool found = false;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            if (!found || *it > accum)
                            {
                                accum = *it;
                                found = true;
                            }
                        }
                        *rout = found ? accum : result_type(0);
                    }
                }
            };
            
            template <class T>
            struct nanmean_accumulator
            {
                using result_type = double;
                
                template <class InputIt, class OutputIt>
                void operator()(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type sum = 0;
                    std::size_t count = 0;
                    for (auto it = first; it != last; ++it, ++out)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            sum += static_cast<result_type>(*it);
                            ++count;
                        }
                        *out = count > 0 ? sum / static_cast<result_type>(count) : 0;
                    }
                }
                
                template <class InputIt, class OutputIt>
                void reverse(InputIt first, InputIt last, OutputIt out) const
                {
                    result_type sum = 0;
                    std::size_t count = 0;
                    auto rit = std::make_reverse_iterator(last);
                    auto rend = std::make_reverse_iterator(first);
                    auto rout = std::make_reverse_iterator(out + std::distance(first, last));
                    for (auto it = rit; it != rend; ++it, ++rout)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            sum += static_cast<result_type>(*it);
                            ++count;
                        }
                        *rout = count > 0 ? sum / static_cast<result_type>(count) : 0;
                    }
                }
            };
            
        } // namespace detail
        
        // --------------------------------------------------------------------
        // xaccumulator - Expression that performs cumulative operations
        // --------------------------------------------------------------------
        template <class E>
        class xaccumulator : public xexpression<xaccumulator<E>>
        {
        public:
            using self_type = xaccumulator<E>;
            using base_type = xexpression<self_type>;
            using xexpression_type = std::decay_t<E>;
            
            using value_type = typename xexpression_type::value_type;
            using reference = value_type;
            using const_reference = value_type;
            using pointer = value_type*;
            using const_pointer = const value_type*;
            using size_type = typename xexpression_type::size_type;
            using difference_type = typename xexpression_type::difference_type;
            
            using shape_type = typename xexpression_type::shape_type;
            using strides_type = typename xexpression_type::strides_type;
            
            using expression_tag = xaccumulator_tag;
            static constexpr bool is_const = true;
            
            // Construction
            template <class EX, class F>
            xaccumulator(EX&& e, F&& func, std::size_t axis, bool reverse = false)
                : m_expression(std::forward<EX>(e))
                , m_functor(std::forward<F>(func))
                , m_axis(normalize_axis(static_cast<std::ptrdiff_t>(axis), e.dimension()))
                , m_reverse(reverse)
                , m_shape(e.shape())
                , m_strides(e.strides())
                , m_backstrides(e.backstrides())
                , m_size(e.size())
            {
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
            std::size_t axis() const noexcept { return m_axis; }
            bool reverse() const noexcept { return m_reverse; }
            
            // Element access
            template <class... Args>
            value_type operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            value_type flat(size_type i) const
            {
                return compute_accumulated(i);
            }
            
            template <class S>
            value_type element(const S& index) const
            {
                return flat(compute_index(index));
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
                const_iterator(const self_type* accum, size_type index)
                    : m_accum(accum), m_index(index) {}
                
                reference operator*() const { return m_accum->flat(m_index); }
                
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                const_iterator& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                
                const_iterator operator+(difference_type n) const { return const_iterator(m_accum, m_index + static_cast<size_type>(n)); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_accum, m_index - static_cast<size_type>(n)); }
                
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                
                bool operator==(const const_iterator& rhs) const { return m_accum == rhs.m_accum && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const const_iterator& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const const_iterator& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const const_iterator& rhs) const { return m_index >= rhs.m_index; }
                
                reference operator[](difference_type n) const { return m_accum->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
                
            private:
                const self_type* m_accum = nullptr;
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
            
            // Disable assignment
            template <class EX>
            disable_xexpression<EX, self_type&> operator=(const EX&) = delete;
            
        private:
            E m_expression;
            std::function<void(const value_type*, const value_type*, value_type*)> m_functor; // simplified type
            std::size_t m_axis;
            bool m_reverse;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            
            value_type compute_accumulated(size_type flat_index) const
            {
                // Unravel flat_index to multi-index
                std::vector<size_type> coords(dimension());
                size_type temp = flat_index;
                if (m_reverse)
                {
                    // For reverse, we need to unravel differently? No, unraveling is same.
                }
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    coords[d] = temp / m_strides[d];
                    temp %= m_strides[d];
                }
                
                // We need to traverse along the specified axis up to the current coordinate.
                size_type axis_len = m_shape[m_axis];
                size_type current_pos = coords[m_axis];
                
                // Build base index with axis coordinate set to 0
                std::vector<size_type> base_coords = coords;
                base_coords[m_axis] = 0;
                
                // Compute flat index of the start of this line
                size_type line_start = 0;
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    line_start += base_coords[d] * m_strides[d];
                }
                
                // Determine the stride along the axis
                size_type axis_stride = m_strides[m_axis];
                
                // If reverse, we start from the end of the axis and move backward
                if (m_reverse)
                {
                    // For reverse accumulation, the value at position `current_pos` is the accumulation
                    // from the end to `current_pos` (inclusive).
                    // So we start from axis_len-1 and go backwards to current_pos.
                    std::vector<value_type> values(axis_len - current_pos);
                    size_type idx = line_start + (axis_len - 1) * axis_stride;
                    for (size_type i = 0; i < axis_len - current_pos; ++i)
                    {
                        values[i] = m_expression.flat(idx);
                        idx -= axis_stride;
                    }
                    // Apply reverse accumulator
                    value_type result;
                    std::vector<value_type> temp_out(values.size());
                    m_functor_reverse(values.data(), values.data() + values.size(), temp_out.data());
                    return temp_out.back();
                }
                else
                {
                    // Forward accumulation: accumulate from start to current_pos
                    std::vector<value_type> values(current_pos + 1);
                    size_type idx = line_start;
                    for (size_type i = 0; i <= current_pos; ++i)
                    {
                        values[i] = m_expression.flat(idx);
                        idx += axis_stride;
                    }
                    std::vector<value_type> temp_out(values.size());
                    m_functor_forward(values.data(), values.data() + values.size(), temp_out.data());
                    return temp_out.back();
                }
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
        // Concrete accumulator class with specific functor
        // --------------------------------------------------------------------
        template <class E, class F>
        class xaccumulator_impl : public xexpression<xaccumulator_impl<E, F>>
        {
        public:
            using self_type = xaccumulator_impl<E, F>;
            using base_type = xexpression<self_type>;
            using xexpression_type = std::decay_t<E>;
            using functor_type = F;
            
            using value_type = typename functor_type::result_type;
            using reference = value_type;
            using const_reference = value_type;
            using pointer = value_type*;
            using const_pointer = const value_type*;
            using size_type = typename xexpression_type::size_type;
            using difference_type = typename xexpression_type::difference_type;
            
            using shape_type = typename xexpression_type::shape_type;
            using strides_type = typename xexpression_type::strides_type;
            
            using expression_tag = xaccumulator_tag;
            static constexpr bool is_const = true;
            
            template <class EX>
            xaccumulator_impl(EX&& e, std::size_t axis, bool reverse = false)
                : m_expression(std::forward<EX>(e))
                , m_axis(normalize_axis(static_cast<std::ptrdiff_t>(axis), e.dimension()))
                , m_reverse(reverse)
                , m_shape(e.shape())
                , m_strides(e.strides())
                , m_backstrides(e.backstrides())
                , m_size(e.size())
                , m_functor()
            {
            }
            
            size_type size() const noexcept { return m_size; }
            size_type dimension() const noexcept { return m_shape.size(); }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return m_expression.layout(); }
            
            const xexpression_type& expression() const noexcept { return m_expression; }
            std::size_t axis() const noexcept { return m_axis; }
            bool reverse() const noexcept { return m_reverse; }
            
            template <class... Args>
            value_type operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            value_type flat(size_type i) const
            {
                return compute_accumulated(i);
            }
            
            // Iterator support (similar to above)
            class const_iterator
            {
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = typename self_type::value_type;
                using difference_type = std::ptrdiff_t;
                using pointer = const value_type*;
                using reference = value_type;
                
                const_iterator() = default;
                const_iterator(const self_type* accum, size_type index)
                    : m_accum(accum), m_index(index) {}
                
                reference operator*() const { return m_accum->flat(m_index); }
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                const_iterator& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                const_iterator operator+(difference_type n) const { return const_iterator(m_accum, m_index + static_cast<size_type>(n)); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_accum, m_index - static_cast<size_type>(n)); }
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                bool operator==(const const_iterator& rhs) const { return m_accum == rhs.m_accum && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const const_iterator& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const const_iterator& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const const_iterator& rhs) const { return m_index >= rhs.m_index; }
                reference operator[](difference_type n) const { return m_accum->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
            private:
                const self_type* m_accum = nullptr;
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
            
            template <class EX>
            disable_xexpression<EX, self_type&> operator=(const EX&) = delete;
            
        private:
            E m_expression;
            std::size_t m_axis;
            bool m_reverse;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            functor_type m_functor;
            
            value_type compute_accumulated(size_type flat_index) const
            {
                std::vector<size_type> coords(dimension());
                size_type temp = flat_index;
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    coords[d] = temp / m_strides[d];
                    temp %= m_strides[d];
                }
                
                size_type axis_len = m_shape[m_axis];
                size_type current_pos = coords[m_axis];
                
                std::vector<size_type> base_coords = coords;
                base_coords[m_axis] = 0;
                
                size_type line_start = 0;
                for (std::size_t d = 0; d < dimension(); ++d)
                {
                    line_start += base_coords[d] * m_strides[d];
                }
                
                size_type axis_stride = m_strides[m_axis];
                
                if (m_reverse)
                {
                    std::vector<typename xexpression_type::value_type> values(axis_len - current_pos);
                    size_type idx = line_start + (axis_len - 1) * axis_stride;
                    for (size_type i = 0; i < axis_len - current_pos; ++i)
                    {
                        values[i] = m_expression.flat(idx);
                        idx -= axis_stride;
                    }
                    std::vector<value_type> temp_out(values.size());
                    m_functor.reverse(values.data(), values.data() + values.size(), temp_out.data());
                    return temp_out.back();
                }
                else
                {
                    std::vector<typename xexpression_type::value_type> values(current_pos + 1);
                    size_type idx = line_start;
                    for (size_type i = 0; i <= current_pos; ++i)
                    {
                        values[i] = m_expression.flat(idx);
                        idx += axis_stride;
                    }
                    std::vector<value_type> temp_out(values.size());
                    m_functor(values.data(), values.data() + values.size(), temp_out.data());
                    return temp_out.back();
                }
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
        // Helper functions to create accumulators
        // --------------------------------------------------------------------
        template <class E>
        inline auto cumsum(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::sum_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto cumprod(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::prod_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto cummin(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::min_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto cummax(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::max_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto cummean(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::mean_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto cumvar(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::variance_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto nancumsum(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::nansum_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto nancumprod(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::nanprod_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto nancummin(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::nanmin_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto nancummax(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::nanmax_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        template <class E>
        inline auto nancummean(E&& e, std::size_t axis = 0, bool reverse = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator_impl<std::decay_t<E>, detail::nanmean_accumulator<value_type>>(
                std::forward<E>(e), axis, reverse);
        }
        
        // --------------------------------------------------------------------
        // diff - difference between consecutive elements along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline auto diff(E&& e, std::size_t n = 1, std::size_t axis = 0)
        {
            auto& expr = e.derived_cast();
            if (n == 0)
                return eval(expr);
            
            std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), expr.dimension());
            auto shape = expr.shape();
            if (shape[ax] <= n)
            {
                XTENSOR_THROW(std::runtime_error, "diff: axis size must be greater than n");
            }
            shape[ax] -= n;
            
            using value_type = typename std::decay_t<E>::value_type;
            xarray_container<value_type> result(shape);
            
            // Compute differences iteratively
            auto temp = eval(expr);
            for (std::size_t k = 0; k < n; ++k)
            {
                auto new_shape = temp.shape();
                new_shape[ax] -= 1;
                xarray_container<value_type> diff_result(new_shape);
                
                // For each slice along axis, compute diff
                std::size_t axis_len = temp.shape()[ax];
                std::size_t slice_size = temp.size() / axis_len;
                std::size_t stride = temp.strides()[ax];
                
                for (std::size_t i = 0; i < slice_size; ++i)
                {
                    std::size_t base = (i / stride) * (stride * axis_len) + (i % stride);
                    for (std::size_t j = 0; j < axis_len - 1; ++j)
                    {
                        std::size_t idx1 = base + j * stride;
                        std::size_t idx2 = base + (j + 1) * stride;
                        diff_result.flat(i * (axis_len - 1) + j) = temp.flat(idx2) - temp.flat(idx1);
                    }
                }
                temp = std::move(diff_result);
                if (k == n - 1)
                    result = temp;
            }
            return result;
        }
        
        // --------------------------------------------------------------------
        // trapz - trapezoidal integration along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline auto trapz(E&& e, double dx = 1.0, std::size_t axis = 0)
        {
            auto& expr = e.derived_cast();
            std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), expr.dimension());
            auto shape = expr.shape();
            if (shape[ax] < 2)
            {
                XTENSOR_THROW(std::runtime_error, "trapz: axis size must be at least 2");
            }
            shape[ax] = 1; // reduce to single element along axis
            using value_type = typename std::decay_t<E>::value_type;
            xarray_container<value_type> result(shape);
            
            std::size_t axis_len = expr.shape()[ax];
            std::size_t slice_size = expr.size() / axis_len;
            std::size_t stride = expr.strides()[ax];
            
            for (std::size_t i = 0; i < slice_size; ++i)
            {
                std::size_t base = (i / stride) * (stride * axis_len) + (i % stride);
                value_type sum = 0;
                for (std::size_t j = 0; j < axis_len - 1; ++j)
                {
                    std::size_t idx1 = base + j * stride;
                    std::size_t idx2 = base + (j + 1) * stride;
                    sum += (expr.flat(idx1) + expr.flat(idx2));
                }
                result.flat(i) = sum * dx * 0.5;
            }
            return result;
        }
        
        template <class E, class X>
        inline auto trapz(E&& e, const X& x, std::size_t axis = 0)
        {
            auto& expr = e.derived_cast();
            std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), expr.dimension());
            if (x.dimension() != 1)
            {
                XTENSOR_THROW(std::runtime_error, "trapz: x must be 1-dimensional");
            }
            if (x.size() != expr.shape()[ax])
            {
                XTENSOR_THROW(std::runtime_error, "trapz: length of x must match axis size");
            }
            
            auto shape = expr.shape();
            shape[ax] = 1;
            using value_type = typename std::decay_t<E>::value_type;
            xarray_container<value_type> result(shape);
            
            std::size_t axis_len = expr.shape()[ax];
            std::size_t slice_size = expr.size() / axis_len;
            std::size_t stride = expr.strides()[ax];
            
            for (std::size_t i = 0; i < slice_size; ++i)
            {
                std::size_t base = (i / stride) * (stride * axis_len) + (i % stride);
                value_type sum = 0;
                for (std::size_t j = 0; j < axis_len - 1; ++j)
                {
                    std::size_t idx1 = base + j * stride;
                    std::size_t idx2 = base + (j + 1) * stride;
                    double dx = static_cast<double>(x(j + 1) - x(j));
                    sum += (expr.flat(idx1) + expr.flat(idx2)) * dx;
                }
                result.flat(i) = sum * 0.5;
            }
            return result;
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XACCUMULATOR_HPP

// core/xaccumulator.hpp