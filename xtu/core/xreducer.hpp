// core/xreducer.hpp

#ifndef XTENSOR_XREDUCER_HPP
#define XTENSOR_XREDUCER_HPP

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
#include <limits>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // reducer function objects
        // --------------------------------------------------------------------
        namespace detail
        {
            template <class T, class E = void>
            struct sum_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    return std::accumulate(first, last, result_type(0));
                }
            };
            
            template <class T, class E = void>
            struct prod_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    return std::accumulate(first, last, result_type(1), std::multiplies<result_type>());
                }
            };
            
            template <class T, class E = void>
            struct mean_fun
            {
                using result_type = double; // default to double for mean
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    auto n = std::distance(first, last);
                    if (n == 0) return result_type(0);
                    return static_cast<result_type>(std::accumulate(first, last, result_type(0))) / static_cast<result_type>(n);
                }
            };
            
            template <class T, class E = void>
            struct variance_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    auto n = std::distance(first, last);
                    if (n <= 1) return result_type(0);
                    result_type mean = mean_fun<T>()(first, last);
                    result_type accum = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        result_type diff = static_cast<result_type>(*it) - mean;
                        accum += diff * diff;
                    }
                    return accum / static_cast<result_type>(n - 1);
                }
            };
            
            template <class T, class E = void>
            struct stddev_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    return std::sqrt(variance_fun<T>()(first, last));
                }
            };
            
            template <class T, class E = void>
            struct amin_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    if (first == last) return result_type(0);
                    return *std::min_element(first, last);
                }
            };
            
            template <class T, class E = void>
            struct amax_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    if (first == last) return result_type(0);
                    return *std::max_element(first, last);
                }
            };
            
            template <class T, class E = void>
            struct all_fun
            {
                using result_type = bool;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    return std::all_of(first, last, [](const T& v) { return static_cast<bool>(v); });
                }
            };
            
            template <class T, class E = void>
            struct any_fun
            {
                using result_type = bool;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    return std::any_of(first, last, [](const T& v) { return static_cast<bool>(v); });
                }
            };
            
            template <class T, class E = void>
            struct norm_l0_fun
            {
                using result_type = std::size_t;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    return std::count_if(first, last, [](const T& v) { return v != T(0); });
                }
            };
            
            template <class T, class E = void>
            struct norm_l1_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type sum = 0;
                    for (auto it = first; it != last; ++it)
                        sum += std::abs(static_cast<result_type>(*it));
                    return sum;
                }
            };
            
            template <class T, class E = void>
            struct norm_l2_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type sum_sq = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        result_type val = static_cast<result_type>(*it);
                        sum_sq += val * val;
                    }
                    return std::sqrt(sum_sq);
                }
            };
            
            template <class T, class E = void>
            struct norm_linf_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    if (first == last) return T(0);
                    return *std::max_element(first, last, [](const T& a, const T& b) {
                        return std::abs(a) < std::abs(b);
                    });
                }
            };
            
            template <class T, class E = void>
            struct norm_sq_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type sum_sq = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        result_type val = static_cast<result_type>(*it);
                        sum_sq += val * val;
                    }
                    return sum_sq;
                }
            };
            
            template <class T, class E = void>
            struct median_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    std::vector<T> temp(first, last);
                    if (temp.empty()) return result_type(0);
                    std::sort(temp.begin(), temp.end());
                    auto n = temp.size();
                    if (n % 2 == 1)
                        return static_cast<result_type>(temp[n / 2]);
                    else
                        return static_cast<result_type>(temp[n / 2 - 1] + temp[n / 2]) / 2.0;
                }
            };
            
            template <class T, class E = void>
            struct ptp_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    if (first == last) return result_type(0);
                    auto minmax = std::minmax_element(first, last);
                    return *minmax.second - *minmax.first;
                }
            };
            
            // Argmin/argmax: return index instead of value
            template <class T, class E = void>
            struct argmin_fun
            {
                using result_type = std::size_t;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    if (first == last) return 0;
                    return static_cast<result_type>(std::distance(first, std::min_element(first, last)));
                }
            };
            
            template <class T, class E = void>
            struct argmax_fun
            {
                using result_type = std::size_t;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    if (first == last) return 0;
                    return static_cast<result_type>(std::distance(first, std::max_element(first, last)));
                }
            };
            
            // NaN-handling versions
            template <class T, class E = void>
            struct nansum_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type sum = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                            sum += *it;
                    }
                    return sum;
                }
            };
            
            template <class T, class E = void>
            struct nanprod_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type prod = 1;
                    for (auto it = first; it != last; ++it)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                            prod *= *it;
                    }
                    return prod;
                }
            };
            
            template <class T, class E = void>
            struct nanmean_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type sum = 0;
                    std::size_t count = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            sum += *it;
                            ++count;
                        }
                    }
                    return count > 0 ? sum / count : result_type(0);
                }
            };
            
            template <class T, class E = void>
            struct nanvar_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type mean = nanmean_fun<T>()(first, last);
                    result_type accum = 0;
                    std::size_t count = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            result_type diff = static_cast<result_type>(*it) - mean;
                            accum += diff * diff;
                            ++count;
                        }
                    }
                    return count > 1 ? accum / (count - 1) : result_type(0);
                }
            };
            
            template <class T, class E = void>
            struct nanstd_fun
            {
                using result_type = double;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    return std::sqrt(nanvar_fun<T>()(first, last));
                }
            };
            
            template <class T, class E = void>
            struct nanmin_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type min_val = std::numeric_limits<result_type>::max();
                    bool found = false;
                    for (auto it = first; it != last; ++it)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            if (!found || *it < min_val)
                            {
                                min_val = *it;
                                found = true;
                            }
                        }
                    }
                    return found ? min_val : result_type(0);
                }
            };
            
            template <class T, class E = void>
            struct nanmax_fun
            {
                using result_type = T;
                
                template <class InputIt>
                result_type operator()(InputIt first, InputIt last) const
                {
                    result_type max_val = std::numeric_limits<result_type>::lowest();
                    bool found = false;
                    for (auto it = first; it != last; ++it)
                    {
                        if (!std::isnan(static_cast<double>(*it)))
                        {
                            if (!found || *it > max_val)
                            {
                                max_val = *it;
                                found = true;
                            }
                        }
                    }
                    return found ? max_val : result_type(0);
                }
            };
            
        } // namespace detail
        
        // --------------------------------------------------------------------
        // xreducer - Expression that reduces along specified axes
        // --------------------------------------------------------------------
        template <class CT, class X, class O>
        class xreducer : public xexpression<xreducer<CT, X, O>>
        {
        public:
            using self_type = xreducer<CT, X, O>;
            using base_type = xexpression<self_type>;
            using xexpression_type = std::decay_t<CT>;
            using axes_type = X;
            using functor_type = O;
            
            using value_type = typename functor_type::result_type;
            using reference = value_type;
            using const_reference = value_type;
            using pointer = value_type*;
            using const_pointer = const value_type*;
            using size_type = typename xexpression_type::size_type;
            using difference_type = typename xexpression_type::difference_type;
            
            using shape_type = std::vector<size_type>;
            using strides_type = std::vector<size_type>;
            
            using expression_tag = xreducer_tag;
            
            static constexpr bool is_const = true;
            
            // Construction
            template <class CTA, class AX, class F>
            xreducer(CTA&& e, AX&& axes, F&& func)
                : m_expression(std::forward<CTA>(e))
                , m_axes(std::forward<AX>(axes))
                , m_functor(std::forward<F>(func))
            {
                normalize_axes();
                compute_shape_and_strides();
                m_size = compute_size(m_shape);
            }
            
            // Size and shape
            size_type size() const noexcept { return m_size; }
            size_type dimension() const noexcept { return m_shape.size(); }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return layout_type::dynamic; }
            
            // Access to underlying expression and axes
            const xexpression_type& expression() const noexcept { return m_expression; }
            const axes_type& axes() const noexcept { return m_axes; }
            const functor_type& functor() const noexcept { return m_functor; }
            
            // Element access - evaluate reduction on the fly
            template <class... Args>
            value_type operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            value_type flat(size_type i) const
            {
                // We need to reduce over the specified axes at the position given by flat index i.
                // The flat index i corresponds to a multi-index in the reduced space.
                // We need to iterate over all elements in the original expression that map to this reduced position.
                return compute_reduction(i);
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
                const_iterator(const self_type* reducer, size_type index)
                    : m_reducer(reducer), m_index(index) {}
                
                reference operator*() const { return m_reducer->flat(m_index); }
                
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                const_iterator& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                
                const_iterator operator+(difference_type n) const { return const_iterator(m_reducer, m_index + static_cast<size_type>(n)); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_reducer, m_index - static_cast<size_type>(n)); }
                
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                
                bool operator==(const const_iterator& rhs) const { return m_reducer == rhs.m_reducer && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const const_iterator& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const const_iterator& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const const_iterator& rhs) const { return m_index >= rhs.m_index; }
                
                reference operator[](difference_type n) const { return m_reducer->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
                
            private:
                const self_type* m_reducer = nullptr;
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
            template <class E>
            disable_xexpression<E, self_type&> operator=(const E&) = delete;
            
        private:
            CT m_expression;
            axes_type m_axes;
            functor_type m_functor;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            std::vector<bool> m_is_reduced_axis; // marks which axes of original expression are reduced
            
            void normalize_axes()
            {
                std::size_t ndim = m_expression.dimension();
                m_is_reduced_axis.resize(ndim, false);
                for (auto axis : m_axes)
                {
                    std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), ndim);
                    m_is_reduced_axis[ax] = true;
                }
            }
            
            void compute_shape_and_strides()
            {
                const auto& expr_shape = m_expression.shape();
                const auto& expr_strides = m_expression.strides();
                std::size_t ndim = expr_shape.size();
                
                // Build reduced shape: skip reduced axes
                for (std::size_t i = 0; i < ndim; ++i)
                {
                    if (!m_is_reduced_axis[i])
                        m_shape.push_back(expr_shape[i]);
                }
                
                // Strides for reduced tensor (row-major)
                if (!m_shape.empty())
                {
                    m_strides.resize(m_shape.size());
                    m_backstrides.resize(m_shape.size());
                    m_strides.back() = 1;
                    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(m_shape.size()) - 2; i >= 0; --i)
                    {
                        m_strides[static_cast<std::size_t>(i)] = m_strides[static_cast<std::size_t>(i) + 1] * m_shape[static_cast<std::size_t>(i) + 1];
                    }
                    for (std::size_t i = 0; i < m_shape.size(); ++i)
                    {
                        m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
                    }
                }
                
                // For reduction computation, we need mapping from reduced index to original indices.
                // We'll store the mapping of reduced dimension index to original dimension index.
                m_reduced_to_original.clear();
                for (std::size_t i = 0; i < ndim; ++i)
                {
                    if (!m_is_reduced_axis[i])
                        m_reduced_to_original.push_back(i);
                }
            }
            
            std::vector<std::size_t> m_reduced_to_original; // mapping from reduced dim index to original dim index
            
            value_type compute_reduction(size_type flat_index) const
            {
                // Unravel flat_index to reduced coordinates
                std::vector<size_type> reduced_coords(m_shape.size());
                size_type temp = flat_index;
                for (std::size_t d = 0; d < m_shape.size(); ++d)
                {
                    reduced_coords[d] = temp / m_strides[d];
                    temp %= m_strides[d];
                }
                
                // Build a base index in the original expression (with reduced axes set to 0)
                std::vector<size_type> base_coords(m_expression.dimension(), 0);
                for (std::size_t i = 0; i < m_reduced_to_original.size(); ++i)
                {
                    base_coords[m_reduced_to_original[i]] = reduced_coords[i];
                }
                
                // We need to iterate over all combinations of the reduced axes.
                // The number of elements to reduce over is product of sizes of reduced axes.
                std::vector<std::size_t> reduced_axes;
                std::vector<size_type> reduced_sizes;
                const auto& expr_shape = m_expression.shape();
                for (std::size_t i = 0; i < m_expression.dimension(); ++i)
                {
                    if (m_is_reduced_axis[i])
                    {
                        reduced_axes.push_back(i);
                        reduced_sizes.push_back(expr_shape[i]);
                    }
                }
                
                // If no reduced axes, return the single element.
                if (reduced_axes.empty())
                {
                    size_type src_index = compute_src_index(base_coords);
                    return static_cast<value_type>(m_expression.flat(src_index));
                }
                
                // We need to iterate over all combinations. Use a temporary vector for the varying coords.
                std::vector<size_type> varying_coords = base_coords;
                std::vector<value_type> values;
                size_type total_combinations = std::accumulate(reduced_sizes.begin(), reduced_sizes.end(),
                                                                size_type(1), std::multiplies<size_type>());
                values.reserve(total_combinations);
                
                // Recursive or iterative combination generation.
                // For simplicity, we'll use a nested loop over the number of reduced axes.
                std::vector<std::size_t> counters(reduced_axes.size(), 0);
                bool done = false;
                while (!done)
                {
                    // Update varying_coords with current counters
                    for (std::size_t i = 0; i < reduced_axes.size(); ++i)
                    {
                        varying_coords[reduced_axes[i]] = counters[i];
                    }
                    size_type src_index = compute_src_index(varying_coords);
                    values.push_back(static_cast<value_type>(m_expression.flat(src_index)));
                    
                    // Increment counters
                    std::size_t pos = reduced_axes.size();
                    while (pos > 0)
                    {
                        --pos;
                        ++counters[pos];
                        if (counters[pos] < reduced_sizes[pos])
                            break;
                        counters[pos] = 0;
                        if (pos == 0)
                            done = true;
                    }
                }
                
                return m_functor(values.begin(), values.end());
            }
            
            size_type compute_src_index(const std::vector<size_type>& coords) const
            {
                size_type idx = 0;
                const auto& strides = m_expression.strides();
                for (std::size_t i = 0; i < coords.size(); ++i)
                {
                    idx += coords[i] * strides[i];
                }
                return idx;
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
        // xreducer with keepdims option
        // --------------------------------------------------------------------
        template <class CT, class X, class O>
        class xreducer_keepdims : public xexpression<xreducer_keepdims<CT, X, O>>
        {
        public:
            using self_type = xreducer_keepdims<CT, X, O>;
            using base_type = xexpression<self_type>;
            using xexpression_type = std::decay_t<CT>;
            using axes_type = X;
            using functor_type = O;
            
            using value_type = typename functor_type::result_type;
            using reference = value_type;
            using const_reference = value_type;
            using pointer = value_type*;
            using const_pointer = const value_type*;
            using size_type = typename xexpression_type::size_type;
            using difference_type = typename xexpression_type::difference_type;
            
            using shape_type = std::vector<size_type>;
            using strides_type = std::vector<size_type>;
            
            using expression_tag = xreducer_tag;
            static constexpr bool is_const = true;
            
            template <class CTA, class AX, class F>
            xreducer_keepdims(CTA&& e, AX&& axes, F&& func, bool keepdims)
                : m_expression(std::forward<CTA>(e))
                , m_axes(std::forward<AX>(axes))
                , m_functor(std::forward<F>(func))
                , m_keepdims(keepdims)
            {
                normalize_axes();
                compute_shape_and_strides();
                m_size = compute_size(m_shape);
            }
            
            size_type size() const noexcept { return m_size; }
            size_type dimension() const noexcept { return m_shape.size(); }
            const shape_type& shape() const noexcept { return m_shape; }
            const strides_type& strides() const noexcept { return m_strides; }
            const strides_type& backstrides() const noexcept { return m_backstrides; }
            layout_type layout() const noexcept { return layout_type::dynamic; }
            
            const xexpression_type& expression() const noexcept { return m_expression; }
            const axes_type& axes() const noexcept { return m_axes; }
            const functor_type& functor() const noexcept { return m_functor; }
            bool keepdims() const noexcept { return m_keepdims; }
            
            template <class... Args>
            value_type operator()(Args... args) const
            {
                size_type index = compute_index_impl(std::index_sequence_for<Args...>(),
                                                     static_cast<size_type>(args)...);
                return flat(index);
            }
            
            value_type flat(size_type i) const
            {
                return compute_reduction(i);
            }
            
            // Iterator support similar to xreducer
            class const_iterator
            {
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = typename self_type::value_type;
                using difference_type = std::ptrdiff_t;
                using pointer = const value_type*;
                using reference = value_type;
                
                const_iterator() = default;
                const_iterator(const self_type* reducer, size_type index)
                    : m_reducer(reducer), m_index(index) {}
                
                reference operator*() const { return m_reducer->flat(m_index); }
                const_iterator& operator++() { ++m_index; return *this; }
                const_iterator operator++(int) { const_iterator tmp = *this; ++*this; return tmp; }
                const_iterator& operator--() { --m_index; return *this; }
                const_iterator operator--(int) { const_iterator tmp = *this; --*this; return tmp; }
                const_iterator& operator+=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) + n); return *this; }
                const_iterator& operator-=(difference_type n) { m_index = static_cast<size_type>(static_cast<difference_type>(m_index) - n); return *this; }
                const_iterator operator+(difference_type n) const { return const_iterator(m_reducer, m_index + static_cast<size_type>(n)); }
                const_iterator operator-(difference_type n) const { return const_iterator(m_reducer, m_index - static_cast<size_type>(n)); }
                difference_type operator-(const const_iterator& rhs) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(rhs.m_index); }
                bool operator==(const const_iterator& rhs) const { return m_reducer == rhs.m_reducer && m_index == rhs.m_index; }
                bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
                bool operator<(const const_iterator& rhs) const { return m_index < rhs.m_index; }
                bool operator<=(const const_iterator& rhs) const { return m_index <= rhs.m_index; }
                bool operator>(const const_iterator& rhs) const { return m_index > rhs.m_index; }
                bool operator>=(const const_iterator& rhs) const { return m_index >= rhs.m_index; }
                reference operator[](difference_type n) const { return m_reducer->flat(static_cast<size_type>(static_cast<difference_type>(m_index) + n)); }
            private:
                const self_type* m_reducer = nullptr;
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
            
            template <class E>
            disable_xexpression<E, self_type&> operator=(const E&) = delete;
            
        private:
            CT m_expression;
            axes_type m_axes;
            functor_type m_functor;
            bool m_keepdims;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            std::vector<bool> m_is_reduced_axis;
            std::vector<std::size_t> m_reduced_to_original;
            
            void normalize_axes()
            {
                std::size_t ndim = m_expression.dimension();
                m_is_reduced_axis.resize(ndim, false);
                for (auto axis : m_axes)
                {
                    std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), ndim);
                    m_is_reduced_axis[ax] = true;
                }
            }
            
            void compute_shape_and_strides()
            {
                const auto& expr_shape = m_expression.shape();
                std::size_t ndim = expr_shape.size();
                
                if (m_keepdims)
                {
                    m_shape = expr_shape;
                    for (std::size_t i = 0; i < ndim; ++i)
                    {
                        if (m_is_reduced_axis[i])
                            m_shape[i] = 1;
                    }
                    // Strides follow row-major
                    if (!m_shape.empty())
                    {
                        m_strides.resize(m_shape.size());
                        m_backstrides.resize(m_shape.size());
                        m_strides.back() = 1;
                        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(m_shape.size()) - 2; i >= 0; --i)
                        {
                            m_strides[static_cast<std::size_t>(i)] = m_strides[static_cast<std::size_t>(i) + 1] * m_shape[static_cast<std::size_t>(i) + 1];
                        }
                        for (std::size_t i = 0; i < m_shape.size(); ++i)
                        {
                            m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
                        }
                    }
                    // Mapping: reduced dimensions still present but size 1
                    m_reduced_to_original.clear();
                    for (std::size_t i = 0; i < ndim; ++i)
                        m_reduced_to_original.push_back(i);
                }
                else
                {
                    for (std::size_t i = 0; i < ndim; ++i)
                    {
                        if (!m_is_reduced_axis[i])
                            m_shape.push_back(expr_shape[i]);
                    }
                    if (!m_shape.empty())
                    {
                        m_strides.resize(m_shape.size());
                        m_backstrides.resize(m_shape.size());
                        m_strides.back() = 1;
                        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(m_shape.size()) - 2; i >= 0; --i)
                        {
                            m_strides[static_cast<std::size_t>(i)] = m_strides[static_cast<std::size_t>(i) + 1] * m_shape[static_cast<std::size_t>(i) + 1];
                        }
                        for (std::size_t i = 0; i < m_shape.size(); ++i)
                        {
                            m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
                        }
                    }
                    m_reduced_to_original.clear();
                    for (std::size_t i = 0; i < ndim; ++i)
                    {
                        if (!m_is_reduced_axis[i])
                            m_reduced_to_original.push_back(i);
                    }
                }
            }
            
            value_type compute_reduction(size_type flat_index) const
            {
                std::vector<size_type> reduced_coords(m_shape.size());
                size_type temp = flat_index;
                for (std::size_t d = 0; d < m_shape.size(); ++d)
                {
                    reduced_coords[d] = temp / m_strides[d];
                    temp %= m_strides[d];
                }
                
                std::vector<size_type> base_coords(m_expression.dimension(), 0);
                if (m_keepdims)
                {
                    for (std::size_t i = 0; i < m_shape.size(); ++i)
                    {
                        base_coords[i] = reduced_coords[i];
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m_reduced_to_original.size(); ++i)
                    {
                        base_coords[m_reduced_to_original[i]] = reduced_coords[i];
                    }
                }
                
                std::vector<std::size_t> reduced_axes;
                std::vector<size_type> reduced_sizes;
                const auto& expr_shape = m_expression.shape();
                for (std::size_t i = 0; i < m_expression.dimension(); ++i)
                {
                    if (m_is_reduced_axis[i])
                    {
                        reduced_axes.push_back(i);
                        reduced_sizes.push_back(expr_shape[i]);
                    }
                }
                
                if (reduced_axes.empty())
                {
                    size_type src_index = compute_src_index(base_coords);
                    return static_cast<value_type>(m_expression.flat(src_index));
                }
                
                std::vector<size_type> varying_coords = base_coords;
                std::vector<value_type> values;
                size_type total_combinations = std::accumulate(reduced_sizes.begin(), reduced_sizes.end(),
                                                                size_type(1), std::multiplies<size_type>());
                values.reserve(total_combinations);
                
                std::vector<std::size_t> counters(reduced_axes.size(), 0);
                bool done = false;
                while (!done)
                {
                    for (std::size_t i = 0; i < reduced_axes.size(); ++i)
                    {
                        varying_coords[reduced_axes[i]] = counters[i];
                    }
                    size_type src_index = compute_src_index(varying_coords);
                    values.push_back(static_cast<value_type>(m_expression.flat(src_index)));
                    
                    std::size_t pos = reduced_axes.size();
                    while (pos > 0)
                    {
                        --pos;
                        ++counters[pos];
                        if (counters[pos] < reduced_sizes[pos])
                            break;
                        counters[pos] = 0;
                        if (pos == 0)
                            done = true;
                    }
                }
                
                return m_functor(values.begin(), values.end());
            }
            
            size_type compute_src_index(const std::vector<size_type>& coords) const
            {
                size_type idx = 0;
                const auto& strides = m_expression.strides();
                for (std::size_t i = 0; i < coords.size(); ++i)
                {
                    idx += coords[i] * strides[i];
                }
                return idx;
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
        // Helper functions to create reducers
        // --------------------------------------------------------------------
        template <class E, class X, class F>
        inline auto reduce(E&& e, X&& axes, F&& func, bool keepdims = false)
        {
            using reducer_type = xreducer_keepdims<std::decay_t<E>, std::decay_t<X>, std::decay_t<F>>;
            return reducer_type(std::forward<E>(e), std::forward<X>(axes), std::forward<F>(func), keepdims);
        }
        
        template <class E, class F>
        inline auto reduce_all(E&& e, F&& func)
        {
            std::vector<std::size_t> axes(e.dimension());
            std::iota(axes.begin(), axes.end(), 0);
            return reduce(std::forward<E>(e), axes, std::forward<F>(func), false);
        }
        
        // Specific reduction functions
        template <class E>
        inline auto sum(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::sum_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto sum(E&& e, std::size_t axis, bool keepdims = false)
        {
            return sum(std::forward<E>(e), std::vector<std::size_t>{axis}, keepdims);
        }
        
        template <class E>
        inline auto sum(E&& e, bool keepdims = false)
        {
            std::vector<std::size_t> axes(e.dimension());
            std::iota(axes.begin(), axes.end(), 0);
            return sum(std::forward<E>(e), axes, keepdims);
        }
        
        template <class E>
        inline auto prod(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::prod_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto prod(E&& e, std::size_t axis, bool keepdims = false)
        {
            return prod(std::forward<E>(e), std::vector<std::size_t>{axis}, keepdims);
        }
        
        template <class E>
        inline auto prod(E&& e, bool keepdims = false)
        {
            std::vector<std::size_t> axes(e.dimension());
            std::iota(axes.begin(), axes.end(), 0);
            return prod(std::forward<E>(e), axes, keepdims);
        }
        
        template <class E>
        inline auto mean(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::mean_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto mean(E&& e, std::size_t axis, bool keepdims = false)
        {
            return mean(std::forward<E>(e), std::vector<std::size_t>{axis}, keepdims);
        }
        
        template <class E>
        inline auto mean(E&& e, bool keepdims = false)
        {
            std::vector<std::size_t> axes(e.dimension());
            std::iota(axes.begin(), axes.end(), 0);
            return mean(std::forward<E>(e), axes, keepdims);
        }
        
        template <class E>
        inline auto variance(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::variance_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto stddev(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::stddev_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto amin(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::amin_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto amin(E&& e, std::size_t axis, bool keepdims = false)
        {
            return amin(std::forward<E>(e), std::vector<std::size_t>{axis}, keepdims);
        }
        
        template <class E>
        inline auto amax(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::amax_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto amax(E&& e, std::size_t axis, bool keepdims = false)
        {
            return amax(std::forward<E>(e), std::vector<std::size_t>{axis}, keepdims);
        }
        
        template <class E>
        inline auto all(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::all_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto any(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::any_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto norm_l0(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::norm_l0_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto norm_l1(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::norm_l1_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto norm_l2(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::norm_l2_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto norm_linf(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::norm_linf_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto norm_sq(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::norm_sq_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto median(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::median_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto ptp(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::ptp_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto argmin(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::argmin_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto argmax(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::argmax_fun<value_type>{}, keepdims);
        }
        
        // NaN variants
        template <class E>
        inline auto nansum(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::nansum_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto nanprod(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::nanprod_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto nanmean(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::nanmean_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto nanvar(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::nanvar_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto nanstd(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::nanstd_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto nanmin(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::nanmin_fun<value_type>{}, keepdims);
        }
        
        template <class E>
        inline auto nanmax(E&& e, const std::vector<std::size_t>& axes, bool keepdims = false)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return reduce(std::forward<E>(e), axes, detail::nanmax_fun<value_type>{}, keepdims);
        }
        
        // --------------------------------------------------------------------
        // trace - sum of diagonal elements
        // --------------------------------------------------------------------
        template <class E>
        inline auto trace(E&& e, std::ptrdiff_t offset = 0, std::size_t axis1 = 0, std::size_t axis2 = 1)
        {
            auto diag = diagonal_view(std::forward<E>(e), offset);
            return sum(diag);
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XREDUCER_HPP

// core/xreducer.hpp