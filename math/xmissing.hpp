// core/xmissing.hpp
#ifndef XTENSOR_XMISSING_HPP
#define XTENSOR_XMISSING_HPP

// ----------------------------------------------------------------------------
// xmissing.hpp – Missing data support for xtensor (xoptional)
// ----------------------------------------------------------------------------
// This header defines xoptional<T>, a type that represents either a value
// of type T or a missing value (NA). It also provides expression support,
// reducers, and utility functions to work with arrays containing optional
// values. Fully integrated with bignumber::BigNumber and FFT acceleration.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <optional>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xreducer.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ========================================================================
    // xoptional<T> – value that may be missing
    // ========================================================================
    template <class T>
    class xoptional
    {
    public:
        using value_type = T;

        // --------------------------------------------------------------------
        // Constructors
        // --------------------------------------------------------------------
        constexpr xoptional() noexcept
            : m_value()
            , m_has_value(false)
        {
        }

        constexpr xoptional(const T& value) noexcept(std::is_nothrow_copy_constructible_v<T>)
            : m_value(value)
            , m_has_value(true)
        {
        }

        constexpr xoptional(T&& value) noexcept(std::is_nothrow_move_constructible_v<T>)
            : m_value(std::move(value))
            , m_has_value(true)
        {
        }

        constexpr xoptional(const xoptional&) = default;
        constexpr xoptional(xoptional&&) = default;

        // Construct from value and boolean flag
        constexpr xoptional(const T& value, bool has_val)
            : m_value(value)
            , m_has_value(has_val)
        {
        }

        constexpr xoptional(T&& value, bool has_val)
            : m_value(std::move(value))
            , m_has_value(has_val)
        {
        }

        // --------------------------------------------------------------------
        // Assignment
        // --------------------------------------------------------------------
        xoptional& operator=(const xoptional&) = default;
        xoptional& operator=(xoptional&&) = default;

        xoptional& operator=(const T& value)
        {
            m_value = value;
            m_has_value = true;
            return *this;
        }

        xoptional& operator=(T&& value)
        {
            m_value = std::move(value);
            m_has_value = true;
            return *this;
        }

        // --------------------------------------------------------------------
        // Observers
        // --------------------------------------------------------------------
        constexpr bool has_value() const noexcept
        {
            return m_has_value;
        }

        constexpr explicit operator bool() const noexcept
        {
            return m_has_value;
        }

        constexpr const T& value() const&
        {
            if (!m_has_value)
                XTENSOR_THROW(std::runtime_error, "xoptional::value: no value");
            return m_value;
        }

        constexpr T& value() &
        {
            if (!m_has_value)
                XTENSOR_THROW(std::runtime_error, "xoptional::value: no value");
            return m_value;
        }

        constexpr T&& value() &&
        {
            if (!m_has_value)
                XTENSOR_THROW(std::runtime_error, "xoptional::value: no value");
            return std::move(m_value);
        }

        constexpr const T& value_or(const T& default_value) const noexcept
        {
            return m_has_value ? m_value : default_value;
        }

        constexpr T value_or(T&& default_value) const
        {
            return m_has_value ? m_value : std::forward<T>(default_value);
        }

        // --------------------------------------------------------------------
        // Modifiers
        // --------------------------------------------------------------------
        void reset() noexcept
        {
            m_has_value = false;
        }

        template <class... Args>
        T& emplace(Args&&... args)
        {
            m_value = T(std::forward<Args>(args)...);
            m_has_value = true;
            return m_value;
        }

        // --------------------------------------------------------------------
        // Comparison operators
        // --------------------------------------------------------------------
        friend constexpr bool operator==(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value && !rhs.m_has_value) return true;
            if (!lhs.m_has_value || !rhs.m_has_value) return false;
            return lhs.m_value == rhs.m_value;
        }

        friend constexpr bool operator!=(const xoptional& lhs, const xoptional& rhs)
        {
            return !(lhs == rhs);
        }

        friend constexpr bool operator<(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return false;
            return lhs.m_value < rhs.m_value;
        }

        friend constexpr bool operator>(const xoptional& lhs, const xoptional& rhs)
        {
            return rhs < lhs;
        }

        friend constexpr bool operator<=(const xoptional& lhs, const xoptional& rhs)
        {
            return !(rhs < lhs);
        }

        friend constexpr bool operator>=(const xoptional& lhs, const xoptional& rhs)
        {
            return !(lhs < rhs);
        }

        // --------------------------------------------------------------------
        // Arithmetic operators (propagate missing)
        // --------------------------------------------------------------------
        friend xoptional operator+(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return xoptional();
            return xoptional(lhs.m_value + rhs.m_value);
        }

        friend xoptional operator-(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return xoptional();
            return xoptional(lhs.m_value - rhs.m_value);
        }

        friend xoptional operator*(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return xoptional();
            // Use FFT multiplication for BigNumber
            if constexpr (std::is_same_v<T, bignumber::BigNumber>)
            {
                if (config::use_fft_multiply)
                    return xoptional(bignumber::fft_multiply(lhs.m_value, rhs.m_value));
            }
            return xoptional(lhs.m_value * rhs.m_value);
        }

        friend xoptional operator/(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return xoptional();
            return xoptional(lhs.m_value / rhs.m_value);
        }

        // Unary minus
        xoptional operator-() const
        {
            if (!m_has_value) return xoptional();
            return xoptional(-m_value);
        }

        // --------------------------------------------------------------------
        // Logical operators
        // --------------------------------------------------------------------
        friend xoptional<bool> operator&&(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return xoptional<bool>();
            return xoptional<bool>(lhs.m_value && rhs.m_value);
        }

        friend xoptional<bool> operator||(const xoptional& lhs, const xoptional& rhs)
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return xoptional<bool>();
            return xoptional<bool>(lhs.m_value || rhs.m_value);
        }

        xoptional<bool> operator!() const
        {
            if (!m_has_value) return xoptional<bool>();
            return xoptional<bool>(!m_value);
        }

    private:
        T m_value;
        bool m_has_value;
    };

    // ========================================================================
    // Type traits
    // ========================================================================
    template <class T>
    struct is_xoptional : std::false_type {};

    template <class T>
    struct is_xoptional<xoptional<T>> : std::true_type {};

    template <class T>
    inline constexpr bool is_xoptional_v = is_xoptional<T>::value;

    // ========================================================================
    // Utility functions for optional arrays
    // ========================================================================

    // ------------------------------------------------------------------------
    // has_value – returns boolean mask of valid elements
    // ------------------------------------------------------------------------
    template <class E>
    inline auto has_value(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        static_assert(is_xoptional_v<value_type>, "has_value requires xoptional elements");

        xarray_container<bool> result(expr.shape());
        for (size_type i = 0; i < expr.size(); ++i)
            result.flat(i) = expr.flat(i).has_value();
        return result;
    }

    // ------------------------------------------------------------------------
    // value – extract values, replacing missing with default
    // ------------------------------------------------------------------------
    template <class E>
    inline auto value(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        using opt_type = typename E::value_type;
        using value_type = typename opt_type::value_type;
        static_assert(is_xoptional_v<opt_type>, "value requires xoptional elements");

        xarray_container<value_type> result(expr.shape());
        for (size_type i = 0; i < expr.size(); ++i)
            result.flat(i) = expr.flat(i).value();
        return result;
    }

    template <class E, class T>
    inline auto value(const xexpression<E>& e, const T& default_val)
    {
        const auto& expr = e.derived_cast();
        using opt_type = typename E::value_type;
        using value_type = typename opt_type::value_type;
        static_assert(is_xoptional_v<opt_type>, "value requires xoptional elements");

        xarray_container<value_type> result(expr.shape());
        for (size_type i = 0; i < expr.size(); ++i)
            result.flat(i) = expr.flat(i).value_or(default_val);
        return result;
    }

    // ------------------------------------------------------------------------
    // missing – create an xoptional with missing value
    // ------------------------------------------------------------------------
    template <class T>
    inline xoptional<T> missing()
    {
        return xoptional<T>();
    }

    template <class T>
    inline xoptional<T> missing(const T&)
    {
        return xoptional<T>();
    }

    // ------------------------------------------------------------------------
    // optional – create an xoptional with a value
    // ------------------------------------------------------------------------
    template <class T>
    inline xoptional<std::decay_t<T>> optional(T&& value)
    {
        return xoptional<std::decay_t<T>>(std::forward<T>(value));
    }

    // ========================================================================
    // Reducers for optional values
    // ========================================================================

    namespace detail
    {
        template <class T>
        struct reducer_sum_optional
        {
            using value_type = xoptional<T>;

            value_type init() const noexcept
            {
                return xoptional<T>(T(0));
            }

            value_type operator()(const value_type& a, const value_type& b) const
            {
                if (!a.has_value()) return b;
                if (!b.has_value()) return a;
                return xoptional<T>(a.value() + b.value());
            }

            value_type finalize(const value_type& v, size_type) const
            {
                return v;
            }
        };

        template <class T>
        struct reducer_prod_optional
        {
            using value_type = xoptional<T>;

            value_type init() const noexcept
            {
                return xoptional<T>(T(1));
            }

            value_type operator()(const value_type& a, const value_type& b) const
            {
                if (!a.has_value()) return b;
                if (!b.has_value()) return a;
                if constexpr (std::is_same_v<T, bignumber::BigNumber>)
                {
                    if (config::use_fft_multiply)
                        return xoptional<T>(bignumber::fft_multiply(a.value(), b.value()));
                }
                return xoptional<T>(a.value() * b.value());
            }

            value_type finalize(const value_type& v, size_type) const
            {
                return v;
            }
        };

        template <class T>
        struct reducer_mean_optional
        {
            using value_type = xoptional<T>;

            struct state
            {
                T sum = T(0);
                size_t count = 0;
            };

            state init() const noexcept
            {
                return state{};
            }

            state operator()(const state& s, const value_type& v) const
            {
                if (!v.has_value()) return s;
                return state{s.sum + v.value(), s.count + 1};
            }

            state merge(const state& s1, const state& s2) const
            {
                return state{s1.sum + s2.sum, s1.count + s2.count};
            }

            value_type finalize(const state& s, size_type) const
            {
                if (s.count == 0) return value_type();
                return value_type(s.sum / T(s.count));
            }
        };
    } // namespace detail

    template <class E>
    inline auto sum(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        using inner_type = typename value_type::value_type;
        auto reducer = detail::reducer_sum_optional<inner_type>();
        return xreducer<const E&, std::vector<size_type>, decltype(reducer)>(
            e.derived_cast(), std::vector<size_type>{}, std::move(reducer)
        )();
    }

    template <class E>
    inline auto prod(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        using inner_type = typename value_type::value_type;
        auto reducer = detail::reducer_prod_optional<inner_type>();
        return xreducer<const E&, std::vector<size_type>, decltype(reducer)>(
            e.derived_cast(), std::vector<size_type>{}, std::move(reducer)
        )();
    }

    template <class E>
    inline auto mean(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        using inner_type = typename value_type::value_type;
        auto reducer = detail::reducer_mean_optional<inner_type>();
        return xreducer<const E&, std::vector<size_type>, decltype(reducer)>(
            e.derived_cast(), std::vector<size_type>{}, std::move(reducer)
        )();
    }

    // ========================================================================
    // Missing value propagation for expressions
    // ========================================================================
    // The xoptional_assembly class handles expression evaluation with missingness

    template <class E>
    class xoptional_assembly : public xexpression<xoptional_assembly<E>>
    {
    public:
        using value_type = typename E::value_type;
        using size_type = xt::size_type;
        using shape_type = xt::shape_type;

        explicit xoptional_assembly(const E& expr)
            : m_expression(expr)
        {
        }

        const shape_type& shape() const
        {
            return m_expression.shape();
        }

        size_type size() const
        {
            return m_expression.size();
        }

        size_type dimension() const
        {
            return m_expression.dimension();
        }

        value_type flat(size_type i) const
        {
            return m_expression.flat(i);
        }

        template <class... Args>
        value_type operator()(Args... args) const
        {
            return m_expression(args...);
        }

    private:
        E m_expression;
    };

    // ========================================================================
    // Convenience functions for creating optional arrays
    // ========================================================================
    template <class T>
    inline xarray_container<xoptional<T>> optional_array(const shape_type& shape)
    {
        return xarray_container<xoptional<T>>(shape);
    }

    template <class T>
    inline xarray_container<xoptional<T>> optional_array(const shape_type& shape, const T& fill_value)
    {
        return xarray_container<xoptional<T>>(shape, xoptional<T>(fill_value));
    }

} // namespace xt

#endif // XTENSOR_XMISSING_HPP                      auto& val = result.element(coords);
                            if (is_missing(val, sentinel))
                            {
                                if (!is_missing(last_valid, sentinel))
                                    val = last_valid;
                            }
                            else
                            {
                                last_valid = val;
                            }
                        }
                    }
                    else if (dir == fill_direction::backward)
                    {
                        typename E::value_type next_valid = sentinel;
                        for (std::size_t i = axis_len; i-- > 0; )
                        {
                            coords[ax] = i;
                            auto& val = result.element(coords);
                            if (is_missing(val, sentinel))
                            {
                                if (!is_missing(next_valid, sentinel))
                                    val = next_valid;
                            }
                            else
                            {
                                next_valid = val;
                            }
                        }
                    }
                    else // nearest
                    {
                        // Two passes: forward then backward, combine
                        auto forward_result = eval(result);
                        auto backward_result = eval(result);
                        
                        // Forward pass
                        {
                            typename E::value_type last_valid = sentinel;
                            for (std::size_t i = 0; i < axis_len; ++i)
                            {
                                coords[ax] = i;
                                auto& val = forward_result.element(coords);
                                if (is_missing(val, sentinel))
                                {
                                    if (!is_missing(last_valid, sentinel))
                                        val = last_valid;
                                }
                                else
                                {
                                    last_valid = val;
                                }
                            }
                        }
                        // Backward pass
                        {
                            typename E::value_type next_valid = sentinel;
                            for (std::size_t i = axis_len; i-- > 0; )
                            {
                                coords[ax] = i;
                                auto& val = backward_result.element(coords);
                                if (is_missing(val, sentinel))
                                {
                                    if (!is_missing(next_valid, sentinel))
                                        val = next_valid;
                                }
                                else
                                {
                                    next_valid = val;
                                }
                            }
                        }
                        // Combine: use forward if available, else backward
                        for (std::size_t i = 0; i < axis_len; ++i)
                        {
                            coords[ax] = i;
                            auto& orig = result.element(coords);
                            if (is_missing(orig, sentinel))
                            {
                                auto fwd = forward_result.element(coords);
                                auto bwd = backward_result.element(coords);
                                if (!is_missing(fwd, sentinel))
                                    orig = fwd;
                                else if (!is_missing(bwd, sentinel))
                                    orig = bwd;
                            }
                        }
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Interpolate missing values linearly along an axis
            // --------------------------------------------------------------------
            template <class E>
            inline auto interpolate_missing_linear(
                const xexpression<E>& e, std::size_t axis,
                const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                auto result = eval(e);
                std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), result.dimension());
                std::size_t axis_len = result.shape()[ax];
                std::size_t num_slices = result.size() / axis_len;
                
                using value_type = typename E::value_type;
                
                for (std::size_t slice = 0; slice < num_slices; ++slice)
                {
                    std::vector<std::size_t> coords(result.dimension(), 0);
                    std::size_t temp = slice;
                    for (std::size_t d = 0; d < result.dimension(); ++d)
                    {
                        if (d == ax) continue;
                        std::size_t stride_after = 1;
                        for (std::size_t k = d + 1; k < result.dimension(); ++k)
                            if (k != ax) stride_after *= result.shape()[k];
                        coords[d] = temp / stride_after;
                        temp %= stride_after;
                    }
                    
                    // Find valid points
                    std::vector<std::size_t> valid_indices;
                    std::vector<value_type> valid_values;
                    for (std::size_t i = 0; i < axis_len; ++i)
                    {
                        coords[ax] = i;
                        auto val = result.element(coords);
                        if (!is_missing(val, sentinel))
                        {
                            valid_indices.push_back(i);
                            valid_values.push_back(val);
                        }
                    }
                    
                    if (valid_indices.size() < 2)
                    {
                        // Not enough points for interpolation, fill with nearest
                        if (!valid_indices.empty())
                        {
                            value_type fill_val = valid_values[0];
                            for (std::size_t i = 0; i < axis_len; ++i)
                            {
                                coords[ax] = i;
                                if (is_missing(result.element(coords), sentinel))
                                    result.element(coords) = fill_val;
                            }
                        }
                        continue;
                    }
                    
                    // Interpolate between valid points
                    for (std::size_t k = 0; k < valid_indices.size() - 1; ++k)
                    {
                        std::size_t start_idx = valid_indices[k];
                        std::size_t end_idx = valid_indices[k + 1];
                        value_type start_val = valid_values[k];
                        value_type end_val = valid_values[k + 1];
                        
                        if (end_idx > start_idx + 1)
                        {
                            value_type step = (end_val - start_val) / static_cast<value_type>(end_idx - start_idx);
                            for (std::size_t i = start_idx + 1; i < end_idx; ++i)
                            {
                                coords[ax] = i;
                                result.element(coords) = start_val + step * static_cast<value_type>(i - start_idx);
                            }
                        }
                    }
                    
                    // Extrapolate beyond ends using constant fill
                    if (valid_indices.front() > 0)
                    {
                        value_type first_val = valid_values.front();
                        for (std::size_t i = 0; i < valid_indices.front(); ++i)
                        {
                            coords[ax] = i;
                            if (is_missing(result.element(coords), sentinel))
                                result.element(coords) = first_val;
                        }
                    }
                    if (valid_indices.back() < axis_len - 1)
                    {
                        value_type last_val = valid_values.back();
                        for (std::size_t i = valid_indices.back() + 1; i < axis_len; ++i)
                        {
                            coords[ax] = i;
                            if (is_missing(result.element(coords), sentinel))
                                result.element(coords) = last_val;
                        }
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Replace missing values with mean/median of non-missing along axis
            // --------------------------------------------------------------------
            template <class E>
            inline auto fill_missing_mean(const xexpression<E>& e, std::size_t axis,
                                          const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                auto result = eval(e);
                std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), result.dimension());
                std::size_t axis_len = result.shape()[ax];
                std::size_t num_slices = result.size() / axis_len;
                
                using value_type = typename E::value_type;
                
                for (std::size_t slice = 0; slice < num_slices; ++slice)
                {
                    std::vector<std::size_t> coords(result.dimension(), 0);
                    std::size_t temp = slice;
                    for (std::size_t d = 0; d < result.dimension(); ++d)
                    {
                        if (d == ax) continue;
                        std::size_t stride_after = 1;
                        for (std::size_t k = d + 1; k < result.dimension(); ++k)
                            if (k != ax) stride_after *= result.shape()[k];
                        coords[d] = temp / stride_after;
                        temp %= stride_after;
                    }
                    
                    value_type sum = 0;
                    std::size_t count = 0;
                    for (std::size_t i = 0; i < axis_len; ++i)
                    {
                        coords[ax] = i;
                        auto val = result.element(coords);
                        if (!is_missing(val, sentinel))
                        {
                            sum += val;
                            ++count;
                        }
                    }
                    
                    if (count > 0)
                    {
                        value_type mean_val = sum / static_cast<value_type>(count);
                        for (std::size_t i = 0; i < axis_len; ++i)
                        {
                            coords[ax] = i;
                            if (is_missing(result.element(coords), sentinel))
                                result.element(coords) = mean_val;
                        }
                    }
                }
                return result;
            }
            
            template <class E>
            inline auto fill_missing_median(const xexpression<E>& e, std::size_t axis,
                                            const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                auto result = eval(e);
                std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), result.dimension());
                std::size_t axis_len = result.shape()[ax];
                std::size_t num_slices = result.size() / axis_len;
                
                using value_type = typename E::value_type;
                
                for (std::size_t slice = 0; slice < num_slices; ++slice)
                {
                    std::vector<std::size_t> coords(result.dimension(), 0);
                    std::size_t temp = slice;
                    for (std::size_t d = 0; d < result.dimension(); ++d)
                    {
                        if (d == ax) continue;
                        std::size_t stride_after = 1;
                        for (std::size_t k = d + 1; k < result.dimension(); ++k)
                            if (k != ax) stride_after *= result.shape()[k];
                        coords[d] = temp / stride_after;
                        temp %= stride_after;
                    }
                    
                    std::vector<value_type> valid_vals;
                    for (std::size_t i = 0; i < axis_len; ++i)
                    {
                        coords[ax] = i;
                        auto val = result.element(coords);
                        if (!is_missing(val, sentinel))
                            valid_vals.push_back(val);
                    }
                    
                    if (!valid_vals.empty())
                    {
                        std::sort(valid_vals.begin(), valid_vals.end());
                        value_type median_val;
                        if (valid_vals.size() % 2 == 1)
                            median_val = valid_vals[valid_vals.size() / 2];
                        else
                            median_val = (valid_vals[valid_vals.size() / 2 - 1] + valid_vals[valid_vals.size() / 2]) / static_cast<value_type>(2);
                        
                        for (std::size_t i = 0; i < axis_len; ++i)
                        {
                            coords[ax] = i;
                            if (is_missing(result.element(coords), sentinel))
                                result.element(coords) = median_val;
                        }
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Return a mask of missing values
            // --------------------------------------------------------------------
            template <class E>
            inline auto missing_mask(const xexpression<E>& e,
                                     const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                xarray_container<bool> mask(expr.shape());
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    mask.flat(i) = is_missing(expr.flat(i), sentinel);
                }
                return mask;
            }
            
            template <class E>
            inline auto nan_mask(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                xarray_container<bool> mask(expr.shape());
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    for (std::size_t i = 0; i < expr.size(); ++i)
                        mask.flat(i) = std::isnan(expr.flat(i));
                }
                else
                {
                    std::fill(mask.begin(), mask.end(), false);
                }
                return mask;
            }
            
            // --------------------------------------------------------------------
            // Remove rows/columns with any/all missing values (for 2D)
            // --------------------------------------------------------------------
            template <class E>
            inline auto drop_rows_with_missing(const xexpression<E>& e, bool any = true,
                                               const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                if (expr.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "drop_rows_with_missing: requires 2-D expression");
                }
                
                std::size_t n_rows = expr.shape()[0];
                std::size_t n_cols = expr.shape()[1];
                
                std::vector<bool> keep_row(n_rows, true);
                for (std::size_t i = 0; i < n_rows; ++i)
                {
                    bool row_has_missing = false;
                    bool row_all_missing = true;
                    for (std::size_t j = 0; j < n_cols; ++j)
                    {
                        bool missing = is_missing(expr(i, j), sentinel);
                        if (missing) row_has_missing = true;
                        else row_all_missing = false;
                    }
                    if (any)
                        keep_row[i] = !row_has_missing;
                    else
                        keep_row[i] = !row_all_missing;
                }
                
                std::size_t kept_count = std::count(keep_row.begin(), keep_row.end(), true);
                using value_type = typename E::value_type;
                xarray_container<value_type> result(std::vector<std::size_t>{kept_count, n_cols});
                
                std::size_t out_row = 0;
                for (std::size_t i = 0; i < n_rows; ++i)
                {
                    if (keep_row[i])
                    {
                        for (std::size_t j = 0; j < n_cols; ++j)
                            result(out_row, j) = expr(i, j);
                        ++out_row;
                    }
                }
                return result;
            }
            
            template <class E>
            inline auto drop_columns_with_missing(const xexpression<E>& e, bool any = true,
                                                  const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                if (expr.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "drop_columns_with_missing: requires 2-D expression");
                }
                
                std::size_t n_rows = expr.shape()[0];
                std::size_t n_cols = expr.shape()[1];
                
                std::vector<bool> keep_col(n_cols, true);
                for (std::size_t j = 0; j < n_cols; ++j)
                {
                    bool col_has_missing = false;
                    bool col_all_missing = true;
                    for (std::size_t i = 0; i < n_rows; ++i)
                    {
                        bool missing = is_missing(expr(i, j), sentinel);
                        if (missing) col_has_missing = true;
                        else col_all_missing = false;
                    }
                    if (any)
                        keep_col[j] = !col_has_missing;
                    else
                        keep_col[j] = !col_all_missing;
                }
                
                std::size_t kept_count = std::count(keep_col.begin(), keep_col.end(), true);
                using value_type = typename E::value_type;
                xarray_container<value_type> result(std::vector<std::size_t>{n_rows, kept_count});
                
                std::size_t out_col = 0;
                for (std::size_t j = 0; j < n_cols; ++j)
                {
                    if (keep_col[j])
                    {
                        for (std::size_t i = 0; i < n_rows; ++i)
                            result(i, out_col) = expr(i, j);
                        ++out_col;
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Replace missing values with a specific value in-place
            // --------------------------------------------------------------------
            template <class E>
            inline void replace_missing_inplace(xexpression<E>& e, const typename E::value_type& new_value,
                                                const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                auto& expr = e.derived_cast();
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (is_missing(expr.flat(i), sentinel))
                        expr.flat(i) = new_value;
                }
            }
            
            template <class E>
            inline void replace_nan_inplace(xexpression<E>& e, const typename E::value_type& new_value)
            {
                auto& expr = e.derived_cast();
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    for (std::size_t i = 0; i < expr.size(); ++i)
                    {
                        if (std::isnan(expr.flat(i)))
                            expr.flat(i) = new_value;
                    }
                }
            }
            
            template <class E>
            inline void replace_inf_inplace(xexpression<E>& e, const typename E::value_type& new_value)
            {
                auto& expr = e.derived_cast();
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    for (std::size_t i = 0; i < expr.size(); ++i)
                    {
                        if (std::isinf(expr.flat(i)))
                            expr.flat(i) = new_value;
                    }
                }
            }
            
            // --------------------------------------------------------------------
            // Convenience functions that dispatch to appropriate methods
            // --------------------------------------------------------------------
            template <class E>
            inline auto dropna(const xexpression<E>& e, std::size_t axis = 0, const std::string& how = "any")
            {
                const auto& expr = e.derived_cast();
                bool any = (how == "any");
                
                if (expr.dimension() == 1)
                {
                    return drop_nan(e);
                }
                else if (expr.dimension() == 2)
                {
                    if (axis == 0)
                        return drop_rows_with_missing(e, any, missing_sentinel<typename E::value_type>::value());
                    else
                        return drop_columns_with_missing(e, any, missing_sentinel<typename E::value_type>::value());
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "dropna for dimension > 2 not fully supported");
                }
                return eval(e);
            }
            
            template <class E>
            inline auto fillna(const xexpression<E>& e, const typename E::value_type& value)
            {
                return fill_nan(e, value);
            }
            
            template <class E>
            inline auto fillna(const xexpression<E>& e, const std::string& method, std::size_t axis = 0)
            {
                if (method == "ffill" || method == "pad")
                    return fill_missing_directional(e, axis, fill_direction::forward);
                else if (method == "bfill" || method == "backfill")
                    return fill_missing_directional(e, axis, fill_direction::backward);
                else if (method == "nearest")
                    return fill_missing_directional(e, axis, fill_direction::nearest);
                else
                    XTENSOR_THROW(std::invalid_argument, "fillna: unknown method");
                return eval(e);
            }
            
            template <class E>
            inline auto interpolate_na(const xexpression<E>& e, std::size_t axis = 0, const std::string& method = "linear")
            {
                if (method == "linear")
                    return interpolate_missing_linear(e, axis);
                else if (method == "mean")
                    return fill_missing_mean(e, axis);
                else if (method == "median")
                    return fill_missing_median(e, axis);
                else
                    XTENSOR_THROW(std::invalid_argument, "interpolate_na: unsupported method");
                return eval(e);
            }
            
        } // namespace missing
        
        // Bring into xt namespace for convenience
        using missing::isnan;
        using missing::isinf;
        using missing::isfinite;
        using missing::ismissing;
        using missing::count_missing;
        using missing::count_nan;
        using missing::count_inf;
        using missing::count_finite;
        using missing::any_missing;
        using missing::all_missing;
        using missing::any_nan;
        using missing::all_nan;
        using missing::drop_missing;
        using missing::drop_nan;
        using missing::fill_missing;
        using missing::fill_nan;
        using missing::fill_inf;
        using missing::interpolate_missing_linear;
        using missing::missing_mask;
        using missing::nan_mask;
        using missing::dropna;
        using missing::fillna;
        using missing::interpolate_na;
        using missing::replace_missing_inplace;
        using missing::replace_nan_inplace;
        using missing::replace_inf_inplace;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XMISSING_HPP

// math/xmissing.hpp