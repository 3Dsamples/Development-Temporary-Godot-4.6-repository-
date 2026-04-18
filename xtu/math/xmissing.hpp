// math/xmissing.hpp

#ifndef XTENSOR_XMISSING_HPP
#define XTENSOR_XMISSING_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xfunction.hpp"
#include "../core/xview.hpp"
#include "xsorting.hpp"

#include <cmath>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace missing
        {
            // --------------------------------------------------------------------
            // Missing value detection utilities
            // --------------------------------------------------------------------
            
            // Default missing value sentinel for numeric types
            template <class T>
            struct missing_sentinel
            {
                static T value()
                {
                    if constexpr (std::is_floating_point_v<T>)
                        return std::numeric_limits<T>::quiet_NaN();
                    else
                        return std::numeric_limits<T>::max();
                }
            };
            
            // Check if a value is missing (NaN or sentinel)
            template <class T>
            inline bool is_missing(const T& val, const T& sentinel = missing_sentinel<T>::value())
            {
                if constexpr (std::is_floating_point_v<T>)
                {
                    return std::isnan(val);
                }
                else
                {
                    return val == sentinel;
                }
            }
            
            // Check if a value is NaN (for floating point)
            template <class T>
            inline bool is_nan(const T& val)
            {
                if constexpr (std::is_floating_point_v<T>)
                {
                    return std::isnan(val);
                }
                else
                {
                    return false;
                }
            }
            
            // Check if a value is infinite
            template <class T>
            inline bool is_inf(const T& val)
            {
                if constexpr (std::is_floating_point_v<T>)
                {
                    return std::isinf(val);
                }
                else
                {
                    return false;
                }
            }
            
            // Check if a value is finite
            template <class T>
            inline bool is_finite(const T& val)
            {
                if constexpr (std::is_floating_point_v<T>)
                {
                    return std::isfinite(val);
                }
                else
                {
                    return true;
                }
            }
            
            // --------------------------------------------------------------------
            // Element-wise missing value checks (return boolean expression)
            // --------------------------------------------------------------------
            template <class E>
            inline auto isnan(const xexpression<E>& e)
            {
                return make_xfunction(
                    [](const typename E::value_type& v) -> bool {
                        if constexpr (std::is_floating_point_v<typename E::value_type>)
                            return std::isnan(v);
                        else
                            return false;
                    },
                    e.derived_cast()
                );
            }
            
            template <class E>
            inline auto isinf(const xexpression<E>& e)
            {
                return make_xfunction(
                    [](const typename E::value_type& v) -> bool {
                        if constexpr (std::is_floating_point_v<typename E::value_type>)
                            return std::isinf(v);
                        else
                            return false;
                    },
                    e.derived_cast()
                );
            }
            
            template <class E>
            inline auto isfinite(const xexpression<E>& e)
            {
                return make_xfunction(
                    [](const typename E::value_type& v) -> bool {
                        if constexpr (std::is_floating_point_v<typename E::value_type>)
                            return std::isfinite(v);
                        else
                            return true;
                    },
                    e.derived_cast()
                );
            }
            
            template <class E>
            inline auto ismissing(const xexpression<E>& e, 
                                  const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                return make_xfunction(
                    [sentinel](const typename E::value_type& v) -> bool {
                        return is_missing(v, sentinel);
                    },
                    e.derived_cast()
                );
            }
            
            // --------------------------------------------------------------------
            // Count missing values
            // --------------------------------------------------------------------
            template <class E>
            inline std::size_t count_missing(const xexpression<E>& e,
                                              const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                std::size_t count = 0;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (is_missing(expr.flat(i), sentinel))
                        ++count;
                }
                return count;
            }
            
            template <class E>
            inline std::size_t count_nan(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    std::size_t count = 0;
                    for (std::size_t i = 0; i < expr.size(); ++i)
                    {
                        if (std::isnan(expr.flat(i)))
                            ++count;
                    }
                    return count;
                }
                return 0;
            }
            
            template <class E>
            inline std::size_t count_inf(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    std::size_t count = 0;
                    for (std::size_t i = 0; i < expr.size(); ++i)
                    {
                        if (std::isinf(expr.flat(i)))
                            ++count;
                    }
                    return count;
                }
                return 0;
            }
            
            template <class E>
            inline std::size_t count_finite(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                std::size_t count = 0;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (is_finite(expr.flat(i)))
                        ++count;
                }
                return count;
            }
            
            // --------------------------------------------------------------------
            // Check if any/all values are missing
            // --------------------------------------------------------------------
            template <class E>
            inline bool any_missing(const xexpression<E>& e,
                                    const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (is_missing(expr.flat(i), sentinel))
                        return true;
                }
                return false;
            }
            
            template <class E>
            inline bool all_missing(const xexpression<E>& e,
                                    const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                if (expr.size() == 0) return true;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (!is_missing(expr.flat(i), sentinel))
                        return false;
                }
                return true;
            }
            
            template <class E>
            inline bool any_nan(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    for (std::size_t i = 0; i < expr.size(); ++i)
                        if (std::isnan(expr.flat(i))) return true;
                }
                return false;
            }
            
            template <class E>
            inline bool all_nan(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                if (expr.size() == 0) return true;
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    for (std::size_t i = 0; i < expr.size(); ++i)
                        if (!std::isnan(expr.flat(i))) return false;
                }
                return true;
            }
            
            // --------------------------------------------------------------------
            // Drop missing values (flattened output)
            // --------------------------------------------------------------------
            template <class E>
            inline auto drop_missing(const xexpression<E>& e,
                                     const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                std::vector<value_type> result;
                result.reserve(expr.size() - count_missing(expr, sentinel));
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (!is_missing(expr.flat(i), sentinel))
                        result.push_back(expr.flat(i));
                }
                return result;
            }
            
            template <class E>
            inline auto drop_nan(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                if constexpr (std::is_floating_point_v<value_type>)
                {
                    std::vector<value_type> result;
                    result.reserve(expr.size());
                    for (std::size_t i = 0; i < expr.size(); ++i)
                    {
                        if (!std::isnan(expr.flat(i)))
                            result.push_back(expr.flat(i));
                    }
                    return result;
                }
                else
                {
                    std::vector<value_type> result(expr.size());
                    std::copy(expr.begin(), expr.end(), result.begin());
                    return result;
                }
            }
            
            // Drop missing values along an axis (returns a container with variable length slices)
            // This is complex; we'll implement a version that returns a list of vectors for each slice.
            template <class E>
            inline std::vector<std::vector<typename E::value_type>> drop_missing_axis(
                const xexpression<E>& e, std::size_t axis,
                const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                const auto& expr = e.derived_cast();
                std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), expr.dimension());
                std::size_t num_slices = expr.size() / expr.shape()[ax];
                std::size_t axis_len = expr.shape()[ax];
                
                std::vector<std::vector<typename E::value_type>> result(num_slices);
                
                for (std::size_t slice = 0; slice < num_slices; ++slice)
                {
                    std::vector<std::size_t> coords(expr.dimension(), 0);
                    std::size_t temp = slice;
                    for (std::size_t d = 0; d < expr.dimension(); ++d)
                    {
                        if (d == ax) continue;
                        std::size_t stride_after = 1;
                        for (std::size_t k = d + 1; k < expr.dimension(); ++k)
                            if (k != ax) stride_after *= expr.shape()[k];
                        coords[d] = temp / stride_after;
                        temp %= stride_after;
                    }
                    
                    for (std::size_t i = 0; i < axis_len; ++i)
                    {
                        coords[ax] = i;
                        auto val = expr.element(coords);
                        if (!is_missing(val, sentinel))
                            result[slice].push_back(val);
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Fill missing values with a constant
            // --------------------------------------------------------------------
            template <class E>
            inline auto fill_missing(const xexpression<E>& e, const typename E::value_type& fill_value,
                                     const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                auto result = eval(e);
                for (std::size_t i = 0; i < result.size(); ++i)
                {
                    if (is_missing(result.flat(i), sentinel))
                        result.flat(i) = fill_value;
                }
                return result;
            }
            
            template <class E>
            inline auto fill_nan(const xexpression<E>& e, const typename E::value_type& fill_value)
            {
                auto result = eval(e);
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    for (std::size_t i = 0; i < result.size(); ++i)
                    {
                        if (std::isnan(result.flat(i)))
                            result.flat(i) = fill_value;
                    }
                }
                return result;
            }
            
            template <class E>
            inline auto fill_inf(const xexpression<E>& e, const typename E::value_type& fill_value)
            {
                auto result = eval(e);
                if constexpr (std::is_floating_point_v<typename E::value_type>)
                {
                    for (std::size_t i = 0; i < result.size(); ++i)
                    {
                        if (std::isinf(result.flat(i)))
                            result.flat(i) = fill_value;
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Fill missing with forward/backward fill along an axis
            // --------------------------------------------------------------------
            enum class fill_direction
            {
                forward,
                backward,
                nearest
            };
            
            template <class E>
            inline auto fill_missing_directional(
                const xexpression<E>& e, std::size_t axis, fill_direction dir = fill_direction::forward,
                const typename E::value_type& sentinel = missing_sentinel<typename E::value_type>::value())
            {
                auto result = eval(e);
                std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), result.dimension());
                std::size_t axis_len = result.shape()[ax];
                std::size_t num_slices = result.size() / axis_len;
                
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
                    
                    if (dir == fill_direction::forward)
                    {
                        typename E::value_type last_valid = sentinel;
                        for (std::size_t i = 0; i < axis_len; ++i)
                        {
                            coords[ax] = i;
                            auto& val = result.element(coords);
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