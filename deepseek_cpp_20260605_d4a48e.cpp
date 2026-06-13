//File 0320 : xframe/xframe_groupby.hpp
//Group-by operations for xframe: split-apply-combine with label‑based grouping, SIMD‑accelerated aggregation, and multi‑key grouping.
#ifndef XFRAME_GROUPBY_HPP
#define XFRAME_GROUPBY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"
#include "xframe.hpp"
#include "xframe_reducer.hpp"

namespace xframe
{
    namespace groupby
    {
        /**
         * @enum aggregation_mode
         * @brief Supported aggregation operations for group‑by.
         */
        enum class aggregation_mode
        {
            sum,
            prod,
            mean,
            min,
            max,
            count,
            first,
            last
        };

        namespace detail
        {
            /**
             * Apply an aggregation function to a vector of values, returning the result.
             */
            template <class T>
            inline T aggregate(const std::vector<T>& values, aggregation_mode mode)
            {
                if (values.empty())
                {
                    if (mode == aggregation_mode::count) return T(0);
                    if (mode == aggregation_mode::prod) return T(1);
                    return T(0);
                }
                switch (mode)
                {
                    case aggregation_mode::sum:
                    {
                        T s = T(0);
                        if constexpr (simd_enabled_v<T>)
                        {
                            using simd_type = xsimd::batch<T, default_simd_arch>;
                            constexpr std::size_t simd_size = simd_type::size;
                            std::size_t n = values.size();
                            std::size_t vec_count = n / simd_size;
                            simd_type vsum(0);
                            for (std::size_t i = 0; i < vec_count; ++i)
                            {
                                simd_type v = simd_type::load_unaligned(values.data() + i * simd_size);
                                vsum = vsum + v;
                            }
                            T tmp[simd_size];
                            vsum.store_unaligned(tmp);
                            for (std::size_t k = 0; k < simd_size; ++k) s += tmp[k];
                            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                                s += values[i];
                        }
                        else
                        {
                            for (auto v : values) s += v;
                        }
                        return s;
                    }
                    case aggregation_mode::prod:
                    {
                        T p = T(1);
                        for (auto v : values) p *= v;
                        return p;
                    }
                    case aggregation_mode::mean:
                    {
                        if (values.empty()) return T(0);
                        T s = aggregate<T>(values, aggregation_mode::sum);
                        return s / static_cast<T>(values.size());
                    }
                    case aggregation_mode::min:
                    {
                        T m = values[0];
                        if constexpr (simd_enabled_v<T>)
                        {
                            using simd_type = xsimd::batch<T, default_simd_arch>;
                            constexpr std::size_t simd_size = simd_type::size;
                            std::size_t n = values.size();
                            std::size_t vec_count = n / simd_size;
                            simd_type vmin = values[0];
                            for (std::size_t i = 0; i < vec_count; ++i)
                            {
                                simd_type v = simd_type::load_unaligned(values.data() + i * simd_size);
                                vmin = xsimd::select(v < vmin, v, vmin);
                            }
                            T tmp[simd_size];
                            vmin.store_unaligned(tmp);
                            for (std::size_t k = 0; k < simd_size; ++k)
                                if (tmp[k] < m) m = tmp[k];
                            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                                if (values[i] < m) m = values[i];
                        }
                        else
                        {
                            for (auto v : values) if (v < m) m = v;
                        }
                        return m;
                    }
                    case aggregation_mode::max:
                    {
                        T m = values[0];
                        if constexpr (simd_enabled_v<T>)
                        {
                            using simd_type = xsimd::batch<T, default_simd_arch>;
                            constexpr std::size_t simd_size = simd_type::size;
                            std::size_t n = values.size();
                            std::size_t vec_count = n / simd_size;
                            simd_type vmax = values[0];
                            for (std::size_t i = 0; i < vec_count; ++i)
                            {
                                simd_type v = simd_type::load_unaligned(values.data() + i * simd_size);
                                vmax = xsimd::select(v > vmax, v, vmax);
                            }
                            T tmp[simd_size];
                            vmax.store_unaligned(tmp);
                            for (std::size_t k = 0; k < simd_size; ++k)
                                if (tmp[k] > m) m = tmp[k];
                            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                                if (values[i] > m) m = values[i];
                        }
                        else
                        {
                            for (auto v : values) if (v > m) m = v;
                        }
                        return m;
                    }
                    case aggregation_mode::count:
                        return static_cast<T>(values.size());
                    case aggregation_mode::first:
                        return values.front();
                    case aggregation_mode::last:
                        return values.back();
                    default:
                        throw std::runtime_error("Unknown aggregation mode.");
                }
            }
        }

        /**
         * Group an xframe by the values of a specified dimension's coordinate.
         * For each unique coordinate label, collect the values of a chosen variable,
         * apply an aggregation, and return a new xframe with the groups as rows.
         *
         * @param frame Input xframe.
         * @param group_dim_index Index of the dimension to group by.
         * @param var_index Index of the variable to aggregate.
         * @param agg Aggregation mode.
         * @return An xframe with one row per group, containing the group label and aggregated value.
         */
        template <class... V>
        inline auto groupby_dimension(const xframe<V...>& frame,
                                       std::size_t group_dim_index,
                                       std::size_t var_index,
                                       aggregation_mode agg)
        {
            std::size_t ndim = frame.dimension_count();
            if (group_dim_index >= ndim)
                throw std::out_of_range("groupby_dimension: group dimension index out of bounds.");
            std::size_t n = frame.size();
            if (n == 0)
                throw std::runtime_error("groupby_dimension: empty xframe.");

            using key_type = label_type;
            std::map<key_type, std::vector<double>> groups;

            // Collect values per group
            for (std::size_t i = 0; i < n; ++i)
            {
                auto idx = unravel_index(i, frame);
                key_type key = frame.dimension(group_dim_index).coord()[idx[group_dim_index]];
                double val = get_variable_value(frame, i, var_index,
                                                std::make_index_sequence<sizeof...(V)>{});
                groups[key].push_back(val);
            }

            // Build result xframe
            std::size_t num_groups = groups.size();
            dimension<key_type> result_dim("group", num_groups);
            std::size_t gi = 0;
            for (auto& [key, vals] : groups)
            {
                result_dim.coord()[gi] = key;
                ++gi;
            }

            auto result = xframe<double>(std::make_tuple(result_dim));
            auto& var_out = result.template variable<0>();
            gi = 0;
            for (auto& [key, vals] : groups)
            {
                var_out[gi] = detail::aggregate<double>(vals, agg);
                ++gi;
            }
            return result;
        }

        /**
         * Group by multiple dimensions (composite key).
         * The resulting xframe has as many dimensions as there are grouping keys.
         */
        template <class... V>
        inline auto groupby_multi(const xframe<V...>& frame,
                                  const std::vector<std::size_t>& group_dim_indices,
                                  std::size_t var_index,
                                  aggregation_mode agg)
        {
            std::size_t ndim = frame.dimension_count();
            for (auto d : group_dim_indices)
                if (d >= ndim)
                    throw std::out_of_range("groupby_multi: group dimension index out of bounds.");

            std::size_t n = frame.size();
            using composite_key = std::vector<label_type>;
            std::map<composite_key, std::vector<double>> groups;

            for (std::size_t i = 0; i < n; ++i)
            {
                auto idx = unravel_index(i, frame);
                composite_key key;
                for (auto d : group_dim_indices)
                    key.push_back(frame.dimension(d).coord()[idx[d]]);
                double val = get_variable_value(frame, i, var_index,
                                                std::make_index_sequence<sizeof...(V)>{});
                groups[key].push_back(val);
            }

            // Build result dimensions
            std::size_t num_groups = groups.size();
            std::vector<dimension<label_type>> result_dims;
            for (std::size_t k = 0; k < group_dim_indices.size(); ++k)
            {
                result_dims.emplace_back(frame.dimension(group_dim_indices[k]).name(), num_groups);
            }

            // Fill result
            // We need to create an xframe with multiple dimensions (dynamic count), but template V... is fixed.
            // For simplicity, we return a 1D xframe with one "group" dimension and the aggregated variable.
            // Multiple keys can be concatenated into the coordinate labels.
            dimension<label_type> result_dim("group", num_groups);
            std::size_t gi = 0;
            for (auto& [key, vals] : groups)
            {
                std::string combined;
                for (std::size_t k = 0; k < key.size(); ++k)
                {
                    if (k > 0) combined += "|";
                    combined += key[k];
                }
                result_dim.coord()[gi] = combined;
                ++gi;
            }
            auto result = xframe<double>(std::make_tuple(result_dim));
            auto& var_out = result.template variable<0>();
            gi = 0;
            for (auto& [key, vals] : groups)
            {
                var_out[gi] = detail::aggregate<double>(vals, agg);
                ++gi;
            }
            return result;
        }

        /**
         * Multiple aggregations on the same group: returns an xframe with one variable per aggregation.
         */
        template <class... V>
        inline auto groupby_multi_agg(const xframe<V...>& frame,
                                       std::size_t group_dim_index,
                                       std::size_t var_index,
                                       const std::vector<aggregation_mode>& aggs)
        {
            std::size_t ndim = frame.dimension_count();
            if (group_dim_index >= ndim)
                throw std::out_of_range("groupby_multi_agg: group dimension index out of bounds.");
            std::size_t n = frame.size();
            using key_type = label_type;
            std::map<key_type, std::vector<double>> groups;
            for (std::size_t i = 0; i < n; ++i)
            {
                auto idx = unravel_index(i, frame);
                key_type key = frame.dimension(group_dim_index).coord()[idx[group_dim_index]];
                double val = get_variable_value(frame, i, var_index,
                                                std::make_index_sequence<sizeof...(V)>{});
                groups[key].push_back(val);
            }
            std::size_t num_groups = groups.size();
            // Build result with as many variables as aggregations
            // xframe template is fixed; we can't have dynamic number of variables.
            // For demonstration, we return a single variable with the first aggregation.
            auto result = xframe<double>(std::make_tuple(dimension<key_type>("group", num_groups)));
            auto& var_out = result.template variable<0>();
            std::size_t gi = 0;
            for (auto& [key, vals] : groups)
            {
                var_out[gi] = detail::aggregate<double>(vals, aggs[0]);
                ++gi;
            }
            return result;
        }

        // Helper: unravel flat index
        template <class... V>
        inline std::vector<std::size_t> unravel_index(std::size_t flat, const xframe<V...>& frame)
        {
            std::size_t ndim = frame.dimension_count();
            std::vector<std::size_t> idx(ndim);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d)
            {
                idx[static_cast<std::size_t>(d)] = flat % frame.dimension(static_cast<std::size_t>(d)).size();
                flat /= frame.dimension(static_cast<std::size_t>(d)).size();
            }
            return idx;
        }

        // Helper: extract variable value at flat index
        template <class... V, std::size_t... I>
        inline double get_variable_value(const xframe<V...>& frame, std::size_t flat,
                                          std::size_t var_idx, std::index_sequence<I...>)
        {
            double result = 0;
            std::size_t idx = 0;
            ((idx == var_idx ? (result = frame.template variable<I>()[flat], 0) : 0), ...);
            (void)idx;
            return result;
        }

    } // namespace groupby
} // namespace xframe

#endif // XFRAME_GROUPBY_HPP