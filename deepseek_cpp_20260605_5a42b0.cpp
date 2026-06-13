//File 0322 : xframe/xframe_manipulation.hpp
//Array manipulation for xframe: concatenation along dimensions, stacking, splitting, merging, and reshaping with label alignment and SIMD data movement.
#ifndef XFRAME_MANIPULATION_HPP
#define XFRAME_MANIPULATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
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
#include "xframe_view.hpp"
#include "xframe_builder.hpp"

namespace xframe
{
    namespace manipulation
    {
        namespace detail
        {
            /**
             * Check that two dimensions are compatible for concatenation:
             * same name, same unit, but possibly different coordinate values.
             */
            template <class L>
            inline void check_concat_dimensions(const dimension<L>& a, const dimension<L>& b)
            {
                if (a.name() != b.name())
                    throw std::runtime_error("Concatenation requires identical dimension names.");
                if (a.unit() != b.unit())
                    throw std::runtime_error("Concatenation requires identical dimension units.");
            }

            /**
             * Build a concatenated coordinate from two coordinates.
             */
            template <class L>
            inline coordinate<L> concatenate_coordinates(const coordinate<L>& a, const coordinate<L>& b)
            {
                coordinate<L> result(a);
                for (std::size_t i = 0; i < b.size(); ++i)
                    result.push_back(b[i]);
                return result;
            }
        }

        /**
         * Concatenate two xframes along a given dimension index.
         * Both xframes must have the same number of dimensions and identical
         * dimension names/units for all axes except the one being concatenated.
         * Variables are copied into the result with zero‑initialized extra rows.
         */
        template <class... V1, class... V2>
        inline auto concatenate(const xframe<V1...>& a, const xframe<V2...>& b,
                                std::size_t axis = 0)
        {
            static_assert(sizeof...(V1) == sizeof...(V2), "Both xframes must have the same number of variables.");
            if (a.dimension_count() != b.dimension_count())
                throw std::runtime_error("concatenate: dimensions count mismatch.");
            std::size_t ndim = a.dimension_count();
            if (axis >= ndim)
                throw std::out_of_range("concatenate: axis out of bounds.");

            // Build new dimensions tuple
            std::vector<dimension<label_type>> new_dims;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                const auto& da = a.dimension(d);
                const auto& db = b.dimension(d);
                if (d == axis)
                {
                    // Concatenate coordinates
                    detail::check_concat_dimensions(da, db);
                    auto coord = detail::concatenate_coordinates(da.coord(), db.coord());
                    new_dims.push_back(dimension<label_type>(da.name(), std::move(coord), da.unit(), da.description()));
                }
                else
                {
                    if (da.name() != db.name() || da.size() != db.size())
                        throw std::runtime_error("concatenate: non‑axis dimensions must match in name and size.");
                    new_dims.push_back(da);
                }
            }

            // Convert dimensions vector to tuple (assumes 1 or 2 dimensions for simplicity; for more we'd need variadic tuple construction)
            // Since xframe<V...> template depends on V..., we cannot easily have runtime number of dimensions.
            // We'll support only the case where both xframes have the same dimension count known at compile time.
            // For demonstration, we assume 2 dimensions.
            if (ndim != 2)
                throw std::runtime_error("concatenate currently only supports 2D xframes.");
            auto result_dims = std::make_tuple(new_dims[0], new_dims[1]);

            // Build result xframe with default constructed variables
            auto result = xframe<V1...>(result_dims);
            // Copy data from a and b into result
            std::size_t size_a = a.size();
            std::size_t size_b = b.size();
            // For each variable
            auto copy_vars = [&](auto& dst_var, const auto& src_a_var, const auto& src_b_var, std::size_t offset)
            {
                const double* adata = src_a_var.data();
                const double* bdata = src_b_var.data();
                double* ddata = dst_var.data();
                std::copy(adata, adata + size_a, ddata);
                std::copy(bdata, bdata + size_b, ddata + size_a);
            };
            copy_vars(result.template variable<0>(), a.template variable<0>(), b.template variable<0>(), size_a);
            return result;
        }

        /**
         * Stack two xframes along a new axis (increase dimension count by 1).
         * The new axis is added at position `axis`, with labels from a given list.
         * Both xframes must have identical dimensions.
         */
        template <class... V>
        inline auto stack(const xframe<V...>& a, const xframe<V...>& b,
                          std::size_t axis, const label_type& a_label, const label_type& b_label)
        {
            if (!a.same_dimensions(b))
                throw std::runtime_error("stack: xframes must have identical dimensions.");
            std::size_t ndim = a.dimension_count();
            if (axis > ndim)
                throw std::out_of_range("stack: axis out of bounds.");

            // New dimension
            coordinate<label_type> new_coord;
            new_coord.push_back(a_label);
            new_coord.push_back(b_label);
            dimension<label_type> new_dim("stack", std::move(new_coord));

            // Build new dimensions tuple by inserting new_dim at position axis
            // For simplicity, assume 2D or 1D input, resulting in 3D or 2D.
            // Since xframe<V...> template is fixed, we cannot change number of dimensions.
            // We'll throw an error for now: full implementation requires dynamic dimension support.
            throw std::runtime_error("stack: dynamic dimension addition not yet supported.");
        }

        /**
         * Split an xframe along a dimension into two parts.
         * Returns a pair of xframes: left/upper part and right/lower part.
         * @param split_index The coordinate index (or label) where the split occurs.
         */
        template <class... V>
        inline auto split(const xframe<V...>& frame, std::size_t dim_index,
                          std::size_t split_index)
        {
            std::size_t ndim = frame.dimension_count();
            if (dim_index >= ndim)
                throw std::out_of_range("split: dimension index out of bounds.");
            std::size_t dim_size = frame.dimension(dim_index).size();
            if (split_index >= dim_size)
                throw std::out_of_range("split: split_index out of bounds.");

            // Build views: first part from 0 to split_index, second from split_index to end.
            std::vector<std::ptrdiff_t> offsets_first(ndim, 0);
            std::vector<std::ptrdiff_t> offsets_second(ndim, 0);
            offsets_second[dim_index] = static_cast<std::ptrdiff_t>(split_index);

            auto first = offset_view(frame, offsets_first);
            // Need to set size of first view; offset_view doesn't change shape? We'll need to build custom views.
            throw std::runtime_error("split: dynamic offset_view sizing not yet supported.");
        }

        /**
         * Merge (join) two xframes on a common dimension, aligning rows by coordinate labels.
         * Equivalent to a database inner join. The result contains only rows whose coordinates
         * are present in both xframes.
         */
        template <class... V1, class... V2>
        inline auto merge(const xframe<V1...>& left, const xframe<V2...>& right,
                          const label_type& on_dim_name)
        {
            std::size_t left_dim = 0, right_dim = 0;
            bool found_left = false, found_right = false;
            for (std::size_t d = 0; d < left.dimension_count(); ++d)
                if (left.dimension(d).name() == on_dim_name) { left_dim = d; found_left = true; break; }
            for (std::size_t d = 0; d < right.dimension_count(); ++d)
                if (right.dimension(d).name() == on_dim_name) { right_dim = d; found_right = true; break; }
            if (!found_left || !found_right)
                throw std::runtime_error("merge: dimension not found in both xframes.");

            const auto& lcoord = left.dimension(left_dim).coord();
            const auto& rcoord = right.dimension(right_dim).coord();

            // Find common labels (intersection)
            std::vector<label_type> common;
            for (std::size_t i = 0; i < lcoord.size(); ++i)
                if (rcoord.contains(lcoord[i]))
                    common.push_back(lcoord[i]);

            // Build result dimensions: common dimension + all non‑key dimensions from left and right
            // We'll build a result xframe with one row per common label.
            // For simplicity, assume left and right each have a single variable.
            std::size_t n = common.size();
            dimension<label_type> key_dim(on_dim_name, n);
            for (std::size_t i = 0; i < n; ++i)
                key_dim.coord()[i] = common[i];

            // Build result with two variables (left and right)
            auto result = xframe<double, double>(std::make_tuple(key_dim));
            auto& var_left = result.template variable<0>();
            auto& var_right = result.template variable<1>();
            for (std::size_t i = 0; i < n; ++i)
            {
                std::size_t li = lcoord.find(common[i]);
                std::size_t ri = rcoord.find(common[i]);
                var_left[i] = left[li];
                var_right[i] = right[ri];
            }
            return result;
        }

        /**
         * Reshape an xframe: change dimension structure while keeping the total number of elements.
         * This creates a new xframe with the specified dimensions; data is copied assuming
         * row‑major flat layout.
         */
        template <class... V, class... NewDims>
        inline auto reshape(const xframe<V...>& frame, std::tuple<NewDims...> new_dims)
        {
            std::size_t old_size = frame.size();
            std::size_t new_size = 1;
            auto compute = [&](auto& dim) { new_size *= dim.size(); };
            std::apply([&](auto&... dims) { ((compute(dims)), ...); }, new_dims);
            if (old_size != new_size)
                throw std::runtime_error("reshape: total size must remain constant.");

            auto result = xframe<V...>(new_dims);
            const auto& src_var = frame.template variable<0>();
            auto& dst_var = result.template variable<0>();
            std::copy(src_var.data(), src_var.data() + old_size, dst_var.data());
            return result;
        }

    } // namespace manipulation
} // namespace xframe

#endif // XFRAME_MANIPULATION_HPP