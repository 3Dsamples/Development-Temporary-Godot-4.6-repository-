//File 0062 : views/xview_utils.hpp
//View utilities: helper functions for creating slices, range views, diagonal views, and common view transformations.
#ifndef XTENSOR_XVIEW_UTILS_HPP
#define XTENSOR_XVIEW_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xslice.hpp"
#include "../core/xarray.hpp"
#include "../core/xmanipulation.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xdynamic_view.hpp"

namespace xt
{
    namespace view_utils
    {
        /**
         * Create a 1D slice (range) along a given axis, keeping other dimensions.
         */
        template <class E>
        inline auto slice_along_axis(E&& e, std::size_t axis, std::ptrdiff_t start,
                                     std::ptrdiff_t stop, std::ptrdiff_t step = 1)
        {
            auto ndim = e.dimension();
            if (axis >= ndim) throw std::out_of_range("slice_along_axis: axis out of bounds.");
            // Build slice tuple: all axes are xall except the target axis.
            std::vector<std::size_t> slices;
            for (std::size_t i = 0; i < ndim; ++i)
            {
                if (i == axis)
                    slices.push_back(static_cast<std::size_t>(start)); // simplified
            }
            // Better: use dynamic_view with xrange
            auto sl = xrange(start, stop, step);
            // We'll use a lambda to build the dynamic_view
            return dynamic_view(std::forward<E>(e), [&]() {
                // For each dimension: if i == axis, return sl, else return xall
                // But this requires variadic; we can't do it at runtime easily.
            }());
            // For simplicity, return a strided_view with explicit shape/strides
            auto shape = e.shape();
            shape[axis] = static_cast<std::size_t>(std::ceil(static_cast<double>(stop - start) / step));
            auto strides = compute_strides(shape);
            strides[axis] = e.strides()[axis] * static_cast<std::size_t>(step);
            return strided_view(std::forward<E>(e), shape, strides, start * e.strides()[axis]);
        }

        /**
         * Extract a diagonal view of a 2D matrix.
         */
        template <class E>
        inline auto diagonal(E&& e, std::ptrdiff_t offset = 0)
        {
            auto sh = e.shape();
            if (sh.size() != 2) throw std::runtime_error("diagonal requires 2D array.");
            std::size_t nrows = sh[0], ncols = sh[1];
            std::size_t diag_len = 0;
            std::size_t row_start = 0, col_start = 0;
            if (offset >= 0)
            {
                diag_len = std::min(nrows, ncols - static_cast<std::size_t>(offset));
                col_start = static_cast<std::size_t>(offset);
            }
            else
            {
                diag_len = std::min(nrows - static_cast<std::size_t>(-offset), ncols);
                row_start = static_cast<std::size_t>(-offset);
            }
            std::vector<std::size_t> new_shape = {diag_len};
            std::vector<std::size_t> new_strides = {e.strides()[0] + e.strides()[1]};
            std::size_t offset_base = row_start * e.strides()[0] + col_start * e.strides()[1];
            return strided_view(std::forward<E>(e), new_shape, new_strides, offset_base);
        }

        /**
         * Flip along an axis (returns a view, no copy).
         */
        template <class E>
        inline auto flip_view(E&& e, std::size_t axis)
        {
            auto shape = e.shape();
            std::size_t ndim = shape.size();
            if (axis >= ndim) throw std::out_of_range("flip_view: axis out of bounds.");
            std::vector<std::size_t> new_strides = e.strides();
            std::ptrdiff_t axis_len = static_cast<std::ptrdiff_t>(shape[axis]);
            new_strides[axis] = -e.strides()[axis];
            std::size_t offset = (axis_len - 1) * e.strides()[axis];
            return strided_view(std::forward<E>(e), shape, new_strides, offset);
        }

        /**
         * Roll along an axis (view, not copy).
         */
        template <class E>
        inline auto roll_view(E&& e, std::ptrdiff_t shift, std::size_t axis)
        {
            auto shape = e.shape();
            std::size_t ndim = shape.size();
            if (axis >= ndim) throw std::out_of_range("roll_view: axis out of bounds.");
            std::ptrdiff_t len = static_cast<std::ptrdiff_t>(shape[axis]);
            shift = ((shift % len) + len) % len;
            if (shift == 0) return dynamic_view(std::forward<E>(e)); // identity
            // We'll need to build a new strided view with offset
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < ndim; ++i) block_size *= shape[i];
            std::size_t offset = shift * e.strides()[axis];
            return strided_view(std::forward<E>(e), shape, e.strides(), offset);
        }

        /**
         * Create a lower triangular view (sets upper triangle to zero via masking).
         */
        template <class E>
        inline auto tril(E&& e, std::ptrdiff_t k = 0)
        {
            auto sh = e.shape();
            if (sh.size() != 2) throw std::runtime_error("tril requires 2D array.");
            using value_type = typename std::decay_t<E>::value_type;
            auto result = xt::eval(std::forward<E>(e));
            for (std::size_t i = 0; i < sh[0]; ++i)
                for (std::size_t j = 0; j < sh[1]; ++j)
                    if (static_cast<std::ptrdiff_t>(j) > static_cast<std::ptrdiff_t>(i) + k)
                        result(i, j) = value_type(0);
            return result;
        }

        /**
         * Create an upper triangular view.
         */
        template <class E>
        inline auto triu(E&& e, std::ptrdiff_t k = 0)
        {
            auto sh = e.shape();
            if (sh.size() != 2) throw std::runtime_error("triu requires 2D array.");
            using value_type = typename std::decay_t<E>::value_type;
            auto result = xt::eval(std::forward<E>(e));
            for (std::size_t i = 0; i < sh[0]; ++i)
                for (std::size_t j = 0; j < sh[1]; ++j)
                    if (static_cast<std::ptrdiff_t>(j) < static_cast<std::ptrdiff_t>(i) + k)
                        result(i, j) = value_type(0);
            return result;
        }

        /**
         * Create a view that selects indices from a list along a given axis.
         */
        template <class E, class Indices>
        inline auto take_along_axis(E&& e, const Indices& indices, std::size_t axis)
        {
            auto base_shape = e.shape();
            auto idx_shape = indices.shape();
            std::size_t ndim = base_shape.size();
            if (axis >= ndim) throw std::out_of_range("take_along_axis: axis out of bounds.");

            // Result shape = indices shape, but replace the axis dimension with indices' axis dim?
            // Standard take_along_axis: result shape = indices shape.
            // We'll implement as a lazy view that maps output indices to input via indices array.
            struct take_functor
            {
                const E* base;
                const Indices* idx;
                std::size_t axis;

                auto operator()(std::vector<std::size_t> out_idx) const
                {
                    // Replace the coordinate along axis with the index from indices
                    std::size_t replacement = (*idx).element(out_idx.begin(), out_idx.end());
                    out_idx[axis] = replacement;
                    return (*base).element(out_idx.begin(), out_idx.end());
                }
            };

            // For simplicity, return an evaluated copy
            auto result_shape = idx_shape;
            using value_type = typename std::decay_t<E>::value_type;
            xarray_container<uvector<value_type>> result(result_shape);
            for (std::size_t i = 0; i < result.size(); ++i)
            {
                auto out_idx = unravel_index(i, result_shape);
                result[i] = take_functor{&e, &indices, axis}(out_idx);
            }
            return result;
        }

        /**
         * Create a sliding window view (as strided view) for 1D/2D convolution.
         */
        template <class E>
        inline auto sliding_window(E&& e, const std::vector<std::size_t>& window_shape,
                                    const std::vector<std::size_t>& strides = {})
        {
            auto base_shape = e.shape();
            std::size_t ndim = base_shape.size();
            if (window_shape.size() != ndim) throw std::runtime_error("sliding_window: window_shape must match base rank.");
            std::vector<std::size_t> win_strides = strides.empty() ? std::vector<std::size_t>(ndim, 1) : strides;
            if (win_strides.size() != ndim) throw std::runtime_error("sliding_window: strides size mismatch.");

            // Output shape: (base - window) / stride + 1
            std::vector<std::size_t> out_shape(ndim);
            for (std::size_t d = 0; d < ndim; ++d)
                out_shape[d] = (base_shape[d] - window_shape[d]) / win_strides[d] + 1;

            // Append window dimensions to output shape
            out_shape.insert(out_shape.end(), window_shape.begin(), window_shape.end());

            // Strides for the sliding window: base strides * win_strides, then original strides for inner window
            auto base_strides = e.strides();
            std::vector<std::size_t> out_strides(2 * ndim);
            for (std::size_t d = 0; d < ndim; ++d)
                out_strides[d] = base_strides[d] * win_strides[d];
            for (std::size_t d = 0; d < ndim; ++d)
                out_strides[ndim + d] = base_strides[d];

            return strided_view(std::forward<E>(e), out_shape, out_strides, 0);
        }

        /**
         * Compute the cumulative product of dimensions of a shape.
         */
        inline std::vector<std::size_t> cumprod_shape(const std::vector<std::size_t>& shape)
        {
            std::vector<std::size_t> result(shape.size());
            std::partial_sum(shape.begin(), shape.end(), result.begin(), std::multiplies<std::size_t>());
            return result;
        }
    }
} // namespace xt

#endif // XTENSOR_XVIEW_UTILS_HPP