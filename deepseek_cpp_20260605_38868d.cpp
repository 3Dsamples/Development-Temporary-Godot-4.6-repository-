//File 0317 : xframe/xframe_xtensor.hpp
//Bridge between xframe and xtensor: convert variables and dimensions to xtensor arrays for high-performance 2D/3D simulation and advanced math.
#ifndef XFRAME_XTENSOR_HPP
#define XFRAME_XTENSOR_HPP

#include <cstddef>
#include <cstdint>
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

// Include xtensor (assuming the previously rewritten xtensor is available)
#include "../core/xtensor.hpp"
#include "../core/xarray.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"

namespace xframe
{
    namespace bridge
    {
        /**
         * Extract a variable from an xframe as a 1D xtensor array.
         * The xtensor array shares the dimension labels as a separate coordinate?
         * Only the values are copied.
         */
        template <class... V, std::size_t I = 0>
        inline auto variable_to_xtensor(const xframe<V...>& frame)
        {
            const auto& var = frame.template variable<I>();
            xt::xarray<double> result({var.size()});
            std::copy(var.data(), var.data() + var.size(), result.data());
            return result;
        }

        /**
         * Convert a 2D xframe (with two dimensions and one variable) into a 2D xtensor array.
         * The shape is (dim0.size(), dim1.size()). Values are copied row‑wise.
         */
        template <class... V>
        inline auto to_xtensor2d(const xframe<V...>& frame)
        {
            if (frame.dimension_count() != 2)
                throw std::runtime_error("to_xtensor2d: xframe must have exactly 2 dimensions.");
            std::size_t rows = frame.dimension(0).size();
            std::size_t cols = frame.dimension(1).size();
            xt::xarray<double> result({rows, cols});
            const auto& var = frame.template variable<0>();
            const double* src = var.data();
            double* dst = result.data();
            std::copy(src, src + rows * cols, dst);
            return result;
        }

        /**
         * Convert a 3D xframe (three dimensions, one variable) into a 3D xtensor array.
         */
        template <class... V>
        inline auto to_xtensor3d(const xframe<V...>& frame)
        {
            if (frame.dimension_count() != 3)
                throw std::runtime_error("to_xtensor3d: xframe must have exactly 3 dimensions.");
            auto sh = frame.shape();
            xt::xarray<double> result({sh[0], sh[1], sh[2]});
            const auto& var = frame.template variable<0>();
            const double* src = var.data();
            double* dst = result.data();
            std::copy(src, src + result.size(), dst);
            return result;
        }

        /**
         * Build an xframe from a 2D xtensor array and two dimensions.
         * The variable is named "value" by default.
         */
        template <class L = label_type>
        inline auto from_xtensor2d(const xt::xarray<double>& arr,
                                   const dimension<L>& dim0,
                                   const dimension<L>& dim1,
                                   const L& var_name = L("value"))
        {
            auto sh = arr.shape();
            if (sh.size() != 2 || sh[0] != dim0.size() || sh[1] != dim1.size())
                throw std::runtime_error("from_xtensor2d: shape mismatch.");
            auto var = variable<double, L>(sh[0] * sh[1], var_name);
            const double* src = arr.data();
            double* dst = var.data();
            std::copy(src, src + var.size(), dst);
            return xframe<decltype(var)>(std::make_tuple(dim0, dim1), {std::move(var)});
        }

        /**
         * Build an xframe from a 1D xtensor array and one dimension.
         */
        template <class L = label_type>
        inline auto from_xtensor1d(const xt::xarray<double>& arr,
                                   const dimension<L>& dim,
                                   const L& var_name = L("value"))
        {
            if (arr.dimension() != 1 || arr.size() != dim.size())
                throw std::runtime_error("from_xtensor1d: shape mismatch.");
            auto var = variable<double, L>(arr.size(), var_name);
            const double* src = arr.data();
            double* dst = var.data();
            std::copy(src, src + var.size(), dst);
            return xframe<decltype(var)>(std::make_tuple(dim), {std::move(var)});
        }

        /**
         * Extract multiple variables from an xframe as separate 1D xtensor arrays.
         * Returns a tuple of xtensor arrays corresponding to each variable.
         */
        template <class... V>
        inline auto variables_to_xtensors(const xframe<V...>& frame)
        {
            return extract_variables_impl(frame, std::make_index_sequence<sizeof...(V)>{});
        }

        template <class... V, std::size_t... I>
        inline auto extract_variables_impl(const xframe<V...>& frame, std::index_sequence<I...>)
        {
            return std::make_tuple(variable_to_xtensor<V...>(frame)...);
        }

        /**
         * Element‑wise operation: apply a functor (as xtensor expression) to an xframe variable.
         * Returns a new xframe with the same dimensions and the transformed variable.
         */
        template <class... V, class Func>
        inline auto transform_variable(const xframe<V...>& frame, Func&& f)
        {
            auto var = frame.template variable<0>();
            xt::xarray<double> tmp = variable_to_xtensor(frame);
            auto transformed = f(tmp);
            auto result_var = variable<double, label_type>(var.size(), var.name());
            const double* src = transformed.data();
            double* dst = result_var.data();
            std::copy(src, src + var.size(), dst);
            return xframe<decltype(result_var)>(std::make_tuple(frame.dimension(0)), {std::move(result_var)});
        }

        /**
         * Apply a 2D convolution (from xtensor's signal module) to an xframe's variable.
         * Assumes the xframe has 2 dimensions and one variable.
         */
        template <class... V>
        inline auto convolve2d_xframe(const xframe<V...>& frame,
                                      const xt::xarray<double>& kernel)
        {
            auto arr = to_xtensor2d(frame);
            auto convolved = xt::signal::convolve2d(arr, kernel);
            // The result is of different shape (full convolution); for simplicity, take 'same' crop.
            // We'll return as a new xframe with the original dimensions, using the central part.
            std::size_t kh = kernel.shape()[0], kw = kernel.shape()[1];
            std::size_t pad_h = (kh - 1) / 2, pad_w = (kw - 1) / 2;
            auto sh = arr.shape();
            xt::xarray<double> cropped = xt::view(convolved,
                                                  xt::range(pad_h, pad_h + sh[0]),
                                                  xt::range(pad_w, pad_w + sh[1]));
            return from_xtensor2d(cropped, frame.dimension(0), frame.dimension(1));
        }
    } // namespace bridge
} // namespace xframe

#endif // XFRAME_XTENSOR_HPP