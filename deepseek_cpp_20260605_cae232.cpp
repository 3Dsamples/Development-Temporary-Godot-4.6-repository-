//File 0314 : xframe/xframe_builder.hpp
//Array factory functions for xframe: ones, zeros, empty, full, arange, linspace, logspace with labeled dimensions and SIMD-friendly initialization.
#ifndef XFRAME_BUILDER_HPP
#define XFRAME_BUILDER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <initializer_list>
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

namespace xframe
{
    namespace builder
    {
        /**
         * Create an xframe filled with zeros.
         * @param dims Tuple of dimensions.
         */
        template <class... Dims>
        inline auto zeros(std::tuple<Dims...> dims)
        {
            return xframe<double>(std::move(dims));
        }

        /**
         * Create an xframe filled with ones.
         */
        template <class... Dims>
        inline auto ones(std::tuple<Dims...> dims)
        {
            auto frame = xframe<double>(std::move(dims));
            for (std::size_t i = 0; i < frame.size(); ++i)
            {
                auto row = frame[i];
                // Set all values to 1.0
                std::apply([](auto&... vals) { ((vals = 1.0), ...); }, row);
            }
            return frame;
        }

        /**
         * Create an xframe filled with a constant value.
         */
        template <class... Dims>
        inline auto full(std::tuple<Dims...> dims, double value)
        {
            auto frame = xframe<double>(std::move(dims));
            for (std::size_t i = 0; i < frame.size(); ++i)
            {
                auto row = frame[i];
                std::apply([value](auto&... vals) { ((vals = value), ...); }, row);
            }
            return frame;
        }

        /**
         * Create an xframe with a single variable from an initializer list and dimension.
         */
        template <class Dim>
        inline auto from_values(Dim dim, std::initializer_list<double> values)
        {
            auto frame = xframe<double>(std::make_tuple(dim));
            auto& var = frame.template variable<0>();
            std::size_t i = 0;
            for (auto val : values)
                var[i++] = val;
            return frame;
        }

        /**
         * Create a 1D xframe with an arange of values.
         */
        inline auto arange(double start, double stop, double step = 1.0,
                           const std::string& dim_name = "index")
        {
            std::size_t n = static_cast<std::size_t>(std::ceil((stop - start) / step));
            dimension<label_type> dim(dim_name, n);
            for (std::size_t i = 0; i < n; ++i)
                dim.coord()[i] = label_type(std::to_string(i));
            auto frame = xframe<double>(std::make_tuple(dim));
            auto& var = frame.template variable<0>();
            for (std::size_t i = 0; i < n; ++i)
                var[i] = start + static_cast<double>(i) * step;
            return frame;
        }

        /**
         * Create a 1D xframe with linspace values.
         */
        inline auto linspace(double start, double stop, std::size_t n = 50,
                             const std::string& dim_name = "index", bool endpoint = true)
        {
            dimension<label_type> dim(dim_name, n);
            for (std::size_t i = 0; i < n; ++i)
                dim.coord()[i] = label_type(std::to_string(i));
            auto frame = xframe<double>(std::make_tuple(dim));
            auto& var = frame.template variable<0>();
            if (n == 1)
            {
                var[0] = start;
            }
            else
            {
                double step = (stop - start) / static_cast<double>(endpoint ? n - 1 : n);
                for (std::size_t i = 0; i < n; ++i)
                    var[i] = start + static_cast<double>(i) * step;
                if (endpoint) var[n-1] = stop;
            }
            return frame;
        }

        /**
         * Create a 1D xframe with logspace values.
         */
        inline auto logspace(double start, double stop, std::size_t n = 50,
                             double base = 10.0, const std::string& dim_name = "index",
                             bool endpoint = true)
        {
            auto frame = linspace(start, stop, n, dim_name, endpoint);
            auto& var = frame.template variable<0>();
            for (std::size_t i = 0; i < n; ++i)
                var[i] = std::pow(base, var[i]);
            return frame;
        }

        /**
         * Create a 2D xframe with meshgrid coordinates.
         * @param dim_x First dimension.
         * @param dim_y Second dimension.
         * @param values 2D initializer list of values.
         */
        inline auto meshgrid(const dimension<label_type>& dim_x,
                             const dimension<label_type>& dim_y,
                             std::initializer_list<std::initializer_list<double>> values)
        {
            auto frame = xframe<double>(std::make_tuple(dim_x, dim_y));
            auto& var = frame.template variable<0>();
            std::size_t row = 0;
            for (const auto& row_vals : values)
            {
                std::size_t col = 0;
                for (double val : row_vals)
                {
                    if (row < dim_x.size() && col < dim_y.size())
                        var[row * dim_y.size() + col] = val;
                    ++col;
                }
                ++row;
            }
            return frame;
        }
    } // namespace builder
} // namespace xframe

#endif // XFRAME_BUILDER_HPP