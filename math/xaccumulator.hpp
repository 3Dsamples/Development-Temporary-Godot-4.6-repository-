// include/xtu/math/xaccumulator.hpp
// xtensor-unified - Cumulative operations (cumsum, cumprod, cummax, cummin)
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_MATH_XACCUMULATOR_HPP
#define XTU_MATH_XACCUMULATOR_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/containers/xtensor.hpp"
#include "xtu/views/xview.hpp"

XTU_NAMESPACE_BEGIN
namespace math {

namespace detail {
    // Check if value is NaN (specialized for floating point)
    template <class T>
    bool is_nan(const T& val) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::isnan(val);
        } else {
            return false;
        }
    }

    // Helper to traverse and accumulate along a specified axis
    template <class E, class Accumulator, class ResultContainer>
    void accumulate_along_axis(const E& expr, ResultContainer& result, size_t axis, Accumulator acc) {
        using value_type = typename E::value_type;
        size_t ndim = expr.dimension();
        XTU_ASSERT_MSG(axis < ndim, "Axis out of range");
        size_t axis_size = expr.shape()[axis];
        
        // Create index arrays for source and destination
        std::vector<size_t> src_idx(ndim);
        std::vector<size_t> dst_idx(ndim);
        
        // Recursive traversal over all dimensions except axis
        std::function<void(size_t)> traverse = [&](size_t dim) {
            if (dim == ndim) {
                // We have a complete set of fixed indices for all dimensions except axis.
                // Perform cumulative operation along axis.
                value_type running = acc.identity();
                for (size_t k = 0; k < axis_size; ++k) {
                    src_idx[axis] = k;
                    running = acc(running, expr(src_idx));
                    dst_idx[axis] = k;
                    result(dst_idx) = running;
                }
                return;
            }
            if (dim == axis) {
                // Skip axis dimension in recursion; we handle it inside the leaf
                traverse(dim + 1);
                return;
            }
            for (size_t i = 0; i < expr.shape()[dim]; ++i) {
                src_idx[dim] = i;
                dst_idx[dim] = i;
                traverse(dim + 1);
            }
        };
        traverse(0);
    }

    // Accumulator functors
    template <class T>
    struct cumsum_accumulator {
        using value_type = T;
        value_type running;
        cumsum_accumulator() : running(identity()) {}
        static value_type identity() { return value_type(0); }
        value_type operator()(const value_type& acc, const value_type& val) {
            running = acc + val;
            return running;
        }
    };

    template <class T>
    struct cumprod_accumulator {
        using value_type = T;
        value_type running;
        cumprod_accumulator() : running(identity()) {}
        static value_type identity() { return value_type(1); }
        value_type operator()(const value_type& acc, const value_type& val) {
            running = acc * val;
            return running;
        }
    };

    template <class T>
    struct cummax_accumulator {
        using value_type = T;
        value_type running;
        cummax_accumulator() : running(identity()) {}
        static value_type identity() { return std::numeric_limits<T>::lowest(); }
        value_type operator()(const value_type& acc, const value_type& val) {
            running = (acc > val) ? acc : val;
            return running;
        }
    };

    template <class T>
    struct cummin_accumulator {
        using value_type = T;
        value_type running;
        cummin_accumulator() : running(identity()) {}
        static value_type identity() { return std::numeric_limits<T>::max(); }
        value_type operator()(const value_type& acc, const value_type& val) {
            running = (acc < val) ? acc : val;
            return running;
        }
    };

    // NaN-aware accumulators: skip NaN, propagate NaN only if all are NaN
    template <class T>
    struct nancumsum_accumulator {
        using value_type = T;
        value_type running;
        bool has_valid;
        nancumsum_accumulator() : running(identity()), has_valid(false) {}
        static value_type identity() { return value_type(0); }
        value_type operator()(const value_type& acc, const value_type& val) {
            if (is_nan(val)) {
                // Keep previous running value; if no valid yet, keep as is (which is 0)
                if (!has_valid) {
                    running = std::numeric_limits<T>::quiet_NaN();
                }
            } else {
                if (!has_valid) {
                    running = val;
                    has_valid = true;
                } else {
                    running = running + val;
                }
            }
            return running;
        }
    };

    template <class T>
    struct nancumprod_accumulator {
        using value_type = T;
        value_type running;
        bool has_valid;
        nancumprod_accumulator() : running(identity()), has_valid(false) {}
        static value_type identity() { return value_type(1); }
        value_type operator()(const value_type& acc, const value_type& val) {
            if (is_nan(val)) {
                if (!has_valid) {
                    running = std::numeric_limits<T>::quiet_NaN();
                }
                // else keep previous running
            } else {
                if (!has_valid) {
                    running = val;
                    has_valid = true;
                } else {
                    running = running * val;
                }
            }
            return running;
        }
    };

    template <class T>
    struct nancummax_accumulator {
        using value_type = T;
        value_type running;
        bool has_valid;
        nancummax_accumulator() : running(identity()), has_valid(false) {}
        static value_type identity() { return std::numeric_limits<T>::lowest(); }
        value_type operator()(const value_type& acc, const value_type& val) {
            if (is_nan(val)) {
                if (!has_valid) {
                    running = std::numeric_limits<T>::quiet_NaN();
                }
            } else {
                if (!has_valid) {
                    running = val;
                    has_valid = true;
                } else {
                    running = (running > val) ? running : val;
                }
            }
            return running;
        }
    };

    template <class T>
    struct nancummin_accumulator {
        using value_type = T;
        value_type running;
        bool has_valid;
        nancummin_accumulator() : running(identity()), has_valid(false) {}
        static value_type identity() { return std::numeric_limits<T>::max(); }
        value_type operator()(const value_type& acc, const value_type& val) {
            if (is_nan(val)) {
                if (!has_valid) {
                    running = std::numeric_limits<T>::quiet_NaN();
                }
            } else {
                if (!has_valid) {
                    running = val;
                    has_valid = true;
                } else {
                    running = (running < val) ? running : val;
                }
            }
            return running;
        }
    };
} // namespace detail

// #############################################################################
// cumsum
// #############################################################################
template <class E>
auto cumsum(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::cumsum_accumulator<value_type>());
    return result;
}

// #############################################################################
// cumprod
// #############################################################################
template <class E>
auto cumprod(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::cumprod_accumulator<value_type>());
    return result;
}

// #############################################################################
// cummax
// #############################################################################
template <class E>
auto cummax(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::cummax_accumulator<value_type>());
    return result;
}

// #############################################################################
// cummin
// #############################################################################
template <class E>
auto cummin(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::cummin_accumulator<value_type>());
    return result;
}

// #############################################################################
// nancumsum (ignoring NaN)
// #############################################################################
template <class E>
auto nancumsum(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::nancumsum_accumulator<value_type>());
    return result;
}

// #############################################################################
// nancumprod (ignoring NaN)
// #############################################################################
template <class E>
auto nancumprod(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::nancumprod_accumulator<value_type>());
    return result;
}

// #############################################################################
// nancummax (ignoring NaN)
// #############################################################################
template <class E>
auto nancummax(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::nancummax_accumulator<value_type>());
    return result;
}

// #############################################################################
// nancummin (ignoring NaN)
// #############################################################################
template <class E>
auto nancummin(const xexpression<E>& e, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result(expr.shape());
    detail::accumulate_along_axis(expr, result, axis, detail::nancummin_accumulator<value_type>());
    return result;
}

} // namespace math

// Bring into main namespace for convenience
using math::cumsum;
using math::cumprod;
using math::cummax;
using math::cummin;
using math::nancumsum;
using math::nancumprod;
using math::nancummax;
using math::nancummin;

XTU_NAMESPACE_END

#endif // XTU_MATH_XACCUMULATOR_HPP