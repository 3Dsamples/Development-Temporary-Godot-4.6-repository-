//File 0036 : core/xbuilder.hpp
//Array factory functions: ones, zeros, empty, full, arange, linspace, logspace, eye, meshgrid, and like-variants with C++17, SIMD, and low memory.
#ifndef XTENSOR_XBUILDER_HPP
#define XTENSOR_XBUILDER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xarray.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xbroadcast.hpp"
#include "xgenerator.hpp"
#include "xoperation.hpp"

namespace xt
{
    /****************************************
     * ones – lazy expression of ones
     ****************************************/
    template <class T, class S>
    inline auto ones(S&& shape) noexcept
    {
        return broadcast(T(1), std::forward<S>(shape));
    }

    template <class T, class I, std::size_t L>
    inline auto ones(const I (&shape)[L]) noexcept
    {
        return broadcast(T(1), shape);
    }

    /****************************************
     * zeros – lazy expression of zeros
     ****************************************/
    template <class T, class S>
    inline auto zeros(S&& shape) noexcept
    {
        return broadcast(T(0), std::forward<S>(shape));
    }

    template <class T, class I, std::size_t L>
    inline auto zeros(const I (&shape)[L]) noexcept
    {
        return broadcast(T(0), shape);
    }

    /****************************************
     * full – lazy expression filled with value
     ****************************************/
    template <class T, class S>
    inline auto full(S&& shape, T value) noexcept
    {
        return broadcast(value, std::forward<S>(shape));
    }

    /****************************************
     * empty – allocated container, uninitialized
     ****************************************/
    template <class T, class S>
    inline auto empty(const S& shape)
    {
        using container_type = xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        return container_type(shape);
    }

    template <class T, class I, std::size_t N>
    inline auto empty(const std::array<I, N>& shape)
    {
        using container_type = xtensor_container<uvector<T>, N, DEFAULT_LAYOUT>;
        return container_type(xtl::forward_sequence<typename container_type::shape_type>(shape));
    }

    /****************************************
     * empty_like – container like input expression
     ****************************************/
    template <class E>
    inline auto empty_like(const xexpression<E>& e)
    {
        using temporary_type = typename xcontainer_inner_types<E>::temporary_type;
        auto res = temporary_type::from_shape(e.derived_cast().shape());
        return res;
    }

    /****************************************
     * full_like – filled container like input
     ****************************************/
    template <class E>
    inline auto full_like(const xexpression<E>& e, typename E::value_type fill_value)
    {
        using temporary_type = typename xcontainer_inner_types<E>::temporary_type;
        auto res = temporary_type::from_shape(e.derived_cast().shape());
        res.fill(fill_value);
        return res;
    }

    template <class E>
    inline auto zeros_like(const xexpression<E>& e)
    {
        return full_like(e, typename E::value_type(0));
    }

    template <class E>
    inline auto ones_like(const xexpression<E>& e)
    {
        return full_like(e, typename E::value_type(1));
    }

    /****************************************
     * arange – evenly spaced values [start, stop)
     ****************************************/
    namespace detail
    {
        template <class T, class S>
        class arange_generator
        {
        public:
            using value_type = T;
            using step_type = S;

            arange_generator(T start, T stop, S step, std::size_t num_steps, bool endpoint = false) noexcept
                : m_start(start), m_stop(stop), m_step(step), m_num_steps(num_steps), m_endpoint(endpoint) {}

            template <class... Args>
            T operator()(Args... args) const
            {
                if constexpr (sizeof...(Args) == 0)
                    return static_cast<T>(m_start);
                else
                    return access_impl(args...);
            }

            template <class It>
            T element(It first, It /*last*/) const
            {
                return access_impl(*first);
            }

        private:
            T m_start;
            T m_stop;
            step_type m_step;
            std::size_t m_num_steps;
            bool m_endpoint;

            template <class... Args>
            T access_impl(Args... args) const
            {
                if constexpr (sizeof...(Args) == 1)
                {
                    std::size_t idx = static_cast<std::size_t>(std::get<0>(std::forward_as_tuple(args...)));
                    if (m_endpoint && m_num_steps > 1 && idx == m_num_steps - 1)
                        return static_cast<T>(m_stop);
                    return static_cast<T>(m_start + m_step * static_cast<T>(idx));
                }
                else
                {
                    return static_cast<T>(m_start);
                }
            }
        };

        template <class T, class S>
        auto arange_impl(T start, T stop, S step) noexcept
        {
            std::size_t shape;
            if constexpr (std::is_integral_v<T> && std::is_integral_v<S>)
            {
                if (step == S(0)) throw std::runtime_error("arange: step must not be zero.");
                shape = static_cast<std::size_t>(std::ceil(static_cast<double>(stop - start) / static_cast<double>(step)));
                if (shape == 0) shape = 1;
            }
            else
            {
                if (step == S(0)) throw std::runtime_error("arange: step must not be zero.");
                shape = static_cast<std::size_t>(std::ceil(static_cast<double>(stop - start) / static_cast<double>(step)));
                if (shape == 0) shape = 1;
            }
            return detail::make_xgenerator(arange_generator<T, S>(start, stop, step, shape), std::array<std::size_t, 1>{shape});
        }
    }

    template <class T, class S = T>
    inline auto arange(T start, T stop, S step = 1) noexcept
    {
        return detail::arange_impl(start, stop, step);
    }

    template <class T>
    inline auto arange(T stop) noexcept
    {
        return arange(T(0), stop, T(1));
    }

    /****************************************
     * linspace – num_samples evenly spaced [start, stop]
     ****************************************/
    template <class T>
    inline auto linspace(T start, T stop, std::size_t num_samples = 50, bool endpoint = true) noexcept
    {
        using fp_type = std::common_type_t<T, double>;
        fp_type step = fp_type(stop - start) / std::fmax(fp_type(1), fp_type(num_samples - (endpoint ? 1 : 0)));
        return detail::make_xgenerator(
            detail::arange_generator<fp_type, fp_type>(fp_type(start), fp_type(stop), step, num_samples, endpoint),
            std::array<std::size_t, 1>{num_samples});
    }

    /****************************************
     * logspace – log-scaled evenly spaced [base^start, base^stop]
     ****************************************/
    template <class T>
    inline auto logspace(T start, T stop, std::size_t num_samples = 50, T base = 10, bool endpoint = true) noexcept
    {
        return pow(base, linspace(start, stop, num_samples, endpoint));
    }

    /****************************************
     * eye – identity matrix
     ****************************************/
    namespace detail
    {
        template <class T>
        class eye_fn
        {
        public:
            using value_type = T;
            eye_fn(std::ptrdiff_t k) : m_k(k) {}

            template <class It>
            T operator()(const It& begin, const It& end) const
            {
                using lvalue_type = typename std::iterator_traits<It>::value_type;
                return *(end - 2) + static_cast<lvalue_type>(m_k) == *(end - 1) ? T(1) : T(0);
            }

        private:
            std::ptrdiff_t m_k;
        };
    }

    template <class T>
    inline auto eye(const std::vector<std::size_t>& shape, std::ptrdiff_t k = 0)
    {
        return detail::make_xgenerator(detail::fn_impl<detail::eye_fn<T>>(detail::eye_fn<T>(k)), shape);
    }

    template <class T>
    inline auto eye(std::size_t n, std::ptrdiff_t k = 0)
    {
        return eye<T>({n, n}, k);
    }

    /****************************************
     * meshgrid – coordinate matrices from coordinate vectors
     ****************************************/
    template <class... Es>
    inline auto meshgrid(Es&&... es)
    {
        auto arrays = std::make_tuple(xt::eval(std::forward<Es>(es))...);
        constexpr std::size_t ndim = sizeof...(Es);
        std::array<std::size_t, ndim> shape;
        std::size_t dim = 0;
        std::apply([&](auto&&... arr) {
            ((shape[dim++] = arr.size()), ...);
        }, arrays);

        std::vector<xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>>> results;
        dim = 0;
        std::apply([&](auto&&... arr) {
            auto build = [&](auto& a) {
                xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(
                    std::vector<std::size_t>(shape.begin(), shape.end()));
                for (std::size_t i = 0; i < result.size(); ++i)
                {
                    auto idx = unravel_index(i, result.shape());
                    result[i] = static_cast<double>(a[idx[dim]]);
                }
                results.push_back(std::move(result));
            };
            (build(arr), ...);
        }, arrays);
        return results;
    }

    /****************************************
     * concatenate / stack / vstack / hstack (already in xmanipulation)
     ****************************************/

    /****************************************
     * from_shape – creates container from shape
     ****************************************/
    template <class T, class S>
    inline auto from_shape(const S& shape)
    {
        return empty<T>(shape);
    }

} // namespace xt

#endif // XTENSOR_XBUILDER_HPP