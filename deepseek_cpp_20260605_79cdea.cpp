//File 0053 : core/xvectorize.hpp
//Vectorize scalar functions over arrays with SIMD-accelerated evaluation and automatic broadcasting.
#ifndef XTENSOR_XVECTORIZE_HPP
#define XTENSOR_XVECTORIZE_HPP

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xarray.hpp"

namespace xt
{
    /**
     * @class xvectorize
     * @brief Wraps a scalar function for element-wise application to expressions.
     *
     * The scalar function is applied to each element of the input expression(s),
     * returning a new expression that lazily evaluates the function.
     * Supports SIMD evaluation when the function provides a simd_apply overload.
     */
    template <class Func>
    class xvectorize
    {
    public:
        explicit xvectorize(Func f) noexcept : m_f(std::move(f)) {}

        /**
         * Apply the vectorized function to a single expression.
         */
        template <class E>
        auto operator()(E&& e) const
        {
            return detail::make_xfunction(m_f, std::forward<E>(e));
        }

        /**
         * Apply the vectorized function to two expressions.
         */
        template <class E1, class E2>
        auto operator()(E1&& e1, E2&& e2) const
        {
            return detail::make_xfunction(m_f, std::forward<E1>(e1), std::forward<E2>(e2));
        }

        /**
         * Apply the vectorized function to three expressions.
         */
        template <class E1, class E2, class E3>
        auto operator()(E1&& e1, E2&& e2, E3&& e3) const
        {
            return detail::make_xfunction(m_f, std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
        }

        /**
         * Apply the vectorized function to four expressions.
         */
        template <class E1, class E2, class E3, class E4>
        auto operator()(E1&& e1, E2&& e2, E3&& e3, E4&& e4) const
        {
            return detail::make_xfunction(m_f, std::forward<E1>(e1), std::forward<E2>(e2),
                                          std::forward<E3>(e3), std::forward<E4>(e4));
        }

        /**
         * Apply the vectorized function to five expressions.
         */
        template <class E1, class E2, class E3, class E4, class E5>
        auto operator()(E1&& e1, E2&& e2, E3&& e3, E4&& e4, E5&& e5) const
        {
            return detail::make_xfunction(m_f, std::forward<E1>(e1), std::forward<E2>(e2),
                                          std::forward<E3>(e3), std::forward<E4>(e4), std::forward<E5>(e5));
        }

    private:
        Func m_f;
    };

    /**
     * Free function to create a vectorized callable from a scalar function.
     */
    template <class Func>
    inline auto vectorize(Func&& f) noexcept
    {
        return xvectorize<std::decay_t<Func>>(std::forward<Func>(f));
    }

    /*********************************************
     * Predefined vectorized wrappers for common math functions
     *********************************************/
    namespace detail
    {
        // Functor that calls std::func via operator()
        template <class Func>
        struct std_function_wrapper
        {
            Func func;
            template <class... Args>
            auto operator()(Args&&... args) const -> decltype(func(std::forward<Args>(args)...))
            {
                return func(std::forward<Args>(args)...);
            }
        };
    }

    // Create vectorized versions of common functions
    inline auto vabs = vectorize([](auto x) { return std::abs(x); });
    inline auto vsqrt = vectorize([](auto x) { return std::sqrt(x); });
    inline auto vexp = vectorize([](auto x) { return std::exp(x); });
    inline auto vlog = vectorize([](auto x) { return std::log(x); });
    inline auto vsin = vectorize([](auto x) { return std::sin(x); });
    inline auto vcos = vectorize([](auto x) { return std::cos(x); });
    inline auto vtan = vectorize([](auto x) { return std::tan(x); });
    inline auto vasin = vectorize([](auto x) { return std::asin(x); });
    inline auto vacos = vectorize([](auto x) { return std::acos(x); });
    inline auto vatan = vectorize([](auto x) { return std::atan(x); });
    inline auto vsinh = vectorize([](auto x) { return std::sinh(x); });
    inline auto vcosh = vectorize([](auto x) { return std::cosh(x); });
    inline auto vtanh = vectorize([](auto x) { return std::tanh(x); });
    inline auto vceil = vectorize([](auto x) { return std::ceil(x); });
    inline auto vfloor = vectorize([](auto x) { return std::floor(x); });
    inline auto vround = vectorize([](auto x) { return std::round(x); });
    inline auto vtrunc = vectorize([](auto x) { return std::trunc(x); });

    // Vectorized binary functions
    inline auto vpow = vectorize([](auto x, auto y) { return std::pow(x, y); });
    inline auto vatan2 = vectorize([](auto y, auto x) { return std::atan2(y, x); });
    inline auto vhypot = vectorize([](auto x, auto y) { return std::hypot(x, y); });
    inline auto vfmod = vectorize([](auto x, auto y) { return std::fmod(x, y); });
    inline auto vmax = vectorize([](auto x, auto y) { return x > y ? x : y; });
    inline auto vmin = vectorize([](auto x, auto y) { return x < y ? x : y; });

} // namespace xt

#endif // XTENSOR_XVECTORIZE_HPP