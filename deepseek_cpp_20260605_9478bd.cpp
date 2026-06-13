//File 0024 : core/xcomplex.hpp
//Complex number support with SIMD-accelerated operations and math functions for complex arrays.
#ifndef XTENSOR_XCOMPLEX_HPP
#define XTENSOR_XCOMPLEX_HPP

#include <complex>
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
    /*********************************************
     * Complex functors (with SIMD when possible)
     *********************************************/
    namespace math
    {
        // real part
        struct real_fun
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::real(c);
            }
            template <class T>
            auto simd_apply(const xsimd::batch<std::complex<T>, default_simd_arch>& c) const noexcept
            {
                return xsimd::real(c);
            }
        };

        // imag part
        struct imag_fun
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::imag(c);
            }
            template <class T>
            auto simd_apply(const xsimd::batch<std::complex<T>, default_simd_arch>& c) const noexcept
            {
                return xsimd::imag(c);
            }
        };

        // complex conjugate
        struct conj_fun
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::conj(c);
            }
            template <class T>
            auto simd_apply(const xsimd::batch<std::complex<T>, default_simd_arch>& c) const noexcept
            {
                return xsimd::conj(c);
            }
        };

        // magnitude (abs)
        struct abs_fun
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::abs(c);
            }
            template <class T>
            auto simd_apply(const xsimd::batch<std::complex<T>, default_simd_arch>& c) const noexcept
            {
                return xsimd::abs(c);
            }
        };

        // argument
        struct arg_fun
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::arg(c);
            }
        };

        // norm (squared magnitude)
        struct norm_fun
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::norm(c);
            }
        };

        // proj (Riemann sphere projection)
        struct proj_fun
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::proj(c);
            }
        };

        // complex from polar
        struct polar_fun
        {
            template <class T>
            auto operator()(const T& rho, const T& theta) const noexcept
            {
                return std::polar(rho, theta);
            }
        };

        // complex exponential
        struct exp_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::exp(c);
            }
            template <class T>
            auto simd_apply(const xsimd::batch<std::complex<T>, default_simd_arch>& c) const noexcept
            {
                return xsimd::exp(c);
            }
        };

        // complex logarithm
        struct log_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::log(c);
            }
        };

        // complex square root
        struct sqrt_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::sqrt(c);
            }
        };

        // complex power (scalar exponent)
        struct pow_fun_complex
        {
            template <class T, class U>
            auto operator()(const std::complex<T>& x, const U& y) const noexcept
            {
                return std::pow(x, y);
            }
        };

        // complex sine
        struct sin_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::sin(c);
            }
        };

        // complex cosine
        struct cos_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::cos(c);
            }
        };

        // complex tangent
        struct tan_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::tan(c);
            }
        };

        // complex hyperbolic sine
        struct sinh_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::sinh(c);
            }
        };

        // complex hyperbolic cosine
        struct cosh_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::cosh(c);
            }
        };

        // complex hyperbolic tangent
        struct tanh_fun_complex
        {
            template <class T>
            auto operator()(const std::complex<T>& c) const noexcept
            {
                return std::tanh(c);
            }
        };
    }

    /*********************************************
     * Free functions for complex operations
     *********************************************/

    /**
     * Return the real part of each element.
     */
    template <class E>
    inline auto real(E&& e)
    {
        return detail::make_xfunction<math::real_fun>(std::forward<E>(e));
    }

    /**
     * Return the imaginary part of each element.
     */
    template <class E>
    inline auto imag(E&& e)
    {
        return detail::make_xfunction<math::imag_fun>(std::forward<E>(e));
    }

    /**
     * Return the complex conjugate of each element.
     */
    template <class E>
    inline auto conj(E&& e)
    {
        return detail::make_xfunction<math::conj_fun>(std::forward<E>(e));
    }

    /**
     * Return the magnitude (absolute value) of each complex element.
     */
    template <class E>
    inline auto abs(E&& e)
    {
        return detail::make_xfunction<math::abs_fun>(std::forward<E>(e));
    }

    /**
     * Return the phase angle of each complex element.
     */
    template <class E>
    inline auto arg(E&& e)
    {
        return detail::make_xfunction<math::arg_fun>(std::forward<E>(e));
    }

    /**
     * Return the squared magnitude of each complex element.
     */
    template <class E>
    inline auto norm(E&& e)
    {
        return detail::make_xfunction<math::norm_fun>(std::forward<E>(e));
    }

    /**
     * Return the projection onto the Riemann sphere.
     */
    template <class E>
    inline auto proj(E&& e)
    {
        return detail::make_xfunction<math::proj_fun>(std::forward<E>(e));
    }

    /**
     * Create complex numbers from polar magnitudes and angles.
     */
    template <class E1, class E2>
    inline auto polar(E1&& r, E2&& theta)
    {
        return detail::make_xfunction<math::polar_fun>(std::forward<E1>(r), std::forward<E2>(theta));
    }

    // Overload standard math functions for complex expressions

    template <class E>
    inline auto exp(E&& e)
    {
        return detail::make_xfunction<math::exp_fun_complex>(std::forward<E>(e));
    }

    template <class E>
    inline auto log(E&& e)
    {
        return detail::make_xfunction<math::log_fun_complex>(std::forward<E>(e));
    }

    template <class E>
    inline auto sqrt(E&& e)
    {
        return detail::make_xfunction<math::sqrt_fun_complex>(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline auto pow(E1&& e1, E2&& e2)
    {
        return detail::make_xfunction<math::pow_fun_complex>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E>
    inline auto sin(E&& e)
    {
        return detail::make_xfunction<math::sin_fun_complex>(std::forward<E>(e));
    }

    template <class E>
    inline auto cos(E&& e)
    {
        return detail::make_xfunction<math::cos_fun_complex>(std::forward<E>(e));
    }

    template <class E>
    inline auto tan(E&& e)
    {
        return detail::make_xfunction<math::tan_fun_complex>(std::forward<E>(e));
    }

    template <class E>
    inline auto sinh(E&& e)
    {
        return detail::make_xfunction<math::sinh_fun_complex>(std::forward<E>(e));
    }

    template <class E>
    inline auto cosh(E&& e)
    {
        return detail::make_xfunction<math::cosh_fun_complex>(std::forward<E>(e));
    }

    template <class E>
    inline auto tanh(E&& e)
    {
        return detail::make_xfunction<math::tanh_fun_complex>(std::forward<E>(e));
    }

    // Convenience: Construct complex array from real and imaginary parts
    template <class E1, class E2>
    inline auto complex_array(E1&& real_part, E2&& imag_part)
    {
        using value_type = std::complex<typename std::decay_t<E1>::value_type>;
        // Make xfunction that constructs std::complex from two expressions
        struct make_complex_fun
        {
            template <class T>
            auto operator()(T r, T i) const { return std::complex<T>(r, i); }
        };
        return detail::make_xfunction<make_complex_fun>(std::forward<E1>(real_part), std::forward<E2>(imag_part));
    }

} // namespace xt

#endif // XTENSOR_XCOMPLEX_HPP