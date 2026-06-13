//File 0016 : core/xtensor_simd.hpp
//SIMD abstraction layer with generic batch operations, architecture dispatching, and math functions.
#ifndef XTENSOR_SIMD_HPP
#define XTENSOR_SIMD_HPP

#include <type_traits>
#include <utility>
#include <cmath>

#include "xtensor_config.hpp"
#include <xsimd/xsimd.hpp>

namespace xt
{
    namespace xt_simd
    {
        using namespace xsimd;

        // Default arch from config
        using default_arch = xt::default_simd_arch;

        // Batch type for a scalar type on default arch
        template <class T>
        using batch = xsimd::batch<T, default_arch>;

        // Boolean batch type
        template <class T>
        using batch_bool = xsimd::batch_bool<T, default_arch>;

        // Check if a type has SIMD registers on default arch
        template <class T>
        inline constexpr bool has_simd_register_v = xsimd::has_simd_register<T, default_arch>::value;

        // Get the default SIMD value type for an expression's value type
        template <class E>
        using xsimd_default_value_type = typename std::decay_t<E>::value_type;

        // Aligned load/store wrappers
        template <class T>
        inline batch<T> load_aligned(const T* ptr) noexcept
        {
            return batch<T>::load_aligned(ptr);
        }

        template <class T>
        inline batch<T> load_unaligned(const T* ptr) noexcept
        {
            return batch<T>::load_unaligned(ptr);
        }

        template <class T>
        inline void store_aligned(T* ptr, const batch<T>& b) noexcept
        {
            b.store_aligned(ptr);
        }

        template <class T>
        inline void store_unaligned(T* ptr, const batch<T>& b) noexcept
        {
            b.store_unaligned(ptr);
        }

        // Select operation: if cond then a else b (element-wise)
        template <class T>
        inline batch<T> select(const batch_bool<T>& cond, const batch<T>& a, const batch<T>& b) noexcept
        {
            return xsimd::select(cond, a, b);
        }

        // Bitwise operations on batches
        template <class T>
        inline batch<T> bitwise_and(const batch<T>& a, const batch<T>& b) noexcept
        {
            return a & b;
        }

        template <class T>
        inline batch<T> bitwise_or(const batch<T>& a, const batch<T>& b) noexcept
        {
            return a | b;
        }

        template <class T>
        inline batch<T> bitwise_xor(const batch<T>& a, const batch<T>& b) noexcept
        {
            return a ^ b;
        }

        template <class T>
        inline batch<T> bitwise_not(const batch<T>& a) noexcept
        {
            return ~a;
        }

        // Shift operations
        template <class T>
        inline batch<T> shift_left(const batch<T>& a, int n) noexcept
        {
            return a << n;
        }

        template <class T>
        inline batch<T> shift_right(const batch<T>& a, int n) noexcept
        {
            return a >> n;
        }

        // Math functions for SIMD batches (overloads for batch types)
        // We define them in a separate namespace to avoid ambiguity with std, using xsimd's implementations.
        namespace math
        {
            using xsimd::abs;
            using xsimd::fabs;
            using xsimd::fmod;
            using xsimd::remainder;
            using xsimd::fma;
            using xsimd::fmax;
            using xsimd::fmin;
            using xsimd::fdim;
            using xsimd::exp;
            using xsimd::exp2;
            using xsimd::expm1;
            using xsimd::log;
            using xsimd::log10;
            using xsimd::log2;
            using xsimd::log1p;
            using xsimd::pow;
            using xsimd::sqrt;
            using xsimd::cbrt;
            using xsimd::hypot;
            using xsimd::sin;
            using xsimd::cos;
            using xsimd::tan;
            using xsimd::asin;
            using xsimd::acos;
            using xsimd::atan;
            using xsimd::atan2;
            using xsimd::sinh;
            using xsimd::cosh;
            using xsimd::tanh;
            using xsimd::asinh;
            using xsimd::acosh;
            using xsimd::atanh;
            using xsimd::erf;
            using xsimd::erfc;
            using xsimd::tgamma;
            using xsimd::lgamma;
            using xsimd::ceil;
            using xsimd::floor;
            using xsimd::trunc;
            using xsimd::round;
            using xsimd::nearbyint;
            using xsimd::rint;
            using xsimd::isfinite;
            using xsimd::isinf;
            using xsimd::isnan;
        }

        // Scalar-vector promotion: if one argument is scalar, broadcast to batch
        template <class T>
        inline batch<T> promote_to_batch(T scalar) noexcept
        {
            return batch<T>(scalar);
        }

        // Functor wrappers for SIMD operations: allow custom functors to have simd_apply methods
        namespace functor
        {
            // Example plus functor with simd_apply
            struct simd_plus
            {
                template <class T>
                batch<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a + b;
                }
                template <class T>
                batch<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a + b;
                }
            };

            struct simd_minus
            {
                template <class T>
                batch<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a - b;
                }
                template <class T>
                batch<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a - b;
                }
            };

            struct simd_multiplies
            {
                template <class T>
                batch<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a * b;
                }
                template <class T>
                batch<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a * b;
                }
            };

            struct simd_divides
            {
                template <class T>
                batch<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a / b;
                }
                template <class T>
                batch<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a / b;
                }
            };

            struct simd_negate
            {
                template <class T>
                batch<T> operator()(const batch<T>& a) const noexcept
                {
                    return -a;
                }
                template <class T>
                batch<T> simd_apply(const batch<T>& a) const noexcept
                {
                    return -a;
                }
            };

            struct simd_abs
            {
                template <class T>
                batch<T> operator()(const batch<T>& a) const noexcept
                {
                    return xt_simd::math::abs(a);
                }
                template <class T>
                batch<T> simd_apply(const batch<T>& a) const noexcept
                {
                    return xt_simd::math::abs(a);
                }
            };

            struct simd_sqrt
            {
                template <class T>
                batch<T> operator()(const batch<T>& a) const noexcept
                {
                    return xt_simd::math::sqrt(a);
                }
                template <class T>
                batch<T> simd_apply(const batch<T>& a) const noexcept
                {
                    return xt_simd::math::sqrt(a);
                }
            };

            // Comparison functors returning boolean batches
            struct simd_less
            {
                template <class T>
                batch_bool<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a < b;
                }
                template <class T>
                batch_bool<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a < b;
                }
            };

            struct simd_less_equal
            {
                template <class T>
                batch_bool<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a <= b;
                }
                template <class T>
                batch_bool<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a <= b;
                }
            };

            struct simd_greater
            {
                template <class T>
                batch_bool<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a > b;
                }
                template <class T>
                batch_bool<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a > b;
                }
            };

            struct simd_greater_equal
            {
                template <class T>
                batch_bool<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a >= b;
                }
                template <class T>
                batch_bool<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a >= b;
                }
            };

            struct simd_equal
            {
                template <class T>
                batch_bool<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a == b;
                }
                template <class T>
                batch_bool<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a == b;
                }
            };

            struct simd_not_equal
            {
                template <class T>
                batch_bool<T> operator()(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a != b;
                }
                template <class T>
                batch_bool<T> simd_apply(const batch<T>& a, const batch<T>& b) const noexcept
                {
                    return a != b;
                }
            };

        } // namespace functor

    } // namespace xt_simd

    // Bring xt_simd into xt for convenience? No, avoid pollution.
    // But in other files we may use xt_simd::batch etc.

} // namespace xt

#endif // XTENSOR_SIMD_HPP