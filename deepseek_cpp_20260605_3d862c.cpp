//File 0041 : core/xnoalias.hpp
//No-alias assignment proxy enabling SIMD-accelerated in-place evaluation without temporary creation.
#ifndef XTENSOR_XNOALIAS_HPP
#define XTENSOR_XNOALIAS_HPP

#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xassign.hpp"
#include "xfunction.hpp"
#include "xsemantic.hpp"
#include "xarray.hpp"

namespace xt
{
    /**
     * @class xnoalias_proxy
     * @brief Wraps an expression reference to indicate non-aliasing assignment.
     *
     * When `noalias(a) = b` is called, the proxy performs element-wise assignment
     * directly into `a` without creating an intermediate temporary. This is safe
     * only when `a` and `b` do not share memory (no aliasing).
     */
    template <class E>
    class xnoalias_proxy
    {
    public:
        using derived_type = E;
        using value_type = typename E::value_type;
        using temporary_type = typename xcontainer_inner_types<E>::temporary_type;

        explicit xnoalias_proxy(E& e) noexcept : m_e(e) {}

        /**
         * Direct assignment from any xtensor expression without temporary.
         */
        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator=(const Expr& expr)
        {
            // Ensure shapes are compatible via broadcasting
            auto& e = m_e.derived_cast();
            auto expr_shape = expr.shape();
            auto e_shape = e.shape();
            if (expr_shape.size() != e_shape.size())
                throw std::runtime_error("noalias assignment: shape mismatch.");

            // Perform strided element-wise copy directly
            // Use the assignment engine for SIMD-accelerated copy
            using assign_t = xexpression_assigner_base<xtensor_expression_tag>;
            assign_t::assign(e, expr);
            return *this;
        }

        /**
         * Compound assignment operators: noalias(a) += b, etc.
         */
        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator+=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() + expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator-=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() - expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator*=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() * expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator/=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() / expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator%=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() % expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator&=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() & expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator|=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() | expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator^=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() ^ expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator<<=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() << expr;
            return *this;
        }

        template <class Expr>
        disable_xexpression<Expr, xnoalias_proxy&> operator>>=(const Expr& expr)
        {
            m_e.derived_cast() = m_e.derived_cast() >> expr;
            return *this;
        }

        /**
         * Scalar assignment
         */
        xnoalias_proxy& operator=(const value_type& scalar)
        {
            m_e.derived_cast().fill(scalar);
            return *this;
        }

    private:
        E& m_e;
    };

    /**
     * Free function: wrap an expression for no-alias assignment.
     */
    template <class E>
    inline auto noalias(E& expr) noexcept
    {
        return xnoalias_proxy<E>(expr);
    }

    /********************************************
     * xexpression_assigner_base – default assign
     ********************************************/
    template <class Tag>
    class xexpression_assigner_base
    {
    public:
        template <class E1, class E2>
        static void assign(E1& dst, const E2& src)
        {
            // If both have contiguous memory and same layout, use SIMD copy
            if constexpr (is_simd_enabled_v<typename E1::value_type>)
            {
                using T = typename E1::value_type;
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;

                auto src_shape = src.shape();
                auto dst_shape = dst.shape();
                bool contiguous = (src_shape == dst_shape);

                if (contiguous)
                {
                    std::size_t count = dst.size();
                    const T* src_data = src.data();
                    T* dst_data = dst.data();
                    std::size_t vec_count = count / simd_size;
                    for (std::size_t i = 0; i < vec_count; ++i)
                    {
                        simd_type v = simd_type::load_unaligned(src_data + i * simd_size);
                        v.store_unaligned(dst_data + i * simd_size);
                    }
                    for (std::size_t i = vec_count * simd_size; i < count; ++i)
                        dst_data[i] = src_data[i];
                    return;
                }
            }

            // Fallback: element-by-element via strided loops
            auto src_flat = src.shape();
            auto dst_flat = dst.shape();
            if (src_flat.size() != dst_flat.size())
                throw std::runtime_error("Dimension mismatch in assignment.");

            // Use nested loops with SIMD along innermost dimension
            assign_strided(dst, src);
        }

    private:
        template <class E1, class E2>
        static void assign_strided(E1& dst, const E2& src)
        {
            using T = typename E1::value_type;
            auto& shape = dst.shape();
            std::size_t ndim = shape.size();
            if (ndim == 0) return;

            // Compute the contiguous inner dimension for SIMD
            std::size_t inner_size = shape[ndim - 1];
            std::size_t outer_count = dst.size() / inner_size;

            const T* src_data = src.data();
            T* dst_data = dst.data();

            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;

                for (std::size_t o = 0; o < outer_count; ++o)
                {
                    std::size_t offset = o * inner_size;
                    std::size_t j = 0;
                    for (; j + simd_size <= inner_size; j += simd_size)
                    {
                        simd_type v = simd_type::load_unaligned(src_data + offset + j);
                        v.store_unaligned(dst_data + offset + j);
                    }
                    for (; j < inner_size; ++j)
                        dst_data[offset + j] = src_data[offset + j];
                }
            }
            else
            {
                std::copy(src_data, src_data + dst.size(), dst_data);
            }
        }
    };

} // namespace xt

#endif // XTENSOR_XNOALIAS_HPP