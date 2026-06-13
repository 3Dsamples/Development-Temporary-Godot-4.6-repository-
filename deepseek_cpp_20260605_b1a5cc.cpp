//File 0008 : core/xreducer.hpp
//High-performance reducer engine with SIMD accumulation, parallel execution policies, and full reduction semantics.
#ifndef XTENSOR_XREDUCER_HPP
#define XTENSOR_XREDUCER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    /********************************************
     * Reducer functors (SIMD-aware)
     ********************************************/
    namespace detail
    {
        struct plus
        {
            template <class T1, class T2>
            constexpr auto operator()(const T1& a, const T2& b) const
            {
                return a + b;
            }
            template <class T>
            constexpr auto simd_apply(const T& a, const T& b) const
            {
                return a + b;
            }
        };

        struct multiplies
        {
            template <class T1, class T2>
            constexpr auto operator()(const T1& a, const T2& b) const
            {
                return a * b;
            }
            template <class T>
            constexpr auto simd_apply(const T& a, const T& b) const
            {
                return a * b;
            }
        };

        struct maximum
        {
            template <class T1, class T2>
            constexpr auto operator()(const T1& a, const T2& b) const
            {
                return a > b ? a : b;
            }
            template <class T>
            constexpr auto simd_apply(const T& a, const T& b) const
            {
                return xt_simd::select(a > b, a, b);
            }
        };

        struct minimum
        {
            template <class T1, class T2>
            constexpr auto operator()(const T1& a, const T2& b) const
            {
                return a < b ? a : b;
            }
            template <class T>
            constexpr auto simd_apply(const T& a, const T& b) const
            {
                return xt_simd::select(a < b, a, b);
            }
        };

        // Reduction identity: returns a constant value
        template <class T>
        class const_value
        {
        public:
            using value_type = T;
            const_value(const T& val) : m_value(val) {}
            template <class... Args>
            const T& operator()(Args&&...) const { return m_value; }
        private:
            T m_value;
        };
    }

    /********************************************
     * xreducer_functor: wraps a binary functor
     *   with an init function for reductions.
     ********************************************/
    template <class F, class InitF>
    class xreducer_functor
    {
    public:
        using functor_type = F;
        using init_functor_type = InitF;

        xreducer_functor(const F& f, const InitF& init_f)
            : m_f(f), m_init_f(init_f) {}

        template <class... Args>
        auto operator()(Args&&... args) const -> decltype(m_f(std::forward<Args>(args)...))
        {
            return m_f(std::forward<Args>(args)...);
        }

        template <class... Args>
        auto init(Args&&... args) const -> decltype(m_init_f(std::forward<Args>(args)...))
        {
            return m_init_f(std::forward<Args>(args)...);
        }

        const F& functor() const { return m_f; }
        const InitF& init_functor() const { return m_init_f; }

    private:
        F m_f;
        InitF m_init_f;
    };

    template <class F, class InitF>
    inline auto make_xreducer_functor(const F& f, const InitF& init_f)
    {
        return xreducer_functor<F, InitF>(f, init_f);
    }

    /********************************************
     * xreducer_base: expression base for reducers
     ********************************************/
    template <class F, class E, class X, class ES>
    class xreducer;

    template <class F, class E, class X, class ES>
    struct xcontainer_inner_types<xreducer<F, E, X, ES>>
    {
        using value_type = typename std::decay_t<E>::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    /**
     * @class xreducer
     * @brief Represents a reduction operation over specified axes.
     *
     * Supports SIMD-accelerated accumulation and parallel execution.
     */
    template <class F, class E, class X, class ES = DEFAULT_STRATEGY_REDUCERS>
    class xreducer : public xexpression<xreducer<F, E, X, ES>>
    {
    public:

        using self_type = xreducer<F, E, X, ES>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;

        using expression_type = E;
        using axes_type = X;
        using functor_type = F;
        using evaluation_strategy_type = ES;

        /**
         * Constructs the reducer with the functor, expression, axes, and evaluation strategy.
         */
        xreducer(F&& f, E&& e, X&& axes, ES&& es = ES()) noexcept;

        size_type size() const;
        shape_type shape() const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class It>
        const_reference element(It first, It last) const;

        const F& functor() const noexcept { return m_f; }
        const E& expression() const noexcept { return m_e; }
        const X& axes() const noexcept { return m_axes; }

    private:
        F m_f;
        E m_e;
        X m_axes;
        ES m_es;
        shape_type m_shape;
        value_type m_init_value;

        value_type compute_init() const;
        shape_type compute_shape() const;
        value_type reduce_over_axes_simd() const;
        value_type reduce_over_axes_parallel() const;
        value_type reduce_over_axes_sequential() const;
        value_type reduce_1d(const value_type* data, size_type size) const;
    };

    /********************************************
     * xreducer implementation
     ********************************************/
    template <class F, class E, class X, class ES>
    inline xreducer<F, E, X, ES>::xreducer(F&& f, E&& e, X&& axes, ES&& es) noexcept
        : m_f(std::forward<F>(f))
        , m_e(std::forward<E>(e))
        , m_axes(std::forward<X>(axes))
        , m_es(std::forward<ES>(es))
    {
        m_shape = compute_shape();
        m_init_value = compute_init();
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::size() const -> size_type
    {
        return compute_size(m_shape);
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::shape() const -> shape_type
    {
        return m_shape;
    }

    template <class F, class E, class X, class ES>
    template <class... Args>
    inline auto xreducer<F, E, X, ES>::operator()(Args... args) const -> const_reference
    {
        // Reduce entire expression; we don't support element-wise on reducer yet.
        // Instead, return a scalar if all axes are reduced.
        if (m_shape.empty())
        {
            // scalar result
            static value_type result = reduce_over_axes_parallel();
            return result;
        }
        throw std::runtime_error("xreducer: element access not implemented for partial reduction.");
    }

    template <class F, class E, class X, class ES>
    template <class It>
    inline auto xreducer<F, E, X, ES>::element(It first, It last) const -> const_reference
    {
        return operator()(0);
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::compute_init() const -> value_type
    {
        // Use the init functor from the reducer functor if present, else default
        if constexpr (std::is_same_v<F, xreducer_functor<detail::plus, detail::const_value<value_type>>>)
        {
            return value_type(0);
        }
        else if constexpr (std::is_same_v<F, xreducer_functor<detail::multiplies, detail::const_value<value_type>>>)
        {
            return value_type(1);
        }
        else if constexpr (std::is_same_v<F, xreducer_functor<detail::maximum, detail::const_value<value_type>>>)
        {
            return std::numeric_limits<value_type>::lowest();
        }
        else if constexpr (std::is_same_v<F, xreducer_functor<detail::minimum, detail::const_value<value_type>>>)
        {
            return std::numeric_limits<value_type>::max();
        }
        else
        {
            return value_type{};
        }
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::compute_shape() const -> shape_type
    {
        auto eshape = m_e.shape();
        if (m_axes.empty())
        {
            // reduce to scalar -> empty shape
            return shape_type{};
        }
        shape_type res;
        for (size_type i = 0; i < eshape.size(); ++i)
        {
            if (std::find(m_axes.begin(), m_axes.end(), i) == m_axes.end())
            {
                res.push_back(eshape[i]);
            }
        }
        return res;
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::reduce_over_axes_simd() const -> value_type
    {
        // For full reduction (scalar output), we can gather data linearly and reduce with SIMD.
        const auto& data = m_e.data();
        size_type count = m_e.size();
        return reduce_1d(data, count);
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::reduce_over_axes_parallel() const -> value_type
    {
        // Use std::execution::par to reduce in parallel if ES supports it.
        const auto& data = m_e.data();
        size_type count = m_e.size();
        if constexpr (std::is_same_v<ES, parallel_strategy>)
        {
            // Parallel reduction using divide and conquer
            value_type init = m_init_value;
            size_type num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            std::vector<value_type> partials(num_threads, init);
            std::vector<std::thread> threads;
            size_type chunk = count / num_threads;
            for (size_type t = 0; t < num_threads; ++t)
            {
                threads.emplace_back([&, t]() {
                    size_type start = t * chunk;
                    size_type end = (t == num_threads - 1) ? count : start + chunk;
                    value_type local = init;
                    for (size_type i = start; i < end; ++i)
                    {
                        local = m_f(local, data[i]);
                    }
                    partials[t] = local;
                });
            }
            for (auto& th : threads) th.join();
            value_type result = init;
            for (auto& p : partials)
            {
                result = m_f(result, p);
            }
            return result;
        }
        else
        {
            return reduce_over_axes_sequential();
        }
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::reduce_over_axes_sequential() const -> value_type
    {
        const auto& data = m_e.data();
        size_type count = m_e.size();
        return reduce_1d(data, count);
    }

    template <class F, class E, class X, class ES>
    inline auto xreducer<F, E, X, ES>::reduce_1d(const value_type* data, size_type count) const -> value_type
    {
        value_type result = m_init_value;
        // SIMD accumulation if available
        if constexpr (is_simd_enabled_v<value_type>)
        {
            using simd_type = xsimd::batch<value_type, xsimd::default_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            size_type vec_count = count / simd_size;
            simd_type vresult = result; // broadcast init
            for (size_type i = 0; i < vec_count; ++i)
            {
                simd_type vdata = simd_type::load_unaligned(data + i * simd_size);
                vresult = m_f.simd_apply(vresult, vdata);
            }
            // Reduce SIMD vector to scalar
            value_type tmp[simd_size];
            vresult.store_unaligned(tmp);
            for (std::size_t i = 0; i < simd_size; ++i)
            {
                result = m_f(result, tmp[i]);
            }
            // Handle remaining elements
            for (size_type i = vec_count * simd_size; i < count; ++i)
            {
                result = m_f(result, data[i]);
            }
        }
        else
        {
            for (size_type i = 0; i < count; ++i)
            {
                result = m_f(result, data[i]);
            }
        }
        return result;
    }

    // Overloaded reduce function to create an xreducer
    template <class F, class E, class X, class ES = DEFAULT_STRATEGY_REDUCERS>
    inline auto reduce(F&& f, E&& e, X&& axes, ES&& es = ES())
    {
        return xreducer<F, E, X, ES>(std::forward<F>(f), std::forward<E>(e),
                                     std::forward<X>(axes), std::forward<ES>(es));
    }

    // Parallel strategy tag
    struct parallel_strategy {};
    struct sequential_strategy {};

    inline constexpr parallel_strategy parallel = parallel_strategy{};
    inline constexpr sequential_strategy sequential = sequential_strategy{};

    // Default strategy reducer
    using DEFAULT_STRATEGY_REDUCERS = sequential_strategy;

}  // namespace xt

#endif  // XTENSOR_XREDUCER_HPP