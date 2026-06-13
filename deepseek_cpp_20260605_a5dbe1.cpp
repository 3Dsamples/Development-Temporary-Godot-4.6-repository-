//File 0009 : core/xaccumulator.hpp
//Accumulator operations (cumsum, cumprod, cummax, cummin) with SIMD prefix scans, parallel chunking, and full expression integration.
#ifndef XTENSOR_XACCUMULATOR_HPP
#define XTENSOR_XACCUMULATOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <functional>
#include <numeric>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xreducer.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    /********************************************
     * accumulator policies (SIMD-enabled)
     ********************************************/
    namespace detail
    {
        template <class F>
        struct accumulator_functor
        {
            F m_f;
            using simd_value_type = xsimd::batch<double, 4>; // placeholder for generic SIMD type

            template <class T>
            auto operator()(const T& acc, const T& val) const
            {
                return m_f(acc, val);
            }

            template <class T>
            auto simd_apply(const T& acc, const T& val) const
            {
                return m_f.simd_apply(acc, val);
            }
        };

        struct plus_accum : accumulator_functor<plus>
        {
            plus_accum() : accumulator_functor<plus>{plus{}} {}
        };
        struct multiplies_accum : accumulator_functor<multiplies>
        {
            multiplies_accum() : accumulator_functor<multiplies>{multiplies{}} {}
        };
        struct maximum_accum : accumulator_functor<maximum>
        {
            maximum_accum() : accumulator_functor<maximum>{maximum{}} {}
        };
        struct minimum_accum : accumulator_functor<minimum>
        {
            minimum_accum() : accumulator_functor<minimum>{minimum{}} {}
        };
    }

    /********************************************
     * Sequential inclusive scan with SIMD
     ********************************************/
    namespace detail
    {
        template <class InputIt, class OutputIt, class T, class BinaryOp>
        inline void simd_inclusive_scan(InputIt first, InputIt last, OutputIt d_first,
                                         T init, BinaryOp binary_op)
        {
            using value_type = T;
            std::size_t n = static_cast<std::size_t>(std::distance(first, last));
            if (n == 0) return;
            // SIMD chunks if available
            if constexpr (is_simd_enabled_v<value_type>)
            {
                using simd_type = xsimd::batch<value_type, xsimd::default_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                if (n >= simd_size)
                {
                    simd_type vacc = init;
                    std::size_t i = 0;
                    // Process full SIMD registers, performing inter-register scan manually
                    for (; i + simd_size <= n; i += simd_size)
                    {
                        simd_type chunk = simd_type::load_unaligned(first + i);
                        // To perform inclusive scan within a register we use a shuffle-based approach
                        simd_type shifted = chunk;
                        // shift and accumulate within the register
                        for (std::size_t s = 1; s < simd_size; s <<= 1)
                        {
                            simd_type tmp = xsimd::shift_left(shifted, s);
                            shifted = binary_op.simd_apply(shifted, tmp);
                        }
                        // Add previous accumulator to all elements
                        simd_type result = binary_op.simd_apply(vacc, shifted);
                        result.store_unaligned(d_first + i);
                        // Update vacc with the last element of the chunk
                        value_type last_arr[simd_size];
                        result.store_unaligned(last_arr);
                        vacc = last_arr[simd_size - 1];
                    }
                    // handle tail elements with scalar scan
                    value_type carry = vacc;
                    for (; i < n; ++i)
                    {
                        carry = binary_op(carry, first[i]);
                        d_first[i] = carry;
                    }
                    return;
                }
            }
            // Scalar fallback
            value_type acc = init;
            for (std::size_t i = 0; i < n; ++i)
            {
                acc = binary_op(acc, first[i]);
                d_first[i] = acc;
            }
        }

        // Parallel inclusive scan using divide-and-conquer with two-pass (up-sweep, down-sweep)
        template <class RandomIt, class OutputIt, class T, class BinaryOp>
        inline void parallel_inclusive_scan(RandomIt first, RandomIt last, OutputIt d_first,
                                             T init, BinaryOp binary_op)
        {
            std::size_t n = static_cast<std::size_t>(std::distance(first, last));
            if (n == 0) return;
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            if (n < num_threads * 1024)
            {
                simd_inclusive_scan(first, last, d_first, init, binary_op);
                return;
            }
            std::size_t chunk = (n + num_threads - 1) / num_threads;
            std::vector<T> partial_sums(num_threads, init);
            std::vector<std::thread> threads;
            // Phase 1: compute chunk partial sums
            for (unsigned int t = 0; t < num_threads; ++t)
            {
                threads.emplace_back([&, t]() {
                    std::size_t start = t * chunk;
                    std::size_t end = std::min(start + chunk, n);
                    T acc = init;
                    for (std::size_t i = start; i < end; ++i)
                        acc = binary_op(acc, first[i]);
                    partial_sums[t] = acc;
                });
            }
            for (auto& th : threads) th.join();

            // Phase 2: prefix sum over partial_sums (sequential, small)
            std::vector<T> prefix(num_threads, init);
            prefix[0] = partial_sums[0];
            for (unsigned int t = 1; t < num_threads; ++t)
                prefix[t] = binary_op(prefix[t - 1], partial_sums[t]);

            // Phase 3: combine: each thread applies offset = prefix[t-1] (if t>0) else init
            threads.clear();
            for (unsigned int t = 0; t < num_threads; ++t)
            {
                threads.emplace_back([&, t]() {
                    std::size_t start = t * chunk;
                    std::size_t end = std::min(start + chunk, n);
                    T offset = (t == 0) ? init : prefix[t - 1];
                    T acc = offset;
                    for (std::size_t i = start; i < end; ++i)
                    {
                        acc = binary_op(acc, first[i]);
                        d_first[i] = acc;
                    }
                });
            }
            for (auto& th : threads) th.join();
        }
    }

    /********************************************
     * xaccumulator expression node
     ********************************************/
    template <class F, class E>
    class xaccumulator;

    template <class F, class E>
    struct xcontainer_inner_types<xaccumulator<F, E>>
    {
        using value_type = typename std::decay_t<E>::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename std::decay_t<E>::shape_type;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    template <class F, class E>
    class xaccumulator : public xexpression<xaccumulator<F, E>>
    {
    public:
        using self_type = xaccumulator<F, E>;
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
        using expression_type = E;

        xaccumulator(F&& f, E&& e, value_type init)
            : m_f(std::forward<F>(f)), m_e(std::forward<E>(e)), m_init(init)
        {
            evaluate();
        }

        size_type size() const { return m_result.size(); }
        shape_type shape() const { return m_e.shape(); }

        template <class... Args>
        const_reference operator()(Args... args) const { return m_result(args...); }
        template <class It>
        const_reference element(It first, It last) const { return m_result.element(first, last); }

        const temporary_type& result() const { return m_result; }

    private:
        F m_f;
        E m_e;
        value_type m_init;
        temporary_type m_result;

        void evaluate()
        {
            m_result.resize(m_e.shape(), m_init);
            const auto& e_data = m_e.data();
            auto* res_data = m_result.data();
            detail::parallel_inclusive_scan(e_data, e_data + m_e.size(), res_data, m_init, m_f);
        }
    };

    /********************************************
     * Free functions: cumsum, cumprod, etc.
     ********************************************/
    namespace accumulator
    {
        template <class E>
        inline auto cumsum(E&& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator<detail::plus_accum, E>(detail::plus_accum{},
                                                        std::forward<E>(e), value_type(0));
        }

        template <class E>
        inline auto cumprod(E&& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator<detail::multiplies_accum, E>(detail::multiplies_accum{},
                                                              std::forward<E>(e), value_type(1));
        }

        template <class E>
        inline auto cummax(E&& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator<detail::maximum_accum, E>(detail::maximum_accum{},
                                                           std::forward<E>(e),
                                                           std::numeric_limits<value_type>::lowest());
        }

        template <class E>
        inline auto cummin(E&& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            return xaccumulator<detail::minimum_accum, E>(detail::minimum_accum{},
                                                           std::forward<E>(e),
                                                           std::numeric_limits<value_type>::max());
        }

        // Optional axis-specific cum operations can be added here as needed.
    }

}  // namespace xt

#endif  // XTENSOR_XACCUMULATOR_HPP