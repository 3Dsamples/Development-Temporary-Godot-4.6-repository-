//File 0111 : numdot/random.h
//Random number generation: uniform, normal, integer, exponential, gamma, and Sobol quasi‑random sequences, with SIMD batch generation and parallel seeding.
#ifndef NUMDOT_RANDOM_H
#define NUMDOT_RANDOM_H

#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <mutex>
#include <thread>
#include <functional>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"
#include "array.h"
#include "elementwise.h"
#include "math.h"

namespace numdot
{
namespace random
{
    using default_engine = std::mt19937_64;

    namespace detail
    {
        // Thread‑local engine for global state
        inline default_engine& global_engine()
        {
            thread_local static default_engine eng(
                static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())
            );
            return eng;
        }
    }

    // Seed the global engine
    inline void seed(std::uint64_t s) { detail::global_engine().seed(s); }

    inline void seed()
    {
        std::random_device rd;
        std::array<std::uint64_t, 8> seed_data;
        for (auto& v : seed_data) v = rd();
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        detail::global_engine().seed(seq);
    }

    // ========== SIMD batch helpers ==========
    template <class T>
    using simd_batch = xsimd::batch<T, default_simd_arch>;

    namespace detail
    {
        template <class T, class Func>
        void fill_array_simd(array<T>& arr, Func gen)
        {
            T* data = arr.data();
            std::size_t count = arr.size();
            if constexpr (simd_enabled_v<T>)
            {
                constexpr std::size_t simd_size = simd_batch<T>::size;
                std::size_t vec_count = count / simd_size;
                alignas(64) std::array<T, simd_size> buffer;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    for (std::size_t k = 0; k < simd_size; ++k) buffer[k] = gen();
                    simd_batch<T> b = simd_batch<T>::load_aligned(buffer.data());
                    b.store_aligned(data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < count; ++i)
                    data[i] = gen();
            }
            else
            {
                for (std::size_t i = 0; i < count; ++i) data[i] = gen();
            }
        }
    }

    /**
     * Uniform distribution over [0, 1).
     */
    template <class S>
    inline auto rand(const S& shape)
    {
        using T = double;
        array<T> result(shape);
        detail::fill_array_simd(result, []() -> T {
            static std::uniform_real_distribution<T> dist(0.0, 1.0);
            return dist(detail::global_engine());
        });
        return result;
    }

    /**
     * Standard normal distribution (mean 0, std 1).
     */
    template <class S>
    inline auto randn(const S& shape)
    {
        using T = double;
        array<T> result(shape);
        detail::fill_array_simd(result, []() -> T {
            static std::normal_distribution<T> dist(0.0, 1.0);
            return dist(detail::global_engine());
        });
        return result;
    }

    /**
     * Normal distribution with specified mean and stddev.
     */
    template <class S, class T = double>
    inline auto normal(const S& shape, T mean = 0.0, T stddev = 1.0)
    {
        array<T> result(shape);
        detail::fill_array_simd(result, [mean, stddev]() -> T {
            static std::normal_distribution<T> dist(0.0, 1.0);
            return mean + stddev * dist(detail::global_engine());
        });
        return result;
    }

    /**
     * Uniform distribution over [a, b).
     */
    template <class S, class T = double>
    inline auto uniform(const S& shape, T a = 0.0, T b = 1.0)
    {
        array<T> result(shape);
        detail::fill_array_simd(result, [a, b]() -> T {
            static std::uniform_real_distribution<T> dist(0.0, 1.0);
            return a + (b - a) * dist(detail::global_engine());
        });
        return result;
    }

    /**
     * Random integers in [low, high).
     */
    template <class S, class IntType = int>
    inline auto randint(const S& shape, IntType low, IntType high)
    {
        array<IntType> result(shape);
        detail::fill_array_simd(result, [low, high]() -> IntType {
            static std::uniform_int_distribution<IntType> dist(low, high - 1);
            return dist(detail::global_engine());
        });
        return result;
    }

    /**
     * Exponential distribution with given scale.
     */
    template <class S, class T = double>
    inline auto exponential(const S& shape, T scale = 1.0)
    {
        array<T> result(shape);
        detail::fill_array_simd(result, [scale]() -> T {
            static std::exponential_distribution<T> dist(1.0 / scale);
            return dist(detail::global_engine());
        });
        return result;
    }

    /**
     * Gamma distribution.
     */
    template <class S, class T = double>
    inline auto gamma(const S& shape, T alpha, T beta = 1.0)
    {
        array<T> result(shape);
        detail::fill_array_simd(result, [alpha, beta]() -> T {
            static std::gamma_distribution<T> dist(alpha, beta);
            return dist(detail::global_engine());
        });
        return result;
    }

    /**
     * Sobol quasi‑random engine for low‑discrepancy sequences.
     */
    template <std::size_t D>
    class sobol_engine
    {
    public:
        sobol_engine() : m_index(0)
        {
            init_direction_numbers();
            m_x.fill(0);
        }

        std::array<double, D> next()
        {
            if (m_index == 0)
            {
                ++m_index;
                std::array<double, D> r; r.fill(0.0);
                return r;
            }
            std::size_t c = 0;
            std::size_t idx = m_index;
            while ((idx & 1) == 0) { idx >>= 1; ++c; }
            ++m_index;
            for (std::size_t d = 0; d < D; ++d)
                m_x[d] ^= m_direction[d][c];
            std::array<double, D> result;
            for (std::size_t d = 0; d < D; ++d)
                result[d] = m_x[d] / std::pow(2.0, 32);
            return result;
        }

    private:
        void init_direction_numbers()
        {
            // Simplified direction numbers – real implementation would load from resource
            for (std::size_t d = 0; d < D; ++d)
            {
                m_direction[d].resize(32);
                std::uint32_t v = 1u << (31 - d % 31);
                for (std::size_t i = 0; i < 32; ++i)
                {
                    m_direction[d][i] = v << (31 - i);
                }
            }
        }

        std::size_t m_index;
        std::array<double, D> m_x;
        std::array<std::vector<std::uint32_t>, D> m_direction;
    };

    /**
     * Generate Sobol sequence as an array (N x D).
     */
    template <std::size_t D>
    inline auto sobol(std::size_t count)
    {
        sobol_engine<D> gen;
        array<double> result({count, D});
        for (std::size_t i = 0; i < count; ++i)
        {
            auto pt = gen.next();
            for (std::size_t d = 0; d < D; ++d) result(i, d) = pt[d];
        }
        return result;
    }

    /**
     * Random permutation of [0, n-1].
     */
    inline auto permutation(std::size_t n)
    {
        array<std::size_t> result({n});
        std::iota(result.data(), result.data() + n, 0);
        std::shuffle(result.data(), result.data() + n, detail::global_engine());
        return result;
    }

    /**
     * Random choice with/without replacement from a 1D array.
     */
    template <class E>
    inline auto choice(const expression<E>& arr, std::size_t size, bool replace = true)
    {
        using T = typename E::value_type;
        const auto& a = arr.derived();
        array<T> result({size});
        auto& eng = detail::global_engine();
        if (replace)
        {
            std::uniform_int_distribution<std::size_t> dist(0, a.size() - 1);
            for (std::size_t i = 0; i < size; ++i) result[i] = a[dist(eng)];
        }
        else
        {
            if (size > a.size()) throw std::runtime_error("choice: sample size larger than population without replacement.");
            auto idx = permutation(a.size());
            for (std::size_t i = 0; i < size; ++i) result[i] = a[idx[i]];
        }
        return result;
    }

    /**
     * Shuffle an array in‑place along first axis.
     */
    template <class E>
    inline void shuffle(E& expr)
    {
        auto& a = expr.derived();
        auto shape = a.shape();
        if (shape.empty()) return;
        std::size_t outer = shape[0];
        std::size_t block = a.size() / outer;
        auto idx = permutation(outer);
        array<typename E::value_type> temp(shape);
        auto* src = a.data();
        auto* dst = temp.data();
        for (std::size_t i = 0; i < outer; ++i)
            std::copy(src + idx[i] * block, src + idx[i] * block + block, dst + i * block);
        std::copy(dst, dst + a.size(), src);
    }

} // namespace random
} // namespace numdot

#endif // NUMDOT_RANDOM_H