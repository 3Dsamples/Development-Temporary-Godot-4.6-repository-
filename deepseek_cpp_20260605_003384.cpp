//File 0014 : core/xrandom.hpp
//Random number generation with SIMD-accelerated distributions, parallel seeding, and array generation for simulations.
#ifndef XTENSOR_XRANDOM_HPP
#define XTENSOR_XRANDOM_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <execution>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xreducer.hpp"
#include "xaccumulator.hpp"
#include "xeval.hpp"
#include "xmanipulation.hpp"
#include "xio.hpp"
#include "xsort.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt {
namespace random {

    // Default random engine: 64-bit Mersenne Twister
    using default_engine_type = std::mt19937_64;

    // Seeder: thread_local engine for global state
    namespace detail {
        inline default_engine_type& get_global_engine() {
            thread_local static default_engine_type engine(
                static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())
            );
            return engine;
        }
    }

    // Seed the global engine with a specific value
    inline void seed(uint64_t s) {
        detail::get_global_engine().seed(s);
    }

    // Seed using entropy source
    inline void seed() {
        std::random_device rd;
        std::array<uint64_t, 8> seed_data;
        for (auto& v : seed_data) v = rd();
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        detail::get_global_engine().seed(seq);
    }

    // SIMD batch type for random operations
    template <class T>
    using simd_batch = xsimd::batch<T, xsimd::default_arch>;

    // Generate multiple uniform random numbers into a buffer using SIMD box-muller or direct.
    namespace detail {
        // Fill a range with uniform [0,1) doubles from an engine
        template <class Engine, class OutputIt>
        void fill_uniform(Engine& eng, OutputIt first, std::size_t count) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (std::size_t i = 0; i < count; ++i) {
                *first++ = dist(eng);
            }
        }

        // Fill a buffer of doubles with standard normal random numbers using Box-Muller SIMD style.
        // We generate pairs of uniform and convert to normal via Box-Muller, storing in SIMD batches.
        template <class Engine, class OutputIt>
        void fill_normal(Engine& eng, OutputIt first, std::size_t count, double mean = 0.0, double stddev = 1.0) {
            // Use Box-Muller: z0 = sqrt(-2 ln U1) cos(2 pi U2), z1 = sqrt(-2 ln U1) sin(2 pi U2)
            // We'll process pairs and pack into SIMD when count is large.
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            const double two_pi = 2.0 * xt::numeric_constants<double>::PI;
            auto out = first;
            for (std::size_t i = 0; i < count / 2; ++i) {
                double u1 = dist(eng);
                double u2 = dist(eng);
                double r = std::sqrt(-2.0 * std::log(u1));
                double z0 = r * std::cos(two_pi * u2);
                double z1 = r * std::sin(two_pi * u2);
                if constexpr (xt::is_simd_enabled_v<double>) {
                    // Store in SIMD batch of size 2? We'll just store scalar and let later SIMD fill handle it.
                    *out++ = mean + stddev * z0;
                    *out++ = mean + stddev * z1;
                } else {
                    *out++ = mean + stddev * z0;
                    *out++ = mean + stddev * z1;
                }
            }
            // Handle odd element
            if (count % 2 != 0) {
                double u1 = dist(eng);
                double u2 = dist(eng);
                double r = std::sqrt(-2.0 * std::log(u1));
                *out++ = mean + stddev * (r * std::cos(two_pi * u2));
            }
        }

        // Fill a 1D array with values from a callback using SIMD chunking
        template <class T, class Func>
        void fill_array_simd(xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>& arr, Func gen) {
            T* data = arr.data();
            std::size_t count = arr.size();
            if constexpr (is_simd_enabled_v<T>) {
                using batch_type = simd_batch<T>;
                constexpr std::size_t simd_size = batch_type::size;
                // Process in SIMD chunks: we need to generate a batch of random numbers.
                // Since gen() returns a scalar, we can generate simd_size scalars and load into batch.
                alignas(64) std::array<T, simd_size> buffer;
                std::size_t i = 0;
                for (; i + simd_size <= count; i += simd_size) {
                    for (std::size_t j = 0; j < simd_size; ++j) {
                        buffer[j] = gen();
                    }
                    batch_type b = batch_type::load_aligned(buffer.data());
                    b.store_aligned(data + i);
                }
                for (; i < count; ++i) {
                    data[i] = gen();
                }
            } else {
                std::generate(data, data + count, gen);
            }
        }
    } // namespace detail

    /********************************************
     * xrandom_engine class
     ********************************************/
    class xrandom_engine {
    public:
        using engine_type = default_engine_type;

        xrandom_engine() : m_engine(std::random_device{}()) {}
        explicit xrandom_engine(uint64_t s) : m_engine(s) {}

        void seed(uint64_t s) { m_engine.seed(s); }
        engine_type& engine() { return m_engine; }

        // Generate a uniform random number in [0,1)
        double uniform() {
            return std::uniform_real_distribution<double>(0.0, 1.0)(m_engine);
        }

        // Generate a standard normal random number
        double normal() {
            return std::normal_distribution<double>(0.0, 1.0)(m_engine);
        }

        // SIMD batch of uniform [0,1) doubles
        simd_batch<double> uniform_batch() {
            std::array<double, simd_batch<double>::size> arr;
            detail::fill_uniform(m_engine, arr.data(), arr.size());
            return simd_batch<double>::load_unaligned(arr.data());
        }

        // SIMD batch of standard normal doubles via Box-Muller
        simd_batch<double> normal_batch() {
            std::array<double, simd_batch<double>::size> arr;
            detail::fill_normal(m_engine, arr.data(), arr.size(), 0.0, 1.0);
            return simd_batch<double>::load_unaligned(arr.data());
        }

    private:
        engine_type m_engine;
    };

    // Global engine accessor
    inline xrandom_engine& global_engine() {
        static xrandom_engine instance;
        return instance;
    }

    /********************************************
     * rand - uniform random numbers in [0,1)
     ********************************************/
    template <class S>
    inline auto rand(const S& shape) {
        using T = double;
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        detail::fill_array_simd(result, []() -> T {
            return std::uniform_real_distribution<T>(0.0, 1.0)(detail::get_global_engine());
        });
        return result;
    }

    /********************************************
     * randint - random integers in [low, high)
     ********************************************/
    template <class S, class IntType = int>
    inline auto randint(const S& shape, IntType low, IntType high) {
        xarray_container<xt::uvector<IntType>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        std::uniform_int_distribution<IntType> dist(low, high - 1);
        detail::fill_array_simd(result, [&dist]() mutable -> IntType {
            return dist(detail::get_global_engine());
        });
        return result;
    }

    /********************************************
     * randn - standard normal (mean=0, std=1)
     ********************************************/
    template <class S>
    inline auto randn(const S& shape) {
        using T = double;
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::normal_distribution<T> dist(0.0, 1.0);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    /********************************************
     * normal - normal with given mean and stddev
     ********************************************/
    template <class S, class T = double>
    inline auto normal(const S& shape, T mean = 0.0, T stddev = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::normal_distribution<T> dist(mean, stddev);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    /********************************************
     * uniform - uniform [a,b)
     ********************************************/
    template <class S, class T = double>
    inline auto uniform(const S& shape, T a = 0.0, T b = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::uniform_real_distribution<T> dist(a, b);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    /********************************************
     * exponential - exponential with given scale (1/lambda)
     ********************************************/
    template <class S, class T = double>
    inline auto exponential(const S& shape, T scale = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::exponential_distribution<T> dist(1.0 / scale);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    /********************************************
     * gamma - gamma distribution
     ********************************************/
    template <class S, class T = double>
    inline auto gamma(const S& shape, T alpha, T beta = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::gamma_distribution<T> dist(alpha, beta);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    /********************************************
     * binomial
     ********************************************/
    template <class S, class IntType = int>
    inline auto binomial(const S& shape, IntType n, double p) {
        xarray_container<xt::uvector<IntType>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::binomial_distribution<IntType> dist(n, p);
        detail::fill_array_simd(result, [&]() -> IntType { return dist(eng); });
        return result;
    }

    /********************************************
     * poisson
     ********************************************/
    template <class S, class IntType = int>
    inline auto poisson(const S& shape, double mean) {
        xarray_container<xt::uvector<IntType>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::poisson_distribution<IntType> dist(mean);
        detail::fill_array_simd(result, [&]() -> IntType { return dist(eng); });
        return result;
    }

    /********************************************
     * choice - random sample with/without replacement from a 1D array
     ********************************************/
    template <class E>
    inline auto choice(const E& arr, std::size_t size, bool replace = true) {
        using T = typename std::decay_t<E>::value_type;
        const auto& data = arr.data();
        std::size_t n = arr.size();
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({size});
        auto& eng = detail::get_global_engine();
        if (replace) {
            std::uniform_int_distribution<std::size_t> idx_dist(0, n - 1);
            for (std::size_t i = 0; i < size; ++i) {
                result[i] = data[idx_dist(eng)];
            }
        } else {
            if (size > n) throw std::runtime_error("choice: sample size larger than population without replacement.");
            std::vector<std::size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), eng);
            for (std::size_t i = 0; i < size; ++i) {
                result[i] = data[indices[i]];
            }
        }
        return result;
    }

    /********************************************
     * shuffle - in-place shuffle of array along first axis
     ********************************************/
    template <class E>
    inline void shuffle(E&& expr) {
        auto& arr = expr.derived_cast();
        auto shape = arr.shape();
        if (shape.empty()) return;
        std::size_t outer = shape[0];
        std::size_t block = arr.size() / outer;
        std::vector<std::size_t> indices(outer);
        std::iota(indices.begin(), indices.end(), 0);
        auto& eng = detail::get_global_engine();
        std::shuffle(indices.begin(), indices.end(), eng);
        // Permute blocks
        using T = typename std::decay_t<E>::value_type;
        std::vector<T> temp(arr.size());
        std::copy(arr.data(), arr.data() + arr.size(), temp.begin());
        for (std::size_t i = 0; i < outer; ++i) {
            std::size_t src_start = indices[i] * block;
            std::size_t dst_start = i * block;
            std::copy(temp.begin() + src_start, temp.begin() + src_start + block,
                      arr.data() + dst_start);
        }
    }

    /********************************************
     * permutation - random permutation of [0, n-1]
     ********************************************/
    inline auto permutation(std::size_t n) {
        std::vector<std::size_t> v(n);
        std::iota(v.begin(), v.end(), 0);
        auto& eng = detail::get_global_engine();
        std::shuffle(v.begin(), v.end(), eng);
        xarray_container<xt::uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        std::copy(v.begin(), v.end(), result.data());
        return result;
    }

    /********************************************
     * Random expression placeholder (for advanced usage)
     ********************************************/
    template <class T>
    class random_expression {
    public:
        using value_type = T;
        using engine_type = default_engine_type;

        random_expression() = default;

        // Generate an xarray with given shape using a callable
        template <class S, class Func>
        auto generate(const S& shape, Func f) {
            xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
            detail::fill_array_simd(result, f);
            return result;
        }
    };

} // namespace random
} // namespace xt

#endif // XTENSOR_XRANDOM_HPP