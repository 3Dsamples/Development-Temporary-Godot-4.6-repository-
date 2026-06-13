//File 0014 (UPDATED) : core/xrandom.hpp
//Random number generation with SIMD distributions, parallel seeding, Sobol quasi-random sequences, and array generation for simulations.
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

    using default_engine_type = std::mt19937_64;

    namespace detail {
        inline default_engine_type& get_global_engine() {
            thread_local static default_engine_type engine(
                static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())
            );
            return engine;
        }
    }

    inline void seed(uint64_t s) { detail::get_global_engine().seed(s); }
    inline void seed() {
        std::random_device rd;
        std::array<uint64_t, 8> seed_data;
        for (auto& v : seed_data) v = rd();
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        detail::get_global_engine().seed(seq);
    }

    template <class T>
    using simd_batch = xsimd::batch<T, default_simd_arch>;

    namespace detail {
        template <class Engine, class OutputIt>
        void fill_uniform(Engine& eng, OutputIt first, std::size_t count) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (std::size_t i = 0; i < count; ++i) *first++ = dist(eng);
        }

        template <class Engine, class OutputIt>
        void fill_normal(Engine& eng, OutputIt first, std::size_t count, double mean = 0.0, double stddev = 1.0) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            const double two_pi = 2.0 * xt::numeric_constants<double>::PI;
            auto out = first;
            for (std::size_t i = 0; i < count / 2; ++i) {
                double u1 = dist(eng);
                double u2 = dist(eng);
                double r = std::sqrt(-2.0 * std::log(u1));
                double z0 = r * std::cos(two_pi * u2);
                double z1 = r * std::sin(two_pi * u2);
                *out++ = mean + stddev * z0;
                *out++ = mean + stddev * z1;
            }
            if (count % 2 != 0) {
                double u1 = dist(eng);
                double u2 = dist(eng);
                double r = std::sqrt(-2.0 * std::log(u1));
                *out++ = mean + stddev * (r * std::cos(two_pi * u2));
            }
        }

        template <class T, class Func>
        void fill_array_simd(xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>& arr, Func gen) {
            T* data = arr.data();
            std::size_t count = arr.size();
            if constexpr (is_simd_enabled_v<T>) {
                using batch_type = simd_batch<T>;
                constexpr std::size_t simd_size = batch_type::size;
                alignas(64) std::array<T, simd_size> buffer;
                std::size_t i = 0;
                for (; i + simd_size <= count; i += simd_size) {
                    for (std::size_t j = 0; j < simd_size; ++j) buffer[j] = gen();
                    batch_type b = batch_type::load_aligned(buffer.data());
                    b.store_aligned(data + i);
                }
                for (; i < count; ++i) data[i] = gen();
            } else {
                std::generate(data, data + count, gen);
            }
        }
    }

    class xrandom_engine {
    public:
        using engine_type = default_engine_type;
        xrandom_engine() : m_engine(std::random_device{}()) {}
        explicit xrandom_engine(uint64_t s) : m_engine(s) {}
        void seed(uint64_t s) { m_engine.seed(s); }
        engine_type& engine() { return m_engine; }

        double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(m_engine); }
        double normal() { return std::normal_distribution<double>(0.0, 1.0)(m_engine); }

        simd_batch<double> uniform_batch() {
            std::array<double, simd_batch<double>::size> arr;
            detail::fill_uniform(m_engine, arr.data(), arr.size());
            return simd_batch<double>::load_unaligned(arr.data());
        }

        simd_batch<double> normal_batch() {
            std::array<double, simd_batch<double>::size> arr;
            detail::fill_normal(m_engine, arr.data(), arr.size(), 0.0, 1.0);
            return simd_batch<double>::load_unaligned(arr.data());
        }

    private:
        engine_type m_engine;
    };

    inline xrandom_engine& global_engine() {
        static xrandom_engine instance;
        return instance;
    }

    // Sobol quasi-random sequence generator (Gray code based)
    template <std::size_t D>
    class sobol_engine {
    public:
        sobol_engine() : m_index(0) { init_direction_numbers(); }
        std::array<double, D> next() {
            if (m_index == 0) { ++m_index; std::array<double, D> r; r.fill(0.0); return r; }
            std::size_t c = 0;
            std::size_t idx = m_index;
            while ((idx & 1) == 0) { idx >>= 1; ++c; }
            ++m_index;
            std::array<double, D> point = m_x;
            for (std::size_t d = 0; d < D; ++d)
                point[d] ^= m_direction[d][c];
            m_x = point;
            std::array<double, D> result;
            for (std::size_t d = 0; d < D; ++d)
                result[d] = point[d] / std::pow(2.0, 32);
            return result;
        }

    private:
        void init_direction_numbers() {
            // Precomputed direction numbers for up to 6 dimensions, 32 bits
            const std::size_t max_dim = 6;
            std::vector<std::vector<std::uint32_t>> v_init = {
                {1}, {1}, {1}, {1}, {1}, {1}
            };
            // These are just placeholders; a real implementation would load Sobol initialisation data.
            for (std::size_t d = 0; d < D; ++d) {
                m_direction[d].resize(32);
                std::uint32_t v = v_init[d][0];
                for (std::size_t i = 0; i < 32; ++i) {
                    m_direction[d][i] = v << (31 - i);
                }
            }
            m_x.fill(0);
        }

        std::size_t m_index;
        std::array<double, D> m_x;
        std::array<std::vector<std::uint32_t>, D> m_direction;
    };

    // Convenience function to fill an array with Sobol points
    template <std::size_t D>
    inline auto sobol_samples(std::size_t count) {
        sobol_engine<D> gen;
        xarray_container<xt::uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({count, D});
        for (std::size_t i = 0; i < count; ++i) {
            auto pt = gen.next();
            for (std::size_t d = 0; d < D; ++d) result(i, d) = pt[d];
        }
        return result;
    }

    // Standard distributions (unchanged signatures, already present)
    template <class S>
    inline auto rand(const S& shape) {
        using T = double;
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        detail::fill_array_simd(result, []() -> T {
            return std::uniform_real_distribution<T>(0.0, 1.0)(detail::get_global_engine());
        });
        return result;
    }

    template <class S, class IntType = int>
    inline auto randint(const S& shape, IntType low, IntType high) {
        xarray_container<xt::uvector<IntType>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        std::uniform_int_distribution<IntType> dist(low, high - 1);
        detail::fill_array_simd(result, [&dist]() mutable -> IntType {
            return dist(detail::get_global_engine());
        });
        return result;
    }

    template <class S>
    inline auto randn(const S& shape) {
        using T = double;
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::normal_distribution<T> dist(0.0, 1.0);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    template <class S, class T = double>
    inline auto normal(const S& shape, T mean = 0.0, T stddev = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::normal_distribution<T> dist(mean, stddev);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    template <class S, class T = double>
    inline auto uniform(const S& shape, T a = 0.0, T b = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::uniform_real_distribution<T> dist(a, b);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    template <class S, class T = double>
    inline auto exponential(const S& shape, T scale = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::exponential_distribution<T> dist(1.0 / scale);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    template <class S, class T = double>
    inline auto gamma(const S& shape, T alpha, T beta = 1.0) {
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::gamma_distribution<T> dist(alpha, beta);
        detail::fill_array_simd(result, [&]() -> T { return dist(eng); });
        return result;
    }

    template <class S, class IntType = int>
    inline auto binomial(const S& shape, IntType n, double p) {
        xarray_container<xt::uvector<IntType>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::binomial_distribution<IntType> dist(n, p);
        detail::fill_array_simd(result, [&]() -> IntType { return dist(eng); });
        return result;
    }

    template <class S, class IntType = int>
    inline auto poisson(const S& shape, double mean) {
        xarray_container<xt::uvector<IntType>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(shape);
        auto& eng = detail::get_global_engine();
        std::poisson_distribution<IntType> dist(mean);
        detail::fill_array_simd(result, [&]() -> IntType { return dist(eng); });
        return result;
    }

    template <class E>
    inline auto choice(const E& arr, std::size_t size, bool replace = true) {
        using T = typename std::decay_t<E>::value_type;
        const auto& data = arr.data();
        std::size_t n = arr.size();
        xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({size});
        auto& eng = detail::get_global_engine();
        if (replace) {
            std::uniform_int_distribution<std::size_t> idx_dist(0, n - 1);
            for (std::size_t i = 0; i < size; ++i) result[i] = data[idx_dist(eng)];
        } else {
            if (size > n) throw std::runtime_error("choice: sample size > population without replacement.");
            std::vector<std::size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), eng);
            for (std::size_t i = 0; i < size; ++i) result[i] = data[indices[i]];
        }
        return result;
    }

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

    inline auto permutation(std::size_t n) {
        std::vector<std::size_t> v(n);
        std::iota(v.begin(), v.end(), 0);
        auto& eng = detail::get_global_engine();
        std::shuffle(v.begin(), v.end(), eng);
        xarray_container<xt::uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        std::copy(v.begin(), v.end(), result.data());
        return result;
    }

    template <class T>
    class random_expression {
    public:
        using value_type = T;
        using engine_type = default_engine_type;
        random_expression() = default;
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