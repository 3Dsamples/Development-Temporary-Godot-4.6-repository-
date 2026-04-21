// core/xrandom.hpp
#ifndef XTENSOR_XRANDOM_HPP
#define XTENSOR_XRANDOM_HPP

// ----------------------------------------------------------------------------
// xrandom.hpp – Random number generation for xtensor arrays
// ----------------------------------------------------------------------------
// This header defines functions to generate random arrays with various
// distributions: uniform, normal, binomial, poisson, etc. It also provides
// a random engine wrapper and supports BigNumber random generation (by
// generating random limbs). All functions return xarray_container with
// the requested shape and value type.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>
#include <limits>
#include <cmath>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ========================================================================
    // Random engine management
    // ========================================================================
    using default_engine_type = std::mt19937_64;

    // Set the seed for the default random engine
    void random_seed(uint64_t seed);
    // Get a reference to the thread‑local default random engine
    default_engine_type& get_default_random_engine();
    // Replace the default random engine with a user‑provided one
    void set_default_random_engine(const default_engine_type& engine);

    // ========================================================================
    // Core distributions (templated, engine parameter defaults to thread‑local)
    // ========================================================================

    // Generate uniformly distributed floating‑point numbers in [low, high)
    template <class T = double, class E = default_engine_type>
    auto random_uniform(const shape_type& shape, T low = 0.0, T high = 1.0, E& engine = get_default_random_engine());

    // Generate uniformly distributed integers in [low, high]
    template <class T = int, class E = default_engine_type>
    auto random_integers(const shape_type& shape, T low, T high, E& engine = get_default_random_engine());

    // Generate normally (Gaussian) distributed numbers with given mean and stddev
    template <class T = double, class E = default_engine_type>
    auto random_normal(const shape_type& shape, T mean = 0.0, T stddev = 1.0, E& engine = get_default_random_engine());

    // Generate binomial distributed integers
    template <class T = int, class E = default_engine_type>
    auto random_binomial(const shape_type& shape, int trials, double prob, E& engine = get_default_random_engine());

    // Generate poisson distributed integers
    template <class T = int, class E = default_engine_type>
    auto random_poisson(const shape_type& shape, double mean, E& engine = get_default_random_engine());

    // Generate exponentially distributed numbers
    template <class T = double, class E = default_engine_type>
    auto random_exponential(const shape_type& shape, T lambda = 1.0, E& engine = get_default_random_engine());

    // Generate gamma distributed numbers
    template <class T = double, class E = default_engine_type>
    auto random_gamma(const shape_type& shape, T alpha, T beta = 1.0, E& engine = get_default_random_engine());

    // Generate chi‑squared distributed numbers
    template <class T = double, class E = default_engine_type>
    auto random_chisquare(const shape_type& shape, T degrees, E& engine = get_default_random_engine());

    // Generate Student's t‑distributed numbers
    template <class T = double, class E = default_engine_type>
    auto random_student_t(const shape_type& shape, T degrees, E& engine = get_default_random_engine());

    // Generate Weibull distributed numbers
    template <class T = double, class E = default_engine_type>
    auto random_weibull(const shape_type& shape, T a, T b, E& engine = get_default_random_engine());

    // ========================================================================
    // Bernoulli and sampling
    // ========================================================================

    // Generate a boolean mask with given success probability
    template <class E = default_engine_type>
    auto random_bernoulli(const shape_type& shape, double p = 0.5, E& engine = get_default_random_engine());

    // Random choice from a given array (with or without replacement)
    template <class E, class Eng = default_engine_type>
    auto random_choice(const xexpression<E>& a, size_type num_samples, bool replace = true, Eng& engine = get_default_random_engine());

    // Random permutation of integers 0..n-1
    template <class Eng = default_engine_type>
    auto random_permutation(size_type n, Eng& engine = get_default_random_engine());

    // Shuffle an array (returns a shuffled copy)
    template <class E, class Eng = default_engine_type>
    auto random_shuffle(const xexpression<E>& e, Eng& engine = get_default_random_engine());

    // ========================================================================
    // Convenience aliases (NumPy‑like)
    // ========================================================================

    // Uniform in [0,1) – alias for random_uniform
    template <class T = double, class E = default_engine_type>
    auto rand(const shape_type& shape, E& engine = get_default_random_engine());

    // Standard normal – alias for random_normal(0,1)
    template <class T = double, class E = default_engine_type>
    auto randn(const shape_type& shape, E& engine = get_default_random_engine());

    // Uniform integers – alias for random_integers
    template <class T = int, class E = default_engine_type>
    auto randi(const shape_type& shape, T low, T high, E& engine = get_default_random_engine());

    // ========================================================================
    // BigNumber specific generators (limb‑based)
    // ========================================================================

    // Generate random BigNumbers with up to `num_bits` random bits
    template <class E = default_engine_type>
    auto random_bignumber_bits(const shape_type& shape, size_t num_bits, E& engine = get_default_random_engine());

    // Generate random BigNumbers uniformly in [0, max_val)
    template <class E = default_engine_type>
    auto random_bignumber_lt(const shape_type& shape, const bignumber::BigNumber& max_val, E& engine = get_default_random_engine());

    // Generate random BigNumbers uniformly in [low, high]
    template <class E = default_engine_type>
    auto random_bignumber_range(const shape_type& shape, const bignumber::BigNumber& low, const bignumber::BigNumber& high, E& engine = get_default_random_engine());

    // Approximate normal distribution for BigNumber (Box‑Muller)
    template <class E = default_engine_type>
    auto random_bignumber_normal(const shape_type& shape, const bignumber::BigNumber& mean, const bignumber::BigNumber& stddev, E& engine = get_default_random_engine());

    // Approximate Poisson distribution for BigNumber (Knuth's method)
    template <class E = default_engine_type>
    auto random_bignumber_poisson(const shape_type& shape, const bignumber::BigNumber& lambda, E& engine = get_default_random_engine());

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    // Set the seed for the default random engine
    inline void random_seed(uint64_t seed) { get_default_random_engine().seed(seed); }

    // Get a reference to the thread‑local default random engine
    inline default_engine_type& get_default_random_engine()
    { static thread_local default_engine_type engine(std::chrono::steady_clock::now().time_since_epoch().count()); return engine; }

    // Replace the default random engine with a user‑provided one
    inline void set_default_random_engine(const default_engine_type& engine) { get_default_random_engine() = engine; }

    // Generate uniformly distributed floating‑point numbers in [low, high)
    template <class T, class E>
    auto random_uniform(const shape_type& shape, T low, T high, E& engine)
    { static_assert(std::is_floating_point_v<T>, "Uniform requires floating point"); xarray_container<T> result(shape); std::uniform_real_distribution<T> dist(low, high); for (auto& v : result) v = dist(engine); return result; }

    // Generate uniformly distributed integers in [low, high]
    template <class T, class E>
    auto random_integers(const shape_type& shape, T low, T high, E& engine)
    { static_assert(std::is_integral_v<T>, "Integers requires integral type"); xarray_container<T> result(shape); std::uniform_int_distribution<T> dist(low, high); for (auto& v : result) v = dist(engine); return result; }

    // Generate normally (Gaussian) distributed numbers with given mean and stddev
    template <class T, class E>
    auto random_normal(const shape_type& shape, T mean, T stddev, E& engine)
    { static_assert(std::is_floating_point_v<T>, "Normal requires floating point"); xarray_container<T> result(shape); std::normal_distribution<T> dist(mean, stddev); for (auto& v : result) v = dist(engine); return result; }

    // Generate binomial distributed integers
    template <class T, class E>
    auto random_binomial(const shape_type& shape, int trials, double prob, E& engine)
    { static_assert(std::is_integral_v<T>, "Binomial requires integral type"); xarray_container<T> result(shape); std::binomial_distribution<T> dist(trials, prob); for (auto& v : result) v = dist(engine); return result; }

    // Generate poisson distributed integers
    template <class T, class E>
    auto random_poisson(const shape_type& shape, double mean, E& engine)
    { static_assert(std::is_integral_v<T>, "Poisson requires integral type"); xarray_container<T> result(shape); std::poisson_distribution<T> dist(mean); for (auto& v : result) v = dist(engine); return result; }

    // Generate exponentially distributed numbers
    template <class T, class E>
    auto random_exponential(const shape_type& shape, T lambda, E& engine)
    { static_assert(std::is_floating_point_v<T>, "Exponential requires floating point"); xarray_container<T> result(shape); std::exponential_distribution<T> dist(lambda); for (auto& v : result) v = dist(engine); return result; }

    // Generate gamma distributed numbers
    template <class T, class E>
    auto random_gamma(const shape_type& shape, T alpha, T beta, E& engine)
    { static_assert(std::is_floating_point_v<T>, "Gamma requires floating point"); xarray_container<T> result(shape); std::gamma_distribution<T> dist(alpha, beta); for (auto& v : result) v = dist(engine); return result; }

    // Generate chi‑squared distributed numbers
    template <class T, class E>
    auto random_chisquare(const shape_type& shape, T degrees, E& engine)
    { static_assert(std::is_floating_point_v<T>, "Chi-squared requires floating point"); xarray_container<T> result(shape); std::chi_squared_distribution<T> dist(degrees); for (auto& v : result) v = dist(engine); return result; }

    // Generate Student's t‑distributed numbers
    template <class T, class E>
    auto random_student_t(const shape_type& shape, T degrees, E& engine)
    { static_assert(std::is_floating_point_v<T>, "Student-t requires floating point"); xarray_container<T> result(shape); std::student_t_distribution<T> dist(degrees); for (auto& v : result) v = dist(engine); return result; }

    // Generate Weibull distributed numbers
    template <class T, class E>
    auto random_weibull(const shape_type& shape, T a, T b, E& engine)
    { static_assert(std::is_floating_point_v<T>, "Weibull requires floating point"); xarray_container<T> result(shape); std::weibull_distribution<T> dist(a, b); for (auto& v : result) v = dist(engine); return result; }

    // Generate a boolean mask with given success probability
    template <class E>
    auto random_bernoulli(const shape_type& shape, double p, E& engine)
    { xarray_container<bool> result(shape); std::bernoulli_distribution dist(p); for (auto& v : result) v = dist(engine); return result; }

    // Random choice from a given array (with or without replacement)
    template <class E, class Eng>
    auto random_choice(const xexpression<E>& a, size_type num_samples, bool replace, Eng& engine)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Random permutation of integers 0..n-1
    template <class Eng>
    auto random_permutation(size_type n, Eng& engine)
    { xarray_container<size_type> result({n}); std::iota(result.begin(), result.end(), 0); std::shuffle(result.begin(), result.end(), engine); return result; }

    // Shuffle an array (returns a shuffled copy)
    template <class E, class Eng>
    auto random_shuffle(const xexpression<E>& e, Eng& engine)
    { auto result = xarray_container<typename E::value_type>(e); std::shuffle(result.begin(), result.end(), engine); return result; }

    // Uniform in [0,1) – alias for random_uniform
    template <class T, class E>
    auto rand(const shape_type& shape, E& engine)
    { return random_uniform<T>(shape, T(0), T(1), engine); }

    // Standard normal – alias for random_normal(0,1)
    template <class T, class E>
    auto randn(const shape_type& shape, E& engine)
    { return random_normal<T>(shape, T(0), T(1), engine); }

    // Uniform integers – alias for random_integers
    template <class T, class E>
    auto randi(const shape_type& shape, T low, T high, E& engine)
    { return random_integers<T>(shape, low, high, engine); }

    // Generate random BigNumbers with up to `num_bits` random bits
    template <class E>
    auto random_bignumber_bits(const shape_type& shape, size_t num_bits, E& engine)
    { /* TODO: implement limb‑wise generation */ return xarray_container<bignumber::BigNumber>(shape); }

    // Generate random BigNumbers uniformly in [0, max_val)
    template <class E>
    auto random_bignumber_lt(const shape_type& shape, const bignumber::BigNumber& max_val, E& engine)
    { /* TODO: implement rejection sampling */ return xarray_container<bignumber::BigNumber>(shape); }

    // Generate random BigNumbers uniformly in [low, high]
    template <class E>
    auto random_bignumber_range(const shape_type& shape, const bignumber::BigNumber& low, const bignumber::BigNumber& high, E& engine)
    { /* TODO: implement */ return xarray_container<bignumber::BigNumber>(shape); }

    // Approximate normal distribution for BigNumber (Box‑Muller)
    template <class E>
    auto random_bignumber_normal(const shape_type& shape, const bignumber::BigNumber& mean, const bignumber::BigNumber& stddev, E& engine)
    { /* TODO: implement via double and scale */ return xarray_container<bignumber::BigNumber>(shape); }

    // Approximate Poisson distribution for BigNumber (Knuth's method)
    template <class E>
    auto random_bignumber_poisson(const shape_type& shape, const bignumber::BigNumber& lambda, E& engine)
    { /* TODO: implement */ return xarray_container<bignumber::BigNumber>(shape); }

} // namespace xt

#endif // XTENSOR_XRANDOM_HPPontainer<T> result(shape);
            std::fisher_f_distribution<T> dist(m, n);
            for (auto& v : result)
                v = dist(engine);
            return result;
        }

        // --------------------------------------------------------------------
        // Rayleigh distribution
        // --------------------------------------------------------------------
        template <class T = double, class E = default_engine_type>
        inline auto rayleigh(const shape_type& shape, T sigma = 1.0, E& engine = detail::get_default_engine())
        {
            static_assert(std::is_floating_point_v<T>, "rayleigh requires floating point");
            xarray_container<T> result(shape);
            // Rayleigh is not in std::; implement via Weibull or transform
            std::weibull_distribution<T> dist(sigma * std::sqrt(T(2)), T(2));
            for (auto& v : result)
                v = dist(engine);
            return result;
        }

        // --------------------------------------------------------------------
        // Cauchy distribution
        // --------------------------------------------------------------------
        template <class T = double, class E = default_engine_type>
        inline auto cauchy(const shape_type& shape, T location = 0.0, T scale = 1.0, E& engine = detail::get_default_engine())
        {
            static_assert(std::is_floating_point_v<T>, "cauchy requires floating point");
            xarray_container<T> result(shape);
            std::cauchy_distribution<T> dist(location, scale);
            for (auto& v : result)
                v = dist(engine);
            return result;
        }

        // --------------------------------------------------------------------
        // Bernoulli distribution (boolean mask)
        // --------------------------------------------------------------------
        template <class E = default_engine_type>
        inline auto bernoulli(const shape_type& shape, double p = 0.5, E& engine = detail::get_default_engine())
        {
            xarray_container<bool> result(shape);
            std::bernoulli_distribution dist(p);
            for (auto& v : result)
                v = dist(engine);
            return result;
        }

        // ====================================================================
        // Sampling functions
        // ====================================================================

        // --------------------------------------------------------------------
        // Random choice with replacement (uniform)
        // --------------------------------------------------------------------
        template <class T, class E = default_engine_type>
        inline auto choice(const xexpression<T>& a, size_type num_samples,
                           bool replace = true, E& engine = detail::get_default_engine())
        {
            const auto& arr = a.derived_cast();
            using value_type = typename T::value_type;
            if (!replace && static_cast<size_type>(num_samples) > arr.size())
                XTENSOR_THROW(std::invalid_argument, "Sample larger than population without replacement");

            std::vector<value_type> population(arr.begin(), arr.end());
            xarray_container<value_type> result({num_samples});

            if (replace)
            {
                std::uniform_int_distribution<size_t> dist(0, population.size() - 1);
                for (auto& v : result)
                    v = population[dist(engine)];
            }
            else
            {
                std::shuffle(population.begin(), population.end(), engine);
                std::copy_n(population.begin(), num_samples, result.begin());
            }
            return result;
        }

        // --------------------------------------------------------------------
        // Weighted random choice
        // --------------------------------------------------------------------
        template <class T, class W, class E = default_engine_type>
        inline auto choice(const xexpression<T>& a, size_type num_samples,
                           const xexpression<W>& weights, bool replace = true,
                           E& engine = detail::get_default_engine())
        {
            const auto& arr = a.derived_cast();
            const auto& w = weights.derived_cast();
            if (w.size() != arr.size())
                XTENSOR_THROW(std::invalid_argument, "Weights size must match population size");

            using value_type = typename T::value_type;
            std::vector<double> cum_weights(w.size());
            double sum = 0.0;
            for (size_t i = 0; i < w.size(); ++i)
            {
                sum += static_cast<double>(w.flat(i));
                cum_weights[i] = sum;
            }
            if (sum <= 0.0)
                XTENSOR_THROW(std::invalid_argument, "Sum of weights must be positive");

            std::uniform_real_distribution<double> dist(0.0, sum);
            xarray_container<value_type> result({num_samples});

            if (replace)
            {
                for (auto& v : result)
                {
                    double r = dist(engine);
                    auto it = std::lower_bound(cum_weights.begin(), cum_weights.end(), r);
                    size_t idx = std::distance(cum_weights.begin(), it);
                    if (idx >= arr.size()) idx = arr.size() - 1;
                    v = arr.flat(idx);
                }
            }
            else
            {
                // Without replacement: use systematic sampling (approximate)
                std::vector<size_t> indices(arr.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), engine);
                // Weighted selection from shuffled indices is complex; simplified:
                for (size_t i = 0; i < num_samples && i < arr.size(); ++i)
                    result.flat(i) = arr.flat(indices[i]);
            }
            return result;
        }

        // --------------------------------------------------------------------
        // Permutation of 0..n-1
        // --------------------------------------------------------------------
        template <class E = default_engine_type>
        inline auto permutation(size_type n, E& engine = detail::get_default_engine())
        {
            xarray_container<size_type> result({n});
            std::iota(result.begin(), result.end(), 0);
            std::shuffle(result.begin(), result.end(), engine);
            return result;
        }

        // --------------------------------------------------------------------
        // Shuffle an array (returns a copy)
        // --------------------------------------------------------------------
        template <class E, class Eng = default_engine_type>
        inline auto shuffle(const xexpression<E>& e, Eng& engine = detail::get_default_engine())
        {
            auto result = xarray_container<typename E::value_type>(e);
            std::shuffle(result.begin(), result.end(), engine);
            return result;
        }

        // ====================================================================
        // Convenience aliases (NumPy‑like)
        // ====================================================================

        template <class T = double, class E = default_engine_type>
        inline auto rand(const shape_type& shape, E& engine = detail::get_default_engine())
        {
            return uniform<T, E>(shape, 0.0, 1.0, engine);
        }

        template <class T = double, class E = default_engine_type>
        inline auto randn(const shape_type& shape, E& engine = detail::get_default_engine())
        {
            return normal<T, E>(shape, 0.0, 1.0, engine);
        }

        template <class T = int, class E = default_engine_type>
        inline auto randi(const shape_type& shape, T low, T high, E& engine = detail::get_default_engine())
        {
            return randint<T, E>(shape, low, high, engine);
        }

        // ====================================================================
        // BigNumber specializations (FFT‑aware arithmetic used downstream)
        // ====================================================================

        // --------------------------------------------------------------------
        // Uniform BigNumber in [0, 2^bits - 1]
        // --------------------------------------------------------------------
        template <class E = default_engine_type>
        inline auto bignumber_bits(const shape_type& shape, size_t num_bits, E& engine = detail::get_default_engine())
        {
            xarray_container<bignumber::BigNumber> result(shape);
            for (auto& v : result)
                v = detail::random_bignumber_bits(engine, num_bits);
            return result;
        }

        // --------------------------------------------------------------------
        // Uniform BigNumber in [0, max_val)
        // --------------------------------------------------------------------
        template <class E = default_engine_type>
        inline auto bignumber_uniform(const shape_type& shape, const bignumber::BigNumber& max_val,
                                      E& engine = detail::get_default_engine())
        {
            xarray_container<bignumber::BigNumber> result(shape);
            for (auto& v : result)
                v = detail::random_bignumber_lt(engine, max_val);
            return result;
        }

        // --------------------------------------------------------------------
        // Uniform BigNumber in [low, high]
        // --------------------------------------------------------------------
        template <class E = default_engine_type>
        inline auto bignumber_range(const shape_type& shape,
                                    const bignumber::BigNumber& low,
                                    const bignumber::BigNumber& high,
                                    E& engine = detail::get_default_engine())
        {
            xarray_container<bignumber::BigNumber> result(shape);
            bignumber::BigNumber range = high - low + bignumber::BigNumber(1);
            for (auto& v : result)
                v = low + detail::random_bignumber_lt(engine, range);
            return result;
        }

        // --------------------------------------------------------------------
        // Normal approximation for BigNumber (Box‑Muller)
        // --------------------------------------------------------------------
        template <class E = default_engine_type>
        inline auto bignumber_normal(const shape_type& shape,
                                     const bignumber::BigNumber& mean,
                                     const bignumber::BigNumber& stddev,
                                     E& engine = detail::get_default_engine())
        {
            // Use double-precision normal and scale; FFT multiplication used when
            // scaling with stddev (since it's BigNumber).
            xarray_container<bignumber::BigNumber> result(shape);
            std::normal_distribution<double> dist(0.0, 1.0);
            for (auto& v : result)
            {
                double sample = dist(engine);
                // Convert sample to BigNumber (assume conversion exists)
                bignumber::BigNumber bn_sample(sample);
                v = mean + bignumber::fft_multiply(bn_sample, stddev);
            }
            return result;
        }

        // --------------------------------------------------------------------
        // Poisson for BigNumber (Knuth's method)
        // --------------------------------------------------------------------
        template <class E = default_engine_type>
        inline auto bignumber_poisson(const shape_type& shape,
                                      const bignumber::BigNumber& lambda,
                                      E& engine = detail::get_default_engine())
        {
            xarray_container<bignumber::BigNumber> result(shape);
            // Convert lambda to double for the Poisson algorithm
            double L = std::exp(-bignumber::to_double(lambda));
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            for (auto& v : result)
            {
                int64_t k = 0;
                double p = 1.0;
                do {
                    ++k;
                    p *= dist(engine);
                } while (p > L);
                v = bignumber::BigNumber(k - 1);
            }
            return result;
        }

    } // namespace random

    // ========================================================================
    // Bring random functions into xt namespace for convenience
    // ========================================================================
    using random::seed;
    using random::get_default_random_engine;
    using random::set_default_random_engine;
    using random::rand;
    using random::randn;
    using random::randi;
    using random::uniform;
    using random::normal;
    using random::binomial;
    using random::poisson;
    using random::exponential;
    using random::gamma;
    using random::chi_squared;
    using random::student_t;
    using random::weibull;
    using random::geometric;
    using random::lognormal;
    using random::negative_binomial;
    using random::extreme_value;
    using random::fisher_f;
    using random::rayleigh;
    using random::cauchy;
    using random::bernoulli;
    using random::choice;
    using random::permutation;
    using random::shuffle;

} // namespace xt

#endif // XTENSOR_XRANDOM_HPPT>>(shape, min_val, max_val);
        }
        
        template <class T = int, std::size_t N>
        inline auto randint(const std::array<std::size_t, N>& shape, T min_val, T max_val)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xuniform_distribution<T> dist(min_val, max_val);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Normal (Gaussian)
        template <class T = double>
        inline auto randn(const svector<std::size_t>& shape, T mean = 0, T stddev = 1)
        {
            return detail::make_random<T, xnormal_distribution<T>>(shape, mean, stddev);
        }
        
        template <class T = double, std::size_t N>
        inline auto randn(const std::array<std::size_t, N>& shape, T mean = 0, T stddev = 1)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xnormal_distribution<T> dist(mean, stddev);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Bernoulli
        template <class T = bool>
        inline auto bernoulli(const svector<std::size_t>& shape, double p = 0.5)
        {
            return detail::make_random<T, xbernoulli_distribution<T>>(shape, p);
        }
        
        template <class T = bool, std::size_t N>
        inline auto bernoulli(const std::array<std::size_t, N>& shape, double p = 0.5)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xbernoulli_distribution<T> dist(p);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Binomial
        template <class T = int>
        inline auto binomial(const svector<std::size_t>& shape, T t, double p = 0.5)
        {
            return detail::make_random<T, xbinomial_distribution<T>>(shape, t, p);
        }
        
        template <class T = int, std::size_t N>
        inline auto binomial(const std::array<std::size_t, N>& shape, T t, double p = 0.5)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xbinomial_distribution<T> dist(t, p);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Poisson
        template <class T = int>
        inline auto poisson(const svector<std::size_t>& shape, double mean = 1.0)
        {
            return detail::make_random<T, xpoisson_distribution<T>>(shape, mean);
        }
        
        template <class T = int, std::size_t N>
        inline auto poisson(const std::array<std::size_t, N>& shape, double mean = 1.0)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xpoisson_distribution<T> dist(mean);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Exponential
        template <class T = double>
        inline auto exponential(const svector<std::size_t>& shape, T lambda = 1.0)
        {
            return detail::make_random<T, xexponential_distribution<T>>(shape, lambda);
        }
        
        template <class T = double, std::size_t N>
        inline auto exponential(const std::array<std::size_t, N>& shape, T lambda = 1.0)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xexponential_distribution<T> dist(lambda);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Gamma
        template <class T = double>
        inline auto gamma(const svector<std::size_t>& shape, T alpha, T beta)
        {
            return detail::make_random<T, xgamma_distribution<T>>(shape, alpha, beta);
        }
        
        template <class T = double, std::size_t N>
        inline auto gamma(const std::array<std::size_t, N>& shape, T alpha, T beta)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xgamma_distribution<T> dist(alpha, beta);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Beta
        template <class T = double>
        inline auto beta(const svector<std::size_t>& shape, T alpha, T beta)
        {
            return detail::make_random<T, xbeta_distribution<T>>(shape, alpha, beta);
        }
        
        template <class T = double, std::size_t N>
        inline auto beta(const std::array<std::size_t, N>& shape, T alpha, T beta)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xbeta_distribution<T> dist(alpha, beta);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Chi-squared
        template <class T = double>
        inline auto chisquare(const svector<std::size_t>& shape, T n)
        {
            return detail::make_random<T, xchisquared_distribution<T>>(shape, n);
        }
        
        template <class T = double, std::size_t N>
        inline auto chisquare(const std::array<std::size_t, N>& shape, T n)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xchisquared_distribution<T> dist(n);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Student's t
        template <class T = double>
        inline auto student_t(const svector<std::size_t>& shape, T n)
        {
            return detail::make_random<T, xstudent_t_distribution<T>>(shape, n);
        }
        
        template <class T = double, std::size_t N>
        inline auto student_t(const std::array<std::size_t, N>& shape, T n)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xstudent_t_distribution<T> dist(n);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Fisher F
        template <class T = double>
        inline auto fisher_f(const svector<std::size_t>& shape, T d1, T d2)
        {
            return detail::make_random<T, xfisher_f_distribution<T>>(shape, d1, d2);
        }
        
        template <class T = double, std::size_t N>
        inline auto fisher_f(const std::array<std::size_t, N>& shape, T d1, T d2)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xfisher_f_distribution<T> dist(d1, d2);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Log-normal
        template <class T = double>
        inline auto lognormal(const svector<std::size_t>& shape, T m, T s)
        {
            return detail::make_random<T, xlognormal_distribution<T>>(shape, m, s);
        }
        
        template <class T = double, std::size_t N>
        inline auto lognormal(const std::array<std::size_t, N>& shape, T m, T s)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xlognormal_distribution<T> dist(m, s);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Weibull
        template <class T = double>
        inline auto weibull(const svector<std::size_t>& shape, T a, T b)
        {
            return detail::make_random<T, xweibull_distribution<T>>(shape, a, b);
        }
        
        template <class T = double, std::size_t N>
        inline auto weibull(const std::array<std::size_t, N>& shape, T a, T b)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xweibull_distribution<T> dist(a, b);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Extreme value (Gumbel)
        template <class T = double>
        inline auto gumbel(const svector<std::size_t>& shape, T a, T b)
        {
            return detail::make_random<T, xextreme_value_distribution<T>>(shape, a, b);
        }
        
        template <class T = double, std::size_t N>
        inline auto gumbel(const std::array<std::size_t, N>& shape, T a, T b)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xextreme_value_distribution<T> dist(a, b);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Choice - random sample from given values
        template <class T>
        inline auto choice(const svector<std::size_t>& shape, const std::vector<T>& values,
                           const std::vector<double>& probs = {})
        {
            xarray_container<T> result(shape);
            xrandom_engine gen;
            if (probs.empty())
            {
                std::uniform_int_distribution<std::size_t> dist(0, values.size() - 1);
                for (auto& v : result)
                {
                    v = values[dist(gen)];
                }
            }
            else
            {
                if (probs.size() != values.size())
                {
                    XTENSOR_THROW(std::invalid_argument, "choice: probs size must match values size");
                }
                std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
                for (auto& v : result)
                {
                    v = values[dist(gen)];
                }
            }
            return result;
        }
        
        template <class T, std::size_t N>
        inline auto choice(const std::array<std::size_t, N>& shape, const std::vector<T>& values,
                           const std::vector<double>& probs = {})
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            if (probs.empty())
            {
                std::uniform_int_distribution<std::size_t> dist(0, values.size() - 1);
                for (auto& v : result)
                {
                    v = values[dist(gen)];
                }
            }
            else
            {
                if (probs.size() != values.size())
                {
                    XTENSOR_THROW(std::invalid_argument, "choice: probs size must match values size");
                }
                std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
                for (auto& v : result)
                {
                    v = values[dist(gen)];
                }
            }
            return result;
        }
        
        // Permutation - random permutation of integers
        inline auto permutation(std::size_t n)
        {
            std::vector<std::size_t> result(n);
            std::iota(result.begin(), result.end(), 0);
            xrandom_engine gen;
            std::shuffle(result.begin(), result.end(), gen.engine());
            return result;
        }
        
        inline auto permutation(const svector<std::size_t>& shape, std::size_t n)
        {
            xarray_container<std::size_t> result(shape);
            std::vector<std::size_t> pool(n);
            std::iota(pool.begin(), pool.end(), 0);
            xrandom_engine gen;
            for (auto& v : result)
            {
                std::uniform_int_distribution<std::size_t> dist(0, pool.size() - 1);
                std::size_t idx = dist(gen);
                v = pool[idx];
            }
            return result;
        }
        
        // Shuffle - randomly shuffle elements of an array
        template <class E>
        inline auto shuffle(const xexpression<E>& expr)
        {
            auto result = eval(expr);
            xrandom_engine gen;
            std::shuffle(result.begin(), result.end(), gen.engine());
            return result;
        }
        
        template <class E>
        inline void shuffle(xexpression<E>& expr)
        {
            auto& e = expr.derived_cast();
            xrandom_engine gen;
            std::shuffle(e.begin(), e.end(), gen.engine());
        }
        
        // --------------------------------------------------------------------
        // Parallel random generation (using multiple threads)
        // --------------------------------------------------------------------
        namespace detail
        {
            template <class T, class Dist, class... Args>
            inline void parallel_fill_random(xexpression<T>& e, Dist& dist_template, std::size_t num_threads = 0)
            {
                if (num_threads == 0)
                {
                    num_threads = std::thread::hardware_concurrency();
                    if (num_threads == 0) num_threads = 4;
                }
                
                auto& expr = e.derived_cast();
                std::size_t total_size = expr.size();
                if (total_size < num_threads * 1000)
                {
                    // Not worth parallelizing
                    xrandom_engine gen;
                    auto dist = dist_template;
                    for (auto& v : expr) v = dist(gen);
                    return;
                }
                
                std::vector<std::thread> threads;
                std::size_t chunk_size = total_size / num_threads;
                
                for (std::size_t t = 0; t < num_threads; ++t)
                {
                    std::size_t start = t * chunk_size;
                    std::size_t end = (t == num_threads - 1) ? total_size : start + chunk_size;
                    
                    threads.emplace_back([&expr, start, end, &dist_template]() {
                        xrandom_engine gen;
                        auto dist = dist_template;
                        for (std::size_t i = start; i < end; ++i)
                        {
                            expr.flat(i) = dist(gen);
                        }
                    });
                }
                
                for (auto& t : threads)
                {
                    t.join();
                }
            }
        }
        
        template <class T>
        inline auto parallel_random(const svector<std::size_t>& shape, T min_val = 0, T max_val = 1, std::size_t num_threads = 0)
        {
            xarray_container<T> result(shape);
            xuniform_distribution<T> dist(min_val, max_val);
            detail::parallel_fill_random(result, dist, num_threads);
            return result;
        }
        
        template <class T>
        inline auto parallel_randn(const svector<std::size_t>& shape, T mean = 0, T stddev = 1, std::size_t num_threads = 0)
        {
            xarray_container<T> result(shape);
            xnormal_distribution<T> dist(mean, stddev);
            detail::parallel_fill_random(result, dist, num_threads);
            return result;
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XRANDOM_HPP

// math/xrandom.hpp