// math/xrandom.hpp

#ifndef XTENSOR_XRANDOM_HPP
#define XTENSOR_XRANDOM_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"

#include <random>
#include <chrono>
#include <functional>
#include <type_traits>
#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <numeric>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // xrandom_engine - Default random engine (Mersenne Twister 19937)
        // --------------------------------------------------------------------
        class xrandom_engine
        {
        public:
            using result_type = std::mt19937::result_type;
            
            xrandom_engine()
            {
                std::random_device rd;
                if (rd.entropy() > 0)
                {
                    std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
                    m_engine.seed(seq);
                }
                else
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
                    m_engine.seed(static_cast<result_type>(nanos));
                }
            }
            
            explicit xrandom_engine(result_type seed)
                : m_engine(seed)
            {
            }
            
            template <class Sseq>
            explicit xrandom_engine(Sseq& seq)
                : m_engine(seq)
            {
            }
            
            void seed(result_type value = result_type())
            {
                m_engine.seed(value);
            }
            
            template <class Sseq>
            void seed(Sseq& seq)
            {
                m_engine.seed(seq);
            }
            
            result_type operator()()
            {
                return m_engine();
            }
            
            void discard(unsigned long long z)
            {
                m_engine.discard(z);
            }
            
            static constexpr result_type min()
            {
                return std::mt19937::min();
            }
            
            static constexpr result_type max()
            {
                return std::mt19937::max();
            }
            
            std::mt19937& engine() { return m_engine; }
            const std::mt19937& engine() const { return m_engine; }
            
        private:
            std::mt19937 m_engine;
        };
        
        // Global random engine (thread-safe via internal locking)
        namespace detail
        {
            inline xrandom_engine& get_global_random_engine()
            {
                static xrandom_engine global_engine;
                return global_engine;
            }
            
            inline std::mutex& get_random_mutex()
            {
                static std::mutex mtx;
                return mtx;
            }
        }
        
        inline void random_seed(xrandom_engine::result_type seed)
        {
            std::lock_guard<std::mutex> lock(detail::get_random_mutex());
            detail::get_global_random_engine().seed(seed);
        }
        
        template <class Sseq>
        inline void random_seed(Sseq& seq)
        {
            std::lock_guard<std::mutex> lock(detail::get_random_mutex());
            detail::get_global_random_engine().seed(seq);
        }
        
        // --------------------------------------------------------------------
        // xrandom_generator - Base class for distribution generators
        // --------------------------------------------------------------------
        template <class Distribution, class Engine = xrandom_engine>
        class xrandom_generator
        {
        public:
            using distribution_type = Distribution;
            using engine_type = Engine;
            using result_type = typename distribution_type::result_type;
            
            xrandom_generator() = default;
            
            explicit xrandom_generator(const distribution_type& dist)
                : m_distribution(dist)
            {
            }
            
            xrandom_generator(const distribution_type& dist, const engine_type& eng)
                : m_distribution(dist), m_engine(eng)
            {
            }
            
            template <class... Args>
            result_type operator()(Args&&... args)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                return m_distribution(m_engine, std::forward<Args>(args)...);
            }
            
            template <class... Args>
            result_type generate(Args&&... args)
            {
                return this->operator()(std::forward<Args>(args)...);
            }
            
            void seed(typename engine_type::result_type seed_val)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_engine.seed(seed_val);
            }
            
            distribution_type& distribution() { return m_distribution; }
            const distribution_type& distribution() const { return m_distribution; }
            engine_type& engine() { return m_engine; }
            const engine_type& engine() const { return m_engine; }
            
        private:
            distribution_type m_distribution;
            engine_type m_engine;
            std::mutex m_mutex;
        };
        
        // --------------------------------------------------------------------
        // Uniform distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xuniform_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::conditional_t<
                std::is_floating_point<T>::value,
                std::uniform_real_distribution<T>,
                std::uniform_int_distribution<T>
            >;
            
            xuniform_distribution() = default;
            
            xuniform_distribution(T a, T b)
                : m_distribution(a, b)
            {
                if (a >= b)
                {
                    XTENSOR_THROW(std::invalid_argument, "uniform: invalid range [a, b]");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g)
            {
                return m_distribution(g);
            }
            
            T a() const { return m_distribution.a(); }
            T b() const { return m_distribution.b(); }
            void param(T a, T b) { m_distribution.param(typename distribution_type::param_type(a, b)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Normal (Gaussian) distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xnormal_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::normal_distribution<T>;
            
            xnormal_distribution() = default;
            xnormal_distribution(T mean, T stddev) : m_distribution(mean, stddev)
            {
                if (stddev <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "normal: stddev must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            T mean() const { return m_distribution.mean(); }
            T stddev() const { return m_distribution.stddev(); }
            void param(T mean, T stddev) { m_distribution.param(typename distribution_type::param_type(mean, stddev)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Bernoulli distribution (0 or 1)
        // --------------------------------------------------------------------
        template <class T = bool, class Engine = xrandom_engine>
        class xbernoulli_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::bernoulli_distribution;
            
            xbernoulli_distribution() = default;
            explicit xbernoulli_distribution(double p) : m_distribution(p)
            {
                if (p < 0.0 || p > 1.0)
                {
                    XTENSOR_THROW(std::invalid_argument, "bernoulli: p must be in [0,1]");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return static_cast<T>(m_distribution(g)); }
            
            double p() const { return m_distribution.p(); }
            void param(double p) { m_distribution.param(typename distribution_type::param_type(p)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Binomial distribution
        // --------------------------------------------------------------------
        template <class T = int, class Engine = xrandom_engine>
        class xbinomial_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::binomial_distribution<T>;
            
            xbinomial_distribution() = default;
            xbinomial_distribution(T t, double p) : m_distribution(t, p)
            {
                if (t < 0 || p < 0.0 || p > 1.0)
                {
                    XTENSOR_THROW(std::invalid_argument, "binomial: t >= 0 and p in [0,1]");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            T t() const { return m_distribution.t(); }
            double p() const { return m_distribution.p(); }
            void param(T t, double p) { m_distribution.param(typename distribution_type::param_type(t, p)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Poisson distribution
        // --------------------------------------------------------------------
        template <class T = int, class Engine = xrandom_engine>
        class xpoisson_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::poisson_distribution<T>;
            
            xpoisson_distribution() = default;
            explicit xpoisson_distribution(double mean) : m_distribution(mean)
            {
                if (mean <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "poisson: mean must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            double mean() const { return m_distribution.mean(); }
            void param(double mean) { m_distribution.param(typename distribution_type::param_type(mean)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Exponential distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xexponential_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::exponential_distribution<T>;
            
            xexponential_distribution() = default;
            explicit xexponential_distribution(T lambda) : m_distribution(lambda)
            {
                if (lambda <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "exponential: lambda must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            T lambda() const { return m_distribution.lambda(); }
            void param(T lambda) { m_distribution.param(typename distribution_type::param_type(lambda)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Gamma distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xgamma_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::gamma_distribution<T>;
            
            xgamma_distribution() = default;
            xgamma_distribution(T alpha, T beta) : m_distribution(alpha, beta)
            {
                if (alpha <= 0 || beta <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "gamma: alpha and beta must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            T alpha() const { return m_distribution.alpha(); }
            T beta() const { return m_distribution.beta(); }
            void param(T alpha, T beta) { m_distribution.param(typename distribution_type::param_type(alpha, beta)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Beta distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xbeta_distribution
        {
        public:
            using result_type = T;
            
            xbeta_distribution() = default;
            xbeta_distribution(T alpha, T beta) : m_alpha(alpha), m_beta(beta)
            {
                if (alpha <= 0 || beta <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "beta: alpha and beta must be positive");
                }
                m_gamma_alpha = std::gamma_distribution<T>(alpha, 1.0);
                m_gamma_beta = std::gamma_distribution<T>(beta, 1.0);
            }
            
            template <class URNG>
            T operator()(URNG& g)
            {
                T x = m_gamma_alpha(g);
                T y = m_gamma_beta(g);
                return x / (x + y);
            }
            
            T alpha() const { return m_alpha; }
            T beta() const { return m_beta; }
            void param(T alpha, T beta)
            {
                m_alpha = alpha;
                m_beta = beta;
                m_gamma_alpha.param(typename std::gamma_distribution<T>::param_type(alpha, 1.0));
                m_gamma_beta.param(typename std::gamma_distribution<T>::param_type(beta, 1.0));
            }
            
        private:
            T m_alpha = 1.0;
            T m_beta = 1.0;
            std::gamma_distribution<T> m_gamma_alpha;
            std::gamma_distribution<T> m_gamma_beta;
        };
        
        // --------------------------------------------------------------------
        // Chi-squared distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xchisquared_distribution
        {
        public:
            using result_type = T;
            
            xchisquared_distribution() = default;
            explicit xchisquared_distribution(T n) : m_gamma(n / 2.0, 2.0), m_n(n)
            {
                if (n <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "chisquared: n must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_gamma(g); }
            
            T n() const { return m_n; }
            void param(T n)
            {
                m_n = n;
                m_gamma.param(typename std::gamma_distribution<T>::param_type(n / 2.0, 2.0));
            }
            
        private:
            std::gamma_distribution<T> m_gamma;
            T m_n = 1.0;
        };
        
        // --------------------------------------------------------------------
        // Student's t-distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xstudent_t_distribution
        {
        public:
            using result_type = T;
            
            xstudent_t_distribution() = default;
            explicit xstudent_t_distribution(T n) : m_n(n)
            {
                if (n <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "student_t: n must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g)
            {
                std::normal_distribution<T> normal(0.0, 1.0);
                std::chi_squared_distribution<T> chisq(m_n);
                T z = normal(g);
                T v = chisq(g);
                return z / std::sqrt(v / m_n);
            }
            
            T n() const { return m_n; }
            void param(T n) { m_n = n; }
            
        private:
            T m_n = 1.0;
        };
        
        // --------------------------------------------------------------------
        // Fisher F-distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xfisher_f_distribution
        {
        public:
            using result_type = T;
            
            xfisher_f_distribution() = default;
            xfisher_f_distribution(T d1, T d2) : m_d1(d1), m_d2(d2)
            {
                if (d1 <= 0 || d2 <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "fisher_f: d1 and d2 must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g)
            {
                std::chi_squared_distribution<T> chisq1(m_d1);
                std::chi_squared_distribution<T> chisq2(m_d2);
                T x1 = chisq1(g) / m_d1;
                T x2 = chisq2(g) / m_d2;
                return x1 / x2;
            }
            
            T d1() const { return m_d1; }
            T d2() const { return m_d2; }
            void param(T d1, T d2) { m_d1 = d1; m_d2 = d2; }
            
        private:
            T m_d1 = 1.0;
            T m_d2 = 1.0;
        };
        
        // --------------------------------------------------------------------
        // Log-normal distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xlognormal_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::lognormal_distribution<T>;
            
            xlognormal_distribution() = default;
            xlognormal_distribution(T m, T s) : m_distribution(m, s)
            {
                if (s <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "lognormal: s must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            T m() const { return m_distribution.m(); }
            T s() const { return m_distribution.s(); }
            void param(T m, T s) { m_distribution.param(typename distribution_type::param_type(m, s)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Weibull distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xweibull_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::weibull_distribution<T>;
            
            xweibull_distribution() = default;
            xweibull_distribution(T a, T b) : m_distribution(a, b)
            {
                if (a <= 0 || b <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "weibull: a and b must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            T a() const { return m_distribution.a(); }
            T b() const { return m_distribution.b(); }
            void param(T a, T b) { m_distribution.param(typename distribution_type::param_type(a, b)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Extreme value distribution (Gumbel)
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xextreme_value_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::extreme_value_distribution<T>;
            
            xextreme_value_distribution() = default;
            xextreme_value_distribution(T a, T b) : m_distribution(a, b)
            {
                if (b <= 0)
                {
                    XTENSOR_THROW(std::invalid_argument, "extreme_value: b must be positive");
                }
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            T a() const { return m_distribution.a(); }
            T b() const { return m_distribution.b(); }
            void param(T a, T b) { m_distribution.param(typename distribution_type::param_type(a, b)); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Discrete distribution
        // --------------------------------------------------------------------
        template <class T = int, class Engine = xrandom_engine>
        class xdiscrete_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::discrete_distribution<T>;
            
            xdiscrete_distribution() = default;
            
            template <class InputIt>
            xdiscrete_distribution(InputIt first, InputIt last)
                : m_distribution(first, last)
            {
            }
            
            xdiscrete_distribution(std::initializer_list<double> weights)
                : m_distribution(weights)
            {
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            std::vector<double> probabilities() const { return m_distribution.probabilities(); }
            
            template <class InputIt>
            void param(InputIt first, InputIt last)
            {
                m_distribution.param(typename distribution_type::param_type(first, last));
            }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Piecewise constant distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xpiecewise_constant_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::piecewise_constant_distribution<T>;
            
            xpiecewise_constant_distribution() = default;
            
            template <class InputItB, class InputItW>
            xpiecewise_constant_distribution(InputItB first_b, InputItB last_b, InputItW first_w)
                : m_distribution(first_b, last_b, first_w)
            {
            }
            
            xpiecewise_constant_distribution(std::initializer_list<T> boundaries,
                                             std::initializer_list<double> weights)
                : m_distribution(boundaries, weights)
            {
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            std::vector<T> intervals() const { return m_distribution.intervals(); }
            std::vector<double> densities() const { return m_distribution.densities(); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Piecewise linear distribution
        // --------------------------------------------------------------------
        template <class T = double, class Engine = xrandom_engine>
        class xpiecewise_linear_distribution
        {
        public:
            using result_type = T;
            using distribution_type = std::piecewise_linear_distribution<T>;
            
            xpiecewise_linear_distribution() = default;
            
            template <class InputItB, class InputItW>
            xpiecewise_linear_distribution(InputItB first_b, InputItB last_b, InputItW first_w)
                : m_distribution(first_b, last_b, first_w)
            {
            }
            
            xpiecewise_linear_distribution(std::initializer_list<T> boundaries,
                                           std::initializer_list<double> weights)
                : m_distribution(boundaries, weights)
            {
            }
            
            template <class URNG>
            T operator()(URNG& g) { return m_distribution(g); }
            
            std::vector<T> intervals() const { return m_distribution.intervals(); }
            std::vector<double> densities() const { return m_distribution.densities(); }
            
        private:
            distribution_type m_distribution;
        };
        
        // --------------------------------------------------------------------
        // Random tensor generation functions
        // --------------------------------------------------------------------
        
        namespace detail
        {
            template <class T, class Gen, class Dist>
            inline void fill_random(xexpression<T>& e, Gen& gen, Dist& dist)
            {
                auto& expr = e.derived_cast();
                for (auto& v : expr)
                {
                    v = dist(gen);
                }
            }
            
            template <class T, class Dist, class... Args>
            inline auto make_random(const svector<std::size_t>& shape, Args&&... args)
            {
                xarray_container<T> result(shape);
                xrandom_engine gen;
                Dist dist(std::forward<Args>(args)...);
                fill_random(result, gen, dist);
                return result;
            }
        }
        
        // Uniform random
        template <class T = double>
        inline auto random(const svector<std::size_t>& shape, T min_val = 0, T max_val = 1)
        {
            return detail::make_random<T, xuniform_distribution<T>>(shape, min_val, max_val);
        }
        
        template <class T = double, std::size_t N>
        inline auto random(const std::array<std::size_t, N>& shape, T min_val = 0, T max_val = 1)
        {
            xtensor_container<T, N> result(shape);
            xrandom_engine gen;
            xuniform_distribution<T> dist(min_val, max_val);
            detail::fill_random(result, gen, dist);
            return result;
        }
        
        // Uniform integer
        template <class T = int>
        inline auto randint(const svector<std::size_t>& shape, T min_val, T max_val)
        {
            return detail::make_random<T, xuniform_distribution<T>>(shape, min_val, max_val);
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