// include/xtu/utils/xbenchmark.hpp
// xtensor-unified - Benchmarking and timing utilities
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_UTILS_XBENCHMARK_HPP
#define XTU_UTILS_XBENCHMARK_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/stats/xstats.hpp"

XTU_NAMESPACE_BEGIN
namespace utils {

// #############################################################################
// High-resolution timer
// #############################################################################
class Timer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using duration = clock::duration;

private:
    time_point m_start;
    time_point m_end;
    bool m_running;

public:
    Timer() : m_running(false) {}

    void start() {
        m_start = clock::now();
        m_running = true;
    }

    void stop() {
        if (m_running) {
            m_end = clock::now();
            m_running = false;
        }
    }

    void reset() {
        m_running = false;
    }

    double elapsed_seconds() const {
        if (m_running) {
            auto now = clock::now();
            return std::chrono::duration<double>(now - m_start).count();
        }
        return std::chrono::duration<double>(m_end - m_start).count();
    }

    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }

    double elapsed_microseconds() const {
        return elapsed_seconds() * 1000000.0;
    }

    double elapsed_nanoseconds() const {
        return elapsed_seconds() * 1000000000.0;
    }
};

// #############################################################################
// Scoped timer (RAII)
// #############################################################################
class ScopedTimer {
private:
    Timer m_timer;
    std::string m_name;
    std::function<void(const std::string&, double)> m_callback;

public:
    explicit ScopedTimer(const std::string& name = "",
                         std::function<void(const std::string&, double)> callback = nullptr)
        : m_name(name), m_callback(callback) {
        m_timer.start();
    }

    ~ScopedTimer() {
        m_timer.stop();
        double elapsed = m_timer.elapsed_seconds();
        if (m_callback) {
            m_callback(m_name, elapsed);
        } else if (!m_name.empty()) {
            std::cout << "[TIMER] " << m_name << ": " << elapsed << " s" << std::endl;
        }
    }

    double elapsed() const { return m_timer.elapsed_seconds(); }
};

// #############################################################################
// Benchmark result with statistics
// #############################################################################
struct BenchmarkResult {
    std::string name;
    size_t iterations;
    double total_time_sec;
    double mean_time_sec;
    double median_time_sec;
    double min_time_sec;
    double max_time_sec;
    double stddev_time_sec;
    std::vector<double> individual_times;

    std::string to_string() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "Benchmark: " << name << "\n";
        oss << "  Iterations: " << iterations << "\n";
        oss << "  Total:      " << total_time_sec << " s\n";
        oss << "  Mean:       " << mean_time_sec * 1000.0 << " ms\n";
        oss << "  Median:     " << median_time_sec * 1000.0 << " ms\n";
        oss << "  Min:        " << min_time_sec * 1000.0 << " ms\n";
        oss << "  Max:        " << max_time_sec * 1000.0 << " ms\n";
        oss << "  StdDev:     " << stddev_time_sec * 1000.0 << " ms";
        return oss.str();
    }
};

// #############################################################################
// Benchmark runner
// #############################################################################
class BenchmarkRunner {
private:
    std::map<std::string, BenchmarkResult> m_results;
    size_t m_default_iterations;
    size_t m_warmup_iterations;

public:
    explicit BenchmarkRunner(size_t default_iterations = 10, size_t warmup_iterations = 2)
        : m_default_iterations(default_iterations), m_warmup_iterations(warmup_iterations) {}

    // Run a single benchmark
    template <class Func>
    BenchmarkResult run(const std::string& name, Func&& func, size_t iterations = 0) {
        if (iterations == 0) iterations = m_default_iterations;

        BenchmarkResult result;
        result.name = name;
        result.iterations = iterations;
        result.individual_times.reserve(iterations + m_warmup_iterations);

        Timer timer;
        // Warmup
        for (size_t i = 0; i < m_warmup_iterations; ++i) {
            func();
        }
        // Measurement
        for (size_t i = 0; i < iterations; ++i) {
            timer.start();
            func();
            timer.stop();
            result.individual_times.push_back(timer.elapsed_seconds());
        }

        // Compute statistics
        if (!result.individual_times.empty()) {
            result.total_time_sec = std::accumulate(result.individual_times.begin(),
                                                     result.individual_times.end(), 0.0);
            result.mean_time_sec = result.total_time_sec / static_cast<double>(iterations);
            std::sort(result.individual_times.begin(), result.individual_times.end());
            result.min_time_sec = result.individual_times.front();
            result.max_time_sec = result.individual_times.back();
            size_t mid = iterations / 2;
            if (iterations % 2 == 0) {
                result.median_time_sec = (result.individual_times[mid - 1] + result.individual_times[mid]) / 2.0;
            } else {
                result.median_time_sec = result.individual_times[mid];
            }
            double sum_sq = 0.0;
            for (double t : result.individual_times) {
                double diff = t - result.mean_time_sec;
                sum_sq += diff * diff;
            }
            result.stddev_time_sec = std::sqrt(sum_sq / static_cast<double>(iterations));
        }

        m_results[name] = result;
        return result;
    }

    // Compare two functions
    template <class Func1, class Func2>
    void compare(const std::string& name1, Func1&& func1,
                 const std::string& name2, Func2&& func2,
                 size_t iterations = 0) {
        auto res1 = run(name1, std::forward<Func1>(func1), iterations);
        auto res2 = run(name2, std::forward<Func2>(func2), iterations);
        double speedup = res1.mean_time_sec / res2.mean_time_sec;
        std::cout << "Comparison:\n";
        std::cout << "  " << name1 << ": " << res1.mean_time_sec * 1000.0 << " ms\n";
        std::cout << "  " << name2 << ": " << res2.mean_time_sec * 1000.0 << " ms\n";
        std::cout << "  Speedup: " << speedup << "x\n";
    }

    // Print all results
    void print_all() const {
        for (const auto& kv : m_results) {
            std::cout << kv.second.to_string() << "\n\n";
        }
    }

    // Get result by name
    const BenchmarkResult& get_result(const std::string& name) const {
        auto it = m_results.find(name);
        if (it == m_results.end()) {
            XTU_THROW(std::out_of_range, "Benchmark not found: " + name);
        }
        return it->second;
    }

    // Clear all results
    void clear() { m_results.clear(); }

    // Set default iterations
    void set_default_iterations(size_t n) { m_default_iterations = n; }
    void set_warmup_iterations(size_t n) { m_warmup_iterations = n; }
};

// #############################################################################
// Global benchmark runner instance
// #############################################################################
inline BenchmarkRunner& global_benchmark() {
    static BenchmarkRunner runner;
    return runner;
}

// #############################################################################
// Convenience macros for quick benchmarking
// #############################################################################
#define XTU_BENCHMARK(name, func, iterations) \
    ::xtu::utils::global_benchmark().run(name, func, iterations)

#define XTU_BENCHMARK_ONCE(name, func) \
    ::xtu::utils::global_benchmark().run(name, func, 1)

#define XTU_BENCHMARK_COMPARE(name1, func1, name2, func2, iterations) \
    ::xtu::utils::global_benchmark().compare(name1, func1, name2, func2, iterations)

#define XTU_TIME_IT(name, code) \
    do { \
        ::xtu::utils::Timer _timer; \
        _timer.start(); \
        code; \
        _timer.stop(); \
        std::cout << "[TIMER] " << name << ": " << _timer.elapsed_seconds() << " s" << std::endl; \
    } while(0)

// #############################################################################
// Throughput measurement (operations per second)
// #############################################################################
struct ThroughputResult {
    std::string name;
    size_t operations;
    double total_time_sec;
    double ops_per_sec;
    double ns_per_op;
};

template <class Func>
ThroughputResult measure_throughput(const std::string& name, Func&& func, size_t operations) {
    Timer timer;
    timer.start();
    for (size_t i = 0; i < operations; ++i) {
        func();
    }
    timer.stop();
    double elapsed = timer.elapsed_seconds();
    ThroughputResult result;
    result.name = name;
    result.operations = operations;
    result.total_time_sec = elapsed;
    result.ops_per_sec = static_cast<double>(operations) / elapsed;
    result.ns_per_op = (elapsed * 1000000000.0) / static_cast<double>(operations);
    return result;
}

// #############################################################################
// Memory bandwidth measurement helper
// #############################################################################
template <class T>
double measure_bandwidth(size_t num_elements, std::function<void(T*, size_t)> kernel) {
    std::vector<T> data(num_elements);
    Timer timer;
    timer.start();
    kernel(data.data(), num_elements);
    timer.stop();
    double seconds = timer.elapsed_seconds();
    double bytes = static_cast<double>(num_elements * sizeof(T));
    return bytes / seconds;
}

// #############################################################################
// Simple progress reporter for long-running operations
// #############################################################################
class ProgressReporter {
private:
    size_t m_total;
    size_t m_current;
    size_t m_last_percent;
    Timer m_timer;
    std::string m_prefix;

public:
    explicit ProgressReporter(size_t total, const std::string& prefix = "Progress")
        : m_total(total), m_current(0), m_last_percent(0), m_prefix(prefix) {
        m_timer.start();
    }

    void update(size_t current) {
        m_current = current;
        size_t percent = (m_current * 100) / m_total;
        if (percent > m_last_percent) {
            m_last_percent = percent;
            double elapsed = m_timer.elapsed_seconds();
            double eta = (elapsed / static_cast<double>(m_current)) * static_cast<double>(m_total - m_current);
            std::cout << "\r" << m_prefix << ": " << percent << "% "
                      << "(" << m_current << "/" << m_total << ") "
                      << "ETA: " << format_time(eta) << "    " << std::flush;
        }
    }

    void increment(size_t n = 1) {
        update(m_current + n);
    }

    void finish() {
        m_timer.stop();
        std::cout << "\r" << m_prefix << ": 100% "
                  << "(" << m_total << "/" << m_total << ") "
                  << "Total: " << format_time(m_timer.elapsed_seconds())
                  << "                    " << std::endl;
    }

private:
    static std::string format_time(double seconds) {
        if (seconds < 60.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << seconds << "s";
            return oss.str();
        } else if (seconds < 3600.0) {
            int mins = static_cast<int>(seconds / 60.0);
            int secs = static_cast<int>(seconds) % 60;
            std::ostringstream oss;
            oss << mins << "m " << secs << "s";
            return oss.str();
        } else {
            int hours = static_cast<int>(seconds / 3600.0);
            int mins = (static_cast<int>(seconds) % 3600) / 60;
            std::ostringstream oss;
            oss << hours << "h " << mins << "m";
            return oss.str();
        }
    }
};

} // namespace utils

// Bring into main namespace for convenience
using utils::Timer;
using utils::ScopedTimer;
using utils::BenchmarkResult;
using utils::BenchmarkRunner;
using utils::global_benchmark;
using utils::measure_throughput;
using utils::ThroughputResult;
using utils::measure_bandwidth;
using utils::ProgressReporter;

XTU_NAMESPACE_END

#endif // XTU_UTILS_XBENCHMARK_HPP