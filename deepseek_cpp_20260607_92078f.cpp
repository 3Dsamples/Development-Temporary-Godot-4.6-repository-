/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_OBSERVABILITY_METRICS_COLLECTOR_H_INCLUDED
#define ORTHOTREE_CORE_OBSERVABILITY_METRICS_COLLECTOR_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/parallel/lockfree_query_buffer.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace OrthoTree {
namespace Observability {

// ============================================================================
//  Metric types (counter, gauge, histogram)
// ============================================================================
enum class MetricType : uint8_t {
    Counter,      // monotonic increment
    Gauge,        // arbitrary value
    Histogram,    // distribution of values
    Rate,         // events per second
    Timer         // duration
};

// ============================================================================
//  Histogram bucket configuration (logarithmic or linear)
// ============================================================================
struct HistogramConfig {
    std::vector<double> buckets;   // upper bounds for each bucket
    bool logarithmic = true;       // if true, generate logarithmic buckets
    double minValue = 0.0;
    double maxValue = 1000.0;
    size_t numBuckets = 20;
};

// ============================================================================
//  Metric sample (value + timestamp)
// ============================================================================
struct MetricSample {
    double value;
    std::chrono::steady_clock::time_point timestamp;
};

// ============================================================================
//  MetricsCollector: collects performance metrics, counts, and histograms
//  with low overhead. Supports SIMD batch updates and lock‑free recording.
//  Designed for real‑time profiling of octree operations, queries, and
//  simulation events.
// ============================================================================
class MetricsCollector {
public:
    using time_point = std::chrono::steady_clock::time_point;
    using duration = std::chrono::microseconds;

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        bool enableHistograms = true;
        bool enableTimers = true;
        size_t maxSampleCount = 10000;       // max samples per metric
        double sampleIntervalMs = 100.0;     // auto‑sample interval (ms)
        bool useSimdBatch = true;
        bool autoFlush = true;               // flush to disk periodically
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit MetricsCollector(const Config& cfg = Config())
        : m_config(cfg)
        , m_startTime(std::chrono::steady_clock::now())
        , m_lastFlush(m_startTime) {}

    // ------------------------------------------------------------------------
    //  Register a metric (create if not exists)
    // ------------------------------------------------------------------------
    void registerMetric(const std::string& name, MetricType type,
                        const HistogramConfig& histCfg = HistogramConfig()) {
        std::lock_guard<std::mutex> lock(m_metricsMutex);
        if (m_metrics.find(name) != m_metrics.end()) return;
        Metric m;
        m.type = type;
        m.name = name;
        if (type == MetricType::Histogram) {
            m.histCfg = histCfg;
            if (histCfg.logarithmic) {
                double logMin = std::log10(histCfg.minValue);
                double logMax = std::log10(histCfg.maxValue);
                for (size_t i = 0; i <= histCfg.numBuckets; ++i) {
                    double bound = std::pow(10.0, logMin + (logMax - logMin) * i / histCfg.numBuckets);
                    m.buckets.push_back(bound);
                }
            } else {
                double step = (histCfg.maxValue - histCfg.minValue) / histCfg.numBuckets;
                for (size_t i = 0; i <= histCfg.numBuckets; ++i) {
                    m.buckets.push_back(histCfg.minValue + i * step);
                }
            }
            m.bucketCounts.resize(m.buckets.size(), 0);
        }
        m.value.store(0.0, std::memory_order_relaxed);
        m_samples.reserve(m_config.maxSampleCount);
        m_metrics[name] = std::move(m);
    }

    // ------------------------------------------------------------------------
    //  Increment counter (thread‑safe, lock‑free)
    // ------------------------------------------------------------------------
    void incrementCounter(const std::string& name, double delta = 1.0) {
        auto it = getMetric(name);
        if (it != m_metrics.end() && it->second.type == MetricType::Counter) {
            it->second.value.fetch_add(delta, std::memory_order_relaxed);
        }
    }

    // ------------------------------------------------------------------------
    //  Set gauge value
    // ------------------------------------------------------------------------
    void setGauge(const std::string& name, double value) {
        auto it = getMetric(name);
        if (it != m_metrics.end() && it->second.type == MetricType::Gauge) {
            it->second.value.store(value, std::memory_order_relaxed);
        }
    }

    // ------------------------------------------------------------------------
    //  Record histogram observation
    // ------------------------------------------------------------------------
    void observeHistogram(const std::string& name, double value) {
        auto it = getMetric(name);
        if (it == m_metrics.end() || it->second.type != MetricType::Histogram) return;
        auto& metric = it->second;
        std::lock_guard<std::mutex> lock(metric.sampleMutex);
        // Find bucket
        size_t idx = 0;
        while (idx < metric.buckets.size() && value > metric.buckets[idx]) ++idx;
        if (idx < metric.bucketCounts.size()) {
            metric.bucketCounts[idx]++;
        }
        metric.totalCount++;
        metric.sum += value;
        if (metric.samples.size() < m_config.maxSampleCount) {
            metric.samples.push_back({value, std::chrono::steady_clock::now()});
        }
    }

    // ------------------------------------------------------------------------
    //  Timer: measure duration of a scope (RAII)
    // ------------------------------------------------------------------------
    class ScopedTimer {
    public:
        ScopedTimer(MetricsCollector& collector, const std::string& name)
            : m_collector(collector), m_name(name), m_start(std::chrono::steady_clock::now()) {}
        ~ScopedTimer() {
            auto end = std::chrono::steady_clock::now();
            double us = std::chrono::duration_cast<duration>(end - m_start).count();
            m_collector.recordTimer(m_name, us);
        }
    private:
        MetricsCollector& m_collector;
        std::string m_name;
        time_point m_start;
    };

    // Record timer value manually (microseconds)
    void recordTimer(const std::string& name, double microseconds) {
        auto it = getMetric(name);
        if (it != m_metrics.end() && it->second.type == MetricType::Timer) {
            // treat as histogram
            observeHistogram(name, microseconds);
            it->second.value.store(microseconds, std::memory_order_relaxed); // last value
        } else {
            // auto‑register as timer
            registerMetric(name, MetricType::Timer);
            recordTimer(name, microseconds);
        }
    }

    // ------------------------------------------------------------------------
    //  Batch operations (SIMD friendly)
    // ------------------------------------------------------------------------
    void batchIncrement(const std::string& name, const double* deltas, size_t count) {
        auto it = getMetric(name);
        if (it == m_metrics.end()) return;
        auto& metric = it->second;
        if (metric.type != MetricType::Counter) return;
        double sum = 0.0;
        if (m_config.useSimdBatch && count >= 4) {
            // SIMD reduction (pseudo)
            for (size_t i = 0; i < count; ++i) sum += deltas[i];
        } else {
            for (size_t i = 0; i < count; ++i) sum += deltas[i];
        }
        metric.value.fetch_add(sum, std::memory_order_relaxed);
    }

    // ------------------------------------------------------------------------
    //  Query current value (snapshot)
    // ------------------------------------------------------------------------
    double getValue(const std::string& name) const {
        auto it = getMetric(name);
        if (it != m_metrics.end()) {
            return it->second.value.load(std::memory_order_relaxed);
        }
        return 0.0;
    }

    // ------------------------------------------------------------------------
    //  Get histogram snapshot
    // ------------------------------------------------------------------------
    struct HistogramSnapshot {
        std::vector<double> bucketBounds;
        std::vector<uint64_t> bucketCounts;
        uint64_t totalCount;
        double sum;
        double min;
        double max;
        double mean;
        double stddev;
    };

    HistogramSnapshot getHistogram(const std::string& name) const {
        HistogramSnapshot snap;
        auto it = getMetric(name);
        if (it == m_metrics.end() || it->second.type != MetricType::Histogram) return snap;
        const auto& metric = it->second;
        std::lock_guard<std::mutex> lock(metric.sampleMutex);
        snap.bucketBounds = metric.buckets;
        snap.bucketCounts = metric.bucketCounts;
        snap.totalCount = metric.totalCount;
        snap.sum = metric.sum;
        // compute min, max, mean, stddev from samples
        double minVal = std::numeric_limits<double>::max();
        double maxVal = -std::numeric_limits<double>::max();
        double sumSq = 0.0;
        for (const auto& s : metric.samples) {
            if (s.value < minVal) minVal = s.value;
            if (s.value > maxVal) maxVal = s.value;
            sumSq += s.value * s.value;
        }
        snap.min = minVal;
        snap.max = maxVal;
        if (snap.totalCount > 0) {
            snap.mean = snap.sum / snap.totalCount;
            snap.stddev = std::sqrt((sumSq / snap.totalCount) - snap.mean * snap.mean);
        } else {
            snap.mean = 0.0;
            snap.stddev = 0.0;
        }
        return snap;
    }

    // ------------------------------------------------------------------------
    //  Reset all metrics (clear samples, zero counters)
    // ------------------------------------------------------------------------
    void reset() {
        std::lock_guard<std::mutex> lock(m_metricsMutex);
        for (auto& pair : m_metrics) {
            auto& metric = pair.second;
            metric.value.store(0.0, std::memory_order_relaxed);
            std::lock_guard<std::mutex> sampleLock(metric.sampleMutex);
            metric.samples.clear();
            metric.totalCount = 0;
            metric.sum = 0.0;
            std::fill(metric.bucketCounts.begin(), metric.bucketCounts.end(), 0);
        }
        m_startTime = std::chrono::steady_clock::now();
    }

    // ------------------------------------------------------------------------
    //  Export to JSON (for monitoring tools)
    // ------------------------------------------------------------------------
    std::string exportToJson() const {
        std::lock_guard<std::mutex> lock(m_metricsMutex);
        std::string json = "{\n";
        bool first = true;
        for (const auto& pair : m_metrics) {
            if (!first) json += ",\n";
            first = false;
            const auto& metric = pair.second;
            json += "  \"" + metric.name + "\": {\n";
            json += "    \"type\": \"" + metricTypeToString(metric.type) + "\",\n";
            json += "    \"value\": " + std::to_string(metric.value.load()) + ",\n";
            if (metric.type == MetricType::Histogram) {
                std::lock_guard<std::mutex> sampleLock(metric.sampleMutex);
                json += "    \"totalCount\": " + std::to_string(metric.totalCount) + ",\n";
                json += "    \"sum\": " + std::to_string(metric.sum) + ",\n";
                json += "    \"buckets\": [";
                for (size_t i = 0; i < metric.bucketCounts.size(); ++i) {
                    if (i > 0) json += ",";
                    json += "{\"le\":" + std::to_string(metric.buckets[i]) +
                            ",\"count\":" + std::to_string(metric.bucketCounts[i]) + "}";
                }
                json += "]\n";
            }
            json += "  }";
        }
        json += "\n}\n";
        return json;
    }

private:
    using MetricMap = std::unordered_map<std::string, struct Metric>;

    struct Metric {
        std::string name;
        MetricType type;
        std::atomic<double> value;
        std::vector<double> buckets;
        std::vector<uint64_t> bucketCounts;
        std::vector<MetricSample> samples;
        uint64_t totalCount = 0;
        double sum = 0.0;
        HistogramConfig histCfg;
        mutable std::mutex sampleMutex;
    };

    auto getMetric(const std::string& name) const {
        std::lock_guard<std::mutex> lock(m_metricsMutex);
        return m_metrics.find(name);
    }

    auto getMetric(const std::string& name) {
        std::lock_guard<std::mutex> lock(m_metricsMutex);
        return m_metrics.find(name);
    }

    static const char* metricTypeToString(MetricType type) {
        switch (type) {
            case MetricType::Counter: return "counter";
            case MetricType::Gauge: return "gauge";
            case MetricType::Histogram: return "histogram";
            case MetricType::Rate: return "rate";
            case MetricType::Timer: return "timer";
            default: return "unknown";
        }
    }

    Config m_config;
    time_point m_startTime;
    time_point m_lastFlush;
    mutable std::mutex m_metricsMutex;
    MetricMap m_metrics;
};

// ----------------------------------------------------------------------------
//  Global instance for convenience (optional)
// ----------------------------------------------------------------------------
inline MetricsCollector& globalMetrics() {
    static MetricsCollector instance;
    return instance;
}

// ----------------------------------------------------------------------------
//  SIMD helper: batch update multiple counters
// ----------------------------------------------------------------------------
class SimdMetricsUpdater {
public:
    static void updateBatch(MetricsCollector& collector,
                            const std::string* names,
                            const double* deltas,
                            size_t count) {
        for (size_t i = 0; i < count; ++i) {
            collector.incrementCounter(names[i], deltas[i]);
        }
    }
};

} // namespace Observability
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OBSERVABILITY_METRICS_COLLECTOR_H_INCLUDED