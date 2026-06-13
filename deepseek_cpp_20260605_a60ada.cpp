//File 0345 : xframe/xframe_trace.hpp
//Execution tracing and logging for xframe operations: records memory usage, operation counts, timings, and SIMD utilization to aid performance analysis and debugging.
#ifndef XFRAME_XFRAME_TRACE_HPP
#define XFRAME_XFRAME_TRACE_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xframe_config.hpp"

namespace xframe {
namespace trace {

    /**
     * @enum trace_level
     * @brief Verbosity levels for tracing.
     */
    enum class trace_level : int { none = 0, error, warning, info, debug, simd };

    /**
     * @struct trace_record
     * @brief A single trace entry.
     */
    struct trace_record {
        std::chrono::high_resolution_clock::time_point timestamp;
        std::string operation;
        std::size_t size;
        double duration_us;
        bool simd_used;
    };

    /**
     * @class xframe_tracer
     * @brief Singleton tracer collecting execution statistics.
     *
     * Records every major operation (copy, arithmetic, reduction, etc.)
     * along with timing, data size, and whether SIMD was utilized.
     * The collected trace can be exported to JSON or CSV for offline
     * performance profiling.
     */
    class xframe_tracer {
    public:
        static xframe_tracer& instance() {
            static xframe_tracer tracer;
            return tracer;
        }

        /**
         * Set the minimum level for recording.
         */
        void set_level(trace_level lvl) { m_level = lvl; }

        /**
         * Enable or disable tracing globally.
         */
        void enable(bool on) { m_enabled = on; }
        bool is_enabled() const noexcept { return m_enabled; }

        /**
         * Record an operation.
         * @param op_name Descriptive name (e.g., "add", "matmul").
         * @param element_count Number of elements processed.
         * @param elapsed_us Wall‑clock time in microseconds.
         * @param simd_accelerated True if SIMD was used.
         */
        void record(const std::string& op_name, std::size_t element_count,
                    double elapsed_us, bool simd_accelerated) {
            if (!m_enabled) return;
            std::lock_guard<std::mutex> lock(m_mutex);
            m_records.push_back({
                std::chrono::high_resolution_clock::now(),
                op_name, element_count, elapsed_us, simd_accelerated
            });
            // Update aggregate counters
            auto& agg = m_aggregates[op_name];
            agg.count++;
            agg.total_elements += element_count;
            agg.total_time_us += elapsed_us;
            if (simd_accelerated) agg.simd_count++;
        }

        /**
         * RAII helper to time a scope.
         */
        class scoped_timer {
        public:
            scoped_timer(const std::string& name, std::size_t count, bool simd)
                : m_name(name), m_count(count), m_simd(simd),
                  m_start(std::chrono::high_resolution_clock::now()) {}
            ~scoped_timer() {
                auto end = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(end - m_start).count();
                xframe_tracer::instance().record(m_name, m_count, us, m_simd);
            }
        private:
            std::string m_name;
            std::size_t m_count;
            bool m_simd;
            std::chrono::high_resolution_clock::time_point m_start;
        };

        /**
         * Export all recorded traces to a CSV file.
         */
        void save_csv(const std::string& filename) const {
            std::ofstream f(filename);
            f << "timestamp,operation,size,duration_us,simd\n";
            for (auto& r : m_records) {
                f << std::chrono::duration_cast<std::chrono::microseconds>(
                       r.timestamp.time_since_epoch()).count() << ","
                  << r.operation << "," << r.size << ","
                  << r.duration_us << "," << (r.simd_used ? "1" : "0") << "\n";
            }
        }

        /**
         * Export aggregate statistics to a CSV file.
         */
        void save_aggregates(const std::string& filename) const {
            std::ofstream f(filename);
            f << "operation,count,total_elements,total_time_us,simd_count\n";
            for (auto& [op, agg] : m_aggregates) {
                f << op << "," << agg.count << "," << agg.total_elements << ","
                  << agg.total_time_us << "," << agg.simd_count << "\n";
            }
        }

        /**
         * Print a summary of operations to stdout.
         */
        void print_summary() const {
            std::cout << "=== xframe trace summary ===\n";
            for (auto& [op, agg] : m_aggregates) {
                std::cout << op << ": " << agg.count << " calls, "
                          << agg.total_elements << " elements, "
                          << agg.total_time_us << " µs, "
                          << "SIMD used " << (agg.simd_count * 100.0 / std::max(agg.count, 1u)) << "%\n";
            }
        }

    private:
        xframe_tracer() = default;
        bool m_enabled = false;
        trace_level m_level = trace_level::info;
        mutable std::mutex m_mutex;
        std::vector<trace_record> m_records;

        struct aggregate {
            std::size_t count = 0;
            std::size_t total_elements = 0;
            double total_time_us = 0.0;
            std::size_t simd_count = 0;
        };
        std::unordered_map<std::string, aggregate> m_aggregates;
    };

    // Convenience macros for tracing
    #ifndef XFRAME_NO_TRACE
        #define XFRAME_TRACE_SCOPED(name, count, simd) \
            xframe::trace::xframe_tracer::scoped_timer _xframe_timer_(name, count, simd)
    #else
        #define XFRAME_TRACE_SCOPED(name, count, simd) ((void)0)
    #endif

} // namespace trace
} // namespace xframe

#endif // XFRAME_XFRAME_TRACE_HPP