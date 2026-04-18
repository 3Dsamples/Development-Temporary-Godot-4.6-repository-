// include/xtu/godot/xprofiling.hpp
// xtensor-unified - Profiling and debugging for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPROFILING_HPP
#define XTU_GODOT_XPROFILING_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class EngineDebugger;
class EngineProfiler;
class ProfilerFrame;
class MemoryTracker;
class Logger;
class CallStack;

// #############################################################################
// Log verbosity levels
// #############################################################################
enum class LogLevel : uint8_t {
    LEVEL_DEBUG = 0,
    LEVEL_INFO = 1,
    LEVEL_WARNING = 2,
    LEVEL_ERROR = 3,
    LEVEL_FATAL = 4
};

// #############################################################################
// Profiler category types
// #############################################################################
enum class ProfilerCategory : uint16_t {
    CATEGORY_ENGINE = 0,
    CATEGORY_PHYSICS = 1,
    CATEGORY_RENDERING = 2,
    CATEGORY_AUDIO = 3,
    CATEGORY_SCRIPT = 4,
    CATEGORY_NETWORK = 5,
    CATEGORY_IO = 6,
    CATEGORY_CUSTOM = 1000
};

// #############################################################################
// Memory allocation types
// #############################################################################
enum class MemoryAllocType : uint8_t {
    ALLOC_STANDARD = 0,
    ALLOC_POOL = 1,
    ALLOC_STATIC = 2,
    ALLOC_GPU = 3,
    ALLOC_AUDIO = 4
};

// #############################################################################
// Debugger message types
// #############################################################################
enum class DebuggerMessageType : uint8_t {
    MSG_NONE = 0,
    MSG_PROFILE_FRAME = 1,
    MSG_MEMORY_SNAPSHOT = 2,
    MSG_BREAKPOINT_HIT = 3,
    MSG_LOG_MESSAGE = 4,
    MSG_PERFORMANCE_WARNING = 5,
    MSG_CUSTOM = 6
};

// #############################################################################
// ScopedTimer - RAII timer for function profiling
// #############################################################################
class ScopedTimer {
private:
    std::chrono::high_resolution_clock::time_point m_start;
    const char* m_name;
    ProfilerCategory m_category;
    static thread_local std::stack<ScopedTimer*> s_active_timers;

public:
    ScopedTimer(const char* name, ProfilerCategory category = ProfilerCategory::CATEGORY_ENGINE)
        : m_name(name), m_category(category) {
        m_start = std::chrono::high_resolution_clock::now();
        s_active_timers.push(this);
    }

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();
        EngineProfiler::get_singleton()->add_timing(m_name, m_category, duration);
        s_active_timers.pop();
    }

    static const ScopedTimer* current() {
        return s_active_timers.empty() ? nullptr : s_active_timers.top();
    }
};

thread_local std::stack<ScopedTimer*> ScopedTimer::s_active_timers;

// #############################################################################
// ProfilerFrame - Single frame profiling data
// #############################################################################
struct ProfilerFrame {
    uint64_t frame_number = 0;
    double frame_time = 0.0;
    double physics_time = 0.0;
    double render_time = 0.0;
    double audio_time = 0.0;
    double script_time = 0.0;
    double idle_time = 0.0;
    size_t draw_calls = 0;
    size_t vertices_rendered = 0;
    size_t objects_culled = 0;
    size_t memory_used = 0;
    size_t memory_peak = 0;
    std::unordered_map<std::string, double> custom_timings;
    std::unordered_map<std::string, size_t> counters;

    void reset() {
        frame_number = 0;
        frame_time = 0.0;
        physics_time = 0.0;
        render_time = 0.0;
        audio_time = 0.0;
        script_time = 0.0;
        idle_time = 0.0;
        draw_calls = 0;
        vertices_rendered = 0;
        objects_culled = 0;
        memory_used = 0;
        memory_peak = 0;
        custom_timings.clear();
        counters.clear();
    }
};

// #############################################################################
// MemoryAllocation - Single allocation record
// #############################################################################
struct MemoryAllocation {
    void* ptr = nullptr;
    size_t size = 0;
    MemoryAllocType type = MemoryAllocType::ALLOC_STANDARD;
    const char* file = nullptr;
    int line = 0;
    uint64_t timestamp = 0;
    std::string tag;
};

// #############################################################################
// MemoryTracker - Memory allocation tracking
// #############################################################################
class MemoryTracker {
private:
    static MemoryTracker* s_singleton;
    std::unordered_map<void*, MemoryAllocation> m_allocations;
    std::atomic<size_t> m_total_allocated{0};
    std::atomic<size_t> m_peak_allocated{0};
    std::atomic<size_t> m_allocation_count{0};
    std::mutex m_mutex;
    bool m_tracking_enabled = true;
    std::ofstream m_log_file;
    std::queue<MemoryAllocation> m_pending_logs;
    std::thread m_log_worker;
    std::atomic<bool> m_log_worker_running{false};

public:
    static MemoryTracker* get_singleton() { return s_singleton; }

    MemoryTracker() {
        s_singleton = this;
        start_log_worker();
    }

    ~MemoryTracker() {
        m_log_worker_running = false;
        if (m_log_worker.joinable()) m_log_worker.join();
        s_singleton = nullptr;
    }

    void set_tracking_enabled(bool enabled) { m_tracking_enabled = enabled; }
    bool is_tracking_enabled() const { return m_tracking_enabled; }

    void* allocate(size_t size, MemoryAllocType type = MemoryAllocType::ALLOC_STANDARD,
                   const char* file = nullptr, int line = 0, const std::string& tag = "") {
        void* ptr = std::malloc(size);
        if (!ptr) return nullptr;

        if (m_tracking_enabled) {
            std::lock_guard<std::mutex> lock(m_mutex);
            MemoryAllocation alloc;
            alloc.ptr = ptr;
            alloc.size = size;
            alloc.type = type;
            alloc.file = file;
            alloc.line = line;
            alloc.timestamp = get_timestamp();
            alloc.tag = tag;
            m_allocations[ptr] = alloc;

            size_t current = m_total_allocated.fetch_add(size) + size;
            size_t peak = m_peak_allocated.load();
            while (current > peak && !m_peak_allocated.compare_exchange_weak(peak, current)) {}

            ++m_allocation_count;
        }

        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        std::free(ptr);

        if (m_tracking_enabled) {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_allocations.find(ptr);
            if (it != m_allocations.end()) {
                m_total_allocated -= it->second.size;
                m_allocations.erase(it);
            }
        }
    }

    template <typename T>
    T* allocate_t(size_t count = 1, const char* file = nullptr, int line = 0) {
        return static_cast<T*>(allocate(sizeof(T) * count, MemoryAllocType::ALLOC_STANDARD, file, line));
    }

    size_t get_total_allocated() const { return m_total_allocated.load(); }
    size_t get_peak_allocated() const { return m_peak_allocated.load(); }
    size_t get_allocation_count() const { return m_allocation_count.load(); }

    std::vector<MemoryAllocation> get_allocation_snapshot() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<MemoryAllocation> result;
        result.reserve(m_allocations.size());
        for (const auto& kv : m_allocations) {
            result.push_back(kv.second);
        }
        return result;
    }

    void dump_to_file(const std::string& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::ofstream file(path);
        if (!file.is_open()) return;

        file << "=== Memory Allocation Dump ===\n";
        file << "Total Allocated: " << format_bytes(m_total_allocated.load()) << "\n";
        file << "Peak Allocated: " << format_bytes(m_peak_allocated.load()) << "\n";
        file << "Active Allocations: " << m_allocations.size() << "\n\n";

        for (const auto& kv : m_allocations) {
            const auto& a = kv.second;
            file << "[" << a.ptr << "] " << a.size << " bytes";
            if (a.file) file << " @ " << a.file << ":" << a.line;
            if (!a.tag.empty()) file << " [" << a.tag << "]";
            file << "\n";
        }
    }

    void check_leaks() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_allocations.empty()) {
            std::cerr << "WARNING: " << m_allocations.size() << " memory leaks detected!\n";
            for (const auto& kv : m_allocations) {
                const auto& a = kv.second;
                std::cerr << "  Leak: " << a.size << " bytes";
                if (a.file) std::cerr << " at " << a.file << ":" << a.line;
                std::cerr << "\n";
            }
        }
    }

private:
    uint64_t get_timestamp() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    std::string format_bytes(size_t bytes) const {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit = 0;
        double size = static_cast<double>(bytes);
        while (size >= 1024.0 && unit < 4) {
            size /= 1024.0;
            ++unit;
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit]);
        return std::string(buf);
    }

    void start_log_worker() {
        m_log_worker_running = true;
        m_log_worker = std::thread([this]() {
            while (m_log_worker_running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                // Process pending logs
            }
        });
    }
};

// #############################################################################
// Logger - Logging system
// #############################################################################
class Logger {
public:
    struct LogEntry {
        LogLevel level = LogLevel::LEVEL_INFO;
        std::string message;
        const char* file = nullptr;
        int line = 0;
        const char* function = nullptr;
        uint64_t timestamp = 0;
    };

private:
    static Logger* s_singleton;
    std::vector<LogEntry> m_log_history;
    std::queue<LogEntry> m_pending_logs;
    std::mutex m_mutex;
    LogLevel m_min_level = LogLevel::LEVEL_INFO;
    bool m_log_to_file = false;
    std::string m_log_file_path;
    std::ofstream m_log_file;
    bool m_log_to_console = true;
    std::unordered_map<std::string, bool> m_category_filters;

public:
    static Logger* get_singleton() { return s_singleton; }

    Logger() {
        s_singleton = this;
    }

    ~Logger() {
        if (m_log_file.is_open()) m_log_file.close();
        s_singleton = nullptr;
    }

    void set_min_level(LogLevel level) { m_min_level = level; }
    LogLevel get_min_level() const { return m_min_level; }

    void set_log_to_file(bool enabled, const std::string& path = "") {
        m_log_to_file = enabled;
        m_log_file_path = path;
        if (enabled && !path.empty()) {
            if (m_log_file.is_open()) m_log_file.close();
            m_log_file.open(path, std::ios::app);
        }
    }

    void set_log_to_console(bool enabled) { m_log_to_console = enabled; }

    void set_category_filter(const std::string& category, bool enabled) {
        m_category_filters[category] = enabled;
    }

    void log(LogLevel level, const std::string& message, const char* file = nullptr,
             int line = 0, const char* function = nullptr, const std::string& category = "") {
        if (level < m_min_level) return;
        if (!category.empty()) {
            auto it = m_category_filters.find(category);
            if (it != m_category_filters.end() && !it->second) return;
        }

        LogEntry entry;
        entry.level = level;
        entry.message = message;
        entry.file = file;
        entry.line = line;
        entry.function = function;
        entry.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_log_history.push_back(entry);
            if (m_log_history.size() > 10000) {
                m_log_history.erase(m_log_history.begin(), m_log_history.begin() + 5000);
            }
        }

        if (m_log_to_console) {
            print_log(entry);
        }

        if (m_log_to_file && m_log_file.is_open()) {
            write_log_to_file(entry);
        }
    }

    void debug(const std::string& msg, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        log(LogLevel::LEVEL_DEBUG, msg, file, line, func);
    }

    void info(const std::string& msg, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        log(LogLevel::LEVEL_INFO, msg, file, line, func);
    }

    void warning(const std::string& msg, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        log(LogLevel::LEVEL_WARNING, msg, file, line, func);
    }

    void error(const std::string& msg, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        log(LogLevel::LEVEL_ERROR, msg, file, line, func);
    }

    void fatal(const std::string& msg, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        log(LogLevel::LEVEL_FATAL, msg, file, line, func);
    }

    std::vector<LogEntry> get_log_history() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_log_history;
    }

    void clear_history() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_log_history.clear();
    }

private:
    void print_log(const LogEntry& entry) {
        std::ostream& out = (entry.level >= LogLevel::LEVEL_WARNING) ? std::cerr : std::cout;
        out << "[" << level_to_string(entry.level) << "] " << entry.message;
        if (entry.file) {
            out << " (" << entry.file << ":" << entry.line << ")";
        }
        out << std::endl;
    }

    void write_log_to_file(const LogEntry& entry) {
        m_log_file << "[" << level_to_string(entry.level) << "] "
                   << format_timestamp(entry.timestamp) << " "
                   << entry.message;
        if (entry.file) {
            m_log_file << " (" << entry.file << ":" << entry.line << ")";
        }
        m_log_file << std::endl;
    }

    const char* level_to_string(LogLevel level) const {
        switch (level) {
            case LogLevel::LEVEL_DEBUG: return "DEBUG";
            case LogLevel::LEVEL_INFO: return "INFO";
            case LogLevel::LEVEL_WARNING: return "WARN";
            case LogLevel::LEVEL_ERROR: return "ERROR";
            case LogLevel::LEVEL_FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }

    std::string format_timestamp(uint64_t us) const {
        auto sec = us / 1000000;
        auto min = sec / 60;
        auto hour = min / 60;
        char buf[32];
        snprintf(buf, sizeof(buf), "%02zu:%02zu:%02zu.%06zu",
                 static_cast<size_t>(hour % 24),
                 static_cast<size_t>(min % 60),
                 static_cast<size_t>(sec % 60),
                 static_cast<size_t>(us % 1000000));
        return std::string(buf);
    }
};

// #############################################################################
// EngineProfiler - Performance profiling
// #############################################################################
class EngineProfiler : public Object {
    XTU_GODOT_REGISTER_CLASS(EngineProfiler, Object)

private:
    static EngineProfiler* s_singleton;
    std::vector<ProfilerFrame> m_frame_history;
    ProfilerFrame m_current_frame;
    std::unordered_map<std::string, std::vector<uint64_t>> m_timings;
    std::chrono::high_resolution_clock::time_point m_frame_start;
    std::mutex m_mutex;
    bool m_profiling_enabled = false;
    size_t m_max_history_frames = 600;
    uint64_t m_frame_number = 0;

public:
    static EngineProfiler* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EngineProfiler"); }

    EngineProfiler() { s_singleton = this; }
    ~EngineProfiler() { s_singleton = nullptr; }

    void set_profiling_enabled(bool enabled) { m_profiling_enabled = enabled; }
    bool is_profiling_enabled() const { return m_profiling_enabled; }

    void begin_frame() {
        if (!m_profiling_enabled) return;
        m_frame_start = std::chrono::high_resolution_clock::now();
        m_current_frame.reset();
        m_current_frame.frame_number = ++m_frame_number;
    }

    void end_frame() {
        if (!m_profiling_enabled) return;
        auto end = std::chrono::high_resolution_clock::now();
        m_current_frame.frame_time = std::chrono::duration<double>(end - m_frame_start).count();

        std::lock_guard<std::mutex> lock(m_mutex);
        m_frame_history.push_back(m_current_frame);
        if (m_frame_history.size() > m_max_history_frames) {
            m_frame_history.erase(m_frame_history.begin());
        }
    }

    void add_timing(const char* name, ProfilerCategory category, int64_t microseconds) {
        if (!m_profiling_enabled) return;
        std::string key(name);
        m_timings[key].push_back(static_cast<uint64_t>(microseconds));
        if (m_timings[key].size() > 1000) {
            m_timings[key].erase(m_timings[key].begin());
        }

        switch (category) {
            case ProfilerCategory::CATEGORY_PHYSICS:
                m_current_frame.physics_time += static_cast<double>(microseconds) / 1000000.0;
                break;
            case ProfilerCategory::CATEGORY_RENDERING:
                m_current_frame.render_time += static_cast<double>(microseconds) / 1000000.0;
                break;
            case ProfilerCategory::CATEGORY_AUDIO:
                m_current_frame.audio_time += static_cast<double>(microseconds) / 1000000.0;
                break;
            case ProfilerCategory::CATEGORY_SCRIPT:
                m_current_frame.script_time += static_cast<double>(microseconds) / 1000000.0;
                break;
            default:
                break;
        }
        m_current_frame.custom_timings[name] += static_cast<double>(microseconds) / 1000000.0;
    }

    void increment_counter(const std::string& name, size_t count = 1) {
        if (!m_profiling_enabled) return;
        m_current_frame.counters[name] += count;
    }

    void set_draw_calls(size_t count) { m_current_frame.draw_calls = count; }
    void set_vertices_rendered(size_t count) { m_current_frame.vertices_rendered = count; }
    void set_objects_culled(size_t count) { m_current_frame.objects_culled = count; }

    void update_memory_stats() {
        m_current_frame.memory_used = MemoryTracker::get_singleton()->get_total_allocated();
        m_current_frame.memory_peak = MemoryTracker::get_singleton()->get_peak_allocated();
    }

    const ProfilerFrame& get_current_frame() const { return m_current_frame; }

    std::vector<ProfilerFrame> get_frame_history() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_frame_history;
    }

    double get_average_frame_time(size_t frames = 60) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_frame_history.empty()) return 0.0;
        size_t count = std::min(frames, m_frame_history.size());
        double total = 0.0;
        for (size_t i = m_frame_history.size() - count; i < m_frame_history.size(); ++i) {
            total += m_frame_history[i].frame_time;
        }
        return total / static_cast<double>(count);
    }

    double get_fps() const {
        double avg = get_average_frame_time(10);
        return avg > 0.0 ? 1.0 / avg : 0.0;
    }

    std::string generate_report() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::ostringstream oss;
        oss << "=== Engine Profiler Report ===\n";
        oss << "FPS: " << std::fixed << std::setprecision(1) << get_fps() << "\n";
        oss << "Frame Time (avg): " << get_average_frame_time() * 1000.0 << " ms\n";
        oss << "Physics Time: " << m_current_frame.physics_time * 1000.0 << " ms\n";
        oss << "Render Time: " << m_current_frame.render_time * 1000.0 << " ms\n";
        oss << "Script Time: " << m_current_frame.script_time * 1000.0 << " ms\n";
        oss << "Draw Calls: " << m_current_frame.draw_calls << "\n";
        oss << "Memory Used: " << format_bytes(m_current_frame.memory_used) << "\n";
        oss << "Memory Peak: " << format_bytes(m_current_frame.memory_peak) << "\n";
        return oss.str();
    }

private:
    std::string format_bytes(size_t bytes) const {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit = 0;
        double size = static_cast<double>(bytes);
        while (size >= 1024.0 && unit < 3) {
            size /= 1024.0;
            ++unit;
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit]);
        return std::string(buf);
    }
};

// #############################################################################
// CallStack - Stack trace collection
// #############################################################################
class CallStack {
public:
    struct Frame {
        std::string function;
        std::string file;
        int line = 0;
        void* address = nullptr;
    };

private:
    static thread_local std::vector<Frame> s_call_stack;
    static thread_local std::string s_current_function;

public:
    static void push(const std::string& function, const char* file = nullptr, int line = 0) {
        Frame frame;
        frame.function = function;
        frame.file = file ? file : "";
        frame.line = line;
        s_call_stack.push_back(frame);
        s_current_function = function;
    }

    static void pop() {
        if (!s_call_stack.empty()) {
            s_call_stack.pop_back();
            s_current_function = s_call_stack.empty() ? "" : s_call_stack.back().function;
        }
    }

    static const std::string& current_function() { return s_current_function; }

    static std::vector<Frame> get_stack() { return s_call_stack; }

    static std::string format_stack() {
        std::ostringstream oss;
        oss << "Call Stack:\n";
        for (auto it = s_call_stack.rbegin(); it != s_call_stack.rend(); ++it) {
            oss << "  at " << it->function;
            if (!it->file.empty()) {
                oss << " (" << it->file << ":" << it->line << ")";
            }
            oss << "\n";
        }
        return oss.str();
    }
};

thread_local std::vector<CallStack::Frame> CallStack::s_call_stack;
thread_local std::string CallStack::s_current_function;

// #############################################################################
// ScopedFunction - RAII function call tracking
// #############################################################################
class ScopedFunction {
public:
    ScopedFunction(const std::string& name, const char* file = nullptr, int line = 0) {
        CallStack::push(name, file, line);
    }

    ~ScopedFunction() {
        CallStack::pop();
    }
};

// #############################################################################
// EngineDebugger - Remote debugging
// #############################################################################
class EngineDebugger : public Object {
    XTU_GODOT_REGISTER_CLASS(EngineDebugger, Object)

private:
    static EngineDebugger* s_singleton;
    bool m_active = false;
    int m_port = 6007;
    std::vector<String> m_breakpoints;
    std::unordered_map<String, Variant> m_watched_expressions;
    std::mutex m_mutex;

public:
    static EngineDebugger* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EngineDebugger"); }

    EngineDebugger() { s_singleton = this; }
    ~EngineDebugger() { s_singleton = nullptr; }

    void start(int port = 6007) {
        m_port = port;
        m_active = true;
    }

    void stop() {
        m_active = false;
    }

    bool is_active() const { return m_active; }

    void add_breakpoint(const String& file, int line) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_breakpoints.push_back(file + ":" + String::num(line));
    }

    void remove_breakpoint(const String& file, int line) {
        std::lock_guard<std::mutex> lock(m_mutex);
        String key = file + ":" + String::num(line);
        auto it = std::find(m_breakpoints.begin(), m_breakpoints.end(), key);
        if (it != m_breakpoints.end()) m_breakpoints.erase(it);
    }

    bool has_breakpoint(const String& file, int line) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        String key = file + ":" + String::num(line);
        return std::find(m_breakpoints.begin(), m_breakpoints.end(), key) != m_breakpoints.end();
    }

    void send_message(DebuggerMessageType type, const Variant& data) {
        // Send to connected debugger client
    }

    void send_profile_frame(const ProfilerFrame& frame) {
        // Send frame data
    }
};

// #############################################################################
// Profiling macros
// #############################################################################
#define XTU_PROFILE_SCOPE(name) \
    ::xtu::godot::ScopedTimer _xtu_timer_##__LINE__(name, ::xtu::godot::ProfilerCategory::CATEGORY_ENGINE)

#define XTU_PROFILE_SCOPE_CAT(name, category) \
    ::xtu::godot::ScopedTimer _xtu_timer_##__LINE__(name, category)

#define XTU_PROFILE_FUNCTION() \
    ::xtu::godot::ScopedTimer _xtu_timer_##__LINE__(__FUNCTION__, ::xtu::godot::ProfilerCategory::CATEGORY_ENGINE)

#define XTU_TRACK_FUNCTION() \
    ::xtu::godot::ScopedFunction _xtu_func_##__LINE__(__FUNCTION__, __FILE__, __LINE__)

#define XTU_LOG_DEBUG(msg) \
    ::xtu::godot::Logger::get_singleton()->debug(msg, __FILE__, __LINE__, __FUNCTION__)

#define XTU_LOG_INFO(msg) \
    ::xtu::godot::Logger::get_singleton()->info(msg, __FILE__, __LINE__, __FUNCTION__)

#define XTU_LOG_WARNING(msg) \
    ::xtu::godot::Logger::get_singleton()->warning(msg, __FILE__, __LINE__, __FUNCTION__)

#define XTU_LOG_ERROR(msg) \
    ::xtu::godot::Logger::get_singleton()->error(msg, __FILE__, __LINE__, __FUNCTION__)

#define XTU_LOG_FATAL(msg) \
    ::xtu::godot::Logger::get_singleton()->fatal(msg, __FILE__, __LINE__, __FUNCTION__)

} // namespace godot

// Bring into main namespace
using godot::EngineDebugger;
using godot::EngineProfiler;
using godot::MemoryTracker;
using godot::Logger;
using godot::CallStack;
using godot::ProfilerFrame;
using godot::LogLevel;
using godot::ProfilerCategory;
using godot::ScopedTimer;
using godot::ScopedFunction;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPROFILING_HPP