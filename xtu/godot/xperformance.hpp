// include/xtu/godot/xperformance.hpp
// xtensor-unified - Performance monitoring for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPERFORMANCE_HPP
#define XTU_GODOT_XPERFORMANCE_HPP

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xcore.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace main {

// #############################################################################
// Forward declarations
// #############################################################################
class Performance;

// #############################################################################
// Performance monitor types (matching Godot's Performance.Monitor enum)
// #############################################################################
enum class PerformanceMonitor : uint8_t {
    TIME_FPS = 0,
    TIME_PROCESS = 1,
    TIME_PHYSICS_PROCESS = 2,
    TIME_NAVIGATION_PROCESS = 3,
    MEMORY_STATIC = 4,
    MEMORY_STATIC_MAX = 5,
    MEMORY_MESSAGE_BUFFER_MAX = 6,
    OBJECT_COUNT = 7,
    OBJECT_RESOURCE_COUNT = 8,
    OBJECT_NODE_COUNT = 9,
    OBJECT_ORPHAN_NODE_COUNT = 10,
    RENDER_TOTAL_OBJECTS_IN_FRAME = 11,
    RENDER_TOTAL_PRIMITIVES_IN_FRAME = 12,
    RENDER_TOTAL_DRAW_CALLS_IN_FRAME = 13,
    RENDER_VIDEO_MEM_USED = 14,
    RENDER_TEXTURE_MEM_USED = 15,
    RENDER_BUFFER_MEM_USED = 16,
    PHYSICS_2D_ACTIVE_OBJECTS = 17,
    PHYSICS_2D_ISLAND_COUNT = 18,
    PHYSICS_3D_ACTIVE_OBJECTS = 19,
    PHYSICS_3D_ISLAND_COUNT = 20,
    AUDIO_OUTPUT_LATENCY = 21,
    NAVIGATION_ACTIVE_MAPS = 22,
    NAVIGATION_REGION_COUNT = 23,
    NAVIGATION_AGENT_COUNT = 24,
    MONITOR_MAX = 25
};

// #############################################################################
// Performance - Global performance monitoring singleton
// #############################################################################
class Performance : public Object {
    XTU_GODOT_REGISTER_CLASS(Performance, Object)

private:
    static Performance* s_singleton;
    
    std::unordered_map<PerformanceMonitor, std::atomic<double>> m_monitors;
    std::unordered_map<String, std::atomic<double>> m_custom_monitors;
    mutable std::mutex m_mutex;
    
    // Rolling averages for FPS
    static constexpr size_t FPS_HISTORY_SIZE = 60;
    std::array<double, FPS_HISTORY_SIZE> m_fps_history{};
    size_t m_fps_index = 0;
    double m_fps_accum = 0.0;
    int m_fps_samples = 0;

public:
    static Performance* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("Performance"); }

    Performance() {
        s_singleton = this;
        initialize_monitors();
    }

    ~Performance() { s_singleton = nullptr; }

    // Get monitor value
    double get_monitor(PerformanceMonitor monitor) const {
        auto it = m_monitors.find(monitor);
        if (it != m_monitors.end()) {
            return it->second.load();
        }
        return 0.0;
    }

    // Set monitor value (for internal engine use)
    void set_monitor(PerformanceMonitor monitor, double value) {
        auto it = m_monitors.find(monitor);
        if (it != m_monitors.end()) {
            it->second.store(value);
        }
    }

    // Custom monitors (for scripts)
    void add_custom_monitor(const String& name, double initial_value = 0.0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_custom_monitors[name].store(initial_value);
    }

    void remove_custom_monitor(const String& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_custom_monitors.erase(name);
    }

    bool has_custom_monitor(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_custom_monitors.find(name) != m_custom_monitors.end();
    }

    double get_custom_monitor(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_custom_monitors.find(name);
        return it != m_custom_monitors.end() ? it->second.load() : 0.0;
    }

    void set_custom_monitor(const String& name, double value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_custom_monitors.find(name);
        if (it != m_custom_monitors.end()) {
            it->second.store(value);
        }
    }

    std::vector<String> get_custom_monitor_names() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> names;
        for (const auto& kv : m_custom_monitors) {
            names.push_back(kv.first);
        }
        return names;
    }

    // Update frame-based monitors
    void update_frame(double delta) {
        // Update FPS
        m_fps_accum += delta;
        m_fps_samples++;
        if (m_fps_accum >= 1.0) {
            double fps = static_cast<double>(m_fps_samples) / m_fps_accum;
            m_fps_history[m_fps_index] = fps;
            m_fps_index = (m_fps_index + 1) % FPS_HISTORY_SIZE;
            
            // Compute average FPS
            double avg_fps = 0.0;
            size_t count = 0;
            for (size_t i = 0; i < FPS_HISTORY_SIZE; ++i) {
                if (m_fps_history[i] > 0.0) {
                    avg_fps += m_fps_history[i];
                    ++count;
                }
            }
            if (count > 0) {
                set_monitor(PerformanceMonitor::TIME_FPS, avg_fps / static_cast<double>(count));
            }
            
            m_fps_accum = 0.0;
            m_fps_samples = 0;
        }
    }

    // Convenience accessors for common monitors
    double get_fps() const { return get_monitor(PerformanceMonitor::TIME_FPS); }
    double get_process_time() const { return get_monitor(PerformanceMonitor::TIME_PROCESS); }
    double get_physics_time() const { return get_monitor(PerformanceMonitor::TIME_PHYSICS_PROCESS); }
    double get_static_memory() const { return get_monitor(PerformanceMonitor::MEMORY_STATIC); }
    double get_static_memory_max() const { return get_monitor(PerformanceMonitor::MEMORY_STATIC_MAX); }
    double get_object_count() const { return get_monitor(PerformanceMonitor::OBJECT_COUNT); }
    double get_resource_count() const { return get_monitor(PerformanceMonitor::OBJECT_RESOURCE_COUNT); }
    double get_node_count() const { return get_monitor(PerformanceMonitor::OBJECT_NODE_COUNT); }
    double get_draw_calls() const { return get_monitor(PerformanceMonitor::RENDER_TOTAL_DRAW_CALLS_IN_FRAME); }

private:
    void initialize_monitors() {
        for (int i = 0; i < static_cast<int>(PerformanceMonitor::MONITOR_MAX); ++i) {
            m_monitors[static_cast<PerformanceMonitor>(i)].store(0.0);
        }
    }
};

} // namespace main

// Bring into main namespace
using main::Performance;
using main::PerformanceMonitor;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPERFORMANCE_HPP