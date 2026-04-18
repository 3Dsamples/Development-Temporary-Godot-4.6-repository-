// include/xtu/godot/xmain_timer.hpp
// xtensor-unified - Frame pacing and timer synchronization for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XMAIN_TIMER_HPP
#define XTU_GODOT_XMAIN_TIMER_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>

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
class MainTimerSync;

// #############################################################################
// Frame pacing mode
// #############################################################################
enum class FramePacingMode : uint8_t {
    PACING_IDLE = 0,      // Sleep when idle
    PACING_VSYNC = 1,     // Wait for vsync
    PACING_FIXED = 2,     // Fixed timestep
    PACING_UNLIMITED = 3  // No frame limit
};

// #############################################################################
// MainTimerSync - Frame pacing and synchronization
// #############################################################################
class MainTimerSync : public Object {
    XTU_GODOT_REGISTER_CLASS(MainTimerSync, Object)

private:
    static MainTimerSync* s_singleton;
    
    FramePacingMode m_pacing_mode = FramePacingMode::PACING_VSYNC;
    uint32_t m_target_fps = 60;
    bool m_use_fixed_timestep = false;
    uint32_t m_fixed_timestep_hz = 60;
    double m_time_scale = 1.0;
    bool m_low_processor_usage = true;
    
    std::chrono::high_resolution_clock::time_point m_frame_start;
    std::chrono::high_resolution_clock::time_point m_last_frame_time;
    double m_frame_delta = 0.0;
    double m_physics_delta = 1.0 / 60.0;
    double m_accumulated_time = 0.0;
    uint64_t m_frame_count = 0;
    
    mutable std::mutex m_mutex;
    std::atomic<bool> m_exit_requested{false};

public:
    static MainTimerSync* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("MainTimerSync"); }

    MainTimerSync() {
        s_singleton = this;
        m_frame_start = std::chrono::high_resolution_clock::now();
        m_last_frame_time = m_frame_start;
    }

    ~MainTimerSync() { s_singleton = nullptr; }

    // Configuration
    void set_pacing_mode(FramePacingMode mode) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pacing_mode = mode;
    }

    FramePacingMode get_pacing_mode() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_pacing_mode;
    }

    void set_target_fps(uint32_t fps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_target_fps = std::max(1u, fps);
    }

    uint32_t get_target_fps() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_target_fps;
    }

    void set_use_fixed_timestep(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_use_fixed_timestep = use;
    }

    bool get_use_fixed_timestep() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_use_fixed_timestep;
    }

    void set_fixed_timestep_hz(uint32_t hz) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_fixed_timestep_hz = std::max(1u, hz);
        m_physics_delta = 1.0 / static_cast<double>(m_fixed_timestep_hz);
    }

    uint32_t get_fixed_timestep_hz() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_fixed_timestep_hz;
    }

    void set_time_scale(double scale) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_time_scale = std::max(0.0, scale);
    }

    double get_time_scale() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_time_scale;
    }

    void set_low_processor_usage(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_low_processor_usage = enable;
    }

    bool get_low_processor_usage() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_low_processor_usage;
    }

    // Frame timing
    void begin_frame() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto now = std::chrono::high_resolution_clock::now();
        m_frame_delta = std::chrono::duration<double>(now - m_last_frame_time).count();
        m_last_frame_time = now;
        ++m_frame_count;
    }

    void end_frame() {
        std::lock_guard<std::mutex> lock(m_mutex);
        pace_frame();
    }

    double get_frame_delta() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_frame_delta * m_time_scale;
    }

    double get_physics_delta() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_physics_delta * m_time_scale;
    }

    uint64_t get_frame_count() const {
        return m_frame_count;
    }

    // Fixed timestep accumulation
    bool should_physics_step() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_use_fixed_timestep) {
            return true;
        }
        m_accumulated_time += m_frame_delta * m_time_scale;
        if (m_accumulated_time >= m_physics_delta) {
            m_accumulated_time -= m_physics_delta;
            return true;
        }
        return false;
    }

    void request_exit() {
        m_exit_requested = true;
    }

    bool is_exit_requested() const {
        return m_exit_requested;
    }

    // Sleep to maintain target frame rate
    void pace_frame() {
        if (m_pacing_mode == FramePacingMode::PACING_UNLIMITED) {
            return;
        }

        if (m_pacing_mode == FramePacingMode::PACING_FIXED) {
            double target_frame_time = 1.0 / static_cast<double>(m_target_fps);
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - m_frame_start).count();
            double remaining = target_frame_time - elapsed;

            if (remaining > 0.0) {
                if (m_low_processor_usage) {
                    // Sleep for most of the time, then spin for precision
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(remaining * 0.9));
                    // Spin for remaining
                    while (remaining > 0.0) {
                        now = std::chrono::high_resolution_clock::now();
                        elapsed = std::chrono::duration<double>(now - m_frame_start).count();
                        remaining = target_frame_time - elapsed;
                    }
                } else {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(remaining));
                }
            }
            m_frame_start = std::chrono::high_resolution_clock::now();
        } else if (m_pacing_mode == FramePacingMode::PACING_IDLE) {
            if (m_low_processor_usage) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        // PACING_VSYNC is handled by the graphics driver
    }

    // Wait for a specified duration (platform-aware)
    void delay_usec(uint32_t usec) {
        std::this_thread::sleep_for(std::chrono::microseconds(usec));
    }

    void delay_msec(uint32_t msec) {
        std::this_thread::sleep_for(std::chrono::milliseconds(msec));
    }
};

} // namespace main

// Bring into main namespace
using main::MainTimerSync;
using main::FramePacingMode;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XMAIN_TIMER_HPP