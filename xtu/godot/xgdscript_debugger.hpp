// include/xtu/godot/xgdscript_debugger.hpp
// xtensor-unified - GDScript Debugger for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XGDSCRIPT_DEBUGGER_HPP
#define XTU_GODOT_XGDSCRIPT_DEBUGGER_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xgdscript.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace gdscript {

// #############################################################################
// Forward declarations
// #############################################################################
class GDScriptDebugger;
class DebuggerBreakpoint;
class DebuggerStackFrame;
class DebuggerVariable;

// #############################################################################
// Debugger state
// #############################################################################
enum class DebuggerState : uint8_t {
    STATE_RUNNING = 0,
    STATE_PAUSED = 1,
    STATE_STEPPING = 2,
    STATE_BREAK = 3
};

// #############################################################################
// Step mode
// #############################################################################
enum class StepMode : uint8_t {
    STEP_NONE = 0,
    STEP_INTO = 1,
    STEP_OVER = 2,
    STEP_OUT = 3
};

// #############################################################################
// DebuggerBreakpoint - Breakpoint information
// #############################################################################
struct DebuggerBreakpoint {
    String file;
    int line = 0;
    bool enabled = true;
    String condition;
    int hit_count = 0;
    int ignore_count = 0;
    int temporary_id = 0;
};

// #############################################################################
// DebuggerStackFrame - Call stack frame
// #############################################################################
struct DebuggerStackFrame {
    String function_name;
    String script_path;
    int line = 0;
    int column = 0;
    uint64_t instance_id = 0;
    std::vector<String> local_vars;
    std::vector<Variant> local_values;
};

// #############################################################################
// DebuggerVariable - Inspected variable
// #############################################################################
struct DebuggerVariable {
    String name;
    Variant value;
    VariantType type = VariantType::NIL;
    std::vector<DebuggerVariable> children;
    bool expanded = false;
    bool editable = true;
};

// #############################################################################
// GDScriptDebugger - Main debugger class
// #############################################################################
class GDScriptDebugger : public Object {
    XTU_GODOT_REGISTER_CLASS(GDScriptDebugger, Object)

private:
    static GDScriptDebugger* s_singleton;
    
    DebuggerState m_state = DebuggerState::STATE_RUNNING;
    StepMode m_step_mode = StepMode::STEP_NONE;
    
    std::unordered_map<String, std::vector<DebuggerBreakpoint>> m_breakpoints;
    std::vector<DebuggerStackFrame> m_stack_frames;
    std::vector<DebuggerVariable> m_variables;
    
    int m_current_stack_depth = 0;
    String m_error_message;
    
    std::function<void()> m_continue_callback;
    std::mutex m_mutex;
    std::atomic<bool> m_debugging{false};

public:
    static GDScriptDebugger* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("GDScriptDebugger"); }

    GDScriptDebugger() { s_singleton = this; }
    ~GDScriptDebugger() { s_singleton = nullptr; }

    // #########################################################################
    // State management
    // #########################################################################
    void start() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_debugging = true;
        m_state = DebuggerState::STATE_RUNNING;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_debugging = false;
        m_state = DebuggerState::STATE_RUNNING;
        m_continue_callback = nullptr;
    }

    bool is_debugging() const { return m_debugging; }
    DebuggerState get_state() const { return m_state; }

    void pause() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_debugging) {
            m_state = DebuggerState::STATE_PAUSED;
        }
    }

    void resume() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_debugging && m_state == DebuggerState::STATE_PAUSED) {
            m_state = DebuggerState::STATE_RUNNING;
            if (m_continue_callback) {
                m_continue_callback();
                m_continue_callback = nullptr;
            }
        }
    }

    void step(StepMode mode) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_debugging) {
            m_step_mode = mode;
            m_state = DebuggerState::STATE_STEPPING;
            if (m_continue_callback) {
                m_continue_callback();
                m_continue_callback = nullptr;
            }
        }
    }

    // #########################################################################
    // Breakpoint management
    // #########################################################################
    void add_breakpoint(const String& file, int line, const String& condition = "") {
        std::lock_guard<std::mutex> lock(m_mutex);
        DebuggerBreakpoint bp;
        bp.file = file;
        bp.line = line;
        bp.condition = condition;
        bp.enabled = true;
        m_breakpoints[file].push_back(bp);
        emit_signal("breakpoint_added", file, line);
    }

    void remove_breakpoint(const String& file, int line) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_breakpoints.find(file);
        if (it != m_breakpoints.end()) {
            auto& bps = it->second;
            bps.erase(std::remove_if(bps.begin(), bps.end(),
                [line](const DebuggerBreakpoint& bp) { return bp.line == line; }), bps.end());
            emit_signal("breakpoint_removed", file, line);
        }
    }

    void enable_breakpoint(const String& file, int line, bool enabled) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_breakpoints.find(file);
        if (it != m_breakpoints.end()) {
            for (auto& bp : it->second) {
                if (bp.line == line) {
                    bp.enabled = enabled;
                    break;
                }
            }
        }
    }

    std::vector<DebuggerBreakpoint> get_breakpoints(const String& file) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_breakpoints.find(file);
        return it != m_breakpoints.end() ? it->second : std::vector<DebuggerBreakpoint>();
    }

    void clear_breakpoints() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_breakpoints.clear();
        emit_signal("breakpoints_cleared");
    }

    // #########################################################################
    // Execution hooks (called from GDScriptVM)
    // #########################################################################
    bool on_line_reached(const String& file, int line, int stack_depth) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_debugging) return false;

        m_current_stack_depth = stack_depth;

        // Check for breakpoints
        bool should_break = false;
        auto it = m_breakpoints.find(file);
        if (it != m_breakpoints.end()) {
            for (const auto& bp : it->second) {
                if (bp.line == line && bp.enabled) {
                    if (bp.condition.empty() || evaluate_condition(bp.condition)) {
                        should_break = true;
                        break;
                    }
                }
            }
        }

        // Handle step modes
        if (m_step_mode != StepMode::STEP_NONE) {
            if (m_step_mode == StepMode::STEP_INTO) {
                should_break = true;
            } else if (m_step_mode == StepMode::STEP_OVER) {
                if (stack_depth <= m_step_stack_depth) {
                    should_break = true;
                }
            } else if (m_step_mode == StepMode::STEP_OUT) {
                if (stack_depth < m_step_stack_depth) {
                    should_break = true;
                }
            }
        }

        if (should_break) {
            m_state = DebuggerState::STATE_BREAK;
            m_step_mode = StepMode::STEP_NONE;
            capture_stack_frames();
            emit_signal("break", file, line);
            
            // Wait for continue
            wait_for_continue();
        }

        return should_break;
    }

    void on_function_enter(const String& function, const String& file, int line) {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_current_stack_depth;
        emit_signal("function_enter", function, file, line);
    }

    void on_function_exit(const String& function) {
        std::lock_guard<std::mutex> lock(m_mutex);
        --m_current_stack_depth;
        emit_signal("function_exit", function);
    }

    void on_error(const String& error, const String& file, int line) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_error_message = error;
        m_state = DebuggerState::STATE_BREAK;
        capture_stack_frames();
        emit_signal("error", error, file, line);
        wait_for_continue();
    }

    // #########################################################################
    // Stack and variable inspection
    // #########################################################################
    std::vector<DebuggerStackFrame> get_stack_frames() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stack_frames;
    }

    std::vector<DebuggerVariable> get_variables(int frame_index = 0) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (frame_index >= 0 && frame_index < static_cast<int>(m_stack_frames.size())) {
            return get_variables_for_frame(frame_index);
        }
        return {};
    }

    Variant evaluate_expression(const String& expression, int frame_index = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Evaluate expression in the context of the given stack frame
        return Variant();
    }

    void set_variable(const String& name, const Variant& value, int frame_index = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Set variable value in the given stack frame
    }

    // #########################################################################
    // Error information
    // #########################################################################
    String get_error_message() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_error_message;
    }

    void clear_error() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_error_message.clear();
    }

private:
    int m_step_stack_depth = 0;

    void wait_for_continue() {
        std::condition_variable cv;
        std::mutex wait_mutex;
        bool continued = false;
        
        m_continue_callback = [&]() {
            std::lock_guard<std::mutex> lock(wait_mutex);
            continued = true;
            cv.notify_one();
        };
        
        std::unique_lock<std::mutex> lock(wait_mutex);
        cv.wait(lock, [&]() { return continued; });
    }

    void capture_stack_frames() {
        m_stack_frames.clear();
        // Capture current call stack from GDScriptVM
        // This would integrate with the VM's stack walking
    }

    std::vector<DebuggerVariable> get_variables_for_frame(int frame_index) const {
        std::vector<DebuggerVariable> vars;
        // Extract local variables from the specified stack frame
        return vars;
    }

    bool evaluate_condition(const String& condition) const {
        // Evaluate breakpoint condition expression
        return true;
    }
};

// #############################################################################
// Debugger integration with GDScriptFunction
// #############################################################################
class GDScriptFunctionDebugInfo {
public:
    struct LineInfo {
        int bytecode_offset = 0;
        int line = 0;
        int column = 0;
        String source_file;
    };

private:
    std::vector<LineInfo> m_line_infos;
    std::vector<String> m_local_var_names;
    std::unordered_map<int, int> m_offset_to_line;

public:
    void add_line_info(int offset, int line, int column, const String& file) {
        LineInfo info;
        info.bytecode_offset = offset;
        info.line = line;
        info.column = column;
        info.source_file = file;
        m_line_infos.push_back(info);
        m_offset_to_line[offset] = static_cast<int>(m_line_infos.size() - 1);
    }

    int get_line_from_offset(int offset) const {
        auto it = m_offset_to_line.find(offset);
        if (it != m_offset_to_line.end()) {
            return m_line_infos[it->second].line;
        }
        return -1;
    }

    String get_file_from_offset(int offset) const {
        auto it = m_offset_to_line.find(offset);
        if (it != m_offset_to_line.end()) {
            return m_line_infos[it->second].source_file;
        }
        return String();
    }

    void add_local_var(const String& name) {
        m_local_var_names.push_back(name);
    }

    const std::vector<String>& get_local_vars() const { return m_local_var_names; }
};

// #############################################################################
// DebuggerMessage - Protocol messages for remote debugging
// #############################################################################
class DebuggerMessage {
public:
    enum Type {
        MSG_NONE = 0,
        MSG_ATTACH = 1,
        MSG_DETACH = 2,
        MSG_BREAK = 3,
        MSG_CONTINUE = 4,
        MSG_STEP = 5,
        MSG_STACK = 6,
        MSG_VARIABLES = 7,
        MSG_EVALUATE = 8,
        MSG_BREAKPOINT_ADD = 9,
        MSG_BREAKPOINT_REMOVE = 10,
        MSG_ERROR = 11
    };

    Type type = MSG_NONE;
    io::json::JsonValue data;

    String serialize() const {
        io::json::JsonValue json;
        json["type"] = io::json::JsonValue(static_cast<int>(type));
        json["data"] = data;
        return json.dump().c_str();
    }

    static DebuggerMessage parse(const String& str) {
        DebuggerMessage msg;
        io::json::JsonValue json = io::json::JsonValue::parse(str.to_std_string());
        msg.type = static_cast<Type>(static_cast<int>(json["type"].as_number()));
        msg.data = json["data"];
        return msg;
    }

    static DebuggerMessage make_break(const String& file, int line, const std::vector<DebuggerStackFrame>& stack) {
        DebuggerMessage msg;
        msg.type = MSG_BREAK;
        msg.data["file"] = io::json::JsonValue(file.to_std_string());
        msg.data["line"] = io::json::JsonValue(line);
        io::json::JsonValue stack_arr;
        for (const auto& frame : stack) {
            io::json::JsonValue frame_json;
            frame_json["function"] = io::json::JsonValue(frame.function_name.to_std_string());
            frame_json["file"] = io::json::JsonValue(frame.script_path.to_std_string());
            frame_json["line"] = io::json::JsonValue(frame.line);
            stack_arr.as_array().push_back(frame_json);
        }
        msg.data["stack"] = stack_arr;
        return msg;
    }

    static DebuggerMessage make_step(StepMode mode) {
        DebuggerMessage msg;
        msg.type = MSG_STEP;
        msg.data["mode"] = io::json::JsonValue(static_cast<int>(mode));
        return msg;
    }

    static DebuggerMessage make_continue() {
        DebuggerMessage msg;
        msg.type = MSG_CONTINUE;
        return msg;
    }
};

} // namespace gdscript

// Bring into main namespace
using gdscript::GDScriptDebugger;
using gdscript::DebuggerState;
using gdscript::StepMode;
using gdscript::DebuggerBreakpoint;
using gdscript::DebuggerStackFrame;
using gdscript::DebuggerVariable;
using gdscript::DebuggerMessage;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XGDSCRIPT_DEBUGGER_HPP