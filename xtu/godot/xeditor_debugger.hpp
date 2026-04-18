// include/xtu/godot/xeditor_debugger.hpp
// xtensor-unified - Editor debugger for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_DEBUGGER_HPP
#define XTU_GODOT_XEDITOR_DEBUGGER_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xprofiling.hpp"
#include "xtu/godot/xgui.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorDebuggerNode;
class ScriptEditorDebugger;
class EditorProfiler;
class EditorVisualProfiler;
class EditorNetworkProfiler;

// #############################################################################
// Debugger message types
// #############################################################################
enum class DebuggerMessageType : uint8_t {
    MSG_NONE = 0,
    MSG_ATTACH = 1,
    MSG_DETACH = 2,
    MSG_BREAK = 3,
    MSG_CONTINUE = 4,
    MSG_STEP = 5,
    MSG_STACK_FRAME = 6,
    MSG_VARIABLES = 7,
    MSG_OUTPUT = 8,
    MSG_ERROR = 9,
    MSG_PROFILE_FRAME = 10,
    MSG_NETWORK = 11
};

// #############################################################################
// Debugger break mode
// #############################################################################
enum class DebuggerBreakMode : uint8_t {
    BREAK_NONE = 0,
    BREAK_STEP_INTO = 1,
    BREAK_STEP_OVER = 2,
    BREAK_STEP_OUT = 3,
    BREAK_PAUSED = 4
};

// #############################################################################
// Stack frame info
// #############################################################################
struct DebuggerStackFrame {
    String file;
    String function;
    int line = 0;
    int column = 0;
    String source;
};

// #############################################################################
// Variable info
// #############################################################################
struct DebuggerVariable {
    String name;
    Variant value;
    VariantType type = VariantType::NIL;
    std::vector<DebuggerVariable> children;
    bool expanded = false;
};

// #############################################################################
// ScriptEditorDebugger - Main debugging interface
// #############################################################################
class ScriptEditorDebugger : public Control {
    XTU_GODOT_REGISTER_CLASS(ScriptEditorDebugger, Control)

private:
    static ScriptEditorDebugger* s_singleton;
    TabContainer* m_tabs = nullptr;
    Tree* m_stack_tree = nullptr;
    Tree* m_variables_tree = nullptr;
    Tree* m_breakpoints_tree = nullptr;
    TextEdit* m_output_log = nullptr;
    LineEdit* m_expression_input = nullptr;
    Button* m_continue_btn = nullptr;
    Button* m_step_over_btn = nullptr;
    Button* m_step_into_btn = nullptr;
    Button* m_step_out_btn = nullptr;
    Button* m_stop_btn = nullptr;
    String m_remote_peer;
    bool m_attached = false;
    DebuggerBreakMode m_break_mode = DebuggerBreakMode::BREAK_NONE;
    std::vector<DebuggerStackFrame> m_stack_frames;
    std::vector<DebuggerVariable> m_variables;
    std::unordered_map<String, std::vector<int>> m_breakpoints;
    std::mutex m_mutex;
    std::queue<std::pair<DebuggerMessageType, Variant>> m_message_queue;

public:
    static ScriptEditorDebugger* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ScriptEditorDebugger"); }

    ScriptEditorDebugger() {
        s_singleton = this;
        build_ui();
    }

    ~ScriptEditorDebugger() { s_singleton = nullptr; }

    void attach(const String& address, int port = 6007) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_remote_peer = address + ":" + String::num(port);
        m_attached = true;
        m_break_mode = DebuggerBreakMode::BREAK_PAUSED;
        update_ui_state();
        send_message(DebuggerMessageType::MSG_ATTACH, Variant());
    }

    void detach() {
        std::lock_guard<std::mutex> lock(m_mutex);
        send_message(DebuggerMessageType::MSG_DETACH, Variant());
        m_attached = false;
        m_break_mode = DebuggerBreakMode::BREAK_NONE;
        update_ui_state();
    }

    bool is_attached() const { return m_attached; }

    void continue_execution() {
        if (!m_attached) return;
        send_message(DebuggerMessageType::MSG_CONTINUE, Variant());
        m_break_mode = DebuggerBreakMode::BREAK_NONE;
        update_ui_state();
    }

    void step_over() {
        if (!m_attached) return;
        send_message(DebuggerMessageType::MSG_STEP, 0);
    }

    void step_into() {
        if (!m_attached) return;
        send_message(DebuggerMessageType::MSG_STEP, 1);
    }

    void step_out() {
        if (!m_attached) return;
        send_message(DebuggerMessageType::MSG_STEP, 2);
    }

    void stop() {
        if (!m_attached) return;
        send_message(DebuggerMessageType::MSG_BREAK, Variant());
    }

    void add_breakpoint(const String& file, int line) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_breakpoints[file].push_back(line);
        if (m_attached) {
            send_breakpoint(file, line, true);
        }
        refresh_breakpoints();
    }

    void remove_breakpoint(const String& file, int line) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_breakpoints.find(file);
        if (it != m_breakpoints.end()) {
            auto& lines = it->second;
            lines.erase(std::remove(lines.begin(), lines.end(), line), lines.end());
            if (m_attached) {
                send_breakpoint(file, line, false);
            }
        }
        refresh_breakpoints();
    }

    void evaluate_expression(const String& expr) {
        if (!m_attached) return;
        // Send expression evaluation request
    }

    void process_messages() {
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_message_queue.empty()) {
            auto [type, data] = m_message_queue.front();
            m_message_queue.pop();
            handle_message(type, data);
        }
    }

    void _process(double delta) override {
        process_messages();
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();
        m_continue_btn = new Button();
        m_continue_btn->set_text("Continue");
        m_continue_btn->connect("pressed", this, "continue_execution");
        toolbar->add_child(m_continue_btn);

        m_step_over_btn = new Button();
        m_step_over_btn->set_text("Step Over");
        m_step_over_btn->connect("pressed", this, "step_over");
        toolbar->add_child(m_step_over_btn);

        m_step_into_btn = new Button();
        m_step_into_btn->set_text("Step Into");
        m_step_into_btn->connect("pressed", this, "step_into");
        toolbar->add_child(m_step_into_btn);

        m_step_out_btn = new Button();
        m_step_out_btn->set_text("Step Out");
        m_step_out_btn->connect("pressed", this, "step_out");
        toolbar->add_child(m_step_out_btn);

        m_stop_btn = new Button();
        m_stop_btn->set_text("Stop");
        m_stop_btn->connect("pressed", this, "stop");
        toolbar->add_child(m_stop_btn);

        main->add_child(toolbar);

        // Tabs
        m_tabs = new TabContainer();
        m_tabs->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(m_tabs);

        // Stack tab
        m_stack_tree = new Tree();
        m_stack_tree->set_column_title(0, "Function");
        m_stack_tree->set_column_title(1, "File");
        m_stack_tree->set_column_title(2, "Line");
        m_tabs->add_child(m_stack_tree);
        m_tabs->set_tab_title(0, "Stack");

        // Variables tab
        m_variables_tree = new Tree();
        m_variables_tree->set_column_title(0, "Name");
        m_variables_tree->set_column_title(1, "Value");
        m_variables_tree->set_column_title(2, "Type");
        m_tabs->add_child(m_variables_tree);
        m_tabs->set_tab_title(1, "Variables");

        // Breakpoints tab
        m_breakpoints_tree = new Tree();
        m_breakpoints_tree->set_column_title(0, "File");
        m_breakpoints_tree->set_column_title(1, "Line");
        m_tabs->add_child(m_breakpoints_tree);
        m_tabs->set_tab_title(2, "Breakpoints");

        // Output tab
        m_output_log = new TextEdit();
        m_output_log->set_readonly(true);
        m_tabs->add_child(m_output_log);
        m_tabs->set_tab_title(3, "Output");

        // Expression evaluator
        HBoxContainer* expr_row = new HBoxContainer();
        expr_row->add_child(new Label(">"));
        m_expression_input = new LineEdit();
        m_expression_input->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_expression_input->connect("text_entered", this, "on_expression_entered");
        expr_row->add_child(m_expression_input);
        main->add_child(expr_row);

        update_ui_state();
    }

    void update_ui_state() {
        bool paused = m_break_mode != DebuggerBreakMode::BREAK_NONE;
        m_continue_btn->set_disabled(!m_attached || !paused);
        m_step_over_btn->set_disabled(!m_attached || !paused);
        m_step_into_btn->set_disabled(!m_attached || !paused);
        m_step_out_btn->set_disabled(!m_attached || !paused);
        m_stop_btn->set_disabled(!m_attached);
    }

    void send_message(DebuggerMessageType type, const Variant& data) {}
    void send_breakpoint(const String& file, int line, bool add) {}

    void handle_message(DebuggerMessageType type, const Variant& data) {
        switch (type) {
            case DebuggerMessageType::MSG_STACK_FRAME:
                update_stack_frames(data);
                break;
            case DebuggerMessageType::MSG_VARIABLES:
                update_variables(data);
                break;
            case DebuggerMessageType::MSG_OUTPUT:
                append_output(data.as<String>());
                break;
            default:
                break;
        }
    }

    void update_stack_frames(const Variant& data) {}
    void update_variables(const Variant& data) {}
    void append_output(const String& text) {
        m_output_log->set_text(m_output_log->get_text() + text + "\n");
    }

    void refresh_breakpoints() {
        m_breakpoints_tree->clear();
        TreeItem* root = m_breakpoints_tree->create_item();
        for (const auto& kv : m_breakpoints) {
            for (int line : kv.second) {
                TreeItem* item = m_breakpoints_tree->create_item(root);
                item->set_text(0, kv.first);
                item->set_text(1, String::num(line));
            }
        }
    }

    void on_expression_entered(const String& expr) {
        evaluate_expression(expr);
        m_expression_input->clear();
    }
};

// #############################################################################
// EditorProfiler - Performance profiling panel
// #############################################################################
class EditorProfiler : public Control {
    XTU_GODOT_REGISTER_CLASS(EditorProfiler, Control)

private:
    GraphEdit* m_graph = nullptr;
    Tree* m_metrics_tree = nullptr;
    Button* m_record_btn = nullptr;
    Button* m_clear_btn = nullptr;
    bool m_recording = false;
    std::vector<ProfilerFrame> m_history;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("EditorProfiler"); }

    EditorProfiler() {
        build_ui();
    }

    void start_recording() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_recording = true;
        m_record_btn->set_text("Stop");
        EngineProfiler::get_singleton()->set_profiling_enabled(true);
    }

    void stop_recording() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_recording = false;
        m_record_btn->set_text("Start");
        EngineProfiler::get_singleton()->set_profiling_enabled(false);
    }

    void toggle_recording() {
        if (m_recording) stop_recording();
        else start_recording();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_history.clear();
        update_graph();
    }

    void _process(double delta) override {
        if (m_recording) {
            std::lock_guard<std::mutex> lock(m_mutex);
            ProfilerFrame frame = EngineProfiler::get_singleton()->get_current_frame();
            m_history.push_back(frame);
            if (m_history.size() > 600) {
                m_history.erase(m_history.begin());
            }
            update_graph();
            update_metrics();
        }
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        HBoxContainer* toolbar = new HBoxContainer();
        m_record_btn = new Button();
        m_record_btn->set_text("Start");
        m_record_btn->connect("pressed", this, "toggle_recording");
        toolbar->add_child(m_record_btn);

        m_clear_btn = new Button();
        m_clear_btn->set_text("Clear");
        m_clear_btn->connect("pressed", this, "clear");
        toolbar->add_child(m_clear_btn);

        main->add_child(toolbar);

        HSplitContainer* split = new HSplitContainer();
        split->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(split);

        m_graph = new GraphEdit();
        m_graph->set_right_disconnects(true);
        split->add_child(m_graph);

        m_metrics_tree = new Tree();
        m_metrics_tree->set_column_title(0, "Metric");
        m_metrics_tree->set_column_title(1, "Value");
        split->add_child(m_metrics_tree);
    }

    void update_graph() {}
    void update_metrics() {}
};

// #############################################################################
// EditorVisualProfiler - Flame graph profiler
// #############################################################################
class EditorVisualProfiler : public Control {
    XTU_GODOT_REGISTER_CLASS(EditorVisualProfiler, Control)

private:
    Control* m_canvas = nullptr;
    std::unordered_map<String, std::vector<uint64_t>> m_timings;

public:
    static StringName get_class_static() { return StringName("EditorVisualProfiler"); }

    EditorVisualProfiler() {
        build_ui();
    }

    void _draw() override {
        draw_flame_graph();
    }

private:
    void build_ui() {
        m_canvas = new Control();
        m_canvas->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_canvas->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(m_canvas);
    }

    void draw_flame_graph() {}
};

// #############################################################################
// EditorNetworkProfiler - Network traffic monitor
// #############################################################################
class EditorNetworkProfiler : public Control {
    XTU_GODOT_REGISTER_CLASS(EditorNetworkProfiler, Control)

private:
    Tree* m_requests_tree = nullptr;
    bool m_capturing = false;

public:
    static StringName get_class_static() { return StringName("EditorNetworkProfiler"); }

    EditorNetworkProfiler() {
        build_ui();
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        m_requests_tree = new Tree();
        m_requests_tree->set_columns(4);
        m_requests_tree->set_column_title(0, "Method");
        m_requests_tree->set_column_title(1, "URL");
        m_requests_tree->set_column_title(2, "Status");
        m_requests_tree->set_column_title(3, "Size");
        m_requests_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(m_requests_tree);
    }
};

} // namespace editor

// Bring into main namespace
using editor::ScriptEditorDebugger;
using editor::EditorProfiler;
using editor::EditorVisualProfiler;
using editor::EditorNetworkProfiler;
using editor::DebuggerMessageType;
using editor::DebuggerBreakMode;
using editor::DebuggerStackFrame;
using editor::DebuggerVariable;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_DEBUGGER_HPP