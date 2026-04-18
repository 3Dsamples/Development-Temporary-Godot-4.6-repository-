// include/xtu/godot/xeditor_log.hpp
// xtensor-unified - Editor build log and error navigation for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_LOG_HPP
#define XTU_GODOT_XEDITOR_LOG_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xerror.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorLog;
class BuildOutputParser;

// #############################################################################
// Log entry type
// #############################################################################
enum class LogEntryType : uint8_t {
    TYPE_INFO = 0,
    TYPE_WARNING = 1,
    TYPE_ERROR = 2,
    TYPE_SUCCESS = 3,
    TYPE_DEBUG = 4
};

// #############################################################################
// Log entry structure
// #############################################################################
struct LogEntry {
    LogEntryType type = LogEntryType::TYPE_INFO;
    String message;
    String file;
    int line = 0;
    int column = 0;
    uint64_t timestamp = 0;
    String source;  // e.g., "build", "script", "shader"
};

// #############################################################################
// BuildOutputParser - Parse compiler/build tool output
// #############################################################################
class BuildOutputParser : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(BuildOutputParser, RefCounted)

public:
    static StringName get_class_static() { return StringName("BuildOutputParser"); }

    std::vector<LogEntry> parse_gcc_output(const String& output, const String& source = "gcc") {
        std::vector<LogEntry> result;
        // GCC/Clang format: file:line:column: error: message
        std::regex gcc_regex(R"(([^:\s]+):(\d+):(\d+):\s*(error|warning|note):\s*(.+))");
        std::string out = output.to_std_string();
        std::smatch match;
        std::string::const_iterator start = out.begin();
        while (std::regex_search(start, out.end(), match, gcc_regex)) {
            LogEntry entry;
            entry.file = String(match[1].str().c_str());
            entry.line = std::stoi(match[2].str());
            entry.column = std::stoi(match[3].str());
            String level = String(match[4].str().c_str());
            if (level == "error") entry.type = LogEntryType::TYPE_ERROR;
            else if (level == "warning") entry.type = LogEntryType::TYPE_WARNING;
            else entry.type = LogEntryType::TYPE_INFO;
            entry.message = String(match[5].str().c_str());
            entry.source = source;
            entry.timestamp = OS::get_singleton()->get_ticks_msec();
            result.push_back(entry);
            start = match[0].second;
        }
        return result;
    }

    std::vector<LogEntry> parse_msvc_output(const String& output, const String& source = "msvc") {
        std::vector<LogEntry> result;
        // MSVC format: file(line,column): error CXXXX: message
        std::regex msvc_regex(R"(([^(]+)\((\d+),(\d+)\):\s*(error|warning)\s*([^:]+):\s*(.+))");
        std::string out = output.to_std_string();
        std::smatch match;
        std::string::const_iterator start = out.begin();
        while (std::regex_search(start, out.end(), match, msvc_regex)) {
            LogEntry entry;
            entry.file = String(match[1].str().c_str());
            entry.line = std::stoi(match[2].str());
            entry.column = std::stoi(match[3].str());
            String level = String(match[4].str().c_str());
            entry.type = (level == "error") ? LogEntryType::TYPE_ERROR : LogEntryType::TYPE_WARNING;
            entry.message = String(match[6].str().c_str());
            entry.source = source;
            entry.timestamp = OS::get_singleton()->get_ticks_msec();
            result.push_back(entry);
            start = match[0].second;
        }
        return result;
    }

    std::vector<LogEntry> parse_gdscript_error(const String& output) {
        std::vector<LogEntry> result;
        // GDScript format: at file:line - message
        std::regex gd_regex(R"(at\s+([^:]+):(\d+)\s*-\s*(.+))");
        std::string out = output.to_std_string();
        std::smatch match;
        std::string::const_iterator start = out.begin();
        while (std::regex_search(start, out.end(), match, gd_regex)) {
            LogEntry entry;
            entry.file = String(match[1].str().c_str());
            entry.line = std::stoi(match[2].str());
            entry.type = LogEntryType::TYPE_ERROR;
            entry.message = String(match[3].str().c_str());
            entry.source = "gdscript";
            entry.timestamp = OS::get_singleton()->get_ticks_msec();
            result.push_back(entry);
            start = match[0].second;
        }
        return result;
    }

    std::vector<LogEntry> parse_shader_error(const String& output) {
        std::vector<LogEntry> result;
        // Shader error format: ERROR: line - message
        std::regex shader_regex(R"(ERROR:\s*(\d+):\s*(.+))");
        std::string out = output.to_std_string();
        std::smatch match;
        std::string::const_iterator start = out.begin();
        while (std::regex_search(start, out.end(), match, shader_regex)) {
            LogEntry entry;
            entry.line = std::stoi(match[1].str());
            entry.type = LogEntryType::TYPE_ERROR;
            entry.message = String(match[2].str().c_str());
            entry.source = "shader";
            entry.timestamp = OS::get_singleton()->get_ticks_msec();
            result.push_back(entry);
            start = match[0].second;
        }
        return result;
    }
};

// #############################################################################
// EditorLog - Enhanced log viewer with error navigation
// #############################################################################
class EditorLog : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(EditorLog, VBoxContainer)

private:
    RichTextLabel* m_output = nullptr;
    Tree* m_errors_tree = nullptr;
    LineEdit* m_search_box = nullptr;
    Button* m_clear_btn = nullptr;
    Button* m_copy_btn = nullptr;
    CheckBox* m_auto_scroll = nullptr;
    CheckBox* m_show_info = nullptr;
    CheckBox* m_show_warnings = nullptr;
    CheckBox* m_show_errors = nullptr;
    OptionButton* m_filter_source = nullptr;
    std::vector<LogEntry> m_entries;
    std::vector<LogEntry> m_filtered_entries;
    BuildOutputParser* m_parser = nullptr;
    std::mutex m_mutex;
    std::function<void(const String&, int, int)> m_error_selected_callback;

public:
    static StringName get_class_static() { return StringName("EditorLog"); }

    EditorLog() {
        m_parser = new BuildOutputParser();
        build_ui();
    }

    void set_error_selected_callback(std::function<void(const String&, int, int)> cb) {
        m_error_selected_callback = cb;
    }

    void add_entry(const LogEntry& entry) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries.push_back(entry);
        if (m_entries.size() > 10000) {
            m_entries.erase(m_entries.begin(), m_entries.begin() + 5000);
        }
        update_display();
    }

    void add_message(const String& msg, LogEntryType type = LogEntryType::TYPE_INFO,
                     const String& source = "") {
        LogEntry entry;
        entry.type = type;
        entry.message = msg;
        entry.source = source;
        entry.timestamp = OS::get_singleton()->get_ticks_msec();
        add_entry(entry);
    }

    void add_build_output(const String& output, const String& compiler = "gcc") {
        std::vector<LogEntry> parsed;
        if (compiler == "gcc" || compiler == "clang") {
            parsed = m_parser->parse_gcc_output(output, compiler);
        } else if (compiler == "msvc") {
            parsed = m_parser->parse_msvc_output(output, compiler);
        }
        for (const auto& e : parsed) {
            add_entry(e);
        }
    }

    void add_script_error(const String& output) {
        auto parsed = m_parser->parse_gdscript_error(output);
        for (const auto& e : parsed) {
            add_entry(e);
        }
    }

    void add_shader_error(const String& output) {
        auto parsed = m_parser->parse_shader_error(output);
        for (const auto& e : parsed) {
            add_entry(e);
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries.clear();
        m_filtered_entries.clear();
        m_output->clear();
        m_errors_tree->clear();
    }

    void copy_to_clipboard() {
        String text;
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& e : m_filtered_entries) {
            text += format_entry_text(e) + "\n";
        }
        DisplayServer::get_singleton()->clipboard_set(text);
    }

    void export_to_file(const String& path) {
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
        if (!file.is_valid()) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& e : m_entries) {
            file->store_string(format_entry_text(e) + "\n");
        }
    }

    void apply_filter() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_filtered_entries.clear();

        String search = m_search_box->get_text().to_lower();
        int source_idx = m_filter_source->get_selected();
        String source_filter = (source_idx > 0) ? m_filter_source->get_item_text(source_idx) : "";

        for (const auto& e : m_entries) {
            if (!m_show_info->is_pressed() && e.type == LogEntryType::TYPE_INFO) continue;
            if (!m_show_warnings->is_pressed() && e.type == LogEntryType::TYPE_WARNING) continue;
            if (!m_show_errors->is_pressed() && e.type == LogEntryType::TYPE_ERROR) continue;

            if (!search.empty()) {
                if (e.message.to_lower().find(search) == String::npos &&
                    e.file.to_lower().find(search) == String::npos) {
                    continue;
                }
            }

            if (!source_filter.empty() && e.source != source_filter) continue;

            m_filtered_entries.push_back(e);
        }

        refresh_display();
    }

private:
    void build_ui() {
        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();

        m_clear_btn = new Button();
        m_clear_btn->set_text("Clear");
        m_clear_btn->connect("pressed", this, "clear");
        toolbar->add_child(m_clear_btn);

        m_copy_btn = new Button();
        m_copy_btn->set_text("Copy");
        m_copy_btn->connect("pressed", this, "copy_to_clipboard");
        toolbar->add_child(m_copy_btn);

        Button* export_btn = new Button();
        export_btn->set_text("Export");
        export_btn->connect("pressed", this, "on_export_pressed");
        toolbar->add_child(export_btn);

        toolbar->add_child(new VSeparator());

        m_auto_scroll = new CheckBox();
        m_auto_scroll->set_text("Auto Scroll");
        m_auto_scroll->set_pressed(true);
        toolbar->add_child(m_auto_scroll);

        m_show_info = new CheckBox();
        m_show_info->set_text("Info");
        m_show_info->set_pressed(true);
        m_show_info->connect("toggled", this, "on_filter_changed");
        toolbar->add_child(m_show_info);

        m_show_warnings = new CheckBox();
        m_show_warnings->set_text("Warnings");
        m_show_warnings->set_pressed(true);
        m_show_warnings->connect("toggled", this, "on_filter_changed");
        toolbar->add_child(m_show_warnings);

        m_show_errors = new CheckBox();
        m_show_errors->set_text("Errors");
        m_show_errors->set_pressed(true);
        m_show_errors->connect("toggled", this, "on_filter_changed");
        toolbar->add_child(m_show_errors);

        add_child(toolbar);

        // Search bar
        HBoxContainer* search_row = new HBoxContainer();
        search_row->add_child(new Label("Search:"));

        m_search_box = new LineEdit();
        m_search_box->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_search_box->connect("text_changed", this, "on_filter_changed");
        search_row->add_child(m_search_box);

        m_filter_source = new OptionButton();
        m_filter_source->add_item("All Sources");
        m_filter_source->add_item("build");
        m_filter_source->add_item("gdscript");
        m_filter_source->add_item("shader");
        m_filter_source->connect("item_selected", this, "on_filter_changed");
        search_row->add_child(m_filter_source);

        add_child(search_row);

        // Split view
        HSplitContainer* split = new HSplitContainer();
        split->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(split);

        // Output text area
        m_output = new RichTextLabel();
        m_output->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_output->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_output->set_selection_enabled(true);
        m_output->set_scroll_follow(true);
        split->add_child(m_output);

        // Errors tree
        m_errors_tree = new Tree();
        m_errors_tree->set_columns(3);
        m_errors_tree->set_column_title(0, "Error");
        m_errors_tree->set_column_title(1, "File");
        m_errors_tree->set_column_title(2, "Line");
        m_errors_tree->set_hide_root(true);
        m_errors_tree->connect("item_activated", this, "on_error_activated");
        m_errors_tree->connect("item_selected", this, "on_error_selected");
        split->add_child(m_errors_tree);
    }

    void update_display() {
        call_deferred("_update_display_deferred");
    }

    void _update_display_deferred() {
        apply_filter();
    }

    void refresh_display() {
        // Build rich text output
        m_output->clear();
        for (const auto& e : m_filtered_entries) {
            m_output->append_text(format_entry_rich(e) + "\n");
        }

        // Build errors tree
        m_errors_tree->clear();
        TreeItem* root = m_errors_tree->create_item();

        for (size_t i = 0; i < m_filtered_entries.size(); ++i) {
            const auto& e = m_filtered_entries[i];
            if (e.type != LogEntryType::TYPE_ERROR && e.type != LogEntryType::TYPE_WARNING) continue;

            TreeItem* item = m_errors_tree->create_item(root);
            item->set_text(0, e.message);
            item->set_text(1, e.file);
            item->set_text(2, String::num(e.line));
            item->set_metadata(0, static_cast<int64_t>(i));
            item->set_metadata(1, e.file);
            item->set_metadata(2, e.line);
            item->set_metadata(3, e.column);

            // Set color based on type
            Color color = (e.type == LogEntryType::TYPE_ERROR) ?
                EditorTheme::get_singleton()->get_color(ThemeColorRole::COLOR_ERROR) :
                EditorTheme::get_singleton()->get_color(ThemeColorRole::COLOR_WARNING);
            item->set_custom_color(0, color);
        }
    }

    String format_entry_text(const LogEntry& e) const {
        String prefix;
        switch (e.type) {
            case LogEntryType::TYPE_WARNING: prefix = "WARNING: "; break;
            case LogEntryType::TYPE_ERROR: prefix = "ERROR: "; break;
            case LogEntryType::TYPE_SUCCESS: prefix = "SUCCESS: "; break;
            default: prefix = "";
        }
        String result = "[" + e.source + "] " + prefix + e.message;
        if (!e.file.empty()) {
            result += " (" + e.file + ":" + String::num(e.line) + ")";
        }
        return result;
    }

    String format_entry_rich(const LogEntry& e) const {
        String color;
        switch (e.type) {
            case LogEntryType::TYPE_WARNING: color = "#e6b800"; break;
            case LogEntryType::TYPE_ERROR: color = "#e60000"; break;
            case LogEntryType::TYPE_SUCCESS: color = "#00cc00"; break;
            default: color = "#ffffff";
        }
        return "[color=" + color + "]" + format_entry_text(e) + "[/color]";
    }

    void on_filter_changed() {
        apply_filter();
    }

    void on_error_activated() {
        TreeItem* item = m_errors_tree->get_selected();
        if (item && item != m_errors_tree->get_root()) {
            String file = item->get_metadata(1).as<String>();
            int line = static_cast<int>(item->get_metadata(2).as<int64_t>());
            int column = static_cast<int>(item->get_metadata(3).as<int64_t>());
            if (m_error_selected_callback) {
                m_error_selected_callback(file, line, column);
            }
            emit_signal("error_selected", file, line, column);
        }
    }

    void on_error_selected() {
        // Preview error in status bar
    }

    void on_export_pressed() {
        EditorFileDialog* dialog = new EditorFileDialog();
        dialog->set_file_mode(FileDialogMode::MODE_SAVE_FILE);
        dialog->add_filter("*.log", "Log Files");
        dialog->connect("file_selected", this, "export_to_file");
        dialog->popup_centered();
    }
};

// #############################################################################
// ErrorNavigator - Navigate between errors in code
// #############################################################################
class ErrorNavigator : public HBoxContainer {
    XTU_GODOT_REGISTER_CLASS(ErrorNavigator, HBoxContainer)

private:
    Button* m_prev_btn = nullptr;
    Button* m_next_btn = nullptr;
    Label* m_count_label = nullptr;
    std::vector<LogEntry> m_errors;
    int m_current_index = -1;
    std::function<void(const String&, int, int)> m_navigate_callback;

public:
    static StringName get_class_static() { return StringName("ErrorNavigator"); }

    ErrorNavigator() {
        build_ui();
    }

    void set_errors(const std::vector<LogEntry>& errors) {
        m_errors.clear();
        for (const auto& e : errors) {
            if (e.type == LogEntryType::TYPE_ERROR || e.type == LogEntryType::TYPE_WARNING) {
                m_errors.push_back(e);
            }
        }
        m_current_index = m_errors.empty() ? -1 : 0;
        update_ui();
        if (m_current_index >= 0 && m_navigate_callback) {
            const auto& e = m_errors[m_current_index];
            m_navigate_callback(e.file, e.line, e.column);
        }
    }

    void set_navigate_callback(std::function<void(const String&, int, int)> cb) {
        m_navigate_callback = cb;
    }

    void go_prev() {
        if (m_errors.empty()) return;
        m_current_index = (m_current_index - 1 + static_cast<int>(m_errors.size())) % static_cast<int>(m_errors.size());
        update_ui();
        if (m_navigate_callback) {
            const auto& e = m_errors[m_current_index];
            m_navigate_callback(e.file, e.line, e.column);
        }
    }

    void go_next() {
        if (m_errors.empty()) return;
        m_current_index = (m_current_index + 1) % static_cast<int>(m_errors.size());
        update_ui();
        if (m_navigate_callback) {
            const auto& e = m_errors[m_current_index];
            m_navigate_callback(e.file, e.line, e.column);
        }
    }

    void clear() {
        m_errors.clear();
        m_current_index = -1;
        update_ui();
    }

private:
    void build_ui() {
        m_prev_btn = new Button();
        m_prev_btn->set_text("<");
        m_prev_btn->set_tooltip_text("Previous Error");
        m_prev_btn->connect("pressed", this, "go_prev");
        add_child(m_prev_btn);

        m_count_label = new Label();
        m_count_label->set_text("0/0");
        add_child(m_count_label);

        m_next_btn = new Button();
        m_next_btn->set_text(">");
        m_next_btn->set_tooltip_text("Next Error");
        m_next_btn->connect("pressed", this, "go_next");
        add_child(m_next_btn);
    }

    void update_ui() {
        if (m_errors.empty()) {
            m_count_label->set_text("0/0");
            m_prev_btn->set_disabled(true);
            m_next_btn->set_disabled(true);
        } else {
            m_count_label->set_text(String::num(m_current_index + 1) + "/" + String::num(static_cast<int>(m_errors.size())));
            m_prev_btn->set_disabled(false);
            m_next_btn->set_disabled(false);
        }
    }
};

} // namespace editor

// Bring into main namespace
using editor::EditorLog;
using editor::BuildOutputParser;
using editor::ErrorNavigator;
using editor::LogEntry;
using editor::LogEntryType;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_LOG_HPP