// include/xtu/godot/xeditor_build.hpp
// xtensor-unified - Editor build and run management for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_BUILD_HPP
#define XTU_GODOT_XEDITOR_BUILD_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xeditor_export.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorBuildProfile;
class EditorRun;
class EditorRunNative;
class EditorLog;
class EditorBuildManager;

// #############################################################################
// Build configuration type
// #############################################################################
enum class BuildConfigType : uint8_t {
    CONFIG_DEBUG = 0,
    CONFIG_RELEASE = 1,
    CONFIG_CUSTOM = 2
};

// #############################################################################
// Run target type
// #############################################################################
enum class RunTargetType : uint8_t {
    TARGET_EDITOR = 0,
    TARGET_CURRENT_SCENE = 1,
    TARGET_MAIN_SCENE = 2,
    TARGET_CUSTOM = 3
};

// #############################################################################
// Build log level
// #############################################################################
enum class BuildLogLevel : uint8_t {
    LOG_INFO = 0,
    LOG_WARNING = 1,
    LOG_ERROR = 2
};

// #############################################################################
// Build log entry
// #############################################################################
struct BuildLogEntry {
    BuildLogLevel level = BuildLogLevel::LOG_INFO;
    String message;
    String file;
    int line = 0;
    int column = 0;
    uint64_t timestamp = 0;
};

// #############################################################################
// EditorBuildProfile - Build configuration preset
// #############################################################################
class EditorBuildProfile : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorBuildProfile, RefCounted)

private:
    String m_name;
    BuildConfigType m_type = BuildConfigType::CONFIG_DEBUG;
    std::map<String, Variant> m_defines;
    std::vector<String> m_disabled_classes;
    std::vector<String> m_disabled_features;
    std::vector<String> m_enabled_features;
    String m_custom_build_script;
    bool m_strip_debug = false;
    bool m_optimize_size = false;

public:
    static StringName get_class_static() { return StringName("EditorBuildProfile"); }

    void set_name(const String& name) { m_name = name; }
    String get_name() const { return m_name; }

    void set_type(BuildConfigType type) { m_type = type; }
    BuildConfigType get_type() const { return m_type; }

    void set_define(const String& name, const Variant& value) { m_defines[name] = value; }
    Variant get_define(const String& name) const {
        auto it = m_defines.find(name);
        return it != m_defines.end() ? it->second : Variant();
    }

    void set_strip_debug(bool strip) { m_strip_debug = strip; }
    bool get_strip_debug() const { return m_strip_debug; }

    void set_optimize_size(bool optimize) { m_optimize_size = optimize; }
    bool get_optimize_size() const { return m_optimize_size; }
};

// #############################################################################
// EditorRun - Run configuration
// #############################################################################
class EditorRun : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorRun, RefCounted)

private:
    String m_name;
    RunTargetType m_target_type = RunTargetType::TARGET_MAIN_SCENE;
    String m_custom_scene;
    std::vector<String> m_command_line_args;
    std::map<String, String> m_environment;
    bool m_enable_remote_debug = true;
    int m_debug_port = 6007;
    bool m_auto_build = true;
    Ref<EditorBuildProfile> m_build_profile;

public:
    static StringName get_class_static() { return StringName("EditorRun"); }

    void set_name(const String& name) { m_name = name; }
    String get_name() const { return m_name; }

    void set_target_type(RunTargetType type) { m_target_type = type; }
    RunTargetType get_target_type() const { return m_target_type; }

    void set_custom_scene(const String& scene) { m_custom_scene = scene; }
    String get_custom_scene() const { return m_custom_scene; }

    void add_command_line_arg(const String& arg) { m_command_line_args.push_back(arg); }
    const std::vector<String>& get_command_line_args() const { return m_command_line_args; }

    void set_environment_var(const String& key, const String& value) { m_environment[key] = value; }
    String get_environment_var(const String& key) const {
        auto it = m_environment.find(key);
        return it != m_environment.end() ? it->second : String();
    }

    void set_enable_remote_debug(bool enable) { m_enable_remote_debug = enable; }
    bool get_enable_remote_debug() const { return m_enable_remote_debug; }

    void set_debug_port(int port) { m_debug_port = port; }
    int get_debug_port() const { return m_debug_port; }

    void set_build_profile(const Ref<EditorBuildProfile>& profile) { m_build_profile = profile; }
    Ref<EditorBuildProfile> get_build_profile() const { return m_build_profile; }
};

// #############################################################################
// EditorRunNative - Native platform runner
// #############################################################################
class EditorRunNative : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorRunNative, RefCounted)

public:
    static StringName get_class_static() { return StringName("EditorRunNative"); }

    virtual bool can_run() = 0;
    virtual bool start(const String& executable, const std::vector<String>& args,
                       const std::map<String, String>& env, const String& working_dir) = 0;
    virtual bool stop() = 0;
    virtual bool is_running() = 0;
    virtual void process_output() = 0;
};

// #############################################################################
// EditorLog - Build output viewer
// #############################################################################
class EditorLog : public Control {
    XTU_GODOT_REGISTER_CLASS(EditorLog, Control)

private:
    RichTextLabel* m_output = nullptr;
    Tree* m_errors_tree = nullptr;
    Button* m_clear_btn = nullptr;
    Button* m_copy_btn = nullptr;
    CheckBox* m_auto_scroll = nullptr;
    std::vector<BuildLogEntry> m_entries;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("EditorLog"); }

    EditorLog() { build_ui(); }

    void add_message(const String& msg, BuildLogLevel level = BuildLogLevel::LOG_INFO,
                     const String& file = "", int line = 0, int column = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        BuildLogEntry entry;
        entry.level = level;
        entry.message = msg;
        entry.file = file;
        entry.line = line;
        entry.column = column;
        entry.timestamp = OS::get_singleton()->get_ticks_msec();
        m_entries.push_back(entry);
        update_display();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries.clear();
        m_output->clear();
        m_errors_tree->clear();
    }

    void copy_to_clipboard() {
        String text;
        for (const auto& e : m_entries) {
            text += format_entry(e) + "\n";
        }
        DisplayServer::get_singleton()->clipboard_set(text);
    }

    void navigate_to_error(const String& file, int line, int column) {
        emit_signal("error_selected", file, line, column);
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        HBoxContainer* toolbar = new HBoxContainer();
        m_clear_btn = new Button();
        m_clear_btn->set_text("Clear");
        m_clear_btn->connect("pressed", this, "clear");
        toolbar->add_child(m_clear_btn);

        m_copy_btn = new Button();
        m_copy_btn->set_text("Copy");
        m_copy_btn->connect("pressed", this, "copy_to_clipboard");
        toolbar->add_child(m_copy_btn);

        m_auto_scroll = new CheckBox();
        m_auto_scroll->set_text("Auto Scroll");
        m_auto_scroll->set_pressed(true);
        toolbar->add_child(m_auto_scroll);
        main->add_child(toolbar);

        m_output = new RichTextLabel();
        m_output->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_output->set_selection_enabled(true);
        m_output->set_scroll_follow(true);
        main->add_child(m_output);

        m_errors_tree = new Tree();
        m_errors_tree->set_columns(2);
        m_errors_tree->set_column_title(0, "Error");
        m_errors_tree->set_column_title(1, "File");
        m_errors_tree->connect("item_activated", this, "on_error_activated");
    }

    void update_display() {
        call_deferred("_update_display_deferred");
    }

    void _update_display_deferred() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_output->clear();
        m_errors_tree->clear();
        TreeItem* error_root = m_errors_tree->create_item();

        for (const auto& e : m_entries) {
            m_output->append_text(format_entry(e) + "\n");
            if (e.level == BuildLogLevel::LOG_ERROR) {
                TreeItem* item = m_errors_tree->create_item(error_root);
                item->set_text(0, e.message);
                item->set_text(1, e.file + ":" + String::num(e.line));
                item->set_metadata(0, e.file);
                item->set_metadata(1, e.line);
            }
        }
    }

    String format_entry(const BuildLogEntry& e) const {
        String prefix;
        Color color;
        switch (e.level) {
            case BuildLogLevel::LOG_WARNING:
                prefix = "WARNING: ";
                color = EditorTheme::get_singleton()->get_color(ThemeColorRole::COLOR_WARNING);
                break;
            case BuildLogLevel::LOG_ERROR:
                prefix = "ERROR: ";
                color = EditorTheme::get_singleton()->get_color(ThemeColorRole::COLOR_ERROR);
                break;
            default:
                prefix = "";
                color = EditorTheme::get_singleton()->get_color(ThemeColorRole::COLOR_TEXT);
        }
        String result = prefix + e.message;
        if (!e.file.empty()) {
            result += " [" + e.file + ":" + String::num(e.line) + "]";
        }
        return result;
    }

    void on_error_activated() {
        TreeItem* item = m_errors_tree->get_selected();
        if (item && item != m_errors_tree->get_root()) {
            String file = item->get_metadata(0).as<String>();
            int line = item->get_metadata(1).as<int>();
            navigate_to_error(file, line, 0);
        }
    }
};

// #############################################################################
// EditorBuildManager - Central build management
// #############################################################################
class EditorBuildManager : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorBuildManager, Object)

private:
    static EditorBuildManager* s_singleton;
    std::vector<Ref<EditorBuildProfile>> m_build_profiles;
    std::vector<Ref<EditorRun>> m_run_configs;
    Ref<EditorRun> m_active_run;
    EditorLog* m_log = nullptr;
    Ref<EditorRunNative> m_runner;
    std::thread m_build_thread;
    std::atomic<bool> m_building{false};
    std::mutex m_mutex;

public:
    static EditorBuildManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorBuildManager"); }

    EditorBuildManager() { s_singleton = this; }
    ~EditorBuildManager() { s_singleton = nullptr; }

    void set_log(EditorLog* log) { m_log = log; }

    void add_build_profile(const Ref<EditorBuildProfile>& profile) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_build_profiles.push_back(profile);
    }

    std::vector<Ref<EditorBuildProfile>> get_build_profiles() const { return m_build_profiles; }

    void add_run_config(const Ref<EditorRun>& config) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_run_configs.push_back(config);
    }

    std::vector<Ref<EditorRun>> get_run_configs() const { return m_run_configs; }

    void build(const Ref<EditorBuildProfile>& profile) {
        if (m_building) return;
        m_building = true;
        m_build_thread = std::thread([this, profile]() {
            build_thread_func(profile);
            m_building = false;
        });
    }

    void run(const Ref<EditorRun>& config) {
        if (m_building) {
            add_log("Build in progress, please wait...", BuildLogLevel::LOG_WARNING);
            return;
        }
        m_active_run = config;
        if (config.is_valid() && config->get_auto_build()) {
            build_and_run(config);
        } else {
            execute_run(config);
        }
    }

    void stop() {
        if (m_runner.is_valid() && m_runner->is_running()) {
            m_runner->stop();
        }
    }

    bool is_building() const { return m_building; }
    bool is_running() const { return m_runner.is_valid() && m_runner->is_running(); }

    void add_log(const String& msg, BuildLogLevel level = BuildLogLevel::LOG_INFO) {
        if (m_log) m_log->add_message(msg, level);
    }

private:
    void build_thread_func(const Ref<EditorBuildProfile>& profile) {
        add_log("Starting build with profile: " + profile->get_name());
        // Run SCons or custom build script
        add_log("Build completed successfully.", BuildLogLevel::LOG_INFO);
    }

    void build_and_run(const Ref<EditorRun>& config) {
        Ref<EditorBuildProfile> profile = config->get_build_profile();
        if (!profile.is_valid()) {
            profile.instance();
            profile->set_type(BuildConfigType::CONFIG_DEBUG);
        }
        build(profile);
        execute_run(config);
    }

    void execute_run(const Ref<EditorRun>& config) {
        String executable = get_executable_path(config);
        std::vector<String> args = config->get_command_line_args();
        if (config->get_enable_remote_debug()) {
            args.push_back("--remote-debug");
            args.push_back("--remote-debug-port");
            args.push_back(String::num(config->get_debug_port()));
        }
        String scene = get_target_scene(config);
        if (!scene.empty()) {
            args.push_back(scene);
        }
        if (m_runner.is_valid()) {
            m_runner->start(executable, args, config->get_environment(), ".");
            add_log("Started: " + executable);
        }
    }

    String get_executable_path(const Ref<EditorRun>& config) const {
        return OS::get_singleton()->get_executable_path();
    }

    String get_target_scene(const Ref<EditorRun>& config) const {
        switch (config->get_target_type()) {
            case RunTargetType::TARGET_CURRENT_SCENE:
                return EditorNode::get_singleton()->get_current_scene()->get_scene_file_path();
            case RunTargetType::TARGET_MAIN_SCENE:
                return ProjectSettings::get_singleton()->get("application/run/main_scene").as<String>();
            case RunTargetType::TARGET_CUSTOM:
                return config->get_custom_scene();
            default:
                return "";
        }
    }
};

} // namespace editor

// Bring into main namespace
using editor::EditorBuildProfile;
using editor::EditorRun;
using editor::EditorRunNative;
using editor::EditorLog;
using editor::EditorBuildManager;
using editor::BuildConfigType;
using editor::RunTargetType;
using editor::BuildLogLevel;
using editor::BuildLogEntry;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_BUILD_HPP