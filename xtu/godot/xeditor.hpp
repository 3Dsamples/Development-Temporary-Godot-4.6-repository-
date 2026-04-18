// include/xtu/godot/xeditor.hpp
// xtensor-unified - Editor system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_HPP
#define XTU_GODOT_XEDITOR_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xinput.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/parallel/xparallel.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorNode;
class EditorPlugin;
class EditorInspector;
class EditorFileSystem;
class EditorResourcePicker;
class EditorUndoRedoManager;
class EditorSpinSlider;
class EditorAssetInstaller;
class EditorExportPlatform;
class EditorSettings;

// #############################################################################
// Editor plugin types
// #############################################################################
enum class EditorPluginType : uint8_t {
    TYPE_GENERAL = 0,
    TYPE_2D = 1,
    TYPE_3D = 2,
    TYPE_SCRIPT = 3,
    TYPE_ASSET_LIB = 4,
    TYPE_CUSTOM = 5
};

// #############################################################################
// Editor notification types
// #############################################################################
enum class EditorNotification : uint32_t {
    NOTIFICATION_READY = 0,
    NOTIFICATION_ENTER_TREE = 1,
    NOTIFICATION_EXIT_TREE = 2,
    NOTIFICATION_SCENE_CHANGED = 3,
    NOTIFICATION_SCENE_CLOSED = 4,
    NOTIFICATION_RESOURCE_SAVED = 5,
    NOTIFICATION_EDITOR_QUIT = 6,
    NOTIFICATION_EDITOR_SETTINGS_CHANGED = 7,
    NOTIFICATION_PROJECT_SETTINGS_CHANGED = 8
};

// #############################################################################
// Editor file system import modes
// #############################################################################
enum class EditorFileSystemImportMode : uint8_t {
    IMPORT_MODE_SINGLE = 0,
    IMPORT_MODE_MULTIPLE = 1,
    IMPORT_MODE_KEEP = 2
};

// #############################################################################
// Editor export platform types
// #############################################################################
enum class EditorExportPlatformType : uint8_t {
    PLATFORM_WINDOWS = 0,
    PLATFORM_LINUX = 1,
    PLATFORM_MACOS = 2,
    PLATFORM_ANDROID = 3,
    PLATFORM_IOS = 4,
    PLATFORM_WEB = 5,
    PLATFORM_CUSTOM = 6
};

// #############################################################################
// Editor undo redo operation types
// #############################################################################
enum class EditorUndoRedoOperationType : uint8_t {
    OPERATION_ACTION = 0,
    OPERATION_REFERENCE = 1,
    OPERATION_METHOD = 2,
    OPERATION_PROPERTY = 3
};

// #############################################################################
// Editor inspector categories
// #############################################################################
enum class EditorInspectorCategory : uint8_t {
    CATEGORY_TRANSFORM = 0,
    CATEGORY_MATERIAL = 1,
    CATEGORY_PHYSICS = 2,
    CATEGORY_RENDERING = 3,
    CATEGORY_SCRIPT = 4,
    CATEGORY_CUSTOM = 5
};

// #############################################################################
// Editor asset installer states
// #############################################################################
enum class EditorAssetInstallerState : uint8_t {
    STATE_IDLE = 0,
    STATE_DOWNLOADING = 1,
    STATE_EXTRACTING = 2,
    STATE_INSTALLING = 3,
    STATE_COMPLETED = 4,
    STATE_ERROR = 5
};

// #############################################################################
// EditorUndoRedoManager - Undo/redo system
// #############################################################################
class EditorUndoRedoManager : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorUndoRedoManager, Object)

public:
    struct Operation {
        StringName name;
        Object* object = nullptr;
        StringName property;
        Variant old_value;
        Variant new_value;
        EditorUndoRedoOperationType type = EditorUndoRedoOperationType::OPERATION_ACTION;
        uint64_t timestamp = 0;
    };

private:
    static EditorUndoRedoManager* s_singleton;
    std::vector<std::vector<Operation>> m_undo_stack;
    std::vector<std::vector<Operation>> m_redo_stack;
    int32_t m_current_action_level = 0;
    std::vector<Operation> m_current_operations;
    size_t m_max_steps = 100;
    bool m_merge_mode = false;
    std::mutex m_mutex;

public:
    static EditorUndoRedoManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorUndoRedoManager"); }

    EditorUndoRedoManager() { s_singleton = this; }
    ~EditorUndoRedoManager() { s_singleton = nullptr; }

    void create_action(const StringName& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current_action_level == 0) {
            m_current_operations.clear();
        }
        ++m_current_action_level;
    }

    void commit_action(bool execute = true) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current_action_level <= 0) return;
        --m_current_action_level;
        if (m_current_action_level == 0 && !m_current_operations.empty()) {
            m_undo_stack.push_back(m_current_operations);
            m_redo_stack.clear();
            m_current_operations.clear();
            while (m_undo_stack.size() > m_max_steps) {
                m_undo_stack.erase(m_undo_stack.begin());
            }
        }
    }

    void add_do_method(Object* object, const StringName& method, const std::vector<Variant>& args = {}) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current_action_level == 0) return;
        Operation op;
        op.type = EditorUndoRedoOperationType::OPERATION_METHOD;
        op.object = object;
        op.name = method;
        m_current_operations.push_back(op);
    }

    void add_undo_method(Object* object, const StringName& method, const std::vector<Variant>& args = {}) {
        // Mirror for undo
    }

    void add_do_property(Object* object, const StringName& property, const Variant& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current_action_level == 0) return;
        Operation op;
        op.type = EditorUndoRedoOperationType::OPERATION_PROPERTY;
        op.object = object;
        op.property = property;
        op.old_value = object->get(property);
        op.new_value = value;
        m_current_operations.push_back(op);
        object->set(property, value);
    }

    bool undo() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_undo_stack.empty()) return false;
        auto ops = m_undo_stack.back();
        m_undo_stack.pop_back();
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            if (it->type == EditorUndoRedoOperationType::OPERATION_PROPERTY) {
                it->object->set(it->property, it->old_value);
            }
        }
        m_redo_stack.push_back(ops);
        return true;
    }

    bool redo() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_redo_stack.empty()) return false;
        auto ops = m_redo_stack.back();
        m_redo_stack.pop_back();
        for (const auto& op : ops) {
            if (op.type == EditorUndoRedoOperationType::OPERATION_PROPERTY) {
                op.object->set(op.property, op.new_value);
            }
        }
        m_undo_stack.push_back(ops);
        return true;
    }

    void clear_history() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_undo_stack.clear();
        m_redo_stack.clear();
        m_current_operations.clear();
    }

    bool has_undo() const { return !m_undo_stack.empty(); }
    bool has_redo() const { return !m_redo_stack.empty(); }
    StringName get_current_action_name() const { return StringName(); }
};

// #############################################################################
// EditorSettings - Persistent editor configuration
// #############################################################################
class EditorSettings : public Resource {
    XTU_GODOT_REGISTER_CLASS(EditorSettings, Resource)

private:
    static EditorSettings* s_singleton;
    std::unordered_map<StringName, Variant> m_settings;
    std::unordered_map<StringName, PropertyInfo> m_property_info;
    std::string m_config_path;
    std::mutex m_mutex;

public:
    static EditorSettings* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorSettings"); }

    EditorSettings() { s_singleton = this; load(); }
    ~EditorSettings() { s_singleton = nullptr; save(); }

    void set(const StringName& key, const Variant& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_settings[key] = value;
    }

    Variant get(const StringName& key, const Variant& default_val = Variant()) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_settings.find(key);
        return it != m_settings.end() ? it->second : default_val;
    }

    bool has(const StringName& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_settings.find(key) != m_settings.end();
    }

    void erase(const StringName& key) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_settings.erase(key);
    }

    void set_initial_value(const StringName& key, const Variant& value, bool restart_if_changed = false) {
        if (!has(key)) set(key, value);
    }

    void add_property_info(const PropertyInfo& info) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_property_info[info.name] = info;
    }

    void load() {
        // Load from config file
    }

    void save() {
        io::json::JsonValue json;
        for (const auto& kv : m_settings) {
            // Serialize
        }
        io::json::save_json(m_config_path.empty() ? "editor_settings.json" : m_config_path, json);
    }

    void set_config_path(const std::string& path) { m_config_path = path; }
};

// #############################################################################
// EditorFileSystem - File monitoring and importing
// #############################################################################
class EditorFileSystem : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorFileSystem, Object)

private:
    static EditorFileSystem* s_singleton;
    std::string m_root_path;
    std::unordered_map<std::string, uint64_t> m_file_cache;
    std::unordered_set<std::string> m_pending_imports;
    std::queue<std::string> m_import_queue;
    std::atomic<bool> m_scanning{false};
    std::mutex m_mutex;
    std::thread m_worker;

public:
    static EditorFileSystem* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorFileSystem"); }

    EditorFileSystem() { s_singleton = this; }
    ~EditorFileSystem() { s_singleton = nullptr; }

    void set_root_path(const std::string& path) { m_root_path = path; }
    std::string get_root_path() const { return m_root_path; }

    void scan() {
        if (m_scanning) return;
        m_scanning = true;
        m_worker = std::thread([this]() {
            scan_directory(m_root_path);
            process_import_queue();
            m_scanning = false;
        });
    }

    void scan_changes() {
        // Incremental scan
    }

    bool is_scanning() const { return m_scanning; }

    void reimport_files(const std::vector<std::string>& files) {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& f : files) {
            m_import_queue.push(f);
        }
    }

    std::string get_file_type(const std::string& path) const {
        size_t dot = path.find_last_of('.');
        return dot != std::string::npos ? path.substr(dot + 1) : "";
    }

    void update_file(const std::string& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pending_imports.insert(path);
    }

private:
    void scan_directory(const std::string& path) {
        // Recursive directory scan
    }

    void process_import_queue() {
        while (!m_import_queue.empty()) {
            std::string path = m_import_queue.front();
            m_import_queue.pop();
            import_file(path);
        }
    }

    void import_file(const std::string& path) {
        // Import based on extension
    }
};

// #############################################################################
// EditorInspector - Property inspector
// #############################################################################
class EditorInspector : public ScrollContainer {
    XTU_GODOT_REGISTER_CLASS(EditorInspector, ScrollContainer)

private:
    Object* m_object = nullptr;
    std::vector<PropertyInfo> m_properties;
    std::unordered_map<StringName, Control*> m_property_editors;
    bool m_read_only = false;
    bool m_use_doc_hints = true;
    StringName m_section;

public:
    static StringName get_class_static() { return StringName("EditorInspector"); }

    void edit(Object* object) {
        m_object = object;
        if (!object) {
            clear();
            return;
        }
        m_properties = object->get_property_list();
        rebuild();
    }

    Object* get_edited_object() const { return m_object; }

    void refresh() {
        if (m_object) {
            for (const auto& prop : m_properties) {
                auto it = m_property_editors.find(prop.name);
                if (it != m_property_editors.end()) {
                    update_editor(it->second, prop, m_object->get(prop.name));
                }
            }
        }
    }

    void set_read_only(bool read_only) { m_read_only = read_only; }
    bool is_read_only() const { return m_read_only; }

    void set_use_doc_hints(bool use) { m_use_doc_hints = use; }
    bool is_using_doc_hints() const { return m_use_doc_hints; }

private:
    void clear() {
        for (auto& kv : m_property_editors) {
            kv.second->queue_free();
        }
        m_property_editors.clear();
        m_properties.clear();
    }

    void rebuild() {
        clear();
        EditorInspectorCategory current_category = EditorInspectorCategory::CATEGORY_CUSTOM;
        for (const auto& prop : m_properties) {
            if (!(static_cast<uint32_t>(prop.usage) & static_cast<uint32_t>(PropertyUsage::EDITOR))) continue;
            Control* editor = create_editor_for_property(prop);
            if (editor) {
                m_property_editors[prop.name] = editor;
                add_child(editor);
                update_editor(editor, prop, m_object->get(prop.name));
            }
        }
    }

    Control* create_editor_for_property(const PropertyInfo& prop) {
        switch (prop.type) {
            case VariantType::BOOL: return new CheckBox();
            case VariantType::INT: return new SpinBox();
            case VariantType::FLOAT: return new EditorSpinSlider();
            case VariantType::STRING: return new LineEdit();
            case VariantType::VECTOR2: return new Control(); // Vector2 editor
            case VariantType::VECTOR3: return new Control(); // Vector3 editor
            case VariantType::COLOR: return new ColorPickerButton();
            case VariantType::OBJECT: return new EditorResourcePicker();
            default: return new Label();
        }
    }

    void update_editor(Control* editor, const PropertyInfo& prop, const Variant& value) {
        // Update editor control with value
    }

    void _property_changed(const StringName& property, const Variant& new_value) {
        if (m_object && !m_read_only) {
            EditorUndoRedoManager::get_singleton()->create_action("Set " + property.string());
            EditorUndoRedoManager::get_singleton()->add_do_property(m_object, property, new_value);
            EditorUndoRedoManager::get_singleton()->commit_action();
        }
    }
};

// #############################################################################
// EditorResourcePicker - Resource selection control
// #############################################################################
class EditorResourcePicker : public HBoxContainer {
    XTU_GODOT_REGISTER_CLASS(EditorResourcePicker, HBoxContainer)

private:
    Ref<Resource> m_resource;
    StringName m_base_type;
    bool m_editable = true;
    Button* m_button = nullptr;
    TextureRect* m_preview = nullptr;

public:
    static StringName get_class_static() { return StringName("EditorResourcePicker"); }

    EditorResourcePicker() {
        m_button = new Button();
        m_button->set_text("Load");
        m_preview = new TextureRect();
        add_child(m_preview);
        add_child(m_button);
        m_button->connect("pressed", this, "_on_button_pressed");
    }

    void set_base_type(const StringName& type) { m_base_type = type; }
    StringName get_base_type() const { return m_base_type; }

    void set_edited_resource(const Ref<Resource>& res) {
        m_resource = res;
        update_preview();
    }

    Ref<Resource> get_edited_resource() const { return m_resource; }

    void set_editable(bool editable) {
        m_editable = editable;
        m_button->set_disabled(!editable);
    }

    bool is_editable() const { return m_editable; }

    void _on_button_pressed() {
        // Open resource picker dialog
        emit_signal("resource_selected", m_resource);
    }

private:
    void update_preview() {
        if (m_resource.is_valid()) {
            m_button->set_text(m_resource->get_name().string());
        } else {
            m_button->set_text("Empty");
        }
    }
};

// #############################################################################
// EditorSpinSlider - Numeric slider for editor
// #############################################################################
class EditorSpinSlider : public HBoxContainer {
    XTU_GODOT_REGISTER_CLASS(EditorSpinSlider, HBoxContainer)

private:
    LineEdit* m_line_edit = nullptr;
    Slider* m_slider = nullptr;
    float m_value = 0.0f;
    float m_min_value = 0.0f;
    float m_max_value = 100.0f;
    float m_step = 0.01f;
    bool m_use_float = true;

public:
    static StringName get_class_static() { return StringName("EditorSpinSlider"); }

    EditorSpinSlider() {
        m_line_edit = new LineEdit();
        m_slider = new Slider();
        m_slider->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(m_line_edit);
        add_child(m_slider);
        m_line_edit->connect("text_changed", this, "_on_text_changed");
        m_slider->connect("value_changed", this, "_on_slider_changed");
    }

    void set_value(float value) {
        value = std::clamp(value, m_min_value, m_max_value);
        m_value = value;
        m_line_edit->set_text(String::num(value, m_use_float ? 3 : 0));
        m_slider->set_value(value);
    }

    float get_value() const { return m_value; }

    void set_min(float min) { m_min_value = min; m_slider->set_min(min); }
    void set_max(float max) { m_max_value = max; m_slider->set_max(max); }
    void set_step(float step) { m_step = step; m_slider->set_step(step); }

    void _on_text_changed(const String& text) {
        float val = text.to_float();
        if (val != m_value) {
            set_value(val);
            emit_signal("value_changed", m_value);
        }
    }

    void _on_slider_changed(float value) {
        if (value != m_value) {
            set_value(value);
            emit_signal("value_changed", m_value);
        }
    }
};

// #############################################################################
// EditorPlugin - Base class for editor plugins
// #############################################################################
class EditorPlugin : public Node {
    XTU_GODOT_REGISTER_CLASS(EditorPlugin, Node)

private:
    EditorNode* m_editor = nullptr;
    EditorPluginType m_type = EditorPluginType::TYPE_GENERAL;
    bool m_enabled = true;

public:
    static StringName get_class_static() { return StringName("EditorPlugin"); }

    virtual StringName get_plugin_name() const { return StringName(); }
    virtual EditorPluginType get_plugin_type() const { return m_type; }

    virtual void _enter_tree() override {
        m_editor = EditorNode::get_singleton();
        if (m_editor) m_editor->add_plugin(this);
    }

    virtual void _exit_tree() override {
        if (m_editor) m_editor->remove_plugin(this);
    }

    virtual void _notification(int p_what) override {
        switch (static_cast<EditorNotification>(p_what)) {
            case EditorNotification::NOTIFICATION_READY: _ready(); break;
            case EditorNotification::NOTIFICATION_SCENE_CHANGED: _scene_changed(); break;
            default: break;
        }
    }

    virtual void _ready() {}
    virtual void _scene_changed() {}
    virtual void apply_changes() {}
    virtual bool handles(Object* object) const { return false; }
    virtual void edit(Object* object) {}
    virtual void make_visible(bool visible) {}

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void add_control_to_container(Control* control, const StringName& container) {}
    void remove_control_from_container(Control* control) {}

    void add_tool_menu_item(const StringName& name, const StringName& callback) {}
    void add_tool_submenu_item(const StringName& name, PopupMenu* submenu) {}

    EditorNode* get_editor() const { return m_editor; }
    EditorUndoRedoManager* get_undo_redo() const { return EditorUndoRedoManager::get_singleton(); }
};

// #############################################################################
// EditorNode - Main editor window controller
// #############################################################################
class EditorNode : public Node {
    XTU_GODOT_REGISTER_CLASS(EditorNode, Node)

private:
    static EditorNode* s_singleton;
    std::vector<EditorPlugin*> m_plugins;
    Node* m_current_scene = nullptr;
    bool m_quitting = false;
    std::mutex m_mutex;

public:
    static EditorNode* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorNode"); }

    EditorNode() { s_singleton = this; }
    ~EditorNode() { s_singleton = nullptr; }

    void add_plugin(EditorPlugin* plugin) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_plugins.begin(), m_plugins.end(), plugin) == m_plugins.end()) {
            m_plugins.push_back(plugin);
        }
    }

    void remove_plugin(EditorPlugin* plugin) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_plugins.begin(), m_plugins.end(), plugin);
        if (it != m_plugins.end()) m_plugins.erase(it);
    }

    void edit_node(Node* node) {
        for (auto* plugin : m_plugins) {
            if (plugin->is_enabled() && plugin->handles(node)) {
                plugin->edit(node);
            }
        }
    }

    void edit_resource(const Ref<Resource>& res) {
        for (auto* plugin : m_plugins) {
            if (plugin->is_enabled() && plugin->handles(res.ptr())) {
                plugin->edit(res.ptr());
            }
        }
    }

    void set_current_scene(Node* scene) {
        m_current_scene = scene;
        notify_plugins(EditorNotification::NOTIFICATION_SCENE_CHANGED);
    }

    Node* get_current_scene() const { return m_current_scene; }

    void save_scene() {
        if (m_current_scene) {
            // Save scene to file
        }
    }

    void save_all_scenes() {}

    void quit() {
        m_quitting = true;
        notify_plugins(EditorNotification::NOTIFICATION_EDITOR_QUIT);
        get_tree()->quit();
    }

    bool is_quitting() const { return m_quitting; }

    void notify_plugins(EditorNotification what) {
        for (auto* plugin : m_plugins) {
            plugin->notification(static_cast<int>(what));
        }
    }
};

// #############################################################################
// EditorAssetInstaller - Asset library package installer
// #############################################################################
class EditorAssetInstaller : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorAssetInstaller, Object)

private:
    std::string m_asset_path;
    std::string m_target_path;
    EditorAssetInstallerState m_state = EditorAssetInstallerState::STATE_IDLE;
    float m_progress = 0.0f;
    std::string m_error_message;
    std::function<void()> m_completion_callback;

public:
    static StringName get_class_static() { return StringName("EditorAssetInstaller"); }

    void set_asset_path(const std::string& path) { m_asset_path = path; }
    void set_target_path(const std::string& path) { m_target_path = path; }

    void install() {
        m_state = EditorAssetInstallerState::STATE_EXTRACTING;
        // Extract and install asset
        m_state = EditorAssetInstallerState::STATE_COMPLETED;
        m_progress = 1.0f;
        if (m_completion_callback) m_completion_callback();
    }

    EditorAssetInstallerState get_state() const { return m_state; }
    float get_progress() const { return m_progress; }
    std::string get_error_message() const { return m_error_message; }

    void set_completion_callback(std::function<void()> cb) { m_completion_callback = cb; }
};

// #############################################################################
// EditorExportPlatform - Base class for export platforms
// #############################################################################
class EditorExportPlatform : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatform, RefCounted)

public:
    struct ExportPreset {
        StringName name;
        EditorExportPlatformType platform;
        std::unordered_map<StringName, Variant> options;
        std::vector<std::string> include_files;
        std::vector<std::string> exclude_files;
        StringName export_path;
    };

private:
    EditorExportPlatformType m_type = EditorExportPlatformType::PLATFORM_CUSTOM;
    std::vector<ExportPreset> m_presets;

public:
    static StringName get_class_static() { return StringName("EditorExportPlatform"); }

    EditorExportPlatformType get_type() const { return m_type; }

    virtual StringName get_name() const { return StringName(); }
    virtual StringName get_os_name() const { return StringName(); }
    virtual std::vector<StringName> get_binary_extensions() const { return {}; }

    virtual void get_export_options(std::vector<PropertyInfo>& options) const {}
    virtual bool can_export() const { return true; }

    virtual bool export_project(const ExportPreset& preset, const std::string& path,
                                bool debug = false, const std::string& password = "") {
        return false;
    }

    void add_preset(const ExportPreset& preset) { m_presets.push_back(preset); }
    const std::vector<ExportPreset>& get_presets() const { return m_presets; }
};

} // namespace editor

// Bring into main namespace
using editor::EditorNode;
using editor::EditorPlugin;
using editor::EditorInspector;
using editor::EditorFileSystem;
using editor::EditorResourcePicker;
using editor::EditorUndoRedoManager;
using editor::EditorSpinSlider;
using editor::EditorAssetInstaller;
using editor::EditorExportPlatform;
using editor::EditorSettings;
using editor::EditorPluginType;
using editor::EditorNotification;
using editor::EditorFileSystemImportMode;
using editor::EditorExportPlatformType;
using editor::EditorUndoRedoOperationType;
using editor::EditorInspectorCategory;
using editor::EditorAssetInstallerState;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_HPP