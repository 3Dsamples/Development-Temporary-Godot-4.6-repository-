// include/xtu/godot/xeditor_settings.hpp
// xtensor-unified - Editor settings and project configuration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_SETTINGS_HPP
#define XTU_GODOT_XEDITOR_SETTINGS_HPP

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
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xinput.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class ProjectSettingsEditor;
class EditorSettingsDialog;
class InputMapEditor;
class PluginConfigDialog;
class FeatureProfileEditor;

// #############################################################################
// Settings section types
// #############################################################################
enum class SettingsSection : uint8_t {
    SECTION_GENERAL = 0,
    SECTION_INPUT = 1,
    SECTION_RENDERING = 2,
    SECTION_AUDIO = 3,
    SECTION_PHYSICS = 4,
    SECTION_NETWORK = 5,
    SECTION_LOCALIZATION = 6,
    SECTION_EDITOR = 7,
    SECTION_PLUGINS = 8
};

// #############################################################################
// Input action type
// #############################################################################
enum class InputActionType : uint8_t {
    ACTION_BUTTON = 0,
    ACTION_AXIS = 1,
    ACTION_VECTOR = 2
};

// #############################################################################
// Plugin status
// #############################################################################
enum class PluginStatus : uint8_t {
    STATUS_DISABLED = 0,
    STATUS_ENABLED = 1,
    STATUS_ERROR = 2
};

// #############################################################################
// ProjectSettingsEditor - Main project settings window
// #############################################################################
class ProjectSettingsEditor : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(ProjectSettingsEditor, AcceptDialog)

private:
    static ProjectSettingsEditor* s_singleton;
    TabContainer* m_tabs = nullptr;
    Tree* m_settings_tree = nullptr;
    LineEdit* m_search_box = nullptr;
    std::unordered_map<String, PropertyInfo> m_all_properties;
    std::unordered_map<String, Control*> m_property_editors;
    std::mutex m_mutex;

public:
    static ProjectSettingsEditor* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ProjectSettingsEditor"); }

    ProjectSettingsEditor() {
        s_singleton = this;
        set_title("Project Settings");
        build_ui();
        load_settings();
    }

    ~ProjectSettingsEditor() { s_singleton = nullptr; }

    void set_setting(const String& name, const Variant& value) {
        // Set project setting
    }

    Variant get_setting(const String& name) const {
        // Get project setting
        return Variant();
    }

    void add_custom_setting(const String& name, VariantType type, const Variant& default_value,
                            PropertyHint hint = PropertyHint::NONE, const String& hint_string = "") {
        // Register custom setting
    }

    void search(const String& text) {
        std::lock_guard<std::mutex> lock(m_mutex);
        filter_settings(text);
    }

    void _ok_pressed() override {
        save_settings();
        AcceptDialog::_ok_pressed();
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        // Search bar
        HBoxContainer* search_row = new HBoxContainer();
        search_row->add_child(new Label("Search:"));
        m_search_box = new LineEdit();
        m_search_box->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_search_box->connect("text_changed", this, "on_search_changed");
        search_row->add_child(m_search_box);
        main->add_child(search_row);

        // Tabs
        m_tabs = new TabContainer();
        m_tabs->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(m_tabs);

        // General settings tab
        m_settings_tree = new Tree();
        m_settings_tree->set_columns(2);
        m_settings_tree->set_column_title(0, "Setting");
        m_settings_tree->set_column_title(1, "Value");
        m_settings_tree->set_hide_root(true);
        m_tabs->add_child(m_settings_tree);
        m_tabs->set_tab_title(0, "General");

        // Input Map tab
        InputMapEditor* input_editor = new InputMapEditor();
        m_tabs->add_child(input_editor);
        m_tabs->set_tab_title(1, "Input Map");

        // Localization tab
        Control* localization_tab = new Control();
        m_tabs->add_child(localization_tab);
        m_tabs->set_tab_title(2, "Localization");

        // Plugins tab
        Control* plugins_tab = new Control();
        m_tabs->add_child(plugins_tab);
        m_tabs->set_tab_title(3, "Plugins");
    }

    void load_settings() {}
    void save_settings() {}
    void filter_settings(const String& text) {}
    void on_search_changed(const String& text) { search(text); }
};

// #############################################################################
// EditorSettingsDialog - Editor preferences dialog
// #############################################################################
class EditorSettingsDialog : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(EditorSettingsDialog, AcceptDialog)

private:
    Tree* m_sections_tree = nullptr;
    Control* m_content_panel = nullptr;
    LineEdit* m_search_box = nullptr;
    std::unordered_map<String, std::vector<PropertyInfo>> m_section_properties;

public:
    static StringName get_class_static() { return StringName("EditorSettingsDialog"); }

    EditorSettingsDialog() {
        set_title("Editor Settings");
        build_ui();
        load_sections();
    }

    void select_section(const String& section) {
        // Display section properties
    }

    void _ok_pressed() override {
        save_settings();
        AcceptDialog::_ok_pressed();
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        HBoxContainer* search_row = new HBoxContainer();
        search_row->add_child(new Label("Search:"));
        m_search_box = new LineEdit();
        m_search_box->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        search_row->add_child(m_search_box);
        main->add_child(search_row);

        HSplitContainer* split = new HSplitContainer();
        split->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(split);

        m_sections_tree = new Tree();
        m_sections_tree->set_hide_root(true);
        m_sections_tree->connect("item_selected", this, "on_section_selected");
        split->add_child(m_sections_tree);

        m_content_panel = new Control();
        m_content_panel->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        split->add_child(m_content_panel);
    }

    void load_sections() {
        TreeItem* root = m_sections_tree->create_item();
        add_section(root, "Interface", "interface");
        add_section(root, "Text Editor", "text_editor");
        add_section(root, "Script Editor", "script_editor");
        add_section(root, "File System", "file_system");
        add_section(root, "Docks", "docks");
        add_section(root, "Run", "run");
        add_section(root, "Network", "network");
    }

    void add_section(TreeItem* parent, const String& name, const String& key) {
        TreeItem* item = m_sections_tree->create_item(parent);
        item->set_text(0, name);
        item->set_metadata(0, key);
    }

    void on_section_selected() {
        TreeItem* item = m_sections_tree->get_selected();
        if (item) {
            select_section(item->get_metadata(0).as<String>());
        }
    }

    void save_settings() {}
};

// #############################################################################
// InputMapEditor - Input action mapping editor
// #############################################################################
class InputMapEditor : public Control {
    XTU_GODOT_REGISTER_CLASS(InputMapEditor, Control)

private:
    Tree* m_actions_tree = nullptr;
    LineEdit* m_action_name = nullptr;
    Button* m_add_action_btn = nullptr;
    Button* m_add_event_btn = nullptr;
    Button* m_remove_btn = nullptr;
    OptionButton* m_device_filter = nullptr;
    std::unordered_map<String, std::vector<Ref<InputEvent>>> m_pending_events;

public:
    static StringName get_class_static() { return StringName("InputMapEditor"); }

    InputMapEditor() {
        build_ui();
        load_actions();
    }

    void add_action(const String& name, float deadzone = 0.5f) {
        if (!InputMap::get_singleton()->has_action(name)) {
            InputMap::get_singleton()->add_action(name, deadzone);
            refresh_actions();
        }
    }

    void remove_action(const String& name) {
        InputMap::get_singleton()->erase_action(name);
        refresh_actions();
    }

    void add_event_to_action(const String& action, const Ref<InputEvent>& event) {
        InputMap::get_singleton()->action_add_event(action, event);
        refresh_actions();
    }

    void remove_event_from_action(const String& action, int idx) {
        InputMap::get_singleton()->action_erase_event(action, idx);
        refresh_actions();
    }

    void refresh_actions() {
        m_actions_tree->clear();
        TreeItem* root = m_actions_tree->create_item();
        for (const auto& action : InputMap::get_singleton()->get_actions()) {
            TreeItem* item = m_actions_tree->create_item(root);
            item->set_text(0, action.string());
            item->set_metadata(0, action);

            auto events = InputMap::get_singleton()->action_get_events(action);
            for (const auto& event : events) {
                TreeItem* event_item = m_actions_tree->create_item(item);
                event_item->set_text(0, event->as_text().string());
            }
        }
    }

    void listen_for_input(const String& action) {
        // Start input listening mode
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        // Action add bar
        HBoxContainer* add_bar = new HBoxContainer();
        m_action_name = new LineEdit();
        m_action_name->set_placeholder("New Action Name");
        m_action_name->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_bar->add_child(m_action_name);

        m_add_action_btn = new Button();
        m_add_action_btn->set_text("Add");
        m_add_action_btn->connect("pressed", this, "on_add_action");
        add_bar->add_child(m_add_action_btn);
        main->add_child(add_bar);

        // Actions tree
        m_actions_tree = new Tree();
        m_actions_tree->set_columns(1);
        m_actions_tree->set_column_title(0, "Action");
        m_actions_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(m_actions_tree);

        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();
        m_add_event_btn = new Button();
        m_add_event_btn->set_text("Add Event");
        m_add_event_btn->connect("pressed", this, "on_add_event");
        toolbar->add_child(m_add_event_btn);

        m_remove_btn = new Button();
        m_remove_btn->set_text("Remove");
        m_remove_btn->connect("pressed", this, "on_remove");
        toolbar->add_child(m_remove_btn);

        m_device_filter = new OptionButton();
        m_device_filter->add_item("All Devices");
        m_device_filter->add_item("Keyboard");
        m_device_filter->add_item("Mouse");
        m_device_filter->add_item("Joypad");
        toolbar->add_child(m_device_filter);
        main->add_child(toolbar);
    }

    void load_actions() { refresh_actions(); }
    void on_add_action() {
        String name = m_action_name->get_text();
        if (!name.empty()) {
            add_action(name);
            m_action_name->clear();
        }
    }
    void on_add_event() {}
    void on_remove() {}
};

// #############################################################################
// PluginConfigDialog - Plugin creation/configuration
// #############################################################################
class PluginConfigDialog : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(PluginConfigDialog, AcceptDialog)

private:
    LineEdit* m_name_edit = nullptr;
    LineEdit* m_subfolder_edit = nullptr;
    TextEdit* m_description_edit = nullptr;
    LineEdit* m_author_edit = nullptr;
    LineEdit* m_version_edit = nullptr;
    LineEdit* m_script_edit = nullptr;
    CheckBox* m_activate_now = nullptr;

public:
    static StringName get_class_static() { return StringName("PluginConfigDialog"); }

    PluginConfigDialog() {
        set_title("Create Plugin");
        build_ui();
    }

    void _ok_pressed() override {
        String name = m_name_edit->get_text();
        String subfolder = m_subfolder_edit->get_text();
        String description = m_description_edit->get_text();
        String author = m_author_edit->get_text();
        String version = m_version_edit->get_text();
        String script = m_script_edit->get_text();

        create_plugin(name, subfolder, description, author, version, script);
        AcceptDialog::_ok_pressed();
    }

private:
    void build_ui() {
        VBoxContainer* vbox = new VBoxContainer();
        add_child(vbox);

        // Plugin Name
        HBoxContainer* name_row = new HBoxContainer();
        name_row->add_child(new Label("Plugin Name:"));
        m_name_edit = new LineEdit();
        m_name_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        name_row->add_child(m_name_edit);
        vbox->add_child(name_row);

        // Subfolder
        HBoxContainer* folder_row = new HBoxContainer();
        folder_row->add_child(new Label("Subfolder:"));
        m_subfolder_edit = new LineEdit();
        m_subfolder_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        folder_row->add_child(m_subfolder_edit);
        vbox->add_child(folder_row);

        // Description
        vbox->add_child(new Label("Description:"));
        m_description_edit = new TextEdit();
        m_description_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        vbox->add_child(m_description_edit);

        // Author
        HBoxContainer* author_row = new HBoxContainer();
        author_row->add_child(new Label("Author:"));
        m_author_edit = new LineEdit();
        m_author_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        author_row->add_child(m_author_edit);
        vbox->add_child(author_row);

        // Version
        HBoxContainer* version_row = new HBoxContainer();
        version_row->add_child(new Label("Version:"));
        m_version_edit = new LineEdit();
        m_version_edit->set_text("1.0");
        version_row->add_child(m_version_edit);
        vbox->add_child(version_row);

        // Script
        HBoxContainer* script_row = new HBoxContainer();
        script_row->add_child(new Label("Script:"));
        m_script_edit = new LineEdit();
        m_script_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        script_row->add_child(m_script_edit);
        vbox->add_child(script_row);

        // Activate now
        m_activate_now = new CheckBox();
        m_activate_now->set_text("Activate now");
        m_activate_now->set_pressed(true);
        vbox->add_child(m_activate_now);
    }

    void create_plugin(const String& name, const String& subfolder, const String& description,
                       const String& author, const String& version, const String& script) {
        String plugin_path = "res://addons/" + subfolder + "/";
        DirAccess::make_dir_recursive(plugin_path);

        // Create plugin.cfg
        String cfg_content = "[plugin]\n";
        cfg_content += "name=\"" + name + "\"\n";
        cfg_content += "description=\"" + description + "\"\n";
        cfg_content += "author=\"" + author + "\"\n";
        cfg_content += "version=\"" + version + "\"\n";
        cfg_content += "script=\"" + script + "\"\n";

        Ref<FileAccess> cfg_file = FileAccess::open(plugin_path + "plugin.cfg", FileAccess::WRITE);
        if (cfg_file.is_valid()) {
            cfg_file->store_string(cfg_content);
            cfg_file->close();
        }

        // Create plugin script
        String script_content = "@tool\nextends EditorPlugin\n\n";
        script_content += "func _enter_tree():\n\tpass\n\n";
        script_content += "func _exit_tree():\n\tpass\n";

        Ref<FileAccess> script_file = FileAccess::open(plugin_path + script, FileAccess::WRITE);
        if (script_file.is_valid()) {
            script_file->store_string(script_content);
            script_file->close();
        }
    }
};

// #############################################################################
// FeatureProfileEditor - Platform feature configuration
// #############################################################################
class FeatureProfileEditor : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(FeatureProfileEditor, AcceptDialog)

private:
    Tree* m_features_tree = nullptr;
    LineEdit* m_profile_name = nullptr;
    OptionButton* m_profile_select = nullptr;
    std::unordered_map<String, std::unordered_map<String, bool>> m_profiles;

public:
    static StringName get_class_static() { return StringName("FeatureProfileEditor"); }

    FeatureProfileEditor() {
        set_title("Feature Profiles");
        build_ui();
        load_profiles();
    }

    void add_profile(const String& name) {
        m_profiles[name] = {};
        refresh_profile_list();
    }

    void remove_profile(const String& name) {
        m_profiles.erase(name);
        refresh_profile_list();
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        HBoxContainer* profile_row = new HBoxContainer();
        profile_row->add_child(new Label("Profile:"));
        m_profile_select = new OptionButton();
        m_profile_select->connect("item_selected", this, "on_profile_selected");
        profile_row->add_child(m_profile_select);

        m_profile_name = new LineEdit();
        m_profile_name->set_placeholder("New Profile Name");
        profile_row->add_child(m_profile_name);

        Button* add_btn = new Button();
        add_btn->set_text("Add");
        add_btn->connect("pressed", this, "on_add_profile");
        profile_row->add_child(add_btn);
        main->add_child(profile_row);

        m_features_tree = new Tree();
        m_features_tree->set_columns(2);
        m_features_tree->set_column_title(0, "Feature");
        m_features_tree->set_column_title(1, "Enabled");
        m_features_tree->set_hide_root(true);
        m_features_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(m_features_tree);
    }

    void load_profiles() {
        add_profile("Default");
        add_profile("Mobile");
        add_profile("Web");
    }

    void refresh_profile_list() {
        m_profile_select->clear();
        for (const auto& kv : m_profiles) {
            m_profile_select->add_item(kv.first);
        }
    }

    void on_profile_selected(int idx) {}
    void on_add_profile() {
        String name = m_profile_name->get_text();
        if (!name.empty()) {
            add_profile(name);
            m_profile_name->clear();
        }
    }
};

} // namespace editor

// Bring into main namespace
using editor::ProjectSettingsEditor;
using editor::EditorSettingsDialog;
using editor::InputMapEditor;
using editor::PluginConfigDialog;
using editor::FeatureProfileEditor;
using editor::SettingsSection;
using editor::InputActionType;
using editor::PluginStatus;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_SETTINGS_HPP