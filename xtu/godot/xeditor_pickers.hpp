// include/xtu/godot/xeditor_pickers.hpp
// xtensor-unified - Editor resource pickers and dialogs for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_PICKERS_HPP
#define XTU_GODOT_XEDITOR_PICKERS_HPP

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
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorResourcePicker;
class EditorQuickOpen;
class EditorFileDialog;
class EditorDirDialog;
class EditorPropertySelector;

// #############################################################################
// Resource picker mode
// #############################################################################
enum class ResourcePickerMode : uint8_t {
    MODE_SINGLE = 0,
    MODE_MULTIPLE = 1,
    MODE_FOLDER = 2
};

// #############################################################################
// Quick open search mode
// #############################################################################
enum class QuickOpenMode : uint8_t {
    MODE_ALL = 0,
    MODE_SCENE = 1,
    MODE_SCRIPT = 2,
    MODE_RESOURCE = 3,
    MODE_NODE = 4
};

// #############################################################################
// File dialog mode
// #############################################################################
enum class FileDialogMode : uint8_t {
    MODE_OPEN_FILE = 0,
    MODE_OPEN_FILES = 1,
    MODE_OPEN_DIR = 2,
    MODE_OPEN_ANY = 3,
    MODE_SAVE_FILE = 4
};

// #############################################################################
// File dialog access
// #############################################################################
enum class FileDialogAccess : uint8_t {
    ACCESS_RESOURCES = 0,
    ACCESS_USERDATA = 1,
    ACCESS_FILESYSTEM = 2
};

// #############################################################################
// Search result item
// #############################################################################
struct SearchResultItem {
    String path;
    String name;
    String type;
    float score = 0.0f;
    Ref<Texture2D> icon;
};

// #############################################################################
// EditorResourcePicker - Resource selection control
// #############################################################################
class EditorResourcePicker : public HBoxContainer {
    XTU_GODOT_REGISTER_CLASS(EditorResourcePicker, HBoxContainer)

private:
    Button* m_button = nullptr;
    TextureRect* m_preview = nullptr;
    Label* m_label = nullptr;
    Ref<Resource> m_resource;
    String m_base_type;
    bool m_editable = true;
    bool m_toggle_mode = false;
    ResourcePickerMode m_mode = ResourcePickerMode::MODE_SINGLE;

public:
    static StringName get_class_static() { return StringName("EditorResourcePicker"); }

    EditorResourcePicker() {
        build_ui();
    }

    void set_base_type(const String& type) {
        m_base_type = type;
        update_button_text();
    }

    String get_base_type() const { return m_base_type; }

    void set_edited_resource(const Ref<Resource>& res) {
        m_resource = res;
        update_preview();
        update_button_text();
        emit_signal("resource_changed", res);
    }

    Ref<Resource> get_edited_resource() const { return m_resource; }

    void set_editable(bool editable) {
        m_editable = editable;
        m_button->set_disabled(!editable);
    }

    bool is_editable() const { return m_editable; }

    void set_toggle_mode(bool enable) {
        m_toggle_mode = enable;
        m_button->set_toggle_mode(enable);
    }

    bool is_toggle_mode() const { return m_toggle_mode; }

    void set_picker_mode(ResourcePickerMode mode) { m_mode = mode; }
    ResourcePickerMode get_picker_mode() const { return m_mode; }

    void _on_button_pressed() {
        if (m_toggle_mode) {
            // Toggle dropdown
        } else {
            show_resource_selector();
        }
    }

    void show_resource_selector() {
        EditorFileDialog* dialog = new EditorFileDialog();
        dialog->set_file_mode(FileDialogMode::MODE_OPEN_FILE);
        dialog->set_access(FileDialogAccess::ACCESS_RESOURCES);
        dialog->add_filter("*." + m_base_type + "; *." + m_base_type.to_lower());
        dialog->connect("file_selected", this, "on_resource_selected");
        dialog->popup_centered();
    }

    void on_resource_selected(const String& path) {
        Ref<Resource> res = ResourceLoader::load(path, m_base_type);
        if (res.is_valid()) {
            set_edited_resource(res);
        }
    }

    void clear() {
        m_resource = Ref<Resource>();
        update_preview();
        update_button_text();
    }

    bool can_drop_data(const vec2f& pos, const Variant& data) const {
        if (!m_editable) return false;
        // Check if dropped data is valid resource
        return true;
    }

    void drop_data(const vec2f& pos, const Variant& data) {
        // Handle resource drop
    }

private:
    void build_ui() {
        m_preview = new TextureRect();
        m_preview->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
        m_preview->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
        m_preview->set_custom_minimum_size(vec2f(32, 32));
        add_child(m_preview);

        m_label = new Label();
        m_label->set_clip_text(true);
        m_label->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(m_label);

        m_button = new Button();
        m_button->set_text("Load");
        m_button->set_focus_mode(Control::FOCUS_NONE);
        m_button->connect("pressed", this, "_on_button_pressed");
        add_child(m_button);
    }

    void update_preview() {
        if (m_resource.is_valid()) {
            // Generate preview
        } else {
            m_preview->set_texture(Ref<Texture2D>());
        }
    }

    void update_button_text() {
        if (m_resource.is_valid()) {
            String path = m_resource->get_path();
            if (path.empty()) {
                m_label->set_text("[built-in] " + m_resource->get_class().string());
            } else {
                m_label->set_text(path.get_file());
            }
            m_button->set_text("Edit");
        } else {
            m_label->set_text("[empty] " + m_base_type);
            m_button->set_text("Load");
        }
    }
};

// #############################################################################
// EditorQuickOpen - Quick open dialog with fuzzy search
// #############################################################################
class EditorQuickOpen : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(EditorQuickOpen, AcceptDialog)

private:
    LineEdit* m_search_box = nullptr;
    Tree* m_results_tree = nullptr;
    Label* m_preview_label = nullptr;
    CheckBox* m_match_case = nullptr;
    OptionButton* m_mode_select = nullptr;
    QuickOpenMode m_mode = QuickOpenMode::MODE_ALL;
    std::vector<SearchResultItem> m_all_items;
    std::vector<SearchResultItem> m_filtered_items;
    std::function<void(const String&)> m_callback;
    std::mutex m_mutex;
    std::thread m_index_thread;
    bool m_indexing = false;

public:
    static StringName get_class_static() { return StringName("EditorQuickOpen"); }

    EditorQuickOpen() {
        set_title("Quick Open");
        build_ui();
        start_indexing();
    }

    ~EditorQuickOpen() {
        if (m_index_thread.joinable()) m_index_thread.join();
    }

    void set_search_mode(QuickOpenMode mode) {
        m_mode = mode;
        m_mode_select->select(static_cast<int>(mode));
        filter_items();
    }

    void set_callback(std::function<void(const String&)> cb) { m_callback = cb; }

    void popup_centered() override {
        AcceptDialog::popup_centered();
        m_search_box->grab_focus();
        m_search_box->select_all();
    }

    void _ok_pressed() override {
        TreeItem* selected = m_results_tree->get_selected();
        if (selected) {
            String path = selected->get_metadata(0).as<String>();
            if (m_callback) m_callback(path);
        }
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

        m_mode_select = new OptionButton();
        m_mode_select->add_item("All");
        m_mode_select->add_item("Scenes");
        m_mode_select->add_item("Scripts");
        m_mode_select->add_item("Resources");
        m_mode_select->add_item("Nodes");
        m_mode_select->connect("item_selected", this, "on_mode_changed");
        search_row->add_child(m_mode_select);

        main->add_child(search_row);

        // Options
        HBoxContainer* options_row = new HBoxContainer();
        m_match_case = new CheckBox();
        m_match_case->set_text("Match Case");
        m_match_case->connect("toggled", this, "on_match_case_toggled");
        options_row->add_child(m_match_case);
        main->add_child(options_row);

        // Results tree
        m_results_tree = new Tree();
        m_results_tree->set_columns(1);
        m_results_tree->set_column_title(0, "Results");
        m_results_tree->set_hide_root(true);
        m_results_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_results_tree->connect("item_selected", this, "on_item_selected");
        m_results_tree->connect("item_activated", this, "on_item_activated");
        main->add_child(m_results_tree);

        // Preview
        m_preview_label = new Label();
        m_preview_label->set_text("No selection");
        main->add_child(m_preview_label);
    }

    void start_indexing() {
        m_indexing = true;
        m_index_thread = std::thread([this]() {
            index_filesystem("res://");
            m_indexing = false;
        });
    }

    void index_filesystem(const String& path) {
        Ref<DirAccess> dir = DirAccess::open(path);
        if (!dir.is_valid()) return;

        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item == "." || item == "..") continue;
            String full_path = path + "/" + item;
            if (dir->current_is_dir()) {
                index_filesystem(full_path);
            } else {
                SearchResultItem si;
                si.path = full_path;
                si.name = item;
                si.type = item.get_extension();
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_all_items.push_back(si);
                }
            }
        }
        dir->list_dir_end();
    }

    void filter_items() {
        String query = m_search_box->get_text().to_lower();
        m_filtered_items.clear();

        if (query.empty()) {
            m_filtered_items = m_all_items;
        } else {
            for (const auto& item : m_all_items) {
                if (item.name.to_lower().find(query) != String::npos) {
                    m_filtered_items.push_back(item);
                }
            }
        }

        // Filter by mode
        if (m_mode != QuickOpenMode::MODE_ALL) {
            std::vector<SearchResultItem> mode_filtered;
            String ext_filter;
            switch (m_mode) {
                case QuickOpenMode::MODE_SCENE: ext_filter = "tscn,scn"; break;
                case QuickOpenMode::MODE_SCRIPT: ext_filter = "gd,cs"; break;
                case QuickOpenMode::MODE_RESOURCE: ext_filter = "tres,res"; break;
                default: break;
            }
            for (const auto& item : m_filtered_items) {
                if (ext_filter.find(item.type) != String::npos) {
                    mode_filtered.push_back(item);
                }
            }
            m_filtered_items = std::move(mode_filtered);
        }

        update_results();
    }

    void update_results() {
        m_results_tree->clear();
        TreeItem* root = m_results_tree->create_item();

        for (const auto& item : m_filtered_items) {
            TreeItem* ti = m_results_tree->create_item(root);
            ti->set_text(0, item.name);
            ti->set_metadata(0, item.path);
            ti->set_icon(0, get_icon_for_type(item.type));
        }
    }

    Ref<Texture2D> get_icon_for_type(const String& type) {
        // Return appropriate icon
        return Ref<Texture2D>();
    }

    void on_search_changed(const String& text) { filter_items(); }
    void on_mode_changed(int idx) { set_search_mode(static_cast<QuickOpenMode>(idx)); }
    void on_match_case_toggled(bool pressed) { filter_items(); }

    void on_item_selected() {
        TreeItem* selected = m_results_tree->get_selected();
        if (selected) {
            m_preview_label->set_text(selected->get_metadata(0).as<String>());
        }
    }

    void on_item_activated() {
        _ok_pressed();
    }
};

// #############################################################################
// EditorFileDialog - File selection dialog
// #############################################################################
class EditorFileDialog : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(EditorFileDialog, AcceptDialog)

private:
    LineEdit* m_path_edit = nullptr;
    LineEdit* m_file_edit = nullptr;
    Tree* m_file_tree = nullptr;
    ItemList* m_favorites_list = nullptr;
    ItemList* m_recent_list = nullptr;
    OptionButton* m_filter_select = nullptr;
    FileDialogMode m_file_mode = FileDialogMode::MODE_OPEN_FILE;
    FileDialogAccess m_access = FileDialogAccess::ACCESS_RESOURCES;
    String m_current_path = "res://";
    std::vector<String> m_filters;
    std::vector<String> m_filter_descriptions;
    std::vector<String> m_selected_files;
    std::vector<String> m_favorites;
    std::vector<String> m_recent;

public:
    static StringName get_class_static() { return StringName("EditorFileDialog"); }

    EditorFileDialog() {
        build_ui();
        load_favorites();
        load_recent();
        refresh();
    }

    void set_file_mode(FileDialogMode mode) {
        m_file_mode = mode;
        update_ui_for_mode();
    }

    FileDialogMode get_file_mode() const { return m_file_mode; }

    void set_access(FileDialogAccess access) {
        m_access = access;
        if (access == FileDialogAccess::ACCESS_RESOURCES) {
            m_current_path = "res://";
        }
        refresh();
    }

    FileDialogAccess get_access() const { return m_access; }

    void set_current_path(const String& path) {
        if (DirAccess::dir_exists(path)) {
            m_current_path = path;
            m_path_edit->set_text(path);
            refresh();
        }
    }

    String get_current_path() const { return m_current_path; }

    void set_current_file(const String& file) {
        m_file_edit->set_text(file);
    }

    String get_current_file() const { return m_file_edit->get_text(); }

    void add_filter(const String& filter, const String& description = "") {
        m_filters.push_back(filter);
        m_filter_descriptions.push_back(description.empty() ? filter : description);
        update_filter_select();
    }

    void clear_filters() {
        m_filters.clear();
        m_filter_descriptions.clear();
        update_filter_select();
    }

    void add_favorite(const String& path) {
        if (std::find(m_favorites.begin(), m_favorites.end(), path) == m_favorites.end()) {
            m_favorites.push_back(path);
            save_favorites();
            refresh_favorites();
        }
    }

    void remove_favorite(const String& path) {
        auto it = std::find(m_favorites.begin(), m_favorites.end(), path);
        if (it != m_favorites.end()) {
            m_favorites.erase(it);
            save_favorites();
            refresh_favorites();
        }
    }

    void add_recent(const String& path) {
        auto it = std::find(m_recent.begin(), m_recent.end(), path);
        if (it != m_recent.end()) {
            m_recent.erase(it);
        }
        m_recent.insert(m_recent.begin(), path);
        if (m_recent.size() > 20) {
            m_recent.pop_back();
        }
        save_recent();
        refresh_recent();
    }

    std::vector<String> get_selected_files() const { return m_selected_files; }

    void _ok_pressed() override {
        String path = m_current_path;
        String file = m_file_edit->get_text();

        if (m_file_mode == FileDialogMode::MODE_OPEN_DIR) {
            m_selected_files = {path};
        } else if (m_file_mode == FileDialogMode::MODE_OPEN_FILE ||
                   m_file_mode == FileDialogMode::MODE_SAVE_FILE) {
            if (!file.empty()) {
                m_selected_files = {path + "/" + file};
            }
        }

        if (!m_selected_files.empty()) {
            add_recent(path);
            emit_signal("file_selected", m_selected_files[0]);
            AcceptDialog::_ok_pressed();
        }
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        // Path bar
        HBoxContainer* path_row = new HBoxContainer();
        path_row->add_child(new Label("Path:"));
        m_path_edit = new LineEdit();
        m_path_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_path_edit->connect("text_entered", this, "on_path_entered");
        path_row->add_child(m_path_edit);

        Button* up_btn = new Button();
        up_btn->set_text("Up");
        up_btn->connect("pressed", this, "on_navigate_up");
        path_row->add_child(up_btn);

        Button* refresh_btn = new Button();
        refresh_btn->set_text("Refresh");
        refresh_btn->connect("pressed", this, "refresh");
        path_row->add_child(refresh_btn);

        main->add_child(path_row);

        // Main area
        HSplitContainer* split = new HSplitContainer();
        split->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(split);

        // Sidebar
        VBoxContainer* sidebar = new VBoxContainer();
        sidebar->add_child(new Label("Favorites"));
        m_favorites_list = new ItemList();
        m_favorites_list->connect("item_activated", this, "on_favorite_activated");
        sidebar->add_child(m_favorites_list);

        sidebar->add_child(new Label("Recent"));
        m_recent_list = new ItemList();
        m_recent_list->connect("item_activated", this, "on_recent_activated");
        sidebar->add_child(m_recent_list);

        split->add_child(sidebar);

        // File tree
        m_file_tree = new Tree();
        m_file_tree->set_columns(1);
        m_file_tree->set_column_title(0, "Name");
        m_file_tree->set_hide_root(true);
        m_file_tree->connect("item_selected", this, "on_tree_item_selected");
        m_file_tree->connect("item_activated", this, "on_tree_item_activated");
        split->add_child(m_file_tree);

        // File name bar
        HBoxContainer* file_row = new HBoxContainer();
        file_row->add_child(new Label("File:"));
        m_file_edit = new LineEdit();
        m_file_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_file_edit->connect("text_changed", this, "on_file_changed");
        file_row->add_child(m_file_edit);
        main->add_child(file_row);

        // Filter bar
        HBoxContainer* filter_row = new HBoxContainer();
        filter_row->add_child(new Label("Filter:"));
        m_filter_select = new OptionButton();
        m_filter_select->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_filter_select->connect("item_selected", this, "on_filter_changed");
        filter_row->add_child(m_filter_select);
        main->add_child(filter_row);

        update_ui_for_mode();
    }

    void update_ui_for_mode() {
        bool is_save = (m_file_mode == FileDialogMode::MODE_SAVE_FILE);
        set_ok_button_text(is_save ? "Save" : "Open");
        m_file_edit->set_editable(is_save || m_file_mode == FileDialogMode::MODE_OPEN_FILE);
    }

    void refresh() {
        build_file_tree();
        refresh_favorites();
        refresh_recent();
    }

    void build_file_tree() {
        m_file_tree->clear();
        TreeItem* root = m_file_tree->create_item();
        root->set_text(0, m_current_path);
        root->set_metadata(0, m_current_path);

        Ref<DirAccess> dir = DirAccess::open(m_current_path);
        if (!dir.is_valid()) return;

        std::vector<String> dirs;
        std::vector<String> files;

        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item == "." || item == "..") continue;
            if (dir->current_is_dir()) {
                dirs.push_back(item);
            } else if (file_matches_filter(item)) {
                files.push_back(item);
            }
        }
        dir->list_dir_end();

        std::sort(dirs.begin(), dirs.end());
        std::sort(files.begin(), files.end());

        for (const auto& d : dirs) {
            TreeItem* ti = m_file_tree->create_item(root);
            ti->set_text(0, d);
            ti->set_icon(0, get_folder_icon());
            ti->set_metadata(0, m_current_path + "/" + d);
        }
        for (const auto& f : files) {
            TreeItem* ti = m_file_tree->create_item(root);
            ti->set_text(0, f);
            ti->set_icon(0, get_file_icon(f));
            ti->set_metadata(0, m_current_path + "/" + f);
        }
    }

    bool file_matches_filter(const String& filename) const {
        if (m_filters.empty()) return true;
        for (const auto& filter : m_filters) {
            if (filename.match(filter)) return true;
        }
        return false;
    }

    void update_filter_select() {
        m_filter_select->clear();
        for (size_t i = 0; i < m_filter_descriptions.size(); ++i) {
            m_filter_select->add_item(m_filter_descriptions[i]);
        }
        if (m_filter_descriptions.empty()) {
            m_filter_select->add_item("All Files (*)");
        }
    }

    void refresh_favorites() {
        m_favorites_list->clear();
        for (const auto& fav : m_favorites) {
            m_favorites_list->add_item(fav);
        }
    }

    void refresh_recent() {
        m_recent_list->clear();
        for (const auto& rec : m_recent) {
            m_recent_list->add_item(rec);
        }
    }

    void load_favorites() {}
    void save_favorites() {}
    void load_recent() {}
    void save_recent() {}

    Ref<Texture2D> get_folder_icon() { return Ref<Texture2D>(); }
    Ref<Texture2D> get_file_icon(const String& path) { return Ref<Texture2D>(); }

    void on_path_entered(const String& path) { set_current_path(path); }
    void on_navigate_up() {
        if (m_current_path != "res://") {
            String parent = m_current_path.get_base_dir();
            if (parent.empty()) parent = "res://";
            set_current_path(parent);
        }
    }
    void on_tree_item_selected() {
        TreeItem* item = m_file_tree->get_selected();
        if (item) {
            String path = item->get_metadata(0).as<String>();
            if (DirAccess::dir_exists(path)) {
                // Directory selected - don't set as file
            } else {
                m_file_edit->set_text(item->get_text(0));
            }
        }
    }
    void on_tree_item_activated() {
        TreeItem* item = m_file_tree->get_selected();
        if (item) {
            String path = item->get_metadata(0).as<String>();
            if (DirAccess::dir_exists(path)) {
                set_current_path(path);
            } else {
                m_file_edit->set_text(item->get_text(0));
                _ok_pressed();
            }
        }
    }
    void on_file_changed(const String& text) {}
    void on_filter_changed(int idx) { refresh(); }
    void on_favorite_activated(int idx) {
        set_current_path(m_favorites[idx]);
    }
    void on_recent_activated(int idx) {
        set_current_path(m_recent[idx]);
    }
};

// #############################################################################
// EditorDirDialog - Directory selection dialog
// #############################################################################
class EditorDirDialog : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(EditorDirDialog, AcceptDialog)

private:
    Tree* m_dir_tree = nullptr;
    LineEdit* m_path_edit = nullptr;
    String m_selected_path;

public:
    static StringName get_class_static() { return StringName("EditorDirDialog"); }

    EditorDirDialog() {
        set_title("Select Directory");
        build_ui();
    }

    void set_current_path(const String& path) {
        m_path_edit->set_text(path);
        // Expand tree to path
    }

    String get_selected_path() const { return m_selected_path; }

    void _ok_pressed() override {
        m_selected_path = m_path_edit->get_text();
        AcceptDialog::_ok_pressed();
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        main->add_child(new Label("Directory:"));

        m_dir_tree = new Tree();
        m_dir_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_dir_tree->connect("item_selected", this, "on_item_selected");
        main->add_child(m_dir_tree);

        HBoxContainer* path_row = new HBoxContainer();
        path_row->add_child(new Label("Path:"));
        m_path_edit = new LineEdit();
        m_path_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        path_row->add_child(m_path_edit);
        main->add_child(path_row);
    }

    void on_item_selected() {
        TreeItem* item = m_dir_tree->get_selected();
        if (item) {
            m_path_edit->set_text(item->get_metadata(0).as<String>());
        }
    }
};

// #############################################################################
// EditorPropertySelector - Property path selector
// #############################################################################
class EditorPropertySelector : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(EditorPropertySelector, AcceptDialog)

private:
    Tree* m_properties_tree = nullptr;
    LineEdit* m_search_box = nullptr;
    String m_selected_property;
    Object* m_target_object = nullptr;

public:
    static StringName get_class_static() { return StringName("EditorPropertySelector"); }

    EditorPropertySelector() {
        set_title("Select Property");
        build_ui();
    }

    void set_target_object(Object* obj) {
        m_target_object = obj;
        build_property_tree();
    }

    String get_selected_property() const { return m_selected_property; }

    void _ok_pressed() override {
        TreeItem* selected = m_properties_tree->get_selected();
        if (selected) {
            m_selected_property = selected->get_metadata(0).as<String>();
        }
        AcceptDialog::_ok_pressed();
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        m_search_box = new LineEdit();
        m_search_box->set_placeholder("Search properties...");
        m_search_box->connect("text_changed", this, "on_search_changed");
        main->add_child(m_search_box);

        m_properties_tree = new Tree();
        m_properties_tree->set_columns(1);
        m_properties_tree->set_hide_root(true);
        m_properties_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_properties_tree->connect("item_activated", this, "on_item_activated");
        main->add_child(m_properties_tree);
    }

    void build_property_tree() {
        m_properties_tree->clear();
        if (!m_target_object) return;

        TreeItem* root = m_properties_tree->create_item();
        std::vector<PropertyInfo> props = m_target_object->get_property_list();
        for (const auto& prop : props) {
            if (!(static_cast<uint32_t>(prop.usage) & static_cast<uint32_t>(PropertyUsage::EDITOR))) continue;
            TreeItem* item = m_properties_tree->create_item(root);
            item->set_text(0, prop.name.string());
            item->set_metadata(0, prop.name);
        }
    }

    void on_search_changed(const String& text) {
        // Filter properties
    }

    void on_item_activated() {
        _ok_pressed();
    }
};

} // namespace editor

// Bring into main namespace
using editor::EditorResourcePicker;
using editor::EditorQuickOpen;
using editor::EditorFileDialog;
using editor::EditorDirDialog;
using editor::EditorPropertySelector;
using editor::ResourcePickerMode;
using editor::QuickOpenMode;
using editor::FileDialogMode;
using editor::FileDialogAccess;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_PICKERS_HPP