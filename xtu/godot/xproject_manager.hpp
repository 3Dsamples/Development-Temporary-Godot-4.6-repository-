// include/xtu/godot/xproject_manager.hpp
// xtensor-unified - Project Manager for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPROJECT_MANAGER_HPP
#define XTU_GODOT_XPROJECT_MANAGER_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xeditor_theme.hpp"
#include "xtu/godot/xeditor_pickers.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class ProjectManager;
class ProjectDialog;
class ProjectListItem;
class ProjectList;
class ProjectTag;

// #############################################################################
// Project info structure
// #############################################################################
struct ProjectInfo {
    String path;
    String name;
    String description;
    String icon_path;
    String main_scene;
    String godot_version;
    uint64_t last_modified = 0;
    std::vector<String> tags;
    bool favorite = false;
    bool is_valid = false;
    bool missing = false;
};

// #############################################################################
// Project template type
// #############################################################################
enum class ProjectTemplateType : uint8_t {
    TEMPLATE_2D = 0,
    TEMPLATE_3D = 1,
    TEMPLATE_XR = 2,
    TEMPLATE_MOBILE = 3,
    TEMPLATE_EMPTY = 4,
    TEMPLATE_CUSTOM = 5
};

// #############################################################################
// Project sort method
// #############################################################################
enum class ProjectSortMethod : uint8_t {
    SORT_NAME = 0,
    SORT_MODIFIED = 1,
    SORT_CREATED = 2,
    SORT_PATH = 3
};

// #############################################################################
// ProjectList - Scrollable list of projects
// #############################################################################
class ProjectList : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(ProjectList, VBoxContainer)

private:
    std::vector<ProjectInfo> m_projects;
    std::vector<ProjectInfo> m_filtered_projects;
    VBoxContainer* m_list_container = nullptr;
    LineEdit* m_search_box = nullptr;
    OptionButton* m_sort_select = nullptr;
    OptionButton* m_tag_filter = nullptr;
    CheckBox* m_favorites_only = nullptr;
    ProjectSortMethod m_sort_method = ProjectSortMethod::SORT_MODIFIED;
    std::unordered_set<String> m_available_tags;
    std::function<void(const ProjectInfo&)> m_selection_callback;
    ProjectInfo m_selected_project;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("ProjectList"); }

    ProjectList() {
        build_ui();
    }

    void set_projects(const std::vector<ProjectInfo>& projects) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_projects = projects;
        m_available_tags.clear();
        for (const auto& p : m_projects) {
            for (const auto& t : p.tags) {
                m_available_tags.insert(t);
            }
        }
        update_tag_filter();
        filter_and_sort();
    }

    void set_selection_callback(std::function<void(const ProjectInfo&)> cb) {
        m_selection_callback = cb;
    }

    ProjectInfo get_selected_project() const { return m_selected_project; }

    void refresh() {
        filter_and_sort();
    }

    void clear_selection() {
        m_selected_project = ProjectInfo();
    }

private:
    void build_ui() {
        // Search bar
        HBoxContainer* search_row = new HBoxContainer();
        m_search_box = new LineEdit();
        m_search_box->set_placeholder("Search projects...");
        m_search_box->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_search_box->connect("text_changed", this, "on_search_changed");
        search_row->add_child(m_search_box);

        m_sort_select = new OptionButton();
        m_sort_select->add_item("Name");
        m_sort_select->add_item("Last Modified");
        m_sort_select->add_item("Path");
        m_sort_select->connect("item_selected", this, "on_sort_changed");
        search_row->add_child(m_sort_select);

        add_child(search_row);

        // Filter row
        HBoxContainer* filter_row = new HBoxContainer();
        m_tag_filter = new OptionButton();
        m_tag_filter->add_item("All Tags");
        m_tag_filter->connect("item_selected", this, "on_tag_changed");
        filter_row->add_child(m_tag_filter);

        m_favorites_only = new CheckBox();
        m_favorites_only->set_text("Favorites Only");
        m_favorites_only->connect("toggled", this, "on_favorites_toggled");
        filter_row->add_child(m_favorites_only);
        add_child(filter_row);

        // Scroll container for project items
        ScrollContainer* scroll = new ScrollContainer();
        scroll->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        scroll->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(scroll);

        m_list_container = new VBoxContainer();
        m_list_container->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        scroll->add_child(m_list_container);
    }

    void update_tag_filter() {
        m_tag_filter->clear();
        m_tag_filter->add_item("All Tags");
        for (const auto& tag : m_available_tags) {
            m_tag_filter->add_item(tag);
        }
    }

    void filter_and_sort() {
        m_filtered_projects = m_projects;

        // Filter by search
        String search = m_search_box->get_text().to_lower();
        if (!search.empty()) {
            std::vector<ProjectInfo> filtered;
            for (const auto& p : m_filtered_projects) {
                if (p.name.to_lower().find(search) != String::npos ||
                    p.path.to_lower().find(search) != String::npos) {
                    filtered.push_back(p);
                }
            }
            m_filtered_projects = std::move(filtered);
        }

        // Filter by tag
        int tag_idx = m_tag_filter->get_selected();
        if (tag_idx > 0) {
            String tag = m_tag_filter->get_item_text(tag_idx);
            std::vector<ProjectInfo> filtered;
            for (const auto& p : m_filtered_projects) {
                if (std::find(p.tags.begin(), p.tags.end(), tag) != p.tags.end()) {
                    filtered.push_back(p);
                }
            }
            m_filtered_projects = std::move(filtered);
        }

        // Filter favorites
        if (m_favorites_only->is_pressed()) {
            std::vector<ProjectInfo> filtered;
            for (const auto& p : m_filtered_projects) {
                if (p.favorite) filtered.push_back(p);
            }
            m_filtered_projects = std::move(filtered);
        }

        // Sort
        switch (m_sort_method) {
            case ProjectSortMethod::SORT_NAME:
                std::sort(m_filtered_projects.begin(), m_filtered_projects.end(),
                    [](const ProjectInfo& a, const ProjectInfo& b) { return a.name < b.name; });
                break;
            case ProjectSortMethod::SORT_MODIFIED:
                std::sort(m_filtered_projects.begin(), m_filtered_projects.end(),
                    [](const ProjectInfo& a, const ProjectInfo& b) { return a.last_modified > b.last_modified; });
                break;
            case ProjectSortMethod::SORT_PATH:
                std::sort(m_filtered_projects.begin(), m_filtered_projects.end(),
                    [](const ProjectInfo& a, const ProjectInfo& b) { return a.path < b.path; });
                break;
        }

        rebuild_list();
    }

    void rebuild_list() {
        // Clear existing items
        for (int i = 0; i < m_list_container->get_child_count(); ++i) {
            m_list_container->get_child(i)->queue_free();
        }

        for (const auto& info : m_filtered_projects) {
            ProjectListItem* item = new ProjectListItem(info);
            item->connect("selected", this, "on_project_selected", info);
            item->connect("double_clicked", this, "on_project_double_clicked", info);
            m_list_container->add_child(item);
        }
    }

    void on_search_changed(const String&) { filter_and_sort(); }
    void on_sort_changed(int idx) {
        m_sort_method = static_cast<ProjectSortMethod>(idx);
        filter_and_sort();
    }
    void on_tag_changed(int) { filter_and_sort(); }
    void on_favorites_toggled(bool) { filter_and_sort(); }

    void on_project_selected(const ProjectInfo& info) {
        m_selected_project = info;
        if (m_selection_callback) m_selection_callback(info);
    }

    void on_project_double_clicked(const ProjectInfo& info) {
        emit_signal("project_activated", info.path);
    }
};

// #############################################################################
// ProjectListItem - Individual project entry
// #############################################################################
class ProjectListItem : public HBoxContainer {
    XTU_GODOT_REGISTER_CLASS(ProjectListItem, HBoxContainer)

private:
    ProjectInfo m_info;
    TextureRect* m_icon = nullptr;
    VBoxContainer* m_text_container = nullptr;
    Label* m_name_label = nullptr;
    Label* m_path_label = nullptr;
    Label* m_version_label = nullptr;
    Button* m_favorite_btn = nullptr;
    Button* m_menu_btn = nullptr;
    bool m_selected = false;

public:
    static StringName get_class_static() { return StringName("ProjectListItem"); }

    ProjectListItem(const ProjectInfo& info) : m_info(info) {
        build_ui();
    }

    void set_selected(bool selected) {
        m_selected = selected;
        update_style();
    }

    bool is_selected() const { return m_selected; }

    ProjectInfo get_info() const { return m_info; }

    void _gui_input(const Ref<InputEvent>& event) override {
        if (auto* mb = dynamic_cast<InputEventMouseButton*>(event.ptr())) {
            if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
                emit_signal("selected", m_info);
                if (mb->is_double_click()) {
                    emit_signal("double_clicked", m_info);
                }
            }
        }
    }

private:
    void build_ui() {
        set_h_size_flags(SIZE_EXPAND | SIZE_FILL);

        m_icon = new TextureRect();
        m_icon->set_custom_minimum_size(vec2f(48, 48));
        m_icon->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
        add_child(m_icon);

        m_text_container = new VBoxContainer();
        m_text_container->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(m_text_container);

        HBoxContainer* name_row = new HBoxContainer();
        m_name_label = new Label();
        m_name_label->set_text(m_info.name);
        m_name_label->add_theme_font_override("font", EditorFonts::get_singleton()->get_main_font());
        name_row->add_child(m_name_label);

        if (m_info.favorite) {
            Label* star = new Label();
            star->set_text("★");
            name_row->add_child(star);
        }

        m_text_container->add_child(name_row);

        m_path_label = new Label();
        m_path_label->set_text(m_info.path);
        m_path_label->set_modulate(Color(0.7f, 0.7f, 0.7f, 1.0f));
        m_text_container->add_child(m_path_label);

        HBoxContainer* info_row = new HBoxContainer();
        m_version_label = new Label();
        m_version_label->set_text("Godot " + m_info.godot_version);
        m_version_label->set_modulate(Color(0.5f, 0.5f, 0.5f, 1.0f));
        info_row->add_child(m_version_label);
        m_text_container->add_child(info_row);

        // Favorite button
        m_favorite_btn = new Button();
        m_favorite_btn->set_text(m_info.favorite ? "★" : "☆");
        m_favorite_btn->set_flat(true);
        m_favorite_btn->connect("pressed", this, "on_favorite_pressed");
        add_child(m_favorite_btn);

        // Menu button
        m_menu_btn = new Button();
        m_menu_btn->set_text("...");
        m_menu_btn->set_flat(true);
        m_menu_btn->connect("pressed", this, "on_menu_pressed");
        add_child(m_menu_btn);
    }

    void update_style() {
        if (m_selected) {
            set_modulate(Color(1.0f, 1.0f, 1.0f, 1.0f));
        }
    }

    void on_favorite_pressed() {
        m_info.favorite = !m_info.favorite;
        m_favorite_btn->set_text(m_info.favorite ? "★" : "☆");
        emit_signal("favorite_toggled", m_info.path, m_info.favorite);
    }

    void on_menu_pressed() {
        PopupMenu* menu = new PopupMenu();
        menu->add_item("Open");
        menu->add_item("Show in File Manager");
        menu->add_item("Rename");
        menu->add_item("Duplicate");
        menu->add_item("Delete");
        menu->add_separator();
        menu->add_item("Edit Tags");
        menu->connect("id_pressed", this, "on_menu_action");
        menu->set_position(m_menu_btn->get_screen_position() + vec2f(0, m_menu_btn->get_size().y()));
        menu->popup();
    }

    void on_menu_action(int id) {
        switch (id) {
            case 0: emit_signal("project_activated", m_info.path); break;
            case 1: OS::get_singleton()->shell_open(m_info.path); break;
            case 2: emit_signal("rename_requested", m_info); break;
            case 3: emit_signal("duplicate_requested", m_info); break;
            case 4: emit_signal("delete_requested", m_info); break;
            case 6: emit_signal("edit_tags_requested", m_info); break;
        }
    }
};

// #############################################################################
// ProjectDialog - New project creation dialog
// #############################################################################
class ProjectDialog : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(ProjectDialog, AcceptDialog)

private:
    LineEdit* m_name_edit = nullptr;
    LineEdit* m_path_edit = nullptr;
    OptionButton* m_template_select = nullptr;
    CheckBox* m_create_folder = nullptr;
    CheckBox* m_initialize_git = nullptr;
    TextEdit* m_description_edit = nullptr;
    String m_selected_path;

public:
    static StringName get_class_static() { return StringName("ProjectDialog"); }

    ProjectDialog() {
        set_title("Create New Project");
        build_ui();
    }

    String get_selected_path() const { return m_selected_path; }

    void _ok_pressed() override {
        String name = m_name_edit->get_text();
        String path = m_path_edit->get_text();

        if (m_create_folder->is_pressed()) {
            path = path + "/" + name;
        }

        if (create_project(name, path)) {
            m_selected_path = path;
            AcceptDialog::_ok_pressed();
        }
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        add_child(main);

        // Project Name
        HBoxContainer* name_row = new HBoxContainer();
        name_row->add_child(new Label("Project Name:"));
        m_name_edit = new LineEdit();
        m_name_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_name_edit->connect("text_changed", this, "on_name_changed");
        name_row->add_child(m_name_edit);
        main->add_child(name_row);

        // Project Path
        HBoxContainer* path_row = new HBoxContainer();
        path_row->add_child(new Label("Project Path:"));
        m_path_edit = new LineEdit();
        m_path_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_path_edit->set_text(OS::get_singleton()->get_user_data_dir() + "/projects");
        path_row->add_child(m_path_edit);

        Button* browse_btn = new Button();
        browse_btn->set_text("Browse");
        browse_btn->connect("pressed", this, "on_browse_pressed");
        path_row->add_child(browse_btn);
        main->add_child(path_row);

        // Template
        HBoxContainer* template_row = new HBoxContainer();
        template_row->add_child(new Label("Template:"));
        m_template_select = new OptionButton();
        m_template_select->add_item("2D");
        m_template_select->add_item("3D");
        m_template_select->add_item("XR");
        m_template_select->add_item("Mobile");
        m_template_select->add_item("Empty");
        template_row->add_child(m_template_select);
        main->add_child(template_row);

        // Options
        m_create_folder = new CheckBox();
        m_create_folder->set_text("Create folder");
        m_create_folder->set_pressed(true);
        main->add_child(m_create_folder);

        m_initialize_git = new CheckBox();
        m_initialize_git->set_text("Initialize Git repository");
        m_initialize_git->set_pressed(true);
        main->add_child(m_initialize_git);

        // Description
        main->add_child(new Label("Description (optional):"));
        m_description_edit = new TextEdit();
        m_description_edit->set_custom_minimum_size(vec2f(0, 80));
        main->add_child(m_description_edit);
    }

    void on_name_changed(const String& name) {
        // Auto-update path
    }

    void on_browse_pressed() {
        EditorFileDialog* dialog = new EditorFileDialog();
        dialog->set_file_mode(FileDialogMode::MODE_OPEN_DIR);
        dialog->set_access(FileDialogAccess::ACCESS_FILESYSTEM);
        dialog->connect("dir_selected", this, "on_path_selected");
        dialog->popup_centered();
    }

    void on_path_selected(const String& path) {
        m_path_edit->set_text(path);
    }

    bool create_project(const String& name, const String& path) {
        if (!DirAccess::make_dir_recursive(path)) {
            return false;
        }

        // Create project.godot file
        String config_path = path + "/project.godot";
        Ref<FileAccess> file = FileAccess::open(config_path, FileAccess::WRITE);
        if (!file.is_valid()) return false;

        file->store_string("; Engine configuration file.\n");
        file->store_string("; It's best edited using the editor UI and not directly.\n");
        file->store_string("; The format may change in future versions.\n\n");
        file->store_string("[application]\n");
        file->store_string("config/name=\"" + name + "\"\n");
        file->store_string("config/description=\"" + m_description_edit->get_text() + "\"\n");
        file->store_string("config/icon=\"res://icon.svg\"\n\n");

        // Create template files based on selection
        int template_idx = m_template_select->get_selected();
        create_template_files(path, static_cast<ProjectTemplateType>(template_idx));

        // Initialize Git if requested
        if (m_initialize_git->is_pressed()) {
            OS::get_singleton()->execute("git", {"init", path});
        }

        return true;
    }

    void create_template_files(const String& path, ProjectTemplateType type) {
        // Create default scene
        String scene_path = path + "/main.tscn";
        // Create default script
        // Create icon
    }
};

// #############################################################################
// ProjectManager - Main project manager window
// #############################################################################
class ProjectManager : public Window {
    XTU_GODOT_REGISTER_CLASS(ProjectManager, Window)

private:
    static ProjectManager* s_singleton;
    ProjectList* m_project_list = nullptr;
    Button* m_new_btn = nullptr;
    Button* m_import_btn = nullptr;
    Button* m_scan_btn = nullptr;
    Button* m_open_btn = nullptr;
    Button* m_remove_btn = nullptr;
    Label* m_status_label = nullptr;
    std::vector<String> m_search_paths;
    std::thread m_scan_thread;
    std::atomic<bool> m_scanning{false};

public:
    static ProjectManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ProjectManager"); }

    ProjectManager() {
        s_singleton = this;
        set_title("Godot Project Manager");
        build_ui();
        load_settings();
        start_scan();
    }

    ~ProjectManager() {
        if (m_scan_thread.joinable()) m_scan_thread.join();
        s_singleton = nullptr;
    }

    void add_search_path(const String& path) {
        if (std::find(m_search_paths.begin(), m_search_paths.end(), path) == m_search_paths.end()) {
            m_search_paths.push_back(path);
        }
    }

    void scan_projects() {
        if (m_scanning) return;
        start_scan();
    }

    void open_project(const String& path) {
        String editor_path = OS::get_singleton()->get_executable_path().get_base_dir() + "/godot";
        std::vector<String> args = {"--editor", "--path", path};
        OS::get_singleton()->execute(editor_path, args);
        // Or if running as editor already, switch project
    }

    void create_new_project() {
        ProjectDialog* dialog = new ProjectDialog();
        dialog->connect("confirmed", this, "on_project_created");
        dialog->popup_centered();
    }

    void import_project() {
        EditorFileDialog* dialog = new EditorFileDialog();
        dialog->set_file_mode(FileDialogMode::MODE_OPEN_FILE);
        dialog->add_filter("*.godot", "Godot Project");
        dialog->connect("file_selected", this, "on_project_imported");
        dialog->popup_centered();
    }

    void remove_project(const String& path) {
        // Remove from config list, don't delete files
    }

private:
    void build_ui() {
        VBoxContainer* main = new VBoxContainer();
        main->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
        add_child(main);

        // Header
        HBoxContainer* header = new HBoxContainer();
        Label* title = new Label();
        title->set_text("Godot Engine");
        title->add_theme_font_override("font", EditorFonts::get_singleton()->get_main_font());
        title->add_theme_font_size_override("font_size", 24);
        header->add_child(title);
        main->add_child(header);

        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();
        m_new_btn = new Button();
        m_new_btn->set_text("New Project");
        m_new_btn->connect("pressed", this, "create_new_project");
        toolbar->add_child(m_new_btn);

        m_import_btn = new Button();
        m_import_btn->set_text("Import");
        m_import_btn->connect("pressed", this, "import_project");
        toolbar->add_child(m_import_btn);

        m_scan_btn = new Button();
        m_scan_btn->set_text("Scan");
        m_scan_btn->connect("pressed", this, "scan_projects");
        toolbar->add_child(m_scan_btn);
        main->add_child(toolbar);

        // Project list
        m_project_list = new ProjectList();
        m_project_list->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_project_list->connect("project_activated", this, "on_project_activated");
        m_project_list->connect("rename_requested", this, "on_rename_requested");
        m_project_list->connect("delete_requested", this, "on_delete_requested");
        main->add_child(m_project_list);

        // Bottom bar
        HBoxContainer* bottom = new HBoxContainer();
        m_status_label = new Label();
        m_status_label->set_text("Ready");
        bottom->add_child(m_status_label);

        m_open_btn = new Button();
        m_open_btn->set_text("Open");
        m_open_btn->set_disabled(true);
        m_open_btn->connect("pressed", this, "on_open_pressed");
        bottom->add_child(m_open_btn);

        m_remove_btn = new Button();
        m_remove_btn->set_text("Remove from List");
        m_remove_btn->set_disabled(true);
        bottom->add_child(m_remove_btn);
        main->add_child(bottom);

        m_project_list->connect("selected", this, "on_project_selected");
    }

    void load_settings() {
        String config_path = OS::get_singleton()->get_user_data_dir() + "/projects.json";
        if (FileAccess::file_exists(config_path)) {
            String content = FileAccess::get_file_as_string(config_path);
            io::json::JsonValue json = io::json::JsonValue::parse(content.to_std_string());
            // Load search paths and favorite projects
        }
        if (m_search_paths.empty()) {
            m_search_paths.push_back(OS::get_singleton()->get_user_data_dir() + "/projects");
        }
    }

    void save_settings() {
        io::json::JsonValue json;
        io::json::JsonValue paths_arr;
        for (const auto& p : m_search_paths) {
            paths_arr.as_array().push_back(io::json::JsonValue(p.to_std_string()));
        }
        json["search_paths"] = paths_arr;
        String config_path = OS::get_singleton()->get_user_data_dir() + "/projects.json";
        Ref<FileAccess> file = FileAccess::open(config_path, FileAccess::WRITE);
        if (file.is_valid()) {
            file->store_string(json.dump(2).c_str());
        }
    }

    void start_scan() {
        m_scanning = true;
        m_status_label->set_text("Scanning for projects...");
        if (m_scan_thread.joinable()) m_scan_thread.join();
        m_scan_thread = std::thread([this]() {
            std::vector<ProjectInfo> projects;
            for (const auto& path : m_search_paths) {
                scan_directory(path, projects);
            }
            call_deferred("_scan_completed", projects);
        });
    }

    void scan_directory(const String& path, std::vector<ProjectInfo>& projects) {
        Ref<DirAccess> dir = DirAccess::open(path);
        if (!dir.is_valid()) return;

        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item == "." || item == "..") continue;
            String full_path = path + "/" + item;
            if (dir->current_is_dir()) {
                String project_file = full_path + "/project.godot";
                if (FileAccess::file_exists(project_file)) {
                    ProjectInfo info = parse_project_file(project_file);
                    info.path = full_path;
                    projects.push_back(info);
                } else {
                    scan_directory(full_path, projects);
                }
            }
        }
        dir->list_dir_end();
    }

    ProjectInfo parse_project_file(const String& path) {
        ProjectInfo info;
        info.is_valid = true;
        info.last_modified = FileAccess::get_modified_time(path);

        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return info;

        String content = file->get_as_text();
        // Parse INI format
        std::vector<String> lines = content.split("\n");
        String current_section;
        for (const auto& line : lines) {
            String trimmed = line.strip_edges();
            if (trimmed.empty() || trimmed[0] == ';') continue;
            if (trimmed[0] == '[' && trimmed[trimmed.length()-1] == ']') {
                current_section = trimmed.substr(1, trimmed.length() - 2);
            } else if (current_section == "application") {
                size_t eq = trimmed.find('=');
                if (eq != String::npos) {
                    String key = trimmed.substr(0, eq).strip_edges();
                    String value = trimmed.substr(eq + 1).strip_edges();
                    if (value.length() >= 2 && value[0] == '"' && value[value.length()-1] == '"') {
                        value = value.substr(1, value.length() - 2);
                    }
                    if (key == "config/name") info.name = value;
                    else if (key == "config/description") info.description = value;
                    else if (key == "config/icon") info.icon_path = value;
                    else if (key == "run/main_scene") info.main_scene = value;
                }
            }
        }

        if (info.name.empty()) {
            info.name = path.get_base_dir().get_file();
        }

        return info;
    }

    void _scan_completed(const std::vector<ProjectInfo>& projects) {
        m_project_list->set_projects(projects);
        m_status_label->set_text("Found " + String::num(static_cast<int>(projects.size())) + " projects");
        m_scanning = false;
    }

    void on_project_selected(const ProjectInfo& info) {
        m_open_btn->set_disabled(false);
        m_remove_btn->set_disabled(false);
    }

    void on_project_activated(const String& path) {
        open_project(path);
    }

    void on_open_pressed() {
        ProjectInfo info = m_project_list->get_selected_project();
        if (info.is_valid) {
            open_project(info.path);
        }
    }

    void on_project_created() {
        scan_projects();
    }

    void on_project_imported(const String& path) {
        String project_dir = path.get_base_dir();
        if (std::find(m_search_paths.begin(), m_search_paths.end(), project_dir) == m_search_paths.end()) {
            m_search_paths.push_back(project_dir);
            save_settings();
        }
        scan_projects();
    }

    void on_rename_requested(const ProjectInfo& info) {}
    void on_delete_requested(const ProjectInfo& info) {}
};

} // namespace editor

// Bring into main namespace
using editor::ProjectManager;
using editor::ProjectDialog;
using editor::ProjectList;
using editor::ProjectListItem;
using editor::ProjectInfo;
using editor::ProjectTemplateType;
using editor::ProjectSortMethod;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPROJECT_MANAGER_HPP