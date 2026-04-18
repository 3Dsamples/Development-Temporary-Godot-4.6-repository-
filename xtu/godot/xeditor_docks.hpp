// include/xtu/godot/xeditor_docks.hpp
// xtensor-unified - Editor docks and viewport plugins for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_DOCKS_HPP
#define XTU_GODOT_XEDITOR_DOCKS_HPP

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
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xgraphics.hpp"
#include "xtu/godot/xphysics2d.hpp"
#include "xtu/godot/xphysics3d.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class SceneTreeDock;
class InspectorDock;
class FileSystemDock;
class NodeDock;
class CanvasItemEditorPlugin;
class SpatialEditorPlugin;
class ScriptEditorPlugin;

// #############################################################################
// Gizmo handle types
// #############################################################################
enum class GizmoHandle : uint8_t {
    HANDLE_NONE = 0,
    HANDLE_X = 1,
    HANDLE_Y = 2,
    HANDLE_Z = 3,
    HANDLE_XY = 4,
    HANDLE_XZ = 5,
    HANDLE_YZ = 6,
    HANDLE_XYZ = 7
};

// #############################################################################
// Gizmo mode
// #############################################################################
enum class GizmoMode : uint8_t {
    MODE_MOVE = 0,
    MODE_ROTATE = 1,
    MODE_SCALE = 2,
    MODE_SELECT = 3
};

// #############################################################################
// Transform snap mode
// #############################################################################
enum class TransformSnapMode : uint8_t {
    SNAP_NONE = 0,
    SNAP_GRID = 1,
    SNAP_RELATIVE = 2,
    SNAP_ABSOLUTE = 3
};

// #############################################################################
// SceneTreeDock - Scene hierarchy management
// #############################################################################
class SceneTreeDock : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(SceneTreeDock, VBoxContainer)

private:
    Tree* m_scene_tree = nullptr;
    LineEdit* m_filter_input = nullptr;
    Button* m_add_node_btn = nullptr;
    Button* m_instance_scene_btn = nullptr;
    Button* m_delete_btn = nullptr;
    Button* m_duplicate_btn = nullptr;
    Button* m_reparent_btn = nullptr;
    Button* m_visibility_btn = nullptr;
    Node* m_edited_scene = nullptr;
    std::unordered_map<Node*, TreeItem*> m_node_to_item;
    bool m_updating = false;

public:
    static StringName get_class_static() { return StringName("SceneTreeDock"); }

    SceneTreeDock() {
        build_ui();
    }

    void set_edited_scene(Node* scene) {
        m_edited_scene = scene;
        refresh_tree();
    }

    Node* get_edited_scene() const { return m_edited_scene; }

    void refresh_tree() {
        m_updating = true;
        m_scene_tree->clear();
        m_node_to_item.clear();
        if (m_edited_scene) {
            TreeItem* root = m_scene_tree->create_item();
            build_tree_item(root, m_edited_scene);
        }
        m_updating = false;
    }

    void add_node(Node* parent, Node* node) {
        if (!parent || !node) return;
        parent->add_child(node);
        refresh_tree();
        select_node(node);
    }

    void delete_node(Node* node) {
        if (!node || node == m_edited_scene) return;
        Node* parent = node->get_parent();
        if (parent) {
            parent->remove_child(node);
            node->queue_free();
            refresh_tree();
        }
    }

    void duplicate_node(Node* node) {
        if (!node) return;
        Node* duplicate = node->duplicate();
        Node* parent = node->get_parent();
        if (parent && duplicate) {
            parent->add_child(duplicate);
            duplicate->set_owner(m_edited_scene);
            refresh_tree();
            select_node(duplicate);
        }
    }

    void reparent_node(Node* node, Node* new_parent) {
        if (!node || !new_parent || node == new_parent) return;
        Node* old_parent = node->get_parent();
        if (old_parent) {
            old_parent->remove_child(node);
            new_parent->add_child(node);
            refresh_tree();
        }
    }

    void select_node(Node* node) {
        auto it = m_node_to_item.find(node);
        if (it != m_node_to_item.end()) {
            it->second->select(0);
            emit_signal("node_selected", node);
        }
    }

    Node* get_selected_node() const {
        TreeItem* selected = m_scene_tree->get_selected();
        if (selected) {
            return selected->get_metadata(0).as<Node*>();
        }
        return nullptr;
    }

    void set_filter(const String& filter) {
        // Filter tree items
    }

private:
    void build_ui() {
        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();
        m_filter_input = new LineEdit();
        m_filter_input->set_placeholder("Filter nodes...");
        m_filter_input->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_filter_input->connect("text_changed", this, "on_filter_changed");
        toolbar->add_child(m_filter_input);

        m_add_node_btn = new Button();
        m_add_node_btn->set_text("+");
        m_add_node_btn->connect("pressed", this, "on_add_node");
        toolbar->add_child(m_add_node_btn);

        m_instance_scene_btn = new Button();
        m_instance_scene_btn->set_text("Instance");
        m_instance_scene_btn->connect("pressed", this, "on_instance_scene");
        toolbar->add_child(m_instance_scene_btn);

        add_child(toolbar);

        // Scene tree
        m_scene_tree = new Tree();
        m_scene_tree->set_columns(1);
        m_scene_tree->set_column_title(0, "Scene");
        m_scene_tree->set_hide_root(true);
        m_scene_tree->set_allow_rmb_select(true);
        m_scene_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_scene_tree->connect("item_selected", this, "on_item_selected");
        m_scene_tree->connect("item_activated", this, "on_item_activated");
        add_child(m_scene_tree);

        // Bottom toolbar
        HBoxContainer* bottom_toolbar = new HBoxContainer();
        m_delete_btn = new Button();
        m_delete_btn->set_text("Delete");
        m_delete_btn->connect("pressed", this, "on_delete");
        bottom_toolbar->add_child(m_delete_btn);

        m_duplicate_btn = new Button();
        m_duplicate_btn->set_text("Duplicate");
        m_duplicate_btn->connect("pressed", this, "on_duplicate");
        bottom_toolbar->add_child(m_duplicate_btn);

        m_reparent_btn = new Button();
        m_reparent_btn->set_text("Reparent");
        m_reparent_btn->connect("pressed", this, "on_reparent");
        bottom_toolbar->add_child(m_reparent_btn);

        m_visibility_btn = new Button();
        m_visibility_btn->set_text("Visibility");
        m_visibility_btn->connect("pressed", this, "on_toggle_visibility");
        bottom_toolbar->add_child(m_visibility_btn);

        add_child(bottom_toolbar);
    }

    void build_tree_item(TreeItem* parent_item, Node* node) {
        TreeItem* item = m_scene_tree->create_item(parent_item);
        item->set_text(0, node->get_name().string());
        item->set_icon(0, get_icon_for_node(node));
        item->set_metadata(0, node);
        m_node_to_item[node] = item;

        for (int i = 0; i < node->get_child_count(); ++i) {
            build_tree_item(item, node->get_child(i));
        }
    }

    Ref<Texture2D> get_icon_for_node(Node* node) {
        // Return appropriate icon based on node type
        return Ref<Texture2D>();
    }

    void on_filter_changed(const String& text) { set_filter(text); }
    void on_item_selected() {
        Node* node = get_selected_node();
        if (node) emit_signal("node_selected", node);
    }
    void on_item_activated() {
        Node* node = get_selected_node();
        if (node) emit_signal("node_activated", node);
    }
    void on_add_node() {
        // Show add node dialog
    }
    void on_instance_scene() {
        // Show open scene dialog
    }
    void on_delete() {
        Node* node = get_selected_node();
        if (node) delete_node(node);
    }
    void on_duplicate() {
        Node* node = get_selected_node();
        if (node) duplicate_node(node);
    }
    void on_reparent() {
        // Show reparent dialog
    }
    void on_toggle_visibility() {
        Node* node = get_selected_node();
        if (node) {
            if (auto* n2d = dynamic_cast<Node2D*>(node)) {
                n2d->set_visible(!n2d->is_visible());
            } else if (auto* n3d = dynamic_cast<Node3D*>(node)) {
                n3d->set_visible(!n3d->is_visible());
            }
        }
    }
};

// #############################################################################
// InspectorDock - Property inspector panel
// #############################################################################
class InspectorDock : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(InspectorDock, VBoxContainer)

private:
    EditorInspector* m_inspector = nullptr;
    LineEdit* m_search_box = nullptr;
    Button* m_refresh_btn = nullptr;
    Button* m_back_btn = nullptr;
    Button* m_forward_btn = nullptr;
    Object* m_current_object = nullptr;
    std::vector<Object*> m_history;
    int m_history_pos = -1;

public:
    static StringName get_class_static() { return StringName("InspectorDock"); }

    InspectorDock() {
        build_ui();
    }

    void edit(Object* object) {
        if (m_current_object && m_current_object != object) {
            add_to_history(m_current_object);
        }
        m_current_object = object;
        m_inspector->edit(object);
        update_history_buttons();
    }

    Object* get_edited_object() const { return m_current_object; }

    void refresh() {
        m_inspector->refresh();
    }

    void go_back() {
        if (m_history_pos > 0) {
            --m_history_pos;
            m_current_object = m_history[m_history_pos];
            m_inspector->edit(m_current_object);
            update_history_buttons();
        }
    }

    void go_forward() {
        if (m_history_pos < static_cast<int>(m_history.size()) - 1) {
            ++m_history_pos;
            m_current_object = m_history[m_history_pos];
            m_inspector->edit(m_current_object);
            update_history_buttons();
        }
    }

    void search(const String& text) {
        // Filter properties in inspector
    }

private:
    void build_ui() {
        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();

        m_back_btn = new Button();
        m_back_btn->set_text("<");
        m_back_btn->set_disabled(true);
        m_back_btn->connect("pressed", this, "go_back");
        toolbar->add_child(m_back_btn);

        m_forward_btn = new Button();
        m_forward_btn->set_text(">");
        m_forward_btn->set_disabled(true);
        m_forward_btn->connect("pressed", this, "go_forward");
        toolbar->add_child(m_forward_btn);

        m_search_box = new LineEdit();
        m_search_box->set_placeholder("Filter properties...");
        m_search_box->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_search_box->connect("text_changed", this, "on_search_changed");
        toolbar->add_child(m_search_box);

        m_refresh_btn = new Button();
        m_refresh_btn->set_text("Refresh");
        m_refresh_btn->connect("pressed", this, "refresh");
        toolbar->add_child(m_refresh_btn);

        add_child(toolbar);

        // Inspector
        m_inspector = new EditorInspector();
        m_inspector->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(m_inspector);
    }

    void add_to_history(Object* object) {
        // Remove forward history
        if (m_history_pos < static_cast<int>(m_history.size()) - 1) {
            m_history.erase(m_history.begin() + m_history_pos + 1, m_history.end());
        }
        m_history.push_back(object);
        m_history_pos = static_cast<int>(m_history.size()) - 1;
        if (m_history.size() > 50) {
            m_history.erase(m_history.begin());
            --m_history_pos;
        }
    }

    void update_history_buttons() {
        m_back_btn->set_disabled(m_history_pos <= 0);
        m_forward_btn->set_disabled(m_history_pos >= static_cast<int>(m_history.size()) - 1);
    }

    void on_search_changed(const String& text) { search(text); }
};

// #############################################################################
// FileSystemDock - File browser panel
// #############################################################################
class FileSystemDock : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(FileSystemDock, VBoxContainer)

private:
    Tree* m_file_tree = nullptr;
    ItemList* m_file_list = nullptr;
    LineEdit* m_path_edit = nullptr;
    LineEdit* m_search_box = nullptr;
    Button* m_refresh_btn = nullptr;
    Button* m_favorite_btn = nullptr;
    Button* m_up_btn = nullptr;
    String m_current_path = "res://";
    std::vector<String> m_favorites;
    std::unordered_map<String, Ref<Texture2D>> m_thumbnail_cache;

public:
    static StringName get_class_static() { return StringName("FileSystemDock"); }

    FileSystemDock() {
        build_ui();
        refresh();
    }

    void set_current_path(const String& path) {
        if (DirAccess::dir_exists(path)) {
            m_current_path = path;
            m_path_edit->set_text(path);
            refresh();
        }
    }

    String get_current_path() const { return m_current_path; }

    void refresh() {
        build_file_tree();
        build_file_list();
    }

    void navigate_up() {
        if (m_current_path != "res://") {
            String parent = m_current_path.get_base_dir();
            if (parent.empty()) parent = "res://";
            set_current_path(parent);
        }
    }

    void add_favorite(const String& path) {
        if (std::find(m_favorites.begin(), m_favorites.end(), path) == m_favorites.end()) {
            m_favorites.push_back(path);
        }
    }

    void remove_favorite(const String& path) {
        auto it = std::find(m_favorites.begin(), m_favorites.end(), path);
        if (it != m_favorites.end()) m_favorites.erase(it);
    }

    void search(const String& text) {
        // Filter files
    }

    void create_folder() {
        // Show create folder dialog
    }

    void create_script() {
        // Show create script dialog
    }

    void create_scene() {
        // Create new scene
    }

    void delete_selected() {
        // Delete selected files/folders
    }

    void rename_selected() {
        // Rename selected file/folder
    }

    void duplicate_selected() {
        // Duplicate selected file
    }

private:
    void build_ui() {
        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();

        m_up_btn = new Button();
        m_up_btn->set_text("Up");
        m_up_btn->connect("pressed", this, "navigate_up");
        toolbar->add_child(m_up_btn);

        m_refresh_btn = new Button();
        m_refresh_btn->set_text("Refresh");
        m_refresh_btn->connect("pressed", this, "refresh");
        toolbar->add_child(m_refresh_btn);

        m_favorite_btn = new Button();
        m_favorite_btn->set_text("Favorite");
        m_favorite_btn->connect("pressed", this, "on_toggle_favorite");
        toolbar->add_child(m_favorite_btn);

        m_path_edit = new LineEdit();
        m_path_edit->set_text(m_current_path);
        m_path_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_path_edit->connect("text_entered", this, "on_path_entered");
        toolbar->add_child(m_path_edit);

        m_search_box = new LineEdit();
        m_search_box->set_placeholder("Search...");
        m_search_box->connect("text_changed", this, "on_search_changed");
        toolbar->add_child(m_search_box);

        add_child(toolbar);

        // Split view
        HSplitContainer* split = new HSplitContainer();
        split->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(split);

        // Directory tree
        m_file_tree = new Tree();
        m_file_tree->set_hide_root(true);
        m_file_tree->connect("item_selected", this, "on_tree_item_selected");
        split->add_child(m_file_tree);

        // File list
        m_file_list = new ItemList();
        m_file_list->set_icon_mode(ItemList::ICON_MODE_TOP);
        m_file_list->set_fixed_icon_size(vec2i(64, 64));
        m_file_list->connect("item_activated", this, "on_item_activated");
        split->add_child(m_file_list);
    }

    void build_file_tree() {
        m_file_tree->clear();
        TreeItem* root = m_file_tree->create_item();
        root->set_text(0, "res://");
        root->set_metadata(0, "res://");
        add_directory_to_tree(root, "res://");
    }

    void add_directory_to_tree(TreeItem* parent, const String& path) {
        Ref<DirAccess> dir = DirAccess::open(path);
        if (!dir.is_valid()) return;
        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item == "." || item == "..") continue;
            if (dir->current_is_dir()) {
                TreeItem* ti = m_file_tree->create_item(parent);
                ti->set_text(0, item);
                ti->set_metadata(0, path + "/" + item);
            }
        }
        dir->list_dir_end();
    }

    void build_file_list() {
        m_file_list->clear();
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
            } else {
                files.push_back(item);
            }
        }
        dir->list_dir_end();

        std::sort(dirs.begin(), dirs.end());
        std::sort(files.begin(), files.end());

        for (const auto& d : dirs) {
            int idx = m_file_list->add_item(d);
            m_file_list->set_item_icon(idx, get_folder_icon());
            m_file_list->set_item_metadata(idx, m_current_path + "/" + d);
        }
        for (const auto& f : files) {
            int idx = m_file_list->add_item(f);
            m_file_list->set_item_icon(idx, get_file_icon(f));
            m_file_list->set_item_metadata(idx, m_current_path + "/" + f);
        }
    }

    Ref<Texture2D> get_folder_icon() { return Ref<Texture2D>(); }
    Ref<Texture2D> get_file_icon(const String& path) { return Ref<Texture2D>(); }

    void on_path_entered(const String& path) { set_current_path(path); }
    void on_search_changed(const String& text) { search(text); }
    void on_tree_item_selected() {
        TreeItem* item = m_file_tree->get_selected();
        if (item) {
            set_current_path(item->get_metadata(0).as<String>());
        }
    }
    void on_item_activated(int idx) {
        String path = m_file_list->get_item_metadata(idx).as<String>();
        if (DirAccess::dir_exists(path)) {
            set_current_path(path);
        } else {
            // Open file
            emit_signal("file_selected", path);
        }
    }
    void on_toggle_favorite() {
        if (std::find(m_favorites.begin(), m_favorites.end(), m_current_path) != m_favorites.end()) {
            remove_favorite(m_current_path);
        } else {
            add_favorite(m_current_path);
        }
    }
};

// #############################################################################
// NodeDock - Node signals and groups panel
// #############################################################################
class NodeDock : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(NodeDock, VBoxContainer)

private:
    TabContainer* m_tabs = nullptr;
    Tree* m_signals_tree = nullptr;
    Tree* m_groups_tree = nullptr;
    Node* m_current_node = nullptr;

public:
    static StringName get_class_static() { return StringName("NodeDock"); }

    NodeDock() {
        build_ui();
    }

    void set_node(Node* node) {
        m_current_node = node;
        refresh_signals();
        refresh_groups();
    }

    void refresh_signals() {
        m_signals_tree->clear();
        if (!m_current_node) return;
        TreeItem* root = m_signals_tree->create_item();
        // Get signal list from ClassDB
        // ...
    }

    void refresh_groups() {
        m_groups_tree->clear();
        if (!m_current_node) return;
        TreeItem* root = m_groups_tree->create_item();
        for (const auto& group : m_current_node->get_groups()) {
            TreeItem* item = m_groups_tree->create_item(root);
            item->set_text(0, group.string());
        }
    }

    void add_to_group(const StringName& group) {
        if (m_current_node) {
            m_current_node->add_to_group(group);
            refresh_groups();
        }
    }

    void remove_from_group(const StringName& group) {
        if (m_current_node) {
            m_current_node->remove_from_group(group);
            refresh_groups();
        }
    }

private:
    void build_ui() {
        m_tabs = new TabContainer();
        m_tabs->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(m_tabs);

        // Signals tab
        m_signals_tree = new Tree();
        m_signals_tree->set_columns(1);
        m_signals_tree->set_column_title(0, "Signal");
        m_tabs->add_child(m_signals_tree);
        m_tabs->set_tab_title(0, "Signals");

        // Groups tab
        VBoxContainer* groups_vbox = new VBoxContainer();
        m_groups_tree = new Tree();
        m_groups_tree->set_columns(1);
        m_groups_tree->set_column_title(0, "Group");
        m_groups_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        groups_vbox->add_child(m_groups_tree);

        HBoxContainer* group_buttons = new HBoxContainer();
        Button* add_btn = new Button();
        add_btn->set_text("Add");
        add_btn->connect("pressed", this, "on_add_group");
        group_buttons->add_child(add_btn);
        Button* remove_btn = new Button();
        remove_btn->set_text("Remove");
        remove_btn->connect("pressed", this, "on_remove_group");
        group_buttons->add_child(remove_btn);
        groups_vbox->add_child(group_buttons);

        m_tabs->add_child(groups_vbox);
        m_tabs->set_tab_title(1, "Groups");
    }

    void on_add_group() {
        // Show add group dialog
    }

    void on_remove_group() {
        TreeItem* selected = m_groups_tree->get_selected();
        if (selected) {
            remove_from_group(selected->get_text(0).c_str());
        }
    }
};

// #############################################################################
// CanvasItemEditorPlugin - 2D editor viewport
// #############################################################################
class CanvasItemEditorPlugin : public EditorPlugin {
    XTU_GODOT_REGISTER_CLASS(CanvasItemEditorPlugin, EditorPlugin)

private:
    Control* m_viewport_container = nullptr;
    SubViewport* m_viewport = nullptr;
    HBoxContainer* m_toolbar = nullptr;
    Button* m_move_mode_btn = nullptr;
    Button* m_rotate_mode_btn = nullptr;
    Button* m_scale_mode_btn = nullptr;
    Button* m_snap_toggle_btn = nullptr;
    SpinBox* m_snap_x_spin = nullptr;
    SpinBox* m_snap_y_spin = nullptr;
    GizmoMode m_gizmo_mode = GizmoMode::MODE_MOVE;
    bool m_snap_enabled = false;
    vec2f m_snap_step = {8, 8};
    std::vector<Node2D*> m_selected_nodes;
    std::unordered_map<Node2D*, Transform2D> m_initial_transforms;

public:
    static StringName get_class_static() { return StringName("CanvasItemEditorPlugin"); }

    StringName get_plugin_name() const override { return StringName("2D"); }

    void _enter_tree() override {
        EditorPlugin::_enter_tree();
        build_ui();
    }

    void _exit_tree() override {
        EditorPlugin::_exit_tree();
    }

    bool handles(Object* object) const override {
        return dynamic_cast<CanvasItem*>(object) != nullptr;
    }

    void edit(Object* object) override {
        CanvasItem* item = dynamic_cast<CanvasItem*>(object);
        if (item) {
            m_selected_nodes.clear();
            if (auto* n2d = dynamic_cast<Node2D*>(item)) {
                m_selected_nodes.push_back(n2d);
            }
            update_gizmo();
        }
    }

    void make_visible(bool visible) override {
        m_viewport_container->set_visible(visible);
    }

    void set_gizmo_mode(GizmoMode mode) {
        m_gizmo_mode = mode;
        update_toolbar();
    }

    void set_snap_enabled(bool enabled) {
        m_snap_enabled = enabled;
        m_snap_toggle_btn->set_pressed(enabled);
    }

private:
    void build_ui() {
        m_viewport_container = new Control();
        m_viewport_container->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_viewport_container->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_control_to_container(m_viewport_container, "main");

        m_viewport = new SubViewport();
        m_viewport->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_viewport->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_viewport_container->add_child(m_viewport);

        // Create 2D camera
        Camera2D* camera = new Camera2D();
        camera->set_current(true);
        m_viewport->add_child(camera);

        build_toolbar();
    }

    void build_toolbar() {
        m_toolbar = new HBoxContainer();

        m_move_mode_btn = new Button();
        m_move_mode_btn->set_text("Move");
        m_move_mode_btn->set_toggle_mode(true);
        m_move_mode_btn->set_pressed(true);
        m_move_mode_btn->connect("pressed", this, "on_mode_move");
        m_toolbar->add_child(m_move_mode_btn);

        m_rotate_mode_btn = new Button();
        m_rotate_mode_btn->set_text("Rotate");
        m_rotate_mode_btn->set_toggle_mode(true);
        m_rotate_mode_btn->connect("pressed", this, "on_mode_rotate");
        m_toolbar->add_child(m_rotate_mode_btn);

        m_scale_mode_btn = new Button();
        m_scale_mode_btn->set_text("Scale");
        m_scale_mode_btn->set_toggle_mode(true);
        m_scale_mode_btn->connect("pressed", this, "on_mode_scale");
        m_toolbar->add_child(m_scale_mode_btn);

        m_toolbar->add_child(new VSeparator());

        m_snap_toggle_btn = new Button();
        m_snap_toggle_btn->set_text("Snap");
        m_snap_toggle_btn->set_toggle_mode(true);
        m_snap_toggle_btn->connect("toggled", this, "on_snap_toggled");
        m_toolbar->add_child(m_snap_toggle_btn);

        m_snap_x_spin = new SpinBox();
        m_snap_x_spin->set_min(1);
        m_snap_x_spin->set_max(256);
        m_snap_x_spin->set_value(8);
        m_snap_x_spin->connect("value_changed", this, "on_snap_x_changed");
        m_toolbar->add_child(m_snap_x_spin);

        m_snap_y_spin = new SpinBox();
        m_snap_y_spin->set_min(1);
        m_snap_y_spin->set_max(256);
        m_snap_y_spin->set_value(8);
        m_snap_y_spin->connect("value_changed", this, "on_snap_y_changed");
        m_toolbar->add_child(m_snap_y_spin);

        add_control_to_container(m_toolbar, "toolbar");
    }

    void update_toolbar() {
        m_move_mode_btn->set_pressed(m_gizmo_mode == GizmoMode::MODE_MOVE);
        m_rotate_mode_btn->set_pressed(m_gizmo_mode == GizmoMode::MODE_ROTATE);
        m_scale_mode_btn->set_pressed(m_gizmo_mode == GizmoMode::MODE_SCALE);
    }

    void update_gizmo() {
        m_viewport->update();
    }

    void on_mode_move() { set_gizmo_mode(GizmoMode::MODE_MOVE); }
    void on_mode_rotate() { set_gizmo_mode(GizmoMode::MODE_ROTATE); }
    void on_mode_scale() { set_gizmo_mode(GizmoMode::MODE_SCALE); }
    void on_snap_toggled(bool pressed) { m_snap_enabled = pressed; }
    void on_snap_x_changed(float value) { m_snap_step.x() = value; }
    void on_snap_y_changed(float value) { m_snap_step.y() = value; }
};

// #############################################################################
// SpatialEditorPlugin - 3D editor viewport
// #############################################################################
class SpatialEditorPlugin : public EditorPlugin {
    XTU_GODOT_REGISTER_CLASS(SpatialEditorPlugin, EditorPlugin)

private:
    Control* m_viewport_container = nullptr;
    SubViewport* m_viewport = nullptr;
    HBoxContainer* m_toolbar = nullptr;
    Button* m_move_mode_btn = nullptr;
    Button* m_rotate_mode_btn = nullptr;
    Button* m_scale_mode_btn = nullptr;
    Button* m_snap_toggle_btn = nullptr;
    SpinBox* m_snap_translate_spin = nullptr;
    SpinBox* m_snap_rotate_spin = nullptr;
    SpinBox* m_snap_scale_spin = nullptr;
    OptionButton* m_coord_system_select = nullptr;
    GizmoMode m_gizmo_mode = GizmoMode::MODE_MOVE;
    bool m_snap_enabled = false;
    float m_translate_snap = 1.0f;
    float m_rotate_snap = 15.0f;
    float m_scale_snap = 0.1f;
    bool m_local_coords = false;
    std::vector<Node3D*> m_selected_nodes;

public:
    static StringName get_class_static() { return StringName("SpatialEditorPlugin"); }

    StringName get_plugin_name() const override { return StringName("3D"); }

    void _enter_tree() override {
        EditorPlugin::_enter_tree();
        build_ui();
    }

    void _exit_tree() override {
        EditorPlugin::_exit_tree();
    }

    bool handles(Object* object) const override {
        return dynamic_cast<Node3D*>(object) != nullptr;
    }

    void edit(Object* object) override {
        Node3D* n3d = dynamic_cast<Node3D*>(object);
        if (n3d) {
            m_selected_nodes.clear();
            m_selected_nodes.push_back(n3d);
            update_gizmo();
        }
    }

    void make_visible(bool visible) override {
        m_viewport_container->set_visible(visible);
    }

private:
    void build_ui() {
        m_viewport_container = new Control();
        m_viewport_container->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_viewport_container->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_control_to_container(m_viewport_container, "main");

        m_viewport = new SubViewport();
        m_viewport->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_viewport->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_viewport_container->add_child(m_viewport);

        // Create 3D camera
        Camera3D* camera = new Camera3D();
        camera->set_current(true);
        camera->set_position(vec3f(0, 2, 5));
        camera->look_at(vec3f(0, 0, 0));
        m_viewport->add_child(camera);

        // Add environment
        WorldEnvironment* env = new WorldEnvironment();
        Ref<Environment> env_res;
        env_res.instance();
        env_res->set_background_mode(Environment::BG_MODE_COLOR);
        env_res->set_ambient_light_color(Color(0.3f, 0.3f, 0.3f));
        env->set_environment(env_res);
        m_viewport->add_child(env);

        // Add lights
        DirectionalLight3D* light = new DirectionalLight3D();
        light->set_rotation(vec3f(-0.5f, 0.5f, 0));
        m_viewport->add_child(light);

        build_toolbar();
    }

    void build_toolbar() {
        m_toolbar = new HBoxContainer();

        m_move_mode_btn = new Button();
        m_move_mode_btn->set_text("Move");
        m_move_mode_btn->set_toggle_mode(true);
        m_move_mode_btn->set_pressed(true);
        m_toolbar->add_child(m_move_mode_btn);

        m_rotate_mode_btn = new Button();
        m_rotate_mode_btn->set_text("Rotate");
        m_rotate_mode_btn->set_toggle_mode(true);
        m_toolbar->add_child(m_rotate_mode_btn);

        m_scale_mode_btn = new Button();
        m_scale_mode_btn->set_text("Scale");
        m_scale_mode_btn->set_toggle_mode(true);
        m_toolbar->add_child(m_scale_mode_btn);

        m_toolbar->add_child(new VSeparator());

        m_snap_toggle_btn = new Button();
        m_snap_toggle_btn->set_text("Snap");
        m_snap_toggle_btn->set_toggle_mode(true);
        m_toolbar->add_child(m_snap_toggle_btn);

        m_snap_translate_spin = new SpinBox();
        m_snap_translate_spin->set_min(0.01f);
        m_snap_translate_spin->set_max(100.0f);
        m_snap_translate_spin->set_step(0.1f);
        m_snap_translate_spin->set_value(1.0f);
        m_toolbar->add_child(m_snap_translate_spin);

        m_snap_rotate_spin = new SpinBox();
        m_snap_rotate_spin->set_min(0.1f);
        m_snap_rotate_spin->set_max(180.0f);
        m_snap_rotate_spin->set_step(1.0f);
        m_snap_rotate_spin->set_value(15.0f);
        m_toolbar->add_child(m_snap_rotate_spin);

        m_snap_scale_spin = new SpinBox();
        m_snap_scale_spin->set_min(0.01f);
        m_snap_scale_spin->set_max(10.0f);
        m_snap_scale_spin->set_step(0.01f);
        m_snap_scale_spin->set_value(0.1f);
        m_toolbar->add_child(m_snap_scale_spin);

        m_toolbar->add_child(new VSeparator());

        m_coord_system_select = new OptionButton();
        m_coord_system_select->add_item("Global");
        m_coord_system_select->add_item("Local");
        m_toolbar->add_child(m_coord_system_select);

        add_control_to_container(m_toolbar, "toolbar");
    }

    void update_gizmo() {
        m_viewport->update();
    }
};

} // namespace editor

// Bring into main namespace
using editor::SceneTreeDock;
using editor::InspectorDock;
using editor::FileSystemDock;
using editor::NodeDock;
using editor::CanvasItemEditorPlugin;
using editor::SpatialEditorPlugin;
using editor::GizmoHandle;
using editor::GizmoMode;
using editor::TransformSnapMode;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_DOCKS_HPP