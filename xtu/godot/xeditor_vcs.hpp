// include/xtu/godot/xeditor_vcs.hpp
// xtensor-unified - Version control integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_VCS_HPP
#define XTU_GODOT_XEDITOR_VCS_HPP

#include <atomic>
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
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorVCSInterface;
class GitPlugin;
class VersionControlEditorPlugin;

// #############################################################################
// VCS file status
// #############################################################################
enum class VCSFileStatus : uint8_t {
    STATUS_NORMAL = 0,
    STATUS_MODIFIED = 1,
    STATUS_ADDED = 2,
    STATUS_DELETED = 3,
    STATUS_RENAMED = 4,
    STATUS_CONFLICT = 5,
    STATUS_IGNORED = 6,
    STATUS_UNTRACKED = 7,
    STATUS_UPTODATE = 8
};

// #############################################################################
// VCS change type
// #############################################################################
enum class VCSChangeType : uint8_t {
    CHANGE_MODIFY = 0,
    CHANGE_ADD = 1,
    CHANGE_DELETE = 2,
    CHANGE_RENAME = 3
};

// #############################################################################
// VCS commit info
// #############################################################################
struct VCSCommitInfo {
    String hash;
    String author;
    String email;
    String message;
    uint64_t timestamp = 0;
    std::vector<String> files;
};

// #############################################################################
// VCS diff hunk
// #############################################################################
struct VCSDiffHunk {
    int old_start = 0;
    int old_lines = 0;
    int new_start = 0;
    int new_lines = 0;
    String content;
};

// #############################################################################
// VCS file diff
// #############################################################################
struct VCSFileDiff {
    String path;
    String old_path;  // for renames
    VCSChangeType type = VCSChangeType::CHANGE_MODIFY;
    std::vector<VCSDiffHunk> hunks;
};

// #############################################################################
// EditorVCSInterface - Base class for VCS backends
// #############################################################################
class EditorVCSInterface : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorVCSInterface, Object)

public:
    static StringName get_class_static() { return StringName("EditorVCSInterface"); }

    virtual String get_vcs_name() const = 0;
    virtual bool initialize(const String& project_path) = 0;
    virtual void shutdown() = 0;

    virtual bool is_available() const { return true; }
    virtual bool is_initialized() const = 0;

    virtual VCSFileStatus get_file_status(const String& path) = 0;
    virtual void get_file_status_async(const std::vector<String>& paths,
                                       std::function<void(std::unordered_map<String, VCSFileStatus>)> callback) {}

    virtual bool stage_file(const String& path) = 0;
    virtual bool unstage_file(const String& path) = 0;
    virtual bool discard_file(const String& path) = 0;

    virtual bool commit(const String& message, const std::vector<String>& files) = 0;

    virtual bool push(const String& remote = "origin", const String& branch = "") = 0;
    virtual bool pull(const String& remote = "origin", const String& branch = "") = 0;
    virtual bool fetch(const String& remote = "origin") = 0;

    virtual std::vector<String> get_modified_files() = 0;
    virtual std::vector<String> get_staged_files() = 0;
    virtual std::vector<String> get_untracked_files() = 0;

    virtual std::vector<VCSCommitInfo> get_commit_history(int max_count = 100) = 0;
    virtual VCSFileDiff get_diff(const String& path) = 0;

    virtual std::vector<String> get_branches() = 0;
    virtual String get_current_branch() = 0;
    virtual bool create_branch(const String& name) = 0;
    virtual bool checkout_branch(const String& name) = 0;
    virtual bool merge_branch(const String& name) = 0;
};

// #############################################################################
// GitPlugin - Git implementation
// #############################################################################
class GitPlugin : public EditorVCSInterface {
    XTU_GODOT_REGISTER_CLASS(GitPlugin, EditorVCSInterface)

private:
    String m_repo_path;
    bool m_initialized = false;
    std::mutex m_mutex;
    std::queue<std::function<void()>> m_async_tasks;
    std::thread m_worker;
    std::atomic<bool> m_worker_running{false};
    std::unordered_map<String, VCSFileStatus> m_status_cache;
    std::chrono::steady_clock::time_point m_last_status_update;

public:
    static StringName get_class_static() { return StringName("GitPlugin"); }

    ~GitPlugin() { shutdown(); }

    String get_vcs_name() const override { return "Git"; }

    bool initialize(const String& project_path) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_repo_path = project_path;
        if (!is_git_repository()) {
            return false;
        }
        start_worker();
        m_initialized = true;
        return true;
    }

    void shutdown() override {
        stop_worker();
        std::lock_guard<std::mutex> lock(m_mutex);
        m_initialized = false;
    }

    bool is_initialized() const override { return m_initialized; }

    VCSFileStatus get_file_status(const String& path) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_status_cache.find(path);
        if (it != m_status_cache.end()) {
            return it->second;
        }
        String output = execute_git({"status", "--porcelain", "--", path});
        return parse_status(output);
    }

    bool stage_file(const String& path) override {
        String output = execute_git({"add", "--", path});
        emit_signal("file_staged", path);
        return true;
    }

    bool unstage_file(const String& path) override {
        String output = execute_git({"reset", "HEAD", "--", path});
        emit_signal("file_unstaged", path);
        return true;
    }

    bool discard_file(const String& path) override {
        String output = execute_git({"checkout", "--", path});
        emit_signal("file_discarded", path);
        return true;
    }

    bool commit(const String& message, const std::vector<String>& files) override {
        std::vector<String> args = {"commit", "-m", message};
        args.insert(args.end(), files.begin(), files.end());
        String output = execute_git(args);
        emit_signal("committed", message);
        return true;
    }

    bool push(const String& remote, const String& branch) override {
        std::vector<String> args = {"push"};
        if (!remote.empty()) args.push_back(remote);
        if (!branch.empty()) args.push_back(branch);
        String output = execute_git(args);
        emit_signal("pushed");
        return true;
    }

    bool pull(const String& remote, const String& branch) override {
        std::vector<String> args = {"pull"};
        if (!remote.empty()) args.push_back(remote);
        if (!branch.empty()) args.push_back(branch);
        String output = execute_git(args);
        emit_signal("pulled");
        return true;
    }

    bool fetch(const String& remote) override {
        std::vector<String> args = {"fetch"};
        if (!remote.empty()) args.push_back(remote);
        String output = execute_git(args);
        return true;
    }

    std::vector<String> get_modified_files() override {
        String output = execute_git({"status", "--porcelain"});
        return parse_modified_files(output);
    }

    std::vector<String> get_staged_files() override {
        String output = execute_git({"diff", "--cached", "--name-only"});
        return parse_file_list(output);
    }

    std::vector<String> get_untracked_files() override {
        String output = execute_git({"ls-files", "--others", "--exclude-standard"});
        return parse_file_list(output);
    }

    std::vector<VCSCommitInfo> get_commit_history(int max_count) override {
        String output = execute_git({"log", "--format=%H|%an|%ae|%at|%s", "-n", String::num(max_count)});
        return parse_commit_history(output);
    }

    VCSFileDiff get_diff(const String& path) override {
        VCSFileDiff diff;
        diff.path = path;
        String output = execute_git({"diff", "--unified=3", "--", path});
        diff.hunks = parse_diff_hunks(output);
        return diff;
    }

    std::vector<String> get_branches() override {
        String output = execute_git({"branch", "--format=%(refname:short)"});
        return parse_file_list(output);
    }

    String get_current_branch() override {
        String output = execute_git({"branch", "--show-current"});
        return output.strip_edges();
    }

    bool create_branch(const String& name) override {
        String output = execute_git({"branch", name});
        return true;
    }

    bool checkout_branch(const String& name) override {
        String output = execute_git({"checkout", name});
        emit_signal("branch_changed", name);
        return true;
    }

    bool merge_branch(const String& name) override {
        String output = execute_git({"merge", name});
        return true;
    }

    void refresh_status() {
        enqueue_task([this]() {
            std::lock_guard<std::mutex> lock(m_mutex);
            String output = execute_git({"status", "--porcelain"});
            m_status_cache = parse_status_map(output);
            m_last_status_update = std::chrono::steady_clock::now();
        });
    }

private:
    bool is_git_repository() const {
        String output = execute_git({"rev-parse", "--git-dir"});
        return !output.empty() && output.find(".git") != String::npos;
    }

    String execute_git(const std::vector<String>& args) const {
        std::vector<String> full_args = {"git", "-C", m_repo_path};
        full_args.insert(full_args.end(), args.begin(), args.end());

        String cmd;
        for (const auto& a : full_args) {
            cmd += a + " ";
        }

        FILE* pipe = popen(cmd.utf8(), "r");
        if (!pipe) return String();

        std::string result;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        pclose(pipe);
        return String(result.c_str());
    }

    VCSFileStatus parse_status(const String& output) const {
        if (output.empty()) return VCSFileStatus::STATUS_NORMAL;
        char status = output[0];
        switch (status) {
            case 'M': return VCSFileStatus::STATUS_MODIFIED;
            case 'A': return VCSFileStatus::STATUS_ADDED;
            case 'D': return VCSFileStatus::STATUS_DELETED;
            case 'R': return VCSFileStatus::STATUS_RENAMED;
            case 'C': return VCSFileStatus::STATUS_CONFLICT;
            case '?': return VCSFileStatus::STATUS_UNTRACKED;
            default: return VCSFileStatus::STATUS_NORMAL;
        }
    }

    std::unordered_map<String, VCSFileStatus> parse_status_map(const String& output) const {
        std::unordered_map<String, VCSFileStatus> result;
        auto lines = output.split("\n");
        for (const auto& line : lines) {
            if (line.length() < 4) continue;
            char x = line[0];
            char y = line[1];
            String file = line.substr(3).strip_edges();

            VCSFileStatus status = VCSFileStatus::STATUS_NORMAL;
            if (x == 'M' || y == 'M') status = VCSFileStatus::STATUS_MODIFIED;
            else if (x == 'A' || y == 'A') status = VCSFileStatus::STATUS_ADDED;
            else if (x == 'D' || y == 'D') status = VCSFileStatus::STATUS_DELETED;
            else if (x == 'R' || y == 'R') status = VCSFileStatus::STATUS_RENAMED;
            else if (x == 'U' || y == 'U' || x == 'A' && y == 'A' || x == 'D' && y == 'D') {
                status = VCSFileStatus::STATUS_CONFLICT;
            } else if (x == '?') {
                status = VCSFileStatus::STATUS_UNTRACKED;
            }

            if (status != VCSFileStatus::STATUS_NORMAL) {
                result[file] = status;
            }
        }
        return result;
    }

    std::vector<String> parse_modified_files(const String& output) const {
        std::vector<String> result;
        auto lines = output.split("\n");
        for (const auto& line : lines) {
            if (line.length() >= 4) {
                result.push_back(line.substr(3).strip_edges());
            }
        }
        return result;
    }

    std::vector<String> parse_file_list(const String& output) const {
        std::vector<String> result;
        auto lines = output.split("\n");
        for (const auto& line : lines) {
            String trimmed = line.strip_edges();
            if (!trimmed.empty()) {
                result.push_back(trimmed);
            }
        }
        return result;
    }

    std::vector<VCSCommitInfo> parse_commit_history(const String& output) const {
        std::vector<VCSCommitInfo> result;
        auto lines = output.split("\n");
        for (const auto& line : lines) {
            auto parts = line.split("|");
            if (parts.size() >= 5) {
                VCSCommitInfo info;
                info.hash = parts[0];
                info.author = parts[1];
                info.email = parts[2];
                info.timestamp = parts[3].to_int();
                info.message = parts[4];
                result.push_back(info);
            }
        }
        return result;
    }

    std::vector<VCSDiffHunk> parse_diff_hunks(const String& output) const {
        std::vector<VCSDiffHunk> result;
        auto lines = output.split("\n");
        VCSDiffHunk current;
        for (const auto& line : lines) {
            if (line.begins_with("@@")) {
                if (!current.content.empty()) {
                    result.push_back(current);
                }
                current = VCSDiffHunk();
                // Parse @@ -old_start,old_lines +new_start,new_lines @@
                size_t minus = line.find('-');
                size_t plus = line.find('+');
                size_t at = line.find('@', plus);
                if (minus != String::npos && plus != String::npos) {
                    String old_part = line.substr(minus + 1, plus - minus - 1).strip_edges();
                    String new_part = line.substr(plus + 1, at - plus - 1).strip_edges();
                    auto old_nums = old_part.split(",");
                    auto new_nums = new_part.split(",");
                    if (old_nums.size() >= 1) current.old_start = old_nums[0].to_int();
                    if (old_nums.size() >= 2) current.old_lines = old_nums[1].to_int();
                    if (new_nums.size() >= 1) current.new_start = new_nums[0].to_int();
                    if (new_nums.size() >= 2) current.new_lines = new_nums[1].to_int();
                }
            } else if (!line.empty()) {
                current.content += line + "\n";
            }
        }
        if (!current.content.empty()) {
            result.push_back(current);
        }
        return result;
    }

    void start_worker() {
        m_worker_running = true;
        m_worker = std::thread([this]() { worker_loop(); });
    }

    void stop_worker() {
        m_worker_running = false;
        if (m_worker.joinable()) m_worker.join();
    }

    void worker_loop() {
        while (m_worker_running) {
            std::function<void()> task;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_async_tasks.empty()) {
                    task = std::move(m_async_tasks.front());
                    m_async_tasks.pop();
                }
            }
            if (task) {
                task();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void enqueue_task(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_async_tasks.push(task);
    }
};

// #############################################################################
// VersionControlEditorPlugin - VCS UI integration
// #############################################################################
class VersionControlEditorPlugin : public EditorPlugin {
    XTU_GODOT_REGISTER_CLASS(VersionControlEditorPlugin, EditorPlugin)

private:
    Ref<EditorVCSInterface> m_vcs;
    Button* m_commit_btn = nullptr;
    Button* m_push_btn = nullptr;
    Button* m_pull_btn = nullptr;
    Button* m_refresh_btn = nullptr;
    Label* m_branch_label = nullptr;
    Tree* m_file_tree = nullptr;
    VBoxContainer* m_panel = nullptr;
    AcceptDialog* m_commit_dialog = nullptr;
    TextEdit* m_commit_message = nullptr;
    Tree* m_commit_files_tree = nullptr;

public:
    static StringName get_class_static() { return StringName("VersionControlEditorPlugin"); }

    StringName get_plugin_name() const override { return StringName("VersionControl"); }

    void _enter_tree() override {
        EditorPlugin::_enter_tree();
        build_ui();
        initialize_vcs();
    }

    void _exit_tree() override {
        if (m_vcs.is_valid()) {
            m_vcs->shutdown();
        }
        remove_control_from_container(m_panel);
        EditorPlugin::_exit_tree();
    }

    void set_vcs(const Ref<EditorVCSInterface>& vcs) {
        if (m_vcs.is_valid()) {
            m_vcs->shutdown();
        }
        m_vcs = vcs;
        if (vcs.is_valid()) {
            vcs->connect("file_staged", this, "on_file_changed");
            vcs->connect("file_unstaged", this, "on_file_changed");
            vcs->connect("committed", this, "on_committed");
            vcs->connect("branch_changed", this, "on_branch_changed");
            refresh();
        }
    }

    void refresh() {
        if (!m_vcs.is_valid()) return;
        m_branch_label->set_text(m_vcs->get_current_branch());
        refresh_file_tree();
    }

    void show_commit_dialog() {
        if (!m_vcs.is_valid()) return;

        m_commit_files_tree->clear();
        TreeItem* root = m_commit_files_tree->create_item();

        auto staged = m_vcs->get_staged_files();
        for (const auto& file : staged) {
            TreeItem* item = m_commit_files_tree->create_item(root);
            item->set_text(0, file);
            item->set_checked(0, true);
        }

        m_commit_message->set_text("");
        m_commit_dialog->popup_centered();
    }

    void commit() {
        if (!m_vcs.is_valid()) return;

        String message = m_commit_message->get_text();
        std::vector<String> files;
        TreeItem* root = m_commit_files_tree->get_root();
        TreeItem* item = root->get_first_child();
        while (item) {
            if (item->is_checked(0)) {
                files.push_back(item->get_text(0));
            }
            item = item->get_next();
        }

        if (!message.empty() && !files.empty()) {
            m_vcs->commit(message, files);
        }
    }

    void push() {
        if (m_vcs.is_valid()) {
            m_vcs->push();
        }
    }

    void pull() {
        if (m_vcs.is_valid()) {
            m_vcs->pull();
        }
    }

private:
    void build_ui() {
        m_panel = new VBoxContainer();

        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();

        m_commit_btn = new Button();
        m_commit_btn->set_text("Commit");
        m_commit_btn->connect("pressed", this, "show_commit_dialog");
        toolbar->add_child(m_commit_btn);

        m_push_btn = new Button();
        m_push_btn->set_text("Push");
        m_push_btn->connect("pressed", this, "push");
        toolbar->add_child(m_push_btn);

        m_pull_btn = new Button();
        m_pull_btn->set_text("Pull");
        m_pull_btn->connect("pressed", this, "pull");
        toolbar->add_child(m_pull_btn);

        m_refresh_btn = new Button();
        m_refresh_btn->set_text("Refresh");
        m_refresh_btn->connect("pressed", this, "refresh");
        toolbar->add_child(m_refresh_btn);

        toolbar->add_child(new VSeparator());

        m_branch_label = new Label();
        m_branch_label->set_text("main");
        toolbar->add_child(m_branch_label);

        m_panel->add_child(toolbar);

        // File tree
        m_file_tree = new Tree();
        m_file_tree->set_columns(2);
        m_file_tree->set_column_title(0, "File");
        m_file_tree->set_column_title(1, "Status");
        m_file_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_file_tree->connect("item_activated", this, "on_item_activated");
        m_file_tree->connect("item_rmb_selected", this, "on_item_rmb");
        m_panel->add_child(m_file_tree);

        add_control_to_container(m_panel, "bottom");

        // Commit dialog
        build_commit_dialog();
    }

    void build_commit_dialog() {
        m_commit_dialog = new AcceptDialog();
        m_commit_dialog->set_title("Commit Changes");
        m_commit_dialog->connect("confirmed", this, "commit");

        VBoxContainer* main = new VBoxContainer();
        m_commit_dialog->add_child(main);

        main->add_child(new Label("Commit Message:"));
        m_commit_message = new TextEdit();
        m_commit_message->set_custom_minimum_size(vec2f(400, 80));
        main->add_child(m_commit_message);

        main->add_child(new Label("Files to commit:"));
        m_commit_files_tree = new Tree();
        m_commit_files_tree->set_columns(1);
        m_commit_files_tree->set_select_mode(Tree::SELECT_ROW);
        m_commit_files_tree->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        main->add_child(m_commit_files_tree);
    }

    void initialize_vcs() {
        String project_path = "res://";
        Ref<GitPlugin> git;
        git.instance();
        if (git->initialize(project_path)) {
            set_vcs(git);
        }
    }

    void refresh_file_tree() {
        m_file_tree->clear();
        if (!m_vcs.is_valid()) return;

        TreeItem* root = m_file_tree->create_item();

        auto modified = m_vcs->get_modified_files();
        for (const auto& file : modified) {
            VCSFileStatus status = m_vcs->get_file_status(file);
            TreeItem* item = m_file_tree->create_item(root);
            item->set_text(0, file);
            item->set_text(1, status_to_string(status));
            item->set_metadata(0, file);
            set_item_color(item, status);
        }
    }

    String status_to_string(VCSFileStatus status) const {
        switch (status) {
            case VCSFileStatus::STATUS_MODIFIED: return "Modified";
            case VCSFileStatus::STATUS_ADDED: return "Added";
            case VCSFileStatus::STATUS_DELETED: return "Deleted";
            case VCSFileStatus::STATUS_RENAMED: return "Renamed";
            case VCSFileStatus::STATUS_CONFLICT: return "Conflict!";
            case VCSFileStatus::STATUS_UNTRACKED: return "Untracked";
            default: return "Normal";
        }
    }

    void set_item_color(TreeItem* item, VCSFileStatus status) {
        Color color;
        switch (status) {
            case VCSFileStatus::STATUS_MODIFIED: color = Color(1.0f, 0.8f, 0.2f); break;
            case VCSFileStatus::STATUS_ADDED: color = Color(0.2f, 1.0f, 0.2f); break;
            case VCSFileStatus::STATUS_DELETED: color = Color(1.0f, 0.2f, 0.2f); break;
            case VCSFileStatus::STATUS_CONFLICT: color = Color(1.0f, 0.0f, 0.0f); break;
            case VCSFileStatus::STATUS_UNTRACKED: color = Color(0.5f, 0.5f, 0.5f); break;
            default: return;
        }
        item->set_custom_color(0, color);
        item->set_custom_color(1, color);
    }

    void on_file_changed(const String& file) { refresh_file_tree(); }
    void on_committed(const String& message) { refresh_file_tree(); }
    void on_branch_changed(const String& branch) { m_branch_label->set_text(branch); }

    void on_item_activated() {
        TreeItem* item = m_file_tree->get_selected();
        if (item && item != m_file_tree->get_root()) {
            String file = item->get_metadata(0).as<String>();
            VCSFileDiff diff = m_vcs->get_diff(file);
            show_diff_dialog(file, diff);
        }
    }

    void on_item_rmb(const vec2f& pos) {
        TreeItem* item = m_file_tree->get_selected();
        if (!item || item == m_file_tree->get_root()) return;

        String file = item->get_metadata(0).as<String>();
        VCSFileStatus status = m_vcs->get_file_status(file);

        PopupMenu* menu = new PopupMenu();
        if (status == VCSFileStatus::STATUS_UNTRACKED || status == VCSFileStatus::STATUS_MODIFIED) {
            menu->add_item("Stage", 0);
        }
        if (status == VCSFileStatus::STATUS_ADDED || status == VCSFileStatus::STATUS_MODIFIED) {
            menu->add_item("Unstage", 1);
        }
        menu->add_item("Discard", 2);
        menu->add_separator();
        menu->add_item("Show Diff", 3);

        menu->connect("id_pressed", this, "on_menu_action", file);
        menu->set_position(pos);
        menu->popup();
    }

    void on_menu_action(int id, const String& file) {
        if (!m_vcs.is_valid()) return;
        switch (id) {
            case 0: m_vcs->stage_file(file); break;
            case 1: m_vcs->unstage_file(file); break;
            case 2: m_vcs->discard_file(file); break;
            case 3: {
                VCSFileDiff diff = m_vcs->get_diff(file);
                show_diff_dialog(file, diff);
                break;
            }
        }
    }

    void show_diff_dialog(const String& file, const VCSFileDiff& diff) {
        AcceptDialog* dialog = new AcceptDialog();
        dialog->set_title("Diff: " + file);

        VBoxContainer* main = new VBoxContainer();
        dialog->add_child(main);

        RichTextLabel* diff_view = new RichTextLabel();
        diff_view->set_custom_minimum_size(vec2f(600, 400));
        diff_view->set_use_bbcode(true);

        String text = "[code]";
        for (const auto& hunk : diff.hunks) {
            text += hunk.content;
        }
        text += "[/code]";
        diff_view->set_text(text);

        main->add_child(diff_view);
        dialog->popup_centered();
    }
};

} // namespace editor

// Bring into main namespace
using editor::EditorVCSInterface;
using editor::GitPlugin;
using editor::VersionControlEditorPlugin;
using editor::VCSFileStatus;
using editor::VCSChangeType;
using editor::VCSCommitInfo;
using editor::VCSFileDiff;
using editor::VCSDiffHunk;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_VCS_HPP