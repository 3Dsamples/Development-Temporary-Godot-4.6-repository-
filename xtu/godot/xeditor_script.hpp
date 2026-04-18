// include/xtu/godot/xeditor_script.hpp
// xtensor-unified - Editor code editing tools for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_SCRIPT_HPP
#define XTU_GODOT_XEDITOR_SCRIPT_HPP

#include <algorithm>
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

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xgdscript.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class CodeEditor;
class ScriptEditor;
class ScriptTextEditor;
class EditorHelp;
class ScriptCreateDialog;
class EditorAutoloadSettings;

// #############################################################################
// Syntax highlighting theme
// #############################################################################
enum class SyntaxTheme : uint8_t {
    THEME_DEFAULT = 0,
    THEME_DARK = 1,
    THEME_LIGHT = 2,
    THEME_MONOKAI = 3,
    THEME_SOLARIZED = 4,
    THEME_CUSTOM = 5
};

// #############################################################################
// Code completion trigger
// #############################################################################
enum class CompletionTrigger : uint8_t {
    TRIGGER_MANUAL = 0,
    TRIGGER_DOT = 1,
    TRIGGER_AUTO = 2,
    TRIGGER_PAREN = 3
};

// #############################################################################
// Script template type
// #############################################################################
enum class ScriptTemplateType : uint8_t {
    TEMPLATE_EMPTY = 0,
    TEMPLATE_NODE = 1,
    TEMPLATE_RESOURCE = 2,
    TEMPLATE_PLUGIN = 3,
    TEMPLATE_CUSTOM = 4
};

// #############################################################################
// Syntax token types
// #############################################################################
enum class SyntaxTokenType : uint8_t {
    TOKEN_TEXT = 0,
    TOKEN_KEYWORD = 1,
    TOKEN_TYPE = 2,
    TOKEN_FUNCTION = 3,
    TOKEN_NUMBER = 4,
    TOKEN_STRING = 5,
    TOKEN_COMMENT = 6,
    TOKEN_SYMBOL = 7,
    TOKEN_ERROR = 8
};

// #############################################################################
// Syntax token
// #############################################################################
struct SyntaxToken {
    SyntaxTokenType type = SyntaxTokenType::TOKEN_TEXT;
    int start_line = 0;
    int start_column = 0;
    int end_line = 0;
    int end_column = 0;
    String text;
};

// #############################################################################
// Code completion item
// #############################################################################
struct CompletionItem {
    String label;
    String detail;
    String insert_text;
    String documentation;
    VariantType return_type = VariantType::NIL;
    int priority = 0;
    bool is_function = false;
    bool is_property = false;
    bool is_constant = false;
};

// #############################################################################
// CodeEditor - Base text editor with syntax highlighting
// #############################################################################
class CodeEditor : public Control {
    XTU_GODOT_REGISTER_CLASS(CodeEditor, Control)

protected:
    String m_text;
    std::vector<String> m_lines;
    std::vector<SyntaxToken> m_tokens;
    SyntaxTheme m_theme = SyntaxTheme::THEME_DARK;
    std::unordered_map<SyntaxTokenType, Color> m_colors;
    int m_caret_line = 0;
    int m_caret_column = 0;
    int m_selection_start_line = 0;
    int m_selection_start_column = 0;
    int m_selection_end_line = 0;
    int m_selection_end_column = 0;
    bool m_line_numbers = true;
    bool m_highlight_current_line = true;
    bool m_brace_matching = true;
    bool m_code_folding = true;
    bool m_show_whitespace = false;
    int m_tab_size = 4;
    float m_line_height = 20.0f;
    float m_char_width = 10.0f;
    std::vector<int> m_folded_lines;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("CodeEditor"); }

    CodeEditor() {
        initialize_colors();
    }

    void set_text(const String& text) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_text = text;
        update_lines();
        retokenize();
        update();
    }

    String get_text() const { return m_text; }

    void insert_text_at_caret(const String& text) {
        if (m_lines.empty()) {
            m_lines.push_back(text);
        } else {
            m_lines[m_caret_line] = m_lines[m_caret_line].substr(0, m_caret_column) +
                                    text +
                                    m_lines[m_caret_line].substr(m_caret_column);
        }
        m_caret_column += text.length();
        rebuild_text();
        retokenize();
        update();
    }

    void delete_selection() {
        if (!has_selection()) return;
        // Remove selected text
        rebuild_text();
        retokenize();
        update();
    }

    void copy() {}
    void cut() {}
    void paste() {}

    void undo() {}
    void redo() {}

    void set_line_numbers(bool show) { m_line_numbers = show; update(); }
    bool get_line_numbers() const { return m_line_numbers; }

    void set_highlight_current_line(bool highlight) { m_highlight_current_line = highlight; update(); }
    bool get_highlight_current_line() const { return m_highlight_current_line; }

    void set_brace_matching(bool match) { m_brace_matching = match; update(); }
    bool get_brace_matching() const { return m_brace_matching; }

    void set_theme(SyntaxTheme theme) {
        m_theme = theme;
        initialize_colors();
        update();
    }

    SyntaxTheme get_theme() const { return m_theme; }

    void set_tab_size(int size) { m_tab_size = size; update(); }
    int get_tab_size() const { return m_tab_size; }

    bool has_selection() const {
        return m_selection_start_line != m_selection_end_line ||
               m_selection_start_column != m_selection_end_column;
    }

    void goto_line(int line) {
        m_caret_line = std::clamp(line, 0, static_cast<int>(m_lines.size()) - 1);
        m_caret_column = 0;
        update();
    }

    int find_text(const String& text, int from_line, int from_column, bool case_sensitive = true) {
        // Search implementation
        return -1;
    }

    void replace_text(const String& find, const String& replace, bool all = false) {}

    void add_breakpoint(int line) {}
    void remove_breakpoint(int line) {}
    void toggle_breakpoint(int line) {}

    virtual std::vector<CompletionItem> get_completions(const String& prefix) { return {}; }
    virtual String get_tooltip_at(int line, int column) { return String(); }
    virtual void on_symbol_clicked(const String& symbol) {}

    void _draw() override {
        draw_line_numbers();
        draw_text();
        draw_caret();
        draw_selection();
        if (m_brace_matching) draw_brace_match();
        if (m_highlight_current_line) draw_current_line_highlight();
    }

    void _gui_input(const Ref<InputEvent>& event) override {
        // Handle keyboard and mouse
    }

protected:
    void initialize_colors() {
        switch (m_theme) {
            case SyntaxTheme::THEME_DARK:
                m_colors[SyntaxTokenType::TOKEN_TEXT] = Color(0.9f, 0.9f, 0.9f);
                m_colors[SyntaxTokenType::TOKEN_KEYWORD] = Color(0.8f, 0.4f, 0.7f);
                m_colors[SyntaxTokenType::TOKEN_TYPE] = Color(0.4f, 0.8f, 0.7f);
                m_colors[SyntaxTokenType::TOKEN_FUNCTION] = Color(0.5f, 0.7f, 1.0f);
                m_colors[SyntaxTokenType::TOKEN_NUMBER] = Color(0.7f, 0.8f, 0.5f);
                m_colors[SyntaxTokenType::TOKEN_STRING] = Color(0.9f, 0.7f, 0.4f);
                m_colors[SyntaxTokenType::TOKEN_COMMENT] = Color(0.5f, 0.6f, 0.5f);
                m_colors[SyntaxTokenType::TOKEN_SYMBOL] = Color(0.9f, 0.9f, 0.9f);
                m_colors[SyntaxTokenType::TOKEN_ERROR] = Color(1.0f, 0.3f, 0.3f);
                break;
            default:
                break;
        }
    }

    void update_lines() {
        m_lines.clear();
        size_t start = 0;
        while (start < m_text.length()) {
            size_t end = m_text.find("\n", start);
            if (end == String::npos) {
                m_lines.push_back(m_text.substr(start));
                break;
            }
            m_lines.push_back(m_text.substr(start, end - start));
            start = end + 1;
        }
        if (m_lines.empty()) m_lines.push_back("");
    }

    void rebuild_text() {
        String result;
        for (size_t i = 0; i < m_lines.size(); ++i) {
            if (i > 0) result += "\n";
            result += m_lines[i];
        }
        m_text = result;
    }

    virtual void retokenize() {
        m_tokens.clear();
        // Tokenization implemented by derived classes
    }

    void draw_line_numbers() {}
    void draw_text() {}
    void draw_caret() {}
    void draw_selection() {}
    void draw_brace_match() {}
    void draw_current_line_highlight() {}
};

// #############################################################################
// ScriptTextEditor - GDScript-specific text editor
// #############################################################################
class ScriptTextEditor : public CodeEditor {
    XTU_GODOT_REGISTER_CLASS(ScriptTextEditor, CodeEditor)

private:
    Ref<Script> m_script;
    std::unordered_map<String, std::vector<CompletionItem>> m_completion_cache;
    bool m_parse_errors = false;
    std::vector<std::pair<int, String>> m_error_messages;

public:
    static StringName get_class_static() { return StringName("ScriptTextEditor"); }

    void set_script(const Ref<Script>& script) {
        m_script = script;
        if (script.is_valid()) {
            set_text(script->get_source_code());
        }
    }

    Ref<Script> get_script() const { return m_script; }

    void save() {
        if (m_script.is_valid()) {
            m_script->set_source_code(get_text());
            ResourceSaver::save(m_script, m_script->get_path());
        }
    }

    std::vector<CompletionItem> get_completions(const String& prefix) override {
        std::vector<CompletionItem> result;
        // Get completions from GDScript analyzer
        return result;
    }

    String get_tooltip_at(int line, int column) override {
        // Get symbol documentation
        return String();
    }

    void on_symbol_clicked(const String& symbol) override {
        // Jump to definition
    }

protected:
    void retokenize() override {
        // GDScript-specific tokenization
    }
};

// #############################################################################
// ScriptEditor - Main script editing window
// #############################################################################
class ScriptEditor : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(ScriptEditor, VBoxContainer)

private:
    static ScriptEditor* s_singleton;
    TabContainer* m_tab_container = nullptr;
    std::vector<ScriptTextEditor*> m_open_editors;
    std::unordered_map<String, ScriptTextEditor*> m_editor_by_path;
    EditorHelp* m_help = nullptr;
    Button* m_save_button = nullptr;
    Button* m_save_all_button = nullptr;
    Button* m_find_button = nullptr;
    Button* m_replace_button = nullptr;
    LineEdit* m_search_box = nullptr;

public:
    static ScriptEditor* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ScriptEditor"); }

    ScriptEditor() {
        s_singleton = this;
        build_ui();
    }

    ~ScriptEditor() { s_singleton = nullptr; }

    void open_script(const String& path) {
        auto it = m_editor_by_path.find(path);
        if (it != m_editor_by_path.end()) {
            m_tab_container->set_current_tab(m_tab_container->get_tab_idx_from_control(it->second));
            return;
        }

        Ref<Script> script = ResourceLoader::load(path);
        if (!script.is_valid()) return;

        ScriptTextEditor* editor = new ScriptTextEditor();
        editor->set_script(script);
        editor->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);

        m_tab_container->add_child(editor);
        m_tab_container->set_tab_title(editor, path.get_file());
        m_open_editors.push_back(editor);
        m_editor_by_path[path] = editor;
    }

    void close_script(const String& path) {
        auto it = m_editor_by_path.find(path);
        if (it == m_editor_by_path.end()) return;

        ScriptTextEditor* editor = it->second;
        m_tab_container->remove_child(editor);
        auto vec_it = std::find(m_open_editors.begin(), m_open_editors.end(), editor);
        if (vec_it != m_open_editors.end()) m_open_editors.erase(vec_it);
        m_editor_by_path.erase(it);
        delete editor;
    }

    void save_current() {
        ScriptTextEditor* editor = get_current_editor();
        if (editor) editor->save();
    }

    void save_all() {
        for (auto* editor : m_open_editors) {
            editor->save();
        }
    }

    ScriptTextEditor* get_current_editor() const {
        int idx = m_tab_container->get_current_tab();
        if (idx >= 0 && idx < static_cast<int>(m_open_editors.size())) {
            return m_open_editors[idx];
        }
        return nullptr;
    }

    void show_help(const String& topic) {
        if (m_help) {
            m_help->show_topic(topic);
        }
    }

    void search_in_files(const String& text) {}
    void replace_in_files(const String& find, const String& replace) {}

private:
    void build_ui() {
        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();

        m_save_button = new Button();
        m_save_button->set_text("Save");
        m_save_button->connect("pressed", this, "save_current");

        m_save_all_button = new Button();
        m_save_all_button->set_text("Save All");
        m_save_all_button->connect("pressed", this, "save_all");

        m_find_button = new Button();
        m_find_button->set_text("Find");

        m_replace_button = new Button();
        m_replace_button->set_text("Replace");

        m_search_box = new LineEdit();
        m_search_box->set_placeholder("Search...");

        toolbar->add_child(m_save_button);
        toolbar->add_child(m_save_all_button);
        toolbar->add_child(new VSeparator());
        toolbar->add_child(m_search_box);
        toolbar->add_child(m_find_button);
        toolbar->add_child(m_replace_button);

        add_child(toolbar);

        // Tab container
        m_tab_container = new TabContainer();
        m_tab_container->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_tab_container->connect("tab_changed", this, "on_tab_changed");
        add_child(m_tab_container);

        // Help panel (collapsible)
        m_help = new EditorHelp();
        m_help->set_visible(false);
        add_child(m_help);
    }

    void on_tab_changed(int tab) {}
};

// #############################################################################
// EditorHelp - Documentation viewer
// #############################################################################
class EditorHelp : public Control {
    XTU_GODOT_REGISTER_CLASS(EditorHelp, Control)

private:
    RichTextLabel* m_content = nullptr;
    LineEdit* m_search = nullptr;
    Tree* m_toc = nullptr;
    std::unordered_map<String, String> m_doc_cache;

public:
    static StringName get_class_static() { return StringName("EditorHelp"); }

    EditorHelp() {
        build_ui();
    }

    void show_topic(const String& topic) {
        String content = get_documentation(topic);
        m_content->set_text(content);
    }

    void search(const String& query) {
        // Search documentation
    }

private:
    void build_ui() {
        HSplitContainer* split = new HSplitContainer();
        split->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        split->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_child(split);

        // Table of contents
        VBoxContainer* left_panel = new VBoxContainer();
        m_search = new LineEdit();
        m_search->set_placeholder("Search...");
        left_panel->add_child(m_search);

        m_toc = new Tree();
        m_toc->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        left_panel->add_child(m_toc);
        split->add_child(left_panel);

        // Content
        m_content = new RichTextLabel();
        m_content->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_content->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_content->set_selection_enabled(true);
        m_content->set_use_bbcode(true);
        split->add_child(m_content);
    }

    String get_documentation(const String& topic) {
        // Fetch from cache or load from disk
        return String();
    }
};

// #############################################################################
// ScriptCreateDialog - New script creation dialog
// #############################################################################
class ScriptCreateDialog : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(ScriptCreateDialog, AcceptDialog)

private:
    LineEdit* m_path_edit = nullptr;
    LineEdit* m_class_name_edit = nullptr;
    OptionButton* m_language_select = nullptr;
    OptionButton* m_template_select = nullptr;
    OptionButton* m_base_class_select = nullptr;
    CheckBox* m_builtin_script = nullptr;
    CheckBox* m_tool_script = nullptr;
    String m_initial_path;

public:
    static StringName get_class_static() { return StringName("ScriptCreateDialog"); }

    ScriptCreateDialog() {
        set_title("Create Script");
        build_ui();
    }

    void set_initial_path(const String& path) { m_initial_path = path; }

    void _ok_pressed() override {
        String path = m_path_edit->get_text();
        String class_name = m_class_name_edit->get_text();
        String language = m_language_select->get_item_text(m_language_select->get_selected());
        String base_class = m_base_class_select->get_item_text(m_base_class_select->get_selected());

        String script_content = generate_script_template(class_name, base_class);
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
        if (file.is_valid()) {
            file->store_string(script_content);
            file->close();
        }

        AcceptDialog::_ok_pressed();
    }

private:
    void build_ui() {
        VBoxContainer* vbox = new VBoxContainer();
        add_child(vbox);

        // Path
        HBoxContainer* path_row = new HBoxContainer();
        path_row->add_child(new Label("Path:"));
        m_path_edit = new LineEdit();
        m_path_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        path_row->add_child(m_path_edit);
        Button* browse_btn = new Button();
        browse_btn->set_text("Browse");
        path_row->add_child(browse_btn);
        vbox->add_child(path_row);

        // Class Name
        HBoxContainer* name_row = new HBoxContainer();
        name_row->add_child(new Label("Class Name:"));
        m_class_name_edit = new LineEdit();
        m_class_name_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        name_row->add_child(m_class_name_edit);
        vbox->add_child(name_row);

        // Language
        HBoxContainer* lang_row = new HBoxContainer();
        lang_row->add_child(new Label("Language:"));
        m_language_select = new OptionButton();
        m_language_select->add_item("GDScript");
        m_language_select->add_item("C#");
        lang_row->add_child(m_language_select);
        vbox->add_child(lang_row);

        // Template
        HBoxContainer* template_row = new HBoxContainer();
        template_row->add_child(new Label("Template:"));
        m_template_select = new OptionButton();
        m_template_select->add_item("Empty");
        m_template_select->add_item("Node");
        m_template_select->add_item("Resource");
        template_row->add_child(m_template_select);
        vbox->add_child(template_row);

        // Base Class
        HBoxContainer* base_row = new HBoxContainer();
        base_row->add_child(new Label("Base Class:"));
        m_base_class_select = new OptionButton();
        m_base_class_select->add_item("Node");
        m_base_class_select->add_item("Node2D");
        m_base_class_select->add_item("Node3D");
        m_base_class_select->add_item("Control");
        m_base_class_select->add_item("Resource");
        m_base_class_select->add_item("RefCounted");
        base_row->add_child(m_base_class_select);
        vbox->add_child(base_row);

        // Options
        m_builtin_script = new CheckBox();
        m_builtin_script->set_text("Built-in Script");
        vbox->add_child(m_builtin_script);

        m_tool_script = new CheckBox();
        m_tool_script->set_text("Tool Script");
        vbox->add_child(m_tool_script);
    }

    String generate_script_template(const String& class_name, const String& base_class) {
        String result = "extends " + base_class + "\n\n";
        result += "class_name " + class_name + "\n\n";
        result += "# Called when the node enters the scene tree for the first time.\n";
        result += "func _ready():\n";
        result += "\tpass # Replace with function body.\n\n";
        result += "# Called every frame. 'delta' is the elapsed time since the previous frame.\n";
        result += "func _process(delta):\n";
        result += "\tpass\n";
        return result;
    }
};

// #############################################################################
// EditorAutoloadSettings - Autoload singleton configuration
// #############################################################################
class EditorAutoloadSettings : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(EditorAutoloadSettings, AcceptDialog)

private:
    Tree* m_autoload_list = nullptr;
    LineEdit* m_name_edit = nullptr;
    LineEdit* m_path_edit = nullptr;
    CheckBox* m_singleton_check = nullptr;
    std::vector<std::pair<String, String>> m_autoloads;

public:
    static StringName get_class_static() { return StringName("EditorAutoloadSettings"); }

    EditorAutoloadSettings() {
        set_title("Autoload");
        build_ui();
    }

    void add_autoload(const String& name, const String& path) {
        m_autoloads.push_back({name, path});
        refresh_list();
    }

    void remove_autoload(int idx) {
        if (idx >= 0 && idx < static_cast<int>(m_autoloads.size())) {
            m_autoloads.erase(m_autoloads.begin() + idx);
            refresh_list();
        }
    }

    std::vector<std::pair<String, String>> get_autoload_list() const {
        return m_autoloads;
    }

private:
    void build_ui() {
        VBoxContainer* vbox = new VBoxContainer();
        add_child(vbox);

        m_autoload_list = new Tree();
        m_autoload_list->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_autoload_list->set_columns(2);
        m_autoload_list->set_column_title(0, "Name");
        m_autoload_list->set_column_title(1, "Path");
        vbox->add_child(m_autoload_list);

        HBoxContainer* add_row = new HBoxContainer();
        m_name_edit = new LineEdit();
        m_name_edit->set_placeholder("Name");
        add_row->add_child(m_name_edit);
        m_path_edit = new LineEdit();
        m_path_edit->set_placeholder("Path");
        m_path_edit->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        add_row->add_child(m_path_edit);
        Button* browse_btn = new Button();
        browse_btn->set_text("...");
        add_row->add_child(browse_btn);
        Button* add_btn = new Button();
        add_btn->set_text("Add");
        add_btn->connect("pressed", this, "on_add_pressed");
        add_row->add_child(add_btn);
        vbox->add_child(add_row);

        m_singleton_check = new CheckBox();
        m_singleton_check->set_text("Enable");
        m_singleton_check->set_pressed(true);
        vbox->add_child(m_singleton_check);
    }

    void on_add_pressed() {
        String name = m_name_edit->get_text();
        String path = m_path_edit->get_text();
        if (!name.empty() && !path.empty()) {
            add_autoload(name, path);
            m_name_edit->clear();
            m_path_edit->clear();
        }
    }

    void refresh_list() {
        m_autoload_list->clear();
        TreeItem* root = m_autoload_list->create_item();
        for (const auto& item : m_autoloads) {
            TreeItem* ti = m_autoload_list->create_item(root);
            ti->set_text(0, item.first);
            ti->set_text(1, item.second);
        }
    }
};

} // namespace editor

// Bring into main namespace
using editor::CodeEditor;
using editor::ScriptEditor;
using editor::ScriptTextEditor;
using editor::EditorHelp;
using editor::ScriptCreateDialog;
using editor::EditorAutoloadSettings;
using editor::SyntaxTheme;
using editor::CompletionTrigger;
using editor::ScriptTemplateType;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_SCRIPT_HPP