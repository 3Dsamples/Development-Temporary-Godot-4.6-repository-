// include/xtu/godot/xeditor_animation.hpp
// xtensor-unified - Editor animation tools for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_ANIMATION_HPP
#define XTU_GODOT_XEDITOR_ANIMATION_HPP

#include <algorithm>
#include <cstddef>
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
#include "xtu/godot/xanimation.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/interp/xinterp.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class AnimationTrackEditor;
class AnimationBezierEditor;
class AnimationPlayerEditor;
class AnimationKeyEditor;
class AnimationCurveEdit;

// #############################################################################
// Keyframe edit mode
// #############################################################################
enum class KeyframeEditMode : uint8_t {
    EDIT_MODE_NONE = 0,
    EDIT_MODE_MOVE = 1,
    EDIT_MODE_SCALE = 2,
    EDIT_MODE_ROTATE = 3,
    EDIT_MODE_TANGENT = 4
};

// #############################################################################
// Timeline snap mode
// #############################################################################
enum class TimelineSnapMode : uint8_t {
    SNAP_NONE = 0,
    SNAP_SECONDS = 1,
    SNAP_FRAMES = 2,
    SNAP_MARKERS = 3
};

// #############################################################################
// Track editor types
// #############################################################################
enum class TrackEditorType : uint8_t {
    EDITOR_VALUE = 0,
    EDITOR_TRANSFORM_2D = 1,
    EDITOR_TRANSFORM_3D = 2,
    EDITOR_BEZIER = 3,
    EDITOR_AUDIO = 4,
    EDITOR_METHOD = 5,
    EDITOR_BLEND_SHAPE = 6
};

// #############################################################################
// Keyframe selection info
// #############################################################################
struct KeyframeSelection {
    int track_idx = -1;
    int key_idx = -1;
    float time = 0.0f;
    Variant value;
    bool selected = false;
};

// #############################################################################
// AnimationTrackEditor - Main track editing widget
// #############################################################################
class AnimationTrackEditor : public Control {
    XTU_GODOT_REGISTER_CLASS(AnimationTrackEditor, Control)

private:
    Ref<Animation> m_animation;
    AnimationPlayer* m_player = nullptr;
    std::vector<std::unique_ptr<AnimationTrackEditorPlugin>> m_plugins;
    std::vector<KeyframeSelection> m_selected_keys;
    KeyframeEditMode m_edit_mode = KeyframeEditMode::EDIT_MODE_NONE;
    TimelineSnapMode m_snap_mode = TimelineSnapMode::SNAP_NONE;
    float m_snap_step = 0.1f;
    float m_timeline_position = 0.0f;
    float m_timeline_zoom = 1.0f;
    float m_timeline_scroll = 0.0f;
    bool m_playing = false;
    bool m_loop = true;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("AnimationTrackEditor"); }

    AnimationTrackEditor() {
        register_default_plugins();
    }

    void set_animation(const Ref<Animation>& anim) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_animation = anim;
        update_tracks();
    }

    Ref<Animation> get_animation() const { return m_animation; }

    void set_player(AnimationPlayer* player) { m_player = player; }
    AnimationPlayer* get_player() const { return m_player; }

    void set_timeline_position(float pos) {
        m_timeline_position = pos;
        update();
        if (m_player && m_playing) {
            m_player->seek(pos, true);
        }
    }

    float get_timeline_position() const { return m_timeline_position; }

    void set_zoom(float zoom) {
        m_timeline_zoom = std::clamp(zoom, 0.1f, 10.0f);
        update();
    }

    float get_zoom() const { return m_timeline_zoom; }

    void set_snap_mode(TimelineSnapMode mode) { m_snap_mode = mode; }
    TimelineSnapMode get_snap_mode() const { return m_snap_mode; }

    void set_snap_step(float step) { m_snap_step = step; }
    float get_snap_step() const { return m_snap_step; }

    void add_track(AnimationTrackType type, int at_pos = -1) {
        if (!m_animation.is_valid()) return;
        int idx = m_animation->add_track(type, at_pos);
        update_tracks();
        emit_signal("track_added", idx);
    }

    void remove_track(int idx) {
        if (!m_animation.is_valid()) return;
        m_animation->remove_track(idx);
        update_tracks();
        emit_signal("track_removed", idx);
    }

    void insert_key(int track_idx, float time, const Variant& value) {
        if (!m_animation.is_valid()) return;
        AnimationTrack* track = m_animation->get_track(track_idx);
        if (!track) return;
        track->insert_key(time, value);
        update();
        emit_signal("key_inserted", track_idx, time);
    }

    void duplicate_keys() {
        // Duplicate selected keys
    }

    void delete_selected_keys() {
        if (!m_animation.is_valid()) return;
        for (const auto& sel : m_selected_keys) {
            if (sel.track_idx >= 0 && sel.key_idx >= 0) {
                AnimationTrack* track = m_animation->get_track(sel.track_idx);
                if (track) track->remove_key(sel.key_idx);
            }
        }
        m_selected_keys.clear();
        update();
    }

    void select_all_keys() {
        m_selected_keys.clear();
        if (!m_animation.is_valid()) return;
        for (int t = 0; t < m_animation->get_track_count(); ++t) {
            AnimationTrack* track = m_animation->get_track(t);
            for (int k = 0; k < static_cast<int>(track->get_key_count()); ++k) {
                KeyframeSelection sel;
                sel.track_idx = t;
                sel.key_idx = k;
                sel.time = track->get_key(k).time;
                sel.value = track->get_key(k).value;
                sel.selected = true;
                m_selected_keys.push_back(sel);
            }
        }
        update();
    }

    void copy_keys() {}
    void paste_keys() {}

    void play() {
        if (m_player) {
            m_player->play();
            m_playing = true;
        }
    }

    void stop() {
        if (m_player) {
            m_player->stop();
            m_playing = false;
        }
    }

    bool is_playing() const { return m_playing; }

    void set_loop(bool loop) {
        m_loop = loop;
        if (m_animation.is_valid()) {
            m_animation->set_loop_mode(loop ? AnimationLoopMode::LINEAR : AnimationLoopMode::NONE);
        }
    }

    bool get_loop() const { return m_loop; }

    void _draw() override {
        if (!m_animation.is_valid()) return;
        draw_timeline();
        draw_tracks();
        draw_keyframes();
        draw_cursor();
    }

    void _gui_input(const Ref<InputEvent>& event) override {
        // Handle mouse/keyboard for keyframe editing
    }

private:
    void register_default_plugins() {
        // Register built-in track editors
    }

    void update_tracks() {
        // Refresh track list
    }

    void draw_timeline() {
        // Draw time ruler
    }

    void draw_tracks() {
        // Draw track headers
    }

    void draw_keyframes() {
        // Draw keyframe diamonds
    }

    void draw_cursor() {
        // Draw playhead cursor
    }

    float snap_time(float time) const {
        switch (m_snap_mode) {
            case TimelineSnapMode::SNAP_SECONDS:
                return std::round(time / m_snap_step) * m_snap_step;
            case TimelineSnapMode::SNAP_FRAMES:
                return std::round(time * 60.0f) / 60.0f;
            default:
                return time;
        }
    }
};

// #############################################################################
// AnimationTrackEditorPlugin - Base for track-specific editors
// #############################################################################
class AnimationTrackEditorPlugin : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(AnimationTrackEditorPlugin, RefCounted)

public:
    static StringName get_class_static() { return StringName("AnimationTrackEditorPlugin"); }

    virtual TrackEditorType get_type() const = 0;
    virtual String get_name() const = 0;
    virtual bool handles_track(const AnimationTrack* track) const = 0;
    virtual Control* create_editor(AnimationTrackEditor* editor, int track_idx) = 0;
    virtual void draw_track_background(CanvasItem* canvas, const Rect2& rect, AnimationTrack* track, float zoom, float scroll) {}
    virtual void draw_keyframe(CanvasItem* canvas, const Rect2& rect, const AnimationKey<Variant>& key, bool selected) {}
    virtual Variant interpolate_value(const AnimationTrack* track, float time) const { return Variant(); }
};

// #############################################################################
// AnimationBezierEditor - Bezier curve editor for property tracks
// #############################################################################
class AnimationBezierEditor : public Control {
    XTU_GODOT_REGISTER_CLASS(AnimationBezierEditor, Control)

private:
    Ref<Animation> m_animation;
    int m_track_idx = -1;
    std::vector<vec2f> m_curve_points;
    std::vector<vec2f> m_tangent_in;
    std::vector<vec2f> m_tangent_out;
    int m_selected_point = -1;
    int m_hovered_point = -1;
    vec2f m_view_offset;
    float m_view_zoom = 1.0f;
    bool m_show_tangents = true;
    bool m_snap_enabled = false;
    vec2f m_snap_step = {0.1f, 0.1f};

public:
    static StringName get_class_static() { return StringName("AnimationBezierEditor"); }

    void set_animation(const Ref<Animation>& anim, int track_idx) {
        m_animation = anim;
        m_track_idx = track_idx;
        rebuild_curve();
        update();
    }

    void set_show_tangents(bool show) { m_show_tangents = show; update(); }
    bool get_show_tangents() const { return m_show_tangents; }

    void set_snap_enabled(bool enabled) { m_snap_enabled = enabled; }
    bool is_snap_enabled() const { return m_snap_enabled; }

    void set_view_offset(const vec2f& offset) { m_view_offset = offset; update(); }
    vec2f get_view_offset() const { return m_view_offset; }

    void set_view_zoom(float zoom) { m_view_zoom = std::clamp(zoom, 0.1f, 10.0f); update(); }
    float get_view_zoom() const { return m_view_zoom; }

    void add_point(const vec2f& pos) {
        if (!m_animation.is_valid() || m_track_idx < 0) return;
        AnimationTrack* track = m_animation->get_track(m_track_idx);
        if (!track) return;
        track->insert_key(pos.x(), pos.y(), AnimationInterpolationType::CUBIC);
        rebuild_curve();
        update();
    }

    void remove_selected_point() {
        if (m_selected_point < 0) return;
        AnimationTrack* track = m_animation->get_track(m_track_idx);
        if (!track) return;
        track->remove_key(m_selected_point);
        m_selected_point = -1;
        rebuild_curve();
        update();
    }

    void _draw() override {
        draw_grid();
        draw_curve();
        if (m_show_tangents) draw_tangents();
        draw_points();
    }

    void _gui_input(const Ref<InputEvent>& event) override {
        // Handle point selection, dragging, tangent editing
    }

private:
    void rebuild_curve() {
        m_curve_points.clear();
        m_tangent_in.clear();
        m_tangent_out.clear();
        if (!m_animation.is_valid() || m_track_idx < 0) return;
        AnimationTrack* track = m_animation->get_track(m_track_idx);
        if (!track) return;
        for (int i = 0; i < static_cast<int>(track->get_key_count()); ++i) {
            const auto& key = track->get_key(i);
            m_curve_points.push_back({key.time, key.value.as<float>()});
            m_tangent_in.push_back({key.time - 0.1f, key.value.as<float>() - key.in_tangent});
            m_tangent_out.push_back({key.time + 0.1f, key.value.as<float>() + key.out_tangent});
        }
    }

    void draw_grid() {}
    void draw_curve() {}
    void draw_tangents() {}
    void draw_points() {}
};

// #############################################################################
// AnimationPlayerEditor - AnimationPlayer node editor
// #############################################################################
class AnimationPlayerEditor : public VBoxContainer {
    XTU_GODOT_REGISTER_CLASS(AnimationPlayerEditor, VBoxContainer)

private:
    AnimationPlayer* m_player = nullptr;
    AnimationTrackEditor* m_track_editor = nullptr;
    Button* m_play_button = nullptr;
    Button* m_stop_button = nullptr;
    Button* m_loop_button = nullptr;
    LineEdit* m_animation_name = nullptr;
    OptionButton* m_animation_list = nullptr;
    HSlider* m_timeline_slider = nullptr;
    Label* m_time_label = nullptr;
    SpinBox* m_zoom_spin = nullptr;
    bool m_updating = false;

public:
    static StringName get_class_static() { return StringName("AnimationPlayerEditor"); }

    AnimationPlayerEditor() {
        build_ui();
    }

    void set_player(AnimationPlayer* player) {
        m_player = player;
        m_track_editor->set_player(player);
        refresh_animation_list();
    }

    AnimationPlayer* get_player() const { return m_player; }

    void refresh_animation_list() {
        if (!m_player) return;
        m_updating = true;
        m_animation_list->clear();
        for (const auto& name : m_player->get_animation_list()) {
            m_animation_list->add_item(name.string());
        }
        StringName current = m_player->get_current_animation();
        if (current) {
            for (int i = 0; i < m_animation_list->get_item_count(); ++i) {
                if (m_animation_list->get_item_text(i) == current.string()) {
                    m_animation_list->select(i);
                    break;
                }
            }
        }
        m_updating = false;
        on_animation_selected(m_animation_list->get_selected());
    }

    void on_animation_selected(int idx) {
        if (m_updating || !m_player) return;
        String name = m_animation_list->get_item_text(idx);
        m_player->set_current_animation(name.c_str());
        Ref<Animation> anim = m_player->get_animation(name.c_str());
        m_track_editor->set_animation(anim);
        m_animation_name->set_text(name);
    }

    void on_play_pressed() {
        if (!m_player) return;
        if (m_player->is_playing()) {
            m_player->stop();
        } else {
            m_player->play();
        }
        update_play_button();
    }

    void on_stop_pressed() {
        if (m_player) m_player->stop();
        update_play_button();
    }

    void on_loop_toggled(bool pressed) {
        if (m_track_editor) m_track_editor->set_loop(pressed);
    }

    void on_timeline_changed(float value) {
        if (m_player) m_player->seek(value);
        if (m_track_editor) m_track_editor->set_timeline_position(value);
        update_time_label();
    }

    void on_zoom_changed(float value) {
        if (m_track_editor) m_track_editor->set_zoom(value);
    }

    void _process(double delta) override {
        if (m_player && m_player->is_playing()) {
            float pos = m_player->get_position();
            m_timeline_slider->set_value(pos);
            if (m_track_editor) m_track_editor->set_timeline_position(pos);
            update_time_label();
        }
        update_play_button();
    }

private:
    void build_ui() {
        // Create UI controls
        m_play_button = new Button();
        m_play_button->set_text("Play");
        m_play_button->connect("pressed", this, "on_play_pressed");

        m_stop_button = new Button();
        m_stop_button->set_text("Stop");
        m_stop_button->connect("pressed", this, "on_stop_pressed");

        m_loop_button = new Button();
        m_loop_button->set_text("Loop");
        m_loop_button->set_toggle_mode(true);
        m_loop_button->connect("toggled", this, "on_loop_toggled");

        m_animation_name = new LineEdit();
        m_animation_list = new OptionButton();
        m_animation_list->connect("item_selected", this, "on_animation_selected");

        m_timeline_slider = new HSlider();
        m_timeline_slider->set_h_size_flags(SIZE_EXPAND | SIZE_FILL);
        m_timeline_slider->connect("value_changed", this, "on_timeline_changed");

        m_time_label = new Label();
        m_time_label->set_text("0.00s");

        m_zoom_spin = new SpinBox();
        m_zoom_spin->set_min(0.1f);
        m_zoom_spin->set_max(10.0f);
        m_zoom_spin->set_step(0.1f);
        m_zoom_spin->set_value(1.0f);
        m_zoom_spin->connect("value_changed", this, "on_zoom_changed");

        m_track_editor = new AnimationTrackEditor();
        m_track_editor->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);

        // Layout
        HBoxContainer* toolbar = new HBoxContainer();
        toolbar->add_child(m_play_button);
        toolbar->add_child(m_stop_button);
        toolbar->add_child(m_loop_button);
        toolbar->add_child(m_animation_name);
        toolbar->add_child(m_animation_list);

        HBoxContainer* timeline_bar = new HBoxContainer();
        timeline_bar->add_child(m_time_label);
        timeline_bar->add_child(m_timeline_slider);
        timeline_bar->add_child(m_zoom_spin);

        add_child(toolbar);
        add_child(timeline_bar);
        add_child(m_track_editor);
    }

    void update_play_button() {
        if (m_player && m_player->is_playing()) {
            m_play_button->set_text("Pause");
        } else {
            m_play_button->set_text("Play");
        }
    }

    void update_time_label() {
        float pos = m_player ? m_player->get_position() : 0.0f;
        float len = m_player ? m_player->get_current_animation_length() : 0.0f;
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f / %.2f", pos, len);
        m_time_label->set_text(buf);
    }
};

// #############################################################################
// AnimationKeyEditor - Popup for editing individual keyframe
// #############################################################################
class AnimationKeyEditor : public AcceptDialog {
    XTU_GODOT_REGISTER_CLASS(AnimationKeyEditor, AcceptDialog)

private:
    AnimationTrack* m_track = nullptr;
    int m_key_idx = -1;
    float m_time = 0.0f;
    Variant m_value;
    AnimationInterpolationType m_interp = AnimationInterpolationType::LINEAR;
    float m_in_tangent = 0.0f;
    float m_out_tangent = 0.0f;

    SpinBox* m_time_spin = nullptr;
    Control* m_value_editor = nullptr;
    OptionButton* m_interp_option = nullptr;
    SpinBox* m_in_tangent_spin = nullptr;
    SpinBox* m_out_tangent_spin = nullptr;

public:
    static StringName get_class_static() { return StringName("AnimationKeyEditor"); }

    AnimationKeyEditor() {
        set_title("Edit Keyframe");
        build_ui();
    }

    void edit_key(AnimationTrack* track, int key_idx) {
        m_track = track;
        m_key_idx = key_idx;
        if (!track || key_idx < 0) return;

        const auto& key = track->get_key(key_idx);
        m_time = key.time;
        m_value = key.value;
        m_interp = key.interpolation;
        m_in_tangent = key.in_tangent;
        m_out_tangent = key.out_tangent;

        m_time_spin->set_value(m_time);
        update_value_editor();
        m_interp_option->select(static_cast<int>(m_interp));
        m_in_tangent_spin->set_value(m_in_tangent);
        m_out_tangent_spin->set_value(m_out_tangent);

        bool show_tangents = (m_interp == AnimationInterpolationType::CUBIC);
        m_in_tangent_spin->set_visible(show_tangents);
        m_out_tangent_spin->set_visible(show_tangents);

        popup_centered();
    }

    void _ok_pressed() override {
        if (!m_track || m_key_idx < 0) return;

        m_time = m_time_spin->get_value();
        m_interp = static_cast<AnimationInterpolationType>(m_interp_option->get_selected());
        m_in_tangent = m_in_tangent_spin->get_value();
        m_out_tangent = m_out_tangent_spin->get_value();

        m_track->remove_key(m_key_idx);
        AnimationKey<Variant> new_key{m_time, m_value, m_interp, m_in_tangent, m_out_tangent};
        m_track->add_key(new_key);

        AcceptDialog::_ok_pressed();
    }

private:
    void build_ui() {
        VBoxContainer* vbox = new VBoxContainer();
        add_child(vbox);

        // Time
        HBoxContainer* time_row = new HBoxContainer();
        time_row->add_child(new Label("Time:"));
        m_time_spin = new SpinBox();
        m_time_spin->set_min(0.0f);
        m_time_spin->set_max(3600.0f);
        m_time_spin->set_step(0.01f);
        time_row->add_child(m_time_spin);
        vbox->add_child(time_row);

        // Value (placeholder - actual depends on type)
        HBoxContainer* value_row = new HBoxContainer();
        value_row->add_child(new Label("Value:"));
        m_value_editor = new LineEdit();
        value_row->add_child(m_value_editor);
        vbox->add_child(value_row);

        // Interpolation
        HBoxContainer* interp_row = new HBoxContainer();
        interp_row->add_child(new Label("Interpolation:"));
        m_interp_option = new OptionButton();
        m_interp_option->add_item("Nearest");
        m_interp_option->add_item("Linear");
        m_interp_option->add_item("Cubic");
        interp_row->add_child(m_interp_option);
        vbox->add_child(interp_row);

        // Tangents
        HBoxContainer* in_tan_row = new HBoxContainer();
        in_tan_row->add_child(new Label("In Tangent:"));
        m_in_tangent_spin = new SpinBox();
        m_in_tangent_spin->set_min(-1000.0f);
        m_in_tangent_spin->set_max(1000.0f);
        m_in_tangent_spin->set_step(0.1f);
        in_tan_row->add_child(m_in_tangent_spin);
        vbox->add_child(in_tan_row);

        HBoxContainer* out_tan_row = new HBoxContainer();
        out_tan_row->add_child(new Label("Out Tangent:"));
        m_out_tangent_spin = new SpinBox();
        m_out_tangent_spin->set_min(-1000.0f);
        m_out_tangent_spin->set_max(1000.0f);
        m_out_tangent_spin->set_step(0.1f);
        out_tan_row->add_child(m_out_tangent_spin);
        vbox->add_child(out_tan_row);
    }

    void update_value_editor() {
        // Update based on variant type
    }
};

} // namespace editor

// Bring into main namespace
using editor::AnimationTrackEditor;
using editor::AnimationBezierEditor;
using editor::AnimationPlayerEditor;
using editor::AnimationKeyEditor;
using editor::AnimationTrackEditorPlugin;
using editor::KeyframeEditMode;
using editor::TimelineSnapMode;
using editor::TrackEditorType;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_ANIMATION_HPP