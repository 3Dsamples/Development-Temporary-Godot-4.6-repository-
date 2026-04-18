// include/xtu/godot/xwebxr.hpp
// xtensor-unified - WebXR Integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XWEBXR_HPP
#define XTU_GODOT_XWEBXR_HPP

#include <atomic>
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
#include "xtu/godot/xxr.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"

#ifdef XTU_OS_WEB
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace webxr {

// #############################################################################
// Forward declarations
// #############################################################################
class WebXRInterface;
class WebXRSession;
class WebXRFrame;

// #############################################################################
// WebXR session mode
// #############################################################################
enum class WebXRSessionMode : uint8_t {
    SESSION_MODE_INLINE = 0,
    SESSION_MODE_IMMERSIVE_VR = 1,
    SESSION_MODE_IMMERSIVE_AR = 2
};

// #############################################################################
// WebXR reference space type
// #############################################################################
enum class WebXRReferenceSpaceType : uint8_t {
    REFERENCE_SPACE_VIEWER = 0,
    REFERENCE_SPACE_LOCAL = 1,
    REFERENCE_SPACE_LOCAL_FLOOR = 2,
    REFERENCE_SPACE_BOUNDED_FLOOR = 3,
    REFERENCE_SPACE_UNBOUNDED = 4
};

// #############################################################################
// WebXR input source handedness
// #############################################################################
enum class WebXRHandedness : uint8_t {
    HANDEDNESS_NONE = 0,
    HANDEDNESS_LEFT = 1,
    HANDEDNESS_RIGHT = 2
};

// #############################################################################
// WebXR target ray mode
// #############################################################################
enum class WebXRTargetRayMode : uint8_t {
    TARGET_RAY_GAZE = 0,
    TARGET_RAY_TRACKED_POINTER = 1,
    TARGET_RAY_SCREEN = 2
};

// #############################################################################
// WebXR input source
// #############################################################################
struct WebXRInputSource {
    int id = -1;
    WebXRHandedness handedness = WebXRHandedness::HANDEDNESS_NONE;
    WebXRTargetRayMode target_ray_mode = WebXRTargetRayMode::TARGET_RAY_TRACKED_POINTER;
    mat4f grip_transform;
    mat4f target_ray_transform;
    std::vector<String> profiles;
    bool has_grip = false;
    bool has_thumbstick = false;
    bool has_touchpad = false;
    bool has_trigger = false;
    bool has_squeeze = false;
    float trigger_value = 0.0f;
    float squeeze_value = 0.0f;
    vec2f thumbstick_axes;
    vec2f touchpad_axes;
    bool trigger_pressed = false;
    bool squeeze_pressed = false;
    bool thumbstick_pressed = false;
    bool touchpad_pressed = false;
    bool primary_button_pressed = false;
    bool secondary_button_pressed = false;
};

// #############################################################################
// WebXRView - Per-eye view data
// #############################################################################
struct WebXRView {
    int eye = 0;
    mat4f view_transform;
    mat4f projection_matrix;
    vec4f viewport;
};

// #############################################################################
// WebXRInterface - Main WebXR implementation
// #############################################################################
class WebXRInterface : public XRInterface {
    XTU_GODOT_REGISTER_CLASS(WebXRInterface, XRInterface)

private:
    static WebXRInterface* s_singleton;
    
    WebXRSessionMode m_session_mode = WebXRSessionMode::SESSION_MODE_IMMERSIVE_VR;
    WebXRReferenceSpaceType m_reference_space_type = WebXRReferenceSpaceType::REFERENCE_SPACE_LOCAL_FLOOR;
    
    bool m_initialized = false;
    bool m_session_active = false;
    bool m_session_requested = false;
    bool m_session_supported = false;
    bool m_session_ready = false;
    
    std::vector<WebXRView> m_views;
    std::vector<WebXRInputSource> m_input_sources;
    mat4f m_head_transform;
    vec2f m_render_target_size = {1024, 1024};
    vec2f m_display_geometry = {1024, 1024};
    
    String m_required_features;
    String m_optional_features;
    String m_requested_reference_space_types;
    
    mutable std::mutex m_mutex;
    std::function<void(const String&)> m_js_callback;

public:
    static WebXRInterface* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("WebXRInterface"); }

    WebXRInterface() {
        s_singleton = this;
        m_required_features = "local-floor";
        m_optional_features = "hand-tracking";
    }

    ~WebXRInterface() { s_singleton = nullptr; }

    StringName get_name() const override { return StringName("WebXR"); }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

#ifdef XTU_OS_WEB
        // Check WebXR support via JavaScript
        check_webxr_support();
        m_initialized = true;
        return true;
#else
        return false;
#endif
    }

    void uninitialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_session_active) {
            end_session();
        }
        m_initialized = false;
    }

    bool is_initialized() const override { return m_initialized; }

    // #########################################################################
    // Session management
    // #########################################################################
    void set_session_mode(WebXRSessionMode mode) { m_session_mode = mode; }
    WebXRSessionMode get_session_mode() const { return m_session_mode; }

    void set_required_features(const String& features) { m_required_features = features; }
    String get_required_features() const { return m_required_features; }

    void set_optional_features(const String& features) { m_optional_features = features; }
    String get_optional_features() const { return m_optional_features; }

    bool is_session_supported(WebXRSessionMode mode) const {
#ifdef XTU_OS_WEB
        // Query browser support
        return EM_ASM_INT({
            return navigator.xr && navigator.xr.isSessionSupported($0) ? 1 : 0;
        }, mode == WebXRSessionMode::SESSION_MODE_IMMERSIVE_VR ? "immersive-vr" : "immersive-ar");
#else
        return false;
#endif
    }

    void request_session() {
        if (m_session_requested || m_session_active) return;
        m_session_requested = true;

#ifdef XTU_OS_WEB
        String session_mode_str = session_mode_to_string(m_session_mode);
        EM_ASM({
            const mode = UTF8ToString($0);
            const requiredFeatures = UTF8ToString($1).split(',').filter(f => f);
            const optionalFeatures = UTF8ToString($2).split(',').filter(f => f);
            
            navigator.xr.requestSession(mode, {
                requiredFeatures: requiredFeatures,
                optionalFeatures: optionalFeatures
            }).then(session => {
                Module.webxr_session = session;
                Module.webxr_on_session_started();
            }).catch(err => {
                console.error('WebXR session request failed:', err);
                Module.webxr_on_session_failed();
            });
        }, session_mode_str.utf8(), m_required_features.utf8(), m_optional_features.utf8());
#endif
    }

    void end_session() {
        if (!m_session_active) return;

#ifdef XTU_OS_WEB
        EM_ASM({
            if (Module.webxr_session) {
                Module.webxr_session.end();
                Module.webxr_session = null;
            }
        });
#endif
        m_session_active = false;
        m_session_ready = false;
    }

    bool is_session_active() const { return m_session_active; }

    // #########################################################################
    // Reference spaces
    // #########################################################################
    void set_reference_space_type(WebXRReferenceSpaceType type) {
        m_reference_space_type = type;
    }

    WebXRReferenceSpaceType get_reference_space_type() const {
        return m_reference_space_type;
    }

    String get_reference_space_type_string() const {
        switch (m_reference_space_type) {
            case WebXRReferenceSpaceType::REFERENCE_SPACE_VIEWER: return "viewer";
            case WebXRReferenceSpaceType::REFERENCE_SPACE_LOCAL: return "local";
            case WebXRReferenceSpaceType::REFERENCE_SPACE_LOCAL_FLOOR: return "local-floor";
            case WebXRReferenceSpaceType::REFERENCE_SPACE_BOUNDED_FLOOR: return "bounded-floor";
            case WebXRReferenceSpaceType::REFERENCE_SPACE_UNBOUNDED: return "unbounded";
            default: return "local-floor";
        }
    }

    // #########################################################################
    // Rendering
    // #########################################################################
    void set_render_target_size(const vec2i& size) {
        m_render_target_size = vec2f(size.x(), size.y());
    }

    vec2i get_render_target_size() const {
        return vec2i(m_render_target_size.x(), m_render_target_size.y());
    }

    std::vector<WebXRView> get_views() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_views;
    }

    int get_view_count() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int>(m_views.size());
    }

    WebXRView get_view(int idx) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return idx >= 0 && idx < static_cast<int>(m_views.size()) ? m_views[idx] : WebXRView();
    }

    mat4f get_head_transform() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_head_transform;
    }

    // #########################################################################
    // XRInterface overrides
    // #########################################################################
    XRPlayAreaMode get_play_area_mode() const override {
        return XRPlayAreaMode::PLAY_AREA_ROOMSCALE;
    }

    vec2f get_play_area() const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_display_geometry;
    }

    mat4f get_transform_for_view(int view, const mat4f& cam_transform) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (view >= 0 && view < static_cast<int>(m_views.size())) {
            return m_views[view].view_transform;
        }
        return mat4f::identity();
    }

    mat4f get_projection_for_view(int view, float aspect, float near, float far) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (view >= 0 && view < static_cast<int>(m_views.size())) {
            return m_views[view].projection_matrix;
        }
        return perspective(60.0f, aspect, near, far);
    }

    std::vector<Ref<XRTracker>> get_trackers() override {
        std::vector<Ref<XRTracker>> trackers;
        std::lock_guard<std::mutex> lock(m_mutex);

        // Head tracker
        Ref<XRTracker> head;
        head.instance();
        head->set_name("head");
        head->set_type(XRTrackerType::TRACKER_HEAD);
        trackers.push_back(head);

        // Input source trackers
        for (size_t i = 0; i < m_input_sources.size(); ++i) {
            Ref<XRTracker> controller;
            controller.instance();
            controller->set_name(String("controller_") + String::num(static_cast<int>(i)));
            controller->set_type(m_input_sources[i].handedness == WebXRHandedness::HANDEDNESS_LEFT ?
                                 XRTrackerType::TRACKER_CONTROLLER_LEFT :
                                 XRTrackerType::TRACKER_CONTROLLER_RIGHT);
            trackers.push_back(controller);
        }

        return trackers;
    }

    void process() override {
        if (!m_session_active) return;

#ifdef XTU_OS_WEB
        // Update frame data via JavaScript
        EM_ASM({
            if (Module.webxr_session && Module.webxr_frame) {
                Module.webxr_update_frame_data();
            }
        });
#endif
    }

    void pre_render() override {
        if (!m_session_active || !m_session_ready) return;

#ifdef XTU_OS_WEB
        EM_ASM({
            if (Module.webxr_session) {
                Module.webxr_begin_frame();
            }
        });
#endif
    }

    void post_render() override {
        if (!m_session_active || !m_session_ready) return;

#ifdef XTU_OS_WEB
        EM_ASM({
            if (Module.webxr_session && Module.webxr_frame) {
                Module.webxr_end_frame();
            }
        });
#endif
    }

    // #########################################################################
    // Input handling
    // #########################################################################
    std::vector<WebXRInputSource> get_input_sources() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_input_sources;
    }

    WebXRInputSource get_input_source(int id) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& src : m_input_sources) {
            if (src.id == id) return src;
        }
        return WebXRInputSource();
    }

    bool is_button_pressed(int controller_id, const String& button) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& src : m_input_sources) {
            if (src.id == controller_id) {
                if (button == "trigger") return src.trigger_pressed;
                if (button == "squeeze") return src.squeeze_pressed;
                if (button == "thumbstick") return src.thumbstick_pressed;
                if (button == "touchpad") return src.touchpad_pressed;
                if (button == "primary") return src.primary_button_pressed;
                if (button == "secondary") return src.secondary_button_pressed;
            }
        }
        return false;
    }

    vec2f get_axis(int controller_id, const String& axis) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& src : m_input_sources) {
            if (src.id == controller_id) {
                if (axis == "thumbstick") return src.thumbstick_axes;
                if (axis == "touchpad") return src.touchpad_axes;
            }
        }
        return vec2f(0, 0);
    }

    // #########################################################################
    // JavaScript callbacks (called from JS)
    // #########################################################################
    void _on_session_started() {
        m_session_active = true;
        m_session_requested = false;
        emit_signal("session_started");
    }

    void _on_session_failed() {
        m_session_active = false;
        m_session_requested = false;
        emit_signal("session_failed");
    }

    void _on_session_ended() {
        m_session_active = false;
        m_session_ready = false;
        emit_signal("session_ended");
    }

    void _on_session_ready() {
        m_session_ready = true;
        emit_signal("session_ready");
    }

    void _update_views(const String& views_json) {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Parse JSON and update m_views
    }

    void _update_input_sources(const String& sources_json) {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Parse JSON and update m_input_sources
    }

    void _update_head_transform(const std::vector<float>& matrix) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (matrix.size() >= 16) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    m_head_transform[i][j] = matrix[i * 4 + j];
                }
            }
        }
    }

    void _update_display_geometry(float width, float height) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_display_geometry = vec2f(width, height);
    }

private:
#ifdef XTU_OS_WEB
    void check_webxr_support() {
        m_session_supported = EM_ASM_INT({
            return (typeof navigator !== 'undefined' && navigator.xr) ? 1 : 0;
        });
    }

    static String session_mode_to_string(WebXRSessionMode mode) {
        switch (mode) {
            case WebXRSessionMode::SESSION_MODE_IMMERSIVE_VR: return "immersive-vr";
            case WebXRSessionMode::SESSION_MODE_IMMERSIVE_AR: return "immersive-ar";
            default: return "inline";
        }
    }
#endif
};

// #############################################################################
// JavaScript bridge functions (for Emscripten)
// #############################################################################
#ifdef XTU_OS_WEB
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    void webxr_on_session_started() {
        WebXRInterface::get_singleton()->_on_session_started();
    }

    EMSCRIPTEN_KEEPALIVE
    void webxr_on_session_failed() {
        WebXRInterface::get_singleton()->_on_session_failed();
    }

    EMSCRIPTEN_KEEPALIVE
    void webxr_on_session_ended() {
        WebXRInterface::get_singleton()->_on_session_ended();
    }

    EMSCRIPTEN_KEEPALIVE
    void webxr_on_session_ready() {
        WebXRInterface::get_singleton()->_on_session_ready();
    }

    EMSCRIPTEN_KEEPALIVE
    void webxr_update_views(const char* json) {
        WebXRInterface::get_singleton()->_update_views(String(json));
    }

    EMSCRIPTEN_KEEPALIVE
    void webxr_update_input_sources(const char* json) {
        WebXRInterface::get_singleton()->_update_input_sources(String(json));
    }

    EMSCRIPTEN_KEEPALIVE
    void webxr_update_head_transform(float* matrix) {
        std::vector<float> vec(matrix, matrix + 16);
        WebXRInterface::get_singleton()->_update_head_transform(vec);
    }

    EMSCRIPTEN_KEEPALIVE
    void webxr_update_display_geometry(float width, float height) {
        WebXRInterface::get_singleton()->_update_display_geometry(width, height);
    }
}
#endif

} // namespace webxr

// Bring into main namespace
using webxr::WebXRInterface;
using webxr::WebXRSessionMode;
using webxr::WebXRReferenceSpaceType;
using webxr::WebXRHandedness;
using webxr::WebXRTargetRayMode;
using webxr::WebXRInputSource;
using webxr::WebXRView;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XWEBXR_HPP