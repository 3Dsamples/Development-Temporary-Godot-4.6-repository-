// include/xtu/godot/xmobile_vr.hpp
// xtensor-unified - Mobile VR (Cardboard/Daydream) Integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XMOBILE_VR_HPP
#define XTU_GODOT_XMOBILE_VR_HPP

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
#include "xtu/godot/xxr.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/godot/xinput.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace mobile_vr {

// #############################################################################
// Forward declarations
// #############################################################################
class MobileVRInterface;
class MobileVRLensDistortion;
class MobileVRController;

// #############################################################################
// Mobile VR display mode
// #############################################################################
enum class MobileVRDisplayMode : uint8_t {
    MODE_MONO = 0,
    MODE_STEREO = 1
};

// #############################################################################
// Mobile VR lens distortion parameters
// #############################################################################
struct MobileVRLensParams {
    float k1 = 0.22f;      // Primary radial distortion coefficient
    float k2 = 0.26f;      // Secondary radial distortion coefficient
    float screen_width_m = 0.062f;
    float screen_height_m = 0.035f;
    float screen_to_lens_m = 0.039f;
    float inter_lens_m = 0.064f;
    float vertical_alignment_m = 0.035f;
    float tray_to_lens_m = 0.0f;
    vec2f lens_center_offset = {0, 0};
    std::vector<float> inverse_distortion;
};

// #############################################################################
// Mobile VR tracking mode
// #############################################################################
enum class MobileVRTrackingMode : uint8_t {
    TRACKING_NONE = 0,
    TRACKING_ROTATION = 1,
    TRACKING_POSITION = 2
};

// #############################################################################
// Mobile VR controller type
// #############################################################################
enum class MobileVRControllerType : uint8_t {
    CONTROLLER_NONE = 0,
    CONTROLLER_CARDBOARD = 1,
    CONTROLLER_DAYDREAM = 2,
    CONTROLLER_GEAR_VR = 3
};

// #############################################################################
// Mobile VR controller state
// #############################################################################
struct MobileVRControllerState {
    bool connected = false;
    MobileVRControllerType type = MobileVRControllerType::CONTROLLER_NONE;
    mat4f transform;
    bool touch_pressed = false;
    vec2f touch_position;
    bool app_button_pressed = false;
    bool home_button_pressed = false;
    bool click_button_pressed = false;
    bool volume_up_pressed = false;
    bool volume_down_pressed = false;
    vec3f gyroscope;
    vec3f accelerometer;
    int battery_level = 0;
};

// #############################################################################
// MobileVRLensDistortion - Lens distortion correction
// #############################################################################
class MobileVRLensDistortion : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(MobileVRLensDistortion, RefCounted)

private:
    MobileVRLensParams m_params;
    Ref<Texture2D> m_distortion_texture;
    bool m_vignette_enabled = true;
    float m_vignette_size = 0.5f;
    float m_border_size = 0.02f;
    Color m_border_color = {0, 0, 0, 1};

public:
    static StringName get_class_static() { return StringName("MobileVRLensDistortion"); }

    void set_params(const MobileVRLensParams& params) { m_params = params; }
    const MobileVRLensParams& get_params() const { return m_params; }

    void set_k1(float k1) { m_params.k1 = k1; }
    float get_k1() const { return m_params.k1; }

    void set_k2(float k2) { m_params.k2 = k2; }
    float get_k2() const { return m_params.k2; }

    void set_screen_width_m(float width) { m_params.screen_width_m = width; }
    float get_screen_width_m() const { return m_params.screen_width_m; }

    void set_screen_height_m(float height) { m_params.screen_height_m = height; }
    float get_screen_height_m() const { return m_params.screen_height_m; }

    void set_inter_lens_m(float distance) { m_params.inter_lens_m = distance; }
    float get_inter_lens_m() const { return m_params.inter_lens_m; }

    void set_vignette_enabled(bool enabled) { m_vignette_enabled = enabled; }
    bool is_vignette_enabled() const { return m_vignette_enabled; }

    void set_vignette_size(float size) { m_vignette_size = size; }
    float get_vignette_size() const { return m_vignette_size; }

    vec2f distort(const vec2f& uv, bool left_eye) const {
        float lens_shift = left_eye ? -m_params.inter_lens_m * 0.5f : m_params.inter_lens_m * 0.5f;
        vec2f centered = uv - vec2f(0.5f + lens_shift / m_params.screen_width_m, 0.5f);
        float r2 = centered.length_sq();
        float distortion = 1.0f + m_params.k1 * r2 + m_params.k2 * r2 * r2;
        return vec2f(0.5f) + centered * distortion;
    }

    vec2f undistort(const vec2f& uv, bool left_eye) const {
        // Newton-Raphson iteration to find inverse distortion
        vec2f result = uv;
        for (int i = 0; i < 5; ++i) {
            vec2f distorted = distort(result, left_eye);
            vec2f error = uv - distorted;
            result += error;
        }
        return result;
    }

    float get_vignette(const vec2f& uv) const {
        if (!m_vignette_enabled) return 1.0f;
        vec2f centered = uv - vec2f(0.5f);
        float dist = centered.length() * 2.0f;
        return 1.0f - std::clamp((dist - m_vignette_size) / (1.0f - m_vignette_size), 0.0f, 1.0f);
    }
};

// #############################################################################
// MobileVRInterface - Main mobile VR implementation
// #############################################################################
class MobileVRInterface : public XRInterface {
    XTU_GODOT_REGISTER_CLASS(MobileVRInterface, XRInterface)

private:
    static MobileVRInterface* s_singleton;

    bool m_initialized = false;
    MobileVRDisplayMode m_display_mode = MobileVRDisplayMode::MODE_STEREO;
    MobileVRTrackingMode m_tracking_mode = MobileVRTrackingMode::TRACKING_ROTATION;

    Ref<MobileVRLensDistortion> m_lens_distortion;
    MobileVRControllerState m_controller_state;

    mat4f m_head_transform;
    mat4f m_eye_transforms[2];
    mat4f m_eye_projections[2];
    vec4f m_eye_viewports[2];
    vec2i m_render_target_size = {1920, 1080};

    vec3f m_gyroscope;
    vec3f m_accelerometer;
    vec3f m_magnetometer;
    quatf m_head_rotation;
    vec3f m_head_position;

    float m_neck_horizontal_offset = 0.08f;
    float m_neck_vertical_offset = 0.075f;
    float m_eye_depth = 0.08f;

    bool m_low_persistence = false;
    float m_framebuffer_scale = 1.0f;
    float m_oversample = 1.5f;

    mutable std::mutex m_mutex;

public:
    static MobileVRInterface* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("MobileVRInterface"); }

    MobileVRInterface() {
        s_singleton = this;
        m_lens_distortion.instance();
    }

    ~MobileVRInterface() { s_singleton = nullptr; }

    StringName get_name() const override { return StringName("MobileVR"); }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

        // Initialize sensors
#ifdef XTU_OS_ANDROID
        // Enable gyroscope and accelerometer
#endif
#ifdef XTU_OS_IOS
        // Enable CoreMotion
#endif
        m_initialized = true;
        return true;
    }

    void uninitialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_initialized = false;
    }

    bool is_initialized() const override { return m_initialized; }

    // #########################################################################
    // Configuration
    // #########################################################################
    void set_display_mode(MobileVRDisplayMode mode) { m_display_mode = mode; }
    MobileVRDisplayMode get_display_mode() const { return m_display_mode; }

    void set_tracking_mode(MobileVRTrackingMode mode) { m_tracking_mode = mode; }
    MobileVRTrackingMode get_tracking_mode() const { return m_tracking_mode; }

    Ref<MobileVRLensDistortion> get_lens_distortion() const { return m_lens_distortion; }

    void set_neck_horizontal_offset(float offset) { m_neck_horizontal_offset = offset; }
    float get_neck_horizontal_offset() const { return m_neck_horizontal_offset; }

    void set_neck_vertical_offset(float offset) { m_neck_vertical_offset = offset; }
    float get_neck_vertical_offset() const { return m_neck_vertical_offset; }

    void set_eye_depth(float depth) { m_eye_depth = depth; }
    float get_eye_depth() const { return m_eye_depth; }

    void set_framebuffer_scale(float scale) { m_framebuffer_scale = scale; }
    float get_framebuffer_scale() const { return m_framebuffer_scale; }

    void set_oversample(float oversample) { m_oversample = std::max(1.0f, oversample); }
    float get_oversample() const { return m_oversample; }

    // #########################################################################
    // Sensor data
    // #########################################################################
    vec3f get_gyroscope() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_gyroscope;
    }

    vec3f get_accelerometer() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_accelerometer;
    }

    vec3f get_magnetometer() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_magnetometer;
    }

    // #########################################################################
    // Head tracking
    // #########################################################################
    void recenter() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_head_rotation = quatf();
        m_head_position = vec3f(0);
        update_head_transform();
    }

    quatf get_head_rotation() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_head_rotation;
    }

    vec3f get_head_position() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_head_position;
    }

    // #########################################################################
    // Controller
    // #########################################################################
    MobileVRControllerState get_controller_state() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_controller_state;
    }

    bool is_controller_connected() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_controller_state.connected;
    }

    mat4f get_controller_transform() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_controller_state.transform;
    }

    // #########################################################################
    // Rendering
    // #########################################################################
    void set_render_target_size(const vec2i& size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_render_target_size = size;
        update_eye_params();
    }

    vec2i get_render_target_size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_render_target_size;
    }

    vec2i get_eye_render_target_size() const {
        vec2i base = get_render_target_size();
        base.x() = static_cast<int>(base.x() * m_framebuffer_scale * m_oversample);
        base.y() = static_cast<int>(base.y() * m_framebuffer_scale * m_oversample);
        return base;
    }

    // #########################################################################
    // XRInterface overrides
    // #########################################################################
    XRPlayAreaMode get_play_area_mode() const override {
        return m_tracking_mode == MobileVRTrackingMode::TRACKING_POSITION ?
               XRPlayAreaMode::PLAY_AREA_ROOMSCALE : XRPlayAreaMode::PLAY_AREA_3DOF;
    }

    vec2f get_play_area() const override { return vec2f(2, 2); }

    mat4f get_transform_for_view(int view, const mat4f& cam_transform) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (view >= 0 && view < 2) {
            return m_eye_transforms[view];
        }
        return mat4f::identity();
    }

    mat4f get_projection_for_view(int view, float aspect, float near, float far) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (view >= 0 && view < 2) {
            return m_eye_projections[view];
        }
        return perspective(60.0f, aspect, near, far);
    }

    std::vector<Ref<XRTracker>> get_trackers() override {
        std::vector<Ref<XRTracker>> trackers;

        Ref<XRTracker> head;
        head.instance();
        head->set_name("head");
        head->set_type(XRTrackerType::TRACKER_HEAD);
        trackers.push_back(head);

        if (m_controller_state.connected) {
            Ref<XRTracker> controller;
            controller.instance();
            controller->set_name("controller");
            controller->set_type(XRTrackerType::TRACKER_CONTROLLER_RIGHT);
            trackers.push_back(controller);
        }

        return trackers;
    }

    void process() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        update_sensors();
        update_head_transform();
        update_eye_params();
    }

    // #########################################################################
    // Distortion rendering
    // #########################################################################
    void render_distorted(RID left_eye_texture, RID right_eye_texture, RID target) {
        // Apply barrel distortion and composite to screen
    }

private:
    void update_sensors() {
#ifdef XTU_OS_ANDROID
        // Read gyroscope and accelerometer via JNI
#endif
#ifdef XTU_OS_IOS
        // Read CoreMotion data
#endif
        // Integrate gyroscope for head rotation
        float dt = 0.016f; // Assume 60fps
        vec3f delta = m_gyroscope * dt;
        m_head_rotation = m_head_rotation * quatf(delta.x(), delta.y(), delta.z(), 0).normalized();
    }

    void update_head_transform() {
        mat4f rot = rotate(mat4f::identity(), m_head_rotation);
        mat4f trans = translate(mat4f::identity(), m_head_position);
        m_head_transform = trans * rot;
    }

    void update_eye_params() {
        const MobileVRLensParams& p = m_lens_distortion->get_params();
        float aspect = static_cast<float>(m_render_target_size.x()) / m_render_target_size.y();
        float fov = 2.0f * std::atan(p.screen_width_m / (2.0f * p.screen_to_lens_m));

        mat4f base_proj = perspective(fov, aspect, 0.1f, 100.0f);

        float ipd = p.inter_lens_m;
        vec3f head_pos = m_head_position;
        vec3f neck_offset(0, m_neck_vertical_offset, m_eye_depth);
        vec3f left_eye_pos = head_pos + neck_offset + vec3f(-ipd * 0.5f, 0, 0);
        vec3f right_eye_pos = head_pos + neck_offset + vec3f(ipd * 0.5f, 0, 0);

        mat4f head_rot = rotate(mat4f::identity(), m_head_rotation);
        m_eye_transforms[0] = translate(mat4f::identity(), left_eye_pos) * head_rot;
        m_eye_transforms[1] = translate(mat4f::identity(), right_eye_pos) * head_rot;

        m_eye_projections[0] = base_proj;
        m_eye_projections[1] = base_proj;

        m_eye_viewports[0] = vec4f(0, 0, 0.5f, 1.0f);
        m_eye_viewports[1] = vec4f(0.5f, 0, 0.5f, 1.0f);
    }

    void update_controller() {
        // Poll Daydream/Cardboard controller via platform APIs
    }
};

} // namespace mobile_vr

// Bring into main namespace
using mobile_vr::MobileVRInterface;
using mobile_vr::MobileVRLensDistortion;
using mobile_vr::MobileVRDisplayMode;
using mobile_vr::MobileVRTrackingMode;
using mobile_vr::MobileVRControllerType;
using mobile_vr::MobileVRLensParams;
using mobile_vr::MobileVRControllerState;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XMOBILE_VR_HPP