// include/xtu/godot/xopenxr.hpp
// xtensor-unified - OpenXR integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XOPENXR_HPP
#define XTU_GODOT_XOPENXR_HPP

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
#include "xtu/godot/xinput.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"

#ifdef XTU_USE_OPENXR
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class OpenXRInterface;
class OpenXRActionMap;
class OpenXRHandTracking;
class OpenXRExtensionWrapper;

// #############################################################################
// OpenXR session state
// #############################################################################
enum class OpenXRSessionState : uint8_t {
    STATE_IDLE = 0,
    STATE_READY = 1,
    STATE_SYNCHRONIZED = 2,
    STATE_VISIBLE = 3,
    STATE_FOCUSED = 4,
    STATE_STOPPING = 5,
    STATE_EXITING = 6,
    STATE_LOSS_PENDING = 7
};

// #############################################################################
// OpenXR view configuration type
// #############################################################################
enum class OpenXRViewConfigType : uint8_t {
    VIEW_CONFIG_MONO = 0,
    VIEW_CONFIG_STEREO = 1
};

// #############################################################################
// OpenXR action type
// #############################################################################
enum class OpenXRActionType : uint8_t {
    ACTION_BOOLEAN = 0,
    ACTION_FLOAT = 1,
    ACTION_VECTOR2 = 2,
    ACTION_POSE = 3,
    ACTION_VIBRATION = 4
};

// #############################################################################
// OpenXR hand joint
// #############################################################################
enum class OpenXRHandJoint : uint8_t {
    JOINT_PALM = 0,
    JOINT_WRIST = 1,
    JOINT_THUMB_METACARPAL = 2,
    JOINT_THUMB_PROXIMAL = 3,
    JOINT_THUMB_DISTAL = 4,
    JOINT_THUMB_TIP = 5,
    JOINT_INDEX_METACARPAL = 6,
    JOINT_INDEX_PROXIMAL = 7,
    JOINT_INDEX_INTERMEDIATE = 8,
    JOINT_INDEX_DISTAL = 9,
    JOINT_INDEX_TIP = 10,
    JOINT_MIDDLE_METACARPAL = 11,
    JOINT_MIDDLE_PROXIMAL = 12,
    JOINT_MIDDLE_INTERMEDIATE = 13,
    JOINT_MIDDLE_DISTAL = 14,
    JOINT_MIDDLE_TIP = 15,
    JOINT_RING_METACARPAL = 16,
    JOINT_RING_PROXIMAL = 17,
    JOINT_RING_INTERMEDIATE = 18,
    JOINT_RING_DISTAL = 19,
    JOINT_RING_TIP = 20,
    JOINT_PINKY_METACARPAL = 21,
    JOINT_PINKY_PROXIMAL = 22,
    JOINT_PINKY_INTERMEDIATE = 23,
    JOINT_PINKY_DISTAL = 24,
    JOINT_PINKY_TIP = 25,
    JOINT_MAX = 26
};

// #############################################################################
// OpenXR action binding
// #############################################################################
struct OpenXRActionBinding {
    String name;
    String localized_name;
    OpenXRActionType type = OpenXRActionType::ACTION_BOOLEAN;
    std::vector<String> paths;
    bool active = false;
};

// #############################################################################
// OpenXRActionMap - Action binding and state management
// #############################################################################
class OpenXRActionMap : public Resource {
    XTU_GODOT_REGISTER_CLASS(OpenXRActionMap, Resource)

private:
    std::vector<OpenXRActionBinding> m_actions;
    std::unordered_map<String, size_t> m_action_index;
    std::unordered_map<String, Variant> m_action_states;
    mutable std::mutex m_mutex;
#ifdef XTU_USE_OPENXR
    XrActionSet m_action_set = XR_NULL_HANDLE;
#endif

public:
    static StringName get_class_static() { return StringName("OpenXRActionMap"); }

    void add_action(const OpenXRActionBinding& action) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_action_index[action.name] = m_actions.size();
        m_actions.push_back(action);
    }

    void remove_action(const String& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_action_index.find(name);
        if (it != m_action_index.end()) {
            m_actions.erase(m_actions.begin() + it->second);
            rebuild_index();
        }
    }

    bool has_action(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_action_index.find(name) != m_action_index.end();
    }

    void set_action_state(const String& name, const Variant& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_action_states[name] = value;
    }

    Variant get_action_state(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_action_states.find(name);
        return it != m_action_states.end() ? it->second : Variant();
    }

    float get_action_float(const String& name) const {
        return get_action_state(name).as<float>();
    }

    bool get_action_bool(const String& name) const {
        return get_action_state(name).as<bool>();
    }

    vec2f get_action_vec2(const String& name) const {
        return get_action_state(name).as<vec2f>();
    }

    mat4f get_action_pose(const String& name) const {
        return get_action_state(name).as<mat4f>();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_actions.clear();
        m_action_index.clear();
        m_action_states.clear();
    }

    std::vector<String> get_action_names() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> names;
        for (const auto& a : m_actions) names.push_back(a.name);
        return names;
    }

#ifdef XTU_USE_OPENXR
    void set_action_set(XrActionSet set) { m_action_set = set; }
    XrActionSet get_action_set() const { return m_action_set; }
#endif

private:
    void rebuild_index() {
        m_action_index.clear();
        for (size_t i = 0; i < m_actions.size(); ++i) {
            m_action_index[m_actions[i].name] = i;
        }
    }
};

// #############################################################################
// OpenXRHandTracking - Hand joint tracking
// #############################################################################
class OpenXRHandTracking : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(OpenXRHandTracking, RefCounted)

private:
    std::array<mat4f, static_cast<size_t>(OpenXRHandJoint::JOINT_MAX)> m_joint_transforms;
    std::array<vec3f, static_cast<size_t>(OpenXRHandJoint::JOINT_MAX)> m_joint_velocities;
    std::array<float, static_cast<size_t>(OpenXRHandJoint::JOINT_MAX)> m_joint_radii;
    bool m_is_active = false;
    bool m_is_left_hand = false;
    std::mutex m_mutex;
#ifdef XTU_USE_OPENXR
    XrHandTrackerEXT m_hand_tracker = XR_NULL_HANDLE;
#endif

public:
    static StringName get_class_static() { return StringName("OpenXRHandTracking"); }

    void set_active(bool active) { m_is_active = active; }
    bool is_active() const { return m_is_active; }

    void set_left_hand(bool left) { m_is_left_hand = left; }
    bool is_left_hand() const { return m_is_left_hand; }

    mat4f get_joint_transform(OpenXRHandJoint joint) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_joint_transforms[static_cast<size_t>(joint)];
    }

    vec3f get_joint_velocity(OpenXRHandJoint joint) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_joint_velocities[static_cast<size_t>(joint)];
    }

    float get_joint_radius(OpenXRHandJoint joint) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_joint_radii[static_cast<size_t>(joint)];
    }

    void update_joint_transforms(const std::array<mat4f, static_cast<size_t>(OpenXRHandJoint::JOINT_MAX)>& transforms) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_joint_transforms = transforms;
    }

    // Gesture detection
    bool is_pinching(float threshold = 0.02f) const {
        vec3f thumb_tip = get_joint_transform(OpenXRHandJoint::JOINT_THUMB_TIP).get_origin();
        vec3f index_tip = get_joint_transform(OpenXRHandJoint::JOINT_INDEX_TIP).get_origin();
        return (thumb_tip - index_tip).length() < threshold;
    }

    bool is_grabbing(float curl_threshold = 0.7f) const {
        // Simplified grab detection based on finger curl
        return get_finger_curl() > curl_threshold;
    }

    float get_finger_curl() const {
        float curl = 0.0f;
        for (int i = static_cast<int>(OpenXRHandJoint::JOINT_INDEX_METACARPAL);
             i <= static_cast<int>(OpenXRHandJoint::JOINT_PINKY_TIP); ++i) {
            // Compute curl based on joint angles
        }
        return curl;
    }

#ifdef XTU_USE_OPENXR
    void set_hand_tracker(XrHandTrackerEXT tracker) { m_hand_tracker = tracker; }
    XrHandTrackerEXT get_hand_tracker() const { return m_hand_tracker; }
#endif
};

// #############################################################################
// OpenXRInterface - Main OpenXR implementation
// #############################################################################
class OpenXRInterface : public XRInterface {
    XTU_GODOT_REGISTER_CLASS(OpenXRInterface, XRInterface)

private:
    static OpenXRInterface* s_singleton;
    
#ifdef XTU_USE_OPENXR
    XrInstance m_instance = XR_NULL_HANDLE;
    XrSystemId m_system_id = XR_NULL_SYSTEM_ID;
    XrSession m_session = XR_NULL_HANDLE;
    XrSessionState m_session_state = XR_SESSION_STATE_UNKNOWN;
    XrSpace m_local_space = XR_NULL_HANDLE;
    XrSpace m_stage_space = XR_NULL_HANDLE;
    XrSpace m_view_space = XR_NULL_HANDLE;
    XrViewConfigurationType m_view_config_type = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    std::vector<XrView> m_views;
    std::vector<XrViewConfigurationView> m_config_views;
    std::vector<XrCompositionLayerProjectionView> m_projection_views;
    std::vector<int64_t> m_swapchain_formats;
#endif

    bool m_initialized = false;
    bool m_running = false;
    OpenXRSessionState m_state = OpenXRSessionState::STATE_IDLE;
    Ref<OpenXRActionMap> m_action_map;
    std::vector<Ref<OpenXRHandTracking>> m_hand_trackers;
    XRPlayAreaMode m_play_area_mode = XRPlayAreaMode::PLAY_AREA_ROOMSCALE;
    vec2f m_play_area_size = {2, 2};
    String m_runtime_name;
    String m_runtime_version;
    std::mutex m_mutex;

public:
    static OpenXRInterface* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("OpenXRInterface"); }

    OpenXRInterface() {
        s_singleton = this;
        m_action_map.instance();
        initialize_hand_trackers();
    }

    ~OpenXRInterface() {
        shutdown();
        s_singleton = nullptr;
    }

    StringName get_name() const override { return StringName("OpenXR"); }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

#ifdef XTU_USE_OPENXR
        if (!create_instance()) return false;
        if (!get_system_id()) return false;
        if (!create_session()) return false;
        if (!create_spaces()) return false;
        if (!create_action_map()) return false;
#endif

        m_initialized = true;
        return true;
    }

    void uninitialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_OPENXR
        if (m_session) xrDestroySession(m_session);
        if (m_instance) xrDestroyInstance(m_instance);
        m_session = XR_NULL_HANDLE;
        m_instance = XR_NULL_HANDLE;
#endif
        m_initialized = false;
        m_running = false;
    }

    bool is_initialized() const override { return m_initialized; }

    void process() override {
        if (!m_initialized || !m_running) return;
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_OPENXR
        poll_events();
        if (m_session_state >= XR_SESSION_STATE_READY) {
            update_action_states();
            update_hand_tracking();
        }
#endif
    }

    void pre_render() override {
        if (!m_initialized || !m_running) return;
#ifdef XTU_USE_OPENXR
        std::lock_guard<std::mutex> lock(m_mutex);
        XrFrameWaitInfo frame_wait_info = {XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState frame_state = {XR_TYPE_FRAME_STATE};
        xrWaitFrame(m_session, &frame_wait_info, &frame_state);
        
        XrFrameBeginInfo frame_begin_info = {XR_TYPE_FRAME_BEGIN_INFO};
        xrBeginFrame(m_session, &frame_begin_info);
        
        if (frame_state.shouldRender) {
            update_view_poses(frame_state.predictedDisplayTime);
        }
#endif
    }

    void post_render() override {
        if (!m_initialized || !m_running) return;
#ifdef XTU_USE_OPENXR
        std::lock_guard<std::mutex> lock(m_mutex);
        XrFrameEndInfo frame_end_info = {XR_TYPE_FRAME_END_INFO};
        frame_end_info.displayTime = m_predicted_display_time;
        frame_end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        frame_end_info.layerCount = 1;
        
        XrCompositionLayerProjection layer = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        layer.space = m_local_space;
        layer.viewCount = static_cast<uint32_t>(m_projection_views.size());
        layer.views = m_projection_views.data();
        
        const XrCompositionLayerBaseHeader* layers[] = {
            reinterpret_cast<const XrCompositionLayerBaseHeader*>(&layer)
        };
        frame_end_info.layers = layers;
        
        xrEndFrame(m_session, &frame_end_info);
#endif
    }

    XRPlayAreaMode get_play_area_mode() const override { return m_play_area_mode; }
    vec2f get_play_area() const override { return m_play_area_size; }

    mat4f get_transform_for_view(int view, const mat4f& cam_transform) override {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_OPENXR
        if (view >= 0 && view < static_cast<int>(m_views.size())) {
            XrView& xr_view = m_views[view];
            return xr_pose_to_mat4(xr_view.pose);
        }
#endif
        return mat4f::identity();
    }

    mat4f get_projection_for_view(int view, float aspect, float near, float far) override {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_OPENXR
        if (view >= 0 && view < static_cast<int>(m_views.size())) {
            XrView& xr_view = m_views[view];
            return xr_fov_to_projection(xr_view.fov, near, far);
        }
#endif
        return perspective(60.0f, aspect, near, far);
    }

    std::vector<Ref<XRTracker>> get_trackers() override {
        std::vector<Ref<XRTracker>> trackers;
        // Enumerate OpenXR action spaces
        return trackers;
    }

    void trigger_haptic_pulse(const StringName& action, int tracker_id, float frequency, float amplitude, float duration) override {
#ifdef XTU_USE_OPENXR
        // Send haptic pulse via OpenXR
#endif
    }

    Ref<OpenXRActionMap> get_action_map() const { return m_action_map; }

    Ref<OpenXRHandTracking> get_hand_tracker(bool left_hand) const {
        return left_hand ? m_hand_trackers[0] : m_hand_trackers[1];
    }

    String get_runtime_name() const { return m_runtime_name; }
    String get_runtime_version() const { return m_runtime_version; }

private:
#ifdef XTU_USE_OPENXR
    uint64_t m_predicted_display_time = 0;

    bool create_instance() {
        // Enumerate extensions and create XrInstance
        return true;
    }

    bool get_system_id() {
        XrSystemGetInfo system_info = {XR_TYPE_SYSTEM_GET_INFO};
        system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        return xrGetSystem(m_instance, &system_info, &m_system_id) == XR_SUCCESS;
    }

    bool create_session() {
        // Create graphics binding and XrSession
        return true;
    }

    bool create_spaces() {
        XrReferenceSpaceCreateInfo space_info = {XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        space_info.poseInReferenceSpace.orientation.w = 1.0f;
        xrCreateReferenceSpace(m_session, &space_info, &m_local_space);
        
        space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
        xrCreateReferenceSpace(m_session, &space_info, &m_stage_space);
        
        space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
        xrCreateReferenceSpace(m_session, &space_info, &m_view_space);
        
        return true;
    }

    bool create_action_map() {
        // Create XrActionSet and actions
        return true;
    }

    void poll_events() {
        XrEventDataBuffer event = {XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(m_instance, &event) == XR_SUCCESS) {
            switch (event.type) {
                case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                    auto* state_event = reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
                    m_session_state = state_event->state;
                    m_state = convert_session_state(state_event->state);
                    if (state_event->state == XR_SESSION_STATE_READY) {
                        XrSessionBeginInfo begin_info = {XR_TYPE_SESSION_BEGIN_INFO};
                        begin_info.primaryViewConfigurationType = m_view_config_type;
                        xrBeginSession(m_session, &begin_info);
                        m_running = true;
                    } else if (state_event->state == XR_SESSION_STATE_STOPPING) {
                        xrEndSession(m_session);
                        m_running = false;
                    } else if (state_event->state == XR_SESSION_STATE_EXITING) {
                        m_running = false;
                    }
                    break;
                }
            }
            event.type = XR_TYPE_EVENT_DATA_BUFFER;
        }
    }

    void update_action_states() {
        XrActiveActionSet active_set = {m_action_map->get_action_set(), XR_NULL_PATH};
        XrActionsSyncInfo sync_info = {XR_TYPE_ACTIONS_SYNC_INFO};
        sync_info.countActiveActionSets = 1;
        sync_info.activeActionSets = &active_set;
        xrSyncActions(m_session, &sync_info);
        
        // Query action states and update m_action_map
    }

    void update_hand_tracking() {
        // Update hand joint locations
    }

    void update_view_poses(uint64_t display_time) {
        m_predicted_display_time = display_time;
        
        XrViewLocateInfo view_locate_info = {XR_TYPE_VIEW_LOCATE_INFO};
        view_locate_info.viewConfigurationType = m_view_config_type;
        view_locate_info.displayTime = display_time;
        view_locate_info.space = m_local_space;
        
        XrViewState view_state = {XR_TYPE_VIEW_STATE};
        uint32_t view_count;
        xrLocateViews(m_session, &view_locate_info, &view_state, 0, &view_count, nullptr);
        m_views.resize(view_count);
        m_projection_views.resize(view_count);
        xrLocateViews(m_session, &view_locate_info, &view_state, view_count, &view_count, m_views.data());
        
        for (size_t i = 0; i < view_count; ++i) {
            m_projection_views[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            m_projection_views[i].pose = m_views[i].pose;
            m_projection_views[i].fov = m_views[i].fov;
        }
    }

    OpenXRSessionState convert_session_state(XrSessionState state) const {
        switch (state) {
            case XR_SESSION_STATE_READY: return OpenXRSessionState::STATE_READY;
            case XR_SESSION_STATE_SYNCHRONIZED: return OpenXRSessionState::STATE_SYNCHRONIZED;
            case XR_SESSION_STATE_VISIBLE: return OpenXRSessionState::STATE_VISIBLE;
            case XR_SESSION_STATE_FOCUSED: return OpenXRSessionState::STATE_FOCUSED;
            case XR_SESSION_STATE_STOPPING: return OpenXRSessionState::STATE_STOPPING;
            case XR_SESSION_STATE_EXITING: return OpenXRSessionState::STATE_EXITING;
            case XR_SESSION_STATE_LOSS_PENDING: return OpenXRSessionState::STATE_LOSS_PENDING;
            default: return OpenXRSessionState::STATE_IDLE;
        }
    }

    mat4f xr_pose_to_mat4(const XrPosef& pose) const {
        quatf rot(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        vec3f pos(pose.position.x, pose.position.y, pose.position.z);
        return translate(mat4f::identity(), pos) * rotate(mat4f::identity(), rot);
    }

    mat4f xr_fov_to_projection(const XrFovf& fov, float near, float far) const {
        float tan_left = std::tan(fov.angleLeft);
        float tan_right = std::tan(fov.angleRight);
        float tan_down = std::tan(fov.angleDown);
        float tan_up = std::tan(fov.angleUp);
        
        float tan_width = tan_right - tan_left;
        float tan_height = tan_up - tan_down;
        
        mat4f result(0);
        result[0][0] = 2.0f / tan_width;
        result[1][1] = 2.0f / tan_height;
        result[2][0] = (tan_right + tan_left) / tan_width;
        result[2][1] = (tan_up + tan_down) / tan_height;
        result[2][2] = -(far + near) / (far - near);
        result[2][3] = -1.0f;
        result[3][2] = -(2.0f * far * near) / (far - near);
        return result;
    }
#endif

    void initialize_hand_trackers() {
        for (int i = 0; i < 2; ++i) {
            Ref<OpenXRHandTracking> tracker;
            tracker.instance();
            tracker->set_left_hand(i == 0);
            m_hand_trackers.push_back(tracker);
        }
    }
};

} // namespace godot

// Bring into main namespace
using godot::OpenXRInterface;
using godot::OpenXRActionMap;
using godot::OpenXRHandTracking;
using godot::OpenXRSessionState;
using godot::OpenXRViewConfigType;
using godot::OpenXRActionType;
using godot::OpenXRHandJoint;
using godot::OpenXRActionBinding;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XOPENXR_HPP