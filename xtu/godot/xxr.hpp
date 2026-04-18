// include/xtu/godot/xxr.hpp
// xtensor-unified - XR (AR/VR) server for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XXR_HPP
#define XTU_GODOT_XXR_HPP

#include <algorithm>
#include <atomic>
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
#include "xtu/godot/xinput.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class XRServer;
class XRInterface;
class XROrigin3D;
class XRAnchor3D;
class XRController3D;
class XRHand3D;
class XRPose;
class XRTracker;

// #############################################################################
// XR tracker types
// #############################################################################
enum class XRTrackerType : uint8_t {
    TRACKER_UNKNOWN = 0,
    TRACKER_HEAD = 1,
    TRACKER_CONTROLLER_LEFT = 2,
    TRACKER_CONTROLLER_RIGHT = 3,
    TRACKER_HAND_LEFT = 4,
    TRACKER_HAND_RIGHT = 5,
    TRACKER_BODY = 6,
    TRACKER_FACE = 7,
    TRACKER_EYE_LEFT = 8,
    TRACKER_EYE_RIGHT = 9,
    TRACKER_CUSTOM = 10
};

// #############################################################################
// XR hand joints
// #############################################################################
enum class XRHandJoints : uint8_t {
    HAND_JOINT_PALM = 0,
    HAND_JOINT_WRIST = 1,
    HAND_JOINT_THUMB_METACARPAL = 2,
    HAND_JOINT_THUMB_PROXIMAL = 3,
    HAND_JOINT_THUMB_DISTAL = 4,
    HAND_JOINT_THUMB_TIP = 5,
    HAND_JOINT_INDEX_METACARPAL = 6,
    HAND_JOINT_INDEX_PROXIMAL = 7,
    HAND_JOINT_INDEX_INTERMEDIATE = 8,
    HAND_JOINT_INDEX_DISTAL = 9,
    HAND_JOINT_INDEX_TIP = 10,
    HAND_JOINT_MIDDLE_METACARPAL = 11,
    HAND_JOINT_MIDDLE_PROXIMAL = 12,
    HAND_JOINT_MIDDLE_INTERMEDIATE = 13,
    HAND_JOINT_MIDDLE_DISTAL = 14,
    HAND_JOINT_MIDDLE_TIP = 15,
    HAND_JOINT_RING_METACARPAL = 16,
    HAND_JOINT_RING_PROXIMAL = 17,
    HAND_JOINT_RING_INTERMEDIATE = 18,
    HAND_JOINT_RING_DISTAL = 19,
    HAND_JOINT_RING_TIP = 20,
    HAND_JOINT_PINKY_METACARPAL = 21,
    HAND_JOINT_PINKY_PROXIMAL = 22,
    HAND_JOINT_PINKY_INTERMEDIATE = 23,
    HAND_JOINT_PINKY_DISTAL = 24,
    HAND_JOINT_PINKY_TIP = 25,
    HAND_JOINT_MAX = 26
};

// #############################################################################
// XR hand finger gestures
// #############################################################################
enum class XRHandGesture : uint32_t {
    GESTURE_NONE = 0,
    GESTURE_PINCH = 1 << 0,
    GESTURE_POINT = 1 << 1,
    GESTURE_GRAB = 1 << 2,
    GESTURE_THUMB_UP = 1 << 3,
    GESTURE_FIST = 1 << 4,
    GESTURE_OPEN_HAND = 1 << 5
};

// #############################################################################
// XR tracking confidence
// #############################################################################
enum class XRTrackingConfidence : uint8_t {
    TRACKING_CONFIDENCE_NONE = 0,
    TRACKING_CONFIDENCE_LOW = 1,
    TRACKING_CONFIDENCE_HIGH = 2
};

// #############################################################################
// XR anchor tracking state
// #############################################################################
enum class XRAnchorTrackingState : uint8_t {
    ANCHOR_TRACKING_NONE = 0,
    ANCHOR_TRACKING_PENDING = 1,
    ANCHOR_TRACKING_TRACKING = 2,
    ANCHOR_TRACKING_STOPPED = 3
};

// #############################################################################
// XR play area mode
// #############################################################################
enum class XRPlayAreaMode : uint8_t {
    PLAY_AREA_UNKNOWN = 0,
    PLAY_AREA_3DOF = 1,
    PLAY_AREA_SITTING = 2,
    PLAY_AREA_ROOMSCALE = 3,
    PLAY_AREA_STAGE = 4
};

// #############################################################################
// XRPose - Transform with tracking confidence
// #############################################################################
class XRPose : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(XRPose, RefCounted)

private:
    mat4f m_transform;
    vec3f m_linear_velocity;
    vec3f m_angular_velocity;
    XRTrackingConfidence m_confidence = XRTrackingConfidence::TRACKING_CONFIDENCE_NONE;
    bool m_has_transform = false;
    bool m_has_velocity = false;

public:
    static StringName get_class_static() { return StringName("XRPose"); }

    void set_transform(const mat4f& t) { m_transform = t; m_has_transform = true; }
    mat4f get_transform() const { return m_transform; }
    bool has_transform() const { return m_has_transform; }

    void set_linear_velocity(const vec3f& v) { m_linear_velocity = v; m_has_velocity = true; }
    vec3f get_linear_velocity() const { return m_linear_velocity; }

    void set_angular_velocity(const vec3f& v) { m_angular_velocity = v; }
    vec3f get_angular_velocity() const { return m_angular_velocity; }
    bool has_velocity() const { return m_has_velocity; }

    void set_confidence(XRTrackingConfidence conf) { m_confidence = conf; }
    XRTrackingConfidence get_confidence() const { return m_confidence; }

    void clear() {
        m_has_transform = false;
        m_has_velocity = false;
        m_confidence = XRTrackingConfidence::TRACKING_CONFIDENCE_NONE;
    }
};

// #############################################################################
// XRTracker - Individual tracked device
// #############################################################################
class XRTracker : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(XRTracker, RefCounted)

private:
    StringName m_name;
    XRTrackerType m_type = XRTrackerType::TRACKER_UNKNOWN;
    int m_id = 0;
    Ref<XRPose> m_pose;
    std::vector<bool> m_input_states;
    std::vector<float> m_axis_states;
    bool m_is_active = false;

public:
    static StringName get_class_static() { return StringName("XRTracker"); }

    XRTracker() { m_pose.instance(); }

    void set_name(const StringName& name) { m_name = name; }
    StringName get_name() const { return m_name; }

    void set_type(XRTrackerType type) { m_type = type; }
    XRTrackerType get_type() const { return m_type; }

    void set_id(int id) { m_id = id; }
    int get_id() const { return m_id; }

    Ref<XRPose> get_pose() { return m_pose; }

    void set_active(bool active) { m_is_active = active; }
    bool is_active() const { return m_is_active; }

    bool is_tracking_hands() const {
        return m_type == XRTrackerType::TRACKER_HAND_LEFT ||
               m_type == XRTrackerType::TRACKER_HAND_RIGHT;
    }

    bool is_tracking_controller() const {
        return m_type == XRTrackerType::TRACKER_CONTROLLER_LEFT ||
               m_type == XRTrackerType::TRACKER_CONTROLLER_RIGHT;
    }
};

// #############################################################################
// XRInterface - Backend interface for OpenXR/ARKit/ARCore/etc.
// #############################################################################
class XRInterface : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(XRInterface, RefCounted)

private:
    StringName m_name;
    bool m_initialized = false;
    bool m_primary = false;

public:
    static StringName get_class_static() { return StringName("XRInterface"); }

    virtual StringName get_name() const { return m_name; }
    virtual uint32_t get_capabilities() const { return 0; }

    virtual bool initialize() { m_initialized = true; return true; }
    virtual void uninitialize() { m_initialized = false; }

    virtual bool is_initialized() const { return m_initialized; }
    virtual bool is_primary() const { return m_primary; }
    virtual void set_primary(bool primary) { m_primary = primary; }

    virtual XRPlayAreaMode get_play_area_mode() const { return XRPlayAreaMode::PLAY_AREA_3DOF; }
    virtual vec2f get_play_area() const { return vec2f(1, 1); }
    virtual bool get_anchor_detection_is_enabled() const { return false; }
    virtual void set_anchor_detection_is_enabled(bool enable) {}

    virtual int get_camera_feed_id() const { return 0; }

    virtual mat4f get_transform_for_view(int view, const mat4f& cam_transform) {
        return mat4f::identity();
    }

    virtual mat4f get_projection_for_view(int view, float aspect, float near, float far) {
        return perspective(60.0f, aspect, near, far);
    }

    virtual std::vector<Ref<XRTracker>> get_trackers() { return {}; }
    virtual Ref<XRPose> get_tracker_pose(int tracker_id) { return Ref<XRPose>(); }

    virtual void process() {}
    virtual void pre_render() {}
    virtual void post_render() {}

    virtual void trigger_haptic_pulse(const StringName& action, int tracker_id, float frequency, float amplitude, float duration) {}
};

// #############################################################################
// XRServer - Global XR manager singleton
// #############################################################################
class XRServer : public Object {
    XTU_GODOT_REGISTER_CLASS(XRServer, Object)

private:
    static XRServer* s_singleton;
    Ref<XRInterface> m_primary_interface;
    std::vector<Ref<XRInterface>> m_interfaces;
    XROrigin3D* m_world_origin = nullptr;
    mat4f m_world_scale;
    float m_world_scale_factor = 1.0f;
    mat4f m_reference_frame;
    std::unordered_map<int, Ref<XRTracker>> m_trackers;
    std::mutex m_mutex;

public:
    static XRServer* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("XRServer"); }

    XRServer() { s_singleton = this; }
    ~XRServer() { s_singleton = nullptr; }

    void add_interface(const Ref<XRInterface>& interface) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_interfaces.begin(), m_interfaces.end(), interface) == m_interfaces.end()) {
            m_interfaces.push_back(interface);
        }
    }

    void remove_interface(const Ref<XRInterface>& interface) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_interfaces.begin(), m_interfaces.end(), interface);
        if (it != m_interfaces.end()) m_interfaces.erase(it);
    }

    Ref<XRInterface> find_interface(const StringName& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& iface : m_interfaces) {
            if (iface->get_name() == name) return iface;
        }
        return Ref<XRInterface>();
    }

    std::vector<Ref<XRInterface>> get_interfaces() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_interfaces;
    }

    void set_primary_interface(const Ref<XRInterface>& interface) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_primary_interface.is_valid()) {
            m_primary_interface->set_primary(false);
        }
        m_primary_interface = interface;
        if (interface.is_valid()) {
            interface->set_primary(true);
        }
    }

    Ref<XRInterface> get_primary_interface() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_primary_interface;
    }

    void set_world_origin(XROrigin3D* origin) { m_world_origin = origin; }
    XROrigin3D* get_world_origin() const { return m_world_origin; }

    void set_world_scale(float scale) {
        m_world_scale_factor = scale;
        m_world_scale = scale(mat4f::identity(), vec3f(scale, scale, scale));
    }

    float get_world_scale() const { return m_world_scale_factor; }

    mat4f get_reference_frame() const {
        if (m_world_origin) {
            return m_world_origin->get_global_transform() * m_world_scale;
        }
        return m_world_scale;
    }

    void center_on_hmd(XRInterface::RotationMode mode, bool keep_height) {
        // Recenter tracking space
    }

    Ref<XRTracker> get_tracker(int tracker_id) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_trackers.find(tracker_id);
        return it != m_trackers.end() ? it->second : Ref<XRTracker>();
    }

    std::vector<Ref<XRTracker>> get_trackers(XRTrackerType type_mask) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Ref<XRTracker>> result;
        for (const auto& kv : m_trackers) {
            if (static_cast<int>(kv.second->get_type()) & static_cast<int>(type_mask)) {
                result.push_back(kv.second);
            }
        }
        return result;
    }

    void process() {
        if (m_primary_interface.is_valid()) {
            m_primary_interface->process();
        }
    }

    void pre_render() {
        if (m_primary_interface.is_valid()) {
            m_primary_interface->pre_render();
        }
    }

    void post_render() {
        if (m_primary_interface.is_valid()) {
            m_primary_interface->post_render();
        }
    }
};

// #############################################################################
// XROrigin3D - World tracking origin
// #############################################################################
class XROrigin3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(XROrigin3D, Node3D)

private:
    bool m_current = true;
    float m_world_scale = 1.0f;

public:
    static StringName get_class_static() { return StringName("XROrigin3D"); }

    void set_current(bool current) {
        m_current = current;
        if (current) {
            XRServer::get_singleton()->set_world_origin(this);
        }
    }

    bool is_current() const { return m_current; }

    void set_world_scale(float scale) {
        m_world_scale = scale;
        if (m_current) {
            XRServer::get_singleton()->set_world_scale(scale);
        }
    }

    float get_world_scale() const { return m_world_scale; }

    void _enter_tree() override {
        Node3D::_enter_tree();
        if (m_current) {
            XRServer::get_singleton()->set_world_origin(this);
            XRServer::get_singleton()->set_world_scale(m_world_scale);
        }
    }

    void _exit_tree() override {
        if (m_current && XRServer::get_singleton()->get_world_origin() == this) {
            XRServer::get_singleton()->set_world_origin(nullptr);
        }
        Node3D::_exit_tree();
    }
};

// #############################################################################
// XRAnchor3D - Spatial anchor
// #############################################################################
class XRAnchor3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(XRAnchor3D, Node3D)

private:
    StringName m_anchor_id;
    XRAnchorTrackingState m_tracking_state = XRAnchorTrackingState::ANCHOR_TRACKING_NONE;
    bool m_is_active = true;

public:
    static StringName get_class_static() { return StringName("XRAnchor3D"); }

    void set_anchor_id(const StringName& id) { m_anchor_id = id; }
    StringName get_anchor_id() const { return m_anchor_id; }

    XRAnchorTrackingState get_tracking_state() const { return m_tracking_state; }

    bool get_is_active() const { return m_is_active; }

    vec3f get_size() const { return vec3f(0); }
};

// #############################################################################
// XRController3D - VR controller
// #############################################################################
class XRController3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(XRController3D, Node3D)

private:
    int m_tracker_id = -1;
    StringName m_controller_name;
    float m_rumble = 0.0f;
    std::unordered_map<StringName, float> m_axis_values;
    std::unordered_map<StringName, bool> m_button_states;

public:
    static StringName get_class_static() { return StringName("XRController3D"); }

    void set_tracker_id(int id) { m_tracker_id = id; }
    int get_tracker_id() const { return m_tracker_id; }

    void set_controller_name(const StringName& name) { m_controller_name = name; }
    StringName get_controller_name() const { return m_controller_name; }

    bool is_button_pressed(const StringName& button) const {
        auto it = m_button_states.find(button);
        return it != m_button_states.end() && it->second;
    }

    float get_axis(const StringName& axis) const {
        auto it = m_axis_values.find(axis);
        return it != m_axis_values.end() ? it->second : 0.0f;
    }

    vec2f get_vector2(const StringName& axis_x, const StringName& axis_y) const {
        return vec2f(get_axis(axis_x), get_axis(axis_y));
    }

    void trigger_haptic_pulse(const StringName& action, float frequency, float amplitude, float duration) {
        XRServer::get_singleton()->get_primary_interface()->trigger_haptic_pulse(action, m_tracker_id, frequency, amplitude, duration);
    }

    void _process(double delta) override {
        auto tracker = XRServer::get_singleton()->get_tracker(m_tracker_id);
        if (tracker.is_valid() && tracker->is_active()) {
            Ref<XRPose> pose = tracker->get_pose();
            if (pose.is_valid() && pose->has_transform()) {
                XROrigin3D* origin = XRServer::get_singleton()->get_world_origin();
                mat4f world_transform = origin ? origin->get_global_transform() : mat4f::identity();
                set_global_transform(world_transform * pose->get_transform());
            }
        }
    }
};

// #############################################################################
// XRHand3D - Hand tracking
// #############################################################################
class XRHand3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(XRHand3D, Node3D)

private:
    int m_tracker_id = -1;
    bool m_is_left_hand = false;
    std::vector<mat4f> m_joint_transforms;
    std::vector<vec3f> m_joint_radii;
    uint32_t m_active_gestures = 0;
    float m_pinch_amount = 0.0f;
    float m_grab_amount = 0.0f;

public:
    static StringName get_class_static() { return StringName("XRHand3D"); }

    XRHand3D() {
        m_joint_transforms.resize(static_cast<size_t>(XRHandJoints::HAND_JOINT_MAX), mat4f::identity());
        m_joint_radii.resize(static_cast<size_t>(XRHandJoints::HAND_JOINT_MAX), 0.01f);
    }

    void set_tracker_id(int id) { m_tracker_id = id; }
    int get_tracker_id() const { return m_tracker_id; }

    void set_left_hand(bool left) { m_is_left_hand = left; }
    bool is_left_hand() const { return m_is_left_hand; }

    mat4f get_joint_transform(XRHandJoints joint) const {
        return m_joint_transforms[static_cast<size_t>(joint)];
    }

    vec3f get_joint_radius(XRHandJoints joint) const {
        return m_joint_radii[static_cast<size_t>(joint)];
    }

    bool has_gesture(XRHandGesture gesture) const {
        return (m_active_gestures & static_cast<uint32_t>(gesture)) != 0;
    }

    float get_pinch_amount() const { return m_pinch_amount; }
    float get_grab_amount() const { return m_grab_amount; }

    void _process(double delta) override {
        // Update from tracker
        auto tracker = XRServer::get_singleton()->get_tracker(m_tracker_id);
        if (tracker.is_valid() && tracker->is_active()) {
            // Update joint transforms
        }
    }
};

// #############################################################################
// OpenXRInterface - OpenXR backend implementation
// #############################################################################
class OpenXRInterface : public XRInterface {
    XTU_GODOT_REGISTER_CLASS(OpenXRInterface, XRInterface)

private:
    bool m_initialized = false;
    uint64_t m_instance = 0;
    uint64_t m_session = 0;
    uint64_t m_system_id = 0;
    XRPlayAreaMode m_play_area_mode = XRPlayAreaMode::PLAY_AREA_ROOMSCALE;
    vec2f m_play_area_size = {2, 2};

public:
    static StringName get_class_static() { return StringName("OpenXRInterface"); }

    StringName get_name() const override { return StringName("OpenXR"); }

    bool initialize() override {
        // Initialize OpenXR loader and create instance
        m_initialized = true;
        return true;
    }

    void uninitialize() override {
        m_initialized = false;
    }

    bool is_initialized() const override { return m_initialized; }

    XRPlayAreaMode get_play_area_mode() const override { return m_play_area_mode; }
    vec2f get_play_area() const override { return m_play_area_size; }

    mat4f get_transform_for_view(int view, const mat4f& cam_transform) override {
        // Get eye poses from OpenXR
        return mat4f::identity();
    }

    mat4f get_projection_for_view(int view, float aspect, float near, float far) override {
        // Get projection matrices from OpenXR
        return perspective(60.0f, aspect, near, far);
    }

    std::vector<Ref<XRTracker>> get_trackers() override {
        // Enumerate action spaces
        return {};
    }

    void trigger_haptic_pulse(const StringName& action, int tracker_id, float frequency, float amplitude, float duration) override {
        // Apply haptic feedback
    }
};

} // namespace godot

// Bring into main namespace
using godot::XRServer;
using godot::XRInterface;
using godot::XROrigin3D;
using godot::XRAnchor3D;
using godot::XRController3D;
using godot::XRHand3D;
using godot::XRPose;
using godot::XRTracker;
using godot::OpenXRInterface;
using godot::XRTrackerType;
using godot::XRHandJoints;
using godot::XRHandGesture;
using godot::XRTrackingConfidence;
using godot::XRAnchorTrackingState;
using godot::XRPlayAreaMode;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XXR_HPP