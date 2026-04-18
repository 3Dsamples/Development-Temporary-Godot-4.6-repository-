// include/xtu/godot/xarkit.hpp
// xtensor-unified - ARKit Integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XARKIT_HPP
#define XTU_GODOT_XARKIT_HPP

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
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"

#ifdef XTU_OS_IOS
#import <ARKit/ARKit.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace arkit {

// #############################################################################
// Forward declarations
// #############################################################################
class ARKitInterface;
class ARKitSession;
class ARKitAnchor;
class ARKitPlaneAnchor;

// #############################################################################
// ARKit tracking state
// #############################################################################
enum class ARKitTrackingState : uint8_t {
    STATE_NOT_AVAILABLE = 0,
    STATE_LIMITED = 1,
    STATE_NORMAL = 2
};

// #############################################################################
// ARKit tracking reason (when limited)
// #############################################################################
enum class ARKitTrackingReason : uint8_t {
    REASON_NONE = 0,
    REASON_INITIALIZING = 1,
    REASON_RELOCALIZING = 2,
    REASON_EXCESSIVE_MOTION = 3,
    REASON_INSUFFICIENT_FEATURES = 4
};

// #############################################################################
// ARKit plane alignment
// #############################################################################
enum class ARKitPlaneAlignment : uint8_t {
    ALIGNMENT_HORIZONTAL = 0,
    ALIGNMENT_VERTICAL = 1
};

// #############################################################################
// ARKit plane classification
// #############################################################################
enum class ARKitPlaneClassification : uint8_t {
    CLASSIFICATION_NONE = 0,
    CLASSIFICATION_WALL = 1,
    CLASSIFICATION_FLOOR = 2,
    CLASSIFICATION_CEILING = 3,
    CLASSIFICATION_TABLE = 4,
    CLASSIFICATION_SEAT = 5,
    CLASSIFICATION_WINDOW = 6,
    CLASSIFICATION_DOOR = 7
};

// #############################################################################
// ARKit world mapping status
// #############################################################################
enum class ARKitWorldMappingStatus : uint8_t {
    MAPPING_NOT_AVAILABLE = 0,
    MAPPING_LIMITED = 1,
    MAPPING_EXTENDING = 2,
    MAPPING_MAPPED = 3
};

// #############################################################################
// ARKit light estimation
// #############################################################################
struct ARKitLightEstimate {
    float ambient_intensity = 1000.0f;
    float ambient_color_temperature = 6500.0f;
    vec3f primary_light_direction;
    float primary_light_intensity = 0.0f;
    bool valid = false;
};

// #############################################################################
// ARKitAnchor - Base anchor class
// #############################################################################
class ARKitAnchor : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ARKitAnchor, RefCounted)

private:
    String m_identifier;
    mat4f m_transform;
    bool m_tracking = true;

public:
    static StringName get_class_static() { return StringName("ARKitAnchor"); }

    void set_identifier(const String& id) { m_identifier = id; }
    String get_identifier() const { return m_identifier; }

    void set_transform(const mat4f& transform) { m_transform = transform; }
    mat4f get_transform() const { return m_transform; }

    void set_tracking(bool tracking) { m_tracking = tracking; }
    bool is_tracking() const { return m_tracking; }
};

// #############################################################################
// ARKitPlaneAnchor - Detected plane
// #############################################################################
class ARKitPlaneAnchor : public ARKitAnchor {
    XTU_GODOT_REGISTER_CLASS(ARKitPlaneAnchor, ARKitAnchor)

private:
    ARKitPlaneAlignment m_alignment = ARKitPlaneAlignment::ALIGNMENT_HORIZONTAL;
    ARKitPlaneClassification m_classification = ARKitPlaneClassification::CLASSIFICATION_NONE;
    vec3f m_center;
    vec3f m_extent;
    std::vector<vec3f> m_boundary_vertices;
    bool m_updated = false;
    bool m_removed = false;

public:
    static StringName get_class_static() { return StringName("ARKitPlaneAnchor"); }

    void set_alignment(ARKitPlaneAlignment alignment) { m_alignment = alignment; }
    ARKitPlaneAlignment get_alignment() const { return m_alignment; }

    void set_classification(ARKitPlaneClassification classification) { m_classification = classification; }
    ARKitPlaneClassification get_classification() const { return m_classification; }

    void set_center(const vec3f& center) { m_center = center; }
    vec3f get_center() const { return m_center; }

    void set_extent(const vec3f& extent) { m_extent = extent; }
    vec3f get_extent() const { return m_extent; }

    void set_boundary_vertices(const std::vector<vec3f>& vertices) { m_boundary_vertices = vertices; }
    const std::vector<vec3f>& get_boundary_vertices() const { return m_boundary_vertices; }

    void set_updated(bool updated) { m_updated = updated; }
    bool is_updated() const { return m_updated; }

    void set_removed(bool removed) { m_removed = removed; }
    bool is_removed() const { return m_removed; }
};

// #############################################################################
// ARKitInterface - Main ARKit implementation
// #############################################################################
class ARKitInterface : public XRInterface {
    XTU_GODOT_REGISTER_CLASS(ARKitInterface, XRInterface)

private:
    static ARKitInterface* s_singleton;

    bool m_initialized = false;
    bool m_session_running = false;
    ARKitTrackingState m_tracking_state = ARKitTrackingState::STATE_NOT_AVAILABLE;
    ARKitTrackingReason m_tracking_reason = ARKitTrackingReason::REASON_NONE;
    ARKitWorldMappingStatus m_world_mapping = ARKitWorldMappingStatus::MAPPING_NOT_AVAILABLE;

    mat4f m_camera_transform;
    mat4f m_projection_matrix;
    vec2i m_camera_resolution = {1920, 1440};
    Ref<Texture2D> m_camera_feed_texture;

    ARKitLightEstimate m_light_estimate;

    bool m_plane_detection_enabled = true;
    bool m_light_estimation_enabled = true;
    bool m_people_occlusion_enabled = false;
    bool m_world_tracking_enabled = true;

    std::unordered_map<String, Ref<ARKitPlaneAnchor>> m_plane_anchors;
    std::vector<String> m_updated_anchors;
    std::vector<String> m_removed_anchors;

    mutable std::mutex m_mutex;

public:
    static ARKitInterface* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ARKitInterface"); }

    ARKitInterface() { s_singleton = this; }
    ~ARKitInterface() { s_singleton = nullptr; }

    StringName get_name() const override { return StringName("ARKit"); }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

#ifdef XTU_OS_IOS
        if (@available(iOS 11.0, *)) {
            if (!ARWorldTrackingConfiguration.isSupported) {
                return false;
            }
        } else {
            return false;
        }
#endif
        m_initialized = true;
        return true;
    }

    void uninitialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_session_running) {
            stop_session();
        }
        m_initialized = false;
    }

    bool is_initialized() const override { return m_initialized; }

    // #########################################################################
    // Session management
    // #########################################################################
    void start_session() {
        if (m_session_running) return;
        m_session_running = true;
        emit_signal("session_started");
    }

    void stop_session() {
        if (!m_session_running) return;
        m_session_running = false;
        emit_signal("session_ended");
    }

    void reset_session() {
        stop_session();
        m_plane_anchors.clear();
        start_session();
    }

    bool is_session_running() const { return m_session_running; }

    // #########################################################################
    // Configuration
    // #########################################################################
    void set_plane_detection_enabled(bool enabled) { m_plane_detection_enabled = enabled; }
    bool is_plane_detection_enabled() const { return m_plane_detection_enabled; }

    void set_light_estimation_enabled(bool enabled) { m_light_estimation_enabled = enabled; }
    bool is_light_estimation_enabled() const { return m_light_estimation_enabled; }

    void set_people_occlusion_enabled(bool enabled) { m_people_occlusion_enabled = enabled; }
    bool is_people_occlusion_enabled() const { return m_people_occlusion_enabled; }

    void set_world_tracking_enabled(bool enabled) { m_world_tracking_enabled = enabled; }
    bool is_world_tracking_enabled() const { return m_world_tracking_enabled; }

    // #########################################################################
    // Tracking state
    // #########################################################################
    ARKitTrackingState get_tracking_state() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_tracking_state;
    }

    ARKitTrackingReason get_tracking_reason() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_tracking_reason;
    }

    ARKitWorldMappingStatus get_world_mapping_status() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_world_mapping;
    }

    // #########################################################################
    // Camera and projection
    // #########################################################################
    mat4f get_camera_transform() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_camera_transform;
    }

    mat4f get_projection_matrix() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_projection_matrix;
    }

    vec2i get_camera_resolution() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_camera_resolution;
    }

    Ref<Texture2D> get_camera_feed_texture() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_camera_feed_texture;
    }

    // #########################################################################
    // Light estimation
    // #########################################################################
    ARKitLightEstimate get_light_estimate() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_light_estimate;
    }

    // #########################################################################
    // Plane anchors
    // #########################################################################
    std::vector<Ref<ARKitPlaneAnchor>> get_plane_anchors() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Ref<ARKitPlaneAnchor>> result;
        for (const auto& kv : m_plane_anchors) {
            result.push_back(kv.second);
        }
        return result;
    }

    Ref<ARKitPlaneAnchor> get_plane_anchor(const String& identifier) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_plane_anchors.find(identifier);
        return it != m_plane_anchors.end() ? it->second : Ref<ARKitPlaneAnchor>();
    }

    std::vector<String> get_updated_anchors() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto result = std::move(m_updated_anchors);
        m_updated_anchors.clear();
        return result;
    }

    std::vector<String> get_removed_anchors() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto result = std::move(m_removed_anchors);
        m_removed_anchors.clear();
        return result;
    }

    // #########################################################################
    // Hit testing
    // #########################################################################
    std::vector<Dictionary> hit_test(const vec2f& screen_point, bool existing_planes = true,
                                      bool estimated_planes = true, bool feature_points = true) {
        std::vector<Dictionary> results;
        // Perform hit test against planes and feature points
        return results;
    }

    // #########################################################################
    // XRInterface overrides
    // #########################################################################
    XRPlayAreaMode get_play_area_mode() const override {
        return XRPlayAreaMode::PLAY_AREA_3DOF;
    }

    vec2f get_play_area() const override { return vec2f(1, 1); }

    mat4f get_transform_for_view(int view, const mat4f& cam_transform) override {
        return get_camera_transform();
    }

    mat4f get_projection_for_view(int view, float aspect, float near, float far) override {
        return get_projection_matrix();
    }

    std::vector<Ref<XRTracker>> get_trackers() override {
        std::vector<Ref<XRTracker>> trackers;
        return trackers;
    }

    void process() override {
        // Update tracking on main thread
    }

    // #########################################################################
    // Internal callbacks (called from Objective-C bridge)
    // #########################################################################
    void _on_frame_updated(const mat4f& camera_transform, const mat4f& projection,
                           const ARKitLightEstimate& light_estimate) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_camera_transform = camera_transform;
        m_projection_matrix = projection;
        m_light_estimate = light_estimate;
    }

    void _on_plane_added(const String& identifier, const mat4f& transform,
                          ARKitPlaneAlignment alignment, const vec3f& center, const vec3f& extent) {
        std::lock_guard<std::mutex> lock(m_mutex);
        Ref<ARKitPlaneAnchor> anchor;
        anchor.instance();
        anchor->set_identifier(identifier);
        anchor->set_transform(transform);
        anchor->set_alignment(alignment);
        anchor->set_center(center);
        anchor->set_extent(extent);
        m_plane_anchors[identifier] = anchor;
        emit_signal("plane_added", identifier);
    }

    void _on_plane_updated(const String& identifier, const mat4f& transform,
                            const vec3f& center, const vec3f& extent) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_plane_anchors.find(identifier);
        if (it != m_plane_anchors.end()) {
            it->second->set_transform(transform);
            it->second->set_center(center);
            it->second->set_extent(extent);
            it->second->set_updated(true);
            m_updated_anchors.push_back(identifier);
            emit_signal("plane_updated", identifier);
        }
    }

    void _on_plane_removed(const String& identifier) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_plane_anchors.erase(identifier);
        m_removed_anchors.push_back(identifier);
        emit_signal("plane_removed", identifier);
    }

    void _on_tracking_state_changed(int state, int reason) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_tracking_state = static_cast<ARKitTrackingState>(state);
        m_tracking_reason = static_cast<ARKitTrackingReason>(reason);
        emit_signal("tracking_state_changed", state, reason);
    }
};

} // namespace arkit

// Bring into main namespace
using arkit::ARKitInterface;
using arkit::ARKitAnchor;
using arkit::ARKitPlaneAnchor;
using arkit::ARKitTrackingState;
using arkit::ARKitTrackingReason;
using arkit::ARKitPlaneAlignment;
using arkit::ARKitPlaneClassification;
using arkit::ARKitWorldMappingStatus;
using arkit::ARKitLightEstimate;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XARKIT_HPP