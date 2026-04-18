// include/xtu/godot/xarcore.hpp
// xtensor-unified - ARCore Integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XARCORE_HPP
#define XTU_GODOT_XARCORE_HPP

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

#ifdef XTU_OS_ANDROID
#include <arcore_c_api.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace arcore {

// #############################################################################
// Forward declarations
// #############################################################################
class ARCoreInterface;
class ARCoreSession;
class ARCoreAnchor;
class ARCorePlaneAnchor;
class ARCorePointCloud;

// #############################################################################
// ARCore tracking state
// #############################################################################
enum class ARCoreTrackingState : uint8_t {
    STATE_STOPPED = 0,
    STATE_TRACKING = 1,
    STATE_PAUSED = 2
};

// #############################################################################
// ARCore tracking failure reason
// #############################################################################
enum class ARCoreTrackingFailureReason : uint8_t {
    REASON_NONE = 0,
    REASON_BAD_STATE = 1,
    REASON_INSUFFICIENT_LIGHT = 2,
    REASON_EXCESSIVE_MOTION = 3,
    REASON_INSUFFICIENT_FEATURES = 4,
    REASON_CAMERA_UNAVAILABLE = 5
};

// #############################################################################
// ARCore plane type
// #############################################################################
enum class ARCorePlaneType : uint8_t {
    TYPE_HORIZONTAL_UPWARD = 0,
    TYPE_HORIZONTAL_DOWNWARD = 1,
    TYPE_VERTICAL = 2
};

// #############################################################################
// ARCore plane detection mode
// #############################################################################
enum class ARCorePlaneDetectionMode : uint8_t {
    MODE_NONE = 0,
    MODE_HORIZONTAL = 1,
    MODE_VERTICAL = 2,
    MODE_HORIZONTAL_AND_VERTICAL = 3
};

// #############################################################################
// ARCore cloud anchor state
// #############################################################################
enum class ARCoreCloudAnchorState : uint8_t {
    STATE_NONE = 0,
    STATE_TASK_IN_PROGRESS = 1,
    STATE_SUCCESS = 2,
    STATE_ERROR_INTERNAL = 3,
    STATE_ERROR_NOT_AUTHORIZED = 4,
    STATE_ERROR_SERVICE_UNAVAILABLE = 5,
    STATE_ERROR_RESOURCE_EXHAUSTED = 6,
    STATE_ERROR_HOSTING_DATASET_PROCESSING_FAILED = 7,
    STATE_ERROR_CLOUD_ID_NOT_FOUND = 8,
    STATE_ERROR_RESOLVING_LOCALIZATION_NO_MATCH = 9,
    STATE_ERROR_RESOLVING_SDK_VERSION_TOO_OLD = 10,
    STATE_ERROR_RESOLVING_SDK_VERSION_TOO_NEW = 11
};

// #############################################################################
// ARCore light estimation
// #############################################################################
struct ARCoreLightEstimate {
    float ambient_intensity = 0.0f;
    vec3f ambient_color;
    vec3f primary_light_direction;
    float primary_light_intensity = 0.0f;
    float ambient_spherical_harmonics[27];
    bool valid = false;
};

// #############################################################################
// ARCore depth point
// #############################################################################
struct ARCoreDepthPoint {
    vec3f position;
    float confidence = 0.0f;
};

// #############################################################################
// ARCoreAnchor - Base anchor class
// #############################################################################
class ARCoreAnchor : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ARCoreAnchor, RefCounted)

private:
    String m_identifier;
    mat4f m_transform;
    ARCoreTrackingState m_tracking_state = ARCoreTrackingState::STATE_STOPPED;

public:
    static StringName get_class_static() { return StringName("ARCoreAnchor"); }

    void set_identifier(const String& id) { m_identifier = id; }
    String get_identifier() const { return m_identifier; }

    void set_transform(const mat4f& transform) { m_transform = transform; }
    mat4f get_transform() const { return m_transform; }

    void set_tracking_state(ARCoreTrackingState state) { m_tracking_state = state; }
    ARCoreTrackingState get_tracking_state() const { return m_tracking_state; }
};

// #############################################################################
// ARCorePlaneAnchor - Detected plane
// #############################################################################
class ARCorePlaneAnchor : public ARCoreAnchor {
    XTU_GODOT_REGISTER_CLASS(ARCorePlaneAnchor, ARCoreAnchor)

private:
    ARCorePlaneType m_type = ARCorePlaneType::TYPE_HORIZONTAL_UPWARD;
    vec3f m_center;
    vec3f m_extent;
    std::vector<vec3f> m_boundary_polygon;
    bool m_subsumed_by_another = false;

public:
    static StringName get_class_static() { return StringName("ARCorePlaneAnchor"); }

    void set_type(ARCorePlaneType type) { m_type = type; }
    ARCorePlaneType get_type() const { return m_type; }

    void set_center(const vec3f& center) { m_center = center; }
    vec3f get_center() const { return m_center; }

    void set_extent(const vec3f& extent) { m_extent = extent; }
    vec3f get_extent() const { return m_extent; }

    void set_boundary_polygon(const std::vector<vec3f>& polygon) { m_boundary_polygon = polygon; }
    const std::vector<vec3f>& get_boundary_polygon() const { return m_boundary_polygon; }

    void set_subsumed(bool subsumed) { m_subsumed_by_another = subsumed; }
    bool is_subsumed() const { return m_subsumed_by_another; }
};

// #############################################################################
// ARCorePointCloud - Feature points
// #############################################################################
class ARCorePointCloud : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ARCorePointCloud, RefCounted)

private:
    std::vector<vec4f> m_points; // xyz + confidence
    uint64_t m_timestamp = 0;

public:
    static StringName get_class_static() { return StringName("ARCorePointCloud"); }

    void set_points(const std::vector<vec4f>& points) { m_points = points; }
    const std::vector<vec4f>& get_points() const { return m_points; }

    int get_point_count() const { return static_cast<int>(m_points.size()); }

    vec3f get_point(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_points.size()) ?
               vec3f(m_points[idx].x(), m_points[idx].y(), m_points[idx].z()) : vec3f(0);
    }

    float get_confidence(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_points.size()) ? m_points[idx].w() : 0.0f;
    }

    void set_timestamp(uint64_t ts) { m_timestamp = ts; }
    uint64_t get_timestamp() const { return m_timestamp; }
};

// #############################################################################
// ARCoreInterface - Main ARCore implementation
// #############################################################################
class ARCoreInterface : public XRInterface {
    XTU_GODOT_REGISTER_CLASS(ARCoreInterface, XRInterface)

private:
    static ARCoreInterface* s_singleton;

    bool m_initialized = false;
    bool m_session_running = false;
    ARCoreTrackingState m_tracking_state = ARCoreTrackingState::STATE_STOPPED;
    ARCoreTrackingFailureReason m_failure_reason = ARCoreTrackingFailureReason::REASON_NONE;

    mat4f m_camera_transform;
    mat4f m_projection_matrix;
    vec2i m_camera_resolution = {1920, 1080};
    Ref<Texture2D> m_camera_feed_texture;

    ARCoreLightEstimate m_light_estimate;

    ARCorePlaneDetectionMode m_plane_detection_mode = ARCorePlaneDetectionMode::MODE_HORIZONTAL_AND_VERTICAL;
    bool m_light_estimation_enabled = true;
    bool m_depth_enabled = false;
    bool m_instant_placement_enabled = false;
    bool m_cloud_anchors_enabled = false;

    std::unordered_map<String, Ref<ARCorePlaneAnchor>> m_plane_anchors;
    Ref<ARCorePointCloud> m_point_cloud;
    std::vector<String> m_updated_anchors;
    std::vector<String> m_removed_anchors;

    String m_cloud_anchor_api_key;

    mutable std::mutex m_mutex;

#ifdef XTU_OS_ANDROID
    ArSession* m_ar_session = nullptr;
    ArFrame* m_ar_frame = nullptr;
#endif

public:
    static ARCoreInterface* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ARCoreInterface"); }

    ARCoreInterface() {
        s_singleton = this;
        m_point_cloud.instance();
    }

    ~ARCoreInterface() {
        if (m_session_running) stop_session();
        s_singleton = nullptr;
    }

    StringName get_name() const override { return StringName("ARCore"); }

    bool initialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

#ifdef XTU_OS_ANDROID
        ArStatus status = ArSession_create(nullptr, nullptr, &m_ar_session);
        if (status != AR_SUCCESS) return false;

        ArConfig* config = nullptr;
        ArConfig_create(m_ar_session, &config);
        ArConfig_setLightEstimationMode(m_ar_session, config, AR_LIGHT_ESTIMATION_MODE_AMBIENT_INTENSITY);
        ArSession_configure(m_ar_session, config);
        ArConfig_destroy(config);

        m_initialized = true;
        return true;
#else
        return false;
#endif
    }

    void uninitialize() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_session_running) {
            stop_session();
        }
#ifdef XTU_OS_ANDROID
        if (m_ar_session) {
            ArSession_destroy(m_ar_session);
            m_ar_session = nullptr;
        }
#endif
        m_initialized = false;
    }

    bool is_initialized() const override { return m_initialized; }

    // #########################################################################
    // Session management
    // #########################################################################
    void start_session() {
        if (m_session_running) return;
#ifdef XTU_OS_ANDROID
        ArStatus status = ArSession_resume(m_ar_session);
        if (status == AR_SUCCESS) {
            m_session_running = true;
            emit_signal("session_started");
        }
#endif
    }

    void pause_session() {
#ifdef XTU_OS_ANDROID
        ArSession_pause(m_ar_session);
#endif
        m_session_running = false;
    }

    void stop_session() {
        pause_session();
        m_plane_anchors.clear();
        emit_signal("session_ended");
    }

    void reset_session() {
        stop_session();
        start_session();
    }

    bool is_session_running() const { return m_session_running; }

    // #########################################################################
    // Configuration
    // #########################################################################
    void set_plane_detection_mode(ARCorePlaneDetectionMode mode) {
        m_plane_detection_mode = mode;
        apply_configuration();
    }

    ARCorePlaneDetectionMode get_plane_detection_mode() const { return m_plane_detection_mode; }

    void set_light_estimation_enabled(bool enabled) {
        m_light_estimation_enabled = enabled;
        apply_configuration();
    }

    bool is_light_estimation_enabled() const { return m_light_estimation_enabled; }

    void set_depth_enabled(bool enabled) {
        m_depth_enabled = enabled;
        apply_configuration();
    }

    bool is_depth_enabled() const { return m_depth_enabled; }

    void set_instant_placement_enabled(bool enabled) {
        m_instant_placement_enabled = enabled;
        apply_configuration();
    }

    bool is_instant_placement_enabled() const { return m_instant_placement_enabled; }

    void set_cloud_anchors_enabled(bool enabled) { m_cloud_anchors_enabled = enabled; }
    bool is_cloud_anchors_enabled() const { return m_cloud_anchors_enabled; }

    void set_cloud_anchor_api_key(const String& key) { m_cloud_anchor_api_key = key; }
    String get_cloud_anchor_api_key() const { return m_cloud_anchor_api_key; }

    // #########################################################################
    // Tracking state
    // #########################################################################
    ARCoreTrackingState get_tracking_state() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_tracking_state;
    }

    ARCoreTrackingFailureReason get_failure_reason() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_failure_reason;
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
    ARCoreLightEstimate get_light_estimate() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_light_estimate;
    }

    // #########################################################################
    // Plane anchors
    // #########################################################################
    std::vector<Ref<ARCorePlaneAnchor>> get_plane_anchors() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Ref<ARCorePlaneAnchor>> result;
        for (const auto& kv : m_plane_anchors) {
            result.push_back(kv.second);
        }
        return result;
    }

    Ref<ARCorePlaneAnchor> get_plane_anchor(const String& identifier) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_plane_anchors.find(identifier);
        return it != m_plane_anchors.end() ? it->second : Ref<ARCorePlaneAnchor>();
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
    // Point cloud
    // #########################################################################
    Ref<ARCorePointCloud> get_point_cloud() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_point_cloud;
    }

    // #########################################################################
    // Hit testing
    // #########################################################################
    std::vector<Dictionary> hit_test(const vec2f& screen_point) {
        std::vector<Dictionary> results;
#ifdef XTU_OS_ANDROID
        if (!m_ar_frame) return results;

        ArHitResultList* hit_result_list = nullptr;
        ArHitResultList_create(m_ar_session, &hit_result_list);
        ArFrame_hitTest(m_ar_session, m_ar_frame, screen_point.x(), screen_point.y(), hit_result_list);

        int32_t hit_count = 0;
        ArHitResultList_getSize(m_ar_session, hit_result_list, &hit_count);

        for (int i = 0; i < hit_count; ++i) {
            ArHitResult* hit_result = nullptr;
            ArHitResult_create(m_ar_session, &hit_result);
            ArHitResultList_getItem(m_ar_session, hit_result_list, i, hit_result);

            ArPose* pose = nullptr;
            ArPose_create(m_ar_session, nullptr, &pose);
            ArHitResult_getHitPose(m_ar_session, hit_result, pose);

            float raw[7];
            ArPose_getPoseRaw(m_ar_session, pose, raw);
            mat4f transform = pose_to_mat4(raw);

            ArTrackable* trackable = nullptr;
            ArHitResult_acquireTrackable(m_ar_session, hit_result, &trackable);

            Dictionary dict;
            dict["transform"] = transform;
            dict["distance"] = ArHitResult_getDistance(m_ar_session, hit_result);
            results.push_back(dict);

            ArTrackable_release(trackable);
            ArPose_destroy(pose);
            ArHitResult_destroy(hit_result);
        }

        ArHitResultList_destroy(hit_result_list);
#endif
        return results;
    }

    // #########################################################################
    // Cloud anchors
    // #########################################################################
    String host_cloud_anchor(const mat4f& pose) {
        // Host anchor to ARCore Cloud
        return String();
    }

    ARCoreCloudAnchorState resolve_cloud_anchor(const String& cloud_id) {
        // Resolve cloud anchor
        return ARCoreCloudAnchorState::STATE_NONE;
    }

    // #########################################################################
    // Depth API
    // #########################################################################
    std::vector<ARCoreDepthPoint> get_depth_points() {
        std::vector<ARCoreDepthPoint> result;
        return result;
    }

    float get_depth_at_point(const vec2f& screen_point) {
        return 0.0f;
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
#ifdef XTU_OS_ANDROID
        if (!m_ar_session || !m_session_running) return;

        if (m_ar_frame) {
            ArFrame_destroy(m_ar_frame);
        }
        ArFrame_create(m_ar_session, &m_ar_frame);
        ArSession_update(m_ar_session, m_ar_frame);

        ArTrackingState tracking_state;
        ArCamera* camera = nullptr;
        ArFrame_acquireCamera(m_ar_session, m_ar_frame, &camera);
        ArCamera_getTrackingState(m_ar_session, camera, &tracking_state);

        std::lock_guard<std::mutex> lock(m_mutex);
        m_tracking_state = convert_tracking_state(tracking_state);

        if (tracking_state == AR_TRACKING_STATE_TRACKING) {
            ArPose* pose = nullptr;
            ArPose_create(m_ar_session, nullptr, &pose);
            ArCamera_getDisplayOrientedPose(m_ar_session, camera, pose);
            float raw[7];
            ArPose_getPoseRaw(m_ar_session, pose, raw);
            m_camera_transform = pose_to_mat4(raw);
            ArPose_destroy(pose);

            update_light_estimate(camera);
            update_planes();
            update_point_cloud();
        }

        ArCamera_release(camera);
#endif
    }

private:
    void apply_configuration() {
#ifdef XTU_OS_ANDROID
        if (!m_ar_session) return;

        ArConfig* config = nullptr;
        ArConfig_create(m_ar_session, &config);
        ArSession_getConfig(m_ar_session, config);

        ArPlaneFindingMode plane_mode = AR_PLANE_FINDING_MODE_DISABLED;
        switch (m_plane_detection_mode) {
            case ARCorePlaneDetectionMode::MODE_HORIZONTAL:
                plane_mode = AR_PLANE_FINDING_MODE_HORIZONTAL;
                break;
            case ARCorePlaneDetectionMode::MODE_VERTICAL:
                plane_mode = AR_PLANE_FINDING_MODE_VERTICAL;
                break;
            case ARCorePlaneDetectionMode::MODE_HORIZONTAL_AND_VERTICAL:
                plane_mode = AR_PLANE_FINDING_MODE_HORIZONTAL_AND_VERTICAL;
                break;
            default:
                break;
        }
        ArConfig_setPlaneFindingMode(m_ar_session, config, plane_mode);

        ArLightEstimationMode light_mode = m_light_estimation_enabled ?
            AR_LIGHT_ESTIMATION_MODE_AMBIENT_INTENSITY : AR_LIGHT_ESTIMATION_MODE_DISABLED;
        ArConfig_setLightEstimationMode(m_ar_session, config, light_mode);

        if (m_depth_enabled) {
            ArConfig_setDepthMode(m_ar_session, config, AR_DEPTH_MODE_AUTOMATIC);
        }

        if (m_instant_placement_enabled) {
            ArConfig_setInstantPlacementMode(m_ar_session, config, AR_INSTANT_PLACEMENT_MODE_LOCAL_Y_UP);
        }

        ArSession_configure(m_ar_session, config);
        ArConfig_destroy(config);
#endif
    }

    ARCoreTrackingState convert_tracking_state(ArTrackingState state) {
        switch (state) {
            case AR_TRACKING_STATE_TRACKING: return ARCoreTrackingState::STATE_TRACKING;
            case AR_TRACKING_STATE_PAUSED: return ARCoreTrackingState::STATE_PAUSED;
            default: return ARCoreTrackingState::STATE_STOPPED;
        }
    }

    mat4f pose_to_mat4(const float raw[7]) {
        quatf rot(raw[0], raw[1], raw[2], raw[3]);
        vec3f pos(raw[4], raw[5], raw[6]);
        return translate(mat4f::identity(), pos) * rotate(mat4f::identity(), rot);
    }

    void update_light_estimate(ArCamera* camera) {
#ifdef XTU_OS_ANDROID
        ArLightEstimate* light_estimate = nullptr;
        ArLightEstimate_create(m_ar_session, &light_estimate);
        ArFrame_getLightEstimate(m_ar_session, m_ar_frame, light_estimate);

        ArLightEstimateState state;
        ArLightEstimate_getState(m_ar_session, light_estimate, &state);

        if (state == AR_LIGHT_ESTIMATE_STATE_VALID) {
            ArLightEstimate_getAmbientIntensity(m_ar_session, light_estimate, &m_light_estimate.ambient_intensity);
            float color[3];
            ArLightEstimate_getColorCorrection(m_ar_session, light_estimate, color);
            m_light_estimate.ambient_color = vec3f(color[0], color[1], color[2]);
            m_light_estimate.valid = true;
        } else {
            m_light_estimate.valid = false;
        }

        ArLightEstimate_destroy(light_estimate);
#endif
    }

    void update_planes() {
#ifdef XTU_OS_ANDROID
        ArTrackableList* plane_list = nullptr;
        ArTrackableList_create(m_ar_session, &plane_list);
        ArFrame_getUpdatedTrackables(m_ar_session, m_ar_frame, AR_TRACKABLE_PLANE, plane_list);

        int32_t list_size;
        ArTrackableList_getSize(m_ar_session, plane_list, &list_size);

        for (int i = 0; i < list_size; ++i) {
            ArTrackable* trackable = nullptr;
            ArTrackableList_acquireItem(m_ar_session, plane_list, i, &trackable);
            ArPlane* plane = ArAsPlane(trackable);

            ArTrackingState tracking_state;
            ArTrackable_getTrackingState(m_ar_session, trackable, &tracking_state);

            if (tracking_state == AR_TRACKING_STATE_STOPPED) {
                // Remove plane
                String identifier = String::num(ArTrackable_getHashCode(m_ar_session, trackable));
                _on_plane_removed(identifier);
            } else {
                update_plane_data(plane);
            }

            ArTrackable_release(trackable);
        }

        ArTrackableList_destroy(plane_list);
#endif
    }

    void update_plane_data(ArPlane* plane) {
        String identifier = String::num(ArTrackable_getHashCode(m_ar_session, ArAsTrackable(plane)));

        ArPose* pose = nullptr;
        ArPose_create(m_ar_session, nullptr, &pose);
        ArPlane_getCenterPose(m_ar_session, plane, pose);
        float raw[7];
        ArPose_getPoseRaw(m_ar_session, pose, raw);
        mat4f transform = pose_to_mat4(raw);
        ArPose_destroy(pose);

        ArPlaneType plane_type;
        ArPlane_getType(m_ar_session, plane, &plane_type);
        ARCorePlaneType type = static_cast<ARCorePlaneType>(plane_type);

        float center_x, center_y, center_z;
        ArPlane_getCenter(m_ar_session, plane, &center_x, &center_y, &center_z);
        vec3f center(center_x, center_y, center_z);

        float extent_x, extent_z;
        ArPlane_getExtentX(m_ar_session, plane, &extent_x);
        ArPlane_getExtentZ(m_ar_session, plane, &extent_z);
        vec3f extent(extent_x, 0, extent_z);

        auto it = m_plane_anchors.find(identifier);
        if (it == m_plane_anchors.end()) {
            _on_plane_added(identifier, transform, type, center, extent);
        } else {
            _on_plane_updated(identifier, transform, center, extent);
        }
    }

    void update_point_cloud() {
#ifdef XTU_OS_ANDROID
        ArPointCloud* point_cloud = nullptr;
        ArFrame_acquirePointCloud(m_ar_session, m_ar_frame, &point_cloud);

        int32_t point_count;
        ArPointCloud_getNumberOfPoints(m_ar_session, point_cloud, &point_count);

        std::vector<vec4f> points(point_count);
        if (point_count > 0) {
            const float* point_data = nullptr;
            ArPointCloud_getData(m_ar_session, point_cloud, &point_data);

            for (int i = 0; i < point_count; ++i) {
                points[i] = vec4f(point_data[i * 4], point_data[i * 4 + 1],
                                  point_data[i * 4 + 2], point_data[i * 4 + 3]);
            }

            int64_t timestamp;
            ArPointCloud_getTimestamp(m_ar_session, point_cloud, &timestamp);
            m_point_cloud->set_points(points);
            m_point_cloud->set_timestamp(static_cast<uint64_t>(timestamp));
        }

        ArPointCloud_release(point_cloud);
#endif
    }

    void _on_plane_added(const String& identifier, const mat4f& transform,
                          ARCorePlaneType type, const vec3f& center, const vec3f& extent) {
        std::lock_guard<std::mutex> lock(m_mutex);
        Ref<ARCorePlaneAnchor> anchor;
        anchor.instance();
        anchor->set_identifier(identifier);
        anchor->set_transform(transform);
        anchor->set_type(type);
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
};

} // namespace arcore

// Bring into main namespace
using arcore::ARCoreInterface;
using arcore::ARCoreAnchor;
using arcore::ARCorePlaneAnchor;
using arcore::ARCorePointCloud;
using arcore::ARCoreTrackingState;
using arcore::ARCoreTrackingFailureReason;
using arcore::ARCorePlaneType;
using arcore::ARCorePlaneDetectionMode;
using arcore::ARCoreCloudAnchorState;
using arcore::ARCoreLightEstimate;
using arcore::ARCoreDepthPoint;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XARCORE_HPP