// include/xtu/godot/xscene_advanced.hpp
// xtensor-unified - Advanced Scene Nodes for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XSCENE_ADVANCED_HPP
#define XTU_GODOT_XSCENE_ADVANCED_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xphysics3d.hpp"
#include "xtu/godot/xpathfinding.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/graphics/xintersection.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Part 1: Timer - High-precision timer node
// #############################################################################

class Timer : public Node {
    XTU_GODOT_REGISTER_CLASS(Timer, Node)

public:
    enum TimerProcessMode {
        TIMER_PROCESS_IDLE,
        TIMER_PROCESS_PHYSICS
    };

private:
    float m_wait_time = 1.0f;
    bool m_one_shot = false;
    bool m_autostart = false;
    TimerProcessMode m_process_mode = TIMER_PROCESS_IDLE;
    float m_time_left = 0.0f;
    bool m_paused = false;
    bool m_running = false;

public:
    static StringName get_class_static() { return StringName("Timer"); }

    void set_wait_time(float time) { m_wait_time = std::max(0.0f, time); }
    float get_wait_time() const { return m_wait_time; }

    void set_one_shot(bool one_shot) { m_one_shot = one_shot; }
    bool is_one_shot() const { return m_one_shot; }

    void set_autostart(bool autostart) { m_autostart = autostart; }
    bool is_autostart() const { return m_autostart; }

    void set_process_mode(TimerProcessMode mode) { m_process_mode = mode; }
    TimerProcessMode get_process_mode() const { return m_process_mode; }

    void set_paused(bool paused) { m_paused = paused; }
    bool is_paused() const { return m_paused; }

    void start(float time = -1.0f) {
        if (time >= 0.0f) m_wait_time = time;
        m_time_left = m_wait_time;
        m_running = true;
    }

    void stop() {
        m_time_left = 0.0f;
        m_running = false;
    }

    bool is_stopped() const { return !m_running; }
    float get_time_left() const { return m_time_left; }

    void _ready() override {
        if (m_autostart) start();
    }

    void _process(double delta) override {
        if (m_process_mode == TIMER_PROCESS_IDLE && m_running && !m_paused) {
            update_timer(static_cast<float>(delta));
        }
    }

    void _physics_process(double delta) override {
        if (m_process_mode == TIMER_PROCESS_PHYSICS && m_running && !m_paused) {
            update_timer(static_cast<float>(delta));
        }
    }

private:
    void update_timer(float delta) {
        m_time_left -= delta;
        if (m_time_left <= 0.0f) {
            emit_signal("timeout");
            if (m_one_shot) {
                stop();
            } else {
                m_time_left = m_wait_time;
            }
        }
    }
};

// #############################################################################
// Part 2: RemoteTransform3D - Force transform synchronization
// #############################################################################

class RemoteTransform3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(RemoteTransform3D, Node3D)

public:
    enum UpdateMode {
        UPDATE_CONTINUOUS,
        UPDATE_DISCRETE,
        UPDATE_CAPTURE
    };

private:
    NodePath m_remote_path;
    bool m_update_position = true;
    bool m_update_rotation = true;
    bool m_update_scale = true;
    bool m_use_global_coordinates = true;
    UpdateMode m_update_mode = UPDATE_CONTINUOUS;

public:
    static StringName get_class_static() { return StringName("RemoteTransform3D"); }

    void set_remote_node(const NodePath& path) { m_remote_path = path; }
    NodePath get_remote_node() const { return m_remote_path; }

    void set_update_position(bool update) { m_update_position = update; }
    bool get_update_position() const { return m_update_position; }

    void set_update_rotation(bool update) { m_update_rotation = update; }
    bool get_update_rotation() const { return m_update_rotation; }

    void set_update_scale(bool update) { m_update_scale = update; }
    bool get_update_scale() const { return m_update_scale; }

    void set_use_global_coordinates(bool use) { m_use_global_coordinates = use; }
    bool get_use_global_coordinates() const { return m_use_global_coordinates; }

    void set_update_mode(UpdateMode mode) { m_update_mode = mode; }
    UpdateMode get_update_mode() const { return m_update_mode; }

    void force_update() {
        Node* remote = get_node_or_null(m_remote_path);
        Node3D* remote_3d = dynamic_cast<Node3D*>(remote);
        if (!remote_3d) return;

        if (m_use_global_coordinates) {
            if (m_update_position) remote_3d->set_global_position(get_global_position());
            if (m_update_rotation) remote_3d->set_global_rotation(get_global_rotation());
            if (m_update_scale) remote_3d->set_global_scale(get_global_scale());
        } else {
            if (m_update_position) remote_3d->set_position(get_position());
            if (m_update_rotation) remote_3d->set_rotation(get_rotation());
            if (m_update_scale) remote_3d->set_scale(get_scale());
        }
    }

    void _process(double delta) override {
        if (m_update_mode == UPDATE_CONTINUOUS) {
            force_update();
        }
    }

    void _physics_process(double delta) override {
        if (m_update_mode == UPDATE_DISCRETE) {
            force_update();
        }
    }

    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
            force_update();
        }
    }
};

// #############################################################################
// Part 3: VisibleOnScreenNotifier3D - Visibility detection
// #############################################################################

class VisibleOnScreenNotifier3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(VisibleOnScreenNotifier3D, Node3D)

private:
    aabb m_aabb = aabb(vec3f(-1, -1, -1), vec3f(1, 1, 1));
    bool m_enable = true;
    bool m_visible = false;
    int m_max_screen = 0;

public:
    static StringName get_class_static() { return StringName("VisibleOnScreenNotifier3D"); }

    void set_aabb(const aabb& bounds) { m_aabb = bounds; }
    aabb get_aabb() const { return m_aabb; }

    void set_enable(bool enable) { m_enable = enable; }
    bool is_enabled() const { return m_enable; }

    bool is_on_screen() const { return m_visible; }

    void _process(double delta) override {
        if (!m_enable) return;

        Viewport* viewport = get_viewport();
        if (!viewport) return;

        Camera3D* camera = viewport->get_camera_3d();
        if (!camera) return;

        mat4f cam_transform = camera->get_global_transform();
        aabb world_aabb = get_global_transform().xform(m_aabb);

        frustum cam_frustum = frustum(camera->get_camera_projection() * cam_transform.affine_inverse());
        bool now_visible = cam_frustum.intersects(world_aabb);

        if (now_visible != m_visible) {
            m_visible = now_visible;
            if (m_visible) {
                emit_signal("screen_entered");
            } else {
                emit_signal("screen_exited");
            }
        }
    }

    bool is_visible_in_frustum(const frustum& f) const {
        aabb world_aabb = get_global_transform().xform(m_aabb);
        return f.intersects(world_aabb);
    }
};

// #############################################################################
// Part 4: Path3D - 3D curve path
// #############################################################################

class Path3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(Path3D, Node3D)

public:
    struct CurvePoint {
        vec3f position;
        vec3f in_tangent;
        vec3f out_tangent;
    };

private:
    std::vector<CurvePoint> m_curve;
    bool m_baked = false;
    std::vector<vec3f> m_baked_points;
    float m_bake_interval = 0.2f;
    bool m_closed = false;
    float m_curve_length = 0.0f;
    std::vector<float> m_cumulative_lengths;

public:
    static StringName get_class_static() { return StringName("Path3D"); }

    void set_curve(const std::vector<CurvePoint>& points) {
        m_curve = points;
        m_baked = false;
    }

    const std::vector<CurvePoint>& get_curve() const { return m_curve; }

    void add_point(const vec3f& pos, const vec3f& in_tan = vec3f(0), const vec3f& out_tan = vec3f(0)) {
        m_curve.push_back({pos, in_tan, out_tan});
        m_baked = false;
    }

    void remove_point(int idx) {
        if (idx >= 0 && idx < static_cast<int>(m_curve.size())) {
            m_curve.erase(m_curve.begin() + idx);
            m_baked = false;
        }
    }

    void set_closed(bool closed) { m_closed = closed; m_baked = false; }
    bool is_closed() const { return m_closed; }

    void set_bake_interval(float interval) { m_bake_interval = std::max(0.01f, interval); m_baked = false; }
    float get_bake_interval() const { return m_bake_interval; }

    void bake() {
        m_baked_points.clear();
        m_cumulative_lengths.clear();
        if (m_curve.size() < 2) {
            m_curve_length = 0.0f;
            return;
        }

        int segments = m_closed ? static_cast<int>(m_curve.size()) : static_cast<int>(m_curve.size() - 1);
        float total_length = 0.0f;

        for (int i = 0; i < segments; ++i) {
            int idx0 = i;
            int idx1 = (i + 1) % m_curve.size();

            const CurvePoint& p0 = m_curve[idx0];
            const CurvePoint& p1 = m_curve[idx1];

            float segment_length = estimate_curve_length(p0, p1);
            int samples = std::max(2, static_cast<int>(segment_length / m_bake_interval));

            for (int j = 0; j < samples; ++j) {
                float t = static_cast<float>(j) / static_cast<float>(samples - 1);
                vec3f pt = interpolate_hermite(p0.position, p1.position, p0.out_tangent, p1.in_tangent, t);
                m_baked_points.push_back(pt);

                if (j > 0) {
                    float dist = (pt - m_baked_points[m_baked_points.size() - 2]).length();
                    total_length += dist;
                }
                m_cumulative_lengths.push_back(total_length);
            }
        }

        m_curve_length = total_length;
        m_baked = true;
    }

    vec3f sample_point(float offset) const {
        if (!m_baked || m_baked_points.empty()) return vec3f(0);
        if (offset <= 0.0f) return m_baked_points.front();
        if (offset >= m_curve_length) return m_baked_points.back();

        // Binary search for the segment
        auto it = std::lower_bound(m_cumulative_lengths.begin(), m_cumulative_lengths.end(), offset);
        size_t idx = std::distance(m_cumulative_lengths.begin(), it);
        if (idx >= m_baked_points.size()) return m_baked_points.back();
        if (idx == 0) return m_baked_points.front();

        float t0 = m_cumulative_lengths[idx - 1];
        float t1 = m_cumulative_lengths[idx];
        float t = (offset - t0) / (t1 - t0);
        return m_baked_points[idx - 1] * (1.0f - t) + m_baked_points[idx] * t;
    }

    vec3f sample_tangent(float offset) const {
        if (!m_baked || m_baked_points.size() < 2) return vec3f(0, 0, 1);
        float delta = m_curve_length * 0.01f;
        vec3f p1 = sample_point(std::max(0.0f, offset - delta));
        vec3f p2 = sample_point(std::min(m_curve_length, offset + delta));
        return (p2 - p1).normalized();
    }

    float get_curve_length() const { return m_curve_length; }

private:
    float estimate_curve_length(const CurvePoint& p0, const CurvePoint& p1) const {
        return (p0.position - p1.position).length();
    }

    vec3f interpolate_hermite(const vec3f& p0, const vec3f& p1, const vec3f& t0, const vec3f& t1, float t) const {
        float t2 = t * t;
        float t3 = t2 * t;
        float h1 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float h2 = -2.0f * t3 + 3.0f * t2;
        float h3 = t3 - 2.0f * t2 + t;
        float h4 = t3 - t2;
        return p0 * h1 + p1 * h2 + t0 * h3 + t1 * h4;
    }
};

// #############################################################################
// Part 5: PathFollow3D - Follow a Path3D
// #############################################################################

class PathFollow3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(PathFollow3D, Node3D)

public:
    enum RotationMode {
        ROTATION_NONE,
        ROTATION_Y,
        ROTATION_XY,
        ROTATION_XYZ,
        ROTATION_ORIENTED
    };

private:
    NodePath m_path_path;
    Path3D* m_path = nullptr;
    float m_progress = 0.0f;
    float m_progress_ratio = 0.0f;
    float m_h_offset = 0.0f;
    float m_v_offset = 0.0f;
    vec3f m_cubic_interp = vec3f(1, 0, 0);
    bool m_loop = true;
    bool m_use_global = true;
    RotationMode m_rotation_mode = ROTATION_ORIENTED;
    float m_tilt = 0.0f;

public:
    static StringName get_class_static() { return StringName("PathFollow3D"); }

    void set_path_path(const NodePath& path) { m_path_path = path; }
    NodePath get_path_path() const { return m_path_path; }

    void set_progress(float progress) {
        if (m_path) {
            m_progress = std::clamp(progress, 0.0f, m_path->get_curve_length());
            m_progress_ratio = m_progress / m_path->get_curve_length();
            update_transform();
        }
    }

    float get_progress() const { return m_progress; }

    void set_progress_ratio(float ratio) {
        m_progress_ratio = std::clamp(ratio, 0.0f, 1.0f);
        if (m_path) {
            m_progress = m_progress_ratio * m_path->get_curve_length();
            update_transform();
        }
    }

    float get_progress_ratio() const { return m_progress_ratio; }

    void set_h_offset(float offset) { m_h_offset = offset; update_transform(); }
    float get_h_offset() const { return m_h_offset; }

    void set_v_offset(float offset) { m_v_offset = offset; update_transform(); }
    float get_v_offset() const { return m_v_offset; }

    void set_loop(bool loop) { m_loop = loop; }
    bool is_loop() const { return m_loop; }

    void set_rotation_mode(RotationMode mode) { m_rotation_mode = mode; update_transform(); }
    RotationMode get_rotation_mode() const { return m_rotation_mode; }

    void set_tilt(float tilt) { m_tilt = tilt; update_transform(); }
    float get_tilt() const { return m_tilt; }

    void _ready() override {
        Node3D::_ready();
        m_path = dynamic_cast<Path3D*>(get_node_or_null(m_path_path));
        if (m_path && !m_path->is_closed()) {
            m_path->bake();
        }
        update_transform();
    }

    void _process(double delta) override {
        if (m_path) {
            update_transform();
        }
    }

private:
    void update_transform() {
        if (!m_path) return;

        vec3f path_pos = m_path->sample_point(m_progress);
        vec3f tangent = m_path->sample_tangent(m_progress);
        vec3f normal = vec3f(0, 1, 0);
        vec3f binormal = cross(tangent, normal).normalized();
        normal = cross(binormal, tangent).normalized();

        // Apply offsets
        path_pos += normal * m_v_offset + binormal * m_h_offset;

        // Set position
        if (m_use_global) {
            mat4f global_transform = m_path->get_global_transform();
            set_global_position(global_transform.xform(path_pos));
        } else {
            set_position(path_pos);
        }

        // Set rotation
        switch (m_rotation_mode) {
            case ROTATION_NONE:
                break;
            case ROTATION_Y: {
                float angle = std::atan2(tangent.x(), tangent.z());
                set_rotation(quatf(vec3f(0, 1, 0), angle));
                break;
            }
            case ROTATION_ORIENTED: {
                quatf q = quatf::from_basis(binormal, normal, tangent);
                if (m_tilt != 0.0f) {
                    q = q * quatf(tangent, m_tilt);
                }
                set_rotation(q);
                break;
            }
            default:
                break;
        }
    }
};

// #############################################################################
// Part 6: VehicleBody3D - Vehicle physics
// #############################################################################

class VehicleBody3D : public RigidBody3D {
    XTU_GODOT_REGISTER_CLASS(VehicleBody3D, RigidBody3D)

public:
    struct Wheel {
        NodePath wheel_path;
        float suspension_stiffness = 5.88f;
        float suspension_damping = 0.88f;
        float suspension_compression = 0.83f;
        float suspension_max_force = 6000.0f;
        float wheel_radius = 0.5f;
        float wheel_rest_length = 0.15f;
        float wheel_friction_slip = 10.5f;
        float roll_influence = 0.1f;
        vec3f suspension_attach_point;
        bool use_as_steering = false;
        bool use_as_traction = false;
        float engine_force = 0.0f;
        float brake = 0.0f;
        float steering = 0.0f;
    };

private:
    std::vector<Wheel> m_wheels;
    float m_engine_force = 0.0f;
    float m_brake = 0.0f;
    float m_steering = 0.0f;
    float m_mass = 40.0f;
    float m_friction = 10.5f;
    bool m_engine_enabled = true;

public:
    static StringName get_class_static() { return StringName("VehicleBody3D"); }

    void add_wheel(const Wheel& wheel) {
        m_wheels.push_back(wheel);
    }

    void remove_wheel(int idx) {
        if (idx >= 0 && idx < static_cast<int>(m_wheels.size())) {
            m_wheels.erase(m_wheels.begin() + idx);
        }
    }

    Wheel* get_wheel(int idx) {
        return idx >= 0 && idx < static_cast<int>(m_wheels.size()) ? &m_wheels[idx] : nullptr;
    }

    int get_wheel_count() const { return static_cast<int>(m_wheels.size()); }

    void set_engine_force(float force) { m_engine_force = force; }
    float get_engine_force() const { return m_engine_force; }

    void set_brake(float brake) { m_brake = brake; }
    float get_brake() const { return m_brake; }

    void set_steering(float steering) { m_steering = std::clamp(steering, -1.0f, 1.0f); }
    float get_steering() const { return m_steering; }

    void set_mass(float mass) override {
        RigidBody3D::set_mass(mass);
        m_mass = mass;
    }

    void _physics_process(double delta) override {
        if (!m_engine_enabled) return;

        for (auto& wheel : m_wheels) {
            // Apply wheel forces using ray-cast suspension
            update_wheel(wheel, static_cast<float>(delta));
        }
    }

private:
    void update_wheel(Wheel& wheel, float delta) {
        Node3D* wheel_node = dynamic_cast<Node3D*>(get_node_or_null(wheel.wheel_path));
        if (!wheel_node) return;

        vec3f attach_point = get_global_transform().xform(wheel.suspension_attach_point);
        vec3f ray_dir = -get_global_transform().basis.y();

        PhysicsDirectSpaceState3D* space = PhysicsServer3D::get_singleton()->space_get_direct_state(get_world_3d()->get_space());
        if (!space) return;

        float ray_length = wheel.wheel_radius + wheel.suspension_rest_length;
        PhysicsDirectSpaceState3D::RayResult result;

        if (space->intersect_ray(attach_point, attach_point + ray_dir * ray_length, m_collision_mask, true, false, result)) {
            float distance = (attach_point - result.position).length();
            float compression = wheel.suspension_rest_length - (distance - wheel.wheel_radius);
            compression = std::max(0.0f, compression);

            // Suspension force
            float suspension_force = compression * wheel.suspension_stiffness;
            vec3f force = result.normal * suspension_force;
            apply_force(force, attach_point - get_global_position());

            // Damping
            vec3f velocity_at_wheel = get_linear_velocity() + cross(get_angular_velocity(), attach_point - get_global_position());
            float compression_velocity = dot(velocity_at_wheel, -ray_dir);
            float damping_force = compression_velocity * wheel.suspension_damping;
            apply_force(result.normal * damping_force, attach_point - get_global_position());

            // Update wheel position
            vec3f wheel_pos = result.position + ray_dir * wheel.wheel_radius;
            wheel_node->set_global_position(wheel_pos);

            // Traction force
            if (wheel.use_as_traction) {
                vec3f traction_dir = get_global_transform().basis.x();
                vec3f traction_force = traction_dir * (wheel.engine_force > 0 ? m_engine_force : wheel.engine_force);
                apply_force(traction_force, attach_point - get_global_position());
            }

            // Steering
            if (wheel.use_as_steering) {
                float steering_angle = m_steering * wheel.steering * 0.5f;
                wheel_node->set_rotation(quatf(vec3f(0, 1, 0), steering_angle));
            }
        }
    }
};

// #############################################################################
// Part 7: WorldEnvironment enhancements
// #############################################################################

class WorldEnvironmentAdvanced : public WorldEnvironment {
    XTU_GODOT_REGISTER_CLASS(WorldEnvironmentAdvanced, WorldEnvironment)

private:
    Ref<Environment> m_environment;
    Ref<CameraAttributes> m_camera_attributes;

public:
    static StringName get_class_static() { return StringName("WorldEnvironmentAdvanced"); }

    void set_environment(const Ref<Environment>& env) override {
        m_environment = env;
        apply_environment();
    }

    Ref<Environment> get_environment() const override { return m_environment; }

    void set_camera_attributes(const Ref<CameraAttributes>& attr) {
        m_camera_attributes = attr;
    }

    Ref<CameraAttributes> get_camera_attributes() const { return m_camera_attributes; }

    void apply_environment() {
        if (!m_environment.is_valid()) return;

        RenderingServer* rs = RenderingServer::get_singleton();
        RID scenario = get_world_3d()->get_scenario();

        // Apply environment settings
        rs->scenario_set_environment(scenario, m_environment->get_rid());

        // Apply fog
        if (m_environment->is_fog_enabled()) {
            rs->scenario_set_fog_enabled(scenario, true);
            rs->scenario_set_fog_mode(scenario, static_cast<RenderingServer::FogMode>(m_environment->get_fog_mode()));
            rs->scenario_set_fog_density(scenario, m_environment->get_fog_density());
            rs->scenario_set_fog_light_color(scenario, m_environment->get_fog_light_color());
            rs->scenario_set_fog_height(scenario, m_environment->get_fog_height());
            rs->scenario_set_fog_height_density(scenario, m_environment->get_fog_height_density());
        }

        // Apply SSAO
        if (m_environment->is_ssao_enabled()) {
            rs->scenario_set_ssao_enabled(scenario, true);
            rs->scenario_set_ssao_radius(scenario, m_environment->get_ssao_radius());
            rs->scenario_set_ssao_intensity(scenario, m_environment->get_ssao_intensity());
            rs->scenario_set_ssao_quality(scenario, static_cast<RenderingServer::EnvironmentSSAOQuality>(m_environment->get_ssao_quality()));
        }

        // Apply glow
        if (m_environment->is_glow_enabled()) {
            rs->scenario_set_glow_enabled(scenario, true);
            rs->scenario_set_glow_intensity(scenario, m_environment->get_glow_intensity());
            rs->scenario_set_glow_blend_mode(scenario, static_cast<RenderingServer::GlowBlendMode>(m_environment->get_glow_blend_mode()));
        }

        // Apply tonemap
        rs->scenario_set_tonemapper(scenario, static_cast<RenderingServer::ToneMapper>(m_environment->get_tonemapper()));
        rs->scenario_set_tonemap_exposure(scenario, m_environment->get_tonemap_exposure());
    }

    void _enter_tree() override {
        WorldEnvironment::_enter_tree();
        apply_environment();
    }
};

} // namespace godot

// Bring into main namespace
using godot::Timer;
using godot::RemoteTransform3D;
using godot::VisibleOnScreenNotifier3D;
using godot::Path3D;
using godot::PathFollow3D;
using godot::VehicleBody3D;
using godot::WorldEnvironmentAdvanced;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XSCENE_ADVANCED_HPP