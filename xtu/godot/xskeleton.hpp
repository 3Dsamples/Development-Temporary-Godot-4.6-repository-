// include/xtu/godot/xskeleton.hpp
// xtensor-unified - Skeleton and Skinning for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XSKELETON_HPP
#define XTU_GODOT_XSKELETON_HPP

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
#include "xtu/godot/xanimation.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/graphics/xmesh.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class Skeleton3D;
class Skin;
class SkinReference;
class BoneAttachment3D;
class SkeletonModifier3D;
class PhysicalBone3D;

// #############################################################################
// Skin - Skinning data resource
// #############################################################################
class Skin : public Resource {
    XTU_GODOT_REGISTER_CLASS(Skin, Resource)

public:
    struct BoneBind {
        StringName bone_name;
        mat4f inverse_bind_matrix;
    };

private:
    std::vector<BoneBind> m_binds;
    std::unordered_map<StringName, size_t> m_bone_index;

public:
    static StringName get_class_static() { return StringName("Skin"); }

    void add_bind(const StringName& bone_name, const mat4f& inverse_bind) {
        m_bone_index[bone_name] = m_binds.size();
        m_binds.push_back({bone_name, inverse_bind});
    }

    void remove_bind(int idx) {
        if (idx >= 0 && idx < static_cast<int>(m_binds.size())) {
            m_bone_index.erase(m_binds[idx].bone_name);
            m_binds.erase(m_binds.begin() + idx);
            rebuild_index();
        }
    }

    int get_bind_count() const { return static_cast<int>(m_binds.size()); }

    const BoneBind& get_bind(int idx) const { return m_binds[idx]; }

    int find_bone(const StringName& bone_name) const {
        auto it = m_bone_index.find(bone_name);
        return it != m_bone_index.end() ? static_cast<int>(it->second) : -1;
    }

    void clear() {
        m_binds.clear();
        m_bone_index.clear();
    }

    void set_bind_pose(const Skeleton3D* skeleton) {
        if (!skeleton) return;
        m_binds.clear();
        for (int i = 0; i < skeleton->get_bone_count(); ++i) {
            StringName name = skeleton->get_bone_name(i);
            mat4f rest = skeleton->get_bone_rest(i);
            mat4f global_rest = skeleton->get_bone_global_rest(i);
            mat4f inv_bind = (global_rest * rest).affine_inverse();
            add_bind(name, inv_bind);
        }
    }

private:
    void rebuild_index() {
        m_bone_index.clear();
        for (size_t i = 0; i < m_binds.size(); ++i) {
            m_bone_index[m_binds[i].bone_name] = i;
        }
    }
};

// #############################################################################
// Skeleton3D - Skeletal hierarchy node
// #############################################################################
class Skeleton3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(Skeleton3D, Node3D)

public:
    enum AnimationUpdateMode {
        UPDATE_CONTINUOUS,
        UPDATE_DISCRETE,
        UPDATE_CAPTURE
    };

private:
    struct Bone {
        StringName name;
        int parent = -1;
        mat4f rest;
        mat4f pose;
        mat4f custom_pose;
        mat4f global_pose;
        mat4f global_rest;
        bool enabled = true;
        float physical_bone_radius = 0.1f;
    };

    std::vector<Bone> m_bones;
    std::unordered_map<StringName, int> m_bone_name_map;
    AnimationUpdateMode m_update_mode = UPDATE_CONTINUOUS;
    bool m_animate_physical_bones = true;
    bool m_reset_physical_bones_on_start = true;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("Skeleton3D"); }

    int add_bone(const StringName& name, int parent = -1, const mat4f& rest = mat4f::identity()) {
        std::lock_guard<std::mutex> lock(m_mutex);
        int idx = static_cast<int>(m_bones.size());
        Bone bone;
        bone.name = name;
        bone.parent = parent;
        bone.rest = rest;
        bone.pose = rest;
        bone.custom_pose = rest;
        bone.global_pose = rest;
        bone.global_rest = rest;
        m_bones.push_back(bone);
        m_bone_name_map[name] = idx;
        update_global_poses();
        return idx;
    }

    void remove_bone(int idx) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        m_bone_name_map.erase(m_bones[idx].name);
        // Update children's parent indices
        for (size_t i = 0; i < m_bones.size(); ++i) {
            if (m_bones[i].parent == idx) {
                m_bones[i].parent = m_bones[idx].parent;
            } else if (m_bones[i].parent > idx) {
                --m_bones[i].parent;
            }
        }
        m_bones.erase(m_bones.begin() + idx);
        rebuild_name_map();
        update_global_poses();
    }

    int get_bone_count() const { return static_cast<int>(m_bones.size()); }

    int find_bone(const StringName& name) const {
        auto it = m_bone_name_map.find(name);
        return it != m_bone_name_map.end() ? it->second : -1;
    }

    StringName get_bone_name(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) ? m_bones[idx].name : StringName();
    }

    int get_bone_parent(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) ? m_bones[idx].parent : -1;
    }

    void set_bone_rest(int idx, const mat4f& rest) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        m_bones[idx].rest = rest;
        update_global_rest();
        update_global_poses();
    }

    mat4f get_bone_rest(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) ? m_bones[idx].rest : mat4f::identity();
    }

    void set_bone_pose(int idx, const mat4f& pose) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        m_bones[idx].pose = pose;
    }

    mat4f get_bone_pose(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) ? m_bones[idx].pose : mat4f::identity();
    }

    void set_bone_custom_pose(int idx, const mat4f& pose) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        m_bones[idx].custom_pose = pose;
    }

    mat4f get_bone_custom_pose(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) ? m_bones[idx].custom_pose : mat4f::identity();
    }

    void set_bone_enabled(int idx, bool enabled) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        m_bones[idx].enabled = enabled;
    }

    bool is_bone_enabled(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) && m_bones[idx].enabled;
    }

    mat4f get_bone_global_pose(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) ? m_bones[idx].global_pose : mat4f::identity();
    }

    mat4f get_bone_global_rest(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_bones.size()) ? m_bones[idx].global_rest : mat4f::identity();
    }

    void update_global_poses() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (size_t i = 0; i < m_bones.size(); ++i) {
            mat4f local_pose = m_bones[i].pose * m_bones[i].custom_pose;
            if (m_bones[i].parent >= 0) {
                m_bones[i].global_pose = m_bones[m_bones[i].parent].global_pose * local_pose;
            } else {
                m_bones[i].global_pose = get_global_transform() * local_pose;
            }
        }
    }

    void update_global_rest() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (size_t i = 0; i < m_bones.size(); ++i) {
            if (m_bones[i].parent >= 0) {
                m_bones[i].global_rest = m_bones[m_bones[i].parent].global_rest * m_bones[i].rest;
            } else {
                m_bones[i].global_rest = get_global_transform() * m_bones[i].rest;
            }
        }
    }

    void localize_rests() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (size_t i = 0; i < m_bones.size(); ++i) {
            if (m_bones[i].parent >= 0) {
                m_bones[i].rest = m_bones[m_bones[i].parent].global_rest.affine_inverse() * m_bones[i].global_rest;
            } else {
                m_bones[i].rest = get_global_transform().affine_inverse() * m_bones[i].global_rest;
            }
        }
    }

    void reset_bone_poses() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& bone : m_bones) {
            bone.pose = bone.rest;
            bone.custom_pose = mat4f::identity();
        }
        update_global_poses();
    }

    void set_bone_pose_position(int idx, const vec3f& pos) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        m_bones[idx].pose = translate(m_bones[idx].pose, pos);
    }

    void set_bone_pose_rotation(int idx, const quatf& rot) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        mat4f t = m_bones[idx].pose;
        vec3f pos(t[3][0], t[3][1], t[3][2]);
        m_bones[idx].pose = translate(mat4f::identity(), pos) * rotate(mat4f::identity(), rot);
    }

    void set_bone_pose_scale(int idx, const vec3f& scale) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx < 0 || idx >= static_cast<int>(m_bones.size())) return;
        mat4f t = m_bones[idx].pose;
        vec3f pos(t[3][0], t[3][1], t[3][2]);
        quatf rot = quatf::from_matrix(t);
        m_bones[idx].pose = translate(mat4f::identity(), pos) * rotate(mat4f::identity(), rot) * scale(mat4f::identity(), scale);
    }

    vec3f get_bone_pose_position(int idx) const {
        mat4f pose = get_bone_pose(idx);
        return vec3f(pose[3][0], pose[3][1], pose[3][2]);
    }

    quatf get_bone_pose_rotation(int idx) const {
        return quatf::from_matrix(get_bone_pose(idx));
    }

    vec3f get_bone_pose_scale(int idx) const {
        mat4f pose = get_bone_pose(idx);
        return vec3f(
            vec3f(pose[0][0], pose[0][1], pose[0][2]).length(),
            vec3f(pose[1][0], pose[1][1], pose[1][2]).length(),
            vec3f(pose[2][0], pose[2][1], pose[2][2]).length()
        );
    }

    void clear_bones() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_bones.clear();
        m_bone_name_map.clear();
    }

    void set_animate_physical_bones(bool animate) { m_animate_physical_bones = animate; }
    bool get_animate_physical_bones() const { return m_animate_physical_bones; }

    void _process(double delta) override {
        if (m_update_mode == UPDATE_CONTINUOUS) {
            update_global_poses();
        }
    }

    void _physics_process(double delta) override {
        if (m_update_mode == UPDATE_DISCRETE) {
            update_global_poses();
        }
    }

    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
            update_global_poses();
        }
    }

private:
    void rebuild_name_map() {
        m_bone_name_map.clear();
        for (size_t i = 0; i < m_bones.size(); ++i) {
            m_bone_name_map[m_bones[i].name] = static_cast<int>(i);
        }
    }
};

// #############################################################################
// BoneAttachment3D - Attach nodes to bones
// #############################################################################
class BoneAttachment3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(BoneAttachment3D, Node3D)

private:
    NodePath m_skeleton_path;
    int m_bone_idx = -1;
    StringName m_bone_name;
    bool m_use_external_skeleton = false;
    NodePath m_external_skeleton_path;

public:
    static StringName get_class_static() { return StringName("BoneAttachment3D"); }

    void set_skeleton_path(const NodePath& path) { m_skeleton_path = path; }
    NodePath get_skeleton_path() const { return m_skeleton_path; }

    void set_bone_name(const StringName& name) { m_bone_name = name; }
    StringName get_bone_name() const { return m_bone_name; }

    void set_use_external_skeleton(bool use) { m_use_external_skeleton = use; }
    bool get_use_external_skeleton() const { return m_use_external_skeleton; }

    void set_external_skeleton(const NodePath& path) { m_external_skeleton_path = path; }
    NodePath get_external_skeleton() const { return m_external_skeleton_path; }

    void _process(double delta) override {
        Skeleton3D* skeleton = nullptr;
        if (m_use_external_skeleton) {
            skeleton = dynamic_cast<Skeleton3D*>(get_node_or_null(m_external_skeleton_path));
        } else {
            Node* parent = get_parent();
            while (parent) {
                skeleton = dynamic_cast<Skeleton3D*>(parent);
                if (skeleton) break;
                parent = parent->get_parent();
            }
        }

        if (!skeleton) return;

        if (m_bone_idx < 0 || skeleton->get_bone_name(m_bone_idx) != m_bone_name) {
            m_bone_idx = skeleton->find_bone(m_bone_name);
        }

        if (m_bone_idx >= 0) {
            set_global_transform(skeleton->get_bone_global_pose(m_bone_idx));
        }
    }
};

// #############################################################################
// PhysicalBone3D - Physics-driven bone
// #############################################################################
class PhysicalBone3D : public PhysicsBody3DNode {
    XTU_GODOT_REGISTER_CLASS(PhysicalBone3D, PhysicsBody3DNode)

private:
    NodePath m_skeleton_path;
    int m_bone_idx = -1;
    StringName m_bone_name;
    float m_bone_offset = 0.0f;
    float m_joint_rotation_spring_stiffness = 10.0f;
    float m_joint_rotation_spring_damping = 2.0f;
    float m_joint_linear_spring_stiffness = 10.0f;
    float m_joint_linear_spring_damping = 2.0f;
    float m_body_offset = 0.0f;
    bool m_simulate = true;
    bool m_static_body = false;

public:
    static StringName get_class_static() { return StringName("PhysicalBone3D"); }

    void set_bone_name(const StringName& name) { m_bone_name = name; }
    StringName get_bone_name() const { return m_bone_name; }

    void set_simulate(bool simulate) { m_simulate = simulate; }
    bool is_simulated() const { return m_simulate; }

    void set_static_body(bool static_body) { m_static_body = static_body; }
    bool is_static_body() const { return m_static_body; }

    void set_joint_rotation_stiffness(float stiffness) { m_joint_rotation_spring_stiffness = stiffness; }
    float get_joint_rotation_stiffness() const { return m_joint_rotation_spring_stiffness; }

    void set_joint_rotation_damping(float damping) { m_joint_rotation_spring_damping = damping; }
    float get_joint_rotation_damping() const { return m_joint_rotation_spring_damping; }

    void _ready() override {
        PhysicsBody3DNode::_ready();
        find_skeleton_and_bone();
    }

    void _physics_process(double delta) override {
        if (!m_simulate || m_static_body) return;

        Skeleton3D* skeleton = find_skeleton();
        if (!skeleton || m_bone_idx < 0) return;

        mat4f bone_pose = skeleton->get_bone_global_pose(m_bone_idx);
        mat4f body_transform = get_global_transform();

        // Apply spring forces to match bone pose
        vec3f bone_pos(bone_pose[3][0], bone_pose[3][1], bone_pose[3][2]);
        vec3f body_pos(body_transform[3][0], body_transform[3][1], body_transform[3][2]);

        vec3f displacement = bone_pos - body_pos;
        vec3f spring_force = displacement * m_joint_linear_spring_stiffness;
        vec3f damping_force = -get_linear_velocity() * m_joint_linear_spring_damping;

        apply_central_force(spring_force + damping_force);

        // Rotation spring
        quatf bone_rot = quatf::from_matrix(bone_pose);
        quatf body_rot = quatf::from_matrix(body_transform);
        quatf delta_rot = bone_rot * body_rot.inverse();

        vec3f axis;
        float angle;
        delta_rot.get_axis_angle(axis, angle);
        vec3f torque = axis * angle * m_joint_rotation_spring_stiffness;
        vec3f damping_torque = -get_angular_velocity() * m_joint_rotation_spring_damping;

        apply_torque(torque + damping_torque);
    }

private:
    Skeleton3D* find_skeleton() {
        return dynamic_cast<Skeleton3D*>(get_node_or_null(m_skeleton_path));
    }

    void find_skeleton_and_bone() {
        Node* parent = get_parent();
        while (parent) {
            if (auto* skeleton = dynamic_cast<Skeleton3D*>(parent)) {
                m_skeleton_path = skeleton->get_path();
                m_bone_idx = skeleton->find_bone(m_bone_name);
                break;
            }
            parent = parent->get_parent();
        }
    }
};

// #############################################################################
// SkinnedMeshInstance - Mesh instance with skinning
// #############################################################################
class SkinnedMeshInstance : public MeshInstance3D {
    XTU_GODOT_REGISTER_CLASS(SkinnedMeshInstance, MeshInstance3D)

private:
    Ref<Skin> m_skin;
    NodePath m_skeleton_path;
    Skeleton3D* m_skeleton = nullptr;

public:
    static StringName get_class_static() { return StringName("SkinnedMeshInstance"); }

    void set_skin(const Ref<Skin>& skin) { m_skin = skin; }
    Ref<Skin> get_skin() const { return m_skin; }

    void set_skeleton_path(const NodePath& path) { m_skeleton_path = path; }
    NodePath get_skeleton_path() const { return m_skeleton_path; }

    void _ready() override {
        MeshInstance3D::_ready();
        m_skeleton = dynamic_cast<Skeleton3D*>(get_node_or_null(m_skeleton_path));
    }

    void _process(double delta) override {
        if (!m_skeleton || !m_skin.is_valid() || !get_mesh().is_valid()) return;

        // Update skinning uniforms
        RenderingServer* rs = RenderingServer::get_singleton();
        RID mesh_rid = get_mesh()->get_rid();
        RID skeleton_rid = m_skeleton->get_rid();

        if (m_skin->get_bind_count() > 0) {
            std::vector<mat4f> skinning_matrices;
            skinning_matrices.reserve(m_skin->get_bind_count());

            for (int i = 0; i < m_skin->get_bind_count(); ++i) {
                const Skin::BoneBind& bind = m_skin->get_bind(i);
                int bone_idx = m_skeleton->find_bone(bind.bone_name);
                if (bone_idx >= 0) {
                    mat4f bone_pose = m_skeleton->get_bone_global_pose(bone_idx);
                    skinning_matrices.push_back(bone_pose * bind.inverse_bind_matrix);
                } else {
                    skinning_matrices.push_back(mat4f::identity());
                }
            }

            rs->mesh_set_skinning_data(mesh_rid, skinning_matrices);
        }
    }
};

} // namespace godot

// Bring into main namespace
using godot::Skeleton3D;
using godot::Skin;
using godot::BoneAttachment3D;
using godot::PhysicalBone3D;
using godot::SkinnedMeshInstance;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XSKELETON_HPP