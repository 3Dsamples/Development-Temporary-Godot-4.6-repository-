// skeleton_3d.cpp
#include "skeleton_3d.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace lighting {

struct Skeleton3D::Impl {
    std::vector<BoneData> bones;
    std::unordered_map<std::string, int> name_to_index;
    bool dirty = true;

    // Animation blending (simplified: track name -> weight)
    std::unordered_map<std::string, float> blend_weights;

    // Lighting flags
    bool cast_shadow = true;
    int gi_mode = 2; // dynamic by default for animated skeletons

    // Cached skin matrices array for GPU upload
    std::vector<float> skin_matrix_data; // 4x4 matrices as float[16] per bone

    void compute_global_poses();
    void compute_skin_matrices();
    void update_skin_data_array();
};

Skeleton3D::Skeleton3D() : pimpl(std::make_unique<Impl>()) {}
Skeleton3D::~Skeleton3D() = default;

int Skeleton3D::add_bone(const char* name, int parent_index) {
    int idx = (int)pimpl->bones.size();
    BoneData bone;
    bone.name = name;
    bone.parent = parent_index;
    pimpl->bones.push_back(bone);
    pimpl->name_to_index[name] = idx;
    pimpl->dirty = true;
    return idx;
}

void Skeleton3D::remove_bone(int index) {
    if (index < 0 || index >= (int)pimpl->bones.size()) return;
    pimpl->name_to_index.erase(pimpl->bones[index].name);
    pimpl->bones.erase(pimpl->bones.begin() + index);
    // Renumber remaining bones? For simplicity, we don't reparent.
    pimpl->dirty = true;
}

int Skeleton3D::get_bone_count() const { return (int)pimpl->bones.size(); }
int Skeleton3D::find_bone(const char* name) const {
    auto it = pimpl->name_to_index.find(name);
    return (it != pimpl->name_to_index.end()) ? it->second : -1;
}
const char* Skeleton3D::get_bone_name(int index) const {
    return (index >=0 && index < (int)pimpl->bones.size()) ? pimpl->bones[index].name.c_str() : "";
}
void Skeleton3D::set_bone_parent(int index, int parent_index) {
    if (index >=0 && index < (int)pimpl->bones.size())
        pimpl->bones[index].parent = parent_index;
    pimpl->dirty = true;
}
int Skeleton3D::get_bone_parent(int index) const {
    return (index >=0 && index < (int)pimpl->bones.size()) ? pimpl->bones[index].parent : -1;
}

void Skeleton3D::set_bone_rest_pose(int index, const Transform3D& rest) {
    if (index >=0 && index < (int)pimpl->bones.size())
        pimpl->bones[index].rest_pose = rest;
    pimpl->dirty = true;
}
Transform3D Skeleton3D::get_bone_rest_pose(int index) const {
    if (index >=0 && index < (int)pimpl->bones.size())
        return pimpl->bones[index].rest_pose;
    return Transform3D();
}
void Skeleton3D::set_bone_inverse_bind(int index, const Transform3D& inv_bind) {
    if (index >=0 && index < (int)pimpl->bones.size())
        pimpl->bones[index].inverse_bind = inv_bind;
}
Transform3D Skeleton3D::get_bone_inverse_bind(int index) const {
    if (index >=0 && index < (int)pimpl->bones.size())
        return pimpl->bones[index].inverse_bind;
    return Transform3D();
}

void Skeleton3D::set_bone_local_pose(int index, const Transform3D& pose) {
    if (index >=0 && index < (int)pimpl->bones.size()) {
        pimpl->bones[index].local_pose = pose;
        pimpl->dirty = true;
    }
}
Transform3D Skeleton3D::get_bone_local_pose(int index) const {
    return (index >=0 && index < (int)pimpl->bones.size()) ? pimpl->bones[index].local_pose : Transform3D();
}
void Skeleton3D::set_bone_global_pose(int index, const Transform3D& pose) {
    if (index >=0 && index < (int)pimpl->bones.size()) {
        pimpl->bones[index].global_pose = pose;
        // Mark that we have overridden global pose; need to propagate to children?
        // For simplicity, we keep dirty flag to recompute from parent.
        pimpl->dirty = true;
    }
}
Transform3D Skeleton3D::get_bone_global_pose(int index) const {
    return (index >=0 && index < (int)pimpl->bones.size()) ? pimpl->bones[index].global_pose : Transform3D();
}

Transform3D Skeleton3D::get_bone_skin_matrix(int index) const {
    if (index >=0 && index < (int)pimpl->bones.size())
        return pimpl->bones[index].skin_pose;
    return Transform3D();
}

void Skeleton3D::set_animation_blend(int track, float weight) {
    // In a full implementation, track would be an animation track index.
    // For simplicity, we ignore track.
}
void Skeleton3D::clear_animation_tracks() { pimpl->blend_weights.clear(); }

void Skeleton3D::Impl::compute_global_poses() {
    // Recursively compute from root
    for (size_t i = 0; i < bones.size(); ++i) {
        if (bones[i].parent == -1) {
            bones[i].global_pose = bones[i].local_pose;
            // accumulate with skeleton's own transform? The skeleton node itself provides global transform.
        }
    }
    // Now propagate to children (topological order required). Simple loop multiple times.
    bool changed = true;
    int pass = 0;
    while (changed && pass < (int)bones.size()) {
        changed = false;
        for (size_t i = 0; i < bones.size(); ++i) {
            int parent = bones[i].parent;
            if (parent >= 0 && parent != (int)i) {
                if (bones[parent].global_pose.is_valid()) { // placeholder: need flag
                    bones[i].global_pose = bones[parent].global_pose * bones[i].local_pose;
                    changed = true;
                }
            }
        }
        ++pass;
    }
}

void Skeleton3D::Impl::compute_skin_matrices() {
    for (size_t i = 0; i < bones.size(); ++i) {
        bones[i].skin_pose = bones[i].global_pose * bones[i].inverse_bind;
    }
}

void Skeleton3D::Impl::update_skin_data_array() {
    skin_matrix_data.resize(bones.size() * 16);
    for (size_t i = 0; i < bones.size(); ++i) {
        const double* m = bones[i].skin_pose.matrix; // assuming Transform3D has 16 doubles
        for (int j = 0; j < 16; ++j)
            skin_matrix_data[i*16 + j] = (float)m[j];
    }
}

void Skeleton3D::update_bone_transforms() {
    if (!pimpl->dirty) return;
    pimpl->compute_global_poses();
    pimpl->compute_skin_matrices();
    pimpl->update_skin_data_array();
    pimpl->dirty = false;
}

void Skeleton3D::reset_to_rest_pose() {
    for (size_t i = 0; i < pimpl->bones.size(); ++i) {
        pimpl->bones[i].local_pose = pimpl->bones[i].rest_pose;
    }
    pimpl->dirty = true;
    update_bone_transforms();
}

void Skeleton3D::set_dirty(bool dirty) { pimpl->dirty = dirty; }
bool Skeleton3D::is_dirty() const { return pimpl->dirty; }

void Skeleton3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool Skeleton3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void Skeleton3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int Skeleton3D::get_gi_mode() const { return pimpl->gi_mode; }

void Skeleton3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // Send skin matrices to rendering server for all meshes that use this skeleton.
    if (!pimpl->dirty) return;
    update_bone_transforms(); // ensure latest
    // Placeholder: call RenderingServer::skeleton_update_bones(skeleton_rid, skin_matrix_data.data())
}

} // namespace lighting