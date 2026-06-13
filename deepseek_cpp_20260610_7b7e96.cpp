// bone_attachment_3d.cpp
#include "bone_attachment_3d.h"
#include <cstring>
#include <cmath>

namespace lighting {

struct BoneAttachment3D::Impl {
    char skeleton_path[256] = {0};
    char bone_name[64] = {0};
    int bone_index = -1;             // -1 = not set, otherwise index
    bool use_external_skeleton = false;
    Transform3D relative_offset;     // local offset relative to bone
    Transform3D absolute_offset;     // global offset (used only if non‑identity)
    bool cast_shadow = true;
    int gi_mode = 2;                 // dynamic by default
    // cached pointers and state
    Skeleton3D* skeleton = nullptr;
    bool dirty = true;
    Transform3D last_bone_pose;      // for detecting changes
};

BoneAttachment3D::BoneAttachment3D() : pimpl(std::make_unique<Impl>()) {
    pimpl->relative_offset.set_identity();
    pimpl->absolute_offset.set_identity();
}
BoneAttachment3D::~BoneAttachment3D() = default;

void BoneAttachment3D::set_skeleton_path(const char* path) {
    strncpy(pimpl->skeleton_path, path, 255);
    pimpl->skeleton_path[255] = 0;
    pimpl->dirty = true;
}
const char* BoneAttachment3D::get_skeleton_path() const { return pimpl->skeleton_path; }

void BoneAttachment3D::set_bone_name(const char* name) {
    strncpy(pimpl->bone_name, name, 63);
    pimpl->bone_name[63] = 0;
    pimpl->bone_index = -1;
    pimpl->dirty = true;
}
const char* BoneAttachment3D::get_bone_name() const { return pimpl->bone_name; }

void BoneAttachment3D::set_bone_index(int index) {
    pimpl->bone_index = index;
    pimpl->bone_name[0] = 0;
    pimpl->dirty = true;
}
int BoneAttachment3D::get_bone_index() const { return pimpl->bone_index; }

void BoneAttachment3D::set_use_external_skeleton(bool enable) { pimpl->use_external_skeleton = enable; }
bool BoneAttachment3D::get_use_external_skeleton() const { return pimpl->use_external_skeleton; }

void BoneAttachment3D::set_relative_offset(const Transform3D& offset) {
    pimpl->relative_offset = offset;
    pimpl->dirty = true;
}
Transform3D BoneAttachment3D::get_relative_offset() const { return pimpl->relative_offset; }

void BoneAttachment3D::set_absolute_offset(const Transform3D& offset) {
    pimpl->absolute_offset = offset;
    pimpl->dirty = true;
}
Transform3D BoneAttachment3D::get_absolute_offset() const { return pimpl->absolute_offset; }

void BoneAttachment3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool BoneAttachment3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void BoneAttachment3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int BoneAttachment3D::get_gi_mode() const { return pimpl->gi_mode; }

void BoneAttachment3D::update_attachment() {
    if (!pimpl->dirty && pimpl->skeleton) return;

    // Resolve skeleton node if path changed
    if (!pimpl->skeleton || pimpl->use_external_skeleton) {
        // External skeleton would be set via another method; here we try to find it.
        // Simplified: assume the skeleton is in the scene tree with the given path.
        // In a real engine, we would call get_node(pimpl->skeleton_path)
        pimpl->skeleton = nullptr; // placeholder – user must set external skeleton manually
    }

    // If we have a skeleton, get bone index
    if (pimpl->skeleton) {
        if (pimpl->bone_index >= 0) {
            // index already valid
        } else if (pimpl->bone_name[0] != 0) {
            pimpl->bone_index = pimpl->skeleton->find_bone(pimpl->bone_name);
        }
    }

    if (pimpl->skeleton && pimpl->bone_index >= 0 && pimpl->bone_index < pimpl->skeleton->get_bone_count()) {
        Transform3D bone_global = pimpl->skeleton->get_bone_global_pose(pimpl->bone_index);
        // compute final world transform for this attachment
        Transform3D final_transform;
        if (pimpl->absolute_offset.is_identity()) {
            final_transform = bone_global * pimpl->relative_offset;
        } else {
            final_transform = pimpl->absolute_offset;
        }
        set_global_transform(final_transform);
        pimpl->last_bone_pose = bone_global;
    }
    pimpl->dirty = false;
}

void BoneAttachment3D::process(double delta) {
    Node3D::process(delta);
    update_attachment();
}

void BoneAttachment3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // Notify render server about shadow/gi settings for this node
    // (The attached node's visual instance will inherit these flags)
}

} // namespace lighting