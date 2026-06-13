// skeleton_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace lighting {

// ============================================================================
// Skeleton3D – bone hierarchy for skinning and animation
// Supports bone transforms (inverse bind pose), animation blending,
// and real‑time bone updates for lighting (shadows, GI).
// ============================================================================

struct BoneData {
    std::string name;
    int parent = -1;
    Transform3D rest_pose;         // local transform
    Transform3D inverse_bind;      // from bone space to model space for skinning
    Transform3D local_pose;        // current local (animated) transform
    Transform3D global_pose;       // cached world transform
    Transform3D skin_pose;         // combined for skinning: global_pose * inverse_bind
    bool enabled = true;
};

class Skeleton3D : public Node3D {
public:
    Skeleton3D();
    ~Skeleton3D();

    // ------------------------------------------------------------------------
    // Bone management
    // ------------------------------------------------------------------------
    int add_bone(const char* name, int parent_index = -1);
    void remove_bone(int index);
    int get_bone_count() const;
    int find_bone(const char* name) const;
    const char* get_bone_name(int index) const;
    void set_bone_parent(int index, int parent_index);
    int get_bone_parent(int index) const;

    // ------------------------------------------------------------------------
    // Rest pose (bind pose) and inverse bind matrices
    // ------------------------------------------------------------------------
    void set_bone_rest_pose(int index, const Transform3D& rest);
    Transform3D get_bone_rest_pose(int index) const;
    void set_bone_inverse_bind(int index, const Transform3D& inv_bind);
    Transform3D get_bone_inverse_bind(int index) const;

    // ------------------------------------------------------------------------
    // Current animated transforms (local to skeleton)
    // ------------------------------------------------------------------------
    void set_bone_local_pose(int index, const Transform3D& pose);
    Transform3D get_bone_local_pose(int index) const;
    void set_bone_global_pose(int index, const Transform3D& pose); // direct world override
    Transform3D get_bone_global_pose(int index) const;

    // ------------------------------------------------------------------------
    // Skin matrix (for vertex shaders)
    // ------------------------------------------------------------------------
    Transform3D get_bone_skin_matrix(int index) const;

    // ------------------------------------------------------------------------
    // Animation binding (for animation players)
    // ------------------------------------------------------------------------
    void set_animation_blend(int track, float weight);
    void clear_animation_tracks();

    // ------------------------------------------------------------------------
    // Update skeleton (recompute global poses and skin matrices)
    // ------------------------------------------------------------------------
    void update_bone_transforms();  // call after any bone transform changes
    void reset_to_rest_pose();

    // ------------------------------------------------------------------------
    // Performance & caching
    // ------------------------------------------------------------------------
    void set_dirty(bool dirty = true);
    bool is_dirty() const;

    // ------------------------------------------------------------------------
    // Lighting integration – skeletal meshes cast shadows and receive GI
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_gi_mode(int mode);   // 0=off, 1=static (baked), 2=dynamic
    int get_gi_mode() const;

    // ------------------------------------------------------------------------
    // Rendering server sync (update bone matrices for GPU)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting