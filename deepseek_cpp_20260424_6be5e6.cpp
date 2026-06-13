// genesis/engine/mesh.h

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include "genesis/datatypes.h"
#include "genesis/engine/bvh.h"

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Mesh class: represents a 3D triangle mesh with optional attributes.
// Supports loading from OBJ/STL, building BVH, computing normals, etc.
//------------------------------------------------------------------------------
class Mesh {
public:
    // Vertex attribute structure
    struct Vertex {
        datatypes::Vector3 position;
        datatypes::Vector3 normal;
        datatypes::Vector2 texcoord;
        datatypes::Vector4 color;
        // Skinning data (optional)
        std::array<int32_t, 4> bone_indices = {-1, -1, -1, -1};
        std::array<float, 4> bone_weights = {0.0f, 0.0f, 0.0f, 0.0f};

        Vertex() : position(0), normal(0,0,1), texcoord(0), color(1,1,1,1) {}
        explicit Vertex(const datatypes::Vector3& pos) : position(pos), normal(0,0,1), texcoord(0), color(1,1,1,1) {}
    };

    // Submesh/material group
    struct SubMesh {
        std::string material_name;
        std::vector<uint32_t> indices;   // indices into mesh's vertex array
        uint32_t index_offset = 0;
        uint32_t index_count = 0;
        datatypes::AABB bounds;
    };

    // Bone/skeleton data for skinning
    struct Bone {
        std::string name;
        datatypes::Matrix4r bind_pose;     // transform from bone space to model space
        datatypes::Matrix4r inverse_bind;  // transform from model space to bone space
        int32_t parent_index = -1;
        std::vector<int32_t> children;
    };

    // Animation keyframe
    struct Keyframe {
        float time;
        std::vector<datatypes::Transformr> bone_transforms; // local transforms
    };

    struct AnimationClip {
        std::string name;
        float duration;
        std::vector<Keyframe> keyframes;
    };

    // Constructors
    Mesh();
    ~Mesh();

    // Clear all data
    void clear();

    // Load from file (OBJ, STL, PLY, etc.)
    bool load_file(const std::string& path);

    // Create primitive shapes
    static Mesh create_box(const datatypes::Vector3& size = datatypes::Vector3(1,1,1));
    static Mesh create_sphere(float radius = 1.0f, int segments = 32);
    static Mesh create_cylinder(float radius = 1.0f, float height = 2.0f, int segments = 32);
    static Mesh create_plane(const datatypes::Vector2& size = datatypes::Vector2(1,1));
    static Mesh create_capsule(float radius = 1.0f, float height = 2.0f, int segments = 16);
    static Mesh create_torus(float outer_radius = 1.0f, float inner_radius = 0.3f, int segments = 32, int sides = 16);

    // Vertex and index access
    const std::vector<Vertex>& vertices() const { return vertices_; }
    std::vector<Vertex>& vertices() { return vertices_; }
    const std::vector<uint32_t>& indices() const { return indices_; }
    std::vector<uint32_t>& indices() { return indices_; }

    void set_vertices(const std::vector<datatypes::Vector3>& positions);
    void set_indices(const std::vector<uint32_t>& indices);

    // Compute normals (if not present)
    void compute_normals(bool smooth = true);
    void compute_tangents(); // for normal mapping

    // Bounding volume
    datatypes::AABB compute_aabb() const;
    const datatypes::AABB& get_aabb() const { return aabb_; }

    // BVH acceleration structure
    void build_bvh();
    const BVH* get_bvh() const { return bvh_.get(); }
    BVH* get_bvh() { return bvh_.get(); }

    // Transformations
    void apply_transform(const datatypes::Matrix4r& transform);
    void translate(const datatypes::Vector3& offset);
    void rotate(const datatypes::Quat& rotation);
    void scale(const datatypes::Vector3& scale);

    // Submesh / material management
    void add_submesh(const SubMesh& submesh);
    const std::vector<SubMesh>& submeshes() const { return submeshes_; }
    std::vector<SubMesh>& submeshes() { return submeshes_; }

    // Skeleton and skinning
    void add_bone(const Bone& bone);
    const std::vector<Bone>& bones() const { return bones_; }
    void build_skinning_data();
    void apply_skinning(const std::vector<datatypes::Matrix4r>& bone_transforms);

    // Animation
    void add_animation(const AnimationClip& clip);
    const std::vector<AnimationClip>& animations() const { return animations_; }
    void sample_animation(const std::string& clip_name, float time, std::vector<datatypes::Matrix4r>& out_transforms) const;

    // Ray casting
    struct RayHit {
        float t = std::numeric_limits<float>::max();
        datatypes::Vector3 point;
        datatypes::Vector3 normal;
        uint32_t triangle_index = 0;
        uint32_t submesh_index = 0;
        bool hit = false;
    };
    RayHit ray_cast(const datatypes::Ray& ray) const;

    // Closest point query
    struct ClosestPointResult {
        datatypes::Vector3 point;
        uint32_t triangle_index = 0;
        float distance_sq = std::numeric_limits<float>::max();
    };
    ClosestPointResult closest_point(const datatypes::Vector3& query) const;

    // Collision mesh generation (convex decomposition, etc.)
    Mesh create_convex_hull() const; // placeholder, would need external lib
    std::vector<Mesh> create_convex_decomposition() const;

    // Statistics
    size_t vertex_count() const { return vertices_.size(); }
    size_t triangle_count() const { return indices_.size() / 3; }
    float surface_area() const;
    float volume() const;

    // Name for identification
    void set_name(const std::string& name) { name_ = name; }
    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    std::vector<SubMesh> submeshes_;
    std::vector<Bone> bones_;
    std::vector<AnimationClip> animations_;
    datatypes::AABB aabb_;
    std::unique_ptr<BVH> bvh_;

    // OBJ loading helpers
    struct ObjFaceVertex {
        int v_idx, vt_idx, vn_idx;
        bool operator==(const ObjFaceVertex& other) const {
            return v_idx == other.v_idx && vt_idx == other.vt_idx && vn_idx == other.vn_idx;
        }
    };
    struct ObjFaceVertexHash {
        size_t operator()(const ObjFaceVertex& fv) const {
            return ((std::hash<int>()(fv.v_idx) ^ (std::hash<int>()(fv.vt_idx) << 1)) >> 1) ^ (std::hash<int>()(fv.vn_idx) << 1);
        }
    };

    bool load_obj(const std::string& path);
    bool load_stl(const std::string& path, bool is_ascii = false);
    bool load_ply(const std::string& path);

    // Helper for building primitive shapes
    void add_triangle(const datatypes::Vector3& v0, const datatypes::Vector3& v1, const datatypes::Vector3& v2);
    void add_quad(const datatypes::Vector3& v0, const datatypes::Vector3& v1,
                  const datatypes::Vector3& v2, const datatypes::Vector3& v3);

    // Update AABB from vertices
    void update_aabb();
};

} // namespace engine
} // namespace genesis