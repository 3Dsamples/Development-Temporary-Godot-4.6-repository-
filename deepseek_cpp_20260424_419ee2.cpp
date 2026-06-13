// genesis/engine/bvh.h

#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <functional>
#include <stack>
#include <array>
#include "genesis/datatypes.h"

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Bounding Volume Hierarchy (BVH) for fast collision detection and ray casting.
// Implements a binary tree of axis-aligned bounding boxes (AABB).
//------------------------------------------------------------------------------

class BVH {
public:
    // Node structure for the BVH tree
    struct Node {
        datatypes::AABB bounds;           // Bounding box of this node
        uint32_t left_child;              // Index of left child (0 if leaf)
        uint32_t right_child;             // Index of right child (0 if leaf)
        uint32_t primitive_index;         // First primitive index (if leaf)
        uint32_t primitive_count;         // Number of primitives (0 for internal nodes)
        uint8_t split_axis;               // Axis along which split occurred (0=x,1=y,2=z)
        bool is_leaf() const { return right_child == 0; }
    };

    // Primitive type (triangle or other geometry)
    struct Primitive {
        datatypes::Vector3 v0, v1, v2;    // Triangle vertices
        datatypes::Vector3 centroid;      // Precomputed centroid for sorting
        uint32_t original_index;          // Index in original mesh
        uint32_t material_id;             // Material identifier
    };

    // Statistics for the BVH build
    struct Stats {
        size_t node_count = 0;
        size_t leaf_count = 0;
        size_t max_depth = 0;
        size_t total_primitive_references = 0;
        float build_time_ms = 0.0f;
    };

    // Constructor
    BVH();
    ~BVH();

    // Build the BVH from a list of triangles (vertices and indices)
    void build(const std::vector<datatypes::Vector3>& vertices,
               const std::vector<uint32_t>& indices);

    // Build from a list of primitives directly
    void build(std::vector<Primitive> primitives);

    // Clear the BVH
    void clear();

    // Refit the BVH bounds without rebuilding (for deformable meshes)
    void refit(const std::vector<datatypes::Vector3>& vertices);

    // Query functions

    // Find all primitives that intersect a given AABB
    void intersect_aabb(const datatypes::AABB& query_aabb,
                        std::vector<uint32_t>& out_primitive_indices) const;

    // Find all primitives that intersect a ray
    struct RayHit {
        uint32_t primitive_index = 0;
        float t = std::numeric_limits<float>::max();
        datatypes::Vector3 point;
        datatypes::Vector3 normal;
        bool hit = false;
    };
    RayHit intersect_ray(const datatypes::Ray& ray) const;

    // Find closest point on the mesh to a given query point
    struct ClosestPointResult {
        uint32_t primitive_index = 0;
        datatypes::Vector3 point;
        float distance_sq = std::numeric_limits<float>::max();
    };
    ClosestPointResult closest_point(const datatypes::Vector3& query) const;

    // Collision detection between this BVH and another
    void intersect_bvh(const BVH& other,
                       std::vector<std::pair<uint32_t, uint32_t>>& out_primitive_pairs) const;

    // Traversal with a custom callback
    void traverse(const std::function<bool(const Node&)>& node_callback) const;

    // Get accessors
    const std::vector<Node>& get_nodes() const { return nodes_; }
    const std::vector<Primitive>& get_primitives() const { return primitives_; }
    const Stats& get_stats() const { return stats_; }

    // For visualization: get all node bounding boxes
    void get_node_boxes(std::vector<datatypes::AABB>& out_boxes,
                        std::vector<uint32_t>& out_levels) const;

private:
    std::vector<Node> nodes_;
    std::vector<Primitive> primitives_;
    std::vector<uint32_t> primitive_indices_; // Temporary during build
    Stats stats_;

    // Build configuration
    static constexpr uint32_t MAX_PRIMS_PER_LEAF = 4;
    static constexpr uint32_t BINS_COUNT = 16;  // For SAH binning

    // Internal build methods
    struct BuildTask {
        uint32_t node_index;
        uint32_t start;
        uint32_t count;
        uint32_t depth;
    };
    void build_recursive(uint32_t node_index, uint32_t start, uint32_t count, uint32_t depth);
    uint32_t find_best_split(uint32_t start, uint32_t count, uint32_t& split_axis);
    void partition_primitives(uint32_t start, uint32_t count, uint32_t axis, float split_pos,
                              uint32_t& out_mid);

    // SAH (Surface Area Heuristic) methods
    struct Bin {
        datatypes::AABB bounds;
        uint32_t count = 0;
    };
    float compute_sah_cost(uint32_t start, uint32_t count, uint32_t axis, float split_pos) const;

    // Primitive bounds computation
    datatypes::AABB compute_primitive_bounds(uint32_t start, uint32_t count) const;
    datatypes::Vector3 compute_primitive_centroid(uint32_t prim_idx) const;

    // Ray-triangle intersection (Möller–Trumbore)
    bool ray_triangle_intersect(const datatypes::Ray& ray,
                                const Primitive& prim,
                                float& out_t,
                                datatypes::Vector3& out_normal) const;

    // Point-triangle distance and closest point
    datatypes::Vector3 closest_point_on_triangle(const datatypes::Vector3& p,
                                                 const Primitive& prim) const;

    // Refit helper
    void refit_node(uint32_t node_index, const std::vector<datatypes::Vector3>& vertices);
};

//------------------------------------------------------------------------------
// Utility functions for BVH operations
//------------------------------------------------------------------------------

// Build a BVH for a set of points (e.g., particles) where each point is a sphere
class PointBVH {
public:
    struct Point {
        datatypes::Vector3 position;
        float radius;
        uint32_t index;
    };

    void build(const std::vector<Point>& points);
    void clear();

    // Query points within radius of a given point
    void query_radius(const datatypes::Vector3& center, float radius,
                      std::vector<uint32_t>& out_indices) const;

    // Query k-nearest neighbors
    void query_knn(const datatypes::Vector3& center, uint32_t k,
                   std::vector<std::pair<uint32_t, float>>& out_results) const;

private:
    struct PointNode {
        datatypes::AABB bounds;
        uint32_t left = 0;
        uint32_t right = 0;
        uint32_t point_start = 0;
        uint32_t point_count = 0;
    };
    std::vector<PointNode> nodes_;
    std::vector<Point> points_;
};

} // namespace engine
} // namespace genesis