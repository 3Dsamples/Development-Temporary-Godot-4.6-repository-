// genesis/engine/bvh.cpp

#include "genesis/engine/bvh.h"
#include <cmath>
#include <stack>
#include <queue>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstring>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// BVH Implementation
//------------------------------------------------------------------------------

BVH::BVH() {
    // Start with a single empty node (index 0 is reserved as null)
    nodes_.clear();
    nodes_.push_back(Node()); // node 0 is sentinel
    primitives_.clear();
}

BVH::~BVH() = default;

void BVH::clear() {
    nodes_.clear();
    nodes_.push_back(Node());
    primitives_.clear();
    primitive_indices_.clear();
    stats_ = Stats();
}

void BVH::build(const std::vector<datatypes::Vector3>& vertices,
                const std::vector<uint32_t>& indices) {
    clear();
    if (indices.size() < 3) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert triangles to primitives
    primitives_.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3) {
        Primitive prim;
        prim.v0 = vertices[indices[i]];
        prim.v1 = vertices[indices[i+1]];
        prim.v2 = vertices[indices[i+2]];
        prim.centroid = (prim.v0 + prim.v1 + prim.v2) * (1.0f/3.0f);
        prim.original_index = static_cast<uint32_t>(i / 3);
        prim.material_id = 0;
        primitives_.push_back(prim);
    }
    build(std::move(primitives_));
    // build already clears and sets primitives_, we just update stats time
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.build_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
}

void BVH::build(std::vector<Primitive> primitives) {
    clear();
    primitives_ = std::move(primitives);
    if (primitives_.empty()) return;

    primitive_indices_.resize(primitives_.size());
    std::iota(primitive_indices_.begin(), primitive_indices_.end(), 0);

    // Create root node
    nodes_.resize(2); // node 0 sentinel, node 1 root
    Node& root = nodes_[1];
    root.bounds = compute_primitive_bounds(0, static_cast<uint32_t>(primitives_.size()));
    root.left_child = 0;
    root.right_child = 0;
    root.primitive_index = 0;
    root.primitive_count = static_cast<uint32_t>(primitives_.size());

    // Build recursively
    build_recursive(1, 0, root.primitive_count, 0);

    stats_.node_count = nodes_.size() - 1; // exclude sentinel
}

void BVH::build_recursive(uint32_t node_index, uint32_t start, uint32_t count, uint32_t depth) {
    stats_.max_depth = std::max(stats_.max_depth, static_cast<size_t>(depth));
    Node& node = nodes_[node_index];

    // Leaf condition
    if (count <= MAX_PRIMS_PER_LEAF) {
        node.primitive_index = start;
        node.primitive_count = count;
        node.left_child = 0;
        node.right_child = 0;
        stats_.leaf_count++;
        stats_.total_primitive_references += count;
        return;
    }

    // Find best split
    uint32_t split_axis;
    uint32_t split_pos = find_best_split(start, count, split_axis);
    if (split_pos == start || split_pos == start + count) {
        // Could not find a valid split, make leaf
        node.primitive_index = start;
        node.primitive_count = count;
        node.left_child = 0;
        node.right_child = 0;
        stats_.leaf_count++;
        stats_.total_primitive_references += count;
        return;
    }

    // Partition primitives around the split point
    float split_val = compute_primitive_centroid(primitive_indices_[split_pos])[split_axis];
    uint32_t mid;
    partition_primitives(start, count, split_axis, split_val, mid);

    // Create children
    uint32_t left_child = static_cast<uint32_t>(nodes_.size());
    uint32_t right_child = left_child + 1;
    nodes_.resize(nodes_.size() + 2);

    node.left_child = left_child;
    node.right_child = right_child;
    node.split_axis = static_cast<uint8_t>(split_axis);
    node.primitive_index = 0;
    node.primitive_count = 0;

    // Compute child bounds and recurse
    Node& left = nodes_[left_child];
    left.bounds = compute_primitive_bounds(start, mid - start);
    left.primitive_index = start;
    left.primitive_count = mid - start;

    Node& right = nodes_[right_child];
    right.bounds = compute_primitive_bounds(mid, start + count - mid);
    right.primitive_index = mid;
    right.primitive_count = start + count - mid;

    build_recursive(left_child, start, mid - start, depth + 1);
    build_recursive(right_child, mid, start + count - mid, depth + 1);
}

uint32_t BVH::find_best_split(uint32_t start, uint32_t count, uint32_t& split_axis) {
    datatypes::AABB centroid_bounds;
    for (uint32_t i = start; i < start + count; ++i) {
        centroid_bounds.expand(compute_primitive_centroid(primitive_indices_[i]));
    }
    datatypes::Vector3 extents = centroid_bounds.max - centroid_bounds.min;

    // Choose axis with largest extent
    split_axis = 0;
    if (extents[1] > extents[split_axis]) split_axis = 1;
    if (extents[2] > extents[split_axis]) split_axis = 2;

    if (extents[split_axis] < 1e-6f) {
        // Degenerate, just split in middle
        return start + count / 2;
    }

    // SAH binning
    std::array<Bin, BINS_COUNT> bins;
    float bin_width = extents[split_axis] / BINS_COUNT;
    float bin_start = centroid_bounds.min[split_axis];

    // Initialize bins
    for (auto& bin : bins) {
        bin.bounds = datatypes::AABB();
        bin.count = 0;
    }

    // Fill bins
    for (uint32_t i = start; i < start + count; ++i) {
        const Primitive& prim = primitives_[primitive_indices_[i]];
        float centroid_val = prim.centroid[split_axis];
        int bin_idx = static_cast<int>((centroid_val - bin_start) / bin_width);
        bin_idx = std::clamp(bin_idx, 0, static_cast<int>(BINS_COUNT - 1));
        bins[bin_idx].bounds.expand(prim.v0);
        bins[bin_idx].bounds.expand(prim.v1);
        bins[bin_idx].bounds.expand(prim.v2);
        bins[bin_idx].count++;
    }

    // Compute prefix sums of counts and bounds
    std::array<datatypes::AABB, BINS_COUNT> prefix_bounds;
    std::array<uint32_t, BINS_COUNT> prefix_counts;
    datatypes::AABB accum_bounds;
    uint32_t accum_count = 0;
    for (size_t i = 0; i < BINS_COUNT; ++i) {
        accum_count += bins[i].count;
        accum_bounds.expand(bins[i].bounds);
        prefix_counts[i] = accum_count;
        prefix_bounds[i] = accum_bounds;
    }

    // Compute suffix sums
    std::array<datatypes::AABB, BINS_COUNT> suffix_bounds;
    accum_bounds = datatypes::AABB();
    for (int i = static_cast<int>(BINS_COUNT) - 1; i >= 0; --i) {
        accum_bounds.expand(bins[i].bounds);
        suffix_bounds[i] = accum_bounds;
    }

    // Find best split by SAH cost
    float best_cost = std::numeric_limits<float>::max();
    uint32_t best_split_pos = start;
    for (size_t i = 0; i < BINS_COUNT - 1; ++i) {
        if (prefix_counts[i] == 0 || prefix_counts[i] == count) continue;
        float cost_left = prefix_bounds[i].surfaceArea() * static_cast<float>(prefix_counts[i]);
        float cost_right = suffix_bounds[i+1].surfaceArea() * static_cast<float>(count - prefix_counts[i]);
        float cost = cost_left + cost_right;
        if (cost < best_cost) {
            best_cost = cost;
            // The split position in terms of primitive index order: we'll use the first primitive
            // whose centroid is greater than the bin boundary.
            // For simplicity, we just return the index of the first primitive in the next bin.
            best_split_pos = start + prefix_counts[i];
        }
    }

    return best_split_pos;
}

void BVH::partition_primitives(uint32_t start, uint32_t count, uint32_t axis, float split_pos,
                               uint32_t& out_mid) {
    uint32_t left = start;
    uint32_t right = start + count - 1;
    while (left <= right) {
        if (compute_primitive_centroid(primitive_indices_[left])[axis] < split_pos) {
            ++left;
        } else {
            std::swap(primitive_indices_[left], primitive_indices_[right]);
            --right;
        }
    }
    out_mid = left; // left is the start of the right partition
}

datatypes::AABB BVH::compute_primitive_bounds(uint32_t start, uint32_t count) const {
    datatypes::AABB bounds;
    for (uint32_t i = start; i < start + count; ++i) {
        const Primitive& prim = primitives_[primitive_indices_[i]];
        bounds.expand(prim.v0);
        bounds.expand(prim.v1);
        bounds.expand(prim.v2);
    }
    return bounds;
}

datatypes::Vector3 BVH::compute_primitive_centroid(uint32_t prim_idx) const {
    return primitives_[prim_idx].centroid;
}

float BVH::compute_sah_cost(uint32_t start, uint32_t count, uint32_t axis, float split_pos) const {
    datatypes::AABB left_bounds, right_bounds;
    uint32_t left_count = 0, right_count = 0;
    for (uint32_t i = start; i < start + count; ++i) {
        const Primitive& prim = primitives_[primitive_indices_[i]];
        if (prim.centroid[axis] < split_pos) {
            left_bounds.expand(prim.v0);
            left_bounds.expand(prim.v1);
            left_bounds.expand(prim.v2);
            ++left_count;
        } else {
            right_bounds.expand(prim.v0);
            right_bounds.expand(prim.v1);
            right_bounds.expand(prim.v2);
            ++right_count;
        }
    }
    float cost = left_bounds.surfaceArea() * static_cast<float>(left_count) +
                 right_bounds.surfaceArea() * static_cast<float>(right_count);
    return cost;
}

void BVH::refit(const std::vector<datatypes::Vector3>& vertices) {
    if (nodes_.size() <= 1) return;
    // Update primitive positions
    for (auto& prim : primitives_) {
        uint32_t idx = prim.original_index * 3;
        prim.v0 = vertices[idx];
        prim.v1 = vertices[idx+1];
        prim.v2 = vertices[idx+2];
        prim.centroid = (prim.v0 + prim.v1 + prim.v2) * (1.0f/3.0f);
    }
    // Refit nodes bottom-up using a stack
    std::vector<uint32_t> postorder;
    std::stack<uint32_t> stack;
    stack.push(1);
    while (!stack.empty()) {
        uint32_t idx = stack.top();
        stack.pop();
        postorder.push_back(idx);
        Node& node = nodes_[idx];
        if (!node.is_leaf()) {
            stack.push(node.right_child);
            stack.push(node.left_child);
        }
    }
    std::reverse(postorder.begin(), postorder.end());
    for (uint32_t idx : postorder) {
        refit_node(idx, vertices);
    }
}

void BVH::refit_node(uint32_t node_index, const std::vector<datatypes::Vector3>& vertices) {
    Node& node = nodes_[node_index];
    if (node.is_leaf()) {
        datatypes::AABB bounds;
        for (uint32_t i = 0; i < node.primitive_count; ++i) {
            const Primitive& prim = primitives_[primitive_indices_[node.primitive_index + i]];
            bounds.expand(prim.v0);
            bounds.expand(prim.v1);
            bounds.expand(prim.v2);
        }
        node.bounds = bounds;
    } else {
        node.bounds = nodes_[node.left_child].bounds;
        node.bounds.expand(nodes_[node.right_child].bounds);
    }
}

void BVH::intersect_aabb(const datatypes::AABB& query_aabb,
                         std::vector<uint32_t>& out_primitive_indices) const {
    if (nodes_.size() <= 1) return;
    std::stack<uint32_t> stack;
    stack.push(1);
    while (!stack.empty()) {
        uint32_t idx = stack.top();
        stack.pop();
        const Node& node = nodes_[idx];
        if (!node.bounds.intersects(query_aabb)) continue;
        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.primitive_count; ++i) {
                out_primitive_indices.push_back(primitive_indices_[node.primitive_index + i]);
            }
        } else {
            stack.push(node.left_child);
            stack.push(node.right_child);
        }
    }
}

BVH::RayHit BVH::intersect_ray(const datatypes::Ray& ray) const {
    RayHit best_hit;
    if (nodes_.size() <= 1) return best_hit;

    std::stack<uint32_t> stack;
    stack.push(1);
    while (!stack.empty()) {
        uint32_t idx = stack.top();
        stack.pop();
        const Node& node = nodes_[idx];
        float tmin, tmax;
        if (!node.bounds.rayIntersect(ray, tmin, tmax) || tmin > best_hit.t) continue;
        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.primitive_count; ++i) {
                const Primitive& prim = primitives_[primitive_indices_[node.primitive_index + i]];
                float t;
                datatypes::Vector3 normal;
                if (ray_triangle_intersect(ray, prim, t, normal) && t < best_hit.t) {
                    best_hit.hit = true;
                    best_hit.t = t;
                    best_hit.primitive_index = prim.original_index;
                    best_hit.point = ray.pointAt(t);
                    best_hit.normal = normal;
                }
            }
        } else {
            // Order children by ray direction for better culling
            // Not implemented for brevity, but could push farther child first
            stack.push(node.left_child);
            stack.push(node.right_child);
        }
    }
    return best_hit;
}

BVH::ClosestPointResult BVH::closest_point(const datatypes::Vector3& query) const {
    ClosestPointResult result;
    if (nodes_.size() <= 1) return result;

    struct QueueItem {
        uint32_t node_idx;
        float dist_sq;
        bool operator<(const QueueItem& other) const { return dist_sq > other.dist_sq; }
    };
    std::priority_queue<QueueItem> pq;
    pq.push({1, nodes_[1].bounds.distanceToPoint(query)});

    while (!pq.empty()) {
        QueueItem item = pq.top();
        pq.pop();
        if (item.dist_sq > result.distance_sq) break;

        const Node& node = nodes_[item.node_idx];
        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.primitive_count; ++i) {
                const Primitive& prim = primitives_[primitive_indices_[node.primitive_index + i]];
                datatypes::Vector3 cp = closest_point_on_triangle(query, prim);
                float dsq = (cp - query).squaredNorm();
                if (dsq < result.distance_sq) {
                    result.distance_sq = dsq;
                    result.point = cp;
                    result.primitive_index = prim.original_index;
                }
            }
        } else {
            float dleft = nodes_[node.left_child].bounds.distanceToPoint(query);
            float dright = nodes_[node.right_child].bounds.distanceToPoint(query);
            pq.push({node.left_child, dleft});
            pq.push({node.right_child, dright});
        }
    }
    return result;
}

void BVH::intersect_bvh(const BVH& other,
                        std::vector<std::pair<uint32_t, uint32_t>>& out_primitive_pairs) const {
    if (nodes_.size() <= 1 || other.nodes_.size() <= 1) return;

    struct Pair {
        uint32_t a, b;
    };
    std::stack<Pair> stack;
    stack.push({1, 1});

    while (!stack.empty()) {
        Pair p = stack.top();
        stack.pop();
        const Node& node_a = nodes_[p.a];
        const Node& node_b = other.nodes_[p.b];
        if (!node_a.bounds.intersects(node_b.bounds)) continue;

        if (node_a.is_leaf() && node_b.is_leaf()) {
            // Test all primitives in both leaves
            for (uint32_t i = 0; i < node_a.primitive_count; ++i) {
                uint32_t prim_a = primitive_indices_[node_a.primitive_index + i];
                for (uint32_t j = 0; j < node_b.primitive_count; ++j) {
                    uint32_t prim_b = other.primitive_indices_[node_b.primitive_index + j];
                    out_primitive_pairs.emplace_back(prim_a, prim_b);
                }
            }
        } else if (node_a.is_leaf()) {
            stack.push({p.a, node_b.left_child});
            stack.push({p.a, node_b.right_child});
        } else if (node_b.is_leaf()) {
            stack.push({node_a.left_child, p.b});
            stack.push({node_a.right_child, p.b});
        } else {
            // Descend into the node with larger surface area first (heuristic)
            if (node_a.bounds.surfaceArea() > node_b.bounds.surfaceArea()) {
                stack.push({node_a.left_child, p.b});
                stack.push({node_a.right_child, p.b});
            } else {
                stack.push({p.a, node_b.left_child});
                stack.push({p.a, node_b.right_child});
            }
        }
    }
}

void BVH::traverse(const std::function<bool(const Node&)>& node_callback) const {
    if (nodes_.size() <= 1) return;
    std::stack<uint32_t> stack;
    stack.push(1);
    while (!stack.empty()) {
        uint32_t idx = stack.top();
        stack.pop();
        const Node& node = nodes_[idx];
        if (node_callback(node)) {
            if (!node.is_leaf()) {
                stack.push(node.left_child);
                stack.push(node.right_child);
            }
        }
    }
}

void BVH::get_node_boxes(std::vector<datatypes::AABB>& out_boxes,
                         std::vector<uint32_t>& out_levels) const {
    if (nodes_.size() <= 1) return;
    std::queue<std::pair<uint32_t, uint32_t>> q; // node_idx, level
    q.push({1, 0});
    while (!q.empty()) {
        auto [idx, level] = q.front();
        q.pop();
        const Node& node = nodes_[idx];
        out_boxes.push_back(node.bounds);
        out_levels.push_back(level);
        if (!node.is_leaf()) {
            q.push({node.left_child, level + 1});
            q.push({node.right_child, level + 1});
        }
    }
}

bool BVH::ray_triangle_intersect(const datatypes::Ray& ray,
                                 const Primitive& prim,
                                 float& out_t,
                                 datatypes::Vector3& out_normal) const {
    const datatypes::Vector3& v0 = prim.v0;
    const datatypes::Vector3& v1 = prim.v1;
    const datatypes::Vector3& v2 = prim.v2;
    datatypes::Vector3 e1 = v1 - v0;
    datatypes::Vector3 e2 = v2 - v0;
    datatypes::Vector3 h = ray.direction.cross(e2);
    float a = e1.dot(h);
    if (std::abs(a) < 1e-7f) return false;
    float f = 1.0f / a;
    datatypes::Vector3 s = ray.origin - v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;
    datatypes::Vector3 q = s.cross(e1);
    float v = f * ray.direction.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * e2.dot(q);
    if (t > 1e-6f) {
        out_t = t;
        out_normal = e1.cross(e2).normalized();
        return true;
    }
    return false;
}

datatypes::Vector3 BVH::closest_point_on_triangle(const datatypes::Vector3& p,
                                                  const Primitive& prim) const {
    const datatypes::Vector3& a = prim.v0;
    const datatypes::Vector3& b = prim.v1;
    const datatypes::Vector3& c = prim.v2;
    datatypes::Vector3 ab = b - a;
    datatypes::Vector3 ac = c - a;
    datatypes::Vector3 ap = p - a;
    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    datatypes::Vector3 bp = p - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    datatypes::Vector3 cp = p - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + ac * w;
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

//------------------------------------------------------------------------------
// PointBVH Implementation
//------------------------------------------------------------------------------

void PointBVH::build(const std::vector<Point>& points) {
    points_ = points;
    if (points_.empty()) return;

    // Build simple recursive split based on median (similar to BVH but for points)
    std::vector<uint32_t> indices(points_.size());
    std::iota(indices.begin(), indices.end(), 0);
    nodes_.clear();
    nodes_.push_back(PointNode()); // sentinel

    struct BuildTask {
        uint32_t parent;
        uint32_t start;
        uint32_t count;
        bool is_left;
    };
    std::stack<BuildTask> tasks;
    tasks.push({0, 0, static_cast<uint32_t>(points_.size()), false});

    while (!tasks.empty()) {
        BuildTask task = tasks.top();
        tasks.pop();

        datatypes::AABB bounds;
        for (uint32_t i = task.start; i < task.start + task.count; ++i) {
            bounds.expand(points_[indices[i]].position);
        }

        uint32_t node_idx = static_cast<uint32_t>(nodes_.size());
        nodes_.emplace_back();
        PointNode& node = nodes_.back();
        node.bounds = bounds;

        if (task.parent != 0) {
            if (task.is_left) nodes_[task.parent].left = node_idx;
            else nodes_[task.parent].right = node_idx;
        }

        if (task.count <= 16) {
            node.point_start = task.start;
            node.point_count = task.count;
            continue;
        }

        // Find longest axis
        datatypes::Vector3 ext = bounds.max - bounds.min;
        int axis = 0;
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;

        // Sort points along axis using nth_element
        auto mid = indices.begin() + task.start + task.count/2;
        std::nth_element(indices.begin() + task.start, mid, indices.begin() + task.start + task.count,
            [this, axis](uint32_t a, uint32_t b) {
                return points_[a].position[axis] < points_[b].position[axis];
            });
        uint32_t mid_idx = static_cast<uint32_t>(task.start + task.count/2);
        tasks.push({node_idx, task.start, mid_idx - task.start, true});
        tasks.push({node_idx, mid_idx, task.start + task.count - mid_idx, false});
    }
}

void PointBVH::clear() {
    nodes_.clear();
    points_.clear();
}

void PointBVH::query_radius(const datatypes::Vector3& center, float radius,
                            std::vector<uint32_t>& out_indices) const {
    if (nodes_.empty()) return;
    float r2 = radius * radius;
    std::stack<uint32_t> stack;
    stack.push(1);
    while (!stack.empty()) {
        uint32_t idx = stack.top();
        stack.pop();
        const PointNode& node = nodes_[idx];
        if (node.bounds.distanceToPoint(center) > radius) continue;
        if (node.left == 0 && node.right == 0) {
            for (uint32_t i = 0; i < node.point_count; ++i) {
                const Point& pt = points_[node.point_start + i];
                if ((pt.position - center).squaredNorm() <= r2) {
                    out_indices.push_back(pt.index);
                }
            }
        } else {
            stack.push(node.left);
            stack.push(node.right);
        }
    }
}

void PointBVH::query_knn(const datatypes::Vector3& center, uint32_t k,
                         std::vector<std::pair<uint32_t, float>>& out_results) const {
    if (nodes_.empty() || k == 0) return;
    out_results.clear();
    // Use priority queue for nodes and a max-heap for nearest neighbors
    struct NodeDist {
        uint32_t idx;
        float dist_sq;
        bool operator<(const NodeDist& other) const { return dist_sq > other.dist_sq; }
    };
    std::priority_queue<NodeDist> pq;
    pq.push({1, nodes_[1].bounds.distanceToPoint(center)});
    // Max-heap for results (keep smallest distances)
    auto cmp = [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
        return a.second < b.second;
    };
    std::vector<std::pair<uint32_t, float>> heap;
    heap.reserve(k);

    while (!pq.empty()) {
        NodeDist nd = pq.top();
        pq.pop();
        if (heap.size() == k && nd.dist_sq > heap.front().second) break;
        const PointNode& node = nodes_[nd.idx];
        if (node.left == 0 && node.right == 0) {
            for (uint32_t i = 0; i < node.point_count; ++i) {
                const Point& pt = points_[node.point_start + i];
                float d2 = (pt.position - center).squaredNorm();
                if (heap.size() < k) {
                    heap.emplace_back(pt.index, d2);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (d2 < heap.front().second) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {pt.index, d2};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        } else {
            float dl = nodes_[node.left].bounds.distanceToPoint(center);
            float dr = nodes_[node.right].bounds.distanceToPoint(center);
            pq.push({node.left, dl});
            pq.push({node.right, dr});
        }
    }
    std::sort(heap.begin(), heap.end(), [](auto& a, auto& b) { return a.second < b.second; });
    out_results = std::move(heap);
}

} // namespace engine
} // namespace genesis