//File group name : OrthoTree Math
//File 0086 : core/math/approx_radius.h
//Approximate radius search: find all points within radius, allowing relative error epsilon.
//Uses kd‑tree with pruning based on (1+ε) tolerance. Returns vector of point indices.

#ifndef ORTHOTREE_CORE_MATH_APPROX_RADIUS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_APPROX_RADIUS_H_INCLUDED

#include "../../build_config.h"
#include "basic/vector.h"
#include "kdtree.h"
#include "math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  ApproximateRadiusSearch: wrapper around KdTree that performs radius search
//  with a relative error epsilon. For each point, if the true distance to query
//  is d, the algorithm returns points with distance <= (1+ε) * radius.
//  It uses early pruning to reduce the number of distance checks.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class ApproximateRadiusSearch {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, N>;
    using kdtree_type = KdTree<T, N>;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Constructor: wraps an existing KdTree (must be built).
    // ------------------------------------------------------------------------
    explicit ApproximateRadiusSearch(const kdtree_type& tree, T epsilon = T(0.1))
        : m_tree(tree), m_epsilon(epsilon) {}

    // ------------------------------------------------------------------------
    //  Approximate radius search. Returns indices of points whose distance to
    //  query is <= (1 + epsilon) * radius.
    // ------------------------------------------------------------------------
    std::vector<size_type> query(const point_type& query, T radius) const {
        std::vector<size_type> result;
        if (radius <= T(0)) return result;
        T radiusSq = radius * radius;
        T thresholdSq = radiusSq * (T(1) + m_epsilon) * (T(1) + m_epsilon);
        // Use recursive traversal with pruning based on bounding sphere.
        const auto& nodes = m_tree.m_nodes;
        const auto& points = m_tree.m_points;
        const auto& indices = m_tree.m_indices;
        if (nodes.empty()) return result;
        searchNode(0, query, radiusSq, thresholdSq, result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Set epsilon (relative error).
    // ------------------------------------------------------------------------
    void setEpsilon(T eps) noexcept { m_epsilon = eps; }
    T epsilon() const noexcept { return m_epsilon; }

private:
    // Recursive traversal with approximate bounding sphere pruning.
    void searchNode(size_type nodeIdx, const point_type& query,
                    T radiusSq, T thresholdSq,
                    std::vector<size_type>& out) const {
        const auto& node = m_tree.m_nodes[nodeIdx];
        // Quick rejection: if the node's bounding sphere is farther than
        // (1+ε)*radius, skip.
        T sphereDist2 = (node.center - query).squaredLength();
        T sphereBound2 = sphereDist2 - node.radius * node.radius;
        if (sphereBound2 > thresholdSq) return;

        if (node.isLeaf) {
            for (size_type i = node.leaf.start; i < node.leaf.start + node.leaf.count; ++i) {
                size_type idx = m_tree.m_indices[i];
                T d2 = (m_tree.m_points[idx] - query).squaredLength();
                if (d2 <= radiusSq) {
                    out.push_back(idx);
                } else if (d2 <= thresholdSq) {
                    // Within approximate radius but not exact; we can include
                    // or reject based on application. We'll include as approximate.
                    out.push_back(idx);
                }
            }
            return;
        }
        // Process children in order of increasing distance.
        T leftDist2 = m_tree.m_nodes[node.children.left].center.squaredDistanceTo(query);
        T rightDist2 = m_tree.m_nodes[node.children.right].center.squaredDistanceTo(query);
        if (leftDist2 < rightDist2) {
            searchNode(node.children.left, query, radiusSq, thresholdSq, out);
            searchNode(node.children.right, query, radiusSq, thresholdSq, out);
        } else {
            searchNode(node.children.right, query, radiusSq, thresholdSq, out);
            searchNode(node.children.left, query, radiusSq, thresholdSq, out);
        }
    }

    const kdtree_type& m_tree;
    T m_epsilon;
};

// ----------------------------------------------------------------------------
//  Helper to create approximate radius search wrapper.
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
ApproximateRadiusSearch<T, N> makeApproxRadiusSearch(const KdTree<T, N>& tree,
                                                     T epsilon = T(0.1)) {
    return ApproximateRadiusSearch<T, N>(tree, epsilon);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller.
// ----------------------------------------------------------------------------
class ApproxRadiusEnvironment {
public:
    static ApproxRadiusEnvironment& instance() {
        static ApproxRadiusEnvironment env;
        return env;
    }
    void setDefaultEpsilon(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultEpsilon = eps;
    }
    T defaultEpsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultEpsilon;
    }
private:
    ApproxRadiusEnvironment() : m_defaultEpsilon(T(0.1)) {}
    mutable std::mutex m_mutex;
    T m_defaultEpsilon;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_APPROX_RADIUS_H_INCLUDED