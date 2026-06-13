/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file partitioning.h
 * @brief Spatial partitioning heuristics and splitting algorithms.
 *
 * This file provides algorithms for splitting node bounding boxes in octrees
 * and BVHs. It includes:
 * - Surface area heuristic (SAH) for BVH construction
 * - Median split (for LBVH)
 * - Equal‑volume split (for uniform grids)
 * - Axis selection based on longest side
 *
 * These functions are used during tree construction to decide where to split
 * a node and along which axis, balancing traversal cost and overlap.
 */

#ifndef ORTHOTREE_DETAIL_PARTITIONING_H_INCLUDED
#define ORTHOTREE_DETAIL_PARTITIONING_H_INCLUDED

#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "common.h"
#include "inplace_vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace OrthoTree {
namespace detail {

// ============================================================================
//  Helper: compute bounding box of a set of points or AABBs
// ============================================================================

/**
 * @brief Compute the union AABB of a range of boxes.
 * @tparam Iter Iterator over AxisAlignedBox<T,Dim>
 */
template <Dimension Dim, typename T, typename Iter>
Math::AxisAlignedBox<T, Dim> computeUnionAABB(Iter first, Iter last) noexcept {
    Math::AxisAlignedBox<T, Dim> result;
    for (auto it = first; it != last; ++it) {
        result.extend(*it);
    }
    return result;
}

/**
 * @brief Compute the centroid AABB of a range of boxes (i.e., min/max of centroids).
 */
template <Dimension Dim, typename T, typename Iter>
Math::AxisAlignedBox<T, Dim> computeCentroidAABB(Iter first, Iter last) noexcept {
    using Vector = Math::Vector<T, Dim>;
    Vector minCentroid = Vector(std::numeric_limits<T>::max());
    Vector maxCentroid = Vector(std::numeric_limits<T>::lowest());
    for (auto it = first; it != last; ++it) {
        Vector centroid = it->center();
        minCentroid = minCentroid.componentWiseMin(centroid);
        maxCentroid = maxCentroid.componentWiseMax(centroid);
    }
    return Math::AxisAlignedBox<T, Dim>(minCentroid, maxCentroid);
}

// ============================================================================
//  Axis selection heuristics
// ============================================================================

/**
 * @brief Choose the longest axis of a bounding box.
 * @return 0=x, 1=y, 2=z (for 3D) or 0=x,1=y (for 2D).
 */
template <Dimension Dim, typename T>
uint8_t selectLongestAxis(const Math::AxisAlignedBox<T, Dim>& box) noexcept {
    Math::Vector<T, Dim> extents = box.extents();
    uint8_t bestAxis = 0;
    T maxExtent = extents[0];
    for (std::size_t i = 1; i < static_cast<std::size_t>(Dim); ++i) {
        if (extents[i] > maxExtent) {
            maxExtent = extents[i];
            bestAxis = static_cast<uint8_t>(i);
        }
    }
    return bestAxis;
}

/**
 * @brief Choose axis with largest centroid spread (for LBVH).
 */
template <Dimension Dim, typename T, typename Iter>
uint8_t selectAxisBySpread(Iter first, Iter last) noexcept {
    Math::AxisAlignedBox<T, Dim> centroidAABB = computeCentroidAABB<Dim, T>(first, last);
    return selectLongestAxis(centroidAABB);
}

// ============================================================================
//  Surface area heuristic (SAH) for BVH
// ============================================================================

/**
 * @brief Compute surface area of an AABB (2D: area, 3D: surface area).
 */
template <Dimension Dim, typename T>
T surfaceArea(const Math::AxisAlignedBox<T, Dim>& box) noexcept {
    if constexpr (Dim == Dim2) {
        Math::Vector<T, 2> ext = box.extents();
        return ext[0] * ext[1];
    } else if constexpr (Dim == Dim3) {
        Math::Vector<T, 3> ext = box.extents();
        return T(2) * (ext[0]*ext[1] + ext[0]*ext[2] + ext[1]*ext[2]);
    }
    return T(0);
}

/**
 * @brief Cost of a node with N primitives if we do not split.
 */
template <typename T>
T leafCost(T numPrimitives, T traversalCost = T(1), T intersectionCost = T(1)) noexcept {
    return numPrimitives * intersectionCost;
}

/**
 * @brief Cost of splitting a node into two children with given bounding boxes and primitive counts.
 * @param parentArea Surface area of parent (or we compute from child areas).
 */
template <typename T>
T splitCost(const Math::AxisAlignedBox<T, Dim2>& leftBox,
            const Math::AxisAlignedBox<T, Dim2>& rightBox,
            T leftCount, T rightCount,
            T traversalCost = T(1), T intersectionCost = T(1)) noexcept {
    if constexpr (Dim == Dim2) {
        T leftArea = surfaceArea(leftBox);
        T rightArea = surfaceArea(rightBox);
        T parentArea = surfaceArea(leftBox.hull(rightBox)); // or precomputed
        return traversalCost + (leftArea / parentArea) * leftCount * intersectionCost +
                               (rightArea / parentArea) * rightCount * intersectionCost;
    } else if constexpr (Dim == Dim3) {
        T leftArea = surfaceArea(leftBox);
        T rightArea = surfaceArea(rightBox);
        T parentArea = surfaceArea(leftBox.hull(rightBox));
        return traversalCost + (leftArea / parentArea) * leftCount * intersectionCost +
                               (rightArea / parentArea) * rightCount * intersectionCost;
    }
}

// ----------------------------------------------------------------------------
//  SAH split finding (sort centroids along axis and evaluate)
// ----------------------------------------------------------------------------

/**
 * @brief Find best split plane using SAH for a set of bounding boxes.
 * @tparam Iter Iterator over AxisAlignedBox<T,Dim>.
 * @param first, last Range of boxes.
 * @param centroidAxis Axis to sort centroids on.
 * @param bestSplitPos Output split position (coordinate value).
 * @return Minimum cost (or infinity if cannot split).
 */
template <Dimension Dim, typename T, typename Iter>
T findBestSAHSplit(Iter first, Iter last, uint8_t centroidAxis, T& bestSplitPos) {
    constexpr T INF = std::numeric_limits<T>::max();
    if (std::distance(first, last) <= 1) return INF;

    // Create array of indices with centroids
    using BoxType = typename std::iterator_traits<Iter>::value_type;
    std::vector<std::pair<T, BoxType>> centroids;
    centroids.reserve(std::distance(first, last));
    for (auto it = first; it != last; ++it) {
        T c = it->center()[centroidAxis];
        centroids.emplace_back(c, *it);
    }
    std::sort(centroids.begin(), centroids.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Precompute prefix/suffix bounding boxes and counts
    std::size_t N = centroids.size();
    std::vector<Math::AxisAlignedBox<T, Dim>> prefixBox(N);
    std::vector<Math::AxisAlignedBox<T, Dim>> suffixBox(N);
    std::vector<size_t> prefixCount(N), suffixCount(N);

    Math::AxisAlignedBox<T, Dim> cur;
    for (std::size_t i = 0; i < N; ++i) {
        cur.extend(centroids[i].second);
        prefixBox[i] = cur;
        prefixCount[i] = i + 1;
    }
    cur = Math::AxisAlignedBox<T, Dim>();
    for (std::size_t i = N; i-- > 0;) {
        cur.extend(centroids[i].second);
        suffixBox[i] = cur;
        suffixCount[i] = N - i;
    }

    T bestCost = INF;
    bestSplitPos = T(0);
    for (std::size_t i = 1; i < N; ++i) {
        // Split after i-1, left has i elements, right has N-i
        T cost = splitCost(prefixBox[i-1], suffixBox[i],
                           static_cast<T>(prefixCount[i-1]),
                           static_cast<T>(suffixCount[i]));
        if (cost < bestCost) {
            bestCost = cost;
            bestSplitPos = (centroids[i-1].first + centroids[i].first) * T(0.5);
        }
    }
    return bestCost;
}

// ============================================================================
//  Median split (for LBVH, based on Morton codes)
// ============================================================================

/**
 * @brief Partition a range of Morton codes into left/right halves by median.
 * @tparam Iter RandomAccessIterator of MortonCode.
 * @return Iterator to middle (first element of right half).
 */
template <typename Iter>
Iter medianSplit(Iter first, Iter last) {
    std::size_t n = std::distance(first, last);
    if (n <= 1) return last;
    auto mid = first + n / 2;
    std::nth_element(first, mid, last);
    return mid;
}

// ----------------------------------------------------------------------------
//  Equal‑volume split (for uniform grid or octree levels)
// ----------------------------------------------------------------------------

/**
 * @brief Split an AABB into 2^Dim children of equal volume.
 * @param box Parent box.
 * @return Array of child AABBs.
 */
template <Dimension Dim, typename T>
std::array<Math::AxisAlignedBox<T, Dim>, (Dim == Dim2 ? 4 : 8)>
equalVolumeSplit(const Math::AxisAlignedBox<T, Dim>& box) noexcept {
    constexpr uint8_t numChildren = (Dim == Dim2 ? 4 : 8);
    std::array<Math::AxisAlignedBox<T, Dim>, numChildren> children;
    Math::Vector<T, Dim> min = box.min();
    Math::Vector<T, Dim> max = box.max();
    Math::Vector<T, Dim> mid = box.center();

    for (uint8_t i = 0; i < numChildren; ++i) {
        Math::Vector<T, Dim> childMin = min;
        Math::Vector<T, Dim> childMax = max;
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            if ((i >> d) & 1) {
                childMin[d] = mid[d];
            } else {
                childMax[d] = mid[d];
            }
        }
        children[i] = Math::AxisAlignedBox<T, Dim>(childMin, childMax);
    }
    return children;
}

// ============================================================================
//  Object‑median split (based on object's centroid position along axis)
// ============================================================================

/**
 * @brief Split range of boxes based on median of centroids along given axis.
 * @tparam Iter Iterator of AxisAlignedBox.
 * @return Iterator to split point.
 */
template <Dimension Dim, typename T, typename Iter>
Iter objectMedianSplit(Iter first, Iter last, uint8_t axis) {
    using BoxType = typename std::iterator_traits<Iter>::value_type;
    std::vector<std::pair<T, BoxType>> temp;
    temp.reserve(std::distance(first, last));
    for (auto it = first; it != last; ++it) {
        temp.emplace_back(it->center()[axis], *it);
    }
    std::sort(temp.begin(), temp.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::size_t mid = temp.size() / 2;
    // Copy back? Simpler: we could modify original range via swap.
    // For simplicity, we assume we can reorder the input range using nth_element on centroids.
    // We'll implement using nth_element directly.
    std::vector<T> centroids;
    centroids.reserve(std::distance(first, last));
    for (auto it = first; it != last; ++it) centroids.push_back(it->center()[axis]);
    auto midIt = first + std::distance(first, last) / 2;
    std::nth_element(first, midIt, last,
        [axis](const BoxType& a, const BoxType& b) {
            return a.center()[axis] < b.center()[axis];
        });
    return midIt;
}

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_PARTITIONING_H_INCLUDED