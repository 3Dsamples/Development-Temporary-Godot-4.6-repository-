//File 0046 : core/math/spatial_partitioning.h
//Spatial partitioning algorithms for BVH/octree construction: median split, SAH full, SAH binned, Morton pre‑sort; uses AABB from geometry_primitives.h.
#ifndef CORE_MATH_SPATIAL_PARTITIONING_H
#define CORE_MATH_SPATIAL_PARTITIONING_H

#include "vector_math.h"
#include "geometry_primitives.h"
#include "space_filling_curves.h"
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdint>

namespace SimulationMath {
namespace spatial_partition {

// -----------------------------------------------------------------------------
// 1. Median split: partition range [first,last) by centroid axis; returns iterator to first element of right child.
// -----------------------------------------------------------------------------
template <typename PrimitiveIterator, typename CentroidFunc>
PrimitiveIterator median_split(PrimitiveIterator first, PrimitiveIterator last,
                               CentroidFunc centroid_func, int axis) noexcept {
    size_t n = std::distance(first, last);
    if (n <= 1) return first;
    auto mid = first + n / 2;
    std::nth_element(first, mid, last,
        [&](const auto& a, const auto& b) {
            auto ca = centroid_func(a);
            auto cb = centroid_func(b);
            float va = (axis == 0) ? vector_math::get_x(ca) : (axis == 1) ? vector_math::get_y(ca) : vector_math::get_z(ca);
            float vb = (axis == 0) ? vector_math::get_x(cb) : (axis == 1) ? vector_math::get_y(cb) : vector_math::get_z(cb);
            return va < vb;
        });
    return mid;
}

// -----------------------------------------------------------------------------
// 2. Compute AABB for a range of primitives given a function that extracts AABB.
// -----------------------------------------------------------------------------
template <typename PrimitiveIterator, typename AABBFunc>
geometry::AABB compute_range_aabb(PrimitiveIterator first, PrimitiveIterator last,
                                  AABBFunc&& aabb_func) noexcept {
    geometry::AABB box;
    for (auto it = first; it != last; ++it)
        box.extend(aabb_func(*it));
    return box;
}

// -----------------------------------------------------------------------------
// 3. Full SAH split: evaluate all possible splits and pick minimum cost.
//    Returns the split iterator and left/right AABBs.
// -----------------------------------------------------------------------------
template <typename PrimitiveIterator, typename CentroidFunc>
struct SahResult {
    PrimitiveIterator split_pos;
    geometry::AABB left_aabb;
    geometry::AABB right_aabb;
    float cost;
};

template <typename PrimitiveIterator, typename CentroidFunc>
SahResult<PrimitiveIterator, CentroidFunc> sah_full_split(
    PrimitiveIterator first, PrimitiveIterator last,
    CentroidFunc centroid_func, int axis, const geometry::AABB& node_bbox) noexcept
{
    size_t n = std::distance(first, last);
    SahResult<PrimitiveIterator, CentroidFunc> result;
    result.cost = std::numeric_limits<float>::max();
    if (n <= 1) {
        result.split_pos = first;
        result.left_aabb = geometry::AABB();
        result.right_aabb = geometry::AABB();
        return result;
    }

    // Sort primitives by centroid on axis
    std::sort(first, last, [&](const auto& a, const auto& b) {
        auto ca = centroid_func(a); auto cb = centroid_func(b);
        float fa = (axis==0)?vector_math::get_x(ca):(axis==1)?vector_math::get_y(ca):vector_math::get_z(ca);
        float fb = (axis==0)?vector_math::get_x(cb):(axis==1)?vector_math::get_y(cb):vector_math::get_z(cb);
        return fa < fb;
    });

    // Precompute right-side AABBs
    std::vector<geometry::AABB> right_aabbs(n);
    right_aabbs[n-1] = geometry::AABB();
    right_aabbs[n-1].extend(centroid_func(*(first + n - 1)));
    for (size_t i = n-1; i > 0; --i) {
        right_aabbs[i-1] = right_aabbs[i];
        right_aabbs[i-1].extend(centroid_func(*(first + i - 1)));
    }

    geometry::AABB left_aabb;
    for (size_t i = 0; i < n - 1; ++i) {
        left_aabb.extend(centroid_func(*(first + i)));
        const auto& right_aabb = right_aabbs[i+1];
        float cost_val = left_aabb.surface_area() * (i + 1) + right_aabb.surface_area() * (n - i - 1);
        if (cost_val < result.cost) {
            result.cost = cost_val;
            result.split_pos = first + i + 1;
            result.left_aabb = left_aabb;
            result.right_aabb = right_aabb;
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// 4. Binned SAH split: divide centroids into bins, compute cost per bin boundary.
// -----------------------------------------------------------------------------
template <int Bins = 32, typename PrimitiveIterator, typename CentroidFunc>
SahResult<PrimitiveIterator, CentroidFunc> sah_binned_split(
    PrimitiveIterator first, PrimitiveIterator last,
    CentroidFunc centroid_func, int axis, const geometry::AABB& node_bbox) noexcept
{
    size_t n = std::distance(first, last);
    SahResult<PrimitiveIterator, CentroidFunc> result;
    result.cost = std::numeric_limits<float>::max();
    if (n <= 2) {
        result.split_pos = first + n/2;
        result.left_aabb  = compute_range_aabb(first, result.split_pos, [&](const auto& p){ return centroid_func(p); });
        result.right_aabb = compute_range_aabb(result.split_pos, last, [&](const auto& p){ return centroid_func(p); });
        return result;
    }

    // Determine axis range of centroids
    float axis_min = std::numeric_limits<float>::max();
    float axis_max = std::numeric_limits<float>::lowest();
    std::vector<float> projs(n);
    std::vector<geometry::AABB> aabbs(n);
    for (size_t i = 0; i < n; ++i) {
        auto c = centroid_func(*(first + i));
        float val = (axis==0)?vector_math::get_x(c):(axis==1)?vector_math::get_y(c):vector_math::get_z(c);
        projs[i] = val;
        aabbs[i] = geometry::AABB(c, c);
        if (val < axis_min) axis_min = val;
        if (val > axis_max) axis_max = val;
    }
    if (axis_max <= axis_min) {
        // degenerate – fall back to median
        auto mid = first + n/2;
        result.split_pos = mid;
        result.left_aabb  = compute_range_aabb(first, mid, [&](const auto& p){ return centroid_func(p); });
        result.right_aabb = compute_range_aabb(mid, last, [&](const auto& p){ return centroid_func(p); });
        result.cost = result.left_aabb.surface_area() * (n/2) + result.right_aabb.surface_area() * (n - n/2);
        return result;
    }

    float scale = (Bins - 1) / (axis_max - axis_min + 1e-12f);
    std::array<float, Bins> binCount{};
    std::array<geometry::AABB, Bins> binBox;
    for (auto& bb : binBox) bb = geometry::AABB();

    for (size_t i = 0; i < n; ++i) {
        int bin = static_cast<int>(scale * (projs[i] - axis_min));
        bin = std::max(0, std::min(bin, Bins-1));
        binCount[bin] += 1.0f;
        binBox[bin].extend(aabbs[i].min);
        binBox[bin].extend(aabbs[i].max);
    }

    // Sweep from left
    std::array<float, Bins> leftCost{};
    std::array<geometry::AABB, Bins> leftBox;
    leftBox[0] = binBox[0];
    leftCost[0] = binCount[0] * leftBox[0].surface_area();
    for (int i = 1; i < Bins; ++i) {
        leftBox[i] = leftBox[i-1];
        leftBox[i].extend(binBox[i].min);
        leftBox[i].extend(binBox[i].max);
        leftCost[i] = leftCost[i-1] + binCount[i] * leftBox[i].surface_area();
    }

    // Sweep from right
    std::array<float, Bins> rightCost{};
    std::array<geometry::AABB, Bins> rightBox;
    rightBox[Bins-1] = binBox[Bins-1];
    rightCost[Bins-1] = binCount[Bins-1] * rightBox[Bins-1].surface_area();
    for (int i = Bins-2; i >= 0; --i) {
        rightBox[i] = rightBox[i+1];
        rightBox[i].extend(binBox[i].min);
        rightBox[i].extend(binBox[i].max);
        rightCost[i] = rightCost[i+1] + binCount[i] * rightBox[i].surface_area();
    }

    // Find best bin split
    float best_cost = std::numeric_limits<float>::max();
    int best_bin = 0;
    for (int i = 0; i < Bins-1; ++i) {
        float cost = leftCost[i] + rightCost[i+1];
        if (cost < best_cost) {
            best_cost = cost;
            best_bin = i;
        }
    }

    // Partition primitives by the bin split (exact into left/right groups)
    auto mid = std::partition(first, last, [&](const auto& prim) {
        auto c = centroid_func(prim);
        float val = (axis==0)?vector_math::get_x(c):(axis==1)?vector_math::get_y(c):vector_math::get_z(c);
        int bin = static_cast<int>(scale * (val - axis_min));
        bin = std::max(0, std::min(bin, Bins-1));
        return bin <= best_bin;
    });

    result.split_pos  = mid;
    result.left_aabb  = compute_range_aabb(first, mid, [&](const auto& p){ return centroid_func(p); });
    result.right_aabb = compute_range_aabb(mid, last, [&](const auto& p){ return centroid_func(p); });
    result.cost = best_cost;
    return result;
}

// -----------------------------------------------------------------------------
// 5. Morton pre‑sort: compute Morton codes and sort primitives, returning the sorted range (in‑place).
// -----------------------------------------------------------------------------
template <typename PrimitiveIterator, typename MortonFunc>
void morton_sort(PrimitiveIterator first, PrimitiveIterator last, MortonFunc&& morton_func) noexcept {
    size_t n = std::distance(first, last);
    if (n <= 1) return;
    std::vector<uint64_t> keys(n);
    std::vector<std::remove_reference_t<decltype(*first)>> temp(first, last);
    for (size_t i = 0; i < n; ++i)
        keys[i] = morton_func(temp[i]);
    // sort indices
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });
    // write back sorted
    for (size_t i = 0; i < n; ++i)
        *first++ = temp[idx[i]];
}

} // namespace spatial_partition
} // namespace SimulationMath

#endif // CORE_MATH_SPATIAL_PARTITIONING_H