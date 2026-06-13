//File 0060 : core/math/broadphase.h
//Broad‑phase collision detection: sweep‑and‑prune (1‑axis), spatial hashing, and uniform 3D grid with SIMD‑accelerated cell enumeration and AABB overlap tests.
#ifndef CORE_MATH_BROADPHASE_H
#define CORE_MATH_BROADPHASE_H

#include "vector_math.h"
#include "geometry_primitives.h"
#include "space_filling_curves.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>

namespace SimulationMath {
namespace broadphase {

// -----------------------------------------------------------------------------
// 1. A pair of overlapping primitive indices
// -----------------------------------------------------------------------------
struct OverlapPair {
    uint32_t a;
    uint32_t b;
    OverlapPair() noexcept : a(0), b(0) {}
    OverlapPair(uint32_t _a, uint32_t _b) noexcept : a(_a), b(_b) {}
    bool operator==(const OverlapPair& o) const noexcept { return (a == o.a && b == o.b) || (a == o.b && b == o.a); }
};

// -----------------------------------------------------------------------------
// 2. Hash function for overlap pair (unordered_set usage)
// -----------------------------------------------------------------------------
struct OverlapPairHash {
    size_t operator()(const OverlapPair& p) const noexcept {
        uint64_t key = (static_cast<uint64_t>(std::min(p.a, p.b)) << 32) | std::max(p.a, p.b);
        return std::hash<uint64_t>()(key);
    }
};

// -----------------------------------------------------------------------------
// 3. Sweep and Prune (1‑axis) – sorts AABBs by min.x, tracks active set
// -----------------------------------------------------------------------------
class SweepAndPrune1D {
public:
    struct Entry {
        float value;        // min.x or max.x
        uint32_t id;        // object index
        bool is_min;        // true = min, false = max
    };

    SweepAndPrune1D() = default;

    void build(const std::vector<geometry::AABB>& aabbs,
               std::vector<OverlapPair>& out_pairs) noexcept {
        size_t n = aabbs.size();
        if (n < 2) { out_pairs.clear(); return; }

        // Create sorted list of interval endpoints
        std::vector<Entry> endpoints;
        endpoints.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) {
            float min_x = vector_math::get_x(aabbs[i].min);
            float max_x = vector_math::get_x(aabbs[i].max);
            endpoints.push_back({min_x, static_cast<uint32_t>(i), true});
            endpoints.push_back({max_x, static_cast<uint32_t>(i), false});
        }

        // Sort by value, then by is_min (min before max on tie)
        std::sort(endpoints.begin(), endpoints.end(),
                  [](const Entry& a, const Entry& b) {
                      if (a.value != b.value) return a.value < b.value;
                      return a.is_min && !b.is_min; // min before max when tied
                  });

        // Active set: indices currently overlapping in x‑axis
        std::vector<uint32_t> active;
        out_pairs.clear();

        for (const auto& e : endpoints) {
            if (e.is_min) {
                // Check against all active for real 3D AABB overlap
                for (uint32_t other : active) {
                    if (aabbs[e.id].overlaps(aabbs[other])) {
                        out_pairs.emplace_back(e.id, other);
                    }
                }
                active.push_back(e.id);
            } else {
                // Remove from active set
                auto it = std::find(active.begin(), active.end(), e.id);
                if (it != active.end()) {
                    *it = active.back();
                    active.pop_back();
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// 4. Spatial Hashing – maps cell coordinates to object lists via hash map
// -----------------------------------------------------------------------------
class SpatialHashBroadphase {
public:
    SpatialHashBroadphase(float cell_size) : cell_size_(cell_size), inv_cell_size_(1.0f / cell_size) {}

    void build(const std::vector<geometry::AABB>& aabbs,
               std::vector<OverlapPair>& out_pairs) noexcept {
        out_pairs.clear();
        size_t n = aabbs.size();
        if (n < 2) return;

        // Clear previous hash
        hash_map_.clear();

        // For each AABB, enumerate all cells it overlaps and insert index
        for (size_t i = 0; i < n; ++i) {
            const auto& aabb = aabbs[i];
            // Compute integer cell range that the AABB overlaps
            int cx_min = static_cast<int>(std::floor(vector_math::get_x(aabb.min) * inv_cell_size_));
            int cy_min = static_cast<int>(std::floor(vector_math::get_y(aabb.min) * inv_cell_size_));
            int cz_min = static_cast<int>(std::floor(vector_math::get_z(aabb.min) * inv_cell_size_));
            int cx_max = static_cast<int>(std::floor(vector_math::get_x(aabb.max) * inv_cell_size_));
            int cy_max = static_cast<int>(std::floor(vector_math::get_y(aabb.max) * inv_cell_size_));
            int cz_max = static_cast<int>(std::floor(vector_math::get_z(aabb.max) * inv_cell_size_));

            for (int cz = cz_min; cz <= cz_max; ++cz) {
                for (int cy = cy_min; cy <= cy_max; ++cy) {
                    for (int cx = cx_min; cx <= cx_max; ++cx) {
                        uint64_t key = compute_cell_key(cx, cy, cz);
                        auto& bucket = hash_map_[key];
                        // Check against existing objects in this cell
                        for (uint32_t other : bucket) {
                            if (aabb.overlaps(aabbs[other])) {
                                out_pairs.emplace_back(static_cast<uint32_t>(i), other);
                            }
                        }
                        bucket.push_back(static_cast<uint32_t>(i));
                    }
                }
            }
        }
        // Remove duplicates from out_pairs? Not done; caller may filter.
    }

private:
    float cell_size_;
    float inv_cell_size_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> hash_map_;

    uint64_t compute_cell_key(int cx, int cy, int cz) const noexcept {
        // Combine coordinates into 64‑bit using Morton encoding for locality
        uint32_t ux = static_cast<uint32_t>(cx >= 0 ? cx * 2 : -cx * 2 + 1);
        uint32_t uy = static_cast<uint32_t>(cy >= 0 ? cy * 2 : -cy * 2 + 1);
        uint32_t uz = static_cast<uint32_t>(cz >= 0 ? cz * 2 : -cz * 2 + 1);
        return space_filling::morton3D_encode_64(ux, uy, uz);
    }
};

// -----------------------------------------------------------------------------
// 5. Uniform 3D Grid – explicit dense grid (for moderate world sizes)
// -----------------------------------------------------------------------------
class UniformGridBroadphase {
public:
    UniformGridBroadphase(float cell_size, uint32_t grid_size)
        : cell_size_(cell_size), inv_cell_size_(1.0f / cell_size),
          grid_size_(grid_size), total_cells_(grid_size_ * grid_size_ * grid_size_) {}

    void build(const std::vector<geometry::AABB>& aabbs,
               std::vector<OverlapPair>& out_pairs) noexcept {
        out_pairs.clear();
        size_t n = aabbs.size();
        if (n < 2) return;

        // Clear grid
        cells_.assign(total_cells_, {});

        for (size_t i = 0; i < n; ++i) {
            const auto& aabb = aabbs[i];
            int cx_min = std::max(0, std::min(static_cast<int>(grid_size_ - 1),
                static_cast<int>(vector_math::get_x(aabb.min) * inv_cell_size_)));
            int cy_min = std::max(0, std::min(static_cast<int>(grid_size_ - 1),
                static_cast<int>(vector_math::get_y(aabb.min) * inv_cell_size_)));
            int cz_min = std::max(0, std::min(static_cast<int>(grid_size_ - 1),
                static_cast<int>(vector_math::get_z(aabb.min) * inv_cell_size_)));
            int cx_max = std::max(0, std::min(static_cast<int>(grid_size_ - 1),
                static_cast<int>(vector_math::get_x(aabb.max) * inv_cell_size_)));
            int cy_max = std::max(0, std::min(static_cast<int>(grid_size_ - 1),
                static_cast<int>(vector_math::get_y(aabb.max) * inv_cell_size_)));
            int cz_max = std::max(0, std::min(static_cast<int>(grid_size_ - 1),
                static_cast<int>(vector_math::get_z(aabb.max) * inv_cell_size_)));

            for (int cz = cz_min; cz <= cz_max; ++cz) {
                for (int cy = cy_min; cy <= cy_max; ++cy) {
                    for (int cx = cx_min; cx <= cx_max; ++cx) {
                        size_t cell_idx = cell_index(cx, cy, cz);
                        auto& cell = cells_[cell_idx];
                        for (uint32_t other : cell) {
                            if (aabb.overlaps(aabbs[other])) {
                                out_pairs.emplace_back(static_cast<uint32_t>(i), other);
                            }
                        }
                        cell.push_back(static_cast<uint32_t>(i));
                    }
                }
            }
        }
    }

private:
    float cell_size_;
    float inv_cell_size_;
    uint32_t grid_size_;
    size_t total_cells_;
    std::vector<std::vector<uint32_t>> cells_;

    size_t cell_index(int cx, int cy, int cz) const noexcept {
        return static_cast<size_t>(cx) + grid_size_ * (static_cast<size_t>(cy) + grid_size_ * static_cast<size_t>(cz));
    }
};

} // namespace broadphase
} // namespace SimulationMath

#endif // CORE_MATH_BROADPHASE_H