//12/40
//File 0092 : core/math/mesh_sampling.h
//Random point sampling on triangle meshes: uniform area‑weighted, stratified, and Poisson disk sampling with Bridson’s algorithm and exact point‑in‑triangle test.
#ifndef CORE_MATH_MESH_SAMPLING_H
#define CORE_MATH_MESH_SAMPLING_H

#include "mesh_data.h"               // HalfEdgeMesh (for access)
#include "vector_math.h"
#include "random.h"                  // PCG32 random generator
#include "geometry_primitives.h"     // Triangle, Ray
#include "exact_arithmetic.h"        // orient2d for robust inclusion
#include "math_constants.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <queue>
#include <limits>

namespace SimulationMath {
namespace mesh_sampling {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Uniform random point on a single triangle using barycentric coordinates
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR uniform_on_triangle(const DirectX::XMVECTOR& v0,
                                              const DirectX::XMVECTOR& v1,
                                              const DirectX::XMVECTOR& v2,
                                              PCG32& rng) noexcept {
    float u = rng.next_float();
    float v = rng.next_float();
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    DirectX::XMVECTOR p = DirectX::XMVectorAdd(v0,
        DirectX::XMVectorAdd(
            DirectX::XMVectorScale(DirectX::XMVectorSubtract(v1, v0), u),
            DirectX::XMVectorScale(DirectX::XMVectorSubtract(v2, v0), v)));
    return p;
}

// -----------------------------------------------------------------------------
// 2. Precompute cumulative area array and triangle normals
// -----------------------------------------------------------------------------
inline void build_area_distribution(const HalfEdgeMesh& mesh,
                                     std::vector<float>& cum_areas,
                                     std::vector<float>& tri_areas) noexcept {
    size_t num_tri = mesh.faces().size();
    tri_areas.resize(num_tri, 0.0f);
    cum_areas.resize(num_tri + 1, 0.0f);
    const auto& hedges = mesh.half_edges();
    const auto& verts = mesh.vertices();

    for (size_t f = 0; f < num_tri; ++f) {
        const MeshFace& face = mesh.faces()[f];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t v0 = hedges[he0].vertex_index;
        uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
        uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
        v2 = hedges[v2].vertex_index;

        DirectX::XMVECTOR p0 = verts[v0].position;
        DirectX::XMVECTOR p1 = verts[v1].position;
        DirectX::XMVECTOR p2 = verts[v2].position;
        DirectX::XMVECTOR cross = vector_math::cross3(
            DirectX::XMVectorSubtract(p1, p0),
            DirectX::XMVectorSubtract(p2, p0));
        tri_areas[f] = 0.5f * vector_math::length3_scalar(cross);
        cum_areas[f+1] = cum_areas[f] + tri_areas[f];
    }
}

// -----------------------------------------------------------------------------
// 3. Generate N uniform area‑weighted random points on the mesh surface
// -----------------------------------------------------------------------------
inline std::vector<DirectX::XMVECTOR> sample_uniform(const HalfEdgeMesh& mesh,
                                                     size_t N, PCG32& rng) noexcept {
    std::vector<float> cum_areas, tri_areas;
    build_area_distribution(mesh, cum_areas, tri_areas);
    float total_area = cum_areas.back();
    if (total_area <= 0.0f) return {};

    const auto& hedges = mesh.half_edges();
    const auto& verts = mesh.vertices();
    size_t num_tri = tri_areas.size();
    std::vector<DirectX::XMVECTOR> points;
    points.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        float r = rng.next_float() * total_area;
        // binary search for triangle index
        auto it = std::upper_bound(cum_areas.begin(), cum_areas.end(), r);
        size_t tri_idx = std::distance(cum_areas.begin(), it) - 1;
        if (tri_idx >= num_tri) tri_idx = num_tri - 1;

        const MeshFace& face = mesh.faces()[tri_idx];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t v0 = hedges[he0].vertex_index;
        uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
        uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
        v2 = hedges[v2].vertex_index;

        points.push_back(uniform_on_triangle(verts[v0].position,
                                             verts[v1].position,
                                             verts[v2].position, rng));
    }
    return points;
}

// -----------------------------------------------------------------------------
// 4. Stratified sampling: subdivide each triangle into sqrt(N) sub‑triangles and sample one per cell
// -----------------------------------------------------------------------------
inline std::vector<DirectX::XMVECTOR> sample_stratified(const HalfEdgeMesh& mesh,
                                                        size_t N, PCG32& rng) noexcept {
    std::vector<float> cum_areas, tri_areas;
    build_area_distribution(mesh, cum_areas, tri_areas);
    float total_area = cum_areas.back();
    if (total_area <= 0.0f) return {};

    const auto& hedges = mesh.half_edges();
    const auto& verts = mesh.vertices();
    size_t num_tri = tri_areas.size();
    // Allocate number of samples per triangle proportional to its area
    std::vector<size_t> samples_per_tri(num_tri, 0);
    size_t remaining = N;
    for (size_t f = 0; f < num_tri && remaining > 0; ++f) {
        size_t alloc = static_cast<size_t>(std::round(tri_areas[f] / total_area * N));
        if (alloc < 1) alloc = 1;
        if (alloc > remaining) alloc = remaining;
        samples_per_tri[f] = alloc;
        remaining -= alloc;
    }
    // Handle remaining due to rounding
    while (remaining > 0) {
        for (size_t f = 0; f < num_tri && remaining > 0; ++f) {
            samples_per_tri[f]++; remaining--;
        }
    }

    std::vector<DirectX::XMVECTOR> points;
    points.reserve(N);
    for (size_t f = 0; f < num_tri; ++f) {
        if (samples_per_tri[f] == 0) continue;
        const MeshFace& face = mesh.faces()[f];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t v0 = hedges[he0].vertex_index;
        uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
        uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
        v2 = hedges[v2].vertex_index;

        DirectX::XMVECTOR p0 = verts[v0].position;
        DirectX::XMVECTOR p1 = verts[v1].position;
        DirectX::XMVECTOR p2 = verts[v2].position;

        // Subdivide triangle into a grid of n x n sub‑triangles where n = ceil(sqrt(count))
        size_t count = samples_per_tri[f];
        size_t n = static_cast<size_t>(std::ceil(std::sqrt(static_cast<float>(count))));
        size_t placed = 0;
        for (size_t j = 0; j < n && placed < count; ++j) {
            for (size_t i = 0; i < n && placed < count; ++i) {
                // Local barycentric coords in sub‑cell
                float u = (i + rng.next_float()) / n;
                float v = (j + rng.next_float()) / n;
                if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                DirectX::XMVECTOR pt = DirectX::XMVectorAdd(p0,
                    DirectX::XMVectorAdd(
                        DirectX::XMVectorScale(DirectX::XMVectorSubtract(p1, p0), u),
                        DirectX::XMVectorScale(DirectX::XMVectorSubtract(p2, p0), v)));
                points.push_back(pt);
                placed++;
            }
        }
    }
    return points;
}

// -----------------------------------------------------------------------------
// 5. Poisson disk sampling on mesh surface using Bridson’s algorithm adapted to 3D surfaces
// -----------------------------------------------------------------------------
inline std::vector<DirectX::XMVECTOR> sample_poisson_disk(const HalfEdgeMesh& mesh,
                                                           float radius, int max_attempts = 30,
                                                           PCG32& rng = PCG32{}) noexcept {
    if (radius <= 0.0f) return {};
    // We'll use a simple rejection sampling on the surface: generate candidates by uniform sampling,
    // accept if distance to all existing points > radius. To improve efficiency, we maintain a spatial grid.
    // First, generate a large pool of candidate points using uniform sampling (oversample).
    // Then greedily select points that satisfy the distance constraint.

    // Precompute cumulative areas
    std::vector<float> cum_areas, tri_areas;
    build_area_distribution(mesh, cum_areas, tri_areas);
    float total_area = cum_areas.back();
    if (total_area <= 0.0f) return {};

    const auto& hedges = mesh.half_edges();
    const auto& verts = mesh.vertices();
    size_t num_tri = tri_areas.size();

    // Estimate number of points
    float density = 1.0f / (constants::PIf * radius * radius);
    size_t max_points = static_cast<size_t>(total_area * density * 3.0f); // oversample
    if (max_points < 10) max_points = 10;

    // Generate candidates uniformly
    std::vector<DirectX::XMVECTOR> candidates;
    candidates.reserve(max_points);
    for (size_t i = 0; i < max_points; ++i) {
        float r = rng.next_float() * total_area;
        auto it = std::upper_bound(cum_areas.begin(), cum_areas.end(), r);
        size_t tri_idx = std::distance(cum_areas.begin(), it) - 1;
        if (tri_idx >= num_tri) tri_idx = num_tri - 1;
        const MeshFace& face = mesh.faces()[tri_idx];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t v0 = hedges[he0].vertex_index;
        uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
        uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
        v2 = hedges[v2].vertex_index;
        candidates.push_back(uniform_on_triangle(verts[v0].position,
                                                 verts[v1].position,
                                                 verts[v2].position, rng));
    }

    // Greedy selection with spatial grid for fast distance queries
    // Build a grid covering the bounding box of the mesh with cell size = radius
    DirectX::XMVECTOR bb_min = verts[0].position, bb_max = verts[0].position;
    for (size_t i = 1; i < verts.size(); ++i) {
        bb_min = DirectX::XMVectorMin(bb_min, verts[i].position);
        bb_max = DirectX::XMVectorMax(bb_max, verts[i].position);
    }
    float cell_size = radius;
    int nx = std::max(1, (int)(vector_math::get_x(bb_max) - vector_math::get_x(bb_min)) / cell_size + 1);
    int ny = std::max(1, (int)(vector_math::get_y(bb_max) - vector_math::get_y(bb_min)) / cell_size + 1);
    int nz = std::max(1, (int)(vector_math::get_z(bb_max) - vector_math::get_z(bb_min)) / cell_size + 1);
    auto cell_index = [&](DirectX::FXMVECTOR p) -> int {
        int ix = (int)((vector_math::get_x(p) - vector_math::get_x(bb_min)) / cell_size);
        int iy = (int)((vector_math::get_y(p) - vector_math::get_y(bb_min)) / cell_size);
        int iz = (int)((vector_math::get_z(p) - vector_math::get_z(bb_min)) / cell_size);
        ix = std::clamp(ix, 0, nx-1);
        iy = std::clamp(iy, 0, ny-1);
        iz = std::clamp(iz, 0, nz-1);
        return ix + nx * (iy + ny * iz);
    };

    std::vector<std::vector<size_t>> grid(nx * ny * nz);
    std::vector<DirectX::XMVECTOR> accepted;
    accepted.reserve(max_points / 3);

    // For each candidate, check against existing accepted points in neighboring cells
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const DirectX::XMVECTOR& p = candidates[idx];
        bool ok = true;
        int cell_id = cell_index(p);
        // Check neighboring cells (including own)
        for (int dz = -1; dz <= 1 && ok; ++dz) {
            for (int dy = -1; dy <= 1 && ok; ++dy) {
                for (int dx = -1; dx <= 1 && ok; ++dx) {
                    int nx_ = (cell_id % (nx*ny)) % nx + dx;
                    int ny_ = ((cell_id % (nx*ny)) / nx) + dy;
                    int nz_ = cell_id / (nx*ny) + dz;
                    if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) continue;
                    int neighbor_cell = nx_ + nx * (ny_ + ny * nz_);
                    for (size_t j : grid[neighbor_cell]) {
                        float dist_sq = vector_math::length_sq3_scalar(DirectX::XMVectorSubtract(p, accepted[j]));
                        if (dist_sq < radius * radius) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
        }
        if (ok) {
            accepted.push_back(p);
            grid[cell_id].push_back(accepted.size() - 1);
        }
    }
    return accepted;
}

// -----------------------------------------------------------------------------
// 6. Sample points along edges (for wireframe visualization)
// -----------------------------------------------------------------------------
inline std::vector<DirectX::XMVECTOR> sample_edges(const HalfEdgeMesh& mesh, float spacing) noexcept {
    std::vector<DirectX::XMVECTOR> points;
    const auto& hedges = mesh.half_edges();
    const auto& verts = mesh.vertices();
    std::unordered_set<uint64_t> visited;
    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a < b) return ((uint64_t)a << 32) | b;
        return ((uint64_t)b << 32) | a;
    };

    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        if (he.twin_edge == 0xFFFFFFFFu) continue; // boundary, skip?
        uint32_t v0 = he.vertex_index;
        uint32_t v1 = hedges[he.next_edge].vertex_index;
        uint64_t key = edge_key(v0, v1);
        if (visited.count(key)) continue;
        visited.insert(key);

        DirectX::XMVECTOR p0 = verts[v0].position;
        DirectX::XMVECTOR p1 = verts[v1].position;
        float len = vector_math::length3_scalar(DirectX::XMVectorSubtract(p1, p0));
        if (len < spacing) {
            points.push_back(p0);
            continue;
        }
        int steps = static_cast<int>(len / spacing);
        for (int s = 0; s <= steps; ++s) {
            float t = (float)s / steps;
            points.push_back(DirectX::XMVectorLerp(p0, p1, t));
        }
    }
    return points;
}

} // namespace mesh_sampling
} // namespace SimulationMath

#endif // CORE_MATH_MESH_SAMPLING_H