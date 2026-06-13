//14/40
//File 0094 : core/math/mesh_voxelization.h
//Solid voxelization of closed triangle meshes: triangle‑AABB overlap marking, exact inside test for boundary voxels, flood‑fill from outside seed, and dense 3D grid export.
#ifndef CORE_MATH_MESH_VOXELIZATION_H
#define CORE_MATH_MESH_VOXELIZATION_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "geometry_primitives.h"     // AABB, Triangle
#include "exact_arithmetic.h"        // orient2d, orient3d for robust tests
#include "math_constants.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <limits>

namespace SimulationMath {
namespace voxelization {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. VoxelGrid structure: dense 3D binary array
// -----------------------------------------------------------------------------
struct VoxelGrid {
    int nx, ny, nz;
    DirectX::XMVECTOR origin;   // world coordinates of voxel (0,0,0) corner
    float voxel_size;
    std::vector<int> data;      // 0 = unknown, 1 = solid, 2 = empty (outside)

    VoxelGrid() noexcept : nx(0), ny(0), nz(0), voxel_size(1.0f) {}

    bool in_bounds(int ix, int iy, int iz) const noexcept {
        return ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz;
    }

    size_t index(int ix, int iy, int iz) const noexcept {
        return static_cast<size_t>(ix) + nx * (static_cast<size_t>(iy) + ny * static_cast<size_t>(iz));
    }

    int at(int ix, int iy, int iz) const noexcept {
        if (!in_bounds(ix, iy, iz)) return 0;
        return data[index(ix, iy, iz)];
    }

    void set(int ix, int iy, int iz, int val) noexcept {
        if (in_bounds(ix, iy, iz))
            data[index(ix, iy, iz)] = val;
    }

    DirectX::XMVECTOR center(int ix, int iy, int iz) const noexcept {
        return DirectX::XMVectorAdd(origin,
            DirectX::XMVectorSet((ix + 0.5f) * voxel_size,
                                  (iy + 0.5f) * voxel_size,
                                  (iz + 0.5f) * voxel_size, 0.0f));
    }
};

// -----------------------------------------------------------------------------
// 2. Triangle‑AABB overlap test (conservative)
// -----------------------------------------------------------------------------
inline bool triangle_aabb_overlap(const DirectX::XMVECTOR& v0,
                                  const DirectX::XMVECTOR& v1,
                                  const DirectX::XMVECTOR& v2,
                                  const geometry::AABB& box) noexcept {
    // Use separating axis theorem (SAT) with 13 axes.
    // For brevity, we use the quick AABB of triangle and then overlap test.
    geometry::AABB tri_box;
    tri_box.min = DirectX::XMVectorMin(v0, DirectX::XMVectorMin(v1, v2));
    tri_box.max = DirectX::XMVectorMax(v0, DirectX::XMVectorMax(v1, v2));
    if (!tri_box.overlaps(box)) return false;
    // Additional axes: triangle normal, edge cross products, etc. (not implemented for brevity,
    // but the AABB overlap is fast and conservative; false positives are acceptable for voxelization.)
    return true;
}

// -----------------------------------------------------------------------------
// 3. Exact inside test for a point (same as in mesh_boolean)
// -----------------------------------------------------------------------------
inline bool exact_inside_test(const DirectX::XMVECTOR& point, const HalfEdgeMesh& mesh) noexcept {
    double px = vector_math::get_x(point);
    double py = vector_math::get_y(point);
    double pz = vector_math::get_z(point);
    int winding = 0;
    const auto& hedges = mesh.half_edges();
    const auto& verts = mesh.vertices();
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const MeshFace& face = mesh.faces()[f];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t v0 = hedges[he0].vertex_index;
        uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
        uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
        v2 = hedges[v2].vertex_index;

        double ax = vector_math::get_x(verts[v0].position);
        double ay = vector_math::get_y(verts[v0].position);
        double az = vector_math::get_z(verts[v0].position);
        double bx = vector_math::get_x(verts[v1].position);
        double by = vector_math::get_y(verts[v1].position);
        double bz = vector_math::get_z(verts[v1].position);
        double cx = vector_math::get_x(verts[v2].position);
        double cy = vector_math::get_y(verts[v2].position);
        double cz = vector_math::get_z(verts[v2].position);

        double nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
        double ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
        double nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        double denom = nx; // ray direction (1,0,0)
        if (std::fabs(denom) < 1e-30) continue;
        double t = - ((px - ax) * nx + (py - ay) * ny + (pz - az) * nz) / denom;
        if (t <= 0.0) continue;

        double ix = px + t;
        double iy = py;
        double iz = pz;
        double o1 = orient2d(ay, az, by, bz, iy, iz);
        double o2 = orient2d(by, bz, cy, cz, iy, iz);
        double o3 = orient2d(cy, cz, ay, az, iy, iz);
        if ((o1 >= 0.0 && o2 >= 0.0 && o3 >= 0.0) || (o1 <= 0.0 && o2 <= 0.0 && o3 <= 0.0)) {
            winding++;
        }
    }
    return (winding % 2) == 1;
}

// -----------------------------------------------------------------------------
// 4. Find an outside seed voxel (a corner of the grid that is guaranteed outside)
// -----------------------------------------------------------------------------
inline void find_outside_seed(const VoxelGrid& grid, int& sx, int& sy, int& sz) noexcept {
    // The grid bounding box is slightly larger than the mesh, so the corner is outside.
    sx = 0; sy = 0; sz = 0;
}

// -----------------------------------------------------------------------------
// 5. Mark voxels that intersect the mesh (triangle‑AABB overlap) as boundary.
//    For each triangle, compute its AABB, then iterate over voxels within that AABB and set to boundary (value 3).
// -----------------------------------------------------------------------------
inline void mark_boundary_voxels(const HalfEdgeMesh& mesh, VoxelGrid& grid) noexcept {
    const auto& hedges = mesh.half_edges();
    const auto& verts = mesh.vertices();
    // Use a temporary value for boundary (3)
    const int BOUNDARY = 3;
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
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
        geometry::AABB tri_bbox;
        tri_bbox.min = DirectX::XMVectorMin(p0, DirectX::XMVectorMin(p1, p2));
        tri_bbox.max = DirectX::XMVectorMax(p0, DirectX::XMVectorMax(p1, p2));

        // Compute voxel index range that this triangle's AABB overlaps
        float inv_vs = 1.0f / grid.voxel_size;
        int ix_min = std::max(0, (int)((vector_math::get_x(tri_bbox.min) - vector_math::get_x(grid.origin)) * inv_vs));
        int iy_min = std::max(0, (int)((vector_math::get_y(tri_bbox.min) - vector_math::get_y(grid.origin)) * inv_vs));
        int iz_min = std::max(0, (int)((vector_math::get_z(tri_bbox.min) - vector_math::get_z(grid.origin)) * inv_vs));
        int ix_max = std::min(grid.nx-1, (int)((vector_math::get_x(tri_bbox.max) - vector_math::get_x(grid.origin)) * inv_vs));
        int iy_max = std::min(grid.ny-1, (int)((vector_math::get_y(tri_bbox.max) - vector_math::get_y(grid.origin)) * inv_vs));
        int iz_max = std::min(grid.nz-1, (int)((vector_math::get_z(tri_bbox.max) - vector_math::get_z(grid.origin)) * inv_vs));

        for (int iz = iz_min; iz <= iz_max; ++iz) {
            for (int iy = iy_min; iy <= iy_max; ++iy) {
                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    // Quick AABB overlap (conservative, but we can also do exact SAT; for voxelization, AABB overlap is fine)
                    geometry::AABB voxel_box;
                    voxel_box.min = DirectX::XMVectorSet(
                        (ix) * grid.voxel_size + vector_math::get_x(grid.origin),
                        (iy) * grid.voxel_size + vector_math::get_y(grid.origin),
                        (iz) * grid.voxel_size + vector_math::get_z(grid.origin), 0.0f);
                    voxel_box.max = DirectX::XMVectorAdd(voxel_box.min,
                        DirectX::XMVectorReplicate(grid.voxel_size));
                    if (triangle_aabb_overlap(p0, p1, p2, voxel_box)) {
                        grid.set(ix, iy, iz, BOUNDARY);
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 6. Flood‑fill from an outside seed: mark all reachable non‑boundary voxels as empty (outside).
// -----------------------------------------------------------------------------
inline void flood_fill_outside(VoxelGrid& grid, int start_x, int start_y, int start_z) noexcept {
    const int EMPTY = 2;
    const int BOUNDARY = 3;
    if (!grid.in_bounds(start_x, start_y, start_z)) return;
    if (grid.at(start_x, start_y, start_z) == BOUNDARY) return;

    std::queue<std::tuple<int,int,int>> q;
    q.push({start_x, start_y, start_z});
    grid.set(start_x, start_y, start_z, EMPTY);

    const int neigh[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

    while (!q.empty()) {
        auto [x, y, z] = q.front(); q.pop();
        for (int k = 0; k < 6; ++k) {
            int nx_ = x + neigh[k][0];
            int ny_ = y + neigh[k][1];
            int nz_ = z + neigh[k][2];
            if (!grid.in_bounds(nx_, ny_, nz_)) continue;
            int val = grid.at(nx_, ny_, nz_);
            if (val == EMPTY || val == BOUNDARY) continue; // already visited or boundary
            grid.set(nx_, ny_, nz_, EMPTY);
            q.push({nx_, ny_, nz_});
        }
    }
}

// -----------------------------------------------------------------------------
// 7. Main voxelization routine:
//    a) create grid, mark boundary voxels,
//    b) classify boundary voxels as inside/outside using exact test,
//    c) flood fill outside from a seed,
//    d) remaining unknown voxels become inside (solid).
// -----------------------------------------------------------------------------
inline VoxelGrid voxelize_mesh(const HalfEdgeMesh& mesh, float voxel_size) noexcept {
    VoxelGrid grid;
    if (mesh.vertex_count() == 0) return grid;

    const auto& verts = mesh.vertices();
    DirectX::XMVECTOR bb_min = verts[0].position;
    DirectX::XMVECTOR bb_max = verts[0].position;
    for (size_t i = 1; i < verts.size(); ++i) {
        bb_min = DirectX::XMVectorMin(bb_min, verts[i].position);
        bb_max = DirectX::XMVectorMax(bb_max, verts[i].position);
    }
    bb_min = DirectX::XMVectorSubtract(bb_min, DirectX::XMVectorReplicate(voxel_size));
    bb_max = DirectX::XMVectorAdd(bb_max, DirectX::XMVectorReplicate(voxel_size));
    grid.origin = bb_min;
    grid.voxel_size = voxel_size;

    float sizeX = vector_math::get_x(bb_max) - vector_math::get_x(bb_min);
    float sizeY = vector_math::get_y(bb_max) - vector_math::get_y(bb_min);
    float sizeZ = vector_math::get_z(bb_max) - vector_math::get_z(bb_min);
    grid.nx = std::max(1, (int)(sizeX / voxel_size) + 1);
    grid.ny = std::max(1, (int)(sizeY / voxel_size) + 1);
    grid.nz = std::max(1, (int)(sizeZ / voxel_size) + 1);

    size_t total = static_cast<size_t>(grid.nx) * grid.ny * grid.nz;
    grid.data.assign(total, 0); // all unknown

    // Step 1: mark boundary voxels (value 3)
    mark_boundary_voxels(mesh, grid);

    // Step 2: classify boundary voxels as inside (1) or outside (2) using exact test
    for (int iz = 0; iz < grid.nz; ++iz) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int ix = 0; ix < grid.nx; ++ix) {
                if (grid.at(ix, iy, iz) == 3) {
                    DirectX::XMVECTOR p = grid.center(ix, iy, iz);
                    bool inside = exact_inside_test(p, mesh);
                    grid.set(ix, iy, iz, inside ? 1 : 2);
                }
            }
        }
    }

    // Step 3: flood fill outside from a corner seed
    int sx, sy, sz;
    find_outside_seed(grid, sx, sy, sz);
    flood_fill_outside(grid, sx, sy, sz);

    // Step 4: any remaining unknown voxels (0) are inside (solid)
    for (int iz = 0; iz < grid.nz; ++iz) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int ix = 0; ix < grid.nx; ++ix) {
                if (grid.at(ix, iy, iz) == 0) {
                    grid.set(ix, iy, iz, 1); // solid
                }
            }
        }
    }

    return grid;
}

// -----------------------------------------------------------------------------
// 8. Export solid voxel centers as a point list
// -----------------------------------------------------------------------------
inline std::vector<DirectX::XMVECTOR> export_solid_voxel_centers(const VoxelGrid& grid) noexcept {
    std::vector<DirectX::XMVECTOR> centers;
    for (int iz = 0; iz < grid.nz; ++iz)
        for (int iy = 0; iy < grid.ny; ++iy)
            for (int ix = 0; ix < grid.nx; ++ix)
                if (grid.at(ix, iy, iz) == 1) // solid
                    centers.push_back(grid.center(ix, iy, iz));
    return centers;
}

} // namespace voxelization
} // namespace SimulationMath

#endif // CORE_MATH_MESH_VOXELIZATION_H