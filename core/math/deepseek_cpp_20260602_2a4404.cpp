//File 0101 : core/math/mesh_signed_distance_field.h
//Signed distance field (SDF) generation from closed triangle meshes: exact inside test, closest point via spatial query, dense 3D grid output with gradient computation.
#ifndef CORE_MATH_MESH_SIGNED_DISTANCE_FIELD_H
#define CORE_MATH_MESH_SIGNED_DISTANCE_FIELD_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "geometry_primitives.h"
#include "mesh_distance.h"           // MeshDistanceQuery, exact_inside_test
#include "math_constants.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace SimulationMath {
namespace sdf {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Signed distance field grid structure
// -----------------------------------------------------------------------------
struct SDFGrid {
    int nx, ny, nz;
    DirectX::XMVECTOR origin;   // world coordinate of corner (0,0,0)
    float voxel_size;
    std::vector<float> data;    // signed distance at each voxel center (negative inside)

    SDFGrid() noexcept : nx(0), ny(0), nz(0), voxel_size(1.0f) {}

    bool in_bounds(int ix, int iy, int iz) const noexcept {
        return ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz;
    }

    size_t index(int ix, int iy, int iz) const noexcept {
        return static_cast<size_t>(ix) + nx * (static_cast<size_t>(iy) + ny * static_cast<size_t>(iz));
    }

    float at(int ix, int iy, int iz) const noexcept {
        if (!in_bounds(ix, iy, iz)) return 0.0f;
        return data[index(ix, iy, iz)];
    }

    void set(int ix, int iy, int iz, float val) noexcept {
        if (in_bounds(ix, iy, iz))
            data[index(ix, iy, iz)] = val;
    }

    DirectX::XMVECTOR center(int ix, int iy, int iz) const noexcept {
        return DirectX::XMVectorAdd(origin,
            DirectX::XMVectorSet((ix + 0.5f) * voxel_size,
                                  (iy + 0.5f) * voxel_size,
                                  (iz + 0.5f) * voxel_size, 0.0f));
    }

    // Compute gradient of the SDF at a voxel using central differences (in world units)
    DirectX::XMVECTOR gradient(int ix, int iy, int iz) const noexcept {
        float inv_2h = 0.5f / voxel_size;
        float gx = (at(ix+1, iy, iz) - at(ix-1, iy, iz)) * inv_2h;
        float gy = (at(ix, iy+1, iz) - at(ix, iy-1, iz)) * inv_2h;
        float gz = (at(ix, iy, iz+1) - at(ix, iy, iz-1)) * inv_2h;
        return DirectX::XMVectorSet(gx, gy, gz, 0.0f);
    }
};

// -----------------------------------------------------------------------------
// 2. Compute signed distance field for a closed triangle mesh
//    resolution: number of voxels along the largest dimension
//    margin: extra space around the mesh (in multiples of voxel_size)
// -----------------------------------------------------------------------------
inline SDFGrid compute_signed_distance_field(const HalfEdgeMesh& mesh,
                                             int resolution = 128,
                                             float margin = 4.0f) noexcept {
    SDFGrid sdf;
    if (mesh.vertex_count() == 0) return sdf;

    // Bounding box of the mesh
    const auto& verts = mesh.vertices();
    DirectX::XMVECTOR bb_min = verts[0].position;
    DirectX::XMVECTOR bb_max = verts[0].position;
    for (size_t i = 1; i < verts.size(); ++i) {
        bb_min = DirectX::XMVectorMin(bb_min, verts[i].position);
        bb_max = DirectX::XMVectorMax(bb_max, verts[i].position);
    }

    // Determine cell size based on resolution along the longest axis
    float size_x = vector_math::get_x(bb_max) - vector_math::get_x(bb_min);
    float size_y = vector_math::get_y(bb_max) - vector_math::get_y(bb_min);
    float size_z = vector_math::get_z(bb_max) - vector_math::get_z(bb_min);
    float max_size = std::max({size_x, size_y, size_z});
    if (max_size <= 0.0f) max_size = 1.0f;
    float cell_size = max_size / (resolution - 2.0f * margin); // reserve margin cells

    // Apply margin
    sdf.voxel_size = cell_size;
    DirectX::XMVECTOR margin_vec = DirectX::XMVectorReplicate(margin * cell_size);
    bb_min = DirectX::XMVectorSubtract(bb_min, margin_vec);
    bb_max = DirectX::XMVectorAdd(bb_max, margin_vec);
    sdf.origin = bb_min;

    size_x = vector_math::get_x(bb_max) - vector_math::get_x(bb_min);
    size_y = vector_math::get_y(bb_max) - vector_math::get_y(bb_min);
    size_z = vector_math::get_z(bb_max) - vector_math::get_z(bb_min);
    sdf.nx = std::max(1, (int)(size_x / cell_size) + 1);
    sdf.ny = std::max(1, (int)(size_y / cell_size) + 1);
    sdf.nz = std::max(1, (int)(size_z / cell_size) + 1);

    size_t total = static_cast<size_t>(sdf.nx) * sdf.ny * sdf.nz;
    sdf.data.resize(total, 0.0f);

    // Build spatial query structure on the mesh for fast closest point queries
    MeshDistanceQuery dist_query(mesh, cell_size);

    // For each voxel center, compute unsigned distance to mesh, then determine sign with exact inside test
    for (int iz = 0; iz < sdf.nz; ++iz) {
        for (int iy = 0; iy < sdf.ny; ++iy) {
            for (int ix = 0; ix < sdf.nx; ++ix) {
                DirectX::XMVECTOR p = sdf.center(ix, iy, iz);
                DirectX::XMVECTOR closest;
                float dist = dist_query.closest_point(p, mesh, closest); // unsigned distance
                bool inside = exact_inside_test(p, mesh);
                float signed_dist = inside ? -dist : dist;
                sdf.set(ix, iy, iz, signed_dist);
            }
        }
    }
    return sdf;
}

// -----------------------------------------------------------------------------
// 3. Trilinear interpolation of SDF at an arbitrary point (world coordinates)
// -----------------------------------------------------------------------------
inline float sample_sdf(const SDFGrid& sdf, DirectX::FXMVECTOR world_point) noexcept {
    float x = (vector_math::get_x(world_point) - vector_math::get_x(sdf.origin)) / sdf.voxel_size - 0.5f;
    float y = (vector_math::get_y(world_point) - vector_math::get_y(sdf.origin)) / sdf.voxel_size - 0.5f;
    float z = (vector_math::get_z(world_point) - vector_math::get_z(sdf.origin)) / sdf.voxel_size - 0.5f;
    int ix0 = std::clamp(static_cast<int>(std::floor(x)), 0, sdf.nx - 2);
    int iy0 = std::clamp(static_cast<int>(std::floor(y)), 0, sdf.ny - 2);
    int iz0 = std::clamp(static_cast<int>(std::floor(z)), 0, sdf.nz - 2);
    float fx = x - ix0, fy = y - iy0, fz = z - iz0;

    float v000 = sdf.at(ix0,   iy0,   iz0);
    float v100 = sdf.at(ix0+1, iy0,   iz0);
    float v010 = sdf.at(ix0,   iy0+1, iz0);
    float v110 = sdf.at(ix0+1, iy0+1, iz0);
    float v001 = sdf.at(ix0,   iy0,   iz0+1);
    float v101 = sdf.at(ix0+1, iy0,   iz0+1);
    float v011 = sdf.at(ix0,   iy0+1, iz0+1);
    float v111 = sdf.at(ix0+1, iy0+1, iz0+1);

    auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
    return lerp(lerp(lerp(v000, v100, fx), lerp(v010, v110, fx), fy),
                lerp(lerp(v001, v101, fx), lerp(v011, v111, fx), fy), fz);
}

} // namespace sdf
} // namespace SimulationMath

#endif // CORE_MATH_MESH_SIGNED_DISTANCE_FIELD_H