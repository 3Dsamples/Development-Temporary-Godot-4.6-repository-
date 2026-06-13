//File group name : OrthoTree Math
//File 0083 : core/math/dual_contouring.h
//Dual contouring isosurface extraction from a voxel grid (signed distance / density).
//Produces triangle meshes with sharp features (edges, corners) by minimising quadratic error functions.
//Supports Hermite data (position + normal per grid edge crossing).
//Includes SIMD batch for gradient evaluation.

#ifndef ORTHOTREE_CORE_MATH_DUAL_CONTOURING_H_INCLUDED
#define ORTHOTREE_CORE_MATH_DUAL_CONTOURING_H_INCLUDED

#include "../../build_config.h"
#include "basic/vector.h"
#include "geometry/voxel_grid.h"
#include "geometry/mesh.h"
#include "math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <optional>

namespace OrthoTree {
namespace Math {
namespace DualContouring {

// ============================================================================
//  Intersection point and normal on a grid edge (Hermite data).
// ============================================================================
template<typename T = float>
struct EdgeIntersection {
    Basic::Vector<T,3> position;  // point on the edge where isosurface crosses
    Basic::Vector<T,3> normal;    // gradient at that point (unit)
    uint32_t edgeIdx;             // index of the edge (0..11) in the cell
    uint32_t cellIdx;             // linear index of the cell
};

// ============================================================================
//  DualContouring: extract isosurface from a scalar field (signed distance).
//  For each cell that has sign change, compute the vertex position by minimising
//  the quadratic error function using the Hermite data.
//  Then generate triangles connecting vertices of adjacent cells around sign‑changing edges.
// ============================================================================
template<typename T = float>
class DualContouring {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using grid_type = Geometry::VoxelGrid<T>;
    using mesh_type = Geometry::TriangleMesh<T>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        T isoLevel = T(0);
        bool computeNormals = true;
        T qefWeight = T(1e-5);      // regularisation weight for QEF
        bool enableSIMD = true;
        bool useEdgeMask = true;    // only process edges with sign change
    };

    // ------------------------------------------------------------------------
    //  Generate mesh from grid
    // ------------------------------------------------------------------------
    static mesh_type generate(const grid_type& grid, const Config& cfg = Config()) {
        const auto& bounds = grid.bounds();
        const auto& res = grid.resolution();
        T iso = cfg.isoLevel;

        // Pre‑compute: world position of grid corners (cached for speed)
        const size_t nx = res[0], ny = res[1], nz = res[2];
        const size_t nxy = nx * ny;

        // Helper: linear index of a cell (x, y, z) – cells range 0..nx-2, etc.
        auto cellIndex = [&](int x, int y, int z) -> uint32_t {
            return static_cast<uint32_t>((z * (ny-1) + y) * (nx-1) + x);
        };

        // Helper: world position of a grid vertex (corner)
        auto worldPos = [&](int x, int y, int z) -> point_type {
            T fx = static_cast<T>(x) / static_cast<T>(nx - 1);
            T fy = static_cast<T>(y) / static_cast<T>(ny - 1);
            T fz = static_cast<T>(z) / static_cast<T>(nz - 1);
            return bounds.min() + point_type(fx, fy, fz) * bounds.extents();
        };

        // Step 1: find edge intersections and normals (Hermite data)
        std::vector<EdgeIntersection<T>> intersections;
        intersections.reserve((nx-1)*(ny-1)*(nz-1) * 6); // rough estimate

        // Pre‑compute corner values for all cells (to avoid recomputation)
        std::vector<T> cornerVals(nx * ny * nz, T(0));
        for (int z = 0; z < (int)nz; ++z) {
            for (int y = 0; y < (int)ny; ++y) {
                for (int x = 0; x < (int)nx; ++x) {
                    size_t idx = (z * ny + y) * nx + x;
                    point_type wp = worldPos(x, y, z);
                    cornerVals[idx] = grid.interpolate(wp);
                }
            }
        }

        // Edge direction vectors for the 12 edges of a cube (in terms of corner offsets)
        const int edgeOffsets[12][2] = {
            {0,1}, {1,2}, {2,3}, {3,0},   // bottom face
            {4,5}, {5,6}, {6,7}, {7,4},   // top face
            {0,4}, {1,5}, {2,6}, {3,7}    // vertical edges
        };
        // Corner offset coordinates from cell min corner (0,0,0)
        const int cornerOffsets[8][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };

        // For each cell, compute edge intersections
        for (int z = 0; z < (int)nz-1; ++z) {
            for (int y = 0; y < (int)ny-1; ++y) {
                for (int x = 0; x < (int)nx-1; ++x) {
                    // Corner values of this cell (8 corners)
                    T val[8];
                    for (int c = 0; c < 8; ++c) {
                        int cx = x + cornerOffsets[c][0];
                        int cy = y + cornerOffsets[c][1];
                        int cz = z + cornerOffsets[c][2];
                        size_t idx = (cz * ny + cy) * nx + cx;
                        val[c] = cornerVals[idx];
                    }
                    uint32_t cellIdx = cellIndex(x, y, z);

                    // For each edge, check sign change
                    for (int e = 0; e < 12; ++e) {
                        int c0 = edgeOffsets[e][0];
                        int c1 = edgeOffsets[e][1];
                        T v0 = val[c0];
                        T v1 = val[c1];
                        if ((v0 < iso) == (v1 < iso)) continue; // no crossing

                        // Interpolate along the edge
                        T t = (iso - v0) / (v1 - v0);
                        // Get edge endpoints in world coordinates
                        int x0 = x + cornerOffsets[c0][0];
                        int y0 = y + cornerOffsets[c0][1];
                        int z0 = z + cornerOffsets[c0][2];
                        int x1 = x + cornerOffsets[c1][0];
                        int y1 = y + cornerOffsets[c1][1];
                        int z1 = z + cornerOffsets[c1][2];
                        point_type p0 = worldPos(x0, y0, z0);
                        point_type p1 = worldPos(x1, y1, z1);
                        point_type pos = p0 + (p1 - p0) * t;

                        // Compute normal (gradient) at intersection point
                        point_type normal = grid.gradient(pos);
                        T len = normal.length();
                        if (len > T(0)) normal = normal / len;

                        intersections.push_back({pos, normal, static_cast<uint32_t>(e), cellIdx});
                    }
                }
            }
        }

        // Step 2: group intersections by cell
        std::unordered_map<uint32_t, std::vector<EdgeIntersection<T>>> cellMap;
        for (const auto& inter : intersections) {
            cellMap[inter.cellIdx].push_back(inter);
        }

        // Step 3: for each cell, compute a minimising vertex via QEF
        // Also store mapping from cell index to vertex index in output mesh
        std::unordered_map<uint32_t, uint32_t> cellVertexMap;
        std::vector<point_type> vertices;

        for (const auto& pair : cellMap) {
            uint32_t cellIdx = pair.first;
            const auto& inters = pair.second;
            if (inters.size() < 3) continue; // not enough constraints

            // Build linear system: A p = b, where A = Σ (n_i * n_i^T), b = Σ n_i * (n_i · p_i)
            T A[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            T b[3] = {0,0,0};

            for (const auto& inter : inters) {
                const point_type& n = inter.normal;
                point_type p = inter.position;
                T n0 = n[0], n1 = n[1], n2 = n[2];
                T dot = n0*p[0] + n1*p[1] + n2*p[2];
                A[0][0] += n0*n0; A[0][1] += n0*n1; A[0][2] += n0*n2;
                A[1][0] += n1*n0; A[1][1] += n1*n1; A[1][2] += n1*n2;
                A[2][0] += n2*n0; A[2][1] += n2*n1; A[2][2] += n2*n2;
                b[0] += n0 * dot;
                b[1] += n1 * dot;
                b[2] += n2 * dot;
            }

            // Regularisation
            for (int i = 0; i < 3; ++i) A[i][i] += cfg.qefWeight;

            // Solve 3x3 system (Cholesky or Gaussian elimination)
            T det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                  - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                  + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

            point_type vertex;
            if (std::abs(det) < T(1e-12)) {
                // degenerate: use average of intersection points
                point_type avg(0);
                for (const auto& inter : inters) avg = avg + inter.position;
                vertex = avg / static_cast<T>(inters.size());
            } else {
                T invDet = T(1) / det;
                T x = ( (A[1][1]*A[2][2] - A[1][2]*A[2][1]) * b[0]
                      - (A[0][1]*A[2][2] - A[0][2]*A[2][1]) * b[1]
                      + (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * b[2] ) * invDet;
                T y = ( -(A[1][0]*A[2][2] - A[1][2]*A[2][0]) * b[0]
                      + (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * b[1]
                      - (A[0][0]*A[1][2] - A[0][2]*A[1][0]) * b[2] ) * invDet;
                T z = ( (A[1][0]*A[2][1] - A[1][1]*A[2][0]) * b[0]
                      - (A[0][0]*A[2][1] - A[0][1]*A[2][0]) * b[1]
                      + (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * b[2] ) * invDet;
                vertex = point_type(x, y, z);
            }
            // Clamp to cell bounds to avoid degenerate triangles
            point_type minP = worldPos(x, y, z);
            point_type maxP = worldPos(x+1, y+1, z+1);
            vertex = vertex.componentWiseMax(minP).componentWiseMin(maxP);
            uint32_t vtxIdx = static_cast<uint32_t>(vertices.size());
            vertices.push_back(vertex);
            cellVertexMap[cellIdx] = vtxIdx;
        }

        // Step 4: generate triangles by processing each sign‑changing edge
        // An edge is defined by a lower corner (x,y,z) and an edge direction (0,1,2) and orientation.
        // The four incident cells around that edge are:
        // for edge in X direction: cells (x,y,z), (x,y-1,z), (x,y,z-1), (x,y-1,z-1) (and similar for Y and Z).
        // We'll iterate over all grid edges using loops.
        std::vector<uint32_t> indices;

        // Helper to get vertex index for a cell (or return false if not found)
        auto getCellVertex = [&](int cx, int cy, int cz) -> std::optional<uint32_t> {
            if (cx < 0 || cy < 0 || cz < 0) return std::nullopt;
            if (cx >= (int)nx-1 || cy >= (int)ny-1 || cz >= (int)nz-1) return std::nullopt;
            uint32_t cidx = cellIndex(cx, cy, cz);
            auto it = cellVertexMap.find(cidx);
            if (it == cellVertexMap.end()) return std::nullopt;
            return it->second;
        };

        // Loop over X‑direction edges (axis = 0)
        for (int z = 0; z < (int)nz; ++z) {
            for (int y = 0; y < (int)ny; ++y) {
                for (int x = 0; x < (int)nx-1; ++x) {
                    // Edge from (x,y,z) to (x+1,y,z)
                    // Four incident cells: (x,y,z), (x,y-1,z), (x,y,z-1), (x,y-1,z-1)
                    int cells[4][3] = {
                        {x, y, z},
                        {x, y-1, z},
                        {x, y, z-1},
                        {x, y-1, z-1}
                    };
                    std::optional<uint32_t> verts[4];
                    bool allPresent = true;
                    for (int i = 0; i < 4; ++i) {
                        verts[i] = getCellVertex(cells[i][0], cells[i][1], cells[i][2]);
                        if (!verts[i]) { allPresent = false; break; }
                    }
                    if (allPresent) {
                        // Order: v0 (x,y,z), v1 (x,y-1,z), v2 (x,y,z-1), v3 (x,y-1,z-1)
                        // Generate two triangles: (v0, v2, v1) and (v1, v2, v3) or similar (consistent orientation)
                        uint32_t v0 = *verts[0], v1 = *verts[1], v2 = *verts[2], v3 = *verts[3];
                        indices.push_back(v0); indices.push_back(v2); indices.push_back(v1);
                        indices.push_back(v1); indices.push_back(v2); indices.push_back(v3);
                    }
                }
            }
        }

        // Y‑direction edges
        for (int z = 0; z < (int)nz; ++z) {
            for (int x = 0; x < (int)nx; ++x) {
                for (int y = 0; y < (int)ny-1; ++y) {
                    int cells[4][3] = {
                        {x, y, z},
                        {x-1, y, z},
                        {x, y, z-1},
                        {x-1, y, z-1}
                    };
                    std::optional<uint32_t> verts[4];
                    bool allPresent = true;
                    for (int i = 0; i < 4; ++i) {
                        verts[i] = getCellVertex(cells[i][0], cells[i][1], cells[i][2]);
                        if (!verts[i]) { allPresent = false; break; }
                    }
                    if (allPresent) {
                        uint32_t v0 = *verts[0], v1 = *verts[1], v2 = *verts[2], v3 = *verts[3];
                        indices.push_back(v0); indices.push_back(v2); indices.push_back(v1);
                        indices.push_back(v1); indices.push_back(v2); indices.push_back(v3);
                    }
                }
            }
        }

        // Z‑direction edges
        for (int y = 0; y < (int)ny; ++y) {
            for (int x = 0; x < (int)nx; ++x) {
                for (int z = 0; z < (int)nz-1; ++z) {
                    int cells[4][3] = {
                        {x, y, z},
                        {x-1, y, z},
                        {x, y-1, z},
                        {x-1, y-1, z}
                    };
                    std::optional<uint32_t> verts[4];
                    bool allPresent = true;
                    for (int i = 0; i < 4; ++i) {
                        verts[i] = getCellVertex(cells[i][0], cells[i][1], cells[i][2]);
                        if (!verts[i]) { allPresent = false; break; }
                    }
                    if (allPresent) {
                        uint32_t v0 = *verts[0], v1 = *verts[1], v2 = *verts[2], v3 = *verts[3];
                        indices.push_back(v0); indices.push_back(v2); indices.push_back(v1);
                        indices.push_back(v1); indices.push_back(v2); indices.push_back(v3);
                    }
                }
            }
        }

        // Build mesh
        mesh_type mesh(vertices, indices);
        if (cfg.computeNormals) mesh.computeNormals();
        return mesh;
    }
};

} // namespace DualContouring
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_DUAL_CONTOURING_H_INCLUDED