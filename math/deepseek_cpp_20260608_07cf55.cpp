// File 465: modules/integration/unified_point_cloud_generator.h
// Volumetric Poisson‑disk point cloud generator for closed triangle meshes.
// Fills the interior of a mesh with points separated by a minimum distance,
// then builds a TreeNSearch BVH for immediate neighbour queries.  Suitable
// for SPH fluid initialisation, granular materials, and sensor simulation.
// All geometry queries (inside/outside, ray‑cast) are fully implemented.

#ifndef INTEGRATION_UNIFIED_POINT_CLOUD_GENERATOR_H
#define INTEGRATION_UNIFIED_POINT_CLOUD_GENERATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/aabb.h"
#include "core/math/random_number_generator.h"
#include "core/typedefs.h"

#include "../../gaia/src/mesh/tri_mesh.h"             // input surface
#include "../../gaia/src/bvh/bvh.h"                   // Gaia BVH for ray-cast inside test
#include "../../gaia/src/bvh/query.h"                 // ray-triangle intersection
#include "../../treesearch/point_set_search.h"        // output BVH

namespace unified {

class UnifiedPointCloudGenerator : public RefCounted {
    GDCLASS(UnifiedPointCloudGenerator, RefCounted);

public:
    // -------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------
    real_t min_distance = 0.1f;          // minimum separation between points
    int    max_points = 10000;           // maximum number of points to generate
    int    dart_attempts = 30;           // candidates per active point
    int    random_seed = 12345;          // reproducible seed

    // -------------------------------------------------------------------
    // Set the input closed triangle mesh and build the point cloud.
    // The result is stored internally; access via get_points() and
    // get_bvh().
    // -------------------------------------------------------------------
    void generate(const gaia::mesh::TriMesh &p_mesh);

    // -------------------------------------------------------------------
    // Access the generated points.
    // -------------------------------------------------------------------
    const LocalVector<Vector3> &get_points() const { return points; }
    int get_point_count() const { return points.size(); }

    // -------------------------------------------------------------------
    // Access the TreeNSearch BVH built from the points (valid after
    // generate()).  Can be used directly for SPH neighbour queries.
    // -------------------------------------------------------------------
    const treesearch::PointSetSearch &get_bvh() const { return bvh; }

protected:
    static void _bind_methods();

private:
    // Output points (world space positions).
    LocalVector<Vector3> points;

    // TreeNSearch BVH over the generated points.
    treesearch::PointSetSearch bvh;

    // Spatial grid used during Poisson‑disk generation.
    real_t cell_size;                        // cell edge = min_distance / sqrt(3)
    LocalVector<LocalVector<int>> grid;     // list of point indices per cell
    AABB mesh_bounds;                        // bounding box of the input mesh
    Vector3i grid_dim;                       // number of cells in x, y, z

    // Gaia BVH built from the surface triangles for fast inside‑outside tests.
    gaia::bvh::BVH surface_bvh;
    LocalVector<Vector3> surface_vertices;   // copy of mesh vertices
    LocalVector<int>    surface_indices;     // flat triangle indices (3 per tri)

    // --- internal helpers ---

    // Inside‑outside test using ray‑casting from the point in +X direction.
    bool is_inside_mesh(const Vector3 &p_world) const;

    // Convert world position to grid cell index.
    Vector3i world_to_cell(const Vector3 &p) const;

    // Grid cell flat index.
    int cell_index(const Vector3i &p_cell) const;

    // Add a point to the grid.
    void add_point_to_grid(int p_point_idx);

    // Check if a point violates the minimum distance with existing points
    // in neighbouring cells.
    bool is_too_close(const Vector3 &p_point) const;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedPointCloudGenerator::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_min_distance", "dist"), &UnifiedPointCloudGenerator::set_min_distance);
    ClassDB::bind_method(D_METHOD("get_min_distance"), &UnifiedPointCloudGenerator::get_min_distance);
    ClassDB::bind_method(D_METHOD("set_max_points", "max"), &UnifiedPointCloudGenerator::set_max_points);
    ClassDB::bind_method(D_METHOD("get_max_points"), &UnifiedPointCloudGenerator::get_max_points);
    ClassDB::bind_method(D_METHOD("set_dart_attempts", "attempts"), &UnifiedPointCloudGenerator::set_dart_attempts);
    ClassDB::bind_method(D_METHOD("get_dart_attempts"), &UnifiedPointCloudGenerator::get_dart_attempts);
    ClassDB::bind_method(D_METHOD("set_random_seed", "seed"), &UnifiedPointCloudGenerator::set_random_seed);
    ClassDB::bind_method(D_METHOD("get_random_seed"), &UnifiedPointCloudGenerator::get_random_seed);
    ClassDB::bind_method(D_METHOD("generate", "mesh"), &UnifiedPointCloudGenerator::generate);
    ClassDB::bind_method(D_METHOD("get_points"), &UnifiedPointCloudGenerator::get_points);
    ClassDB::bind_method(D_METHOD("get_bvh"), &UnifiedPointCloudGenerator::get_bvh);
}

// Property setters/getters.
void UnifiedPointCloudGenerator::set_min_distance(real_t v) { min_distance = MAX(v, 0.001f); }
real_t UnifiedPointCloudGenerator::get_min_distance() const { return min_distance; }
void UnifiedPointCloudGenerator::set_max_points(int v) { max_points = MAX(v, 1); }
int UnifiedPointCloudGenerator::get_max_points() const { return max_points; }
void UnifiedPointCloudGenerator::set_dart_attempts(int v) { dart_attempts = MAX(v, 1); }
int UnifiedPointCloudGenerator::get_dart_attempts() const { return dart_attempts; }
void UnifiedPointCloudGenerator::set_random_seed(int v) { random_seed = v; }
int UnifiedPointCloudGenerator::get_random_seed() const { return random_seed; }

// ---------------------------------------------------------------------------
// Main generation entry point.
// ---------------------------------------------------------------------------
void UnifiedPointCloudGenerator::generate(const gaia::mesh::TriMesh &p_mesh) {
    points.clear();
    grid.clear();

    int tri_count = p_mesh.triangle_count();
    if (tri_count < 4) return; // need closed mesh with at least 4 tri

    // Copy surface data for efficient access.
    surface_vertices.resize(p_mesh.vertex_count());
    for (int i = 0; i < surface_vertices.size(); ++i) {
        surface_vertices[i] = p_mesh.get_vertex(i);
    }
    surface_indices.resize(tri_count * 3);
    for (int t = 0; t < tri_count; ++t) {
        auto tri = p_mesh.get_triangle(t);
        surface_indices[t*3]     = tri.v0;
        surface_indices[t*3 + 1] = tri.v1;
        surface_indices[t*3 + 2] = tri.v2;
    }

    // Build AABB of the mesh for bounding.
    mesh_bounds = AABB(surface_vertices[0], Vector3());
    for (int i = 1; i < surface_vertices.size(); ++i) {
        mesh_bounds.expand_to(surface_vertices[i]);
    }
    // Expand slightly to allow points near the boundary.
    mesh_bounds.grow_by(min_distance * 0.5f);

    // Build Gaia BVH over surface triangles for inside tests.
    LocalVector<AABB> tri_aabbs(tri_count);
    for (int t = 0; t < tri_count; ++t) {
        Vector3 v0 = surface_vertices[surface_indices[t*3]];
        Vector3 v1 = surface_vertices[surface_indices[t*3+1]];
        Vector3 v2 = surface_vertices[surface_indices[t*3+2]];
        AABB box(v0, Vector3());
        box.expand_to(v1);
        box.expand_to(v2);
        tri_aabbs[t] = box;
    }
    surface_bvh.build_final(tri_aabbs);

    // Set up spatial grid for Poisson‑disk.
    cell_size = min_distance / Math::sqrt(3.0f); // ensures at most one point per cell
    if (cell_size < CMP_EPSILON) cell_size = CMP_EPSILON;
    grid_dim.x = (int)Math::ceil(mesh_bounds.size.x / cell_size) + 1;
    grid_dim.y = (int)Math::ceil(mesh_bounds.size.y / cell_size) + 1;
    grid_dim.z = (int)Math::ceil(mesh_bounds.size.z / cell_size) + 1;
    int total_cells = grid_dim.x * grid_dim.y * grid_dim.z;
    grid.resize(total_cells);
    for (int i = 0; i < total_cells; ++i) grid[i].clear();

    // Random number generator.
    RandomNumberGenerator rng;
    rng.set_seed(random_seed);

    // --- Poisson‑disk algorithm ---

    // 1. Generate the first point randomly inside the mesh.
    Vector3 first_point;
    bool found = false;
    for (int attempt = 0; attempt < 500; ++attempt) {
        Vector3 p(
            rng.randf_range(mesh_bounds.position.x, mesh_bounds.position.x + mesh_bounds.size.x),
            rng.randf_range(mesh_bounds.position.y, mesh_bounds.position.y + mesh_bounds.size.y),
            rng.randf_range(mesh_bounds.position.z, mesh_bounds.position.z + mesh_bounds.size.z)
        );
        if (is_inside_mesh(p)) {
            first_point = p;
            found = true;
            break;
        }
    }
    if (!found) return; // no interior point found (mesh may be open or too thin)

    points.push_back(first_point);
    add_point_to_grid(0);

    // Active list contains indices of points that are candidates for generating neighbours.
    LocalVector<int> active;
    active.push_back(0);

    while (!active.is_empty() && points.size() < max_points) {
        // Pick a random active point.
        int active_idx = rng.randi() % active.size();
        int pt_idx = active[active_idx];
        Vector3 center = points[pt_idx];
        bool accepted = false;

        // Generate candidates in a shell between r and 2r around the center.
        for (int k = 0; k < dart_attempts; ++k) {
            // Uniform random direction.
            real_t theta = rng.randf_range(0.0f, Math_TAU);
            real_t phi   = Math::acos(rng.randf_range(-1.0f, 1.0f));
            real_t r     = rng.randf_range(min_distance, 2.0f * min_distance);
            Vector3 offset(
                r * Math::sin(phi) * Math::cos(theta),
                r * Math::sin(phi) * Math::sin(theta),
                r * Math::cos(phi)
            );
            Vector3 candidate = center + offset;

            // Check inside mesh and minimum distance.
            if (!is_inside_mesh(candidate)) continue;
            if (is_too_close(candidate)) continue;

            // Accept point.
            int new_idx = points.size();
            points.push_back(candidate);
            add_point_to_grid(new_idx);
            active.push_back(new_idx);
            accepted = true;
            if (points.size() >= max_points) break;
        }

        // If no candidate accepted, remove the point from active list.
        if (!accepted) {
            active.remove_at(active_idx);
        }
    }

    // Build the TreeNSearch BVH from the generated points.
    bvh.build(points);
}

// ---------------------------------------------------------------------------
// Inside test: shoot a ray in +X and count intersections with surface.
// ---------------------------------------------------------------------------
bool UnifiedPointCloudGenerator::is_inside_mesh(const Vector3 &p_world) const {
    int hits = 0;
    Vector3 ray_origin = p_world;
    Vector3 ray_dir(1.0f, 0.0f, 0.0f);
    real_t far = mesh_bounds.size.x * 2.0f;
    // Use Gaia BVH to query triangles intersecting the ray.
    AABB ray_aabb(ray_origin, Vector3());
    ray_aabb.expand_to(ray_origin + ray_dir * far);
    surface_bvh.query_intersect(ray_aabb, [&](int prim) {
        if (prim < 0 || prim >= surface_indices.size() / 3) return;
        int idx = prim * 3;
        Vector3 v0 = surface_vertices[surface_indices[idx]];
        Vector3 v1 = surface_vertices[surface_indices[idx+1]];
        Vector3 v2 = surface_vertices[surface_indices[idx+2]];
        real_t t, u, v;
        if (gaia::bvh::intersect_ray_triangle(ray_origin, ray_dir, v0, v1, v2, t, u, v)) {
            if (t > 0.0f && t < far) hits++;
        }
    });
    // Odd number of hits = inside (for closed manifold).
    return (hits & 1) != 0;
}

// ---------------------------------------------------------------------------
// Spatial grid helpers.
// ---------------------------------------------------------------------------
Vector3i UnifiedPointCloudGenerator::world_to_cell(const Vector3 &p) const {
    Vector3 rel = p - mesh_bounds.position;
    return Vector3i(
        (int)(rel.x / cell_size),
        (int)(rel.y / cell_size),
        (int)(rel.z / cell_size)
    );
}

int UnifiedPointCloudGenerator::cell_index(const Vector3i &p_cell) const {
    if (p_cell.x < 0 || p_cell.x >= grid_dim.x ||
        p_cell.y < 0 || p_cell.y >= grid_dim.y ||
        p_cell.z < 0 || p_cell.z >= grid_dim.z)
        return -1;
    return p_cell.x + grid_dim.x * (p_cell.y + grid_dim.y * p_cell.z);
}

void UnifiedPointCloudGenerator::add_point_to_grid(int p_point_idx) {
    Vector3i cell = world_to_cell(points[p_point_idx]);
    int cidx = cell_index(cell);
    if (cidx >= 0 && cidx < grid.size()) {
        grid[cidx].push_back(p_point_idx);
    }
}

// ---------------------------------------------------------------------------
// Minimum distance check: look at 3x3x3 neighbour cells.
// ---------------------------------------------------------------------------
bool UnifiedPointCloudGenerator::is_too_close(const Vector3 &p_point) const {
    Vector3i center_cell = world_to_cell(p_point);
    real_t min_dist_sq = min_distance * min_distance;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                Vector3i cell = center_cell + Vector3i(dx, dy, dz);
                int cidx = cell_index(cell);
                if (cidx < 0) continue;
                for (int idx : grid[cidx]) {
                    if (points[idx].distance_squared_to(p_point) < min_dist_sq) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_POINT_CLOUD_GENERATOR_H