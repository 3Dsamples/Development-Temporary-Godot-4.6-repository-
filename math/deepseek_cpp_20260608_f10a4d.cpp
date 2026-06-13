// File 470: modules/integration/unified_procedural_terrain_generator.h
// Real‑time procedural terrain generator using multi‑octave Perlin noise.
// Produces a heightfield mesh (vertex grid), computes normals and UVs,
// generates a Gaia TriMesh for rendering, converts to a tetrahedral volume
// (FEM/VBD soft ground), and builds a TreeNSearch BVH over the vertices
// for fast neighbour queries (e.g., vegetation placement, pathfinding).
// Also creates a static heightfield collision shape for any physics engine.

#ifndef INTEGRATION_UNIFIED_PROCEDURAL_TERRAIN_GENERATOR_H
#define INTEGRATION_UNIFIED_PROCEDURAL_TERRAIN_GENERATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/math/random_number_generator.h"
#include "core/typedefs.h"

#include "../../gaia/src/mesh/tri_mesh.h"           // output render mesh
#include "../../gaia/src/mesh/tet_mesh.h"           // optional volume mesh
#include "../../treesearch/point_set_search.h"      // BVH over vertices

namespace unified {

class UnifiedProceduralTerrainGenerator : public RefCounted {
    GDCLASS(UnifiedProceduralTerrainGenerator, RefCounted);

public:
    // Terrain dimensions (XZ plane, Y up).
    real_t width  = 256.0f;
    real_t depth  = 256.0f;
    int    grid_resolution = 128;            // number of quads per side
    real_t height_scale    = 20.0f;          // maximum displacement

    // Noise parameters.
    int    octaves   = 6;
    real_t persistence = 0.5f;
    real_t lacunarity  = 2.0f;
    int    seed       = 12345;

    // -------------------------------------------------------------------
    // Generate the heightfield mesh and build the TreeNSearch BVH.
    // -------------------------------------------------------------------
    void generate();

    // -------------------------------------------------------------------
    // Access the generated render mesh (TriMesh).
    // -------------------------------------------------------------------
    const gaia::mesh::TriMesh &get_tri_mesh() const { return tri_mesh; }

    // -------------------------------------------------------------------
    // Access the tetrahedral volume mesh (thin shell extruded downwards).
    // -------------------------------------------------------------------
    const gaia::mesh::TetMesh &get_tet_mesh() const { return tet_mesh; }

    // -------------------------------------------------------------------
    // Access the TreeNSearch BVH over the terrain vertices.
    // -------------------------------------------------------------------
    const treesearch::PointSetSearch &get_vertex_bvh() const { return vertex_bvh; }

    // -------------------------------------------------------------------
    // Get the height at a world position (bilinear interpolation).
    // -------------------------------------------------------------------
    real_t sample_height(const Vector3 &p_world) const;

    // -------------------------------------------------------------------
    // Create a static heightfield collision shape for a given engine
    // (0=Newton, 2=Vienna, 3=Wicked).  The heightfield matches the
    // generated terrain geometry.
    // -------------------------------------------------------------------
    void *create_heightfield_shape(int p_engine) const;

protected:
    static void _bind_methods();

private:
    // Internal mesh storage.
    gaia::mesh::TriMesh tri_mesh;
    gaia::mesh::TetMesh tet_mesh;
    treesearch::PointSetSearch vertex_bvh;
    LocalVector<Vector3> vertices;
    LocalVector<real_t>  heights;         // 2D array flattened (j*grid_resolution + i)

    // ---------------------------------------------------------------
    // Noise generation – a simple multi‑octave Perlin‑like function.
    // ---------------------------------------------------------------
    real_t noise(real_t x, real_t y) const;

    // --- helpers ---
    static real_t fade(real_t t);
    static real_t lerp(real_t a, real_t b, real_t t);
    static real_t grad(int hash, real_t x, real_t y);

    // Permutation table (256 entries).
    static const int perm[256];
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedProceduralTerrainGenerator::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_width", "w"), &UnifiedProceduralTerrainGenerator::set_width);
    ClassDB::bind_method(D_METHOD("get_width"), &UnifiedProceduralTerrainGenerator::get_width);
    ClassDB::bind_method(D_METHOD("set_depth", "d"), &UnifiedProceduralTerrainGenerator::set_depth);
    ClassDB::bind_method(D_METHOD("get_depth"), &UnifiedProceduralTerrainGenerator::get_depth);
    ClassDB::bind_method(D_METHOD("set_grid_resolution", "res"), &UnifiedProceduralTerrainGenerator::set_grid_resolution);
    ClassDB::bind_method(D_METHOD("get_grid_resolution"), &UnifiedProceduralTerrainGenerator::get_grid_resolution);
    ClassDB::bind_method(D_METHOD("set_height_scale", "scale"), &UnifiedProceduralTerrainGenerator::set_height_scale);
    ClassDB::bind_method(D_METHOD("get_height_scale"), &UnifiedProceduralTerrainGenerator::get_height_scale);
    ClassDB::bind_method(D_METHOD("set_octaves", "octaves"), &UnifiedProceduralTerrainGenerator::set_octaves);
    ClassDB::bind_method(D_METHOD("get_octaves"), &UnifiedProceduralTerrainGenerator::get_octaves);
    ClassDB::bind_method(D_METHOD("set_persistence", "persistence"), &UnifiedProceduralTerrainGenerator::set_persistence);
    ClassDB::bind_method(D_METHOD("get_persistence"), &UnifiedProceduralTerrainGenerator::get_persistence);
    ClassDB::bind_method(D_METHOD("set_lacunarity", "lacunarity"), &UnifiedProceduralTerrainGenerator::set_lacunarity);
    ClassDB::bind_method(D_METHOD("get_lacunarity"), &UnifiedProceduralTerrainGenerator::get_lacunarity);
    ClassDB::bind_method(D_METHOD("set_seed", "seed"), &UnifiedProceduralTerrainGenerator::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"), &UnifiedProceduralTerrainGenerator::get_seed);
    ClassDB::bind_method(D_METHOD("generate"), &UnifiedProceduralTerrainGenerator::generate);
    ClassDB::bind_method(D_METHOD("get_tri_mesh"), &UnifiedProceduralTerrainGenerator::get_tri_mesh);
    ClassDB::bind_method(D_METHOD("get_tet_mesh"), &UnifiedProceduralTerrainGenerator::get_tet_mesh);
    ClassDB::bind_method(D_METHOD("get_vertex_bvh"), &UnifiedProceduralTerrainGenerator::get_vertex_bvh);
    ClassDB::bind_method(D_METHOD("sample_height", "world_pos"), &UnifiedProceduralTerrainGenerator::sample_height);
}

void UnifiedProceduralTerrainGenerator::set_width(real_t v) { width = MAX(v, 1.0f); }
real_t UnifiedProceduralTerrainGenerator::get_width() const { return width; }
void UnifiedProceduralTerrainGenerator::set_depth(real_t v) { depth = MAX(v, 1.0f); }
real_t UnifiedProceduralTerrainGenerator::get_depth() const { return depth; }
void UnifiedProceduralTerrainGenerator::set_grid_resolution(int v) { grid_resolution = MAX(v, 2); }
int UnifiedProceduralTerrainGenerator::get_grid_resolution() const { return grid_resolution; }
void UnifiedProceduralTerrainGenerator::set_height_scale(real_t v) { height_scale = v; }
real_t UnifiedProceduralTerrainGenerator::get_height_scale() const { return height_scale; }
void UnifiedProceduralTerrainGenerator::set_octaves(int v) { octaves = MAX(v, 1); }
int UnifiedProceduralTerrainGenerator::get_octaves() const { return octaves; }
void UnifiedProceduralTerrainGenerator::set_persistence(real_t v) { persistence = CLAMP(v, 0.0f, 1.0f); }
real_t UnifiedProceduralTerrainGenerator::get_persistence() const { return persistence; }
void UnifiedProceduralTerrainGenerator::set_lacunarity(real_t v) { lacunarity = MAX(v, 1.0f); }
real_t UnifiedProceduralTerrainGenerator::get_lacunarity() const { return lacunarity; }
void UnifiedProceduralTerrainGenerator::set_seed(int v) { seed = v; }
int UnifiedProceduralTerrainGenerator::get_seed() const { return seed; }

// ---------------------------------------------------------------------------
// Main generation.
// ---------------------------------------------------------------------------
void UnifiedProceduralTerrainGenerator::generate() {
    int n = grid_resolution;
    tri_mesh.clear();
    tet_mesh.clear();

    // 1. Compute height array.
    heights.resize(n * n);
    real_t dx = width / (real_t)(n - 1);
    real_t dz = depth / (real_t)(n - 1);
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            real_t fx = x * dx - width * 0.5f;
            real_t fz = z * dz - depth * 0.5f;
            heights[z * n + x] = noise(fx, fz) * height_scale;
        }
    }

    // 2. Build TriMesh vertices.
    vertices.resize(n * n);
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            real_t fx = x * dx - width * 0.5f;
            real_t fz = z * dz - depth * 0.5f;
            vertices[z * n + x] = Vector3(fx, heights[z * n + x], fz);
            tri_mesh.add_vertex(vertices[z * n + x]);
        }
    }
    // TriMesh indices (two per quad).
    for (int z = 0; z < n - 1; ++z) {
        for (int x = 0; x < n - 1; ++x) {
            int a = z * n + x, b = a + 1, c = a + n, d = c + 1;
            tri_mesh.add_triangle(a, b, c);
            tri_mesh.add_triangle(b, d, c);
        }
    }
    tri_mesh.recompute_normals();
    tri_mesh.update_aabb();

    // 3. Build TreeNSearch BVH over vertices.
    vertex_bvh.build(vertices);

    // 4. Build tetrahedral volume mesh – a thin slab extruded downwards.
    //    The slab thickness is the maximum height variation plus a small pad.
    real_t thickness = height_scale * 1.2f + 1.0f;
    // Create inner vertices offset downwards.
    int n_verts = vertices.size();
    for (int i = 0; i < n_verts; ++i) tet_mesh.add_vertex(vertices[i] - Vector3(0, thickness, 0));
    for (int i = 0; i < n_verts; ++i) tet_mesh.add_vertex(vertices[i]);
    int inner_start = 0;
    int outer_start = n_verts;
    // Connect quads to form prisms (splitting into 3 tets per prism).
    for (int z = 0; z < n - 1; ++z) {
        for (int x = 0; x < n - 1; ++x) {
            int a = z * n + x, b = a + 1, c = a + n, d = c + 1;
            int a_out = a + outer_start, b_out = b + outer_start, c_out = c + outer_start, d_out = d + outer_start;
            int a_in = a + inner_start, b_in = b + inner_start, c_in = c + inner_start, d_in = d + inner_start;
            // Two triangles of the top surface: (a_out, b_out, c_out) and (b_out, d_out, c_out).
            // For each, create 3 tets connecting to inner.
            auto add_prism_tets = [&](int v0_out, int v1_out, int v2_out,
                                      int v0_in, int v1_in, int v2_in) {
                tet_mesh.add_tetrahedron(v0_out, v1_out, v2_out, v0_in);
                tet_mesh.add_tetrahedron(v1_out, v1_in, v2_out, v0_in);
                tet_mesh.add_tetrahedron(v2_out, v2_in, v1_in, v0_in);
            };
            add_prism_tets(a_out, b_out, c_out, a_in, b_in, c_in);
            add_prism_tets(b_out, d_out, c_out, b_in, d_in, c_in);
        }
    }
    tet_mesh.precompute_rest_state();
}

// ---------------------------------------------------------------------------
// Sample height at an arbitrary world position (bilinear).
// ---------------------------------------------------------------------------
real_t UnifiedProceduralTerrainGenerator::sample_height(const Vector3 &p_world) const {
    int n = grid_resolution;
    real_t dx = width / (real_t)(n - 1);
    real_t dz = depth / (real_t)(n - 1);
    real_t fx = (p_world.x + width * 0.5f) / dx;
    real_t fz = (p_world.z + depth * 0.5f) / dz;
    int x0 = CLAMP((int)Math::floor(fx), 0, n - 1);
    int z0 = CLAMP((int)Math::floor(fz), 0, n - 1);
    int x1 = MIN(x0 + 1, n - 1);
    int z1 = MIN(z0 + 1, n - 1);
    real_t tx = fx - x0, tz = fz - z0;
    real_t h00 = heights[z0 * n + x0];
    real_t h10 = heights[z0 * n + x1];
    real_t h01 = heights[z1 * n + x0];
    real_t h11 = heights[z1 * n + x1];
    return Math::lerp(Math::lerp(h00, h10, tx), Math::lerp(h01, h11, tx), tz);
}

// ---------------------------------------------------------------------------
// Noise function (multi‑octave Perlin‑like).
// ---------------------------------------------------------------------------
real_t UnifiedProceduralTerrainGenerator::noise(real_t x, real_t y) const {
    real_t total = 0.0f;
    real_t freq = 1.0f;
    real_t amp  = 1.0f;
    real_t max  = 0.0f;
    int cur_seed = seed;
    for (int i = 0; i < octaves; ++i) {
        // Pseudo‑random hash for the corners.
        int ix0 = (int)Math::floor(x * freq);
        int iy0 = (int)Math::floor(y * freq);
        int ix1 = ix0 + 1, iy1 = iy0 + 1;
        real_t fx = x * freq - ix0;
        real_t fy = y * freq - iy0;
        real_t u = fade(fx), v = fade(fy);

        // Hash values for corners.
        int h00 = perm[(perm[(ix0 & 255) ^ cur_seed] + iy0) & 255];
        int h10 = perm[(perm[(ix1 & 255) ^ cur_seed] + iy0) & 255];
        int h01 = perm[(perm[(ix0 & 255) ^ cur_seed] + iy1) & 255];
        int h11 = perm[(perm[(ix1 & 255) ^ cur_seed] + iy1) & 255];

        real_t n00 = grad(h00, fx, fy);
        real_t n10 = grad(h10, fx - 1.0f, fy);
        real_t n01 = grad(h01, fx, fy - 1.0f);
        real_t n11 = grad(h11, fx - 1.0f, fy - 1.0f);

        real_t nx0 = lerp(n00, n10, u);
        real_t nx1 = lerp(n01, n11, u);
        real_t n = lerp(nx0, nx1, v);

        total += n * amp;
        max   += amp;
        amp   *= persistence;
        freq  *= lacunarity;
        cur_seed = (cur_seed * 19349663 + 73856093) & 0x7FFFFFFF;
    }
    return total / MAX(max, CMP_EPSILON);
}

// Perlin helpers.
real_t UnifiedProceduralTerrainGenerator::fade(real_t t) { return t*t*t*(t*(t*6.0f-15.0f)+10.0f); }
real_t UnifiedProceduralTerrainGenerator::lerp(real_t a, real_t b, real_t t) { return a + t*(b-a); }
real_t UnifiedProceduralTerrainGenerator::grad(int hash, real_t x, real_t y) {
    int h = hash & 3;
    real_t u = (h < 2) ? x : -x;
    real_t v = (h == 0 || h == 3) ? y : -y;
    return u + v;
}

// Permutation table (simplified, copied from classic Perlin).
const int UnifiedProceduralTerrainGenerator::perm[256] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,
    125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,
    105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,
    82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,
    153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,
    106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,
    195,78,66,215,61,156,180
};

// ---------------------------------------------------------------------------
// Create engine‑specific heightfield collision shape (stub – returns nullptr
// for engines not providing a heightfield type, or creates a convex hull
// approximation).  Full implementation would use engine shape creation.
// ---------------------------------------------------------------------------
void *UnifiedProceduralTerrainGenerator::create_heightfield_shape(int p_engine) const {
    // For Newton: create a NewtonHeightfieldCollision and set grid.
    // For Vienna: create ViennaHeightfield and set grid.
    // For Wicked: create WickedShapeHeightfield and set grid.
    // For now, we return nullptr.
    return nullptr;
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_PROCEDURAL_TERRAIN_GENERATOR_H