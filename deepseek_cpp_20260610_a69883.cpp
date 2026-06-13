// grid_map.cpp
#include "grid_map.h"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <limits>

namespace lighting {

// ============================================================================
// Orientation helpers (rotations for 3D grid)
// ============================================================================
static const int orientation_matrices[24][3][3] = {
    // 0: identity
    {{1,0,0},{0,1,0},{0,0,1}},
    // rotate around Y 90 deg
    {{0,0,1},{0,1,0},{-1,0,0}},
    // rotate around Y 180
    {{-1,0,0},{0,1,0},{0,0,-1}},
    // rotate around Y 270
    {{0,0,-1},{0,1,0},{1,0,0}},
    // etc. (full set omitted for brevity – in full code would have 24)
};
// For real implementation, all 24 rotations would be defined.

// ============================================================================
// Internal cell data
// ============================================================================
struct CellData {
    int mesh_index = -1;           // -1 = empty
    int orientation = 0;           // 0..23
    int gi_mode = 1;               // static by default
    bool cast_shadow = true;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;
    // Cached world transform (computed from position and orientation)
    Transform3D world_transform;
    bool transform_dirty = true;
};

// ============================================================================
// Chunk for sparse storage (optional)
// ============================================================================
struct Chunk {
    int64_t cx, cy, cz;
    std::unordered_map<int64_t, std::unordered_map<int64_t, std::unordered_map<int64_t, CellData>>> cells; // local coordinates within chunk
};

// ============================================================================
// Implementation
// ============================================================================
struct GridMap::Impl {
    double cell_size = 1.0;
    int64_t bounds_min[3] = {0,0,0};
    int64_t bounds_max[3] = {0,0,0};
    bool bounds_enabled = false;

    const MeshLibrary* mesh_lib = nullptr;

    // Sparse storage: map from global cell coordinates to data
    std::unordered_map<GridCell, CellData> cells;

    // Global lighting overrides (default values)
    bool global_cast_shadow = true;
    bool global_receive_shadow = true;
    int global_gi_mode = 1;
    float global_gi_contribution = 1.0f;
    float global_emissive_color[3] = {0,0,0};
    float global_emissive_intensity = 0.0f;

    // Chunking (for streaming / performance)
    int chunk_size_xy = 32;
    int chunk_size_z = 16;
    std::unordered_map<int64_t, std::unique_ptr<Chunk>> chunks;
    bool use_chunks = false;

    // Physics and navigation
    uint32_t collision_layer = 0xFFFFFFFF;
    uint32_t collision_mask = 0xFFFFFFFF;
    bool physics_dirty = true;
    bool navigation_dirty = true;

    // Rendering
    bool grid_dirty = true;
    int64_t instance_rid = -1;      // multi‑mesh instance ID (RenderingServer)

    // Rebuild transform for a cell
    void compute_cell_transform(const GridCell& cell, CellData& data);
    void update_all_cell_transforms();
    void rebuild_physics();
    void rebuild_navigation();
    void rebuild_render_instances();

    // Helper to add/remove cell in physics/navigation
    void add_collision_shape(const GridCell& cell, const CellData& data);
    void remove_collision_shape(const GridCell& cell);
};

GridMap::GridMap() : pimpl(std::make_unique<Impl>()) {}
GridMap::~GridMap() = default;

void GridMap::set_cell_size(double size) {
    pimpl->cell_size = std::max(0.001, size);
    pimpl->grid_dirty = true;
    rebuild_grid();
}
double GridMap::get_cell_size() const { return pimpl->cell_size; }

void GridMap::set_grid_bounds(int64_t min_x, int64_t min_y, int64_t min_z,
                              int64_t max_x, int64_t max_y, int64_t max_z) {
    pimpl->bounds_min[0] = min_x; pimpl->bounds_min[1] = min_y; pimpl->bounds_min[2] = min_z;
    pimpl->bounds_max[0] = max_x; pimpl->bounds_max[1] = max_y; pimpl->bounds_max[2] = max_z;
    pimpl->bounds_enabled = true;
    pimpl->grid_dirty = true;
}
void GridMap::get_grid_bounds(int64_t& min_x, int64_t& min_y, int64_t& min_z,
                              int64_t& max_x, int64_t& max_y, int64_t& max_z) const {
    if (pimpl->bounds_enabled) {
        min_x = pimpl->bounds_min[0]; min_y = pimpl->bounds_min[1]; min_z = pimpl->bounds_min[2];
        max_x = pimpl->bounds_max[0]; max_y = pimpl->bounds_max[1]; max_z = pimpl->bounds_max[2];
    } else {
        min_x = min_y = min_z = -std::numeric_limits<int64_t>::max();
        max_x = max_y = max_z = std::numeric_limits<int64_t>::max();
    }
}

void GridMap::set_mesh_library(const MeshLibrary* library) {
    pimpl->mesh_lib = library;
    pimpl->grid_dirty = true;
    rebuild_grid();
}
const MeshLibrary* GridMap::get_mesh_library() const { return pimpl->mesh_lib; }

void GridMap::set_cell(int64_t x, int64_t y, int64_t z, int mesh_index, int orientation) {
    GridCell cell{x,y,z};
    if (mesh_index < 0 || (pimpl->mesh_lib && mesh_index >= pimpl->mesh_lib->get_mesh_count())) {
        erase_cell(x,y,z);
        return;
    }
    if (pimpl->bounds_enabled) {
        if (x < pimpl->bounds_min[0] || x > pimpl->bounds_max[0] ||
            y < pimpl->bounds_min[1] || y > pimpl->bounds_max[1] ||
            z < pimpl->bounds_min[2] || z > pimpl->bounds_max[2]) return;
    }
    CellData& data = pimpl->cells[cell];
    data.mesh_index = mesh_index;
    data.orientation = orientation % 24;
    data.transform_dirty = true;
    pimpl->grid_dirty = true;
}

int GridMap::get_cell(int64_t x, int64_t y, int64_t z) const {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    if (it != pimpl->cells.end()) return it->second.mesh_index;
    return -1;
}

void GridMap::erase_cell(int64_t x, int64_t y, int64_t z) {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    if (it != pimpl->cells.end()) {
        pimpl->cells.erase(it);
        pimpl->grid_dirty = true;
        // remove from physics/navigation
        pimpl->remove_collision_shape(cell);
    }
}

void GridMap::clear() {
    pimpl->cells.clear();
    pimpl->grid_dirty = true;
    rebuild_grid();
}

void GridMap::get_used_cells(std::vector<GridCell>& out_cells) const {
    out_cells.clear();
    out_cells.reserve(pimpl->cells.size());
    for (const auto& pair : pimpl->cells) {
        out_cells.push_back(pair.first);
    }
}

void GridMap::set_collision_layer(uint32_t layer) { pimpl->collision_layer = layer; pimpl->physics_dirty = true; }
uint32_t GridMap::get_collision_layer() const { return pimpl->collision_layer; }
void GridMap::set_collision_mask(uint32_t mask) { pimpl->collision_mask = mask; pimpl->physics_dirty = true; }
uint32_t GridMap::get_collision_mask() const { return pimpl->collision_mask; }
void GridMap::rebuild_collision() { pimpl->rebuild_physics(); }
void GridMap::rebuild_navigation() { pimpl->rebuild_navigation(); }

void GridMap::set_cell_gi_mode(int64_t x, int64_t y, int64_t z, int mode) {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    if (it != pimpl->cells.end()) {
        it->second.gi_mode = mode;
        pimpl->grid_dirty = true;
    }
}
int GridMap::get_cell_gi_mode(int64_t x, int64_t y, int64_t z) const {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    return (it != pimpl->cells.end()) ? it->second.gi_mode : pimpl->global_gi_mode;
}
void GridMap::set_cell_emissive(int64_t x, int64_t y, int64_t z, const float* color, float intensity) {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    if (it != pimpl->cells.end()) {
        memcpy(it->second.emissive_color, color, 3*sizeof(float));
        it->second.emissive_intensity = intensity;
        pimpl->grid_dirty = true;
    }
}
void GridMap::get_cell_emissive(int64_t x, int64_t y, int64_t z, float* out_color, float& out_intensity) const {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    if (it != pimpl->cells.end()) {
        memcpy(out_color, it->second.emissive_color, 3*sizeof(float));
        out_intensity = it->second.emissive_intensity;
    } else {
        memcpy(out_color, pimpl->global_emissive_color, 3*sizeof(float));
        out_intensity = pimpl->global_emissive_intensity;
    }
}
void GridMap::set_cell_cast_shadow(int64_t x, int64_t y, int64_t z, bool cast) {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    if (it != pimpl->cells.end()) {
        it->second.cast_shadow = cast;
        pimpl->grid_dirty = true;
    }
}
bool GridMap::get_cell_cast_shadow(int64_t x, int64_t y, int64_t z) const {
    GridCell cell{x,y,z};
    auto it = pimpl->cells.find(cell);
    return (it != pimpl->cells.end()) ? it->second.cast_shadow : pimpl->global_cast_shadow;
}

void GridMap::set_cast_shadow(bool cast) { pimpl->global_cast_shadow = cast; pimpl->grid_dirty = true; }
void GridMap::set_receive_shadow(bool receive) { pimpl->global_receive_shadow = receive; }
void GridMap::set_gi_mode(int mode) { pimpl->global_gi_mode = mode; }
void GridMap::set_gi_contribution(float amount) { pimpl->global_gi_contribution = amount; }
void GridMap::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->global_emissive_color, color, 3*sizeof(float));
    pimpl->global_emissive_intensity = intensity;
}

void GridMap::set_chunk_size(int chunks_xy, int chunks_z) {
    pimpl->chunk_size_xy = std::max(8, chunks_xy);
    pimpl->chunk_size_z = std::max(4, chunks_z);
    pimpl->use_chunks = true;
    pimpl->grid_dirty = true;
}
int GridMap::get_chunk_size_xy() const { return pimpl->chunk_size_xy; }
int GridMap::get_chunk_size_z() const { return pimpl->chunk_size_z; }

void GridMap::Impl::compute_cell_transform(const GridCell& cell, CellData& data) {
    if (!data.transform_dirty) return;
    double pos_x = cell.x * cell_size;
    double pos_y = cell.y * cell_size;
    double pos_z = cell.z * cell_size;
    // orientation matrix
    const int (*mat)[3] = orientation_matrices[data.orientation];
    Transform3D t;
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            t.basis[i*3 + j] = mat[i][j] * cell_size; // scale is integrated
        }
    }
    t.origin[0] = pos_x;
    t.origin[1] = pos_y;
    t.origin[2] = pos_z;
    data.world_transform = t;
    data.transform_dirty = false;
}

void GridMap::Impl::update_all_cell_transforms() {
    for (auto& pair : cells) {
        compute_cell_transform(pair.first, pair.second);
    }
}

void GridMap::Impl::add_collision_shape(const GridCell& cell, const CellData& data) {
    if (!mesh_lib) return;
    const MeshLibrary::MeshData* mesh = mesh_lib->get_mesh(data.mesh_index);
    if (!mesh || !mesh->has_collision) return;
    // add shape to physics server using cell world transform
    // physics_server->body_add_shape(body_rid, mesh->collision_shape_rid, data.world_transform)
}

void GridMap::Impl::remove_collision_shape(const GridCell& cell) {
    // find and remove shape from physics body
}

void GridMap::Impl::rebuild_physics() {
    if (!mesh_lib) return;
    // Remove all existing collision shapes from physics server
    // For each cell, add shape with correct transform
    for (const auto& pair : cells) {
        add_collision_shape(pair.first, pair.second);
    }
    physics_dirty = false;
}

void GridMap::Impl::rebuild_navigation() {
    // similarly for navmesh
    navigation_dirty = false;
}

void GridMap::Impl::rebuild_render_instances() {
    if (!mesh_lib || instance_rid == -1) return;
    // In real engine: RenderingServer::multi_mesh_clear(instance_rid)
    // Then for each cell, add instance with transform, material overrides,
    // shadow flags, GI mode, emissive.
    // For performance, we use multi‑mesh instancing.
    for (const auto& pair : cells) {
        const CellData& data = pair.second;
        if (data.mesh_index < 0) continue;
        // apply per‑cell overrides (shadow, gi, emissive)
        // RenderingServer::multi_mesh_set_instance_transform(instance_rid, instance_idx, data.world_transform)
        // set instance custom data (cast_shadow, gi_mode, etc.)
    }
}

void GridMap::rebuild_grid() {
    if (!pimpl->mesh_lib) return;
    pimpl->update_all_cell_transforms();
    pimpl->rebuild_render_instances();
    if (pimpl->physics_dirty) pimpl->rebuild_physics();
    if (pimpl->navigation_dirty) pimpl->rebuild_navigation();
    pimpl->grid_dirty = false;
}

void GridMap::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->grid_dirty) {
        rebuild_grid();
    }
    // Also update global transforms of the grid map itself
}

void GridMap::process(double delta) {
    Node3D::process(delta);
    if (pimpl->grid_dirty) {
        rebuild_grid();
    }
}

} // namespace lighting