// grid_map.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace lighting {

// ============================================================================
// GridMap – places mesh instances on a 3D grid.
// Each cell stores a mesh index (from a MeshLibrary) and transforms (flip, rotate).
// Supports large worlds via chunked storage, physics collisions, navigation,
// and full lighting (shadows, GI, emissive per cell).
// ============================================================================

class MeshLibrary; // forward declaration (asset container)

struct GridCell {
    int32_t x = 0, y = 0, z = 0;
    bool operator==(const GridCell& o) const { return x==o.x && y==o.y && z==o.z; }
};

namespace std {
    template<> struct hash<GridCell> {
        size_t operator()(const GridCell& c) const {
            return ((c.x * 73856093) ^ (c.y * 19349663) ^ (c.z * 83492791));
        }
    };
}

class GridMap : public Node3D {
public:
    GridMap();
    ~GridMap();

    // ------------------------------------------------------------------------
    // Grid dimensions and cell size
    // ------------------------------------------------------------------------
    void set_cell_size(double size);
    double get_cell_size() const;
    void set_grid_bounds(int64_t min_x, int64_t min_y, int64_t min_z,
                         int64_t max_x, int64_t max_y, int64_t max_z); // 0 = unlimited
    void get_grid_bounds(int64_t& min_x, int64_t& min_y, int64_t& min_z,
                         int64_t& max_x, int64_t& max_y, int64_t& max_z) const;

    // ------------------------------------------------------------------------
    // Mesh library (asset collection)
    // ------------------------------------------------------------------------
    void set_mesh_library(const MeshLibrary* library);
    const MeshLibrary* get_mesh_library() const;

    // ------------------------------------------------------------------------
    // Cell operations
    // ------------------------------------------------------------------------
    void set_cell(int64_t x, int64_t y, int64_t z, int mesh_index,
                  int orientation = 0); // orientation: 0..23 (rotations/reflections)
    int get_cell(int64_t x, int64_t y, int64_t z) const;
    void erase_cell(int64_t x, int64_t y, int64_t z);
    void clear();

    // ------------------------------------------------------------------------
    // Get all occupied cells (for iteration)
    // ------------------------------------------------------------------------
    void get_used_cells(std::vector<GridCell>& out_cells) const;

    // ------------------------------------------------------------------------
    // Collision and navigation (physics)
    // ------------------------------------------------------------------------
    void set_collision_layer(uint32_t layer);
    uint32_t get_collision_layer() const;
    void set_collision_mask(uint32_t mask);
    uint32_t get_collision_mask() const;
    void rebuild_collision();   // regenerate physics shapes from current cells
    void rebuild_navigation();  // rebuild navmesh from cells

    // ------------------------------------------------------------------------
    // Lighting & GI per cell (some cells can be static, others dynamic)
    // ------------------------------------------------------------------------
    void set_cell_gi_mode(int64_t x, int64_t y, int64_t z, int mode); // 0=off,1=static,2=dynamic
    int get_cell_gi_mode(int64_t x, int64_t y, int64_t z) const;
    void set_cell_emissive(int64_t x, int64_t y, int64_t z, const float* color, float intensity);
    void get_cell_emissive(int64_t x, int64_t y, int64_t z, float* out_color, float& out_intensity) const;
    void set_cell_cast_shadow(int64_t x, int64_t y, int64_t z, bool cast);
    bool get_cell_cast_shadow(int64_t x, int64_t y, int64_t z) const;

    // ------------------------------------------------------------------------
    // Global grid overrides (applied to all cells unless overridden)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;

    // ------------------------------------------------------------------------
    // Performance: chunked loading (optional)
    // ------------------------------------------------------------------------
    void set_chunk_size(int chunks_xy, int chunks_z);
    int get_chunk_size_xy() const;
    int get_chunk_size_z() const;

    // ------------------------------------------------------------------------
    // Rendering server sync (rebuilds instance buffers)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

    // ------------------------------------------------------------------------
    // Force rebuild (call after many cell changes)
    // ------------------------------------------------------------------------
    void rebuild_grid();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting