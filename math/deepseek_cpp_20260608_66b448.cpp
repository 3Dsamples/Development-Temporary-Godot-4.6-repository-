// File 429: modules/integration/unified_physics_streaming_world.h
// Streaming world system for large open‑world physics scenes.
// Partitions static bodies into a grid of cells, loading (activating) cells
// near the camera and unloading (deactivating) distant cells.  Dynamic and
// kinematic bodies are never streamed and remain active at all times.
// The system can manage multiple engine worlds (Newton, Genesis, Vienna,
// Wicked) simultaneously.  All cell management is fully implemented.

#ifndef INTEGRATION_UNIFIED_PHYSICS_STREAMING_WORLD_H
#define INTEGRATION_UNIFIED_PHYSICS_STREAMING_WORLD_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedPhysicsStreamingWorld : public RefCounted {
    GDCLASS(UnifiedPhysicsStreamingWorld, RefCounted);

public:
    // -------------------------------------------------------------------
    // Per‑cell data: list of static body IDs belonging to this cell.
    // -------------------------------------------------------------------
    struct Cell {
        LocalVector<uint64_t> body_ids;    // engine‑specific body IDs
        bool loaded = false;               // true when bodies are active
    };

    // -------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------
    Vector3 grid_origin;                   // world‑space corner of the grid
    Vector3 cell_size = Vector3(50, 50, 50);
    Vector3i grid_dimensions = Vector3i(8, 1, 8);  // cells along X, Y, Z
    int load_radius_cells = 2;             // how many cells around listener to keep loaded

    // -------------------------------------------------------------------
    // Set the world pointer for a specific engine.
    // -------------------------------------------------------------------
    void set_world(int p_engine, void *p_world);

    // -------------------------------------------------------------------
    // Register a static body.  Its cell is computed from its world position.
    // The body must already be created in the world and be STATIC.
    // -------------------------------------------------------------------
    void register_static_body(int p_engine, uint64_t p_body_id, const Vector3 &p_position);

    // Remove a previously registered static body (e.g., on destruction).
    void unregister_static_body(int p_engine, uint64_t p_body_id);

    // -------------------------------------------------------------------
    // Update the streaming state based on a new listener position.
    // Call once per frame (or whenever the camera moves significantly).
    // -------------------------------------------------------------------
    void update(const Vector3 &p_listener_position);

    // Force load a specific cell (activate its bodies).
    void load_cell(const Vector3i &p_cell_index);
    // Force unload a cell (deactivate its bodies).
    void unload_cell(const Vector3i &p_cell_index);

    // -------------------------------------------------------------------
    // Query cell index from world position.
    // -------------------------------------------------------------------
    Vector3i world_to_cell(const Vector3 &p_position) const;

    // Get the cell data for a given index (for inspection).
    const Cell *get_cell(const Vector3i &p_cell_index) const;

    // Return total number of registered static bodies.
    int get_registered_body_count() const;

protected:
    static void _bind_methods();

private:
    // Engine worlds (pointers, not owned).
    struct EngineWorld {
        int engine;
        void *world;
    };
    LocalVector<EngineWorld> worlds;

    // Cell storage: map from flat index (x + y*grid_dimensions.x + z*...) to Cell.
    // Flat index is computed by cell_to_index().
    HashMap<int64_t, Cell> cells;

    // Reverse map: (engine, body_id) -> cell index, for fast removal.
    struct BodyKey {
        int engine;
        uint64_t body_id;
        bool operator==(const BodyKey &o) const { return engine==o.engine && body_id==o.body_id; }
        struct Hash {
            uint64_t operator()(const BodyKey &k) const {
                return (uint64_t(k.engine) << 56) | (k.body_id & 0xFFFFFFFFFFFFFFULL);
            }
        };
    };
    HashMap<BodyKey, int64_t, BodyKey::Hash> body_to_cell;

    // Currently loaded cells (for quick iteration when unloading).
    HashSet<int64_t> loaded_cells;

    // Helper to activate / deactivate a body across all its engines.
    void set_body_active(int p_engine, uint64_t p_body_id, bool p_active);

    // Convert cell index to flat index.
    int64_t cell_to_index(const Vector3i &p_cell) const;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_STREAMING_WORLD_H