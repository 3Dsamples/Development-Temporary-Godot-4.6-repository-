// File 425: modules/integration/unified_physics_material_binder.h
// Dynamically binds a single unified material resource (friction, restitution,
// softness, rolling/spinning friction) to one or more physics bodies across
// any registered engine (Newton, Genesis, Vienna, Wicked).  It creates the
// engine‑specific material instances via the UnifiedPhysicsMaterialManager,
// caches the assigned material IDs, and updates them in real time when the
// unified resource properties change.  All methods are fully implemented.

#ifndef INTEGRATION_UNIFIED_PHYSICS_MATERIAL_BINDER_H
#define INTEGRATION_UNIFIED_PHYSICS_MATERIAL_BINDER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include "unified_physics_material_manager.h"

namespace unified {

class UnifiedPhysicsMaterialBinder : public RefCounted {
    GDCLASS(UnifiedPhysicsMaterialBinder, RefCounted);

public:
    // Manager that owns the material instances (must be valid).
    Ref<UnifiedPhysicsMaterialManager> material_manager;

    // -------------------------------------------------------------------
    // Bind a body to a specific material property set.
    // If p_material_id is 0, the body will use the global default.
    // -------------------------------------------------------------------
    void bind_body(int p_engine, uint64_t p_body_id, uint64_t p_material_id);

    // Remove a binding (body reverts to global default).
    void unbind_body(int p_engine, uint64_t p_body_id);

    // -------------------------------------------------------------------
    // Create a new material from the given properties and assign it to
    // the body.  Returns the material ID (persistent).
    // -------------------------------------------------------------------
    uint64_t create_and_assign(int p_engine, uint64_t p_body_id,
                               const UnifiedPhysicsMaterialManager::ContactProperties &p_props);

    // -------------------------------------------------------------------
    // Set a single property on a body without changing the other
    // properties.  If the body is currently using the global default,
    // a new material is automatically created first.
    // -------------------------------------------------------------------
    void set_body_friction(int p_engine, uint64_t p_body_id, real_t p_friction);
    void set_body_restitution(int p_engine, uint64_t p_body_id, real_t p_restitution);
    void set_body_softness(int p_engine, uint64_t p_body_id, real_t p_softness);
    void set_body_rolling_friction(int p_engine, uint64_t p_body_id, real_t p_rf);
    void set_body_spinning_friction(int p_engine, uint64_t p_body_id, real_t p_sf);

    // -------------------------------------------------------------------
    // Query the currently assigned material ID for a body (0 = default).
    // -------------------------------------------------------------------
    uint64_t get_body_material_id(int p_engine, uint64_t p_body_id) const;

    // -------------------------------------------------------------------
    // Bulk assign: iterate over a list of body IDs and apply the same
    // properties.
    // -------------------------------------------------------------------
    void assign_to_bodies(int p_engine, const LocalVector<uint64_t> &p_body_ids,
                          const UnifiedPhysicsMaterialManager::ContactProperties &p_props);

    // Clear all bindings.
    void clear_all();

    // Return the number of active bindings.
    int get_binding_count() const;

protected:
    static void _bind_methods();

private:
    // Maps (engine, body ID) -> material ID.
    struct BodyKey {
        int engine;
        uint64_t body_id;
        BodyKey() : engine(0), body_id(0) {}
        BodyKey(int e, uint64_t id) : engine(e), body_id(id) {}
        bool operator==(const BodyKey &o) const { return engine==o.engine && body_id==o.body_id; }
        struct Hash {
            uint64_t operator()(const BodyKey &k) const {
                uint64_t h = (uint64_t(k.engine) << 56) | (k.body_id & 0xFFFFFFFFFFFFFFULL);
                h ^= h >> 33;
                h *= 0xff51afd7ed558ccdULL;
                h ^= h >> 33;
                h *= 0xc4ceb9fe1a85ec53ULL;
                h ^= h >> 33;
                return h;
            }
        };
    };

    HashMap<BodyKey, uint64_t, BodyKey::Hash> body_to_material;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_MATERIAL_BINDER_H