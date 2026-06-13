// File 427: modules/integration/unified_physics_prefab_serializer.h
// Compact binary prefab serializer for the unified physics pipeline.
// Saves and restores the complete state of all physics bodies, joints,
// vehicles, cloth instances, and particle systems across all registered
// engines (Newton, Genesis, Vienna, Wicked) using a custom versioned
// binary stream.  No external dependencies beyond Godot's FileAccess.

#ifndef INTEGRATION_UNIFIED_PHYSICS_PREFAB_SERIALIZER_H
#define INTEGRATION_UNIFIED_PHYSICS_PREFAB_SERIALIZER_H

#include "core/io/file_access.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedPhysicsPrefabSerializer {
public:
    // File format version identifier.
    static constexpr uint32_t MAGIC = 0x55504648;   // 'UPFH'
    static constexpr uint32_t VERSION = 1;

    // -------------------------------------------------------------------
    // Save all physics data of the unified server to a binary file.
    // The server is expected to expose methods to iterate over all bodies,
    // joints, vehicles, cloths, and particles for each engine.
    // -------------------------------------------------------------------
    Error save_to_file(const String &p_path,
                       const HashMap<int, void *> &p_engine_worlds) const;

    // -------------------------------------------------------------------
    // Load the physics data from a binary file and reconstruct all
    // objects into the provided engine worlds.  Existing objects are
    // not automatically removed; call clear_worlds() before loading if
    // a full reset is desired.
    // -------------------------------------------------------------------
    Error load_from_file(const String &p_path,
                         const HashMap<int, void *> &p_engine_worlds) const;

    // -------------------------------------------------------------------
    // Convenience: clear all worlds (destroy every body, joint, vehicle,
    // cloth, and particle system).  This is safe to call before loading.
    // -------------------------------------------------------------------
    static void clear_worlds(const HashMap<int, void *> &p_engine_worlds);

private:
    // -------------------------------------------------------------------
    // Low‑level binary I/O helpers.
    // -------------------------------------------------------------------
    static void write_uint32(Ref<FileAccess> p_f, uint32_t p_val);
    static uint32_t read_uint32(Ref<FileAccess> p_f);
    static void write_uint64(Ref<FileAccess> p_f, uint64_t p_val);
    static uint64_t read_uint64(Ref<FileAccess> p_f);
    static void write_real(Ref<FileAccess> p_f, real_t p_val);
    static real_t read_real(Ref<FileAccess> p_f);
    static void write_vector3(Ref<FileAccess> p_f, const Vector3 &p_vec);
    static Vector3 read_vector3(Ref<FileAccess> p_f);
    static void write_transform(Ref<FileAccess> p_f, const Transform3D &p_xform);
    static Transform3D read_transform(Ref<FileAccess> p_f);
    static void write_basis(Ref<FileAccess> p_f, const Basis &p_basis);
    static Basis read_basis(Ref<FileAccess> p_f);
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_PREFAB_SERIALIZER_H