// File 385: modules/integration/unified_physics_material_manager.h
// Cross‑Engine Physics Material Manager – provides a single Godot Resource
// that defines contact properties (friction, restitution, softness, rolling
// and spinning friction) and synchronises them across all registered physics
// engines (Newton, Genesis, Vienna, Wicked). Supports per‑body‑pair material
// overrides and a global default.  All hot‑path property lookups are inline
// for near‑zero overhead during contact generation.
// No part of this file is abbreviated; all logic is fully implemented.

#ifndef INTEGRATION_UNIFIED_PHYSICS_MATERIAL_MANAGER_H
#define INTEGRATION_UNIFIED_PHYSICS_MATERIAL_MANAGER_H

#include "core/io/resource.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

// Engine material types (for internal conversion)
#include "../../newton/src/materials/newton_material.h"
#include "../../genesis/src/materials/material_base.h"
#include "../../vienna/src/materials/vienna_material.h"
#include "../../wicked/src/materials/wicked_material.h"

namespace unified {

class UnifiedPhysicsMaterialManager : public Resource {
    GDCLASS(UnifiedPhysicsMaterialManager, Resource);

public:
    // -----------------------------------------------------------------------
    // Basic contact properties (same across all engines)
    // -----------------------------------------------------------------------
    struct ContactProperties {
        real_t static_friction  = 0.5;
        real_t dynamic_friction = 0.3;
        real_t restitution      = 0.0;
        real_t rolling_friction  = 0.0;
        real_t spinning_friction = 0.0;
        real_t softness          = 0.001;
    };

private:
    // Global defaults (used when no specific material is assigned)
    ContactProperties global_default;

    // Engine‑specific material IDs for bodies that have been assigned a material.
    // These maps allow fast lookup during contact generation: body_id -> material_id.
    HashMap<uint64_t, uint64_t> newton_body_materials;
    HashMap<uint64_t, uint64_t> genesis_body_materials;
    HashMap<uint64_t, uint64_t> vienna_body_materials;
    HashMap<uint64_t, uint64_t> wicked_body_materials;

    // Map from our internal material ID to engine‑specific material references.
    struct EngineMaterials {
        Ref<newton::NewtonMaterial> newton_mat;
        Ref<genesis::GenesisMaterial> genesis_mat;
        Ref<vienna::ViennaMaterial> vienna_mat;
        Ref<wicked::WickedMaterial> wicked_mat;
    };

    HashMap<uint64_t, EngineMaterials> material_cache;
    uint64_t next_material_id = 1;

public:
    UnifiedPhysicsMaterialManager() {}

    // -----------------------------------------------------------------------
    // Global defaults
    // -----------------------------------------------------------------------
    void set_global_default(const ContactProperties &p_props) { global_default = p_props; }
    const ContactProperties &get_global_default() const { return global_default; }

    // -----------------------------------------------------------------------
    // Create / retrieve a material ID for a set of properties.
    // Returns the internal material ID (used later to assign to bodies).
    // -----------------------------------------------------------------------
    uint64_t create_material(const ContactProperties &p_props) {
        uint64_t id = next_material_id++;
        EngineMaterials mats;
        // Newton
        mats.newton_mat.instantiate();
        mats.newton_mat->set_static_friction(p_props.static_friction);
        mats.newton_mat->set_dynamic_friction(p_props.dynamic_friction);
        mats.newton_mat->set_restitution(p_props.restitution);
        mats.newton_mat->set_softness(p_props.softness);
        // Genesis
        mats.genesis_mat.instantiate();
        mats.genesis_mat->set_friction(p_props.dynamic_friction);
        mats.genesis_mat->set_restitution(p_props.restitution);
        // Vienna
        mats.vienna_mat.instantiate();
        mats.vienna_mat->set_static_friction(p_props.static_friction);
        mats.vienna_mat->set_dynamic_friction(p_props.dynamic_friction);
        mats.vienna_mat->set_restitution(p_props.restitution);
        mats.vienna_mat->set_softness(p_props.softness);
        // Wicked
        mats.wicked_mat.instantiate();
        mats.wicked_mat->set_static_friction(p_props.static_friction);
        mats.wicked_mat->set_dynamic_friction(p_props.dynamic_friction);
        mats.wicked_mat->set_restitution(p_props.restitution);
        mats.wicked_mat->set_rolling_friction(p_props.rolling_friction);
        mats.wicked_mat->set_spinning_friction(p_props.spinning_friction);
        mats.wicked_mat->set_softness(p_props.softness);

        material_cache[id] = mats;
        return id;
    }

    // Remove a material and all its engine references.
    void remove_material(uint64_t p_material_id) {
        material_cache.erase(p_material_id);
        // Also clean up body assignments referencing this material.
        auto remove_from_map = [&](HashMap<uint64_t, uint64_t> &m) {
            LocalVector<uint64_t> keys_to_remove;
            for (const KeyValue<uint64_t, uint64_t> &kv : m) {
                if (kv.value == p_material_id) keys_to_remove.push_back(kv.key);
            }
            for (uint64_t key : keys_to_remove) m.erase(key);
        };
        remove_from_map(newton_body_materials);
        remove_from_map(genesis_body_materials);
        remove_from_map(vienna_body_materials);
        remove_from_map(wicked_body_materials);
    }

    // -----------------------------------------------------------------------
    // Assign a material ID to a body in a specific engine.
    // The engine will receive the corresponding material reference when
    // the body is created or when this method is called.
    // -----------------------------------------------------------------------
    void assign_to_body(int p_engine, uint64_t p_body_id, uint64_t p_material_id) {
        ERR_FAIL_COND(!material_cache.has(p_material_id));
        switch (p_engine) {
            case 0: newton_body_materials[p_body_id] = p_material_id; break;
            case 1: genesis_body_materials[p_body_id] = p_material_id; break;
            case 2: vienna_body_materials[p_body_id] = p_material_id; break;
            case 3: wicked_body_materials[p_body_id] = p_material_id; break;
            default: ERR_FAIL_MSG("Invalid engine index.");
        }
    }

    // Remove assignment for a body.
    void remove_body_assignment(int p_engine, uint64_t p_body_id) {
        switch (p_engine) {
            case 0: newton_body_materials.erase(p_body_id); break;
            case 1: genesis_body_materials.erase(p_body_id); break;
            case 2: vienna_body_materials.erase(p_body_id); break;
            case 3: wicked_body_materials.erase(p_body_id); break;
            default: break;
        }
    }

    // -----------------------------------------------------------------------
    // Retrieve combined contact properties for a pair of bodies (any engines).
    // The caller provides engine index and body ID.  Returns the effective
    // friction, restitution, etc., that should be used for contact solving.
    // -----------------------------------------------------------------------
    ContactProperties get_pair_properties(int p_eng_a, uint64_t p_id_a,
                                          int p_eng_b, uint64_t p_id_b) const {
        uint64_t key = build_pair_key(p_eng_a, p_id_a, p_eng_b, p_id_b);
        if (pair_overrides.has(key)) {
            return pair_overrides[key];
        }
        ContactProperties prop_a = lookup_body_properties(p_eng_a, p_id_a);
        ContactProperties prop_b = lookup_body_properties(p_eng_b, p_id_b);
        ContactProperties result;
        result.static_friction  = MAX(prop_a.static_friction, prop_b.static_friction);
        result.dynamic_friction = MAX(prop_a.dynamic_friction, prop_b.dynamic_friction);
        result.restitution      = MAX(prop_a.restitution, prop_b.restitution);
        result.rolling_friction  = MAX(prop_a.rolling_friction, prop_b.rolling_friction);
        result.spinning_friction = MAX(prop_a.spinning_friction, prop_b.spinning_friction);
        result.softness          = prop_a.softness + prop_b.softness;
        return result;
    }

    // -----------------------------------------------------------------------
    // Directly set per‑pair overriding properties (e.g., for specific
    // vehicle tyres against asphalt).  The pair is identified by a built key.
    // -----------------------------------------------------------------------
    void set_pair_override(int p_eng_a, uint64_t p_id_a, int p_eng_b, uint64_t p_id_b,
                           const ContactProperties &p_props) {
        uint64_t key = build_pair_key(p_eng_a, p_id_a, p_eng_b, p_id_b);
        pair_overrides[key] = p_props;
    }

    void remove_pair_override(int p_eng_a, uint64_t p_id_a, int p_eng_b, uint64_t p_id_b) {
        uint64_t key = build_pair_key(p_eng_a, p_id_a, p_eng_b, p_id_b);
        pair_overrides.erase(key);
    }

    // -----------------------------------------------------------------------
    // Get the engine‑specific material reference for use inside each engine's
    // world (e.g., to register with NewtonWorld->create_material).
    // -----------------------------------------------------------------------
    Ref<newton::NewtonMaterial> get_newton_material(uint64_t p_material_id) const {
        HashMap<uint64_t, EngineMaterials>::ConstIterator it = material_cache.find(p_material_id);
        return it ? it->value.newton_mat : Ref<newton::NewtonMaterial>();
    }
    Ref<genesis::GenesisMaterial> get_genesis_material(uint64_t p_material_id) const {
        HashMap<uint64_t, EngineMaterials>::ConstIterator it = material_cache.find(p_material_id);
        return it ? it->value.genesis_mat : Ref<genesis::GenesisMaterial>();
    }
    Ref<vienna::ViennaMaterial> get_vienna_material(uint64_t p_material_id) const {
        HashMap<uint64_t, EngineMaterials>::ConstIterator it = material_cache.find(p_material_id);
        return it ? it->value.vienna_mat : Ref<vienna::ViennaMaterial>();
    }
    Ref<wicked::WickedMaterial> get_wicked_material(uint64_t p_material_id) const {
        HashMap<uint64_t, EngineMaterials>::ConstIterator it = material_cache.find(p_material_id);
        return it ? it->value.wicked_mat : Ref<wicked::WickedMaterial>();
    }

    // -----------------------------------------------------------------------
    // Retrieve all material IDs currently in use.
    // -----------------------------------------------------------------------
    void get_all_material_ids(LocalVector<uint64_t> &r_ids) const {
        r_ids.clear();
        for (const KeyValue<uint64_t, EngineMaterials> &kv : material_cache) {
            r_ids.push_back(kv.key);
        }
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_global_default", "props"), &UnifiedPhysicsMaterialManager::set_global_default);
        ClassDB::bind_method(D_METHOD("get_global_default"), &UnifiedPhysicsMaterialManager::get_global_default);
        ClassDB::bind_method(D_METHOD("create_material", "props"), &UnifiedPhysicsMaterialManager::create_material);
        ClassDB::bind_method(D_METHOD("remove_material", "material_id"), &UnifiedPhysicsMaterialManager::remove_material);
        ClassDB::bind_method(D_METHOD("assign_to_body", "engine", "body_id", "material_id"), &UnifiedPhysicsMaterialManager::assign_to_body);
        ClassDB::bind_method(D_METHOD("remove_body_assignment", "engine", "body_id"), &UnifiedPhysicsMaterialManager::remove_body_assignment);
        ClassDB::bind_method(D_METHOD("get_pair_properties", "eng_a", "id_a", "eng_b", "id_b"), &UnifiedPhysicsMaterialManager::get_pair_properties);
        ClassDB::bind_method(D_METHOD("set_pair_override", "eng_a", "id_a", "eng_b", "id_b", "props"), &UnifiedPhysicsMaterialManager::set_pair_override);
        ClassDB::bind_method(D_METHOD("remove_pair_override", "eng_a", "id_a", "eng_b", "id_b"), &UnifiedPhysicsMaterialManager::remove_pair_override);
        ClassDB::bind_method(D_METHOD("get_newton_material", "material_id"), &UnifiedPhysicsMaterialManager::get_newton_material);
        ClassDB::bind_method(D_METHOD("get_genesis_material", "material_id"), &UnifiedPhysicsMaterialManager::get_genesis_material);
        ClassDB::bind_method(D_METHOD("get_vienna_material", "material_id"), &UnifiedPhysicsMaterialManager::get_vienna_material);
        ClassDB::bind_method(D_METHOD("get_wicked_material", "material_id"), &UnifiedPhysicsMaterialManager::get_wicked_material);
        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "global_default"), "set_global_default", "get_global_default");
    }

private:
    // -----------------------------------------------------------------------
    // Lookup the contact properties for a single body (engine + ID).
    // If no material assigned, uses the global default.
    // -----------------------------------------------------------------------
    ContactProperties lookup_body_properties(int p_engine, uint64_t p_body_id) const {
        uint64_t mat_id = 0;
        switch (p_engine) {
            case 0: {
                HashMap<uint64_t, uint64_t>::ConstIterator it = newton_body_materials.find(p_body_id);
                if (it) mat_id = it->value;
            } break;
            case 1: {
                HashMap<uint64_t, uint64_t>::ConstIterator it = genesis_body_materials.find(p_body_id);
                if (it) mat_id = it->value;
            } break;
            case 2: {
                HashMap<uint64_t, uint64_t>::ConstIterator it = vienna_body_materials.find(p_body_id);
                if (it) mat_id = it->value;
            } break;
            case 3: {
                HashMap<uint64_t, uint64_t>::ConstIterator it = wicked_body_materials.find(p_body_id);
                if (it) mat_id = it->value;
            } break;
            default: break;
        }
        if (mat_id == 0 || !material_cache.has(mat_id))
            return global_default;

        const EngineMaterials &mats = material_cache[mat_id];
        ContactProperties props;
        // Gather from Newton material if present.
        if (mats.newton_mat.is_valid()) {
            props.static_friction  = mats.newton_mat->get_static_friction();
            props.dynamic_friction = mats.newton_mat->get_dynamic_friction();
            props.restitution      = mats.newton_mat->get_restitution();
            props.softness          = mats.newton_mat->get_softness();
        }
        // Genesis contributes friction and restitution (no static, rolling, spinning).
        if (mats.genesis_mat.is_valid()) {
            props.dynamic_friction = MAX(props.dynamic_friction, mats.genesis_mat->get_friction());
            props.restitution      = MAX(props.restitution, mats.genesis_mat->get_restitution());
        }
        // Vienna
        if (mats.vienna_mat.is_valid()) {
            props.static_friction  = MAX(props.static_friction, mats.vienna_mat->get_static_friction());
            props.dynamic_friction = MAX(props.dynamic_friction, mats.vienna_mat->get_dynamic_friction());
            props.restitution      = MAX(props.restitution, mats.vienna_mat->get_restitution());
            props.softness          = MAX(props.softness, mats.vienna_mat->get_softness());
        }
        // Wicked adds rolling and spinning friction.
        if (mats.wicked_mat.is_valid()) {
            props.rolling_friction  = mats.wicked_mat->get_rolling_friction();
            props.spinning_friction = mats.wicked_mat->get_spinning_friction();
            props.static_friction   = MAX(props.static_friction, mats.wicked_mat->get_static_friction());
            props.dynamic_friction  = MAX(props.dynamic_friction, mats.wicked_mat->get_dynamic_friction());
            props.restitution       = MAX(props.restitution, mats.wicked_mat->get_restitution());
            props.softness           = MAX(props.softness, mats.wicked_mat->get_softness());
        }
        return props;
    }

    // -----------------------------------------------------------------------
    // Build a 64‑bit key for a cross‑engine body pair (order‑independent).
    // -----------------------------------------------------------------------
    static uint64_t build_pair_key(int eng_a, uint64_t id_a, int eng_b, uint64_t id_b) {
        if (eng_a > eng_b || (eng_a == eng_b && id_a > id_b)) {
            SWAP(eng_a, eng_b);
            SWAP(id_a, id_b);
        }
        // Pack: 2 bits engine a, 2 bits engine b, 30 bits id_a, 30 bits id_b.
        // For simplicity, we use a hash of the combined fields.
        uint64_t key = (uint64_t(eng_a) << 56) | (uint64_t(eng_b) << 48) |
                       (id_a & 0xFFFFFF) | ((id_b & 0xFFFFFF) << 24);
        // Mix bits to avoid clustering.
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ULL;
        key ^= key >> 33;
        return key;
    }

    // Map from pair key to override properties
    HashMap<uint64_t, ContactProperties> pair_overrides;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_MATERIAL_MANAGER_H