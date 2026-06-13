// File 426: modules/integration/unified_physics_material_binder.cpp
// Implementation of the material binder.  All operations go through the
// UnifiedPhysicsMaterialManager to create, update, and assign materials.

#include "unified_physics_material_binder.h"
#include "unified_physics_material_manager.h"

namespace unified {

void UnifiedPhysicsMaterialBinder::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_material_manager", "manager"),
        &UnifiedPhysicsMaterialBinder::set_material_manager);
    ClassDB::bind_method(D_METHOD("get_material_manager"),
        &UnifiedPhysicsMaterialBinder::get_material_manager);
    ClassDB::bind_method(D_METHOD("bind_body", "engine", "body_id", "material_id"),
        &UnifiedPhysicsMaterialBinder::bind_body);
    ClassDB::bind_method(D_METHOD("unbind_body", "engine", "body_id"),
        &UnifiedPhysicsMaterialBinder::unbind_body);
    ClassDB::bind_method(D_METHOD("create_and_assign", "engine", "body_id", "props"),
        &UnifiedPhysicsMaterialBinder::create_and_assign);
    ClassDB::bind_method(D_METHOD("set_body_friction", "engine", "body_id", "friction"),
        &UnifiedPhysicsMaterialBinder::set_body_friction);
    ClassDB::bind_method(D_METHOD("set_body_restitution", "engine", "body_id", "restitution"),
        &UnifiedPhysicsMaterialBinder::set_body_restitution);
    ClassDB::bind_method(D_METHOD("set_body_softness", "engine", "body_id", "softness"),
        &UnifiedPhysicsMaterialBinder::set_body_softness);
    ClassDB::bind_method(D_METHOD("set_body_rolling_friction", "engine", "body_id", "rf"),
        &UnifiedPhysicsMaterialBinder::set_body_rolling_friction);
    ClassDB::bind_method(D_METHOD("set_body_spinning_friction", "engine", "body_id", "sf"),
        &UnifiedPhysicsMaterialBinder::set_body_spinning_friction);
    ClassDB::bind_method(D_METHOD("get_body_material_id", "engine", "body_id"),
        &UnifiedPhysicsMaterialBinder::get_body_material_id);
    ClassDB::bind_method(D_METHOD("assign_to_bodies", "engine", "body_ids", "props"),
        &UnifiedPhysicsMaterialBinder::assign_to_bodies);
    ClassDB::bind_method(D_METHOD("clear_all"),
        &UnifiedPhysicsMaterialBinder::clear_all);
    ClassDB::bind_method(D_METHOD("get_binding_count"),
        &UnifiedPhysicsMaterialBinder::get_binding_count);
}

// ---------------------------------------------------------------------------
// Bind a body to an existing material ID.
// ---------------------------------------------------------------------------
void UnifiedPhysicsMaterialBinder::bind_body(int p_engine, uint64_t p_body_id,
                                              uint64_t p_material_id) {
    ERR_FAIL_COND(!material_manager.is_valid());
    BodyKey key(p_engine, p_body_id);
    if (p_material_id != 0 && material_manager->get_newton_material(p_material_id).is_null()) {
        // The material ID is invalid; create a new one from global defaults.
        UnifiedPhysicsMaterialManager::ContactProperties props =
            material_manager->get_global_default();
        p_material_id = material_manager->create_material(props);
    }
    body_to_material[key] = p_material_id;
    material_manager->assign_to_body(p_engine, p_body_id, p_material_id);
}

// ---------------------------------------------------------------------------
// Unbind a body (revert to global defaults).
// ---------------------------------------------------------------------------
void UnifiedPhysicsMaterialBinder::unbind_body(int p_engine, uint64_t p_body_id) {
    BodyKey key(p_engine, p_body_id);
    body_to_material.erase(key);
    if (material_manager.is_valid()) {
        material_manager->remove_body_assignment(p_engine, p_body_id);
    }
}

// ---------------------------------------------------------------------------
// Create a fresh material from properties and assign it to a body.
// ---------------------------------------------------------------------------
uint64_t UnifiedPhysicsMaterialBinder::create_and_assign(
    int p_engine, uint64_t p_body_id,
    const UnifiedPhysicsMaterialManager::ContactProperties &p_props) {
    ERR_FAIL_COND_V(!material_manager.is_valid(), 0);
    uint64_t mat_id = material_manager->create_material(p_props);
    BodyKey key(p_engine, p_body_id);
    body_to_material[key] = mat_id;
    material_manager->assign_to_body(p_engine, p_body_id, mat_id);
    return mat_id;
}

// ---------------------------------------------------------------------------
// Per‑property setters – they read the current material (or create one),
// modify the property, and update the material manager.
// ---------------------------------------------------------------------------
void UnifiedPhysicsMaterialBinder::set_body_friction(int p_engine, uint64_t p_body_id,
                                                     real_t p_friction) {
    ERR_FAIL_COND(!material_manager.is_valid());
    uint64_t mat_id = get_body_material_id(p_engine, p_body_id);
    UnifiedPhysicsMaterialManager::ContactProperties props;
    if (mat_id != 0) {
        // Retrieve current properties from the manager via material instances.
        // We'll need a way to get the stored props from the material ID.
        // Since MaterialManager stores engine materials but not a single
        // ContactProperties struct per ID, we'll create a new material each time
        // to apply a property change.  To preserve existing other properties,
        // we can fetch the Newton material (if any) to read its current values,
        // modify the desired field, then call create_and_assign.
        // For simplicity, we'll assume the user uses create_and_assign for first
        // setup and set_body_* as a convenience that replaces entire material.
        // But the requirement says "set a single property without changing the other
        // properties."  So we'll retrieve the current combined props from the manager
        // using get_pair_properties? That takes two bodies.  Instead, we'll access
        // the Newton material to get its properties (as a fallback).  If not
        // available, we'll use the global defaults as base.
        Ref<newton::NewtonMaterial> newton_mat = material_manager->get_newton_material(mat_id);
        if (newton_mat.is_valid()) {
            props.static_friction = newton_mat->get_static_friction();
            props.dynamic_friction = newton_mat->get_dynamic_friction();
            props.restitution = newton_mat->get_restitution();
            props.softness = newton_mat->get_softness();
        } else {
            props = material_manager->get_global_default();
        }
    } else {
        props = material_manager->get_global_default();
    }
    props.dynamic_friction = p_friction;
    props.static_friction = p_friction;
    create_and_assign(p_engine, p_body_id, props);
}

void UnifiedPhysicsMaterialBinder::set_body_restitution(int p_engine, uint64_t p_body_id,
                                                        real_t p_restitution) {
    // Similar pattern: get current props, modify restitution, re‑assign.
    ERR_FAIL_COND(!material_manager.is_valid());
    uint64_t mat_id = get_body_material_id(p_engine, p_body_id);
    UnifiedPhysicsMaterialManager::ContactProperties props =
        mat_id != 0 ? get_material_properties(mat_id) : material_manager->get_global_default();
    props.restitution = p_restitution;
    create_and_assign(p_engine, p_body_id, props);
}

void UnifiedPhysicsMaterialBinder::set_body_softness(int p_engine, uint64_t p_body_id,
                                                     real_t p_softness) {
    ERR_FAIL_COND(!material_manager.is_valid());
    uint64_t mat_id = get_body_material_id(p_engine, p_body_id);
    UnifiedPhysicsMaterialManager::ContactProperties props =
        mat_id != 0 ? get_material_properties(mat_id) : material_manager->get_global_default();
    props.softness = p_softness;
    create_and_assign(p_engine, p_body_id, props);
}

void UnifiedPhysicsMaterialBinder::set_body_rolling_friction(int p_engine, uint64_t p_body_id,
                                                             real_t p_rf) {
    ERR_FAIL_COND(!material_manager.is_valid());
    uint64_t mat_id = get_body_material_id(p_engine, p_body_id);
    UnifiedPhysicsMaterialManager::ContactProperties props =
        mat_id != 0 ? get_material_properties(mat_id) : material_manager->get_global_default();
    props.rolling_friction = p_rf;
    create_and_assign(p_engine, p_body_id, props);
}

void UnifiedPhysicsMaterialBinder::set_body_spinning_friction(int p_engine, uint64_t p_body_id,
                                                              real_t p_sf) {
    ERR_FAIL_COND(!material_manager.is_valid());
    uint64_t mat_id = get_body_material_id(p_engine, p_body_id);
    UnifiedPhysicsMaterialManager::ContactProperties props =
        mat_id != 0 ? get_material_properties(mat_id) : material_manager->get_global_default();
    props.spinning_friction = p_sf;
    create_and_assign(p_engine, p_body_id, props);
}

// ---------------------------------------------------------------------------
// Retrieve the current material ID for a body.
// ---------------------------------------------------------------------------
uint64_t UnifiedPhysicsMaterialBinder::get_body_material_id(int p_engine, uint64_t p_body_id) const {
    BodyKey key(p_engine, p_body_id);
    HashMap<BodyKey, uint64_t, BodyKey::Hash>::ConstIterator it = body_to_material.find(key);
    return it ? it->value : 0;
}

// ---------------------------------------------------------------------------
// Bulk assignment.
// ---------------------------------------------------------------------------
void UnifiedPhysicsMaterialBinder::assign_to_bodies(
    int p_engine, const LocalVector<uint64_t> &p_body_ids,
    const UnifiedPhysicsMaterialManager::ContactProperties &p_props) {
    ERR_FAIL_COND(!material_manager.is_valid());
    uint64_t mat_id = material_manager->create_material(p_props);
    for (uint64_t id : p_body_ids) {
        BodyKey key(p_engine, id);
        body_to_material[key] = mat_id;
        material_manager->assign_to_body(p_engine, id, mat_id);
    }
}

// ---------------------------------------------------------------------------
// Clear all bindings.
// ---------------------------------------------------------------------------
void UnifiedPhysicsMaterialBinder::clear_all() {
    if (material_manager.is_valid()) {
        for (const KeyValue<BodyKey, uint64_t> &kv : body_to_material) {
            material_manager->remove_body_assignment(kv.key.engine, kv.key.body_id);
        }
    }
    body_to_material.clear();
}

int UnifiedPhysicsMaterialBinder::get_binding_count() const {
    return body_to_material.size();
}

// ---------------------------------------------------------------------------
// Helper: retrieve the stored ContactProperties for a material ID.
// ---------------------------------------------------------------------------
UnifiedPhysicsMaterialManager::ContactProperties
UnifiedPhysicsMaterialBinder::get_material_properties(uint64_t p_mat_id) const {
    UnifiedPhysicsMaterialManager::ContactProperties props;
    if (!material_manager.is_valid()) return props;
    Ref<newton::NewtonMaterial> nm = material_manager->get_newton_material(p_mat_id);
    if (nm.is_valid()) {
        props.static_friction = nm->get_static_friction();
        props.dynamic_friction = nm->get_dynamic_friction();
        props.restitution = nm->get_restitution();
        props.softness = nm->get_softness();
    }
    Ref<vienna::ViennaMaterial> vm = material_manager->get_vienna_material(p_mat_id);
    if (vm.is_valid()) {
        props.dynamic_friction = MAX(props.dynamic_friction, vm->get_dynamic_friction());
        props.restitution = MAX(props.restitution, vm->get_restitution());
        props.softness = MAX(props.softness, vm->get_softness());
    }
    Ref<wicked::WickedMaterial> wm = material_manager->get_wicked_material(p_mat_id);
    if (wm.is_valid()) {
        props.rolling_friction = wm->get_rolling_friction();
        props.spinning_friction = wm->get_spinning_friction();
        props.dynamic_friction = MAX(props.dynamic_friction, wm->get_dynamic_friction());
        props.restitution = MAX(props.restitution, wm->get_restitution());
        props.softness = MAX(props.softness, wm->get_softness());
    }
    // If no engine material is available, fallback to global default.
    if (!nm.is_valid() && !vm.is_valid() && !wm.is_valid()) {
        props = material_manager->get_global_default();
    }
    return props;
}

} // namespace unified