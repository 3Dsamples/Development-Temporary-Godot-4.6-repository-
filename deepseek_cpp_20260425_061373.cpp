// genesis/engine/material_property_manager.cpp

// NOTE: The header material_property_manager.h must be extended with:
//   public:
//     void sync_entities(const std::vector<std::shared_ptr<BaseEntity>>& entities);
//   private:
//     std::unordered_map<uint64_t, std::weak_ptr<BaseEntity>> entity_map_;
// This is required to enable the manager to resolve entity IDs.
// The code below assumes these additions exist.

#include "genesis/engine/material_property_manager.h" // Corresponding header
#include "genesis/engine/entities/base_entity.h"      // BaseEntity
#include "genesis/engine/entities/soft_tissue_entity.h" // SoftTissueEntity
#include "genesis/engine/entities/fem_entity.h"       // FEMEntity
#include "genesis/engine/solvers/soft_tissue_solver.h" // Solver
#include "genesis/engine/entities/restoration_controller.h" // Restoration
#include "genesis/engine/entities/jiggle_mapper.h"    // Jiggle mapper
#include "genesis/engine/fresnel_material_modifier.h" // Fresnel
#include <algorithm>                                  // std::find_if

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Construction / Destruction
//------------------------------------------------------------------------------
MaterialPropertyManager::MaterialPropertyManager() = default;
MaterialPropertyManager::~MaterialPropertyManager() = default;

//------------------------------------------------------------------------------
// Setup of external components
//------------------------------------------------------------------------------
void MaterialPropertyManager::set_soft_tissue_solver(std::shared_ptr<SoftTissueSolver> solver) {
    solver_ = solver;
}
void MaterialPropertyManager::set_restoration_controller(std::shared_ptr<RestorationController> ctrl) {
    restoration_ctrl_ = ctrl;
}
void MaterialPropertyManager::set_jiggle_mapper(std::shared_ptr<JiggleMapper> mapper) {
    jiggle_mapper_ = mapper;
}
void MaterialPropertyManager::set_fresnel_modifier(std::shared_ptr<FresnelMaterialModifier> mod) {
    fresnel_modifier_ = mod;
}

//------------------------------------------------------------------------------
// Global properties
//------------------------------------------------------------------------------
void MaterialPropertyManager::set_global_properties(const MaterialPropertySet& props) {
    global_props_ = props;
    // Global props are not automatically applied to all entities;
    // entities that lack per‑entity settings will use these when apply_properties is called.
}
MaterialPropertySet MaterialPropertyManager::get_global_properties() const {
    return global_props_;
}

//------------------------------------------------------------------------------
// Per‑entity property setting
//------------------------------------------------------------------------------
void MaterialPropertyManager::set_entity_properties(uint64_t entity_id, const MaterialPropertySet& props) {
    entity_props_[entity_id] = props;               // Store override
    apply_to_entity(entity_id, props);              // Immediately apply

    // Notify callback
    if (change_callback_) {
        change_callback_(entity_id, props);
    }
}

MaterialPropertySet MaterialPropertyManager::get_entity_properties(uint64_t entity_id) const {
    auto it = entity_props_.find(entity_id);
    if (it != entity_props_.end()) {
        return it->second;                          // Return stored per‑entity props
    }
    return global_props_;                           // Fallback to global
}

//------------------------------------------------------------------------------
// Individual setters (convenience)
//------------------------------------------------------------------------------
void MaterialPropertyManager::set_stiffness(uint64_t entity_id, double value) {
    auto props = get_entity_properties(entity_id);
    props.stiffness = value;
    set_entity_properties(entity_id, props);
}
void MaterialPropertyManager::set_damping(uint64_t entity_id, double value) {
    auto props = get_entity_properties(entity_id);
    props.damping = value;
    set_entity_properties(entity_id, props);
}
void MaterialPropertyManager::set_restoration(uint64_t entity_id, double value) {
    auto props = get_entity_properties(entity_id);
    props.restoration = value;
    set_entity_properties(entity_id, props);
}
void MaterialPropertyManager::set_jiggle(uint64_t entity_id, double value) {
    auto props = get_entity_properties(entity_id);
    props.jiggle = value;
    set_entity_properties(entity_id, props);
}
void MaterialPropertyManager::set_hardness(uint64_t entity_id, double value) {
    auto props = get_entity_properties(entity_id);
    props.hardness = value;
    set_entity_properties(entity_id, props);
}

//------------------------------------------------------------------------------
// Sync entity map – call once per frame with all scene entities
//------------------------------------------------------------------------------
void MaterialPropertyManager::sync_entities(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    entity_map_.clear();                            // Rebuild map
    for (auto& e : entities) {
        if (e) {
            entity_map_[e->id()] = e;               // Store weak pointer
        }
    }
}

//------------------------------------------------------------------------------
// Apply all pending property changes (can be called after batch edits)
//------------------------------------------------------------------------------
void MaterialPropertyManager::apply_properties() {
    // Apply global properties to all registered entities that lack per‑entity overrides
    for (auto& [id, weak_ent] : entity_map_) {
        if (auto ent = weak_ent.lock()) {
            MaterialPropertySet props = global_props_;
            auto it = entity_props_.find(id);
            if (it != entity_props_.end()) {
                props = it->second;                 // Use per‑entity if present
            }
            apply_to_entity(id, props);
        }
    }
}

//------------------------------------------------------------------------------
// Callback hook
//------------------------------------------------------------------------------
void MaterialPropertyManager::set_change_callback(
    std::function<void(uint64_t, const MaterialPropertySet&)> callback) {
    change_callback_ = callback;
}

//==============================================================================
// Internal application methods
//==============================================================================

void MaterialPropertyManager::apply_to_entity(uint64_t id, const MaterialPropertySet& props) {
    // Locate the entity
    std::shared_ptr<BaseEntity> entity;
    auto it = entity_map_.find(id);
    if (it != entity_map_.end()) {
        entity = it->second.lock();
    }
    if (!entity) return;                            // Entity not found (may have been destroyed)

    // If entity is SoftTissueEntity, directly set its soft config
    if (auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity)) {
        SoftTissueConfig cfg = soft->soft_config(); // Get current config
        cfg.youngs_modulus = props.stiffness;
        cfg.viscosity = props.damping;
        cfg.restoration_stiffness = props.restoration;
        // Map hardness to yield stress? hardness 0‑1 → yield between 1e3 and 1e5
        double yield = 1e3 + props.hardness * (1e5 - 1e3);
        cfg.yield_stress = yield;
        soft->set_soft_config(cfg);                 // Apply
    }
    // If generic FEM, set its FEM config (limited)
    else if (auto fem = std::dynamic_pointer_cast<FEMEntity>(entity)) {
        FEMConfig cfg = fem->fem_config();
        cfg.youngs_modulus = props.stiffness;
        // Note: FEMConfig doesn't have damping/restoration, so we skip those.
        fem->set_fem_config(cfg);
    }

    // Propagate to attached components
    apply_to_solver(id, props);
    apply_to_restoration(id, props);
    apply_to_jiggle(id, props);
    apply_to_fresnel(id, props);
}

void MaterialPropertyManager::apply_to_solver(uint64_t id, const MaterialPropertySet& props) {
    auto solver = solver_.lock();
    if (!solver) return;
    // The solver operates on entities during its step; we can set per‑entity config
    // via its set_soft_config? Actually, SoftTissueSolver only has a global config.
    // We could extend it with per‑entity overrides, but for now we rely on the entity's own config.
    // We do nothing here because the entity already carries its parameters.
    // If the solver later supports per‑entity overrides, this is the hook.
}

void MaterialPropertyManager::apply_to_restoration(uint64_t id, const MaterialPropertySet& props) {
    auto rc = restoration_ctrl_.lock();
    if (!rc) return;
    RestorationConfig cfg = rc->config();
    cfg.stiffness = props.restoration;              // map restoration to stiffness
    cfg.damping = props.damping;                    // damping influences return speed
    rc->set_config(cfg);
}

void MaterialPropertyManager::apply_to_jiggle(uint64_t id, const MaterialPropertySet& props) {
    auto mapper = jiggle_mapper_.lock();
    if (!mapper) return;
    // The jiggle mapper uses per‑bone parameters; we set global parameters
    BoneJiggleParams params;
    params.stiffness = props.jiggle * 10.0;         // Convert jiggle (0‑1? originally stiffness) to actual stiffness
    params.damping = props.damping;                 // Use damping as base
    params.influence = 1.0;                         // full effect
    mapper->set_global_params(params);
}

void MaterialPropertyManager::apply_to_fresnel(uint64_t id, const MaterialPropertySet& props) {
    auto fresnel = fresnel_modifier_.lock();
    if (!fresnel) return;
    FresnelConfig cfg = fresnel->config();
    // Update target/base values
    cfg.target_stiffness = props.stiffness;
    cfg.target_damping = props.damping;
    cfg.target_restoration = props.restoration;
    cfg.target_hardness = props.hardness;
    fresnel->set_config(cfg);
    fresnel->compute_fresnel(true);                 // Recompute and apply
}

} // namespace engine
} // namespace genesis