// genesis/engine/interaction_pipeline.cpp
#include "genesis/engine/interaction_pipeline.h"    // Corresponding header
#include "genesis/engine/scene.h"                   // Scene for entity iteration
#include "genesis/engine/interaction_manager.h"     // InteractionManager
#include "genesis/engine/solvers/soft_tissue_solver.h" // SoftTissueSolver
#include "genesis/engine/entities/restoration_controller.h" // RestorationController
#include "genesis/engine/entities/jiggle_mapper.h"  // JiggleMapper
#include "genesis/engine/fresnel_material_modifier.h" // FresnelMaterialModifier
#include "genesis/engine/material_property_manager.h" // MaterialPropertyManager
#include "genesis/engine/entities/base_entity.h"    // BaseEntity for scene sync
#include <sstream>                                  // std::ostringstream for status

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Construction / Destruction
//------------------------------------------------------------------------------
InteractionPipeline::InteractionPipeline() = default;  // Default constructor
InteractionPipeline::~InteractionPipeline() = default; // Destructor

//------------------------------------------------------------------------------
// Set components
//------------------------------------------------------------------------------
void InteractionPipeline::set_interaction_manager(std::shared_ptr<InteractionManager> mgr) {
    interaction_mgr_ = mgr;                         // Store interaction manager
}
void InteractionPipeline::set_soft_tissue_solver(std::shared_ptr<SoftTissueSolver> solver) {
    soft_tissue_solver_ = solver;                   // Store soft tissue solver
}
void InteractionPipeline::set_restoration_controller(std::shared_ptr<RestorationController> ctrl) {
    restoration_ctrl_ = ctrl;                       // Store restoration controller
}
void InteractionPipeline::set_jiggle_mapper(std::shared_ptr<JiggleMapper> mapper) {
    jiggle_mapper_ = mapper;                        // Store jiggle mapper
}
void InteractionPipeline::set_fresnel_modifier(std::shared_ptr<FresnelMaterialModifier> mod) {
    fresnel_modifier_ = mod;                        // Store Fresnel modifier
}
void InteractionPipeline::set_material_property_manager(std::shared_ptr<MaterialPropertyManager> mgr) {
    material_property_mgr_ = mgr;                   // Store material property manager
}

//------------------------------------------------------------------------------
// Main update – orchestrate all subsystems
//------------------------------------------------------------------------------
void InteractionPipeline::update(double dt, Scene& scene) {
    // Sync entity lists across managers that need them
    sync_managers_with_scene(scene);                // Provide entity map to material manager

    // 1. Process interactions (poke, punch, grab, bend)
    if (interaction_mgr_) {
        interaction_mgr_->update(dt, scene);        // Apply forces to entities

        // Notify restoration controller about active interactions
        if (restoration_ctrl_) {
            bool any_active = interaction_mgr_->active_interaction_count() > 0;
            restoration_ctrl_->notify_interaction_active(any_active); // Soften restoration during interaction
        }
    }

    // 2. Run soft tissue solver (viscoelasticity, bending, plasticity)
    if (soft_tissue_solver_) {
        // Collect all entities from the scene to pass to solver
        std::vector<std::shared_ptr<BaseEntity>> entities;
        for (size_t i = 0; i < scene.entities().size(); ++i) {
            entities.push_back(scene.entities()[i]); // Copy shared pointers
        }
        soft_tissue_solver_->step(dt, entities);    // Apply material forces
    }

    // 3. Apply restoration (balloon‑like return to rest shape)
    if (restoration_ctrl_) {
        restoration_ctrl_->update(dt);              // Compute and apply restoring forces
    }

    // 4. Jiggle mapper (secondary bone‑driven oscillation)
    if (jiggle_mapper_ && jiggle_mapper_->is_bound()) {
        jiggle_mapper_->update(dt);                 // Advance spring simulation, write vertices
    }

    // 5. Fresnel material modifier (update vertex colours, adjust per‑element stiffness)
    if (fresnel_modifier_ && fresnel_modifier_->is_bound()) {
        fresnel_modifier_->compute_fresnel(true);   // Recompute factors and apply to entity
    }

    // 6. Material property manager – propagate any outstanding changes
    if (material_property_mgr_) {
        material_property_mgr_->apply_properties(); // Push to entities and controllers
    }

    // 7. Build a status string and invoke the hook
    if (on_status_update) {
        std::ostringstream status;
        status << "Pipeline:";
        if (interaction_mgr_) status << " interactions=" << interaction_mgr_->active_interaction_count();
        if (restoration_ctrl_) status << " restoring=" << (restoration_ctrl_->is_attached() ? "yes" : "no");
        if (jiggle_mapper_) status << " jiggle=" << (jiggle_mapper_->is_bound() ? "active" : "idle");
        if (fresnel_modifier_) status << " fresnel=" << (fresnel_modifier_->is_bound() ? "active" : "idle");
        on_status_update(status.str());             // Call the UI hook
    }
}

//------------------------------------------------------------------------------
// Helper: sync entity lists with managers that need them
//------------------------------------------------------------------------------
void InteractionPipeline::sync_managers_with_scene(Scene& scene) {
    if (material_property_mgr_) {
        // Material manager needs a map of all entities to apply property overrides
        material_property_mgr_->sync_entities(scene.entities());
    }
    // Other managers (restoration, jiggle) are bound directly to entities,
    // so they don't need a scene-wide sync.
}

} // namespace engine
} // namespace genesis