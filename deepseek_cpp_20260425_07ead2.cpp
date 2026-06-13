// genesis/engine/interaction_pipeline.h
#pragma once

//------------------------------------------------------------------------------
// InteractionPipeline – central coordinator that runs every simulation step.
// It wires together:
//   - InteractionManager (poke/punch/grab)
//   - SoftTissueSolver (viscoelastic restoration and bending)
//   - RestorationController (balloon‑like return to shape)
//   - JiggleMapper (bone‑driven secondary motion)
//   - FresnelMaterialModifier (surface‑driven stiffness changes)
//   - MaterialPropertyManager (global/per‑entity parameter control)
// Each component is optional (set to nullptr to disable).
// The pipeline provides a single update() that is called from the Scene.
// All UI hooks are left as public member function pointers.
//------------------------------------------------------------------------------

#include <memory>
#include <functional>

namespace genesis {
namespace engine {

class Scene;
class InteractionManager;
class SoftTissueSolver;
class RestorationController;
class JiggleMapper;
class FresnelMaterialModifier;
class MaterialPropertyManager;

class InteractionPipeline {
public:
    InteractionPipeline();
    ~InteractionPipeline();

    //----------------------------------------------------------------------
    // Set the components (use these before calling update)
    //----------------------------------------------------------------------
    void set_interaction_manager(std::shared_ptr<InteractionManager> mgr);
    void set_soft_tissue_solver(std::shared_ptr<SoftTissueSolver> solver);
    void set_restoration_controller(std::shared_ptr<RestorationController> ctrl);
    void set_jiggle_mapper(std::shared_ptr<JiggleMapper> mapper);
    void set_fresnel_modifier(std::shared_ptr<FresnelMaterialModifier> mod);
    void set_material_property_manager(std::shared_ptr<MaterialPropertyManager> mgr);

    //----------------------------------------------------------------------
    // Main update – call every simulation frame (e.g., from Scene::step)
    //----------------------------------------------------------------------
    void update(double dt, Scene& scene);

    //----------------------------------------------------------------------
    // Hook for future UI: called after each update with a status string
    //----------------------------------------------------------------------
    std::function<void(const std::string&)> on_status_update;

private:
    std::shared_ptr<InteractionManager> interaction_mgr_;
    std::shared_ptr<SoftTissueSolver> soft_tissue_solver_;
    std::shared_ptr<RestorationController> restoration_ctrl_;
    std::shared_ptr<JiggleMapper> jiggle_mapper_;
    std::shared_ptr<FresnelMaterialModifier> fresnel_modifier_;
    std::shared_ptr<MaterialPropertyManager> material_property_mgr_;

    // Helpers
    void sync_managers_with_scene(Scene& scene);
};

} // namespace engine
} // namespace genesis