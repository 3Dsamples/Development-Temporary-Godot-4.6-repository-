// genesis/engine/material_property_manager.h
#pragma once

//------------------------------------------------------------------------------
// MaterialPropertyManager – provides a unified interface to set and get
// material parameters (stiffness, damping, restoration, jiggle, hardness) for
// all deformable entities in a scene.  It can be driven by external sources
// (Fresnel, bone weights, UI sliders) and pushes updates to the appropriate
// entity/solver.  All UI hooks are left as callbacks.
//------------------------------------------------------------------------------

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace genesis {
namespace engine {

class BaseEntity;
class SoftTissueEntity;
class FEMEntity;
class SoftTissueSolver;
class RestorationController;
class JiggleMapper;
class FresnelMaterialModifier;

//------------------------------------------------------------------------------
// MaterialPropertySet – per‑entity property snapshot
//------------------------------------------------------------------------------
struct MaterialPropertySet {
    double stiffness   = 500.0;    // Young's modulus equivalent (Pa)
    double damping     = 50.0;     // Viscous damping (Pa·s)
    double restoration = 200.0;    // Restoration stiffness (N/m)
    double jiggle      = 20.0;     // Jiggle damping inversely (higher = stiffer jiggle)
    double hardness    = 0.5;      // 0‑1 plastic yield multiplier
};

//------------------------------------------------------------------------------
// Manager class
//------------------------------------------------------------------------------
class MaterialPropertyManager {
public:
    MaterialPropertyManager();
    ~MaterialPropertyManager();

    //----------------------------------------------------------------------
    // Initialisation
    //----------------------------------------------------------------------
    // Provide references to the solvers and controllers (can be nullptr if
    // not used).  The manager only holds weak pointers.
    void set_soft_tissue_solver(std::shared_ptr<SoftTissueSolver> solver);
    void set_restoration_controller(std::shared_ptr<RestorationController> controller);
    void set_jiggle_mapper(std::shared_ptr<JiggleMapper> mapper);
    void set_fresnel_modifier(std::shared_ptr<FresnelMaterialModifier> modifier);

    //----------------------------------------------------------------------
    // Per‑entity property access
    //----------------------------------------------------------------------
    // Set target material properties for a specific entity (by ID).
    // If the entity is a SoftTissueEntity, the values are applied directly;
    // otherwise they are stored for the solver to query.
    void set_entity_properties(uint64_t entity_id, const MaterialPropertySet& props);

    // Get the currently active material properties of an entity (reads from
    // entity or from cached target).
    MaterialPropertySet get_entity_properties(uint64_t entity_id) const;

    //----------------------------------------------------------------------
    // Global override (applies to all entities unless per‑entity set)
    //----------------------------------------------------------------------
    void set_global_properties(const MaterialPropertySet& props);
    MaterialPropertySet get_global_properties() const;

    //----------------------------------------------------------------------
    // Push all property changes to the respective solvers/controllers
    // (called automatically when properties change, or can be invoked
    // manually after batch edits).
    //----------------------------------------------------------------------
    void apply_properties();

    //----------------------------------------------------------------------
    // Hook for future UI to be notified when any property changes
    //----------------------------------------------------------------------
    void set_change_callback(std::function<void(uint64_t entity_id, const MaterialPropertySet&)> callback);

    //----------------------------------------------------------------------
    // Convenience methods to set individual parameters of an entity
    //----------------------------------------------------------------------
    void set_stiffness(uint64_t entity_id, double value);
    void set_damping(uint64_t entity_id, double value);
    void set_restoration(uint64_t entity_id, double value);
    void set_jiggle(uint64_t entity_id, double value);
    void set_hardness(uint64_t entity_id, double value);

private:
    // Global default properties
    MaterialPropertySet global_props_;

    // Per‑entity overrides
    std::unordered_map<uint64_t, MaterialPropertySet> entity_props_;

    // Weak pointers to external components
    std::weak_ptr<SoftTissueSolver> solver_;
    std::weak_ptr<RestorationController> restoration_ctrl_;
    std::weak_ptr<JiggleMapper> jiggle_mapper_;
    std::weak_ptr<FresnelMaterialModifier> fresnel_modifier_;

    // Callback for UI
    std::function<void(uint64_t, const MaterialPropertySet&)> change_callback_;

    // Apply to a single entity (internal)
    void apply_to_entity(uint64_t id, const MaterialPropertySet& props);
    void apply_to_solver(uint64_t id, const MaterialPropertySet& props);
    void apply_to_restoration(uint64_t id, const MaterialPropertySet& props);
    void apply_to_jiggle(uint64_t id, const MaterialPropertySet& props);
    void apply_to_fresnel(uint64_t id, const MaterialPropertySet& props);
};

} // namespace engine
} // namespace genesis