// genesis/engine/entities/restoration_controller.h
#pragma once

//------------------------------------------------------------------------------
// RestorationController – attached to a FEMEntity (or SoftTissueEntity) to
// monitor deformation and apply passive restoring forces whenever the entity
// is displaced from its reference shape.  The controller detects when the
// entity is no longer under active interaction and automatically increases
// restoration to return it to the rest state, then idles.
//------------------------------------------------------------------------------

#include "genesis/datatypes.h"                     // Vector3, real
#include <memory>                                  // std::shared_ptr
#include <functional>                              // std::function for hook

namespace genesis {
namespace engine {

// Forward declarations
class FEMEntity;
class BaseEntity;

//------------------------------------------------------------------------------
// Configuration for the restoration controller
//------------------------------------------------------------------------------
struct RestorationConfig {
    // Spring stiffness (N/m) used while actively returning to shape
    double stiffness = 2000.0;
    // Damping coefficient (N·s/m) to avoid oscillation
    double damping = 100.0;
    
    // Maximum force per node to avoid instability
    double max_force_per_node = 500.0;
    
    // Threshold of average displacement below which the entity is considered
    // "at rest" and forces are released (m)
    double rest_threshold = 0.001;
    
    // Time (seconds) the controller waits after external forces cease before
    // fully engaging restoration (allows transient user touches)
    double activation_delay = 0.1;
    
    // If true, the controller auto‑detects when active interaction is happening
    // and reduces restoration; if false, it always applies full restoration.
    bool auto_modulate = true;
};

//------------------------------------------------------------------------------
// RestorationController class
//------------------------------------------------------------------------------
class RestorationController {
public:
    RestorationController();
    ~RestorationController();

    //----------------------------------------------------------------------
    // Binding
    //----------------------------------------------------------------------
    // Attach to a specific FEM entity (the controller will store a weak pointer)
    void attach(std::shared_ptr<FEMEntity> entity);
    void detach();
    bool is_attached() const;

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_config(const RestorationConfig& config);
    const RestorationConfig& config() const;

    //----------------------------------------------------------------------
    // Update (called every simulation step, e.g., from Scene or solver)
    //----------------------------------------------------------------------
    // dt = time step
    void update(double dt);

    //----------------------------------------------------------------------
    // Force external interaction notification – tells the controller that an
    // active force (poke/punch/grab) is currently acting on the entity.
    // The controller can optionally reduce its own restoring force to allow
    // the deformation.
    //----------------------------------------------------------------------
    void notify_interaction_active(bool active);

    //----------------------------------------------------------------------
    // Hook for future UI integration
    //----------------------------------------------------------------------
    // Callback receives the current average restoration force being applied,
    // or other status information.
    void set_status_callback(std::function<void(double avg_force, bool restoring)> callback);

private:
    std::weak_ptr<FEMEntity> entity_;              // The deformable entity
    RestorationConfig config_;

    // Internal state
    double time_since_last_interaction_ = 1.0;     // Accumulator for activation delay
    bool interaction_active_ = false;              // External interaction flag
    bool currently_restoring_ = false;             // Whether we are actively applying forces

    // Reference positions (copied from entity at attach time)
    std::vector<datatypes::Vector3> reference_positions_;

    // Hook
    std::function<void(double, bool)> status_callback_;

    // Helper
    void apply_restoring_forces();
    double compute_average_displacement() const;
};

} // namespace engine
} // namespace genesis