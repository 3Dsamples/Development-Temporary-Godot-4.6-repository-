// genesis/engine/entities/pbd_entity.h

#pragma once

//------------------------------------------------------------------------------
// PBD Entity class - Position Based Dynamics particle system.
// Manages particles, constraints, and provides interface for PBD solver.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/particle_entity.h" // Base class: ParticleEntity
#include "genesis/datatypes.h"                      // Vector3, real, etc.
#include <vector>                                   // std::vector
#include <memory>                                   // std::shared_ptr
#include <string>                                   // std::string

namespace genesis {
namespace engine {

// Forward declaration of constraint types (defined in pbd_solver.h)
class PBDConstraint;
class DistanceConstraint;
class BendingConstraint;
class VolumeConstraint;
class ShapeMatchingConstraint;
class PinConstraint;
class AttachmentConstraint;

//------------------------------------------------------------------------------
// PBD Configuration
//------------------------------------------------------------------------------
struct PBDConfig {
    // Solver parameters (defaults, can be overridden by solver)
    int iterations = 5;                             // Constraint solver iterations
    double velocity_damping = 0.0;                  // Global velocity damping
    double max_velocity = 100.0;                    // Maximum allowed velocity magnitude
    bool enable_continuous_collision = false;       // Use CCD for fast moving particles
    
    // Constraint stiffness defaults
    double default_stiffness = 1.0;                 // Default stiffness for constraints
};

//------------------------------------------------------------------------------
// PBD Entity class - particle system with constraints for PBD simulation
//------------------------------------------------------------------------------
class PBDEntity : public ParticleEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    PBDEntity();                                    // Default constructor
    explicit PBDEntity(const std::string& name);    // Named constructor
    virtual ~PBDEntity();                           // Destructor

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_pbd_config(const PBDConfig& config);   // Set PBD configuration
    const PBDConfig& pbd_config() const { return pbd_config_; }

    //----------------------------------------------------------------------
    // Constraint management
    //----------------------------------------------------------------------
    // Add a distance constraint between two particles
    void add_distance_constraint(int idx0, int idx1, double rest_length = -1.0, double stiffness = 1.0);
    
    // Add a bending constraint (dihedral angle) between four particles
    void add_bending_constraint(int idx0, int idx1, int idx2, int idx3, double rest_angle = -1.0, double stiffness = 1.0);
    
    // Add a volume constraint for a tetrahedron
    void add_volume_constraint(int idx0, int idx1, int idx2, int idx3, double rest_volume = -1.0, double stiffness = 1.0);
    
    // Add a shape matching constraint for a cluster of particles
    void add_shape_matching_constraint(const std::vector<int>& indices, double stiffness = 1.0);
    
    // Pin a particle to a fixed world position
    void add_pin_constraint(int particle_idx, const datatypes::Vector3& fixed_position = datatypes::Vector3(0.0));
    
    // Attach a particle to another entity (e.g., rigid body)
    void add_attachment_constraint(int particle_idx, std::shared_ptr<BaseEntity> target, const datatypes::Vector3& local_offset);
    
    // Remove all constraints
    void clear_constraints();
    
    // Get all constraints (for solver)
    const std::vector<std::shared_ptr<PBDConstraint>>& constraints() const { return constraints_; }
    std::vector<std::shared_ptr<PBDConstraint>>& constraints() { return constraints_; }
    
    //----------------------------------------------------------------------
    // PBD-specific state (positions from previous step for velocity update)
    //----------------------------------------------------------------------
    const std::vector<datatypes::Vector3>& prev_positions() const { return prev_positions_; }
    std::vector<datatypes::Vector3>& prev_positions() { return prev_positions_; }
    void set_prev_positions(const std::vector<datatypes::Vector3>& prev);
    void store_prev_positions();                    // Copy current positions to prev_positions
    
    // Temporary delta array for constraint projection (cleared each iteration)
    const std::vector<datatypes::Vector3>& delta() const { return delta_; }
    std::vector<datatypes::Vector3>& delta() { return delta_; }
    void clear_delta();                             // Reset delta to zero
    
    // Inverse mass array (convenience access)
    std::vector<double> inverse_masses() const;
    void compute_inverse_masses();                  // Fill inv_mass_ array from masses_
    
    //----------------------------------------------------------------------
    // Overrides from ParticleEntity/BaseEntity
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;     // PBD uses custom integration, minimal here
    virtual void reset() override;
    
    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "PBDEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    PBDConfig pbd_config_;                          // PBD configuration
    
    std::vector<std::shared_ptr<PBDConstraint>> constraints_; // All constraints for this entity
    
    // PBD state arrays
    std::vector<datatypes::Vector3> prev_positions_; // Positions from previous step (for velocity update)
    std::vector<datatypes::Vector3> delta_;          // Temporary delta for constraint projection
    std::vector<double> inv_mass_;                   // Cached inverse masses
    
    // Helper to add constraint with automatic rest length/volume computation
    double compute_rest_length(int idx0, int idx1) const;
    double compute_rest_angle(int idx0, int idx1, int idx2, int idx3) const;
    double compute_rest_volume(int idx0, int idx1, int idx2, int idx3) const;
};

} // namespace engine
} // namespace genesis