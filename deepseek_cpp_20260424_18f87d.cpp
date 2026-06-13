// genesis/engine/entities/hybrid_entity.h

#pragma once

//------------------------------------------------------------------------------
// Hybrid Entity class - combines multiple physics representations (e.g., MPM + FEM)
// Useful for coupling particle methods with mesh-based solids, or embedding rigid
// bodies within deformable materials.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/base_entity.h"   // Base class: BaseEntity
#include "genesis/datatypes.h"                      // Vector3, Matrix3, Quat, AABB
#include <vector>                                   // std::vector for particles and nodes
#include <memory>                                   // std::shared_ptr, std::weak_ptr
#include <unordered_map>                            // std::unordered_map for coupling pairs
#include <string>                                   // std::string

namespace genesis {
namespace engine {

// Forward declarations of component entity types
class MPMEntity;
class SPHEntity;
class FEMEntity;
class RigidEntity;
class PBDEntity;

//------------------------------------------------------------------------------
// Coupling type between primary and secondary representations
//------------------------------------------------------------------------------
enum class HybridCouplingType : uint8_t {
    NONE = 0,                                       // No coupling (independent)
    ONE_WAY = 1,                                    // Primary affects secondary (e.g., MPM drives FEM)
    TWO_WAY = 2,                                    // Mutual interaction
    KINEMATIC = 3                                   // Secondary follows primary kinematically
};

//------------------------------------------------------------------------------
// Hybrid entity configuration
//------------------------------------------------------------------------------
struct HybridConfig {
    // Primary and secondary entity types (set by factory or manually)
    std::string primary_type;                       // "MPM", "SPH", "FEM", "Rigid", etc.
    std::string secondary_type;                     // Secondary representation type
    
    // Coupling settings
    HybridCouplingType coupling_type = HybridCouplingType::TWO_WAY;
    double coupling_stiffness = 1000.0;             // Penalty stiffness for constraints
    double coupling_damping = 10.0;                 // Damping coefficient for coupling forces
    double coupling_distance = 0.05;                // Maximum distance for coupling pairs
    
    // Transfer settings (for MPM <-> FEM)
    bool transfer_velocity = true;                  // Transfer velocity between representations
    bool transfer_stress = false;                   // Transfer stress/deformation state
    double transfer_relaxation = 0.5;               // Relaxation factor for state blending
    
    // Visualization
    bool visualize_primary = true;                  // Show primary representation
    bool visualize_secondary = false;               // Show secondary representation
};

//------------------------------------------------------------------------------
// Coupling pair: maps a particle/node from primary to a set in secondary
//------------------------------------------------------------------------------
struct CouplingPair {
    uint32_t primary_index;                         // Index in primary entity
    std::vector<uint32_t> secondary_indices;        // Indices in secondary entity (e.g., FEM nodes)
    std::vector<double> weights;                    // Interpolation weights (sum to 1)
    double distance = 0.0;                          // Initial distance for penalty method
};

//------------------------------------------------------------------------------
// HybridEntity class - manages two coupled physics representations
//------------------------------------------------------------------------------
class HybridEntity : public BaseEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    HybridEntity();                                 // Default constructor
    explicit HybridEntity(const std::string& name); // Named constructor
    virtual ~HybridEntity();                        // Destructor

    //----------------------------------------------------------------------
    // Component management
    //----------------------------------------------------------------------
    // Set the primary entity (takes ownership or shares)
    void set_primary_entity(std::shared_ptr<BaseEntity> entity);
    // Set the secondary entity
    void set_secondary_entity(std::shared_ptr<BaseEntity> entity);
    
    // Access components
    std::shared_ptr<BaseEntity> primary_entity() const { return primary_; }
    std::shared_ptr<BaseEntity> secondary_entity() const { return secondary_; }
    
    // Check types
    bool has_primary() const { return primary_ != nullptr; }
    bool has_secondary() const { return secondary_ != nullptr; }
    
    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_hybrid_config(const HybridConfig& config);
    const HybridConfig& hybrid_config() const { return config_; }
    
    //----------------------------------------------------------------------
    // Coupling management
    //----------------------------------------------------------------------
    // Build coupling pairs automatically (e.g., nearest neighbors or barycentric mapping)
    void build_coupling_pairs();
    // Manually add a coupling pair
    void add_coupling_pair(const CouplingPair& pair);
    // Clear all coupling pairs
    void clear_coupling_pairs();
    // Get coupling pairs (for debugging)
    const std::vector<CouplingPair>& coupling_pairs() const { return coupling_pairs_; }
    
    //----------------------------------------------------------------------
    // Synchronization and transfer
    //----------------------------------------------------------------------
    // Transfer state from primary to secondary (one-way)
    void transfer_primary_to_secondary();
    // Transfer state from secondary to primary (one-way)
    void transfer_secondary_to_primary();
    // Apply coupling forces (two-way interaction)
    void apply_coupling_forces(double dt);
    // Synchronize transforms (e.g., for rigid bodies)
    void synchronize_transforms();
    
    //----------------------------------------------------------------------
    // Overrides from BaseEntity
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;     // Integrate both components and apply coupling
    virtual void integrate_velocity(double dt) override;
    virtual void integrate_position(double dt) override;
    virtual void reset() override;
    virtual datatypes::AABB world_aabb() const override; // Combined AABB
    
    // Force and impulse forwarding
    virtual void apply_force(const datatypes::Vector3& force) override;
    virtual void apply_force(const datatypes::Vector3& force, const datatypes::Vector3& world_point) override;
    virtual void apply_torque(const datatypes::Vector3& torque) override;
    virtual void apply_impulse(const datatypes::Vector3& impulse) override;
    virtual void apply_impulse(const datatypes::Vector3& impulse, const datatypes::Vector3& world_point) override;
    virtual void clear_forces() override;
    
    // Transform override (propagate to both components)
    virtual void set_transform(const datatypes::Transformr& t) override;
    
    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "HybridEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    HybridConfig config_;                           // Hybrid configuration
    std::shared_ptr<BaseEntity> primary_;           // Primary physics representation
    std::shared_ptr<BaseEntity> secondary_;         // Secondary physics representation
    
    std::vector<CouplingPair> coupling_pairs_;      // Mapping between primary and secondary
    
    // Helper methods
    void build_mpm_fem_coupling();                  // Coupling between MPM particles and FEM nodes
    void build_sph_fem_coupling();                  // Coupling between SPH particles and FEM nodes
    void build_pbd_fem_coupling();                  // Coupling between PBD particles and FEM nodes
    void build_rigid_particle_coupling();           // Coupling between rigid body and particles
    
    // Transfer helpers
    void transfer_mpm_to_fem();                     // MPM particle state to FEM nodes
    void transfer_fem_to_mpm();                     // FEM node state to MPM particles
    void transfer_sph_to_fem();                     // SPH to FEM
    void transfer_fem_to_sph();                     // FEM to SPH
    
    // Force application helpers
    void apply_coupling_force_mpm_fem(double dt);   // Penalty forces between MPM and FEM
    void apply_coupling_force_sph_fem(double dt);   // Penalty forces between SPH and FEM
    void apply_coupling_force_rigid_particles(double dt); // Rigid body to particles
};

} // namespace engine
} // namespace genesis