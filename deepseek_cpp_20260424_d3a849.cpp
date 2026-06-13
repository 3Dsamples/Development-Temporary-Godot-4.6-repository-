// genesis/engine/entities/soft_tissue_entity.h
#pragma once

//------------------------------------------------------------------------------
// SoftTissueEntity – extends FEMEntity with flesh-like material behavior,
// passive restoration, and real‑time interaction (poke, punch, bend).
// Uses a co‑rotated FEM formulation with additional viscoelasticity and
// shape‑memory layers to simulate skin, muscle, buttocks, breasts, etc.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/fem_entity.h"     // Base FEM entity
#include "genesis/datatypes.h"                     // Vector3, Matrix3r, real
#include <vector>                                  // std::vector
#include <memory>                                  // std::shared_ptr
#include <functional>                              // std::function for callbacks

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Material model for soft tissue (non‑linear, anisotropic, viscoelastic)
//------------------------------------------------------------------------------
enum class SoftTissueMaterial : uint8_t {
    SKIN = 0,            // Epidermal/dermal layer, stiff under compression
    FAT = 1,             // Subcutaneous fat, low stiffness, high damping
    MUSCLE = 2,          // Active/passive muscle, transversely isotropic
    BREAST_TISSUE = 3,   // Fibroglandular + adipose, nonlinear
    CARTILAGE = 4        // Stiff, nearly incompressible
};

//------------------------------------------------------------------------------
// Configuration for soft tissue behaviour
//------------------------------------------------------------------------------
struct SoftTissueConfig {
    SoftTissueMaterial material = SoftTissueMaterial::MUSCLE;
    
    // Elastic properties (small strain)
    double youngs_modulus = 5e4;       // Pa (flesh ~1e4 – 1e6)
    double poisson_ratio = 0.45;       // near incompressible
    
    // Viscoelasticity (linear Maxwell model)
    double viscosity = 500.0;          // Pa·s, damping coefficient
    double relaxation_time = 0.1;      // seconds, exponential decay of stress
    
    // Restoration / shape memory (passive return to rest)
    double restoration_stiffness = 200.0; // penalty force toward rest (N/m)
    double restoration_damping = 10.0;    // damping toward rest
    
    // Bending resistance (extra stiffness against bending)
    double bending_stiffness = 100.0;     // N·m/rad (additional to material)
    
    // Plasticity / permanent deformation (when stress exceeds yield)
    double yield_stress = 2e4;         // Pa, onset of irreversible change
    double plastic_hardening = 5e3;    // Pa, hardening slope
    
    // Control of active muscle contraction (0 = passive, 1 = fully activated)
    double activation_level = 0.0;     // [0,1]
    double max_contractile_stress = 1e5; // Pa, additional stress when activated
};

//------------------------------------------------------------------------------
// Deformation event (poke, punch, etc.) – passed to callback or recorded
//------------------------------------------------------------------------------
struct DeformationEvent {
    enum class Type { POKE, PUNCH, GRAB, BEND, RELEASE };
    Type type;
    datatypes::Vector3 world_point;     // location on surface
    datatypes::Vector3 direction;       // inward direction (poke/punch)
    double force_magnitude;             // N
    double duration;                    // seconds (0 = instantaneous)
    double timestamp;                   // simulation time
};

//------------------------------------------------------------------------------
// SoftTissueEntity – handles flesh deformation, restoration, and bending
//------------------------------------------------------------------------------
class SoftTissueEntity : public FEMEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    SoftTissueEntity();
    explicit SoftTissueEntity(const std::string& name);
    virtual ~SoftTissueEntity();

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_soft_config(const SoftTissueConfig& config);
    const SoftTissueConfig& soft_config() const { return soft_config_; }
    
    //----------------------------------------------------------------------
    // Deformation control (poke, punch, bend, release)
    //----------------------------------------------------------------------
    // Apply a point force at a surface location (poke/punch)
    void apply_poke(const datatypes::Vector3& world_point,
                    const datatypes::Vector3& direction,
                    double force_magnitude,
                    double duration = 0.0);
    
    // Apply a bending moment (e.g., folding)
    void apply_bend(const datatypes::Vector3& axis_origin,
                    const datatypes::Vector3& axis_direction,
                    double moment_magnitude);
    
    // Release all active external forces (simulate letting go)
    void release_deformation();
    
    // Register callback for deformation events (for haptics, sound, etc.)
    void set_deformation_callback(std::function<void(const DeformationEvent&)> callback);
    
    //----------------------------------------------------------------------
    // Overrides from FEMEntity / BaseEntity
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;       // Adds viscoelasticity & restoration
    virtual void reset() override;
    
    //----------------------------------------------------------------------
    // Query current deformation state
    //----------------------------------------------------------------------
    double get_max_displacement() const;              // Max distance from rest shape
    double get_strain_energy() const;                 // Total elastic energy (J)
    
    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "SoftTissueEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    SoftTissueConfig soft_config_;                    // Tissue material parameters
    
    // Active deformation forces (applied per node during poke/punch)
    std::vector<datatypes::Vector3> external_poke_forces_; // accumulated per node
    
    // Rest positions for restoration (copy of original rest positions)
    std::vector<datatypes::Vector3> original_rest_positions_;
    
    // Per‑element viscous strain (for viscoelasticity)
    std::vector<datatypes::Matrix3r> viscous_strain_;  // F_v (deformation gradient viscous part)
    
    // Bending deformation state (not used in small strain, but for bending constraint)
    bool bending_active_ = false;
    datatypes::Vector3 bend_axis_origin_;
    datatypes::Vector3 bend_axis_direction_;
    double bend_moment_ = 0.0;
    
    // Callback for external events
    std::function<void(const DeformationEvent&)> deformation_callback_;
    
    // Timers for transient forces
    double poke_remaining_time_ = 0.0;
    datatypes::Vector3 poke_point_;
    datatypes::Vector3 poke_direction_;
    double poke_force_ = 0.0;
    
    // Internal methods
    void compute_viscoelastic_stress(double dt);      // Update viscous strain & stress
    void compute_restoration_forces(double dt);       // Passive return to rest shape
    void compute_bending_forces(double dt);           // Apply bending moment as nodal forces
    void distribute_poke_force();                     // Spread poke force to nearby nodes
    void update_viscous_strain(double dt);            // Evolve viscous strain
};

} // namespace engine
} // namespace genesis