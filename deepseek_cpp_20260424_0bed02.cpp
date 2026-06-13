// genesis/engine/entities/mpm_entity.h

#pragma once

//------------------------------------------------------------------------------
// MPM Entity class - represents a collection of Material Point Method particles.
// Contains particle positions, velocities, deformation gradients, and material properties.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/particle_entity.h" // Base class: ParticleEntity
#include "genesis/datatypes.h"                      // Vector3, Matrix3, real, etc.
#include <vector>                                   // std::vector for particle arrays
#include <string>                                   // std::string

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// MPM Material model types
//------------------------------------------------------------------------------
enum class MPMMaterialModel : uint8_t {
    NEO_HOOKEAN = 0,                                // Neo-Hookean hyperelastic
    FIXED_COROTATED = 1,                            // Fixed corotated (StVK with corotation)
    VON_MISES_PLASTIC = 2,                          // Elastoplastic with von Mises yield
    DRUCKER_PRAGER = 3,                             // Drucker-Prager (soil, sand)
    SAND = 4,                                       // Sand-specific model
    SNOW = 5,                                       // Snow model (Stomakhin et al.)
    LIQUID = 6,                                     // Weakly compressible liquid
    ELASTIC = 7                                     // Simple linear elastic
};

//------------------------------------------------------------------------------
// MPM Configuration
//------------------------------------------------------------------------------
struct MPMConfig {
    // Material parameters
    MPMMaterialModel material_model = MPMMaterialModel::NEO_HOOKEAN;
    double youngs_modulus = 1e6;                    // Young's modulus (Pa)
    double poisson_ratio = 0.3;                     // Poisson's ratio
    double density = 1000.0;                        // Rest density (kg/m³)
    
    // Plasticity (if applicable)
    double yield_stress = 1e6;                      // Yield stress (Pa)
    double hardening_modulus = 0.0;                 // Linear hardening modulus
    double friction_angle = 30.0 * M_PI / 180.0;    // For Drucker-Prager / sand (radians)
    double cohesion = 0.0;                          // Cohesion (Pa)
    
    // Numerical parameters
    double hardening_alpha = 0.0;                   // Plastic hardening factor
    double volume_correction_stiffness = 1000.0;    // For volume preservation
    bool enable_affine = true;                      // Use APIC affine matrix
};

//------------------------------------------------------------------------------
// MPM Entity class - container for MPM particles and their state
//------------------------------------------------------------------------------
class MPMEntity : public ParticleEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    MPMEntity();                                    // Default constructor
    explicit MPMEntity(const std::string& name);    // Named constructor
    virtual ~MPMEntity();                           // Destructor

    //----------------------------------------------------------------------
    // Particle management (overrides/extensions)
    //----------------------------------------------------------------------
    // Add particles with positions, velocities, masses, and volumes
    void add_particles(const std::vector<datatypes::Vector3>& positions,
                       const std::vector<datatypes::Vector3>& velocities,
                       const std::vector<double>& masses,
                       const std::vector<double>& volumes);
    
    // Remove particles by index mask
    void remove_particles(const std::vector<bool>& mask);
    
    // Particle volume access
    double particle_volume(size_t i) const;
    const std::vector<double>& particle_volumes() const { return volumes_; }
    void set_particle_volume(size_t i, double volume);
    
    //----------------------------------------------------------------------
    // Deformation state (per particle)
    //----------------------------------------------------------------------
    const std::vector<datatypes::Matrix3r>& deformation_gradients() const { return F_; }
    void set_deformation_gradients(const std::vector<datatypes::Matrix3r>& F);
    datatypes::Matrix3r deformation_gradient(size_t i) const;
    void set_deformation_gradient(size_t i, const datatypes::Matrix3r& F);
    
    // Plastic strain (if plasticity enabled)
    const std::vector<datatypes::Matrix3r>& plastic_strains() const { return Fp_; }
    void set_plastic_strains(const std::vector<datatypes::Matrix3r>& Fp);
    datatypes::Matrix3r plastic_strain(size_t i) const;
    void set_plastic_strain(size_t i, const datatypes::Matrix3r& Fp);
    
    // Affine velocity matrix (for APIC)
    const std::vector<datatypes::Matrix3r>& affine_velocities() const { return C_; }
    void set_affine_velocities(const std::vector<datatypes::Matrix3r>& C);
    datatypes::Matrix3r affine_velocity(size_t i) const;
    void set_affine_velocity(size_t i, const datatypes::Matrix3r& C);
    
    // Plastic Jacobian determinant (for volume tracking)
    const std::vector<double>& plastic_jacobians() const { return Jp_; }
    void set_plastic_jacobians(const std::vector<double>& Jp);
    double plastic_jacobian(size_t i) const;
    void set_plastic_jacobian(size_t i, double Jp);
    
    //----------------------------------------------------------------------
    // Bulk state operations (used by solver)
    //----------------------------------------------------------------------
    void set_particle_states(const std::vector<datatypes::Vector3>& positions,
                             const std::vector<datatypes::Vector3>& velocities,
                             const std::vector<datatypes::Matrix3r>& F,
                             const std::vector<datatypes::Matrix3r>& C);
    
    //----------------------------------------------------------------------
    // Material configuration
    //----------------------------------------------------------------------
    void set_mpm_config(const MPMConfig& config);
    const MPMConfig& mpm_config() const { return mpm_config_; }
    
    // Per-particle material override (optional)
    void set_particle_material(size_t i, uint32_t material_id);
    uint32_t particle_material(size_t i) const;
    const std::vector<uint32_t>& particle_materials() const { return material_ids_; }
    
    //----------------------------------------------------------------------
    // Overrides from BaseEntity/ParticleEntity
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;     // MPM integration is solver-driven, minimal here
    virtual void reset() override;
    virtual datatypes::AABB world_aabb() const override; // Bounding box of all particles
    
    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "MPMEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    // Particle state arrays
    std::vector<double> volumes_;                   // Particle volumes (m³)
    std::vector<datatypes::Matrix3r> F_;            // Deformation gradients
    std::vector<datatypes::Matrix3r> Fp_;           // Plastic part of F
    std::vector<datatypes::Matrix3r> C_;            // Affine velocity matrices (APIC)
    std::vector<double> Jp_;                        // Plastic Jacobian determinant
    std::vector<uint32_t> material_ids_;            // Material ID per particle
    
    MPMConfig mpm_config_;                          // Default material configuration
    
    // Helper methods
    void resize_arrays(size_t new_size);            // Resize all particle arrays to given size
    void initialize_particle_defaults(size_t start, size_t count); // Set default state for new particles
};

} // namespace engine
} // namespace genesis