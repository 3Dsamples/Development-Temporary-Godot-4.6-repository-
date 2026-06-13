// genesis/engine/entities/sph_entity.h

#pragma once

//------------------------------------------------------------------------------
// SPH Entity class - Smoothed Particle Hydrodynamics fluid entity.
// Manages a collection of SPH particles with density, pressure, and neighbor lists.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/particle_entity.h" // Base class: ParticleEntity
#include "genesis/datatypes.h"                      // Vector3, real, etc.
#include <vector>                                   // std::vector
#include <string>                                   // std::string

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// SPH Configuration
//------------------------------------------------------------------------------
struct SPHConfig {
    // Particle properties
    double particle_radius = 0.05;                  // Particle radius (m)
    double rest_density = 1000.0;                   // Reference density (kg/m³)
    
    // Fluid parameters
    double viscosity = 0.01;                        // Dynamic viscosity (Pa·s)
    double surface_tension = 0.072;                 // Surface tension coefficient (N/m)
    double gas_stiffness = 1000.0;                  // Stiffness for pressure (Tait equation)
    double speed_of_sound = 20.0;                   // Speed of sound for weakly compressible SPH
    
    // Kernel properties (automatically computed from radius)
    double kernel_radius = 0.1;                     // Support radius (usually 2*particle_radius)
    
    // Solver options
    bool enable_viscosity = true;                   // Apply viscosity forces
    bool enable_surface_tension = true;             // Apply surface tension forces
    bool enable_turbulence = false;                 // Apply vorticity confinement
    double vorticity_coefficient = 0.01;            // Strength of vorticity force
    
    // Artificial viscosity (for stability)
    double artificial_viscosity_alpha = 0.1;
    double artificial_viscosity_beta = 0.0;
    
    // Boundary handling
    bool is_boundary = false;                       // Whether this entity represents a boundary
};

//------------------------------------------------------------------------------
// SPH Entity class - particle-based fluid representation
//------------------------------------------------------------------------------
class SPHEntity : public ParticleEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    SPHEntity();                                    // Default constructor
    explicit SPHEntity(const std::string& name);    // Named constructor
    virtual ~SPHEntity();                           // Destructor

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_sph_config(const SPHConfig& config);   // Set SPH configuration
    const SPHConfig& sph_config() const { return sph_config_; }
    
    // Boundary flag
    bool is_boundary() const { return sph_config_.is_boundary; }
    void set_is_boundary(bool boundary) { sph_config_.is_boundary = boundary; }

    //----------------------------------------------------------------------
    // Particle state extensions (density, pressure)
    //----------------------------------------------------------------------
    const std::vector<double>& densities() const { return densities_; }
    std::vector<double>& densities() { return densities_; }
    double density(size_t i) const;
    void set_density(size_t i, double rho);
    void set_densities(const std::vector<double>& rho);
    
    const std::vector<double>& pressures() const { return pressures_; }
    std::vector<double>& pressures() { return pressures_; }
    double pressure(size_t i) const;
    void set_pressure(size_t i, double p);
    void set_pressures(const std::vector<double>& p);
    
    const std::vector<datatypes::Vector3>& accelerations() const { return accelerations_; }
    std::vector<datatypes::Vector3>& accelerations() { return accelerations_; }
    datatypes::Vector3 acceleration(size_t i) const;
    void set_acceleration(size_t i, const datatypes::Vector3& a);
    void clear_accelerations();
    
    // Neighbor information (optional, may be managed by solver)
    const std::vector<std::vector<uint32_t>>& neighbors() const { return neighbors_; }
    std::vector<std::vector<uint32_t>>& neighbors() { return neighbors_; }
    void clear_neighbors();
    
    // Material identifier
    uint32_t material_id() const { return material_id_; }
    void set_material_id(uint32_t id) { material_id_ = id; }

    //----------------------------------------------------------------------
    // Overrides from ParticleEntity/BaseEntity
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;     // SPH uses solver-driven integration
    virtual void reset() override;
    virtual void add_particles(const std::vector<datatypes::Vector3>& positions,
                               const std::vector<datatypes::Vector3>& velocities,
                               const std::vector<double>& masses) override;
    virtual void remove_particle(size_t index) override;
    virtual void clear_particles() override;

    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "SPHEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    SPHConfig sph_config_;                          // SPH configuration
    
    // SPH-specific state arrays
    std::vector<double> densities_;                 // Particle densities (kg/m³)
    std::vector<double> pressures_;                 // Particle pressures (Pa)
    std::vector<datatypes::Vector3> accelerations_; // Net acceleration (force/mass)
    std::vector<std::vector<uint32_t>> neighbors_;  // Neighbor indices per particle
    uint32_t material_id_ = 0;                      // Material identifier
    
    // Helper methods
    void resize_state_arrays(size_t new_size);      // Resize all SPH arrays
};

} // namespace engine
} // namespace genesis