// genesis/engine/entities/particle_entity.h

#pragma once

//------------------------------------------------------------------------------
// Particle Entity class - base class for particle-based representations (MPM, SPH, PBD).
// Manages a collection of particles with positions, velocities, masses, and forces.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/base_entity.h"   // Base class: BaseEntity
#include "genesis/datatypes.h"                      // Vector3, real, etc.
#include <vector>                                   // std::vector for particle arrays
#include <string>                                   // std::string

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// ParticleEntity - abstract base for all particle systems
//------------------------------------------------------------------------------
class ParticleEntity : public BaseEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    ParticleEntity();                               // Default constructor
    explicit ParticleEntity(const std::string& name); // Named constructor
    virtual ~ParticleEntity();                      // Destructor

    //----------------------------------------------------------------------
    // Particle management
    //----------------------------------------------------------------------
    // Add a single particle
    void add_particle(const datatypes::Vector3& position,
                      const datatypes::Vector3& velocity = datatypes::Vector3(0.0),
                      double mass = 1.0);
    
    // Add multiple particles at once
    virtual void add_particles(const std::vector<datatypes::Vector3>& positions,
                               const std::vector<datatypes::Vector3>& velocities,
                               const std::vector<double>& masses);
    
    // Remove a particle by index
    void remove_particle(size_t index);
    
    // Remove all particles
    void clear_particles();
    
    // Particle count
    size_t particle_count() const { return positions_.size(); }
    bool empty() const { return positions_.empty(); }
    
    //----------------------------------------------------------------------
    // Particle state access (by index)
    //----------------------------------------------------------------------
    datatypes::Vector3 particle_position(size_t i) const;
    void set_particle_position(size_t i, const datatypes::Vector3& pos);
    
    datatypes::Vector3 particle_velocity(size_t i) const;
    void set_particle_velocity(size_t i, const datatypes::Vector3& vel);
    
    double particle_mass(size_t i) const;
    void set_particle_mass(size_t i, double mass);
    
    datatypes::Vector3 particle_force(size_t i) const;
    void set_particle_force(size_t i, const datatypes::Vector3& force);
    void add_particle_force(size_t i, const datatypes::Vector3& force);
    
    // Inverse mass (computed from mass)
    double particle_inverse_mass(size_t i) const;
    
    //----------------------------------------------------------------------
    // Bulk access (for solvers)
    //----------------------------------------------------------------------
    const std::vector<datatypes::Vector3>& particle_positions() const { return positions_; }
    std::vector<datatypes::Vector3>& particle_positions() { return positions_; }
    void set_positions(const std::vector<datatypes::Vector3>& positions);
    
    const std::vector<datatypes::Vector3>& particle_velocities() const { return velocities_; }
    std::vector<datatypes::Vector3>& particle_velocities() { return velocities_; }
    void set_velocities(const std::vector<datatypes::Vector3>& velocities);
    
    const std::vector<double>& particle_masses() const { return masses_; }
    std::vector<double>& particle_masses() { return masses_; }
    void set_masses(const std::vector<double>& masses);
    
    const std::vector<datatypes::Vector3>& particle_forces() const { return forces_; }
    std::vector<datatypes::Vector3>& particle_forces() { return forces_; }
    
    //----------------------------------------------------------------------
    // Aggregate properties
    //----------------------------------------------------------------------
    double total_mass() const;                      // Sum of all particle masses
    datatypes::Vector3 center_of_mass() const;      // Mass-weighted average position
    datatypes::Vector3 total_momentum() const;      // Sum of mass * velocity
    double kinetic_energy() const;                  // 0.5 * sum(m * v²)
    
    //----------------------------------------------------------------------
    // Forces and impulses (overrides)
    //----------------------------------------------------------------------
    virtual void apply_force(const datatypes::Vector3& force) override;
    virtual void apply_force(const datatypes::Vector3& force, const datatypes::Vector3& world_point) override;
    virtual void apply_impulse(const datatypes::Vector3& impulse) override;
    virtual void apply_impulse(const datatypes::Vector3& impulse, const datatypes::Vector3& world_point) override;
    virtual void clear_forces() override;
    
    // Apply force to a specific particle
    void apply_force_to_particle(size_t i, const datatypes::Vector3& force);
    void apply_impulse_to_particle(size_t i, const datatypes::Vector3& impulse);
    
    //----------------------------------------------------------------------
    // Integration (explicit Euler for particles, can be overridden)
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;
    virtual void integrate_velocity(double dt) override;
    virtual void integrate_position(double dt) override;
    
    //----------------------------------------------------------------------
    // Reset
    //----------------------------------------------------------------------
    virtual void reset() override;
    
    //----------------------------------------------------------------------
    // Queries
    //----------------------------------------------------------------------
    virtual datatypes::AABB world_aabb() const override;
    
    // Find nearest particle to a point
    size_t nearest_particle(const datatypes::Vector3& world_point) const;
    
    // Find particles within a radius
    std::vector<size_t> particles_in_radius(const datatypes::Vector3& world_point, double radius) const;
    
    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "ParticleEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

protected:
    std::vector<datatypes::Vector3> positions_;     // Particle positions (world space)
    std::vector<datatypes::Vector3> velocities_;    // Particle velocities
    std::vector<double> masses_;                    // Particle masses
    std::vector<datatypes::Vector3> forces_;        // Accumulated forces per particle
    
    // Initial state for reset
    std::vector<datatypes::Vector3> initial_positions_;
    std::vector<datatypes::Vector3> initial_velocities_;
    std::vector<double> initial_masses_;
    
    // Helper to validate index
    bool valid_index(size_t i) const { return i < positions_.size(); }
    
    // Update derived quantities when masses change
    void update_mass_properties();
};

} // namespace engine
} // namespace genesis