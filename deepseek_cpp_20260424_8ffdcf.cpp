// genesis/engine/entities/emitter.h

#pragma once

//------------------------------------------------------------------------------
// Emitter entity class - spawns particles (MPM, SPH, or generic) over time.
// Supports various emission shapes, rates, and initial velocity profiles.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/base_entity.h"   // Base class: BaseEntity
#include "genesis/datatypes.h"                      // Vector3, Quat, AABB, etc.
#include <vector>                                   // std::vector for particle buffers
#include <functional>                               // std::function for custom emission callbacks
#include <random>                                   // std::mt19937, distributions for randomness

namespace genesis {
namespace engine {

// Forward declarations
class ParticleEntity;                              // Generic particle container
class MPMEntity;                                   // MPM-specific particle container
class SPHEntity;                                   // SPH-specific particle container

//------------------------------------------------------------------------------
// Emission shape types
//------------------------------------------------------------------------------
enum class EmitterShape : uint8_t {
    POINT = 0,                                      // Emit from a single point
    SPHERE = 1,                                     // Emit within a sphere (surface or volume)
    BOX = 2,                                        // Emit within a box (surface or volume)
    CYLINDER = 3,                                   // Emit within a cylinder
    MESH_SURFACE = 4,                               // Emit from mesh surface
    CUSTOM = 5                                      // Custom emission function
};

//------------------------------------------------------------------------------
// Emission mode (continuous vs burst)
//------------------------------------------------------------------------------
enum class EmitterMode : uint8_t {
    CONTINUOUS = 0,                                 // Emit particles every frame
    BURST = 1,                                      // Emit a fixed number once
    PERIODIC_BURST = 2                              // Emit bursts at intervals
};

//------------------------------------------------------------------------------
// Emitter configuration
//------------------------------------------------------------------------------
struct EmitterConfig {
    // Basic parameters
    EmitterShape shape = EmitterShape::POINT;       // Emission shape
    EmitterMode mode = EmitterMode::CONTINUOUS;     // Emission mode
    double emission_rate = 100.0;                   // Particles per second (continuous) or total count (burst)
    double burst_interval = 1.0;                    // Seconds between bursts (periodic burst mode)
    double lifetime = -1.0;                         // How long emitter is active (-1 = infinite)
    bool enabled = true;                            // Whether emission is currently active
    
    // Shape parameters
    double sphere_radius = 1.0;                     // For sphere shape
    bool sphere_surface_only = false;               // Emit only on surface
    datatypes::Vector3 box_extents = {1.0, 1.0, 1.0}; // For box shape (half extents)
    bool box_surface_only = false;                  // Emit only on box surface
    double cylinder_radius = 1.0;                   // For cylinder shape
    double cylinder_height = 2.0;                   // Cylinder height
    bool cylinder_surface_only = false;             // Emit only on cylinder surface
    
    // Velocity parameters
    datatypes::Vector3 initial_velocity = {0.0, 0.0, 0.0}; // Base velocity (world or local depending on flag)
    bool velocity_in_local_space = true;            // If true, initial_velocity is in emitter's local frame
    double speed_variation = 0.0;                   // Random variation factor for speed (0 = none, 1 = 100%)
    double angle_spread = 0.0;                      // Cone angle spread in radians (0 = directional, π = omnidirectional)
    
    // Particle properties
    double particle_mass = 1.0;                     // Mass of each emitted particle
    double particle_radius = 0.05;                  // Radius (for SPH/MPM)
    uint32_t material_id = 0;                       // Material identifier for emitted particles
    double initial_temperature = 293.0;             // Temperature (for thermal effects)
    
    // Target entity
    std::string target_entity_name;                 // Name of entity to emit into (if empty, creates new)
    bool create_new_entity = false;                 // Whether to spawn a new entity per emission
};

//------------------------------------------------------------------------------
// EmitterEntity class - spawns particles into the simulation
//------------------------------------------------------------------------------
class EmitterEntity : public BaseEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    EmitterEntity();                                // Default constructor
    explicit EmitterEntity(const EmitterConfig& config); // Constructor with config
    explicit EmitterEntity(const std::string& name); // Named constructor
    virtual ~EmitterEntity();                       // Destructor

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_config(const EmitterConfig& config);   // Apply emitter configuration
    const EmitterConfig& config() const { return config_; } // Get configuration

    //----------------------------------------------------------------------
    // Control
    //----------------------------------------------------------------------
    void start();                                   // Enable emission
    void stop();                                    // Disable emission
    void emit_burst(int count);                     // Emit a single burst of given count
    void set_target_entity(std::shared_ptr<ParticleEntity> target); // Set target particle entity
    void set_target_entity(std::shared_ptr<MPMEntity> target); // Set target MPM entity
    void set_target_entity(std::shared_ptr<SPHEntity> target); // Set target SPH entity

    //----------------------------------------------------------------------
    // Update (called each simulation step)
    //----------------------------------------------------------------------
    virtual void update(double dt);                  // Update emission and spawn particles

    //----------------------------------------------------------------------
    // Override physics integration (emitter itself doesn't move by physics)
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;      // Calls update(dt) then base integration

    //----------------------------------------------------------------------
    // State queries
    //----------------------------------------------------------------------
    double get_elapsed_time() const { return elapsed_time_; } // Time since start
    double get_emitted_count() const { return total_emitted_; } // Total particles emitted
    bool is_active() const { return config_.enabled && (config_.lifetime < 0 || elapsed_time_ < config_.lifetime); }

    //----------------------------------------------------------------------
    // Reset
    //----------------------------------------------------------------------
    virtual void reset() override;

    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "EmitterEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    EmitterConfig config_;                          // Emitter configuration
    double elapsed_time_ = 0.0;                     // Total active time
    double time_since_last_burst_ = 0.0;            // Time accumulator for periodic bursts
    double emission_accumulator_ = 0.0;             // Fractional particle count for continuous emission
    size_t total_emitted_ = 0;                      // Total particles emitted
    
    // Target entity pointers (weak to avoid circular references)
    std::weak_ptr<ParticleEntity> target_particle_entity_; // Generic particle target
    std::weak_ptr<MPMEntity> target_mpm_entity_;    // MPM-specific target
    std::weak_ptr<SPHEntity> target_sph_entity_;    // SPH-specific target
    
    // Random number generation
    std::mt19937 rng_;                              // Mersenne Twister RNG engine
    std::uniform_real_distribution<double> uniform_dist_; // Uniform distribution [0,1]
    std::normal_distribution<double> normal_dist_;  // Normal distribution (mean=0, std=1)
    
    // Internal helper methods
    void emit_particles(int count);                 // Emit a specific number of particles
    datatypes::Vector3 generate_position() const;   // Generate random position within emission shape
    datatypes::Vector3 generate_velocity(const datatypes::Vector3& base_vel) const; // Generate varied velocity
    void initialize_random_generator();             // Seed RNG with random device
    void emit_to_entity(int count, std::shared_ptr<ParticleEntity> entity); // Add particles to generic entity
    void emit_to_entity(int count, std::shared_ptr<MPMEntity> entity); // Add particles to MPM entity
    void emit_to_entity(int count, std::shared_ptr<SPHEntity> entity); // Add particles to SPH entity
    std::shared_ptr<ParticleEntity> find_or_create_target(); // Get or create target particle entity
};

} // namespace engine
} // namespace genesis