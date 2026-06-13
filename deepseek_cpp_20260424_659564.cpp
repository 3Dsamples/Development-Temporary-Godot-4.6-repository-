// genesis/engine/entities/emitter.cpp

#include "genesis/engine/entities/emitter.h"       // Include corresponding header
#include "genesis/engine/entities/particle_entity.h" // ParticleEntity target type
#include "genesis/engine/entities/mpm_entity.h"    // MPMEntity target type
#include "genesis/engine/entities/sph_entity.h"    // SPHEntity target type
#include "genesis/engine/scene.h"                  // Scene needed to create new entities
#include <cmath>                                   // std::sin, std::cos, std::sqrt, M_PI
#include <algorithm>                               // std::clamp, std::max, std::min
#include <sstream>                                 // std::ostringstream for repr
#include <random>                                  // std::random_device for seeding
#include <chrono>                                  // std::chrono for seed

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// EmitterEntity construction
//------------------------------------------------------------------------------
EmitterEntity::EmitterEntity()
    : BaseEntity("Emitter")                        // Call base constructor with default name
    , config_()                                    // Default configuration
    , uniform_dist_(0.0, 1.0)                      // Initialize uniform distribution [0,1]
    , normal_dist_(0.0, 1.0)                       // Initialize standard normal distribution
{
    // Seed the random number generator with a truly random value
    initialize_random_generator();                 // Set up RNG with random device
}

EmitterEntity::EmitterEntity(const EmitterConfig& config)
    : BaseEntity("Emitter")                        // Base constructor
    , config_(config)                              // Copy configuration
    , uniform_dist_(0.0, 1.0)                      // Uniform distribution
    , normal_dist_(0.0, 1.0)                       // Normal distribution
{
    initialize_random_generator();                 // Seed RNG
}

EmitterEntity::EmitterEntity(const std::string& name)
    : BaseEntity(name)                             // Base constructor with custom name
    , config_()                                    // Default config
    , uniform_dist_(0.0, 1.0)                      // Uniform distribution
    , normal_dist_(0.0, 1.0)                       // Normal distribution
{
    initialize_random_generator();                 // Seed RNG
}

EmitterEntity::~EmitterEntity() {
    // Virtual destructor (cleanup handled by base class and smart pointers)
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void EmitterEntity::set_config(const EmitterConfig& config) {
    // Store new configuration
    config_ = config;                              // Copy config struct
}

//------------------------------------------------------------------------------
// Control
//------------------------------------------------------------------------------
void EmitterEntity::start() {
    // Enable emission
    config_.enabled = true;                        // Set enabled flag
    // Reset elapsed time if lifetime is finite and we want fresh start
    if (config_.lifetime > 0) {
        elapsed_time_ = 0.0;                       // Restart lifetime counter
    }
    time_since_last_burst_ = 0.0;                  // Reset burst timer
    emission_accumulator_ = 0.0;                   // Reset fractional accumulator
}

void EmitterEntity::stop() {
    // Disable emission
    config_.enabled = false;                       // Clear enabled flag
}

void EmitterEntity::emit_burst(int count) {
    // Emit a fixed number of particles immediately (bypasses rate/mode)
    if (count <= 0) return;                        // Nothing to emit
    emit_particles(count);                         // Delegate to internal emission
    total_emitted_ += count;                       // Update total counter
}

void EmitterEntity::set_target_entity(std::shared_ptr<ParticleEntity> target) {
    // Set the target generic particle entity
    target_particle_entity_ = target;              // Store weak pointer
    // Clear other target types to avoid ambiguity
    target_mpm_entity_.reset();                    // Clear MPM target
    target_sph_entity_.reset();                    // Clear SPH target
}

void EmitterEntity::set_target_entity(std::shared_ptr<MPMEntity> target) {
    // Set the target MPM entity
    target_mpm_entity_ = target;                   // Store weak pointer
    target_particle_entity_.reset();               // Clear generic target
    target_sph_entity_.reset();                    // Clear SPH target
}

void EmitterEntity::set_target_entity(std::shared_ptr<SPHEntity> target) {
    // Set the target SPH entity
    target_sph_entity_ = target;                   // Store weak pointer
    target_particle_entity_.reset();               // Clear generic target
    target_mpm_entity_.reset();                    // Clear MPM target
}

//------------------------------------------------------------------------------
// Update
//------------------------------------------------------------------------------
void EmitterEntity::update(double dt) {
    // Check if emitter is active
    if (!is_active()) return;                      // Not active, nothing to do
    
    // Update elapsed time
    elapsed_time_ += dt;                           // Increment active timer
    
    // Handle different emission modes
    switch (config_.mode) {
        case EmitterMode::CONTINUOUS: {
            // Compute number of particles to emit this step
            double to_emit = config_.emission_rate * dt + emission_accumulator_;
            int count = static_cast<int>(to_emit); // Integer part
            emission_accumulator_ = to_emit - count; // Store fractional remainder
            if (count > 0) {
                emit_particles(count);             // Spawn particles
                total_emitted_ += count;           // Update total
            }
            break;
        }
        case EmitterMode::BURST: {
            // Burst mode: emit only once (handled by start/reset or explicit call)
            // Here we emit if enabled and not yet emitted (tracked by elapsed time == 0?)
            // For simplicity, we rely on explicit emit_burst or on first update
            if (elapsed_time_ <= dt && total_emitted_ == 0) {
                int count = static_cast<int>(config_.emission_rate);
                emit_particles(count);             // Emit burst
                total_emitted_ += count;
            }
            break;
        }
        case EmitterMode::PERIODIC_BURST: {
            // Accumulate time and emit bursts at intervals
            time_since_last_burst_ += dt;          // Add delta time
            if (time_since_last_burst_ >= config_.burst_interval) {
                int bursts = static_cast<int>(time_since_last_burst_ / config_.burst_interval);
                int count_per_burst = static_cast<int>(config_.emission_rate);
                int total_count = bursts * count_per_burst;
                emit_particles(total_count);       // Emit multiple bursts at once
                total_emitted_ += total_count;
                time_since_last_burst_ = std::fmod(time_since_last_burst_, config_.burst_interval);
            }
            break;
        }
    }
}

//------------------------------------------------------------------------------
// Integration override
//------------------------------------------------------------------------------
void EmitterEntity::integrate(double dt) {
    // Emitter itself doesn't have physics-driven motion, but it may be moved manually
    // First, update emission
    update(dt);                                    // Spawn particles based on dt
    // Then call base integration (which does nothing for static emitter, but maintains consistency)
    BaseEntity::integrate(dt);                     // Base integration (position/velocity unchanged)
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
void EmitterEntity::reset() {
    // Reset base entity state
    BaseEntity::reset();                           // Reset transform, velocity, etc.
    // Reset emission state
    elapsed_time_ = 0.0;                           // Clear elapsed time
    time_since_last_burst_ = 0.0;                  // Clear burst timer
    emission_accumulator_ = 0.0;                   // Clear fractional accumulator
    total_emitted_ = 0;                            // Reset emission counter
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string EmitterEntity::repr() const {
    std::ostringstream oss;
    oss << "EmitterEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", active=" << is_active()
        << ", rate=" << config_.emission_rate
        << ", emitted=" << total_emitted_
        << ")";
    return oss.str();
}

std::string EmitterEntity::str() const {
    return "Emitter " + name() + " (" + std::to_string(total_emitted_) + " emitted)";
}

//------------------------------------------------------------------------------
// Internal: Emit particles
//------------------------------------------------------------------------------
void EmitterEntity::emit_particles(int count) {
    // Find or create the target entity to receive particles
    auto target = find_or_create_target();         // Get target particle container
    if (!target) return;                           // No target available, abort
    
    // Determine entity type and call appropriate overload
    if (auto mpm_target = std::dynamic_pointer_cast<MPMEntity>(target)) {
        emit_to_entity(count, mpm_target);         // Emit to MPM entity
    } else if (auto sph_target = std::dynamic_pointer_cast<SPHEntity>(target)) {
        emit_to_entity(count, sph_target);         // Emit to SPH entity
    } else {
        // Fallback to generic particle entity
        emit_to_entity(count, target);             // Emit to generic particle entity
    }
}

std::shared_ptr<ParticleEntity> EmitterEntity::find_or_create_target() {
    // Try to get existing target from weak pointers
    if (auto target = target_particle_entity_.lock()) {
        return target;                             // Return existing generic target
    }
    if (auto target = target_mpm_entity_.lock()) {
        return target;                             // Return MPM target (which derives from ParticleEntity)
    }
    if (auto target = target_sph_entity_.lock()) {
        return target;                             // Return SPH target
    }
    
    // If no target but name specified, search scene
    if (!config_.target_entity_name.empty() && scene_) {
        auto entity = scene_->get_entity(config_.target_entity_name);
        if (entity) {
            if (auto particle_entity = std::dynamic_pointer_cast<ParticleEntity>(entity)) {
                target_particle_entity_ = particle_entity;
                return particle_entity;
            }
        }
    }
    
    // If we should create a new entity and we have a scene
    if (config_.create_new_entity && scene_) {
        // Determine which type to create based on what targets were set or defaults
        // Default to generic ParticleEntity
        auto new_entity = std::make_shared<ParticleEntity>();
        new_entity->set_name(name_ + "_particles");
        scene_->add_entity(new_entity);
        target_particle_entity_ = new_entity;
        return new_entity;
    }
    
    return nullptr;                                // No target available
}

//------------------------------------------------------------------------------
// Position generation based on shape
//------------------------------------------------------------------------------
datatypes::Vector3 EmitterEntity::generate_position() const {
    datatypes::Vector3 local_pos;                  // Position in emitter's local space
    
    switch (config_.shape) {
        case EmitterShape::POINT: {
            local_pos = datatypes::Vector3(0.0);   // Origin
            break;
        }
        case EmitterShape::SPHERE: {
            // Generate random point in sphere (or on surface)
            double u = uniform_dist_(rng_);        // Random [0,1]
            double v = uniform_dist_(rng_);        // Random [0,1]
            double w = uniform_dist_(rng_);        // Random [0,1]
            
            double theta = 2.0 * M_PI * u;         // Azimuthal angle
            double phi = std::acos(2.0 * v - 1.0); // Polar angle
            
            double r = config_.sphere_radius;
            if (!config_.sphere_surface_only) {
                r *= std::cbrt(w);                 // Uniform distribution in volume
            }
            
            double sin_phi = std::sin(phi);
            local_pos[0] = r * sin_phi * std::cos(theta);
            local_pos[1] = r * sin_phi * std::sin(theta);
            local_pos[2] = r * std::cos(phi);
            break;
        }
        case EmitterShape::BOX: {
            // Generate random point in box (or on surface)
            datatypes::Vector3 ext = config_.box_extents;
            if (config_.box_surface_only) {
                // Choose a face (6 faces)
                int face = static_cast<int>(uniform_dist_(rng_) * 6.0);
                double u = uniform_dist_(rng_) * 2.0 - 1.0;
                double v = uniform_dist_(rng_) * 2.0 - 1.0;
                switch (face) {
                    case 0: local_pos = datatypes::Vector3( ext[0], u*ext[1], v*ext[2]); break;
                    case 1: local_pos = datatypes::Vector3(-ext[0], u*ext[1], v*ext[2]); break;
                    case 2: local_pos = datatypes::Vector3(u*ext[0],  ext[1], v*ext[2]); break;
                    case 3: local_pos = datatypes::Vector3(u*ext[0], -ext[1], v*ext[2]); break;
                    case 4: local_pos = datatypes::Vector3(u*ext[0], v*ext[1],  ext[2]); break;
                    case 5: local_pos = datatypes::Vector3(u*ext[0], v*ext[1], -ext[2]); break;
                }
            } else {
                // Uniform in volume
                local_pos[0] = (uniform_dist_(rng_) * 2.0 - 1.0) * ext[0];
                local_pos[1] = (uniform_dist_(rng_) * 2.0 - 1.0) * ext[1];
                local_pos[2] = (uniform_dist_(rng_) * 2.0 - 1.0) * ext[2];
            }
            break;
        }
        case EmitterShape::CYLINDER: {
            double r = config_.cylinder_radius;
            double h = config_.cylinder_height;
            double angle = 2.0 * M_PI * uniform_dist_(rng_);
            
            if (config_.cylinder_surface_only) {
                // On cylindrical surface: radius fixed
                local_pos[0] = r * std::cos(angle);
                local_pos[1] = r * std::sin(angle);
                local_pos[2] = (uniform_dist_(rng_) * 2.0 - 1.0) * h * 0.5;
            } else {
                // In volume
                double radius = r * std::sqrt(uniform_dist_(rng_));
                local_pos[0] = radius * std::cos(angle);
                local_pos[1] = radius * std::sin(angle);
                local_pos[2] = (uniform_dist_(rng_) * 2.0 - 1.0) * h * 0.5;
            }
            break;
        }
        default:
            local_pos = datatypes::Vector3(0.0);   // Fallback to origin
            break;
    }
    
    // Transform to world space
    return transform().transformPoint(local_pos); // Apply emitter's world transform
}

//------------------------------------------------------------------------------
// Velocity generation with variation and spread
//------------------------------------------------------------------------------
datatypes::Vector3 EmitterEntity::generate_velocity(const datatypes::Vector3& base_vel) const {
    datatypes::Vector3 vel = base_vel;             // Start with base velocity
    
    // Apply speed variation
    if (config_.speed_variation > 0.0) {
        double factor = 1.0 + config_.speed_variation * (uniform_dist_(rng_) * 2.0 - 1.0);
        factor = std::max(0.0, factor);            // Don't reverse direction by variation
        vel = vel * factor;                        // Scale magnitude
    }
    
    // Apply angular spread (cone)
    if (config_.angle_spread > 0.0) {
        double speed = vel.norm();                 // Original speed
        if (speed > 1e-12) {
            datatypes::Vector3 dir = vel / speed;  // Base direction
            
            // Generate random perpendicular vectors to form basis
            datatypes::Vector3 perp1, perp2;
            if (std::abs(dir[0]) < 0.9) {
                perp1 = datatypes::Vector3(1,0,0).cross(dir).normalized();
            } else {
                perp1 = datatypes::Vector3(0,1,0).cross(dir).normalized();
            }
            perp2 = dir.cross(perp1).normalized();
            
            // Random azimuth and polar within cone
            double theta = 2.0 * M_PI * uniform_dist_(rng_);
            double phi = config_.angle_spread * uniform_dist_(rng_);
            
            double sin_phi = std::sin(phi);
            datatypes::Vector3 new_dir = dir * std::cos(phi) 
                                       + perp1 * (sin_phi * std::cos(theta))
                                       + perp2 * (sin_phi * std::sin(theta));
            new_dir.normalize();
            vel = new_dir * speed;
        }
    }
    
    return vel;
}

//------------------------------------------------------------------------------
// Random generator initialization
//------------------------------------------------------------------------------
void EmitterEntity::initialize_random_generator() {
    // Seed with a combination of random device and time
    std::random_device rd;                         // Hardware random device
    auto seed = rd() ^ std::chrono::steady_clock::now().time_since_epoch().count();
    rng_.seed(static_cast<unsigned int>(seed));    // Seed Mersenne Twister
}

//------------------------------------------------------------------------------
// Emit to specific entity types
//------------------------------------------------------------------------------
void EmitterEntity::emit_to_entity(int count, std::shared_ptr<ParticleEntity> entity) {
    if (!entity) return;                           // No target
    
    // Prepare vectors for batch addition
    std::vector<datatypes::Vector3> positions;
    std::vector<datatypes::Vector3> velocities;
    std::vector<double> masses;
    positions.reserve(count);
    velocities.reserve(count);
    masses.reserve(count);
    
    // Base velocity in world space
    datatypes::Vector3 base_vel = config_.initial_velocity;
    if (config_.velocity_in_local_space) {
        base_vel = rotation().rotate(base_vel);    // Convert local to world
    }
    
    for (int i = 0; i < count; ++i) {
        positions.push_back(generate_position());  // Random position
        velocities.push_back(generate_velocity(base_vel)); // Varied velocity
        masses.push_back(config_.particle_mass);   // Constant mass
    }
    
    // Add to entity
    entity->add_particles(positions, velocities, masses);
}

void EmitterEntity::emit_to_entity(int count, std::shared_ptr<MPMEntity> entity) {
    if (!entity) return;
    
    std::vector<datatypes::Vector3> positions;
    std::vector<datatypes::Vector3> velocities;
    std::vector<double> masses;
    std::vector<double> volumes;
    positions.reserve(count);
    velocities.reserve(count);
    masses.reserve(count);
    volumes.reserve(count);
    
    datatypes::Vector3 base_vel = config_.initial_velocity;
    if (config_.velocity_in_local_space) {
        base_vel = rotation().rotate(base_vel);
    }
    
    double volume = (4.0/3.0) * M_PI * std::pow(config_.particle_radius, 3);
    
    for (int i = 0; i < count; ++i) {
        positions.push_back(generate_position());
        velocities.push_back(generate_velocity(base_vel));
        masses.push_back(config_.particle_mass);
        volumes.push_back(volume);
    }
    
    entity->add_particles(positions, velocities, masses, volumes);
}

void EmitterEntity::emit_to_entity(int count, std::shared_ptr<SPHEntity> entity) {
    if (!entity) return;
    
    std::vector<datatypes::Vector3> positions;
    std::vector<datatypes::Vector3> velocities;
    std::vector<double> masses;
    positions.reserve(count);
    velocities.reserve(count);
    masses.reserve(count);
    
    datatypes::Vector3 base_vel = config_.initial_velocity;
    if (config_.velocity_in_local_space) {
        base_vel = rotation().rotate(base_vel);
    }
    
    for (int i = 0; i < count; ++i) {
        positions.push_back(generate_position());
        velocities.push_back(generate_velocity(base_vel));
        masses.push_back(config_.particle_mass);
    }
    
    entity->add_particles(positions, velocities, masses);
}

} // namespace engine
} // namespace genesis