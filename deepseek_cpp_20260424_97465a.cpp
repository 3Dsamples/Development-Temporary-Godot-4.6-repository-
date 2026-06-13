// genesis/engine/entities/sph_entity.cpp

#include "genesis/engine/entities/sph_entity.h"     // Include corresponding header
#include <algorithm>                                // std::copy, std::fill, std::min
#include <numeric>                                  // std::accumulate
#include <sstream>                                  // std::ostringstream for repr
#include <cmath>                                    // std::pow, M_PI

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// SPHEntity construction
//------------------------------------------------------------------------------
SPHEntity::SPHEntity()
    : ParticleEntity("SPHEntity")                   // Call base constructor with default name
{
    // Initialize SPH config with default values
    sph_config_ = SPHConfig{};                      // Use default configuration
}

SPHEntity::SPHEntity(const std::string& name)
    : ParticleEntity(name)                          // Base constructor with custom name
{
    sph_config_ = SPHConfig{};                      // Default configuration
}

SPHEntity::~SPHEntity() {
    // Virtual destructor (vectors automatically cleaned up)
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void SPHEntity::set_sph_config(const SPHConfig& config) {
    // Store SPH configuration
    sph_config_ = config;                           // Copy config struct
    // Ensure kernel radius is consistent with particle radius
    if (sph_config_.kernel_radius < 2.0 * sph_config_.particle_radius) {
        sph_config_.kernel_radius = 2.0 * sph_config_.particle_radius; // Set reasonable default
    }
}

//------------------------------------------------------------------------------
// Particle state extensions
//------------------------------------------------------------------------------
double SPHEntity::density(size_t i) const {
    // Return density of particle i
    if (i < densities_.size()) {
        return densities_[i];                       // Return stored density
    }
    return sph_config_.rest_density;                // Default rest density
}

void SPHEntity::set_density(size_t i, double rho) {
    // Set density for a single particle
    if (i < densities_.size()) {
        densities_[i] = rho;                        // Update density
    }
}

void SPHEntity::set_densities(const std::vector<double>& rho) {
    // Set all densities at once
    if (rho.size() == densities_.size()) {
        densities_ = rho;                           // Copy entire vector
    } else {
        size_t n = std::min(rho.size(), densities_.size());
        std::copy(rho.begin(), rho.begin() + n, densities_.begin());
    }
}

double SPHEntity::pressure(size_t i) const {
    // Return pressure of particle i
    if (i < pressures_.size()) {
        return pressures_[i];                       // Return stored pressure
    }
    return 0.0;                                     // Default zero pressure
}

void SPHEntity::set_pressure(size_t i, double p) {
    // Set pressure for a single particle
    if (i < pressures_.size()) {
        pressures_[i] = p;                          // Update pressure
    }
}

void SPHEntity::set_pressures(const std::vector<double>& p) {
    // Set all pressures at once
    if (p.size() == pressures_.size()) {
        pressures_ = p;                             // Copy entire vector
    } else {
        size_t n = std::min(p.size(), pressures_.size());
        std::copy(p.begin(), p.begin() + n, pressures_.begin());
    }
}

datatypes::Vector3 SPHEntity::acceleration(size_t i) const {
    // Return net acceleration of particle i
    if (i < accelerations_.size()) {
        return accelerations_[i];                   // Return stored acceleration
    }
    return datatypes::Vector3(0.0);                 // Default zero acceleration
}

void SPHEntity::set_acceleration(size_t i, const datatypes::Vector3& a) {
    // Set acceleration for a single particle
    if (i < accelerations_.size()) {
        accelerations_[i] = a;                      // Update acceleration
    }
}

void SPHEntity::clear_accelerations() {
    // Reset all accelerations to zero
    std::fill(accelerations_.begin(), accelerations_.end(), datatypes::Vector3(0.0));
}

void SPHEntity::clear_neighbors() {
    // Clear all neighbor lists
    for (auto& n : neighbors_) {
        n.clear();                                  // Clear each particle's neighbor list
    }
    // Optionally shrink to save memory
    neighbors_.clear();
    neighbors_.resize(particle_count());            // Reinitialize with empty vectors
}

//------------------------------------------------------------------------------
// Overrides from ParticleEntity
//------------------------------------------------------------------------------
void SPHEntity::integrate(double dt) {
    // SPH integration is handled by SPHSolver; this is a fallback simple integration
    if (!is_dynamic() || !enabled_) return;
    // Simple explicit Euler using stored accelerations
    for (size_t i = 0; i < velocities_.size(); ++i) {
        velocities_[i] += accelerations_[i] * dt;   // v += a * dt
        positions_[i] += velocities_[i] * dt;       // x += v * dt
    }
    clear_accelerations();                          // Reset accelerations for next step
}

void SPHEntity::reset() {
    // Reset to initial state
    ParticleEntity::reset();                        // Reset positions, velocities, masses, forces
    // Reset SPH-specific state
    densities_.assign(particle_count(), sph_config_.rest_density); // Reset to rest density
    pressures_.assign(particle_count(), 0.0);       // Zero pressure
    accelerations_.assign(particle_count(), datatypes::Vector3(0.0)); // Zero acceleration
    clear_neighbors();                              // Clear neighbor lists
}

void SPHEntity::add_particles(const std::vector<datatypes::Vector3>& positions,
                              const std::vector<datatypes::Vector3>& velocities,
                              const std::vector<double>& masses) {
    // Add multiple particles (override to also extend SPH-specific arrays)
    size_t old_size = particle_count();
    ParticleEntity::add_particles(positions, velocities, masses); // Base class addition
    size_t new_size = particle_count();
    if (new_size > old_size) {
        resize_state_arrays(new_size);              // Ensure SPH arrays are sized correctly
        // Initialize new SPH state to defaults
        for (size_t i = old_size; i < new_size; ++i) {
            densities_[i] = sph_config_.rest_density;
            pressures_[i] = 0.0;
            accelerations_[i] = datatypes::Vector3(0.0);
        }
        // Neighbors will be recomputed by solver
    }
}

void SPHEntity::remove_particle(size_t index) {
    // Remove a single particle and its associated SPH state
    if (!valid_index(index)) return;
    // Erase from SPH-specific arrays
    densities_.erase(densities_.begin() + index);
    pressures_.erase(pressures_.begin() + index);
    accelerations_.erase(accelerations_.begin() + index);
    if (index < neighbors_.size()) {
        neighbors_.erase(neighbors_.begin() + index);
    }
    // Erase from base class arrays
    ParticleEntity::remove_particle(index);
}

void SPHEntity::clear_particles() {
    // Remove all particles and clear SPH state
    ParticleEntity::clear_particles();              // Clear base arrays
    densities_.clear();
    pressures_.clear();
    accelerations_.clear();
    neighbors_.clear();
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string SPHEntity::repr() const {
    std::ostringstream oss;
    oss << "SPHEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", particles=" << particle_count()
        << ", rest_density=" << sph_config_.rest_density
        << ", is_boundary=" << sph_config_.is_boundary
        << ", material=" << material_id_
        << ")";
    return oss.str();
}

std::string SPHEntity::str() const {
    return name() + " (SPH: " + std::to_string(particle_count()) + " particles)";
}

//------------------------------------------------------------------------------
// Private helper methods
//------------------------------------------------------------------------------
void SPHEntity::resize_state_arrays(size_t new_size) {
    // Resize all SPH-specific arrays to the given size
    densities_.resize(new_size, sph_config_.rest_density);
    pressures_.resize(new_size, 0.0);
    accelerations_.resize(new_size, datatypes::Vector3(0.0));
    neighbors_.resize(new_size);                    // Each will be an empty vector
}

} // namespace engine
} // namespace genesis