// genesis/engine/entities/particle_entity.cpp

#include "genesis/engine/entities/particle_entity.h" // Include corresponding header
#include <algorithm>                                 // std::copy, std::fill, std::min, std::max
#include <numeric>                                   // std::accumulate
#include <cmath>                                     // std::sqrt
#include <sstream>                                   // std::ostringstream
#include <limits>                                    // std::numeric_limits

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// ParticleEntity construction
//------------------------------------------------------------------------------
ParticleEntity::ParticleEntity()
    : BaseEntity("ParticleEntity")                  // Call base constructor with default name
{
    // Empty constructor body (vectors are empty initially)
}

ParticleEntity::ParticleEntity(const std::string& name)
    : BaseEntity(name)                              // Base constructor with custom name
{
    // Empty constructor body
}

ParticleEntity::~ParticleEntity() {
    // Virtual destructor (vectors automatically cleaned up)
}

//------------------------------------------------------------------------------
// Particle management
//------------------------------------------------------------------------------
void ParticleEntity::add_particle(const datatypes::Vector3& position,
                                  const datatypes::Vector3& velocity,
                                  double mass) {
    // Add a single particle to the end of the arrays
    positions_.push_back(position);                 // Add position
    velocities_.push_back(velocity);                // Add velocity
    masses_.push_back(mass);                        // Add mass
    forces_.push_back(datatypes::Vector3(0.0));     // Initialize force to zero
    
    // Also store initial state for reset functionality
    initial_positions_.push_back(position);         // Store initial position
    initial_velocities_.push_back(velocity);        // Store initial velocity
    initial_masses_.push_back(mass);                // Store initial mass
}

void ParticleEntity::add_particles(const std::vector<datatypes::Vector3>& positions,
                                   const std::vector<datatypes::Vector3>& velocities,
                                   const std::vector<double>& masses) {
    // Add multiple particles at once (more efficient than repeated push_back)
    size_t old_size = positions_.size();            // Current size
    size_t add_count = positions.size();            // Number to add
    
    // Resize all vectors to accommodate new particles
    positions_.resize(old_size + add_count);        // Expand positions array
    velocities_.resize(old_size + add_count);       // Expand velocities array
    masses_.resize(old_size + add_count);           // Expand masses array
    forces_.resize(old_size + add_count, datatypes::Vector3(0.0)); // Expand forces with zeros
    initial_positions_.resize(old_size + add_count);
    initial_velocities_.resize(old_size + add_count);
    initial_masses_.resize(old_size + add_count);
    
    // Copy provided data into the newly allocated slots
    std::copy(positions.begin(), positions.end(), positions_.begin() + old_size);
    std::copy(velocities.begin(), velocities.end(), velocities_.begin() + old_size);
    std::copy(masses.begin(), masses.end(), masses_.begin() + old_size);
    
    // Copy to initial state arrays as well
    std::copy(positions.begin(), positions.end(), initial_positions_.begin() + old_size);
    std::copy(velocities.begin(), velocities.end(), initial_velocities_.begin() + old_size);
    std::copy(masses.begin(), masses.end(), initial_masses_.begin() + old_size);
}

void ParticleEntity::remove_particle(size_t index) {
    // Remove a single particle by index (preserving order of remaining)
    if (!valid_index(index)) return;                // Check bounds
    
    // Erase the element at the given index from each vector
    positions_.erase(positions_.begin() + index);   // Remove position
    velocities_.erase(velocities_.begin() + index); // Remove velocity
    masses_.erase(masses_.begin() + index);         // Remove mass
    forces_.erase(forces_.begin() + index);         // Remove force
    initial_positions_.erase(initial_positions_.begin() + index);
    initial_velocities_.erase(initial_velocities_.begin() + index);
    initial_masses_.erase(initial_masses_.begin() + index);
}

void ParticleEntity::clear_particles() {
    // Remove all particles
    positions_.clear();                             // Clear positions
    velocities_.clear();                            // Clear velocities
    masses_.clear();                                // Clear masses
    forces_.clear();                                // Clear forces
    initial_positions_.clear();                     // Clear initial state
    initial_velocities_.clear();
    initial_masses_.clear();
}

//------------------------------------------------------------------------------
// Particle state access (by index)
//------------------------------------------------------------------------------
datatypes::Vector3 ParticleEntity::particle_position(size_t i) const {
    // Return position of particle i (bounds checked)
    return valid_index(i) ? positions_[i] : datatypes::Vector3(0.0);
}

void ParticleEntity::set_particle_position(size_t i, const datatypes::Vector3& pos) {
    // Set position of particle i
    if (valid_index(i)) {
        positions_[i] = pos;                        // Update position
    }
}

datatypes::Vector3 ParticleEntity::particle_velocity(size_t i) const {
    // Return velocity of particle i
    return valid_index(i) ? velocities_[i] : datatypes::Vector3(0.0);
}

void ParticleEntity::set_particle_velocity(size_t i, const datatypes::Vector3& vel) {
    // Set velocity of particle i
    if (valid_index(i)) {
        velocities_[i] = vel;                       // Update velocity
    }
}

double ParticleEntity::particle_mass(size_t i) const {
    // Return mass of particle i
    return valid_index(i) ? masses_[i] : 0.0;
}

void ParticleEntity::set_particle_mass(size_t i, double mass) {
    // Set mass of particle i
    if (valid_index(i)) {
        masses_[i] = mass;                          // Update mass
    }
}

datatypes::Vector3 ParticleEntity::particle_force(size_t i) const {
    // Return accumulated force on particle i
    return valid_index(i) ? forces_[i] : datatypes::Vector3(0.0);
}

void ParticleEntity::set_particle_force(size_t i, const datatypes::Vector3& force) {
    // Overwrite accumulated force on particle i
    if (valid_index(i)) {
        forces_[i] = force;                         // Set force directly
    }
}

void ParticleEntity::add_particle_force(size_t i, const datatypes::Vector3& force) {
    // Add to accumulated force on particle i
    if (valid_index(i)) {
        forces_[i] += force;                        // Accumulate force
    }
}

double ParticleEntity::particle_inverse_mass(size_t i) const {
    // Compute inverse mass (0 for infinite mass)
    double m = particle_mass(i);                    // Get mass
    return (m > 1e-12) ? 1.0 / m : 0.0;             // Return inverse or 0 if mass is tiny
}

//------------------------------------------------------------------------------
// Bulk access
//------------------------------------------------------------------------------
void ParticleEntity::set_positions(const std::vector<datatypes::Vector3>& positions) {
    // Replace all particle positions
    if (positions.size() == positions_.size()) {
        positions_ = positions;                     // Copy entire vector
    } else {
        size_t n = std::min(positions.size(), positions_.size());
        std::copy(positions.begin(), positions.begin() + n, positions_.begin());
    }
}

void ParticleEntity::set_velocities(const std::vector<datatypes::Vector3>& velocities) {
    // Replace all particle velocities
    if (velocities.size() == velocities_.size()) {
        velocities_ = velocities;                   // Copy entire vector
    } else {
        size_t n = std::min(velocities.size(), velocities_.size());
        std::copy(velocities.begin(), velocities.begin() + n, velocities_.begin());
    }
}

void ParticleEntity::set_masses(const std::vector<double>& masses) {
    // Replace all particle masses
    if (masses.size() == masses_.size()) {
        masses_ = masses;                           // Copy entire vector
    } else {
        size_t n = std::min(masses.size(), masses_.size());
        std::copy(masses.begin(), masses.begin() + n, masses_.begin());
    }
}

//------------------------------------------------------------------------------
// Aggregate properties
//------------------------------------------------------------------------------
double ParticleEntity::total_mass() const {
    // Compute sum of all particle masses
    return std::accumulate(masses_.begin(), masses_.end(), 0.0); // Sum masses
}

datatypes::Vector3 ParticleEntity::center_of_mass() const {
    // Compute mass-weighted average position
    datatypes::Vector3 com(0.0);                    // Initialize to zero
    double total_m = 0.0;                           // Total mass accumulator
    for (size_t i = 0; i < positions_.size(); ++i) {
        com += positions_[i] * masses_[i];          // Weight position by mass
        total_m += masses_[i];                      // Accumulate total mass
    }
    if (total_m > 0) {
        com /= total_m;                             // Normalize by total mass
    }
    return com;                                     // Return center of mass
}

datatypes::Vector3 ParticleEntity::total_momentum() const {
    // Compute sum of (mass * velocity)
    datatypes::Vector3 momentum(0.0);
    for (size_t i = 0; i < velocities_.size(); ++i) {
        momentum += velocities_[i] * masses_[i];    // m * v
    }
    return momentum;
}

double ParticleEntity::kinetic_energy() const {
    // Compute 0.5 * sum(m * v²)
    double energy = 0.0;
    for (size_t i = 0; i < velocities_.size(); ++i) {
        double v2 = velocities_[i].squaredNorm();   // |v|²
        energy += 0.5 * masses_[i] * v2;            // ½ m v²
    }
    return energy;
}

//------------------------------------------------------------------------------
// Forces and impulses (overrides)
//------------------------------------------------------------------------------
void ParticleEntity::apply_force(const datatypes::Vector3& force) {
    // Distribute force equally among all particles (as a body force)
    if (positions_.empty()) return;                 // No particles to apply to
    datatypes::Vector3 per_particle = force / static_cast<double>(positions_.size());
    for (auto& f : forces_) {
        f += per_particle;                          // Add equal share to each particle
    }
}

void ParticleEntity::apply_force(const datatypes::Vector3& force, const datatypes::Vector3& world_point) {
    // Apply force at a point: find nearest particle and apply there
    if (positions_.empty()) return;
    size_t nearest = nearest_particle(world_point); // Find closest particle
    forces_[nearest] += force;                      // Apply force to that particle
}

void ParticleEntity::apply_impulse(const datatypes::Vector3& impulse) {
    // Instantaneously change velocities of all particles equally
    if (positions_.empty()) return;
    for (size_t i = 0; i < velocities_.size(); ++i) {
        double inv_m = particle_inverse_mass(i);    // Get inverse mass
        velocities_[i] += impulse * inv_m;          // Δv = J / m
    }
}

void ParticleEntity::apply_impulse(const datatypes::Vector3& impulse, const datatypes::Vector3& world_point) {
    // Apply impulse at a point: affects nearest particle
    if (positions_.empty()) return;
    size_t nearest = nearest_particle(world_point);
    double inv_m = particle_inverse_mass(nearest);
    velocities_[nearest] += impulse * inv_m;        // Update velocity of that particle
}

void ParticleEntity::clear_forces() {
    // Reset all force accumulators to zero
    std::fill(forces_.begin(), forces_.end(), datatypes::Vector3(0.0));
}

void ParticleEntity::apply_force_to_particle(size_t i, const datatypes::Vector3& force) {
    // Apply force directly to a specific particle
    if (valid_index(i)) {
        forces_[i] += force;                        // Accumulate force
    }
}

void ParticleEntity::apply_impulse_to_particle(size_t i, const datatypes::Vector3& impulse) {
    // Apply impulse to a specific particle
    if (valid_index(i)) {
        double inv_m = particle_inverse_mass(i);
        velocities_[i] += impulse * inv_m;          // Δv = J / m
    }
}

//------------------------------------------------------------------------------
// Integration (explicit Euler)
//------------------------------------------------------------------------------
void ParticleEntity::integrate(double dt) {
    // Perform explicit Euler integration for all particles
    if (!is_dynamic() || !enabled_) return;         // Only integrate dynamic enabled entities
    integrate_velocity(dt);                         // Update velocities from forces
    integrate_position(dt);                         // Update positions from velocities
    clear_forces();                                 // Clear forces after integration
}

void ParticleEntity::integrate_velocity(double dt) {
    // Update velocities using accumulated forces: v += F/m * dt
    for (size_t i = 0; i < velocities_.size(); ++i) {
        double inv_m = particle_inverse_mass(i);    // Inverse mass
        if (inv_m > 0) {
            velocities_[i] += forces_[i] * (inv_m * dt); // a = F/m, Δv = a * dt
        }
    }
}

void ParticleEntity::integrate_position(double dt) {
    // Update positions using current velocities: x += v * dt
    for (size_t i = 0; i < positions_.size(); ++i) {
        positions_[i] += velocities_[i] * dt;       // Euler position update
    }
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
void ParticleEntity::reset() {
    // Restore particles to their initial state
    BaseEntity::reset();                            // Reset base transform and velocity
    if (initial_positions_.size() == positions_.size()) {
        positions_ = initial_positions_;            // Restore positions
        velocities_ = initial_velocities_;          // Restore velocities
        masses_ = initial_masses_;                  // Restore masses
    }
    clear_forces();                                 // Clear accumulated forces
}

//------------------------------------------------------------------------------
// Queries
//------------------------------------------------------------------------------
datatypes::AABB ParticleEntity::world_aabb() const {
    // Compute bounding box of all particles
    datatypes::AABB aabb;
    for (const auto& pos : positions_) {
        aabb.expand(pos);                           // Expand AABB by each particle position
    }
    return aabb;
}

size_t ParticleEntity::nearest_particle(const datatypes::Vector3& world_point) const {
    // Find index of particle closest to the given world point
    if (positions_.empty()) return 0;
    size_t nearest = 0;
    double min_dist_sq = (positions_[0] - world_point).squaredNorm();
    for (size_t i = 1; i < positions_.size(); ++i) {
        double dist_sq = (positions_[i] - world_point).squaredNorm();
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;                  // Update minimum distance
            nearest = i;                            // Update nearest index
        }
    }
    return nearest;                                 // Return index of closest particle
}

std::vector<size_t> ParticleEntity::particles_in_radius(const datatypes::Vector3& world_point, double radius) const {
    // Return indices of all particles within the given radius
    std::vector<size_t> result;
    double r2 = radius * radius;                    // Squared radius for comparison
    for (size_t i = 0; i < positions_.size(); ++i) {
        if ((positions_[i] - world_point).squaredNorm() <= r2) {
            result.push_back(i);                    // Particle is within radius
        }
    }
    return result;
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string ParticleEntity::repr() const {
    std::ostringstream oss;
    oss << "ParticleEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", count=" << particle_count()
        << ", total_mass=" << total_mass()
        << ")";
    return oss.str();
}

std::string ParticleEntity::str() const {
    return name() + " (Particles: " + std::to_string(particle_count()) + ")";
}

//------------------------------------------------------------------------------
// Private helpers
//------------------------------------------------------------------------------
void ParticleEntity::update_mass_properties() {
    // Could be used to update entity-level mass from particles (if needed)
    // For now, entity mass is independent; override if desired
}

} // namespace engine
} // namespace genesis