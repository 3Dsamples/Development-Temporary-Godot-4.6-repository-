// genesis/engine/entities/mpm_entity.cpp

#include "genesis/engine/entities/mpm_entity.h"     // Include corresponding header
#include <algorithm>                                // std::copy, std::fill, std::max, std::min
#include <cmath>                                    // std::pow, M_PI
#include <sstream>                                  // std::ostringstream for repr
#include <numeric>                                  // std::iota (not used but included)

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// MPMEntity construction
//------------------------------------------------------------------------------
MPMEntity::MPMEntity()
    : ParticleEntity("MPMEntity")                   // Call base class constructor with default name
{
    // Empty constructor body (arrays are empty initially)
}

MPMEntity::MPMEntity(const std::string& name)
    : ParticleEntity(name)                          // Base constructor with custom name
{
    // Empty constructor body
}

MPMEntity::~MPMEntity() {
    // Virtual destructor (cleanup handled by vector destructors)
}

//------------------------------------------------------------------------------
// Particle management
//------------------------------------------------------------------------------
void MPMEntity::add_particles(const std::vector<datatypes::Vector3>& positions,
                              const std::vector<datatypes::Vector3>& velocities,
                              const std::vector<double>& masses,
                              const std::vector<double>& volumes) {
    // Add new particles with full MPM state
    size_t old_size = positions_.size();            // Current number of particles
    size_t add_count = positions.size();            // Number to add
    
    // Resize all particle arrays to accommodate new particles
    resize_arrays(old_size + add_count);            // Allocate space
    
    // Copy position data
    std::copy(positions.begin(), positions.end(), positions_.begin() + old_size);
    // Copy velocity data
    std::copy(velocities.begin(), velocities.end(), velocities_.begin() + old_size);
    // Copy mass data
    std::copy(masses.begin(), masses.end(), masses_.begin() + old_size);
    // Copy volume data
    std::copy(volumes.begin(), volumes.end(), volumes_.begin() + old_size);
    
    // Initialize deformation state for new particles to identity/defaults
    initialize_particle_defaults(old_size, add_count);
}

void MPMEntity::remove_particles(const std::vector<bool>& mask) {
    // Remove particles whose mask entry is true (preserving order of remaining)
    if (mask.size() != positions_.size()) return;   // Mask size must match
    
    size_t write_idx = 0;                           // Index to write kept particles
    for (size_t i = 0; i < positions_.size(); ++i) {
        if (!mask[i]) {                             // Keep this particle
            if (write_idx != i) {
                // Move particle data to the write position
                positions_[write_idx] = positions_[i];
                velocities_[write_idx] = velocities_[i];
                masses_[write_idx] = masses_[i];
                volumes_[write_idx] = volumes_[i];
                F_[write_idx] = F_[i];
                Fp_[write_idx] = Fp_[i];
                C_[write_idx] = C_[i];
                Jp_[write_idx] = Jp_[i];
                material_ids_[write_idx] = material_ids_[i];
                forces_[write_idx] = forces_[i];
            }
            ++write_idx;                            // Advance write position
        }
    }
    // Resize all arrays to the new size (number of kept particles)
    resize_arrays(write_idx);                       // Shrink to fit
}

double MPMEntity::particle_volume(size_t i) const {
    // Return volume of particle i
    if (i < volumes_.size()) {
        return volumes_[i];                         // Return stored volume
    }
    return 0.0;                                     // Out of bounds
}

void MPMEntity::set_particle_volume(size_t i, double volume) {
    // Set volume of particle i
    if (i < volumes_.size()) {
        volumes_[i] = volume;                       // Update volume
    }
}

//------------------------------------------------------------------------------
// Deformation state accessors
//------------------------------------------------------------------------------
void MPMEntity::set_deformation_gradients(const std::vector<datatypes::Matrix3r>& F) {
    // Set all deformation gradients at once
    if (F.size() == F_.size()) {
        F_ = F;                                     // Copy entire vector
    } else {
        size_t n = std::min(F.size(), F_.size());   // Copy up to smaller size
        std::copy(F.begin(), F.begin() + n, F_.begin());
    }
}

datatypes::Matrix3r MPMEntity::deformation_gradient(size_t i) const {
    // Return deformation gradient of particle i
    if (i < F_.size()) {
        return F_[i];                               // Return stored matrix
    }
    return datatypes::Matrix3r(1.0);                // Identity as fallback
}

void MPMEntity::set_deformation_gradient(size_t i, const datatypes::Matrix3r& F) {
    // Set deformation gradient for a single particle
    if (i < F_.size()) {
        F_[i] = F;                                  // Update matrix
    }
}

void MPMEntity::set_plastic_strains(const std::vector<datatypes::Matrix3r>& Fp) {
    // Set all plastic strain tensors
    if (Fp.size() == Fp_.size()) {
        Fp_ = Fp;                                   // Copy entire vector
    } else {
        size_t n = std::min(Fp.size(), Fp_.size());
        std::copy(Fp.begin(), Fp.begin() + n, Fp_.begin());
    }
}

datatypes::Matrix3r MPMEntity::plastic_strain(size_t i) const {
    // Return plastic strain of particle i
    if (i < Fp_.size()) {
        return Fp_[i];                              // Return stored matrix
    }
    return datatypes::Matrix3r(1.0);                // Identity (no plastic strain)
}

void MPMEntity::set_plastic_strain(size_t i, const datatypes::Matrix3r& Fp) {
    // Set plastic strain for a single particle
    if (i < Fp_.size()) {
        Fp_[i] = Fp;                                // Update matrix
    }
}

void MPMEntity::set_affine_velocities(const std::vector<datatypes::Matrix3r>& C) {
    // Set all affine velocity matrices (APIC)
    if (C.size() == C_.size()) {
        C_ = C;                                     // Copy entire vector
    } else {
        size_t n = std::min(C.size(), C_.size());
        std::copy(C.begin(), C.begin() + n, C_.begin());
    }
}

datatypes::Matrix3r MPMEntity::affine_velocity(size_t i) const {
    // Return affine velocity matrix of particle i
    if (i < C_.size()) {
        return C_[i];                               // Return stored matrix
    }
    return datatypes::Matrix3r(0.0);                // Zero matrix as fallback
}

void MPMEntity::set_affine_velocity(size_t i, const datatypes::Matrix3r& C) {
    // Set affine velocity matrix for a single particle
    if (i < C_.size()) {
        C_[i] = C;                                  // Update matrix
    }
}

void MPMEntity::set_plastic_jacobians(const std::vector<double>& Jp) {
    // Set all plastic Jacobian determinants
    if (Jp.size() == Jp_.size()) {
        Jp_ = Jp;                                   // Copy entire vector
    } else {
        size_t n = std::min(Jp.size(), Jp_.size());
        std::copy(Jp.begin(), Jp.begin() + n, Jp_.begin());
    }
}

double MPMEntity::plastic_jacobian(size_t i) const {
    // Return plastic Jacobian of particle i
    if (i < Jp_.size()) {
        return Jp_[i];                              // Return stored value
    }
    return 1.0;                                     // Default: no volume change
}

void MPMEntity::set_plastic_jacobian(size_t i, double Jp) {
    // Set plastic Jacobian for a single particle
    if (i < Jp_.size()) {
        Jp_[i] = Jp;                                // Update value
    }
}

//------------------------------------------------------------------------------
// Bulk state operations
//------------------------------------------------------------------------------
void MPMEntity::set_particle_states(const std::vector<datatypes::Vector3>& positions,
                                    const std::vector<datatypes::Vector3>& velocities,
                                    const std::vector<datatypes::Matrix3r>& F,
                                    const std::vector<datatypes::Matrix3r>& C) {
    // Set multiple state arrays at once (used by solver for efficiency)
    if (positions.size() == positions_.size()) {
        positions_ = positions;                     // Copy positions
    }
    if (velocities.size() == velocities_.size()) {
        velocities_ = velocities;                   // Copy velocities
    }
    if (F.size() == F_.size()) {
        F_ = F;                                     // Copy deformation gradients
    }
    if (C.size() == C_.size()) {
        C_ = C;                                     // Copy affine matrices
    }
}

//------------------------------------------------------------------------------
// Material configuration
//------------------------------------------------------------------------------
void MPMEntity::set_mpm_config(const MPMConfig& config) {
    // Set the default material configuration for this entity
    mpm_config_ = config;                           // Copy config struct
}

void MPMEntity::set_particle_material(size_t i, uint32_t material_id) {
    // Set material ID for a specific particle
    if (i < material_ids_.size()) {
        material_ids_[i] = material_id;             // Update material ID
    }
}

uint32_t MPMEntity::particle_material(size_t i) const {
    // Get material ID of particle i
    if (i < material_ids_.size()) {
        return material_ids_[i];                    // Return stored ID
    }
    return 0;                                       // Default material 0
}

//------------------------------------------------------------------------------
// Overrides
//------------------------------------------------------------------------------
void MPMEntity::integrate(double dt) {
    // MPM integration is handled entirely by MPMSolver.
    // This method is intentionally minimal (solver will directly access particle arrays).
    (void)dt;                                       // Suppress unused warning
    // No-op: solver takes care of all MPM-specific integration.
}

void MPMEntity::reset() {
    // Reset entity to initial state
    ParticleEntity::reset();                        // Reset base particle state
    // Reset deformation gradients to identity
    for (auto& f : F_) {
        f = datatypes::Matrix3r(1.0);               // Identity matrix
    }
    // Reset plastic strains to identity
    for (auto& fp : Fp_) {
        fp = datatypes::Matrix3r(1.0);              // Identity matrix
    }
    // Reset affine matrices to zero
    for (auto& c : C_) {
        c = datatypes::Matrix3r(0.0);               // Zero matrix
    }
    // Reset plastic Jacobians to 1.0
    std::fill(Jp_.begin(), Jp_.end(), 1.0);         // No plastic volume change
}

datatypes::AABB MPMEntity::world_aabb() const {
    // Compute world-space AABB from all particle positions
    datatypes::AABB aabb;                           // Start empty
    for (const auto& pos : positions_) {
        aabb.expand(pos);                           // Expand by each particle position
    }
    return aabb;                                    // Return bounding box
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string MPMEntity::repr() const {
    std::ostringstream oss;
    oss << "MPMEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", particles=" << particle_count()
        << ", material=" << (int)mpm_config_.material_model
        << ", mass=" << total_mass()
        << ")";
    return oss.str();
}

std::string MPMEntity::str() const {
    return name() + " (MPM: " + std::to_string(particle_count()) + " particles)";
}

//------------------------------------------------------------------------------
// Private helper methods
//------------------------------------------------------------------------------
void MPMEntity::resize_arrays(size_t new_size) {
    // Resize all particle state vectors to the given size
    positions_.resize(new_size, datatypes::Vector3(0.0));
    velocities_.resize(new_size, datatypes::Vector3(0.0));
    masses_.resize(new_size, 0.0);
    forces_.resize(new_size, datatypes::Vector3(0.0));
    volumes_.resize(new_size, 0.0);
    F_.resize(new_size, datatypes::Matrix3r(1.0));
    Fp_.resize(new_size, datatypes::Matrix3r(1.0));
    C_.resize(new_size, datatypes::Matrix3r(0.0));
    Jp_.resize(new_size, 1.0);
    material_ids_.resize(new_size, 0);
}

void MPMEntity::initialize_particle_defaults(size_t start, size_t count) {
    // Initialize state for newly added particles in the given range
    for (size_t i = start; i < start + count; ++i) {
        F_[i] = datatypes::Matrix3r(1.0);           // Identity deformation gradient
        Fp_[i] = datatypes::Matrix3r(1.0);          // Identity plastic strain
        C_[i] = datatypes::Matrix3r(0.0);           // Zero affine matrix
        Jp_[i] = 1.0;                               // Unit plastic Jacobian
        material_ids_[i] = 0;                       // Default material ID
        forces_[i] = datatypes::Vector3(0.0);       // Zero initial force
    }
}

} // namespace engine
} // namespace genesis