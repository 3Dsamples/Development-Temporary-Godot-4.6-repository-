// genesis/engine/entities/pbd_entity.cpp

#include "genesis/engine/entities/pbd_entity.h"     // Include corresponding header
#include "genesis/engine/solvers/pbd_solver.h"      // For constraint class definitions
#include "genesis/engine/entities/rigid_entity.h"   // For attachment constraint target
#include <algorithm>                                // std::copy, std::fill, std::min, std::max
#include <cmath>                                    // std::acos, std::sqrt, M_PI
#include <sstream>                                  // std::ostringstream for repr

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// PBDEntity construction
//------------------------------------------------------------------------------
PBDEntity::PBDEntity()
    : ParticleEntity("PBDEntity")                   // Call base constructor with default name
{
    // Initialize inverse mass array (empty initially)
    inv_mass_.clear();
}

PBDEntity::PBDEntity(const std::string& name)
    : ParticleEntity(name)                          // Base constructor with custom name
{
    inv_mass_.clear();
}

PBDEntity::~PBDEntity() {
    // Virtual destructor (constraints are shared_ptr, automatically cleaned up)
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void PBDEntity::set_pbd_config(const PBDConfig& config) {
    // Store PBD configuration
    pbd_config_ = config;                           // Copy config struct
}

//------------------------------------------------------------------------------
// Constraint management
//------------------------------------------------------------------------------
void PBDEntity::add_distance_constraint(int idx0, int idx1, double rest_length, double stiffness) {
    // Add a distance constraint between two particles
    if (idx0 < 0 || idx0 >= static_cast<int>(particle_count())) return; // Validate indices
    if (idx1 < 0 || idx1 >= static_cast<int>(particle_count())) return;
    
    // Compute rest length from current positions if not provided
    if (rest_length < 0.0) {
        rest_length = compute_rest_length(idx0, idx1); // Distance between current positions
    }
    // Use default stiffness if none provided
    if (stiffness < 0.0) {
        stiffness = pbd_config_.default_stiffness;  // Fallback to config default
    }
    // Create and store the constraint
    auto constraint = std::make_shared<DistanceConstraint>(idx0, idx1, rest_length, stiffness);
    constraints_.push_back(constraint);             // Add to constraint list
}

void PBDEntity::add_bending_constraint(int idx0, int idx1, int idx2, int idx3, double rest_angle, double stiffness) {
    // Add a bending (dihedral angle) constraint between two triangles
    if (idx0 < 0 || idx0 >= static_cast<int>(particle_count())) return;
    if (idx1 < 0 || idx1 >= static_cast<int>(particle_count())) return;
    if (idx2 < 0 || idx2 >= static_cast<int>(particle_count())) return;
    if (idx3 < 0 || idx3 >= static_cast<int>(particle_count())) return;
    
    // Compute rest angle if not provided
    if (rest_angle < -M_PI) {                       // Sentinel for "not provided"
        rest_angle = compute_rest_angle(idx0, idx1, idx2, idx3);
    }
    if (stiffness < 0.0) {
        stiffness = pbd_config_.default_stiffness;
    }
    auto constraint = std::make_shared<BendingConstraint>(idx0, idx1, idx2, idx3, rest_angle, stiffness);
    constraints_.push_back(constraint);
}

void PBDEntity::add_volume_constraint(int idx0, int idx1, int idx2, int idx3, double rest_volume, double stiffness) {
    // Add a volume preservation constraint for a tetrahedron
    if (idx0 < 0 || idx0 >= static_cast<int>(particle_count())) return;
    if (idx1 < 0 || idx1 >= static_cast<int>(particle_count())) return;
    if (idx2 < 0 || idx2 >= static_cast<int>(particle_count())) return;
    if (idx3 < 0 || idx3 >= static_cast<int>(particle_count())) return;
    
    // Compute rest volume if not provided
    if (rest_volume < 0.0) {
        rest_volume = compute_rest_volume(idx0, idx1, idx2, idx3);
    }
    if (stiffness < 0.0) {
        stiffness = pbd_config_.default_stiffness;
    }
    auto constraint = std::make_shared<VolumeConstraint>(idx0, idx1, idx2, idx3, rest_volume, stiffness);
    constraints_.push_back(constraint);
}

void PBDEntity::add_shape_matching_constraint(const std::vector<int>& indices, double stiffness) {
    // Add a shape matching constraint for a cluster of particles
    if (indices.empty()) return;
    // Validate all indices
    for (int idx : indices) {
        if (idx < 0 || idx >= static_cast<int>(particle_count())) return;
    }
    if (stiffness < 0.0) {
        stiffness = pbd_config_.default_stiffness;
    }
    // Collect rest positions from current state
    std::vector<datatypes::Vector3> rest_positions;
    rest_positions.reserve(indices.size());
    for (int idx : indices) {
        rest_positions.push_back(positions_[idx]);  // Current position becomes rest shape
    }
    auto constraint = std::make_shared<ShapeMatchingConstraint>(indices, rest_positions, stiffness);
    constraints_.push_back(constraint);
}

void PBDEntity::add_pin_constraint(int particle_idx, const datatypes::Vector3& fixed_position) {
    // Pin a particle to a fixed world position
    if (particle_idx < 0 || particle_idx >= static_cast<int>(particle_count())) return;
    // A pin constraint can be implemented as a zero-length distance constraint to a fixed point
    // We'll create a DistanceConstraint with a dummy particle that has infinite mass
    // For simplicity, we create a specialized PinConstraint (defined in pbd_solver.h)
    auto constraint = std::make_shared<PinConstraint>(particle_idx, fixed_position);
    constraints_.push_back(constraint);
}

void PBDEntity::add_attachment_constraint(int particle_idx, std::shared_ptr<BaseEntity> target, const datatypes::Vector3& local_offset) {
    // Attach a particle to another entity (e.g., rigid body)
    if (particle_idx < 0 || particle_idx >= static_cast<int>(particle_count())) return;
    if (!target) return;
    auto constraint = std::make_shared<AttachmentConstraint>(particle_idx, target, local_offset);
    constraints_.push_back(constraint);
}

void PBDEntity::clear_constraints() {
    // Remove all constraints
    constraints_.clear();                           // Clear vector (shared_ptrs will delete)
}

//------------------------------------------------------------------------------
// PBD-specific state
//------------------------------------------------------------------------------
void PBDEntity::set_prev_positions(const std::vector<datatypes::Vector3>& prev) {
    // Set previous positions array
    if (prev.size() == prev_positions_.size()) {
        prev_positions_ = prev;                     // Copy entire vector
    } else {
        size_t n = std::min(prev.size(), prev_positions_.size());
        std::copy(prev.begin(), prev.begin() + n, prev_positions_.begin());
    }
}

void PBDEntity::store_prev_positions() {
    // Copy current positions to prev_positions (called before position update)
    prev_positions_ = positions_;                   // Deep copy
}

void PBDEntity::clear_delta() {
    // Reset delta array to zero
    std::fill(delta_.begin(), delta_.end(), datatypes::Vector3(0.0));
}

std::vector<double> PBDEntity::inverse_masses() const {
    // Return cached inverse masses (compute if needed)
    return inv_mass_;                               // Already computed via compute_inverse_masses
}

void PBDEntity::compute_inverse_masses() {
    // Compute inverse masses from mass array
    inv_mass_.resize(masses_.size());
    for (size_t i = 0; i < masses_.size(); ++i) {
        inv_mass_[i] = (masses_[i] > 1e-12) ? 1.0 / masses_[i] : 0.0;
    }
}

//------------------------------------------------------------------------------
// Overrides
//------------------------------------------------------------------------------
void PBDEntity::integrate(double dt) {
    // PBD integration is handled entirely by PBDSolver.
    // This method is intentionally minimal.
    (void)dt;                                       // Suppress unused warning
    // No-op: solver directly accesses particle arrays and constraints.
}

void PBDEntity::reset() {
    // Reset to initial state
    ParticleEntity::reset();                        // Reset positions, velocities, masses
    // Recompute inverse masses
    compute_inverse_masses();                       // Update inv_mass_ from masses_
    // Reset PBD state arrays
    prev_positions_ = positions_;                   // Initialize prev positions to current
    delta_.assign(particle_count(), datatypes::Vector3(0.0)); // Reset delta to zero
    // Constraints remain unchanged (they are part of the entity definition)
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string PBDEntity::repr() const {
    std::ostringstream oss;
    oss << "PBDEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", particles=" << particle_count()
        << ", constraints=" << constraints_.size()
        << ", total_mass=" << total_mass()
        << ")";
    return oss.str();
}

std::string PBDEntity::str() const {
    return name() + " (PBD: " + std::to_string(particle_count()) + " particles, " + std::to_string(constraints_.size()) + " constraints)";
}

//------------------------------------------------------------------------------
// Private helper methods
//------------------------------------------------------------------------------
double PBDEntity::compute_rest_length(int idx0, int idx1) const {
    // Compute Euclidean distance between two particles' current positions
    if (idx0 < 0 || idx0 >= static_cast<int>(positions_.size())) return 0.0;
    if (idx1 < 0 || idx1 >= static_cast<int>(positions_.size())) return 0.0;
    return (positions_[idx0] - positions_[idx1]).norm(); // Distance
}

double PBDEntity::compute_rest_angle(int idx0, int idx1, int idx2, int idx3) const {
    // Compute dihedral angle between triangles (0,2,3) and (1,3,2) - standard bending constraint order
    datatypes::Vector3 p0 = positions_[idx0];
    datatypes::Vector3 p1 = positions_[idx1];
    datatypes::Vector3 p2 = positions_[idx2];
    datatypes::Vector3 p3 = positions_[idx3];
    
    // Normal of first triangle (p0, p2, p3)
    datatypes::Vector3 n1 = (p2 - p0).cross(p3 - p0);
    double len1 = n1.norm();
    if (len1 < 1e-12) return 0.0;
    n1 /= len1;
    
    // Normal of second triangle (p1, p3, p2)
    datatypes::Vector3 n2 = (p3 - p1).cross(p2 - p1);
    double len2 = n2.norm();
    if (len2 < 1e-12) return 0.0;
    n2 /= len2;
    
    // Shared edge direction
    datatypes::Vector3 e = p3 - p2;
    double e_len = e.norm();
    if (e_len < 1e-12) return 0.0;
    e /= e_len;
    
    // Compute angle using cross and dot
    double cos_theta = n1.dot(n2);
    double sin_theta = n1.cross(n2).dot(e);
    return std::atan2(sin_theta, cos_theta);        // Dihedral angle in [-π, π]
}

double PBDEntity::compute_rest_volume(int idx0, int idx1, int idx2, int idx3) const {
    // Compute signed volume of tetrahedron
    datatypes::Vector3 p0 = positions_[idx0];
    datatypes::Vector3 p1 = positions_[idx1];
    datatypes::Vector3 p2 = positions_[idx2];
    datatypes::Vector3 p3 = positions_[idx3];
    
    datatypes::Vector3 a = p1 - p0;
    datatypes::Vector3 b = p2 - p0;
    datatypes::Vector3 c = p3 - p0;
    return std::abs(a.dot(b.cross(c))) / 6.0;       // Absolute volume
}

} // namespace engine
} // namespace genesis