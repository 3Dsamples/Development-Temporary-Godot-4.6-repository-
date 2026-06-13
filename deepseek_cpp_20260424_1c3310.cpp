// genesis/engine/entities/hybrid_entity.cpp

#include "genesis/engine/entities/hybrid_entity.h" // Include corresponding header
#include "genesis/engine/entities/mpm_entity.h"    // MPMEntity for casting
#include "genesis/engine/entities/sph_entity.h"    // SPHEntity for casting
#include "genesis/engine/entities/fem_entity.h"    // FEMEntity for casting
#include "genesis/engine/entities/pbd_entity.h"    // PBDEntity for casting
#include "genesis/engine/entities/rigid_entity.h"  // RigidEntity for casting
#include <algorithm>                               // std::min, std::max, std::sort
#include <cmath>                                   // std::sqrt, std::pow
#include <sstream>                                 // std::ostringstream
#include <unordered_set>                           // std::unordered_set for visited tracking
#include <queue>                                   // std::priority_queue for nearest neighbors

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Helper: compute distance between two 3D points
//------------------------------------------------------------------------------
static double distance(const datatypes::Vector3& a, const datatypes::Vector3& b) {
    // Compute Euclidean distance between points
    return (a - b).norm();                         // Vector difference magnitude
}

//------------------------------------------------------------------------------
// HybridEntity construction
//------------------------------------------------------------------------------
HybridEntity::HybridEntity()
    : BaseEntity("HybridEntity")                   // Call base constructor with default name
{
    // Empty constructor body (members initialized via default constructors)
}

HybridEntity::HybridEntity(const std::string& name)
    : BaseEntity(name)                             // Base constructor with custom name
{
    // Empty constructor body
}

HybridEntity::~HybridEntity() {
    // Virtual destructor (smart pointers handle cleanup)
}

//------------------------------------------------------------------------------
// Component management
//------------------------------------------------------------------------------
void HybridEntity::set_primary_entity(std::shared_ptr<BaseEntity> entity) {
    // Set the primary physics representation
    primary_ = entity;                             // Store shared pointer
    if (primary_) {
        primary_->set_scene(scene_);               // Propagate scene if already set
    }
}

void HybridEntity::set_secondary_entity(std::shared_ptr<BaseEntity> entity) {
    // Set the secondary physics representation
    secondary_ = entity;                           // Store shared pointer
    if (secondary_) {
        secondary_->set_scene(scene_);             // Propagate scene if already set
    }
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void HybridEntity::set_hybrid_config(const HybridConfig& config) {
    // Store hybrid configuration
    config_ = config;                              // Copy config struct
}

//------------------------------------------------------------------------------
// Coupling management
//------------------------------------------------------------------------------
void HybridEntity::build_coupling_pairs() {
    // Automatically build coupling pairs based on entity types
    if (!primary_ || !secondary_) return;          // Both components required
    
    // Clear existing pairs before building new ones
    coupling_pairs_.clear();                       // Remove old pairs
    
    // Dispatch to appropriate builder based on dynamic types
    auto primary_mpm = std::dynamic_pointer_cast<MPMEntity>(primary_);
    auto primary_sph = std::dynamic_pointer_cast<SPHEntity>(primary_);
    auto primary_pbd = std::dynamic_pointer_cast<PBDEntity>(primary_);
    auto primary_rigid = std::dynamic_pointer_cast<RigidEntity>(primary_);
    auto secondary_fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    auto secondary_particles = std::dynamic_pointer_cast<ParticleEntity>(secondary_);
    
    if (primary_mpm && secondary_fem) {
        build_mpm_fem_coupling();                  // MPM particles to FEM nodes
    } else if (primary_sph && secondary_fem) {
        build_sph_fem_coupling();                  // SPH particles to FEM nodes
    } else if (primary_pbd && secondary_fem) {
        build_pbd_fem_coupling();                  // PBD particles to FEM nodes
    } else if (primary_rigid && secondary_particles) {
        build_rigid_particle_coupling();           // Rigid body to particles
    }
}

void HybridEntity::add_coupling_pair(const CouplingPair& pair) {
    // Manually add a predefined coupling pair
    coupling_pairs_.push_back(pair);               // Append to list
}

void HybridEntity::clear_coupling_pairs() {
    // Remove all coupling pairs
    coupling_pairs_.clear();                       // Clear vector
}

//------------------------------------------------------------------------------
// Synchronization and transfer
//------------------------------------------------------------------------------
void HybridEntity::transfer_primary_to_secondary() {
    // Transfer state from primary representation to secondary
    if (!primary_ || !secondary_) return;          // Both required
    
    // Dispatch based on types
    if (std::dynamic_pointer_cast<MPMEntity>(primary_) && std::dynamic_pointer_cast<FEMEntity>(secondary_)) {
        transfer_mpm_to_fem();                     // MPM -> FEM
    } else if (std::dynamic_pointer_cast<SPHEntity>(primary_) && std::dynamic_pointer_cast<FEMEntity>(secondary_)) {
        transfer_sph_to_fem();                     // SPH -> FEM
    } else if (std::dynamic_pointer_cast<FEMEntity>(primary_) && std::dynamic_pointer_cast<MPMEntity>(secondary_)) {
        transfer_fem_to_mpm();                     // FEM -> MPM
    } else if (std::dynamic_pointer_cast<FEMEntity>(primary_) && std::dynamic_pointer_cast<SPHEntity>(secondary_)) {
        transfer_fem_to_sph();                     // FEM -> SPH
    }
}

void HybridEntity::transfer_secondary_to_primary() {
    // Transfer state from secondary to primary
    if (!primary_ || !secondary_) return;          // Both required
    
    // Dispatch based on types (reverse of above)
    if (std::dynamic_pointer_cast<MPMEntity>(primary_) && std::dynamic_pointer_cast<FEMEntity>(secondary_)) {
        transfer_fem_to_mpm();                     // FEM -> MPM
    } else if (std::dynamic_pointer_cast<SPHEntity>(primary_) && std::dynamic_pointer_cast<FEMEntity>(secondary_)) {
        transfer_fem_to_sph();                     // FEM -> SPH
    } else if (std::dynamic_pointer_cast<FEMEntity>(primary_) && std::dynamic_pointer_cast<MPMEntity>(secondary_)) {
        transfer_mpm_to_fem();                     // MPM -> FEM
    } else if (std::dynamic_pointer_cast<FEMEntity>(primary_) && std::dynamic_pointer_cast<SPHEntity>(secondary_)) {
        transfer_sph_to_fem();                     // SPH -> FEM
    }
}

void HybridEntity::apply_coupling_forces(double dt) {
    // Apply mutual coupling forces between primary and secondary
    if (!primary_ || !secondary_) return;          // Both required
    if (config_.coupling_type != HybridCouplingType::TWO_WAY) return; // Only for two-way
    
    // Dispatch based on types
    if (std::dynamic_pointer_cast<MPMEntity>(primary_) && std::dynamic_pointer_cast<FEMEntity>(secondary_)) {
        apply_coupling_force_mpm_fem(dt);          // MPM <-> FEM
    } else if (std::dynamic_pointer_cast<SPHEntity>(primary_) && std::dynamic_pointer_cast<FEMEntity>(secondary_)) {
        apply_coupling_force_sph_fem(dt);          // SPH <-> FEM
    } else if (std::dynamic_pointer_cast<RigidEntity>(primary_)) {
        apply_coupling_force_rigid_particles(dt);  // Rigid <-> Particles
    }
}

void HybridEntity::synchronize_transforms() {
    // Synchronize world transforms between components (e.g., for rigid body coupling)
    if (!primary_ || !secondary_) return;          // Both required
    
    // If secondary should follow primary kinematically
    if (config_.coupling_type == HybridCouplingType::KINEMATIC) {
        secondary_->set_transform(primary_->transform()); // Copy transform
    }
}

//------------------------------------------------------------------------------
// Overrides from BaseEntity
//------------------------------------------------------------------------------
void HybridEntity::integrate(double dt) {
    // Integrate both components with coupling
    if (!primary_ || !secondary_) {
        if (primary_) primary_->integrate(dt);     // Only primary
        if (secondary_) secondary_->integrate(dt); // Only secondary
        return;
    }
    
    // Apply coupling forces before integration (if two-way)
    if (config_.coupling_type == HybridCouplingType::TWO_WAY) {
        apply_coupling_forces(dt);                 // Compute and apply penalty forces
    }
    
    // Transfer state if one-way or two-way
    if (config_.coupling_type == HybridCouplingType::ONE_WAY) {
        transfer_primary_to_secondary();           // Primary drives secondary
    }
    
    // Integrate primary component
    primary_->integrate(dt);                       // Advance primary physics
    
    // Transfer after primary integration if needed
    if (config_.coupling_type == HybridCouplingType::TWO_WAY) {
        transfer_primary_to_secondary();           // Update secondary from primary
    }
    
    // Integrate secondary component
    secondary_->integrate(dt);                     // Advance secondary physics
    
    // Transfer back if two-way
    if (config_.coupling_type == HybridCouplingType::TWO_WAY) {
        transfer_secondary_to_primary();           // Update primary from secondary
    }
    
    // Synchronize transforms for kinematic coupling
    if (config_.coupling_type == HybridCouplingType::KINEMATIC) {
        synchronize_transforms();                  // Align transforms
    }
}

void HybridEntity::integrate_velocity(double dt) {
    // Delegate velocity integration to components
    if (primary_) primary_->integrate_velocity(dt); // Primary velocity update
    if (secondary_) secondary_->integrate_velocity(dt); // Secondary velocity update
}

void HybridEntity::integrate_position(double dt) {
    // Delegate position integration to components
    if (primary_) primary_->integrate_position(dt); // Primary position update
    if (secondary_) secondary_->integrate_position(dt); // Secondary position update
}

void HybridEntity::reset() {
    // Reset both components to initial state
    BaseEntity::reset();                           // Reset base state
    if (primary_) primary_->reset();               // Reset primary
    if (secondary_) secondary_->reset();           // Reset secondary
    coupling_pairs_.clear();                       // Clear coupling (may need rebuild)
}

datatypes::AABB HybridEntity::world_aabb() const {
    // Return union of both components' AABBs
    datatypes::AABB aabb;                          // Start empty
    if (primary_) aabb.expand(primary_->world_aabb()); // Add primary bounds
    if (secondary_) aabb.expand(secondary_->world_aabb()); // Add secondary bounds
    return aabb;                                   // Return combined AABB
}

void HybridEntity::apply_force(const datatypes::Vector3& force) {
    // Forward force to both components (split equally? or to primary)
    if (primary_) primary_->apply_force(force * 0.5); // Half to primary
    if (secondary_) secondary_->apply_force(force * 0.5); // Half to secondary
}

void HybridEntity::apply_force(const datatypes::Vector3& force, const datatypes::Vector3& world_point) {
    // Forward force at point to both components
    if (primary_) primary_->apply_force(force * 0.5, world_point);
    if (secondary_) secondary_->apply_force(force * 0.5, world_point);
}

void HybridEntity::apply_torque(const datatypes::Vector3& torque) {
    // Forward torque (mainly relevant for rigid bodies)
    if (primary_) primary_->apply_torque(torque);
    if (secondary_) secondary_->apply_torque(torque);
}

void HybridEntity::apply_impulse(const datatypes::Vector3& impulse) {
    // Forward impulse
    if (primary_) primary_->apply_impulse(impulse * 0.5);
    if (secondary_) secondary_->apply_impulse(impulse * 0.5);
}

void HybridEntity::apply_impulse(const datatypes::Vector3& impulse, const datatypes::Vector3& world_point) {
    // Forward impulse at point
    if (primary_) primary_->apply_impulse(impulse * 0.5, world_point);
    if (secondary_) secondary_->apply_impulse(impulse * 0.5, world_point);
}

void HybridEntity::clear_forces() {
    // Clear force accumulators in both components
    if (primary_) primary_->clear_forces();        // Clear primary forces
    if (secondary_) secondary_->clear_forces();    // Clear secondary forces
}

void HybridEntity::set_transform(const datatypes::Transformr& t) {
    // Set transform on both components
    BaseEntity::set_transform(t);                  // Update base transform
    if (primary_) primary_->set_transform(t);      // Propagate to primary
    if (secondary_) secondary_->set_transform(t);  // Propagate to secondary
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string HybridEntity::repr() const {
    std::ostringstream oss;
    oss << "HybridEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", primary=" << (primary_ ? primary_->type_name() : "none")
        << ", secondary=" << (secondary_ ? secondary_->type_name() : "none")
        << ", coupling=" << (int)config_.coupling_type
        << ", pairs=" << coupling_pairs_.size()
        << ")";
    return oss.str();
}

std::string HybridEntity::str() const {
    return name() + " (Hybrid: " + (primary_ ? primary_->type_name() : "?") + "+" + (secondary_ ? secondary_->type_name() : "?") + ")";
}

//------------------------------------------------------------------------------
// Private: coupling builders
//------------------------------------------------------------------------------
void HybridEntity::build_mpm_fem_coupling() {
    // Build coupling pairs between MPM particles and FEM nodes
    auto mpm = std::dynamic_pointer_cast<MPMEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!mpm || !fem) return;
    
    const auto& mpm_pos = mpm->particle_positions(); // MPM particle positions
    const auto& fem_pos = fem->node_positions();     // FEM node positions
    size_t n_fem_nodes = fem_pos.size();
    
    coupling_pairs_.reserve(mpm_pos.size());       // Preallocate
    
    for (size_t p = 0; p < mpm_pos.size(); ++p) {
        CouplingPair pair;
        pair.primary_index = static_cast<uint32_t>(p); // Particle index
        
        // Find nearest FEM nodes (within coupling distance)
        std::vector<std::pair<uint32_t, double>> nearby; // (node_index, distance)
        for (size_t n = 0; n < n_fem_nodes; ++n) {
            double d = distance(mpm_pos[p], fem_pos[n]); // Distance
            if (d < config_.coupling_distance) {
                nearby.push_back({static_cast<uint32_t>(n), d}); // Store nearby node
            }
        }
        
        if (nearby.empty()) {
            // No nearby nodes, use closest single node
            double min_dist = std::numeric_limits<double>::max();
            uint32_t closest = 0;
            for (size_t n = 0; n < n_fem_nodes; ++n) {
                double d = distance(mpm_pos[p], fem_pos[n]);
                if (d < min_dist) {
                    min_dist = d;
                    closest = static_cast<uint32_t>(n);
                }
            }
            pair.secondary_indices.push_back(closest);
            pair.weights.push_back(1.0);
            pair.distance = min_dist;
        } else {
            // Use inverse distance weighting for multiple nodes
            double sum_inv_dist = 0.0;
            for (const auto& n : nearby) {
                sum_inv_dist += 1.0 / (n.second + 1e-6); // Avoid division by zero
            }
            for (const auto& n : nearby) {
                pair.secondary_indices.push_back(n.first);
                double w = (1.0 / (n.second + 1e-6)) / sum_inv_dist;
                pair.weights.push_back(w);
            }
            pair.distance = 0.0; // Not used for multiple
        }
        
        coupling_pairs_.push_back(pair);
    }
}

void HybridEntity::build_sph_fem_coupling() {
    // Similar to MPM-FEM but for SPH particles
    auto sph = std::dynamic_pointer_cast<SPHEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!sph || !fem) return;
    
    const auto& sph_pos = sph->particle_positions();
    const auto& fem_pos = fem->node_positions();
    size_t n_fem_nodes = fem_pos.size();
    
    coupling_pairs_.reserve(sph_pos.size());
    
    for (size_t p = 0; p < sph_pos.size(); ++p) {
        CouplingPair pair;
        pair.primary_index = static_cast<uint32_t>(p);
        
        // Find nearest FEM node (simpler than full interpolation)
        double min_dist = std::numeric_limits<double>::max();
        uint32_t closest = 0;
        for (size_t n = 0; n < n_fem_nodes; ++n) {
            double d = distance(sph_pos[p], fem_pos[n]);
            if (d < min_dist) {
                min_dist = d;
                closest = static_cast<uint32_t>(n);
            }
        }
        pair.secondary_indices.push_back(closest);
        pair.weights.push_back(1.0);
        pair.distance = min_dist;
        
        coupling_pairs_.push_back(pair);
    }
}

void HybridEntity::build_pbd_fem_coupling() {
    // Build coupling for PBD particles to FEM nodes
    auto pbd = std::dynamic_pointer_cast<PBDEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!pbd || !fem) return;
    
    const auto& pbd_pos = pbd->positions();
    const auto& fem_pos = fem->node_positions();
    size_t n_fem_nodes = fem_pos.size();
    
    coupling_pairs_.reserve(pbd_pos.size());
    
    for (size_t p = 0; p < pbd_pos.size(); ++p) {
        CouplingPair pair;
        pair.primary_index = static_cast<uint32_t>(p);
        
        // Find nearest FEM node
        double min_dist = std::numeric_limits<double>::max();
        uint32_t closest = 0;
        for (size_t n = 0; n < n_fem_nodes; ++n) {
            double d = distance(pbd_pos[p], fem_pos[n]);
            if (d < min_dist) {
                min_dist = d;
                closest = static_cast<uint32_t>(n);
            }
        }
        pair.secondary_indices.push_back(closest);
        pair.weights.push_back(1.0);
        pair.distance = min_dist;
        
        coupling_pairs_.push_back(pair);
    }
}

void HybridEntity::build_rigid_particle_coupling() {
    // Rigid body coupled to all particles (secondary)
    auto rigid = std::dynamic_pointer_cast<RigidEntity>(primary_);
    auto particles = std::dynamic_pointer_cast<ParticleEntity>(secondary_);
    if (!rigid || !particles) return;
    
    const auto& part_pos = particles->particle_positions();
    coupling_pairs_.resize(part_pos.size());
    
    for (size_t p = 0; p < part_pos.size(); ++p) {
        CouplingPair pair;
        pair.primary_index = 0; // Only one rigid body
        pair.secondary_indices.push_back(static_cast<uint32_t>(p));
        pair.weights.push_back(1.0);
        coupling_pairs_[p] = pair;
    }
}

//------------------------------------------------------------------------------
// Private: transfer helpers
//------------------------------------------------------------------------------
void HybridEntity::transfer_mpm_to_fem() {
    // Transfer MPM particle velocities to FEM nodes via interpolation
    auto mpm = std::dynamic_pointer_cast<MPMEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!mpm || !fem) return;
    
    const auto& mpm_vel = mpm->particle_velocities();
    const auto& mpm_pos = mpm->particle_positions();
    std::vector<datatypes::Vector3> fem_vel(fem->node_count(), datatypes::Vector3(0.0));
    std::vector<double> weight_sum(fem->node_count(), 0.0);
    
    for (size_t p = 0; p < coupling_pairs_.size() && p < mpm_pos.size(); ++p) {
        const auto& pair = coupling_pairs_[p];
        datatypes::Vector3 p_vel = mpm_vel[p];
        for (size_t j = 0; j < pair.secondary_indices.size(); ++j) {
            uint32_t node_idx = pair.secondary_indices[j];
            double w = pair.weights[j];
            fem_vel[node_idx] += p_vel * w;
            weight_sum[node_idx] += w;
        }
    }
    
    // Normalize and apply
    for (size_t n = 0; n < fem_vel.size(); ++n) {
        if (weight_sum[n] > 0) {
            fem_vel[n] /= weight_sum[n];
            fem->set_node_velocity(n, fem_vel[n]);
        }
    }
}

void HybridEntity::transfer_fem_to_mpm() {
    // Transfer FEM node velocities to MPM particles
    auto mpm = std::dynamic_pointer_cast<MPMEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!mpm || !fem) return;
    
    const auto& fem_vel = fem->node_velocities();
    auto mpm_vel = mpm->particle_velocities();
    
    for (size_t p = 0; p < coupling_pairs_.size() && p < mpm_vel.size(); ++p) {
        const auto& pair = coupling_pairs_[p];
        datatypes::Vector3 interp_vel(0.0);
        double total_weight = 0.0;
        for (size_t j = 0; j < pair.secondary_indices.size(); ++j) {
            uint32_t node_idx = pair.secondary_indices[j];
            double w = pair.weights[j];
            interp_vel += fem_vel[node_idx] * w;
            total_weight += w;
        }
        if (total_weight > 0) {
            interp_vel /= total_weight;
            // Blend with current velocity
            mpm_vel[p] = mpm_vel[p] * (1.0 - config_.transfer_relaxation) + interp_vel * config_.transfer_relaxation;
        }
    }
    mpm->set_velocities(mpm_vel);
}

void HybridEntity::transfer_sph_to_fem() {
    // Transfer SPH velocities to FEM (similar to MPM)
    auto sph = std::dynamic_pointer_cast<SPHEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!sph || !fem) return;
    
    const auto& sph_vel = sph->particle_velocities();
    std::vector<datatypes::Vector3> fem_vel(fem->node_count(), datatypes::Vector3(0.0));
    std::vector<double> weight_sum(fem->node_count(), 0.0);
    
    for (size_t p = 0; p < coupling_pairs_.size(); ++p) {
        const auto& pair = coupling_pairs_[p];
        if (pair.secondary_indices.empty()) continue;
        uint32_t node_idx = pair.secondary_indices[0];
        fem_vel[node_idx] += sph_vel[p];
        weight_sum[node_idx] += 1.0;
    }
    for (size_t n = 0; n < fem_vel.size(); ++n) {
        if (weight_sum[n] > 0) {
            fem_vel[n] /= weight_sum[n];
            fem->set_node_velocity(n, fem_vel[n]);
        }
    }
}

void HybridEntity::transfer_fem_to_sph() {
    // Transfer FEM velocities to SPH particles
    auto sph = std::dynamic_pointer_cast<SPHEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!sph || !fem) return;
    
    const auto& fem_vel = fem->node_velocities();
    auto sph_vel = sph->particle_velocities();
    
    for (size_t p = 0; p < coupling_pairs_.size() && p < sph_vel.size(); ++p) {
        const auto& pair = coupling_pairs_[p];
        if (pair.secondary_indices.empty()) continue;
        uint32_t node_idx = pair.secondary_indices[0];
        sph_vel[p] = sph_vel[p] * (1.0 - config_.transfer_relaxation) + fem_vel[node_idx] * config_.transfer_relaxation;
    }
    sph->set_velocities(sph_vel);
}

//------------------------------------------------------------------------------
// Private: coupling forces
//------------------------------------------------------------------------------
void HybridEntity::apply_coupling_force_mpm_fem(double dt) {
    // Apply penalty forces between MPM particles and coupled FEM nodes
    auto mpm = std::dynamic_pointer_cast<MPMEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!mpm || !fem) return;
    
    const auto& mpm_pos = mpm->particle_positions();
    const auto& fem_pos = fem->node_positions();
    const auto& fem_vel = fem->node_velocities();
    auto mpm_vel = mpm->particle_velocities();
    
    double stiffness = config_.coupling_stiffness;
    double damping = config_.coupling_damping;
    
    for (const auto& pair : coupling_pairs_) {
        uint32_t p_idx = pair.primary_index;
        if (p_idx >= mpm_pos.size()) continue;
        
        datatypes::Vector3 p_pos = mpm_pos[p_idx];
        datatypes::Vector3 p_vel = mpm_vel[p_idx];
        
        for (size_t j = 0; j < pair.secondary_indices.size(); ++j) {
            uint32_t node_idx = pair.secondary_indices[j];
            if (node_idx >= fem_pos.size()) continue;
            
            datatypes::Vector3 node_pos = fem_pos[node_idx];
            datatypes::Vector3 node_vel = fem_vel[node_idx];
            
            datatypes::Vector3 delta = node_pos - p_pos;
            double dist = delta.norm();
            if (dist < 1e-12) continue;
            
            datatypes::Vector3 dir = delta / dist;
            
            // Penalty spring force
            datatypes::Vector3 spring_force = dir * (stiffness * dist);
            
            // Relative velocity damping
            datatypes::Vector3 rel_vel = node_vel - p_vel;
            double vn = rel_vel.dot(dir);
            datatypes::Vector3 damping_force = dir * (damping * vn);
            
            datatypes::Vector3 total_force = spring_force + damping_force;
            
            // Apply equal and opposite
            mpm->add_particle_force(p_idx, total_force);
            fem->add_node_force(node_idx, -total_force);
        }
    }
}

void HybridEntity::apply_coupling_force_sph_fem(double dt) {
    // Similar to MPM-FEM but for SPH particles
    auto sph = std::dynamic_pointer_cast<SPHEntity>(primary_);
    auto fem = std::dynamic_pointer_cast<FEMEntity>(secondary_);
    if (!sph || !fem) return;
    
    const auto& sph_pos = sph->particle_positions();
    const auto& fem_pos = fem->node_positions();
    const auto& fem_vel = fem->node_velocities();
    auto sph_vel = sph->particle_velocities();
    
    double stiffness = config_.coupling_stiffness;
    double damping = config_.coupling_damping;
    
    for (const auto& pair : coupling_pairs_) {
        uint32_t p_idx = pair.primary_index;
        if (p_idx >= sph_pos.size() || pair.secondary_indices.empty()) continue;
        
        uint32_t node_idx = pair.secondary_indices[0];
        if (node_idx >= fem_pos.size()) continue;
        
        datatypes::Vector3 delta = fem_pos[node_idx] - sph_pos[p_idx];
        double dist = delta.norm();
        if (dist < 1e-12) continue;
        
        datatypes::Vector3 dir = delta / dist;
        datatypes::Vector3 spring_force = dir * (stiffness * dist);
        datatypes::Vector3 rel_vel = fem_vel[node_idx] - sph_vel[p_idx];
        double vn = rel_vel.dot(dir);
        datatypes::Vector3 damping_force = dir * (damping * vn);
        datatypes::Vector3 total_force = spring_force + damping_force;
        
        sph->add_particle_force(p_idx, total_force);
        fem->add_node_force(node_idx, -total_force);
    }
}

void HybridEntity::apply_coupling_force_rigid_particles(double dt) {
    // Apply forces between rigid body and particles (secondary)
    auto rigid = std::dynamic_pointer_cast<RigidEntity>(primary_);
    auto particles = std::dynamic_pointer_cast<ParticleEntity>(secondary_);
    if (!rigid || !particles) return;
    
    const auto& part_pos = particles->particle_positions();
    const auto& part_vel = particles->particle_velocities();
    datatypes::Transformr rigid_tf = rigid->transform();
    
    double stiffness = config_.coupling_stiffness;
    double damping = config_.coupling_damping;
    
    for (size_t p = 0; p < part_pos.size() && p < coupling_pairs_.size(); ++p) {
        datatypes::Vector3 target_pos = rigid_tf.transformPoint(datatypes::Vector3(0.0)); // assume coupling to COM for simplicity
        datatypes::Vector3 delta = target_pos - part_pos[p];
        double dist = delta.norm();
        if (dist < 1e-12) continue;
        
        datatypes::Vector3 dir = delta / dist;
        datatypes::Vector3 spring_force = dir * (stiffness * dist);
        datatypes::Vector3 rel_vel = rigid->velocity() - part_vel[p];
        double vn = rel_vel.dot(dir);
        datatypes::Vector3 damping_force = dir * (damping * vn);
        datatypes::Vector3 total_force = spring_force + damping_force;
        
        particles->apply_force_to_particle(p, total_force);
        rigid->apply_force(-total_force, part_pos[p]);
    }
}

} // namespace engine
} // namespace genesis