// genesis/engine/entities/soft_tissue_entity.cpp
#include "genesis/engine/entities/soft_tissue_entity.h" // Corresponding header
#include <algorithm>                                   // std::max, std::min, std::copy
#include <cmath>                                       // std::sqrt, std::pow
#include <numeric>                                     // std::accumulate
#include <sstream>                                     // std::ostringstream
#include <limits>                                      // std::numeric_limits

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Helper: compute cross product of two Vector3
//------------------------------------------------------------------------------
static datatypes::Vector3 cross(const datatypes::Vector3& a, const datatypes::Vector3& b) {
    return datatypes::Vector3(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    );
}

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------
SoftTissueEntity::SoftTissueEntity()
    : FEMEntity("SoftTissueEntity")                  // Base constructor with default name
{
    // External forces array is already sized by FEMEntity, we add our own
    external_poke_forces_.resize(node_count(), datatypes::Vector3(0.0));
    original_rest_positions_ = rest_positions();     // Capture initial rest shape
    viscous_strain_.resize(element_count(), datatypes::Matrix3r(1.0)); // Identity (no viscous strain)
}

SoftTissueEntity::SoftTissueEntity(const std::string& name)
    : FEMEntity(name)                                // Base named constructor
{
    external_poke_forces_.resize(node_count(), datatypes::Vector3(0.0));
    original_rest_positions_ = rest_positions();
    viscous_strain_.resize(element_count(), datatypes::Matrix3r(1.0));
}

SoftTissueEntity::~SoftTissueEntity() = default;     // Virtual destructor

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void SoftTissueEntity::set_soft_config(const SoftTissueConfig& config) {
    soft_config_ = config;                           // Store soft tissue parameters
}

//------------------------------------------------------------------------------
// poke / punch / bend / release
//------------------------------------------------------------------------------
void SoftTissueEntity::apply_poke(const datatypes::Vector3& world_point,
                                  const datatypes::Vector3& direction,
                                  double force_magnitude,
                                  double duration) {
    // Activate a transient poke force for a given duration
    poke_remaining_time_ = duration;                 // If 0, force is applied for one timestep
    poke_point_ = world_point;
    poke_direction_ = direction.normalized();
    poke_force_ = force_magnitude;
    // Distribute the force immediately to nodes (will be applied in integrate)
    distribute_poke_force();
    // Fire callback if registered
    if (deformation_callback_) {
        DeformationEvent event;
        event.type = DeformationEvent::Type::POKE;
        event.world_point = world_point;
        event.direction = poke_direction_;
        event.force_magnitude = force_magnitude;
        event.duration = duration;
        event.timestamp = 0.0;                      // Would be set from simulation time externally
        deformation_callback_(event);
    }
}

void SoftTissueEntity::apply_bend(const datatypes::Vector3& axis_origin,
                                  const datatypes::Vector3& axis_direction,
                                  double moment_magnitude) {
    // Activate a bending moment around the given axis
    bending_active_ = true;
    bend_axis_origin_ = axis_origin;
    bend_axis_direction_ = axis_direction.normalized();
    bend_moment_ = moment_magnitude;
    if (deformation_callback_) {
        DeformationEvent event;
        event.type = DeformationEvent::Type::BEND;
        event.world_point = axis_origin;
        event.direction = bend_axis_direction_;
        event.force_magnitude = moment_magnitude;
        event.duration = 0.0;                       // sustained until release
        event.timestamp = 0.0;
        deformation_callback_(event);
    }
}

void SoftTissueEntity::release_deformation() {
    // Stop all active external forces
    poke_remaining_time_ = 0.0;
    poke_force_ = 0.0;
    bending_active_ = false;
    bend_moment_ = 0.0;
    std::fill(external_poke_forces_.begin(), external_poke_forces_.end(), datatypes::Vector3(0.0));
    if (deformation_callback_) {
        DeformationEvent event;
        event.type = DeformationEvent::Type::RELEASE;
        event.timestamp = 0.0;
        deformation_callback_(event);
    }
}

void SoftTissueEntity::set_deformation_callback(std::function<void(const DeformationEvent&)> callback) {
    deformation_callback_ = callback;                // Store user callback
}

//------------------------------------------------------------------------------
// Integration: main simulation step
//------------------------------------------------------------------------------
void SoftTissueEntity::integrate(double dt) {
    // Add custom soft-tissue forces to the external force accumulators
    // (The FEM solver will later read these via node_external_force())
    if (poke_remaining_time_ > 0.0) {
        poke_remaining_time_ -= dt;                  // Decrease timer
        if (poke_remaining_time_ <= 0.0) {
            poke_force_ = 0.0;                       // Stop force
            std::fill(external_poke_forces_.begin(), external_poke_forces_.end(), datatypes::Vector3(0.0));
        } else {
            // Re-distribute force (could be updated if point moved)
            distribute_poke_force();
        }
        // Add poke forces to the FEM external forces
        for (size_t i = 0; i < node_count(); ++i) {
            add_node_force(i, external_poke_forces_[i]); // Accumulate into FEM force array
        }
    }

    // Restoration: passive spring-damper toward original rest shape
    compute_restoration_forces(dt);                  // Adds forces via add_node_force

    // Bending: if active, apply bending moment forces
    if (bending_active_) {
        compute_bending_forces(dt);                  // Adds forces via add_node_force
    }

    // Viscoelastic damping: simple dashpot proportional to velocity
    double damping_coeff = soft_config_.viscosity;
    for (size_t i = 0; i < node_count(); ++i) {
        datatypes::Vector3 damp_force = node_velocity(i) * (-damping_coeff);
        add_node_force(i, damp_force);               // Viscous drag
    }

    // Let base FEMEntity integrate (which is no-op, but keeps the chain)
    FEMEntity::integrate(dt);
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
void SoftTissueEntity::reset() {
    FEMEntity::reset();                              // Resets positions to rest, clears forces
    // Capture current rest positions as original (in case they were changed)
    original_rest_positions_ = rest_positions();
    external_poke_forces_.assign(node_count(), datatypes::Vector3(0.0));
    viscous_strain_.assign(element_count(), datatypes::Matrix3r(1.0));
    poke_remaining_time_ = 0.0;
    poke_force_ = 0.0;
    bending_active_ = false;
    bend_moment_ = 0.0;
}

//------------------------------------------------------------------------------
// Query max displacement and strain energy
//------------------------------------------------------------------------------
double SoftTissueEntity::get_max_displacement() const {
    double max_d2 = 0.0;
    for (size_t i = 0; i < node_count(); ++i) {
        datatypes::Vector3 diff = node_position(i) - original_rest_positions_[i];
        double d2 = diff.squaredNorm();
        if (d2 > max_d2) max_d2 = d2;
    }
    return std::sqrt(max_d2);                        // Max distance from rest
}

double SoftTissueEntity::get_strain_energy() const {
    // Sum of 0.5 * stiffness * displacement^2 (simplified)
    double energy = 0.0;
    double k = soft_config_.restoration_stiffness;
    for (size_t i = 0; i < node_count(); ++i) {
        datatypes::Vector3 diff = node_position(i) - original_rest_positions_[i];
        energy += 0.5 * k * diff.squaredNorm();
    }
    return energy;
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string SoftTissueEntity::repr() const {
    std::ostringstream oss;
    oss << "SoftTissueEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", nodes=" << node_count()
        << ", material=" << static_cast<int>(soft_config_.material)
        << ", max_disp=" << get_max_displacement()
        << ")";
    return oss.str();
}

std::string SoftTissueEntity::str() const {
    return name() + " (Soft tissue: " + std::to_string(node_count()) + " nodes)";
}

//------------------------------------------------------------------------------
// Private helpers
//------------------------------------------------------------------------------
void SoftTissueEntity::distribute_poke_force() {
    // Find closest surface node (simplified: closest node of the FEM mesh)
    size_t closest = 0;
    double min_dist2 = std::numeric_limits<double>::max();
    for (size_t i = 0; i < node_count(); ++i) {
        double d2 = (node_position(i) - poke_point_).squaredNorm();
        if (d2 < min_dist2) {
            min_dist2 = d2;
            closest = i;
        }
    }
    // Spread force to that node and its immediate neighbors (using tetrahedral connectivity)
    // First, collect neighbor set
    std::vector<bool> affected(node_count(), false);
    affected[closest] = true;
    // Look through all tetrahedra to find neighbors of closest node
    const auto& tets = tetrahedra();
    for (const auto& tet : tets) {
        for (int n = 0; n < 4; ++n) {
            if (tet[n] == static_cast<int>(closest)) {
                // Mark all four nodes of the tetrahedron as affected
                for (int m = 0; m < 4; ++m) {
                    affected[tet[m]] = true;
                }
            }
        }
    }
    // Assign force: primary node gets most, neighbors get less
    double primary_weight = 0.6;
    double neighbor_weight = (1.0 - primary_weight) / (std::count(affected.begin(), affected.end(), true) - 1.0);
    for (size_t i = 0; i < node_count(); ++i) {
        if (affected[i]) {
            double weight = (i == closest) ? primary_weight : neighbor_weight;
            external_poke_forces_[i] = poke_direction_ * (poke_force_ * weight);
        } else {
            external_poke_forces_[i] = datatypes::Vector3(0.0);
        }
    }
}

void SoftTissueEntity::compute_restoration_forces(double dt) {
    // Passive spring-damper returning nodes to original rest positions
    double k = soft_config_.restoration_stiffness;
    double d = soft_config_.restoration_damping;
    for (size_t i = 0; i < node_count(); ++i) {
        datatypes::Vector3 diff = node_position(i) - original_rest_positions_[i];
        datatypes::Vector3 vel = node_velocity(i);
        // Elastic force = -k * displacement
        // Damping force = -d * velocity (only if moving away from rest? we keep simple)
        datatypes::Vector3 force = diff * (-k) + vel * (-d);
        add_node_force(i, force);                    // Accumulate into FEM external forces
    }
}

void SoftTissueEntity::compute_bending_forces(double dt) {
    // Apply a bending moment as nodal forces giving a net torque around bend axis
    if (node_count() == 0) return;
    datatypes::Vector3 axis = bend_axis_direction_;
    datatypes::Vector3 origin = bend_axis_origin_;
    double total_moment = bend_moment_;
    // Compute average radius and assign force magnitude per node
    double avg_radius = 0.0;
    for (size_t i = 0; i < node_count(); ++i) {
        datatypes::Vector3 r = node_position(i) - origin;
        double rad = (r - axis * (axis.dot(r))).norm(); // distance from axis
        avg_radius += rad;
    }
    avg_radius /= node_count();
    if (avg_radius < 1e-12) return;
    // Force magnitude per node to produce total torque:
    // Torque contribution from each node: r × (f_i) ≈ r × (F * (axis × r)?) not exact.
    // We apply a force perpendicular to axis and r, with magnitude = (total_moment / (node_count * avg_radius)).
    // This roughly gives a torque magnitude around axis.
    double force_per_node = total_moment / (node_count() * avg_radius);
    for (size_t i = 0; i < node_count(); ++i) {
        datatypes::Vector3 r = node_position(i) - origin;
        datatypes::Vector3 perp_dir = cross(axis, r).normalized(); // perpendicular to axis and r
        datatypes::Vector3 force = perp_dir * force_per_node;
        add_node_force(i, force);
    }
}

void SoftTissueEntity::update_viscous_strain(double dt) {
    // Maxwell model: evolve viscous strain toward current deformation gradient
    double tau = soft_config_.relaxation_time;
    if (tau <= 0.0) return;
    double alpha = std::exp(-dt / tau);              // Exponential decay toward elastic strain
    for (size_t e = 0; e < element_count(); ++e) {
        datatypes::Matrix3r Fe = deformation_gradient(e); // current total deformation gradient
        // viscous strain decays toward Fe
        viscous_strain_[e] = viscous_strain_[e] * alpha + Fe * (1.0 - alpha);
    }
}

} // namespace engine
} // namespace genesis