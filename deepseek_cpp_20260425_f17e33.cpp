// genesis/engine/interaction_manager.cpp
#include "genesis/engine/interaction_manager.h"   // Corresponding header
#include "genesis/engine/scene.h"                 // Scene for entity lookup and raycasts
#include "genesis/engine/entities/base_entity.h"  // BaseEntity
#include "genesis/engine/entities/fem_entity.h"   // FEMEntity
#include "genesis/engine/entities/pbd_entity.h"   // PBDEntity
#include "genesis/engine/entities/mpm_entity.h"   // MPMEntity
#include "genesis/engine/entities/soft_tissue_entity.h" // SoftTissueEntity
#include <algorithm>                              // std::find, std::max
#include <cmath>                                  // std::sqrt
#include <limits>                                 // std::numeric_limits

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Construction / Destruction
//------------------------------------------------------------------------------
InteractionManager::InteractionManager() = default;  // Default constructor
InteractionManager::~InteractionManager() = default; // Default destructor

//------------------------------------------------------------------------------
// Main update – called each simulation step with scene reference
//------------------------------------------------------------------------------
void InteractionManager::update(double dt, Scene& scene) {
    // Iterate through active interactions, apply forces, and handle expiration
    for (auto it = active_interactions_.begin(); it != active_interactions_.end(); ) {
        ActiveInteraction& interaction = *it;        // Reference to current interaction

        // Decrement remaining time for transient forces
        if (interaction.remaining_time > 0.0) {
            interaction.remaining_time -= dt;        // Reduce timer
            if (interaction.remaining_time <= 0.0) {
                it = active_interactions_.erase(it); // Expired, remove from list
                continue;
            }
        }

        // Resolve target entity if not yet known
        if (interaction.affected_entity_id == 0) {
            if (interaction.event.target_entity_id != 0) {
                interaction.affected_entity_id = interaction.event.target_entity_id; // Use explicit ID
            } else {
                // Perform raycast to find entity under the interaction point
                datatypes::Ray ray;
                ray.origin = interaction.event.world_point - interaction.event.direction * 0.01; // Offset for robustness
                ray.direction = interaction.event.direction;  // Ray direction
                auto hit = scene.ray_cast(ray);              // Cast ray into scene
                if (hit.hit) {
                    interaction.affected_entity_id = hit.entity_id; // Store hit entity ID
                } else {
                    it = active_interactions_.erase(it);     // No entity found, discard interaction
                    continue;
                }
            }
        }

        // Retrieve the target entity from the scene
        std::shared_ptr<BaseEntity> entity = scene.get_entity(interaction.affected_entity_id);
        if (!entity) {
            it = active_interactions_.erase(it);     // Entity no longer exists
            continue;
        }

        // Apply forces based on interaction type
        switch (interaction.event.type) {
            case InteractionEvent::Type::POKE:
            case InteractionEvent::Type::PUNCH: {
                // Calculate force vector
                datatypes::Vector3 force = interaction.event.direction * interaction.event.magnitude;

                // Dispatch to specialized entity methods when possible
                if (auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity)) {
                    soft->apply_poke(interaction.event.world_point, interaction.event.direction,
                                     interaction.event.magnitude, interaction.remaining_time);
                } else if (auto fem = std::dynamic_pointer_cast<FEMEntity>(entity)) {
                    const auto& positions = fem->node_positions();     // Get node positions
                    size_t closest = 0;
                    double min_dist2 = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < positions.size(); ++i) {
                        double d2 = (positions[i] - interaction.event.world_point).squaredNorm();
                        if (d2 < min_dist2) { min_dist2 = d2; closest = i; }
                    }
                    fem->add_node_force(closest, force);              // Apply to nearest FEM node
                    interaction.affected_node_indices = {static_cast<uint32_t>(closest)};
                } else if (auto pbd = std::dynamic_pointer_cast<PBDEntity>(entity)) {
                    const auto& positions = pbd->positions();          // Get particle positions
                    size_t closest = 0;
                    double min_dist2 = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < positions.size(); ++i) {
                        double d2 = (positions[i] - interaction.event.world_point).squaredNorm();
                        if (d2 < min_dist2) { min_dist2 = d2; closest = i; }
                    }
                    pbd->apply_force_to_particle(closest, force);     // Apply to nearest PBD particle
                } else if (auto mpm = std::dynamic_pointer_cast<MPMEntity>(entity)) {
                    const auto& positions = mpm->particle_positions(); // Get MPM particle positions
                    size_t closest = 0;
                    double min_dist2 = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < positions.size(); ++i) {
                        double d2 = (positions[i] - interaction.event.world_point).squaredNorm();
                        if (d2 < min_dist2) { min_dist2 = d2; closest = i; }
                    }
                    mpm->apply_force_to_particle(closest, force);     // Apply to nearest MPM particle
                } else {
                    entity->apply_force(force, interaction.event.world_point); // Fallback: apply at point
                }
                break;
            }
            case InteractionEvent::Type::BEND: {
                // Bending moment application
                if (auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity)) {
                    soft->apply_bend(interaction.event.world_point,   // axis origin
                                     interaction.event.direction,    // axis direction
                                     interaction.event.magnitude);   // moment magnitude
                } else {
                    // Generic bending: apply torque as forces on nodes/particles
                    datatypes::Vector3 axis_origin = interaction.event.world_point;
                    datatypes::Vector3 axis_dir = interaction.event.direction;
                    double moment = interaction.event.magnitude;

                    if (auto fem = std::dynamic_pointer_cast<FEMEntity>(entity)) {
                        const auto& positions = fem->node_positions();
                        double total_weight = 0.0;
                        for (size_t i = 0; i < fem->node_count(); ++i) {
                            datatypes::Vector3 r = positions[i] - axis_origin;
                            double rad = (r - axis_dir * (axis_dir.dot(r))).norm(); // distance from axis
                            total_weight += rad;
                        }
                        if (total_weight < 1e-12) break;
                        for (size_t i = 0; i < fem->node_count(); ++i) {
                            datatypes::Vector3 r = positions[i] - axis_origin;
                            double rad = (r - axis_dir * (axis_dir.dot(r))).norm();
                            if (rad < 1e-6) continue;
                            datatypes::Vector3 force_dir = cross(axis_dir, r).normalized();
                            double force_mag = moment * rad / total_weight;
                            fem->add_node_force(i, force_dir * force_mag); // Apply bending force
                        }
                    } else if (auto pbd = std::dynamic_pointer_cast<PBDEntity>(entity)) {
                        const auto& positions = pbd->positions();
                        double total_weight = 0.0;
                        for (size_t i = 0; i < positions.size(); ++i) {
                            datatypes::Vector3 r = positions[i] - axis_origin;
                            double rad = (r - axis_dir * (axis_dir.dot(r))).norm();
                            total_weight += rad;
                        }
                        if (total_weight < 1e-12) break;
                        for (size_t i = 0; i < positions.size(); ++i) {
                            datatypes::Vector3 r = positions[i] - axis_origin;
                            double rad = (r - axis_dir * (axis_dir.dot(r))).norm();
                            if (rad < 1e-6) continue;
                            datatypes::Vector3 force_dir = cross(axis_dir, r).normalized();
                            double force_mag = moment * rad / total_weight;
                            pbd->apply_force_to_particle(i, force_dir * force_mag);
                        }
                    }
                }
                break;
            }
            case InteractionEvent::Type::GRAB: {
                // Grab: apply spring-damper to keep grabbed point at fixed world location
                double stiffness = 500.0;            // Spring stiffness
                double damping = 10.0;              // Damping coefficient

                if (auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity)) {
                    soft->apply_poke(interaction.event.world_point, interaction.event.direction,
                                     interaction.event.magnitude, -1.0); // Reuse poke with infinite duration
                } else if (auto fem = std::dynamic_pointer_cast<FEMEntity>(entity)) {
                    const auto& positions = fem->node_positions();
                    size_t closest = 0;
                    double min_dist2 = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < positions.size(); ++i) {
                        double d2 = (positions[i] - interaction.event.world_point).squaredNorm();
                        if (d2 < min_dist2) { min_dist2 = d2; closest = i; }
                    }
                    datatypes::Vector3 diff = positions[closest] - interaction.event.world_point;
                    datatypes::Vector3 vel = fem->node_velocity(closest);
                    datatypes::Vector3 spring_force = diff * (-stiffness);   // Hooke's law
                    datatypes::Vector3 damp_force = vel * (-damping);        // Velocity damping
                    fem->add_node_force(closest, spring_force + damp_force);
                    interaction.affected_node_indices = {static_cast<uint32_t>(closest)};
                }
                // PBD, MPM grabs could be added similarly
                break;
            }
            default:
                break;
        }

        ++it; // Advance to next interaction
    }
}

//------------------------------------------------------------------------------
// Add an interaction (public)
//------------------------------------------------------------------------------
void InteractionManager::add_interaction(const InteractionEvent& event) {
    ActiveInteraction active;                        // Create active interaction record
    active.event = event;
    active.remaining_time = event.duration;          // Set timer
    active.affected_entity_id = 0;                   // Will be resolved on next update
    active_interactions_.push_back(active);          // Store in list
}

void InteractionManager::poke(const datatypes::Vector3& world_point,
                              const datatypes::Vector3& direction,
                              double force, double duration) {
    InteractionEvent event;
    event.type = InteractionEvent::Type::POKE;
    event.world_point = world_point;
    event.direction = direction.normalized();
    event.magnitude = force;
    event.duration = duration;
    add_interaction(event);
}

void InteractionManager::punch(const datatypes::Vector3& world_point,
                               const datatypes::Vector3& direction,
                               double force, double duration) {
    InteractionEvent event;
    event.type = InteractionEvent::Type::PUNCH;
    event.world_point = world_point;
    event.direction = direction.normalized();
    event.magnitude = force;
    event.duration = duration;
    add_interaction(event);
}

void InteractionManager::bend(const datatypes::Vector3& axis_origin,
                              const datatypes::Vector3& axis_direction,
                              double moment, double duration) {
    InteractionEvent event;
    event.type = InteractionEvent::Type::BEND;
    event.world_point = axis_origin;                // Axis origin stored in world_point
    event.direction = axis_direction.normalized();   // Axis direction
    event.magnitude = moment;
    event.duration = duration;
    add_interaction(event);
}

void InteractionManager::grab(uint64_t entity_id, const datatypes::Vector3& grab_point) {
    InteractionEvent event;
    event.type = InteractionEvent::Type::GRAB;
    event.target_entity_id = entity_id;
    event.world_point = grab_point;                 // Fixed grab location
    event.duration = -1.0;                          // Infinite until released
    add_interaction(event);
}

void InteractionManager::release(uint64_t entity_id) {
    // Set remaining time to zero for all interactions of given entity (or all)
    for (auto& interaction : active_interactions_) {
        if (entity_id == 0 || interaction.affected_entity_id == entity_id) {
            interaction.remaining_time = 0.0;
        }
    }
}

void InteractionManager::set_event_callback(std::function<void(const InteractionEvent&, const std::string&)> callback) {
    event_callback_ = callback;                     // Store callback function
}

void InteractionManager::clear_event_callback() {
    event_callback_ = nullptr;                      // Remove callback
}

//------------------------------------------------------------------------------
// Private helpers
//------------------------------------------------------------------------------
void InteractionManager::remove_expired_interactions() {
    // Completed interactions are erased inline during update; this is a safety cleanup
    active_interactions_.erase(
        std::remove_if(active_interactions_.begin(), active_interactions_.end(),
            [](const ActiveInteraction& a) { return a.remaining_time == 0.0 && a.event.duration >= 0; }),
        active_interactions_.end());
}

// Anonymous namespace for static cross product function
namespace {
    datatypes::Vector3 cross(const datatypes::Vector3& a, const datatypes::Vector3& b) {
        return datatypes::Vector3(
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        );
    }
}

} // namespace engine
} // namespace genesis