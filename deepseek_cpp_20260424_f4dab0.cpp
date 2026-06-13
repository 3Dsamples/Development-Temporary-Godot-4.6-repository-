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
InteractionManager::InteractionManager() = default;
InteractionManager::~InteractionManager() = default;

//------------------------------------------------------------------------------
// Main update – called each simulation step with scene reference
//------------------------------------------------------------------------------
void InteractionManager::update(double dt, Scene& scene) {
    // Iterate through active interactions
    for (auto it = active_interactions_.begin(); it != active_interactions_.end(); ) {
        ActiveInteraction& interaction = *it;

        // Decrement time for transient interactions
        if (interaction.remaining_time > 0.0) {
            interaction.remaining_time -= dt;
            if (interaction.remaining_time <= 0.0) {
                // Expired; remove this interaction
                it = active_interactions_.erase(it);
                continue;
            }
        }

        // Resolve target entity if not yet known
        if (interaction.affected_entity_id == 0) {
            if (interaction.event.target_entity_id != 0) {
                interaction.affected_entity_id = interaction.event.target_entity_id;
            } else {
                // Raycast to find entity under point
                datatypes::Ray ray;
                // Start a little behind the point along direction for robust hit
                ray.origin = interaction.event.world_point - interaction.event.direction * 0.01;
                ray.direction = interaction.event.direction;
                auto hit = scene.ray_cast(ray);
                if (hit.hit) {
                    interaction.affected_entity_id = hit.entity_id;
                } else {
                    // No entity hit; remove interaction
                    it = active_interactions_.erase(it);
                    continue;
                }
            }
        }

        // Retrieve entity
        std::shared_ptr<BaseEntity> entity = scene.get_entity(interaction.affected_entity_id);
        if (!entity) {
            // Entity disappeared; remove interaction
            it = active_interactions_.erase(it);
            continue;
        }

        // Apply interaction based on type
        switch (interaction.event.type) {
            case InteractionEvent::Type::POKE:
            case InteractionEvent::Type::PUNCH: {
                // Apply a directed force at the world point
                datatypes::Vector3 force = interaction.event.direction * interaction.event.magnitude;

                // Try to use specialized methods if available
                if (auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity)) {
                    soft->apply_poke(interaction.event.world_point, interaction.event.direction,
                                     interaction.event.magnitude, interaction.remaining_time);
                } else if (auto fem = std::dynamic_pointer_cast<FEMEntity>(entity)) {
                    // Spread force to nearby nodes
                    datatypes::Vector3 force_per_node = force * 0.1; // arbitrary fraction
                    // For simplicity, find closest node and apply there
                    const auto& positions = fem->node_positions();
                    size_t closest = 0;
                    double min_dist2 = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < positions.size(); ++i) {
                        double d2 = (positions[i] - interaction.event.world_point).squaredNorm();
                        if (d2 < min_dist2) {
                            min_dist2 = d2;
                            closest = i;
                        }
                    }
                    fem->add_node_force(closest, force);
                    interaction.affected_node_indices = {static_cast<uint32_t>(closest)};
                } else if (auto pbd = std::dynamic_pointer_cast<PBDEntity>(entity)) {
                    const auto& positions = pbd->positions();
                    size_t closest = 0;
                    double min_dist2 = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < positions.size(); ++i) {
                        double d2 = (positions[i] - interaction.event.world_point).squaredNorm();
                        if (d2 < min_dist2) {
                            min_dist2 = d2;
                            closest = i;
                        }
                    }
                    pbd->apply_force_to_particle(closest, force);
                } else if (auto mpm = std::dynamic_pointer_cast<MPMEntity>(entity)) {
                    const auto& positions = mpm->particle_positions();
                    size_t closest = 0;
                    double min_dist2 = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < positions.size(); ++i) {
                        double d2 = (positions[i] - interaction.event.world_point).squaredNorm();
                        if (d2 < min_dist2) {
                            min_dist2 = d2;
                            closest = i;
                        }
                    }
                    mpm->apply_force_to_particle(closest, force);
                } else {
                    // Fallback: apply force to entity's COM
                    entity->apply_force(force, interaction.event.world_point);
                }
                break;
            }
            case InteractionEvent::Type::BEND: {
                // Bending moment applied as forces around axis
                if (auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity)) {
                    soft->apply_bend(interaction.event.world_point,  // axis origin
                                     interaction.event.direction,   // axis direction
                                     interaction.event.magnitude);
                } else {
                    // Generic bending: compute moment arm and apply perpendicular forces to nodes
                    // Use the entity's bounding box to approximate nodes
                    // For generic FEM, apply torque to all nodes based on their position relative to axis
                    auto fem = std::dynamic_pointer_cast<FEMEntity>(entity);
                    if (fem && fem->node_count() > 0) {
                        datatypes::Vector3 axis_origin = interaction.event.world_point;
                        datatypes::Vector3 axis_dir = interaction.event.direction;
                        double moment = interaction.event.magnitude;
                        const auto& positions = fem->node_positions();
                        double total_weight = 0.0;
                        for (size_t i = 0; i < fem->node_count(); ++i) {
                            datatypes::Vector3 r = positions[i] - axis_origin;
                            // perpendicular distance from axis
                            double rad = (r - axis_dir * (axis_dir.dot(r))).norm();
                            total_weight += rad;
                        }
                        if (total_weight < 1e-12) break;
                        for (size_t i = 0; i < fem->node_count(); ++i) {
                            datatypes::Vector3 r = positions[i] - axis_origin;
                            double rad = (r - axis_dir * (axis_dir.dot(r))).norm();
                            if (rad < 1e-6) continue;
                            // Direction perpendicular to axis and r
                            datatypes::Vector3 force_dir = cross(axis_dir, r).normalized();
                            double force_mag = moment * rad / total_weight;
                            fem->add_node_force(i, force_dir * force_mag);
                        }
                    } else if (auto pbd = std::dynamic_pointer_cast<PBDEntity>(entity)) {
                        // Similar but with particles
                        const auto& positions = pbd->positions();
                        datatypes::Vector3 axis_origin = interaction.event.world_point;
                        datatypes::Vector3 axis_dir = interaction.event.direction;
                        double moment = interaction.event.magnitude;
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
                // Grab: apply spring-damper toward a fixed point (simplified as an attraction)
                if (auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity)) {
                    // Use poke with zero duration but sustained force?
                    // We'll apply a spring force toward the grab point, but we need current position of the grabbed node
                    // For now, just re-apply poke each frame.
                    soft->apply_poke(interaction.event.world_point, interaction.event.direction, interaction.event.magnitude, -1.0);
                } else {
                    // Apply a constant force toward the grab point? No, grab should attract to point.
                    // We'll apply a spring force: find closest node and pull toward the world point with stiffness
                    datatypes::Vector3 grab_point = interaction.event.world_point;
                    // Use a fixed stiffness
                    double stiffness = 500.0;
                    // Find closest node
                    auto fem = std::dynamic_pointer_cast<FEMEntity>(entity);
                    if (fem) {
                        const auto& positions = fem->node_positions();
                        size_t closest = 0;
                        double min_dist2 = std::numeric_limits<double>::max();
                        for (size_t i = 0; i < positions.size(); ++i) {
                            double d2 = (positions[i] - grab_point).squaredNorm();
                            if (d2 < min_dist2) { min_dist2 = d2; closest = i; }
                        }
                        datatypes::Vector3 diff = positions[closest] - grab_point;
                        datatypes::Vector3 spring_force = diff * (-stiffness);
                        // Also add damping
                        if (fem->node_count() > 0) {
                            datatypes::Vector3 vel = fem->node_velocity(closest);
                            spring_force += vel * (-10.0); // damping
                        }
                        fem->add_node_force(closest, spring_force);
                        interaction.affected_node_indices = {static_cast<uint32_t>(closest)};
                    }
                    // Similar for PBD, MPM...
                }
                break;
            }
            default:
                break;
        }

        ++it; // Move to next interaction
    }
}

//------------------------------------------------------------------------------
// Add an interaction (public, same as earlier)
//------------------------------------------------------------------------------
void InteractionManager::add_interaction(const InteractionEvent& event) {
    ActiveInteraction active;
    active.event = event;
    active.remaining_time = event.duration;
    active.affected_entity_id = 0;
    active_interactions_.push_back(active);
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
    event.world_point = axis_origin;
    event.direction = axis_direction.normalized();
    event.magnitude = moment;
    event.duration = duration;
    add_interaction(event);
}

void InteractionManager::grab(uint64_t entity_id, const datatypes::Vector3& grab_point) {
    InteractionEvent event;
    event.type = InteractionEvent::Type::GRAB;
    event.target_entity_id = entity_id;
    event.world_point = grab_point;
    event.duration = -1.0;
    add_interaction(event);
}

void InteractionManager::release(uint64_t entity_id) {
    for (auto& interaction : active_interactions_) {
        if (entity_id == 0 || interaction.affected_entity_id == entity_id) {
            interaction.remaining_time = 0.0;
        }
    }
}

void InteractionManager::set_event_callback(std::function<void(const InteractionEvent&, const std::string&)> callback) {
    event_callback_ = callback;
}

void InteractionManager::clear_event_callback() {
    event_callback_ = nullptr;
}

//------------------------------------------------------------------------------
// Private: remove expired interactions (called from update after loop)
//------------------------------------------------------------------------------
void InteractionManager::remove_expired_interactions() {
    // Actually removal is done inline during iteration, so this is a no-op.
    // Kept for potential future use.
    active_interactions_.erase(
        std::remove_if(active_interactions_.begin(), active_interactions_.end(),
            [](const ActiveInteraction& a) { return a.remaining_time == 0.0 && a.event.duration >= 0; }),
        active_interactions_.end());
}

//------------------------------------------------------------------------------
// Static helper: cross product (defined in anonymous namespace)
//------------------------------------------------------------------------------
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