// genesis/engine/solvers/tool_solver.cpp

#include "genesis/engine/solvers/tool_solver.h"
#include "genesis/engine/entities/tool_entity.h"
#include "genesis/engine/entities/rigid_entity.h"
#include "genesis/engine/entities/particle_entity.h"
#include "genesis/engine/entities/mpm_entity.h"
#include "genesis/engine/entities/sph_entity.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// ToolSolver implementation
//------------------------------------------------------------------------------
ToolSolver::ToolSolver(const SolverConfig& config)
    : BaseSolver("ToolSolver") {
    config_ = config;
}

ToolSolver::~ToolSolver() = default;

void ToolSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    update_entity_list(entities);
    initialized_ = true;
}

void ToolSolver::reset() {
    BaseSolver::reset();
    tools_.clear();
    tool_states_.clear();
    tool_configs_.clear();
    all_entities_.clear();
    entities_dirty_ = true;
}

void ToolSolver::step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    
    if (!initialized_ || entities_dirty_) {
        update_entity_list(entities);
    }
    
    // Update tool states based on their current actions
    for (auto& [tool_id, state] : tool_states_) {
        if (!state.is_active) continue;
        
        const ToolConfig& config = tool_configs_[tool_id];
        
        switch (state.current_action) {
            case ToolAction::GRASP:
                handle_grasp(state, config, dt);
                break;
            case ToolAction::RELEASE:
                handle_release(state, config);
                break;
            case ToolAction::PUSH:
            case ToolAction::PULL:
                handle_push(state, config, dt);
                break;
            case ToolAction::CUT:
                handle_cut(state, config, dt);
                break;
            case ToolAction::SPRAY:
                handle_spray(state, config, dt);
                break;
            case ToolAction::ROLL:
                handle_roll(state, config, dt);
                break;
            default:
                break;
        }
        
        // Update tool entity transform from state
        auto tool_it = tools_.find(tool_id);
        if (tool_it != tools_.end()) {
            tool_it->second->set_transform(state.transform);
        }
    }
    
    stats_.constraint_count = tool_states_.size();
}

void ToolSolver::on_entity_added(std::shared_ptr<BaseEntity> entity) {
    entities_dirty_ = true;
}

void ToolSolver::on_entity_removed(std::shared_ptr<BaseEntity> entity) {
    entities_dirty_ = true;
    // Remove from tools if it was a tool
    auto it = tools_.find(entity->id());
    if (it != tools_.end()) {
        tools_.erase(it);
        tool_states_.erase(entity->id());
        tool_configs_.erase(entity->id());
    }
}

void ToolSolver::add_tool(std::shared_ptr<ToolEntity> tool, const ToolConfig& config) {
    if (!tool) return;
    uint64_t id = tool->id();
    tools_[id] = tool;
    tool_configs_[id] = config;
    
    ToolState state;
    state.transform = tool->transform();
    state.is_active = true;
    tool_states_[id] = state;
}

void ToolSolver::remove_tool(uint64_t tool_id) {
    tools_.erase(tool_id);
    tool_states_.erase(tool_id);
    tool_configs_.erase(tool_id);
}

std::shared_ptr<ToolEntity> ToolSolver::get_tool(uint64_t tool_id) const {
    auto it = tools_.find(tool_id);
    if (it != tools_.end()) return it->second;
    return nullptr;
}

void ToolSolver::set_tool_transform(uint64_t tool_id, const datatypes::Transformr& transform) {
    auto it = tool_states_.find(tool_id);
    if (it != tool_states_.end()) {
        it->second.transform = transform;
    }
}

void ToolSolver::set_tool_velocity(uint64_t tool_id, const datatypes::Vector3& linear, const datatypes::Vector3& angular) {
    auto it = tool_states_.find(tool_id);
    if (it != tool_states_.end()) {
        it->second.velocity = linear;
        it->second.angular_velocity = angular;
    }
}

void ToolSolver::set_tool_action(uint64_t tool_id, ToolAction action, double value) {
    auto it = tool_states_.find(tool_id);
    if (it != tool_states_.end()) {
        it->second.current_action = action;
        it->second.action_value = value;
    }
}

void ToolSolver::grasp(uint64_t tool_id, uint64_t target_entity_id) {
    auto it = tool_states_.find(tool_id);
    if (it == tool_states_.end()) return;
    
    ToolState& state = it->second;
    const ToolConfig& config = tool_configs_[tool_id];
    
    // Find target entity
    std::shared_ptr<BaseEntity> target = nullptr;
    for (const auto& e : all_entities_) {
        if (e->id() == target_entity_id) {
            target = e;
            break;
        }
    }
    if (!target) return;
    
    // Check if within grasp tolerance
    datatypes::Vector3 target_pos = target->transform().translation;
    datatypes::Vector3 tool_pos = state.transform.translation;
    double dist = (target_pos - tool_pos).norm();
    if (dist > config.grasp_tolerance) return;
    
    // Compute relative transform from tool to target
    datatypes::Transformr tool_world = state.transform;
    datatypes::Transformr target_world = target->transform();
    datatypes::Transformr relative = tool_world.inverse() * target_world;
    
    state.grasped_objects.push_back({target_entity_id, relative});
    state.current_action = ToolAction::GRASP;
    state.affected_entities.push_back(target_entity_id);
    
    // Wake up the entity
    target->set_dynamic(true);
}

void ToolSolver::release(uint64_t tool_id) {
    auto it = tool_states_.find(tool_id);
    if (it == tool_states_.end()) return;
    
    ToolState& state = it->second;
    const ToolConfig& config = tool_configs_[tool_id];
    
    // Apply release velocity to grasped objects
    for (const auto& [entity_id, relative] : state.grasped_objects) {
        for (auto& e : all_entities_) {
            if (e->id() == entity_id) {
                datatypes::Vector3 release_vel = state.velocity + state.transform.rotation.rotate(relative.translation) * config.release_velocity;
                e->set_velocity(release_vel);
                break;
            }
        }
    }
    
    state.grasped_objects.clear();
    state.current_action = ToolAction::NONE;
    state.affected_entities.clear();
}

bool ToolSolver::is_grasping(uint64_t tool_id) const {
    auto it = tool_states_.find(tool_id);
    if (it != tool_states_.end()) {
        return !it->second.grasped_objects.empty();
    }
    return false;
}

std::vector<uint64_t> ToolSolver::get_grasped_objects(uint64_t tool_id) const {
    std::vector<uint64_t> result;
    auto it = tool_states_.find(tool_id);
    if (it != tool_states_.end()) {
        for (const auto& [id, rel] : it->second.grasped_objects) {
            result.push_back(id);
        }
    }
    return result;
}

void ToolSolver::cut(uint64_t tool_id, const datatypes::Vector3& plane_point, const datatypes::Vector3& plane_normal) {
    auto it = tool_states_.find(tool_id);
    if (it == tool_states_.end()) return;
    
    it->second.cut_plane_point = plane_point;
    it->second.cut_plane_normal = plane_normal.normalized();
    it->second.current_action = ToolAction::CUT;
}

void ToolSolver::spray(uint64_t tool_id, bool enable, double rate) {
    auto it = tool_states_.find(tool_id);
    if (it == tool_states_.end()) return;
    
    if (enable) {
        it->second.current_action = ToolAction::SPRAY;
        it->second.spray_rate = rate;
    } else {
        it->second.current_action = ToolAction::NONE;
    }
}

void ToolSolver::update_entity_list(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    all_entities_ = entities;
    entities_dirty_ = false;
}

void ToolSolver::handle_grasp(ToolState& state, const ToolConfig& config, double dt) {
    // Update grasped objects' transforms to follow the tool
    for (auto& [entity_id, relative] : state.grasped_objects) {
        for (auto& e : all_entities_) {
            if (e->id() == entity_id) {
                datatypes::Transformr new_target_world = state.transform * relative;
                datatypes::Transformr current = e->transform();
                
                // Compute spring-damper force to move to target
                datatypes::Vector3 pos_error = new_target_world.translation - current.translation;
                datatypes::Quat rot_error = new_target_world.rotation * current.rotation.conjugate();
                
                // Position correction force
                datatypes::Vector3 force = pos_error * config.grasp_stiffness;
                force -= e->velocity() * config.grasp_damping;
                
                // Rotation correction torque
                datatypes::Vector3 axis; double angle;
                rot_error.to_axis_angle(axis, angle);
                datatypes::Vector3 torque = axis * (angle * config.grasp_stiffness);
                torque -= e->angular_velocity() * config.grasp_damping;
                
                // Apply forces
                e->apply_force(force);
                e->apply_torque(torque);
                
                // Enforce max force
                double force_mag = force.norm();
                if (force_mag > config.max_force) {
                    force *= config.max_force / force_mag;
                }
                break;
            }
        }
    }
}

void ToolSolver::handle_release(ToolState& state, const ToolConfig& config) {
    state.grasped_objects.clear();
    state.current_action = ToolAction::NONE;
    state.affected_entities.clear();
}

void ToolSolver::handle_push(ToolState& state, const ToolConfig& config, double dt) {
    datatypes::Vector3 center = state.transform.translation;
    double radius = config.interaction_radius;
    
    auto nearby = find_rigid_entities_in_radius(center, radius);
    for (auto& entity : nearby) {
        datatypes::Vector3 entity_pos = entity->transform().translation;
        datatypes::Vector3 dir = entity_pos - center;
        double dist = dir.norm();
        if (dist < 1e-6) continue;
        dir /= dist;
        
        double penetration = radius - dist;
        if (penetration > 0) {
            // Push force proportional to penetration
            double force_mag = penetration * config.push_stiffness;
            datatypes::Vector3 force = dir * force_mag;
            
            // Add velocity-based damping
            datatypes::Vector3 rel_vel = entity->velocity() - state.velocity;
            double vn = rel_vel.dot(dir);
            if (vn > 0) {
                force += dir * (-vn * config.grasp_damping);
            }
            
            entity->apply_force(force);
        }
    }
}

void ToolSolver::handle_cut(ToolState& state, const ToolConfig& config, double dt) {
    // Find entities that can be cut (e.g., MPM or mesh entities)
    for (auto& entity : all_entities_) {
        // Check if entity has a mesh or particles
        auto mpm_entity = std::dynamic_pointer_cast<MPMEntity>(entity);
        if (mpm_entity) {
            // Apply cutting plane to MPM particles
            const auto& positions = mpm_entity->particle_positions();
            std::vector<bool> to_remove(positions.size(), false);
            
            datatypes::Vector3 normal = state.cut_plane_normal;
            datatypes::Vector3 point = state.cut_plane_point;
            
            for (size_t i = 0; i < positions.size(); ++i) {
                double signed_dist = (positions[i] - point).dot(normal);
                if (signed_dist > 0 && signed_dist < config.cut_plane_thickness) {
                    // Mark particle for removal or split
                    to_remove[i] = true;
                }
            }
            
            // Remove marked particles
            mpm_entity->remove_particles(to_remove);
        }
        
        auto sph_entity = std::dynamic_pointer_cast<SPHEntity>(entity);
        if (sph_entity) {
            // Similar for SPH particles
            const auto& positions = sph_entity->particle_positions();
            std::vector<bool> to_remove(positions.size(), false);
            
            datatypes::Vector3 normal = state.cut_plane_normal;
            datatypes::Vector3 point = state.cut_plane_point;
            
            for (size_t i = 0; i < positions.size(); ++i) {
                double signed_dist = (positions[i] - point).dot(normal);
                if (signed_dist > 0 && signed_dist < config.cut_plane_thickness) {
                    to_remove[i] = true;
                }
            }
            
            sph_entity->remove_particles(to_remove);
        }
    }
    
    // Reset action after cut
    state.current_action = ToolAction::NONE;
}

void ToolSolver::handle_spray(ToolState& state, const ToolConfig& config, double dt) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    static std::uniform_real_distribution<double> speed_dist(0.8, 1.2);
    
    // Calculate number of particles to emit this step
    int particles_to_emit = static_cast<int>(config.particles_per_second * state.spray_rate * dt);
    
    // Find or create a particle entity to add particles to
    std::shared_ptr<SPHEntity> spray_target = nullptr;
    for (auto& e : all_entities_) {
        if (e->name() == "spray_particles") {
            spray_target = std::dynamic_pointer_cast<SPHEntity>(e);
            break;
        }
    }
    if (!spray_target) {
        // Create new entity for spray particles - in practice would be done via scene
        return;
    }
    
    // Emit particles
    for (int i = 0; i < particles_to_emit; ++i) {
        datatypes::Vector3 local_offset(
            angle_dist(gen) * config.spray_radius * 0.3,
            angle_dist(gen) * config.spray_radius * 0.3,
            config.spray_radius * 0.5
        );
        datatypes::Vector3 world_pos = state.transform.transformPoint(local_offset);
        
        datatypes::Vector3 spray_dir = state.transform.rotation.rotate(datatypes::Vector3(0, 0, 1));
        double spray_speed = 2.0 * speed_dist(gen);
        datatypes::Vector3 velocity = spray_dir * spray_speed + state.velocity;
        
        double mass = config.particle_density * (4.0/3.0) * M_PI * std::pow(config.particle_radius, 3);
        
        spray_target->add_particle(world_pos, velocity, mass);
    }
}

void ToolSolver::handle_roll(ToolState& state, const ToolConfig& config, double dt) {
    // Apply rolling friction and torque to objects under the roller
    datatypes::Vector3 center = state.transform.translation;
    double radius = config.interaction_radius;
    
    auto nearby = find_rigid_entities_in_radius(center, radius);
    for (auto& entity : nearby) {
        datatypes::Vector3 entity_pos = entity->transform().translation;
        datatypes::Vector3 dir = entity_pos - center;
        double dist = dir.norm();
        if (dist < 1e-6) continue;
        dir /= dist;
        
        // Rolling resistance torque
        datatypes::Vector3 rolling_axis = dir.cross(state.velocity).normalized();
        double rolling_torque_mag = config.friction_coefficient * entity->mass() * 9.81 * radius;
        datatypes::Vector3 torque = rolling_axis * rolling_torque_mag;
        
        entity->apply_torque(torque);
        
        // Linear force to push in direction of roller velocity
        datatypes::Vector3 tangent_vel = state.velocity - dir * state.velocity.dot(dir);
        datatypes::Vector3 force = tangent_vel * (config.push_stiffness * dt);
        entity->apply_force(force);
    }
}

void ToolSolver::apply_grasp_forces(ToolState& state, const ToolConfig& config, double dt) {
    // Already handled in handle_grasp
}

void ToolSolver::update_grasp_transforms(ToolState& state) {
    // Already handled in handle_grasp
}

void ToolSolver::apply_push_forces(const datatypes::Transformr& tool_transform, double radius, double stiffness, double dt) {
    // Already handled in handle_push
}

void ToolSolver::perform_cut(ToolState& state, const ToolConfig& config) {
    // Already handled in handle_cut
}

void ToolSolver::emit_particles(ToolState& state, const ToolConfig& config, double dt) {
    // Already handled in handle_spray
}

std::vector<std::shared_ptr<BaseEntity>> ToolSolver::find_entities_in_radius(const datatypes::Vector3& center, double radius) const {
    std::vector<std::shared_ptr<BaseEntity>> result;
    double r2 = radius * radius;
    for (const auto& e : all_entities_) {
        datatypes::Vector3 pos = e->transform().translation;
        if ((pos - center).squaredNorm() <= r2) {
            result.push_back(e);
        }
    }
    return result;
}

std::vector<std::shared_ptr<RigidEntity>> ToolSolver::find_rigid_entities_in_radius(const datatypes::Vector3& center, double radius) const {
    std::vector<std::shared_ptr<RigidEntity>> result;
    double r2 = radius * radius;
    for (const auto& e : all_entities_) {
        auto rigid = std::dynamic_pointer_cast<RigidEntity>(e);
        if (rigid) {
            datatypes::Vector3 pos = rigid->transform().translation;
            if ((pos - center).squaredNorm() <= r2) {
                result.push_back(rigid);
            }
        }
    }
    return result;
}

std::vector<std::shared_ptr<ParticleEntity>> ToolSolver::find_particle_entities_in_radius(const datatypes::Vector3& center, double radius) const {
    std::vector<std::shared_ptr<ParticleEntity>> result;
    double r2 = radius * radius;
    for (const auto& e : all_entities_) {
        auto particle = std::dynamic_pointer_cast<ParticleEntity>(e);
        if (particle) {
            datatypes::Vector3 pos = particle->transform().translation;
            if ((pos - center).squaredNorm() <= r2) {
                result.push_back(particle);
            }
        }
    }
    return result;
}

bool ToolSolver::can_grasp(uint64_t tool_id, uint64_t entity_id) const {
    auto tool_it = tool_states_.find(tool_id);
    if (tool_it == tool_states_.end()) return false;
    
    const ToolState& state = tool_it->second;
    const ToolConfig& config = tool_configs_.at(tool_id);
    
    // Find entity
    std::shared_ptr<BaseEntity> target = nullptr;
    for (const auto& e : all_entities_) {
        if (e->id() == entity_id) {
            target = e;
            break;
        }
    }
    if (!target) return false;
    
    datatypes::Vector3 target_pos = target->transform().translation;
    datatypes::Vector3 tool_pos = state.transform.translation;
    double dist = (target_pos - tool_pos).norm();
    return dist <= config.grasp_tolerance;
}

void ToolSolver::apply_impulse_to_entity(std::shared_ptr<BaseEntity> entity, const datatypes::Vector3& point, const datatypes::Vector3& impulse) {
    if (!entity) return;
    if (auto rigid = std::dynamic_pointer_cast<RigidEntity>(entity)) {
        rigid->apply_impulse(impulse, point);
    } else {
        // For particle entities, apply to all particles near the point
        if (auto particles = std::dynamic_pointer_cast<ParticleEntity>(entity)) {
            const auto& positions = particles->particle_positions();
            for (size_t i = 0; i < positions.size(); ++i) {
                double dist = (positions[i] - point).norm();
                if (dist < 0.1) {
                    particles->apply_force_to_particle(i, impulse);
                }
            }
        }
    }
}

} // namespace engine
} // namespace genesis