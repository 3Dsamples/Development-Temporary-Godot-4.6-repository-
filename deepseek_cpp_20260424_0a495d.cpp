// genesis/engine/scene.cpp

#include "genesis/engine/scene.h"
#include "genesis/engine/entities/rigid_entity.h"   // placeholder, actual entity types
#include "genesis/engine/entities/particle_entity.h"
#include "genesis/engine/solvers/pbd_solver.h"
#include "genesis/engine/contact_manager.h"
#include <chrono>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// ContactManager (simplified implementation - will be separate file later)
//------------------------------------------------------------------------------
class ContactManager {
public:
    ContactManager() = default;
    void clear() { contacts_.clear(); }
    void add_contact(const ContactPoint& cp) { contacts_.push_back(cp); }
    const std::vector<ContactPoint>& get_contacts() const { return contacts_; }
    std::vector<ContactPoint>& get_contacts() { return contacts_; }
private:
    std::vector<ContactPoint> contacts_;
};

//------------------------------------------------------------------------------
// Scene implementation
//------------------------------------------------------------------------------
Scene::Scene(const SceneConfig& config)
    : config_(config)
    , force_fields_(std::make_unique<ForceFieldManager>())
    , contact_manager_(std::make_unique<ContactManager>())
    , broad_phase_(std::make_unique<BVH>())
{
}

Scene::~Scene() = default;

void Scene::set_config(const SceneConfig& config) {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    config_ = config;
}

void Scene::add_entity(std::shared_ptr<BaseEntity> entity) {
    if (!entity) return;
    std::lock_guard<std::mutex> lock(scene_mutex_);
    entities_.push_back(entity);
    entity_by_id_[entity->id()] = entity;
    entity_by_name_[entity->name()] = entity;
    broad_phase_dirty_ = true;
}

void Scene::remove_entity(uint64_t entity_id) {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    auto it = entity_by_id_.find(entity_id);
    if (it != entity_by_id_.end()) {
        auto entity = it->second;
        entity_by_name_.erase(entity->name());
        entity_by_id_.erase(it);
        auto vec_it = std::find(entities_.begin(), entities_.end(), entity);
        if (vec_it != entities_.end()) entities_.erase(vec_it);
        broad_phase_dirty_ = true;
    }
}

void Scene::remove_entity(const std::string& name) {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    auto it = entity_by_name_.find(name);
    if (it != entity_by_name_.end()) {
        auto entity = it->second;
        entity_by_id_.erase(entity->id());
        entity_by_name_.erase(it);
        auto vec_it = std::find(entities_.begin(), entities_.end(), entity);
        if (vec_it != entities_.end()) entities_.erase(vec_it);
        broad_phase_dirty_ = true;
    }
}

void Scene::clear_entities() {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    entities_.clear();
    entity_by_id_.clear();
    entity_by_name_.clear();
    broad_phase_dirty_ = true;
}

std::shared_ptr<BaseEntity> Scene::get_entity(uint64_t id) const {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    auto it = entity_by_id_.find(id);
    if (it != entity_by_id_.end()) return it->second;
    return nullptr;
}

std::shared_ptr<BaseEntity> Scene::get_entity(const std::string& name) const {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    auto it = entity_by_name_.find(name);
    if (it != entity_by_name_.end()) return it->second;
    return nullptr;
}

void Scene::set_solver(std::shared_ptr<BaseSolver> solver) {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    solver_ = solver;
}

void Scene::reset() {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    current_time_ = 0.0;
    step_count_ = 0;
    for (auto& entity : entities_) {
        entity->reset();
    }
    contacts_.clear();
    contact_manager_->clear();
    stats_ = Stats{};
    broad_phase_dirty_ = true;
}

void Scene::step(double dt) {
    auto step_start = std::chrono::high_resolution_clock::now();
    
    double time_step = (dt > 0.0) ? dt : config_.time_step;
    double sub_dt = time_step / config_.substeps;
    
    for (int s = 0; s < config_.substeps; ++s) {
        substep(sub_dt);
    }
    
    current_time_ += time_step;
    ++step_count_;
    
    auto step_end = std::chrono::high_resolution_clock::now();
    stats_.step_time_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
}

void Scene::substep(double sub_dt) {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    
    // Update time-varying force fields
    force_fields_->update_all(current_time_, sub_dt);
    
    // Apply external forces (gravity, force fields)
    apply_force_fields(sub_dt);
    
    // Integrate velocities (explicit Euler or symplectic)
    integrate_velocities(sub_dt);
    
    // Update broad phase if needed
    update_broad_phase();
    
    // Collision detection
    auto collision_start = std::chrono::high_resolution_clock::now();
    if (config_.enable_collision) {
        detect_collisions(sub_dt);
    }
    auto collision_end = std::chrono::high_resolution_clock::now();
    stats_.collision_time_ms = std::chrono::duration<double, std::milli>(collision_end - collision_start).count();
    
    // Resolve collisions (generate contacts, apply impulses)
    if (config_.enable_collision && !contacts_.empty()) {
        resolve_collisions(sub_dt);
    }
    
    // Solver step (constraints, PBD, etc.)
    auto solver_start = std::chrono::high_resolution_clock::now();
    if (solver_) {
        solver_->step(sub_dt, entities_);
    } else {
        // Default: simple integration if no solver
        for (auto& entity : entities_) {
            if (entity->is_dynamic()) {
                entity->integrate(sub_dt);
            }
        }
    }
    auto solver_end = std::chrono::high_resolution_clock::now();
    stats_.solver_time_ms = std::chrono::duration<double, std::milli>(solver_end - solver_start).count();
    
    // Update entity world transforms
    update_entity_transforms();
    
    // Update stats
    stats_.entity_count = entities_.size();
    size_t particle_count = 0;
    for (auto& e : entities_) {
        if (auto pe = std::dynamic_pointer_cast<ParticleEntity>(e)) {
            particle_count += pe->particle_count();
        }
    }
    stats_.particle_count = particle_count;
    stats_.contact_count = contacts_.size();
}

void Scene::update_broad_phase() {
    if (broad_phase_dirty_) {
        rebuild_broad_phase();
        broad_phase_dirty_ = false;
    }
}

void Scene::rebuild_broad_phase() {
    std::vector<BVH::Primitive> primitives;
    // Map primitive index to entity
    std::vector<uint64_t> prim_to_entity;
    
    for (const auto& entity : entities_) {
        if (!entity->is_collidable()) continue;
        // For each collision shape, create a primitive (AABB) for broad phase
        // For simplicity, we use entity AABB
        datatypes::AABB aabb = entity->world_aabb();
        // Create a pseudo primitive for the BVH
        BVH::Primitive prim;
        prim.centroid = aabb.center();
        prim.original_index = static_cast<uint32_t>(primitives.size());
        primitives.push_back(prim);
        prim_to_entity.push_back(entity->id());
    }
    
    if (!primitives.empty()) {
        broad_phase_->build(primitives);
        // Store entity mapping (we'll need it for queries)
        // In a real implementation, we'd store this mapping in the scene
    } else {
        broad_phase_->clear();
    }
}

void Scene::detect_collisions(double dt) {
    contacts_.clear();
    contact_manager_->clear();
    
    if (entities_.size() < 2) return;
    
    // For broad phase, we could use the BVH to find overlapping pairs
    // For now, brute-force O(N^2) AABB test
    for (size_t i = 0; i < entities_.size(); ++i) {
        auto& a = entities_[i];
        if (!a->is_collidable() || !a->is_enabled()) continue;
        datatypes::AABB aabb_a = a->world_aabb();
        
        for (size_t j = i + 1; j < entities_.size(); ++j) {
            auto& b = entities_[j];
            if (!b->is_collidable() || !b->is_enabled()) continue;
            if (!a->is_dynamic() && !b->is_dynamic()) continue; // both static/kinematic
            
            datatypes::AABB aabb_b = b->world_aabb();
            if (!aabb_a.intersects(aabb_b)) continue;
            
            // Narrow phase: delegate to entity pair
            std::vector<ContactPoint> pair_contacts;
            if (a->collide_with(*b, dt, pair_contacts)) {
                for (auto& cp : pair_contacts) {
                    cp.entity_a = a->id();
                    cp.entity_b = b->id();
                    cp.restitution_coef = config_.restitution;
                    cp.friction_coef = (cp.normal[2] > 0.7) ? config_.static_friction : config_.dynamic_friction;
                    contacts_.push_back(cp);
                    contact_manager_->add_contact(cp);
                }
            }
        }
    }
}

void Scene::resolve_collisions(double dt) {
    // Sort contacts for stability (e.g., by penetration depth)
    std::sort(contacts_.begin(), contacts_.end(),
        [](const ContactPoint& a, const ContactPoint& b) {
            return a.penetration > b.penetration;
        });
    
    // Apply sequential impulses (or PGS solver)
    for (int iter = 0; iter < config_.solver_iterations; ++iter) {
        for (auto& cp : contacts_) {
            auto ent_a = get_entity(cp.entity_a);
            auto ent_b = get_entity(cp.entity_b);
            if (!ent_a || !ent_b) continue;
            
            // Compute relative velocity at contact point
            datatypes::Vector3 vel_a = ent_a->velocity_at_point(cp.position);
            datatypes::Vector3 vel_b = ent_b->velocity_at_point(cp.position);
            datatypes::Vector3 rel_vel = vel_b - vel_a;
            
            double vn = rel_vel.dot(cp.normal);
            double vt = (rel_vel - cp.normal * vn).norm();
            
            // Normal impulse (resolve penetration and relative velocity)
            double effective_mass = 1.0 / (ent_a->inverse_mass(cp.position, cp.normal) +
                                          ent_b->inverse_mass(cp.position, cp.normal));
            
            // Baumgarte stabilization for penetration
            double bias = 0.2 / dt * std::max(0.0, cp.penetration - config_.contact_offset);
            double restitution = cp.restitution_coef;
            double desired_delta_v = -vn - bias;
            if (vn < -1.0) desired_delta_v += restitution * (-vn); // only if separating
            
            double normal_impulse_mag = effective_mass * desired_delta_v;
            // Accumulate impulse (warm starting)
            double new_impulse = std::max(0.0, cp.impulse.norm() + normal_impulse_mag);
            double delta_normal = new_impulse - cp.impulse.norm();
            datatypes::Vector3 normal_impulse = cp.normal * delta_normal;
            
            // Friction impulse (Coulomb)
            datatypes::Vector3 tangent = rel_vel - cp.normal * vn;
            if (tangent.squaredNorm() > 1e-12) {
                tangent.normalize();
                double max_friction = cp.friction_coef * new_impulse;
                double friction_impulse_mag = -vt * effective_mass;
                friction_impulse_mag = std::clamp(friction_impulse_mag, -max_friction, max_friction);
                datatypes::Vector3 friction_impulse = tangent * friction_impulse_mag;
                
                ent_a->apply_impulse(-normal_impulse - friction_impulse, cp.position);
                ent_b->apply_impulse( normal_impulse + friction_impulse, cp.position);
                
                cp.impulse = cp.normal * new_impulse;
                cp.tangent_impulse += friction_impulse;
            } else {
                ent_a->apply_impulse(-normal_impulse, cp.position);
                ent_b->apply_impulse( normal_impulse, cp.position);
                cp.impulse = cp.normal * new_impulse;
            }
        }
    }
}

void Scene::update_entity_transforms() {
    for (auto& entity : entities_) {
        entity->update_world_transform();
    }
}

void Scene::apply_force_fields(double dt) {
    // Apply global gravity
    if (config_.gravity.squaredNorm() > 0) {
        for (auto& entity : entities_) {
            if (entity->is_dynamic()) {
                entity->apply_force(config_.gravity * entity->mass());
            }
        }
    }
    
    // Apply custom force fields (on particles)
    // We'll gather all particles from particle-based entities
    std::vector<datatypes::Vector3> positions;
    std::vector<datatypes::Vector3> velocities;
    std::vector<datatypes::real> masses;
    std::vector<datatypes::Vector3> forces;
    std::vector<size_t> entity_offsets;
    std::vector<std::shared_ptr<BaseEntity>> particle_entities;
    
    for (auto& entity : entities_) {
        if (auto pe = std::dynamic_pointer_cast<ParticleEntity>(entity)) {
            entity_offsets.push_back(positions.size());
            particle_entities.push_back(entity);
            const auto& pos = pe->particle_positions();
            const auto& vel = pe->particle_velocities();
            const auto& mass = pe->particle_masses();
            positions.insert(positions.end(), pos.begin(), pos.end());
            velocities.insert(velocities.end(), vel.begin(), vel.end());
            masses.insert(masses.end(), mass.begin(), mass.end());
            forces.resize(positions.size(), datatypes::Vector3(0));
        }
    }
    
    if (!positions.empty()) {
        force_fields_->apply_all(positions, velocities, masses, forces, dt);
        
        // Distribute forces back to entities
        size_t offset = 0;
        for (size_t i = 0; i < particle_entities.size(); ++i) {
            auto pe = std::dynamic_pointer_cast<ParticleEntity>(particle_entities[i]);
            size_t count = pe->particle_count();
            for (size_t j = 0; j < count; ++j) {
                pe->add_particle_force(j, forces[offset + j]);
            }
            offset += count;
        }
    }
}

void Scene::integrate_velocities(double dt) {
    for (auto& entity : entities_) {
        if (entity->is_dynamic()) {
            // Apply damping
            if (config_.linear_damping > 0) {
                entity->set_velocity(entity->velocity() * (1.0 - config_.linear_damping * dt));
            }
            if (config_.angular_damping > 0 && entity->has_angular()) {
                entity->set_angular_velocity(entity->angular_velocity() * (1.0 - config_.angular_damping * dt));
            }
            // Velocity integration is done inside entity or solver
            entity->integrate_velocity(dt);
        }
    }
}

Scene::RayHit Scene::ray_cast(const datatypes::Ray& ray) const {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    RayHit best_hit;
    
    for (const auto& entity : entities_) {
        if (!entity->is_raycastable()) continue;
        auto hit = entity->ray_cast(ray);
        if (hit.hit && hit.t < best_hit.t) {
            best_hit = hit;
        }
    }
    return best_hit;
}

std::vector<uint64_t> Scene::overlap_aabb(const datatypes::AABB& aabb) const {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    std::vector<uint64_t> result;
    for (const auto& entity : entities_) {
        if (entity->world_aabb().intersects(aabb)) {
            result.push_back(entity->id());
        }
    }
    return result;
}

std::vector<uint64_t> Scene::overlap_sphere(const datatypes::Vector3& center, double radius) const {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    std::vector<uint64_t> result;
    double r2 = radius * radius;
    for (const auto& entity : entities_) {
        datatypes::AABB aabb = entity->world_aabb();
        if (aabb.distanceToPoint(center) <= radius) {
            // Further check could be done
            result.push_back(entity->id());
        }
    }
    return result;
}

} // namespace engine
} // namespace genesis