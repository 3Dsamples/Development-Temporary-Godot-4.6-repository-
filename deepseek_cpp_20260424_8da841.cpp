// genesis/engine/scene.h

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <mutex>
#include "genesis/datatypes.h"
#include "genesis/engine/entities/base_entity.h"
#include "genesis/engine/solvers/base_solver.h"
#include "genesis/engine/force_fields.h"
#include "genesis/engine/bvh.h"

namespace genesis {
namespace engine {

// Forward declarations
class Simulator;
class Entity;
class BaseSolver;
class ForceFieldManager;
class ContactManager;

//------------------------------------------------------------------------------
// Scene configuration
//------------------------------------------------------------------------------
struct SceneConfig {
    std::string name = "default";
    datatypes::Vector3 gravity = {0.0, 0.0, -9.80665};
    double time_step = 0.01;
    int substeps = 1;
    int solver_iterations = 5;
    bool enable_collision = true;
    bool enable_self_collision = true;
    bool enable_contact_islands = true;
    double contact_offset = 0.005;
    double restitution = 0.5;
    double static_friction = 0.5;
    double dynamic_friction = 0.3;
    size_t max_contacts_per_pair = 4;
    bool continuous_collision = false;
    double linear_damping = 0.0;
    double angular_damping = 0.0;
};

//------------------------------------------------------------------------------
// Contact point information
//------------------------------------------------------------------------------
struct ContactPoint {
    datatypes::Vector3 position;        // World space contact position
    datatypes::Vector3 normal;          // Contact normal (from A to B)
    double penetration = 0.0;           // Penetration depth
    datatypes::Vector3 impulse;         // Normal impulse applied
    datatypes::Vector3 tangent_impulse; // Friction impulse
    uint64_t entity_a = 0;              // First entity ID
    uint64_t entity_b = 0;              // Second entity ID
    uint32_t shape_a = 0;               // Shape index in entity A
    uint32_t shape_b = 0;               // Shape index in entity B
    double restitution_coef = 0.5;
    double friction_coef = 0.5;
};

//------------------------------------------------------------------------------
// Scene class: container for entities, solvers, and simulation state.
// Manages the simulation loop and coordinates between components.
//------------------------------------------------------------------------------
class Scene {
public:
    // Constructors
    explicit Scene(const SceneConfig& config = SceneConfig{});
    ~Scene();

    // Prevent copying (heavy resource)
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    // Move allowed
    Scene(Scene&&) = default;
    Scene& operator=(Scene&&) = default;

    //--- Configuration ---
    void set_config(const SceneConfig& config);
    const SceneConfig& config() const { return config_; }

    //--- Entity management ---
    // Add an entity to the scene (takes ownership)
    void add_entity(std::shared_ptr<BaseEntity> entity);
    void remove_entity(uint64_t entity_id);
    void remove_entity(const std::string& name);
    void clear_entities();

    // Access entities
    std::shared_ptr<BaseEntity> get_entity(uint64_t id) const;
    std::shared_ptr<BaseEntity> get_entity(const std::string& name) const;
    const std::vector<std::shared_ptr<BaseEntity>>& entities() const { return entities_; }
    size_t entity_count() const { return entities_.size(); }

    // Find entities by type (dynamic_cast based)
    template<typename T>
    std::vector<std::shared_ptr<T>> find_entities_of_type() const;

    //--- Solver management ---
    void set_solver(std::shared_ptr<BaseSolver> solver);
    std::shared_ptr<BaseSolver> solver() const { return solver_; }

    //--- Force fields ---
    ForceFieldManager& force_field_manager() { return *force_fields_; }
    const ForceFieldManager& force_field_manager() const { return *force_fields_; }

    //--- Simulation control ---
    // Reset simulation to initial state
    void reset();

    // Advance simulation by given time (if not provided, use config time_step)
    void step(double dt = 0.0);

    // Step physics substeps (internal)
    void substep(double sub_dt);

    //--- Time and state ---
    double current_time() const { return current_time_; }
    uint64_t step_count() const { return step_count_; }

    //--- Collision and contacts ---
    void enable_collision(bool enable) { config_.enable_collision = enable; }
    bool is_collision_enabled() const { return config_.enable_collision; }

    // Get current contacts (from last step)
    const std::vector<ContactPoint>& contacts() const { return contacts_; }

    //--- Queries ---
    // Ray cast against all entities
    struct RayHit {
        bool hit = false;
        double t = std::numeric_limits<double>::max();
        datatypes::Vector3 point;
        datatypes::Vector3 normal;
        uint64_t entity_id = 0;
        std::shared_ptr<BaseEntity> entity;
    };
    RayHit ray_cast(const datatypes::Ray& ray) const;

    // Overlap query: return entities whose AABB overlaps the given AABB
    std::vector<uint64_t> overlap_aabb(const datatypes::AABB& aabb) const;

    // Sphere overlap query
    std::vector<uint64_t> overlap_sphere(const datatypes::Vector3& center, double radius) const;

    //--- Broad phase acceleration ---
    void rebuild_broad_phase();
    const BVH& broad_phase() const { return *broad_phase_; }

    //--- Statistics ---
    struct Stats {
        size_t entity_count = 0;
        size_t particle_count = 0;
        size_t constraint_count = 0;
        size_t contact_count = 0;
        double step_time_ms = 0.0;
        double collision_time_ms = 0.0;
        double solver_time_ms = 0.0;
    };
    Stats get_stats() const { return stats_; }

    //--- Debug visualization ---
    void debug_draw_contacts(bool enabled) { debug_draw_contacts_ = enabled; }
    bool debug_draw_contacts() const { return debug_draw_contacts_; }
    void debug_draw_broad_phase(bool enabled) { debug_draw_broad_phase_ = enabled; }

private:
    SceneConfig config_;
    std::vector<std::shared_ptr<BaseEntity>> entities_;
    std::unordered_map<uint64_t, std::shared_ptr<BaseEntity>> entity_by_id_;
    std::unordered_map<std::string, std::shared_ptr<BaseEntity>> entity_by_name_;

    std::shared_ptr<BaseSolver> solver_;
    std::unique_ptr<ForceFieldManager> force_fields_;
    std::unique_ptr<ContactManager> contact_manager_;

    std::unique_ptr<BVH> broad_phase_;    // Broad phase for static/kinematic entities
    bool broad_phase_dirty_ = true;

    double current_time_ = 0.0;
    uint64_t step_count_ = 0;
    std::vector<ContactPoint> contacts_;

    Stats stats_;

    bool debug_draw_contacts_ = false;
    bool debug_draw_broad_phase_ = false;

    mutable std::mutex scene_mutex_; // For thread-safe access

    // Internal helpers
    void update_broad_phase();
    void detect_collisions(double dt);
    void resolve_collisions(double dt);
    void update_entity_transforms();
    void apply_force_fields(double dt);
    void integrate_velocities(double dt);
};

//------------------------------------------------------------------------------
// Template implementation
//------------------------------------------------------------------------------
template<typename T>
std::vector<std::shared_ptr<T>> Scene::find_entities_of_type() const {
    std::vector<std::shared_ptr<T>> result;
    for (const auto& e : entities_) {
        auto casted = std::dynamic_pointer_cast<T>(e);
        if (casted) result.push_back(casted);
    }
    return result;
}

} // namespace engine
} // namespace genesis