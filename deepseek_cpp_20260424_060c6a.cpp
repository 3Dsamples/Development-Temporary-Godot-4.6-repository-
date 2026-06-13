// genesis/engine/solvers/tool_solver.h

#pragma once

#include "genesis/engine/solvers/base_solver.h"
#include "genesis/engine/entities/tool_entity.h"
#include "genesis/datatypes.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Tool types enumeration
//------------------------------------------------------------------------------
enum class ToolType : uint8_t {
    GRIPPER = 0,
    SPATULA = 1,
    CUTTER = 2,
    PUSHER = 3,
    ROLLER = 4,
    SPRAYER = 5,
    CUSTOM = 6
};

//------------------------------------------------------------------------------
// Tool action types
//------------------------------------------------------------------------------
enum class ToolAction : uint8_t {
    NONE = 0,
    GRASP = 1,
    RELEASE = 2,
    PUSH = 3,
    PULL = 4,
    CUT = 5,
    SPRAY = 6,
    ROLL = 7,
    CUSTOM = 8
};

//------------------------------------------------------------------------------
// Tool state structure
//------------------------------------------------------------------------------
struct ToolState {
    datatypes::Transformr transform;        // Current tool pose
    datatypes::Vector3 velocity;            // Linear velocity
    datatypes::Vector3 angular_velocity;    // Angular velocity
    ToolAction current_action = ToolAction::NONE;
    bool is_active = false;
    double action_value = 0.0;              // e.g., grasp force, cut depth, spray rate
    std::vector<uint64_t> affected_entities; // Entities currently interacting with tool
    
    // For grippers
    std::vector<std::pair<uint64_t, datatypes::Transformr>> grasped_objects; // entity id + relative transform
    double grasp_force = 100.0;
    
    // For cutters
    datatypes::Vector3 cut_plane_normal;
    datatypes::Vector3 cut_plane_point;
    double cut_depth = 0.0;
    
    // For sprayers
    double spray_rate = 1.0;
    double spray_radius = 0.5;
    uint32_t sprayed_material_id = 0;
};

//------------------------------------------------------------------------------
// Tool configuration
//------------------------------------------------------------------------------
struct ToolConfig {
    ToolType type = ToolType::GRIPPER;
    
    // Control parameters
    double max_force = 1000.0;
    double max_velocity = 1.0;
    double max_angular_velocity = 3.14159;
    
    // Grasp parameters
    double grasp_tolerance = 0.1;            // max distance for grasp
    double grasp_stiffness = 5000.0;
    double grasp_damping = 50.0;
    double release_velocity = 0.5;
    
    // Cut parameters
    double cut_force_threshold = 500.0;
    double cut_plane_thickness = 0.01;
    
    // Spray parameters
    double particle_radius = 0.02;
    double particle_density = 1000.0;
    int particles_per_second = 100;
    
    // Interaction parameters
    double interaction_radius = 0.5;
    double push_stiffness = 1000.0;
    double friction_coefficient = 0.3;
};

//------------------------------------------------------------------------------
// Tool Solver class - handles tool interactions with the scene.
// Tools can grasp, push, cut, spray particles, etc.
//------------------------------------------------------------------------------
class ToolSolver : public BaseSolver {
public:
    explicit ToolSolver(const SolverConfig& config = SolverConfig{});
    ~ToolSolver() override;

    std::string solver_type() const override { return "Tool"; }

    // Initialization
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Main step
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;

    // Entity management
    void on_entity_added(std::shared_ptr<BaseEntity> entity) override;
    void on_entity_removed(std::shared_ptr<BaseEntity> entity) override;

    // Tool management
    void add_tool(std::shared_ptr<ToolEntity> tool, const ToolConfig& config = ToolConfig{});
    void remove_tool(uint64_t tool_id);
    std::shared_ptr<ToolEntity> get_tool(uint64_t tool_id) const;
    const std::unordered_map<uint64_t, ToolState>& tool_states() const { return tool_states_; }
    
    // Tool control
    void set_tool_transform(uint64_t tool_id, const datatypes::Transformr& transform);
    void set_tool_velocity(uint64_t tool_id, const datatypes::Vector3& linear, const datatypes::Vector3& angular);
    void set_tool_action(uint64_t tool_id, ToolAction action, double value = 0.0);
    
    // Grasping
    void grasp(uint64_t tool_id, uint64_t target_entity_id);
    void release(uint64_t tool_id);
    
    // Cutting
    void cut(uint64_t tool_id, const datatypes::Vector3& plane_point, const datatypes::Vector3& plane_normal);
    
    // Spraying
    void spray(uint64_t tool_id, bool enable, double rate = 1.0);
    
    // Query
    bool is_grasping(uint64_t tool_id) const;
    std::vector<uint64_t> get_grasped_objects(uint64_t tool_id) const;

private:
    // Tool states
    std::unordered_map<uint64_t, std::shared_ptr<ToolEntity>> tools_;
    std::unordered_map<uint64_t, ToolState> tool_states_;
    std::unordered_map<uint64_t, ToolConfig> tool_configs_;
    
    // All entities in the scene (for interaction)
    std::vector<std::shared_ptr<BaseEntity>> all_entities_;
    bool entities_dirty_ = true;
    
    // Internal methods
    void update_entity_list(const std::vector<std::shared_ptr<BaseEntity>>& entities);
    
    // Action handlers
    void handle_grasp(ToolState& state, const ToolConfig& config, double dt);
    void handle_release(ToolState& state, const ToolConfig& config);
    void handle_push(ToolState& state, const ToolConfig& config, double dt);
    void handle_cut(ToolState& state, const ToolConfig& config, double dt);
    void handle_spray(ToolState& state, const ToolConfig& config, double dt);
    void handle_roll(ToolState& state, const ToolConfig& config, double dt);
    
    // Grasp helpers
    void apply_grasp_forces(ToolState& state, const ToolConfig& config, double dt);
    void update_grasp_transforms(ToolState& state);
    
    // Push helpers
    void apply_push_forces(const datatypes::Transformr& tool_transform, double radius, double stiffness, double dt);
    
    // Cut helpers
    void perform_cut(ToolState& state, const ToolConfig& config);
    
    // Spray helpers
    void emit_particles(ToolState& state, const ToolConfig& config, double dt);
    
    // Interaction queries
    std::vector<std::shared_ptr<BaseEntity>> find_entities_in_radius(const datatypes::Vector3& center, double radius) const;
    std::vector<std::shared_ptr<RigidEntity>> find_rigid_entities_in_radius(const datatypes::Vector3& center, double radius) const;
    std::vector<std::shared_ptr<ParticleEntity>> find_particle_entities_in_radius(const datatypes::Vector3& center, double radius) const;
    
    // Utility
    bool can_grasp(uint64_t tool_id, uint64_t entity_id) const;
    void apply_impulse_to_entity(std::shared_ptr<BaseEntity> entity, const datatypes::Vector3& point, const datatypes::Vector3& impulse);
};

} // namespace engine
} // namespace genesis