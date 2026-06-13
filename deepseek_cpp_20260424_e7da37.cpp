// genesis/engine/solvers/pbd_solver.h

#pragma once

#include "genesis/engine/solvers/base_solver.h"
#include "genesis/engine/entities/pbd_entity.h"
#include "genesis/datatypes.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// PBD constraint types
//------------------------------------------------------------------------------
enum class PBDConstraintType : uint8_t {
    DISTANCE = 0,
    BENDING = 1,
    VOLUME = 2,
    SHAPE_MATCHING = 3,
    COLLISION = 4,
    ATTACHMENT = 5,
    PIN = 6,
    TET_VOLUME = 7,
    ANGLE = 8,
    SLIDING = 9,
    CUSTOM = 10
};

//------------------------------------------------------------------------------
// Base class for all PBD constraints
//------------------------------------------------------------------------------
class PBDConstraint {
public:
    PBDConstraintType type;
    double stiffness = 1.0;
    bool enabled = true;
    
    PBDConstraint(PBDConstraintType t, double s = 1.0) : type(t), stiffness(s) {}
    virtual ~PBDConstraint() = default;
    
    // Project the constraint for a single iteration
    virtual void project(const std::vector<datatypes::Vector3>& positions,
                         std::vector<datatypes::Vector3>& delta,
                         const std::vector<double>& inv_mass,
                         double dt) = 0;
    
    // Number of indices this constraint affects
    virtual size_t index_count() const = 0;
    virtual const std::vector<int>& indices() const = 0;
};

//------------------------------------------------------------------------------
// Distance constraint between two particles
//------------------------------------------------------------------------------
class DistanceConstraint : public PBDConstraint {
public:
    int idx0, idx1;
    double rest_length;
    
    DistanceConstraint(int i0, int i1, double length, double stiffness = 1.0)
        : PBDConstraint(PBDConstraintType::DISTANCE, stiffness), idx0(i0), idx1(i1), rest_length(length) {}
    
    void project(const std::vector<datatypes::Vector3>& positions,
                 std::vector<datatypes::Vector3>& delta,
                 const std::vector<double>& inv_mass,
                 double dt) override;
    
    size_t index_count() const override { return 2; }
    const std::vector<int>& indices() const override { static std::vector<int> idx = {idx0, idx1}; return idx; }
};

//------------------------------------------------------------------------------
// Bending constraint (dihedral angle between two triangles)
//------------------------------------------------------------------------------
class BendingConstraint : public PBDConstraint {
public:
    int idx0, idx1, idx2, idx3;
    double rest_angle;
    
    BendingConstraint(int i0, int i1, int i2, int i3, double angle, double stiffness = 1.0)
        : PBDConstraint(PBDConstraintType::BENDING, stiffness), idx0(i0), idx1(i1), idx2(i2), idx3(i3), rest_angle(angle) {}
    
    void project(const std::vector<datatypes::Vector3>& positions,
                 std::vector<datatypes::Vector3>& delta,
                 const std::vector<double>& inv_mass,
                 double dt) override;
    
    size_t index_count() const override { return 4; }
    const std::vector<int>& indices() const override { static std::vector<int> idx = {idx0, idx1, idx2, idx3}; return idx; }
};

//------------------------------------------------------------------------------
// Volume constraint for tetrahedra (preserves volume)
//------------------------------------------------------------------------------
class VolumeConstraint : public PBDConstraint {
public:
    int idx0, idx1, idx2, idx3;
    double rest_volume;
    
    VolumeConstraint(int i0, int i1, int i2, int i3, double vol, double stiffness = 1.0)
        : PBDConstraint(PBDConstraintType::VOLUME, stiffness), idx0(i0), idx1(i1), idx2(i2), idx3(i3), rest_volume(vol) {}
    
    void project(const std::vector<datatypes::Vector3>& positions,
                 std::vector<datatypes::Vector3>& delta,
                 const std::vector<double>& inv_mass,
                 double dt) override;
    
    size_t index_count() const override { return 4; }
    const std::vector<int>& indices() const override { static std::vector<int> idx = {idx0, idx1, idx2, idx3}; return idx; }
};

//------------------------------------------------------------------------------
// Shape matching constraint (for soft bodies)
//------------------------------------------------------------------------------
class ShapeMatchingConstraint : public PBDConstraint {
public:
    std::vector<int> particle_indices;
    std::vector<datatypes::Vector3> rest_positions;
    datatypes::Vector3 rest_com;
    std::vector<datatypes::Vector3> rest_relative;
    double beta = 0.0; // rigidness (0 = full shape match, 1 = rigid)
    
    ShapeMatchingConstraint(const std::vector<int>& indices,
                            const std::vector<datatypes::Vector3>& rest,
                            double stiffness = 1.0);
    
    void project(const std::vector<datatypes::Vector3>& positions,
                 std::vector<datatypes::Vector3>& delta,
                 const std::vector<double>& inv_mass,
                 double dt) override;
    
    size_t index_count() const override { return particle_indices.size(); }
    const std::vector<int>& indices() const override { return particle_indices; }
};

//------------------------------------------------------------------------------
// PBD Solver configuration
//------------------------------------------------------------------------------
struct PBDConfig {
    int iterations = 5;
    int sub_iterations = 1;
    double velocity_damping = 0.0;
    double max_velocity = 100.0;
    bool enable_continuous_collision = false;
    double collision_stiffness = 0.8;
    double collision_friction = 0.2;
    bool use_xpbd = false; // extended PBD for stiffness independence
};

//------------------------------------------------------------------------------
// PBD Solver class - Position Based Dynamics for fast, stable simulation
//------------------------------------------------------------------------------
class PBDSolver : public BaseSolver {
public:
    explicit PBDSolver(const SolverConfig& config = SolverConfig{});
    ~PBDSolver() override;

    std::string solver_type() const override { return "PBD"; }

    // Initialization
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Main step
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;

    // Entity management
    void on_entity_added(std::shared_ptr<BaseEntity> entity) override;
    void on_entity_removed(std::shared_ptr<BaseEntity> entity) override;

    // PBD-specific configuration
    void set_pbd_config(const PBDConfig& config) { pbd_config_ = config; }
    const PBDConfig& pbd_config() const { return pbd_config_; }

    // Constraint management
    void add_constraint(std::shared_ptr<PBDConstraint> constraint, uint64_t entity_id);
    void remove_constraints(uint64_t entity_id);
    void clear_constraints();
    
    // Direct access to constraints for debugging
    const std::vector<std::shared_ptr<PBDConstraint>>& constraints() const { return constraints_; }

private:
    PBDConfig pbd_config_;
    
    // Entity data
    struct EntityPBDState {
        std::shared_ptr<PBDEntity> entity;
        std::vector<datatypes::Vector3> positions;
        std::vector<datatypes::Vector3> prev_positions;
        std::vector<datatypes::Vector3> velocities;
        std::vector<double> inv_mass;
        std::vector<datatypes::Vector3> delta; // temporary for constraint projection
    };
    std::vector<EntityPBDState> entity_states_;
    std::unordered_map<uint64_t, size_t> entity_to_index_;
    bool states_dirty_ = true;
    
    // All constraints in the system (per-entity constraints stored in entity,
    // but we also maintain a global list for efficient solving)
    std::vector<std::shared_ptr<PBDConstraint>> constraints_;
    std::unordered_map<uint64_t, std::vector<size_t>> entity_constraint_indices_;
    
    // Internal methods
    void rebuild_states(const std::vector<std::shared_ptr<BaseEntity>>& entities);
    void predict_positions(double dt);
    void project_constraints(double dt);
    void update_velocities(double dt);
    void apply_collisions(double dt);
    void enforce_max_velocity();
    
    // Helper for constraint solving
    void solve_distance_constraint(EntityPBDState& state, const DistanceConstraint& c, double dt);
    void solve_bending_constraint(EntityPBDState& state, const BendingConstraint& c, double dt);
    void solve_volume_constraint(EntityPBDState& state, const VolumeConstraint& c, double dt);
    void solve_shape_matching_constraint(EntityPBDState& state, ShapeMatchingConstraint& c, double dt);
};

} // namespace engine
} // namespace genesis