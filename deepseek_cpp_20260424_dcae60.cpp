// genesis/engine/solvers/kinematic_solver.h

#pragma once

#include "genesis/engine/solvers/base_solver.h"
#include "genesis/engine/entities/base_entity.h"
#include "genesis/datatypes.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

namespace genesis {
namespace engine {

// Forward declarations
class RigidEntity;
class ArticulatedEntity;

//------------------------------------------------------------------------------
// Joint types for articulated systems
//------------------------------------------------------------------------------
enum class JointType : uint8_t {
    FIXED = 0,
    REVOLUTE = 1,
    PRISMATIC = 2,
    SPHERICAL = 3,
    UNIVERSAL = 4,
    FLOATING = 5,
    PLANAR = 6,
    CONTINUOUS = 7  // revolute without limits
};

//------------------------------------------------------------------------------
// Joint limit structure
//------------------------------------------------------------------------------
struct JointLimits {
    bool has_limits = false;
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();
    double max_effort = std::numeric_limits<double>::infinity();
    double max_velocity = std::numeric_limits<double>::infinity();
    double damping = 0.0;
    double stiffness = 0.0;
};

//------------------------------------------------------------------------------
// Joint description
//------------------------------------------------------------------------------
struct Joint {
    std::string name;
    JointType type = JointType::REVOLUTE;
    
    // Parent and child links (indices into entity list)
    int parent_idx = -1;
    int child_idx = -1;
    
    // Transform from parent frame to joint frame
    datatypes::Transformr parent_to_joint;
    // Transform from child frame to joint frame (in child's local coordinates)
    datatypes::Transformr child_to_joint;
    
    // Joint axis (for revolute/prismatic)
    datatypes::Vector3 axis = {0.0, 0.0, 1.0};
    
    // Joint limits
    JointLimits limits;
    
    // Current state
    double position = 0.0;      // joint coordinate (angle or displacement)
    double velocity = 0.0;
    double acceleration = 0.0;
    double effort = 0.0;
    
    // For spherical joints (orientation quaternion)
    datatypes::Quat orientation;
    datatypes::Vector3 angular_velocity;
};

//------------------------------------------------------------------------------
// Link (rigid body) in articulated chain
//------------------------------------------------------------------------------
struct Link {
    std::string name;
    std::shared_ptr<RigidEntity> entity;
    int parent_joint = -1;
    std::vector<int> child_joints;
    datatypes::Transformr inertial_origin;  // transform from link frame to inertial
};

//------------------------------------------------------------------------------
// Articulated kinematic chain
//------------------------------------------------------------------------------
class ArticulatedSystem {
public:
    ArticulatedSystem() = default;
    
    // Build from URDF or similar description
    void add_link(const Link& link);
    void add_joint(const Joint& joint);
    void finalize();
    
    // Forward kinematics: compute link transforms given joint positions
    void forward_kinematics(const Eigen::VectorXd& q);
    
    // Jacobian computation for a specific end-effector link
    Eigen::MatrixXd compute_jacobian(int link_idx, const datatypes::Vector3& local_point = {0,0,0}) const;
    
    // Inverse kinematics using damped least squares
    bool inverse_kinematics(int ee_link_idx,
                            const datatypes::Transformr& target_pose,
                            Eigen::VectorXd& q,
                            const datatypes::Vector3& local_point = {0,0,0},
                            int max_iterations = 50,
                            double tolerance = 1e-4);
    
    // Accessors
    const std::vector<Link>& links() const { return links_; }
    const std::vector<Joint>& joints() const { return joints_; }
    size_t num_dofs() const { return dof_indices_.size(); }
    
    // Get current world transform of a link
    datatypes::Transformr link_transform(int link_idx) const;
    
private:
    std::vector<Link> links_;
    std::vector<Joint> joints_;
    std::vector<int> dof_indices_;  // mapping from global dof index to joint index
    std::vector<int> joint_q_offset_; // starting index in q vector for each joint
    std::vector<int> parent_link_of_joint_;
    std::vector<std::vector<int>> children_of_link_;
    std::vector<datatypes::Transformr> link_world_poses_;
    
    void compute_link_pose(int link_idx);
};

//------------------------------------------------------------------------------
// Kinematic solver configuration
//------------------------------------------------------------------------------
struct KinematicConfig {
    // IK solver parameters
    int ik_max_iterations = 50;
    double ik_tolerance = 1e-4;
    double ik_damping = 0.01;
    
    // Constraint force parameters
    double baumgarte_factor = 0.1;
    double constraint_force_mixing = 1e-5;
    int position_iterations = 5;
    int velocity_iterations = 3;
    
    // For continuous joints
    bool wrap_angles = true;
    
    // Joint limit handling
    bool enforce_joint_limits = true;
    double joint_limit_stiffness = 1000.0;
};

//------------------------------------------------------------------------------
// KinematicSolver: solves articulated rigid body dynamics with constraints.
// Handles forward/inverse kinematics, joint limits, and constraint forces.
//------------------------------------------------------------------------------
class KinematicSolver : public BaseSolver {
public:
    explicit KinematicSolver(const SolverConfig& config = SolverConfig{});
    ~KinematicSolver() override;

    std::string solver_type() const override { return "Kinematic"; }

    // Initialization
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Main step: updates articulated systems and resolves constraints
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;

    // Entity management
    void on_entity_added(std::shared_ptr<BaseEntity> entity) override;
    void on_entity_removed(std::shared_ptr<BaseEntity> entity) override;

    // Kinematic-specific configuration
    void set_kinematic_config(const KinematicConfig& config) { kin_config_ = config; }
    const KinematicConfig& kinematic_config() const { return kin_config_; }

    // Build an articulated system from entities and joint descriptions
    std::shared_ptr<ArticulatedSystem> create_articulated_system(const std::string& name);
    void add_articulated_system(std::shared_ptr<ArticulatedSystem> system);
    std::shared_ptr<ArticulatedSystem> get_articulated_system(const std::string& name);

    // Forward kinematics on a system
    void forward_kinematics(const std::string& system_name, const Eigen::VectorXd& q);

    // Inverse kinematics
    bool inverse_kinematics(const std::string& system_name,
                            int ee_link_idx,
                            const datatypes::Transformr& target_pose,
                            Eigen::VectorXd& q,
                            const datatypes::Vector3& local_point = {0,0,0});

    // Direct control: set joint positions/velocities (overrides dynamics)
    void set_joint_state(const std::string& system_name, int joint_idx, double pos, double vel = 0.0);
    void set_joint_states(const std::string& system_name, const Eigen::VectorXd& q, const Eigen::VectorXd& qdot);

    // Compute mass matrix for an articulated system
    Eigen::MatrixXd compute_mass_matrix(const ArticulatedSystem& system);

    // Compute inverse dynamics (given q, qdot, qddot, compute joint torques)
    Eigen::VectorXd inverse_dynamics(const ArticulatedSystem& system,
                                     const Eigen::VectorXd& q,
                                     const Eigen::VectorXd& qdot,
                                     const Eigen::VectorXd& qddot,
                                     const std::vector<datatypes::Vector3>& external_forces = {});

private:
    KinematicConfig kin_config_;
    
    // All articulated systems managed by this solver
    std::vector<std::shared_ptr<ArticulatedSystem>> articulated_systems_;
    std::unordered_map<std::string, std::shared_ptr<ArticulatedSystem>> systems_by_name_;
    
    // Per-system state vectors
    struct SystemState {
        Eigen::VectorXd q;
        Eigen::VectorXd qdot;
        Eigen::VectorXd qddot;
        Eigen::VectorXd tau;          // joint efforts
        Eigen::VectorXd tau_ext;      // external torques
        bool needs_update = true;
    };
    std::unordered_map<ArticulatedSystem*, SystemState> system_states_;
    
    // Temporary matrices for dynamics computations
    Eigen::MatrixXd M_;      // mass matrix
    Eigen::VectorXd C_;      // Coriolis/centrifugal terms
    Eigen::VectorXd G_;      // gravity terms
    
    // Internal methods
    void update_system_state(ArticulatedSystem& system, double dt);
    void solve_joint_constraints(ArticulatedSystem& system, double dt);
    void apply_joint_limits(ArticulatedSystem& system, SystemState& state);
    void integrate_articulated_bodies(ArticulatedSystem& system, SystemState& state, double dt);
    
    // Recursive Newton-Euler for inverse dynamics
    void rnea_pass1(const ArticulatedSystem& system,
                    const SystemState& state,
                    std::vector<datatypes::Vector3>& v,
                    std::vector<datatypes::Vector3>& a,
                    std::vector<datatypes::Vector3>& w,
                    std::vector<datatypes::Vector3>& alpha);
    void rnea_pass2(const ArticulatedSystem& system,
                    const SystemState& state,
                    const std::vector<datatypes::Vector3>& v,
                    const std::vector<datatypes::Vector3>& a,
                    const std::vector<datatypes::Vector3>& w,
                    const std::vector<datatypes::Vector3>& alpha,
                    std::vector<datatypes::Vector3>& f,
                    Eigen::VectorXd& tau);
    
    // Composite rigid body algorithm for mass matrix
    void crba(const ArticulatedSystem& system, Eigen::MatrixXd& M);
    
    // Utility: convert between joint types and DOF sizes
    static int joint_dof_count(JointType type);
    static void joint_transform(const Joint& joint, double q, datatypes::Transformr& T);
    static void joint_motion_subspace(const Joint& joint, Eigen::Matrix<double, 6, Eigen::Dynamic>& S);
};

} // namespace engine
} // namespace genesis