// genesis/engine/solvers/kinematic_solver.cpp

#include "genesis/engine/solvers/kinematic_solver.h"
#include "genesis/engine/entities/rigid_entity.h"
#include <algorithm>
#include <numeric>
#include <stack>
#include <queue>
#include <cmath>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return S;
}

inline Eigen::Matrix<double, 6, 1> spatial_velocity(const Eigen::Vector3d& w, const Eigen::Vector3d& v) {
    Eigen::Matrix<double, 6, 1> V;
    V << w, v;
    return V;
}

inline Eigen::Matrix<double, 6, 6> spatial_inertia(double mass, const Eigen::Matrix3d& I, const Eigen::Vector3d& com) {
    Eigen::Matrix<double, 6, 6> M;
    Eigen::Matrix3d skew_com = skew(com);
    M.block<3,3>(0,0) = I + mass * skew_com * skew_com.transpose();
    M.block<3,3>(0,3) = mass * skew_com;
    M.block<3,3>(3,0) = -mass * skew_com;
    M.block<3,3>(3,3) = mass * Eigen::Matrix3d::Identity();
    return M;
}

inline Eigen::Matrix<double, 6, 6> spatial_transform(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    Eigen::Matrix<double, 6, 6> X;
    X.block<3,3>(0,0) = R;
    X.block<3,3>(0,3) = Eigen::Matrix3d::Zero();
    X.block<3,3>(3,0) = -R * skew(p);
    X.block<3,3>(3,3) = R;
    return X;
}

inline Eigen::Matrix<double, 6, 6> spatial_transform_inverse(const Eigen::Matrix<double, 6, 6>& X) {
    Eigen::Matrix<double, 6, 6> Xinv;
    Xinv.block<3,3>(0,0) = X.block<3,3>(0,0).transpose();
    Xinv.block<3,3>(0,3) = Eigen::Matrix3d::Zero();
    Xinv.block<3,3>(3,0) = X.block<3,3>(3,0).transpose();
    Xinv.block<3,3>(3,3) = X.block<3,3>(0,0).transpose();
    return Xinv;
}

inline Eigen::Matrix<double, 6, 1> cross_motion(const Eigen::Matrix<double, 6, 1>& v) {
    Eigen::Matrix<double, 6, 1> vcross;
    vcross << skew(v.head<3>()) * v.head<3>(), skew(v.head<3>()) * v.tail<3>();
    return vcross;
}

inline Eigen::Matrix<double, 6, 1> cross_force(const Eigen::Matrix<double, 6, 1>& v, const Eigen::Matrix<double, 6, 1>& f) {
    Eigen::Matrix<double, 6, 1> fcross;
    fcross << skew(v.head<3>()) * f.head<3>() + skew(v.tail<3>()) * f.tail<3>(),
              skew(v.head<3>()) * f.tail<3>();
    return fcross;
}

//------------------------------------------------------------------------------
// ArticulatedSystem implementation
//------------------------------------------------------------------------------
void ArticulatedSystem::add_link(const Link& link) {
    links_.push_back(link);
}

void ArticulatedSystem::add_joint(const Joint& joint) {
    joints_.push_back(joint);
}

void ArticulatedSystem::finalize() {
    // Build parent/child relationships
    parent_link_of_joint_.resize(joints_.size(), -1);
    children_of_link_.resize(links_.size());
    for (size_t j = 0; j < joints_.size(); ++j) {
        parent_link_of_joint_[j] = joints_[j].parent_idx;
        if (joints_[j].child_idx >= 0) {
            children_of_link_[joints_[j].child_idx].push_back(static_cast<int>(j));
        }
    }
    
    // Compute DOF indices and offsets
    dof_indices_.clear();
    joint_q_offset_.resize(joints_.size(), 0);
    int offset = 0;
    for (size_t j = 0; j < joints_.size(); ++j) {
        joint_q_offset_[j] = offset;
        int dof = joint_dof_count(joints_[j].type);
        for (int d = 0; d < dof; ++d) {
            dof_indices_.push_back(static_cast<int>(j));
        }
        offset += dof;
    }
    
    // Initialize link world poses to identity
    link_world_poses_.assign(links_.size(), datatypes::Transformr());
}

void ArticulatedSystem::forward_kinematics(const Eigen::VectorXd& q) {
    if (q.size() != static_cast<int>(dof_indices_.size())) {
        throw std::invalid_argument("q vector size mismatch");
    }
    
    // Start with base link(s)
    for (size_t i = 0; i < links_.size(); ++i) {
        if (links_[i].parent_joint == -1) {
            link_world_poses_[i] = links_[i].entity->transform();
            compute_link_pose(static_cast<int>(i));
        }
    }
}

void ArticulatedSystem::compute_link_pose(int link_idx) {
    const Link& link = links_[link_idx];
    
    for (int child_joint_idx : link.child_joints) {
        const Joint& joint = joints_[child_joint_idx];
        int child_idx = joint.child_idx;
        if (child_idx < 0) continue;
        
        // Get joint transform based on q
        double q_val = 0.0;
        int dof_offset = joint_q_offset_[child_joint_idx];
        if (joint_dof_count(joint.type) == 1) {
            q_val = q[dof_offset];
        }
        // For multi-dof joints, we would assemble a transform from the quaternion/vector
        
        datatypes::Transformr T_joint;
        joint_transform(joint, q_val, T_joint);
        
        // Compute child's world pose
        datatypes::Transformr parent_pose = link_world_poses_[link_idx];
        datatypes::Transformr child_pose = parent_pose * joint.parent_to_joint * T_joint * joint.child_to_joint.inverse();
        link_world_poses_[child_idx] = child_pose;
        
        // Apply to the actual entity
        if (links_[child_idx].entity) {
            links_[child_idx].entity->set_transform(child_pose);
        }
        
        // Recurse
        compute_link_pose(child_idx);
    }
}

Eigen::MatrixXd ArticulatedSystem::compute_jacobian(int link_idx, const datatypes::Vector3& local_point) const {
    int n_dofs = static_cast<int>(dof_indices_.size());
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n_dofs);
    
    if (link_idx < 0 || link_idx >= static_cast<int>(links_.size())) return J;
    
    // Get world position of the point
    datatypes::Transformr link_pose = link_world_poses_[link_idx];
    datatypes::Vector3 world_point = link_pose.transformPoint(local_point);
    
    // Traverse from link up to base, accumulating Jacobian columns
    int current_link = link_idx;
    while (current_link >= 0) {
        int joint_idx = links_[current_link].parent_joint;
        if (joint_idx < 0) break;
        
        const Joint& joint = joints_[joint_idx];
        datatypes::Transformr joint_pose = link_world_poses_[joint.parent_idx] * joint.parent_to_joint;
        datatypes::Vector3 joint_axis_world = joint_pose.rotation.rotate(joint.axis);
        datatypes::Vector3 joint_pos_world = joint_pose.translation;
        
        int dof = joint_dof_count(joint.type);
        int q_offset = joint_q_offset_[joint_idx];
        
        if (dof == 1) {
            datatypes::Vector3 r = world_point - joint_pos_world;
            if (joint.type == JointType::REVOLUTE || joint.type == JointType::CONTINUOUS) {
                // Angular part
                J.block<3,1>(0, q_offset) = to_eigen(joint_axis_world);
                // Linear part: axis cross r
                J.block<3,1>(3, q_offset) = to_eigen(joint_axis_world.cross(r));
            } else if (joint.type == JointType::PRISMATIC) {
                J.block<3,1>(0, q_offset) = Eigen::Vector3d::Zero();
                J.block<3,1>(3, q_offset) = to_eigen(joint_axis_world);
            }
        }
        // For multi-dof joints, we would add columns for each component
        
        current_link = joint.parent_idx;
    }
    return J;
}

bool ArticulatedSystem::inverse_kinematics(int ee_link_idx,
                                           const datatypes::Transformr& target_pose,
                                           Eigen::VectorXd& q,
                                           const datatypes::Vector3& local_point,
                                           int max_iterations,
                                           double tolerance) {
    if (q.size() != static_cast<int>(dof_indices_.size())) {
        q = Eigen::VectorXd::Zero(dof_indices_.size());
    }
    
    forward_kinematics(q);
    
    double lambda = 0.01; // damping
    for (int iter = 0; iter < max_iterations; ++iter) {
        datatypes::Transformr current_pose = link_world_poses_[ee_link_idx];
        datatypes::Vector3 current_point = current_pose.transformPoint(local_point);
        
        // Error
        Eigen::Vector3d pos_error = to_eigen(target_pose.translation - current_point);
        Eigen::Quaterniond quat_error(target_pose.rotation.w, target_pose.rotation.x,
                                       target_pose.rotation.y, target_pose.rotation.z);
        Eigen::Quaterniond current_quat(current_pose.rotation.w, current_pose.rotation.x,
                                         current_pose.rotation.y, current_pose.rotation.z);
        Eigen::Quaterniond delta_q = quat_error * current_quat.conjugate();
        Eigen::AngleAxisd angle_axis(delta_q);
        Eigen::Vector3d ori_error = angle_axis.angle() * angle_axis.axis();
        
        Eigen::Matrix<double, 6, 1> error;
        error << ori_error, pos_error;
        
        if (error.norm() < tolerance) return true;
        
        // Jacobian
        Eigen::MatrixXd J = compute_jacobian(ee_link_idx, local_point);
        
        // Damped least squares
        Eigen::MatrixXd JJt = J * J.transpose();
        JJt.diagonal().array() += lambda * lambda;
        Eigen::Matrix<double, 6, 1> delta_theta = J.transpose() * JJt.ldlt().solve(error);
        
        // Update q
        q += delta_theta;
        
        // Enforce joint limits (optional)
        for (int i = 0; i < static_cast<int>(joints_.size()); ++i) {
            if (joint_dof_count(joints_[i].type) == 1) {
                int offset = joint_q_offset_[i];
                if (joints_[i].limits.has_limits) {
                    q[offset] = std::clamp(q[offset], joints_[i].limits.lower, joints_[i].limits.upper);
                }
            }
        }
        
        forward_kinematics(q);
        
        // Adaptive damping
        lambda = std::max(lambda * 0.5, 1e-6);
    }
    return false;
}

datatypes::Transformr ArticulatedSystem::link_transform(int link_idx) const {
    if (link_idx >= 0 && link_idx < static_cast<int>(link_world_poses_.size()))
        return link_world_poses_[link_idx];
    return datatypes::Transformr();
}

int ArticulatedSystem::joint_dof_count(JointType type) {
    switch (type) {
        case JointType::REVOLUTE:
        case JointType::PRISMATIC:
        case JointType::CONTINUOUS:
            return 1;
        case JointType::SPHERICAL:
            return 3;
        case JointType::UNIVERSAL:
            return 2;
        case JointType::PLANAR:
            return 3;
        case JointType::FLOATING:
            return 6;
        case JointType::FIXED:
        default:
            return 0;
    }
}

void ArticulatedSystem::joint_transform(const Joint& joint, double q, datatypes::Transformr& T) {
    T = datatypes::Transformr();
    switch (joint.type) {
        case JointType::REVOLUTE:
        case JointType::CONTINUOUS:
            T.rotation = datatypes::Quat(joint.axis, q);
            break;
        case JointType::PRISMATIC:
            T.translation = joint.axis * q;
            break;
        case JointType::FIXED:
            break;
        default:
            break;
    }
}

//------------------------------------------------------------------------------
// KinematicSolver implementation
//------------------------------------------------------------------------------
KinematicSolver::KinematicSolver(const SolverConfig& config)
    : BaseSolver("KinematicSolver") {
    config_ = config;
}

KinematicSolver::~KinematicSolver() = default;

void KinematicSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    // No automatic articulation detection; user must add systems manually.
    initialized_ = true;
}

void KinematicSolver::reset() {
    BaseSolver::reset();
    system_states_.clear();
}

void KinematicSolver::step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    if (!initialized_) return;
    
    for (auto& sys : articulated_systems_) {
        SystemState& state = system_states_[sys.get()];
        if (state.needs_update) {
            // Update state from joint sensors / current poses
            // For now, we just integrate using dynamics
        }
        
        // Compute forward dynamics or inverse dynamics based on control mode
        update_system_state(*sys, dt);
        solve_joint_constraints(*sys, dt);
        apply_joint_limits(*sys, state);
        integrate_articulated_bodies(*sys, state, dt);
        
        // Update forward kinematics with new q
        sys->forward_kinematics(state.q);
    }
}

void KinematicSolver::on_entity_added(std::shared_ptr<BaseEntity> entity) {
    // Could check if entity belongs to an existing system
}

void KinematicSolver::on_entity_removed(std::shared_ptr<BaseEntity> entity) {
    // Remove from any system if needed
}

std::shared_ptr<ArticulatedSystem> KinematicSolver::create_articulated_system(const std::string& name) {
    auto sys = std::make_shared<ArticulatedSystem>();
    articulated_systems_.push_back(sys);
    systems_by_name_[name] = sys;
    system_states_[sys.get()] = SystemState{};
    return sys;
}

void KinematicSolver::add_articulated_system(std::shared_ptr<ArticulatedSystem> system) {
    if (system) {
        articulated_systems_.push_back(system);
        system_states_[system.get()] = SystemState{};
    }
}

std::shared_ptr<ArticulatedSystem> KinematicSolver::get_articulated_system(const std::string& name) {
    auto it = systems_by_name_.find(name);
    if (it != systems_by_name_.end()) return it->second;
    return nullptr;
}

void KinematicSolver::forward_kinematics(const std::string& system_name, const Eigen::VectorXd& q) {
    auto sys = get_articulated_system(system_name);
    if (sys) {
        sys->forward_kinematics(q);
        SystemState& state = system_states_[sys.get()];
        state.q = q;
    }
}

bool KinematicSolver::inverse_kinematics(const std::string& system_name,
                                         int ee_link_idx,
                                         const datatypes::Transformr& target_pose,
                                         Eigen::VectorXd& q,
                                         const datatypes::Vector3& local_point) {
    auto sys = get_articulated_system(system_name);
    if (!sys) return false;
    bool ret = sys->inverse_kinematics(ee_link_idx, target_pose, q, local_point,
                                       kin_config_.ik_max_iterations,
                                       kin_config_.ik_tolerance);
    if (ret) {
        SystemState& state = system_states_[sys.get()];
        state.q = q;
    }
    return ret;
}

void KinematicSolver::set_joint_state(const std::string& system_name, int joint_idx, double pos, double vel) {
    auto sys = get_articulated_system(system_name);
    if (!sys) return;
    SystemState& state = system_states_[sys.get()];
    int offset = sys->joint_q_offset_[joint_idx];
    state.q[offset] = pos;
    state.qdot[offset] = vel;
    state.needs_update = true;
}

void KinematicSolver::set_joint_states(const std::string& system_name, const Eigen::VectorXd& q, const Eigen::VectorXd& qdot) {
    auto sys = get_articulated_system(system_name);
    if (!sys) return;
    SystemState& state = system_states_[sys.get()];
    state.q = q;
    state.qdot = qdot;
    state.needs_update = true;
}

Eigen::MatrixXd KinematicSolver::compute_mass_matrix(const ArticulatedSystem& system) {
    Eigen::MatrixXd M(system.num_dofs(), system.num_dofs());
    crba(system, M);
    return M;
}

Eigen::VectorXd KinematicSolver::inverse_dynamics(const ArticulatedSystem& system,
                                                  const Eigen::VectorXd& q,
                                                  const Eigen::VectorXd& qdot,
                                                  const Eigen::VectorXd& qddot,
                                                  const std::vector<datatypes::Vector3>& external_forces) {
    // Update temporary state
    SystemState temp_state;
    temp_state.q = q;
    temp_state.qdot = qdot;
    temp_state.qddot = qddot;
    
    const auto& links = system.links();
    const auto& joints = system.joints();
    int n = static_cast<int>(links.size());
    
    std::vector<datatypes::Vector3> v(n, datatypes::Vector3(0));
    std::vector<datatypes::Vector3> a(n, datatypes::Vector3(0));
    std::vector<datatypes::Vector3> w(n, datatypes::Vector3(0));
    std::vector<datatypes::Vector3> alpha(n, datatypes::Vector3(0));
    std::vector<datatypes::Vector3> f(n, datatypes::Vector3(0));
    std::vector<datatypes::Vector3> tau(n, datatypes::Vector3(0));
    
    rnea_pass1(system, temp_state, v, a, w, alpha);
    Eigen::VectorXd tau_vec(system.num_dofs());
    rnea_pass2(system, temp_state, v, a, w, alpha, f, tau_vec);
    return tau_vec;
}

void KinematicSolver::update_system_state(ArticulatedSystem& system, double dt) {
    SystemState& state = system_states_[&system];
    int n_dofs = static_cast<int>(system.num_dofs());
    if (state.q.size() != n_dofs) {
        state.q = Eigen::VectorXd::Zero(n_dofs);
        state.qdot = Eigen::VectorXd::Zero(n_dofs);
        state.qddot = Eigen::VectorXd::Zero(n_dofs);
        state.tau = Eigen::VectorXd::Zero(n_dofs);
        state.tau_ext = Eigen::VectorXd::Zero(n_dofs);
    }
}

void KinematicSolver::solve_joint_constraints(ArticulatedSystem& system, double dt) {
    SystemState& state = system_states_[&system];
    int n_dofs = static_cast<int>(system.num_dofs());
    
    // If there are external forces/torques, compute inverse dynamics for feedforward
    if (state.tau_ext.norm() > 0) {
        Eigen::VectorXd tau_id = inverse_dynamics(system, state.q, state.qdot, state.qddot);
        state.tau = state.tau_ext - tau_id;
    }
}

void KinematicSolver::apply_joint_limits(ArticulatedSystem& system, SystemState& state) {
    if (!kin_config_.enforce_joint_limits) return;
    
    const auto& joints = system.joints();
    for (size_t j = 0; j < joints.size(); ++j) {
        if (!joints[j].limits.has_limits) continue;
        int offset = system.joint_q_offset_[j];
        if (joint_dof_count(joints[j].type) == 1) {
            double q = state.q[offset];
            double lower = joints[j].limits.lower;
            double upper = joints[j].limits.upper;
            if (q < lower) {
                state.q[offset] = lower;
                state.qdot[offset] = 0;
                // Could apply penalty force
            } else if (q > upper) {
                state.q[offset] = upper;
                state.qdot[offset] = 0;
            }
        }
    }
}

void KinematicSolver::integrate_articulated_bodies(ArticulatedSystem& system, SystemState& state, double dt) {
    // Simple Euler integration
    state.qdot += state.qddot * dt;
    state.q += state.qdot * dt;
    
    // Apply velocity limits
    const auto& joints = system.joints();
    for (size_t j = 0; j < joints.size(); ++j) {
        int offset = system.joint_q_offset_[j];
        double max_vel = joints[j].limits.max_velocity;
        if (std::isfinite(max_vel)) {
            state.qdot[offset] = std::clamp(state.qdot[offset], -max_vel, max_vel);
        }
    }
}

void KinematicSolver::rnea_pass1(const ArticulatedSystem& system,
                                 const SystemState& state,
                                 std::vector<datatypes::Vector3>& v,
                                 std::vector<datatypes::Vector3>& a,
                                 std::vector<datatypes::Vector3>& w,
                                 std::vector<datatypes::Vector3>& alpha) {
    const auto& links = system.links();
    const auto& joints = system.joints();
    datatypes::Vector3 gravity(0, 0, -9.80665);
    
    for (size_t i = 0; i < links.size(); ++i) {
        if (links[i].parent_joint == -1) {
            // Base link - assume stationary or given velocity
            v[i] = datatypes::Vector3(0);
            w[i] = datatypes::Vector3(0);
            a[i] = -gravity;
            alpha[i] = datatypes::Vector3(0);
        }
    }
    
    // Forward pass
    for (size_t j = 0; j < joints.size(); ++j) {
        const Joint& joint = joints[j];
        int parent = joint.parent_idx;
        int child = joint.child_idx;
        if (parent < 0 || child < 0) continue;
        
        int dof = joint_dof_count(joint.type);
        int offset = system.joint_q_offset_[j];
        double qdot = (dof == 1) ? state.qdot[offset] : 0;
        double qddot = (dof == 1) ? state.qddot[offset] : 0;
        
        datatypes::Vector3 axis_world = links[parent].entity->transform().rotation.rotate(joint.axis);
        
        if (joint.type == JointType::REVOLUTE || joint.type == JointType::CONTINUOUS) {
            w[child] = w[parent] + axis_world * qdot;
            alpha[child] = alpha[parent] + axis_world * qddot + w[parent].cross(axis_world * qdot);
            datatypes::Vector3 r = links[child].entity->transform().translation - links[parent].entity->transform().translation;
            v[child] = v[parent] + w[child].cross(r);
            a[child] = a[parent] + alpha[child].cross(r) + w[child].cross(w[child].cross(r));
        } else if (joint.type == JointType::PRISMATIC) {
            w[child] = w[parent];
            alpha[child] = alpha[parent];
            v[child] = v[parent] + axis_world * qdot + w[parent].cross(links[child].entity->transform().translation - links[parent].entity->transform().translation);
            a[child] = a[parent] + axis_world * qddot + alpha[parent].cross(links[child].entity->transform().translation - links[parent].entity->transform().translation)
                       + w[parent].cross(w[parent].cross(links[child].entity->transform().translation - links[parent].entity->transform().translation))
                       + 2 * w[parent].cross(axis_world * qdot);
        }
    }
}

void KinematicSolver::rnea_pass2(const ArticulatedSystem& system,
                                 const SystemState& state,
                                 const std::vector<datatypes::Vector3>& v,
                                 const std::vector<datatypes::Vector3>& a,
                                 const std::vector<datatypes::Vector3>& w,
                                 const std::vector<datatypes::Vector3>& alpha,
                                 std::vector<datatypes::Vector3>& f,
                                 Eigen::VectorXd& tau) {
    const auto& links = system.links();
    const auto& joints = system.joints();
    int n = static_cast<int>(links.size());
    
    f.assign(n, datatypes::Vector3(0));
    tau = Eigen::VectorXd::Zero(system.num_dofs());
    
    // Backward pass
    for (int i = n - 1; i >= 0; --i) {
        // Compute force on link i: f_i = I_i * a_i + v_i x* I_i v_i
        // For simplicity, use mass * acceleration
        double mass = links[i].entity ? links[i].entity->mass() : 1.0;
        f[i] = a[i] * mass;
        
        int joint_idx = links[i].parent_joint;
        if (joint_idx >= 0) {
            const Joint& joint = joints[joint_idx];
            int dof = joint_dof_count(joint.type);
            int offset = system.joint_q_offset_[joint_idx];
            datatypes::Vector3 axis_world = links[joint.parent_idx].entity->transform().rotation.rotate(joint.axis);
            if (joint.type == JointType::REVOLUTE || joint.type == JointType::CONTINUOUS) {
                tau[offset] = f[i].dot(axis_world);
            } else if (joint.type == JointType::PRISMATIC) {
                tau[offset] = f[i].dot(axis_world);
            }
            // Add to parent force (simplified)
            f[joint.parent_idx] += f[i];
        }
    }
}

void KinematicSolver::crba(const ArticulatedSystem& system, Eigen::MatrixXd& M) {
    int n = static_cast<int>(system.num_dofs());
    M = Eigen::MatrixXd::Zero(n, n);
    // Composite rigid body algorithm - set each qddot unit vector and compute inverse dynamics
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd qddot = Eigen::VectorXd::Unit(n, i);
        Eigen::VectorXd tau = inverse_dynamics(system, q, qdot, qddot);
        M.col(i) = tau;
    }
}

} // namespace engine
} // namespace genesis