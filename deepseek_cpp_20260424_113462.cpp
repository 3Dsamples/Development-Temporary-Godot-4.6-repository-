// genesis/engine/solvers/pbd_solver.cpp

#include "genesis/engine/solvers/pbd_solver.h"
#include "genesis/engine/entities/pbd_entity.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// DistanceConstraint implementation
//------------------------------------------------------------------------------
void DistanceConstraint::project(const std::vector<datatypes::Vector3>& positions,
                                 std::vector<datatypes::Vector3>& delta,
                                 const std::vector<double>& inv_mass,
                                 double dt) {
    if (!enabled) return;
    
    datatypes::Vector3 p0 = positions[idx0] + delta[idx0];
    datatypes::Vector3 p1 = positions[idx1] + delta[idx1];
    datatypes::Vector3 dir = p1 - p0;
    double dist = dir.norm();
    if (dist < 1e-12) return;
    dir /= dist;
    
    double w0 = inv_mass[idx0];
    double w1 = inv_mass[idx1];
    double w_sum = w0 + w1;
    if (w_sum < 1e-12) return;
    
    double correction = (dist - rest_length) * stiffness / w_sum;
    
    delta[idx0] += dir * (correction * w0);
    delta[idx1] -= dir * (correction * w1);
}

//------------------------------------------------------------------------------
// BendingConstraint implementation
//------------------------------------------------------------------------------
void BendingConstraint::project(const std::vector<datatypes::Vector3>& positions,
                                std::vector<datatypes::Vector3>& delta,
                                const std::vector<double>& inv_mass,
                                double dt) {
    if (!enabled) return;
    
    datatypes::Vector3 p0 = positions[idx0] + delta[idx0];
    datatypes::Vector3 p1 = positions[idx1] + delta[idx1];
    datatypes::Vector3 p2 = positions[idx2] + delta[idx2];
    datatypes::Vector3 p3 = positions[idx3] + delta[idx3];
    
    // Compute normals of the two triangles
    datatypes::Vector3 n1 = (p2 - p0).cross(p3 - p0);
    datatypes::Vector3 n2 = (p3 - p1).cross(p2 - p1);
    double len1 = n1.norm();
    double len2 = n2.norm();
    if (len1 < 1e-12 || len2 < 1e-12) return;
    n1 /= len1;
    n2 /= len2;
    
    // Shared edge
    datatypes::Vector3 e = p3 - p2;
    double e_len = e.norm();
    if (e_len < 1e-12) return;
    e /= e_len;
    
    // Current dihedral angle
    double cos_theta = n1.dot(n2);
    double sin_theta = n1.cross(n2).dot(e);
    double theta = std::atan2(sin_theta, cos_theta);
    
    double delta_theta = theta - rest_angle;
    
    // Compute gradient for each vertex
    double w0 = inv_mass[idx0];
    double w1 = inv_mass[idx1];
    double w2 = inv_mass[idx2];
    double w3 = inv_mass[idx3];
    
    // Derivatives from Bridson et al. 2003
    datatypes::Vector3 d0 = n1.cross(p3 - p2) / len1 + n2.cross(p2 - p3) / len2;
    datatypes::Vector3 d1 = n1.cross(p2 - p0) / len1 + n2.cross(p0 - p2) / len2;
    datatypes::Vector3 d2 = n1.cross(p0 - p3) / len1 + n2.cross(p3 - p1) / len2;
    datatypes::Vector3 d3 = n1.cross(p1 - p2) / len1 + n2.cross(p1 - p0) / len2;
    
    double sum_w = w0 * d0.squaredNorm() + w1 * d1.squaredNorm() + 
                   w2 * d2.squaredNorm() + w3 * d3.squaredNorm();
    if (sum_w < 1e-12) return;
    
    double s = -delta_theta * stiffness / sum_w;
    
    delta[idx0] += d0 * (s * w0);
    delta[idx1] += d1 * (s * w1);
    delta[idx2] += d2 * (s * w2);
    delta[idx3] += d3 * (s * w3);
}

//------------------------------------------------------------------------------
// VolumeConstraint implementation
//------------------------------------------------------------------------------
void VolumeConstraint::project(const std::vector<datatypes::Vector3>& positions,
                               std::vector<datatypes::Vector3>& delta,
                               const std::vector<double>& inv_mass,
                               double dt) {
    if (!enabled) return;
    
    datatypes::Vector3 p0 = positions[idx0] + delta[idx0];
    datatypes::Vector3 p1 = positions[idx1] + delta[idx1];
    datatypes::Vector3 p2 = positions[idx2] + delta[idx2];
    datatypes::Vector3 p3 = positions[idx3] + delta[idx3];
    
    // Current volume (signed)
    double volume = (p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0;
    
    // Compute gradients (area vectors of opposite faces)
    datatypes::Vector3 grad0 = (p2 - p1).cross(p3 - p1) / 6.0;
    datatypes::Vector3 grad1 = (p2 - p0).cross(p3 - p0) / 6.0;
    datatypes::Vector3 grad2 = (p1 - p0).cross(p3 - p0) / 6.0;
    datatypes::Vector3 grad3 = (p1 - p0).cross(p2 - p0) / 6.0;
    
    double w0 = inv_mass[idx0];
    double w1 = inv_mass[idx1];
    double w2 = inv_mass[idx2];
    double w3 = inv_mass[idx3];
    double sum_w = w0 * grad0.squaredNorm() + w1 * grad1.squaredNorm() + 
                   w2 * grad2.squaredNorm() + w3 * grad3.squaredNorm();
    if (sum_w < 1e-12) return;
    
    double s = (rest_volume - volume) * stiffness / sum_w;
    
    delta[idx0] += grad0 * (s * w0);
    delta[idx1] -= grad1 * (s * w1); // Note sign differences for each vertex
    delta[idx2] += grad2 * (s * w2);
    delta[idx3] -= grad3 * (s * w3);
}

//------------------------------------------------------------------------------
// ShapeMatchingConstraint implementation
//------------------------------------------------------------------------------
ShapeMatchingConstraint::ShapeMatchingConstraint(const std::vector<int>& indices,
                                                 const std::vector<datatypes::Vector3>& rest,
                                                 double stiffness)
    : PBDConstraint(PBDConstraintType::SHAPE_MATCHING, stiffness)
    , particle_indices(indices)
    , rest_positions(rest) {
    // Compute center of mass
    rest_com = datatypes::Vector3(0);
    for (const auto& p : rest_positions) rest_com += p;
    rest_com /= static_cast<double>(rest_positions.size());
    
    // Compute relative positions
    rest_relative.resize(rest_positions.size());
    for (size_t i = 0; i < rest_positions.size(); ++i) {
        rest_relative[i] = rest_positions[i] - rest_com;
    }
}

void ShapeMatchingConstraint::project(const std::vector<datatypes::Vector3>& positions,
                                      std::vector<datatypes::Vector3>& delta,
                                      const std::vector<double>& inv_mass,
                                      double dt) {
    if (!enabled) return;
    
    size_t n = particle_indices.size();
    
    // Compute current center of mass
    datatypes::Vector3 cur_com(0);
    double total_mass = 0;
    for (size_t i = 0; i < n; ++i) {
        int idx = particle_indices[i];
        double m = 1.0 / (inv_mass[idx] + 1e-12);
        cur_com += (positions[idx] + delta[idx]) * m;
        total_mass += m;
    }
    if (total_mass < 1e-12) return;
    cur_com /= total_mass;
    
    // Compute optimal rotation (Kabsch algorithm via SVD)
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < n; ++i) {
        int idx = particle_indices[i];
        Eigen::Vector3d p = Eigen::Vector3d((positions[idx] + delta[idx] - cur_com)[0],
                                            (positions[idx] + delta[idx] - cur_com)[1],
                                            (positions[idx] + delta[idx] - cur_com)[2]);
        Eigen::Vector3d q = Eigen::Vector3d(rest_relative[i][0], rest_relative[i][1], rest_relative[i][2]);
        A += p * q.transpose();
    }
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    
    // Check for reflection
    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) = -V.col(2);
        R = svd.matrixU() * V.transpose();
    }
    
    // Compute goal positions and move particles
    for (size_t i = 0; i < n; ++i) {
        int idx = particle_indices[i];
        Eigen::Vector3d goal_eigen = Eigen::Vector3d(cur_com[0], cur_com[1], cur_com[2]) + R * Eigen::Vector3d(rest_relative[i][0], rest_relative[i][1], rest_relative[i][2]);
        datatypes::Vector3 goal(goal_eigen.x(), goal_eigen.y(), goal_eigen.z());
        datatypes::Vector3 current = positions[idx] + delta[idx];
        delta[idx] += (goal - current) * stiffness;
    }
}

//------------------------------------------------------------------------------
// PBDSolver implementation
//------------------------------------------------------------------------------
PBDSolver::PBDSolver(const SolverConfig& config)
    : BaseSolver("PBDSolver") {
    config_ = config;
}

PBDSolver::~PBDSolver() = default;

void PBDSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    rebuild_states(entities);
    initialized_ = true;
}

void PBDSolver::reset() {
    BaseSolver::reset();
    entity_states_.clear();
    entity_to_index_.clear();
    constraints_.clear();
    entity_constraint_indices_.clear();
    states_dirty_ = true;
}

void PBDSolver::step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    
    if (!initialized_ || states_dirty_) {
        initialize(entities);
    }
    
    if (entity_states_.empty()) return;
    
    // Clamp dt
    double sub_dt = dt / pbd_config_.sub_iterations;
    
    for (int substep = 0; substep < pbd_config_.sub_iterations; ++substep) {
        // Step 1: Predict positions using velocities and external forces
        predict_positions(sub_dt);
        
        // Step 2: Project constraints iteratively
        project_constraints(sub_dt);
        
        // Step 3: Handle collisions (simplified)
        apply_collisions(sub_dt);
        
        // Step 4: Update velocities
        update_velocities(sub_dt);
        
        // Step 5: Enforce max velocity
        enforce_max_velocity();
    }
    
    // Write back to entities
    for (auto& state : entity_states_) {
        state.entity->set_positions(state.positions);
        state.entity->set_velocities(state.velocities);
    }
    
    stats_.iteration_count = pbd_config_.iterations * pbd_config_.sub_iterations;
    stats_.constraint_count = constraints_.size();
}

void PBDSolver::on_entity_added(std::shared_ptr<BaseEntity> entity) {
    if (std::dynamic_pointer_cast<PBDEntity>(entity)) {
        states_dirty_ = true;
    }
}

void PBDSolver::on_entity_removed(std::shared_ptr<BaseEntity> entity) {
    auto it = entity_to_index_.find(entity->id());
    if (it != entity_to_index_.end()) {
        entity_states_.erase(entity_states_.begin() + it->second);
        remove_constraints(entity->id());
        states_dirty_ = true;
    }
}

void PBDSolver::add_constraint(std::shared_ptr<PBDConstraint> constraint, uint64_t entity_id) {
    constraints_.push_back(constraint);
    entity_constraint_indices_[entity_id].push_back(constraints_.size() - 1);
}

void PBDSolver::remove_constraints(uint64_t entity_id) {
    auto it = entity_constraint_indices_.find(entity_id);
    if (it == entity_constraint_indices_.end()) return;
    
    // Remove constraints from the global list (inefficient but simple; could use erase-remove)
    std::vector<size_t> to_remove = it->second;
    std::sort(to_remove.begin(), to_remove.end(), std::greater<size_t>());
    for (size_t idx : to_remove) {
        constraints_.erase(constraints_.begin() + idx);
    }
    entity_constraint_indices_.erase(it);
    
    // Rebuild indices mapping (since indices shifted)
    entity_constraint_indices_.clear();
    for (size_t i = 0; i < constraints_.size(); ++i) {
        // We don't have a direct way to get entity_id from constraint; assume per-entity constraints
        // In practice, we'd store that mapping separately.
    }
}

void PBDSolver::clear_constraints() {
    constraints_.clear();
    entity_constraint_indices_.clear();
}

void PBDSolver::rebuild_states(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    entity_states_.clear();
    entity_to_index_.clear();
    
    for (const auto& e : entities) {
        auto pbd_entity = std::dynamic_pointer_cast<PBDEntity>(e);
        if (!pbd_entity) continue;
        
        EntityPBDState state;
        state.entity = pbd_entity;
        size_t n = pbd_entity->particle_count();
        state.positions = pbd_entity->positions();
        state.prev_positions = state.positions;
        state.velocities = pbd_entity->velocities();
        state.inv_mass = pbd_entity->inverse_masses();
        state.delta.resize(n, datatypes::Vector3(0));
        
        entity_to_index_[pbd_entity->id()] = entity_states_.size();
        entity_states_.push_back(std::move(state));
    }
    
    // Build constraints from entity definitions
    constraints_.clear();
    entity_constraint_indices_.clear();
    for (auto& state : entity_states_) {
        const auto& entity_constraints = state.entity->constraints();
        for (const auto& c : entity_constraints) {
            constraints_.push_back(c);
            entity_constraint_indices_[state.entity->id()].push_back(constraints_.size() - 1);
        }
    }
    
    states_dirty_ = false;
}

void PBDSolver::predict_positions(double dt) {
    datatypes::Vector3 gravity(0, 0, -9.80665);
    
    for (auto& state : entity_states_) {
        size_t n = state.positions.size();
        for (size_t i = 0; i < n; ++i) {
            if (state.inv_mass[i] < 1e-12) continue;
            
            // Apply damping
            state.velocities[i] *= (1.0 - pbd_config_.velocity_damping);
            
            // Apply external force (gravity)
            state.velocities[i] += gravity * dt;
            
            // Predict position
            state.prev_positions[i] = state.positions[i];
            state.positions[i] += state.velocities[i] * dt;
        }
    }
}

void PBDSolver::project_constraints(double dt) {
    for (int iter = 0; iter < pbd_config_.iterations; ++iter) {
        // Reset delta
        for (auto& state : entity_states_) {
            std::fill(state.delta.begin(), state.delta.end(), datatypes::Vector3(0));
        }
        
        // Solve all constraints
        for (const auto& constraint : constraints_) {
            if (!constraint->enabled) continue;
            
            // For simplicity, assume all indices belong to a single entity.
            // In a full implementation, we'd map indices to the correct state.
            // Here we just use the first entity state (since indices are per-entity)
            // We need to know which entity this constraint belongs to.
            // We'll use a simple heuristic: find state containing the first index.
            int first_idx = constraint->indices()[0];
            EntityPBDState* target_state = nullptr;
            for (auto& state : entity_states_) {
                if (first_idx < static_cast<int>(state.positions.size())) {
                    target_state = &state;
                    break;
                }
                first_idx -= static_cast<int>(state.positions.size());
            }
            if (!target_state) continue;
            
            // Project constraint using that state's delta
            constraint->project(target_state->positions, target_state->delta,
                                target_state->inv_mass, dt);
        }
        
        // Apply accumulated deltas
        for (auto& state : entity_states_) {
            for (size_t i = 0; i < state.positions.size(); ++i) {
                state.positions[i] += state.delta[i];
            }
        }
    }
}

void PBDSolver::update_velocities(double dt) {
    for (auto& state : entity_states_) {
        for (size_t i = 0; i < state.positions.size(); ++i) {
            state.velocities[i] = (state.positions[i] - state.prev_positions[i]) / dt;
        }
    }
}

void PBDSolver::apply_collisions(double dt) {
    // Simple ground collision with plane at z=0
    double floor_y = 0.0;
    double restitution = 0.3;
    double friction = 0.5;
    
    for (auto& state : entity_states_) {
        for (size_t i = 0; i < state.positions.size(); ++i) {
            if (state.positions[i][1] < floor_y) {
                // Penetration correction
                state.positions[i][1] = floor_y;
                
                // Velocity response (simple)
                if (state.velocities[i][1] < 0) {
                    state.velocities[i][1] = -state.velocities[i][1] * restitution;
                    // Friction
                    state.velocities[i][0] *= (1.0 - friction);
                    state.velocities[i][2] *= (1.0 - friction);
                }
            }
        }
    }
}

void PBDSolver::enforce_max_velocity() {
    double max_vel_sq = pbd_config_.max_velocity * pbd_config_.max_velocity;
    for (auto& state : entity_states_) {
        for (auto& v : state.velocities) {
            double v_sq = v.squaredNorm();
            if (v_sq > max_vel_sq) {
                v *= pbd_config_.max_velocity / std::sqrt(v_sq);
            }
        }
    }
}

} // namespace engine
} // namespace genesis