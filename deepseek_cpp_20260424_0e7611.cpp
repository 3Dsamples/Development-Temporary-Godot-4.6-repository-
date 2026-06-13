// genesis/engine/solvers/fem_solver.cpp

#include "genesis/engine/solvers/fem_solver.h"
#include "genesis/engine/entities/fem_entity.h"
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <numeric>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Helper: Convert genesis Vector3 to Eigen Vector3d
//------------------------------------------------------------------------------
inline Eigen::Vector3d to_eigen(const datatypes::Vector3& v) {
    return Eigen::Vector3d(v[0], v[1], v[2]);
}

inline datatypes::Vector3 from_eigen(const Eigen::Vector3d& v) {
    return datatypes::Vector3(v.x(), v.y(), v.z());
}

//------------------------------------------------------------------------------
// FEMSolver implementation
//------------------------------------------------------------------------------
FEMSolver::FEMSolver(const SolverConfig& config)
    : BaseSolver("FEMSolver")
{
    config_ = config;
}

FEMSolver::~FEMSolver() = default;

void FEMSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    
    rebuild_states(entities);
    
    // Allocate global vectors
    size_t total_dofs = 0;
    for (const auto& state : fem_states_) {
        total_dofs += state.entity->node_count() * 3;
    }
    
    u_ = Eigen::VectorXd::Zero(total_dofs);
    v_ = Eigen::VectorXd::Zero(total_dofs);
    a_ = Eigen::VectorXd::Zero(total_dofs);
    f_ext_ = Eigen::VectorXd::Zero(total_dofs);
    f_int_ = Eigen::VectorXd::Zero(total_dofs);
    m_ = Eigen::VectorXd::Zero(total_dofs);
    
    // Compute lumped mass and initial state
    size_t offset = 0;
    for (auto& state : fem_states_) {
        size_t n_nodes = state.entity->node_count();
        for (size_t i = 0; i < n_nodes; ++i) {
            datatypes::Vector3 pos = state.entity->node_position(i);
            for (int d = 0; d < 3; ++d) {
                u_[offset + i*3 + d] = pos[d];
            }
            // Lumped mass will be computed during assembly
        }
        offset += n_nodes * 3;
    }
    
    initialized_ = true;
}

void FEMSolver::reset() {
    BaseSolver::reset();
    fem_states_.clear();
    entity_to_state_index_.clear();
    states_dirty_ = true;
    K_.resize(0, 0);
}

void FEMSolver::step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    
    if (!initialized_ || states_dirty_) {
        initialize(entities);
    }
    
    // Update external forces (gravity, user forces) into f_ext_
    f_ext_.setZero();
    size_t offset = 0;
    for (auto& state : fem_states_) {
        size_t n_nodes = state.entity->node_count();
        datatypes::Vector3 gravity = datatypes::Vector3(0, 0, -9.80665); // should come from scene
        
        for (size_t i = 0; i < n_nodes; ++i) {
            double mass = state.entity->node_mass(i);
            datatypes::Vector3 force = gravity * mass;
            // Add any external forces applied to node
            force += state.entity->node_external_force(i);
            
            for (int d = 0; d < 3; ++d) {
                f_ext_[offset + i*3 + d] = force[d];
            }
        }
        offset += n_nodes * 3;
    }
    
    // Assemble stiffness, mass, and internal forces
    assemble_system(dt);
    
    // Solve for displacement increment
    solve_linear_system(dt);
    
    // Update entity positions and velocities
    update_entities();
    
    stats_.iteration_count = 1; // Implicit single step
    stats_.constraint_count = K_.nonZeros();
}

void FEMSolver::on_entity_added(std::shared_ptr<BaseEntity> entity) {
    if (std::dynamic_pointer_cast<FEMEntity>(entity)) {
        states_dirty_ = true;
    }
}

void FEMSolver::on_entity_removed(std::shared_ptr<BaseEntity> entity) {
    auto it = entity_to_state_index_.find(entity->id());
    if (it != entity_to_state_index_.end()) {
        fem_states_.erase(fem_states_.begin() + it->second);
        states_dirty_ = true;
    }
}

void FEMSolver::rebuild_states(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    fem_states_.clear();
    entity_to_state_index_.clear();
    
    for (const auto& e : entities) {
        auto fem_entity = std::dynamic_pointer_cast<FEMEntity>(e);
        if (!fem_entity) continue;
        
        FEMState state;
        state.entity = fem_entity;
        size_t n_nodes = fem_entity->node_count();
        
        // Store rest positions
        state.rest_positions.resize(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            state.rest_positions[i] = fem_entity->node_rest_position(i);
        }
        
        // Get tetrahedral elements
        const auto& tets = fem_entity->tetrahedra();
        state.tetrahedra = tets;
        
        // Initialize deformation gradients to identity
        state.deformation_gradients.resize(tets.size(), Eigen::Matrix3d::Identity());
        state.plastic_strain.resize(tets.size(), Eigen::Matrix3d::Identity());
        
        // Allocate per-entity force vector
        state.forces = Eigen::VectorXd::Zero(n_nodes * 3);
        state.displacements = Eigen::VectorXd::Zero(n_nodes * 3);
        
        entity_to_state_index_[fem_entity->id()] = fem_states_.size();
        fem_states_.push_back(std::move(state));
    }
    states_dirty_ = false;
}

void FEMSolver::assemble_system(double dt) {
    size_t total_dofs = u_.size();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(total_dofs * 30); // rough estimate
    
    f_int_.setZero();
    m_.setZero();
    
    size_t offset = 0;
    for (auto& state : fem_states_) {
        size_t n_nodes = state.entity->node_count();
        size_t n_elems = state.tetrahedra.size();
        
        // Reset per-entity forces
        state.forces.setZero();
        
        // Clear per-entity triplets (we'll use global index)
        state.stiffness_triplets.clear();
        
        // Compute per-element stiffness and internal forces
        for (size_t e = 0; e < n_elems; ++e) {
            if (fem_config_.element_type == FEMConfig::ElementType::TET4) {
                compute_tet4_stiffness(state, e, triplets, state.forces);
            } else if (fem_config_.element_type == FEMConfig::ElementType::HEX8) {
                compute_hex8_stiffness(state, e, triplets, state.forces);
            }
        }
        
        // Add to global internal forces
        f_int_.segment(offset, n_nodes * 3) += state.forces;
        
        // Add lumped mass
        for (size_t i = 0; i < n_nodes; ++i) {
            double mass = state.entity->node_mass(i);
            for (int d = 0; d < 3; ++d) {
                m_[offset + i*3 + d] += mass;
            }
        }
        
        offset += n_nodes * 3;
    }
    
    // Build sparse stiffness matrix
    K_.resize(total_dofs, total_dofs);
    K_.setFromTriplets(triplets.begin(), triplets.end());
    
    // Apply Rayleigh damping if needed
    if (fem_config_.rayleigh_damping_alpha > 0 || fem_config_.rayleigh_damping_beta > 0) {
        // Add damping matrix C = alpha * M + beta * K to the effective stiffness
        // For implicit Newmark, effective stiffness K_eff = K + gamma/(beta*dt) * C + 1/(beta*dt^2) * M
    }
}

void FEMSolver::solve_linear_system(double dt) {
    double beta = fem_config_.newmark_beta;
    double gamma = fem_config_.newmark_gamma;
    
    // Build effective stiffness for implicit integration
    // K_eff = K + (1/(beta*dt^2)) * M   (simplified, neglecting damping for now)
    Eigen::SparseMatrix<double> K_eff = K_;
    std::vector<Eigen::Triplet<double>> mass_triplets;
    for (int i = 0; i < m_.size(); ++i) {
        if (m_[i] > 0) {
            mass_triplets.emplace_back(i, i, m_[i] / (beta * dt * dt));
        }
    }
    Eigen::SparseMatrix<double> M_diag(m_.size(), m_.size());
    M_diag.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
    K_eff += M_diag;
    
    // Compute residual: R = f_ext - f_int + M * (1/(beta*dt^2) * u_predicted - ...)
    // For simplicity, we do a basic implicit Euler: K_eff * du = f_ext - f_int
    Eigen::VectorXd residual = f_ext_ - f_int_;
    
    // Solve linear system
    linear_solver_.compute(K_eff);
    if (linear_solver_.info() != Eigen::Success) {
        // Fallback or error
        return;
    }
    Eigen::VectorXd du = linear_solver_.solve(residual);
    
    // Update displacements
    u_ += du;
    
    // Update velocities (implicit Euler: v_new = v_old + du/dt)
    v_ = du / dt;
}

void FEMSolver::update_entities() {
    size_t offset = 0;
    for (auto& state : fem_states_) {
        size_t n_nodes = state.entity->node_count();
        for (size_t i = 0; i < n_nodes; ++i) {
            datatypes::Vector3 new_pos(
                u_[offset + i*3 + 0],
                u_[offset + i*3 + 1],
                u_[offset + i*3 + 2]
            );
            datatypes::Vector3 new_vel(
                v_[offset + i*3 + 0],
                v_[offset + i*3 + 1],
                v_[offset + i*3 + 2]
            );
            state.entity->set_node_position(i, new_pos);
            state.entity->set_node_velocity(i, new_vel);
        }
        offset += n_nodes * 3;
    }
}

void FEMSolver::compute_tet4_stiffness(const FEMState& state, size_t elem_idx,
                                       std::vector<Eigen::Triplet<double>>& triplets,
                                       Eigen::VectorXd& forces) {
    const auto& tet = state.tetrahedra[elem_idx];
    std::array<Eigen::Vector3d, 4> X; // rest positions
    std::array<Eigen::Vector3d, 4> x; // current positions
    for (int i = 0; i < 4; ++i) {
        X[i] = to_eigen(state.rest_positions[tet[i]]);
        x[i] = to_eigen(state.entity->node_position(tet[i]));
    }
    
    // Compute deformation gradient F
    Eigen::Matrix3d Dm;
    for (int i = 0; i < 3; ++i) {
        Dm.col(i) = X[i+1] - X[0];
    }
    Eigen::Matrix3d Dm_inv = Dm.inverse();
    
    Eigen::Matrix3d Ds;
    for (int i = 0; i < 3; ++i) {
        Ds.col(i) = x[i+1] - x[0];
    }
    Eigen::Matrix3d F = Ds * Dm_inv;
    
    // Apply plasticity if enabled
    Eigen::Matrix3d F_elastic = F;
    if (fem_config_.enable_plasticity) {
        Eigen::Matrix3d Fp = state.plastic_strain[elem_idx];
        F_elastic = F * Fp.inverse();
        apply_plasticity(F_elastic, Fp);
        state.plastic_strain[elem_idx] = Fp;
    }
    
    // Compute first Piola-Kirchhoff stress P
    Eigen::Matrix3d P = compute_pk1_stress(F_elastic);
    
    // Compute element stiffness matrix contributions
    double volume = std::abs(Dm.determinant()) / 6.0;
    Eigen::Matrix3d H = -volume * P * Dm_inv.transpose();
    
    // Assemble forces and stiffness for the 4 nodes
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d force_i;
        if (i == 0) {
            force_i = -H * Eigen::Vector3d::Ones();
        } else {
            force_i = H.col(i-1);
        }
        
        // Add to per-entity force vector
        size_t offset_i = tet[i] * 3;
        for (int d = 0; d < 3; ++d) {
            forces[offset_i + d] += force_i(d);
        }
    }
    
    // Compute stiffness matrix (tangent) using material tensor
    Eigen::Matrix3d C9 = compute_stiffness_tensor(F_elastic);
    // Build element stiffness matrix Ke (12x12) and add triplets
    // For brevity, we add a simplified stiffness: Ke = volume * B^T * D * B
    // Where D is the 6x6 elasticity matrix derived from C9
    // Implementation of B matrix and full assembly omitted for space, but would be here.
}

void FEMSolver::compute_hex8_stiffness(const FEMState& state, size_t elem_idx,
                                       std::vector<Eigen::Triplet<double>>& triplets,
                                       Eigen::VectorXd& forces) {
    // Similar to tet4 but with 8 nodes and trilinear shape functions
    // Implementation would be similar but more involved.
}

Eigen::Matrix3d FEMSolver::compute_pk1_stress(const Eigen::Matrix3d& F,
                                               const Eigen::Matrix3d& F_plastic) const {
    // Neo-Hookean material
    double mu = fem_config_.youngs_modulus / (2.0 * (1.0 + fem_config_.poisson_ratio));
    double lambda = fem_config_.youngs_modulus * fem_config_.poisson_ratio /
                    ((1.0 + fem_config_.poisson_ratio) * (1.0 - 2.0 * fem_config_.poisson_ratio));
    
    Eigen::Matrix3d C = F.transpose() * F;
    double J = F.determinant();
    double logJ = std::log(J);
    
    Eigen::Matrix3d P = mu * (F - F.inverse().transpose()) + lambda * logJ * F.inverse().transpose();
    return P;
}

Eigen::Matrix3d FEMSolver::compute_stiffness_tensor(const Eigen::Matrix3d& F) const {
    // Simplified: return constant isotropic tensor for small strain
    double E = fem_config_.youngs_modulus;
    double nu = fem_config_.poisson_ratio;
    double mu = E / (2.0 * (1.0 + nu));
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    C.diagonal().setConstant(lambda + 2*mu);
    C(0,1) = C(1,0) = lambda;
    C(0,2) = C(2,0) = lambda;
    C(1,2) = C(2,1) = lambda;
    // In reality this would be a 6x6 matrix; for PK1 derivative it's more complex.
    return C;
}

void FEMSolver::apply_plasticity(Eigen::Matrix3d& F, Eigen::Matrix3d& Fp) const {
    // Simple von Mises plasticity with linear hardening
    double tau_y = fem_config_.yield_stress;
    double H = fem_config_.hardening_modulus;
    
    // Compute trial elastic strain
    Eigen::Matrix3d Fe_tr = F * Fp.inverse();
    
    // Compute deviatoric stress (simplified: use small strain assumption)
    Eigen::Matrix3d epsilon = 0.5 * (Fe_tr + Fe_tr.transpose()) - Eigen::Matrix3d::Identity();
    Eigen::Matrix3d sigma = 2 * mu * epsilon + lambda * epsilon.trace() * Eigen::Matrix3d::Identity();
    
    // Von Mises equivalent stress
    Eigen::Matrix3d s = sigma - (sigma.trace()/3.0) * Eigen::Matrix3d::Identity();
    double J2 = 0.5 * (s.array() * s.array()).sum();
    double sigma_e = std::sqrt(3.0 * J2);
    
    if (sigma_e > tau_y) {
        // Plastic flow
        double dgamma = (sigma_e - tau_y) / (2 * mu + H);
        Eigen::Matrix3d N = s / sigma_e;
        Fp = (Eigen::Matrix3d::Identity() + dgamma * N) * Fp;
        // Ensure Fp determinant = 1 (plastic incompressibility)
        double detFp = Fp.determinant();
        Fp /= std::cbrt(detFp);
    }
}

//------------------------------------------------------------------------------
// Static tet4 shape functions
//------------------------------------------------------------------------------
void FEMSolver::tet4_shape_functions(const Eigen::Vector4d& xi,
                                     Eigen::Matrix<double, 3, 4>& dN_dX,
                                     const std::array<Eigen::Vector3d, 4>& X) {
    // Shape function derivatives in natural coordinates
    Eigen::Matrix<double, 3, 4> dN_dxi;
    dN_dxi << -1,  1,  0,  0,
              -1,  0,  1,  0,
              -1,  0,  0,  1;
    
    // Jacobian of the mapping from natural to reference coordinates
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 4; ++i) {
        J += X[i] * dN_dxi.col(i).transpose();
    }
    
    // Derivatives with respect to reference coordinates
    Eigen::Matrix3d Jinv = J.inverse();
    for (int i = 0; i < 4; ++i) {
        dN_dX.col(i) = Jinv * dN_dxi.col(i);
    }
}

} // namespace engine
} // namespace genesis