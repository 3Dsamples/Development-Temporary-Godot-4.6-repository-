// genesis/engine/solvers/mpm_solver.cpp

#include "genesis/engine/solvers/mpm_solver.h"
#include "genesis/engine/entities/mpm_entity.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// B-spline weight functions (Quadratic B-spline)
//------------------------------------------------------------------------------
double MPMSolver::bspline_weight(double x) {
    double ax = std::abs(x);
    if (ax < 0.5) {
        return 0.75 - ax * ax;
    } else if (ax < 1.5) {
        double t = 1.5 - ax;
        return 0.5 * t * t;
    }
    return 0.0;
}

double MPMSolver::bspline_weight_derivative(double x) {
    double ax = std::abs(x);
    double sign = (x >= 0) ? 1.0 : -1.0;
    if (ax < 0.5) {
        return -2.0 * x;
    } else if (ax < 1.5) {
        return sign * (ax - 1.5);
    }
    return 0.0;
}

void MPMSolver::interpolate_weights(const Eigen::Vector3d& xp, const Grid& grid,
                                    std::array<int, 3>& base_idx,
                                    std::array<double, 3>& wx,
                                    std::array<double, 3>& dwx) {
    Eigen::Vector3d cell_coord = (xp - grid.origin) / grid.dx;
    for (int d = 0; d < 3; ++d) {
        base_idx[d] = static_cast<int>(std::floor(cell_coord[d] - 0.5));
        double dx = cell_coord[d] - (base_idx[d] + 0.5);
        wx[d] = bspline_weight(dx);
        dwx[d] = bspline_weight_derivative(dx) / grid.dx;
    }
}

//------------------------------------------------------------------------------
// MPMSolver implementation
//------------------------------------------------------------------------------
MPMSolver::MPMSolver(const SolverConfig& config)
    : BaseSolver("MPMSolver") {
    config_ = config;
}

MPMSolver::~MPMSolver() = default;

void MPMSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    rebuild_particle_list(entities);
    if (!entity_particles_.empty()) {
        std::vector<datatypes::Vector3> all_positions;
        for (const auto& ep : entity_particles_) {
            for (const auto& p : ep.particles) {
                all_positions.push_back(datatypes::Vector3(p.position.x(), p.position.y(), p.position.z()));
            }
        }
        setup_grid(all_positions);
    }
    initialized_ = true;
}

void MPMSolver::reset() {
    BaseSolver::reset();
    entity_particles_.clear();
    entity_to_index_.clear();
    grid_nodes_.clear();
    particles_dirty_ = true;
}

void MPMSolver::step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    
    if (!initialized_ || particles_dirty_) {
        initialize(entities);
    }
    
    if (entity_particles_.empty()) return;
    
    // CFL condition: limit dt based on max velocity
    double max_vel = 0.0;
    for (const auto& ep : entity_particles_) {
        for (const auto& p : ep.particles) {
            max_vel = std::max(max_vel, p.velocity.norm());
        }
    }
    double dt_cfl = mpm_config_.cfl_factor * mpm_config_.cell_size / (max_vel + 1e-12);
    if (dt > dt_cfl) {
        dt = dt_cfl;
    }
    
    // MPM algorithm
    reset_grid();
    particle_to_grid(dt);
    update_grid_velocities(dt);
    apply_boundary_conditions();
    grid_to_particle(dt);
    
    // Write back to entities
    for (auto& ep : entity_particles_) {
        std::vector<datatypes::Vector3> positions, velocities;
        std::vector<datatypes::Matrix3r> F, C;
        positions.reserve(ep.particles.size());
        velocities.reserve(ep.particles.size());
        F.reserve(ep.particles.size());
        C.reserve(ep.particles.size());
        for (const auto& p : ep.particles) {
            positions.push_back(datatypes::Vector3(p.position.x(), p.position.y(), p.position.z()));
            velocities.push_back(datatypes::Vector3(p.velocity.x(), p.velocity.y(), p.velocity.z()));
            datatypes::Matrix3r F_mat;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    F_mat(i, j) = p.F(i, j);
            F.push_back(F_mat);
            datatypes::Matrix3r C_mat;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    C_mat(i, j) = p.C(i, j);
            C.push_back(C_mat);
        }
        ep.entity->set_particle_states(positions, velocities, F, C);
    }
    
    stats_.particle_count = 0;
    for (const auto& ep : entity_particles_) stats_.particle_count += ep.particles.size();
}

void MPMSolver::on_entity_added(std::shared_ptr<BaseEntity> entity) {
    if (std::dynamic_pointer_cast<MPMEntity>(entity)) {
        particles_dirty_ = true;
    }
}

void MPMSolver::on_entity_removed(std::shared_ptr<BaseEntity> entity) {
    auto it = entity_to_index_.find(entity->id());
    if (it != entity_to_index_.end()) {
        entity_particles_.erase(entity_particles_.begin() + it->second);
        particles_dirty_ = true;
    }
}

void MPMSolver::rebuild_particle_list(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    entity_particles_.clear();
    entity_to_index_.clear();
    
    for (const auto& e : entities) {
        auto mpm_entity = std::dynamic_pointer_cast<MPMEntity>(e);
        if (!mpm_entity) continue;
        
        EntityParticles ep;
        ep.entity = mpm_entity;
        size_t n = mpm_entity->particle_count();
        const auto& pos = mpm_entity->particle_positions();
        const auto& vel = mpm_entity->particle_velocities();
        const auto& F = mpm_entity->deformation_gradients();
        const auto& C = mpm_entity->affine_velocities();
        const auto& mass = mpm_entity->particle_masses();
        const auto& vol = mpm_entity->particle_volumes();
        
        ep.particles.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ep.particles[i].position = Eigen::Vector3d(pos[i][0], pos[i][1], pos[i][2]);
            ep.particles[i].velocity = Eigen::Vector3d(vel[i][0], vel[i][1], vel[i][2]);
            ep.particles[i].mass = mass[i];
            ep.particles[i].volume = vol[i];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    ep.particles[i].F(r, c) = F[i](r, c);
            if (!C.empty()) {
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        ep.particles[i].C(r, c) = C[i](r, c);
            }
            ep.particles[i].material_id = mpm_entity->material_id();
        }
        
        entity_to_index_[mpm_entity->id()] = entity_particles_.size();
        entity_particles_.push_back(std::move(ep));
    }
    particles_dirty_ = false;
}

void MPMSolver::setup_grid(const std::vector<datatypes::Vector3>& particle_positions) {
    if (particle_positions.empty()) {
        grid_.nx = mpm_config_.grid_resolution[0];
        grid_.ny = mpm_config_.grid_resolution[1];
        grid_.nz = mpm_config_.grid_resolution[2];
        grid_.origin = mpm_config_.grid_origin;
    } else {
        datatypes::AABB bbox;
        for (const auto& p : particle_positions) bbox.expand(p);
        datatypes::Vector3 padding(mpm_config_.cell_size * 3);
        bbox.min = bbox.min - padding;
        bbox.max = bbox.max + padding;
        grid_.origin = bbox.min;
        datatypes::Vector3 ext = bbox.max - bbox.min;
        grid_.nx = std::max(4, static_cast<int>(std::ceil(ext[0] / mpm_config_.cell_size)));
        grid_.ny = std::max(4, static_cast<int>(std::ceil(ext[1] / mpm_config_.cell_size)));
        grid_.nz = std::max(4, static_cast<int>(std::ceil(ext[2] / mpm_config_.cell_size)));
    }
    grid_.dx = mpm_config_.cell_size;
    grid_.dy = mpm_config_.cell_size;
    grid_.dz = mpm_config_.cell_size;
    
    grid_domain_.min = grid_.origin;
    grid_domain_.max = grid_.origin + datatypes::Vector3(grid_.nx * grid_.dx,
                                                         grid_.ny * grid_.dy,
                                                         grid_.nz * grid_.dz);
    
    size_t total_nodes = static_cast<size_t>(grid_.nx) * grid_.ny * grid_.nz;
    grid_.nodes.assign(total_nodes, MPMGridNode{});
    grid_nodes_.clear();
    grid_nodes_.resize(total_nodes);
}

void MPMSolver::reset_grid() {
    for (auto& node : grid_.nodes) {
        node.mass = 0.0;
        node.velocity.setZero();
        node.active = false;
    }
}

void MPMSolver::particle_to_grid(double dt) {
    const double dx = grid_.dx;
    const double inv_dx = 1.0 / dx;
    
    for (auto& ep : entity_particles_) {
        for (auto& p : ep.particles) {
            // Compute stress and update deformation gradient
            double hardening = 0.0;
            Eigen::Matrix3d P = compute_pk1_stress(p, hardening);
            
            // Interpolate particle to grid
            std::array<int, 3> base;
            std::array<double, 3> wx, dwx;
            interpolate_weights(p.position, grid_, base, wx, dwx);
            
            Eigen::Matrix3d stress = -p.volume * P * p.F.transpose();
            Eigen::Matrix3d affine = stress * dt + p.mass * p.C;
            
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        int ix = base[0] + i;
                        int iy = base[1] + j;
                        int iz = base[2] + k;
                        if (!grid_.valid(ix, iy, iz)) continue;
                        
                        double w = wx[i] * wx[j] * wx[k];
                        double ddx = dwx[i] * wx[j] * wx[k];
                        double ddy = wx[i] * dwx[j] * wx[k];
                        double ddz = wx[i] * wx[j] * dwx[k];
                        
                        int idx = grid_.idx(ix, iy, iz);
                        grid_.nodes[idx].mass += w * p.mass;
                        
                        Eigen::Vector3d momentum_contrib = w * p.mass * p.velocity;
                        momentum_contrib += affine * Eigen::Vector3d(ddx, ddy, ddz);
                        grid_.nodes[idx].velocity += momentum_contrib;
                        grid_.nodes[idx].active = true;
                    }
                }
            }
        }
    }
    
    // Normalize velocity by mass
    for (auto& node : grid_.nodes) {
        if (node.active && node.mass > 1e-12) {
            node.velocity /= node.mass;
        } else {
            node.velocity.setZero();
        }
    }
}

void MPMSolver::update_grid_velocities(double dt) {
    // Apply gravity and external forces
    Eigen::Vector3d gravity(0, 0, -9.80665);
    
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx(i, j, k);
                if (grid_.nodes[idx].active) {
                    grid_.nodes[idx].velocity += gravity * dt;
                }
            }
        }
    }
    
    // Implicit solve not implemented (explicit only for now)
}

void MPMSolver::apply_boundary_conditions() {
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx(i, j, k);
                if (!grid_.nodes[idx].active) continue;
                
                Eigen::Vector3d pos = grid_.node_position(i, j, k);
                bool at_boundary = false;
                
                // X boundaries
                if (i == 0 && mpm_config_.boundary_type[0] != MPMConfig::BoundaryType::NONE) {
                    at_boundary = true;
                    grid_.nodes[idx].velocity.x() = std::max(0.0, grid_.nodes[idx].velocity.x());
                }
                if (i == grid_.nx - 1 && mpm_config_.boundary_type[1] != MPMConfig::BoundaryType::NONE) {
                    at_boundary = true;
                    grid_.nodes[idx].velocity.x() = std::min(0.0, grid_.nodes[idx].velocity.x());
                }
                // Y boundaries
                if (j == 0 && mpm_config_.boundary_type[2] != MPMConfig::BoundaryType::NONE) {
                    at_boundary = true;
                    grid_.nodes[idx].velocity.y() = std::max(0.0, grid_.nodes[idx].velocity.y());
                }
                if (j == grid_.ny - 1 && mpm_config_.boundary_type[3] != MPMConfig::BoundaryType::NONE) {
                    at_boundary = true;
                    grid_.nodes[idx].velocity.y() = std::min(0.0, grid_.nodes[idx].velocity.y());
                }
                // Z boundaries
                if (k == 0 && mpm_config_.boundary_type[4] != MPMConfig::BoundaryType::NONE) {
                    at_boundary = true;
                    grid_.nodes[idx].velocity.z() = std::max(0.0, grid_.nodes[idx].velocity.z());
                }
                if (k == grid_.nz - 1 && mpm_config_.boundary_type[5] != MPMConfig::BoundaryType::NONE) {
                    at_boundary = true;
                    grid_.nodes[idx].velocity.z() = std::min(0.0, grid_.nodes[idx].velocity.z());
                }
                
                // Sticky boundary zeroes tangential components
                if (at_boundary && mpm_config_.boundary_type[0] == MPMConfig::BoundaryType::STICKY) {
                    if (i == 0 || i == grid_.nx - 1) {
                        grid_.nodes[idx].velocity.y() = 0.0;
                        grid_.nodes[idx].velocity.z() = 0.0;
                    }
                    if (j == 0 || j == grid_.ny - 1) {
                        grid_.nodes[idx].velocity.x() = 0.0;
                        grid_.nodes[idx].velocity.z() = 0.0;
                    }
                    if (k == 0 || k == grid_.nz - 1) {
                        grid_.nodes[idx].velocity.x() = 0.0;
                        grid_.nodes[idx].velocity.y() = 0.0;
                    }
                }
            }
        }
    }
}

void MPMSolver::grid_to_particle(double dt) {
    for (auto& ep : entity_particles_) {
        for (auto& p : ep.particles) {
            std::array<int, 3> base;
            std::array<double, 3> wx, dwx;
            interpolate_weights(p.position, grid_, base, wx, dwx);
            
            Eigen::Vector3d new_vel = Eigen::Vector3d::Zero();
            Eigen::Matrix3d velocity_gradient = Eigen::Matrix3d::Zero();
            Eigen::Vector3d grid_vels[3][3][3];
            
            // Gather grid velocities and compute interpolated velocity
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        int ix = base[0] + i;
                        int iy = base[1] + j;
                        int iz = base[2] + k;
                        if (!grid_.valid(ix, iy, iz)) continue;
                        int idx = grid_.idx(ix, iy, iz);
                        grid_vels[i][j][k] = grid_.nodes[idx].velocity;
                        
                        double w = wx[i] * wx[j] * wx[k];
                        new_vel += w * grid_.nodes[idx].velocity;
                    }
                }
            }
            
            // Compute velocity gradient for deformation gradient update
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        double dwx_val = dwx[i] * wx[j] * wx[k];
                        double dwy_val = wx[i] * dwx[j] * wx[k];
                        double dwz_val = wx[i] * wx[j] * dwx[k];
                        Eigen::Vector3d v = grid_vels[i][j][k];
                        velocity_gradient.col(0) += v * dwx_val;
                        velocity_gradient.col(1) += v * dwy_val;
                        velocity_gradient.col(2) += v * dwz_val;
                    }
                }
            }
            
            // Update deformation gradient: F_new = (I + dt * grad_v) * F
            Eigen::Matrix3d F_new = (Eigen::Matrix3d::Identity() + dt * velocity_gradient) * p.F;
            
            // Apply plasticity
            if (mpm_config_.material_model == MPMConfig::MaterialModel::VON_MISES_PLASTIC ||
                mpm_config_.material_model == MPMConfig::MaterialModel::SAND ||
                mpm_config_.material_model == MPMConfig::MaterialModel::SNOW) {
                p.F = F_new;
                apply_plasticity(p);
            } else {
                p.F = F_new;
            }
            
            // FLIP/PIC blending
            Eigen::Vector3d pic_vel = new_vel;
            Eigen::Vector3d flip_vel = p.velocity + (new_vel - p.velocity);
            p.velocity = (1.0 - mpm_config_.flip_pic_ratio) * pic_vel + mpm_config_.flip_pic_ratio * flip_vel;
            
            // Update position
            p.position += new_vel * dt;
            
            // Update affine matrix for APIC
            if (mpm_config_.enable_affine) {
                Eigen::Matrix3d C_new = Eigen::Matrix3d::Zero();
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            double w = wx[i] * wx[j] * wx[k];
                            Eigen::Vector3d v_diff = grid_vels[i][j][k] - new_vel;
                            C_new += p.mass * w * v_diff * Eigen::Vector3d(
                                (i - 1) * grid_.dx,
                                (j - 1) * grid_.dy,
                                (k - 1) * grid_.dz
                            ).transpose();
                        }
                    }
                }
                p.C = C_new;
            }
            
            // Damping
            if (mpm_config_.velocity_damping > 0) {
                p.velocity *= (1.0 - mpm_config_.velocity_damping);
            }
        }
    }
}

Eigen::Matrix3d MPMSolver::compute_pk1_stress(const MPMParticleState& p, double& hardening) const {
    double mu = mpm_config_.youngs_modulus / (2.0 * (1.0 + mpm_config_.poisson_ratio));
    double lambda = mpm_config_.youngs_modulus * mpm_config_.poisson_ratio /
                    ((1.0 + mpm_config_.poisson_ratio) * (1.0 - 2.0 * mpm_config_.poisson_ratio));
    
    Eigen::Matrix3d F = p.F;
    double J = F.determinant();
    
    switch (mpm_config_.material_model) {
        case MPMConfig::MaterialModel::NEO_HOOKEAN: {
            Eigen::Matrix3d F_inv_T = F.inverse().transpose();
            Eigen::Matrix3d P = mu * (F - F_inv_T) + lambda * std::log(J) * F_inv_T;
            return P;
        }
        case MPMConfig::MaterialModel::FIXED_COROTATED: {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
            Eigen::Matrix3d S = R.transpose() * F;
            Eigen::Matrix3d epsilon = 0.5 * (S + S.transpose()) - Eigen::Matrix3d::Identity();
            Eigen::Matrix3d sigma = lambda * epsilon.trace() * Eigen::Matrix3d::Identity() + 2.0 * mu * epsilon;
            return R * sigma;
        }
        case MPMConfig::MaterialModel::VON_MISES_PLASTIC: {
            // Plasticity handled separately in apply_plasticity
            Eigen::Matrix3d F_elastic = F * p.Fp.inverse();
            double Je = F_elastic.determinant();
            Eigen::Matrix3d F_inv_T = F_elastic.inverse().transpose();
            return mu * (F_elastic - F_inv_T) + lambda * std::log(Je) * F_inv_T;
        }
        default:
            return Eigen::Matrix3d::Zero();
    }
}

void MPMSolver::apply_plasticity(MPMParticleState& p) const {
    double mu = mpm_config_.youngs_modulus / (2.0 * (1.0 + mpm_config_.poisson_ratio));
    double lambda = mpm_config_.youngs_modulus * mpm_config_.poisson_ratio /
                    ((1.0 + mpm_config_.poisson_ratio) * (1.0 - 2.0 * mpm_config_.poisson_ratio));
    double yield_stress = mpm_config_.yield_stress;
    
    Eigen::Matrix3d F = p.F;
    Eigen::Matrix3d Fp = p.Fp;
    Eigen::Matrix3d Fe = F * Fp.inverse();
    
    // SVD of Fe
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Fe, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sig = svd.singularValues();
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // Compute trial Kirchhoff stress principal values
    double Je = sig.prod();
    Eigen::Vector3d tau_trial;
    for (int i = 0; i < 3; ++i) {
        tau_trial[i] = mu * (sig[i] * sig[i] - 1.0) + lambda * std::log(Je);
    }
    
    // von Mises yield criterion
    Eigen::Vector3d s = tau_trial - (tau_trial.sum() / 3.0) * Eigen::Vector3d::Ones();
    double tau_bar = std::sqrt(1.5 * s.dot(s));
    
    if (tau_bar > yield_stress) {
        double alpha = yield_stress / tau_bar;
        Eigen::Vector3d sig_projected = sig;
        // Simple projection: scale back deviatoric part of sig
        for (int i = 0; i < 3; ++i) {
            double sig_m = sig.sum() / 3.0;
            sig_projected[i] = sig_m + alpha * (sig[i] - sig_m);
        }
        // Update Fe and Fp
        Fe = U * sig_projected.asDiagonal() * V.transpose();
        F = Fe * Fp;
        p.Fp = F * Fe.inverse();
        p.F = F;
    }
}

} // namespace engine
} // namespace genesis