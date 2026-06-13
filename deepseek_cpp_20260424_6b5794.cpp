// genesis/engine/solvers/sf_solver.cpp

#include "genesis/engine/solvers/sf_solver.h"
#include "genesis/engine/entities/sf_entity.h"
#include "genesis/engine/mesh.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <limits>
#include <Eigen/IterativeLinearSolvers>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Helper: linear interpolation
//------------------------------------------------------------------------------
inline double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

inline double bilinear_interp(double v00, double v10, double v01, double v11, double tx, double ty) {
    return lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty);
}

inline double trilinear_interp(const double v[8], const double t[3]) {
    return bilinear_interp(
        lerp(v[0], v[1], t[0]),
        lerp(v[4], v[5], t[0]),
        lerp(v[2], v[3], t[0]),
        lerp(v[6], v[7], t[0]),
        t[2], t[1]
    );
}

//------------------------------------------------------------------------------
// SFSolver implementation
//------------------------------------------------------------------------------
SFSolver::SFSolver(const SolverConfig& config)
    : BaseSolver("SFSolver") {
    config_ = config;
}

SFSolver::~SFSolver() = default;

void SFSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    rebuild_states(entities);
    
    // Determine grid domain from all entities
    datatypes::AABB domain;
    for (const auto& state : entity_states_) {
        datatypes::AABB aabb = state.entity->world_aabb();
        domain.expand(aabb.min);
        domain.expand(aabb.max);
    }
    // Add padding
    domain.min = domain.min - datatypes::Vector3(sf_config_.cell_size * 4);
    domain.max = domain.max + datatypes::Vector3(sf_config_.cell_size * 4);
    setup_grid(domain);
    
    initialized_ = true;
}

void SFSolver::reset() {
    BaseSolver::reset();
    entity_states_.clear();
    entity_to_index_.clear();
    solid_obstacles_.clear();
    grid_ = SFGrid{};
    states_dirty_ = true;
}

void SFSolver::step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    
    if (!initialized_ || states_dirty_) {
        initialize(entities);
    }
    
    if (entity_states_.empty()) return;
    
    // CFL condition
    double max_vel = 0.0;
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i <= grid_.nx; ++i) {
                int idx = grid_.idx_u(i, j, k);
                max_vel = std::max(max_vel, std::abs(grid_.u[idx]));
            }
        }
    }
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j <= grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_v(i, j, k);
                max_vel = std::max(max_vel, std::abs(grid_.v[idx]));
            }
        }
    }
    for (int k = 0; k <= grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_w(i, j, k);
                max_vel = std::max(max_vel, std::abs(grid_.w[idx]));
            }
        }
    }
    
    double dt_cfl = sf_config_.cfl_factor * grid_.dx / (max_vel + 1e-12);
    if (dt > dt_cfl) {
        dt = dt_cfl;
    }
    
    // Navier-Stokes steps
    advect_velocity(dt);
    apply_external_forces(dt);
    apply_boundary_conditions();
    compute_pressure(dt);
    apply_pressure_gradient(dt);
    extrapolate_velocity();
    apply_boundary_conditions();
    advect_level_set(dt);
    reinitialize_level_set();
    
    // Update entity states
    update_entity_states();
}

void SFSolver::on_entity_added(std::shared_ptr<BaseEntity> entity) {
    if (std::dynamic_pointer_cast<SFEntity>(entity)) {
        states_dirty_ = true;
    }
}

void SFSolver::on_entity_removed(std::shared_ptr<BaseEntity> entity) {
    auto it = entity_to_index_.find(entity->id());
    if (it != entity_to_index_.end()) {
        entity_states_.erase(entity_states_.begin() + it->second);
        states_dirty_ = true;
    }
}

void SFSolver::build_sdf_from_mesh(const Mesh& mesh, int entity_id) {
    // Compute signed distance field for the mesh on the grid
    // Simple brute force for now; in practice use fast marching / narrow band
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_cell(i, j, k);
                Eigen::Vector3d cell_center(
                    grid_.origin[0] + (i + 0.5) * grid_.dx,
                    grid_.origin[1] + (j + 0.5) * grid_.dy,
                    grid_.origin[2] + (k + 0.5) * grid_.dz
                );
                datatypes::Vector3 query(cell_center.x(), cell_center.y(), cell_center.z());
                auto cp = mesh.closest_point(query);
                double dist = std::sqrt(cp.distance_sq);
                // Determine sign (simplified: assume mesh is closed and we can use normal)
                datatypes::Vector3 dir = query - cp.point;
                double sign = (dir.dot(cp.normal) > 0) ? 1.0 : -1.0;
                grid_.level_set[idx] = sign * dist;
            }
        }
    }
}

void SFSolver::add_solid_obstacle(const datatypes::AABB& aabb) {
    solid_obstacles_.push_back(aabb);
    // Mark solid cells
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_cell(i, j, k);
                Eigen::Vector3d cell_center(
                    grid_.origin[0] + (i + 0.5) * grid_.dx,
                    grid_.origin[1] + (j + 0.5) * grid_.dy,
                    grid_.origin[2] + (k + 0.5) * grid_.dz
                );
                datatypes::Vector3 p(cell_center.x(), cell_center.y(), cell_center.z());
                if (aabb.contains(p)) {
                    grid_.solid[idx] = true;
                    grid_.fluid[idx] = false;
                }
            }
        }
    }
}

void SFSolver::clear_solid_obstacles() {
    solid_obstacles_.clear();
    std::fill(grid_.solid.begin(), grid_.solid.end(), false);
}

void SFSolver::rebuild_states(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    entity_states_.clear();
    entity_to_index_.clear();
    
    for (const auto& e : entities) {
        auto sf_entity = std::dynamic_pointer_cast<SFEntity>(e);
        if (!sf_entity) continue;
        
        EntitySFState state;
        state.entity = sf_entity;
        state.level_set_id = static_cast<int>(entity_states_.size());
        
        // If entity has particles, populate them
        if (sf_entity->has_particles()) {
            const auto& pos = sf_entity->particle_positions();
            const auto& vel = sf_entity->particle_velocities();
            const auto& mass = sf_entity->particle_masses();
            state.particles.resize(pos.size());
            for (size_t i = 0; i < pos.size(); ++i) {
                state.particles[i].position = Eigen::Vector3d(pos[i][0], pos[i][1], pos[i][2]);
                state.particles[i].velocity = Eigen::Vector3d(vel[i][0], vel[i][1], vel[i][2]);
                state.particles[i].mass = mass[i];
                state.particles[i].radius = sf_entity->particle_radius();
                state.particles[i].material_id = sf_entity->material_id();
            }
        }
        
        entity_to_index_[sf_entity->id()] = entity_states_.size();
        entity_states_.push_back(std::move(state));
    }
    states_dirty_ = false;
}

void SFSolver::setup_grid(const datatypes::AABB& domain) {
    grid_.origin = domain.min;
    datatypes::Vector3 ext = domain.max - domain.min;
    
    grid_.dx = sf_config_.cell_size;
    grid_.dy = sf_config_.cell_size;
    grid_.dz = sf_config_.cell_size;
    
    grid_.nx = std::max(8, static_cast<int>(std::ceil(ext[0] / grid_.dx)));
    grid_.ny = std::max(8, static_cast<int>(std::ceil(ext[1] / grid_.dy)));
    grid_.nz = std::max(8, static_cast<int>(std::ceil(ext[2] / grid_.dz)));
    
    // Allocate staggered arrays
    size_t u_size = static_cast<size_t>(grid_.nx + 1) * grid_.ny * grid_.nz;
    size_t v_size = static_cast<size_t>(grid_.nx) * (grid_.ny + 1) * grid_.nz;
    size_t w_size = static_cast<size_t>(grid_.nx) * grid_.ny * (grid_.nz + 1);
    size_t cell_size = static_cast<size_t>(grid_.nx) * grid_.ny * grid_.nz;
    
    grid_.u.assign(u_size, 0.0);
    grid_.v.assign(v_size, 0.0);
    grid_.w.assign(w_size, 0.0);
    grid_.pressure.assign(cell_size, 0.0);
    grid_.level_set.assign(cell_size, std::numeric_limits<double>::max());
    grid_.solid.assign(cell_size, false);
    grid_.fluid.assign(cell_size, false);
    grid_.density.assign(cell_size, sf_config_.density);
    grid_.temperature.assign(cell_size, 293.0);
    
    // Mark fluid cells initially (from entities)
    for (const auto& state : entity_states_) {
        datatypes::AABB aabb = state.entity->world_aabb();
        for (int k = 0; k < grid_.nz; ++k) {
            for (int j = 0; j < grid_.ny; ++j) {
                for (int i = 0; i < grid_.nx; ++i) {
                    int idx = grid_.idx_cell(i, j, k);
                    Eigen::Vector3d cell_center(
                        grid_.origin[0] + (i + 0.5) * grid_.dx,
                        grid_.origin[1] + (j + 0.5) * grid_.dy,
                        grid_.origin[2] + (k + 0.5) * grid_.dz
                    );
                    datatypes::Vector3 p(cell_center.x(), cell_center.y(), cell_center.z());
                    if (aabb.contains(p)) {
                        grid_.fluid[idx] = true;
                        grid_.level_set[idx] = -grid_.dx; // inside
                    }
                }
            }
        }
    }
}

void SFSolver::advect_velocity(double dt) {
    SFGrid new_grid = grid_;
    
    // Advect u
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i <= grid_.nx; ++i) {
                int idx = grid_.idx_u(i, j, k);
                if (grid_.solid[idx]) continue;
                
                Eigen::Vector3d pos(
                    grid_.origin[0] + i * grid_.dx,
                    grid_.origin[1] + (j + 0.5) * grid_.dy,
                    grid_.origin[2] + (k + 0.5) * grid_.dz
                );
                
                Eigen::Vector3d vel = sample_velocity(pos);
                Eigen::Vector3d back_pos = trace_rk3(pos, -dt);
                new_grid.u[idx] = sample_velocity_u(back_pos.x(), back_pos.y(), back_pos.z());
            }
        }
    }
    
    // Advect v
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j <= grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_v(i, j, k);
                Eigen::Vector3d pos(
                    grid_.origin[0] + (i + 0.5) * grid_.dx,
                    grid_.origin[1] + j * grid_.dy,
                    grid_.origin[2] + (k + 0.5) * grid_.dz
                );
                Eigen::Vector3d vel = sample_velocity(pos);
                Eigen::Vector3d back_pos = trace_rk3(pos, -dt);
                new_grid.v[idx] = sample_velocity_v(back_pos.x(), back_pos.y(), back_pos.z());
            }
        }
    }
    
    // Advect w
    for (int k = 0; k <= grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_w(i, j, k);
                Eigen::Vector3d pos(
                    grid_.origin[0] + (i + 0.5) * grid_.dx,
                    grid_.origin[1] + (j + 0.5) * grid_.dy,
                    grid_.origin[2] + k * grid_.dz
                );
                Eigen::Vector3d vel = sample_velocity(pos);
                Eigen::Vector3d back_pos = trace_rk3(pos, -dt);
                new_grid.w[idx] = sample_velocity_w(back_pos.x(), back_pos.y(), back_pos.z());
            }
        }
    }
    
    grid_.u = std::move(new_grid.u);
    grid_.v = std::move(new_grid.v);
    grid_.w = std::move(new_grid.w);
}

void SFSolver::apply_external_forces(double dt) {
    Eigen::Vector3d gravity(0.0, -9.80665, 0.0);
    
    // Add gravity to v component
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j <= grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_v(i, j, k);
                if (!grid_.solid[idx]) {
                    grid_.v[idx] += gravity.y() * dt;
                }
            }
        }
    }
}

void SFSolver::compute_pressure(double dt) {
    assemble_pressure_system(dt);
    
    if (A_.rows() == 0) return;
    
    pressure_solver_.compute(A_);
    if (pressure_solver_.info() == Eigen::Success) {
        solution_ = pressure_solver_.solve(rhs_);
        
        // Copy back to grid pressure
        int n_cells = grid_.nx * grid_.ny * grid_.nz;
        for (int idx = 0; idx < n_cells; ++idx) {
            grid_.pressure[idx] = solution_[idx];
        }
    }
}

void SFSolver::assemble_pressure_system(double dt) {
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    int n_cells = nx * ny * nz;
    
    std::vector<Eigen::Triplet<double>> triplets;
    rhs_ = Eigen::VectorXd::Zero(n_cells);
    
    double scale = dt / (sf_config_.density * grid_.dx * grid_.dx);
    
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = grid_.idx_cell(i, j, k);
                if (grid_.solid[idx] || !grid_.fluid[idx]) {
                    triplets.emplace_back(idx, idx, 1.0);
                    continue;
                }
                
                double diag = 0.0;
                double div = 0.0;
                
                // -x neighbor
                if (i > 0 && grid_.fluid[grid_.idx_cell(i-1, j, k)]) {
                    diag += scale;
                    triplets.emplace_back(idx, grid_.idx_cell(i-1, j, k), -scale);
                }
                div += grid_.u[grid_.idx_u(i+1, j, k)] - grid_.u[grid_.idx_u(i, j, k)];
                
                // +x neighbor
                if (i < nx-1 && grid_.fluid[grid_.idx_cell(i+1, j, k)]) {
                    diag += scale;
                    triplets.emplace_back(idx, grid_.idx_cell(i+1, j, k), -scale);
                }
                
                // -y neighbor
                if (j > 0 && grid_.fluid[grid_.idx_cell(i, j-1, k)]) {
                    diag += scale;
                    triplets.emplace_back(idx, grid_.idx_cell(i, j-1, k), -scale);
                }
                div += grid_.v[grid_.idx_v(i, j+1, k)] - grid_.v[grid_.idx_v(i, j, k)];
                
                // +y neighbor
                if (j < ny-1 && grid_.fluid[grid_.idx_cell(i, j+1, k)]) {
                    diag += scale;
                    triplets.emplace_back(idx, grid_.idx_cell(i, j+1, k), -scale);
                }
                
                // -z neighbor
                if (k > 0 && grid_.fluid[grid_.idx_cell(i, j, k-1)]) {
                    diag += scale;
                    triplets.emplace_back(idx, grid_.idx_cell(i, j, k-1), -scale);
                }
                div += grid_.w[grid_.idx_w(i, j, k+1)] - grid_.w[grid_.idx_w(i, j, k)];
                
                // +z neighbor
                if (k < nz-1 && grid_.fluid[grid_.idx_cell(i, j, k+1)]) {
                    diag += scale;
                    triplets.emplace_back(idx, grid_.idx_cell(i, j, k+1), -scale);
                }
                
                triplets.emplace_back(idx, idx, diag);
                rhs_[idx] = -div / grid_.dx;
            }
        }
    }
    
    A_.resize(n_cells, n_cells);
    A_.setFromTriplets(triplets.begin(), triplets.end());
}

void SFSolver::apply_pressure_gradient(double dt) {
    double scale = dt / (sf_config_.density * grid_.dx);
    
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 1; i < grid_.nx; ++i) {
                int idx_u = grid_.idx_u(i, j, k);
                int idx_c0 = grid_.idx_cell(i-1, j, k);
                int idx_c1 = grid_.idx_cell(i, j, k);
                if (grid_.fluid[idx_c0] && grid_.fluid[idx_c1]) {
                    grid_.u[idx_u] -= scale * (grid_.pressure[idx_c1] - grid_.pressure[idx_c0]);
                }
            }
        }
    }
    
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 1; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx_v = grid_.idx_v(i, j, k);
                int idx_c0 = grid_.idx_cell(i, j-1, k);
                int idx_c1 = grid_.idx_cell(i, j, k);
                if (grid_.fluid[idx_c0] && grid_.fluid[idx_c1]) {
                    grid_.v[idx_v] -= scale * (grid_.pressure[idx_c1] - grid_.pressure[idx_c0]);
                }
            }
        }
    }
    
    for (int k = 1; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx_w = grid_.idx_w(i, j, k);
                int idx_c0 = grid_.idx_cell(i, j, k-1);
                int idx_c1 = grid_.idx_cell(i, j, k);
                if (grid_.fluid[idx_c0] && grid_.fluid[idx_c1]) {
                    grid_.w[idx_w] -= scale * (grid_.pressure[idx_c1] - grid_.pressure[idx_c0]);
                }
            }
        }
    }
}

void SFSolver::extrapolate_velocity() {
    // Simple extrapolation: average neighbors
    SFGrid new_grid = grid_;
    
    for (int iter = 0; iter < 5; ++iter) {
        // Extrapolate u
        for (int k = 0; k < grid_.nz; ++k) {
            for (int j = 0; j < grid_.ny; ++j) {
                for (int i = 0; i <= grid_.nx; ++i) {
                    int idx = grid_.idx_u(i, j, k);
                    if (grid_.fluid[grid_.idx_cell(std::max(0, std::min(i, grid_.nx-1)), j, k)]) continue;
                    
                    double sum = 0.0;
                    int count = 0;
                    if (i > 0 && grid_.fluid[grid_.idx_cell(i-1, j, k)]) { sum += grid_.u[grid_.idx_u(i-1, j, k)]; count++; }
                    if (i < grid_.nx && grid_.fluid[grid_.idx_cell(i, j, k)]) { sum += grid_.u[grid_.idx_u(i+1, j, k)]; count++; }
                    if (j > 0 && grid_.fluid[grid_.idx_cell(std::min(i, grid_.nx-1), j-1, k)]) { sum += grid_.u[grid_.idx_u(i, j-1, k)]; count++; }
                    if (j < grid_.ny-1 && grid_.fluid[grid_.idx_cell(std::min(i, grid_.nx-1), j+1, k)]) { sum += grid_.u[grid_.idx_u(i, j+1, k)]; count++; }
                    if (k > 0 && grid_.fluid[grid_.idx_cell(std::min(i, grid_.nx-1), j, k-1)]) { sum += grid_.u[grid_.idx_u(i, j, k-1)]; count++; }
                    if (k < grid_.nz-1 && grid_.fluid[grid_.idx_cell(std::min(i, grid_.nx-1), j, k+1)]) { sum += grid_.u[grid_.idx_u(i, j, k+1)]; count++; }
                    if (count > 0) new_grid.u[idx] = sum / count;
                }
            }
        }
        grid_.u.swap(new_grid.u);
        
        // Extrapolate v and w similarly (omitted for brevity but would be implemented)
    }
}

void SFSolver::apply_boundary_conditions() {
    // Enforce solid boundaries
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            // Left wall (i=0)
            if (sf_config_.boundary_type[0] == SFConfig::BoundaryType::SOLID) {
                grid_.u[grid_.idx_u(0, j, k)] = 0.0;
            }
            // Right wall (i=nx)
            if (sf_config_.boundary_type[1] == SFConfig::BoundaryType::SOLID) {
                grid_.u[grid_.idx_u(grid_.nx, j, k)] = 0.0;
            }
        }
    }
    // Similar for v and w boundaries
}

void SFSolver::advect_level_set(double dt) {
    std::vector<double> new_phi = grid_.level_set;
    
    for (int k = 0; k < grid_.nz; ++k) {
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                int idx = grid_.idx_cell(i, j, k);
                Eigen::Vector3d pos(
                    grid_.origin[0] + (i + 0.5) * grid_.dx,
                    grid_.origin[1] + (j + 0.5) * grid_.dy,
                    grid_.origin[2] + (k + 0.5) * grid_.dz
                );
                Eigen::Vector3d vel = sample_velocity(pos);
                Eigen::Vector3d back_pos = trace_rk3(pos, -dt);
                new_phi[idx] = sample_level_set(back_pos);
            }
        }
    }
    grid_.level_set = std::move(new_phi);
}

void SFSolver::reinitialize_level_set() {
    // Simple reinitialization using fast marching (simplified: just clamp)
    // In production, would solve Eikonal equation |∇φ| = 1
}

void SFSolver::update_entity_states() {
    for (auto& state : entity_states_) {
        // If using particles, update their velocities from grid
        if (!state.particles.empty()) {
            for (auto& p : state.particles) {
                Eigen::Vector3d vel = sample_velocity(p.position);
                p.velocity = vel;
                p.position += vel * config_.time_step; // should use dt from step
            }
            // Write back to entity
            std::vector<datatypes::Vector3> pos, vel;
            for (const auto& p : state.particles) {
                pos.push_back(datatypes::Vector3(p.position.x(), p.position.y(), p.position.z()));
                vel.push_back(datatypes::Vector3(p.velocity.x(), p.velocity.y(), p.velocity.z()));
            }
            state.entity->set_particle_states(pos, vel);
        }
    }
}

double SFSolver::sample_velocity_u(double x, double y, double z) const {
    // Convert world to grid coordinates
    double gx = (x - grid_.origin[0]) / grid_.dx - 0.5;
    double gy = (y - grid_.origin[1]) / grid_.dy;
    double gz = (z - grid_.origin[2]) / grid_.dz;
    
    int i = static_cast<int>(std::floor(gx));
    int j = static_cast<int>(std::floor(gy));
    int k = static_cast<int>(std::floor(gz));
    
    double fx = gx - i;
    double fy = gy - j;
    double fz = gz - k;
    
    i = std::clamp(i, 0, grid_.nx);
    j = std::clamp(j, 0, grid_.ny-1);
    k = std::clamp(k, 0, grid_.nz-1);
    
    int i1 = std::min(i+1, grid_.nx);
    int j1 = std::min(j+1, grid_.ny-1);
    int k1 = std::min(k+1, grid_.nz-1);
    
    double v000 = grid_.u[grid_.idx_u(i, j, k)];
    double v100 = grid_.u[grid_.idx_u(i1, j, k)];
    double v010 = grid_.u[grid_.idx_u(i, j1, k)];
    double v110 = grid_.u[grid_.idx_u(i1, j1, k)];
    double v001 = grid_.u[grid_.idx_u(i, j, k1)];
    double v101 = grid_.u[grid_.idx_u(i1, j, k1)];
    double v011 = grid_.u[grid_.idx_u(i, j1, k1)];
    double v111 = grid_.u[grid_.idx_u(i1, j1, k1)];
    
    double v00 = lerp(v000, v100, fx);
    double v01 = lerp(v001, v101, fx);
    double v10 = lerp(v010, v110, fx);
    double v11 = lerp(v011, v111, fx);
    double v0 = lerp(v00, v10, fy);
    double v1 = lerp(v01, v11, fy);
    return lerp(v0, v1, fz);
}

double SFSolver::sample_velocity_v(double x, double y, double z) const {
    double gx = (x - grid_.origin[0]) / grid_.dx;
    double gy = (y - grid_.origin[1]) / grid_.dy - 0.5;
    double gz = (z - grid_.origin[2]) / grid_.dz;
    
    int i = static_cast<int>(std::floor(gx));
    int j = static_cast<int>(std::floor(gy));
    int k = static_cast<int>(std::floor(gz));
    
    double fx = gx - i;
    double fy = gy - j;
    double fz = gz - k;
    
    i = std::clamp(i, 0, grid_.nx-1);
    j = std::clamp(j, 0, grid_.ny);
    k = std::clamp(k, 0, grid_.nz-1);
    
    int i1 = std::min(i+1, grid_.nx-1);
    int j1 = std::min(j+1, grid_.ny);
    int k1 = std::min(k+1, grid_.nz-1);
    
    double v000 = grid_.v[grid_.idx_v(i, j, k)];
    double v100 = grid_.v[grid_.idx_v(i1, j, k)];
    double v010 = grid_.v[grid_.idx_v(i, j1, k)];
    double v110 = grid_.v[grid_.idx_v(i1, j1, k)];
    double v001 = grid_.v[grid_.idx_v(i, j, k1)];
    double v101 = grid_.v[grid_.idx_v(i1, j, k1)];
    double v011 = grid_.v[grid_.idx_v(i, j1, k1)];
    double v111 = grid_.v[grid_.idx_v(i1, j1, k1)];
    
    double v00 = lerp(v000, v100, fx);
    double v01 = lerp(v001, v101, fx);
    double v10 = lerp(v010, v110, fx);
    double v11 = lerp(v011, v111, fx);
    double v0 = lerp(v00, v10, fy);
    double v1 = lerp(v01, v11, fy);
    return lerp(v0, v1, fz);
}

double SFSolver::sample_velocity_w(double x, double y, double z) const {
    double gx = (x - grid_.origin[0]) / grid_.dx;
    double gy = (y - grid_.origin[1]) / grid_.dy;
    double gz = (z - grid_.origin[2]) / grid_.dz - 0.5;
    
    int i = static_cast<int>(std::floor(gx));
    int j = static_cast<int>(std::floor(gy));
    int k = static_cast<int>(std::floor(gz));
    
    double fx = gx - i;
    double fy = gy - j;
    double fz = gz - k;
    
    i = std::clamp(i, 0, grid_.nx-1);
    j = std::clamp(j, 0, grid_.ny-1);
    k = std::clamp(k, 0, grid_.nz);
    
    int i1 = std::min(i+1, grid_.nx-1);
    int j1 = std::min(j+1, grid_.ny-1);
    int k1 = std::min(k+1, grid_.nz);
    
    double v000 = grid_.w[grid_.idx_w(i, j, k)];
    double v100 = grid_.w[grid_.idx_w(i1, j, k)];
    double v010 = grid_.w[grid_.idx_w(i, j1, k)];
    double v110 = grid_.w[grid_.idx_w(i1, j1, k)];
    double v001 = grid_.w[grid_.idx_w(i, j, k1)];
    double v101 = grid_.w[grid_.idx_w(i1, j, k1)];
    double v011 = grid_.w[grid_.idx_w(i, j1, k1)];
    double v111 = grid_.w[grid_.idx_w(i1, j1, k1)];
    
    double v00 = lerp(v000, v100, fx);
    double v01 = lerp(v001, v101, fx);
    double v10 = lerp(v010, v110, fx);
    double v11 = lerp(v011, v111, fx);
    double v0 = lerp(v00, v10, fy);
    double v1 = lerp(v01, v11, fy);
    return lerp(v0, v1, fz);
}

Eigen::Vector3d SFSolver::sample_velocity(const Eigen::Vector3d& pos) const {
    return Eigen::Vector3d(
        sample_velocity_u(pos.x(), pos.y(), pos.z()),
        sample_velocity_v(pos.x(), pos.y(), pos.z()),
        sample_velocity_w(pos.x(), pos.y(), pos.z())
    );
}

double SFSolver::sample_level_set(const Eigen::Vector3d& pos) const {
    double gx = (pos.x() - grid_.origin[0]) / grid_.dx - 0.5;
    double gy = (pos.y() - grid_.origin[1]) / grid_.dy - 0.5;
    double gz = (pos.z() - grid_.origin[2]) / grid_.dz - 0.5;
    
    int i = static_cast<int>(std::floor(gx));
    int j = static_cast<int>(std::floor(gy));
    int k = static_cast<int>(std::floor(gz));
    
    double fx = gx - i;
    double fy = gy - j;
    double fz = gz - k;
    
    i = std::clamp(i, 0, grid_.nx-1);
    j = std::clamp(j, 0, grid_.ny-1);
    k = std::clamp(k, 0, grid_.nz-1);
    
    int i1 = std::min(i+1, grid_.nx-1);
    int j1 = std::min(j+1, grid_.ny-1);
    int k1 = std::min(k+1, grid_.nz-1);
    
    double v000 = grid_.level_set[grid_.idx_cell(i, j, k)];
    double v100 = grid_.level_set[grid_.idx_cell(i1, j, k)];
    double v010 = grid_.level_set[grid_.idx_cell(i, j1, k)];
    double v110 = grid_.level_set[grid_.idx_cell(i1, j1, k)];
    double v001 = grid_.level_set[grid_.idx_cell(i, j, k1)];
    double v101 = grid_.level_set[grid_.idx_cell(i1, j, k1)];
    double v011 = grid_.level_set[grid_.idx_cell(i, j1, k1)];
    double v111 = grid_.level_set[grid_.idx_cell(i1, j1, k1)];
    
    double v00 = lerp(v000, v100, fx);
    double v01 = lerp(v001, v101, fx);
    double v10 = lerp(v010, v110, fx);
    double v11 = lerp(v011, v111, fx);
    double v0 = lerp(v00, v10, fy);
    double v1 = lerp(v01, v11, fy);
    return lerp(v0, v1, fz);
}

Eigen::Vector3d SFSolver::trace_rk3(const Eigen::Vector3d& pos, double dt) const {
    Eigen::Vector3d v1 = sample_velocity(pos);
    Eigen::Vector3d p1 = pos + 0.5 * dt * v1;
    Eigen::Vector3d v2 = sample_velocity(p1);
    Eigen::Vector3d p2 = pos + 0.75 * dt * v2;
    Eigen::Vector3d v3 = sample_velocity(p2);
    return pos + dt * (2.0/9.0 * v1 + 3.0/9.0 * v2 + 4.0/9.0 * v3);
}

} // namespace engine
} // namespace genesis