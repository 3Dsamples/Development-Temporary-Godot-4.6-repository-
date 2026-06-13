// genesis/engine/solvers/sph_solver.cpp

#include "genesis/engine/solvers/sph_solver.h"
#include "genesis/engine/entities/sph_entity.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <unordered_set>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// SPHKernel implementation
//------------------------------------------------------------------------------
SPHKernel::SPHKernel(double h) : h_(h), h2_(h * h), h3_(h * h * h) {
    // Cubic spline kernel normalization constants for 3D
    // W_poly6 = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
    // W_spiky = 15 / (pi * h^6) * (h - r)^3
    // W_visc = 15 / (2 * pi * h^3) * (...)
    const double pi = 3.14159265358979323846;
    poly6_coef_ = 315.0 / (64.0 * pi * std::pow(h_, 9));
    spiky_coef_ = 15.0 / (pi * std::pow(h_, 6));
    visc_coef_ = 45.0 / (pi * std::pow(h_, 6));
}

double SPHKernel::W(double r) const {
    if (r >= h_) return 0.0;
    double q = r / h_;
    if (q < 0.5) {
        return poly6_coef_ * std::pow(h2_ - r*r, 3);
    } else {
        return 0.0; // Poly6 only supports r < h, cubic spline would have second branch
    }
}

double SPHKernel::W_normalized(double r) const {
    // For density we often use poly6
    if (r >= h_) return 0.0;
    double diff = h2_ - r*r;
    return poly6_coef_ * diff * diff * diff;
}

Eigen::Vector3d SPHKernel::grad_W(const Eigen::Vector3d& r_vec, double r) const {
    if (r < 1e-12 || r >= h_) return Eigen::Vector3d::Zero();
    // Gradient of spiky kernel: -45/(pi*h^6) * (h - r)^2 * (r_vec / r)
    double coeff = -spiky_coef_ * (h_ - r) * (h_ - r) / r;
    return coeff * r_vec;
}

double SPHKernel::laplacian_W(double r) const {
    if (r < 1e-12 || r >= h_) return 0.0;
    // Laplacian of viscosity kernel: 45/(pi*h^6) * (h - r)
    return visc_coef_ * (h_ - r);
}

//------------------------------------------------------------------------------
// SPHSolver implementation
//------------------------------------------------------------------------------
SPHSolver::SPHSolver(const SolverConfig& config)
    : BaseSolver("SPHSolver"), kernel_(0.1) {
    config_ = config;
}

SPHSolver::~SPHSolver() = default;

void SPHSolver::set_sph_config(const SPHConfig& config) {
    sph_config_ = config;
    kernel_ = SPHKernel(config.kernel_radius);
}

void SPHSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    rebuild_particle_list(entities);
    build_neighbor_search();
    initialized_ = true;
}

void SPHSolver::reset() {
    BaseSolver::reset();
    particles_.clear();
    particle_owner_.clear();
    entity_states_.clear();
    entity_to_index_.clear();
    boundary_planes_.clear();
    bvh_.reset();
    neighbor_cache_.clear();
    particles_dirty_ = true;
    neighbors_dirty_ = true;
}

void SPHSolver::step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);
    
    if (!initialized_ || particles_dirty_) {
        initialize(entities);
    }
    
    if (particles_.empty()) return;
    
    // CFL condition
    double max_vel = 0.0;
    for (const auto& p : particles_) {
        max_vel = std::max(max_vel, p.velocity.norm());
    }
    double dt_cfl = sph_config_.cfl_factor * kernel_.h() / (max_vel + 1e-12);
    dt = std::min(dt, dt_cfl);
    
    double sub_dt = dt / sph_config_.sub_steps;
    
    for (int sub = 0; sub < sph_config_.sub_steps; ++sub) {
        // Find neighbors
        if (neighbors_dirty_) {
            find_neighbors();
            neighbors_dirty_ = false;
        }
        
        // Compute density and pressure
        compute_density_pressure();
        
        // Compute forces
        compute_non_pressure_forces(sub_dt);
        compute_pressure_forces();
        
        // Integrate
        integrate(sub_dt);
        
        // Handle boundaries and collisions
        apply_boundary_conditions(sub_dt);
        handle_collisions(sub_dt);
        
        // Neighbors may change after positions update
        neighbors_dirty_ = true;
    }
    
    // Write back to entities
    update_entities();
    
    stats_.particle_count = particles_.size();
}

void SPHSolver::on_entity_added(std::shared_ptr<BaseEntity> entity) {
    if (std::dynamic_pointer_cast<SPHEntity>(entity)) {
        particles_dirty_ = true;
    }
}

void SPHSolver::on_entity_removed(std::shared_ptr<BaseEntity> entity) {
    auto it = entity_to_index_.find(entity->id());
    if (it != entity_to_index_.end()) {
        entity_states_.erase(entity_states_.begin() + it->second);
        particles_dirty_ = true;
    }
}

void SPHSolver::add_boundary_plane(const datatypes::Vector3& point, const datatypes::Vector3& normal) {
    BoundaryPlane bp;
    bp.point = Eigen::Vector3d(point[0], point[1], point[2]);
    bp.normal = Eigen::Vector3d(normal[0], normal[1], normal[2]).normalized();
    boundary_planes_.push_back(bp);
}

void SPHSolver::clear_boundary_planes() {
    boundary_planes_.clear();
}

void SPHSolver::rebuild_particle_list(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    particles_.clear();
    particle_owner_.clear();
    entity_states_.clear();
    entity_to_index_.clear();
    
    double total_volume = 0.0;
    
    for (const auto& e : entities) {
        auto sph_entity = std::dynamic_pointer_cast<SPHEntity>(e);
        if (!sph_entity) continue;
        
        EntitySPHState state;
        state.entity = sph_entity;
        state.particle_start = particles_.size();
        
        const auto& pos = sph_entity->particle_positions();
        const auto& vel = sph_entity->particle_velocities();
        const auto& mass = sph_entity->particle_masses();
        bool is_boundary = sph_entity->is_boundary();
        
        state.particle_count = pos.size();
        
        for (size_t i = 0; i < pos.size(); ++i) {
            SPHParticleState p;
            p.position = Eigen::Vector3d(pos[i][0], pos[i][1], pos[i][2]);
            p.velocity = Eigen::Vector3d(vel[i][0], vel[i][1], vel[i][2]);
            p.acceleration = Eigen::Vector3d::Zero();
            p.mass = mass[i];
            p.material_id = sph_entity->material_id();
            p.is_boundary = is_boundary;
            p.original_index = static_cast<uint32_t>(particles_.size());
            
            particles_.push_back(p);
            particle_owner_.push_back({sph_entity->id(), i});
            
            if (!is_boundary) {
                total_volume += p.mass / sph_config_.rest_density;
            }
        }
        
        entity_to_index_[sph_entity->id()] = entity_states_.size();
        entity_states_.push_back(state);
    }
    
    // Adjust particle masses to match target density if needed
    if (total_volume > 0) {
        double target_mass = sph_config_.rest_density * total_volume / particles_.size();
        for (auto& p : particles_) {
            if (!p.is_boundary) {
                p.mass = target_mass;
            }
        }
    }
    
    particles_dirty_ = false;
}

void SPHSolver::build_neighbor_search() {
    if (particles_.empty()) return;
    
    bvh_ = std::make_unique<PointBVH>();
    std::vector<PointBVH::Point> points;
    points.reserve(particles_.size());
    for (size_t i = 0; i < particles_.size(); ++i) {
        PointBVH::Point pt;
        pt.position = datatypes::Vector3(particles_[i].position.x(), particles_[i].position.y(), particles_[i].position.z());
        pt.radius = static_cast<float>(kernel_.support_radius());
        pt.index = static_cast<uint32_t>(i);
        points.push_back(pt);
    }
    bvh_->build(points);
    neighbors_dirty_ = true;
}

void SPHSolver::find_neighbors() {
    if (!bvh_ || particles_.empty()) return;
    
    neighbor_cache_.resize(particles_.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < particles_.size(); ++i) {
        std::vector<uint32_t> neighbors;
        datatypes::Vector3 center(particles_[i].position.x(), particles_[i].position.y(), particles_[i].position.z());
        bvh_->query_radius(center, static_cast<float>(kernel_.support_radius()), neighbors);
        
        // Filter to only keep j > i if we want symmetric pairs, but we'll keep all for simplicity
        // and limit to max_neighbors
        if (neighbors.size() > static_cast<size_t>(sph_config_.max_neighbors)) {
            std::nth_element(neighbors.begin(), neighbors.begin() + sph_config_.max_neighbors, neighbors.end(),
                [this, i](uint32_t a, uint32_t b) {
                    double da = (particles_[a].position - particles_[i].position).squaredNorm();
                    double db = (particles_[b].position - particles_[i].position).squaredNorm();
                    return da < db;
                });
            neighbors.resize(sph_config_.max_neighbors);
        }
        neighbor_cache_[i] = std::move(neighbors);
    }
}

void SPHSolver::compute_density_pressure() {
    #pragma omp parallel for
    for (size_t i = 0; i < particles_.size(); ++i) {
        if (particles_[i].is_boundary) {
            particles_[i].density = sph_config_.rest_density;
            particles_[i].pressure = 0.0;
            continue;
        }
        
        double density = particles_[i].mass * kernel_.W_normalized(0.0); // self contribution
        
        for (uint32_t j_idx : neighbor_cache_[i]) {
            if (i == j_idx) continue;
            const auto& pj = particles_[j_idx];
            double r = (particles_[i].position - pj.position).norm();
            if (r < kernel_.support_radius()) {
                density += pj.mass * kernel_.W_normalized(r);
            }
        }
        
        particles_[i].density = std::max(density, sph_config_.rest_density * 0.5);
        particles_[i].pressure = compute_pressure_from_density(particles_[i].density);
    }
}

double SPHSolver::compute_pressure_from_density(double density) const {
    // Tait equation (weakly compressible)
    double ratio = density / sph_config_.rest_density;
    return sph_config_.gas_stiffness * (std::pow(ratio, 7.0) - 1.0);
}

void SPHSolver::compute_non_pressure_forces(double dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles_.size(); ++i) {
        if (particles_[i].is_boundary) continue;
        
        Eigen::Vector3d force = Eigen::Vector3d::Zero();
        
        // Gravity
        force += Eigen::Vector3d(0.0, -9.80665, 0.0) * particles_[i].mass;
        
        // Viscosity
        if (sph_config_.enable_viscosity) {
            compute_viscosity_force(i, neighbor_cache_[i], force);
        }
        
        // Surface tension
        if (sph_config_.enable_surface_tension) {
            compute_surface_tension_force(i, neighbor_cache_[i], force);
        }
        
        // Vorticity confinement
        if (sph_config_.enable_turbulence) {
            compute_vorticity_force(i, neighbor_cache_[i], force);
        }
        
        particles_[i].acceleration = force / particles_[i].mass;
    }
}

void SPHSolver::compute_pressure_forces() {
    #pragma omp parallel for
    for (size_t i = 0; i < particles_.size(); ++i) {
        if (particles_[i].is_boundary) continue;
        
        Eigen::Vector3d pressure_force = Eigen::Vector3d::Zero();
        double rho_i = particles_[i].density;
        double p_i = particles_[i].pressure;
        
        for (uint32_t j_idx : neighbor_cache_[i]) {
            if (i == j_idx) continue;
            const auto& pj = particles_[j_idx];
            
            Eigen::Vector3d r_vec = particles_[i].position - pj.position;
            double r = r_vec.norm();
            if (r < 1e-12) continue;
            
            // Symmetric pressure force (minimizes momentum error)
            double rho_j = pj.density;
            double p_j = pj.pressure;
            
            double avg_pressure = (p_i + p_j) / (2.0 * rho_i * rho_j);
            Eigen::Vector3d grad = kernel_.grad_W(r_vec, r);
            
            pressure_force -= particles_[i].mass * pj.mass * avg_pressure * grad;
        }
        
        particles_[i].acceleration += pressure_force;
    }
}

void SPHSolver::compute_viscosity_force(size_t i, const std::vector<uint32_t>& neighbors, Eigen::Vector3d& force) {
    const auto& pi = particles_[i];
    
    for (uint32_t j_idx : neighbors) {
        if (i == j_idx) continue;
        const auto& pj = particles_[j_idx];
        if (pj.is_boundary) continue;
        
        Eigen::Vector3d r_vec = pi.position - pj.position;
        double r = r_vec.norm();
        if (r < 1e-12 || r >= kernel_.support_radius()) continue;
        
        // Standard SPH viscosity (Monaghan 2005)
        Eigen::Vector3d v_ij = pi.velocity - pj.velocity;
        double lap = kernel_.laplacian_W(r);
        
        // Artificial viscosity (for stability)
        double alpha = sph_config_.artificial_viscosity_alpha;
        double beta = sph_config_.artificial_viscosity_beta;
        double c = sph_config_.speed_of_sound;
        double rho_ij = 0.5 * (pi.density + pj.density);
        double mu = alpha * c * kernel_.h() / rho_ij;
        if (beta > 0) {
            double v_dot_r = v_ij.dot(r_vec);
            if (v_dot_r < 0) {
                mu += beta * kernel_.h() * kernel_.h() * v_dot_r * v_dot_r / (r*r + 0.01 * kernel_.h2_);
            }
        }
        
        force += pj.mass * mu * lap * v_ij;
    }
}

void SPHSolver::compute_surface_tension_force(size_t i, const std::vector<uint32_t>& neighbors, Eigen::Vector3d& force) {
    const auto& pi = particles_[i];
    
    // Compute color field gradient (surface normal)
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double color_laplacian = 0.0;
    
    for (uint32_t j_idx : neighbors) {
        if (i == j_idx) continue;
        const auto& pj = particles_[j_idx];
        if (pj.is_boundary) continue;
        
        Eigen::Vector3d r_vec = pi.position - pj.position;
        double r = r_vec.norm();
        if (r < 1e-12 || r >= kernel_.support_radius()) continue;
        
        double w = kernel_.W_normalized(r);
        normal += pj.mass / pj.density * kernel_.grad_W(r_vec, r);
        color_laplacian += pj.mass / pj.density * kernel_.laplacian_W(r);
    }
    
    double normal_len = normal.norm();
    if (normal_len > 1e-6) {
        normal /= normal_len;
        
        // Curvature from divergence of normal (approximated by color laplacian)
        double curvature = -color_laplacian;
        
        // Surface tension force: -sigma * kappa * n
        force += sph_config_.surface_tension * curvature * normal * pi.mass;
    }
}

void SPHSolver::compute_vorticity_force(size_t i, const std::vector<uint32_t>& neighbors, Eigen::Vector3d& force) {
    const auto& pi = particles_[i];
    
    // Compute vorticity omega = curl v
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    for (uint32_t j_idx : neighbors) {
        if (i == j_idx) continue;
        const auto& pj = particles_[j_idx];
        Eigen::Vector3d r_vec = pi.position - pj.position;
        double r = r_vec.norm();
        if (r < 1e-12 || r >= kernel_.support_radius()) continue;
        
        Eigen::Vector3d v_ij = pj.velocity - pi.velocity;
        omega += pj.mass / pj.density * v_ij.cross(kernel_.grad_W(r_vec, r));
    }
    
    double omega_len = omega.norm();
    if (omega_len > 1e-6) {
        // Compute normalized vorticity location vector
        Eigen::Vector3d eta = Eigen::Vector3d::Zero();
        for (uint32_t j_idx : neighbors) {
            if (i == j_idx) continue;
            const auto& pj = particles_[j_idx];
            Eigen::Vector3d r_vec = pi.position - pj.position;
            double r = r_vec.norm();
            if (r < 1e-12 || r >= kernel_.support_radius()) continue;
            
            eta += pj.mass / pj.density * omega_len * kernel_.grad_W(r_vec, r);
        }
        double eta_len = eta.norm();
        if (eta_len > 1e-6) {
            eta /= eta_len;
            force += sph_config_.vorticity_coefficient * eta.cross(omega) * pi.mass;
        }
    }
}

void SPHSolver::integrate(double dt) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles_.size(); ++i) {
        if (particles_[i].is_boundary) continue;
        
        // Symplectic Euler
        particles_[i].velocity += particles_[i].acceleration * dt;
        particles_[i].position += particles_[i].velocity * dt;
    }
}

void SPHSolver::apply_boundary_conditions(double dt) {
    if (sph_config_.boundary_method == SPHConfig::BoundaryMethod::BOUNDARY_PLANES) {
        double stiffness = sph_config_.boundary_stiffness;
        double damping = sph_config_.boundary_damping;
        
        #pragma omp parallel for
        for (size_t i = 0; i < particles_.size(); ++i) {
            if (particles_[i].is_boundary) continue;
            
            for (const auto& plane : boundary_planes_) {
                Eigen::Vector3d p = particles_[i].position;
                double dist = (p - plane.point).dot(plane.normal);
                double penetration = sph_config_.particle_radius - dist;
                
                if (penetration > 0) {
                    // Position correction
                    particles_[i].position += plane.normal * penetration;
                    
                    // Velocity response
                    double vn = particles_[i].velocity.dot(plane.normal);
                    if (vn < 0) {
                        particles_[i].velocity -= (1.0 + damping) * vn * plane.normal;
                    }
                    
                    // Friction (simplified)
                    Eigen::Vector3d vt = particles_[i].velocity - vn * plane.normal;
                    particles_[i].velocity -= vt * std::min(1.0, stiffness * dt);
                }
            }
        }
    }
    // Dummy particles method would be handled via regular SPH interaction
}

void SPHSolver::handle_collisions(double dt) {
    // Simple particle-particle collision resolution (hard sphere)
    double radius = sph_config_.particle_radius;
    double restitution = 0.5;
    
    for (size_t i = 0; i < particles_.size(); ++i) {
        if (particles_[i].is_boundary) continue;
        
        for (uint32_t j_idx : neighbor_cache_[i]) {
            if (i >= j_idx) continue; // process each pair once
            if (particles_[j_idx].is_boundary) continue;
            
            auto& pi = particles_[i];
            auto& pj = particles_[j_idx];
            
            Eigen::Vector3d r_vec = pi.position - pj.position;
            double r = r_vec.norm();
            double min_dist = 2.0 * radius;
            
            if (r < min_dist && r > 1e-12) {
                Eigen::Vector3d n = r_vec / r;
                double penetration = min_dist - r;
                
                // Mass-weighted position correction
                double w_i = pi.mass;
                double w_j = pj.mass;
                double w_sum = w_i + w_j;
                pi.position += n * (penetration * w_j / w_sum);
                pj.position -= n * (penetration * w_i / w_sum);
                
                // Velocity impulse
                Eigen::Vector3d v_rel = pi.velocity - pj.velocity;
                double vn = v_rel.dot(n);
                if (vn < 0) {
                    double impulse = -(1.0 + restitution) * vn / (1.0/w_i + 1.0/w_j);
                    pi.velocity += n * (impulse / w_i);
                    pj.velocity -= n * (impulse / w_j);
                }
            }
        }
    }
}

void SPHSolver::update_entities() {
    // Group particles by entity and update
    for (auto& state : entity_states_) {
        std::vector<datatypes::Vector3> positions, velocities;
        size_t start = state.particle_start;
        size_t count = state.particle_count;
        
        positions.reserve(count);
        velocities.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            const auto& p = particles_[start + i];
            positions.push_back(datatypes::Vector3(p.position.x(), p.position.y(), p.position.z()));
            velocities.push_back(datatypes::Vector3(p.velocity.x(), p.velocity.y(), p.velocity.z()));
        }
        
        state.entity->set_positions(positions);
        state.entity->set_velocities(velocities);
    }
}

} // namespace engine
} // namespace genesis