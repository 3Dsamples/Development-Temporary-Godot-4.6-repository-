// genesis/engine/entities/sf_entity.cpp

#include "genesis/engine/entities/sf_entity.h"       // Include corresponding header
#include "genesis/engine/mesh.h"                     // For mesh SDF computation
#include <algorithm>                                 // std::copy, std::fill, std::min, std::max
#include <cmath>                                     // std::abs, std::sqrt, std::pow
#include <sstream>                                   // std::ostringstream
#include <queue>                                     // std::priority_queue for fast marching
#include <limits>                                    // std::numeric_limits
#include <cstring>                                   // std::memset

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// SFEntity construction
//------------------------------------------------------------------------------
SFEntity::SFEntity()
    : BaseEntity("SFEntity")                         // Call base constructor with default name
{
    // Empty constructor body (grid uninitialized)
}

SFEntity::SFEntity(const std::string& name)
    : BaseEntity(name)                               // Base constructor with custom name
{
    // Empty constructor body
}

SFEntity::~SFEntity() {
    // Virtual destructor (vectors automatically cleaned up)
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void SFEntity::set_sf_config(const SFConfig& config) {
    // Store fluid configuration
    sf_config_ = config;                             // Copy config struct
}

//------------------------------------------------------------------------------
// Level set grid access
//------------------------------------------------------------------------------
void SFEntity::set_grid(int nx, int ny, int nz, const datatypes::Vector3& origin, double cell_size) {
    // Define the simulation grid dimensions and parameters
    nx_ = nx;                                        // Store grid X dimension
    ny_ = ny;                                        // Store grid Y dimension
    nz_ = nz;                                        // Store grid Z dimension
    origin_ = origin;                                // Store world-space origin
    dx_ = cell_size;                                 // Store uniform cell size
    resize_grid();                                   // Allocate all grid arrays
}

void SFEntity::set_level_set(const std::vector<double>& phi) {
    // Replace entire level set array
    if (phi.size() == phi_.size()) {
        phi_ = phi;                                  // Copy entire vector
    } else {
        size_t n = std::min(phi.size(), phi_.size());
        std::copy(phi.begin(), phi.begin() + n, phi_.begin());
    }
}

double SFEntity::level_set(int i, int j, int k) const {
    // Get signed distance value at cell (i,j,k)
    if (valid_cell(i, j, k)) {
        return phi_[cell_index(i, j, k)];            // Return stored value
    }
    return std::numeric_limits<double>::max();       // Out of bounds: large positive
}

void SFEntity::set_level_set(int i, int j, int k, double value) {
    // Set signed distance value at a specific cell
    if (valid_cell(i, j, k)) {
        phi_[cell_index(i, j, k)] = value;           // Update value
    }
}

//------------------------------------------------------------------------------
// Initialization from geometry
//------------------------------------------------------------------------------
void SFEntity::init_from_mesh(const Mesh& mesh) {
    // Compute signed distance field from a triangle mesh
    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) return;    // Grid must be defined first
    compute_sdf_from_mesh(mesh);                     // Compute SDF (brute force or fast marching)
    // Mark fluid cells (negative SDF)
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                int idx = cell_index(i, j, k);
                flags_[idx] = (phi_[idx] < 0.0) ? 1 : 0; // 1 = fluid, 0 = empty
            }
        }
    }
    // Seed particles if hybrid method enabled
    if (sf_config_.use_particles) {
        seed_particles();                            // Populate interior with particles
    }
}

void SFEntity::init_from_box(const datatypes::AABB& box) {
    // Initialize level set as signed distance to axis-aligned box
    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) return;
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                // Compute cell center world position
                datatypes::Vector3 center = origin_ + datatypes::Vector3(
                    (i + 0.5) * dx_, (j + 0.5) * dx_, (k + 0.5) * dx_);
                // Compute signed distance to box
                datatypes::Vector3 d_min = center - box.min;
                datatypes::Vector3 d_max = box.max - center;
                double dx = std::max({d_min[0], d_min[1], d_min[2]});
                double dy = std::max({d_max[0], d_max[1], d_max[2]});
                double dist = std::max(dx, dy);
                // If inside, distance is negative
                if (box.contains(center)) dist = -dist;
                phi_[cell_index(i, j, k)] = dist;
                flags_[cell_index(i, j, k)] = (dist < 0.0) ? 1 : 0;
            }
        }
    }
    reinitialize_level_set();                        // Ensure proper signed distance property
    if (sf_config_.use_particles) seed_particles();
}

void SFEntity::init_from_sphere(const datatypes::Vector3& center, double radius) {
    // Initialize level set as signed distance to sphere
    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) return;
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                datatypes::Vector3 cell_center = origin_ + datatypes::Vector3(
                    (i + 0.5) * dx_, (j + 0.5) * dx_, (k + 0.5) * dx_);
                double dist = (cell_center - center).norm() - radius;
                phi_[cell_index(i, j, k)] = dist;
                flags_[cell_index(i, j, k)] = (dist < 0.0) ? 1 : 0;
            }
        }
    }
    reinitialize_level_set();
    if (sf_config_.use_particles) seed_particles();
}

void SFEntity::add_solid_obstacle(const Mesh& mesh) {
    // Mark cells inside mesh as solid obstacles
    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) return;
    // Compute SDF for the solid mesh and union with existing solid flags
    std::vector<double> solid_phi(phi_.size(), std::numeric_limits<double>::max());
    // Brute force SDF (simplified: use mesh's closest point query)
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                datatypes::Vector3 p = origin_ + datatypes::Vector3(
                    (i + 0.5) * dx_, (j + 0.5) * dx_, (k + 0.5) * dx_);
                auto cp = mesh.closest_point(p);
                double d = std::sqrt(cp.distance_sq);
                // Determine sign using normal (simplified: assume closed mesh)
                datatypes::Vector3 dir = p - cp.point;
                double sign = (dir.dot(cp.normal) > 0) ? 1.0 : -1.0;
                solid_phi[cell_index(i, j, k)] = sign * d;
            }
        }
    }
    // Update flags: solid if inside mesh (negative SDF) and not already solid
    for (size_t idx = 0; idx < flags_.size(); ++idx) {
        if (solid_phi[idx] < 0.0) {
            flags_[idx] = 2;                         // Mark as solid
            phi_[idx] = std::min(phi_[idx], 0.0);    // Ensure fluid level set doesn't penetrate solid
        }
    }
}

void SFEntity::add_solid_box(const datatypes::AABB& box) {
    // Mark cells inside box as solid
    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) return;
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                datatypes::Vector3 p = origin_ + datatypes::Vector3(
                    (i + 0.5) * dx_, (j + 0.5) * dx_, (k + 0.5) * dx_);
                if (box.contains(p)) {
                    flags_[cell_index(i, j, k)] = 2; // Solid
                    phi_[cell_index(i, j, k)] = std::min(phi_[cell_index(i, j, k)], 0.0);
                }
            }
        }
    }
}

void SFEntity::clear_solids() {
    // Reset solid flags to fluid or empty based on level set
    for (size_t idx = 0; idx < flags_.size(); ++idx) {
        if (flags_[idx] == 2) {
            flags_[idx] = (phi_[idx] < 0.0) ? 1 : 0;
        }
    }
}

//------------------------------------------------------------------------------
// Particle management (hybrid particle-level set)
//------------------------------------------------------------------------------
void SFEntity::add_particles(const std::vector<datatypes::Vector3>& positions,
                             const std::vector<datatypes::Vector3>& velocities,
                             const std::vector<double>& masses) {
    // Add correction particles
    size_t old_size = particle_positions_.size();
    size_t add_count = positions.size();
    particle_positions_.resize(old_size + add_count);
    particle_velocities_.resize(old_size + add_count);
    particle_masses_.resize(old_size + add_count);
    std::copy(positions.begin(), positions.end(), particle_positions_.begin() + old_size);
    std::copy(velocities.begin(), velocities.end(), particle_velocities_.begin() + old_size);
    std::copy(masses.begin(), masses.end(), particle_masses_.begin() + old_size);
}

void SFEntity::remove_particles(const std::vector<bool>& mask) {
    // Remove particles where mask is true
    if (mask.size() != particle_positions_.size()) return;
    size_t write_idx = 0;
    for (size_t i = 0; i < particle_positions_.size(); ++i) {
        if (!mask[i]) {
            if (write_idx != i) {
                particle_positions_[write_idx] = particle_positions_[i];
                particle_velocities_[write_idx] = particle_velocities_[i];
                particle_masses_[write_idx] = particle_masses_[i];
            }
            ++write_idx;
        }
    }
    particle_positions_.resize(write_idx);
    particle_velocities_.resize(write_idx);
    particle_masses_.resize(write_idx);
}

void SFEntity::clear_particles() {
    // Remove all particles
    particle_positions_.clear();
    particle_velocities_.clear();
    particle_masses_.clear();
}

void SFEntity::seed_particles() {
    // Seed particles uniformly inside fluid cells (negative level set)
    if (!sf_config_.use_particles) return;
    clear_particles();
    int ppc = sf_config_.particles_per_cell;
    double mass_per_particle = sf_config_.density * std::pow(dx_, 3) / ppc;
    
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                int idx = cell_index(i, j, k);
                if (phi_[idx] >= 0.0) continue;      // Not inside fluid
                // Seed ppc particles per cell with jittered positions
                for (int p = 0; p < ppc; ++p) {
                    double jitter_x = (static_cast<double>(p % 2) + 0.5) / 2.0;
                    double jitter_y = (static_cast<double>((p / 2) % 2) + 0.5) / 2.0;
                    double jitter_z = (static_cast<double>(p / 4) + 0.5) / 2.0;
                    datatypes::Vector3 pos = origin_ + datatypes::Vector3(
                        (i + jitter_x) * dx_,
                        (j + jitter_y) * dx_,
                        (k + jitter_z) * dx_);
                    particle_positions_.push_back(pos);
                    particle_velocities_.push_back(datatypes::Vector3(0.0));
                    particle_masses_.push_back(mass_per_particle);
                }
            }
        }
    }
}

void SFEntity::set_particle_states(const std::vector<datatypes::Vector3>& positions,
                                   const std::vector<datatypes::Vector3>& velocities) {
    // Update particle positions and velocities (used by solver)
    if (positions.size() == particle_positions_.size()) {
        particle_positions_ = positions;
    } else {
        size_t n = std::min(positions.size(), particle_positions_.size());
        std::copy(positions.begin(), positions.begin() + n, particle_positions_.begin());
    }
    if (velocities.size() == particle_velocities_.size()) {
        particle_velocities_ = velocities;
    } else {
        size_t n = std::min(velocities.size(), particle_velocities_.size());
        std::copy(velocities.begin(), velocities.begin() + n, particle_velocities_.begin());
    }
}

//------------------------------------------------------------------------------
// Overrides from BaseEntity
//------------------------------------------------------------------------------
void SFEntity::integrate(double dt) {
    // SF integration is handled by SFSolver
    (void)dt;                                        // Suppress unused warning
    // No-op: solver directly accesses grid and particle data
}

void SFEntity::reset() {
    // Reset to initial state (re-initialize from original geometry if stored)
    BaseEntity::reset();
    // Clear velocity and pressure fields
    std::fill(u_.begin(), u_.end(), 0.0);
    std::fill(v_.begin(), v_.end(), 0.0);
    std::fill(w_.begin(), w_.end(), 0.0);
    std::fill(pressure_.begin(), pressure_.end(), 0.0);
    // Re-seed particles if hybrid
    if (sf_config_.use_particles) {
        seed_particles();
    }
}

datatypes::AABB SFEntity::world_aabb() const {
    // Return bounding box of the grid
    datatypes::AABB aabb;
    aabb.min = origin_;
    aabb.max = origin_ + datatypes::Vector3(nx_ * dx_, ny_ * dx_, nz_ * dx_);
    return aabb;
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string SFEntity::repr() const {
    std::ostringstream oss;
    oss << "SFEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", grid=" << nx_ << "x" << ny_ << "x" << nz_
        << ", dx=" << dx_
        << ", particles=" << particle_count()
        << ")";
    return oss.str();
}

std::string SFEntity::str() const {
    return name() + " (SF fluid: " + std::to_string(nx_) + "x" + std::to_string(ny_) + "x" + std::to_string(nz_) + ")";
}

//------------------------------------------------------------------------------
// Private helper methods
//------------------------------------------------------------------------------
void SFEntity::resize_grid() {
    // Allocate all grid arrays based on current dimensions
    size_t cell_count = static_cast<size_t>(nx_) * ny_ * nz_;
    size_t u_count = static_cast<size_t>(nx_ + 1) * ny_ * nz_;
    size_t v_count = static_cast<size_t>(nx_) * (ny_ + 1) * nz_;
    size_t w_count = static_cast<size_t>(nx_) * ny_ * (nz_ + 1);
    
    phi_.assign(cell_count, dx_ * 10.0);             // Initialize to large positive (outside)
    pressure_.assign(cell_count, 0.0);
    flags_.assign(cell_count, 0);                    // 0 = empty
    u_.assign(u_count, 0.0);
    v_.assign(v_count, 0.0);
    w_.assign(w_count, 0.0);
}

void SFEntity::compute_sdf_from_mesh(const Mesh& mesh) {
    // Brute force signed distance field computation using mesh closest point
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                datatypes::Vector3 p = origin_ + datatypes::Vector3(
                    (i + 0.5) * dx_, (j + 0.5) * dx_, (k + 0.5) * dx_);
                auto cp = mesh.closest_point(p);
                double dist = std::sqrt(cp.distance_sq);
                // Determine sign using normal (assumes closed mesh)
                datatypes::Vector3 dir = p - cp.point;
                double sign = (dir.dot(cp.normal) > 0) ? 1.0 : -1.0;
                phi_[cell_index(i, j, k)] = sign * dist;
            }
        }
    }
    reinitialize_level_set();                        // Improve SDF quality
}

void SFEntity::reinitialize_level_set() {
    // Fast marching method to reinitialize signed distance function
    // Simplified: only correct cells near interface (band)
    const double band_width = 3.0 * dx_;
    std::vector<double> new_phi = phi_;
    
    for (int iter = 0; iter < sf_config_.reinitialization_steps; ++iter) {
        for (int k = 1; k < nz_-1; ++k) {
            for (int j = 1; j < ny_-1; ++j) {
                for (int i = 1; i < nx_-1; ++i) {
                    int idx = cell_index(i, j, k);
                    if (std::abs(phi_[idx]) > band_width) continue;
                    
                    // Compute gradient magnitude using Godunov scheme
                    double Dx_plus = (phi_[cell_index(i+1, j, k)] - phi_[idx]) / dx_;
                    double Dx_minus = (phi_[idx] - phi_[cell_index(i-1, j, k)]) / dx_;
                    double Dy_plus = (phi_[cell_index(i, j+1, k)] - phi_[idx]) / dx_;
                    double Dy_minus = (phi_[idx] - phi_[cell_index(i, j-1, k)]) / dx_;
                    double Dz_plus = (phi_[cell_index(i, j, k+1)] - phi_[idx]) / dx_;
                    double Dz_minus = (phi_[idx] - phi_[cell_index(i, j, k-1)]) / dx_;
                    
                    double sign = (phi_[idx] > 0) ? 1.0 : -1.0;
                    double a = std::max({sign * Dx_plus, -sign * Dx_minus, 0.0});
                    double b = std::max({sign * Dy_plus, -sign * Dy_minus, 0.0});
                    double c = std::max({sign * Dz_plus, -sign * Dz_minus, 0.0});
                    double grad_norm = std::sqrt(a*a + b*b + c*c);
                    
                    // Evolve toward |∇φ| = 1
                    double dt = 0.5 * dx_;
                    new_phi[idx] = phi_[idx] - dt * sign * (grad_norm - 1.0);
                }
            }
        }
        phi_.swap(new_phi);
        new_phi = phi_;
    }
}

} // namespace engine
} // namespace genesis