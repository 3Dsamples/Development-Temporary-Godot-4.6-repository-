// genesis/engine/solvers/sph_solver.h

#pragma once

#include "genesis/engine/solvers/base_solver.h"
#include "genesis/engine/entities/sph_entity.h"
#include "genesis/engine/bvh.h"
#include "genesis/datatypes.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// SPH Configuration
//------------------------------------------------------------------------------
struct SPHConfig {
    // Particle properties
    double particle_radius = 0.05;
    double rest_density = 1000.0;       // kg/m³
    double particle_mass = 1.0;          // will be computed from density
    
    // Kernel properties
    double kernel_radius = 0.1;          // support radius (usually 2*particle_radius)
    
    // Fluid parameters
    double viscosity = 0.01;             // dynamic viscosity
    double surface_tension = 0.072;      // coefficient (N/m)
    double gas_stiffness = 1000.0;       // for pressure (Tait equation)
    double speed_of_sound = 20.0;        // for weakly compressible SPH
    
    // Time stepping
    double cfl_factor = 0.4;
    int sub_steps = 1;
    
    // Solver options
    bool enable_viscosity = true;
    bool enable_surface_tension = true;
    bool enable_turbulence = false;
    double vorticity_coefficient = 0.01;
    
    // Boundary handling
    enum class BoundaryMethod {
        DUMMY_PARTICLES,
        BOUNDARY_PLANES,
        SDF
    };
    BoundaryMethod boundary_method = BoundaryMethod::DUMMY_PARTICLES;
    double boundary_stiffness = 10000.0;
    double boundary_damping = 0.5;
    
    // Neighbor search
    int max_neighbors = 128;
    
    // Artificial viscosity (for stability)
    double artificial_viscosity_alpha = 0.1;
    double artificial_viscosity_beta = 0.0;
};

//------------------------------------------------------------------------------
// SPH Particle state
//------------------------------------------------------------------------------
struct SPHParticleState {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    double density = 0.0;
    double pressure = 0.0;
    double mass = 0.0;
    uint32_t material_id = 0;
    bool is_boundary = false;
    
    // Temporary for sorting
    uint32_t cell_id = 0;
    uint32_t original_index = 0;
};

//------------------------------------------------------------------------------
// SPH Kernel functions (cubic spline)
//------------------------------------------------------------------------------
class SPHKernel {
public:
    explicit SPHKernel(double h);
    
    double W(double r) const;                    // kernel value
    double W_normalized(double r) const;         // for density interpolation
    Eigen::Vector3d grad_W(const Eigen::Vector3d& r_vec, double r) const;
    double laplacian_W(double r) const;
    
    double support_radius() const { return h_; }
    double h() const { return h_; }
    
    // Precomputed constants
    double poly6_coef() const { return poly6_coef_; }
    double spiky_coef() const { return spiky_coef_; }
    double visc_coef() const { return visc_coef_; }
    
private:
    double h_;
    double h2_;
    double h3_;
    double poly6_coef_;   // for density
    double spiky_coef_;   // for pressure gradient
    double visc_coef_;    // for viscosity laplacian
};

//------------------------------------------------------------------------------
// SPH Solver class - Smoothed Particle Hydrodynamics for fluid simulation
//------------------------------------------------------------------------------
class SPHSolver : public BaseSolver {
public:
    explicit SPHSolver(const SolverConfig& config = SolverConfig{});
    ~SPHSolver() override;

    std::string solver_type() const override { return "SPH"; }

    // Initialization
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Main step
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;

    // Entity management
    void on_entity_added(std::shared_ptr<BaseEntity> entity) override;
    void on_entity_removed(std::shared_ptr<BaseEntity> entity) override;

    // SPH-specific configuration
    void set_sph_config(const SPHConfig& config);
    const SPHConfig& sph_config() const { return sph_config_; }
    
    // Access to particles for visualization
    const std::vector<SPHParticleState>& particles() const { return particles_; }
    
    // Add boundary planes (for BoundaryMethod::BOUNDARY_PLANES)
    void add_boundary_plane(const datatypes::Vector3& point, const datatypes::Vector3& normal);
    void clear_boundary_planes();

private:
    SPHConfig sph_config_;
    SPHKernel kernel_;
    
    // All particles (fluid + boundary)
    std::vector<SPHParticleState> particles_;
    
    // Mapping from global particle index to entity and local index
    struct ParticleOwner {
        uint64_t entity_id;
        size_t local_index;
    };
    std::vector<ParticleOwner> particle_owner_;
    
    // Entity states
    struct EntitySPHState {
        std::shared_ptr<SPHEntity> entity;
        size_t particle_start;
        size_t particle_count;
    };
    std::vector<EntitySPHState> entity_states_;
    std::unordered_map<uint64_t, size_t> entity_to_index_;
    bool particles_dirty_ = true;
    
    // Boundary planes (for simple boundaries)
    struct BoundaryPlane {
        Eigen::Vector3d point;
        Eigen::Vector3d normal;
    };
    std::vector<BoundaryPlane> boundary_planes_;
    
    // Neighbor search acceleration
    std::unique_ptr<PointBVH> bvh_;
    std::vector<std::vector<uint32_t>> neighbor_cache_;
    bool neighbors_dirty_ = true;
    
    // Internal methods
    void rebuild_particle_list(const std::vector<std::shared_ptr<BaseEntity>>& entities);
    void build_neighbor_search();
    void find_neighbors();
    void compute_density_pressure();
    void compute_non_pressure_forces(double dt);
    void compute_pressure_forces();
    void integrate(double dt);
    void apply_boundary_conditions(double dt);
    void handle_collisions(double dt);
    void update_entities();
    
    // SPH specific force terms
    void compute_viscosity_force(size_t i, const std::vector<uint32_t>& neighbors, Eigen::Vector3d& force);
    void compute_surface_tension_force(size_t i, const std::vector<uint32_t>& neighbors, Eigen::Vector3d& force);
    void compute_vorticity_force(size_t i, const std::vector<uint32_t>& neighbors, Eigen::Vector3d& force);
    
    // Equation of state
    double compute_pressure_from_density(double density) const;
};

} // namespace engine
} // namespace genesis