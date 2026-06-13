// genesis/engine/solvers/sf_solver.h

#pragma once

#include "genesis/engine/solvers/base_solver.h"
#include "genesis/engine/entities/sf_entity.h"
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
// Signed Distance Field (SDF) Solver configuration
// Used for fluid simulation and soft body interactions with SDF representation.
//------------------------------------------------------------------------------
struct SFConfig {
    // Grid settings
    double cell_size = 0.05;
    int grid_resolution[3] = {64, 64, 64};
    datatypes::Vector3 grid_origin = {0.0, 0.0, 0.0};
    
    // Fluid properties
    double density = 1000.0;           // kg/m³
    double viscosity = 0.01;           // dynamic viscosity
    double surface_tension = 0.072;    // N/m (water)
    double pressure_stiffness = 100.0; // for weakly compressible SPH-like pressure
    
    // Time integration
    double cfl_factor = 0.4;
    int pressure_iterations = 10;
    double pressure_tolerance = 1e-4;
    
    // Advection scheme
    enum class AdvectionScheme {
        SEMI_LAGRANGIAN,
        MAC_CORMACK,
        BFECC
    };
    AdvectionScheme advection = AdvectionScheme::SEMI_LAGRANGIAN;
    
    // Boundary conditions
    enum class BoundaryType {
        NONE,
        SOLID,
        INFLOW,
        OUTFLOW
    };
    BoundaryType boundary_type[6] = {BoundaryType::SOLID, BoundaryType::SOLID,
                                     BoundaryType::SOLID, BoundaryType::SOLID,
                                     BoundaryType::SOLID, BoundaryType::SOLID};
    datatypes::Vector3 inflow_velocity = {0.0, 0.0, 0.0};
    
    // Solid coupling
    bool enable_solid_coupling = true;
    double solid_friction = 0.3;
    
    // Particle sampling for SDF from mesh
    int samples_per_cell = 8;
};

//------------------------------------------------------------------------------
// SDF Grid data structures (staggered grid for MAC method)
//------------------------------------------------------------------------------
struct SFGrid {
    int nx, ny, nz;
    double dx, dy, dz;
    datatypes::Vector3 origin;
    
    // Staggered velocity components (MAC grid)
    std::vector<double> u; // x-velocity at (i+0.5, j, k)
    std::vector<double> v; // y-velocity at (i, j+0.5, k)
    std::vector<double> w; // z-velocity at (i, j, k+0.5)
    
    // Cell-centered quantities
    std::vector<double> pressure;
    std::vector<double> level_set;      // signed distance
    std::vector<bool> solid;            // solid cell flag
    std::vector<bool> fluid;            // fluid cell flag
    std::vector<double> density;
    std::vector<double> temperature;
    
    // Index helpers
    inline int idx_cell(int i, int j, int k) const {
        return i + nx * (j + ny * k);
    }
    inline int idx_u(int i, int j, int k) const {
        return i + (nx+1) * (j + ny * k);
    }
    inline int idx_v(int i, int j, int k) const {
        return i + nx * (j + (ny+1) * k);
    }
    inline int idx_w(int i, int j, int k) const {
        return i + nx * (j + ny * (k+1));
    }
    inline bool valid_cell(int i, int j, int k) const {
        return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
    }
    inline bool valid_u(int i, int j, int k) const {
        return i >= 0 && i <= nx && j >= 0 && j < ny && k >= 0 && k < nz;
    }
    inline bool valid_v(int i, int j, int k) const {
        return i >= 0 && i < nx && j >= 0 && j <= ny && k >= 0 && k < nz;
    }
    inline bool valid_w(int i, int j, int k) const {
        return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k <= nz;
    }
};

//------------------------------------------------------------------------------
// Particle representation for SDF fluids (optional, for hybrid particle-grid)
//------------------------------------------------------------------------------
struct SFParticle {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    double mass;
    double radius;
    uint32_t material_id;
};

//------------------------------------------------------------------------------
// SF Solver class - Signed Distance Field based fluid simulation.
// Uses level set method with MAC grid for incompressible Navier-Stokes.
//------------------------------------------------------------------------------
class SFSolver : public BaseSolver {
public:
    explicit SFSolver(const SolverConfig& config = SolverConfig{});
    ~SFSolver() override;

    std::string solver_type() const override { return "SF"; }

    // Initialization
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Main step
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;

    // Entity management
    void on_entity_added(std::shared_ptr<BaseEntity> entity) override;
    void on_entity_removed(std::shared_ptr<BaseEntity> entity) override;

    // SF-specific configuration
    void set_sf_config(const SFConfig& config) { sf_config_ = config; }
    const SFConfig& sf_config() const { return sf_config_; }

    // Access to grid for visualization
    const SFGrid& grid() const { return grid_; }
    
    // Build signed distance field from a mesh
    void build_sdf_from_mesh(const Mesh& mesh, int entity_id);
    
    // Add/remove solid obstacles
    void add_solid_obstacle(const datatypes::AABB& aabb);
    void clear_solid_obstacles();

private:
    SFConfig sf_config_;
    SFGrid grid_;
    
    // Entity data
    struct EntitySFState {
        std::shared_ptr<SFEntity> entity;
        std::vector<SFParticle> particles; // if using hybrid representation
        int level_set_id;                  // index into level set fields
    };
    std::vector<EntitySFState> entity_states_;
    std::unordered_map<uint64_t, size_t> entity_to_index_;
    bool states_dirty_ = true;
    
    // Solid obstacles (axis-aligned for simplicity)
    std::vector<datatypes::AABB> solid_obstacles_;
    
    // Temporary arrays for linear solvers
    Eigen::VectorXd rhs_;
    Eigen::VectorXd solution_;
    Eigen::SparseMatrix<double> A_;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> pressure_solver_;
    
    // Internal methods
    void rebuild_states(const std::vector<std::shared_ptr<BaseEntity>>& entities);
    void setup_grid(const datatypes::AABB& domain);
    void advect_velocity(double dt);
    void apply_external_forces(double dt);
    void compute_pressure(double dt);
    void apply_pressure_gradient(double dt);
    void extrapolate_velocity();
    void advect_level_set(double dt);
    void reinitialize_level_set();
    void apply_boundary_conditions();
    void update_entity_states();
    
    // Advection helpers
    double sample_velocity_u(double x, double y, double z) const;
    double sample_velocity_v(double x, double y, double z) const;
    double sample_velocity_w(double x, double y, double z) const;
    Eigen::Vector3d sample_velocity(const Eigen::Vector3d& pos) const;
    double sample_level_set(const Eigen::Vector3d& pos) const;
    Eigen::Vector3d trace_rk3(const Eigen::Vector3d& pos, double dt) const;
    
    // Level set helpers
    void compute_level_set_normals(std::vector<Eigen::Vector3d>& normals) const;
    double compute_curvature(int i, int j, int k) const;
    
    // Linear system assembly for pressure Poisson equation
    void assemble_pressure_system(double dt);
};

} // namespace engine
} // namespace genesis