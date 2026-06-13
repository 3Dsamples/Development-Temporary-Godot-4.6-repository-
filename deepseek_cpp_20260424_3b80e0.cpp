// genesis/engine/solvers/mpm_solver.h

#pragma once

#include "genesis/engine/solvers/base_solver.h"
#include "genesis/engine/entities/mpm_entity.h"
#include "genesis/datatypes.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// MPM configuration
//------------------------------------------------------------------------------
struct MPMConfig {
    // Grid settings
    double cell_size = 0.1;
    int grid_resolution[3] = {32, 32, 32};
    datatypes::Vector3 grid_origin = {0.0, 0.0, 0.0};
    
    // Time integration
    enum class TransferScheme {
        FLIP,
        APIC,
        MLS_MPM
    };
    TransferScheme transfer_scheme = TransferScheme::APIC;
    double flip_pic_ratio = 0.95; // 1.0 = pure FLIP, 0.0 = pure PIC
    
    // Material parameters
    enum class MaterialModel {
        NEO_HOOKEAN,
        FIXED_COROTATED,
        VON_MISES_PLASTIC,
        DRUCKER_PRAGER,
        SAND,
        SNOW,
        LIQUID
    };
    MaterialModel material_model = MaterialModel::NEO_HOOKEAN;
    
    double youngs_modulus = 1e6;
    double poisson_ratio = 0.3;
    double density = 1000.0;
    
    // Plasticity (if applicable)
    double yield_stress = 1e6;
    double hardening_modulus = 0.0;
    double friction_angle = 30.0 * M_PI / 180.0;
    double cohesion = 0.0;
    
    // Damping
    double velocity_damping = 0.0;
    double hardening_alpha = 0.0; // for plastic hardening
    
    // Solver settings
    int num_iterations = 5;
    double cfl_factor = 0.4;
    bool implicit = false;
    bool enable_affine = true; // APIC affine matrix
    
    // Boundary conditions
    enum class BoundaryType {
        NONE,
        STICKY,
        SEPARATE,
        FRICTIONLESS
    };
    BoundaryType boundary_type[6] = {BoundaryType::STICKY, BoundaryType::STICKY,
                                     BoundaryType::STICKY, BoundaryType::STICKY,
                                     BoundaryType::STICKY, BoundaryType::STICKY};
    double friction_coefficient = 0.5;
};

//------------------------------------------------------------------------------
// MPM Grid node
//------------------------------------------------------------------------------
struct MPMGridNode {
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    double mass = 0.0;
    bool active = false;
    bool is_dirichlet = false;
    Eigen::Vector3d prescribed_velocity = Eigen::Vector3d::Zero();
};

//------------------------------------------------------------------------------
// MPM Particle state
//------------------------------------------------------------------------------
struct MPMParticleState {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Matrix3d F;               // deformation gradient
    Eigen::Matrix3d C;               // affine velocity matrix (APIC)
    Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity(); // plastic part
    double mass;
    double volume;
    double Jp = 1.0;                 // plastic Jacobian
    uint32_t material_id = 0;
};

//------------------------------------------------------------------------------
// MPM Solver class - Material Point Method for large deformation solids/fluids
//------------------------------------------------------------------------------
class MPMSolver : public BaseSolver {
public:
    explicit MPMSolver(const SolverConfig& config = SolverConfig{});
    ~MPMSolver() override;

    std::string solver_type() const override { return "MPM"; }

    // Initialization
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Main step
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;

    // Entity management
    void on_entity_added(std::shared_ptr<BaseEntity> entity) override;
    void on_entity_removed(std::shared_ptr<BaseEntity> entity) override;

    // MPM-specific configuration
    void set_mpm_config(const MPMConfig& config) { mpm_config_ = config; }
    const MPMConfig& mpm_config() const { return mpm_config_; }

    // Access to grid for visualization
    const std::vector<MPMGridNode>& grid_nodes() const { return grid_nodes_; }
    const datatypes::AABB& grid_domain() const { return grid_domain_; }

private:
    MPMConfig mpm_config_;
    
    // Grid data
    struct Grid {
        int nx, ny, nz;
        double dx, dy, dz;
        datatypes::Vector3 origin;
        std::vector<MPMGridNode> nodes;
        
        // Index helpers
        inline int idx(int i, int j, int k) const {
            return i + nx * (j + ny * k);
        }
        inline bool valid(int i, int j, int k) const {
            return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
        }
        inline Eigen::Vector3d node_position(int i, int j, int k) const {
            return Eigen::Vector3d(origin[0] + i*dx, origin[1] + j*dy, origin[2] + k*dz);
        }
    };
    Grid grid_;
    datatypes::AABB grid_domain_;
    
    // Particle data per entity
    struct EntityParticles {
        std::shared_ptr<MPMEntity> entity;
        std::vector<MPMParticleState> particles;
    };
    std::vector<EntityParticles> entity_particles_;
    std::unordered_map<uint64_t, size_t> entity_to_index_;
    bool particles_dirty_ = true;

    // Temporary grid arrays
    std::vector<MPMGridNode> grid_nodes_; // alias for grid_.nodes after resize
    
    // Internal methods
    void rebuild_particle_list(const std::vector<std::shared_ptr<BaseEntity>>& entities);
    void setup_grid(const std::vector<datatypes::Vector3>& particle_positions);
    void reset_grid();
    void particle_to_grid(double dt);
    void update_grid_velocities(double dt);
    void grid_to_particle(double dt);
    void apply_boundary_conditions();
    
    // Material model specific methods
    Eigen::Matrix3d compute_pk1_stress(const MPMParticleState& p, double& plastic_hardening) const;
    Eigen::Matrix3d compute_stress_derivative(const MPMParticleState& p) const;
    
    // Plasticity updates
    void apply_plasticity(MPMParticleState& p) const;
    
    // B-spline weight functions
    static double bspline_weight(double x);
    static double bspline_weight_derivative(double x);
    static void interpolate_weights(const Eigen::Vector3d& xp, const Grid& grid,
                                    std::array<int, 3>& base_idx,
                                    std::array<double, 3>& wx,
                                    std::array<double, 3>& dwx);
};

} // namespace engine
} // namespace genesis