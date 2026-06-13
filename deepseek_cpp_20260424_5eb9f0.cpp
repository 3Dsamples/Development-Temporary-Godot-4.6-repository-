// genesis/engine/solvers/fem_solver.h

#pragma once

#include "genesis/engine/solvers/base_solver.h"
#include "genesis/datatypes.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace genesis {
namespace engine {

// Forward declarations
class FEMEntity;
class Mesh;

//------------------------------------------------------------------------------
// FEM Solver configuration extensions
//------------------------------------------------------------------------------
struct FEMConfig {
    // Material properties
    double youngs_modulus = 1e6;      // Pa
    double poisson_ratio = 0.3;
    double density = 1000.0;          // kg/m^3
    double rayleigh_damping_alpha = 0.0;
    double rayleigh_damping_beta = 0.0;
    
    // Time integration
    enum class IntegrationMethod {
        IMPLICIT_EULER,
        NEWMARK_BETA,
        BDF2,
        QUASI_STATIC
    };
    IntegrationMethod integration = IntegrationMethod::IMPLICIT_EULER;
    double newmark_beta = 0.25;
    double newmark_gamma = 0.5;
    
    // Solver settings
    int max_linear_iterations = 100;
    double linear_tolerance = 1e-6;
    int max_newton_iterations = 10;
    double newton_tolerance = 1e-6;
    
    // Element type
    enum class ElementType {
        TET4,   // Linear tetrahedron
        TET10,  // Quadratic tetrahedron
        HEX8,   // Linear hexahedron
        HEX20   // Quadratic hexahedron
    };
    ElementType element_type = ElementType::TET4;
    
    // Plasticity
    bool enable_plasticity = false;
    double yield_stress = 1e6;
    double hardening_modulus = 0.0;
};

//------------------------------------------------------------------------------
// FEM Solver class - solves continuum mechanics using finite element method.
// Supports implicit integration with large deformations (co-rotational or
// hyperelastic materials).
//------------------------------------------------------------------------------
class FEMSolver : public BaseSolver {
public:
    explicit FEMSolver(const SolverConfig& config = SolverConfig{});
    ~FEMSolver() override;

    std::string solver_type() const override { return "FEM"; }

    // Initialization
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Main step
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;

    // Entity management
    void on_entity_added(std::shared_ptr<BaseEntity> entity) override;
    void on_entity_removed(std::shared_ptr<BaseEntity> entity) override;

    // FEM-specific configuration
    void set_fem_config(const FEMConfig& config) { fem_config_ = config; }
    const FEMConfig& fem_config() const { return fem_config_; }

    // Access to internal state (for debugging)
    const Eigen::SparseMatrix<double>& stiffness_matrix() const { return K_; }
    const Eigen::VectorXd& internal_forces() const { return f_int_; }

private:
    FEMConfig fem_config_;
    
    // Per-entity FEM state
    struct FEMState {
        std::shared_ptr<FEMEntity> entity;
        std::vector<datatypes::Vector3> rest_positions;
        std::vector<std::array<int, 4>> tetrahedra;  // for TET4
        std::vector<std::array<int, 8>> hexahedra;   // for HEX8
        std::vector<Eigen::Matrix3d> deformation_gradients;
        std::vector<Eigen::Matrix3d> plastic_strain;
        
        // System matrices for this entity (we may assemble global system)
        std::vector<Eigen::Triplet<double>> stiffness_triplets;
        Eigen::VectorXd forces;
        Eigen::VectorXd displacements;
    };
    
    std::vector<FEMState> fem_states_;
    std::unordered_map<uint64_t, size_t> entity_to_state_index_;
    bool states_dirty_ = true;

    // Global assembled system (for monolithic solve, optional)
    Eigen::SparseMatrix<double> K_;
    Eigen::VectorXd f_ext_;
    Eigen::VectorXd f_int_;
    Eigen::VectorXd u_;
    Eigen::VectorXd v_;
    Eigen::VectorXd a_;
    Eigen::VectorXd m_;   // lumped mass

    // Solver linear algebra
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> linear_solver_;
    
    // Internal methods
    void rebuild_states(const std::vector<std::shared_ptr<BaseEntity>>& entities);
    void assemble_system(double dt);
    void solve_linear_system(double dt);
    void update_entities();
    
    // Element-level computations
    void compute_tet4_stiffness(const FEMState& state, size_t elem_idx,
                                std::vector<Eigen::Triplet<double>>& triplets,
                                Eigen::VectorXd& forces);
    void compute_hex8_stiffness(const FEMState& state, size_t elem_idx,
                                std::vector<Eigen::Triplet<double>>& triplets,
                                Eigen::VectorXd& forces);
    
    // Material models
    Eigen::Matrix3d compute_pk1_stress(const Eigen::Matrix3d& F,
                                       const Eigen::Matrix3d& F_plastic = Eigen::Matrix3d::Identity()) const;
    Eigen::Matrix3d compute_stiffness_tensor(const Eigen::Matrix3d& F) const;
    
    // Plasticity
    void apply_plasticity(Eigen::Matrix3d& F, Eigen::Matrix3d& Fp) const;
    
    // Helper for shape functions and derivatives
    static void tet4_shape_functions(const Eigen::Vector4d& xi,
                                     Eigen::Matrix<double, 3, 4>& dN_dX,
                                     const std::array<Eigen::Vector3d, 4>& X);
};

} // namespace engine
} // namespace genesis