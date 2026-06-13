// genesis/engine/solvers/soft_tissue_solver.h
#pragma once

//------------------------------------------------------------------------------
// SoftTissueSolver – extends BaseSolver to handle soft tissue entities.
// Applies viscoelastic stress, passive restoration, active muscle contraction,
// bending resistance, and plasticity.  Works with SoftTissueEntity or any
// FEMEntity that exposes the required state.
//------------------------------------------------------------------------------

#include "genesis/engine/solvers/base_solver.h"     // Base solver interface
#include "genesis/engine/entities/fem_entity.h"     // FEMEntity for node/element access
#include "genesis/datatypes.h"                     // Vector3, Matrix3r
#include <vector>                                  // std::vector
#include <memory>                                  // std::shared_ptr

namespace genesis {
namespace engine {

// Forward declarations
class SoftTissueEntity;

//------------------------------------------------------------------------------
// SoftTissue solver configuration
//------------------------------------------------------------------------------
struct SoftTissueSolverConfig {
    // Global material properties (used if entity doesn't have its own config)
    double youngs_modulus = 5e4;           // Pa
    double poisson_ratio = 0.45;
    double viscosity = 500.0;             // Pa·s
    double relaxation_time = 0.1;         // s (Maxwell)
    double restoration_stiffness = 200.0;  // N/m
    double restoration_damping = 10.0;    // N·s/m
    double bending_stiffness = 100.0;     // N·m/rad
    double yield_stress = 2e4;            // Pa
    double plastic_hardening = 5e3;       // Pa
    
    // Integration parameters
    int substeps = 1;                     // internal substeps for stability
    bool enable_viscous_stress = true;    // apply Maxwell viscoelastic model
    bool enable_restoration = true;       // passive return to rest shape
    bool enable_bending = true;           // additional bending term
    bool enable_plasticity = false;       // permanent deformation when stressed
};

//------------------------------------------------------------------------------
// SoftTissueSolver class
//------------------------------------------------------------------------------
class SoftTissueSolver : public BaseSolver {
public:
    explicit SoftTissueSolver(const SolverConfig& config = SolverConfig{});
    ~SoftTissueSolver() override;

    std::string solver_type() const override { return "SoftTissue"; }

    // Initialization and stepping
    void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) override;
    void reset() override;

    // Soft tissue specific config
    void set_soft_config(const SoftTissueSolverConfig& config);
    const SoftTissueSolverConfig& soft_config() const;

    // Add an external force to a specific node of an entity (for interactions)
    void apply_nodal_force(uint64_t entity_id, size_t node_idx, const datatypes::Vector3& force);

private:
    SoftTissueSolverConfig soft_config_;           // Solver settings

    // Per‑entity state (cached for performance)
    struct EntityState {
        std::shared_ptr<FEMEntity> entity;         // The FEM entity (could be SoftTissueEntity)
        std::vector<datatypes::Vector3> rest_positions;  // Original rest positions (for restoration)
        std::vector<datatypes::Matrix3r> viscous_strain; // Per‑element viscous strain (Maxwell)
        std::vector<datatypes::Matrix3r> plastic_strain; // Per‑element plastic strain
        std::vector<datatypes::Vector3> external_nodal_forces; // Accumulated interaction forces
    };
    std::vector<EntityState> entity_states_;       // List of all soft tissue entities
    bool states_dirty_ = true;                     // Need to rebuild entity list

    // Internal methods
    void rebuild_entity_list(const std::vector<std::shared_ptr<BaseEntity>>& entities);
    void compute_viscoelastic_forces(EntityState& st, double dt);
    void compute_restoration_forces(EntityState& st, double dt);
    void compute_bending_forces(EntityState& st, double dt);
    void apply_plasticity(EntityState& st, double dt);
    void integrate_entity(EntityState& st, double dt);
    
    // Helper: cross product
    static datatypes::Vector3 cross(const datatypes::Vector3& a, const datatypes::Vector3& b);
};

} // namespace engine
} // namespace genesis