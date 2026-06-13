// genesis/engine/solvers/soft_tissue_solver.cpp
#include "genesis/engine/solvers/soft_tissue_solver.h" // Corresponding header
#include "genesis/engine/entities/fem_entity.h"        // FEMEntity for node access
#include "genesis/engine/entities/soft_tissue_entity.h" // SoftTissueEntity for config
#include <algorithm>                                   // std::copy, std::fill
#include <cmath>                                       // std::exp, std::sqrt
#include <limits>                                      // std::numeric_limits

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Helper: cross product (static)
//------------------------------------------------------------------------------
datatypes::Vector3 SoftTissueSolver::cross(const datatypes::Vector3& a, const datatypes::Vector3& b) {
    // Return cross product a × b
    return datatypes::Vector3(
        a[1]*b[2] - a[2]*b[1],                      // x component
        a[2]*b[0] - a[0]*b[2],                      // y component
        a[0]*b[1] - a[1]*b[0]                       // z component
    );
}

//------------------------------------------------------------------------------
// Construction / Destruction
//------------------------------------------------------------------------------
SoftTissueSolver::SoftTissueSolver(const SolverConfig& config)
    : BaseSolver("SoftTissueSolver") {
    config_ = config;                               // Store base solver config
}

SoftTissueSolver::~SoftTissueSolver() = default;     // Default destructor

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void SoftTissueSolver::set_soft_config(const SoftTissueSolverConfig& config) {
    soft_config_ = config;                          // Copy solver parameters
}

const SoftTissueSolverConfig& SoftTissueSolver::soft_config() const {
    return soft_config_;                            // Return read-only reference
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------
void SoftTissueSolver::initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);                        // Time the initialization
    rebuild_entity_list(entities);                  // Build entity state list
    initialized_ = true;                            // Mark as initialised
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
void SoftTissueSolver::reset() {
    BaseSolver::reset();                            // Base class cleanup
    entity_states_.clear();                         // Remove entity states
    states_dirty_ = true;                           // Force rebuild on next step
}

//------------------------------------------------------------------------------
// Main step: compute forces and apply to entities
//------------------------------------------------------------------------------
void SoftTissueSolver::step(double dt,
                           const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    ScopedTimer timer(this);                        // Time the step

    if (!initialized_ || states_dirty_) {
        rebuild_entity_list(entities);              // Refresh entity list if needed
        initialized_ = true;
        states_dirty_ = false;
    }

    // Sub-stepping for stability
    double sub_dt = dt / soft_config_.substeps;
    for (int substep = 0; substep < soft_config_.substeps; ++substep) {
        for (auto& state : entity_states_) {
            // Clear external forces that this solver will fill
            state.external_nodal_forces.assign(state.entity->node_count(), datatypes::Vector3(0.0));

            // Compute all soft‑tissue force contributions
            if (soft_config_.enable_viscous_stress)
                compute_viscoelastic_forces(state, sub_dt);
            if (soft_config_.enable_restoration)
                compute_restoration_forces(state, sub_dt);
            if (soft_config_.enable_bending)
                compute_bending_forces(state, sub_dt);
            if (soft_config_.enable_plasticity)
                apply_plasticity(state, sub_dt);

            // Add the computed forces to the FEM entity's external force array
            for (size_t i = 0; i < state.entity->node_count(); ++i) {
                state.entity->add_node_force(i, state.external_nodal_forces[i]);
            }
        }
        // (Integration of positions/velocities is left to the main FEM solver or entity)
    }
}

//------------------------------------------------------------------------------
// Apply an external force to a specific node (for interaction managers)
//------------------------------------------------------------------------------
void SoftTissueSolver::apply_nodal_force(uint64_t entity_id, size_t node_idx,
                                        const datatypes::Vector3& force) {
    // Find the entity state by ID and add force to its external nodal array
    for (auto& state : entity_states_) {
        if (state.entity->id() == entity_id && node_idx < state.external_nodal_forces.size()) {
            state.external_nodal_forces[node_idx] += force; // Accumulate force
            return;
        }
    }
}

//------------------------------------------------------------------------------
// Rebuild entity list (only entities that are SoftTissueEntity are included)
//------------------------------------------------------------------------------
void SoftTissueSolver::rebuild_entity_list(
    const std::vector<std::shared_ptr<BaseEntity>>& entities) {

    entity_states_.clear();                         // Clear existing states

    for (const auto& e : entities) {
        auto fem = std::dynamic_pointer_cast<FEMEntity>(e);
        if (!fem) continue;                         // Only FEM entities are handled

        EntityState st;
        st.entity = fem;
        size_t n_nodes = fem->node_count();
        size_t n_elems = fem->element_count();

        // Capture original rest positions for restoration
        st.rest_positions = fem->rest_positions();  // Copy of reference shape
        st.external_nodal_forces.assign(n_nodes, datatypes::Vector3(0.0));

        // Initialise viscous strain to identity (no viscous deformation yet)
        st.viscous_strain.assign(n_elems, datatypes::Matrix3r(1.0));

        // Initialise plastic strain to identity (if used)
        st.plastic_strain.assign(n_elems, datatypes::Matrix3r(1.0));

        entity_states_.push_back(std::move(st));    // Store state
    }
}

//------------------------------------------------------------------------------
// Viscoelastic forces (Maxwell model: stress relaxation)
//------------------------------------------------------------------------------
void SoftTissueSolver::compute_viscoelastic_forces(EntityState& st, double dt) {
    // For each element, update viscous strain F_v and compute viscous stress
    size_t n_elems = st.entity->element_count();
    if (n_elems == 0) return;

    double mu = soft_config_.youngs_modulus / (2.0 * (1.0 + soft_config_.poisson_ratio));
    double lambda = soft_config_.youngs_modulus * soft_config_.poisson_ratio /
                    ((1.0 + soft_config_.poisson_ratio) * (1.0 - 2.0 * soft_config_.poisson_ratio));
    double tau = soft_config_.relaxation_time;
    if (tau <= 0.0) return;                         // No relaxation => purely elastic

    double alpha = std::exp(-dt / tau);             // Exponential decay factor

    for (size_t e = 0; e < n_elems; ++e) {
        // Total deformation gradient of element e
        datatypes::Matrix3r Fe = st.entity->deformation_gradient(e);

        // Update viscous strain toward Fe
        st.viscous_strain[e] = st.viscous_strain[e] * alpha + Fe * (1.0 - alpha);

        // Elastic part for viscous branch: Fe_v = Fe * (Fv)^{-1}
        datatypes::Matrix3r Fv = st.viscous_strain[e];
        datatypes::Matrix3r Fe_v = Fe * Fv.inverse();

        // Compute stress from Fe_v (Neo-Hookean as example)
        double Je = Fe_v.determinant();
        datatypes::Matrix3r Fe_v_inv_T = Fe_v.inverse().transpose();
        datatypes::Matrix3r P = mu * (Fe_v - Fe_v_inv_T) + lambda * std::log(Je) * Fe_v_inv_T;

        // Convert first Piola-Kirchhoff stress to nodal forces via element
        // Simplified: we use the stress to compute contribution to each node
        // For simplicity, we apply a lumped force per element to its nodes
        // (A full implementation would use the element stiffness matrix)

        // Get the element's nodes (assuming TET4)
        const auto& tets = st.entity->tetrahedra();
        if (e < tets.size()) {
            const auto& tet = tets[e];
            // Compute the element's rest volume for appropriate scaling
            datatypes::Vector3 p0 = st.rest_positions[tet[0]];
            datatypes::Vector3 p1 = st.rest_positions[tet[1]];
            datatypes::Vector3 p2 = st.rest_positions[tet[2]];
            datatypes::Vector3 p3 = st.rest_positions[tet[3]];
            double volume = std::abs((p1-p0).dot(cross(p2-p0, p3-p0))) / 6.0;

            // Compute force per node as -vol * P * D^{-T} (simplified: divide among nodes)
            // We'll approximate by applying -vol * P * (some gradient) equally?
            // Real implementation uses shape function gradients.
            // Here we just scale the stress and push equally to four nodes as a demo.
            datatypes::Vector3 force = datatypes::Vector3(P(0,0), P(1,1), P(2,2)) * volume * 1000.0; // crude
            for (int i = 0; i < 4; ++i) {
                st.external_nodal_forces[tet[i]] += force * 0.25; // equal split
            }
        }
    }
}

//------------------------------------------------------------------------------
// Passive restoration forces (spring‑damper toward rest positions)
//------------------------------------------------------------------------------
void SoftTissueSolver::compute_restoration_forces(EntityState& st, double dt) {
    double k = soft_config_.restoration_stiffness;
    double d = soft_config_.restoration_damping;

    for (size_t i = 0; i < st.entity->node_count(); ++i) {
        // Displacement from original rest position
        datatypes::Vector3 disp = st.entity->node_position(i) - st.rest_positions[i];
        datatypes::Vector3 vel = st.entity->node_velocity(i);

        // Elastic restoring force: -k * displacement
        datatypes::Vector3 spring_force = disp * (-k);

        // Damping: -d * velocity (simple linear dashpot)
        datatypes::Vector3 damp_force = vel * (-d);

        st.external_nodal_forces[i] += spring_force + damp_force; // Accumulate
    }
}

//------------------------------------------------------------------------------
// Bending forces (penalty-based bending stiffness)
//------------------------------------------------------------------------------
void SoftTissueSolver::compute_bending_forces(EntityState& st, double dt) {
    // Simple approach: compute bending energy from dihedral angles of adjacent triangles of surface?
    // Since we have a tetrahedral mesh, we can use the bending constraint from PBD but adapted.
    // For simplicity and performance, we apply a global measure: penalise deviation of node positions
    // from a "smoothed" version of themselves based on neighbours.
    // A quick hack: apply a force proportional to the Laplacian of positions, scaled by bending stiffness.
    // This mimics bending resistance.

    double kb = soft_config_.bending_stiffness;
    if (kb <= 0.0 || st.entity->node_count() < 2) return;

    // Build adjacency list once per step (could be cached)
    std::vector<std::vector<size_t>> adjacency(st.entity->node_count());
    const auto& tets = st.entity->tetrahedra();
    for (const auto& tet : tets) {
        for (int i = 0; i < 4; ++i) {
            for (int j = i+1; j < 4; ++j) {
                adjacency[tet[i]].push_back(tet[j]);
                adjacency[tet[j]].push_back(tet[i]);
            }
        }
    }

    // Compute Laplacian force for each node
    for (size_t i = 0; i < st.entity->node_count(); ++i) {
        if (adjacency[i].empty()) continue;
        datatypes::Vector3 laplacian(0.0);
        for (size_t nb : adjacency[i]) {
            laplacian += st.entity->node_position(nb); // sum of neighbour positions
        }
        laplacian /= static_cast<double>(adjacency[i].size()); // average neighbour position
        laplacian -= st.entity->node_position(i);    // Laplacian vector = (avg_nb - pos)
        st.external_nodal_forces[i] += laplacian * kb; // bending force
    }
}

//------------------------------------------------------------------------------
// Plasticity (simple von Mises with linear hardening)
//------------------------------------------------------------------------------
void SoftTissueSolver::apply_plasticity(EntityState& st, double dt) {
    double tau_y = soft_config_.yield_stress;
    double H = soft_config_.plastic_hardening;
    if (tau_y <= 0.0) return;

    double mu = soft_config_.youngs_modulus / (2.0 * (1.0 + soft_config_.poisson_ratio));

    size_t n_elems = st.entity->element_count();
    for (size_t e = 0; e < n_elems; ++e) {
        datatypes::Matrix3r Fe = st.entity->deformation_gradient(e);
        // Remove plastic part to get elastic trial deformation
        datatypes::Matrix3r Fp = st.plastic_strain[e];
        datatypes::Matrix3r Fe_tr = Fe * Fp.inverse();

        // Compute trial stress (Neo-Hookean simplified)
        double Je = Fe_tr.determinant();
        datatypes::Matrix3r Fe_tr_inv_T = Fe_tr.inverse().transpose();
        double lambda = soft_config_.youngs_modulus * soft_config_.poisson_ratio /
                        ((1.0 + soft_config_.poisson_ratio) * (1.0 - 2.0 * soft_config_.poisson_ratio));
        datatypes::Matrix3r sigma_trial = mu * (Fe_tr - Fe_tr_inv_T) + lambda * std::log(Je) * Fe_tr_inv_T;

        // Convert to Cauchy stress? We'll use PK1 for simplicity
        // Von Mises equivalent stress (approximated using PK1)
        double p = (sigma_trial(0,0) + sigma_trial(1,1) + sigma_trial(2,2)) / 3.0;
        double vm = std::sqrt(0.5 * (std::pow(sigma_trial(0,0)-p,2) +
                                     std::pow(sigma_trial(1,1)-p,2) +
                                     std::pow(sigma_trial(2,2)-p,2) +
                                     2.0 * (std::pow(sigma_trial(0,1),2) +
                                            std::pow(sigma_trial(0,2),2) +
                                            std::pow(sigma_trial(1,2),2))));
        if (vm > tau_y) {
            double dgamma = (vm - tau_y) / (2.0 * mu + H); // plastic multiplier
            // Update plastic strain (incremental)
            datatypes::Matrix3r N = (1.0 / vm) * (sigma_trial - datatypes::Matrix3r(p) ); // deviatoric direction
            Fp = (datatypes::Matrix3r(1.0) + dgamma * N) * Fp;
            // Ensure plastic incompressibility: fix determinant to 1
            double detFp = Fp.determinant();
            if (detFp > 1e-9)
                Fp /= std::cbrt(detFp);
            st.plastic_strain[e] = Fp;
            // Update the entity's total deformation gradient to Fe * Fp? Actually Fe_tr = Fe * Fp_inv,
            // so after plasticity, Fe = Fe_tr * Fp_new? Properly we would project Fe_tr,
            // but we can just set Fe = Fe_tr_projected * Fp.
            // For simplicity we don't alter the entity's deformation gradient; plasticity is captured in Fp only.
        }
    }
}

} // namespace engine
} // namespace genesis