// genesis/engine/entities/fem_entity.h

#pragma once

//------------------------------------------------------------------------------
// FEM Entity class - represents a deformable solid simulated with Finite Element Method.
// Contains tetrahedral or hexahedral mesh, material properties, and state vectors.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/base_entity.h"   // Base class: BaseEntity
#include "genesis/datatypes.h"                      // Vector3, Matrix3, etc.
#include <vector>                                   // std::vector for nodes and elements
#include <array>                                    // std::array for fixed-size element indices
#include <memory>                                   // std::shared_ptr
#include <string>                                   // std::string

namespace genesis {
namespace engine {

// Forward declarations
class Mesh;                                        // For visual mesh representation

//------------------------------------------------------------------------------
// Element types supported by FEM
//------------------------------------------------------------------------------
enum class FEMElementType : uint8_t {
    TET4 = 0,                                       // 4-node linear tetrahedron
    TET10 = 1,                                      // 10-node quadratic tetrahedron
    HEX8 = 2,                                       // 8-node linear hexahedron
    HEX20 = 3                                       // 20-node quadratic hexahedron
};

//------------------------------------------------------------------------------
// FEM Material model types
//------------------------------------------------------------------------------
enum class FEMMaterialModel : uint8_t {
    LINEAR_ELASTIC = 0,                             // Hooke's law (small strain)
    NEO_HOOKEAN = 1,                                // Neo-Hookean hyperelastic
    ST_VENANT_KIRCHHOFF = 2,                        // St. Venant-Kirchhoff (large rotation, small strain)
    MOONEY_RIVLIN = 3,                              // Mooney-Rivlin rubber-like
    VON_MISES_PLASTIC = 4                           // Elastoplastic with von Mises yield
};

//------------------------------------------------------------------------------
// FEM Configuration parameters
//------------------------------------------------------------------------------
struct FEMConfig {
    // Material properties
    FEMMaterialModel material_model = FEMMaterialModel::NEO_HOOKEAN;
    double youngs_modulus = 1e6;                    // Young's modulus (Pa)
    double poisson_ratio = 0.3;                     // Poisson's ratio
    double density = 1000.0;                        // Mass density (kg/m³)
    
    // Plasticity (if applicable)
    double yield_stress = 1e6;                      // Yield stress (Pa)
    double hardening_modulus = 0.0;                 // Linear hardening modulus (Pa)
    
    // Damping
    double rayleigh_alpha = 0.0;                    // Mass-proportional damping
    double rayleigh_beta = 0.0;                     // Stiffness-proportional damping
    
    // Integration parameters
    double integration_tolerance = 1e-6;             // Newton solver tolerance
    int max_newton_iterations = 10;                 // Max Newton iterations per step
    bool use_quasi_static = false;                  // If true, neglect inertial terms
};

//------------------------------------------------------------------------------
// FEM Entity class - deformable solid with finite element discretization
//------------------------------------------------------------------------------
class FEMEntity : public BaseEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    FEMEntity();                                    // Default constructor
    explicit FEMEntity(const std::string& name);    // Named constructor
    virtual ~FEMEntity();                           // Destructor

    //----------------------------------------------------------------------
    // Mesh definition
    //----------------------------------------------------------------------
    // Set nodes from vector of positions
    void set_nodes(const std::vector<datatypes::Vector3>& rest_positions);
    // Set tetrahedral elements (each element is 4 node indices)
    void set_tetrahedra(const std::vector<std::array<int, 4>>& tets);
    // Set hexahedral elements (each element is 8 node indices)
    void set_hexahedra(const std::vector<std::array<int, 8>>& hexes);
    // Set element type (must match the provided element data)
    void set_element_type(FEMElementType type);
    
    // Accessors
    size_t node_count() const { return nodes_.size(); }
    size_t element_count() const { return elements_.size(); }
    const std::vector<datatypes::Vector3>& rest_positions() const { return rest_positions_; }
    const std::vector<std::array<int, 4>>& tetrahedra() const { return tetrahedra_; }
    const std::vector<std::array<int, 8>>& hexahedra() const { return hexahedra_; }
    FEMElementType element_type() const { return element_type_; }
    
    // Node positions (current deformed state)
    datatypes::Vector3 node_position(size_t i) const;
    void set_node_position(size_t i, const datatypes::Vector3& pos);
    const std::vector<datatypes::Vector3>& node_positions() const { return current_positions_; }
    void set_node_positions(const std::vector<datatypes::Vector3>& positions);
    
    // Node velocities
    datatypes::Vector3 node_velocity(size_t i) const;
    void set_node_velocity(size_t i, const datatypes::Vector3& vel);
    const std::vector<datatypes::Vector3>& node_velocities() const { return velocities_; }
    void set_node_velocities(const std::vector<datatypes::Vector3>& velocities);
    
    // Node masses (lumped)
    double node_mass(size_t i) const;
    void compute_lumped_masses();                   // Compute masses from density and element volumes
    
    // External forces applied to nodes (cleared each step by solver)
    datatypes::Vector3 node_external_force(size_t i) const;
    void add_node_force(size_t i, const datatypes::Vector3& force);
    void clear_node_forces();
    
    // Rest position for a node
    datatypes::Vector3 node_rest_position(size_t i) const { return rest_positions_[i]; }
    
    //----------------------------------------------------------------------
    // Material and configuration
    //----------------------------------------------------------------------
    void set_fem_config(const FEMConfig& config);
    const FEMConfig& fem_config() const { return fem_config_; }
    
    // Per-element material override (optional)
    void set_element_material(size_t elem_idx, const FEMConfig& config);
    const FEMConfig* get_element_material(size_t elem_idx) const;
    
    //----------------------------------------------------------------------
    // Boundary conditions
    //----------------------------------------------------------------------
    // Fix a node (zero displacement)
    void fix_node(size_t node_idx, bool fixed = true);
    bool is_node_fixed(size_t node_idx) const;
    // Prescribe displacement for a node (overrides fix)
    void prescribe_displacement(size_t node_idx, const datatypes::Vector3& displacement);
    bool has_prescribed_displacement(size_t node_idx) const;
    datatypes::Vector3 prescribed_displacement(size_t node_idx) const;
    // Clear all boundary conditions
    void clear_boundary_conditions();
    
    //----------------------------------------------------------------------
    // State vectors for solver (deformation gradients, plastic strain)
    //----------------------------------------------------------------------
    const std::vector<datatypes::Matrix3r>& deformation_gradients() const { return F_; }
    void set_deformation_gradients(const std::vector<datatypes::Matrix3r>& F);
    const std::vector<datatypes::Matrix3r>& plastic_strains() const { return Fp_; }
    void set_plastic_strains(const std::vector<datatypes::Matrix3r>& Fp);
    
    //----------------------------------------------------------------------
    // Visual mesh (optional, for rendering)
    //----------------------------------------------------------------------
    void set_visual_mesh(std::shared_ptr<Mesh> mesh);
    std::shared_ptr<Mesh> visual_mesh() const { return visual_mesh_; }
    void update_visual_mesh();                      // Update mesh vertices to current deformed positions
    
    //----------------------------------------------------------------------
    // Overrides from BaseEntity
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;     // FEM uses its own solver, so this is minimal
    virtual void reset() override;
    virtual datatypes::AABB world_aabb() const override; // Override to compute from nodes
    
    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "FEMEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    // Mesh data
    std::vector<datatypes::Vector3> rest_positions_; // Undeformed node positions
    std::vector<std::array<int, 4>> tetrahedra_;     // TET4 elements
    std::vector<std::array<int, 8>> hexahedra_;      // HEX8 elements
    FEMElementType element_type_ = FEMElementType::TET4;
    
    // Current state
    std::vector<datatypes::Vector3> current_positions_; // Deformed positions
    std::vector<datatypes::Vector3> velocities_;        // Node velocities
    std::vector<datatypes::Vector3> external_forces_;   // Accumulated external forces
    
    // Mass properties
    std::vector<double> node_masses_;               // Lumped nodal masses
    
    // Material configuration
    FEMConfig fem_config_;                          // Default material for all elements
    std::vector<FEMConfig> element_materials_;      // Per-element override (empty = use default)
    
    // Boundary conditions
    std::vector<bool> fixed_nodes_;                 // True if node has zero displacement
    std::vector<bool> prescribed_nodes_;            // True if node has prescribed displacement
    std::vector<datatypes::Vector3> prescribed_displacements_; // Prescribed displacement values
    
    // Deformation state (per element)
    std::vector<datatypes::Matrix3r> F_;            // Deformation gradients
    std::vector<datatypes::Matrix3r> Fp_;           // Plastic part of deformation gradient
    
    // Visual representation
    std::shared_ptr<Mesh> visual_mesh_;             // Mesh for rendering
    
    // Helper methods
    void resize_state_vectors();                    // Allocate state vectors based on node/element counts
    void compute_element_rest_volume(size_t elem_idx, double& volume) const; // Compute rest volume of element
};

} // namespace engine
} // namespace genesis