// genesis/engine/entities/fem_entity.cpp

#include "genesis/engine/entities/fem_entity.h"     // Include corresponding header
#include "genesis/engine/mesh.h"                    // For visual mesh update
#include <cmath>                                    // std::abs, std::pow, M_PI
#include <algorithm>                                // std::fill, std::max, std::min
#include <sstream>                                  // std::ostringstream for repr
#include <numeric>                                  // std::iota

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// FEMEntity construction
//------------------------------------------------------------------------------
FEMEntity::FEMEntity()
    : BaseEntity("FEMEntity")                       // Call base constructor with default name
{
    resize_state_vectors();                         // Initialize empty vectors
}

FEMEntity::FEMEntity(const std::string& name)
    : BaseEntity(name)                              // Base constructor with custom name
{
    resize_state_vectors();                         // Initialize empty vectors
}

FEMEntity::~FEMEntity() {
    // Virtual destructor (cleanup handled by smart pointers and vectors)
}

//------------------------------------------------------------------------------
// Mesh definition
//------------------------------------------------------------------------------
void FEMEntity::set_nodes(const std::vector<datatypes::Vector3>& rest_positions) {
    // Set rest positions of all nodes
    rest_positions_ = rest_positions;               // Copy rest positions
    current_positions_ = rest_positions;            // Initially, current = rest
    resize_state_vectors();                         // Resize dependent vectors
}

void FEMEntity::set_tetrahedra(const std::vector<std::array<int, 4>>& tets) {
    // Set tetrahedral elements
    tetrahedra_ = tets;                             // Copy tetrahedra list
    element_type_ = FEMElementType::TET4;           // Update element type
    hexahedra_.clear();                             // Clear incompatible element list
    resize_state_vectors();                         // Resize per-element vectors
}

void FEMEntity::set_hexahedra(const std::vector<std::array<int, 8>>& hexes) {
    // Set hexahedral elements
    hexahedra_ = hexes;                             // Copy hexahedra list
    element_type_ = FEMElementType::HEX8;           // Update element type
    tetrahedra_.clear();                            // Clear incompatible element list
    resize_state_vectors();                         // Resize per-element vectors
}

void FEMEntity::set_element_type(FEMElementType type) {
    // Set the element type (must match provided element data)
    element_type_ = type;                           // Store type
}

//------------------------------------------------------------------------------
// Node state accessors
//------------------------------------------------------------------------------
datatypes::Vector3 FEMEntity::node_position(size_t i) const {
    // Return current position of node i (bounds checked)
    if (i < current_positions_.size()) {
        return current_positions_[i];               // Return stored position
    }
    return datatypes::Vector3(0.0);                 // Out of bounds: return zero
}

void FEMEntity::set_node_position(size_t i, const datatypes::Vector3& pos) {
    // Set current position of node i
    if (i < current_positions_.size()) {
        current_positions_[i] = pos;                // Update position
    }
}

void FEMEntity::set_node_positions(const std::vector<datatypes::Vector3>& positions) {
    // Set all node positions at once
    if (positions.size() == current_positions_.size()) {
        current_positions_ = positions;             // Copy entire vector
    } else {
        // Size mismatch: only copy up to smaller size
        size_t n = std::min(positions.size(), current_positions_.size());
        std::copy(positions.begin(), positions.begin() + n, current_positions_.begin());
    }
}

datatypes::Vector3 FEMEntity::node_velocity(size_t i) const {
    // Return velocity of node i
    if (i < velocities_.size()) {
        return velocities_[i];                      // Return stored velocity
    }
    return datatypes::Vector3(0.0);
}

void FEMEntity::set_node_velocity(size_t i, const datatypes::Vector3& vel) {
    // Set velocity of node i
    if (i < velocities_.size()) {
        velocities_[i] = vel;                       // Update velocity
    }
}

void FEMEntity::set_node_velocities(const std::vector<datatypes::Vector3>& velocities) {
    // Set all node velocities at once
    if (velocities.size() == velocities_.size()) {
        velocities_ = velocities;                   // Copy entire vector
    } else {
        size_t n = std::min(velocities.size(), velocities_.size());
        std::copy(velocities.begin(), velocities.begin() + n, velocities_.begin());
    }
}

//------------------------------------------------------------------------------
// Mass properties
//------------------------------------------------------------------------------
double FEMEntity::node_mass(size_t i) const {
    // Return lumped mass of node i
    if (i < node_masses_.size()) {
        return node_masses_[i];                     // Return stored mass
    }
    return 0.0;
}

void FEMEntity::compute_lumped_masses() {
    // Compute lumped nodal masses from element volumes and density
    node_masses_.assign(rest_positions_.size(), 0.0); // Reset masses to zero
    
    size_t num_elements = (element_type_ == FEMElementType::TET4) ? tetrahedra_.size() : hexahedra_.size();
    
    for (size_t e = 0; e < num_elements; ++e) {
        // Get element material configuration
        const FEMConfig* mat = (e < element_materials_.size()) ? &element_materials_[e] : &fem_config_;
        double density = mat->density;
        
        double volume = 0.0;                        // Element rest volume
        compute_element_rest_volume(e, volume);     // Compute volume based on element type
        
        double element_mass = density * volume;     // Total mass of this element
        
        // Distribute mass equally to all nodes of this element (lumped mass)
        int nodes_per_element = (element_type_ == FEMElementType::TET4) ? 4 : 8;
        double node_mass_contrib = element_mass / nodes_per_element;
        
        // Add contribution to each node
        if (element_type_ == FEMElementType::TET4) {
            const auto& tet = tetrahedra_[e];
            for (int i = 0; i < 4; ++i) {
                int node_idx = tet[i];
                node_masses_[node_idx] += node_mass_contrib; // Accumulate mass
            }
        } else {
            const auto& hex = hexahedra_[e];
            for (int i = 0; i < 8; ++i) {
                int node_idx = hex[i];
                node_masses_[node_idx] += node_mass_contrib;
            }
        }
    }
}

//------------------------------------------------------------------------------
// External forces
//------------------------------------------------------------------------------
datatypes::Vector3 FEMEntity::node_external_force(size_t i) const {
    // Return accumulated external force on node i
    if (i < external_forces_.size()) {
        return external_forces_[i];                 // Return stored force
    }
    return datatypes::Vector3(0.0);
}

void FEMEntity::add_node_force(size_t i, const datatypes::Vector3& force) {
    // Add force to node i's accumulator
    if (i < external_forces_.size()) {
        external_forces_[i] += force;               // Accumulate force
    }
}

void FEMEntity::clear_node_forces() {
    // Reset all external force accumulators to zero
    std::fill(external_forces_.begin(), external_forces_.end(), datatypes::Vector3(0.0));
}

//------------------------------------------------------------------------------
// Material configuration
//------------------------------------------------------------------------------
void FEMEntity::set_fem_config(const FEMConfig& config) {
    // Set the default material configuration
    fem_config_ = config;                           // Copy config
}

void FEMEntity::set_element_material(size_t elem_idx, const FEMConfig& config) {
    // Set material override for a specific element
    size_t num_elems = (element_type_ == FEMElementType::TET4) ? tetrahedra_.size() : hexahedra_.size();
    if (elem_idx < num_elems) {
        if (elem_idx >= element_materials_.size()) {
            element_materials_.resize(num_elems, fem_config_); // Resize and fill with default
        }
        element_materials_[elem_idx] = config;      // Store override
    }
}

const FEMConfig* FEMEntity::get_element_material(size_t elem_idx) const {
    // Get material configuration for element (returns default if no override)
    if (elem_idx < element_materials_.size()) {
        return &element_materials_[elem_idx];       // Return override
    }
    return &fem_config_;                            // Return default
}

//------------------------------------------------------------------------------
// Boundary conditions
//------------------------------------------------------------------------------
void FEMEntity::fix_node(size_t node_idx, bool fixed) {
    // Set or clear fixed boundary condition on a node
    if (node_idx < fixed_nodes_.size()) {
        fixed_nodes_[node_idx] = fixed;             // Set fixed flag
        if (fixed) {
            prescribed_nodes_[node_idx] = false;    // Fix overrides prescribed
        }
    }
}

bool FEMEntity::is_node_fixed(size_t node_idx) const {
    // Check if node is fixed
    if (node_idx < fixed_nodes_.size()) {
        return fixed_nodes_[node_idx];              // Return fixed status
    }
    return false;
}

void FEMEntity::prescribe_displacement(size_t node_idx, const datatypes::Vector3& displacement) {
    // Set prescribed displacement for a node
    if (node_idx < prescribed_nodes_.size()) {
        prescribed_nodes_[node_idx] = true;         // Mark as prescribed
        prescribed_displacements_[node_idx] = displacement; // Store displacement
        fixed_nodes_[node_idx] = false;             // Not fixed (prescribed overrides)
    }
}

bool FEMEntity::has_prescribed_displacement(size_t node_idx) const {
    // Check if node has prescribed displacement
    if (node_idx < prescribed_nodes_.size()) {
        return prescribed_nodes_[node_idx];         // Return prescribed status
    }
    return false;
}

datatypes::Vector3 FEMEntity::prescribed_displacement(size_t node_idx) const {
    // Get prescribed displacement value
    if (node_idx < prescribed_displacements_.size()) {
        return prescribed_displacements_[node_idx]; // Return stored displacement
    }
    return datatypes::Vector3(0.0);
}

void FEMEntity::clear_boundary_conditions() {
    // Clear all boundary condition flags
    std::fill(fixed_nodes_.begin(), fixed_nodes_.end(), false);
    std::fill(prescribed_nodes_.begin(), prescribed_nodes_.end(), false);
    // Prescribed displacements don't need clearing but reset to zero
    std::fill(prescribed_displacements_.begin(), prescribed_displacements_.end(), datatypes::Vector3(0.0));
}

//------------------------------------------------------------------------------
// State vectors for solver
//------------------------------------------------------------------------------
void FEMEntity::set_deformation_gradients(const std::vector<datatypes::Matrix3r>& F) {
    // Set deformation gradients for all elements
    if (F.size() == F_.size()) {
        F_ = F;                                     // Copy entire vector
    } else {
        size_t n = std::min(F.size(), F_.size());
        std::copy(F.begin(), F.begin() + n, F_.begin());
    }
}

void FEMEntity::set_plastic_strains(const std::vector<datatypes::Matrix3r>& Fp) {
    // Set plastic strain tensors for all elements
    if (Fp.size() == Fp_.size()) {
        Fp_ = Fp;                                   // Copy entire vector
    } else {
        size_t n = std::min(Fp.size(), Fp_.size());
        std::copy(Fp.begin(), Fp.begin() + n, Fp_.begin());
    }
}

//------------------------------------------------------------------------------
// Visual mesh
//------------------------------------------------------------------------------
void FEMEntity::set_visual_mesh(std::shared_ptr<Mesh> mesh) {
    // Set the visual representation mesh
    visual_mesh_ = mesh;                            // Store shared pointer
}

void FEMEntity::update_visual_mesh() {
    // Update visual mesh vertices to match current deformed positions
    if (!visual_mesh_) return;                      // No mesh to update
    if (current_positions_.empty()) return;         // No positions to copy
    
    auto& vertices = visual_mesh_->vertices();      // Get reference to mesh vertices
    size_t n = std::min(current_positions_.size(), vertices.size());
    for (size_t i = 0; i < n; ++i) {
        vertices[i].position = current_positions_[i]; // Copy position
    }
    visual_mesh_->compute_normals(true);            // Recompute normals for rendering
    visual_mesh_->update_aabb();                    // Update bounding box
}

//------------------------------------------------------------------------------
// Overrides from BaseEntity
//------------------------------------------------------------------------------
void FEMEntity::integrate(double dt) {
    // FEM integration is handled by FEMSolver, not here.
    // This method is a no-op for FEM entities.
    (void)dt;                                       // Suppress unused parameter warning
}

void FEMEntity::reset() {
    // Reset to initial rest configuration
    BaseEntity::reset();                            // Reset base transform and velocity
    current_positions_ = rest_positions_;           // Reset positions to rest state
    std::fill(velocities_.begin(), velocities_.end(), datatypes::Vector3(0.0)); // Zero velocities
    std::fill(external_forces_.begin(), external_forces_.end(), datatypes::Vector3(0.0)); // Clear forces
    
    // Reset deformation gradients to identity
    for (auto& F : F_) {
        F = datatypes::Matrix3r(1.0);               // Identity matrix
    }
    // Reset plastic strains to identity
    for (auto& Fp : Fp_) {
        Fp = datatypes::Matrix3r(1.0);              // Identity matrix
    }
}

datatypes::AABB FEMEntity::world_aabb() const {
    // Compute world-space AABB from current node positions
    datatypes::AABB aabb;                           // Start empty
    for (const auto& pos : current_positions_) {
        aabb.expand(pos);                           // Expand by each node position
    }
    return aabb;                                    // Return computed AABB
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string FEMEntity::repr() const {
    std::ostringstream oss;
    oss << "FEMEntity(id=" << id()
        << ", name=\"" << name() << "\""
        << ", nodes=" << node_count()
        << ", elements=" << element_count()
        << ", material=" << fem_config_.material_model
        << ")";
    return oss.str();
}

std::string FEMEntity::str() const {
    return name() + " (FEM: " + std::to_string(node_count()) + " nodes)";
}

//------------------------------------------------------------------------------
// Private helper methods
//------------------------------------------------------------------------------
void FEMEntity::resize_state_vectors() {
    // Allocate or resize all state vectors based on node/element counts
    size_t n_nodes = rest_positions_.size();
    size_t n_elems = (element_type_ == FEMElementType::TET4) ? tetrahedra_.size() : hexahedra_.size();
    
    // Node-related vectors
    current_positions_.resize(n_nodes, datatypes::Vector3(0.0));
    velocities_.resize(n_nodes, datatypes::Vector3(0.0));
    external_forces_.resize(n_nodes, datatypes::Vector3(0.0));
    node_masses_.resize(n_nodes, 0.0);
    fixed_nodes_.resize(n_nodes, false);
    prescribed_nodes_.resize(n_nodes, false);
    prescribed_displacements_.resize(n_nodes, datatypes::Vector3(0.0));
    
    // Initialize current positions to rest positions
    current_positions_ = rest_positions_;
    
    // Element-related vectors
    F_.resize(n_elems, datatypes::Matrix3r(1.0));   // Identity deformation gradients
    Fp_.resize(n_elems, datatypes::Matrix3r(1.0));  // Identity plastic strains
    
    // Compute masses if nodes exist
    if (n_nodes > 0 && n_elems > 0) {
        compute_lumped_masses();                    // Calculate nodal masses
    }
}

void FEMEntity::compute_element_rest_volume(size_t elem_idx, double& volume) const {
    // Compute rest volume of a given element based on its type
    volume = 0.0;
    if (element_type_ == FEMElementType::TET4) {
        if (elem_idx >= tetrahedra_.size()) return;
        const auto& tet = tetrahedra_[elem_idx];
        datatypes::Vector3 v0 = rest_positions_[tet[0]];
        datatypes::Vector3 v1 = rest_positions_[tet[1]];
        datatypes::Vector3 v2 = rest_positions_[tet[2]];
        datatypes::Vector3 v3 = rest_positions_[tet[3]];
        // Volume of tetrahedron = |(v1-v0) · ((v2-v0) × (v3-v0))| / 6
        datatypes::Vector3 a = v1 - v0;
        datatypes::Vector3 b = v2 - v0;
        datatypes::Vector3 c = v3 - v0;
        volume = std::abs(a.dot(b.cross(c))) / 6.0;
    } else if (element_type_ == FEMElementType::HEX8) {
        if (elem_idx >= hexahedra_.size()) return;
        const auto& hex = hexahedra_[elem_idx];
        // Decompose hexahedron into 5 tetrahedra and sum volumes
        // Simplification: use average of two decompositions for better accuracy
        // For brevity, we use a single decomposition (should be improved in production)
        std::array<std::array<int, 4>, 5> tet_decomp = {{
            {{hex[0], hex[1], hex[3], hex[4]}},
            {{hex[1], hex[2], hex[3], hex[6]}},
            {{hex[1], hex[3], hex[4], hex[6]}},
            {{hex[1], hex[4], hex[5], hex[6]}},
            {{hex[3], hex[4], hex[6], hex[7]}}
        }};
        volume = 0.0;
        for (const auto& tet : tet_decomp) {
            datatypes::Vector3 v0 = rest_positions_[tet[0]];
            datatypes::Vector3 v1 = rest_positions_[tet[1]];
            datatypes::Vector3 v2 = rest_positions_[tet[2]];
            datatypes::Vector3 v3 = rest_positions_[tet[3]];
            datatypes::Vector3 a = v1 - v0;
            datatypes::Vector3 b = v2 - v0;
            datatypes::Vector3 c = v3 - v0;
            volume += std::abs(a.dot(b.cross(c))) / 6.0;
        }
    }
}

} // namespace engine
} // namespace genesis