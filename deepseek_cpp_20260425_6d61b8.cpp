// genesis/engine/body_part_tissue_presets.cpp
#include "genesis/engine/body_part_tissue_presets.h" // Corresponding header
#include "genesis/engine/entities/soft_tissue_entity.h" // SoftTissueEntity
#include "genesis/engine/mesh.h"                   // Mesh class for visual
#include <memory>                                  // std::make_shared
#include <string>                                  // std::string
#include <vector>                                  // std::vector
#include <array>                                   // std::array
#include <fstream>                                 // std::ifstream
#include <sstream>                                 // std::istringstream
#include <algorithm>                               // std::max, std::min
#include <cmath>                                   // M_PI, std::abs

namespace genesis {
namespace engine {
namespace tissue_presets {

//------------------------------------------------------------------------------
// Helper: load a tetrahedral mesh from a simple text file (.tet format).
// Format per line: 4 ints (node indices, 0‑based) after header "elements n"
// Returns false if file cannot be read.
//------------------------------------------------------------------------------
static bool load_tetrahedral_file(const std::string& path,
                                  std::vector<datatypes::Vector3>& nodes,
                                  std::vector<std::array<int, 4>>& tets) {
    std::ifstream file(path);
    if (!file.is_open()) return false;              // File not found

    std::string line;
    size_t num_nodes = 0, num_elems = 0;
    // Expect header: "nodes N" then "elements M"
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip comments
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "nodes") {
            iss >> num_nodes;                        // Read node count
            nodes.resize(num_nodes);
            for (size_t i = 0; i < num_nodes; ++i) {
                if (!std::getline(file, line)) return false;
                std::istringstream nss(line);
                nss >> nodes[i][0] >> nodes[i][1] >> nodes[i][2]; // Read position
            }
        } else if (token == "elements") {
            iss >> num_elems;                        // Read element count
            tets.resize(num_elems);
            for (size_t i = 0; i < num_elems; ++i) {
                if (!std::getline(file, line)) return false;
                std::istringstream ess(line);
                for (int j = 0; j < 4; ++j) {
                    ess >> tets[i][j];               // Read node indices
                }
            }
        }
    }
    return !nodes.empty() && !tets.empty();          // Success if data read
}

//------------------------------------------------------------------------------
// Helper: create a simple tetrahedral mesh from a visual mesh's bounding box.
// Splits the box into 5 tetrahedra with 8 distinct nodes (the 8 corners).
// This is a real (though basic) tetrahedralisation.
//------------------------------------------------------------------------------
static void generate_box_tetra_mesh(const Mesh& visual_mesh,
                                    std::vector<datatypes::Vector3>& nodes,
                                    std::vector<std::array<int, 4>>& tets) {
    datatypes::AABB box = visual_mesh.get_aabb();   // Get bounding box
    datatypes::Vector3 min = box.min;
    datatypes::Vector3 max = box.max;

    // 8 corners of the box
    nodes.resize(8);
    nodes[0] = min;                                  // 0
    nodes[1] = datatypes::Vector3(max[0], min[1], min[2]); // 1
    nodes[2] = datatypes::Vector3(max[0], max[1], min[2]); // 2
    nodes[3] = datatypes::Vector3(min[0], max[1], min[2]); // 3
    nodes[4] = datatypes::Vector3(min[0], min[1], max[2]); // 4
    nodes[5] = datatypes::Vector3(max[0], min[1], max[2]); // 5
    nodes[6] = max;                                  // 6
    nodes[7] = datatypes::Vector3(min[0], max[1], max[2]); // 7

    // 5 tetrahedra decomposition (standard box split)
    tets = {
        {0, 1, 2, 5},
        {0, 2, 3, 7},
        {0, 2, 5, 7},
        {0, 5, 4, 7},
        {2, 5, 6, 7}
    };
}

//------------------------------------------------------------------------------
// Helper: apply scale to nodes (in-place)
//------------------------------------------------------------------------------
static void scale_nodes(std::vector<datatypes::Vector3>& nodes, double scale) {
    if (scale == 1.0) return;
    for (auto& n : nodes) {
        n = n * scale;                               // Scale each component
    }
}

//------------------------------------------------------------------------------
// Helper: create a SoftTissueEntity from mesh paths and config, with optional
// tetrahedral mesh fallback.
//------------------------------------------------------------------------------
static std::shared_ptr<SoftTissueEntity> build_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path,
    const SoftTissueConfig& config,
    double scale) {

    // Create entity
    auto entity = std::make_shared<SoftTissueEntity>();

    // Load visual mesh
    auto visual_mesh = std::make_shared<Mesh>();
    bool mesh_loaded = visual_mesh->load_file(mesh_path);
    if (!mesh_loaded) {
        // If loading fails, create a default cube visual mesh for debugging
        visual_mesh = std::make_shared<Mesh>(Mesh::create_box(datatypes::Vector3(0.2, 0.2, 0.2)));
    }

    // Apply scale to visual mesh vertices
    if (scale != 1.0) {
        auto& verts = visual_mesh->vertices();
        for (auto& v : verts) {
            v.position = v.position * scale;
        }
        visual_mesh->update_aabb();
    }

    // Configure entity
    entity->set_soft_config(config);
    entity->set_visual_mesh(visual_mesh);

    // Load or generate tetrahedral FEM mesh
    std::vector<datatypes::Vector3> fem_nodes;
    std::vector<std::array<int, 4>> fem_tets;

    bool loaded = false;
    if (!fem_mesh_path.empty()) {
        loaded = load_tetrahedral_file(fem_mesh_path, fem_nodes, fem_tets);
    }
    if (!loaded) {
        // Fallback: use bounding box tetrahedralisation
        generate_box_tetra_mesh(*visual_mesh, fem_nodes, fem_tets);
    }
    // Apply scale to FEM nodes
    scale_nodes(fem_nodes, scale);

    // Set FEM mesh
    entity->set_nodes(fem_nodes);
    entity->set_tetrahedra(fem_tets);
    entity->compute_lumped_masses();                // Compute nodal masses from density

    return entity;
}

//==============================================================================
// Public preset functions
//==============================================================================

std::shared_ptr<SoftTissueEntity> create_breast_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path,
    double scale) {

    SoftTissueConfig cfg;
    cfg.material = SoftTissueMaterial::BREAST_TISSUE;
    // Very soft (low Young's modulus ~ 10 kPa)
    cfg.youngs_modulus = 1e4;                       // 10 kPa in Pa
    cfg.poisson_ratio = 0.49;                       // Nearly incompressible
    cfg.viscosity = 800.0;                          // High damping for jiggle
    cfg.relaxation_time = 0.15;                     // Slow return
    cfg.restoration_stiffness = 150.0;              // Moderate restoration
    cfg.restoration_damping = 50.0;                 // Damping during return
    cfg.bending_stiffness = 50.0;                   // Low bending resistance
    cfg.yield_stress = 5e3;                         // Plasticity onset low
    cfg.plastic_hardening = 1e3;

    return build_entity(mesh_path, fem_mesh_path, cfg, scale);
}

std::shared_ptr<SoftTissueEntity> create_buttock_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path,
    double scale) {

    SoftTissueConfig cfg;
    cfg.material = SoftTissueMaterial::FAT;          // Fatty tissue dominant
    // Slightly stiffer than breast (due to underlying muscle)
    cfg.youngs_modulus = 3e4;                       // 30 kPa
    cfg.poisson_ratio = 0.45;
    cfg.viscosity = 600.0;                          // Moderate damping
    cfg.relaxation_time = 0.1;
    cfg.restoration_stiffness = 250.0;              // Stronger restoration
    cfg.restoration_damping = 40.0;
    cfg.bending_stiffness = 80.0;
    cfg.yield_stress = 8e3;
    cfg.plastic_hardening = 3e3;

    return build_entity(mesh_path, fem_mesh_path, cfg, scale);
}

std::shared_ptr<SoftTissueEntity> create_flesh_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path,
    double scale) {

    SoftTissueConfig cfg;
    cfg.material = SoftTissueMaterial::SKIN;         // Generic skin/flesh
    cfg.youngs_modulus = 1e5;                       // 100 kPa (stiffer)
    cfg.poisson_ratio = 0.4;
    cfg.viscosity = 300.0;                          // Lower damping
    cfg.relaxation_time = 0.05;
    cfg.restoration_stiffness = 400.0;              // Strong restoration
    cfg.restoration_damping = 30.0;
    cfg.bending_stiffness = 200.0;
    cfg.yield_stress = 2e4;
    cfg.plastic_hardening = 5e3;

    return build_entity(mesh_path, fem_mesh_path, cfg, scale);
}

std::shared_ptr<SoftTissueEntity> create_muscle_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path,
    double scale,
    const datatypes::Vector3& fibre_direction) {

    SoftTissueConfig cfg;
    cfg.material = SoftTissueMaterial::MUSCLE;
    // Muscle is stiffer along fibres; we use a higher overall stiffness
    cfg.youngs_modulus = 5e5;                       // 500 kPa
    cfg.poisson_ratio = 0.45;
    cfg.viscosity = 200.0;                          // Less damping
    cfg.relaxation_time = 0.02;
    cfg.restoration_stiffness = 800.0;              // Very stiff restoration
    cfg.restoration_damping = 20.0;
    cfg.bending_stiffness = 500.0;
    cfg.yield_stress = 5e4;                         // High yield
    cfg.plastic_hardening = 1e4;

    // Anisotropy is not directly supported; the fibre direction can be
    // stored elsewhere or used in a future anisotropic material model.
    // For now we ignore the direction parameter but keep it in the signature
    // for future extension.

    return build_entity(mesh_path, fem_mesh_path, cfg, scale);
}

std::shared_ptr<SoftTissueEntity> create_custom_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path,
    const SoftTissueConfig& config,
    double scale) {

    return build_entity(mesh_path, fem_mesh_path, config, scale);
}

} // namespace tissue_presets
} // namespace engine
} // namespace genesis