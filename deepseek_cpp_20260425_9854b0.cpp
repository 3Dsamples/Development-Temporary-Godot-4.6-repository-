// genesis/engine/body_part_tissue_presets.h
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "genesis/datatypes.h"                     // Vector3, real

namespace genesis {
namespace engine {

class SoftTissueEntity;
class Scene;

//------------------------------------------------------------------------------
// Predefined anatomical tissue configurations
//------------------------------------------------------------------------------
namespace tissue_presets {

// Create a breast‑like soft tissue entity (non‑linear, very soft, high
// restoration, moderate jiggle).
//   mesh_path    – visual mesh file (OBJ/STL)
//   fem_mesh_path– tetrahedral mesh file (optional, will generate if empty)
//   scale        – uniform scale factor
std::shared_ptr<SoftTissueEntity> create_breast_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path = "",
    double scale = 1.0);

// Buttock tissue – similar to breast but slightly stiffer (due to muscle
// layer) and with higher damping to reduce jiggle.
std::shared_ptr<SoftTissueEntity> create_buttock_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path = "",
    double scale = 1.0);

// Generic flesh (e.g., abdomen, thigh) – medium stiffness, high restoration,
// low jiggle.
std::shared_ptr<SoftTissueEntity> create_flesh_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path = "",
    double scale = 1.0);

// Muscle tissue – anisotropic (simplified by higher stiffness along a given
// fibre direction), active contraction possible.
std::shared_ptr<SoftTissueEntity> create_muscle_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path = "",
    double scale = 1.0,
    const datatypes::Vector3& fibre_direction = datatypes::Vector3(0.0, 0.0, 1.0));

// Generic soft tissue factory using a custom config
std::shared_ptr<SoftTissueEntity> create_custom_entity(
    const std::string& mesh_path,
    const std::string& fem_mesh_path,
    const struct SoftTissueConfig& config,
    double scale = 1.0);

} // namespace tissue_presets

} // namespace engine
} // namespace genesis