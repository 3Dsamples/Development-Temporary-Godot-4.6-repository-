// genesis/engine/fresnel_material_modifier.h
#pragma once

//------------------------------------------------------------------------------
// FresnelMaterialModifier – computes per‑vertex Fresnel factors based on a
// user‑defined direction and a start point, writes them to vertex colours, and
// maps the averaged factor to per‑element material properties of a
// SoftTissueEntity (or any FEMEntity).  Allows real‑time adjustment of
// stiffness, jiggle, hardness, and restoration, controlled by the surface
// orientation relative to the propagation axis.
//------------------------------------------------------------------------------

#include "genesis/datatypes.h"                     // Vector3, Matrix3r, AABB
#include <memory>                                  // std::shared_ptr
#include <vector>                                  // std::vector
#include <functional>                              // std::function

namespace genesis {
namespace engine {

// Forward declarations
class Mesh;
class FEMEntity;
class SoftTissueEntity;
class BaseEntity;

//------------------------------------------------------------------------------
// Configuration for the Fresnel modifier
//------------------------------------------------------------------------------
struct FresnelConfig {
    // Propagation axis (world‑space direction)
    datatypes::Vector3 axis = {0.0, 0.0, 1.0};     // e.g., (0,0,1) = forward direction
    
    // Start point for distance‑based attenuation (world‑space)
    datatypes::Vector3 start_point = {0.0, 0.0, 0.0};
    bool auto_start = true;                        // if true, use mesh centre automatically
    
    // Fresnel exponent (higher = sharper falloff from axis)
    double fresnel_power = 2.0;                    // typical range 0.5 – 5.0
    
    // Distance attenuation along axis (0 = no attenuation, 1 = linear)
    double distance_attenuation = 0.0;             // 0 to 1
    
    // Clamp Fresnel factor range
    double min_factor = 0.0;
    double max_factor = 1.0;
    
    // Material property targets (when factor = 1.0)
    double target_stiffness = 500.0;               // MPa? we use Pa
    double target_damping = 50.0;                  // N·s/m
    double target_restoration = 100.0;             // N/m
    double target_hardness = 0.8;                  // dimensionless (0 = very soft, 1 = rigid)
    
    // Base material properties (when factor = 0.0)
    double base_stiffness = 100.0;
    double base_damping = 10.0;
    double base_restoration = 20.0;
    double base_hardness = 0.1;
};

//------------------------------------------------------------------------------
// FresnelMaterialModifier class
//------------------------------------------------------------------------------
class FresnelMaterialModifier {
public:
    FresnelMaterialModifier();
    ~FresnelMaterialModifier();

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_config(const FresnelConfig& config);
    const FresnelConfig& config() const;
    
    //----------------------------------------------------------------------
    // Bind to an entity and its visual mesh
    //----------------------------------------------------------------------
    // The mesh provides vertex positions and normals; vertex colours are
    // written (or read) as alpha or colour channel.
    // entity is used to apply material changes.
    void bind(std::shared_ptr<BaseEntity> entity,
              std::shared_ptr<Mesh> mesh);
    void unbind();
    bool is_bound() const;
    
    //----------------------------------------------------------------------
    // Compute Fresnel factors and write to vertex colours
    //----------------------------------------------------------------------
    // Optionally updates the vertex colour array of the bound mesh.
    // If 'update_material' is true, also immediately applies the factors to
    // the entity's material properties.
    void compute_fresnel(bool update_material = true);
    
    //----------------------------------------------------------------------
    // Apply current factors to entity material properties (per element)
    //----------------------------------------------------------------------
    // Uses the current vertex colours (or recomputed factors) to set
    // stiffness, damping, restoration, hardness for each element.
    void apply_to_entity();
    
    //----------------------------------------------------------------------
    // Automatic start point: uses the bounding box centre of the mesh
    //----------------------------------------------------------------------
    datatypes::Vector3 compute_auto_start() const;
    
    //----------------------------------------------------------------------
    // Retrieve current per‑vertex factors (for debugging / inspection)
    //----------------------------------------------------------------------
    const std::vector<double>& get_vertex_factors() const;
    
    //----------------------------------------------------------------------
    // Register a callback when material properties change (e.g., for UI)
    //----------------------------------------------------------------------
    void set_update_callback(std::function<void()> callback);

private:
    FresnelConfig config_;
    std::weak_ptr<BaseEntity> entity_;
    std::weak_ptr<Mesh> mesh_;
    std::vector<double> vertex_factors_;   // cached factors (one per vertex)
    bool factors_dirty_ = true;
    std::function<void()> update_callback_;
    
    // Helper: compute Fresnel factor for a given vertex (normal, position)
    double compute_vertex_factor(const datatypes::Vector3& normal,
                                 const datatypes::Vector3& world_pos) const;
};

} // namespace engine
} // namespace genesis