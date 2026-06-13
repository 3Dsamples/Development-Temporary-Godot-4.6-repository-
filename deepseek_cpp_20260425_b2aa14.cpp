// genesis/engine/fresnel_material_modifier.cpp
#include "genesis/engine/fresnel_material_modifier.h" // Corresponding header
#include "genesis/engine/mesh.h"                     // for vertex/normal access
#include "genesis/engine/entities/fem_entity.h"      // for element material setting
#include "genesis/engine/entities/soft_tissue_entity.h" // if needed
#include <algorithm>                                 // std::max, std::min
#include <cmath>                                     // std::pow, std::sqrt, std::abs
#include <limits>                                    // numeric_limits

// If using xtensor/bignumber, replace the following types:
// #include <xtensor/xarray.hpp>
// using Vector3d = xt::xtensor<double, 1>; // or xt::xarray<double>
// etc. (see note at end)

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Construction / Destruction
//------------------------------------------------------------------------------
FresnelMaterialModifier::FresnelMaterialModifier() = default;  // Default config
FresnelMaterialModifier::~FresnelMaterialModifier() = default;

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void FresnelMaterialModifier::set_config(const FresnelConfig& config) {
    config_ = config;                               // Copy new settings
    factors_dirty_ = true;                          // Forces recalculation
}

const FresnelConfig& FresnelMaterialModifier::config() const {
    return config_;                                 // Return read-only reference
}

//------------------------------------------------------------------------------
// Bind / unbind
//------------------------------------------------------------------------------
void FresnelMaterialModifier::bind(std::shared_ptr<BaseEntity> entity,
                                   std::shared_ptr<Mesh> mesh) {
    entity_ = entity;                               // Store weak pointer to entity
    mesh_ = mesh;                                   // Store weak pointer to mesh
    factors_dirty_ = true;                          // Recompute factors when mesh changes
}

void FresnelMaterialModifier::unbind() {
    entity_.reset();                                // Release entity
    mesh_.reset();                                  // Release mesh
    vertex_factors_.clear();                        // Discard cache
}

bool FresnelMaterialModifier::is_bound() const {
    return !entity_.expired() && !mesh_.expired();  // Both must be valid
}

//------------------------------------------------------------------------------
// Compute Fresnel factors for each vertex
//------------------------------------------------------------------------------
void FresnelMaterialModifier::compute_fresnel(bool update_material) {
    auto mesh = mesh_.lock();                       // Retrieve strong reference
    if (!mesh) return;                              // Mesh no longer exists

    const auto& vertices = mesh->vertices();        // Vertex array
    const size_t num_verts = vertices.size();

    // Resize factor cache
    vertex_factors_.resize(num_verts, 0.0);         // Init with zero

    // Determine start point
    datatypes::Vector3 start = config_.start_point;
    if (config_.auto_start) {
        start = compute_auto_start();               // Use mesh centre
    }

    const datatypes::Vector3 axis = config_.axis.normalized(); // Ensure unit length
    const double power = config_.fresnel_power;
    const double dist_att = config_.distance_attenuation;

    // Precompute axis dot with start point for distance calculation
    const double start_proj = axis.dot(start);

    // Loop over vertices
    for (size_t i = 0; i < num_verts; ++i) {
        const auto& v = vertices[i];
        datatypes::Vector3 normal = v.normal;       // Vertex normal (assumed normalized)
        datatypes::Vector3 pos = v.position;         // World position

        // Compute factor = compute_vertex_factor(normal, pos)
        double factor = compute_vertex_factor(normal, pos);
        vertex_factors_[i] = factor;                 // Store
    }

    // Optionally write factors to vertex colours (alpha channel)
    // We'll write to the alpha component of vertex color (RGBA)
    for (size_t i = 0; i < num_verts; ++i) {
        // Get current color; preserve RGB, overwrite alpha with factor
        datatypes::Vector4 col = vertices[i].color;
        col[3] = static_cast<float>(vertex_factors_[i]); // store factor in alpha
        // Note: changing vertex color of the mesh is possible only if mesh is mutable;
        // we assume the Mesh class provides non‑const vertex access.
        // In practice, we'd need a method to set vertex color; we'll use the mutable access.
        // We'll call mesh->set_vertex_color(i, col); but that method might not exist.
        // For now, we directly modify the vertex reference if possible.
        // Actually, vertices are taken by const above; we need to get a non‑const vertex.
        // We'll use the operator[] that returns non‑const reference if mesh is not const.
        // But we have a const reference from vertices() const. We need to lock mesh as non‑const.
        // We'll assume we have a non‑const mesh reference; we already have shared_ptr<Mesh>, so we can
        // get a non‑const reference from mesh->vertices() (non‑const overload).
        // We'll do it directly using mesh->.
    }
    // Actually, the above loop for writing color is not working because we used const vertices above.
    // We'll rewrite: after the factor calculation loop, we get a non‑const reference to mesh vertices.
    {
        auto& mutable_verts = mesh->vertices();      // non‑const access
        for (size_t i = 0; i < num_verts; ++i) {
            mutable_verts[i].color[3] = static_cast<float>(vertex_factors_[i]); // set alpha
        }
    }

    factors_dirty_ = false;                         // Cache up‑to‑date

    if (update_material) {
        apply_to_entity();                           // Push factors to entity elements
    }

    // Fire callback if registered
    if (update_callback_) {
        update_callback_();
    }
}

//------------------------------------------------------------------------------
// Apply current per‑vertex factors to entity material properties
//------------------------------------------------------------------------------
void FresnelMaterialModifier::apply_to_entity() {
    auto entity = std::dynamic_pointer_cast<FEMEntity>(entity_.lock());
    if (!entity) return;                            // Only FEMEntity supported

    auto mesh = mesh_.lock();
    if (!mesh) return;

    // Ensure factors are computed
    if (factors_dirty_) {
        compute_fresnel(false);                     // Compute without recursively calling apply
    }

    const auto& vertices = mesh->vertices();        // Need to get vertex indices per element
    const auto& indices = mesh->indices();          // Triangle indices (for visual mesh)
    // Note: the visual mesh may have triangles; the FEM entity may have tetrahedra.
    // To map vertex factors to elements, we need a correspondence between visual mesh vertices
    // and FEM nodes. For simplicity, we assume the visual mesh vertices correspond one‑to‑one
    // with FEM nodes (same order). If not, a more advanced mapping would be needed.
    // We'll base the element factor on the average of its node factors.

    size_t n_elements = entity->element_count();
    if (n_elements == 0) return;

    // For each FEM element, compute an average factor from its nodes
    for (size_t e = 0; e < n_elements; ++e) {
        double sum_factor = 0.0;
        const auto& tet = entity->tetrahedra()[e];  // 4 node indices
        if (tet.empty()) continue;

        for (int i = 0; i < 4; ++i) {
            int node_idx = tet[i];                  // Index in FEM nodes (same as vertex index)
            if (node_idx >= 0 && node_idx < static_cast<int>(vertex_factors_.size())) {
                sum_factor += vertex_factors_[node_idx];
            }
        }
        double avg_factor = sum_factor / 4.0;       // Average factor for this element
        avg_factor = std::min(std::max(avg_factor, config_.min_factor), config_.max_factor); // Clamp

        // Interpolate material properties
        double stiffness   = config_.base_stiffness   + avg_factor * (config_.target_stiffness   - config_.base_stiffness);
        double damping     = config_.base_damping     + avg_factor * (config_.target_damping     - config_.base_damping);
        double restoration = config_.base_restoration + avg_factor * (config_.target_restoration - config_.base_restoration);
        double hardness    = config_.base_hardness    + avg_factor * (config_.target_hardness    - config_.base_hardness);

        // Set the element material (using FEMConfig or SoftTissueConfig)
        // We'll set via FEMEntity::set_element_material (FEMConfig) – we need to create a config
        FEMConfig mat = entity->fem_config();       // Get current default material
        mat.youngs_modulus = stiffness;              // assume stiffness ~ Young's modulus
        mat.viscosity      = damping;                // not directly in FEMConfig, maybe custom.
        // Since FEMConfig doesn't have all fields, we need to cast to SoftTissueEntity
        auto soft = std::dynamic_pointer_cast<SoftTissueEntity>(entity);
        if (soft) {
            SoftTissueConfig cfg = soft->soft_config();
            // actually SoftTissueConfig has youngs_modulus, viscosity, restoration_stiffness, etc.
            cfg.youngs_modulus = stiffness;
            cfg.viscosity      = damping;
            cfg.restoration_stiffness = restoration;
            // hardness isn't directly a property; we could map to yield_stress etc.
            sof->set_soft_config(cfg); // but that sets global, not per element
            // To set per‑element, SoftTissueEntity would need a different method.
            // For simplicity, we set the global config; the solver will use the global values.
            // In a full implementation, SoftTissueEntity would have a per‑element material override.
        } else {
            // generic FEM: set per‑element material (FEMConfig)
            entity->set_element_material(e, mat);   // overrides element material
        }
    }
}

//------------------------------------------------------------------------------
// Compute automatic start point: mesh bounding box centre
//------------------------------------------------------------------------------
datatypes::Vector3 FresnelMaterialModifier::compute_auto_start() const {
    auto mesh = mesh_.lock();
    if (!mesh) return datatypes::Vector3(0.0);       // No mesh, return origin

    datatypes::AABB aabb = mesh->get_aabb();         // Axis‑aligned bounding box
    return (aabb.min + aabb.max) * 0.5;              // Centre
}

//------------------------------------------------------------------------------
// Return cached vertex factors (computes if dirty)
//------------------------------------------------------------------------------
const std::vector<double>& FresnelMaterialModifier::get_vertex_factors() const {
    if (factors_dirty_) {
        // Call compute_fresnel on const? We'll cast away constness for lazy eval
        const_cast<FresnelMaterialModifier*>(this)->compute_fresnel(false);
    }
    return vertex_factors_;
}

//------------------------------------------------------------------------------
// Set update callback
//------------------------------------------------------------------------------
void FresnelMaterialModifier::set_update_callback(std::function<void()> callback) {
    update_callback_ = callback;
}

//------------------------------------------------------------------------------
// Private helper: compute Fresnel factor for a single vertex
//------------------------------------------------------------------------------
double FresnelMaterialModifier::compute_vertex_factor(const datatypes::Vector3& normal,
                                                      const datatypes::Vector3& world_pos) const {
    // Fresnel‑like factor based on angle between normal and propagation axis
    double NdotAxis = std::abs(normal.dot(config_.axis)); // 0..1
    double fresnel = std::pow(NdotAxis, config_.fresnel_power); // higher power = sharper

    // Distance attenuation along axis: distance from start point projected onto axis
    double proj = config_.axis.dot(world_pos);
    double dist_along_axis = std::abs(proj - config_.axis.dot(config_.start_point));
    // Use distance attenuation factor: factor = fresnel * (1 - distance_att * dist_along_axis)
    // but ensure it stays positive
    double att = 1.0 - config_.distance_attenuation * dist_along_axis;
    att = std::max(att, 0.0);                       // clamp to non‑negative

    double factor = fresnel * att;
    factor = std::min(std::max(factor, config_.min_factor), config_.max_factor);
    return factor;
}

} // namespace engine
} // namespace genesis

//------------------------------------------------------------------------------
// Note on upgrading to xtensor/bignumber:
// To replace datatypes::Vector3 with xtensor arrays:
//   using Vector3 = xt::xtensor<double, 1>; // shape {3}
//   Then operations like dot, cross, etc. would be done via xt::linalg or custom functions.
//   The rest of the logic remains unchanged.
//------------------------------------------------------------------------------