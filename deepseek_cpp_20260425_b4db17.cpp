// genesis/engine/entities/jiggle_mapper.h
#pragma once

//------------------------------------------------------------------------------
// JiggleMapper – applies dynamic secondary motion (jiggle, wobble) to a mesh
// by simulating a spring‑damper per vertex, driven by the movement of the
// bones that influence it.  The jiggle amplitude and stiffness are defined
// per bone, and the effect is blended using standard skinning weights.
// Works seamlessly with skinned entities; leaves a hook for UI to set
// per‑bone parameters (stiffness, damping, influence).  No vertex colours or
// Fresnel are used.
//------------------------------------------------------------------------------

#include "genesis/datatypes.h"                     // Vector3, real
#include <vector>                                  // std::vector
#include <memory>                                  // std::shared_ptr
#include <functional>                              // std::function for hook

namespace genesis {
namespace engine {

// Forward declarations
class BaseEntity;          // For accessing skeleton
class Mesh;                // Visual mesh with skinning data
struct Bone;               // Bone structure (to be defined elsewhere)

//------------------------------------------------------------------------------
// Per‑bone jiggle parameters
//------------------------------------------------------------------------------
struct BoneJiggleParams {
    double stiffness = 200.0;         // Spring stiffness (N/m) for this bone's influence
    double damping = 20.0;            // Damping coefficient (N·s/m)
    double influence = 1.0;           // 0 = no jiggle, 1 = full effect
};

//------------------------------------------------------------------------------
// JiggleMapper class
//------------------------------------------------------------------------------
class JiggleMapper {
public:
    JiggleMapper();
    ~JiggleMapper();

    //----------------------------------------------------------------------
    // Binding
    //----------------------------------------------------------------------
    // Attach to an entity (which must have a skeleton) and its visual mesh
    // (which must have skinning vertex attributes: bone indices and weights).
    void bind(std::shared_ptr<BaseEntity> entity,
              std::shared_ptr<Mesh> mesh);
    void unbind();
    bool is_bound() const;

    //----------------------------------------------------------------------
    // Configuration of per‑bone parameters (hook for future UI)
    //----------------------------------------------------------------------
    // Set the jiggle parameters for a specific bone (0‑based index).
    void set_bone_params(int bone_index, const BoneJiggleParams& params);
    BoneJiggleParams get_bone_params(int bone_index) const;

    // Set global uniform parameters (used for bones not explicitly set).
    void set_global_params(const BoneJiggleParams& params);

    //----------------------------------------------------------------------
    // Update – called after skeletal animation and before rendering
    //----------------------------------------------------------------------
    // dt = time step (seconds)
    // The mapper computes target vertex positions from current bone transforms,
    // advances the spring simulation for each vertex, and writes the resulting
    // displaced positions back into the mesh's vertex buffer.
    void update(double dt);

    //----------------------------------------------------------------------
    // Hook for UI to be notified when jiggle values change
    //----------------------------------------------------------------------
    void set_update_callback(std::function<void()> callback);

private:
    std::weak_ptr<BaseEntity> entity_;             // Entity providing skeleton
    std::weak_ptr<Mesh> mesh_;                     // Visual mesh
    size_t num_vertices_ = 0;

    // Per‑vertex spring state (current offset from animated target)
    std::vector<datatypes::Vector3> offsets_;      // Current jiggle offset per vertex
    std::vector<datatypes::Vector3> velocities_;   // Offset velocity per vertex

    // Per‑bone parameters (indexed by bone index)
    std::vector<BoneJiggleParams> bone_params_;
    BoneJiggleParams global_params_;               // Fallback

    // Cached bone references (from entity's skeleton)
    // We store bone pointers or indices for quick access; for safety use indices.
    size_t num_bones_ = 0;

    // Reference rest positions (used to compute target animation positions)
    std::vector<datatypes::Vector3> rest_vertex_positions_;

    // Internal methods
    void init_spring_state();                      // Initialise offsets and velocities to zero
    void compute_target_positions(std::vector<datatypes::Vector3>& targets) const;
    void advance_springs(const std::vector<datatypes::Vector3>& targets, double dt);
    void write_vertex_positions(const std::vector<datatypes::Vector3>& positions);

    // Hook
    std::function<void()> update_callback_;
};

} // namespace engine
} // namespace genesis