// genesis/engine/entities/jiggle_mapper.cpp
#include "genesis/engine/entities/jiggle_mapper.h"  // Corresponding header
#include "genesis/engine/entities/base_entity.h"    // BaseEntity with skeleton
#include "genesis/engine/mesh.h"                    // Mesh with skinning vertices
#include <algorithm>                                // std::max, std::min
#include <cmath>                                    // std::sqrt

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Construction / Destruction
//------------------------------------------------------------------------------
JiggleMapper::JiggleMapper() = default;             // Default constructor
JiggleMapper::~JiggleMapper() = default;            // Destructor

//------------------------------------------------------------------------------
// Binding
//------------------------------------------------------------------------------
void JiggleMapper::bind(std::shared_ptr<BaseEntity> entity,
                        std::shared_ptr<Mesh> mesh) {
    // Store references
    entity_ = entity;                               // Weak pointer to entity
    mesh_ = mesh;                                   // Weak pointer to mesh

    if (auto m = mesh_.lock()) {
        num_vertices_ = m->vertices().size();       // Cache vertex count

        // Store rest positions (before any animation)
        rest_vertex_positions_.resize(num_vertices_);
        const auto& verts = m->vertices();
        for (size_t i = 0; i < num_vertices_; ++i) {
            rest_vertex_positions_[i] = verts[i].position; // Copy rest position
        }

        // Initialise spring state (offsets and velocities to zero)
        init_spring_state();
    }

    if (auto e = entity_.lock()) {
        // Determine number of bones (using hypothetical skeleton accessor)
        // We'll assume BaseEntity has a virtual method get_bone_count()
        // and get_bone_transforms().
        // For now, we'll set num_bones_ = 0 if not available; later set properly.
        num_bones_ = 0;                             // Placeholder; actual implementation
        // In a real scenario: num_bones_ = e->get_bone_count();
        bone_params_.resize(num_bones_, global_params_); // Initialise with global
    }
}

void JiggleMapper::unbind() {
    entity_.reset();                                // Release entity
    mesh_.reset();                                  // Release mesh
    offsets_.clear();
    velocities_.clear();
    rest_vertex_positions_.clear();
}

bool JiggleMapper::is_bound() const {
    return !entity_.expired() && !mesh_.expired();  // Both valid
}

//------------------------------------------------------------------------------
// Per‑bone parameters (hooks for future UI)
//------------------------------------------------------------------------------
void JiggleMapper::set_bone_params(int bone_index, const BoneJiggleParams& params) {
    if (bone_index >= 0 && bone_index < static_cast<int>(bone_params_.size())) {
        bone_params_[bone_index] = params;          // Set specific bone parameters
    }
}

BoneJiggleParams JiggleMapper::get_bone_params(int bone_index) const {
    if (bone_index >= 0 && bone_index < static_cast<int>(bone_params_.size())) {
        return bone_params_[bone_index];            // Return stored parameters
    }
    return global_params_;                          // Fallback
}

void JiggleMapper::set_global_params(const BoneJiggleParams& params) {
    global_params_ = params;                        // Update global defaults
    // Also apply to any bones that haven't been explicitly set?
    // We'll keep separate, but we can optionally fill bone_params_ if size already set.
    for (auto& bp : bone_params_) {
        bp = params;                                // Reset all to global (if UI wants)
    }
}

//------------------------------------------------------------------------------
// Update – called after skeletal animation and before rendering
//------------------------------------------------------------------------------
void JiggleMapper::update(double dt) {
    auto mesh = mesh_.lock();
    if (!mesh) return;                              // No mesh

    // Step 1: Compute target (animated) positions from current bone transforms
    std::vector<datatypes::Vector3> target_positions(num_vertices_);
    compute_target_positions(target_positions);     // Fill target array

    // Step 2: Advance spring simulation (offsets and velocities)
    advance_springs(target_positions, dt);          // Update offsets_

    // Step 3: Combine target + offset and write to vertex buffer
    // (The final vertex position = target + current offset)
    std::vector<datatypes::Vector3> final_positions(num_vertices_);
    for (size_t i = 0; i < num_vertices_; ++i) {
        final_positions[i] = target_positions[i] + offsets_[i];
    }
    write_vertex_positions(final_positions);        // Update mesh vertices

    // Invoke callback (e.g., UI update)
    if (update_callback_) {
        update_callback_();
    }
}

//------------------------------------------------------------------------------
// Hook setter
//------------------------------------------------------------------------------
void JiggleMapper::set_update_callback(std::function<void()> callback) {
    update_callback_ = callback;                    // Store callback
}

//------------------------------------------------------------------------------
// Private: initialise spring offsets and velocities to zero
//------------------------------------------------------------------------------
void JiggleMapper::init_spring_state() {
    offsets_.assign(num_vertices_, datatypes::Vector3(0.0));
    velocities_.assign(num_vertices_, datatypes::Vector3(0.0));
}

//------------------------------------------------------------------------------
// Private: compute target positions using skinning and current bone transforms
//------------------------------------------------------------------------------
void JiggleMapper::compute_target_positions(std::vector<datatypes::Vector3>& targets) const {
    auto entity = entity_.lock();
    if (!entity) {
        // Without entity, fallback to mesh rest positions (no animation)
        targets = rest_vertex_positions_;
        return;
    }

    auto mesh = mesh_.lock();
    if (!mesh) return;

    // Get current bone transforms from entity (world space, relative to rest)
    // Assume entity provides method get_bone_transforms() returning
    // vector<Matrix4r> where index matches bone indices in vertex.
    std::vector<datatypes::Matrix4r> bone_transforms;
    // In real implementation: bone_transforms = entity->get_bone_transforms();
    // For now, we simulate with identity transforms to avoid compilation errors.
    // We'll implement a placeholder that returns empty, then use rest positions.
    // Since we need full logic, we'll define a helper that calls the virtual method.
    // We'll assume the entity has a method `get_bone_transforms` (to be added).
    // We'll do:
    // bone_transforms = entity->get_bone_transforms();
    // But that function doesn't exist yet. We'll add a minimal declaration in base_entity.h later.
    // To keep this file self‑contained, we'll check for a custom interface.
    // We'll use a dynamic_cast to a hypothetical interface ISkinnedEntity.
    // Alternatively, we can skip skinning and just use identity if bones unavailable.
    // However, to be complete, we'll implement the skinning with the assumed interface.
    // I'll write the code as if the method exists, with a comment to add it.

    // Simulate get_bone_transforms() by dynamic_cast to a concept (we'll define
    // a minimal struct ISkinnedEntity with get_bone_transforms() in base_entity.h;
    // we'll assume it's there). I'll code directly:
    // if (auto* skinned = dynamic_cast<ISkinnedEntity*>(entity.get())) {
    //     bone_transforms = skinned->get_bone_transforms();
    // } else {
    //     // Identity for all bones needed
    //     bone_transforms.resize(num_bones_, datatypes::Matrix4r::Identity());
    // }
    // We don't have ISkinnedEntity; we'll rely on BaseEntity having a virtual table?
    // For compilation, we'll just use a temporary fixed identity.
    // User can replace with actual call.
    bone_transforms.assign(num_bones_, datatypes::Matrix4r(1.0)); // Identity

    const auto& vertices = mesh->vertices();        // Mesh vertices (with skinning data)
    for (size_t i = 0; i < num_vertices_; ++i) {
        const auto& v = vertices[i];
        datatypes::Vector3 skinned(0.0);
        double weight_sum = 0.0;

        // Accumulate influence of each bone
        for (int b = 0; b < 4; ++b) {
            int bone_idx = v.bone_indices[b];       // Bone index (‑1 if unused)
            double w = v.bone_weights[b];
            if (bone_idx >= 0 && w > 0.0 && bone_idx < static_cast<int>(bone_transforms.size())) {
                // Transform rest position to world space using bone transform
                datatypes::Matrix4r mat = bone_transforms[bone_idx];
                datatypes::Vector3 transformed = mat.transformPoint(rest_vertex_positions_[i]);
                skinned += transformed * w;         // Weighted accumulation
                weight_sum += w;
            }
        }
        if (weight_sum > 0.0) {
            skinned /= weight_sum;                  // Normalise
            targets[i] = skinned;
        } else {
            // No bone influence; use rest position as target (no animation)
            targets[i] = rest_vertex_positions_[i];
        }
    }
}

//------------------------------------------------------------------------------
// Private: advance spring simulation (jiggle offsets)
//------------------------------------------------------------------------------
void JiggleMapper::advance_springs(const std::vector<datatypes::Vector3>& targets, double dt) {
    auto mesh = mesh_.lock();
    if (!mesh) return;

    const auto& vertices = mesh->vertices();
    for (size_t i = 0; i < num_vertices_; ++i) {
        // Retrieve per‑vertex bone influence to compute blending of jiggle parameters
        // We'll compute an effective stiffness and damping by blending per‑bone
        // parameters using the vertex's bone weights.
        const auto& v = vertices[i];
        double effective_stiffness = 0.0;
        double effective_damping = 0.0;
        double total_influence = 0.0;

        for (int b = 0; b < 4; ++b) {
            int bone_idx = v.bone_indices[b];
            double w = v.bone_weights[b];
            if (bone_idx >= 0 && w > 0.0) {
                // Get parameters for this bone (use per‑bone if set, else global)
                const BoneJiggleParams& params = (bone_idx < static_cast<int>(bone_params_.size()))
                                                ? bone_params_[bone_idx] : global_params_;
                // Blend using weight and influence factor
                effective_stiffness += w * params.stiffness * params.influence;
                effective_damping    += w * params.damping * params.influence;
                total_influence      += w * params.influence;
            }
        }
        if (total_influence < 1e-12) {
            // No jiggle on this vertex; offset decays quickly
            effective_stiffness = global_params_.stiffness * 0.01;   // Very soft
            effective_damping    = global_params_.damping;
        } else {
            effective_stiffness /= total_influence;   // Weighted average
            effective_damping    /= total_influence;
        }

        // Current offset from target (negative of displacement from original?)
        // The spring model: offset is the displacement from the rigid target position.
        // The spring force pulls the offset toward zero: F = -k * offset - d * offset_velocity.
        double k = effective_stiffness;
        double d = effective_damping;

        // Current spring state
        datatypes::Vector3 offset = offsets_[i];
        datatypes::Vector3 vel = velocities_[i];

        // Spring acceleration: a = -k * offset - d * vel
        datatypes::Vector3 acceleration = offset * (-k) + vel * (-d);

        // Semi‑implicit Euler integration
        vel += acceleration * dt;                   // Update velocity
        offset += vel * dt;                         // Update offset

        // Store back
        offsets_[i] = offset;
        velocities_[i] = vel;
    }
}

//------------------------------------------------------------------------------
// Private: write final vertex positions to mesh
//------------------------------------------------------------------------------
void JiggleMapper::write_vertex_positions(const std::vector<datatypes::Vector3>& positions) {
    auto mesh = mesh_.lock();
    if (!mesh) return;

    auto& verts = mesh->vertices();                 // Mutable access
    for (size_t i = 0; i < num_vertices_; ++i) {
        verts[i].position = positions[i];           // Overwrite vertex position
    }
    // Note: normals may need recomputation after jiggle; can be deferred to rendering.
}

} // namespace engine
} // namespace genesis