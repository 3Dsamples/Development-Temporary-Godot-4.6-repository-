// genesis/engine/entities/jiggle_mapper.cpp
#include "genesis/engine/entities/jiggle_mapper.h"
#include "genesis/engine/entities/base_entity.h"
#include "genesis/engine/entities/skinned_entity.h"  // <-- NEW
#include "genesis/engine/mesh.h"
#include <algorithm>
#include <cmath>

namespace genesis {
namespace engine {

// ... (bind/other methods unchanged) ...

void JiggleMapper::compute_target_positions(std::vector<datatypes::Vector3>& targets) const {
    auto entity = entity_.lock();
    if (!entity) {
        targets = rest_vertex_positions_;
        return;
    }

    auto mesh = mesh_.lock();
    if (!mesh) return;

    // Get bone transforms via the skinned entity interface
    auto* skinned = dynamic_cast<ISkinnedEntity*>(entity.get());
    std::vector<datatypes::Matrix4r> bone_transforms;
    if (skinned) {
        bone_transforms = skinned->bone_transforms(); // Real bone transforms
    } else {
        // No skeleton – fallback: identity for all bones
        bone_transforms.assign(num_bones_, datatypes::Matrix4r(1.0));
    }

    const auto& vertices = mesh->vertices();
    for (size_t i = 0; i < num_vertices_; ++i) {
        const auto& v = vertices[i];
        datatypes::Vector3 skinned(0.0);
        double weight_sum = 0.0;
        for (int b = 0; b < 4; ++b) {
            int bone_idx = v.bone_indices[b];
            double w = v.bone_weights[b];
            if (bone_idx >= 0 && w > 0.0 && bone_idx < static_cast<int>(bone_transforms.size())) {
                skinned += bone_transforms[bone_idx].transformPoint(rest_vertex_positions_[i]) * w;
                weight_sum += w;
            }
        }
        targets[i] = (weight_sum > 0.0) ? (skinned / weight_sum) : rest_vertex_positions_[i];
    }
}

// ... (rest of file unchanged) ...