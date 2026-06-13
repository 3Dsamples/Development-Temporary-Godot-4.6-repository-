// genesis/engine/entities/skinned_entity.h
#pragma once

#include "genesis/datatypes.h"                     // Matrix4r
#include <vector>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// ISkinnedEntity – mix‑in interface for entities that have a skeleton.
// Provides bone transforms that are used by JiggleMapper and other skinning
// tools.
//------------------------------------------------------------------------------
class ISkinnedEntity {
public:
    virtual ~ISkinnedEntity() = default;

    // Number of bones in the skeleton
    virtual size_t bone_count() const = 0;

    // Current world‑space bone transforms (one per bone in order)
    virtual std::vector<datatypes::Matrix4r> bone_transforms() const = 0;

    // Optional: rest‑pose bone transforms (if needed for skinning)
    virtual std::vector<datatypes::Matrix4r> bind_pose_transforms() const = 0;
};

} // namespace engine
} // namespace genesis