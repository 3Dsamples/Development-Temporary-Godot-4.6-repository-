// File 173: modules/newton/src/core/newton_types.h
// Core type aliases for Newton Dynamics to map to Godot's math types.
// Provides shorthand names used throughout the module.

#ifndef NEWTON_CORE_TYPES_H
#define NEWTON_CORE_TYPES_H

#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/vector4.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/math/quaternion.h"
#include "core/typedefs.h"

namespace newton {

// Scalar
using real       = real_t;

// Vectors
using vec3       = Vector3;
using vec4       = Vector4;
using ivec3      = Vector3i;

// Matrices
using mat3       = Basis;          // 3×3 rotation / inertia
using mat4       = Transform3D;    // 4×4 affine transform (basis + origin)

// Quaternion
using quat       = Quaternion;

// Bounding volume
using aabb       = AABB;

// Handles
using body_id    = uint64_t;
using joint_id   = uint64_t;
using material_id = uint64_t;
using shape_id   = uint64_t;

// Engine limits
constexpr int MAX_CONTACTS_PER_MANIFOLD = 8;
constexpr int MAX_SOLVER_ITERATIONS     = 256;

} // namespace newton

#endif // NEWTON_CORE_TYPES_H