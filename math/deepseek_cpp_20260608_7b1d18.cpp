// File 327: modules/wicked/src/core/wicked_types.h
// Core type aliases for the WickedEngine physics module, mapped to Godot's math types.
// Uses real_t precision and standard Godot vector/matrix types for native integration.

#ifndef WICKED_CORE_TYPES_H
#define WICKED_CORE_TYPES_H

#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/math/quaternion.h"
#include "core/typedefs.h"

namespace wicked {

// Scalar type (matches Godot's real_t)
using real = real_t;

// Vectors
using vec3 = Vector3;
using vec4 = Vector4;
using ivec3 = Vector3i;

// Matrices
using mat3 = Basis;            // 3x3 rotation / inertia tensor
using mat4 = Transform3D;      // 4x4 affine transform (basis + origin)

// Quaternion
using quat = Quaternion;

// Bounding volume
using aabb = AABB;

// Handles for bodies, joints, materials, shapes
using body_id      = uint64_t;
using joint_id     = uint64_t;
using material_id  = uint64_t;
using shape_id     = uint64_t;
using vehicle_id   = uint64_t;

// Solver limits
constexpr int MAX_CONTACTS_PER_MANIFOLD = 8;
constexpr int MAX_SOLVER_ITERATIONS     = 256;

} // namespace wicked

#endif // WICKED_CORE_TYPES_H