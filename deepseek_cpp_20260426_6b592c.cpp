// File 172: modules/newton/src/core/newton_types.h
// Newton Dynamics type aliases adapted to Godot's math foundation.
// Provides Newton-specific vector and matrix representations for high‑performance physics.

#ifndef NEWTON_CORE_NEWTON_TYPES_H
#define NEWTON_CORE_NEWTON_TYPES_H

#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/quaternion.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace newton {

// ----- Scalars -----
using real  = real_t;                      // Godot's real_t (double or float)
using u32   = uint32_t;
using i32   = int32_t;

// ----- Vectors -----
using Vec3  = Vector3;                     // Newton uses Vec3 for positions, velocities, forces
using Vec3i = Vector3i;

// ----- Matrices -----
using Mat3  = Basis;                       // 3x3 rotation / inertia tensor
using Mat4  = Transform3D;                 // 4x4 affine transformation (rigid body)

// ----- Quaternion -----
using Quat  = Quaternion;                  // Orientation representation

// ----- Bounding box -----
using AABB  = ::AABB;                      // Godot's AABB

// ----- Constants -----
constexpr real EPSILON      = CMP_EPSILON;
constexpr real INF_REAL     = INFINITY;
constexpr real PI           = Math_PI;
constexpr real GRAVITY_EARTH = 9.80665;

} // namespace newton

#endif // NEWTON_CORE_NEWTON_TYPES_H