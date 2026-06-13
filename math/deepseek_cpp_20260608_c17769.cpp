// File 268: modules/vienna/src/core/vienna_types.h
// Type aliases for ViennaPhysicsEngine, mapped to Godot's math types.

#ifndef VIENNA_CORE_TYPES_H
#define VIENNA_CORE_TYPES_H

#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/math/quaternion.h"
#include "core/typedefs.h"

namespace vienna {

// Scalar
using real   = real_t;

// Vectors
using vec3   = Vector3;
using vec4   = Vector4;
using ivec3  = Vector3i;

// Matrices
using mat3   = Basis;               // 3x3 rotation / inertia
using mat4   = Transform3D;         // 4x4 affine transform

// Quaternion
using quat   = Quaternion;

// Bounding volume
using aabb   = AABB;

// Handles
using body_id    = uint64_t;
using joint_id   = uint64_t;
using material_id = uint64_t;
using shape_id   = uint64_t;
using cloth_id   = uint64_t;

// Particle index
using particle_index = int32_t;

// Maximum constraints and iterations
constexpr int MAX_CONSTRAINTS_PER_BODY = 32;
constexpr int MAX_SOLVER_ITERATIONS     = 256;

} // namespace vienna

#endif // VIENNA_CORE_TYPES_H