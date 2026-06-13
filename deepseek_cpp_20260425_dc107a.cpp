// File 38: modules/gaia/src/types/matrix_types.h

#ifndef GAIA_TYPES_MATRIX_TYPES_H
#define GAIA_TYPES_MATRIX_TYPES_H

#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/math/projection.h"

// ---------------------------------------------------------------------------
// Matrix type aliases for Gaia's original float3x3, float4x4, etc.
// Mapped to Godot's Basis (3x3), Transform3D (4x4 with origin),
// and Projection (4x4 for general transforms).
// ---------------------------------------------------------------------------
namespace gaia::types {

	// 3x3 matrices
	using mat3 = Basis;
	using float3x3 = Basis;
	using double3x3 = Basis;   // real_t precision determined by engine build

	// 4x4 matrices (with translation)
	using mat4 = Transform3D;
	using float4x4 = Transform3D;
	using double4x4 = Transform3D;

	// A 4x4 affine matrix (transform without perspective) – same as Transform3D
	using affine4x4 = Transform3D;

	// A general 4x4 matrix (can contain projection) – aliased to Projection
	using general4x4 = Projection;

} // namespace gaia::types

#endif // GAIA_TYPES_MATRIX_TYPES_H