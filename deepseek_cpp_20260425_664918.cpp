// File 37: modules/gaia/src/types/vector_types.h

#ifndef GAIA_TYPES_VECTOR_TYPES_H
#define GAIA_TYPES_VECTOR_TYPES_H

#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/vector4.h"

// ---------------------------------------------------------------------------
// Vector type aliases for Gaia's original float3, double3, int3, float4, etc.
// These map directly to Godot's native Vector3, Vector3i and Vector4 types.
// ---------------------------------------------------------------------------
namespace gaia::types {

	// -- floating point vectors --
	using float3 = Vector3;
	using double3 = Vector3;      // double precision is controlled by Godot's real_t / build setting
	using float4 = Vector4;
	using double4 = Vector4;

	// -- integer vectors --
	using int3 = Vector3i;
	using uint3 = Vector3i;       // Godot's Vector3i uses int32_t (signed)

	// -- type traits mapping scalar -> vector (used e.g. in old gaia::aabb) --
	template <typename T>
	struct vector_of;

	template <>
	struct vector_of<float> { using type = float3; };

	template <>
	struct vector_of<double> { using type = double3; };

	template <>
	struct vector_of<int32_t> { using type = int3; };

	template <>
	struct vector_of<uint32_t> { using type = uint3; };

} // namespace gaia::types

#endif // GAIA_TYPES_VECTOR_TYPES_H