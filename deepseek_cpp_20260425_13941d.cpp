// File 51: modules/genesis/src/core/genesis_types.h
// Core type aliases mapping Genesis' data structures to Godot's math types.

#ifndef GENESIS_CORE_GENESIS_TYPES_H
#define GENESIS_CORE_GENESIS_TYPES_H

#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace genesis {

// --- Scalar and vector types ---
using real  = real_t;
using vec3  = Vector3;
using vec3i = Vector3i;
using quat  = Quaternion;   // Godot's default orientation

// --- Matrix types ---
using mat3 = Basis;                // 3x3 rotation/scaling matrix
using mat4 = Transform3D;         // 4x4 affine transformation (basis + origin)

// --- Bounding volume ---
using aabb_t = AABB;

// --- Indices ---
using idx_t = int32_t;

// --- Entity and particle IDs ---
using entity_id_t = uint64_t;

// --- Handy constants ---
constexpr real INF_REAL = INFINITY;
constexpr real EPS      = CMP_EPSILON;

// --- Structures that mirror Genesis' core containers (adapted to Godot-friendly names) ---

// A generic array view that wraps a span (for external memory)
template <typename T>
struct ArrayView {
	const T *data;
	int64_t  size;

	ArrayView() : data(nullptr), size(0) {}
	ArrayView(const T *p_data, int64_t p_size) : data(p_data), size(p_size) {}

	const T &operator[](int64_t i) const { return data[i]; }
};

// A mutable view used inside solvers
template <typename T>
struct MutableArrayView {
	T *data;
	int64_t size;

	MutableArrayView() : data(nullptr), size(0) {}
	MutableArrayView(T *p_data, int64_t p_size) : data(p_data), size(p_size) {}

	T &operator[](int64_t i) { return data[i]; }
	const T &operator[](int64_t i) const { return data[i]; }
};

} // namespace genesis

#endif // GENESIS_CORE_GENESIS_TYPES_H