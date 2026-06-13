// File 252: modules/newton/src/utils/newton_mesh_loaders.h
// Utility functions to convert Godot Meshes into Newton collision shapes
// (convex hull and triangle mesh). Supports Mesh, ArrayMesh, and ImmediateMesh.

#ifndef NEWTON_UTILS_MESH_LOADERS_H
#define NEWTON_UTILS_MESH_LOADERS_H

#include "scene/resources/mesh.h"
#include "scene/resources/array_mesh.h"
#include "scene/resources/immediate_mesh.h"
#include "../collision/newton_convex_hull.h"
#include "../collision/newton_collision_tree.h"
#include "../core/newton_types.h"

namespace newton {

class NewtonMeshLoader {
public:
	// Extract all vertex positions from a Godot Mesh (triangulated).
	// Returns true if the mesh contains vertices.
	static bool extract_vertices(const Ref<Mesh> &p_mesh, LocalVector<vec3> &r_vertices);

	// Extract indices (triangles) from a Godot Mesh.
	// The mesh must be triangulated (i.e., PRIMITIVE_TRIANGLES).
	static bool extract_triangle_indices(const Ref<Mesh> &p_mesh, LocalVector<int> &r_indices);

	// Build a convex hull collision shape from the mesh's vertices.
	// A new NewtonCollisionConvexHull is created and filled.
	static Ref<NewtonCollisionConvexHull> create_convex_hull(const Ref<Mesh> &p_mesh);

	// Build a triangle‑mesh collision tree (BVH) from the mesh.
	// Suitable for static level geometry.
	static Ref<NewtonCollisionTree> create_triangle_mesh(const Ref<Mesh> &p_mesh);
};

} // namespace newton

#endif // NEWTON_UTILS_MESH_LOADERS_H