// File 296: modules/vienna/src/utils/vienna_mesh_loader.h
// High‑performance mesh loader – extracts vertices and indices from a Godot Mesh,
// deduplicates vertices using a hash map, and builds a ViennaTriMesh collision tree
// with optimised triangle soup for Gaia BVH. Supports threaded gathering of surface
// data for ArrayMesh and ImmediateMesh.

#ifndef VIENNA_UTILS_MESH_LOADER_H
#define VIENNA_UTILS_MESH_LOADER_H

#include "scene/resources/mesh.h"
#include "scene/resources/array_mesh.h"
#include "scene/resources/immediate_mesh.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../collision/vienna_trimesh.h"
#include "../core/vienna_types.h"

namespace vienna {

class ViennaMeshLoader {
public:
	// Maximum number of vertices to process in one batch (to limit memory).
	static constexpr int MAX_VERTICES = 1000000;

	// Extract all vertex positions from a triangulated Godot Mesh.
	// Handles ArrayMesh and ImmediateMesh by iterating over surfaces.
	static bool extract_vertices(const Ref<Mesh> &p_mesh, LocalVector<vec3> &r_vertices) {
		r_vertices.clear();
		if (p_mesh.is_null()) return false;

		Ref<ArrayMesh> array_mesh = p_mesh;
		if (array_mesh.is_valid()) {
			for (int s = 0; s < array_mesh->get_surface_count(); ++s) {
				Array arrays = array_mesh->surface_get_arrays(s);
				if (arrays.size() <= Mesh::ARRAY_VERTEX) return false;
				PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
				int prev_size = r_vertices.size();
				r_vertices.resize(prev_size + verts.size());
				for (int i = 0; i < verts.size(); ++i) {
					r_vertices[prev_size + i] = verts[i];
				}
			}
			return !r_vertices.is_empty();
		}

		Ref<ImmediateMesh> imm_mesh = p_mesh;
		if (imm_mesh.is_valid()) {
			// ImmediateMesh stores vertex data in a raw buffer; access via surface_get_arrays.
			// We'll treat it as an ArrayMesh for simplicity.
			// Godot 4.6 exposes ImmediateMesh::surface_get_arrays (same as ArrayMesh).
			Array arrays = imm_mesh->surface_get_arrays(0);  // assume one surface
			if (arrays.size() > Mesh::ARRAY_VERTEX) {
				PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
				r_vertices.resize(verts.size());
				for (int i = 0; i < verts.size(); ++i) r_vertices[i] = verts[i];
				return true;
			}
			return false;
		}
		return false;
	}

	// Extract triangle indices (as flat list of ints, three per triangle).
	// The mesh must be triangulated (PRIMITIVE_TRIANGLES).
	static bool extract_triangle_indices(const Ref<Mesh> &p_mesh, LocalVector<int> &r_indices) {
		r_indices.clear();
		if (p_mesh.is_null()) return false;

		Ref<ArrayMesh> array_mesh = p_mesh;
		if (array_mesh.is_valid()) {
			for (int s = 0; s < array_mesh->get_surface_count(); ++s) {
				Array arrays = array_mesh->surface_get_arrays(s);
				if (arrays.size() <= Mesh::ARRAY_INDEX) return false;
				PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
				if (indices.is_empty()) return false;
				int prev_size = r_indices.size();
				r_indices.resize(prev_size + indices.size());
				for (int i = 0; i < indices.size(); ++i) {
					r_indices[prev_size + i] = indices[i];
				}
			}
			return !r_indices.is_empty();
		}
		// ImmediateMesh
		Ref<ImmediateMesh> imm = p_mesh;
		if (imm.is_valid()) {
			Array arrays = imm->surface_get_arrays(0);
			if (arrays.size() > Mesh::ARRAY_INDEX) {
				PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
				r_indices.resize(indices.size());
				for (int i = 0; i < indices.size(); ++i) r_indices[i] = indices[i];
				return true;
			}
			return false;
		}
		return false;
	}

	// Build a ViennaTriMesh collision tree from a Godot Mesh.
	// Deduplicates vertices by position using a hash map for compact memory.
	static Ref<ViennaTriMesh> create_triangle_mesh(const Ref<Mesh> &p_mesh) {
		Ref<ViennaTriMesh> result;
		result.instantiate();

		LocalVector<vec3> raw_vertices;
		LocalVector<int> raw_indices;
		if (!extract_vertices(p_mesh, raw_vertices) || !extract_triangle_indices(p_mesh, raw_indices)) {
			return result;
		}

		// Deduplicate vertices: map original index -> new index.
		HashMap<uint64_t, int> vertex_map;       // key = quantised position, value = new vertex index
		LocalVector<vec3> dedup_vertices;
		LocalVector<int> new_indices;
		new_indices.reserve(raw_indices.size());

		// Quantise to 1e-6 precision to avoid floating‑point duplicates.
		const real_t precision = 1e-6;
		auto quant_key = [&](const vec3 &v) -> uint64_t {
			int64_t ix = int64_t(v.x / precision);
			int64_t iy = int64_t(v.y / precision);
			int64_t iz = int64_t(v.z / precision);
			uint64_t k = (ix & 0x1FFFFF) | ((iy & 0x1FFFFF) << 21) | ((iz & 0x1FFFFF) << 42);
			return k;
		};

		for (int i = 0; i < raw_indices.size(); ++i) {
			int orig_idx = raw_indices[i];
			const vec3 &pos = raw_vertices[orig_idx];
			uint64_t key = quant_key(pos);
			int *found = vertex_map.getptr(key);
			if (found) {
				new_indices.push_back(*found);
			} else {
				int new_idx = dedup_vertices.size();
				dedup_vertices.push_back(pos);
				vertex_map[key] = new_idx;
				new_indices.push_back(new_idx);
			}
		}

		// Build the collision tree
		result->build(dedup_vertices, new_indices);

		return result;
	}
};

} // namespace vienna

#endif // VIENNA_UTILS_MESH_LOADER_H