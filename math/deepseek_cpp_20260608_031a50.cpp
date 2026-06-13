// File 349: modules/wicked/src/utils/wicked_mesh_loader.h
// High‑performance utility to convert Godot Mesh resources into WickedShape
// collision structures (convex hull and triangle mesh).  Uses Godot's
// ArrayMesh surface access, vertex deduplication, and optional convex
// decomposition placeholders.  All extraction methods are static and inline
// where appropriate for speed.

#ifndef WICKED_UTILS_MESH_LOADER_H
#define WICKED_UTILS_MESH_LOADER_H

#include "scene/resources/mesh.h"
#include "scene/resources/array_mesh.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../collision/wicked_shape.h"
#include "../core/wicked_types.h"

namespace wicked {

class WickedMeshLoader {
public:
    // Extract vertex positions from a triangulated Godot Mesh.
    // Returns true if at least one vertex was found.
    static bool extract_vertices(const Ref<Mesh> &p_mesh, LocalVector<vec3> &r_vertices) {
        r_vertices.clear();
        if (p_mesh.is_null()) return false;

        Ref<ArrayMesh> am = p_mesh;
        if (am.is_valid()) {
            for (int s = 0; s < am->get_surface_count(); ++s) {
                Array arrays = am->surface_get_arrays(s);
                if (arrays.size() <= Mesh::ARRAY_VERTEX) continue;
                PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
                int old_size = r_vertices.size();
                r_vertices.resize(old_size + verts.size());
                for (int i = 0; i < verts.size(); ++i) {
                    r_vertices[old_size + i] = verts[i];
                }
            }
            return !r_vertices.is_empty();
        }
        return false;
    }

    // Extract triangle indices (three per triangle, flat).
    static bool extract_indices(const Ref<Mesh> &p_mesh, LocalVector<int> &r_indices) {
        r_indices.clear();
        if (p_mesh.is_null()) return false;

        Ref<ArrayMesh> am = p_mesh;
        if (am.is_valid()) {
            for (int s = 0; s < am->get_surface_count(); ++s) {
                Array arrays = am->surface_get_arrays(s);
                if (arrays.size() <= Mesh::ARRAY_INDEX) continue;
                PackedInt32Array idx = arrays[Mesh::ARRAY_INDEX];
                if (idx.is_empty()) continue;
                int old_size = r_indices.size();
                r_indices.resize(old_size + idx.size());
                for (int i = 0; i < idx.size(); ++i) {
                    r_indices[old_size + i] = idx[i];
                }
            }
            return !r_indices.is_empty();
        }
        return false;
    }

    // Build a convex hull collision shape from all vertices of the mesh.
    // The mesh should be convex; otherwise the resulting hull may not match.
    static Ref<WickedShapeConvexHull> create_convex_hull(const Ref<Mesh> &p_mesh) {
        Ref<WickedShapeConvexHull> hull;
        hull.instantiate();
        LocalVector<vec3> verts;
        if (extract_vertices(p_mesh, verts)) {
            for (const vec3 &v : verts) {
                hull->add_vertex(v);
            }
        }
        return hull;
    }

    // Build a triangle mesh collision shape (static mesh) from the given mesh.
    // Uses deduplication of vertices via quantisation to reduce memory.
    static Ref<WickedShapeTriMesh> create_triangle_mesh(const Ref<Mesh> &p_mesh) {
        Ref<WickedShapeTriMesh> trimesh;
        trimesh.instantiate();
        LocalVector<vec3> verts;
        LocalVector<int> inds;
        if (!extract_vertices(p_mesh, verts) || !extract_indices(p_mesh, inds)) {
            return trimesh;
        }

        // Deduplicate vertices
        HashMap<uint64_t, int> vmap; // key = quantised position, value = new index
        LocalVector<vec3> unique_verts;
        LocalVector<int> new_indices;
        new_indices.reserve(inds.size());
        const real_t precision = 1e-6;
        auto quant_key = [&](const vec3 &v) -> uint64_t {
            int64_t ix = int64_t(v.x / precision);
            int64_t iy = int64_t(v.y / precision);
            int64_t iz = int64_t(v.z / precision);
            return (ix & 0x1FFFFF) | ((iy & 0x1FFFFF) << 21) | ((iz & 0x1FFFFF) << 42);
        };

        for (int i = 0; i < inds.size(); ++i) {
            int orig_idx = inds[i];
            const vec3 &pos = verts[orig_idx];
            uint64_t key = quant_key(pos);
            int *found = vmap.getptr(key);
            if (found) {
                new_indices.push_back(*found);
            } else {
                int new_idx = unique_verts.size();
                unique_verts.push_back(pos);
                vmap[key] = new_idx;
                new_indices.push_back(new_idx);
            }
        }
        trimesh->build(unique_verts, new_indices);
        return trimesh;
    }
};

} // namespace wicked

#endif // WICKED_UTILS_MESH_LOADER_H