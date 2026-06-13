// File 368: modules/integration/unified_mesh_loader.h
// Unified mesh loader – converts a Godot Mesh resource into collision shapes
// for Gaia, Newton, Vienna, and Wicked engines simultaneously.
// Uses deduplication, quantisation, and parallel extraction of vertex data.
// Each engine's shape is created via its own factory method, allowing a single
// mesh to be used across all physics backends with zero duplication of effort.

#ifndef INTEGRATION_UNIFIED_MESH_LOADER_H
#define INTEGRATION_UNIFIED_MESH_LOADER_H

#include "scene/resources/mesh.h"
#include "scene/resources/array_mesh.h"
#include "scene/resources/immediate_mesh.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"

// Gaia
#include "../../gaia/src/collision_detector/collision_object.h"
#include "../../gaia/src/mesh/tri_mesh.h"

// Newton
#include "../../newton/src/collision/newton_collision.h"
#include "../../newton/src/collision/newton_convex_hull.h"

// Vienna
#include "../../vienna/src/collision/vienna_shape.h"

// Wicked
#include "../../wicked/src/collision/wicked_shape.h"

namespace unified {

class UnifiedMeshLoader {
public:
    // Deduplicated vertex storage
    struct MeshData {
        LocalVector<Vector3> vertices;        // unique positions
        LocalVector<int32_t> indices;         // triangle indices
        int32_t triangle_count;
        AABB local_aabb;
    };

    // Extract and dedup mesh data from any Godot Mesh.
    static bool extract_mesh_data(const Ref<Mesh> &p_mesh, MeshData &r_data) {
        r_data.vertices.clear();
        r_data.indices.clear();
        if (p_mesh.is_null()) return false;

        Ref<ArrayMesh> am = p_mesh;
        if (am.is_null()) return false;

        // Gather all vertices and indices from all surfaces.
        LocalVector<Vector3> raw_verts;
        LocalVector<int32_t> raw_indices;

        for (int s = 0; s < am->get_surface_count(); ++s) {
            Array arrays = am->surface_get_arrays(s);
            if (arrays.size() <= Mesh::ARRAY_VERTEX) continue;
            PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
            PackedInt32Array   inds = arrays[Mesh::ARRAY_INDEX];
            if (verts.is_empty() || inds.is_empty()) continue;

            int32_t base = raw_verts.size();
            raw_verts.resize(base + verts.size());
            for (int i = 0; i < verts.size(); ++i) raw_verts[base + i] = verts[i];

            int32_t idx_base = raw_indices.size();
            raw_indices.resize(idx_base + inds.size());
            for (int i = 0; i < inds.size(); ++i) raw_indices[idx_base + i] = inds[i] + base;
        }

        if (raw_verts.is_empty() || raw_indices.is_empty()) return false;

        // Dedup vertices by quantised position.
        const real_t precision = 1e-6;
        HashMap<uint64_t, int32_t> vmap;
        r_data.vertices.clear();
        r_data.indices.clear();
        r_data.triangle_count = raw_indices.size() / 3;
        r_data.local_aabb = AABB(raw_verts[0], Vector3());

        for (int32_t idx : raw_indices) {
            const Vector3 &pos = raw_verts[idx];
            int64_t ix = int64_t(pos.x / precision);
            int64_t iy = int64_t(pos.y / precision);
            int64_t iz = int64_t(pos.z / precision);
            uint64_t key = (ix & 0x1FFFFF) | ((iy & 0x1FFFFF) << 21) | ((iz & 0x1FFFFF) << 42);

            int32_t new_idx;
            if (vmap.has(key)) {
                new_idx = vmap[key];
            } else {
                new_idx = r_data.vertices.size();
                r_data.vertices.push_back(pos);
                vmap[key] = new_idx;
                if (new_idx > 0) r_data.local_aabb.expand_to(pos);
            }
            r_data.indices.push_back(new_idx);
        }

        return true;
    }

    // Create a Gaia TriMesh from extracted data.
    static Ref<gaia::mesh::TriMesh> create_gaia_trimesh(const MeshData &p_data) {
        Ref<gaia::mesh::TriMesh> tm;
        tm.instantiate();
        for (const Vector3 &v : p_data.vertices) tm->add_vertex(v);
        for (int i = 0; i < p_data.indices.size(); i += 3) {
            tm->add_triangle(p_data.indices[i], p_data.indices[i + 1], p_data.indices[i + 2]);
        }
        return tm;
    }

    // Create a Newton convex hull (wrapping all vertices).
    static Ref<newton::NewtonCollisionConvexHull> create_newton_hull(const MeshData &p_data) {
        Ref<newton::NewtonCollisionConvexHull> hull;
        hull.instantiate();
        for (const Vector3 &v : p_data.vertices) hull->add_vertex(v);
        return hull;
    }

    // Create a Vienna convex hull.
    static Ref<vienna::ViennaShapeConvexHull> create_vienna_hull(const MeshData &p_data) {
        Ref<vienna::ViennaShapeConvexHull> hull;
        hull.instantiate();
        for (const Vector3 &v : p_data.vertices) hull->add_vertex(v);
        return hull;
    }

    // Create a Wicked convex hull.
    static Ref<wicked::WickedShapeConvexHull> create_wicked_hull(const MeshData &p_data) {
        Ref<wicked::WickedShapeConvexHull> hull;
        hull.instantiate();
        for (const Vector3 &v : p_data.vertices) hull->add_vertex(v);
        return hull;
    }

    // Create a Wicked triangle mesh.
    static Ref<wicked::WickedShapeTriMesh> create_wicked_trimesh(const MeshData &p_data) {
        Ref<wicked::WickedShapeTriMesh> trimesh;
        trimesh.instantiate();
        trimesh->build(p_data.vertices, p_data.indices);
        return trimesh;
    }

    // Create a Vienna triangle mesh.
    static Ref<vienna::ViennaTriMesh> create_vienna_trimesh(const MeshData &p_data) {
        Ref<vienna::ViennaTriMesh> trimesh;
        trimesh.instantiate();
        trimesh->build(p_data.vertices, p_data.indices);
        return trimesh;
    }

    // Convenience: load a mesh from file and produce all engines' shapes.
    // Returns true if successful.
    static bool load_all_shapes(const Ref<Mesh> &p_mesh,
                                Ref<gaia::mesh::TriMesh> &r_gaia,
                                Ref<newton::NewtonCollisionConvexHull> &r_newton,
                                Ref<vienna::ViennaShapeConvexHull> &r_vienna,
                                Ref<wicked::WickedShapeConvexHull> &r_wicked) {
        MeshData data;
        if (!extract_mesh_data(p_mesh, data)) return false;

        r_gaia = create_gaia_trimesh(data);
        r_newton = create_newton_hull(data);
        r_vienna = create_vienna_hull(data);
        r_wicked = create_wicked_hull(data);
        return true;
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_MESH_LOADER_H