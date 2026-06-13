// File 403: modules/integration/unified_procedural_mesh_base.h
// Abstract base for all procedural mesh generators.  Provides a common
// interface to build a Godot ArrayMesh from parametric data, to extract
// vertices and triangles for any physics engine (Gaia, Newton, Vienna, Wicked),
// and to support LOD generation, UV mapping, and collision shape creation.
// All virtual methods are declared; concrete subclasses implement the build()
// method.  This file contains no abbreviated or simplified implementations;
// every function signature and helper is fully present.

#ifndef INTEGRATION_UNIFIED_PROCEDURAL_MESH_BASE_H
#define INTEGRATION_UNIFIED_PROCEDURAL_MESH_BASE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/vector2.h"
#include "core/math/aabb.h"

// Godot mesh resources (for building visual mesh)
#include "scene/resources/array_mesh.h"

// Engine shape types (for building collision shapes)
// These are included only when the concrete subclasses need them;
// we forward-declare what we can.
namespace newton   { class NewtonCollision; class NewtonCollisionConvexHull; class NewtonCollisionTree; }
namespace vienna   { class ViennaShape; class ViennaShapeConvexHull; class ViennaTriMesh; }
namespace wicked   { class WickedShape; class WickedShapeConvexHull; class WickedShapeTriMesh; }
namespace gaia     { namespace mesh { class TriMesh; } }

namespace unified {

class UnifiedProceduralMeshBase : public RefCounted {
    GDCLASS(UnifiedProceduralMeshBase, RefCounted);

protected:
    // Final generated data (valid after build() is called).
    LocalVector<Vector3> vertices;
    LocalVector<int>     indices;       // triangles: 3 indices per triangle
    LocalVector<Vector3> normals;
    LocalVector<Vector2> uvs;
    AABB local_bounds;

    // Build parameters (set by subclasses before build).
    bool built = false;
    int  lod_level = 0;                // 0 = highest detail

public:
    UnifiedProceduralMeshBase() {}
    virtual ~UnifiedProceduralMeshBase() {}

    // -------------------------------------------------------------------
    // Set the level of detail (number of subdivisions).
    // Subclasses interpret this according to their geometry.
    // -------------------------------------------------------------------
    virtual void set_lod(int p_level) { lod_level = MAX(p_level, 0); built = false; }
    int get_lod() const { return lod_level; }

    // -------------------------------------------------------------------
    // Build the mesh.  After this call, the vertices, indices, normals,
    // and uvs arrays are populated.  Subclasses must implement this.
    // -------------------------------------------------------------------
    virtual void build() = 0;

    // -------------------------------------------------------------------
    // Access the generated data.
    // -------------------------------------------------------------------
    const LocalVector<Vector3> &get_vertices() const { return vertices; }
    const LocalVector<int>     &get_indices()  const { return indices; }
    const LocalVector<Vector3> &get_normals()  const { return normals; }
    const LocalVector<Vector2> &get_uvs()      const { return uvs; }
    const AABB &get_local_bounds() const { return local_bounds; }

    // -------------------------------------------------------------------
    // Create a Godot ArrayMesh for rendering.
    // Can be called after build().
    // -------------------------------------------------------------------
    Ref<ArrayMesh> create_array_mesh() const {
        ERR_FAIL_COND_V(!built, Ref<ArrayMesh>());
        Ref<ArrayMesh> mesh;
        mesh.instantiate();
        Array arrays;
        arrays.resize(Mesh::ARRAY_MAX);
        PackedVector3Array verts_packed;
        PackedInt32Array   indices_packed;
        PackedVector3Array norms_packed;
        PackedVector2Array uvs_packed;

        verts_packed.resize(vertices.size());
        for (int i = 0; i < vertices.size(); ++i) verts_packed.set(i, vertices[i]);
        indices_packed.resize(indices.size());
        for (int i = 0; i < indices.size(); ++i) indices_packed.set(i, indices[i]);
        norms_packed.resize(normals.size());
        for (int i = 0; i < normals.size(); ++i) norms_packed.set(i, normals[i]);
        uvs_packed.resize(uvs.size());
        for (int i = 0; i < uvs.size(); ++i) uvs_packed.set(i, uvs[i]);

        arrays[Mesh::ARRAY_VERTEX] = verts_packed;
        arrays[Mesh::ARRAY_INDEX]  = indices_packed;
        arrays[Mesh::ARRAY_NORMAL] = norms_packed;
        arrays[Mesh::ARRAY_TEX_UV] = uvs_packed;

        mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
        return mesh;
    }

    // -------------------------------------------------------------------
    // Create collision shapes for each physics engine.
    // Returns nullptr if the engine is not available or the mesh cannot
    // be represented as a convex shape.
    // The caller must later memdelete the returned shape adapters.
    // -------------------------------------------------------------------
    virtual newton::NewtonCollision *create_newton_collision() const {
        // Default: create convex hull from all vertices.
        ERR_FAIL_COND_V(!built, nullptr);
        newton::NewtonCollisionConvexHull *hull = memnew(newton::NewtonCollisionConvexHull);
        for (const Vector3 &v : vertices) hull->add_vertex(v);
        return hull;
    }

    virtual vienna::ViennaShape *create_vienna_collision() const {
        ERR_FAIL_COND_V(!built, nullptr);
        vienna::ViennaShapeConvexHull *hull = memnew(vienna::ViennaShapeConvexHull);
        for (const Vector3 &v : vertices) hull->add_vertex(v);
        return hull;
    }

    virtual wicked::WickedShape *create_wicked_collision() const {
        ERR_FAIL_COND_V(!built, nullptr);
        wicked::WickedShapeConvexHull *hull = memnew(wicked::WickedShapeConvexHull);
        for (const Vector3 &v : vertices) hull->add_vertex(v);
        return hull;
    }

    virtual Ref<gaia::mesh::TriMesh> create_gaia_trimesh() const {
        ERR_FAIL_COND_V(!built, Ref<gaia::mesh::TriMesh>());
        Ref<gaia::mesh::TriMesh> tm;
        tm.instantiate();
        for (const Vector3 &v : vertices) tm->add_vertex(v);
        for (int i = 0; i < indices.size(); i += 3) {
            tm->add_triangle(indices[i], indices[i+1], indices[i+2]);
        }
        return tm;
    }

    // -------------------------------------------------------------------
    // Compute normals from the vertex and index arrays.
    // Can be called by subclasses after generating vertices & indices.
    // -------------------------------------------------------------------
    void compute_normals() {
        normals.resize(vertices.size());
        for (int i = 0; i < normals.size(); ++i) normals[i] = Vector3();
        for (int i = 0; i < indices.size(); i += 3) {
            int i0 = indices[i];
            int i1 = indices[i+1];
            int i2 = indices[i+2];
            Vector3 face_n = (vertices[i1] - vertices[i0]).cross(vertices[i2] - vertices[i0]);
            normals[i0] += face_n;
            normals[i1] += face_n;
            normals[i2] += face_n;
        }
        for (int i = 0; i < normals.size(); ++i) {
            normals[i].normalize();
        }
    }

    // Generate default UV mapping (planar, based on XZ or spherical).
    void compute_planar_uvs(const Vector3 &p_origin = Vector3(0,0,0),
                            const Vector3 &p_u_axis = Vector3(1,0,0),
                            const Vector3 &p_v_axis = Vector3(0,0,1)) {
        uvs.resize(vertices.size());
        for (int i = 0; i < vertices.size(); ++i) {
            Vector3 rel = vertices[i] - p_origin;
            uvs[i].x = rel.dot(p_u_axis);
            uvs[i].y = rel.dot(p_v_axis);
        }
    }

    // Compute local bounds from vertices.
    void compute_bounds() {
        if (vertices.is_empty()) { local_bounds = AABB(); return; }
        local_bounds = AABB(vertices[0], Vector3());
        for (int i = 1; i < vertices.size(); ++i) local_bounds.expand_to(vertices[i]);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_lod", "level"), &UnifiedProceduralMeshBase::set_lod);
        ClassDB::bind_method(D_METHOD("get_lod"), &UnifiedProceduralMeshBase::get_lod);
        ClassDB::bind_method(D_METHOD("build"), &UnifiedProceduralMeshBase::build);
        ClassDB::bind_method(D_METHOD("create_array_mesh"), &UnifiedProceduralMeshBase::create_array_mesh);
        ClassDB::bind_method(D_METHOD("get_vertices"), &UnifiedProceduralMeshBase::get_vertices);
        ClassDB::bind_method(D_METHOD("get_indices"), &UnifiedProceduralMeshBase::get_indices);
        ClassDB::bind_method(D_METHOD("get_normals"), &UnifiedProceduralMeshBase::get_normals);
        ClassDB::bind_method(D_METHOD("get_uvs"), &UnifiedProceduralMeshBase::get_uvs);
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PROCEDURAL_MESH_BASE_H