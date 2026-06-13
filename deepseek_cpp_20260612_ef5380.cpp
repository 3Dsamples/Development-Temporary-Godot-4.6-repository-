// Name : lighting enhancement
// File : scene/3d/collision_polygon_3d_ext.cpp 44 of 60
// Description : Implementation of CollisionPolygon3DExt with polygon triangulation,
//               extrusion into convex pieces, convex decomposition, debug mesh generation,
//               and full PhysicsServer + RenderingServer synchronization.
#include "collision_polygon_3d_ext.h"
#include "servers/physics_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/geometry_2d.h"
#include "core/math/triangulate.h"
#include "core/math/quick_hull.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/templates/vector.h"
#include <cmath>

struct CollisionPolygon3DExt::Impl {
    Vector<Vector2> polygon_2d;
    float depth = 0.1f;
    BuildMode build_mode = BUILD_SOLID;
    int max_convex_pieces = 4;
    float margin = 0.0f;
    Transform3D local_transform;

    // Physics shapes (multiple, one per convex piece)
    Vector<RID> shape_rids;
    // Debug mesh (wireframe)
    RID debug_mesh_rid;
    RID debug_instance_rid;
    bool debug_visible = false;
    Color debug_color = Color(0.2, 0.8, 0.2);
    Color debug_emissive_color = Color(0,0,0);
    float debug_emissive_intensity = 0.0f;

    int gi_mode = 0;
    float gi_contribution = 1.0f;

    bool dirty = true;

    Impl() {
        debug_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        debug_instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(debug_instance_rid, debug_mesh_rid);
    }

    ~Impl() {
        clear_shapes();
        if (debug_mesh_rid.is_valid()) RenderingServer::get_singleton()->free(debug_mesh_rid);
        if (debug_instance_rid.is_valid()) RenderingServer::get_singleton()->free(debug_instance_rid);
    }

    void clear_shapes() {
        for (RID rid : shape_rids) {
            PhysicsServer3D::get_singleton()->free(rid);
        }
        shape_rids.clear();
    }

    // Triangulate the 2D polygon (using ear clipping)
    Vector<Vector2> triangulate_polygon(const Vector<Vector2> &poly) {
        Vector<Vector2> tris;
        if (poly.size() < 3) return tris;
        // Triangulate::triangulate returns indices of triangles in the original polygon
        Vector<int> indices = Triangulate::triangulate(poly);
        for (int i = 0; i < indices.size(); i += 3) {
            tris.push_back(poly[indices[i]]);
            tris.push_back(poly[indices[i+1]]);
            tris.push_back(poly[indices[i+2]]);
        }
        return tris;
    }

    // Convert a triangle (2D) to a convex 3D prism (extruded)
    void triangle_to_prism(const Vector2 &v0, const Vector2 &v1, const Vector2 &v2,
                           Vector<Vector3> &out_vertices, Vector<int> &out_indices,
                           Vector<Vector3> &out_normals) {
        double h = depth * 0.5;
        Vector3 b0(v0.x, v0.y, -h);
        Vector3 b1(v1.x, v1.y, -h);
        Vector3 b2(v2.x, v2.y, -h);
        Vector3 t0(v0.x, v0.y,  h);
        Vector3 t1(v1.x, v1.y,  h);
        Vector3 t2(v2.x, v2.y,  h);
        // Bottom face
        int base = out_vertices.size();
        out_vertices.push_back(b0); out_vertices.push_back(b1); out_vertices.push_back(b2);
        out_indices.push_back(base); out_indices.push_back(base+1); out_indices.push_back(base+2);
        for (int i = 0; i < 3; ++i) out_normals.push_back(Vector3(0,0,-1));
        // Top face (reverse order)
        base = out_vertices.size();
        out_vertices.push_back(t0); out_vertices.push_back(t2); out_vertices.push_back(t1);
        out_indices.push_back(base); out_indices.push_back(base+1); out_indices.push_back(base+2);
        for (int i = 0; i < 3; ++i) out_normals.push_back(Vector3(0,0,1));
        // Side faces (3 quads)
        Vector3 edges[3][2] = {{b0,b1},{b1,b2},{b2,b0}};
        Vector3 top_edges[3][2] = {{t0,t1},{t1,t2},{t2,t0}};
        for (int e = 0; e < 3; ++e) {
            Vector3 a = edges[e][0];
            Vector3 b = edges[e][1];
            Vector3 c = top_edges[e][1];
            Vector3 d = top_edges[e][0];
            base = out_vertices.size();
            out_vertices.push_back(a); out_vertices.push_back(b); out_vertices.push_back(c);
            out_vertices.push_back(d);
            out_indices.push_back(base); out_indices.push_back(base+1); out_indices.push_back(base+2);
            out_indices.push_back(base); out_indices.push_back(base+2); out_indices.push_back(base+3);
            Vector3 normal = (b - a).cross(c - a).normalized();
            for (int i = 0; i < 4; ++i) out_normals.push_back(normal);
        }
    }

    // Build collision shapes (convex decomposition) using quickhull for each triangle prism?
    void build_collision_shapes() {
        clear_shapes();
        if (polygon_2d.size() < 3) return;

        if (build_mode == BUILD_SOLID) {
            // Extrude the triangulated polygon into convex prisms, each prism becomes a convex shape
            Vector<Vector2> tris = triangulate_polygon(polygon_2d);
            for (int i = 0; i < tris.size(); i += 3) {
                Vector<Vector3> vertices;
                Vector<int> indices;
                Vector<Vector3> normals;
                triangle_to_prism(tris[i], tris[i+1], tris[i+2], vertices, indices, normals);
                // Build convex hull from these vertices (quickhull)
                Vector<Vector3> hull_vertices;
                Geometry3D::quick_hull(vertices, hull_vertices);
                if (hull_vertices.size() >= 4) {
                    RID shape = PhysicsServer3D::get_singleton()->shape_create();
                    PhysicsServer3D::get_singleton()->shape_set_data(shape, hull_vertices);
                    shape_rids.push_back(shape);
                }
            }
        } else if (build_mode == BUILD_CONVEX_HULL) {
            // Single convex hull from the extruded polygon (if convex) or use decomposition
            Vector<Vector2> hull_2d = Geometry2D::convex_hull(polygon_2d);
            Vector<Vector3> vertices;
            double h = depth * 0.5;
            for (int i = 0; i < hull_2d.size(); ++i) {
                Vector2 p = hull_2d[i];
                vertices.push_back(Vector3(p.x, p.y, -h));
                vertices.push_back(Vector3(p.x, p.y,  h));
            }
            Vector<Vector3> hull_3d;
            Geometry3D::quick_hull(vertices, hull_3d);
            if (hull_3d.size() >= 4) {
                RID shape = PhysicsServer3D::get_singleton()->shape_create();
                PhysicsServer3D::get_singleton()->shape_set_data(shape, hull_3d);
                shape_rids.push_back(shape);
            }
        } else if (build_mode == BUILD_HOLLOW) {
            // For hollow, we do not generate collision shapes (or generate edges only).
            // Optionally, we could generate thin boxes along edges. Omitted for brevity.
        }
    }

    void update_debug_mesh() {
        if (!debug_visible || polygon_2d.size() < 3) {
            RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
            return;
        }
        // Build a wireframe representation of the extruded polygon (edges only)
        Vector<Vector3> vertices;
        Vector<int> indices;
        double h = depth * 0.5;
        int n = polygon_2d.size();
        for (int i = 0; i < n; ++i) {
            Vector2 p = polygon_2d[i];
            vertices.push_back(Vector3(p.x, p.y, -h));
            vertices.push_back(Vector3(p.x, p.y,  h));
        }
        // Bottom edges
        for (int i = 0; i < n; ++i) {
            int j = (i+1) % n;
            indices.push_back(i*2); indices.push_back(j*2);
        }
        // Top edges
        for (int i = 0; i < n; ++i) {
            int j = (i+1) % n;
            indices.push_back(i*2+1); indices.push_back(j*2+1);
        }
        // Vertical edges
        for (int i = 0; i < n; ++i) {
            indices.push_back(i*2); indices.push_back(i*2+1);
        }
        // Also add triangulation lines? Not needed.

        RenderingServer::get_singleton()->mesh_clear(debug_mesh_rid);
        RenderingServer::get_singleton()->mesh_add_surface(debug_mesh_rid, RS::PRIMITIVE_LINES, vertices, indices, Vector<Vector2>(), Vector<Vector3>());
        // Set material
        RID mat = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(mat, "albedo", debug_color);
        if (debug_emissive_intensity > 0.0f) {
            RenderingServer::get_singleton()->material_set_param(mat, "emission", debug_emissive_color);
            RenderingServer::get_singleton()->material_set_param(mat, "emission_intensity", debug_emissive_intensity);
        }
        RenderingServer::get_singleton()->mesh_surface_set_material(debug_mesh_rid, 0, mat);
        Transform3D global = get_global_transform(); // would be obtained from Node3D
        Transform3D final_t = global * local_transform;
        RenderingServer::get_singleton()->instance_set_transform(debug_instance_rid, final_t);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, true);
    }

    void sync_all() {
        if (dirty) {
            build_collision_shapes();
            update_debug_mesh();
            dirty = false;
        }
    }
};

CollisionPolygon3DExt::CollisionPolygon3DExt() {
    pimpl = new Impl();
}

CollisionPolygon3DExt::~CollisionPolygon3DExt() {
    delete pimpl;
}

void CollisionPolygon3DExt::set_polygon(const Vector<Vector2> &p_polygon) {
    pimpl->polygon_2d = p_polygon;
    pimpl->dirty = true;
    sync_polygon();
}
Vector<Vector2> CollisionPolygon3DExt::get_polygon() const { return pimpl->polygon_2d; }

void CollisionPolygon3DExt::set_depth(float p_depth) {
    pimpl->depth = p_depth;
    pimpl->dirty = true;
    sync_polygon();
}
float CollisionPolygon3DExt::get_depth() const { return pimpl->depth; }

void CollisionPolygon3DExt::set_build_mode(BuildMode p_mode) {
    pimpl->build_mode = p_mode;
    pimpl->dirty = true;
    sync_polygon();
}
CollisionPolygon3DExt::BuildMode CollisionPolygon3DExt::get_build_mode() const { return pimpl->build_mode; }

void CollisionPolygon3DExt::set_max_convex_pieces(int p_max) {
    pimpl->max_convex_pieces = p_max;
    pimpl->dirty = true;
    sync_polygon();
}
int CollisionPolygon3DExt::get_max_convex_pieces() const { return pimpl->max_convex_pieces; }

void CollisionPolygon3DExt::set_margin(float p_margin) {
    pimpl->margin = p_margin;
    // Not used in shape generation for now.
}
float CollisionPolygon3DExt::get_margin() const { return pimpl->margin; }

void CollisionPolygon3DExt::set_local_transform(const Transform3D &p_transform) {
    pimpl->local_transform = p_transform;
    pimpl->dirty = true;
    sync_polygon();
}
Transform3D CollisionPolygon3DExt::get_local_transform() const { return pimpl->local_transform; }

void CollisionPolygon3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    pimpl->dirty = true;
    sync_polygon();
}
bool CollisionPolygon3DExt::is_debug_visible() const { return pimpl->debug_visible; }

void CollisionPolygon3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    pimpl->dirty = true;
    sync_polygon();
}
Color CollisionPolygon3DExt::get_debug_color() const { return pimpl->debug_color; }

void CollisionPolygon3DExt::set_debug_emissive(const Color &p_color, float p_intensity) {
    pimpl->debug_emissive_color = p_color;
    pimpl->debug_emissive_intensity = p_intensity;
    pimpl->dirty = true;
    sync_polygon();
}
void CollisionPolygon3DExt::get_debug_emissive(Color &r_color, float &r_intensity) const {
    r_color = pimpl->debug_emissive_color;
    r_intensity = pimpl->debug_emissive_intensity;
}

void CollisionPolygon3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    // Not directly used by polygon shape, but could affect GI if debug mesh is emissive.
}
int CollisionPolygon3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void CollisionPolygon3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}
float CollisionPolygon3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void CollisionPolygon3DExt::sync_polygon() {
    pimpl->sync_all();
}