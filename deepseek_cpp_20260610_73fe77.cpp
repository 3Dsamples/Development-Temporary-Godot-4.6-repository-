// csg_polygon_3d.cpp
#include "csg_polygon_3d.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numbers>

namespace lighting {

// ============================================================================
// Simple ear clipping triangulation for 2D polygon (non‑self‑intersecting)
// ============================================================================
static bool ear_clipping(const std::vector<double>& poly,
                         std::vector<int>& out_triangles) {
    int n = (int)poly.size() / 2;
    if (n < 3) return false;
    // Build index list
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    out_triangles.clear();
    while (indices.size() > 3) {
        bool ear_found = false;
        for (size_t i = 0; i < indices.size(); ++i) {
            int prev = indices[(i + indices.size() - 1) % indices.size()];
            int curr = indices[i];
            int next = indices[(i + 1) % indices.size()];
            // 2D vertices
            double px = poly[prev*2], py = poly[prev*2+1];
            double cx = poly[curr*2], cy = poly[curr*2+1];
            double nx = poly[next*2], ny = poly[next*2+1];
            // Check convexity (cross product sign)
            double cross = (cx - px)*(ny - py) - (cy - py)*(nx - px);
            if (cross <= 0) continue; // reflex or collinear
            // Check if any other vertex lies inside triangle
            bool inside = false;
            for (int v : indices) {
                if (v == prev || v == curr || v == next) continue;
                double vx = poly[v*2], vy = poly[v*2+1];
                // barycentric test
                double w0 = (cx - nx)*(vy - ny) - (cy - ny)*(vx - nx);
                double w1 = (nx - px)*(vy - py) - (ny - py)*(vx - px);
                double w2 = (px - cx)*(vy - cy) - (py - cy)*(vx - cx);
                if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                    inside = true;
                    break;
                }
            }
            if (!inside) {
                out_triangles.push_back(prev);
                out_triangles.push_back(curr);
                out_triangles.push_back(next);
                indices.erase(indices.begin() + i);
                ear_found = true;
                break;
            }
        }
        if (!ear_found) break; // degenerate
    }
    if (indices.size() == 3) {
        out_triangles.push_back(indices[0]);
        out_triangles.push_back(indices[1]);
        out_triangles.push_back(indices[2]);
    }
    return !out_triangles.empty();
}

// ============================================================================
// Triangulate using quick convex decomposition (fallback)
// ============================================================================
static void quick_convex_decomposition(const std::vector<double>& poly,
                                       std::vector<int>& out_triangles) {
    int n = (int)poly.size() / 2;
    if (n < 3) return;
    // Simple fan triangulation (only works for convex polygons)
    for (int i = 1; i < n-1; ++i) {
        out_triangles.push_back(0);
        out_triangles.push_back(i);
        out_triangles.push_back(i+1);
    }
}

// ============================================================================
// Generate extruded mesh from polygon
// ============================================================================
static void generate_extruded_mesh(const std::vector<double>& poly,
                                   double depth,
                                   const std::vector<int>& tri_indices,
                                   std::vector<double>& out_vertices,
                                   std::vector<int>& out_indices,
                                   std::vector<float>& out_normals) {
    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();
    int n = (int)poly.size() / 2;
    if (n < 3 || depth == 0.0) return;

    double half_z = depth * 0.5;
    // bottom and top vertices
    int base_bottom = 0;
    int base_top = n;
    for (int i = 0; i < n; ++i) {
        double x = poly[i*2];
        double y = poly[i*2+1];
        out_vertices.push_back(x);
        out_vertices.push_back(y);
        out_vertices.push_back(-half_z);
        out_normals.push_back(0.0f);
        out_normals.push_back(0.0f);
        out_normals.push_back(-1.0f);
        out_vertices.push_back(x);
        out_vertices.push_back(y);
        out_vertices.push_back(half_z);
        out_normals.push_back(0.0f);
        out_normals.push_back(0.0f);
        out_normals.push_back(1.0f);
    }
    // Extrude side faces (quads between bottom and top)
    int side_start = (int)out_vertices.size() / 3;
    for (int i = 0; i < n; ++i) {
        int next = (i+1) % n;
        double x1 = poly[i*2];
        double y1 = poly[i*2+1];
        double x2 = poly[next*2];
        double y2 = poly[next*2+1];
        // bottom edge vertices (already present)
        // Actually we need to add side vertices (duplicate for flat normals? but we already have bottom and top)
        // For simplicity, we reuse bottom and top vertices and add new quads.
        int v0 = base_bottom + i;
        int v1 = base_bottom + next;
        int v2 = base_top + next;
        int v3 = base_top + i;
        out_indices.push_back(v0);
        out_indices.push_back(v1);
        out_indices.push_back(v2);
        out_indices.push_back(v0);
        out_indices.push_back(v2);
        out_indices.push_back(v3);
    }
    // Add bottom and top caps using triangulation indices
    for (size_t t = 0; t < tri_indices.size(); t += 3) {
        int i0 = tri_indices[t];
        int i1 = tri_indices[t+1];
        int i2 = tri_indices[t+2];
        // bottom cap (indices point to original polygon vertices, map to bottom vertices)
        out_indices.push_back(base_bottom + i0);
        out_indices.push_back(base_bottom + i1);
        out_indices.push_back(base_bottom + i2);
        // top cap (reverse orientation for outward normal)
        out_indices.push_back(base_top + i0);
        out_indices.push_back(base_top + i2);
        out_indices.push_back(base_top + i1);
    }
}

// ============================================================================
// Lathe (rotation) mesh generation
// ============================================================================
static void generate_lathe_mesh(const std::vector<double>& poly,
                                double angle,
                                bool smooth,
                                std::vector<double>& out_vertices,
                                std::vector<int>& out_indices,
                                std::vector<float>& out_normals) {
    // Simplified: rotate profile around Y axis, generate segments
    int n = (int)poly.size() / 2;
    if (n < 2) return;
    int segments = std::max(3, (int)(angle * 8 / (2*std::numbers::pi)) + 1);
    double step = angle / segments;
    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();
    // For each segment, generate vertices
    for (int i = 0; i <= segments; ++i) {
        double phi = i * step;
        double cos_phi = std::cos(phi);
        double sin_phi = std::sin(phi);
        for (int j = 0; j < n; ++j) {
            double x = poly[j*2];
            double y = poly[j*2+1]; // profile radius?
            // actually points are (radius, height) in polygon? assume polygon given as (x,z) with x as radius.
            double r = x;
            double h = y;
            double vx = r * cos_phi;
            double vz = r * sin_phi;
            out_vertices.push_back(vx);
            out_vertices.push_back(h);
            out_vertices.push_back(vz);
            // normal (approximate)
            out_normals.push_back((float)cos_phi);
            out_normals.push_back(0.0f);
            out_normals.push_back((float)sin_phi);
        }
    }
    // indices: quads between rows
    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < n-1; ++j) {
            int i0 = i * n + j;
            int i1 = i * n + j+1;
            int i2 = (i+1) * n + j;
            int i3 = (i+1) * n + j+1;
            out_indices.push_back(i0);
            out_indices.push_back(i1);
            out_indices.push_back(i2);
            out_indices.push_back(i2);
            out_indices.push_back(i1);
            out_indices.push_back(i3);
        }
    }
}

// ============================================================================
// CSGPolygon3D implementation
// ============================================================================
struct CSGPolygon3D::Impl {
    std::vector<double> polygon;              // [x0,y0, x1,y1, ...]
    double depth = 0.0;
    int extrusion_mode = 1;                   // 0=flat,1=extruded,2=lathe
    double lathe_angle = 2.0 * std::numbers::pi; // 360 deg
    bool lathe_smooth = true;
    bool use_ear_clipping = true;
    int max_convex_pieces = 4;
    int material_id = -1;
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool mesh_dirty = true;
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    std::vector<float> uvs;
    int64_t mesh_rid = -1;

    void generate_mesh();
};

CSGPolygon3D::CSGPolygon3D() : pimpl(std::make_unique<Impl>()) {
    set_csg_operation(CSGOperation::UNION);
}
CSGPolygon3D::~CSGPolygon3D() = default;

void CSGPolygon3D::set_polygon(const std::vector<double>& points) {
    pimpl->polygon = points;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
std::vector<double> CSGPolygon3D::get_polygon() const { return pimpl->polygon; }

void CSGPolygon3D::set_depth(double depth) { pimpl->depth = depth; pimpl->mesh_dirty = true; update_csg_mesh(); }
double CSGPolygon3D::get_depth() const { return pimpl->depth; }
void CSGPolygon3D::set_extrusion_mode(int mode) { pimpl->extrusion_mode = mode; pimpl->mesh_dirty = true; update_csg_mesh(); }
int CSGPolygon3D::get_extrusion_mode() const { return pimpl->extrusion_mode; }
void CSGPolygon3D::set_lathe_angle(double angle) { pimpl->lathe_angle = angle; pimpl->mesh_dirty = true; update_csg_mesh(); }
double CSGPolygon3D::get_lathe_angle() const { return pimpl->lathe_angle; }
void CSGPolygon3D::set_lathe_smooth(bool smooth) { pimpl->lathe_smooth = smooth; pimpl->mesh_dirty = true; update_csg_mesh(); }
bool CSGPolygon3D::is_lathe_smooth() const { return pimpl->lathe_smooth; }
void CSGPolygon3D::set_use_ear_clipping(bool use) { pimpl->use_ear_clipping = use; pimpl->mesh_dirty = true; update_csg_mesh(); }
bool CSGPolygon3D::is_ear_clipping() const { return pimpl->use_ear_clipping; }
void CSGPolygon3D::set_max_convex_pieces(int pieces) { pimpl->max_convex_pieces = pieces; }
int CSGPolygon3D::get_max_convex_pieces() const { return pimpl->max_convex_pieces; }
void CSGPolygon3D::set_material(int material_id) { pimpl->material_id = material_id; }
int CSGPolygon3D::get_material() const { return pimpl->material_id; }

void CSGPolygon3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void CSGPolygon3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void CSGPolygon3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void CSGPolygon3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void CSGPolygon3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CSGPolygon3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void CSGPolygon3D::Impl::generate_mesh() {
    if (polygon.size() < 6 || (extrusion_mode == 0 && depth == 0.0)) return;
    std::vector<int> tri_indices;
    if (use_ear_clipping) {
        if (!ear_clipping(polygon, tri_indices)) {
            quick_convex_decomposition(polygon, tri_indices);
        }
    } else {
        quick_convex_decomposition(polygon, tri_indices);
    }
    if (extrusion_mode == 1) {
        generate_extruded_mesh(polygon, depth, tri_indices, vertices, indices, normals);
    } else if (extrusion_mode == 2) {
        generate_lathe_mesh(polygon, lathe_angle, lathe_smooth, vertices, indices, normals);
    } else {
        // flat (2D) – just bottom cap
        vertices.clear(); indices.clear(); normals.clear();
        double z = 0.0;
        int base = 0;
        for (size_t i = 0; i < polygon.size(); i+=2) {
            vertices.push_back(polygon[i]);
            vertices.push_back(polygon[i+1]);
            vertices.push_back(z);
            normals.push_back(0.0f); normals.push_back(0.0f); normals.push_back(1.0f);
        }
        for (size_t t = 0; t < tri_indices.size(); t += 3) {
            indices.push_back(tri_indices[t]);
            indices.push_back(tri_indices[t+1]);
            indices.push_back(tri_indices[t+2]);
        }
    }
}

void CSGPolygon3D::update_csg_mesh() {
    if (!pimpl->mesh_dirty) return;
    pimpl->generate_mesh();
    // Compute AABB
    if (!pimpl->vertices.empty()) {
        double min_x = pimpl->vertices[0], max_x = pimpl->vertices[0];
        double min_y = pimpl->vertices[1], max_y = pimpl->vertices[1];
        double min_z = pimpl->vertices[2], max_z = pimpl->vertices[2];
        for (size_t i = 3; i < pimpl->vertices.size(); i += 3) {
            min_x = std::min(min_x, pimpl->vertices[i]);
            max_x = std::max(max_x, pimpl->vertices[i]);
            min_y = std::min(min_y, pimpl->vertices[i+1]);
            max_y = std::max(max_y, pimpl->vertices[i+1]);
            min_z = std::min(min_z, pimpl->vertices[i+2]);
            max_z = std::max(max_z, pimpl->vertices[i+2]);
        }
        set_aabb(&min_x, &max_x);
        double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
        set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
    }
    _set_mesh_data(pimpl->vertices, pimpl->indices, pimpl->normals, pimpl->uvs);
    pimpl->mesh_dirty = false;
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register for GI
    }
}

} // namespace lighting