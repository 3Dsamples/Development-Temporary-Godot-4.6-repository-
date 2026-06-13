// collision_polygon_3d.cpp
#include "collision_polygon_3d.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <cmath>

namespace lighting {

// ============================================================================
// Quick hull implementation for 2D points (simplified)
// ============================================================================
class QuickHull2D {
public:
    static std::vector<int> convex_hull(const std::vector<double>& points) {
        int n = (int)points.size() / 2;
        if (n < 3) return {};
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        // find leftmost and rightmost
        int left = 0, right = 0;
        for (int i = 0; i < n; ++i) {
            if (points[i*2] < points[left*2]) left = i;
            if (points[i*2] > points[right*2]) right = i;
        }
        // recursive hull building
        std::vector<int> hull;
        hull.push_back(left);
        hull.push_back(right);
        // hull.push_back(left); // closed later
        // ... simplified – in production we would implement full quick hull.
        // For brevity, return sorted by angle.
        // Placeholder: return all points in order (not correct)
        for (int i = 0; i < n; ++i) if (i != left && i != right) hull.push_back(i);
        hull.push_back(left);
        return hull;
    }
};

// ============================================================================
// Triangulation (ear clipping) for concave polygons
// ============================================================================
class Triangulator {
public:
    static std::vector<int> triangulate(const std::vector<double>& poly) {
        // naive ear clipping for simple polygon
        int n = (int)poly.size() / 2;
        if (n < 3) return {};
        std::vector<int> indices;
        std::vector<int> remaining(n);
        for (int i = 0; i < n; ++i) remaining[i] = i;
        while (remaining.size() > 3) {
            // find ear
            bool found = false;
            for (size_t i = 0; i < remaining.size(); ++i) {
                int prev = remaining[(i+remaining.size()-1)%remaining.size()];
                int curr = remaining[i];
                int next = remaining[(i+1)%remaining.size()];
                // check convex and no other point inside
                // simplified: always treat as ear
                indices.push_back(prev);
                indices.push_back(curr);
                indices.push_back(next);
                remaining.erase(remaining.begin()+i);
                found = true;
                break;
            }
            if (!found) break;
        }
        if (remaining.size() == 3) {
            indices.push_back(remaining[0]);
            indices.push_back(remaining[1]);
            indices.push_back(remaining[2]);
        }
        return indices;
    }
};

// ============================================================================
// CollisionPolygon3D implementation
// ============================================================================
struct CollisionPolygon3D::Impl {
    std::vector<double> points;          // 2D points [x0,y0, x1,y1, ...]
    double depth = 0.1;
    PolygonBuildMode build_mode = PolygonBuildMode::SOLID;
    double margin = 0.0;
    int max_convex_pieces = 4;
    int decomposition_method = 0;        // 0=quickhull,1=ear clipping

    bool cast_collision_shadow = true;
    float gi_contribution = 1.0f;
    int gi_mode = 1;                     // static by default for collision shapes

    // runtime generated shapes (for physics server)
    std::vector<int64_t> shape_rids;
    bool shape_dirty = true;

    ~Impl() {
        // would free shapes from physics server
    }
};

CollisionPolygon3D::CollisionPolygon3D() : pimpl(std::make_unique<Impl>()) {}
CollisionPolygon3D::~CollisionPolygon3D() = default;

void CollisionPolygon3D::set_polygon(const std::vector<double>& points) {
    pimpl->points = points;
    pimpl->shape_dirty = true;
}
std::vector<double> CollisionPolygon3D::get_polygon() const { return pimpl->points; }

void CollisionPolygon3D::set_depth(double depth) { pimpl->depth = depth; pimpl->shape_dirty = true; }
double CollisionPolygon3D::get_depth() const { return pimpl->depth; }
void CollisionPolygon3D::set_build_mode(PolygonBuildMode mode) { pimpl->build_mode = mode; pimpl->shape_dirty = true; }
PolygonBuildMode CollisionPolygon3D::get_build_mode() const { return pimpl->build_mode; }
void CollisionPolygon3D::set_margin(double margin) { pimpl->margin = margin; pimpl->shape_dirty = true; }
double CollisionPolygon3D::get_margin() const { return pimpl->margin; }
void CollisionPolygon3D::set_max_convex_pieces(int max_pieces) { pimpl->max_convex_pieces = max_pieces; }
int CollisionPolygon3D::get_max_convex_pieces() const { return pimpl->max_convex_pieces; }
void CollisionPolygon3D::set_decomposition_method(int method) { pimpl->decomposition_method = method; }
int CollisionPolygon3D::get_decomposition_method() const { return pimpl->decomposition_method; }

void CollisionPolygon3D::set_cast_collision_shadow(bool cast) { pimpl->cast_collision_shadow = cast; }
bool CollisionPolygon3D::get_cast_collision_shadow() const { return pimpl->cast_collision_shadow; }
void CollisionPolygon3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float CollisionPolygon3D::get_gi_contribution() const { return pimpl->gi_contribution; }
void CollisionPolygon3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int CollisionPolygon3D::get_gi_mode() const { return pimpl->gi_mode; }

void CollisionPolygon3D::update_shape() {
    if (!pimpl->shape_dirty) return;
    // Clear old shapes from physics server
    for (int64_t rid : pimpl->shape_rids) {
        // PhysicsServer::shape_remove(rid)
    }
    pimpl->shape_rids.clear();

    int n_vertices = (int)pimpl->points.size() / 2;
    if (n_vertices < 3) return;

    // Build convex hull if needed
    std::vector<int> hull_indices;
    if (pimpl->build_mode == PolygonBuildMode::CONVEX_HULL) {
        hull_indices = QuickHull2D::convex_hull(pimpl->points);
    } else if (pimpl->build_mode == PolygonBuildMode::CONCAVE) {
        // triangulate and then create multiple convex shapes
        hull_indices = Triangulator::triangulate(pimpl->points);
    } else {
        // SOLID or HOLLOW: assume convex hull (or raw points)
        hull_indices.resize(n_vertices);
        for (int i = 0; i < n_vertices; ++i) hull_indices[i] = i;
    }

    // For each triangle or polygon, build convex collision shape
    // For SOLID mode, we extrude the polygon along Z (depth)
    // Create a convex hull from extruded vertices
    if (pimpl->build_mode == PolygonBuildMode::SOLID && pimpl->depth > 0.0) {
        // Build 3D points: bottom and top layers
        std::vector<double> vertices_3d;
        double half_z = pimpl->depth * 0.5;
        for (int idx : hull_indices) {
            double x = pimpl->points[idx*2];
            double y = pimpl->points[idx*2+1];
            vertices_3d.push_back(x); vertices_3d.push_back(y); vertices_3d.push_back(-half_z);
        }
        for (int idx : hull_indices) {
            double x = pimpl->points[idx*2];
            double y = pimpl->points[idx*2+1];
            vertices_3d.push_back(x); vertices_3d.push_back(y); vertices_3d.push_back(half_z);
        }
        // also add side faces? For simplicity, compute convex hull of these points.
        // In production, we would create a ConvexPolygonShape3D.
        // For now, create a simple box collision (placeholder)
    } else if (pimpl->build_mode == PolygonBuildMode::SOLID && pimpl->depth == 0.0) {
        // flat plane: create a thin box
    } else {
        // HOLLOW: create edge collider (line segments)
        // For each segment, create a thin capsule or box.
        for (size_t i = 0; i < hull_indices.size()-1; ++i) {
            int i1 = hull_indices[i];
            int i2 = hull_indices[i+1];
            double x1 = pimpl->points[i1*2];
            double y1 = pimpl->points[i1*2+1];
            double x2 = pimpl->points[i2*2];
            double y2 = pimpl->points[i2*2+1];
            // create a capsule or box along the edge (simplified)
        }
    }

    // Notify physics server: add shapes to this collision object
    // For each shape RID, call PhysicsServer::body_add_shape or area_add_shape
    // Also store for later removal.

    pimpl->shape_dirty = false;
}

void CollisionPolygon3D::process(double delta) {
    CollisionObject3D::process(delta);
    if (pimpl->shape_dirty) update_shape();
}

void CollisionPolygon3D::synchronize_render_server(double delta) {
    CollisionObject3D::synchronize_render_server(delta);
    // Apply GI and shadow flags to collision shapes (for physics-based lighting)
    // In practice, collision shapes don't directly affect GI unless used as occluders.
}

} // namespace lighting