// path_3d.cpp
#include "path_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace lighting {

// ============================================================================
// Path3D Implementation
// ============================================================================
struct Path3D::Impl {
    std::vector<PathPoint> points;
    CurveType curve_type = CurveType::CATMULL_ROM;
    bool closed = false;
    int resolution = 20; // subdivisions per segment

    // Baked data
    std::vector<Vector3> baked_points;
    std::vector<float> baked_distances;
    float total_length = 0.0f;
    bool baked_dirty = true;

    // Visualization (debug)
    bool visible = false;   // default invisible
    char material_path[256] = {0};
    float line_width = 0.05f;
    float color[3] = {1.0f, 1.0f, 0.0f};
    int64_t debug_instance = -1;

    void bake();
    Vector3 interpolate_linear(const Vector3& p0, const Vector3& p1, float t) const;
    Vector3 interpolate_catmull(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, float t) const;
    Vector3 interpolate_bezier(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, float t) const;
    Vector3 interpolate_cubic(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, float t) const;
    Vector3 tangent_at_segment(int seg, float t) const;
};

Path3D::Path3D() : pimpl(std::make_unique<Impl>()) {}
Path3D::~Path3D() = default;

void Path3D::add_point(const PathPoint& point) {
    pimpl->points.push_back(point);
    pimpl->baked_dirty = true;
}
void Path3D::insert_point(int index, const PathPoint& point) {
    if (index < 0) index = 0;
    if (index > (int)pimpl->points.size()) index = (int)pimpl->points.size();
    pimpl->points.insert(pimpl->points.begin() + index, point);
    pimpl->baked_dirty = true;
}
void Path3D::remove_point(int index) {
    if (index >=0 && index < (int)pimpl->points.size()) {
        pimpl->points.erase(pimpl->points.begin() + index);
        pimpl->baked_dirty = true;
    }
}
void Path3D::clear_points() { pimpl->points.clear(); pimpl->baked_dirty = true; }
int Path3D::get_point_count() const { return (int)pimpl->points.size(); }
void Path3D::set_point(int index, const PathPoint& point) {
    if (index >=0 && index < (int)pimpl->points.size()) {
        pimpl->points[index] = point;
        pimpl->baked_dirty = true;
    }
}
PathPoint Path3D::get_point(int index) const {
    if (index >=0 && index < (int)pimpl->points.size()) return pimpl->points[index];
    return PathPoint{};
}

void Path3D::set_curve_type(CurveType type) { pimpl->curve_type = type; pimpl->baked_dirty = true; }
CurveType Path3D::get_curve_type() const { return pimpl->curve_type; }
void Path3D::set_closed(bool closed) { pimpl->closed = closed; pimpl->baked_dirty = true; }
bool Path3D::is_closed() const { return pimpl->closed; }
void Path3D::set_curve_resolution(int steps) { pimpl->resolution = steps; pimpl->baked_dirty = true; }
int Path3D::get_curve_resolution() const { return pimpl->resolution; }

void Path3D::Impl::bake() {
    if (points.empty()) {
        baked_points.clear();
        baked_distances.clear();
        total_length = 0.0f;
        baked_dirty = false;
        return;
    }
    int num_segments = points.size() - 1;
    if (closed) num_segments = points.size();
    baked_points.clear();
    baked_distances.clear();
    total_length = 0.0f;

    for (int seg = 0; seg < num_segments; ++seg) {
        int i0 = seg;
        int i1 = (seg + 1) % points.size();
        if (!closed && seg == points.size()-1) break;
        // Get four points for interpolation
        int i_prev = (seg - 1 + points.size()) % points.size();
        int i_next = (seg + 2) % points.size();
        if (!closed) {
            i_prev = (seg > 0) ? seg-1 : seg;
            i_next = (seg+2 < (int)points.size()) ? seg+2 : i1;
        }
        const PathPoint& p0 = points[i_prev];
        const PathPoint& p1 = points[i0];
        const PathPoint& p2 = points[i1];
        const PathPoint& p3 = points[i_next];

        Vector3 prev_pos = (curve_type == CurveType::LINEAR) ? p1.position : p0.position;
        // Not used fully – simplified

        for (int step = 0; step <= pimpl->resolution; ++step) {
            float t = (float)step / pimpl->resolution;
            Vector3 pos;
            if (curve_type == CurveType::LINEAR) {
                pos = interpolate_linear(p1.position, p2.position, t);
            } else if (curve_type == CurveType::CATMULL_ROM) {
                pos = interpolate_catmull(p1.position, p2.position, p0.position, p3.position, t);
            } else if (curve_type == CurveType::BEZIER) {
                pos = interpolate_bezier(p1.position, p1.position + p1.out_tangent, p2.position + p2.in_tangent, p2.position, t);
            } else { // CUBIC_SPLINE
                pos = interpolate_cubic(p1.position, p1.position + p1.out_tangent, p2.position + p2.in_tangent, p2.position, t);
            }
            baked_points.push_back(pos);
        }
    }
    // Compute distances
    for (size_t i = 0; i < baked_points.size(); ++i) {
        if (i == 0) baked_distances.push_back(0.0f);
        else {
            float seg_len = (baked_points[i] - baked_points[i-1]).length();
            total_length += seg_len;
            baked_distances.push_back(total_length);
        }
    }
    baked_dirty = false;
}

Vector3 Path3D::Impl::interpolate_linear(const Vector3& p0, const Vector3& p1, float t) const {
    return p0 + (p1 - p0) * t;
}
Vector3 Path3D::Impl::interpolate_catmull(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, float t) const {
    float t2 = t*t;
    float t3 = t2*t;
    Vector3 a = p1 * 2.0f;
    Vector3 b = (p2 - p0) * t;
    Vector3 c = (p0*2.0f - p1*5.0f + p2*4.0f - p3) * t2;
    Vector3 d = (-p0 + p1*3.0f - p2*3.0f + p3) * t3;
    return (a + b + c + d) * 0.5f;
}
Vector3 Path3D::Impl::interpolate_bezier(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, float t) const {
    float mt = 1.0f - t;
    return p0*mt*mt*mt + p1*3.0f*mt*mt*t + p2*3.0f*mt*t*t + p3*t*t*t;
}
Vector3 Path3D::Impl::interpolate_cubic(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, float t) const {
    return interpolate_bezier(p0,p1,p2,p3,t);
}

Vector3 Path3D::get_point_at_ratio(float t) const {
    if (pimpl->baked_dirty) pimpl->bake();
    if (pimpl->baked_points.empty()) return Vector3(0,0,0);
    if (t <= 0.0f) return pimpl->baked_points.front();
    if (t >= 1.0f) return pimpl->baked_points.back();
    float target_dist = t * pimpl->total_length;
    // binary search in baked_distances
    auto it = std::lower_bound(pimpl->baked_distances.begin(), pimpl->baked_distances.end(), target_dist);
    int idx = std::distance(pimpl->baked_distances.begin(), it);
    if (idx <= 0) return pimpl->baked_points[0];
    if (idx >= (int)pimpl->baked_points.size()) return pimpl->baked_points.back();
    float prev_dist = pimpl->baked_distances[idx-1];
    float seg_len = pimpl->baked_distances[idx] - prev_dist;
    float local_t = (target_dist - prev_dist) / seg_len;
    return pimpl->baked_points[idx-1] + (pimpl->baked_points[idx] - pimpl->baked_points[idx-1]) * local_t;
}

Vector3 Path3D::get_tangent_at_ratio(float t) const {
    // finite difference
    float eps = 0.001f;
    Vector3 p0 = get_point_at_ratio(t - eps);
    Vector3 p1 = get_point_at_ratio(t + eps);
    return (p1 - p0).normalized();
}

float Path3D::get_total_length() const {
    if (pimpl->baked_dirty) pimpl->bake();
    return pimpl->total_length;
}

Vector3 Path3D::get_up_direction_at_ratio(float t) const {
    // simple: assume up is world Y, but tilt can be applied
    // For now, return Vector3(0,1,0)
    return Vector3(0,1,0);
}

float Path3D::get_closest_ratio(const Vector3& world_pos, int max_iterations) const {
    // simple linear search on baked points
    if (pimpl->baked_points.empty()) return 0.0f;
    float min_dist2 = std::numeric_limits<float>::max();
    int best_idx = 0;
    for (size_t i = 0; i < pimpl->baked_points.size(); ++i) {
        Vector3 diff = world_pos - pimpl->baked_points[i];
        float d2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
        if (d2 < min_dist2) {
            min_dist2 = d2;
            best_idx = (int)i;
        }
    }
    // approximate ratio
    float ratio = (float)best_idx / (float)(pimpl->baked_points.size()-1);
    return ratio;
}

void Path3D::bake_points() { pimpl->bake(); }
const std::vector<Vector3>& Path3D::get_baked_points() const {
    if (pimpl->baked_dirty) pimpl->bake();
    return pimpl->baked_points;
}
const std::vector<float>& Path3D::get_baked_distances() const {
    if (pimpl->baked_dirty) pimpl->bake();
    return pimpl->baked_distances;
}

void Path3D::set_visible(bool visible) { pimpl->visible = visible; }
void Path3D::set_material(const char* material_path) { strncpy(pimpl->material_path, material_path, 255); }
void Path3D::set_line_width(float width) { pimpl->line_width = width; }
void Path3D::set_color(float r, float g, float b) { pimpl->color[0]=r; pimpl->color[1]=g; pimpl->color[2]=b; }

void Path3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // Create/update debug visualization mesh for the path.
    if (pimpl->visible && !pimpl->baked_dirty) {
        // Send line segments to render server
    } else if (!pimpl->visible && pimpl->debug_instance != -1) {
        // Remove debug instance
    }
}

// ============================================================================
// PathFollow3D Implementation
// ============================================================================
struct PathFollow3D::Impl {
    Path3D* path = nullptr;
    float ratio = 0.0f;
    float offset = 0.0f;
    int rotation_mode = 1; // 1 = orient to path
    bool cubic_interpolation = true;
    bool auto_advance = false;
    float auto_speed = 1.0f;
    bool loop = true;
    bool cast_shadow = true;
    int gi_mode = 2; // dynamic

    void update_transform();
};

PathFollow3D::PathFollow3D() : pimpl(std::make_unique<Impl>()) {}
PathFollow3D::~PathFollow3D() = default;

void PathFollow3D::set_path(Path3D* path) { pimpl->path = path; update_position(); }
Path3D* PathFollow3D::get_path() const { return pimpl->path; }

void PathFollow3D::set_ratio(float ratio) {
    pimpl->ratio = std::max(0.0f, std::min(1.0f, ratio));
    update_position();
}
float PathFollow3D::get_ratio() const { return pimpl->ratio; }
void PathFollow3D::set_offset(float distance) { pimpl->offset = distance; update_position(); }
float PathFollow3D::get_offset() const { return pimpl->offset; }
void PathFollow3D::set_rotation_mode(int mode) { pimpl->rotation_mode = mode; update_position(); }
int PathFollow3D::get_rotation_mode() const { return pimpl->rotation_mode; }
void PathFollow3D::set_cubic_interpolation(bool enable) { pimpl->cubic_interpolation = enable; update_position(); }
bool PathFollow3D::is_cubic_interpolation_enabled() const { return pimpl->cubic_interpolation; }

void PathFollow3D::update_position() {
    if (!pimpl->path) return;
    float t = pimpl->ratio;
    if (pimpl->offset != 0.0f) {
        float total_len = pimpl->path->get_total_length();
        if (total_len > 0.0f) {
            float offset_ratio = pimpl->offset / total_len;
            t += offset_ratio;
            t = std::fmod(t, 1.0f);
            if (t < 0.0f) t += 1.0f;
        }
    }
    Vector3 pos = pimpl->path->get_point_at_ratio(t);
    Transform3D trans = get_global_transform();
    trans.origin[0] = pos.x;
    trans.origin[1] = pos.y;
    trans.origin[2] = pos.z;
    if (pimpl->rotation_mode > 0) {
        Vector3 tangent = pimpl->path->get_tangent_at_ratio(t);
        // Build orientation from forward direction
        Vector3 forward = tangent.normalized();
        Vector3 up = pimpl->path->get_up_direction_at_ratio(t);
        Vector3 right = up.cross(forward).normalized();
        Vector3 corrected_up = forward.cross(right).normalized();
        // Construct basis in 3x3 matrix
        double basis[9] = {
            right.x, corrected_up.x, forward.x,
            right.y, corrected_up.y, forward.y,
            right.z, corrected_up.z, forward.z
        };
        memcpy(trans.basis, basis, 9*sizeof(double));
    }
    set_global_transform(trans);
}

void PathFollow3D::set_auto_advance(bool enable, float speed) {
    pimpl->auto_advance = enable;
    pimpl->auto_speed = speed;
}
void PathFollow3D::set_loop(bool loop) { pimpl->loop = loop; }
void PathFollow3D::advance(float delta_time) {
    if (!pimpl->auto_advance) return;
    float delta_ratio = pimpl->auto_speed * delta_time / pimpl->path->get_total_length();
    float new_ratio = pimpl->ratio + delta_ratio;
    if (pimpl->loop) {
        new_ratio = std::fmod(new_ratio, 1.0f);
        if (new_ratio < 0.0f) new_ratio += 1.0f;
    } else {
        new_ratio = std::max(0.0f, std::min(1.0f, new_ratio));
    }
    set_ratio(new_ratio);
}

void PathFollow3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool PathFollow3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void PathFollow3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int PathFollow3D::get_gi_mode() const { return pimpl->gi_mode; }

} // namespace lighting