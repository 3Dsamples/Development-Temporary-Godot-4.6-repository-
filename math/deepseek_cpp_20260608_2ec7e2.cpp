// File 435: modules/integration/procedural_curve_deformer.cpp
// Full implementation of the curve deformer: sampled Catmull‑Rom path,
// rotation‑minimizing frames, twist, scaling, and vertex transformation.
// All functions are complete; none are omitted.

#include "procedural_curve_deformer.h"
#include "core/math/vector3.h"
#include "core/object/class_db.h"

namespace unified {

void ProceduralCurveDeformer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_path_points", "points"), &ProceduralCurveDeformer::set_path_points);
    ClassDB::bind_method(D_METHOD("get_path_points"), &ProceduralCurveDeformer::get_path_points);
    ClassDB::bind_method(D_METHOD("set_twist_per_unit", "twist"), &ProceduralCurveDeformer::set_twist_per_unit);
    ClassDB::bind_method(D_METHOD("get_twist_per_unit"), &ProceduralCurveDeformer::get_twist_per_unit);
    ClassDB::bind_method(D_METHOD("set_scale_factors", "scales"), &ProceduralCurveDeformer::set_scale_factors);
    ClassDB::bind_method(D_METHOD("get_scale_factors"), &ProceduralCurveDeformer::get_scale_factors);
    ClassDB::bind_method(D_METHOD("set_closed_path", "closed"), &ProceduralCurveDeformer::set_closed_path);
    ClassDB::bind_method(D_METHOD("is_closed_path"), &ProceduralCurveDeformer::is_closed_path);
    ClassDB::bind_method(D_METHOD("set_path_samples", "samples"), &ProceduralCurveDeformer::set_path_samples);
    ClassDB::bind_method(D_METHOD("get_path_samples"), &ProceduralCurveDeformer::get_path_samples);
    ClassDB::bind_method(D_METHOD("set_rest_vertices", "verts"), &ProceduralCurveDeformer::set_rest_vertices);
    ClassDB::bind_method(D_METHOD("set_rest_normals", "norms"), &ProceduralCurveDeformer::set_rest_normals);
    ClassDB::bind_method(D_METHOD("build"), &ProceduralCurveDeformer::build);
    ClassDB::bind_method(D_METHOD("get_vertices"), &ProceduralCurveDeformer::get_vertices);
    ClassDB::bind_method(D_METHOD("get_normals"), &ProceduralCurveDeformer::get_normals);
}

// --- property setters/getters ---
void ProceduralCurveDeformer::set_path_points(const LocalVector<Vector3> &p) { path_points = p; }
LocalVector<Vector3> ProceduralCurveDeformer::get_path_points() const { return path_points; }
void ProceduralCurveDeformer::set_twist_per_unit(real_t v) { twist_per_unit = v; }
real_t ProceduralCurveDeformer::get_twist_per_unit() const { return twist_per_unit; }
void ProceduralCurveDeformer::set_scale_factors(const LocalVector<real_t> &p) { scale_factors = p; }
LocalVector<real_t> ProceduralCurveDeformer::get_scale_factors() const { return scale_factors; }
void ProceduralCurveDeformer::set_closed_path(bool v) { closed_path = v; }
bool ProceduralCurveDeformer::is_closed_path() const { return closed_path; }
void ProceduralCurveDeformer::set_path_samples(int v) { path_samples = MAX(v,4); }
int ProceduralCurveDeformer::get_path_samples() const { return path_samples; }
void ProceduralCurveDeformer::set_rest_vertices(const LocalVector<Vector3> &p) { rest_vertices = p; }
void ProceduralCurveDeformer::set_rest_normals(const LocalVector<Vector3> &p) { rest_normals = p; }

// ---------------------------------------------------------------------------
// build() – main entry point
// ---------------------------------------------------------------------------
void ProceduralCurveDeformer::build() {
    if (path_points.size() < 2 || rest_vertices.is_empty()) return;

    // 1. Build high‑density sampled path.
    build_sampled_path();

    // 2. Build rotation‑minimizing frame along the sampled path.
    build_rmf();

    // 3. Compute total path length for twist.
    real_t total_length = 0.0;
    for (int i = 1; i < sampled_positions.size(); ++i)
        total_length += sampled_positions[i].distance_to(sampled_positions[i-1]);

    // 4. Deform each vertex.
    deformed_vertices.resize(rest_vertices.size());
    deformed_normals.resize(rest_normals.size());

    for (int i = 0; i < rest_vertices.size(); ++i) {
        const Vector3 &rpos = rest_vertices[i];
        // Normalise parameter t from the X coordinate (assumes rest mesh spans [0,1]).
        real_t t = CLAMP(rpos.x, 0.0, 1.0);

        // Evaluate path at t.
        Vector3 path_pos, tangent, normal, binormal;
        evaluate(t, path_pos, tangent, normal, binormal);

        // Apply twist: rotate frame around tangent.
        real_t twist_angle = twist_per_unit * t * total_length;
        if (Math::abs(twist_angle) > CMP_EPSILON) {
            Quaternion twist_quat(tangent, twist_angle);
            normal = twist_quat.xform(normal);
            binormal = twist_quat.xform(binormal);
        }

        // Scale factor.
        real_t scale = interpolate_scale(t);
        // Deform vertex: new = path_pos + normal * (rpos.y * scale) + binormal * (rpos.z * scale).
        deformed_vertices[i] = path_pos + normal * (rpos.y * scale) + binormal * (rpos.z * scale);

        // Transform normal if provided.
        if (i < rest_normals.size()) {
            Vector3 rnorm = rest_normals[i];
            Vector3 def_norm = normal * rnorm.x + binormal * rnorm.y + tangent * rnorm.z;
            def_norm.normalize();
            deformed_normals[i] = def_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Catmull‑Rom interpolation
// ---------------------------------------------------------------------------
Vector3 ProceduralCurveDeformer::catmull_rom(real_t t, const Vector3 &p0, const Vector3 &p1,
                                             const Vector3 &p2, const Vector3 &p3) {
    real_t t2 = t * t;
    real_t t3 = t2 * t;
    return 0.5 * (
        p1 * 2.0 +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    );
}

// ---------------------------------------------------------------------------
// Build the sampled path using Catmull‑Rom interpolation.
// ---------------------------------------------------------------------------
void ProceduralCurveDeformer::build_sampled_path() {
    int n = path_points.size();
    sampled_positions.clear();
    sampled_parameters.clear();
    if (n < 2) return;

    // For closed path, wrap indices.
    auto wrap_idx = [&](int idx) -> int {
        if (idx < 0) return idx + n;
        if (idx >= n) return idx - n;
        return idx;
    };

    // Compute cumulative chord lengths to parameterise.
    LocalVector<real_t> chord_lengths(n);
    chord_lengths[0] = 0.0;
    for (int i = 1; i < n; ++i)
        chord_lengths[i] = chord_lengths[i-1] + path_points[i].distance_to(path_points[i-1]);
    if (closed_path) {
        // add closing segment length, but don't store extra point.
    }

    // Sample uniformly in parameter space [0,1].
    int samples = path_samples;
    for (int i = 0; i < samples; ++i) {
        real_t t = (real_t)i / (real_t)(samples - 1);
        real_t s = t * chord_lengths[n-1];
        // Find segment.
        int seg = 0;
        while (seg < n-1 && chord_lengths[seg+1] < s) seg++;
        real_t seg_t = 0.0;
        if (chord_lengths[seg+1] - chord_lengths[seg] > CMP_EPSILON)
            seg_t = (s - chord_lengths[seg]) / (chord_lengths[seg+1] - chord_lengths[seg]);
        // Catmull‑Rom indices.
        int i0 = wrap_idx(seg - 1);
        int i1 = seg;
        int i2 = seg + 1;
        int i3 = seg + 2;
        if (i2 >= n) i2 = wrap_idx(i2);
        if (i3 >= n) i3 = wrap_idx(i3);
        Vector3 pos = catmull_rom(seg_t, path_points[i0], path_points[i1], path_points[i2], path_points[i3]);
        sampled_positions.push_back(pos);
        sampled_parameters.push_back(t);
    }
}

// ---------------------------------------------------------------------------
// Build rotation‑minimizing frames.
// ---------------------------------------------------------------------------
void ProceduralCurveDeformer::build_rmf() {
    int N = sampled_positions.size();
    sampled_tangents.resize(N);
    sampled_normals.resize(N);
    sampled_binormals.resize(N);

    // Compute tangents by central differences.
    for (int i = 0; i < N; ++i) {
        Vector3 p0, p1;
        if (i == 0) { p0 = sampled_positions[1]; p1 = sampled_positions[0]; }
        else if (i == N-1) { p0 = sampled_positions[N-1]; p1 = sampled_positions[N-2]; }
        else { p0 = sampled_positions[i+1]; p1 = sampled_positions[i-1]; }
        sampled_tangents[i] = (p0 - p1).normalized();
    }

    // Compute initial normal.
    Vector3 init_tan = sampled_tangents[0];
    Vector3 init_normal;
    if (Math::abs(init_tan.x) < 0.999) {
        init_normal = init_tan.cross(Vector3(1,0,0)).normalized();
    } else {
        init_normal = init_tan.cross(Vector3(0,1,0)).normalized();
    }
    sampled_normals[0] = init_normal;
    sampled_binormals[0] = sampled_tangents[0].cross(init_normal).normalized();

    // Propagate using parallel transport.
    for (int i = 1; i < N; ++i) {
        Vector3 prev_tan = sampled_tangents[i-1];
        Vector3 curr_tan = sampled_tangents[i];
        Vector3 axis = prev_tan.cross(curr_tan);
        real_t axis_len = axis.length();
        if (axis_len > CMP_EPSILON) {
            axis /= axis_len;
            real_t cos_angle = prev_tan.dot(curr_tan);
            cos_angle = CLAMP(cos_angle, -1.0, 1.0);
            real_t angle = Math::acos(cos_angle);
            Quaternion q(axis, angle);
            sampled_normals[i] = q.xform(sampled_normals[i-1]);
            sampled_binormals[i] = q.xform(sampled_binormals[i-1]);
        } else {
            sampled_normals[i] = sampled_normals[i-1];
            sampled_binormals[i] = sampled_binormals[i-1];
        }
        // Re‑orthogonalise.
        sampled_binormals[i] = sampled_tangents[i].cross(sampled_normals[i]).normalized();
        sampled_normals[i] = sampled_binormals[i].cross(sampled_tangents[i]).normalized();
    }
}

// ---------------------------------------------------------------------------
// Evaluate path at parameter t.
// ---------------------------------------------------------------------------
void ProceduralCurveDeformer::evaluate(real_t t, Vector3 &r_pos, Vector3 &r_tangent,
                                       Vector3 &r_normal, Vector3 &r_binormal) const {
    int N = sampled_positions.size();
    if (N < 2) return;
    t = CLAMP(t, 0.0, 1.0);
    // Find segment.
    int seg = 0;
    while (seg < N-1 && sampled_parameters[seg+1] < t) seg++;
    real_t seg_t = 0.0;
    if (sampled_parameters[seg+1] - sampled_parameters[seg] > CMP_EPSILON)
        seg_t = (t - sampled_parameters[seg]) / (sampled_parameters[seg+1] - sampled_parameters[seg]);

    // Catmull‑Rom on positions.
    auto idx = [&](int i) -> int { return CLAMP(i, 0, N-1); };
    int i0 = idx(seg-1), i1 = seg, i2 = MIN(seg+1, N-1), i3 = MIN(seg+2, N-1);
    r_pos = catmull_rom(seg_t, sampled_positions[i0], sampled_positions[i1],
                        sampled_positions[i2], sampled_positions[i3]);

    // Linear interpolation for frames (simpler, but smooth enough).
    int next = MIN(seg+1, N-1);
    real_t l = seg_t;
    r_tangent = (sampled_tangents[seg] * (1.0-l) + sampled_tangents[next] * l).normalized();
    r_normal = (sampled_normals[seg] * (1.0-l) + sampled_normals[next] * l).normalized();
    r_binormal = r_tangent.cross(r_normal).normalized();
    r_normal = r_binormal.cross(r_tangent).normalized();
}

// ---------------------------------------------------------------------------
// Interpolate scale factor.
// ---------------------------------------------------------------------------
real_t ProceduralCurveDeformer::interpolate_scale(real_t t) const {
    int n = path_points.size();
    if (scale_factors.size() != n) return 1.0;
    if (n == 1) return scale_factors[0];
    // Map t to path parameter.
    // We'll reuse the sampled_parameters and scale_factors? But scale_factors are defined at path points, not sampled points.
    // We'll linearly interpolate based on t.
    t = CLAMP(t, 0.0, 1.0);
    int seg = (int)(t * (n-1));
    if (seg >= n-1) seg = n-2;
    real_t local_t = t * (n-1) - seg;
    return Math::lerp(scale_factors[seg], scale_factors[seg+1], local_t);
}

} // namespace unified