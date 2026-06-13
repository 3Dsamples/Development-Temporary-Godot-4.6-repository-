// marker_3d.cpp
#include "marker_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

namespace lighting {

// ============================================================================
// Helper: generate sphere mesh (simple icosphere or UV sphere)
// ============================================================================
static void generate_sphere_mesh(double radius, int segments, int rings,
                                 std::vector<float>& out_vertices,
                                 std::vector<int>& out_indices) {
    out_vertices.clear();
    out_indices.clear();
    // crude UV sphere
    for (int i = 0; i <= rings; ++i) {
        double phi = M_PI * i / rings;
        double sin_phi = sin(phi);
        double cos_phi = cos(phi);
        for (int j = 0; j <= segments; ++j) {
            double theta = 2.0 * M_PI * j / segments;
            double sin_theta = sin(theta);
            double cos_theta = cos(theta);
            double x = radius * sin_phi * cos_theta;
            double y = radius * cos_phi;
            double z = radius * sin_phi * sin_theta;
            out_vertices.push_back((float)x);
            out_vertices.push_back((float)y);
            out_vertices.push_back((float)z);
        }
    }
    for (int i = 0; i < rings; ++i) {
        for (int j = 0; j < segments; ++j) {
            int p0 = i * (segments+1) + j;
            int p1 = i * (segments+1) + j+1;
            int p2 = (i+1) * (segments+1) + j;
            int p3 = (i+1) * (segments+1) + j+1;
            out_indices.push_back(p0); out_indices.push_back(p1); out_indices.push_back(p2);
            out_indices.push_back(p2); out_indices.push_back(p1); out_indices.push_back(p3);
        }
    }
}

// ============================================================================
// Helper: generate cube mesh (axis-aligned)
// ============================================================================
static void generate_cube_mesh(double size,
                               std::vector<float>& out_vertices,
                               std::vector<int>& out_indices) {
    double half = size * 0.5;
    // 8 vertices
    double v[8][3] = {
        {-half, -half, -half}, { half, -half, -half},
        { half, -half,  half}, {-half, -half,  half},
        {-half,  half, -half}, { half,  half, -half},
        { half,  half,  half}, {-half,  half,  half}
    };
    // 12 triangles (2 per face)
    int faces[6][6] = {
        {0,1,2, 0,2,3}, // bottom
        {4,7,6, 4,6,5}, // top
        {0,4,5, 0,5,1}, // front? depends, but covers all
        {1,5,6, 1,6,2},
        {2,6,7, 2,7,3},
        {3,7,4, 3,4,0}
    };
    out_vertices.clear();
    out_indices.clear();
    for (int f = 0; f < 6; ++f) {
        for (int t = 0; t < 6; ++t) {
            int idx = faces[f][t];
            out_vertices.push_back((float)v[idx][0]);
            out_vertices.push_back((float)v[idx][1]);
            out_vertices.push_back((float)v[idx][2]);
            out_indices.push_back(out_indices.size());
        }
    }
}

// ============================================================================
// Helper: generate axes mesh (three lines with colored cones)
// ============================================================================
static void generate_axes_mesh(double length, double thickness,
                               std::vector<float>& out_vertices,
                               std::vector<int>& out_indices,
                               std::vector<float>& out_colors) {
    // Simple: 3 cylinders for axes, with cones at ends. For brevity, we generate a single cube per axis.
    // Not fully implemented for space – but in real code it would be detailed.
    // We'll generate a simple line segment as a thin box.
    out_vertices.clear();
    out_indices.clear();
    out_colors.clear();
    // X axis (red)
    // Y axis (green)
    // Z axis (blue)
    // Placeholder: create a dummy triangle.
    out_vertices = {0,0,0, 0,0.1f,0, 0.1f,0,0};
    out_indices = {0,1,2};
    out_colors = {1,0,0, 0,1,0, 0,0,1};
}

// ============================================================================
// Marker3D implementation
// ============================================================================
struct Marker3D::Impl {
    int gizmo_type = 0;                     // 0=sphere,1=cube,2=axes,3=custom
    double gizmo_size = 0.2;
    float color_rgb[3] = {1.0f, 1.0f, 1.0f};
    float color_alpha = 1.0f;
    int64_t custom_mesh_rid = -1;

    std::string label;
    float label_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    double label_offset[3] = {0.1, 0.1, 0.0};

    float draw_min_dist = 0.0f, draw_max_dist = 100.0f, draw_fade_margin = 0.0f;
    bool always_visible = false;

    bool cast_shadow = false;               // markers usually don't cast shadows
    bool receive_shadow = false;
    int gi_mode = 0;                        // off by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Cached mesh data
    int64_t generated_mesh_rid = -1;
    int64_t instance_rid = -1;
    bool mesh_dirty = true;
    double last_cam_dist = 0.0;
    float opacity = 1.0f;

    std::vector<float> vertices;
    std::vector<int> indices;
    std::vector<float> vertex_colors;       // per‑vertex color (if axes)

    void generate_mesh();
    void update_visibility(double cam_distance);
};

Marker3D::Marker3D() : pimpl(std::make_unique<Impl>()) {}
Marker3D::~Marker3D() = default;

void Marker3D::set_gizmo_type(int type) {
    pimpl->gizmo_type = type;
    pimpl->mesh_dirty = true;
}
int Marker3D::get_gizmo_type() const { return pimpl->gizmo_type; }
void Marker3D::set_gizmo_size(double size) {
    pimpl->gizmo_size = std::max(0.001, size);
    pimpl->mesh_dirty = true;
}
double Marker3D::get_gizmo_size() const { return pimpl->gizmo_size; }
void Marker3D::set_gizmo_color(const float* rgb, float alpha) {
    memcpy(pimpl->color_rgb, rgb, 3*sizeof(float));
    pimpl->color_alpha = alpha;
    pimpl->mesh_dirty = true;
}
void Marker3D::get_gizmo_color(float* out_rgb, float& out_alpha) const {
    memcpy(out_rgb, pimpl->color_rgb, 3*sizeof(float));
    out_alpha = pimpl->color_alpha;
}
void Marker3D::set_custom_mesh(int64_t mesh_rid) {
    pimpl->custom_mesh_rid = mesh_rid;
    pimpl->gizmo_type = 3;
    pimpl->mesh_dirty = true;
}
int64_t Marker3D::get_custom_mesh() const { return pimpl->custom_mesh_rid; }
void Marker3D::set_label(const char* text) {
    pimpl->label = text ? text : "";
}
const char* Marker3D::get_label() const { return pimpl->label.c_str(); }
void Marker3D::set_label_color(const float* rgba) { memcpy(pimpl->label_color, rgba, 4*sizeof(float)); }
void Marker3D::get_label_color(float* out_rgba) const { memcpy(out_rgba, pimpl->label_color, 4*sizeof(float)); }
void Marker3D::set_label_offset(const double* offset) { memcpy(pimpl->label_offset, offset, 3*sizeof(double)); }
void Marker3D::get_label_offset(double* out_offset) const { memcpy(out_offset, pimpl->label_offset, 3*sizeof(double)); }
void Marker3D::set_draw_distance(float min_dist, float max_dist, float fade_margin) {
    pimpl->draw_min_dist = std::max(0.0f, min_dist);
    pimpl->draw_max_dist = std::max(min_dist + 0.01f, max_dist);
    pimpl->draw_fade_margin = fade_margin;
}
void Marker3D::get_draw_distance(float& min_dist, float& max_dist, float& fade_margin) const {
    min_dist = pimpl->draw_min_dist; max_dist = pimpl->draw_max_dist; fade_margin = pimpl->draw_fade_margin;
}
void Marker3D::set_always_visible(bool always) { pimpl->always_visible = always; }
bool Marker3D::is_always_visible() const { return pimpl->always_visible; }

void Marker3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; VisualInstance3D::set_cast_shadow(cast); }
void Marker3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void Marker3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; VisualInstance3D::set_gi_mode(mode); }
void Marker3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void Marker3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void Marker3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void Marker3D::Impl::generate_mesh() {
    if (gizmo_type == 3 && custom_mesh_rid != -1) {
        // use custom mesh directly, no generation needed
        mesh_dirty = false;
        return;
    }
    if (gizmo_type == 0) {
        generate_sphere_mesh(gizmo_size, 16, 16, vertices, indices);
    } else if (gizmo_type == 1) {
        generate_cube_mesh(gizmo_size, vertices, indices);
    } else if (gizmo_type == 2) {
        generate_axes_mesh(gizmo_size, gizmo_size*0.1, vertices, indices, vertex_colors);
    } else {
        // fallback to a small cube
        generate_cube_mesh(gizmo_size, vertices, indices);
    }
    mesh_dirty = false;
}

void Marker3D::Impl::update_visibility(double cam_distance) {
    if (always_visible) {
        opacity = 1.0f;
        set_visible(true);
        return;
    }
    if (cam_distance < draw_min_dist || cam_distance > draw_max_dist) {
        set_visible(false);
        opacity = 0.0f;
        return;
    }
    set_visible(true);
    // fade near boundaries
    if (draw_fade_margin > 0.0f) {
        if (cam_distance < draw_min_dist + draw_fade_margin) {
            float t = (cam_distance - draw_min_dist) / draw_fade_margin;
            opacity = std::clamp(t, 0.0f, 1.0f);
        } else if (cam_distance > draw_max_dist - draw_fade_margin) {
            float t = (draw_max_dist - cam_distance) / draw_fade_margin;
            opacity = std::clamp(t, 0.0f, 1.0f);
        } else {
            opacity = 1.0f;
        }
        // apply opacity to material or instance alpha
    } else {
        opacity = 1.0f;
    }
}

void Marker3D::process(double delta) {
    VisualInstance3D::process(delta);
    // In a real engine, we would get camera distance from the current active camera.
    // For simulation, we approximate.
    double cam_dist = 10.0; // dummy
    pimpl->update_visibility(cam_dist);
}

void Marker3D::synchronize_render_server(double delta) {
    VisualInstance3D::synchronize_render_server(delta);
    if (pimpl->mesh_dirty) {
        pimpl->generate_mesh();
        if (pimpl->vertices.empty()) return;
        // Create or update mesh in rendering server
        if (pimpl->generated_mesh_rid == -1) {
            // pimpl->generated_mesh_rid = RenderingServer::mesh_create();
        }
        // Build vertex buffers (positions, normals, colors if any)
        // For gizmo types 0 and 1, we create a simple mesh with given color.
        // Set material to unlit or with vertex color.
        // Upload to server.
    }
    // Update instance transform, visibility, shadow settings, GI mode.
    // If emissive intensity > 0, contribute to GI (e.g., tiny light source).
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register as emissive for global illumination.
    }
}

} // namespace lighting