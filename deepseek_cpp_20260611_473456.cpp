// marker_3d.cpp
#include "marker_3d.h"
#include <cstring>
#include <cmath>

namespace lighting {

struct Marker3D::Impl {
    bool debug_visible = true;
    double debug_size = 0.1;
    float debug_color[4] = {1.0f, 1.0f, 0.0f, 1.0f}; // yellow

    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool cast_shadow = false;   // markers never cast shadows
    bool receive_shadow = false;
    int gi_mode = 0;            // off by default
    float gi_contribution = 0.0f;

    // Rendering server handle for debug mesh
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;
    bool dirty = true;

    void create_debug_mesh();
};

Marker3D::Marker3D() : pimpl(std::make_unique<Impl>()) {}
Marker3D::~Marker3D() = default;

void Marker3D::set_debug_visible(bool visible) { pimpl->debug_visible = visible; pimpl->dirty = true; }
bool Marker3D::is_debug_visible() const { return pimpl->debug_visible; }
void Marker3D::set_debug_size(double size) { pimpl->debug_size = std::max(0.001, size); pimpl->dirty = true; }
double Marker3D::get_debug_size() const { return pimpl->debug_size; }
void Marker3D::set_debug_color(float r, float g, float b, float a) {
    pimpl->debug_color[0]=r; pimpl->debug_color[1]=g; pimpl->debug_color[2]=b; pimpl->debug_color[3]=a;
    pimpl->dirty = true;
}
void Marker3D::get_debug_color(float* out_rgba) const { memcpy(out_rgba, pimpl->debug_color, 4*sizeof(float)); }

void Marker3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void Marker3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}
void Marker3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
void Marker3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void Marker3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
void Marker3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }

void Marker3D::Impl::create_debug_mesh() {
    // Build a simple cross mesh (three orthogonal lines) to represent the marker.
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> colors; // per vertex color

    double s = debug_size;
    // X axis line: from (-s,0,0) to (s,0,0)
    vertices.push_back(-s); vertices.push_back(0.0); vertices.push_back(0.0);
    vertices.push_back( s); vertices.push_back(0.0); vertices.push_back(0.0);
    // Y axis line
    vertices.push_back(0.0); vertices.push_back(-s); vertices.push_back(0.0);
    vertices.push_back(0.0); vertices.push_back( s); vertices.push_back(0.0);
    // Z axis line
    vertices.push_back(0.0); vertices.push_back(0.0); vertices.push_back(-s);
    vertices.push_back(0.0); vertices.push_back(0.0); vertices.push_back( s);

    // indices for lines (each line is 2 vertices)
    for (int i = 0; i < 6; i += 2) {
        indices.push_back(i);
        indices.push_back(i+1);
    }

    // colors: assign per vertex (red for X, green for Y, blue for Z)
    for (int i = 0; i < 2; ++i) { colors.push_back(1.0f); colors.push_back(0.0f); colors.push_back(0.0f); } // X red
    for (int i = 0; i < 2; ++i) { colors.push_back(0.0f); colors.push_back(1.0f); colors.push_back(0.0f); } // Y green
    for (int i = 0; i < 2; ++i) { colors.push_back(0.0f); colors.push_back(0.0f); colors.push_back(1.0f); } // Z blue

    // In a real engine, we would create a mesh resource and assign it to instance_rid.
    // For now, we store the data and mark as ready.
    // Since rendering server integration is out of scope for this sample, we skip actual upload.
    // But we set dirty false after preparing.
    dirty = false;
}

void Marker3D::process(double delta) {
    Node3D::process(delta);
    // Nothing dynamic.
}

void Marker3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        pimpl->create_debug_mesh();
    }
    if (pimpl->debug_visible && pimpl->mesh_rid != -1) {
        // Update transform and visibility of debug instance.
        // If emissive intensity >0 and gi_mode>0, also contribute to GI (though unlikely for marker).
    } else if (pimpl->mesh_rid != -1) {
        // Hide instance.
    }
    // Update bounding box (small for marker, but for completeness)
    double s = pimpl->debug_size;
    double aabb_min[3] = {-s, -s, -s};
    double aabb_max[3] = {s, s, s};
    set_aabb(aabb_min, aabb_max);
    set_bounding_sphere_radius(s * 1.732);
}

} // namespace lighting