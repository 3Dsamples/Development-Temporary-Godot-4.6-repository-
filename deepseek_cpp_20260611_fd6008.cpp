// reflection_probe.cpp
#include "reflection_probe.h"
#include <cmath>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

namespace lighting {

// ============================================================================
// Helper: Cubemap renderer (simulated – in real engine would call scene renderer)
// ============================================================================
class CubemapRenderer {
public:
    static void render_cubemap(const double* position, int resolution,
                               std::vector<std::vector<float>>& out_faces,
                               uint32_t cull_mask) {
        // In real implementation, render scene 6 times with ortho/perspective.
        // For demo, fill with a gradient.
        out_faces.resize(6);
        for (int face = 0; face < 6; ++face) {
            out_faces[face].assign(resolution * resolution * 3, 0.5f);
            // Add a simple color per face to differentiate
            float r = (face == 0) ? 0.8f : (face == 1) ? 0.2f : 0.5f;
            float g = (face == 2) ? 0.8f : (face == 3) ? 0.2f : 0.5f;
            float b = (face == 4) ? 0.8f : (face == 5) ? 0.2f : 0.5f;
            for (size_t i = 0; i < out_faces[face].size() / 3; ++i) {
                out_faces[face][i*3] = r;
                out_faces[face][i*3+1] = g;
                out_faces[face][i*3+2] = b;
            }
        }
    }
};

// ============================================================================
// ReflectionProbe implementation
// ============================================================================
struct ReflectionProbe::Impl {
    ReflectionUpdateMode update_mode = ReflectionUpdateMode::STATIC;
    float update_freq_hz = 2.0f;         // updates per second
    int resolution = 256;
    bool roughness_filtering = true;

    ReflectionProbeShape shape = ReflectionProbeShape::BOX;
    double extents[3] = {5.0, 5.0, 5.0}; // half extents / radius
    double blend_distance = 1.0;
    float intensity = 1.0f;
    float ambient_color[3] = {0.0f, 0.0f, 0.0f};
    bool ambient_mode = false;            // interior

    uint32_t reflection_mask = 0xFFFFFFFF;

    float gi_contribution = 1.0f;
    float emissive_color[3] = {0.0f, 0.0f, 0.0f};
    float emissive_intensity = 0.0f;
    bool cast_shadow = false;             // debug mesh only

    bool debug_visible = false;
    float debug_color[3] = {0.2f, 0.8f, 1.0f};

    // Captured cubemap
    std::vector<std::vector<float>> captured_faces; // 6 faces, each resolution*resolution*3 floats
    int64_t cubemap_texture_rid = -1;    // GPU texture
    bool captured = false;
    bool capture_dirty = true;

    // Capture async state
    std::atomic<bool> capturing{false};
    std::thread capture_thread;
    std::function<void()> on_capture_complete;

    // Debug mesh for influence area (box/sphere)
    int64_t debug_mesh_rid = -1;
    int64_t debug_instance_rid = -1;

    // Timer for dynamic updates
    double update_timer = 0.0;

    ~Impl() {
        if (capture_thread.joinable()) capture_thread.join();
        if (cubemap_texture_rid != -1) {
            // RenderingServer::texture_free(cubemap_texture_rid);
        }
    }

    void do_capture();
    void upload_cubemap();
    void update_debug_mesh();
};

ReflectionProbe::ReflectionProbe() : pimpl(std::make_unique<Impl>()) {}
ReflectionProbe::~ReflectionProbe() = default;

void ReflectionProbe::set_update_mode(ReflectionUpdateMode mode) { pimpl->update_mode = mode; pimpl->capture_dirty = true; }
ReflectionUpdateMode ReflectionProbe::get_update_mode() const { return pimpl->update_mode; }
void ReflectionProbe::set_update_frequency(float frames_per_second) { pimpl->update_freq_hz = frames_per_second; }
float ReflectionProbe::get_update_frequency() const { return pimpl->update_freq_hz; }
void ReflectionProbe::capture() {
    if (pimpl->capturing) return;
    pimpl->do_capture();
}
void ReflectionProbe::capture_async() {
    if (pimpl->capturing) return;
    pimpl->capturing = true;
    pimpl->capture_thread = std::thread([this]() {
        pimpl->do_capture();
        pimpl->capturing = false;
        if (pimpl->on_capture_complete) pimpl->on_capture_complete();
    });
}
bool ReflectionProbe::is_capturing() const { return pimpl->capturing; }

void ReflectionProbe::set_resolution(int resolution) {
    pimpl->resolution = std::max(16, resolution);
    pimpl->capture_dirty = true;
}
int ReflectionProbe::get_resolution() const { return pimpl->resolution; }
void ReflectionProbe::set_roughness_filtering(bool enable) { pimpl->roughness_filtering = enable; }
bool ReflectionProbe::is_roughness_filtering_enabled() const { return pimpl->roughness_filtering; }

void ReflectionProbe::set_shape(ReflectionProbeShape shape) { pimpl->shape = shape; pimpl->update_debug_mesh(); }
ReflectionProbeShape ReflectionProbe::get_shape() const { return pimpl->shape; }
void ReflectionProbe::set_extents(const double* extents) { memcpy(pimpl->extents, extents, 3*sizeof(double)); pimpl->update_debug_mesh(); }
void ReflectionProbe::get_extents(double* out_extents) const { memcpy(out_extents, pimpl->extents, 3*sizeof(double)); }

void ReflectionProbe::set_blend_distance(double distance) { pimpl->blend_distance = distance; }
double ReflectionProbe::get_blend_distance() const { return pimpl->blend_distance; }
void ReflectionProbe::set_intensity(float intensity) { pimpl->intensity = intensity; }
float ReflectionProbe::get_intensity() const { return pimpl->intensity; }
void ReflectionProbe::set_ambient_color(const float* rgb) { memcpy(pimpl->ambient_color, rgb, 3*sizeof(float)); }
void ReflectionProbe::get_ambient_color(float* out_rgb) const { memcpy(out_rgb, pimpl->ambient_color, 3*sizeof(float)); }
void ReflectionProbe::set_ambient_mode(bool interior) { pimpl->ambient_mode = interior; }
bool ReflectionProbe::get_ambient_mode() const { return pimpl->ambient_mode; }

void ReflectionProbe::set_reflection_mask(uint32_t mask) { pimpl->reflection_mask = mask; }
uint32_t ReflectionProbe::get_reflection_mask() const { return pimpl->reflection_mask; }

void ReflectionProbe::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float ReflectionProbe::get_gi_contribution() const { return pimpl->gi_contribution; }
void ReflectionProbe::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void ReflectionProbe::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}
void ReflectionProbe::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool ReflectionProbe::get_cast_shadow() const { return pimpl->cast_shadow; }

void ReflectionProbe::set_debug_visible(bool visible) { pimpl->debug_visible = visible; pimpl->update_debug_mesh(); }
bool ReflectionProbe::is_debug_visible() const { return pimpl->debug_visible; }
void ReflectionProbe::set_debug_color(const float* rgb) { memcpy(pimpl->debug_color, rgb, 3*sizeof(float)); pimpl->update_debug_mesh(); }
void ReflectionProbe::get_debug_color(float* out_rgb) const { memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float)); }

int64_t ReflectionProbe::get_cubemap_texture_rid() const { return pimpl->cubemap_texture_rid; }

void ReflectionProbe::Impl::do_capture() {
    if (capturing) return;
    capturing = true;
    // Render cubemap at probe position
    Transform3D global = get_global_transform();
    double pos[3] = {global.origin[0], global.origin[1], global.origin[2]};
    CubemapRenderer::render_cubemap(pos, resolution, captured_faces, reflection_mask);
    captured = true;
    capture_dirty = true;
    capturing = false;
}

void ReflectionProbe::Impl::upload_cubemap() {
    if (!capture_dirty || captured_faces.empty()) return;
    if (cubemap_texture_rid == -1) {
        // cubemap_texture_rid = RenderingServer::texture_create_cubemap(resolution, resolution);
    }
    // For each face, upload data
    for (int face = 0; face < 6; ++face) {
        // RenderingServer::texture_set_data(cubemap_texture_rid, face, captured_faces[face].data());
    }
    capture_dirty = false;
}

void ReflectionProbe::Impl::update_debug_mesh() {
    if (!debug_visible) {
        if (debug_instance_rid != -1) {
            // RenderingServer::instance_set_visible(debug_instance_rid, false);
        }
        return;
    }
    // Generate wireframe shape (box or sphere)
    if (shape == ReflectionProbeShape::BOX) {
        // Generate cube wireframe using lines
        double hx = extents[0], hy = extents[1], hz = extents[2];
        std::vector<double> verts = {
            -hx,-hy,-hz,  hx,-hy,-hz,  hx,-hy, hz, -hx,-hy, hz,
            -hx, hy,-hz,  hx, hy,-hz,  hx, hy, hz, -hx, hy, hz
        };
        std::vector<int> edges = {
            0,1, 1,2, 2,3, 3,0,
            4,5, 5,6, 6,7, 7,4,
            0,4, 1,5, 2,6, 3,7
        };
        // Upload as line primitive
    } else if (shape == ReflectionProbeShape::SPHERE) {
        // generate sphere wireframe (latitude/longitude lines)
    }
    // Set material color (unlit, with debug_color)
    Transform3D global = get_global_transform();
    if (debug_instance_rid == -1) {
        // debug_instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(debug_instance_rid, debug_mesh_rid);
    // RenderingServer::instance_set_transform(debug_instance_rid, global);
    // RenderingServer::instance_set_visible(debug_instance_rid, true);
}

void ReflectionProbe::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // Handle periodic capture for dynamic mode
    if (pimpl->update_mode == ReflectionUpdateMode::DYNAMIC && !pimpl->capturing) {
        pimpl->update_timer += delta;
        if (pimpl->update_timer >= 1.0 / pimpl->update_freq_hz) {
            pimpl->update_timer = 0.0;
            capture_async();
        }
    } else if (pimpl->update_mode == ReflectionUpdateMode::STATIC && !pimpl->captured && !pimpl->capturing) {
        capture_async(); // initial capture
    }
    // Upload cubemap if dirty
    if (pimpl->capture_dirty && !pimpl->captured_faces.empty()) {
        pimpl->upload_cubemap();
    }
    // Update debug mesh transform
    if (pimpl->debug_visible && pimpl->debug_instance_rid != -1 && is_transform_dirty()) {
        Transform3D global = get_global_transform();
        // RenderingServer::instance_set_transform(pimpl->debug_instance_rid, global);
    }
    // If emissive and GI contribution > 0, register probe as GI source (placeholder)
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_contribution > 0.0f) {
        // Inject into GI system (e.g., add to light propagation volume)
    }
}

void ReflectionProbe::process(double delta) {
    Node3D::process(delta);
    // nothing extra
}

} // namespace lighting