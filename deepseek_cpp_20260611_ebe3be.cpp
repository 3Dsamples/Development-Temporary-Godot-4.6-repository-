// video_stream_player_3d.cpp
#include "video_stream_player_3d.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>

namespace lighting {

// ============================================================================
// Placeholder VideoStream resource (would be platform‑specific)
// ============================================================================
class VideoStream {
public:
    virtual ~VideoStream() = default;
    virtual bool open(const char* path) = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;
    virtual void play() = 0;
    virtual void stop() = 0;
    virtual bool is_playing() const = 0;
    virtual void set_paused(bool paused) = 0;
    virtual bool is_paused() const = 0;
    virtual void seek(float seconds) = 0;
    virtual float get_position() const = 0;
    virtual float get_duration() const = 0;
    virtual void set_loop(bool loop) = 0;
    virtual bool get_loop() const = 0;
    virtual void set_volume_db(float db) = 0;
    virtual float get_volume_db() const = 0;
    virtual int64_t get_current_texture_rid() const = 0; // GPU texture update each frame
};

// Dummy implementation for simulation
class DummyVideoStream : public VideoStream {
public:
    DummyVideoStream() = default;
    bool open(const char* path) override { m_open = true; return true; }
    void close() override { m_open = false; m_playing = false; }
    bool is_open() const override { return m_open; }
    void play() override { m_playing = true; m_paused = false; }
    void stop() override { m_playing = false; }
    bool is_playing() const override { return m_playing; }
    void set_paused(bool paused) override { m_paused = paused; }
    bool is_paused() const override { return m_paused; }
    void seek(float seconds) override { m_position = seconds; }
    float get_position() const override { return m_position; }
    float get_duration() const override { return 120.0f; }
    void set_loop(bool loop) override { m_loop = loop; }
    bool get_loop() const override { return m_loop; }
    void set_volume_db(float db) override { m_volume_db = db; }
    float get_volume_db() const override { return m_volume_db; }
    int64_t get_current_texture_rid() const override { return m_dummy_texture; }
private:
    bool m_open = false, m_playing = false, m_paused = false, m_loop = false;
    float m_position = 0.0f, m_volume_db = 0.0f;
    int64_t m_dummy_texture = 12345;
};

// ============================================================================
// Helper: generate plane mesh (same as TextureRect3D)
// ============================================================================
static void generate_plane(double width, double height, bool flip_h, bool flip_v,
                           std::vector<double>& out_vertices,
                           std::vector<int>& out_indices,
                           std::vector<float>& out_normals,
                           std::vector<float>& out_uvs) {
    out_vertices.clear(); out_indices.clear(); out_normals.clear(); out_uvs.clear();
    double hw = width * 0.5;
    double hh = height * 0.5;
    double verts[4][3] = {{-hw,-hh,0},{hw,-hh,0},{hw,hh,0},{-hw,hh,0}};
    float u0 = flip_h ? 1.0f : 0.0f, u1 = flip_h ? 0.0f : 1.0f;
    float v0 = flip_v ? 1.0f : 0.0f, v1 = flip_v ? 0.0f : 1.0f;
    float uvs[4][2] = {{u0,v0},{u1,v0},{u1,v1},{u0,v1}};
    for (int i=0;i<4;++i) {
        out_vertices.push_back(verts[i][0]); out_vertices.push_back(verts[i][1]); out_vertices.push_back(verts[i][2]);
        out_normals.push_back(0.0f); out_normals.push_back(0.0f); out_normals.push_back(1.0f);
        out_uvs.push_back(uvs[i][0]); out_uvs.push_back(uvs[i][1]);
    }
    out_indices = {0,1,2, 0,2,3};
}

// ============================================================================
// VideoStreamPlayer3D implementation
// ============================================================================
struct VideoStreamPlayer3D::Impl {
    VideoStream* stream = nullptr;
    bool owned_stream = false;
    char stream_path[256] = {0};
    bool loop = false;
    float volume_db = 0.0f;
    int audio_track = 0;
    char bus_name[64] = "Master";
    double width = 1.0, height = 1.0;
    bool flip_h = false, flip_v = false;
    int billboard_mode = 0;

    // Visual
    bool cast_shadow = false;   // video usually doesn't cast shadow
    bool receive_shadow = true;
    int gi_mode = 1;            // static by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;
    int64_t material_override = -1;

    bool debug_visible = false;
    float debug_color[3] = {0,1,0};

    std::function<void()> finished_callback;

    // Mesh & instance
    bool dirty = true;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    std::vector<float> uvs;

    // Video texture update (each frame)
    int64_t current_texture = -1;

    void regenerate_mesh();
    void update_render_server();
};

VideoStreamPlayer3D::VideoStreamPlayer3D() : pimpl(std::make_unique<Impl>()) {}
VideoStreamPlayer3D::~VideoStreamPlayer3D() = default;

void VideoStreamPlayer3D::set_stream(VideoStream* stream) {
    if (pimpl->owned_stream && pimpl->stream) delete pimpl->stream;
    pimpl->stream = stream;
    pimpl->owned_stream = false;
    pimpl->dirty = true;
}
VideoStream* VideoStreamPlayer3D::get_stream() const { return pimpl->stream; }

void VideoStreamPlayer3D::play() {
    if (pimpl->stream) pimpl->stream->play();
}
void VideoStreamPlayer3D::stop() {
    if (pimpl->stream) pimpl->stream->stop();
}
bool VideoStreamPlayer3D::is_playing() const {
    return pimpl->stream && pimpl->stream->is_playing();
}
void VideoStreamPlayer3D::set_paused(bool paused) {
    if (pimpl->stream) pimpl->stream->set_paused(paused);
}
bool VideoStreamPlayer3D::is_paused() const {
    return pimpl->stream && pimpl->stream->is_paused();
}
void VideoStreamPlayer3D::seek(float seconds) {
    if (pimpl->stream) pimpl->stream->seek(seconds);
}
float VideoStreamPlayer3D::get_position() const {
    return pimpl->stream ? pimpl->stream->get_position() : 0.0f;
}
float VideoStreamPlayer3D::get_duration() const {
    return pimpl->stream ? pimpl->stream->get_duration() : 0.0f;
}
void VideoStreamPlayer3D::set_loop(bool loop) { pimpl->loop = loop; if (pimpl->stream) pimpl->stream->set_loop(loop); }
bool VideoStreamPlayer3D::has_loop() const { return pimpl->loop; }
void VideoStreamPlayer3D::set_volume_db(float db) { pimpl->volume_db = db; if (pimpl->stream) pimpl->stream->set_volume_db(db); }
float VideoStreamPlayer3D::get_volume_db() const { return pimpl->volume_db; }
void VideoStreamPlayer3D::set_audio_track(int track) { pimpl->audio_track = track; }
int VideoStreamPlayer3D::get_audio_track() const { return pimpl->audio_track; }
void VideoStreamPlayer3D::set_bus(const char* bus_name) { strncpy(pimpl->bus_name, bus_name, 63); pimpl->bus_name[63]=0; }
const char* VideoStreamPlayer3D::get_bus() const { return pimpl->bus_name; }

void VideoStreamPlayer3D::set_size(double width, double height) {
    pimpl->width = std::max(0.0, width);
    pimpl->height = std::max(0.0, height);
    pimpl->dirty = true;
}
void VideoStreamPlayer3D::get_size(double& width, double& height) const {
    width = pimpl->width; height = pimpl->height;
}
void VideoStreamPlayer3D::set_flip(bool flip_h, bool flip_v) {
    pimpl->flip_h = flip_h; pimpl->flip_v = flip_v; pimpl->dirty = true;
}
void VideoStreamPlayer3D::get_flip(bool& flip_h, bool& flip_v) const {
    flip_h = pimpl->flip_h; flip_v = pimpl->flip_v;
}
void VideoStreamPlayer3D::set_billboard_mode(int mode) { pimpl->billboard_mode = mode; pimpl->dirty = true; }
int VideoStreamPlayer3D::get_billboard_mode() const { return pimpl->billboard_mode; }

void VideoStreamPlayer3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void VideoStreamPlayer3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void VideoStreamPlayer3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void VideoStreamPlayer3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void VideoStreamPlayer3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void VideoStreamPlayer3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}
void VideoStreamPlayer3D::set_material_override(int64_t material_rid) { pimpl->material_override = material_rid; pimpl->dirty = true; }
int64_t VideoStreamPlayer3D::get_material_override() const { return pimpl->material_override; }

void VideoStreamPlayer3D::set_debug_visible(bool visible) { pimpl->debug_visible = visible; pimpl->dirty = true; }
bool VideoStreamPlayer3D::is_debug_visible() const { return pimpl->debug_visible; }
void VideoStreamPlayer3D::set_debug_color(const float* rgb) { memcpy(pimpl->debug_color, rgb, 3*sizeof(float)); pimpl->dirty = true; }
void VideoStreamPlayer3D::get_debug_color(float* out_rgb) const { memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float)); }
void VideoStreamPlayer3D::set_finished_callback(std::function<void()> callback) { pimpl->finished_callback = callback; }

void VideoStreamPlayer3D::Impl::regenerate_mesh() {
    generate_plane(width, height, flip_h, flip_v, vertices, indices, normals, uvs);
    // Compute AABB
    double min_x=vertices[0], max_x=vertices[0];
    double min_y=vertices[1], max_y=vertices[1];
    double min_z=vertices[2], max_z=vertices[2];
    for (size_t i=3; i<vertices.size(); i+=3) {
        min_x = std::min(min_x, vertices[i]);
        max_x = std::max(max_x, vertices[i]);
        min_y = std::min(min_y, vertices[i+1]);
        max_y = std::max(max_y, vertices[i+1]);
        min_z = std::min(min_z, vertices[i+2]);
        max_z = std::max(max_z, vertices[i+2]);
    }
    set_aabb(&min_x, &max_x);
    double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
}

void VideoStreamPlayer3D::Impl::update_render_server() {
    if (mesh_rid == -1) {
        // mesh_rid = RenderingServer::mesh_create();
    }
    // Clear surfaces and add new one with vertices, normals, uvs, indices.
    // Set material to either material_override or a default video material with current texture.
    if (stream && stream->is_open() && stream->is_playing()) {
        current_texture = stream->get_current_texture_rid();
    } else {
        current_texture = -1;
    }
    // In real engine, we would assign texture to material.
    if (instance_rid == -1) {
        // instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(instance_rid, mesh_rid);
}

void VideoStreamPlayer3D::update_surface() {
    if (!pimpl->dirty) return;
    pimpl->regenerate_mesh();
    pimpl->update_render_server();
    pimpl->dirty = false;
}

void VideoStreamPlayer3D::ready() {
    GeometryInstance3D::ready();
    if (!pimpl->stream) {
        // create dummy stream for simulation
        pimpl->stream = new DummyVideoStream();
        pimpl->owned_stream = true;
    }
    update_surface();
}

void VideoStreamPlayer3D::process(double delta) {
    GeometryInstance3D::process(delta);
    if (pimpl->stream && pimpl->stream->is_playing()) {
        // Update texture each frame if needed (but get_current_texture_rid already does)
        pimpl->current_texture = pimpl->stream->get_current_texture_rid();
        // Check if finished (non‑looping)
        if (!pimpl->loop && pimpl->stream->get_position() >= pimpl->stream->get_duration()-0.05) {
            if (pimpl->finished_callback) pimpl->finished_callback();
        }
        // For billboard mode, we could update instance transform but rendering server may handle.
        // Mark dirty so texture updates.
        pimpl->dirty = true;
    }
    if (pimpl->dirty) update_surface();
}

void VideoStreamPlayer3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->dirty) update_surface();
    // Update instance transform (support billboard)
    if (pimpl->instance_rid != -1) {
        if (pimpl->billboard_mode == 1) {
            // For full billboard, we would set transform to identity and rely on shader.
            // In Godot, the instance's transform is used, so we need to compute a facing-camera matrix.
            // Simplified: we don't implement full billboard here.
        }
        Transform3D global = get_global_transform();
        // RenderingServer::instance_set_transform(pimpl->instance_rid, global);
    }
    // If emissive, add to GI system (placeholder)
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // register emissive surface
    }
}

} // namespace lighting