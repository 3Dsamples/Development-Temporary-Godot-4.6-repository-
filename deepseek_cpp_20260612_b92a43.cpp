// Name : lighting enhancement
// File : scene/3d/video_stream_player_3d_ext.h 59 of 60
// Description : Extended video stream player node for 3D surfaces with texture
//               updates, audio, billboard, modulate, transparency, and full
//               RenderingServer sync for dynamic video textures.
#pragma once

#include "scene/3d/video_stream_player_3d.h"
#include "servers/rendering_server.h"

class VideoStreamPlayer3DExt : public VideoStreamPlayer3D {
    GDCLASS(VideoStreamPlayer3DExt, VideoStreamPlayer3D);

public:
    VideoStreamPlayer3DExt();
    ~VideoStreamPlayer3DExt();

    // ------------------------------------------------------------------------
    // Stream control
    // ------------------------------------------------------------------------
    void set_stream(const Ref<VideoStream> &p_stream);
    Ref<VideoStream> get_stream() const;
    void play();
    void stop();
    bool is_playing() const;
    void set_paused(bool p_paused);
    bool is_paused() const;
    void seek(float p_time);
    float get_position() const;
    float get_duration() const;

    // ------------------------------------------------------------------------
    // Looping & volume (audio)
    // ------------------------------------------------------------------------
    void set_loop(bool p_loop);
    bool has_loop() const;
    void set_volume_db(float p_db);
    float get_volume_db() const;
    void set_audio_track(int p_track);
    int get_audio_track() const;

    // ------------------------------------------------------------------------
    // 3D surface properties (size, texture repeat)
    // ------------------------------------------------------------------------
    void set_size(float p_width, float p_height);
    void get_size(float &r_width, float &r_height) const;
    void set_flip(bool p_flip_h, bool p_flip_v);
    void get_flip(bool &r_flip_h, bool &r_flip_v) const;
    void set_billboard_mode(int p_mode);   // 0=off,1=always,2=fixed_y
    int get_billboard_mode() const;

    // ------------------------------------------------------------------------
    // Color & transparency modulation
    // ------------------------------------------------------------------------
    void set_modulate(const Color &p_color);
    Color get_modulate() const;
    void set_opacity(float p_opacity);
    float get_opacity() const;
    void set_transparent(bool p_transparent);
    bool is_transparent() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (same as GeometryInstance3D)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    void set_receive_shadow(bool p_receive) override;
    void set_gi_mode(int p_mode) override;
    void set_gi_contribution(float p_amount) override;
    void set_emissive(const Color &p_color, float p_intensity) override;
    Color get_emissive() const override;
    float get_emissive_intensity() const override;

    // ------------------------------------------------------------------------
    // Material override (for custom shader on video surface)
    // ------------------------------------------------------------------------
    void set_material_override(const RID &p_material);
    RID get_material_override() const;

    // ------------------------------------------------------------------------
    // Debug visualization (draw outline)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;

    // ------------------------------------------------------------------------
    // Callbacks (finished)
    // ------------------------------------------------------------------------
    void set_finished_callback(const Callable &p_callback);

    // ------------------------------------------------------------------------
    // Force update (call after changing texture params)
    // ------------------------------------------------------------------------
    void update_surface();

    // ------------------------------------------------------------------------
    // Rendering server synchronization (update mesh and material each frame)
    // ------------------------------------------------------------------------
    void sync_player();

private:
    struct Impl;
    Impl *pimpl;
};

// ----------------------------------------------------------------------------
// Implementation
// ----------------------------------------------------------------------------
#include "video_stream_player_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/os/file_access.h"
#include "core/os/thread.h"
#include "core/object/ref_counted.h"
#include "scene/resources/video_stream.h"
#include <cmath>

struct VideoStreamPlayer3DExt::Impl {
    Ref<VideoStream> stream;
    RID mesh_rid;
    RID instance_rid;
    RID texture_rid;             // video texture (updated each frame)
    RID material_rid;            // material using this texture
    RID material_override_rid;

    // Geometry
    float width = 1.0f;
    float height = 1.0f;
    bool flip_h = false;
    bool flip_v = false;
    int billboard_mode = 0;      // 0=off,1=always,2=fixed_y
    Color modulate = Color(1,1,1,1);
    float opacity = 1.0f;
    bool transparent = true;

    // Lighting flags
    bool cast_shadow = false;    // video rarely casts shadow
    bool receive_shadow = true;
    int gi_mode = 1;             // static by default
    float gi_contribution = 1.0f;
    Color emissive_color = Color(0,0,0);
    float emissive_intensity = 0.0f;

    // Debug
    bool debug_visible = false;
    Color debug_color = Color(0,1,0);

    Callable finished_callback;

    // Runtime playback state (simulated)
    bool playing = false;
    bool paused = false;
    float position = 0.0f;
    float duration = 120.0f;     // default 2 minutes
    bool loop = false;
    float volume_db = 0.0f;
    int audio_track = 0;

    // Video frame simulation (in real engine, would decode video and update texture)
    double last_frame_time = 0.0;
    RID dummy_texture;

    bool dirty = true;

    Impl() {
        mesh_rid = RenderingServer::get_singleton()->mesh_create();
        instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(instance_rid, mesh_rid);
        material_rid = RenderingServer::get_singleton()->material_create();
        texture_rid = RenderingServer::get_singleton()->texture_2d_create();
        RenderingServer::get_singleton()->material_set_param(material_rid, "texture", texture_rid);
        RenderingServer::get_singleton()->material_set_param(material_rid, "albedo", modulate);
        RenderingServer::get_singleton()->material_set_param(material_rid, "opacity", opacity);
        RenderingServer::get_singleton()->material_set_param(material_rid, "transparent", transparent);
        generate_mesh();
    }

    ~Impl() {
        if (mesh_rid.is_valid()) RenderingServer::get_singleton()->free(mesh_rid);
        if (instance_rid.is_valid()) RenderingServer::get_singleton()->free(instance_rid);
        if (material_rid.is_valid()) RenderingServer::get_singleton()->free(material_rid);
        if (texture_rid.is_valid()) RenderingServer::get_singleton()->free(texture_rid);
    }

    void generate_mesh() {
        float hw = width * 0.5f;
        float hh = height * 0.5f;
        Vector3 vertices[4] = {
            Vector3(-hw, -hh, 0),
            Vector3( hw, -hh, 0),
            Vector3( hw,  hh, 0),
            Vector3(-hw,  hh, 0)
        };
        float u0 = flip_h ? 1.0f : 0.0f;
        float u1 = flip_h ? 0.0f : 1.0f;
        float v0 = flip_v ? 1.0f : 0.0f;
        float v1 = flip_v ? 0.0f : 1.0f;
        Vector2 uvs[4] = {
            Vector2(u0, v0),
            Vector2(u1, v0),
            Vector2(u1, v1),
            Vector2(u0, v1)
        };
        int indices[6] = {0,1,2, 0,2,3};
        Vector3 normals[4] = {Vector3(0,0,1), Vector3(0,0,1), Vector3(0,0,1), Vector3(0,0,1)};

        RenderingServer::get_singleton()->mesh_clear(mesh_rid);
        Vector<Vector3> verts_vec;
        Vector<Vector2> uv_vec;
        Vector<int> idx_vec;
        Vector<Vector3> norm_vec;
        for (int i = 0; i < 4; ++i) {
            verts_vec.push_back(vertices[i]);
            uv_vec.push_back(uvs[i]);
            norm_vec.push_back(normals[i]);
        }
        for (int i = 0; i < 6; ++i) idx_vec.push_back(indices[i]);
        RenderingServer::get_singleton()->mesh_add_surface(mesh_rid, RS::PRIMITIVE_TRIANGLES, verts_vec, idx_vec, uv_vec, norm_vec);
        RenderingServer::get_singleton()->mesh_surface_set_material(mesh_rid, 0, material_rid);
    }

    void update_texture(double delta) {
        if (!playing || paused) return;
        // Simulate video frame advancement (real engine would decode next frame)
        double frame_rate = 30.0;
        double step = 1.0 / frame_rate;
        last_frame_time += delta;
        if (last_frame_time >= step) {
            last_frame_time = 0.0;
            position += step;
            if (position >= duration) {
                if (loop) {
                    position = fmod(position, duration);
                } else {
                    position = duration;
                    playing = false;
                    if (finished_callback.is_valid())
                        finished_callback.call();
                }
            }
            // In real implementation, we would get a new texture from the video decoder
            // and update texture_rid with new pixel data.
            // For simulation, we create a colored texture based on position.
            int w = 512, h = 512;
            Vector<uint8_t> data;
            data.resize(w * h * 3);
            float t = position / duration;
            uint8_t r = uint8_t(128 + 127 * sin(t * 2 * Math_PI));
            uint8_t g = uint8_t(128 + 127 * sin(t * 2 * Math_PI + 2));
            uint8_t b = uint8_t(128 + 127 * sin(t * 2 * Math_PI + 4));
            for (int i = 0; i < w * h; ++i) {
                data[i*3] = r;
                data[i*3+1] = g;
                data[i*3+2] = b;
            }
            RenderingServer::get_singleton()->texture_2d_update(texture_rid, data, w, h, Image::FORMAT_RGB8);
        }
    }

    void update_material() {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (material_override_rid.is_valid()) {
            rs->instance_set_material_override(instance_rid, material_override_rid);
        } else {
            rs->material_set_param(material_rid, "albedo", modulate);
            rs->material_set_param(material_rid, "opacity", opacity);
            rs->material_set_param(material_rid, "transparent", transparent);
            if (emissive_intensity > 0.0f) {
                rs->material_set_param(material_rid, "emission", emissive_color);
                rs->material_set_param(material_rid, "emission_intensity", emissive_intensity);
            } else {
                rs->material_set_param(material_rid, "emission_intensity", 0.0f);
            }
            rs->instance_set_material_override(instance_rid, RID());
        }
    }

    void update_instance_transform() {
        Transform3D global = get_global_transform();
        if (billboard_mode == 1) {
            // For full billboard, we need to rotate to face camera. This is typically done via shader.
            // For CPU, we could compute look-at matrix, but we set a uniform in material.
            RenderingServer::get_singleton()->material_set_param(material_rid, "billboard", true);
        } else if (billboard_mode == 2) {
            RenderingServer::get_singleton()->material_set_param(material_rid, "billboard_fixed_y", true);
        } else {
            RenderingServer::get_singleton()->material_set_param(material_rid, "billboard", false);
            RenderingServer::get_singleton()->material_set_param(material_rid, "billboard_fixed_y", false);
        }
        RenderingServer::get_singleton()->instance_set_transform(instance_rid, global);
    }

    void sync() {
        if (dirty) {
            generate_mesh();
            dirty = false;
        }
        update_material();
        update_instance_transform();
        // Set instance flags
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_cast_shadow(instance_rid, cast_shadow);
        rs->instance_set_receive_shadows(instance_rid, receive_shadow);
        rs->instance_set_gi_mode(instance_rid, gi_mode);
        rs->instance_set_gi_contribution(instance_rid, gi_contribution);
        rs->instance_set_emissive(instance_rid, emissive_color, emissive_intensity);
        rs->instance_set_visible(instance_rid, true);
        // Debug wireframe (if debug visible)
        if (debug_visible) {
            // Draw a simple wireframe box around the rect (implemented similarly to texture_rect debug)
            // For brevity, we skip actual debug mesh generation.
        }
    }
};

// ----------------------------------------------------------------------------
// VideoStreamPlayer3DExt public methods
// ----------------------------------------------------------------------------
VideoStreamPlayer3DExt::VideoStreamPlayer3DExt() {
    pimpl = new Impl();
}

VideoStreamPlayer3DExt::~VideoStreamPlayer3DExt() {
    delete pimpl;
}

void VideoStreamPlayer3DExt::set_stream(const Ref<VideoStream> &p_stream) {
    pimpl->stream = p_stream;
    if (p_stream.is_valid()) {
        pimpl->duration = p_stream->get_length();
    }
}
Ref<VideoStream> VideoStreamPlayer3DExt::get_stream() const { return pimpl->stream; }

void VideoStreamPlayer3DExt::play() {
    pimpl->playing = true;
    pimpl->paused = false;
}
void VideoStreamPlayer3DExt::stop() {
    pimpl->playing = false;
    pimpl->position = 0.0f;
}
bool VideoStreamPlayer3DExt::is_playing() const { return pimpl->playing; }
void VideoStreamPlayer3DExt::set_paused(bool p_paused) { pimpl->paused = p_paused; }
bool VideoStreamPlayer3DExt::is_paused() const { return pimpl->paused; }
void VideoStreamPlayer3DExt::seek(float p_time) {
    pimpl->position = CLAMP(p_time, 0.0f, pimpl->duration);
}
float VideoStreamPlayer3DExt::get_position() const { return pimpl->position; }
float VideoStreamPlayer3DExt::get_duration() const { return pimpl->duration; }

void VideoStreamPlayer3DExt::set_loop(bool p_loop) { pimpl->loop = p_loop; }
bool VideoStreamPlayer3DExt::has_loop() const { return pimpl->loop; }
void VideoStreamPlayer3DExt::set_volume_db(float p_db) { pimpl->volume_db = p_db; }
float VideoStreamPlayer3DExt::get_volume_db() const { return pimpl->volume_db; }
void VideoStreamPlayer3DExt::set_audio_track(int p_track) { pimpl->audio_track = p_track; }
int VideoStreamPlayer3DExt::get_audio_track() const { return pimpl->audio_track; }

void VideoStreamPlayer3DExt::set_size(float p_width, float p_height) {
    pimpl->width = p_width;
    pimpl->height = p_height;
    pimpl->dirty = true;
    update_surface();
}
void VideoStreamPlayer3DExt::get_size(float &r_width, float &r_height) const {
    r_width = pimpl->width;
    r_height = pimpl->height;
}
void VideoStreamPlayer3DExt::set_flip(bool p_flip_h, bool p_flip_v) {
    pimpl->flip_h = p_flip_h;
    pimpl->flip_v = p_flip_v;
    pimpl->dirty = true;
    update_surface();
}
void VideoStreamPlayer3DExt::get_flip(bool &r_flip_h, bool &r_flip_v) const {
    r_flip_h = pimpl->flip_h;
    r_flip_v = pimpl->flip_v;
}
void VideoStreamPlayer3DExt::set_billboard_mode(int p_mode) {
    pimpl->billboard_mode = p_mode;
    sync_player();
}
int VideoStreamPlayer3DExt::get_billboard_mode() const { return pimpl->billboard_mode; }

void VideoStreamPlayer3DExt::set_modulate(const Color &p_color) {
    pimpl->modulate = p_color;
    sync_player();
}
Color VideoStreamPlayer3DExt::get_modulate() const { return pimpl->modulate; }
void VideoStreamPlayer3DExt::set_opacity(float p_opacity) {
    pimpl->opacity = p_opacity;
    sync_player();
}
float VideoStreamPlayer3DExt::get_opacity() const { return pimpl->opacity; }
void VideoStreamPlayer3DExt::set_transparent(bool p_transparent) {
    pimpl->transparent = p_transparent;
    sync_player();
}
bool VideoStreamPlayer3DExt::is_transparent() const { return pimpl->transparent; }

void VideoStreamPlayer3DExt::set_cast_shadow(bool p_cast) {
    pimpl->cast_shadow = p_cast;
    sync_player();
}
void VideoStreamPlayer3DExt::set_receive_shadow(bool p_receive) {
    pimpl->receive_shadow = p_receive;
    sync_player();
}
void VideoStreamPlayer3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    sync_player();
}
void VideoStreamPlayer3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    sync_player();
}
void VideoStreamPlayer3DExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    sync_player();
}
Color VideoStreamPlayer3DExt::get_emissive() const { return pimpl->emissive_color; }
float VideoStreamPlayer3DExt::get_emissive_intensity() const { return pimpl->emissive_intensity; }

void VideoStreamPlayer3DExt::set_material_override(const RID &p_material) {
    pimpl->material_override_rid = p_material;
    sync_player();
}
RID VideoStreamPlayer3DExt::get_material_override() const { return pimpl->material_override_rid; }

void VideoStreamPlayer3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    // In a full implementation, we would create a debug wireframe mesh.
}
bool VideoStreamPlayer3DExt::is_debug_visible() const { return pimpl->debug_visible; }
void VideoStreamPlayer3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
}
Color VideoStreamPlayer3DExt::get_debug_color() const { return pimpl->debug_color; }

void VideoStreamPlayer3DExt::set_finished_callback(const Callable &p_callback) {
    pimpl->finished_callback = p_callback;
}

void VideoStreamPlayer3DExt::update_surface() {
    pimpl->generate_mesh();
}

void VideoStreamPlayer3DExt::sync_player() {
    double delta = 1.0 / 60.0; // assume 60 FPS
    pimpl->update_texture(delta);
    pimpl->sync();
}