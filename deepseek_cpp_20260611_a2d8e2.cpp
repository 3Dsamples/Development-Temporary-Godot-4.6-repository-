// video_stream_player_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <functional>

namespace lighting {

// ============================================================================
// VideoStreamPlayer3D – plays video on a 3D surface (texture).
// Supports video/audio playback, looping, pause/resume, volume, and full
// lighting (shadows, GI, emissive). Can also act as a billboard.
// ============================================================================

class VideoStream; // forward (resource)

class VideoStreamPlayer3D : public GeometryInstance3D {
public:
    VideoStreamPlayer3D();
    ~VideoStreamPlayer3D();

    // ------------------------------------------------------------------------
    // Stream control
    // ------------------------------------------------------------------------
    void set_stream(VideoStream* stream);
    VideoStream* get_stream() const;
    void play();
    void stop();
    bool is_playing() const;
    void set_paused(bool paused);
    bool is_paused() const;
    void seek(float seconds);
    float get_position() const;
    float get_duration() const;

    // ------------------------------------------------------------------------
    // Looping & volume (audio)
    // ------------------------------------------------------------------------
    void set_loop(bool loop);
    bool has_loop() const;
    void set_volume_db(float db);
    float get_volume_db() const;
    void set_audio_track(int track);
    int get_audio_track() const;
    void set_bus(const char* bus_name);
    const char* get_bus() const;

    // ------------------------------------------------------------------------
    // 3D surface properties (size, texture repeat)
    // ------------------------------------------------------------------------
    void set_size(double width, double height);
    void get_size(double& width, double& height) const;
    void set_flip(bool flip_h, bool flip_v);
    void get_flip(bool& flip_h, bool& flip_v) const;
    void set_billboard_mode(int mode);   // 0=off,1=always,2=fixed_y
    int get_billboard_mode() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (same as GeometryInstance3D)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;
    void get_emissive(float* out_color, float& out_intensity) const override;

    // ------------------------------------------------------------------------
    // Material override (for custom shader on video surface)
    // ------------------------------------------------------------------------
    void set_material_override(int64_t material_rid);
    int64_t get_material_override() const;

    // ------------------------------------------------------------------------
    // Debug visualization (draw outline)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;

    // ------------------------------------------------------------------------
    // Callbacks
    // ------------------------------------------------------------------------
    void set_finished_callback(std::function<void()> callback);

    // ------------------------------------------------------------------------
    // Force update (call after changing texture params)
    // ------------------------------------------------------------------------
    void update_surface();

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void ready() override;
    void process(double delta) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting