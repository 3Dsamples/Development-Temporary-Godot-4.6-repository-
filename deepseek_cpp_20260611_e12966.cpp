// marker_3d.h
#pragma once

#include "visual_instance_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// Marker3D – a debug and utility node representing a position in 3D space.
// Displays a customizable gizmo (sphere, axes, custom mesh) and can optionally
// cast shadows, receive GI, and emit light for debugging lighting setups.
// Optimized for thousands of markers using instance‑based rendering.
// ============================================================================

class Marker3D : public VisualInstance3D {
public:
    Marker3D();
    ~Marker3D();

    // ------------------------------------------------------------------------
    // Gizmo appearance
    // ------------------------------------------------------------------------
    void set_gizmo_type(int type);           // 0 = sphere, 1 = cube, 2 = axes, 3 = custom mesh
    int get_gizmo_type() const;
    void set_gizmo_size(double size);
    double get_gizmo_size() const;
    void set_gizmo_color(const float* rgb, float alpha = 1.0f);
    void get_gizmo_color(float* out_rgb, float& out_alpha) const;

    // ------------------------------------------------------------------------
    // Custom mesh (if gizmo_type == 3)
    // ------------------------------------------------------------------------
    void set_custom_mesh(int64_t mesh_rid);
    int64_t get_custom_mesh() const;

    // ------------------------------------------------------------------------
    // Text label attached to marker
    // ------------------------------------------------------------------------
    void set_label(const char* text);
    const char* get_label() const;
    void set_label_color(const float* rgba);
    void get_label_color(float* out_rgba) const;
    void set_label_offset(const double* offset);
    void get_label_offset(double* out_offset) const;

    // ------------------------------------------------------------------------
    // Visibility and fade
    // ------------------------------------------------------------------------
    void set_draw_distance(float min_dist, float max_dist, float fade_margin = 0.0f);
    void get_draw_distance(float& min_dist, float& max_dist, float& fade_margin) const;
    void set_always_visible(bool always);
    bool is_always_visible() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (markers can be emissive or cast shadows for debugging)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting