// marker_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// Marker3D – simple spatial marker for debugging, editor helpers, or as
// transform handle. Does not cast shadows nor affect GI, but can optionally
// draw a debug shape (cross or axis) when visible.
// ============================================================================

class Marker3D : public Node3D {
public:
    Marker3D();
    ~Marker3D();

    // ------------------------------------------------------------------------
    // Debug visualization
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_size(double size);      // size of cross / axis lines
    double get_debug_size() const;
    void set_debug_color(float r, float g, float b, float a = 1.0f);
    void get_debug_color(float* out_rgba) const;

    // ------------------------------------------------------------------------
    // Lighting & GI (markers are not light‑interactive, but can be emissive
    // for debugging / visualization)
    // ------------------------------------------------------------------------
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;
    void set_cast_shadow(bool cast) override;   // no effect, but for compatibility
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void process(double delta) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting