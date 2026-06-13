// marker_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// Marker3D – simple 3D marker node for scene organization, debugging,
// or as a placeholder for other systems (e.g., spawn points, probe positions).
// Does not cast shadows, does not affect GI, but can be used as a reference
// for placing lights, cameras, or reflection probes. Optionally renders a
// small wireframe shape in the editor or when debug visualization is enabled.
// ============================================================================

class Marker3D : public Node3D {
public:
    Marker3D();
    ~Marker3D();

    // ------------------------------------------------------------------------
    // Visual representation (editor / debug)
    // ------------------------------------------------------------------------
    void set_marker_shape(int shape);   // 0 = sphere, 1 = cube, 2 = cross, 3 = arrow
    int get_marker_shape() const;
    void set_marker_size(double size);
    double get_marker_size() const;
    void set_marker_color(const float* rgb);
    void get_marker_color(float* out_rgb) const;
    void set_debug_visible(bool visible);   // visibility in game view (not recommended)
    bool is_debug_visible() const;

    // ------------------------------------------------------------------------
    // Usage hint (for external tools)
    // ------------------------------------------------------------------------
    void set_metadata(const char* key, const char* value);
    const char* get_metadata(const char* key) const;
    void clear_metadata();

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