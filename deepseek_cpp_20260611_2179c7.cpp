// position_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// Position3D – simple marker node representing a position in space.
// Used as a target, reference point, or for scene organization.
// Does not render anything by default, but can optionally draw a debug icon.
// No shadows, no GI influence.
// ============================================================================

class Position3D : public Node3D {
public:
    Position3D();
    ~Position3D();

    // ------------------------------------------------------------------------
    // Debug visualization (icon)
    // ------------------------------------------------------------------------
    void set_debug_icon_visible(bool visible);
    bool is_debug_icon_visible() const;
    void set_debug_icon_color(const float* rgb);
    void get_debug_icon_color(float* out_rgb) const;
    void set_debug_icon_size(double size);
    double get_debug_icon_size() const;

    // ------------------------------------------------------------------------
    // Metadata (for custom editor tools)
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