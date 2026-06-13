// visibility_notifier_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <functional>

namespace lighting {

// ============================================================================
// VisibilityNotifier3D – detects when the node (or its AABB) becomes visible
// or invisible to any camera. Useful for LOD, enabling/disabling expensive
// systems, or triggering events. Does not directly affect lighting, but can
// be used to toggle lights or GI contributions.
// ============================================================================

class VisibilityNotifier3D : public Node3D {
public:
    VisibilityNotifier3D();
    ~VisibilityNotifier3D();

    // ------------------------------------------------------------------------
    // Bounding volume (used for visibility checks)
    // ------------------------------------------------------------------------
    void set_aabb(const double* min, const double* max);
    void get_aabb(double* out_min, double* out_max) const;
    void set_auto_assign_aabb(bool auto_assign); // if true, compute from children
    bool get_auto_assign_aabb() const;

    // ------------------------------------------------------------------------
    // Visibility state (read‑only)
    // ------------------------------------------------------------------------
    bool is_visible() const;           // currently visible to any camera
    bool is_on_screen() const;         // alias for is_visible

    // ------------------------------------------------------------------------
    // Callbacks
    // ------------------------------------------------------------------------
    void set_enter_callback(std::function<void()> callback);
    void set_exit_callback(std::function<void()> callback);

    // ------------------------------------------------------------------------
    // Debug visualization (draw bounding box)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;

    // ------------------------------------------------------------------------
    // Force update (recheck visibility, called each frame)
    // ------------------------------------------------------------------------
    void update_visibility();

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