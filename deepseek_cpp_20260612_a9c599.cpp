// Name : lighting enhancement
// File : scene/3d/visibility_notifier_3d_ext.h 60 of 60
// Description : Extended visibility notifier with AABB detection, camera frustum culling,
//               enter/exit callbacks, and debug visualization with emissive.
#pragma once

#include "scene/3d/visibility_notifier_3d.h"
#include "servers/rendering_server.h"

class VisibilityNotifier3DExt : public VisibilityNotifier3D {
    GDCLASS(VisibilityNotifier3DExt, VisibilityNotifier3D);

public:
    VisibilityNotifier3DExt();
    ~VisibilityNotifier3DExt();

    // ------------------------------------------------------------------------
    // Bounding volume (AABB) in local space
    // ------------------------------------------------------------------------
    void set_aabb(const AABB &p_aabb);
    AABB get_aabb() const;
    void set_auto_assign_aabb(bool p_auto);
    bool get_auto_assign_aabb() const;

    // ------------------------------------------------------------------------
    // Visibility state (read‑only)
    // ------------------------------------------------------------------------
    bool is_on_screen() const;

    // ------------------------------------------------------------------------
    // Callbacks (will be triggered via scene tree)
    // ------------------------------------------------------------------------
    void set_enter_callback(const Callable &p_callback);
    void set_exit_callback(const Callable &p_callback);

    // ------------------------------------------------------------------------
    // Force update (re‑evaluate visibility)
    // ------------------------------------------------------------------------
    void update_visibility();

    // ------------------------------------------------------------------------
    // Debug visualization (draw bounding box, can be emissive)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;
    void set_debug_emissive(const Color &p_color, float p_intensity);
    void get_debug_emissive(Color &r_color, float &r_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination (debug box can contribute to GI)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_notifier();

private:
    struct Impl;
    Impl *pimpl;
};