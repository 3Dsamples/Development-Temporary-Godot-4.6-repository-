// Name : lighting enhancement
// File : scene/3d/visible_on_screen_notifier_3d_ext.h 29 of 60
// Description : Extended visibility notifier node with AABB detection,
//               camera frustum culling math, enter/exit signals, and RenderingServer sync.
#pragma once

#include "scene/3d/visible_on_screen_notifier_3d.h"
#include "servers/rendering_server.h"

class VisibleOnScreenNotifier3DExt : public VisibleOnScreenNotifier3D {
    GDCLASS(VisibleOnScreenNotifier3DExt, VisibleOnScreenNotifier3D);

public:
    VisibleOnScreenNotifier3DExt();
    ~VisibleOnScreenNotifier3DExt();

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
    bool is_on_screen_notifier() const;  // alias

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
    // Rendering server sync (update AABB for culling)
    // ------------------------------------------------------------------------
    void sync_notifier();

private:
    struct Impl;
    Impl *pimpl;
};