// Name : lighting enhancement
// File : scene/3d/visible_on_screen_notifier_3d_ext.cpp 30 of 60
// Description : Implementation of VisibleOnScreenNotifier3DExt with AABB to world space,
//               frustum culling using RenderingServer's camera frustum planes,
//               enter/exit callbacks, and efficient update logic.
#include "visible_on_screen_notifier_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/aabb.h"
#include "core/math/transform_3d.h"
#include "core/math/math_funcs.h"
#include "core/object/callable.h"

struct VisibleOnScreenNotifier3DExt::Impl {
    AABB local_aabb;
    bool auto_assign_aabb = false;
    bool last_visible = false;
    bool current_visible = false;
    Callable enter_callback;
    Callable exit_callback;

    // For frustum caching (avoid per‑frame allocation)
    Transform3D cached_global_transform;
    AABB cached_world_aabb;

    bool aabb_dirty = true;

    Impl() {
        local_aabb = AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
    }

    void update_world_aabb() {
        Transform3D global = get_global_transform();
        if (global == cached_global_transform && !aabb_dirty) return;
        cached_global_transform = global;
        cached_world_aabb = local_aabb.transformed(global);
        aabb_dirty = false;
    }

    bool check_visibility() {
        update_world_aabb();
        // Ask RenderingServer for camera frustum planes (for the main viewport)
        // In a real engine, we would iterate over all cameras. Simplified: use main camera.
        Vector<Plane> frustum_planes;
        RenderingServer::get_singleton()->camera_get_frustum_planes(RID(), frustum_planes);
        if (frustum_planes.is_empty()) {
            // No camera, assume visible (or use node visibility)
            return true;
        }
        // Test AABB against all frustum planes
        for (const Plane &p : frustum_planes) {
            if (cached_world_aabb.intersects_plane(p) == false) {
                return false;
            }
        }
        return true;
    }
};

VisibleOnScreenNotifier3DExt::VisibleOnScreenNotifier3DExt() {
    pimpl = new Impl();
}

VisibleOnScreenNotifier3DExt::~VisibleOnScreenNotifier3DExt() {
    delete pimpl;
}

void VisibleOnScreenNotifier3DExt::set_aabb(const AABB &p_aabb) {
    pimpl->local_aabb = p_aabb;
    pimpl->aabb_dirty = true;
    sync_notifier();
}
AABB VisibleOnScreenNotifier3DExt::get_aabb() const {
    return pimpl->local_aabb;
}

void VisibleOnScreenNotifier3DExt::set_auto_assign_aabb(bool p_auto) {
    pimpl->auto_assign_aabb = p_auto;
    if (p_auto) {
        // In real engine, compute AABB from all child visual instances.
        // For now, we just keep current.
    }
}
bool VisibleOnScreenNotifier3DExt::get_auto_assign_aabb() const {
    return pimpl->auto_assign_aabb;
}

bool VisibleOnScreenNotifier3DExt::is_on_screen() const {
    return pimpl->current_visible;
}
bool VisibleOnScreenNotifier3DExt::is_on_screen_notifier() const {
    return is_on_screen();
}

void VisibleOnScreenNotifier3DExt::set_enter_callback(const Callable &p_callback) {
    pimpl->enter_callback = p_callback;
}
void VisibleOnScreenNotifier3DExt::set_exit_callback(const Callable &p_callback) {
    pimpl->exit_callback = p_callback;
}

void VisibleOnScreenNotifier3DExt::update_visibility() {
    bool now_visible = pimpl->check_visibility();
    if (now_visible != pimpl->last_visible) {
        if (now_visible && pimpl->enter_callback.is_valid()) {
            pimpl->enter_callback.call();
        } else if (!now_visible && pimpl->exit_callback.is_valid()) {
            pimpl->exit_callback.call();
        }
        pimpl->last_visible = now_visible;
    }
    pimpl->current_visible = now_visible;
}

void VisibleOnScreenNotifier3DExt::sync_notifier() {
    // No direct RenderingServer object, but we can mark dirty.
    pimpl->aabb_dirty = true;
    // Also, we could update the node's global transform cache.
}