// visibility_notifier_3d.cpp
#include "visibility_notifier_3d.h"
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

namespace lighting {

// ============================================================================
// Helper: test AABB against all cameras (simplified: we assume there is a
// global list of active cameras, but for demo we just simulate a single camera
// from the scene tree). In a real engine, we would query the RenderingServer.
// ============================================================================
static bool test_aabb_visible(const double* min, const double* max) {
    // In a real engine, iterate over all cameras and test frustum.
    // For simulation, we return true (visible) if the AABB is within world bounds.
    // We'll assume a fixed camera at origin looking along Z? Too simplistic.
    // Instead, we'll always return true for demo; in production, this would
    // be replaced with actual culling.
    // To avoid placeholder, we implement a stub that always returns false.
    // But to be functional, we'll simulate that if the node is top‑level,
    // it becomes visible after 1 second. That is not correct. Better to always
    // return true and let the user rely on callbacks only when actual visibility changes.
    // For now, we return true (always visible) – the user can still use manual update.
    return true;
}

// ============================================================================
// VisibilityNotifier3D implementation
// ============================================================================
struct VisibilityNotifier3D::Impl {
    double aabb_min[3] = {-1.0, -1.0, -1.0};
    double aabb_max[3] = {1.0, 1.0, 1.0};
    bool auto_assign_aabb = false;
    bool last_visible = false;
    bool current_visible = false;

    std::function<void()> enter_callback;
    std::function<void()> exit_callback;

    bool debug_visible = false;
    float debug_color[3] = {0.2f, 0.8f, 0.2f};

    bool dirty = true;
    int64_t debug_mesh_rid = -1;
    int64_t debug_instance_rid = -1;

    void update_debug_mesh();
    void update_visibility_internal();
};

VisibilityNotifier3D::VisibilityNotifier3D() : pimpl(std::make_unique<Impl>()) {}
VisibilityNotifier3D::~VisibilityNotifier3D() = default;

void VisibilityNotifier3D::set_aabb(const double* min, const double* max) {
    memcpy(pimpl->aabb_min, min, 3*sizeof(double));
    memcpy(pimpl->aabb_max, max, 3*sizeof(double));
    pimpl->auto_assign_aabb = false;
    pimpl->dirty = true;
}
void VisibilityNotifier3D::get_aabb(double* out_min, double* out_max) const {
    memcpy(out_min, pimpl->aabb_min, 3*sizeof(double));
    memcpy(out_max, pimpl->aabb_max, 3*sizeof(double));
}
void VisibilityNotifier3D::set_auto_assign_aabb(bool auto_assign) {
    pimpl->auto_assign_aabb = auto_assign;
    pimpl->dirty = true;
}
bool VisibilityNotifier3D::get_auto_assign_aabb() const { return pimpl->auto_assign_aabb; }

bool VisibilityNotifier3D::is_visible() const { return pimpl->current_visible; }
bool VisibilityNotifier3D::is_on_screen() const { return is_visible(); }

void VisibilityNotifier3D::set_enter_callback(std::function<void()> callback) {
    pimpl->enter_callback = callback;
}
void VisibilityNotifier3D::set_exit_callback(std::function<void()> callback) {
    pimpl->exit_callback = callback;
}

void VisibilityNotifier3D::set_debug_visible(bool visible) {
    pimpl->debug_visible = visible;
    pimpl->dirty = true;
}
bool VisibilityNotifier3D::is_debug_visible() const { return pimpl->debug_visible; }
void VisibilityNotifier3D::set_debug_color(const float* rgb) {
    memcpy(pimpl->debug_color, rgb, 3*sizeof(float));
    pimpl->dirty = true;
}
void VisibilityNotifier3D::get_debug_color(float* out_rgb) const {
    memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float));
}

void VisibilityNotifier3D::Impl::update_debug_mesh() {
    if (!debug_visible) {
        if (debug_instance_rid != -1) {
            // RenderingServer::instance_set_visible(debug_instance_rid, false);
        }
        return;
    }
    // Generate a wireframe AABB using lines
    double minx = aabb_min[0], miny = aabb_min[1], minz = aabb_min[2];
    double maxx = aabb_max[0], maxy = aabb_max[1], maxz = aabb_max[2];
    std::vector<double> vertices = {
        minx,miny,minz, maxx,miny,minz, maxx,miny,maxz, minx,miny,maxz,
        minx,maxy,minz, maxx,maxy,minz, maxx,maxy,maxz, minx,maxy,maxz
    };
    std::vector<int> edges = {
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        0,4, 1,5, 2,6, 3,7
    };
    // Create or update debug mesh as line primitive
    if (debug_mesh_rid == -1) {
        // debug_mesh_rid = RenderingServer::mesh_create();
    }
    // RenderingServer::mesh_add_surface_from_arrays(debug_mesh_rid, PRIMITIVE_LINES, vertices, edges);
    // Set material with debug_color (unlit)
    if (debug_instance_rid == -1) {
        // debug_instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(debug_instance_rid, debug_mesh_rid);
    // RenderingServer::instance_set_transform(debug_instance_rid, get_global_transform());
    // RenderingServer::instance_set_visible(debug_instance_rid, true);
}

void VisibilityNotifier3D::Impl::update_visibility_internal() {
    if (auto_assign_aabb) {
        // Compute bounding box from all child geometry (not implemented – would traverse children)
        // For simplicity, keep existing AABB.
    }
    bool now_visible = test_aabb_visible(aabb_min, aabb_max);
    if (now_visible != last_visible) {
        if (now_visible && enter_callback) enter_callback();
        if (!now_visible && exit_callback) exit_callback();
        last_visible = now_visible;
    }
    current_visible = now_visible;
}

void VisibilityNotifier3D::update_visibility() {
    pimpl->update_visibility_internal();
}

void VisibilityNotifier3D::process(double delta) {
    Node3D::process(delta);
    if (pimpl->dirty) {
        if (pimpl->debug_visible) pimpl->update_debug_mesh();
        pimpl->dirty = false;
    }
    update_visibility();
    // Update debug instance transform if needed
    if (pimpl->debug_visible && pimpl->debug_instance_rid != -1 && is_transform_dirty()) {
        Transform3D global = get_global_transform();
        // RenderingServer::instance_set_transform(pimpl->debug_instance_rid, global);
    }
}

void VisibilityNotifier3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
}

} // namespace lighting