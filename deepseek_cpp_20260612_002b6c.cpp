// Name : lighting enhancement
// File : scene/3d/visibility_notifier_3d_ext.cpp 60 of 60
// Description : Implementation of VisibilityNotifier3DExt with AABB to world space,
//               frustum culling using RenderingServer camera planes,
//               enter/exit callbacks, and debug wireframe box with emissive lighting.
#include "visibility_notifier_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/aabb.h"
#include "core/math/transform_3d.h"
#include "core/math/plane.h"
#include "core/object/callable.h"
#include <vector>

struct VisibilityNotifier3DExt::Impl {
    AABB local_aabb;
    bool auto_assign_aabb = false;
    bool last_visible = false;
    bool current_visible = false;
    Callable enter_callback;
    Callable exit_callback;

    // Debug visualization
    RID debug_mesh_rid;
    RID debug_instance_rid;
    bool debug_visible = false;
    Color debug_color = Color(0, 1, 0);   // green
    Color debug_emissive_color = Color(0,0,0);
    float debug_emissive_intensity = 0.0f;

    int gi_mode = 0;
    float gi_contribution = 1.0f;

    bool aabb_dirty = true;
    Transform3D cached_global;
    AABB cached_world_aabb;

    Impl() {
        debug_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        debug_instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(debug_instance_rid, debug_mesh_rid);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
    }

    ~Impl() {
        if (debug_mesh_rid.is_valid()) RenderingServer::get_singleton()->free(debug_mesh_rid);
        if (debug_instance_rid.is_valid()) RenderingServer::get_singleton()->free(debug_instance_rid);
    }

    void update_world_aabb() {
        if (!aabb_dirty && cached_global == get_global_transform()) return;
        cached_global = get_global_transform();
        cached_world_aabb = local_aabb.transformed(cached_global);
        aabb_dirty = false;
    }

    bool check_visibility() {
        update_world_aabb();
        // Get main camera's frustum planes from RenderingServer
        // In a real engine, we would iterate over all viewports, but for simplicity we use the active camera.
        Vector<Plane> frustum_planes;
        RenderingServer::get_singleton()->camera_get_frustum_planes(RID(), frustum_planes);
        if (frustum_planes.size() == 0) {
            // No camera, assume visible
            return true;
        }
        // Test AABB against each plane
        for (const Plane &p : frustum_planes) {
            if (!cached_world_aabb.intersects_plane(p)) {
                return false;
            }
        }
        return true;
    }

    void update_debug_mesh() {
        if (!debug_visible) {
            RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
            return;
        }
        update_world_aabb();
        Vector3 min = cached_world_aabb.position;
        Vector3 max = cached_world_aabb.position + cached_world_aabb.size;
        // 8 corners of the box
        Vector3 corners[8] = {
            Vector3(min.x, min.y, min.z),
            Vector3(max.x, min.y, min.z),
            Vector3(max.x, min.y, max.z),
            Vector3(min.x, min.y, max.z),
            Vector3(min.x, max.y, min.z),
            Vector3(max.x, max.y, min.z),
            Vector3(max.x, max.y, max.z),
            Vector3(min.x, max.y, max.z)
        };
        // 12 edges
        int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        Vector<Vector3> vertices;
        Vector<int> indices;
        for (int i = 0; i < 12; ++i) {
            vertices.push_back(corners[edges[i][0]]);
            vertices.push_back(corners[edges[i][1]]);
            indices.push_back(i*2);
            indices.push_back(i*2+1);
        }
        RenderingServer::get_singleton()->mesh_clear(debug_mesh_rid);
        if (vertices.size() == 0) return;
        RenderingServer::get_singleton()->mesh_add_surface(debug_mesh_rid, RS::PRIMITIVE_LINES, vertices, indices, Vector<Vector2>(), Vector<Vector3>());
        // Create material
        RID mat = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(mat, "albedo", debug_color);
        if (debug_emissive_intensity > 0.0f) {
            RenderingServer::get_singleton()->material_set_param(mat, "emission", debug_emissive_color);
            RenderingServer::get_singleton()->material_set_param(mat, "emission_intensity", debug_emissive_intensity);
        }
        RenderingServer::get_singleton()->mesh_surface_set_material(debug_mesh_rid, 0, mat);
        // The mesh vertices are already in world space, so instance transform is identity.
        RenderingServer::get_singleton()->instance_set_transform(debug_instance_rid, Transform3D());
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, true);
    }
};

VisibilityNotifier3DExt::VisibilityNotifier3DExt() {
    pimpl = new Impl();
    pimpl->local_aabb = AABB(Vector3(-1,-1,-1), Vector3(2,2,2));
}

VisibilityNotifier3DExt::~VisibilityNotifier3DExt() {
    delete pimpl;
}

void VisibilityNotifier3DExt::set_aabb(const AABB &p_aabb) {
    pimpl->local_aabb = p_aabb;
    pimpl->aabb_dirty = true;
    sync_notifier();
}
AABB VisibilityNotifier3DExt::get_aabb() const { return pimpl->local_aabb; }

void VisibilityNotifier3DExt::set_auto_assign_aabb(bool p_auto) {
    pimpl->auto_assign_aabb = p_auto;
    if (p_auto) {
        // In real engine, compute AABB from all child visual instances.
        // For now, we keep current AABB.
    }
}
bool VisibilityNotifier3DExt::get_auto_assign_aabb() const { return pimpl->auto_assign_aabb; }

bool VisibilityNotifier3DExt::is_on_screen() const {
    return pimpl->current_visible;
}

void VisibilityNotifier3DExt::set_enter_callback(const Callable &p_callback) {
    pimpl->enter_callback = p_callback;
}
void VisibilityNotifier3DExt::set_exit_callback(const Callable &p_callback) {
    pimpl->exit_callback = p_callback;
}

void VisibilityNotifier3DExt::update_visibility() {
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

void VisibilityNotifier3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    pimpl->update_debug_mesh();
}
bool VisibilityNotifier3DExt::is_debug_visible() const { return pimpl->debug_visible; }

void VisibilityNotifier3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    pimpl->update_debug_mesh();
}
Color VisibilityNotifier3DExt::get_debug_color() const { return pimpl->debug_color; }

void VisibilityNotifier3DExt::set_debug_emissive(const Color &p_color, float p_intensity) {
    pimpl->debug_emissive_color = p_color;
    pimpl->debug_emissive_intensity = p_intensity;
    pimpl->update_debug_mesh();
}
void VisibilityNotifier3DExt::get_debug_emissive(Color &r_color, float &r_intensity) const {
    r_color = pimpl->debug_emissive_color;
    r_intensity = pimpl->debug_emissive_intensity;
}

void VisibilityNotifier3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
}
int VisibilityNotifier3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void VisibilityNotifier3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}
float VisibilityNotifier3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void VisibilityNotifier3DExt::sync_notifier() {
    pimpl->aabb_dirty = true;
    pimpl->update_world_aabb();
    if (pimpl->debug_visible) pimpl->update_debug_mesh();
}