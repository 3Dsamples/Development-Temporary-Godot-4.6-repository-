// Name : lighting enhancement
// File : scene/3d/occluder_instance_3d_ext.cpp 28 of 60
// Description : Implementation of OccluderInstance3DExt with shape (box/sphere/mesh),
//               size, cull mask, debug visualization, and full RenderingServer sync.
#include "occluder_instance_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"

struct OccluderInstance3DExt::Impl {
    RID occluder_rid;
    int shape = 0;               // 0 = box, 1 = sphere, 2 = mesh
    Vector3 size = Vector3(1,1,1);   // half extents for box, radius for sphere (x component)
    RID mesh_rid;
    bool enabled = true;
    uint32_t cull_mask = 0xFFFFFFFF;
    bool debug_visible = false;
    Color debug_color = Color(1,0,0);

    bool dirty = true;

    Impl() {
        occluder_rid = RenderingServer::get_singleton()->occluder_create();
    }

    ~Impl() {
        if (occluder_rid.is_valid()) {
            RenderingServer::get_singleton()->free(occluder_rid);
        }
    }

    void sync() {
        if (!dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->occluder_set_enabled(occluder_rid, enabled);
        rs->occluder_set_cull_mask(occluder_rid, cull_mask);
        if (shape == 0) { // box
            rs->occluder_set_box(occluder_rid, size);
        } else if (shape == 1) { // sphere
            rs->occluder_set_sphere(occluder_rid, size.x);
        } else if (shape == 2) { // mesh
            rs->occluder_set_mesh(occluder_rid, mesh_rid);
        }
        rs->occluder_set_transform(occluder_rid, get_global_transform());
        // Debug visualization (wireframe) – for the editor only
        // In real runtime, we would not draw occluders unless debug_visible.
        dirty = false;
    }
};

OccluderInstance3DExt::OccluderInstance3DExt() {
    pimpl = new Impl();
}

OccluderInstance3DExt::~OccluderInstance3DExt() {
    delete pimpl;
}

void OccluderInstance3DExt::set_shape(int p_shape) {
    pimpl->shape = p_shape;
    pimpl->dirty = true;
    sync_occluder();
}
int OccluderInstance3DExt::get_shape() const { return pimpl->shape; }

void OccluderInstance3DExt::set_size(const Vector3 &p_size) {
    pimpl->size = p_size;
    pimpl->dirty = true;
    sync_occluder();
}
Vector3 OccluderInstance3DExt::get_size() const { return pimpl->size; }

void OccluderInstance3DExt::set_mesh(const RID &p_mesh) {
    pimpl->mesh_rid = p_mesh;
    pimpl->dirty = true;
    sync_occluder();
}
RID OccluderInstance3DExt::get_mesh() const { return pimpl->mesh_rid; }

void OccluderInstance3DExt::set_enabled(bool p_enabled) {
    pimpl->enabled = p_enabled;
    pimpl->dirty = true;
    sync_occluder();
}
bool OccluderInstance3DExt::is_enabled() const { return pimpl->enabled; }

void OccluderInstance3DExt::set_cull_mask(uint32_t p_mask) {
    pimpl->cull_mask = p_mask;
    pimpl->dirty = true;
    sync_occluder();
}
uint32_t OccluderInstance3DExt::get_cull_mask() const { return pimpl->cull_mask; }

void OccluderInstance3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    // In rendering server, set debug visibility (if supported)
    RenderingServer::get_singleton()->occluder_set_debug_visible(pimpl->occluder_rid, p_visible);
}
bool OccluderInstance3DExt::is_debug_visible() const { return pimpl->debug_visible; }

void OccluderInstance3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    RenderingServer::get_singleton()->occluder_set_debug_color(pimpl->occluder_rid, p_color);
}
Color OccluderInstance3DExt::get_debug_color() const { return pimpl->debug_color; }

void OccluderInstance3DExt::sync_occluder() {
    pimpl->sync();
}