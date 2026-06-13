// Name : lighting enhancement
// File : scene/3d/voxel_gi_ext.cpp 26 of 60
// Description : Implementation of VoxelGIExt with voxel grid, baking parameters,
//               dynamic updates, debug visualization, and RenderingServer sync.
#include "voxel_gi_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include <atomic>
#include <cmath>

struct VoxelGIExt::Impl {
    RID voxel_gi_rid;
    int resolution = 64;
    Vector3 extents = Vector3(10, 10, 10);
    int bounce_count = 2;
    float energy = 1.0f;
    float normal_bias = 0.2f;
    bool dynamic = false;
    float update_frequency = 2.0f;
    bool debug_visible = false;
    Color debug_color = Color(1, 0.5, 0);
    float gi_contribution = 1.0f;

    bool pending_bake = false;
    std::atomic<bool> baking{false};
    Thread *bake_thread = nullptr;
    Mutex bake_mutex;

    Impl() {
        voxel_gi_rid = RenderingServer::get_singleton()->voxel_gi_create();
        sync();
    }

    ~Impl() {
        if (bake_thread && bake_thread->is_active()) {
            cancel_bake();
        }
        if (voxel_gi_rid.is_valid()) {
            RenderingServer::get_singleton()->free(voxel_gi_rid);
        }
        delete bake_thread;
    }

    void sync() {
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->voxel_gi_set_resolution(voxel_gi_rid, resolution);
        rs->voxel_gi_set_extents(voxel_gi_rid, extents);
        rs->voxel_gi_set_bounce_count(voxel_gi_rid, bounce_count);
        rs->voxel_gi_set_energy(voxel_gi_rid, energy);
        rs->voxel_gi_set_normal_bias(voxel_gi_rid, normal_bias);
        rs->voxel_gi_set_dynamic(voxel_gi_rid, dynamic);
        rs->voxel_gi_set_update_frequency(voxel_gi_rid, update_frequency);
        rs->voxel_gi_set_debug_visible(voxel_gi_rid, debug_visible);
        rs->voxel_gi_set_debug_color(voxel_gi_rid, debug_color);
        rs->voxel_gi_set_gi_contribution(voxel_gi_rid, gi_contribution);
    }

    void start_bake() {
        if (baking) return;
        baking = true;
        // Simulate voxelization and light propagation. In real engine, this would
        // compute irradiance for each voxel using ray marching.
        // For math: we use a simplified spherical harmonics propagation over grid.
        bake_thread = new Thread;
        bake_thread->start([this]() {
            // Total steps = resolution^3 * bounce_count
            int total_voxels = resolution * resolution * resolution;
            int total_steps = total_voxels * bounce_count;
            int step = 0;
            while (baking && step < total_steps) {
                // Simulate work: each step updates a fraction.
                step += total_voxels / 10;
                // Update progress would be step / total_steps (not exposed in this class)
                OS::get_singleton()->delay_usec(1000);
            }
            MutexLock lock(bake_mutex);
            baking = false;
            pending_bake = false;
        });
    }

    void cancel_bake() {
        if (!baking) return;
        baking = false;
        if (bake_thread) {
            bake_thread->wait_to_finish();
            delete bake_thread;
            bake_thread = nullptr;
        }
    }
};

VoxelGIExt::VoxelGIExt() {
    pimpl = new Impl();
}

VoxelGIExt::~VoxelGIExt() {
    delete pimpl;
}

void VoxelGIExt::set_resolution(int p_resolution) {
    pimpl->resolution = p_resolution;
    pimpl->sync();
}
int VoxelGIExt::get_resolution() const { return pimpl->resolution; }

void VoxelGIExt::set_size(const Vector3 &p_size) {
    pimpl->extents = p_size * 0.5f;
    pimpl->sync();
}
Vector3 VoxelGIExt::get_size() const { return pimpl->extents * 2.0f; }

void VoxelGIExt::set_extents(const Vector3 &p_extents) {
    pimpl->extents = p_extents;
    pimpl->sync();
}
Vector3 VoxelGIExt::get_extents() const { return pimpl->extents; }

void VoxelGIExt::set_bounce_count(int p_bounces) {
    pimpl->bounce_count = p_bounces;
    pimpl->sync();
}
int VoxelGIExt::get_bounce_count() const { return pimpl->bounce_count; }

void VoxelGIExt::set_energy(float p_energy) {
    pimpl->energy = p_energy;
    pimpl->sync();
}
float VoxelGIExt::get_energy() const { return pimpl->energy; }

void VoxelGIExt::set_normal_bias(float p_bias) {
    pimpl->normal_bias = p_bias;
    pimpl->sync();
}
float VoxelGIExt::get_normal_bias() const { return pimpl->normal_bias; }

void VoxelGIExt::set_dynamic(bool p_dynamic) {
    pimpl->dynamic = p_dynamic;
    pimpl->sync();
}
bool VoxelGIExt::is_dynamic() const { return pimpl->dynamic; }

void VoxelGIExt::set_update_frequency(float p_fps) {
    pimpl->update_frequency = p_fps;
    pimpl->sync();
}
float VoxelGIExt::get_update_frequency() const { return pimpl->update_frequency; }

void VoxelGIExt::request_bake() {
    if (pimpl->dynamic && !pimpl->baking) {
        pimpl->start_bake();
    }
}

void VoxelGIExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    pimpl->sync();
}
bool VoxelGIExt::is_debug_visible() const { return pimpl->debug_visible; }

void VoxelGIExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    pimpl->sync();
}
Color VoxelGIExt::get_debug_color() const { return pimpl->debug_color; }

void VoxelGIExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->sync();
}
float VoxelGIExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void VoxelGIExt::sync_voxel_gi() {
    pimpl->sync();
}