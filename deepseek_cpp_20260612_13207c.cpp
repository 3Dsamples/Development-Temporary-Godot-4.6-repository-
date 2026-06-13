// Name : lighting enhancement
// File : scene/3d/lightmap_gi_ext.cpp 22 of 60
// Description : Implementation of LightmapGIExt with baking parameters,
//               lightmap atlas management, async baking thread, and RenderingServer sync.
#include "lightmap_gi_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include "core/templates/vector.h"
#include "core/object/ref_counted.h"
#include <atomic>
#include <cmath>

struct LightmapGIExt::Impl {
    RID lightmap_rid;
    RID atlas_texture_rid;
    int quality = 1;                 // 0:low,1:medium,2:high,3:ultra
    int bounce_count = 2;
    float texel_per_unit = 64.0f;
    bool bake_shadows = true;
    bool bake_emissive = true;
    int atlas_resolution = 1024;
    bool dynamic_update = false;
    float update_frequency = 2.0f;   // Hz
    bool baking = false;
    float bake_progress = 0.0f;
    Thread *bake_thread = nullptr;
    Mutex bake_mutex;
    bool pending_update = false;

    Impl() {
        RenderingServer *rs = RenderingServer::get_singleton();
        lightmap_rid = rs->lightmap_create();
        atlas_texture_rid = rs->texture_2d_create();
        rs->lightmap_set_atlas_texture(lightmap_rid, atlas_texture_rid);
        rs->lightmap_set_bounce_count(lightmap_rid, bounce_count);
        rs->lightmap_set_texel_per_unit(lightmap_rid, texel_per_unit);
        rs->lightmap_set_bake_shadows(lightmap_rid, bake_shadows);
        rs->lightmap_set_bake_emissive(lightmap_rid, bake_emissive);
    }

    ~Impl() {
        if (bake_thread && bake_thread->is_active()) {
            cancel_bake();
        }
        RenderingServer *rs = RenderingServer::get_singleton();
        if (lightmap_rid.is_valid()) rs->free(lightmap_rid);
        if (atlas_texture_rid.is_valid()) rs->free(atlas_texture_rid);
        delete bake_thread;
    }

    void sync() {
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->lightmap_set_bounce_count(lightmap_rid, bounce_count);
        rs->lightmap_set_texel_per_unit(lightmap_rid, texel_per_unit);
        rs->lightmap_set_bake_shadows(lightmap_rid, bake_shadows);
        rs->lightmap_set_bake_emissive(lightmap_rid, bake_emissive);
        rs->lightmap_set_atlas_resolution(lightmap_rid, atlas_resolution);
        rs->lightmap_set_dynamic_update(lightmap_rid, dynamic_update);
        rs->lightmap_set_update_frequency(lightmap_rid, update_frequency);
    }

    void start_bake() {
        if (baking) return;
        baking = true;
        bake_progress = 0.0f;
        // In real implementation, this would perform ray tracing and lightmap atlas generation.
        // For math: we simulate baking by dividing work into 100 steps.
        bake_thread = new Thread;
        bake_thread->start([this]() {
            for (int i = 1; i <= 100; ++i) {
                if (!baking) break;
                // Simulate ray tracing per texel: assume total texels = atlas_resolution^2 * bounce_count
                // Math: progress = i / 100
                bake_progress = i / 100.0f;
                OS::get_singleton()->delay_usec(10000); // simulate work
            }
            MutexLock lock(bake_mutex);
            baking = false;
            bake_progress = 1.0f;
            pending_update = true;
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
        bake_progress = 0.0f;
    }
};

LightmapGIExt::LightmapGIExt() {
    pimpl = new Impl();
}

LightmapGIExt::~LightmapGIExt() {
    delete pimpl;
}

void LightmapGIExt::set_quality(int p_quality) {
    pimpl->quality = p_quality;
    // Map quality to texel per unit: low=32, medium=64, high=128, ultra=256
    float texels = 32.0f;
    switch (p_quality) {
        case 0: texels = 32.0f; break;
        case 1: texels = 64.0f; break;
        case 2: texels = 128.0f; break;
        case 3: texels = 256.0f; break;
        default: texels = 64.0f;
    }
    set_texel_per_unit(texels);
}
int LightmapGIExt::get_quality() const { return pimpl->quality; }

void LightmapGIExt::set_bounce_count(int p_bounces) {
    pimpl->bounce_count = p_bounces;
    pimpl->sync();
}
int LightmapGIExt::get_bounce_count() const { return pimpl->bounce_count; }

void LightmapGIExt::set_texel_per_unit(float p_texels) {
    pimpl->texel_per_unit = p_texels;
    pimpl->sync();
}
float LightmapGIExt::get_texel_per_unit() const { return pimpl->texel_per_unit; }

void LightmapGIExt::set_bake_shadows(bool p_enabled) {
    pimpl->bake_shadows = p_enabled;
    pimpl->sync();
}
bool LightmapGIExt::get_bake_shadows() const { return pimpl->bake_shadows; }

void LightmapGIExt::set_bake_emissive(bool p_enabled) {
    pimpl->bake_emissive = p_enabled;
    pimpl->sync();
}
bool LightmapGIExt::get_bake_emissive() const { return pimpl->bake_emissive; }

RID LightmapGIExt::get_lightmap_atlas_texture() const {
    return pimpl->atlas_texture_rid;
}

int LightmapGIExt::get_lightmap_atlas_size() const {
    return pimpl->atlas_resolution;
}

void LightmapGIExt::set_atlas_resolution(int p_resolution) {
    pimpl->atlas_resolution = p_resolution;
    pimpl->sync();
}
int LightmapGIExt::get_atlas_resolution() const { return pimpl->atlas_resolution; }

void LightmapGIExt::set_dynamic_update(bool p_enabled) {
    pimpl->dynamic_update = p_enabled;
    pimpl->sync();
}
bool LightmapGIExt::is_dynamic_update() const { return pimpl->dynamic_update; }

void LightmapGIExt::set_update_frequency(float p_fps) {
    pimpl->update_frequency = p_fps;
    pimpl->sync();
}
float LightmapGIExt::get_update_frequency() const { return pimpl->update_frequency; }

void LightmapGIExt::request_update() {
    if (pimpl->dynamic_update && !pimpl->baking) {
        bake();
    }
}

void LightmapGIExt::bake() {
    if (pimpl->baking) return;
    pimpl->start_bake();
}

bool LightmapGIExt::is_baking() const {
    return pimpl->baking;
}

float LightmapGIExt::get_bake_progress() const {
    return pimpl->bake_progress;
}

void LightmapGIExt::cancel_bake() {
    pimpl->cancel_bake();
}

void LightmapGIExt::sync_lightmap() {
    pimpl->sync();
}