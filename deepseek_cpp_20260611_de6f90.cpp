// Name : lighting enhancement
// File : scene/3d/light_3d_ext.cpp 4 of 60
// Description : Implementation of Light3DExt with real‑time shadow mapping,
//               dynamic resolution, GI flags, and reflection probe intensity.
#include "light_3d_ext.h"
#include "servers/rendering_server.h"

struct Light3DExt::Impl {
    RID light_instance_rid;
    Color color = Color(1, 1, 1);
    float energy = 1.0f;
    float range = 10.0f;
    float attenuation = 1.0f;
    bool shadow_enabled = false;
    float shadow_bias = 0.1f;
    float shadow_normal_bias = 0.1f;
    int shadow_map_resolution = 1024;
    int gi_mode = 2; // dynamic by default
    float gi_contribution = 1.0f;
    float reflection_probe_intensity = 1.0f;
    uint32_t cull_mask = 0xFFFFFFFF;
    bool params_dirty = true;
    bool shadow_dirty = true;

    Impl() {
        light_instance_rid = RenderingServer::get_singleton()->light_create();
    }

    ~Impl() {
        if (light_instance_rid.is_valid()) {
            RenderingServer::get_singleton()->free(light_instance_rid);
        }
    }
};

Light3DExt::Light3DExt() {
    pimpl = new Impl();
}

Light3DExt::~Light3DExt() {
    delete pimpl;
}

void Light3DExt::set_color(const Color &p_color) {
    pimpl->color = p_color;
    pimpl->params_dirty = true;
    sync_light_params();
}

void Light3DExt::set_param(int p_param, float p_value) {
    switch (p_param) {
        case 0: // PARAM_ENERGY
            pimpl->energy = p_value;
            break;
        case 1: // PARAM_RANGE
            pimpl->range = p_value;
            break;
        case 2: // PARAM_ATTENUATION
            pimpl->attenuation = p_value;
            break;
        default:
            return;
    }
    pimpl->params_dirty = true;
    sync_light_params();
}

void Light3DExt::set_shadow_enabled(bool p_enabled) {
    pimpl->shadow_enabled = p_enabled;
    pimpl->shadow_dirty = true;
    sync_shadow_params();
}

void Light3DExt::set_shadow_bias(float p_bias) {
    pimpl->shadow_bias = p_bias;
    pimpl->shadow_dirty = true;
    sync_shadow_params();
}

void Light3DExt::set_shadow_normal_bias(float p_normal_bias) {
    pimpl->shadow_normal_bias = p_normal_bias;
    pimpl->shadow_dirty = true;
    sync_shadow_params();
}

void Light3DExt::set_shadow_map_resolution(int p_resolution) {
    pimpl->shadow_map_resolution = p_resolution;
    pimpl->shadow_dirty = true;
    sync_shadow_params();
}

int Light3DExt::get_shadow_map_resolution() const {
    return pimpl->shadow_map_resolution;
}

void Light3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->params_dirty = true;
    sync_light_params();
}

void Light3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->params_dirty = true;
    sync_light_params();
}

void Light3DExt::set_reflection_probe_intensity(float p_intensity) {
    pimpl->reflection_probe_intensity = p_intensity;
    pimpl->params_dirty = true;
    sync_light_params();
}

float Light3DExt::get_reflection_probe_intensity() const {
    return pimpl->reflection_probe_intensity;
}

void Light3DExt::sync_light_params() {
    if (!pimpl->params_dirty) return;
    RenderingServer *rs = RenderingServer::get_singleton();
    rs->light_set_color(pimpl->light_instance_rid, pimpl->color);
    rs->light_set_param(pimpl->light_instance_rid, RenderingServer::LIGHT_PARAM_ENERGY, pimpl->energy);
    rs->light_set_param(pimpl->light_instance_rid, RenderingServer::LIGHT_PARAM_RANGE, pimpl->range);
    rs->light_set_param(pimpl->light_instance_rid, RenderingServer::LIGHT_PARAM_ATTENUATION, pimpl->attenuation);
    rs->light_set_gi_mode(pimpl->light_instance_rid, pimpl->gi_mode);
    rs->light_set_gi_contribution(pimpl->light_instance_rid, pimpl->gi_contribution);
    rs->light_set_reflection_probe_intensity(pimpl->light_instance_rid, pimpl->reflection_probe_intensity);
    pimpl->params_dirty = false;
}

void Light3DExt::sync_shadow_params() {
    if (!pimpl->shadow_dirty) return;
    RenderingServer *rs = RenderingServer::get_singleton();
    rs->light_set_shadow(pimpl->light_instance_rid, pimpl->shadow_enabled);
    rs->light_set_shadow_bias(pimpl->light_instance_rid, pimpl->shadow_bias);
    rs->light_set_shadow_normal_bias(pimpl->light_instance_rid, pimpl->shadow_normal_bias);
    rs->light_set_shadow_map_resolution(pimpl->light_instance_rid, pimpl->shadow_map_resolution);
    pimpl->shadow_dirty = false;
}

void Light3DExt::set_cull_mask(uint32_t p_mask) {
    pimpl->cull_mask = p_mask;
    RenderingServer::get_singleton()->light_set_cull_mask(pimpl->light_instance_rid, p_mask);
}

uint32_t Light3DExt::get_cull_mask() const {
    return pimpl->cull_mask;
}