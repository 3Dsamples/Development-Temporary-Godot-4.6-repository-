// Name : lighting enhancement
// File : scene/3d/light_3d.cpp file number : 4
// Description : Implementation of advanced 3D light node with shadows, GI, probes, and volumetric effects.

#include "scene/3d/light_3d.h"
#include "servers/rendering_server.h"
#include "servers/rendering/rendering_light_culler.h"
#include "scene/3d/reflection_probe.h"
#include "scene/3d/lightmap_probe.h"
#include "core/os/os.h"
#include "core/math/math_funcs.h"
#include <cmath>
#include <algorithm>

struct Light3D::Impl {
    LightType type = LIGHT_DIRECTIONAL;
    Color color = Color(1, 1, 1);
    float intensity = 1.0f;
    float range = 100.0f;
    float spot_angle = 45.0f;
    float spot_attenuation = 1.0f;
    Vector2 size = Vector2(1, 1);
    float radius = 0.5f;

    bool shadow_enabled = true;
    ShadowTechnique shadow_technique = SHADOW_PCSS;
    int shadow_map_res = 2048;
    int csm_cascade_count = 4;
    float csm_split_lambda = 0.5f;
    float pcss_light_size = 1.0f;
    float vsm_exponent = 40.0f;

    int gi_mode = 1; // static by default
    float gi_contribution = 1.0f;
    bool use_environment = false;
    Ref<Environment> environment;

    bool reflection_probe_enabled = false;
    float reflection_probe_update_rate = 2.0f;
    double reflection_probe_timer = 0.0;
    RID reflection_probe_rid;

    bool volumetric_enabled = true;
    float volumetric_fog_intensity = 1.0f;

    RID light_rid;
    RID instance_rid;

    // Light probe grid
    AABB probe_bounds;
    Vector3i probe_resolution;
    bool has_probe_grid = false;
    Vector<Ref<LightmapProbe>> light_probes;

    Impl() {
        light_rid = RenderingServer::get_singleton()->light_create();
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_ENERGY, intensity);
        RenderingServer::get_singleton()->light_set_color(light_rid, color);
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_RANGE, range);
    }

    ~Impl() {
        if (light_rid.is_valid())
            RenderingServer::get_singleton()->free(light_rid);
        if (reflection_probe_rid.is_valid())
            RenderingServer::get_singleton()->free(reflection_probe_rid);
        if (instance_rid.is_valid())
            RenderingServer::get_singleton()->free(instance_rid);
    }

    void update_light_type() {
        switch (type) {
            case LIGHT_DIRECTIONAL:
                RenderingServer::get_singleton()->light_set_type(light_rid, RenderingServer::LIGHT_DIRECTIONAL);
                break;
            case LIGHT_POINT:
                RenderingServer::get_singleton()->light_set_type(light_rid, RenderingServer::LIGHT_OMNI);
                break;
            case LIGHT_SPOT:
                RenderingServer::get_singleton()->light_set_type(light_rid, RenderingServer::LIGHT_SPOT);
                break;
            case LIGHT_RECTANGLE:
            case LIGHT_DISC:
            case LIGHT_SPHERE:
                // Area lights are handled as separate primitives; for RS, treat as omni with attenuation override.
                RenderingServer::get_singleton()->light_set_type(light_rid, RenderingServer::LIGHT_OMNI);
                break;
        }
    }

    void update_shadow_params() {
        if (!shadow_enabled) {
            RenderingServer::get_singleton()->light_set_shadow(light_rid, false);
            return;
        }
        RenderingServer::get_singleton()->light_set_shadow(light_rid, true);
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_SHADOW_PCF, shadow_technique == SHADOW_PCF ? 1.0f : 0.0f);
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_SHADOW_PCSS, shadow_technique == SHADOW_PCSS ? 1.0f : 0.0f);
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_SHADOW_VSM, shadow_technique == SHADOW_VSM ? 1.0f : 0.0f);
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_SHADOW_CHS, shadow_technique == SHADOW_CHS ? 1.0f : 0.0f);
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_SHADOW_PCSS_LIGHT_SIZE, pcss_light_size);
        RenderingServer::get_singleton()->light_set_param(light_rid, RenderingServer::LIGHT_PARAM_SHADOW_VSM_EXPONENT, vsm_exponent);
        RenderingServer::get_singleton()->light_set_directional_shadow_blend_splits(light_rid, csm_cascade_count, csm_split_lambda);
    }
};

Light3D::Light3D() {
    pimpl = new Impl;
    pimpl->update_light_type();
    pimpl->update_shadow_params();

    // Create a visual instance for the light (invisible by default, but needed for rendering)
    pimpl->instance_rid = RenderingServer::get_singleton()->instance_create();
    RenderingServer::get_singleton()->instance_set_base(pimpl->instance_rid, pimpl->light_rid);
    RenderingServer::get_singleton()->instance_set_visible(pimpl->instance_rid, true);
}

Light3D::~Light3D() {
    delete pimpl;
}

void Light3D::set_light_type(LightType p_type) {
    pimpl->type = p_type;
    pimpl->update_light_type();
    _update_render_server_transform(); // update if needed
}

Light3D::LightType Light3D::get_light_type() const {
    return pimpl->type;
}

void Light3D::set_color(const Color &p_color) {
    pimpl->color = p_color;
    RenderingServer::get_singleton()->light_set_color(pimpl->light_rid, p_color);
}

Color Light3D::get_color() const {
    return pimpl->color;
}

void Light3D::set_intensity(float p_intensity) {
    pimpl->intensity = p_intensity;
    RenderingServer::get_singleton()->light_set_param(pimpl->light_rid, RenderingServer::LIGHT_PARAM_ENERGY, p_intensity);
}

float Light3D::get_intensity() const {
    return pimpl->intensity;
}

void Light3D::set_range(float p_range) {
    pimpl->range = p_range;
    RenderingServer::get_singleton()->light_set_param(pimpl->light_rid, RenderingServer::LIGHT_PARAM_RANGE, p_range);
}

float Light3D::get_range() const {
    return pimpl->range;
}

void Light3D::set_spot_angle(float p_degrees) {
    pimpl->spot_angle = p_degrees;
    float rad = Math::deg2rad(p_degrees);
    float angle = cos(rad);
    RenderingServer::get_singleton()->light_set_param(pimpl->light_rid, RenderingServer::LIGHT_PARAM_SPOT_ANGLE, angle);
}

float Light3D::get_spot_angle() const {
    return pimpl->spot_angle;
}

void Light3D::set_spot_attenuation(float p_attenuation) {
    pimpl->spot_attenuation = p_attenuation;
    RenderingServer::get_singleton()->light_set_param(pimpl->light_rid, RenderingServer::LIGHT_PARAM_SPOT_ATTENUATION, p_attenuation);
}

float Light3D::get_spot_attenuation() const {
    return pimpl->spot_attenuation;
}

void Light3D::set_size(const Vector2 &p_size) {
    pimpl->size = p_size;
    // For area lights, update attenuation and shape in rendering server if supported.
}

Vector2 Light3D::get_size() const {
    return pimpl->size;
}

void Light3D::set_radius(float p_radius) {
    pimpl->radius = p_radius;
}

float Light3D::get_radius() const {
    return pimpl->radius;
}

void Light3D::set_shadow_enabled(bool p_enabled) {
    pimpl->shadow_enabled = p_enabled;
    pimpl->update_shadow_params();
}

bool Light3D::is_shadow_enabled() const {
    return pimpl->shadow_enabled;
}

void Light3D::set_shadow_technique(ShadowTechnique p_technique) {
    pimpl->shadow_technique = p_technique;
    pimpl->update_shadow_params();
}

Light3D::ShadowTechnique Light3D::get_shadow_technique() const {
    return pimpl->shadow_technique;
}

void Light3D::set_shadow_map_resolution(int p_res) {
    pimpl->shadow_map_res = p_res;
    RenderingServer::get_singleton()->light_set_shadow_map_resolution(pimpl->light_rid, p_res);
}

int Light3D::get_shadow_map_resolution() const {
    return pimpl->shadow_map_res;
}

void Light3D::set_csm_cascade_count(int p_count) {
    pimpl->csm_cascade_count = p_count;
    pimpl->update_shadow_params();
}

int Light3D::get_csm_cascade_count() const {
    return pimpl->csm_cascade_count;
}

void Light3D::set_csm_split_lambda(float p_lambda) {
    pimpl->csm_split_lambda = p_lambda;
    pimpl->update_shadow_params();
}

float Light3D::get_csm_split_lambda() const {
    return pimpl->csm_split_lambda;
}

void Light3D::set_pcss_light_size(float p_size) {
    pimpl->pcss_light_size = p_size;
    pimpl->update_shadow_params();
}

float Light3D::get_pcss_light_size() const {
    return pimpl->pcss_light_size;
}

void Light3D::set_vsm_exponent(float p_exp) {
    pimpl->vsm_exponent = p_exp;
    pimpl->update_shadow_params();
}

float Light3D::get_vsm_exponent() const {
    return pimpl->vsm_exponent;
}

void Light3D::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    // Update rendering server's GI contribution for this light.
    if (p_mode == 0) {
        RenderingServer::get_singleton()->light_set_gi_mode(pimpl->light_rid, RenderingServer::LIGHT_GI_MODE_DISABLED);
    } else if (p_mode == 1) {
        RenderingServer::get_singleton()->light_set_gi_mode(pimpl->light_rid, RenderingServer::LIGHT_GI_MODE_STATIC);
    } else if (p_mode == 2) {
        RenderingServer::get_singleton()->light_set_gi_mode(pimpl->light_rid, RenderingServer::LIGHT_GI_MODE_DYNAMIC);
    }
}

int Light3D::get_gi_mode() const {
    return pimpl->gi_mode;
}

void Light3D::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    RenderingServer::get_singleton()->light_set_param(pimpl->light_rid, RenderingServer::LIGHT_PARAM_GI_STRENGTH, p_amount);
}

float Light3D::get_gi_contribution() const {
    return pimpl->gi_contribution;
}

void Light3D::set_use_environment(bool p_use) {
    pimpl->use_environment = p_use;
    if (p_use && pimpl->environment.is_valid()) {
        RenderingServer::get_singleton()->light_set_environment(pimpl->light_rid, pimpl->environment->get_rid());
    } else {
        RenderingServer::get_singleton()->light_set_environment(pimpl->light_rid, RID());
    }
}

bool Light3D::get_use_environment() const {
    return pimpl->use_environment;
}

void Light3D::set_environment(Ref<Environment> p_env) {
    pimpl->environment = p_env;
    if (pimpl->use_environment) {
        RenderingServer::get_singleton()->light_set_environment(pimpl->light_rid, p_env->get_rid());
    }
}

Ref<Environment> Light3D::get_environment() const {
    return pimpl->environment;
}

void Light3D::set_reflection_probe_enabled(bool p_enabled) {
    pimpl->reflection_probe_enabled = p_enabled;
    if (p_enabled && !pimpl->reflection_probe_rid.is_valid()) {
        pimpl->reflection_probe_rid = RenderingServer::get_singleton()->reflection_probe_create();
        RenderingServer::get_singleton()->reflection_probe_set_update_mode(pimpl->reflection_probe_rid,
            pimpl->reflection_probe_update_rate > 0.0f ? RenderingServer::REFLECTION_PROBE_UPDATE_ALWAYS : RenderingServer::REFLECTION_PROBE_UPDATE_ONCE);
    } else if (!p_enabled && pimpl->reflection_probe_rid.is_valid()) {
        RenderingServer::get_singleton()->free(pimpl->reflection_probe_rid);
        pimpl->reflection_probe_rid = RID();
    }
}

bool Light3D::is_reflection_probe_enabled() const {
    return pimpl->reflection_probe_enabled;
}

void Light3D::set_reflection_probe_update_rate(float p_fps) {
    pimpl->reflection_probe_update_rate = p_fps;
    if (pimpl->reflection_probe_rid.is_valid()) {
        RenderingServer::get_singleton()->reflection_probe_set_update_mode(pimpl->reflection_probe_rid,
            p_fps > 0.0f ? RenderingServer::REFLECTION_PROBE_UPDATE_ALWAYS : RenderingServer::REFLECTION_PROBE_UPDATE_ONCE);
    }
}

float Light3D::get_reflection_probe_update_rate() const {
    return pimpl->reflection_probe_update_rate;
}

void Light3D::capture_reflection_probe() {
    if (pimpl->reflection_probe_rid.is_valid()) {
        RenderingServer::get_singleton()->reflection_probe_update(pimpl->reflection_probe_rid);
    }
}

void Light3D::set_light_probe_grid(const AABB &p_bounds, const Vector3i &p_resolution) {
    pimpl->probe_bounds = p_bounds;
    pimpl->probe_resolution = p_resolution;
    pimpl->has_probe_grid = true;
    // Create light probes (irradiance volumes)
    pimpl->light_probes.clear();
    for (int x = 0; x < p_resolution.x; ++x) {
        for (int y = 0; y < p_resolution.y; ++y) {
            for (int z = 0; z < p_resolution.z; ++z) {
                Vector3 pos = p_bounds.position + Vector3(
                    (x + 0.5f) * p_bounds.size.x / p_resolution.x,
                    (y + 0.5f) * p_bounds.size.y / p_resolution.y,
                    (z + 0.5f) * p_bounds.size.z / p_resolution.z);
                Ref<LightmapProbe> probe;
                probe.instantiate();
                probe->set_position(pos);
                probe->set_extents(Vector3(p_bounds.size.x / p_resolution.x, p_bounds.size.y / p_resolution.y, p_bounds.size.z / p_resolution.z));
                pimpl->light_probes.push_back(probe);
            }
        }
    }
}

void Light3D::clear_light_probe_grid() {
    pimpl->has_probe_grid = false;
    pimpl->light_probes.clear();
}

void Light3D::update_light_probes() {
    for (int i = 0; i < pimpl->light_probes.size(); ++i) {
        pimpl->light_probes[i]->capture();
    }
}

void Light3D::set_volumetric_enabled(bool p_enabled) {
    pimpl->volumetric_enabled = p_enabled;
    RenderingServer::get_singleton()->light_set_volumetric_fog(pimpl->light_rid, p_enabled);
}

bool Light3D::is_volumetric_enabled() const {
    return pimpl->volumetric_enabled;
}

void Light3D::set_volumetric_fog_intensity(float p_intensity) {
    pimpl->volumetric_fog_intensity = p_intensity;
    RenderingServer::get_singleton()->light_set_param(pimpl->light_rid, RenderingServer::LIGHT_PARAM_VOLUMETRIC_FOG_STRENGTH, p_intensity);
}

float Light3D::get_volumetric_fog_intensity() const {
    return pimpl->volumetric_fog_intensity;
}

void Light3D::synchronize_render_server(double p_delta) {
    Node3D::synchronize_render_server(p_delta);
    // Update light instance transform
    Transform3D global = get_global_transform();
    RenderingServer::get_singleton()->instance_set_transform(pimpl->instance_rid, global);
    // Update reflection probe transform if enabled
    if (pimpl->reflection_probe_enabled && pimpl->reflection_probe_rid.is_valid()) {
        RenderingServer::get_singleton()->reflection_probe_set_transform(pimpl->reflection_probe_rid, global);
        if (pimpl->reflection_probe_update_rate > 0.0f) {
            pimpl->reflection_probe_timer += p_delta;
            if (pimpl->reflection_probe_timer >= 1.0 / pimpl->reflection_probe_update_rate) {
                pimpl->reflection_probe_timer = 0.0;
                capture_reflection_probe();
            }
        }
    }
    // Update GI contribution if dynamic
    if (pimpl->gi_mode == 2) {
        // For dynamic GI, we may need to update the light's contribution per frame
        RenderingServer::get_singleton()->light_set_gi_mode(pimpl->light_rid, RenderingServer::LIGHT_GI_MODE_DYNAMIC);
    }
}

void Light3D::_transform_changed() {
    Node3D::_transform_changed();
    _update_render_server_transform();
}

void Light3D::_update_render_server_transform() {
    Transform3D global = get_global_transform();
    RenderingServer::get_singleton()->instance_set_transform(pimpl->instance_rid, global);
    if (pimpl->reflection_probe_rid.is_valid())
        RenderingServer::get_singleton()->reflection_probe_set_transform(pimpl->reflection_probe_rid, global);
}

RID Light3D::get_light_rid() const {
    return pimpl->light_rid;
}