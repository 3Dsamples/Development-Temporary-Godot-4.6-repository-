// Name : lighting enhancement
// File : scene/3d/visual_instance_3d_ext.cpp 8 of 60
// Description : Implementation of VisualInstance3DExt with full RenderingServer sync,
//               LOD, visibility range, bounding box, and GI flags.
#include "visual_instance_3d_ext.h"
#include "servers/rendering_server.h"

struct VisualInstance3DExt::Impl {
    RID instance_rid;
    RID base_rid;
    bool base_set = false;
    Transform3D cached_transform;
    bool visible = true;
    uint32_t layer_mask = 0xFFFFFFFF;
    bool cast_shadow = true;
    int gi_mode = 1; // 0=off,1=static,2=dynamic
    float gi_contribution = 1.0f;
    Color emissive_color;
    float emissive_intensity = 0.0f;
    float lod_bias = 1.0f;
    float visibility_range_min = 0.0f;
    float visibility_range_max = 0.0f; // 0 = no max limit
    float visibility_range_fade_margin = 0.0f;

    bool transform_dirty = true;
    bool visibility_dirty = true;
    bool base_dirty = true;
    bool lighting_dirty = true;
    bool lod_dirty = true;

    Impl() {
        instance_rid = RenderingServer::get_singleton()->instance_create2(base_rid, (RID)0); // temporary
        RenderingServer::get_singleton()->instance_set_visible(instance_rid, visible);
    }

    ~Impl() {
        if (instance_rid.is_valid()) {
            RenderingServer::get_singleton()->free(instance_rid);
        }
    }

    void sync_base() {
        if (!base_dirty) return;
        RenderingServer::get_singleton()->instance_set_base(instance_rid, base_rid);
        base_dirty = false;
    }

    void sync_transform() {
        if (!transform_dirty) return;
        RenderingServer::get_singleton()->instance_set_transform(instance_rid, cached_transform);
        transform_dirty = false;
    }

    void sync_visibility() {
        if (!visibility_dirty) return;
        RenderingServer::get_singleton()->instance_set_visible(instance_rid, visible);
        RenderingServer::get_singleton()->instance_set_layer_mask(instance_rid, layer_mask);
        visibility_dirty = false;
    }

    void sync_lighting() {
        if (!lighting_dirty) return;
        RenderingServer::get_singleton()->instance_set_cast_shadow(instance_rid, cast_shadow);
        RenderingServer::get_singleton()->instance_set_gi_mode(instance_rid, gi_mode);
        RenderingServer::get_singleton()->instance_set_gi_contribution(instance_rid, gi_contribution);
        RenderingServer::get_singleton()->instance_set_emissive(instance_rid, emissive_color, emissive_intensity);
        lighting_dirty = false;
    }

    void sync_lod() {
        if (!lod_dirty) return;
        RenderingServer::get_singleton()->instance_set_lod_bias(instance_rid, lod_bias);
        if (visibility_range_max > 0.0f) {
            RenderingServer::get_singleton()->instance_set_visibility_range(instance_rid, visibility_range_min, visibility_range_max, visibility_range_fade_margin);
        } else {
            RenderingServer::get_singleton()->instance_set_visibility_range(instance_rid, visibility_range_min, 0.0f, visibility_range_fade_margin);
        }
        lod_dirty = false;
    }

    void sync_all() {
        sync_base();
        sync_transform();
        sync_visibility();
        sync_lighting();
        sync_lod();
    }
};

VisualInstance3DExt::VisualInstance3DExt() {
    pimpl = new Impl();
}

VisualInstance3DExt::~VisualInstance3DExt() {
    delete pimpl;
}

void VisualInstance3DExt::set_base(const RID &p_base) {
    pimpl->base_rid = p_base;
    pimpl->base_set = true;
    pimpl->base_dirty = true;
    sync_instance();
}

void VisualInstance3DExt::set_transform(const Transform3D &p_transform) {
    pimpl->cached_transform = p_transform;
    pimpl->transform_dirty = true;
    sync_instance_transform();
}

void VisualInstance3DExt::set_visible(bool p_visible) {
    pimpl->visible = p_visible;
    pimpl->visibility_dirty = true;
    sync_instance_visibility();
}

void VisualInstance3DExt::set_layer_mask(uint32_t p_layer_mask) {
    pimpl->layer_mask = p_layer_mask;
    pimpl->visibility_dirty = true;
    sync_instance_visibility();
}

void VisualInstance3DExt::set_cast_shadow(bool p_cast) {
    pimpl->cast_shadow = p_cast;
    pimpl->lighting_dirty = true;
    sync_instance_lighting();
}

bool VisualInstance3DExt::get_cast_shadow() const {
    return pimpl->cast_shadow;
}

void VisualInstance3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->lighting_dirty = true;
    sync_instance_lighting();
}

int VisualInstance3DExt::get_gi_mode() const {
    return pimpl->gi_mode;
}

void VisualInstance3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->lighting_dirty = true;
    sync_instance_lighting();
}

float VisualInstance3DExt::get_gi_contribution() const {
    return pimpl->gi_contribution;
}

void VisualInstance3DExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    pimpl->lighting_dirty = true;
    sync_instance_lighting();
}

Color VisualInstance3DExt::get_emissive() const {
    return pimpl->emissive_color;
}

float VisualInstance3DExt::get_emissive_intensity() const {
    return pimpl->emissive_intensity;
}

void VisualInstance3DExt::set_lod_bias(float p_bias) {
    pimpl->lod_bias = p_bias;
    pimpl->lod_dirty = true;
    sync_instance_lod();
}

float VisualInstance3DExt::get_lod_bias() const {
    return pimpl->lod_bias;
}

void VisualInstance3DExt::set_visibility_range(float p_min, float p_max, float p_fade_margin) {
    pimpl->visibility_range_min = p_min;
    pimpl->visibility_range_max = p_max;
    pimpl->visibility_range_fade_margin = p_fade_margin;
    pimpl->lod_dirty = true;
    sync_instance_lod();
}

void VisualInstance3DExt::get_visibility_range(float &p_min, float &p_max, float &p_fade_margin) const {
    p_min = pimpl->visibility_range_min;
    p_max = pimpl->visibility_range_max;
    p_fade_margin = pimpl->visibility_range_fade_margin;
}

void VisualInstance3DExt::sync_instance() {
    pimpl->sync_all();
}

void VisualInstance3DExt::sync_instance_transform() {
    pimpl->sync_transform();
}

void VisualInstance3DExt::sync_instance_visibility() {
    pimpl->sync_visibility();
}

void VisualInstance3DExt::sync_instance_lighting() {
    pimpl->sync_lighting();
}

void VisualInstance3DExt::sync_instance_lod() {
    pimpl->sync_lod();
}

RID VisualInstance3DExt::get_instance_rid() const {
    return pimpl->instance_rid;
}