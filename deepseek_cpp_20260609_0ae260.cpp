// geometry_instance_3d.cpp
#include "geometry_instance_3d.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace lighting {

struct GeometryInstance3D::Impl {
    // Material handling
    struct MaterialEntry {
        char path[256];
        bool valid;
        MaterialEntry() : path{0}, valid(false) {}
    };
    std::vector<MaterialEntry> surface_materials;
    char material_override_path[256] = {0};
    bool has_material_override = false;

    // Shadow
    bool cast_shadow = true;
    bool receive_shadow = true;
    bool double_sided_shadow = false;

    // Visibility range
    float visibility_begin = 0.0f;
    float visibility_end = 0.0f;      // 0 = infinite
    float visibility_fade_margin = 0.0f;
    int visibility_fade_mode = 0;     // 0=disabled, 1=opacity, 2=scale

    // LOD
    float lod_bias = 1.0f;
    std::vector<float> lod_thresholds; // distances for LOD levels (if empty, use default)

    // Lightmap
    float lightmap_scale = 1.0f;
    int lightmap_index = -1;

    // Culling
    bool ignore_frustum_culling = false;

    // Rendering server handle
    int64_t render_instance_id = -1;
};

GeometryInstance3D::GeometryInstance3D() : pimpl(std::make_unique<Impl>()) {}
GeometryInstance3D::~GeometryInstance3D() = default;

void GeometryInstance3D::set_material(int surface_idx, const char* material_path) {
    if (surface_idx < 0) return;
    // Ensure vector large enough
    if ((size_t)surface_idx >= pimpl->surface_materials.size())
        pimpl->surface_materials.resize(surface_idx + 1);
    pimpl->surface_materials[surface_idx].valid = true;
    strncpy(pimpl->surface_materials[surface_idx].path, material_path, 255);
    pimpl->surface_materials[surface_idx].path[255] = 0;
    _materials_updated();
}

void GeometryInstance3D::clear_material(int surface_idx) {
    if (surface_idx >= 0 && (size_t)surface_idx < pimpl->surface_materials.size()) {
        pimpl->surface_materials[surface_idx].valid = false;
        pimpl->surface_materials[surface_idx].path[0] = 0;
        _materials_updated();
    }
}

void GeometryInstance3D::clear_all_materials() {
    pimpl->surface_materials.clear();
    _materials_updated();
}

bool GeometryInstance3D::has_material(int surface_idx) const {
    return (surface_idx >= 0 && (size_t)surface_idx < pimpl->surface_materials.size() &&
            pimpl->surface_materials[surface_idx].valid);
}

int GeometryInstance3D::get_material_count() const {
    return (int)pimpl->surface_materials.size();
}

void GeometryInstance3D::set_material_override(const char* material_path) {
    pimpl->has_material_override = true;
    strncpy(pimpl->material_override_path, material_path, 255);
    pimpl->material_override_path[255] = 0;
    _materials_updated();
}

void GeometryInstance3D::clear_material_override() {
    pimpl->has_material_override = false;
    _materials_updated();
}

bool GeometryInstance3D::has_material_override() const {
    return pimpl->has_material_override;
}

void GeometryInstance3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool GeometryInstance3D::get_cast_shadow() const { return pimpl->cast_shadow; }

void GeometryInstance3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
bool GeometryInstance3D::get_receive_shadow() const { return pimpl->receive_shadow; }

void GeometryInstance3D::set_double_sided_shadow(bool double_sided) { pimpl->double_sided_shadow = double_sided; }
bool GeometryInstance3D::is_double_sided_shadow() const { return pimpl->double_sided_shadow; }

void GeometryInstance3D::set_visibility_range(float begin, float end, float fade_margin) {
    pimpl->visibility_begin = begin;
    pimpl->visibility_end = end;
    pimpl->visibility_fade_margin = fade_margin;
}
void GeometryInstance3D::get_visibility_range(float& begin, float& end, float& fade_margin) const {
    begin = pimpl->visibility_begin;
    end = pimpl->visibility_end;
    fade_margin = pimpl->visibility_fade_margin;
}
void GeometryInstance3D::set_visibility_range_fade_mode(int mode) { pimpl->visibility_fade_mode = mode; }
int GeometryInstance3D::get_visibility_range_fade_mode() const { return pimpl->visibility_fade_mode; }

void GeometryInstance3D::set_lod_bias(float bias) { pimpl->lod_bias = bias; }
float GeometryInstance3D::get_lod_bias() const { return pimpl->lod_bias; }

void GeometryInstance3D::set_lod_thresholds(const float* thresholds, int count) {
    pimpl->lod_thresholds.assign(thresholds, thresholds + count);
}
const std::vector<float>& GeometryInstance3D::get_lod_thresholds() const { return pimpl->lod_thresholds; }

void GeometryInstance3D::set_lightmap_scale(float scale) { pimpl->lightmap_scale = scale; }
float GeometryInstance3D::get_lightmap_scale() const { return pimpl->lightmap_scale; }
void GeometryInstance3D::set_lightmap_index(int index) { pimpl->lightmap_index = index; }
int GeometryInstance3D::get_lightmap_index() const { return pimpl->lightmap_index; }

void GeometryInstance3D::set_ignore_frustum_culling(bool ignore) { pimpl->ignore_frustum_culling = ignore; }
bool GeometryInstance3D::get_ignore_frustum_culling() const { return pimpl->ignore_frustum_culling; }

void GeometryInstance3D::synchronize_render_server(double delta) {
    VisualInstance3D::synchronize_render_server(delta);
    // Send material overrides and shadow flags to render server
    // Placeholder: actual implementation would call RenderingServer API.
}

void GeometryInstance3D::_materials_updated() {
    // Notify render server that materials changed
    // If instance exists, update materials
    if (get_render_instance_id() != -1) {
        // RenderingServer::instance_set_materials(...)
    }
}

} // namespace lighting