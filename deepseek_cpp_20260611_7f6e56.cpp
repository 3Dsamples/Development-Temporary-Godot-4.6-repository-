// Name : lighting enhancement
// File : scene/3d/light_3d.h file number : 3
// Description : Advanced 3D light node with support for directional, point, spot,
//               area lights, shadows (CSM, PCSS, VSM, CHS), IBL, light probes,
//               reflection probes, volumetric effects, and real-time GI integration.
#pragma once

#include "scene/3d/node_3d.h"
#include "scene/resources/environment.h"
#include "scene/resources/sky.h"
#include "servers/rendering_server.h"
#include "servers/rendering/rendering_light_culler.h"
#include "core/math/color.h"
#include "core/math/transform_3d.h"
#include "core/templates/hash_map.h"

class Light3D : public Node3D {
    GDCLASS(Light3D, Node3D);

public:
    enum LightType {
        LIGHT_DIRECTIONAL,
        LIGHT_POINT,
        LIGHT_SPOT,
        LIGHT_RECTANGLE,
        LIGHT_DISC,
        LIGHT_SPHERE
    };

    enum ShadowTechnique {
        SHADOW_PCF,
        SHADOW_PCSS,
        SHADOW_VSM,
        SHADOW_CHS
    };

    Light3D();
    ~Light3D();

    // Basic light parameters
    void set_light_type(LightType p_type);
    LightType get_light_type() const;
    void set_color(const Color &p_color);
    Color get_color() const;
    void set_intensity(float p_intensity);
    float get_intensity() const;
    void set_range(float p_range);
    float get_range() const;

    // Spot light specific
    void set_spot_angle(float p_degrees);
    float get_spot_angle() const;
    void set_spot_attenuation(float p_attenuation);
    float get_spot_attenuation() const;

    // Area light geometry
    void set_size(const Vector2 &p_size);
    Vector2 get_size() const;
    void set_radius(float p_radius);
    float get_radius() const;

    // Shadows
    void set_shadow_enabled(bool p_enabled);
    bool is_shadow_enabled() const;
    void set_shadow_technique(ShadowTechnique p_technique);
    ShadowTechnique get_shadow_technique() const;
    void set_shadow_map_resolution(int p_res);
    int get_shadow_map_resolution() const;
    void set_csm_cascade_count(int p_count);
    int get_csm_cascade_count() const;
    void set_csm_split_lambda(float p_lambda);
    float get_csm_split_lambda() const;
    void set_pcss_light_size(float p_size);
    float get_pcss_light_size() const;
    void set_vsm_exponent(float p_exp);
    float get_vsm_exponent() const;

    // Global Illumination (GI)
    void set_gi_mode(int p_mode); // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;
    void set_use_environment(bool p_use);
    bool get_use_environment() const;
    void set_environment(Ref<Environment> p_env);
    Ref<Environment> get_environment() const;

    // Reflection probes (for area lights)
    void set_reflection_probe_enabled(bool p_enabled);
    bool is_reflection_probe_enabled() const;
    void set_reflection_probe_update_rate(float p_fps);
    float get_reflection_probe_update_rate() const;
    void capture_reflection_probe();

    // Light probes (irradiance volumes)
    void set_light_probe_grid(const AABB &p_bounds, const Vector3i &p_resolution);
    void clear_light_probe_grid();
    void update_light_probes();

    // Volumetric fog contribution
    void set_volumetric_enabled(bool p_enabled);
    bool is_volumetric_enabled() const;
    void set_volumetric_fog_intensity(float p_intensity);
    float get_volumetric_fog_intensity() const;

    // Render server synchronization
    void synchronize_render_server(double p_delta) override;

    // Get the RenderingServer light RID for debugging
    RID get_light_rid() const;

protected:
    void _transform_changed() override;
    void _update_render_server_transform() override;

private:
    struct Impl;
    Impl *pimpl;
};

#endif // LIGHT_3D_H