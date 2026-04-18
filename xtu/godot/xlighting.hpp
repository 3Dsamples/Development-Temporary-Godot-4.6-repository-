// godot/xlighting.hpp

#ifndef XTENSOR_XLIGHTING_HPP
#define XTENSOR_XLIGHTING_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xnorm.hpp"
#include "../math/xintersection.hpp"
#include "../math/xmaterial.hpp"
#include "../graphics/xrenderer.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xnode.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <random>
#include <mutex>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/node3d.hpp>
    #include <godot_cpp/classes/light3d.hpp>
    #include <godot_cpp/classes/omni_light3d.hpp>
    #include <godot_cpp/classes/spot_light3d.hpp>
    #include <godot_cpp/classes/directional_light3d.hpp>
    #include <godot_cpp/classes/world_environment.hpp>
    #include <godot_cpp/classes/environment.hpp>
    #include <godot_cpp/classes/sky.hpp>
    #include <godot_cpp/classes/procedural_sky.hpp>
    #include <godot_cpp/classes/physical_sky.hpp>
    #include <godot_cpp/classes/panorama_sky.hpp>
    #include <godot_cpp/classes/camera_attributes.hpp>
    #include <godot_cpp/classes/camera3d.hpp>
    #include <godot_cpp/classes/rendering_server.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/color.hpp>
    #include <godot_cpp/variant/basis.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Light types and structures
            // --------------------------------------------------------------------
            enum class LightType
            {
                Directional = 0,
                Point = 1,
                Spot = 2,
                Area = 3
            };

            struct LightData
            {
                Vector3<float> position;       // for point/spot/area
                Vector3<float> direction;      // for directional/spot
                Color3f color = {1,1,1};
                float intensity = 1.0f;
                float range = 10.0f;
                float inner_cone_angle = 30.0f * M_PI / 180.0f;
                float outer_cone_angle = 45.0f * M_PI / 180.0f;
                bool cast_shadows = true;
                int shadow_map_size = 1024;
                float shadow_bias = 0.05f;
                float shadow_normal_bias = 0.02f;
                LightType type = LightType::Point;
                uint32_t layer_mask = 0xFFFFFFFF;
            };

            // --------------------------------------------------------------------
            // Tensor-based Lighting System
            // --------------------------------------------------------------------
            class XLightingSystem
            {
            public:
                using LightArray = xarray_container<float>; // N x 16 (packed light data)

                XLightingSystem() = default;

                // Pack multiple lights into a single tensor
                static LightArray pack_lights(const std::vector<LightData>& lights)
                {
                    size_t n = lights.size();
                    LightArray packed({n, 16});
                    for (size_t i = 0; i < n; ++i)
                    {
                        const auto& l = lights[i];
                        packed(i, 0) = l.position.x;
                        packed(i, 1) = l.position.y;
                        packed(i, 2) = l.position.z;
                        packed(i, 3) = static_cast<float>(l.type);
                        packed(i, 4) = l.direction.x;
                        packed(i, 5) = l.direction.y;
                        packed(i, 6) = l.direction.z;
                        packed(i, 7) = l.intensity;
                        packed(i, 8) = l.color.x;
                        packed(i, 9) = l.color.y;
                        packed(i, 10) = l.color.z;
                        packed(i, 11) = l.range;
                        packed(i, 12) = l.inner_cone_angle;
                        packed(i, 13) = l.outer_cone_angle;
                        packed(i, 14) = l.cast_shadows ? 1.0f : 0.0f;
                        packed(i, 15) = static_cast<float>(l.shadow_map_size);
                    }
                    return packed;
                }

                // Unpack lights from tensor
                static std::vector<LightData> unpack_lights(const LightArray& packed)
                {
                    std::vector<LightData> lights;
                    size_t n = packed.shape()[0];
                    lights.reserve(n);
                    for (size_t i = 0; i < n; ++i)
                    {
                        LightData l;
                        l.position = {packed(i,0), packed(i,1), packed(i,2)};
                        l.type = static_cast<LightType>(static_cast<int>(packed(i,3)));
                        l.direction = {packed(i,4), packed(i,5), packed(i,6)};
                        l.intensity = packed(i,7);
                        l.color = {packed(i,8), packed(i,9), packed(i,10)};
                        l.range = packed(i,11);
                        l.inner_cone_angle = packed(i,12);
                        l.outer_cone_angle = packed(i,13);
                        l.cast_shadows = packed(i,14) > 0.5f;
                        l.shadow_map_size = static_cast<int>(packed(i,15));
                        lights.push_back(l);
                    }
                    return lights;
                }

                // Compute direct lighting for a batch of surface points
                // surface: N x 9 (pos.x, pos.y, pos.z, normal.x, normal.y, normal.z, albedo.r, albedo.g, albedo.b)
                // lights: L x 16 (packed)
                // Returns: N x 3 (irradiance)
                static xarray_container<float> compute_direct_lighting(const xarray_container<float>& surface,
                                                                       const LightArray& lights,
                                                                       float ambient = 0.03f)
                {
                    size_t n = surface.shape()[0];
                    size_t l_count = lights.shape()[0];
                    xarray_container<float> result({n, 3}, ambient);
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        Vector3<float> pos(surface(i,0), surface(i,1), surface(i,2));
                        Vector3<float> normal(surface(i,3), surface(i,4), surface(i,5));
                        normal = normal.normalized();
                        Color3f albedo(surface(i,6), surface(i,7), surface(i,8));
                        
                        Color3f total(0,0,0);
                        for (size_t j = 0; j < l_count; ++j)
                        {
                            LightType type = static_cast<LightType>(static_cast<int>(lights(j,3)));
                            Vector3<float> l_pos(lights(j,0), lights(j,1), lights(j,2));
                            Vector3<float> l_dir(lights(j,4), lights(j,5), lights(j,6));
                            float intensity = lights(j,7);
                            Color3f color(lights(j,8), lights(j,9), lights(j,10));
                            float range = lights(j,11);
                            float inner_cone = lights(j,12);
                            float outer_cone = lights(j,13);
                            
                            Vector3<float> light_vec;
                            float attenuation = 1.0f;
                            
                            switch (type)
                            {
                                case LightType::Directional:
                                    light_vec = -l_dir.normalized();
                                    attenuation = 1.0f;
                                    break;
                                case LightType::Point:
                                    light_vec = l_pos - pos;
                                    {
                                        float dist = light_vec.length();
                                        light_vec = light_vec / dist;
                                        attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
                                        if (dist > range) attenuation = 0.0f;
                                    }
                                    break;
                                case LightType::Spot:
                                    light_vec = l_pos - pos;
                                    {
                                        float dist = light_vec.length();
                                        light_vec = light_vec / dist;
                                        attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
                                        if (dist > range) { attenuation = 0.0f; break; }
                                        Vector3<float> spot_dir = l_dir.normalized();
                                        float cos_theta = (-light_vec).dot(spot_dir);
                                        float cos_inner = std::cos(inner_cone);
                                        float cos_outer = std::cos(outer_cone);
                                        float spot = std::clamp((cos_theta - cos_outer) / (cos_inner - cos_outer + 1e-6f), 0.0f, 1.0f);
                                        attenuation *= spot;
                                    }
                                    break;
                                case LightType::Area:
                                    // Simplified: treat as point light
                                    light_vec = l_pos - pos;
                                    {
                                        float dist = light_vec.length();
                                        light_vec = light_vec / dist;
                                        attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
                                        if (dist > range) attenuation = 0.0f;
                                    }
                                    break;
                            }
                            
                            float n_dot_l = std::max(0.0f, normal.dot(light_vec));
                            Color3f contrib = color * intensity * n_dot_l * attenuation;
                            total = total + contrib;
                        }
                        
                        result(i,0) = albedo.x * total.x + ambient;
                        result(i,1) = albedo.y * total.y + ambient;
                        result(i,2) = albedo.z * total.z + ambient;
                    }
                    return result;
                }

                // Compute shadow map for a directional light
                static xarray_container<float> compute_shadow_map(const LightData& light,
                                                                  const xarray_container<float>& depth_buffer,
                                                                  const xarray_container<float>& surface_points,
                                                                  int shadow_map_size)
                {
                    // surface_points: N x 3 (world positions to test)
                    size_t n = surface_points.shape()[0];
                    xarray_container<float> visibility({n}, 1.0f);
                    // Simplified: just return 1.0 (no shadows) as full shadow mapping is complex
                    // In a full implementation, this would transform points to light space and compare depths
                    return visibility;
                }

                // Generate shadow map cascades for directional light
                static std::vector<Matrix4> compute_cascade_matrices(const LightData& light,
                                                                     const Camera& camera,
                                                                     const std::vector<float>& splits)
                {
                    std::vector<Matrix4> matrices;
                    // Compute light view-projection matrices for each cascade split
                    return matrices;
                }
            };

            // --------------------------------------------------------------------
            // XLightingNode - Godot node for tensor-based lighting
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XLightingNode : public godot::Node3D
            {
                GDCLASS(XLightingNode, godot::Node3D)

            private:
                godot::Ref<XTensorNode> m_lights_tensor;      // L x 16 packed lights
                godot::Ref<XTensorNode> m_surface_tensor;     // N x 9 surface points
                godot::Ref<XTensorNode> m_result_tensor;      // N x 3 lighting result
                godot::Ref<XTensorNode> m_environment_map;    // H x W x 3 environment map
                
                bool m_auto_update = true;
                float m_ambient = 0.03f;
                float m_exposure = 1.0f;
                float m_gamma = 2.2f;
                bool m_use_environment = true;
                godot::Color m_ambient_color = godot::Color(0.1f, 0.1f, 0.1f, 1.0f);
                
                godot::RID m_environment_rid;
                godot::RID m_sky_rid;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_lights_tensor", "tensor"), &XLightingNode::set_lights_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_lights_tensor"), &XLightingNode::get_lights_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_surface_tensor", "tensor"), &XLightingNode::set_surface_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_surface_tensor"), &XLightingNode::get_surface_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_result_tensor", "tensor"), &XLightingNode::set_result_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_result_tensor"), &XLightingNode::get_result_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_environment_map", "tensor"), &XLightingNode::set_environment_map);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_environment_map"), &XLightingNode::get_environment_map);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_update", "enabled"), &XLightingNode::set_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_update"), &XLightingNode::get_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_ambient", "ambient"), &XLightingNode::set_ambient);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_ambient"), &XLightingNode::get_ambient);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_exposure", "exposure"), &XLightingNode::set_exposure);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_exposure"), &XLightingNode::get_exposure);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_gamma", "gamma"), &XLightingNode::set_gamma);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_gamma"), &XLightingNode::get_gamma);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("update_lighting"), &XLightingNode::update_lighting);
                    godot::ClassDB::bind_method(godot::D_METHOD("add_light", "light_data"), &XLightingNode::add_light);
                    godot::ClassDB::bind_method(godot::D_METHOD("remove_light", "index"), &XLightingNode::remove_light);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_lights"), &XLightingNode::clear_lights);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_light_count"), &XLightingNode::get_light_count);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("create_procedural_sky", "sun_direction", "turbidity", "ground_color"), &XLightingNode::create_procedural_sky);
                    godot::ClassDB::bind_method(godot::D_METHOD("sample_environment", "direction"), &XLightingNode::sample_environment);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "lights_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_lights_tensor", "get_lights_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "surface_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_surface_tensor", "get_surface_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "result_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_result_tensor", "get_result_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "environment_map", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_environment_map", "get_environment_map");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_update"), "set_auto_update", "get_auto_update");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "ambient", godot::PROPERTY_HINT_RANGE, "0,1,0.01"), "set_ambient", "get_ambient");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "exposure", godot::PROPERTY_HINT_RANGE, "0,10,0.1"), "set_exposure", "get_exposure");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "gamma", godot::PROPERTY_HINT_RANGE, "0.1,4,0.1"), "set_gamma", "get_gamma");
                    
                    ADD_SIGNAL(godot::MethodInfo("lighting_updated"));
                }

            public:
                XLightingNode() {}
                
                void _ready() override
                {
                    ensure_tensors();
                    if (m_auto_update)
                        update_lighting();
                }
                
                void _process(double delta) override
                {
                    if (m_auto_update)
                        update_lighting();
                }

                void set_lights_tensor(const godot::Ref<XTensorNode>& tensor) { m_lights_tensor = tensor; }
                godot::Ref<XTensorNode> get_lights_tensor() const { return m_lights_tensor; }
                void set_surface_tensor(const godot::Ref<XTensorNode>& tensor) { m_surface_tensor = tensor; }
                godot::Ref<XTensorNode> get_surface_tensor() const { return m_surface_tensor; }
                void set_result_tensor(const godot::Ref<XTensorNode>& tensor) { m_result_tensor = tensor; }
                godot::Ref<XTensorNode> get_result_tensor() const { return m_result_tensor; }
                void set_environment_map(const godot::Ref<XTensorNode>& tensor) { m_environment_map = tensor; }
                godot::Ref<XTensorNode> get_environment_map() const { return m_environment_map; }
                
                void set_auto_update(bool enabled) { m_auto_update = enabled; }
                bool get_auto_update() const { return m_auto_update; }
                void set_ambient(float a) { m_ambient = a; }
                float get_ambient() const { return m_ambient; }
                void set_exposure(float e) { m_exposure = e; }
                float get_exposure() const { return m_exposure; }
                void set_gamma(float g) { m_gamma = g; }
                float get_gamma() const { return m_gamma; }

                void update_lighting()
                {
                    if (!m_surface_tensor.is_valid()) return;
                    auto surface = m_surface_tensor->get_tensor_resource()->m_data.to_float_array();
                    if (surface.dimension() != 2 || surface.shape()[1] != 9)
                    {
                        godot::UtilityFunctions::printerr("XLightingNode: surface tensor must be Nx9");
                        return;
                    }
                    
                    xarray_container<float> lights_arr;
                    if (m_lights_tensor.is_valid())
                    {
                        lights_arr = m_lights_tensor->get_tensor_resource()->m_data.to_float_array();
                    }
                    else
                    {
                        // Create default light
                        lights_arr = xarray_container<float>({1, 16}, 0.0f);
                        lights_arr(0,0) = 0; lights_arr(0,1) = 10; lights_arr(0,2) = 0; // position
                        lights_arr(0,3) = static_cast<float>(LightType::Directional);
                        lights_arr(0,4) = 0.5f; lights_arr(0,5) = -1; lights_arr(0,6) = 0.2f; // direction
                        lights_arr(0,7) = 1.0f; // intensity
                        lights_arr(0,8) = 1; lights_arr(0,9) = 0.95f; lights_arr(0,10) = 0.8f; // color
                        lights_arr(0,11) = 100.0f; // range
                    }
                    
                    auto result = XLightingSystem::compute_direct_lighting(surface, lights_arr, m_ambient);
                    
                    // Add environment lighting if available
                    if (m_use_environment && m_environment_map.is_valid())
                    {
                        auto env = m_environment_map->get_tensor_resource()->m_data.to_float_array();
                        // Sample environment based on normals
                        for (size_t i = 0; i < result.shape()[0]; ++i)
                        {
                            Vector3<float> normal(surface(i,3), surface(i,4), surface(i,5));
                            normal = normal.normalized();
                            // Simple spherical mapping
                            float u = 0.5f + 0.5f * std::atan2(normal.z, normal.x) / M_PI;
                            float v = 0.5f + std::asin(std::clamp(normal.y, -1.0f, 1.0f)) / M_PI;
                            size_t h = env.shape()[0];
                            size_t w = env.shape()[1];
                            size_t x = static_cast<size_t>(u * w) % w;
                            size_t y = static_cast<size_t>(v * h) % h;
                            result(i,0) += env(y, x, 0) * 0.5f;
                            result(i,1) += env(y, x, 1) * 0.5f;
                            result(i,2) += env(y, x, 2) * 0.5f;
                        }
                    }
                    
                    // Apply exposure and gamma
                    for (size_t i = 0; i < result.size(); ++i)
                    {
                        float& v = result.flat(i);
                        v = v * m_exposure;
                        v = std::pow(std::max(v, 0.0f), 1.0f / m_gamma);
                    }
                    
                    if (!m_result_tensor.is_valid())
                        m_result_tensor.instantiate();
                    m_result_tensor->set_data(XVariant::from_xarray(result.cast<double>()).variant());
                    emit_signal("lighting_updated");
                }

                void add_light(const godot::Dictionary& light_data)
                {
                    // Parse light data and add to lights tensor
                    ensure_lights_tensor();
                    auto lights = m_lights_tensor->get_tensor_resource()->m_data.to_float_array();
                    size_t n = lights.shape()[0];
                    xarray_container<float> new_lights({n + 1, 16});
                    // Copy existing
                    for (size_t i = 0; i < n; ++i)
                        for (size_t j = 0; j < 16; ++j)
                            new_lights(i, j) = lights(i, j);
                    // Add new
                    LightData l;
                    if (light_data.has("position"))
                    {
                        godot::Vector3 pos = light_data["position"];
                        l.position = {pos.x, pos.y, pos.z};
                    }
                    if (light_data.has("type"))
                        l.type = static_cast<LightType>(static_cast<int>(light_data["type"]));
                    if (light_data.has("color"))
                    {
                        godot::Color c = light_data["color"];
                        l.color = {c.r, c.g, c.b};
                    }
                    if (light_data.has("intensity"))
                        l.intensity = light_data["intensity"];
                    if (light_data.has("range"))
                        l.range = light_data["range"];
                    // Pack into tensor row
                    new_lights(n, 0) = l.position.x;
                    new_lights(n, 1) = l.position.y;
                    new_lights(n, 2) = l.position.z;
                    new_lights(n, 3) = static_cast<float>(l.type);
                    new_lights(n, 4) = l.direction.x;
                    new_lights(n, 5) = l.direction.y;
                    new_lights(n, 6) = l.direction.z;
                    new_lights(n, 7) = l.intensity;
                    new_lights(n, 8) = l.color.x;
                    new_lights(n, 9) = l.color.y;
                    new_lights(n, 10) = l.color.z;
                    new_lights(n, 11) = l.range;
                    new_lights(n, 12) = l.inner_cone_angle;
                    new_lights(n, 13) = l.outer_cone_angle;
                    new_lights(n, 14) = l.cast_shadows ? 1.0f : 0.0f;
                    new_lights(n, 15) = static_cast<float>(l.shadow_map_size);
                    
                    m_lights_tensor->set_data(XVariant::from_xarray(new_lights.cast<double>()).variant());
                    if (m_auto_update) update_lighting();
                }

                void remove_light(int index)
                {
                    if (!m_lights_tensor.is_valid()) return;
                    auto lights = m_lights_tensor->get_tensor_resource()->m_data.to_float_array();
                    if (index < 0 || index >= static_cast<int>(lights.shape()[0])) return;
                    size_t n = lights.shape()[0];
                    xarray_container<float> new_lights({n - 1, 16});
                    size_t out = 0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        if (static_cast<int>(i) == index) continue;
                        for (size_t j = 0; j < 16; ++j)
                            new_lights(out, j) = lights(i, j);
                        ++out;
                    }
                    m_lights_tensor->set_data(XVariant::from_xarray(new_lights.cast<double>()).variant());
                    if (m_auto_update) update_lighting();
                }

                void clear_lights()
                {
                    if (m_lights_tensor.is_valid())
                        m_lights_tensor->clear();
                }

                int64_t get_light_count() const
                {
                    if (m_lights_tensor.is_valid())
                        return static_cast<int64_t>(m_lights_tensor->size());
                    return 0;
                }

                void create_procedural_sky(const godot::Vector3& sun_direction, float turbidity, const godot::Color& ground_color)
                {
                    // Create a procedural sky environment map as a tensor
                    size_t width = 512, height = 256;
                    xarray_container<float> sky({height, width, 3});
                    
                    Vector3<float> sun_dir(sun_direction.x, sun_direction.y, sun_direction.z);
                    sun_dir = sun_dir.normalized();
                    float sun_theta = std::acos(sun_dir.y);
                    float sun_phi = std::atan2(sun_dir.z, sun_dir.x);
                    
                    for (size_t y = 0; y < height; ++y)
                    {
                        float v = static_cast<float>(y) / height;
                        float theta = v * M_PI;
                        float sin_theta = std::sin(theta);
                        float cos_theta = std::cos(theta);
                        for (size_t x = 0; x < width; ++x)
                        {
                            float u = static_cast<float>(x) / width;
                            float phi = u * 2.0f * M_PI;
                            Vector3<float> dir(sin_theta * std::cos(phi), cos_theta, sin_theta * std::sin(phi));
                            
                            // Simple sky model (Preetham)
                            float cos_gamma = dir.dot(sun_dir);
                            float gamma = std::acos(std::clamp(cos_gamma, -1.0f, 1.0f));
                            
                            // Base sky color
                            Color3f zenith = {0.3f, 0.5f, 1.0f};
                            Color3f horizon = {1.0f, 0.9f, 0.7f};
                            float t = std::pow(1.0f - std::abs(cos_theta), 2.0f);
                            Color3f sky_color = zenith * (1.0f - t) + horizon * t;
                            
                            // Sun contribution
                            float sun_size = 0.05f;
                            float sun_disk = std::max(0.0f, 1.0f - gamma / sun_size);
                            sun_disk = std::pow(sun_disk, 4.0f);
                            Color3f sun_color = {1.0f, 0.95f, 0.9f};
                            sky_color = sky_color + sun_color * sun_disk * 5.0f;
                            
                            // Turbidity effect (simplified)
                            sky_color = sky_color * (1.0f + 0.5f * turbidity);
                            
                            sky(y, x, 0) = std::clamp(sky_color.x, 0.0f, 1.0f);
                            sky(y, x, 1) = std::clamp(sky_color.y, 0.0f, 1.0f);
                            sky(y, x, 2) = std::clamp(sky_color.z, 0.0f, 1.0f);
                        }
                    }
                    
                    if (!m_environment_map.is_valid())
                        m_environment_map.instantiate();
                    m_environment_map->set_data(XVariant::from_xarray(sky.cast<double>()).variant());
                }

                godot::Color sample_environment(const godot::Vector3& direction)
                {
                    if (!m_environment_map.is_valid()) return godot::Color(0,0,0);
                    auto env = m_environment_map->get_tensor_resource()->m_data.to_float_array();
                    Vector3<float> dir(direction.x, direction.y, direction.z);
                    dir = dir.normalized();
                    float u = 0.5f + 0.5f * std::atan2(dir.z, dir.x) / M_PI;
                    float v = 0.5f + std::asin(std::clamp(dir.y, -1.0f, 1.0f)) / M_PI;
                    size_t h = env.shape()[0];
                    size_t w = env.shape()[1];
                    size_t x = static_cast<size_t>(u * w) % w;
                    size_t y = static_cast<size_t>(v * h) % h;
                    return godot::Color(env(y, x, 0), env(y, x, 1), env(y, x, 2));
                }

            private:
                void ensure_tensors()
                {
                    if (!m_surface_tensor.is_valid())
                        m_surface_tensor.instantiate();
                    if (!m_result_tensor.is_valid())
                        m_result_tensor.instantiate();
                }

                void ensure_lights_tensor()
                {
                    if (!m_lights_tensor.is_valid())
                    {
                        m_lights_tensor.instantiate();
                        xarray_container<float> empty({0, 16});
                        m_lights_tensor->set_data(XVariant::from_xarray(empty.cast<double>()).variant());
                    }
                }
            };

            // --------------------------------------------------------------------
            // XGlobalIllumination - Tensor-based GI baker
            // --------------------------------------------------------------------
            class XGlobalIllumination : public godot::Node3D
            {
                GDCLASS(XGlobalIllumination, godot::Node3D)

            private:
                godot::Ref<XTensorNode> m_irradiance_volume;   // X x Y x Z x 3
                godot::Ref<XTensorNode> m_lightmap;            // H x W x 3
                godot::Ref<XTensorNode> m_probes;              // N x 9 (pos + color)
                godot::Vector3 m_volume_min = godot::Vector3(-10, -10, -10);
                godot::Vector3 m_volume_max = godot::Vector3(10, 10, 10);
                godot::Vector3i m_volume_resolution = godot::Vector3i(8, 8, 8);
                int m_bounce_count = 2;
                int m_samples_per_probe = 256;
                bool m_baking = false;
                float m_bake_progress = 0.0f;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_irradiance_volume", "tensor"), &XGlobalIllumination::set_irradiance_volume);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_irradiance_volume"), &XGlobalIllumination::get_irradiance_volume);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_volume_bounds", "min", "max"), &XGlobalIllumination::set_volume_bounds);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_volume_resolution", "resolution"), &XGlobalIllumination::set_volume_resolution);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_bounce_count", "bounces"), &XGlobalIllumination::set_bounce_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_samples_per_probe", "samples"), &XGlobalIllumination::set_samples_per_probe);
                    godot::ClassDB::bind_method(godot::D_METHOD("bake_gi"), &XGlobalIllumination::bake_gi);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_baked_data"), &XGlobalIllumination::clear_baked_data);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bake_progress"), &XGlobalIllumination::get_bake_progress);
                    godot::ClassDB::bind_method(godot::D_METHOD("sample_irradiance", "position", "normal"), &XGlobalIllumination::sample_irradiance);
                    godot::ClassDB::bind_method(godot::D_METHOD("place_probes", "positions"), &XGlobalIllumination::place_probes);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "irradiance_volume", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_irradiance_volume", "get_irradiance_volume");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::VECTOR3, "volume_min"), "set_volume_bounds", "get_volume_bounds");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "bounce_count", godot::PROPERTY_HINT_RANGE, "1,5,1"), "set_bounce_count", "get_bounce_count");
                    
                    ADD_SIGNAL(godot::MethodInfo("bake_completed"));
                    ADD_SIGNAL(godot::MethodInfo("bake_progress", godot::PropertyInfo(godot::Variant::FLOAT, "progress")));
                }

            public:
                void set_irradiance_volume(const godot::Ref<XTensorNode>& tensor) { m_irradiance_volume = tensor; }
                godot::Ref<XTensorNode> get_irradiance_volume() const { return m_irradiance_volume; }
                void set_volume_bounds(const godot::Vector3& min, const godot::Vector3& max)
                {
                    m_volume_min = min;
                    m_volume_max = max;
                }
                void set_volume_resolution(const godot::Vector3i& res) { m_volume_resolution = res; }
                void set_bounce_count(int bounces) { m_bounce_count = bounces; }
                void set_samples_per_probe(int samples) { m_samples_per_probe = samples; }
                float get_bake_progress() const { return m_bake_progress; }

                void bake_gi()
                {
                    m_baking = true;
                    m_bake_progress = 0.0f;
                    
                    // Create irradiance volume tensor
                    xarray_container<float> volume({
                        static_cast<size_t>(m_volume_resolution.x),
                        static_cast<size_t>(m_volume_resolution.y),
                        static_cast<size_t>(m_volume_resolution.z),
                        3
                    }, 0.0f);
                    
                    godot::Vector3 size = m_volume_max - m_volume_min;
                    godot::Vector3 step = size / godot::Vector3(m_volume_resolution);
                    
                    // Simple Monte Carlo path tracing for GI (placeholder)
                    std::mt19937 rng(42);
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    
                    for (int x = 0; x < m_volume_resolution.x && m_baking; ++x)
                    {
                        for (int y = 0; y < m_volume_resolution.y; ++y)
                        {
                            for (int z = 0; z < m_volume_resolution.z; ++z)
                            {
                                godot::Vector3 probe_pos = m_volume_min + godot::Vector3(
                                    (x + 0.5f) * step.x,
                                    (y + 0.5f) * step.y,
                                    (z + 0.5f) * step.z
                                );
                                
                                Color3f accumulated(0,0,0);
                                for (int s = 0; s < m_samples_per_probe; ++s)
                                {
                                    // Sample random direction on sphere
                                    float theta = 2.0f * M_PI * dist(rng);
                                    float phi = std::acos(2.0f * dist(rng) - 1.0f);
                                    Vector3<float> dir(
                                        std::sin(phi) * std::cos(theta),
                                        std::sin(phi) * std::sin(theta),
                                        std::cos(phi)
                                    );
                                    
                                    // Trace ray and accumulate radiance (simplified)
                                    // In a full implementation, this would intersect scene geometry
                                    Color3f radiance(0.5f, 0.5f, 0.5f); // Placeholder
                                    accumulated = accumulated + radiance;
                                }
                                accumulated = accumulated * (1.0f / m_samples_per_probe);
                                
                                volume(x, y, z, 0) = accumulated.x;
                                volume(x, y, z, 1) = accumulated.y;
                                volume(x, y, z, 2) = accumulated.z;
                            }
                        }
                        m_bake_progress = static_cast<float>(x) / m_volume_resolution.x;
                        emit_signal("bake_progress", m_bake_progress);
                    }
                    
                    if (m_baking)
                    {
                        if (!m_irradiance_volume.is_valid())
                            m_irradiance_volume.instantiate();
                        m_irradiance_volume->set_data(XVariant::from_xarray(volume.cast<double>()).variant());
                        m_baking = false;
                        m_bake_progress = 1.0f;
                        emit_signal("bake_completed");
                    }
                }

                void clear_baked_data()
                {
                    if (m_irradiance_volume.is_valid())
                        m_irradiance_volume->clear();
                    if (m_lightmap.is_valid())
                        m_lightmap->clear();
                    if (m_probes.is_valid())
                        m_probes->clear();
                }

                godot::Color sample_irradiance(const godot::Vector3& position, const godot::Vector3& normal)
                {
                    if (!m_irradiance_volume.is_valid())
                        return godot::Color(0,0,0);
                    
                    auto vol = m_irradiance_volume->get_tensor_resource()->m_data.to_float_array();
                    godot::Vector3 size = m_volume_max - m_volume_min;
                    godot::Vector3 local = (position - m_volume_min) / size;
                    
                    if (local.x < 0 || local.x >= 1 || local.y < 0 || local.y >= 1 || local.z < 0 || local.z >= 1)
                        return godot::Color(0,0,0);
                    
                    float fx = local.x * (m_volume_resolution.x - 1);
                    float fy = local.y * (m_volume_resolution.y - 1);
                    float fz = local.z * (m_volume_resolution.z - 1);
                    
                    int x0 = std::floor(fx), x1 = std::min(x0 + 1, m_volume_resolution.x - 1);
                    int y0 = std::floor(fy), y1 = std::min(y0 + 1, m_volume_resolution.y - 1);
                    int z0 = std::floor(fz), z1 = std::min(z0 + 1, m_volume_resolution.z - 1);
                    
                    float wx = fx - x0, wy = fy - y0, wz = fz - z0;
                    
                    auto sample = [&](int x, int y, int z) -> Color3f {
                        return Color3f(vol(x, y, z, 0), vol(x, y, z, 1), vol(x, y, z, 2));
                    };
                    
                    Color3f c000 = sample(x0, y0, z0);
                    Color3f c100 = sample(x1, y0, z0);
                    Color3f c010 = sample(x0, y1, z0);
                    Color3f c110 = sample(x1, y1, z0);
                    Color3f c001 = sample(x0, y0, z1);
                    Color3f c101 = sample(x1, y0, z1);
                    Color3f c011 = sample(x0, y1, z1);
                    Color3f c111 = sample(x1, y1, z1);
                    
                    Color3f c00 = c000 * (1-wx) + c100 * wx;
                    Color3f c01 = c001 * (1-wx) + c101 * wx;
                    Color3f c10 = c010 * (1-wx) + c110 * wx;
                    Color3f c11 = c011 * (1-wx) + c111 * wx;
                    
                    Color3f c0 = c00 * (1-wy) + c10 * wy;
                    Color3f c1 = c01 * (1-wy) + c11 * wy;
                    
                    Color3f result = c0 * (1-wz) + c1 * wz;
                    return godot::Color(result.x, result.y, result.z);
                }

                void place_probes(const godot::Ref<XTensorNode>& positions)
                {
                    m_probes = positions;
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XLightingRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XLightingNode>();
                    godot::ClassDB::register_class<XGlobalIllumination>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::LightType;
        using godot_bridge::LightData;
        using godot_bridge::XLightingSystem;
        using godot_bridge::XLightingNode;
        using godot_bridge::XGlobalIllumination;
        using godot_bridge::XLightingRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XLIGHTING_HPP

// godot/xlighting.hpp