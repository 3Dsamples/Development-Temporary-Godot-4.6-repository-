// godot/xshader.hpp

#ifndef XTENSOR_XSHADER_HPP
#define XTENSOR_XSHADER_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
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
#include <fstream>
#include <sstream>
#include <regex>
#include <mutex>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/rd_shader_file.hpp>
    #include <godot_cpp/classes/rd_shader_spirv.hpp>
    #include <godot_cpp/classes/rd_pipeline_specialization_constant.hpp>
    #include <godot_cpp/classes/rd_shader_source.hpp>
    #include <godot_cpp/classes/rendering_device.hpp>
    #include <godot_cpp/classes/rendering_server.hpp>
    #include <godot_cpp/classes/shader.hpp>
    #include <godot_cpp/classes/shader_material.hpp>
    #include <godot_cpp/classes/material.hpp>
    #include <godot_cpp/classes/standard_material3d.hpp>
    #include <godot_cpp/classes/orm_material3d.hpp>
    #include <godot_cpp/classes/resource_loader.hpp>
    #include <godot_cpp/classes/file_access.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/color.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Shader types and language support
            // --------------------------------------------------------------------
            enum class ShaderType
            {
                Vertex,
                Fragment,
                Compute,
                TessellationControl,
                TessellationEvaluation,
                Geometry,
                Task,
                Mesh
            };

            enum class ShadingLanguage
            {
                GLSL,
                HLSL,
                Metal,
                SPIRV
            };

            enum class ShaderStage
            {
                VERTEX = 0,
                FRAGMENT = 1,
                COMPUTE = 2,
                TESS_CONTROL = 3,
                TESS_EVAL = 4,
                GEOMETRY = 5
            };

            // --------------------------------------------------------------------
            // Shader reflection data
            // --------------------------------------------------------------------
            struct ShaderUniform
            {
                std::string name;
                uint32_t binding;
                uint32_t set;
                enum Type { SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4, TEXTURE, SAMPLER, BUFFER, SUBPASS_INPUT } type;
                size_t array_size = 1;
                size_t size_bytes = 0;
                bool is_storage = false;
                bool is_readonly = false;
            };

            struct ShaderAttribute
            {
                std::string name;
                uint32_t location;
                uint32_t format; // VK_FORMAT_*
                size_t offset = 0;
            };

            struct ShaderPushConstant
            {
                std::string name;
                size_t offset = 0;
                size_t size = 0;
                ShaderStage stage;
            };

            struct ShaderReflection
            {
                std::vector<ShaderUniform> uniforms;
                std::vector<ShaderAttribute> inputs;
                std::vector<ShaderAttribute> outputs;
                std::vector<ShaderPushConstant> push_constants;
                std::vector<std::pair<uint32_t, uint32_t>> specialization_constants;
                size_t local_size_x = 1;
                size_t local_size_y = 1;
                size_t local_size_z = 1;
                std::string entry_point = "main";
            };

            // --------------------------------------------------------------------
            // SPIR-V Compilation and reflection utilities
            // --------------------------------------------------------------------
            namespace spirv
            {
                // Simplified SPIR-V parsing for reflection
                inline ShaderReflection reflect_spirv(const std::vector<uint32_t>& spirv_code)
                {
                    ShaderReflection result;
                    // In a real implementation, this would parse SPIR-V binary using
                    // SPIRV-Cross or similar library to extract bindings, attributes, etc.
                    // For now, we return an empty reflection (placeholder).
                    return result;
                }

                // Cross-compile GLSL to SPIR-V (requires glslang or similar)
                inline std::vector<uint32_t> glsl_to_spirv(const std::string& glsl_source,
                                                           ShaderType type,
                                                           const std::string& entry_point = "main")
                {
                    // Placeholder - would call glslangValidator API
                    return {};
                }

                // Cross-compile HLSL to SPIR-V (requires DXC or glslang)
                inline std::vector<uint32_t> hlsl_to_spirv(const std::string& hlsl_source,
                                                           ShaderType type,
                                                           const std::string& entry_point = "main")
                {
                    // Placeholder - would call DXC
                    return {};
                }

                // Compile to SPIR-V from source with auto-detection
                inline std::vector<uint32_t> compile_to_spirv(const std::string& source,
                                                              ShadingLanguage lang,
                                                              ShaderType type,
                                                              const std::string& entry_point = "main")
                {
                    switch (lang)
                    {
                        case ShadingLanguage::GLSL:
                            return glsl_to_spirv(source, type, entry_point);
                        case ShadingLanguage::HLSL:
                            return hlsl_to_spirv(source, type, entry_point);
                        default:
                            return {};
                    }
                }
            }

            // --------------------------------------------------------------------
            // Tensor-based Shader Parameter Block
            // --------------------------------------------------------------------
            class ShaderParameterBlock
            {
            public:
                // Pack shader parameters into a contiguous buffer for GPU upload
                static xarray_container<float> pack_parameters(const std::map<std::string, godot::Variant>& params,
                                                                const std::vector<ShaderUniform>& uniforms)
                {
                    // Calculate required buffer size
                    size_t total_size = 0;
                    for (const auto& u : uniforms)
                    {
                        if (u.type == ShaderUniform::BUFFER || u.type == ShaderUniform::TEXTURE) continue;
                        total_size += u.size_bytes * u.array_size;
                        // Align to 16 bytes (vec4 alignment)
                        total_size = (total_size + 15) & ~15;
                    }
                    size_t num_floats = total_size / sizeof(float);
                    xarray_container<float> buffer({num_floats}, 0.0f);
                    
                    size_t offset = 0;
                    for (const auto& u : uniforms)
                    {
                        auto it = params.find(u.name);
                        if (it == params.end()) continue;
                        
                        // Write parameter to buffer at offset
                        // This is a simplified placeholder - real implementation would handle
                        // std140/std430 layout rules.
                        switch (u.type)
                        {
                            case ShaderUniform::SCALAR:
                                if (it->second.get_type() == godot::Variant::FLOAT)
                                    buffer(offset) = static_cast<float>(it->second);
                                else if (it->second.get_type() == godot::Variant::INT)
                                    buffer(offset) = static_cast<float>(static_cast<int>(it->second));
                                offset += 4;
                                break;
                            case ShaderUniform::VEC2:
                                if (it->second.get_type() == godot::Variant::VECTOR2)
                                {
                                    godot::Vector2 v = it->second;
                                    buffer(offset) = v.x;
                                    buffer(offset+1) = v.y;
                                }
                                offset += 4;
                                break;
                            case ShaderUniform::VEC3:
                                if (it->second.get_type() == godot::Variant::VECTOR3)
                                {
                                    godot::Vector3 v = it->second;
                                    buffer(offset) = v.x;
                                    buffer(offset+1) = v.y;
                                    buffer(offset+2) = v.z;
                                }
                                else if (it->second.get_type() == godot::Variant::COLOR)
                                {
                                    godot::Color c = it->second;
                                    buffer(offset) = c.r;
                                    buffer(offset+1) = c.g;
                                    buffer(offset+2) = c.b;
                                }
                                offset += 4;
                                break;
                            case ShaderUniform::VEC4:
                                if (it->second.get_type() == godot::Variant::COLOR)
                                {
                                    godot::Color c = it->second;
                                    buffer(offset) = c.r;
                                    buffer(offset+1) = c.g;
                                    buffer(offset+2) = c.b;
                                    buffer(offset+3) = c.a;
                                }
                                else if (it->second.get_type() == godot::Variant::QUATERNION)
                                {
                                    godot::Quaternion q = it->second;
                                    buffer(offset) = q.x;
                                    buffer(offset+1) = q.y;
                                    buffer(offset+2) = q.z;
                                    buffer(offset+3) = q.w;
                                }
                                offset += 4;
                                break;
                            case ShaderUniform::MAT4:
                                if (it->second.get_type() == godot::Variant::TRANSFORM3D)
                                {
                                    godot::Transform3D t = it->second;
                                    // Write 4x3 matrix as 3 vec4s
                                    for (int col = 0; col < 3; ++col)
                                    {
                                        godot::Vector3 v = t.basis.get_column(col);
                                        buffer(offset + col*4) = v.x;
                                        buffer(offset + col*4 + 1) = v.y;
                                        buffer(offset + col*4 + 2) = v.z;
                                        buffer(offset + col*4 + 3) = (col == 3) ? 1.0f : 0.0f;
                                    }
                                    // Origin as fourth column
                                    buffer(offset + 12) = t.origin.x;
                                    buffer(offset + 13) = t.origin.y;
                                    buffer(offset + 14) = t.origin.z;
                                    buffer(offset + 15) = 1.0f;
                                }
                                offset += 16;
                                break;
                            default:
                                break;
                        }
                        // Align to 16-byte boundary
                        offset = (offset + 15) & ~15;
                    }
                    return buffer;
                }

                // Create uniform buffer from tensor
                static xarray_container<float> create_uniform_buffer(const ShaderReflection& reflection,
                                                                     const std::map<std::string, godot::Variant>& params)
                {
                    return pack_parameters(params, reflection.uniforms);
                }
            };

            // --------------------------------------------------------------------
            // XShader - Godot resource for tensor-based shaders
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XShader : public godot::Resource
            {
                GDCLASS(XShader, godot::Resource)

            public:
                enum ShaderType
                {
                    TYPE_SPATIAL = 0,
                    TYPE_CANVAS_ITEM = 1,
                    TYPE_PARTICLES = 2,
                    TYPE_SKY = 3,
                    TYPE_FOG = 4,
                    TYPE_COMPUTE = 5
                };

            private:
                godot::String m_code;
                ShaderType m_shader_type = TYPE_SPATIAL;
                ShadingLanguage m_language = ShadingLanguage::GLSL;
                godot::Ref<godot::RDShaderFile> m_rd_shader;
                godot::RID m_shader_rid;
                godot::RID m_compute_pipeline_rid;
                ShaderReflection m_reflection;
                std::vector<uint32_t> m_spirv_vertex;
                std::vector<uint32_t> m_spirv_fragment;
                std::vector<uint32_t> m_spirv_compute;
                bool m_compiled = false;
                std::string m_compile_error;
                
                godot::Ref<XTensorNode> m_parameters_tensor;
                godot::Ref<XTensorNode> m_output_tensor;
                std::map<std::string, godot::Ref<godot::Texture2D>> m_textures;
                std::map<std::string, godot::Ref<godot::Texture3D>> m_textures_3d;
                std::map<std::string, godot::RID> m_buffers;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_code", "code"), &XShader::set_code);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_code"), &XShader::get_code);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_shader_type", "type"), &XShader::set_shader_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_shader_type"), &XShader::get_shader_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_language", "language"), &XShader::set_language);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_language"), &XShader::get_language);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("compile"), &XShader::compile);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_compiled"), &XShader::is_compiled);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_compile_error"), &XShader::get_compile_error);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_reflection"), &XShader::get_reflection);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("set_parameter", "name", "value"), &XShader::set_parameter);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_parameter", "name"), &XShader::get_parameter);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_texture", "name", "texture"), &XShader::set_texture);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_buffer", "name", "buffer_rid"), &XShader::set_buffer);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_parameters_tensor", "tensor"), &XShader::set_parameters_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_parameters_tensor"), &XShader::get_parameters_tensor);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("dispatch_compute", "work_groups_x", "work_groups_y", "work_groups_z"), &XShader::dispatch_compute);
                    godot::ClassDB::bind_method(godot::D_METHOD("create_material"), &XShader::create_material);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_rd_shader"), &XShader::get_rd_shader);
                    
                    godot::ClassDB::bind_static_method("XShader", godot::D_METHOD("load_from_file", "path"), &XShader::load_from_file);
                    godot::ClassDB::bind_static_method("XShader", godot::D_METHOD("create_compute", "code", "entry_point"), &XShader::create_compute);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "code", godot::PROPERTY_HINT_TYPE_STRING, godot::PROPERTY_USAGE_DEFAULT, "Shader"), "set_code", "get_code");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "shader_type", godot::PROPERTY_HINT_ENUM, "Spatial,CanvasItem,Particles,Sky,Fog,Compute"), "set_shader_type", "get_shader_type");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "language", godot::PROPERTY_HINT_ENUM, "GLSL,HLSL,Metal,SPIRV"), "set_language", "get_language");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "parameters_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_parameters_tensor", "get_parameters_tensor");
                    
                    ADD_SIGNAL(godot::MethodInfo("compiled", godot::PropertyInfo(godot::Variant::BOOL, "success")));
                    
                    BIND_ENUM_CONSTANT(TYPE_SPATIAL);
                    BIND_ENUM_CONSTANT(TYPE_CANVAS_ITEM);
                    BIND_ENUM_CONSTANT(TYPE_PARTICLES);
                    BIND_ENUM_CONSTANT(TYPE_SKY);
                    BIND_ENUM_CONSTANT(TYPE_FOG);
                    BIND_ENUM_CONSTANT(TYPE_COMPUTE);
                }

            public:
                XShader() {}
                ~XShader() { cleanup(); }

                void set_code(const godot::String& code) { m_code = code; m_compiled = false; }
                godot::String get_code() const { return m_code; }
                void set_shader_type(ShaderType type) { m_shader_type = type; m_compiled = false; }
                ShaderType get_shader_type() const { return m_shader_type; }
                void set_language(ShadingLanguage lang) { m_language = lang; m_compiled = false; }
                ShadingLanguage get_language() const { return m_language; }

                bool compile()
                {
                    cleanup();
                    m_compile_error.clear();
                    
                    if (m_code.is_empty())
                    {
                        m_compile_error = "Shader code is empty";
                        emit_signal("compiled", false);
                        return false;
                    }
                    
                    std::string source = m_code.utf8().get_data();
                    
                    // Extract shader stages from source (for GLSL, based on preprocessor)
                    std::string vertex_source, fragment_source, compute_source;
                    parse_stages(source, vertex_source, fragment_source, compute_source);
                    
                    godot::RenderingDevice* rd = godot::RenderingServer::get_singleton()->get_rendering_device();
                    if (!rd)
                    {
                        m_compile_error = "RenderingDevice not available";
                        emit_signal("compiled", false);
                        return false;
                    }
                    
                    m_rd_shader.instantiate();
                    
                    if (m_shader_type == TYPE_COMPUTE)
                    {
                        ShaderType type = ShaderType::Compute;
                        m_spirv_compute = spirv::compile_to_spirv(compute_source.empty() ? source : compute_source,
                                                                  m_language, type, "main");
                        if (m_spirv_compute.empty())
                        {
                            m_compile_error = "Failed to compile compute shader to SPIR-V";
                            emit_signal("compiled", false);
                            return false;
                        }
                        godot::Ref<godot::RDShaderSPIRV> spirv;
                        spirv.instantiate();
                        spirv->set_stage_bytecode(godot::RenderingDevice::SHADER_STAGE_COMPUTE, m_spirv_compute);
                        m_rd_shader->set_spirv(spirv);
                        m_shader_rid = rd->shader_create_from_spirv(spirv);
                        
                        // Create compute pipeline
                        godot::RID pipeline_rid = rd->compute_pipeline_create(m_shader_rid);
                        m_compute_pipeline_rid = pipeline_rid;
                    }
                    else
                    {
                        ShaderType vtype = ShaderType::Vertex;
                        ShaderType ftype = ShaderType::Fragment;
                        m_spirv_vertex = spirv::compile_to_spirv(vertex_source.empty() ? extract_stage(source, "vertex") : vertex_source,
                                                                 m_language, vtype, "vertex");
                        m_spirv_fragment = spirv::compile_to_spirv(fragment_source.empty() ? extract_stage(source, "fragment") : fragment_source,
                                                                   m_language, ftype, "fragment");
                        if (m_spirv_vertex.empty() || m_spirv_fragment.empty())
                        {
                            m_compile_error = "Failed to compile vertex/fragment shader to SPIR-V";
                            emit_signal("compiled", false);
                            return false;
                        }
                        godot::Ref<godot::RDShaderSPIRV> spirv;
                        spirv.instantiate();
                        spirv->set_stage_bytecode(godot::RenderingDevice::SHADER_STAGE_VERTEX, m_spirv_vertex);
                        spirv->set_stage_bytecode(godot::RenderingDevice::SHADER_STAGE_FRAGMENT, m_spirv_fragment);
                        m_rd_shader->set_spirv(spirv);
                        m_shader_rid = rd->shader_create_from_spirv(spirv);
                    }
                    
                    // Reflect shader to get uniform bindings
                    if (m_shader_type == TYPE_COMPUTE)
                        m_reflection = spirv::reflect_spirv(m_spirv_compute);
                    else
                        m_reflection = spirv::reflect_spirv(m_spirv_vertex); // vertex contains most info
                    
                    m_compiled = true;
                    emit_signal("compiled", true);
                    return true;
                }

                bool is_compiled() const { return m_compiled; }
                godot::String get_compile_error() const { return godot::String(m_compile_error.c_str()); }
                
                godot::Dictionary get_reflection() const
                {
                    godot::Dictionary dict;
                    if (!m_compiled) return dict;
                    
                    godot::Array uniforms;
                    for (const auto& u : m_reflection.uniforms)
                    {
                        godot::Dictionary u_dict;
                        u_dict["name"] = godot::String(u.name.c_str());
                        u_dict["binding"] = u.binding;
                        u_dict["set"] = u.set;
                        u_dict["type"] = static_cast<int>(u.type);
                        u_dict["size"] = static_cast<int64_t>(u.size_bytes);
                        uniforms.append(u_dict);
                    }
                    dict["uniforms"] = uniforms;
                    dict["local_size"] = godot::Vector3i(
                        static_cast<int>(m_reflection.local_size_x),
                        static_cast<int>(m_reflection.local_size_y),
                        static_cast<int>(m_reflection.local_size_z)
                    );
                    return dict;
                }

                void set_parameter(const godot::String& name, const godot::Variant& value)
                {
                    std::string n = name.utf8().get_data();
                    m_parameters[n] = value;
                }

                godot::Variant get_parameter(const godot::String& name) const
                {
                    std::string n = name.utf8().get_data();
                    auto it = m_parameters.find(n);
                    if (it != m_parameters.end())
                        return it->second;
                    return godot::Variant();
                }

                void set_texture(const godot::String& name, const godot::Ref<godot::Texture2D>& texture)
                {
                    std::string n = name.utf8().get_data();
                    m_textures[n] = texture;
                }

                void set_buffer(const godot::String& name, const godot::RID& buffer_rid)
                {
                    std::string n = name.utf8().get_data();
                    m_buffers[n] = buffer_rid;
                }

                void set_parameters_tensor(const godot::Ref<XTensorNode>& tensor)
                {
                    m_parameters_tensor = tensor;
                }

                godot::Ref<XTensorNode> get_parameters_tensor() const
                {
                    return m_parameters_tensor;
                }

                void dispatch_compute(int work_groups_x, int work_groups_y, int work_groups_z)
                {
                    if (!m_compiled || m_shader_type != TYPE_COMPUTE || !m_compute_pipeline_rid.is_valid())
                    {
                        godot::UtilityFunctions::printerr("XShader: not compiled as compute shader");
                        return;
                    }
                    
                    godot::RenderingDevice* rd = godot::RenderingServer::get_singleton()->get_rendering_device();
                    if (!rd) return;
                    
                    // Build uniform sets from parameters
                    godot::Array uniform_sets = build_uniform_sets(rd);
                    
                    // Dispatch
                    rd->draw_list_begin(rd->screen_get_framebuffer_format());
                    rd->draw_list_bind_compute_pipeline(m_compute_pipeline_rid);
                    for (int i = 0; i < uniform_sets.size(); ++i)
                    {
                        godot::RID set_rid = uniform_sets[i];
                        if (set_rid.is_valid())
                            rd->draw_list_bind_uniform_set(set_rid, i);
                    }
                    rd->draw_list_dispatch(work_groups_x, work_groups_y, work_groups_z);
                    rd->draw_list_end();
                }

                godot::Ref<godot::ShaderMaterial> create_material()
                {
                    godot::Ref<godot::ShaderMaterial> mat;
                    mat.instantiate();
                    if (m_compiled && m_rd_shader.is_valid())
                    {
                        // Convert RDShaderFile to standard Shader (Godot 4 requires this)
                        // Placeholder: set shader on material
                    }
                    return mat;
                }

                godot::Ref<godot::RDShaderFile> get_rd_shader() const
                {
                    return m_rd_shader;
                }

                godot::RID get_shader_rid() const { return m_shader_rid; }

                static godot::Ref<XShader> load_from_file(const godot::String& path)
                {
                    godot::Ref<godot::FileAccess> f = godot::FileAccess::open(path, godot::FileAccess::READ);
                    if (!f.is_valid())
                    {
                        godot::UtilityFunctions::printerr("XShader: cannot open file", path);
                        return godot::Ref<XShader>();
                    }
                    std::string code = f->get_as_text().utf8().get_data();
                    
                    godot::Ref<XShader> shader;
                    shader.instantiate();
                    shader->set_code(godot::String(code.c_str()));
                    
                    // Detect shader type from extension or content
                    if (path.ends_with(".comp") || code.find("#pragma compute") != std::string::npos)
                        shader->set_shader_type(TYPE_COMPUTE);
                    
                    shader->compile();
                    return shader;
                }

                static godot::Ref<XShader> create_compute(const godot::String& code, const godot::String& entry_point)
                {
                    godot::Ref<XShader> shader;
                    shader.instantiate();
                    shader->set_code(code);
                    shader->set_shader_type(TYPE_COMPUTE);
                    shader->compile();
                    return shader;
                }

            private:
                std::map<std::string, godot::Variant> m_parameters;

                void cleanup()
                {
                    godot::RenderingDevice* rd = godot::RenderingServer::get_singleton()->get_rendering_device();
                    if (rd)
                    {
                        if (m_shader_rid.is_valid())
                            rd->free_rid(m_shader_rid);
                        if (m_compute_pipeline_rid.is_valid())
                            rd->free_rid(m_compute_pipeline_rid);
                        for (auto& p : m_uniform_sets)
                            if (p.second.is_valid())
                                rd->free_rid(p.second);
                    }
                    m_shader_rid = godot::RID();
                    m_compute_pipeline_rid = godot::RID();
                    m_uniform_sets.clear();
                }

                std::map<int, godot::RID> m_uniform_sets;

                void parse_stages(const std::string& source, std::string& vertex, std::string& fragment, std::string& compute)
                {
                    // Simple parser: look for #pragma shader_stage
                    std::regex stage_regex(R"(#pragma\s+shader_stage\s*\(\s*(\w+)\s*\))");
                    std::smatch match;
                    std::string::const_iterator search_start(source.cbegin());
                    size_t last_pos = 0;
                    std::string current_stage;
                    while (std::regex_search(search_start, source.cend(), match, stage_regex))
                    {
                        size_t pos = match.position();
                        if (!current_stage.empty())
                        {
                            std::string stage_code = source.substr(last_pos, pos - last_pos);
                            if (current_stage == "vertex") vertex = stage_code;
                            else if (current_stage == "fragment") fragment = stage_code;
                            else if (current_stage == "compute") compute = stage_code;
                        }
                        current_stage = match[1].str();
                        last_pos = pos + match.length();
                        search_start = match.suffix().first;
                    }
                    if (!current_stage.empty())
                    {
                        std::string stage_code = source.substr(last_pos);
                        if (current_stage == "vertex") vertex = stage_code;
                        else if (current_stage == "fragment") fragment = stage_code;
                        else if (current_stage == "compute") compute = stage_code;
                    }
                }

                std::string extract_stage(const std::string& source, const std::string& stage_name)
                {
                    // If no pragmas, return whole source (assume single stage)
                    return source;
                }

                godot::Array build_uniform_sets(godot::RenderingDevice* rd)
                {
                    godot::Array sets;
                    // Group uniforms by set index
                    std::map<uint32_t, std::vector<ShaderUniform>> set_map;
                    for (const auto& u : m_reflection.uniforms)
                        set_map[u.set].push_back(u);
                    
                    for (const auto& p : set_map)
                    {
                        uint32_t set_idx = p.first;
                        const auto& uniforms = p.second;
                        
                        // Create uniform buffer if needed
                        xarray_container<float> ub_data = ShaderParameterBlock::create_uniform_buffer(m_reflection, m_parameters);
                        godot::PackedByteArray ub_bytes;
                        ub_bytes.resize(static_cast<int>(ub_data.size() * sizeof(float)));
                        std::memcpy(ub_bytes.ptrw(), ub_data.data(), ub_bytes.size());
                        
                        godot::RID ub_rid = rd->uniform_buffer_create(ub_bytes.size(), ub_bytes);
                        
                        // Create uniform set
                        std::vector<godot::RID> bindings;
                        for (const auto& u : uniforms)
                        {
                            if (u.type == ShaderUniform::TEXTURE || u.type == ShaderUniform::SAMPLER)
                            {
                                auto tex_it = m_textures.find(u.name);
                                if (tex_it != m_textures.end() && tex_it->second.is_valid())
                                {
                                    godot::RID tex_rid = tex_it->second->get_rid();
                                    godot::RID sampler_rid = rd->sampler_create(godot::RenderingDevice::SAMPLER_FILTER_LINEAR,
                                                                                godot::RenderingDevice::SAMPLER_REPEAT_MODE_REPEAT,
                                                                                godot::RenderingDevice::SAMPLER_REPEAT_MODE_REPEAT);
                                    bindings.push_back(tex_rid);
                                    bindings.push_back(sampler_rid);
                                }
                            }
                            else if (u.type == ShaderUniform::BUFFER)
                            {
                                auto buf_it = m_buffers.find(u.name);
                                if (buf_it != m_buffers.end())
                                    bindings.push_back(buf_it->second);
                            }
                            else
                            {
                                bindings.push_back(ub_rid);
                            }
                        }
                        
                        godot::RID set_rid = rd->uniform_set_create(bindings, m_shader_rid, set_idx);
                        if (set_rid.is_valid())
                        {
                            m_uniform_sets[set_idx] = set_rid;
                            sets.append(set_rid);
                        }
                    }
                    return sets;
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // XMaterial - Tensor-based material
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XMaterial : public godot::Material
            {
                GDCLASS(XMaterial, godot::Material)

            private:
                godot::Ref<XShader> m_shader;
                godot::Ref<XTensorNode> m_parameters_tensor;
                godot::Dictionary m_texture_params;
                bool m_dirty = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_shader", "shader"), &XMaterial::set_shader);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_shader"), &XMaterial::get_shader);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_parameter", "name", "value"), &XMaterial::set_parameter);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_parameter", "name"), &XMaterial::get_parameter);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_parameters_tensor", "tensor"), &XMaterial::set_parameters_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_parameters_tensor"), &XMaterial::get_parameters_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_texture", "name", "texture"), &XMaterial::set_texture);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_texture", "name"), &XMaterial::get_texture);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "shader", godot::PROPERTY_HINT_RESOURCE_TYPE, "XShader"), "set_shader", "get_shader");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "parameters_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_parameters_tensor", "get_parameters_tensor");
                }

            public:
                void set_shader(const godot::Ref<XShader>& shader)
                {
                    m_shader = shader;
                    m_dirty = true;
                }
                
                godot::Ref<XShader> get_shader() const { return m_shader; }
                
                void set_parameter(const godot::String& name, const godot::Variant& value)
                {
                    m_parameters[name] = value;
                    m_dirty = true;
                }
                
                godot::Variant get_parameter(const godot::String& name) const
                {
                    auto it = m_parameters.find(name);
                    if (it != m_parameters.end()) return it->second;
                    return godot::Variant();
                }
                
                void set_parameters_tensor(const godot::Ref<XTensorNode>& tensor)
                {
                    m_parameters_tensor = tensor;
                    m_dirty = true;
                }
                
                godot::Ref<XTensorNode> get_parameters_tensor() const { return m_parameters_tensor; }
                
                void set_texture(const godot::String& name, const godot::Ref<godot::Texture2D>& texture)
                {
                    m_texture_params[name] = texture;
                }
                
                godot::Ref<godot::Texture2D> get_texture(const godot::String& name) const
                {
                    auto it = m_texture_params.find(name);
                    if (it != m_texture_params.end())
                        return it->second;
                    return godot::Ref<godot::Texture2D>();
                }
                
                virtual godot::RID _get_shader_rid() const override
                {
                    if (m_shader.is_valid())
                        return m_shader->get_shader_rid();
                    return godot::RID();
                }

            private:
                std::map<godot::String, godot::Variant> m_parameters;
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XShaderRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XShader>();
                    godot::ClassDB::register_class<XMaterial>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::ShaderType;
        using godot_bridge::ShadingLanguage;
        using godot_bridge::ShaderUniform;
        using godot_bridge::ShaderAttribute;
        using godot_bridge::ShaderReflection;
        using godot_bridge::ShaderParameterBlock;
        using godot_bridge::XShader;
        using godot_bridge::XMaterial;
        using godot_bridge::XShaderRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XSHADER_HPP

// godot/xshader.hpp