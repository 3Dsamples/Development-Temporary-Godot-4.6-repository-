// include/xtu/godot/xglslang.hpp
// xtensor-unified - GLSL to SPIR-V Compiler for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XGLSLANG_HPP
#define XTU_GODOT_XGLSLANG_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"

#ifdef XTU_USE_GLSLANG
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <SPIRV/GlslangToSpv.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace glslang {

// #############################################################################
// Forward declarations
// #############################################################################
class GLSLangCompiler;
class SPIRVCompiler;
class ShaderCache;

// #############################################################################
// Shader stage types
// #############################################################################
enum class ShaderStage : uint8_t {
    STAGE_VERTEX = 0,
    STAGE_FRAGMENT = 1,
    STAGE_COMPUTE = 2,
    STAGE_TESSELLATION_CONTROL = 3,
    STAGE_TESSELLATION_EVALUATION = 4,
    STAGE_GEOMETRY = 5,
    STAGE_TASK = 6,
    STAGE_MESH = 7,
    STAGE_RAY_GEN = 8,
    STAGE_RAY_ANY_HIT = 9,
    STAGE_RAY_CLOSEST_HIT = 10,
    STAGE_RAY_MISS = 11,
    STAGE_RAY_INTERSECTION = 12,
    STAGE_CALLABLE = 13
};

// #############################################################################
// Shader source type
// #############################################################################
enum class ShaderSourceType : uint8_t {
    SOURCE_GLSL = 0,
    SOURCE_HLSL = 1
};

// #############################################################################
// Optimization level
// #############################################################################
enum class OptimizationLevel : uint8_t {
    OPT_NONE = 0,
    OPT_SIZE = 1,
    OPT_PERFORMANCE = 2
};

// #############################################################################
// Compilation options
// #############################################################################
struct CompileOptions {
    ShaderSourceType source_type = ShaderSourceType::SOURCE_GLSL;
    OptimizationLevel optimization = OptimizationLevel::OPT_PERFORMANCE;
    bool debug_info = false;
    bool strip_debug_info = true;
    bool auto_map_locations = true;
    bool auto_map_bindings = true;
    bool invert_y = false;
    bool use_ubo_binding = true;
    std::vector<std::string> defines;
    std::vector<std::string> include_paths;
    int vulkan_version = VK_API_VERSION_1_2;
    bool separate_shader_objects = true;
    bool generate_remapped_info = false;
};

// #############################################################################
// Compilation result
// #############################################################################
struct CompileResult {
    std::vector<uint32_t> spirv;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    std::map<std::string, int> uniform_bindings;
    std::map<std::string, int> texture_bindings;
    std::map<std::string, int> input_locations;
    std::map<std::string, int> output_locations;
    bool success = false;
};

// #############################################################################
// GLSLangCompiler - Main shader compiler
// #############################################################################
class GLSLangCompiler : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(GLSLangCompiler, RefCounted)

private:
    static GLSLangCompiler* s_singleton;
    bool m_initialized = false;
    mutable std::mutex m_mutex;
    std::unordered_map<uint64_t, CompileResult> m_cache;
    bool m_cache_enabled = true;
    size_t m_max_cache_size = 256;

public:
    static GLSLangCompiler* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("GLSLangCompiler"); }

    GLSLangCompiler() { s_singleton = this; }
    ~GLSLangCompiler() { shutdown(); s_singleton = nullptr; }

    bool initialize() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

#ifdef XTU_USE_GLSLANG
        return glslang::InitializeProcess();
#else
        return false;
#endif
        m_initialized = true;
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_GLSLANG
        if (m_initialized) {
            glslang::FinalizeProcess();
        }
#endif
        m_initialized = false;
        m_cache.clear();
    }

    bool is_initialized() const { return m_initialized; }

    void set_cache_enabled(bool enabled) { m_cache_enabled = enabled; }
    bool is_cache_enabled() const { return m_cache_enabled; }

    void clear_cache() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
    }

    // #########################################################################
    // Main compilation entry
    // #########################################################################
    CompileResult compile(const String& source, ShaderStage stage,
                          const CompileOptions& options = {}) {
        uint64_t hash = compute_hash(source, stage, options);

        if (m_cache_enabled) {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_cache.find(hash);
            if (it != m_cache.end()) {
                return it->second;
            }
        }

        CompileResult result;
#ifdef XTU_USE_GLSLANG
        EShLanguage lang = stage_to_eshlang(stage);
        result = compile_glsl(source.to_std_string(), lang, options);
#else
        result.errors.push_back("GLSLang support not compiled");
#endif

        if (m_cache_enabled && result.success) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_cache.size() >= m_max_cache_size) {
                m_cache.erase(m_cache.begin());
            }
            m_cache[hash] = result;
        }

        return result;
    }

    // #########################################################################
    // Compile from file
    // #########################################################################
    CompileResult compile_file(const String& path, ShaderStage stage,
                               const CompileOptions& options = {}) {
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) {
            CompileResult result;
            result.errors.push_back("Failed to open file: " + path.to_std_string());
            return result;
        }

        String source = file->get_as_text();
        return compile(source, stage, options);
    }

    // #########################################################################
    // Preprocess only (no compilation)
    // #########################################################################
    String preprocess(const String& source, ShaderStage stage,
                      const CompileOptions& options = {}) {
#ifdef XTU_USE_GLSLANG
        EShLanguage lang = stage_to_eshlang(stage);
        std::string processed = preprocess_glsl(source.to_std_string(), lang, options);
        return String(processed.c_str());
#else
        return source;
#endif
    }

    // #########################################################################
    // Validate SPIR-V binary
    // #########################################################################
    bool validate_spirv(const std::vector<uint32_t>& spirv) {
#ifdef XTU_USE_GLSLANG
        spv_const_binary_t binary = {spirv.data(), spirv.size()};
        spv_diagnostic diagnostic = nullptr;
        spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_2);
        spv_result_t result = spvValidate(context, &binary, &diagnostic);
        spvContextDestroy(context);
        if (diagnostic) spvDiagnosticDestroy(diagnostic);
        return result == SPV_SUCCESS;
#else
        return false;
#endif
    }

    // #########################################################################
    // Get SPIR-V disassembly (for debugging)
    // #########################################################################
    String disassemble_spirv(const std::vector<uint32_t>& spirv) {
#ifdef XTU_USE_GLSLANG
        spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_2);
        spv_text text = nullptr;
        spv_diagnostic diagnostic = nullptr;
        spvBinaryToText(context, spirv.data(), spirv.size(),
                        SPV_BINARY_TO_TEXT_OPTION_INDENT, &text, &diagnostic);
        spvContextDestroy(context);

        if (text) {
            String result(text->str, text->length);
            spvTextDestroy(text);
            return result;
        }
#endif
        return String();
    }

private:
    uint64_t compute_hash(const String& source, ShaderStage stage, const CompileOptions& options) {
        uint64_t h = std::hash<std::string>{}(source.to_std_string());
        h = hash_combine(h, static_cast<uint32_t>(stage));
        h = hash_combine(h, static_cast<uint32_t>(options.source_type));
        h = hash_combine(h, static_cast<uint32_t>(options.optimization));
        h = hash_combine(h, options.debug_info ? 1 : 0);
        for (const auto& def : options.defines) {
            h = hash_combine(h, std::hash<std::string>{}(def));
        }
        return h;
    }

    uint64_t hash_combine(uint64_t seed, uint64_t v) const {
        return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

#ifdef XTU_USE_GLSLANG
    EShLanguage stage_to_eshlang(ShaderStage stage) {
        switch (stage) {
            case ShaderStage::STAGE_VERTEX: return EShLangVertex;
            case ShaderStage::STAGE_FRAGMENT: return EShLangFragment;
            case ShaderStage::STAGE_COMPUTE: return EShLangCompute;
            case ShaderStage::STAGE_TESSELLATION_CONTROL: return EShLangTessControl;
            case ShaderStage::STAGE_TESSELLATION_EVALUATION: return EShLangTessEvaluation;
            case ShaderStage::STAGE_GEOMETRY: return EShLangGeometry;
            case ShaderStage::STAGE_TASK: return EShLangTaskNV;
            case ShaderStage::STAGE_MESH: return EShLangMeshNV;
            case ShaderStage::STAGE_RAY_GEN: return EShLangRayGen;
            case ShaderStage::STAGE_RAY_ANY_HIT: return EShLangAnyHit;
            case ShaderStage::STAGE_RAY_CLOSEST_HIT: return EShLangClosestHit;
            case ShaderStage::STAGE_RAY_MISS: return EShLangMiss;
            case ShaderStage::STAGE_RAY_INTERSECTION: return EShLangIntersect;
            case ShaderStage::STAGE_CALLABLE: return EShLangCallable;
            default: return EShLangVertex;
        }
    }

    CompileResult compile_glsl(const std::string& source, EShLanguage stage,
                               const CompileOptions& options) {
        CompileResult result;

        glslang::TShader shader(stage);
        const char* src = source.c_str();
        shader.setStrings(&src, 1);

        int client_input_semantics_version = options.vulkan_version;
        glslang::EShTargetClientVersion client_version = glslang::EShTargetVulkan_1_2;
        glslang::EShTargetLanguageVersion target_version = glslang::EShTargetSpv_1_5;

        shader.setEnvInput(options.source_type == ShaderSourceType::SOURCE_GLSL ?
                           glslang::EShSourceGlsl : glslang::EShSourceHlsl,
                           stage, client_version, 450);
        shader.setEnvClient(glslang::EShClientVulkan, client_version);
        shader.setEnvTarget(glslang::EShTargetSpv, target_version);

        // Set defines
        for (const auto& def : options.defines) {
            shader.setPreamble(def.c_str());
        }

        // Parse
        EShMessages messages = EShMsgDefault;
        if (options.debug_info) messages = (EShMessages)(messages | EShMsgDebugInfo);
        if (options.source_type == ShaderSourceType::SOURCE_HLSL) {
            messages = (EShMessages)(messages | EShMsgReadHlsl);
        }

        if (!shader.parse(&glslang::DefaultTBuiltInResource, 450, false, messages)) {
            result.errors.push_back(shader.getInfoLog());
            result.errors.push_back(shader.getInfoDebugLog());
            return result;
        }

        glslang::TProgram program;
        program.addShader(&shader);

        if (!program.link(messages)) {
            result.errors.push_back(program.getInfoLog());
            result.errors.push_back(program.getInfoDebugLog());
            return result;
        }

        // Generate SPIR-V
        glslang::SpvOptions spvOptions;
        spvOptions.generateDebugInfo = options.debug_info;
        spvOptions.stripDebugInfo = options.strip_debug_info;
        spvOptions.disableOptimizer = (options.optimization == OptimizationLevel::OPT_NONE);
        spvOptions.optimizeSize = (options.optimization == OptimizationLevel::OPT_SIZE);

        std::vector<uint32_t> spirv;
        glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &spvOptions);

        if (spirv.empty()) {
            result.errors.push_back("SPIR-V generation failed");
            return result;
        }

        result.spirv = std::move(spirv);
        result.success = true;

        // Extract reflection info
        if (options.auto_map_bindings || options.auto_map_locations) {
            extract_reflection(program, stage, result);
        }

        return result;
    }

    std::string preprocess_glsl(const std::string& source, EShLanguage stage,
                                const CompileOptions& options) {
        glslang::TShader shader(stage);
        const char* src = source.c_str();
        shader.setStrings(&src, 1);
        shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 450);

        EShMessages messages = EShMsgDefault;
        if (options.source_type == ShaderSourceType::SOURCE_HLSL) {
            messages = (EShMessages)(messages | EShMsgReadHlsl);
        }
        messages = (EShMessages)(messages | EShMsgOnlyPreprocessor);

        if (!shader.preprocess(&glslang::DefaultTBuiltInResource, 450,
                               ENoProfile, false, false, messages, nullptr,
                               glslang::TShader::ForbidIncluder())) {
            return std::string();
        }

        return std::string(shader.getPreprocessedCode());
    }

    void extract_reflection(const glslang::TProgram& program, EShLanguage stage,
                            CompileResult& result) {
        const auto* intermediate = program.getIntermediate(stage);
        if (!intermediate) return;

        // Extract uniform bindings
        for (int i = 0; i < intermediate->getNumUniforms(); ++i) {
            const auto& uniform = intermediate->getUniform(i);
            result.uniform_bindings[uniform.name] = uniform.getBinding();
        }

        // Extract texture bindings
        for (int i = 0; i < intermediate->getNumUniformBlocks(); ++i) {
            const auto& block = intermediate->getUniformBlock(i);
            result.uniform_bindings[block.name] = block.getBinding();
        }

        // Extract input/output locations
        for (int i = 0; i < intermediate->getNumPipeInputs(); ++i) {
            const auto& input = intermediate->getPipeInput(i);
            result.input_locations[input.name] = input.getLocation();
        }
        for (int i = 0; i < intermediate->getNumPipeOutputs(); ++i) {
            const auto& output = intermediate->getPipeOutput(i);
            result.output_locations[output.name] = output.getLocation();
        }
    }
#endif
};

// #############################################################################
// SPIRVCompiler - Low-level SPIR-V manipulation
// #############################################################################
class SPIRVCompiler : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(SPIRVCompiler, RefCounted)

public:
    static StringName get_class_static() { return StringName("SPIRVCompiler"); }

    // Optimize SPIR-V binary
    std::vector<uint32_t> optimize(const std::vector<uint32_t>& spirv,
                                    OptimizationLevel level = OptimizationLevel::OPT_PERFORMANCE) {
#ifdef XTU_USE_GLSLANG
        spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_2);
        spv_optimizer optimizer = spvOptimizerCreate(SPV_ENV_VULKAN_1_2);

        if (level == OptimizationLevel::OPT_SIZE) {
            spvOptimizerRegisterSizePasses(optimizer);
        } else if (level == OptimizationLevel::OPT_PERFORMANCE) {
            spvOptimizerRegisterPerformancePasses(optimizer);
        }

        spv_binary input_binary = {spirv.data(), spirv.size()};
        spv_binary output_binary = nullptr;
        spv_result_t result = spvOptimizerRun(optimizer, input_binary, &output_binary, nullptr);

        std::vector<uint32_t> output;
        if (result == SPV_SUCCESS && output_binary) {
            output.assign(output_binary->code, output_binary->code + output_binary->wordCount);
            spvBinaryDestroy(output_binary);
        }

        spvOptimizerDestroy(optimizer);
        spvContextDestroy(context);
        return output.empty() ? spirv : output;
#else
        return spirv;
#endif
    }

    // Strip debug information
    std::vector<uint32_t> strip_debug(const std::vector<uint32_t>& spirv) {
#ifdef XTU_USE_GLSLANG
        spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_2);
        spv_optimizer optimizer = spvOptimizerCreate(SPV_ENV_VULKAN_1_2);
        spvOptimizerRegisterStripDebugInfoPass(optimizer);

        spv_binary input_binary = {spirv.data(), spirv.size()};
        spv_binary output_binary = nullptr;
        spv_result_t result = spvOptimizerRun(optimizer, input_binary, &output_binary, nullptr);

        std::vector<uint32_t> output;
        if (result == SPV_SUCCESS && output_binary) {
            output.assign(output_binary->code, output_binary->code + output_binary->wordCount);
            spvBinaryDestroy(output_binary);
        }

        spvOptimizerDestroy(optimizer);
        spvContextDestroy(context);
        return output.empty() ? spirv : output;
#else
        return spirv;
#endif
    }
};

} // namespace glslang

// Bring into main namespace
using glslang::GLSLangCompiler;
using glslang::SPIRVCompiler;
using glslang::ShaderStage;
using glslang::ShaderSourceType;
using glslang::OptimizationLevel;
using glslang::CompileOptions;
using glslang::CompileResult;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XGLSLANG_HPP