// Name : lighting enhancement updated
// File : variant_texture_pass.h 82 of 63
// Description : Host-side controller for variant_texture.glsl compute shader.
//               Generates procedural variant textures (solid, checker, noise, random,
//               missing texture, blend) to replace missing textures or add variation.
#pragma once

#include "servers/rendering_server.h"
#include "core/math/color.h"

// ----------------------------------------------------------------------------
// Variant modes (matches GLSL enum)
// ----------------------------------------------------------------------------
enum VariantMode {
    VARIANT_SOLID = 0,
    VARIANT_CHECKERBOARD = 1,
    VARIANT_PERLIN_NOISE = 2,
    VARIANT_RANDOM = 3,
    VARIANT_MISSING_TEXTURE = 4,
    VARIANT_BLEND = 5
};

// ----------------------------------------------------------------------------
// Parameters for variant texture generation
// ----------------------------------------------------------------------------
struct VariantTextureParams {
    VariantMode mode = VARIANT_SOLID;
    Color color1 = Color(1, 1, 1, 1);
    Color color2 = Color(0, 0, 0, 1);
    float intensity = 0.5f;      // blend factor (0 = only input, 1 = only variant)
    float time = 0.0f;           // for animated noise
    float scale = 8.0f;          // pattern scale (checker frequency, noise tiling)
    int seed = 42;
    bool has_input = false;      // set automatically if input texture is valid
};

// ----------------------------------------------------------------------------
// Variant texture compute pass
// ----------------------------------------------------------------------------
class VariantTexturePass {
public:
    VariantTexturePass();
    ~VariantTexturePass();

    // Set generation parameters
    void set_params(const VariantTextureParams &p_params);

    // Generate a variant texture (or blend with input).
    // If p_input_texture is valid, it will be used as base; otherwise a new texture is created.
    // The output texture will be created or resized to (width, height) and filled.
    void execute(const RID &p_input_texture, RID &out_texture, int width, int height);

private:
    VariantTextureParams m_params;
    RID m_compute_shader;
    RID m_uniform_buffer;
    RID m_temp_output;

    void _create_shader();
    void _update_uniforms(int width, int height);
    void _bind_textures(const RID &p_input, const RID &p_output);
    void _dispatch(int width, int height);
};