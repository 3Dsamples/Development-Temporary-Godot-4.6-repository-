// Name : lighting enhancement updated
// File : variant_texture.h 80 of 63
// Description : Host-side controller for variant_texture.glsl compute shader.
//               Generates procedural textures (solid, checker, noise, random, missing)
//               and blends with input textures. Resolves missing texture fallbacks.
#pragma once

#include "servers/rendering_server.h"
#include "core/math/color.h"
#include "core/math/vector2.h"
#include "core/math/math_funcs.h"

// ----------------------------------------------------------------------------
// Variant generation mode (matches GLSL)
// ----------------------------------------------------------------------------
enum VariantMode {
    VARIANT_SOLID = 0,
    VARIANT_CHECKER = 1,
    VARIANT_NOISE = 2,
    VARIANT_RANDOM = 3,
    VARIANT_MISSING = 4,
    VARIANT_BLEND = 5
};

// ----------------------------------------------------------------------------
// Parameters for variant texture generation
// ----------------------------------------------------------------------------
struct VariantParams {
    VariantMode mode = VARIANT_SOLID;
    Color color1 = Color(1, 1, 1, 1);
    Color color2 = Color(0, 0, 0, 1);
    float intensity = 1.0f;      // blend factor for VARIANT_BLEND
    float time = 0.0f;           // for animated noise (time uniform)
    float scale = 8.0f;          // pattern scale (tiles per unit)
    int seed = 42;               // random seed
    bool has_input = false;      // whether input texture is provided
};

// ----------------------------------------------------------------------------
// Variant texture generator
// ----------------------------------------------------------------------------
class VariantTextureGenerator {
public:
    VariantTextureGenerator();
    ~VariantTextureGenerator();

    // Set the parameters for the next generation
    void set_params(const VariantParams &p_params);

    // Set the input texture (optional, for blending or missing fallback)
    void set_input_texture(const RID &p_input);

    // Generate a texture of given dimensions. Returns RID of generated texture.
    // If output_texture is provided (non-null), it will be updated; otherwise a new texture is created.
    RID generate(int p_width, int p_height, RID *p_output_texture = nullptr);

    // Directly apply to an existing texture (overwrites content)
    void apply_to_texture(const RID &p_target_texture);

private:
    VariantParams m_params;
    RID m_input_texture;
    RID m_compute_shader;
    RID m_uniform_buffer;
    RID m_fallback_output; // used when no output texture provided

    void _update_uniforms(int p_width, int p_height);
    void _dispatch(RID p_target, int p_width, int p_height);
};