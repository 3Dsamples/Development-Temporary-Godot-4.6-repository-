// Name : lighting enhancement updated
// File : texture_manager.h 84 of 63
// Description : Manages texture resources, automatically generates variant textures
//               for missing ones using VariantTexturePass, and applies them to materials.
#pragma once

#include "servers/rendering_server.h"
#include "core/math/color.h"
#include "variant_texture_pass.h"  // includes VariantTexturePass and parameters
#include "scene/resources/material.h"

// ----------------------------------------------------------------------------
// Texture manager configuration
// ----------------------------------------------------------------------------
struct TextureManagerConfig {
    bool auto_generate_missing = true;          // auto-create variant texture for missing textures
    VariantMode default_mode = VARIANT_MISSING_TEXTURE;
    Color default_color1 = Color(1, 0, 1, 1);   // magenta
    Color default_color2 = Color(0, 0, 0, 1);
    float default_scale = 8.0f;
    int default_seed = 42;
};

// ----------------------------------------------------------------------------
// Main texture manager class
// ----------------------------------------------------------------------------
class TextureManager {
public:
    TextureManager();
    ~TextureManager();

    // Configure the manager
    void set_config(const TextureManagerConfig &p_config);
    const TextureManagerConfig &get_config() const;

    // Generate a variant texture from parameters (or using defaults)
    // If input_texture is valid, it will be blended; otherwise created from scratch.
    // Returns the RID of the generated texture (may be a new texture or the input if no change).
    RID generate_variant_texture(const RID &p_input_texture,
                                 const VariantTextureParams &p_params,
                                 int width, int height);

    // Get a default missing texture (cached, reusable)
    RID get_missing_texture();

    // Ensure that a material has all its texture parameters filled (auto-generate if missing)
    void ensure_material_textures(const RID &p_material, const StringName &p_tex_param);

    // Set a texture on a material, optionally generating a variant if the texture is invalid
    void set_material_texture(const RID &p_material, const StringName &p_param,
                              const RID &p_texture, int width = 512, int height = 512);

    // Cleanup unused textures (call periodically)
    void cleanup_unused();

private:
    TextureManagerConfig m_config;
    VariantTexturePass m_variant_pass;
    RID m_cached_missing_texture;
    int m_missing_width, m_missing_height;

    // Internal: generate or retrieve cached missing texture
    RID _get_or_create_missing_texture(int width, int height);
};