// Name : lighting enhancement updated
// File : texture_manager.cpp 85 of 63
// Description : Implementation of texture manager with full math for variant generation.
//               Uses VariantTexturePass to generate procedural textures for missing assets.
#include "texture_manager.h"
#include "core/io/resource_loader.h"
#include "scene/resources/texture.h"
#include "servers/rendering_server.h"
#include <unordered_map>

// ----------------------------------------------------------------------------
// Constructor / Destructor
// ----------------------------------------------------------------------------
TextureManager::TextureManager() {
    m_cached_missing_texture = RID();
    m_missing_width = m_missing_height = 0;
}

TextureManager::~TextureManager() {
    if (m_cached_missing_texture.is_valid()) {
        RenderingServer::get_singleton()->free(m_cached_missing_texture);
    }
}

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------
void TextureManager::set_config(const TextureManagerConfig &p_config) {
    m_config = p_config;
}

const TextureManagerConfig &TextureManager::get_config() const {
    return m_config;
}

// ----------------------------------------------------------------------------
// Generate variant texture using VariantTexturePass
// ----------------------------------------------------------------------------
RID TextureManager::generate_variant_texture(const RID &p_input_texture,
                                             const VariantTextureParams &p_params,
                                             int width, int height) {
    VariantTextureParams params = p_params;
    // Set has_input automatically based on whether input is valid
    params.has_input = p_input_texture.is_valid() && RenderingServer::get_singleton()->texture_get_width(p_input_texture) > 0;

    RID output;
    m_variant_pass.set_params(params);
    m_variant_pass.execute(p_input_texture, output, width, height);
    return output;
}

// ----------------------------------------------------------------------------
// Get or create cached missing texture (pink/black checker)
// ----------------------------------------------------------------------------
RID TextureManager::_get_or_create_missing_texture(int width, int height) {
    if (m_cached_missing_texture.is_valid() && m_missing_width == width && m_missing_height == height) {
        return m_cached_missing_texture;
    }

    // If cached exists but size differs, free it
    if (m_cached_missing_texture.is_valid()) {
        RenderingServer::get_singleton()->free(m_cached_missing_texture);
        m_cached_missing_texture = RID();
    }

    VariantTextureParams params;
    params.mode = VARIANT_MISSING_TEXTURE;
    params.color1 = Color(1, 0, 1, 1);
    params.color2 = Color(0, 0, 0, 1);
    params.scale = 8.0f;
    params.has_input = false;

    m_cached_missing_texture = generate_variant_texture(RID(), params, width, height);
    m_missing_width = width;
    m_missing_height = height;
    return m_cached_missing_texture;
}

RID TextureManager::get_missing_texture() {
    // Default size 512x512 for missing texture
    return _get_or_create_missing_texture(512, 512);
}

// ----------------------------------------------------------------------------
// Ensure that a material has a valid texture for a given parameter
// ----------------------------------------------------------------------------
void TextureManager::ensure_material_textures(const RID &p_material, const StringName &p_tex_param) {
    RenderingServer *rs = RenderingServer::get_singleton();
    RID current_tex = rs->material_get_param(p_material, p_tex_param);
    if (!current_tex.is_valid()) {
        // No texture set: assign missing texture
        RID missing = get_missing_texture();
        rs->material_set_param(p_material, p_tex_param, missing);
    } else {
        // Check if the texture is valid (non-zero dimensions)
        int w = rs->texture_get_width(current_tex);
        int h = rs->texture_get_height(current_tex);
        if (w == 0 || h == 0) {
            // Invalid texture, replace with missing
            RID missing = get_missing_texture();
            rs->material_set_param(p_material, p_tex_param, missing);
        }
    }
}

// ----------------------------------------------------------------------------
// Set material texture with automatic variant generation if texture is invalid
// ----------------------------------------------------------------------------
void TextureManager::set_material_texture(const RID &p_material, const StringName &p_param,
                                          const RID &p_texture, int width, int height) {
    RenderingServer *rs = RenderingServer::get_singleton();
    RID final_tex = p_texture;

    // Validate input texture
    bool tex_valid = false;
    if (p_texture.is_valid()) {
        int w = rs->texture_get_width(p_texture);
        int h = rs->texture_get_height(p_texture);
        tex_valid = (w > 0 && h > 0);
    }

    if (!tex_valid && m_config.auto_generate_missing) {
        // Generate a variant texture using default parameters
        VariantTextureParams params;
        params.mode = m_config.default_mode;
        params.color1 = m_config.default_color1;
        params.color2 = m_config.default_color2;
        params.scale = m_config.default_scale;
        params.seed = m_config.default_seed;
        params.has_input = false;
        final_tex = generate_variant_texture(RID(), params, width, height);
    } else if (!tex_valid) {
        // No auto-generation, set to null (or skip)
        final_tex = RID();
    }

    rs->material_set_param(p_material, p_param, final_tex);
}

// ----------------------------------------------------------------------------
// Cleanup: release unused cached textures (currently only missing texture is cached)
// ----------------------------------------------------------------------------
void TextureManager::cleanup_unused() {
    // For now, nothing to do; missing texture is kept permanently.
    // In a more advanced version, you could track usage counts.
}