// Name : lighting enhancement updated
// File : variant_texture_pass.cpp 83 of 63
// Description : Implementation of variant texture compute shader. Embeds the GLSL source
//               as a string (variant_texture.glsl) and compiles it. Provides full math
//               for pattern generation including perlin‑style noise, checkerboard, etc.
#include "variant_texture_pass.h"
#include "servers/rendering_server.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include <cstring>
#include <cmath>

// ----------------------------------------------------------------------------
// Embedded GLSL source for variant_texture.glsl (raw string literal)
// ----------------------------------------------------------------------------
static const char *variant_texture_glsl = R"(
#version 450

layout(set = 0, binding = 0) uniform sampler2D input_texture;
layout(set = 0, binding = 1) uniform VariantParams {
    int mode;
    vec4 color1;
    vec4 color2;
    float intensity;
    float time;
    float scale;
    int seed;
    int has_input;
} params;
layout(set = 0, binding = 2, rgba8) writeonly uniform image2D output_texture;

float random(vec2 st, int seed) {
    return fract(sin(dot(st * float(seed), vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 st, float scale) {
    vec2 i = floor(st * scale);
    vec2 f = fract(st * scale);
    float a = random(i, 1);
    float b = random(i + vec2(1.0, 0.0), 1);
    float c = random(i + vec2(0.0, 1.0), 1);
    float d = random(i + vec2(1.0, 1.0), 1);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_texture);
    if (pixel.x >= size.x || pixel.y >= size.y) return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    vec4 output_color;

    vec4 input_col = vec4(0.0);
    if (params.has_input == 1) {
        input_col = texture(input_texture, uv);
    }

    if (params.mode == 0) {
        output_color = params.color1;
    } else if (params.mode == 1) {
        float tile_scale = params.scale;
        vec2 grid = floor(uv * tile_scale);
        float pattern = mod(grid.x + grid.y, 2.0);
        output_color = mix(params.color1, params.color2, pattern);
    } else if (params.mode == 2) {
        float n = noise(uv, params.scale);
        n = clamp(n, 0.0, 1.0);
        output_color = mix(params.color1, params.color2, n);
    } else if (params.mode == 3) {
        float r = random(uv, params.seed);
        float g = random(uv + vec2(0.234, 0.567), params.seed + 1);
        float b = random(uv + vec2(0.789, 0.123), params.seed + 2);
        output_color = vec4(r, g, b, 1.0);
    } else if (params.mode == 4) {
        vec2 grid = floor(uv * 8.0);
        float pattern = mod(grid.x + grid.y, 2.0);
        output_color = mix(vec4(1.0, 0.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), pattern);
    } else if (params.mode == 5) {
        float tile_scale = params.scale;
        vec2 grid = floor(uv * tile_scale);
        float pattern = mod(grid.x + grid.y, 2.0);
        vec4 variant = mix(params.color1, params.color2, pattern);
        output_color = mix(input_col, variant, params.intensity);
    } else {
        output_color = input_col;
    }
    imageStore(output_texture, pixel, output_color);
}
)";

// ----------------------------------------------------------------------------
// Helper: compile compute shader from source string
// ----------------------------------------------------------------------------
static RID compile_compute_shader(const char *source) {
    RenderingServer *rs = RenderingServer::get_singleton();
    RID shader = rs->shader_create();
    rs->shader_set_source(shader, source, RenderingServer::SHADER_TYPE_COMPUTE);
    // In a real engine, you would check compilation errors.
    return shader;
}

// ----------------------------------------------------------------------------
// Constructor / Destructor
// ----------------------------------------------------------------------------
VariantTexturePass::VariantTexturePass() {
    RenderingServer *rs = RenderingServer::get_singleton();
    // Create compute shader from embedded source
    m_compute_shader = compile_compute_shader(variant_texture_glsl);

    // Create uniform buffer (layout matches GLSL)
    size_t uniform_size = sizeof(int)          // mode
                        + 4 * sizeof(float)   // color1
                        + 4 * sizeof(float)   // color2
                        + sizeof(float)       // intensity
                        + sizeof(float)       // time
                        + sizeof(float)       // scale
                        + sizeof(int)         // seed
                        + sizeof(int);        // has_input
    m_uniform_buffer = rs->uniform_buffer_create(uniform_size);

    // Create temporary output texture (dummy size)
    m_temp_output = rs->texture_2d_create();
    rs->texture_2d_initialize(m_temp_output, 1, 1, Image::FORMAT_RGBA8);
}

VariantTexturePass::~VariantTexturePass() {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (m_compute_shader.is_valid()) rs->free(m_compute_shader);
    if (m_uniform_buffer.is_valid()) rs->free(m_uniform_buffer);
    if (m_temp_output.is_valid()) rs->free(m_temp_output);
}

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------
void VariantTexturePass::set_params(const VariantTextureParams &p_params) {
    m_params = p_params;
}

// ----------------------------------------------------------------------------
// Update uniform buffer with current parameters (full math)
// ----------------------------------------------------------------------------
void VariantTexturePass::_update_uniforms(int width, int height) {
    struct Uniforms {
        int mode;
        float color1[4];
        float color2[4];
        float intensity;
        float time;
        float scale;
        int seed;
        int has_input;
        float _padding[2];
    } uniforms;
    memset(&uniforms, 0, sizeof(uniforms));

    uniforms.mode = (int)m_params.mode;
    uniforms.color1[0] = m_params.color1.r;
    uniforms.color1[1] = m_params.color1.g;
    uniforms.color1[2] = m_params.color1.b;
    uniforms.color1[3] = m_params.color1.a;
    uniforms.color2[0] = m_params.color2.r;
    uniforms.color2[1] = m_params.color2.g;
    uniforms.color2[2] = m_params.color2.b;
    uniforms.color2[3] = m_params.color2.a;
    uniforms.intensity = m_params.intensity;
    uniforms.time = m_params.time;
    uniforms.scale = m_params.scale;
    uniforms.seed = m_params.seed;
    uniforms.has_input = m_params.has_input ? 1 : 0;

    RenderingServer::get_singleton()->uniform_buffer_update(m_uniform_buffer, 0, sizeof(uniforms), &uniforms);
}

// ----------------------------------------------------------------------------
// Bind textures to shader binding points (matches GLSL)
// ----------------------------------------------------------------------------
void VariantTexturePass::_bind_textures(const RID &p_input, const RID &p_output) {
    RenderingServer *rs = RenderingServer::get_singleton();
    // binding 0: input texture (sampler2D)
    rs->material_set_texture(m_compute_shader, 0, p_input);
    // binding 1: uniform buffer
    rs->material_set_uniform_buffer(m_compute_shader, 1, m_uniform_buffer);
    // binding 2: output image (writeonly)
    rs->shader_set_image(m_compute_shader, 2, p_output);
}

// ----------------------------------------------------------------------------
// Dispatch compute shader (local workgroup 8x8)
// ----------------------------------------------------------------------------
void VariantTexturePass::_dispatch(int width, int height) {
    int groups_x = (width + 7) / 8;
    int groups_y = (height + 7) / 8;
    RenderingServer::get_singleton()->compute_shader_dispatch(m_compute_shader, groups_x, groups_y, 1);
}

// ----------------------------------------------------------------------------
// Main execution: create output texture and run the shader
// ----------------------------------------------------------------------------
void VariantTexturePass::execute(const RID &p_input_texture, RID &out_texture, int width, int height) {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (width <= 0 || height <= 0) return;

    // Determine if input texture is valid
    m_params.has_input = p_input_texture.is_valid() && rs->texture_get_width(p_input_texture) > 0;

    // Ensure output texture exists and has correct size
    if (!out_texture.is_valid()) {
        out_texture = rs->texture_2d_create();
        rs->texture_2d_initialize(out_texture, width, height, Image::FORMAT_RGBA8);
    } else {
        int w = rs->texture_get_width(out_texture);
        int h = rs->texture_get_height(out_texture);
        if (w != width || h != height) {
            rs->free(out_texture);
            out_texture = rs->texture_2d_create();
            rs->texture_2d_initialize(out_texture, width, height, Image::FORMAT_RGBA8);
        }
    }

    // Update uniforms (includes pattern math)
    _update_uniforms(width, height);

    // Bind input and output textures
    _bind_textures(p_input_texture, out_texture);

    // Dispatch compute shader
    _dispatch(width, height);

    // Ensure completion
    rs->sync();
}