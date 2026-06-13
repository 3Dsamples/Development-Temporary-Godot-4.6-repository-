// Name : lighting enhancement updated
// File : variant_texture.glsl
// Description : Compute shader that generates a procedural variant texture.
//               Can be used as a fallback for missing textures or to add
//               variation to existing textures (e.g., noise overlay, color tint,
//               checkerboard, or random patterns). Supports blending with an
//               optional input texture.
// ----------------------------------------------------------------------------
// Inputs:
//   binding 0 : sampler2D input_texture (can be null/black if missing)
//   binding 1 : uniform parameters
//   binding 2 : writeonly image2D output_texture
// ----------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform sampler2D input_texture;
layout(set = 0, binding = 1) uniform VariantParams {
    int mode;              // 0 = solid color, 1 = checkerboard, 2 = perlin noise, 3 = random color, 4 = missing texture, 5 = blend with input
    vec4 color1;           // primary color (RGBA)
    vec4 color2;           // secondary color (checkerboard / noise threshold)
    float intensity;       // blend factor (0 = only input, 1 = only variant)
    float time;            // for animated noise
    float scale;           // pattern scale
    int seed;              // random seed
    int has_input;         // 1 if input_texture is valid
} params;

layout(set = 0, binding = 2, rgba8) writeonly uniform image2D output_texture;

// ----------------------------------------------------------------------------
// Helper: pseudo-random function
// ----------------------------------------------------------------------------
float random(vec2 st, int seed) {
    return fract(sin(dot(st * float(seed), vec2(12.9898, 78.233))) * 43758.5453123);
}

// ----------------------------------------------------------------------------
// Simple value noise (2D)
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Main compute shader
// ----------------------------------------------------------------------------
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_texture);
    if (pixel.x >= size.x || pixel.y >= size.y) return;

    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    vec4 output_color;

    // Sample input texture if available and mode requires it
    vec4 input_col = vec4(0.0);
    if (params.has_input == 1) {
        input_col = texture(input_texture, uv);
    }

    if (params.mode == 0) {
        // Solid color
        output_color = params.color1;
    } else if (params.mode == 1) {
        // Checkerboard
        float tile_scale = params.scale;
        vec2 grid = floor(uv * tile_scale);
        float pattern = mod(grid.x + grid.y, 2.0);
        output_color = mix(params.color1, params.color2, pattern);
    } else if (params.mode == 2) {
        // Value noise
        float n = noise(uv, params.scale);
        n = clamp(n, 0.0, 1.0);
        output_color = mix(params.color1, params.color2, n);
    } else if (params.mode == 3) {
        // Random color per pixel (static)
        float r = random(uv, params.seed);
        float g = random(uv + vec2(0.234, 0.567), params.seed + 1);
        float b = random(uv + vec2(0.789, 0.123), params.seed + 2);
        output_color = vec4(r, g, b, 1.0);
    } else if (params.mode == 4) {
        // "Missing texture" pattern: magenta/black checker
        vec2 grid = floor(uv * 8.0);
        float pattern = mod(grid.x + grid.y, 2.0);
        output_color = mix(vec4(1.0, 0.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), pattern);
    } else if (params.mode == 5) {
        // Blend input with variant
        vec4 variant;
        // Use a checkerboard pattern as variant example
        float tile_scale = params.scale;
        vec2 grid = floor(uv * tile_scale);
        float pattern = mod(grid.x + grid.y, 2.0);
        variant = mix(params.color1, params.color2, pattern);
        output_color = mix(input_col, variant, params.intensity);
    } else {
        // Fallback: input texture or black
        output_color = input_col;
    }

    imageStore(output_texture, pixel, output_color);
}