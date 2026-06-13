// gpu_particles_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// GPUParticles3D – high‑performance GPU particle system with full lighting
// Supports dynamic emission, shadows, GI, collisions, and motion blur.
// ============================================================================

enum class ParticleDrawOrder {
    INDICES,
    VIEW_DEPTH,
    LIFETIME
};

enum class ParticleEmissionShape {
    POINT,
    SPHERE,
    BOX,
    ELLIPSOID,
    CONE,
    TORUS
};

struct ParticleParameters {
    // Emission
    int amount = 1000;
    float lifetime = 5.0f;
    float preprocess = 0.0f;
    float explosiveness = 0.0f;
    float randomness = 0.0f;
    ParticleEmissionShape emission_shape = ParticleEmissionShape::POINT;
    double emission_shape_extents[3] = {1.0, 1.0, 1.0};
    double direction[3] = {0.0, 1.0, 0.0};
    float spread = 0.0f;
    float flatness = 0.0f;

    // Initial velocity
    float initial_velocity_min = 0.0f;
    float initial_velocity_max = 0.0f;
    float angular_velocity_min = 0.0f;
    float angular_velocity_max = 0.0f;

    // Gravity & damping
    float gravity[3] = {0.0f, -9.8f, 0.0f};
    float damping = 0.0f;

    // Particle properties
    float size_min = 0.05f;
    float size_max = 0.05f;
    float size_curve = 0.0f;   // 0=constant, 1=curve
    float size_curve_scale = 1.0f;

    float rotation_min = 0.0f;
    float rotation_max = 0.0f;
    float rotation_speed_min = 0.0f;
    float rotation_speed_max = 0.0f;

    // Color & material
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f}; // RGBA
    float color_ramp_texture = 0;   // texture ID (placeholder)
    char material_path[256] = {0};

    // Lighting & shadows
    bool cast_shadow = true;
    bool receive_shadow = false;
    bool use_gi = true;      // global illumination
    bool use_emission = false;
    float emission_energy = 1.0f;
};

class GPUParticles3D : public GeometryInstance3D {
public:
    GPUParticles3D();
    ~GPUParticles3D();

    // ------------------------------------------------------------------------
    // Particle system control
    // ------------------------------------------------------------------------
    void set_emitting(bool emitting);
    bool is_emitting() const;
    void restart();
    void set_one_shot(bool one_shot);
    bool is_one_shot() const;
    void set_lifetime(float seconds);
    float get_lifetime() const;

    // ------------------------------------------------------------------------
    // Parameter access
    // ------------------------------------------------------------------------
    void set_parameters(const ParticleParameters& params);
    const ParticleParameters& get_parameters() const;

    // ------------------------------------------------------------------------
    // Draw order & trail settings
    // ------------------------------------------------------------------------
    void set_draw_order(ParticleDrawOrder order);
    ParticleDrawOrder get_draw_order() const;
    void set_trail_enabled(bool enabled);
    bool is_trail_enabled() const;
    void set_trail_length(float seconds);
    float get_trail_length() const;

    // ------------------------------------------------------------------------
    // Collision & interaction
    // ------------------------------------------------------------------------
    void set_collision_enabled(bool enabled);
    bool is_collision_enabled() const;
    void set_collision_radius(float radius);
    float get_collision_radius() const;
    void set_collision_mask(uint32_t mask);
    uint32_t get_collision_mask() const;

    // ------------------------------------------------------------------------
    // Sub‑emitters (particles that spawn particles)
    // ------------------------------------------------------------------------
    void set_sub_emitter(int64_t particle_instance_id, float at_end);
    void clear_sub_emitter();

    // ------------------------------------------------------------------------
    // GPU buffers & synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void update_particle_buffers();   // upload any changed parameters

    // ------------------------------------------------------------------------
    // Shader uniforms (for custom particle shaders)
    // ------------------------------------------------------------------------
    void set_shader_uniform(const char* name, const float* value, int size);
    void clear_shader_uniform(const char* name);

protected:
    void _update_bounding_volume() override; // compute from active particles

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting