// gpu_particles_collision_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// GPUParticlesCollision3D – collision primitive (sphere, box, SDF, heightfield)
// that influences GPU particles. Supports bounce, friction, and attenuation.
// Integrated with lighting: can be visible (debug), cast shadows, affect GI
// via particle interaction (optional). Optimized for many collisions using
// BVH and GPU textures for heightfield and SDF.
// ============================================================================

enum class CollisionShape3DType : uint8_t {
    SPHERE,
    BOX,
    SDF,            // signed distance field (3D texture)
    HEIGHTFIELD
};

enum class CollisionFalloff : uint8_t {
    CONSTANT,
    LINEAR,
    QUADRATIC
};

class GPUParticlesCollision3D : public Node3D {
public:
    GPUParticlesCollision3D();
    ~GPUParticlesCollision3D();

    // ------------------------------------------------------------------------
    // Shape parameters
    // ------------------------------------------------------------------------
    void set_shape_type(CollisionShape3DType type);
    CollisionShape3DType get_shape_type() const;
    void set_extents(const double* extents);   // half extents for sphere/box
    void get_extents(double* out_extents) const;
    void set_radius(double radius);
    double get_radius() const;

    // ------------------------------------------------------------------------
    // Heightfield data (if type = HEIGHTFIELD)
    // ------------------------------------------------------------------------
    void set_heightfield_data(int width, int depth, const std::vector<float>& heights,
                              double min_height, double max_height,
                              const double* origin = nullptr, double cell_size = 1.0);
    void clear_heightfield();

    // ------------------------------------------------------------------------
    // SDF texture (if type = SDF)
    // ------------------------------------------------------------------------
    void set_sdf_texture(int64_t texture_rid, double texture_to_world_scale);
    int64_t get_sdf_texture() const;

    // ------------------------------------------------------------------------
    // Collision response
    // ------------------------------------------------------------------------
    void set_bounce(float bounce);          // 0 = stop, 1 = perfect reflection
    float get_bounce() const;
    void set_friction(float friction);      // 0 = no friction, 1 = stop tangent
    float get_friction() const;
    void set_attenuation(float attenuation); // velocity reduction when inside collision
    float get_attenuation() const;
    void set_falloff(CollisionFalloff falloff);
    CollisionFalloff get_falloff() const;
    void set_max_distance(float max_dist);
    float get_max_distance() const;

    // ------------------------------------------------------------------------
    // Culling mask (which particle systems are affected)
    // ------------------------------------------------------------------------
    void set_cull_mask(uint32_t mask);
    uint32_t get_cull_mask() const;

    // ------------------------------------------------------------------------
    // Visual representation (debug)
    // ------------------------------------------------------------------------
    void set_visible(bool visible);
    bool is_visible() const;
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_gi_mode(int mode);            // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Rendering server sync
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting