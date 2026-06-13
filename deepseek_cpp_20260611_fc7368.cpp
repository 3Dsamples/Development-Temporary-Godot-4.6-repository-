// gpu_particles_attractor_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// Attractor shape types
// ============================================================================
enum class AttractorShape : uint8_t {
    SPHERE,
    BOX,
    VECTOR_FIELD,
    POINT
};

// ============================================================================
// GPUParticlesAttractor3D – defines a force field (attraction/repulsion)
// that influences GPU particles. Supports linear/quadratic falloff, strength,
// and axis‑aligned or custom direction. Integrated with lighting: can be
// emissive (visual glow) and can affect GI via particle influence (optional).
// ============================================================================

class GPUParticlesAttractor3D : public Node3D {
public:
    GPUParticlesAttractor3D();
    ~GPUParticlesAttractor3D();

    // ------------------------------------------------------------------------
    // Shape and direction
    // ------------------------------------------------------------------------
    void set_shape(AttractorShape shape);
    AttractorShape get_shape() const;
    void set_size(const double* size);          // half extents for box/sphere
    void get_size(double* out_size) const;
    void set_direction(const double* dir);      // for vector field / point
    void get_direction(double* out_dir) const;

    // ------------------------------------------------------------------------
    // Force parameters
    // ------------------------------------------------------------------------
    void set_strength(float strength);          // positive = attract, negative = repel
    float get_strength() const;
    void set_falloff(float falloff);            // 0 = constant, 1 = linear, 2 = quadratic
    float get_falloff() const;
    void set_attenuation(float attenuation);    // distance attenuation factor
    float get_attenuation() const;
    void set_max_distance(float max_dist);
    float get_max_distance() const;

    // ------------------------------------------------------------------------
    // Axis‑aligned attraction (for box shape: attract along local axes)
    // ------------------------------------------------------------------------
    void set_axis_enabled(bool x, bool y, bool z);
    void get_axis_enabled(bool& x, bool& y, bool& z) const;

    // ------------------------------------------------------------------------
    // Lighting & GI (visual representation)
    // ------------------------------------------------------------------------
    void set_visible(bool visible);
    bool is_visible() const;
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_gi_mode(int mode);                 // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Rendering server sync (updates attractor data for particle system)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting