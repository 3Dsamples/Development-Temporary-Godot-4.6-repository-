// Name : lighting enhancement
// File : scene/3d/cpu_particles_3d_ext.h 15 of 60
// Description : Extended CPU particles node with emission parameters, trails,
//               collisions, shadow and GI support, and full RenderingServer sync.
#pragma once

#include "scene/3d/cpu_particles_3d.h"
#include "servers/rendering_server.h"

class CPUParticles3DExt : public CPUParticles3D {
    GDCLASS(CPUParticles3DExt, CPUParticles3D);

public:
    CPUParticles3DExt();
    ~CPUParticles3DExt();

    // ------------------------------------------------------------------------
    // Emission control
    // ------------------------------------------------------------------------
    void set_emitting(bool p_emitting) override;
    bool is_emitting() const override;
    void set_one_shot(bool p_one_shot) override;
    bool is_one_shot() const override;
    void set_amount(int p_amount) override;
    int get_amount() const override;

    // ------------------------------------------------------------------------
    // Particle parameters
    // ------------------------------------------------------------------------
    void set_lifetime(float p_lifetime) override;
    float get_lifetime() const override;
    void set_preprocess(float p_preprocess) override;
    float get_preprocess() const override;
    void set_explosiveness(float p_explosiveness) override;
    float get_explosiveness() const override;
    void set_randomness(float p_randomness) override;
    float get_randomness() const override;

    // ------------------------------------------------------------------------
    // Emission shape
    // ------------------------------------------------------------------------
    void set_emission_shape(int p_shape) override;
    int get_emission_shape() const override;
    void set_emission_shape_extents(const Vector3 &p_extents) override;
    Vector3 get_emission_shape_extents() const override;
    void set_emission_points(const Vector<Vector3> &p_points) override;
    Vector<Vector3> get_emission_points() const override;

    // ------------------------------------------------------------------------
    // Particle visuals
    // ------------------------------------------------------------------------
    void set_draw_order(int p_order) override;
    int get_draw_order() const override;
    void set_material(const RID &p_material) override;
    RID get_material() const override;
    void set_color(const Color &p_color) override;
    Color get_color() const override;
    void set_color_ramp(const RID &p_color_ramp);
    RID get_color_ramp() const;

    // ------------------------------------------------------------------------
    // Trails
    // ------------------------------------------------------------------------
    void set_trail_enabled(bool p_enabled) override;
    bool is_trail_enabled() const override;
    void set_trail_length(float p_length) override;
    float get_trail_length() const override;

    // ------------------------------------------------------------------------
    // Collision (with scene geometry)
    // ------------------------------------------------------------------------
    void set_collision_enabled(bool p_enabled) override;
    bool is_collision_enabled() const override;
    void set_collision_radius(float p_radius) override;
    float get_collision_radius() const override;

    // ------------------------------------------------------------------------
    // Sub‑emitter (spawns child particles)
    // ------------------------------------------------------------------------
    void set_sub_emitter(int p_sub_emitter) override;
    int get_sub_emitter() const override;

    // ------------------------------------------------------------------------
    // Lighting & shadows
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    bool get_cast_shadow() const override;
    void set_gi_mode(int p_mode) override;
    int get_gi_mode() const override;

    // ------------------------------------------------------------------------
    // Rendering server synchronization (push all parameters)
    // ------------------------------------------------------------------------
    void sync_particles();

private:
    struct Impl;
    Impl *pimpl;
};