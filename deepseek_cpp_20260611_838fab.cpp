// Name : lighting enhancement
// File : scene/3d/node_3d_ext.h 1 of 60
// Description : High‑performance extension to Godot's Node3D with integrated lighting,
//               render server caching, dirty flags, and physics interpolation for real‑time 3D.
#pragma once

#include "scene/3d/node_3d.h"
#include "servers/rendering_server.h"

class Node3DExt : public Node3D {
    GDCLASS(Node3DExt, Node3D);

public:
    Node3DExt();
    ~Node3DExt();

    // ------------------------------------------------------------------------
    // Transform and visibility overrides (with rendering server sync)
    // ------------------------------------------------------------------------
    void set_transform(const Transform3D &p_transform) override;
    void set_global_transform(const Transform3D &p_global) override;
    void set_visible(bool p_visible) override;

    // ------------------------------------------------------------------------
    // Lighting flags for dynamic global illumination and shadows
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    bool get_cast_shadow() const override;
    void set_gi_mode(int p_mode) override;
    int get_gi_mode() const override;
    void set_gi_contribution(float p_amount) override;
    float get_gi_contribution() const override;
    void set_emissive(const Color &p_color, float p_intensity) override;
    Color get_emissive() const override;
    float get_emissive_intensity() const override;

    // ------------------------------------------------------------------------
    // Rendering server instance management
    // ------------------------------------------------------------------------
    RID get_render_instance_id() const;
    void sync_render_server_transform(); // called each frame to push transform
    void sync_render_server_visibility();
    void sync_render_server_lighting_params();

    // ------------------------------------------------------------------------
    // Physics interpolation (for smooth motion at variable frame rates)
    // ------------------------------------------------------------------------
    void set_physics_interpolated(bool p_enabled);
    bool is_physics_interpolated() const;
    void set_physics_fraction(double p_frac); // fraction between physics ticks
    void apply_interpolated_transform();

    // ------------------------------------------------------------------------
    // Performance: skip frustum culling for specific nodes (e.g., directional lights)
    // ------------------------------------------------------------------------
    void set_always_visible(bool p_visible);
    bool is_always_visible() const;

protected:
    void _transform_changed() override;
    void _update_render_server_transform() override;

private:
    struct Impl;
    Impl *pimpl;
};