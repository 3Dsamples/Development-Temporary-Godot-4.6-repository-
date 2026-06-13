// Name : lighting enhancement
// File : scene/3d/node_3d.h 1 of xxx
// Description : Base 3D node with double‑precision transforms, hierarchical dirty propagation,
//               physics interpolation, and integration hooks for advanced lighting systems.
#pragma once

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/templates/list.h"
#include "core/templates/vector.h"
#include "servers/rendering_server.h"

class Node3D : public Object {
    GDCLASS(Node3D, Object);

public:
    Node3D();
    ~Node3D();

    // Transform hierarchy
    void set_parent(Node3D *p_parent);
    Node3D *get_parent() const;
    void add_child(Node3D *p_child);
    void remove_child(Node3D *p_child);
    void get_children(List<Node3D *> *r_children) const;

    // Local / world transforms
    void set_transform(const Transform3D &p_transform);
    const Transform3D &get_transform() const;
    Transform3D get_global_transform() const;
    void set_global_transform(const Transform3D &p_global);

    // Notify that transform changed (dirty propagation)
    void update_transform();
    bool is_transform_dirty() const;

    // Name and visibility
    void set_name(const String &p_name);
    String get_name() const;
    void set_visible(bool p_visible);
    bool is_visible() const;

    // Physics interpolation (smooth motion)
    void set_physics_interpolated(bool p_enabled);
    bool is_physics_interpolated() const;
    void synchronize_render_server(double p_delta);

    // Lighting integration (to be overridden by visual instances)
    virtual void set_cast_shadow(bool p_cast);
    virtual bool get_cast_shadow() const;
    virtual void set_gi_mode(int p_mode);
    virtual int get_gi_mode() const;
    virtual void set_gi_contribution(float p_amount);
    virtual float get_gi_contribution() const;
    virtual void set_emissive(const Color &p_color, float p_intensity);
    virtual Color get_emissive() const;
    virtual float get_emissive_intensity() const;

protected:
    virtual void _transform_changed();
    virtual void _update_render_server_transform();

    void _notify_transform_dirty();

private:
    struct Impl;
    Impl *pimpl;
};