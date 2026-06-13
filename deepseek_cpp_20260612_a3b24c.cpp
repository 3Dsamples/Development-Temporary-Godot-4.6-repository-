// Name : lighting enhancement
// File : scene/3d/collision_object_3d_ext.h 39 of 60
// Description : Extended collision object with shape management, layer masks,
//               ray/area callbacks, and full RenderingServer synchronization for GI.
#pragma once

#include "scene/3d/collision_object_3d.h"
#include "servers/rendering_server.h"

class CollisionObject3DExt : public CollisionObject3D {
    GDCLASS(CollisionObject3DExt, CollisionObject3D);

public:
    CollisionObject3DExt();
    ~CollisionObject3DExt();

    // ------------------------------------------------------------------------
    // Shape management
    // ------------------------------------------------------------------------
    int add_shape(const RID &p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false);
    void remove_shape(int p_shape_id);
    void clear_shapes();
    int get_shape_count() const;
    RID get_shape(int p_shape_id) const;
    void set_shape_transform(int p_shape_id, const Transform3D &p_transform);
    Transform3D get_shape_transform(int p_shape_id) const;
    void set_shape_disabled(int p_shape_id, bool p_disabled);
    bool is_shape_disabled(int p_shape_id) const;

    // ------------------------------------------------------------------------
    // Layer and mask (collision detection and culling)
    // ------------------------------------------------------------------------
    void set_collision_layer(uint32_t p_layer);
    uint32_t get_collision_layer() const;
    void set_collision_mask(uint32_t p_mask);
    uint32_t get_collision_mask() const;
    void set_collision_priority(float p_priority);
    float get_collision_priority() const;

    // ------------------------------------------------------------------------
    // Ray and area callbacks (signals)
    // ------------------------------------------------------------------------
    void set_ray_pickable(bool p_enabled);
    bool is_ray_pickable() const;
    void set_area_pickable(bool p_enabled);
    bool is_area_pickable() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (collision objects can influence GI via emissive)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode) override;
    int get_gi_mode() const override;
    void set_gi_contribution(float p_amount) override;
    float get_gi_contribution() const override;
    void set_emissive(const Color &p_color, float p_intensity) override;
    Color get_emissive() const override;
    float get_emissive_intensity() const override;

    // ------------------------------------------------------------------------
    // Rendering server synchronization (update collision data)
    // ------------------------------------------------------------------------
    void sync_collision_object();

private:
    struct Impl;
    Impl *pimpl;
};