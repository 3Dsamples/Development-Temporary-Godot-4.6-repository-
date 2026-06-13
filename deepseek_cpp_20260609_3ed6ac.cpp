// node_3d.h
// High‑performance 3D transform node with double‑precision world coordinates,
// hierarchical transforms, and dirty flag propagation for large‑world support.
#pragma once
#include <cstdint>
#include <vector>
#include <atomic>

namespace lighting {

// Double‑precision for galactic / microscopic scale.
struct Transform3D {
    double origin[3];  // translation (x, y, z)
    double basis[9];   // 3x3 rotation + scale row‑major
    double scale[3];   // optional separate scale

    Transform3D();
    void set_identity();
    void translate(double x, double y, double z);
    void rotate_x(double rad);
    void rotate_y(double rad);
    void rotate_z(double rad);
    void scale_local(double sx, double sy, double sz);
    Transform3D inverse() const;
};

class Node3D {
public:
    Node3D();
    virtual ~Node3D();

    // Transform hierarchy
    void set_parent(Node3D* parent);
    void add_child(Node3D* child);
    void remove_child(Node3D* child);

    // Local / world transforms
    void set_transform(const Transform3D& local);
    const Transform3D& get_transform() const;
    Transform3D get_global_transform() const;
    void set_global_transform(const Transform3D& world);

    // Notify that transform changed (dirty propagation)
    void update_transform();
    bool is_transform_dirty() const;

    // Name and visibility
    void set_name(const char* name);
    const char* get_name() const;
    void set_visible(bool visible);
    bool is_visible() const;

    // For physics interpolation (smooth motion)
    void set_physics_interpolated(bool enabled);
    void synchronize_render_server(double delta);

protected:
    virtual void _transform_changed();

private:
    struct Impl;
    Impl* pimpl;
};

} // namespace lighting