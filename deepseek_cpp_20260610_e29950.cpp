// collision_shape_3d.cpp
#include "collision_shape_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace lighting {

// ============================================================================
// Utility to create physics server shape from variant
// ============================================================================
static int64_t create_physics_shape(const ShapeVariant& shape) {
    // In a real engine, this would call PhysicsServer3D::shape_create()
    // and then set shape parameters. Here we return a dummy handle.
    // Placeholder: return incremented ID.
    static int64_t next_id = 1000;
    return next_id++;
}

// ============================================================================
// CollisionShape3D implementation
// ============================================================================
struct CollisionShape3D::Impl {
    ShapeVariant shape;
    ShapeType3D shape_type = ShapeType3D::EMPTY;
    Transform3D local_transform;       // relative to parent CollisionObject3D
    int gi_mode = 1;                   // static by default
    float gi_contribution = 1.0f;

    int64_t physics_shape_rid = -1;    // handle in physics server
    bool shape_dirty = true;
    bool transform_dirty = true;

    // cached parent collision object (set when added to tree)
    CollisionObject3D* parent_collision = nullptr;
    int shape_index_in_parent = -1;

    ~Impl() {
        if (physics_shape_rid != -1) {
            // PhysicsServer::shape_free(physics_shape_rid)
        }
    }
};

CollisionShape3D::CollisionShape3D() : pimpl(std::make_unique<Impl>()) {
    pimpl->local_transform.set_identity();
}
CollisionShape3D::~CollisionShape3D() = default;

void CollisionShape3D::set_shape(const ShapeVariant& shape) {
    pimpl->shape = shape;
    pimpl->shape_type = static_cast<ShapeType3D>(shape.index());
    pimpl->shape_dirty = true;
    update_shape();
}
const ShapeVariant& CollisionShape3D::get_shape() const { return pimpl->shape; }

ShapeType3D CollisionShape3D::get_shape_type() const { return pimpl->shape_type; }

void CollisionShape3D::set_sphere_radius(double radius) {
    if (pimpl->shape_type != ShapeType3D::SPHERE) {
        pimpl->shape = SphereShape3D{};
        pimpl->shape_type = ShapeType3D::SPHERE;
    }
    auto& sphere = std::get<SphereShape3D>(pimpl->shape);
    sphere.radius = radius;
    pimpl->shape_dirty = true;
}
double CollisionShape3D::get_sphere_radius() const {
    if (pimpl->shape_type == ShapeType3D::SPHERE)
        return std::get<SphereShape3D>(pimpl->shape).radius;
    return 0.0;
}

void CollisionShape3D::set_box_half_extents(const double* extents) {
    if (pimpl->shape_type != ShapeType3D::BOX) {
        pimpl->shape = BoxShape3D{};
        pimpl->shape_type = ShapeType3D::BOX;
    }
    auto& box = std::get<BoxShape3D>(pimpl->shape);
    memcpy(box.half_extents, extents, 3*sizeof(double));
    pimpl->shape_dirty = true;
}
void CollisionShape3D::get_box_half_extents(double* out_extents) const {
    if (pimpl->shape_type == ShapeType3D::BOX) {
        memcpy(out_extents, std::get<BoxShape3D>(pimpl->shape).half_extents, 3*sizeof(double));
    } else {
        out_extents[0]=out_extents[1]=out_extents[2]=0.5;
    }
}

void CollisionShape3D::set_capsule_radius(double radius) {
    if (pimpl->shape_type != ShapeType3D::CAPSULE) {
        pimpl->shape = CapsuleShape3D{};
        pimpl->shape_type = ShapeType3D::CAPSULE;
    }
    auto& capsule = std::get<CapsuleShape3D>(pimpl->shape);
    capsule.radius = radius;
    pimpl->shape_dirty = true;
}
double CollisionShape3D::get_capsule_radius() const {
    return (pimpl->shape_type == ShapeType3D::CAPSULE) ? std::get<CapsuleShape3D>(pimpl->shape).radius : 0.0;
}
void CollisionShape3D::set_capsule_height(double height) {
    if (pimpl->shape_type != ShapeType3D::CAPSULE) {
        pimpl->shape = CapsuleShape3D{};
        pimpl->shape_type = ShapeType3D::CAPSULE;
    }
    auto& capsule = std::get<CapsuleShape3D>(pimpl->shape);
    capsule.height = height;
    pimpl->shape_dirty = true;
}
double CollisionShape3D::get_capsule_height() const {
    return (pimpl->shape_type == ShapeType3D::CAPSULE) ? std::get<CapsuleShape3D>(pimpl->shape).height : 0.0;
}

void CollisionShape3D::set_cylinder_radius(double radius) {
    if (pimpl->shape_type != ShapeType3D::CYLINDER) {
        pimpl->shape = CylinderShape3D{};
        pimpl->shape_type = ShapeType3D::CYLINDER;
    }
    auto& cylinder = std::get<CylinderShape3D>(pimpl->shape);
    cylinder.radius = radius;
    pimpl->shape_dirty = true;
}
double CollisionShape3D::get_cylinder_radius() const {
    return (pimpl->shape_type == ShapeType3D::CYLINDER) ? std::get<CylinderShape3D>(pimpl->shape).radius : 0.0;
}
void CollisionShape3D::set_cylinder_height(double height) {
    if (pimpl->shape_type != ShapeType3D::CYLINDER) {
        pimpl->shape = CylinderShape3D{};
        pimpl->shape_type = ShapeType3D::CYLINDER;
    }
    auto& cylinder = std::get<CylinderShape3D>(pimpl->shape);
    cylinder.height = height;
    pimpl->shape_dirty = true;
}
double CollisionShape3D::get_cylinder_height() const {
    return (pimpl->shape_type == ShapeType3D::CYLINDER) ? std::get<CylinderShape3D>(pimpl->shape).height : 0.0;
}

void CollisionShape3D::set_convex_mesh(const std::vector<double>& vertices, const std::vector<int>& indices) {
    ConvexPolyhedronShape3D convex;
    convex.vertices = vertices;
    convex.indices = indices;
    pimpl->shape = convex;
    pimpl->shape_type = ShapeType3D::CONVEX_POLYHEDRON;
    pimpl->shape_dirty = true;
}

void CollisionShape3D::set_concave_mesh(const std::vector<double>& vertices, const std::vector<int>& indices) {
    ConcaveMeshShape3D concave;
    concave.vertices = vertices;
    concave.indices = indices;
    pimpl->shape = concave;
    pimpl->shape_type = ShapeType3D::CONCAVE_MESH;
    pimpl->shape_dirty = true;
}

void CollisionShape3D::set_heightfield(int width, int depth, const std::vector<float>& heights, double min_h, double max_h) {
    HeightfieldShape3D hf;
    hf.width = width;
    hf.depth = depth;
    hf.heights = heights;
    hf.min_height = min_h;
    hf.max_height = max_h;
    pimpl->shape = hf;
    pimpl->shape_type = ShapeType3D::HEIGHTFIELD;
    pimpl->shape_dirty = true;
}

void CollisionShape3D::set_plane(const double* normal, double distance) {
    PlaneShape3D plane;
    memcpy(plane.normal, normal, 3*sizeof(double));
    plane.distance = distance;
    pimpl->shape = plane;
    pimpl->shape_type = ShapeType3D::PLANE;
    pimpl->shape_dirty = true;
}

void CollisionShape3D::set_local_transform(const Transform3D& transform) {
    pimpl->local_transform = transform;
    pimpl->transform_dirty = true;
    if (pimpl->parent_collision && pimpl->shape_index_in_parent >= 0) {
        // Update shape transform in parent physics body/area
        // PhysicsServer::body_set_shape_transform(parent_rid, shape_index, local_transform)
    }
}
Transform3D CollisionShape3D::get_local_transform() const { return pimpl->local_transform; }

void CollisionShape3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int CollisionShape3D::get_gi_mode() const { return pimpl->gi_mode; }
void CollisionShape3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float CollisionShape3D::get_gi_contribution() const { return pimpl->gi_contribution; }

void CollisionShape3D::update_shape(bool force) {
    if (!force && !pimpl->shape_dirty) return;

    // Create new physics shape if needed
    if (pimpl->physics_shape_rid == -1) {
        pimpl->physics_shape_rid = create_physics_shape(pimpl->shape);
    } else {
        // Update existing shape parameters: PhysicsServer::shape_set_data(rid, shape_data)
    }

    // If attached to parent CollisionObject3D, replace shape
    if (pimpl->parent_collision) {
        // Remove old shape, add new one with current transform
        // (In real engine, we would have shape_index_in_parent)
        if (pimpl->shape_index_in_parent >= 0) {
            // PhysicsServer::body_set_shape(parent_rid, shape_index, physics_shape_rid, local_transform)
        }
    }
    pimpl->shape_dirty = false;
}

void CollisionShape3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // Ensure shape is updated before physics step
    if (pimpl->shape_dirty) update_shape();

    // Update GI contribution: if this shape is used for light probes or VCT,
    // we may need to register or update its contribution.
    if (pimpl->gi_mode > 0 && pimpl->gi_contribution > 0.0f) {
        // Placeholder: add shape geometry to global illumination system
        // (e.g., inject into VCT, DFAO SDF, or light probe sampling)
    }
}

// When added to a CollisionObject3D, set parent pointer
// This would be called by the parent during child notification.
// For completeness, we override _ready and _enter_tree.

} // namespace lighting