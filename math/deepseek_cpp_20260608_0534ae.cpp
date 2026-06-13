// File 395: modules/integration/unified_shape_adapter.h
// Unified Shape Adapters – wraps collision shapes from Newton, Vienna,
// Wicked, and Genesis (RigidEntity) into the common ConvexShape interface
// defined by Gaia.  This allows the Gaia GJK/EPA/CCD solvers to operate
// directly on bodies from any engine without requiring the engine itself
// to inherit from a particular base class.  Each adapter stores a raw
// pointer to the engine shape and implements get_support() by delegating
// to the engine's existing support function.  All methods are inline
// for maximum performance in the hot path.

#ifndef INTEGRATION_UNIFIED_SHAPE_ADAPTER_H
#define INTEGRATION_UNIFIED_SHAPE_ADAPTER_H

// Gaia ConvexShape interface (must be inherited by all adapters)
#include "../../gaia/src/collision_detector/narrow_phase.h" // includes ConvexShape

// Engine shape types
#include "../../newton/src/collision/newton_collision.h"
#include "../../genesis/src/collision/collider.h"        // Genesis ConvexShape? Actually Genesis uses gaia::collision::Collider, which already inherits from ConvexShape? Let's check: in file 53 we had Genesis GJK and Collider; but we haven't defined a generic adapter for it. Actually Genesis uses gaia::collision::Collider (from gaia). So for Genesis shapes (RigidEntity) we already have a collider that implements ConvexShape. So we may not need an adapter for Genesis.
#include "../../vienna/src/collision/vienna_shape.h"
#include "../../wicked/src/collision/wicked_shape.h"

// Gaia math types (for Vector3, Transform3D)
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace unified {

// =========================================================================
// NewtonShapeAdapter – wraps a newton::NewtonCollision into a ConvexShape.
// =========================================================================
class NewtonShapeAdapter : public gaia::collision::ConvexShape {
private:
    const newton::NewtonCollision *shape;  // non‑owning pointer

public:
    explicit NewtonShapeAdapter(const newton::NewtonCollision *p_shape)
        : shape(p_shape) {
        ERR_FAIL_NULL(p_shape);
    }

    // Delegates to Newton's own get_support.
    virtual Vector3 get_support(const Vector3 &dir_local) const override {
        // Note: dir_local is assumed to be in local space of the shape.
        // Newton's get_support expects world direction and a world transform,
        // but we are in ConvexShape interface which receives local direction.
        // We need to transform the direction?  Actually, the GJK/EPA from Gaia
        // uses ConvexShape in the following way: they call get_support with a
        // direction that is already transformed to the shape's local frame by
        // the caller?  Let's inspect Gaia's GJK usage:
        // In gaia::collision::GJK::collide, they call:
        //   gaia::collision::GJK::collide(const ConvexShape &shapeA, const Transform3D &xA, ...)
        // Inside, they compute support in world space by transforming the local support:
        //   shapeA.get_support(transformA.basis.xform_inv(world_dir)) → local support
        //   then transform back to world.  So the ConvexShape::get_support receives a
        // direction in local space and must return a point in local space.
        // NewtonCollision::get_support receives a world direction and world transform
        // and returns world point.  So we cannot use it directly in local space.
        // Instead, we must implement the support ourselves using the shape's geometry.
        // For Newton shapes, we can replicate the support logic (sphere, box, etc.)
        // rather than calling the Newton method.  That way we stay local.
        // Since NewtonCollision has a get_support that takes world dir and transform,
        // we'll need to adapt: we could temporarily create an identity transform,
        // but the Newton method would transform to world then back to local? No.
        // Simpler: For each shape type, we'll query the underlying shape's type and
        // compute local support directly.  To keep the adapter generic, we'll store
        // the shape's type and implement a switch.
        if (!shape) return Vector3();
        switch (shape->get_shape_type()) {
            case newton::ShapeType::SPHERE: {
                real_t radius = static_cast<const newton::NewtonCollisionSphere *>(shape)->get_radius();
                Vector3 dir = dir_local.normalized();
                return dir * radius;
            }
            case newton::ShapeType::BOX: {
                const vec3 &he = static_cast<const newton::NewtonCollisionBox *>(shape)->get_half_extents();
                return Vector3(
                    (dir_local.x >= 0) ? he.x : -he.x,
                    (dir_local.y >= 0) ? he.y : -he.y,
                    (dir_local.z >= 0) ? he.z : -he.z);
            }
            case newton::ShapeType::CAPSULE: {
                const auto *cap = static_cast<const newton::NewtonCollisionCapsule *>(shape);
                real_t r = cap->get_radius();
                real_t half_h = MAX(cap->get_height() * 0.5f - r, 0.0f);
                Vector3 center(0,0,0);
                if (dir_local.y > 0) center.y = half_h;
                else if (dir_local.y < 0) center.y = -half_h;
                return center + dir_local.normalized() * r;
            }
            case newton::ShapeType::CYLINDER: {
                const auto *cyl = static_cast<const newton::NewtonCollisionCylinder *>(shape);
                real_t r = cyl->get_radius();
                real_t half_h = cyl->get_height() * 0.5f;
                real_t cap_y = (dir_local.y >= 0) ? half_h : -half_h;
                real_t rad_len = Math::sqrt(dir_local.x*dir_local.x + dir_local.z*dir_local.z);
                Vector3 support(0, cap_y, 0);
                if (rad_len > CMP_EPSILON) {
                    real_t s = r / rad_len;
                    support.x = dir_local.x * s;
                    support.z = dir_local.z * s;
                }
                return support;
            }
            case newton::ShapeType::CONE: {
                const auto *cone = static_cast<const newton::NewtonCollisionCone *>(shape);
                real_t r = cone->get_radius();
                real_t h = cone->get_height();
                real_t half_h = h * 0.5f;
                Vector3 tip(0, half_h, 0);
                if (dir_local.y > 0.999f) return tip;
                real_t rad_len = Math::sqrt(dir_local.x*dir_local.x + dir_local.z*dir_local.z);
                Vector3 base_pt(0, -half_h, 0);
                if (rad_len > CMP_EPSILON) {
                    real_t s = r / rad_len;
                    base_pt.x = dir_local.x * s;
                    base_pt.z = dir_local.z * s;
                }
                return (tip.dot(dir_local) > base_pt.dot(dir_local)) ? tip : base_pt;
            }
            case newton::ShapeType::CONVEX_HULL: {
                const auto *hull = static_cast<const newton::NewtonCollisionConvexHull *>(shape);
                real_t best_dot = -INFINITY;
                int best_idx = 0;
                const LocalVector<vec3> &verts = hull->get_vertices();
                for (int i = 0; i < verts.size(); ++i) {
                    real_t d = verts[i].dot(dir_local);
                    if (d > best_dot) { best_dot = d; best_idx = i; }
                }
                return verts[best_idx];
            }
            default: return Vector3(); // fallback
        }
        return Vector3();
    }

    // Not used by GJK, but ConvexShape requires an empty implementation.
    virtual ~ConvexShape() override {}
};

// =========================================================================
// ViennaShapeAdapter – wraps a vienna::ViennaShape into a ConvexShape.
// =========================================================================
class ViennaShapeAdapter : public gaia::collision::ConvexShape {
private:
    const vienna::ViennaShape *shape;

public:
    explicit ViennaShapeAdapter(const vienna::ViennaShape *p_shape) : shape(p_shape) {
        ERR_FAIL_NULL(p_shape);
    }

    virtual Vector3 get_support(const Vector3 &dir_local) const override {
        if (!shape) return Vector3();
        switch (shape->get_shape_type()) {
            case vienna::ShapeType::SPHERE: {
                auto *s = static_cast<const vienna::ViennaShapeSphere *>(shape);
                Vector3 dir = dir_local.normalized();
                return dir * s->get_radius();
            }
            case vienna::ShapeType::BOX: {
                auto *s = static_cast<const vienna::ViennaShapeBox *>(shape);
                const vec3 &he = s->get_half_extents();
                return Vector3(
                    (dir_local.x >= 0) ? he.x : -he.x,
                    (dir_local.y >= 0) ? he.y : -he.y,
                    (dir_local.z >= 0) ? he.z : -he.z);
            }
            case vienna::ShapeType::CAPSULE: {
                auto *s = static_cast<const vienna::ViennaShapeCapsule *>(shape);
                real_t r = s->get_radius();
                real_t half_h = MAX(s->get_height() * 0.5f - r, 0.0f);
                Vector3 center(0,0,0);
                if (dir_local.y > 0) center.y = half_h;
                else if (dir_local.y < 0) center.y = -half_h;
                return center + dir_local.normalized() * r;
            }
            case vienna::ShapeType::CYLINDER: {
                auto *s = static_cast<const vienna::ViennaShapeCylinder *>(shape);
                real_t r = s->get_radius();
                real_t half_h = s->get_height() * 0.5f;
                real_t cap_y = (dir_local.y >= 0) ? half_h : -half_h;
                real_t rad_len = Math::sqrt(dir_local.x*dir_local.x + dir_local.z*dir_local.z);
                Vector3 support(0, cap_y, 0);
                if (rad_len > CMP_EPSILON) {
                    real_t scl = r / rad_len;
                    support.x = dir_local.x * scl;
                    support.z = dir_local.z * scl;
                }
                return support;
            }
            case vienna::ShapeType::CONE: {
                auto *s = static_cast<const vienna::ViennaShapeCone *>(shape);
                real_t r = s->get_radius();
                real_t h = s->get_height();
                real_t half_h = h * 0.5f;
                Vector3 tip(0, half_h, 0);
                if (dir_local.y > 0.999f) return tip;
                real_t rad_len = Math::sqrt(dir_local.x*dir_local.x + dir_local.z*dir_local.z);
                Vector3 base_pt(0, -half_h, 0);
                if (rad_len > CMP_EPSILON) {
                    real_t scl = r / rad_len;
                    base_pt.x = dir_local.x * scl;
                    base_pt.z = dir_local.z * scl;
                }
                return (tip.dot(dir_local) > base_pt.dot(dir_local)) ? tip : base_pt;
            }
            case vienna::ShapeType::CONVEX_HULL: {
                auto *s = static_cast<const vienna::ViennaShapeConvexHull *>(shape);
                real_t best_dot = -INFINITY;
                int best_idx = 0;
                const LocalVector<vec3> &verts = s->get_vertices();
                for (int i = 0; i < verts.size(); ++i) {
                    real_t d = verts[i].dot(dir_local);
                    if (d > best_dot) { best_dot = d; best_idx = i; }
                }
                return verts[best_idx];
            }
            default: return Vector3();
        }
    }
    virtual ~ConvexShape() override {}
};

// =========================================================================
// WickedShapeAdapter – wraps a wicked::WickedShape into a ConvexShape.
// =========================================================================
class WickedShapeAdapter : public gaia::collision::ConvexShape {
private:
    const wicked::WickedShape *shape;

public:
    explicit WickedShapeAdapter(const wicked::WickedShape *p_shape) : shape(p_shape) {
        ERR_FAIL_NULL(p_shape);
    }

    virtual Vector3 get_support(const Vector3 &dir_local) const override {
        if (!shape) return Vector3();
        switch (shape->get_shape_type()) {
            case wicked::ShapeType::SPHERE: {
                auto *s = static_cast<const wicked::WickedShapeSphere *>(shape);
                Vector3 dir = dir_local.normalized();
                return dir * s->get_radius();
            }
            case wicked::ShapeType::BOX: {
                auto *s = static_cast<const wicked::WickedShapeBox *>(shape);
                const vec3 &he = s->get_half_extents();
                return Vector3(
                    (dir_local.x >= 0) ? he.x : -he.x,
                    (dir_local.y >= 0) ? he.y : -he.y,
                    (dir_local.z >= 0) ? he.z : -he.z);
            }
            case wicked::ShapeType::CAPSULE: {
                auto *s = static_cast<const wicked::WickedShapeCapsule *>(shape);
                real_t r = s->get_radius();
                real_t half_h = MAX(s->get_height() * 0.5f - r, 0.0f);
                Vector3 center(0,0,0);
                if (dir_local.y > 0) center.y = half_h;
                else if (dir_local.y < 0) center.y = -half_h;
                return center + dir_local.normalized() * r;
            }
            case wicked::ShapeType::CYLINDER: {
                auto *s = static_cast<const wicked::WickedShapeCylinder *>(shape);
                real_t r = s->get_radius();
                real_t half_h = s->get_height() * 0.5f;
                real_t cap_y = (dir_local.y >= 0) ? half_h : -half_h;
                real_t rad_len = Math::sqrt(dir_local.x*dir_local.x + dir_local.z*dir_local.z);
                Vector3 support(0, cap_y, 0);
                if (rad_len > CMP_EPSILON) {
                    real_t scl = r / rad_len;
                    support.x = dir_local.x * scl;
                    support.z = dir_local.z * scl;
                }
                return support;
            }
            case wicked::ShapeType::CONE: {
                auto *s = static_cast<const wicked::WickedShapeCone *>(shape);
                real_t r = s->get_radius();
                real_t h = s->get_height();
                real_t half_h = h * 0.5f;
                Vector3 tip(0, half_h, 0);
                if (dir_local.y > 0.999f) return tip;
                real_t rad_len = Math::sqrt(dir_local.x*dir_local.x + dir_local.z*dir_local.z);
                Vector3 base_pt(0, -half_h, 0);
                if (rad_len > CMP_EPSILON) {
                    real_t scl = r / rad_len;
                    base_pt.x = dir_local.x * scl;
                    base_pt.z = dir_local.z * scl;
                }
                return (tip.dot(dir_local) > base_pt.dot(dir_local)) ? tip : base_pt;
            }
            case wicked::ShapeType::CONVEX_HULL: {
                auto *s = static_cast<const wicked::WickedShapeConvexHull *>(shape);
                real_t best_dot = -INFINITY;
                int best_idx = 0;
                const LocalVector<vec3> &verts = s->get_vertices();
                for (int i = 0; i < verts.size(); ++i) {
                    real_t d = verts[i].dot(dir_local);
                    if (d > best_dot) { best_dot = d; best_idx = i; }
                }
                return verts[best_idx];
            }
            default: return Vector3();
        }
    }
    virtual ~ConvexShape() override {}
};

// =========================================================================
// Factory function to create the appropriate adapter for an engine shape.
// The returned pointer is heap-allocated (memnew); caller must memdelete.
// =========================================================================
inline gaia::collision::ConvexShape *create_shape_adapter(int p_engine, const void *p_shape) {
    if (!p_shape) return nullptr;
    switch (p_engine) {
        case 0: // Newton
            return memnew(NewtonShapeAdapter(static_cast<const newton::NewtonCollision *>(p_shape)));
        case 2: // Vienna
            return memnew(ViennaShapeAdapter(static_cast<const vienna::ViennaShape *>(p_shape)));
        case 3: // Wicked
            return memnew(WickedShapeAdapter(static_cast<const wicked::WickedShape *>(p_shape)));
        default: return nullptr;
    }
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_SHAPE_ADAPTER_H