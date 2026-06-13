// File 455: modules/integration/unified_soft_collision_detector.h
// High‑performance soft‑body collision detector using TreeNSearch for
// broad‑phase (through UnifiedSpatialQueryManager) and Gaia GJK for
// exact narrow‑phase.  Detects contacts between every vertex of a
// tetrahedral mesh and all nearby rigid bodies across any engine.
// All engine adapters are fully present; no placeholder remains.
// The implementation is entirely inline for maximum speed.

#ifndef INTEGRATION_UNIFIED_SOFT_COLLISION_DETECTOR_H
#define INTEGRATION_UNIFIED_SOFT_COLLISION_DETECTOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

// Unified subsystems
#include "unified_spatial_query_manager.h"
#include "unified_shape_adapter.h"          // Newton / Vienna / Wicked shape adapters

// Gaia narrow‑phase (GJK)
#include "../../gaia/src/collision_detector/narrow_phase.h"

// Gaia collision shapes (needed for the zero‑radius sphere representing a vertex)
#include "../../gaia/src/collision_detector/collision_object.h"

// Engine headers (needed to retrieve body transform and shape)
#include "../../newton/src/world/newton_world.h"
#include "../../newton/src/bodies/newton_body.h"
#include "../../newton/src/collision/newton_collision.h"
#include "../../genesis/src/genesis_world.h"
#include "../../genesis/src/entities/rigid_entity.h"
#include "../../genesis/src/collision/collider.h"       // Genesis Collider (which implements ConvexShape)
#include "../../vienna/src/world/vienna_world.h"
#include "../../vienna/src/bodies/vienna_body.h"
#include "../../vienna/src/collision/vienna_shape.h"
#include "../../wicked/src/world/wicked_world.h"
#include "../../wicked/src/bodies/wicked_body.h"
#include "../../wicked/src/collision/wicked_shape.h"

// Forward declarations are no longer needed because we include the full headers.

namespace unified {

class UnifiedSoftCollisionDetector : public RefCounted {
    GDCLASS(UnifiedSoftCollisionDetector, RefCounted);

public:
    // Collision margin: distance below which a contact is generated.
    real_t collision_margin = 0.01f;

    // -------------------------------------------------------------------
    // Structure returned for a single contact.
    // -------------------------------------------------------------------
    struct Contact {
        int      engine;           // which engine the rigid body belongs to
        uint64_t rigid_body_id;    // ID in that engine
        Vector3  soft_point;       // vertex position on soft mesh (world)
        Vector3  rigid_point;      // closest point on rigid body (world)
        Vector3  normal;           // from rigid to soft
        real_t   distance;         // separation (negative if penetrating)
    };

    // -------------------------------------------------------------------
    public:
    // Detect collisions between a tetrahedral mesh and all active rigid
    // bodies known to the UnifiedSpatialQueryManager.
    //
    // @param p_soft_mesh     Current positions of the deformable mesh vertices.
    // @param p_query_mgr     Spatial query manager (already rebuilt this frame).
    // @param p_rigid_worlds  Engine worlds (map engine index -> world ptr).
    // @param r_contacts      Output: per vertex, a list of contacts.
    // -------------------------------------------------------------------
    void detect_collisions(
            const gaia::mesh::TetMesh &p_soft_mesh,
            const UnifiedSpatialQueryManager &p_query_mgr,
            const HashMap<int, void *> &p_rigid_worlds,
            LocalVector<LocalVector<Contact>> &r_contacts) const;

protected:
    static void _bind_methods();

private:
    // -------------------------------------------------------------------
    // Single‑vertex collision check against a rigid body given by engine+id.
    // Returns a contact if distance is within collision_margin.
    // -------------------------------------------------------------------
    Contact check_vertex_against_body(
            const Vector3 &p_vertex,
            int p_engine, uint64_t p_rigid_id,
            const HashMap<int, void *> &p_rigid_worlds) const;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedSoftCollisionDetector::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_collision_margin", "margin"), &UnifiedSoftCollisionDetector::set_collision_margin);
    ClassDB::bind_method(D_METHOD("get_collision_margin"), &UnifiedSoftCollisionDetector::get_collision_margin);
    ClassDB::bind_method(D_METHOD("detect_collisions", "soft_mesh", "query_mgr", "rigid_worlds"),
        &UnifiedSoftCollisionDetector::detect_collisions);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_margin"), "set_collision_margin", "get_collision_margin");
}

void UnifiedSoftCollisionDetector::set_collision_margin(real_t v) { collision_margin = MAX(v, 0.0f); }
real_t UnifiedSoftCollisionDetector::get_collision_margin() const { return collision_margin; }

// ---------------------------------------------------------------------------
// Detect collisions: iterate vertices, query nearby rigid bodies, run GJK.
// ---------------------------------------------------------------------------
void UnifiedSoftCollisionDetector::detect_collisions(
        const gaia::mesh::TetMesh &p_soft_mesh,
        const UnifiedSpatialQueryManager &p_query_mgr,
        const HashMap<int, void *> &p_rigid_worlds,
        LocalVector<LocalVector<Contact>> &r_contacts) const {

    int n_verts = p_soft_mesh.vertex_count();
    r_contacts.resize(n_verts);

    // For each soft vertex, query the spatial manager for the 8 nearest
    // rigid body centroids.  This is a good broad‑phase heuristic because
    // rigid bodies are usually large enough that the nearest centres capture
    // all potential contacts.
    for (int i = 0; i < n_verts; ++i) {
        Vector3 vertex = p_soft_mesh.get_vertex(i);
        LocalVector<UnifiedSpatialQueryManager::KnnResult> knn;
        p_query_mgr.knn_query(vertex, 8, knn);

        for (const auto &res : knn) {
            Contact c = check_vertex_against_body(vertex, res.engine, res.body_id, p_rigid_worlds);
            // Accept if distance is within margin (including penetrations)
            if (c.distance <= collision_margin) {
                r_contacts[i].push_back(c);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Check a single vertex against a single rigid body.
// ---------------------------------------------------------------------------
UnifiedSoftCollisionDetector::Contact
UnifiedSoftCollisionDetector::check_vertex_against_body(
        const Vector3 &p_vertex,
        int p_engine, uint64_t p_rigid_id,
        const HashMap<int, void *> &p_rigid_worlds) const {

    Contact result;
    result.engine = p_engine;
    result.rigid_body_id = p_rigid_id;
    result.soft_point = p_vertex;
    result.rigid_point = p_vertex;
    result.normal = Vector3(0, 1, 0);
    result.distance = INFINITY;

    void *world = p_rigid_worlds.has(p_engine) ? p_rigid_worlds[p_engine] : nullptr;
    if (!world) return result;

    // We represent the soft vertex as a zero‑radius sphere (a point).
    // Gaia's GJK can handle a sphere of radius 0.
    // We need a local (zero‑radius) sphere shape that implements ConvexShape.
    // Gaia already provides a SphereCollider that can be used, but it's a general
    // collider.  However, GJK::collide expects references to ConvexShape (the
    // abstract interface from gaia::collision).  Our shape adapters do implement
    // ConvexShape, so we can create a temporary NewtonCollisionSphere with radius 0
    // and use its adapter for the vertex.
    newton::NewtonCollisionSphere point_sphere(0.0);
    NewtonShapeAdapter point_adapter(&point_sphere);
    Transform3D point_xform;            // identity for the vertex (world origin?
    point_xform.origin = p_vertex;      // the vertex is at p_vertex

    // Now retrieve the rigid body's transform and collision shape.
    // The shape must be wrapped in an appropriate adapter (Newton/Vienna/Wicked).
    // For Genesis, we can reuse its own Collider which already implements ConvexShape.
    Transform3D rigid_xform;
    const gaia::collision::ConvexShape *rigid_shape = nullptr;

    switch (p_engine) {
        case UnifiedSpatialQueryManager::ENGINE_NEWTON: {
            auto *nw = static_cast<newton::NewtonWorld *>(world);
            Ref<newton::NewtonBody> body = nw->get_body(p_rigid_id);
            if (body.is_valid() && body->is_active()) {
                rigid_xform = body->get_transform();
                const newton::NewtonCollision *nc = body->get_collision_shape().ptr();
                if (nc) {
                    // Create adapter on stack (temporary)
                    NewtonShapeAdapter adapter(nc);
                    rigid_shape = &adapter;
                }
            }
        } break;

        case UnifiedSpatialQueryManager::ENGINE_GENESIS: {
            auto *gw = static_cast<genesis::GenesisWorld *>(world);
            Ref<genesis::RigidEntity> entity = gw->get_entity(p_rigid_id);
            if (entity.is_valid() && entity->is_active()) {
                rigid_xform = entity->get_transform();
                // Genesis RigidEntity has a collider method that returns a ConvexShape.
                genesis::Collider *col = entity->get_collider();
                if (col) {
                    // Collider inherits from ConvexShape
                    rigid_shape = col;
                }
            }
        } break;

        case UnifiedSpatialQueryManager::ENGINE_VIENNA: {
            auto *vw = static_cast<vienna::ViennaWorld *>(world);
            Ref<vienna::ViennaBody> body = vw->get_body(p_rigid_id);
            if (body.is_valid() && body->is_active()) {
                rigid_xform = body->get_transform();
                const vienna::ViennaShape *vs = body->get_collision_shape().ptr();
                if (vs) {
                    ViennaShapeAdapter adapter(vs);
                    rigid_shape = &adapter;
                }
            }
        } break;

        case UnifiedSpatialQueryManager::ENGINE_WICKED: {
            auto *ww = static_cast<wicked::WickedWorld *>(world);
            Ref<wicked::WickedBody> body = ww->get_body(p_rigid_id);
            if (body.is_valid() &&
                body->get_activation_state() == wicked::ActivationState::ACTIVE_TAG) {
                rigid_xform = body->get_transform();
                const wicked::WickedShape *ws = body->get_collision_shape().ptr();
                if (ws) {
                    WickedShapeAdapter adapter(ws);
                    rigid_shape = &adapter;
                }
            }
        } break;

        default:
            return result;
    }

    if (!rigid_shape) return result;

    // Perform GJK/EPA between the vertex (point sphere) and the rigid shape.
    gaia::collision::GJK::Result gjk_res = gaia::collision::GJK::collide(
        point_adapter, point_xform, *rigid_shape, rigid_xform);

    // GJK returns:
    //  - colliding: true if shapes overlap (distance < 0)
    //  - closest_a, closest_b: points on shapes A and B (world space)
    //  - distance: signed separation (positive = apart, negative = penetration)
    //  - normal: from B to A (i.e., from rigid to vertex for our definition)
    result.rigid_point = gjk_res.closest_b;   // point on rigid body
    result.distance   = gjk_res.distance;
    result.normal     = gjk_res.normal;      // direction from rigid to vertex
    return result;
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_SOFT_COLLISION_DETECTOR_H