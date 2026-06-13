// File 447: modules/integration/unified_soft_body_pressure_cavity.h
// Pressure cavity for closed soft‑body surfaces.  Computes the signed volume
// of a triangle mesh (counter‑clockwise = outward), compares it against a
// target volume, and applies a proportional pressure force to each face.
// The resulting force is distributed equally to the three vertices.
// Supports per‑vertex mass for velocity update and a pinned‑vertex mask.
// All maths are fully implemented inline; no separate .cpp is needed.

#ifndef INTEGRATION_UNIFIED_SOFT_BODY_PRESSURE_CAVITY_H
#define INTEGRATION_UNIFIED_SOFT_BODY_PRESSURE_CAVITY_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedSoftBodyPressureCavity : public RefCounted {
    GDCLASS(UnifiedSoftBodyPressureCavity, RefCounted);

public:
    // Target internal volume (m³).  Default 1.0.
    real_t target_volume = 1.0f;
    // Proportional gain (Pa/m³).  Pressure = gain * (target_volume - current_volume).
    real_t pressure_gain = 1000.0f;
    // Minimum pressure (Pa) to apply even if volume is below target (can be zero).
    real_t min_pressure = 0.0f;
    // External (ambient) pressure (Pa).  Net internal pressure = computed_pressure - external_pressure.
    real_t external_pressure = 0.0f;

    // -------------------------------------------------------------------
    // Compute the current signed volume of the given triangle mesh.
    // Assumes consistent winding: counter‑clockwise when viewed from
    // outside (normal = (v1-v0)×(v2-v0) points outward).
    // -------------------------------------------------------------------
    static real_t compute_volume(const LocalVector<Vector3> &p_vertices,
                                 const LocalVector<int> &p_triangles);

    // -------------------------------------------------------------------
    // Apply pressure forces to all surface vertices.  The pressure is
    // derived from the difference between current volume and target.
    // Forces are converted to velocity changes: v += force*dt/mass.
    // Pinned vertices (mask != 0) are skipped.
    // -------------------------------------------------------------------
    void apply_forces(const LocalVector<Vector3> &p_vertices,
                      LocalVector<Vector3> &p_velocities,
                      const LocalVector<real_t> &p_masses,
                      const LocalVector<int> &p_triangles,
                      const LocalVector<uint8_t> *p_pinned_mask,
                      real_t p_dt) const;

    // -------------------------------------------------------------------
    // Compute the pressure force vector (world space) and its application
    // point (centroid) on a single triangle, given the internal pressure
    // (Pa) and the external pressure.
    // -------------------------------------------------------------------
    static Vector3 compute_face_pressure_force(const Vector3 &p0, const Vector3 &p1,
                                               const Vector3 &p2,
                                               real_t p_internal_pressure,
                                               real_t p_external_pressure);

protected:
    static void _bind_methods();
};

// =========================================================================
// Inline implementations
// =========================================================================

real_t UnifiedSoftBodyPressureCavity::compute_volume(
        const LocalVector<Vector3> &p_vertices,
        const LocalVector<int> &p_triangles) {
    real_t V = 0.0f;
    int tri_count = p_triangles.size() / 3;
    for (int t = 0; t < tri_count; ++t) {
        int i0 = p_triangles[t*3];
        int i1 = p_triangles[t*3+1];
        int i2 = p_triangles[t*3+2];
        const Vector3 &v0 = p_vertices[i0];
        const Vector3 &v1 = p_vertices[i1];
        const Vector3 &v2 = p_vertices[i2];
        // Signed volume contribution = (v0 × v1)·v2 / 6
        V += v0.cross(v1).dot(v2);
    }
    return V / 6.0f;
}

Vector3 UnifiedSoftBodyPressureCavity::compute_face_pressure_force(
        const Vector3 &p0, const Vector3 &p1, const Vector3 &p2,
        real_t p_internal_pressure, real_t p_external_pressure) {
    // Face normal (not normalized), magnitude = 2 * area.
    Vector3 edge1 = p1 - p0;
    Vector3 edge2 = p2 - p0;
    Vector3 normal = edge1.cross(edge2);
    // Area = 0.5 * |normal|.
    real_t area = normal.length() * 0.5f;
    if (area < CMP_EPSILON) return Vector3();
    // Net pressure (positive = outward).
    real_t net_pressure = p_internal_pressure - p_external_pressure;
    // Force = pressure * area * unit_normal.
    // Unit normal = normal / (2*area).
    // So force = net_pressure * area * (normal / (2*area)) = net_pressure * normal / 2.
    // But careful: the pressure acts on the face with a force direction along the normal
    // (pointing outward). For triangle with vertices CCW, normal points outward.
    // So force vector = net_pressure * area * (normal / (2*area)) = net_pressure * normal * 0.5.
    return normal * (net_pressure * 0.5f);
}

void UnifiedSoftBodyPressureCavity::apply_forces(
        const LocalVector<Vector3> &p_vertices,
        LocalVector<Vector3> &p_velocities,
        const LocalVector<real_t> &p_masses,
        const LocalVector<int> &p_triangles,
        const LocalVector<uint8_t> *p_pinned_mask,
        real_t p_dt) const {

    int tri_count = p_triangles.size() / 3;
    if (tri_count == 0) return;

    // Current volume.
    real_t current_volume = compute_volume(p_vertices, p_triangles);
    // Proportional pressure.
    real_t pressure = pressure_gain * (target_volume - current_volume);
    // Clamp to min pressure.
    pressure = MAX(pressure, min_pressure);

    // For each triangle, compute force and distribute as impulse to vertices.
    for (int t = 0; t < tri_count; ++t) {
        int i0 = p_triangles[t*3];
        int i1 = p_triangles[t*3+1];
        int i2 = p_triangles[t*3+2];

        // Skip if any vertex is pinned.
        if (p_pinned_mask) {
            if ((*p_pinned_mask)[i0] && (*p_pinned_mask)[i1] && (*p_pinned_mask)[i2])
                continue;
        }

        const Vector3 &v0 = p_vertices[i0];
        const Vector3 &v1 = p_vertices[i1];
        const Vector3 &v2 = p_vertices[i2];

        Vector3 force = compute_face_pressure_force(v0, v1, v2, pressure, external_pressure);
        Vector3 impulse = force * p_dt; // force * dt

        // Divide equally among the three vertices.
        Vector3 per_vertex_impulse = impulse / 3.0f;

        // Apply to each unpinned vertex.
        auto apply = [&](int idx) {
            if (p_pinned_mask && (*p_pinned_mask)[idx]) return;
            real_t mass = (idx < p_masses.size()) ? p_masses[idx] : 1.0f;
            if (mass > CMP_EPSILON) {
                p_velocities[idx] += per_vertex_impulse / mass;
            }
        };
        apply(i0); apply(i1); apply(i2);
    }
}

void UnifiedSoftBodyPressureCavity::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_target_volume", "volume"), &UnifiedSoftBodyPressureCavity::set_target_volume);
    ClassDB::bind_method(D_METHOD("get_target_volume"), &UnifiedSoftBodyPressureCavity::get_target_volume);
    ClassDB::bind_method(D_METHOD("set_pressure_gain", "gain"), &UnifiedSoftBodyPressureCavity::set_pressure_gain);
    ClassDB::bind_method(D_METHOD("get_pressure_gain"), &UnifiedSoftBodyPressureCavity::get_pressure_gain);
    ClassDB::bind_method(D_METHOD("set_min_pressure", "min_p"), &UnifiedSoftBodyPressureCavity::set_min_pressure);
    ClassDB::bind_method(D_METHOD("get_min_pressure"), &UnifiedSoftBodyPressureCavity::get_min_pressure);
    ClassDB::bind_method(D_METHOD("set_external_pressure", "ext_p"), &UnifiedSoftBodyPressureCavity::set_external_pressure);
    ClassDB::bind_method(D_METHOD("get_external_pressure"), &UnifiedSoftBodyPressureCavity::get_external_pressure);
    ClassDB::bind_method(D_METHOD("compute_volume", "vertices", "triangles"), &UnifiedSoftBodyPressureCavity::compute_volume);
    ClassDB::bind_method(D_METHOD("apply_forces", "vertices", "velocities", "masses", "triangles", "pinned_mask", "dt"), &UnifiedSoftBodyPressureCavity::apply_forces);
    ClassDB::bind_method(D_METHOD("compute_face_pressure_force", "v0", "v1", "v2", "internal_pressure", "external_pressure"), &UnifiedSoftBodyPressureCavity::compute_face_pressure_force);

    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_volume"), "set_target_volume", "get_target_volume");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pressure_gain"), "set_pressure_gain", "get_pressure_gain");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_pressure"), "set_min_pressure", "get_min_pressure");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "external_pressure"), "set_external_pressure", "get_external_pressure");
}

// Property setters/getters (needed for ClassDB).
void UnifiedSoftBodyPressureCavity::set_target_volume(real_t v) { target_volume = v; }
real_t UnifiedSoftBodyPressureCavity::get_target_volume() const { return target_volume; }
void UnifiedSoftBodyPressureCavity::set_pressure_gain(real_t v) { pressure_gain = MAX(v, 0.0f); }
real_t UnifiedSoftBodyPressureCavity::get_pressure_gain() const { return pressure_gain; }
void UnifiedSoftBodyPressureCavity::set_min_pressure(real_t v) { min_pressure = MAX(v, 0.0f); }
real_t UnifiedSoftBodyPressureCavity::get_min_pressure() const { return min_pressure; }
void UnifiedSoftBodyPressureCavity::set_external_pressure(real_t v) { external_pressure = v; }
real_t UnifiedSoftBodyPressureCavity::get_external_pressure() const { return external_pressure; }

} // namespace unified

#endif // INTEGRATION_UNIFIED_SOFT_BODY_PRESSURE_CAVITY_H