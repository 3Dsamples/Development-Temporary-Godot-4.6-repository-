// File 439: modules/integration/unified_wind_system.cpp
// Implementation of the UnifiedWindSystem.  All force computation and
// vertex update code is defined here.

#include "unified_wind_system.h"
#include "core/math/vector3.h"
#include "core/variant/variant.h"
#include "core/object/class_db.h"

namespace unified {

void UnifiedWindSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_wind_velocity", "velocity"), &UnifiedWindSystem::set_wind_velocity);
    ClassDB::bind_method(D_METHOD("get_wind_velocity"), &UnifiedWindSystem::get_wind_velocity);
    ClassDB::bind_method(D_METHOD("set_air_density", "density"), &UnifiedWindSystem::set_air_density);
    ClassDB::bind_method(D_METHOD("get_air_density"), &UnifiedWindSystem::get_air_density);
    ClassDB::bind_method(D_METHOD("set_drag_coefficient", "cd"), &UnifiedWindSystem::set_drag_coefficient);
    ClassDB::bind_method(D_METHOD("get_drag_coefficient"), &UnifiedWindSystem::get_drag_coefficient);
    ClassDB::bind_method(D_METHOD("set_lift_coefficient", "cl"), &UnifiedWindSystem::set_lift_coefficient);
    ClassDB::bind_method(D_METHOD("get_lift_coefficient"), &UnifiedWindSystem::get_lift_coefficient);
    ClassDB::bind_method(D_METHOD("set_turbulence_intensity", "intensity"), &UnifiedWindSystem::set_turbulence_intensity);
    ClassDB::bind_method(D_METHOD("get_turbulence_intensity"), &UnifiedWindSystem::get_turbulence_intensity);
    ClassDB::bind_method(D_METHOD("set_noise_scale", "scale"), &UnifiedWindSystem::set_noise_scale);
    ClassDB::bind_method(D_METHOD("get_noise_scale"), &UnifiedWindSystem::get_noise_scale);
    ClassDB::bind_method(D_METHOD("set_noise_octaves", "octaves"), &UnifiedWindSystem::set_noise_octaves);
    ClassDB::bind_method(D_METHOD("get_noise_octaves"), &UnifiedWindSystem::get_noise_octaves);
    ClassDB::bind_method(D_METHOD("apply_forces", "vertices", "velocities", "masses", "triangles", "dt", "pinned_mask"),
        &UnifiedWindSystem::apply_forces, DEFVAL(nullptr));
    ClassDB::bind_method(D_METHOD("compute_face_force", "p0", "p1", "p2", "vel0", "vel1", "vel2"),
        &UnifiedWindSystem::compute_face_force);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "wind_velocity"), "set_wind_velocity", "get_wind_velocity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "air_density"), "set_air_density", "get_air_density");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "drag_coefficient"), "set_drag_coefficient", "get_drag_coefficient");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lift_coefficient"), "set_lift_coefficient", "get_lift_coefficient");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "turbulence_intensity", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_turbulence_intensity", "get_turbulence_intensity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "noise_scale"), "set_noise_scale", "get_noise_scale");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "noise_octaves", PROPERTY_HINT_RANGE, "1,5,1"), "set_noise_octaves", "get_noise_octaves");
}

// ---------------------------------------------------------------------------
// Turbulence: a simple pseudo‑random 3D offset based on position.
// Uses sinusoidal functions to mimic Perlin‑like variation.
// ---------------------------------------------------------------------------
Vector3 UnifiedWindSystem::turbulence_offset(const Vector3 &p_world_pos) const {
    real_t scale = noise_scale;
    Vector3 p = p_world_pos * scale;
    real_t val = 0.0f;
    real_t amp = 1.0f;
    real_t freq = 1.0f;
    for (int i = 0; i < noise_octaves; ++i) {
        val += amp * Math::sin(p.x * freq) * Math::cos(p.z * freq) * Math::sin(p.y * freq * 1.7f);
        amp *= 0.5f;
        freq *= 2.0f;
    }
    // Use three shifted versions for X,Y,Z turbulence.
    real_t tx = Math::sin(p.x * 3.1f + 1.0f) * Math::cos(p.y * 2.7f) * Math::sin(p.z * 1.3f);
    real_t ty = Math::cos(p.x * 2.3f) * Math::sin(p.y * 3.7f + 2.0f) * Math::cos(p.z * 1.1f);
    real_t tz = Math::sin(p.x * 1.9f) * Math::cos(p.y * 1.3f) * Math::sin(p.z * 3.1f + 0.5f);
    return Vector3(tx, ty, tz) * turbulence_intensity * wind_velocity.length() * 0.5f;
}

// ---------------------------------------------------------------------------
// Compute force on a single triangle.
// ---------------------------------------------------------------------------
Vector3 UnifiedWindSystem::compute_face_force(const Vector3 &p0, const Vector3 &p1,
                                               const Vector3 &p2,
                                               const Vector3 &p_vel0,
                                               const Vector3 &p_vel1,
                                               const Vector3 &p_vel2) const {
    // Triangle centroid.
    Vector3 centroid = (p0 + p1 + p2) / 3.0f;
    // Average velocity of the triangle.
    Vector3 avg_vel = (p_vel0 + p_vel1 + p_vel2) / 3.0f;

    // Face normal (not necessarily unit – area = 0.5 * |cross|).
    Vector3 edge1 = p1 - p0;
    Vector3 edge2 = p2 - p0;
    Vector3 face_normal = edge1.cross(edge2);
    real_t area = face_normal.length() * 0.5f;
    if (area < CMP_EPSILON) return Vector3();

    Vector3 unit_normal = face_normal / (area * 2.0f); // length = 2*area, so unit.
    // Wind velocity at centroid = base wind + turbulence.
    Vector3 wind_at = wind_velocity + turbulence_offset(centroid);
    // Relative velocity of wind to surface.
    Vector3 v_rel = wind_at - avg_vel;

    // Drag: F_drag = 0.5 * rho * Cd * area * |v_rel| * v_rel
    real_t speed = v_rel.length();
    if (speed < CMP_EPSILON) return Vector3();

    real_t drag_mag = 0.5f * air_density * drag_coefficient * area * speed;
    Vector3 drag_force = drag_mag * v_rel;

    // Lift: perpendicular to wind direction and surface normal.
    // F_lift = 0.5 * rho * Cl * area * |v_rel|^2 * (v_rel x n) x v_rel normalized.
    Vector3 lift_force;
    if (lift_coefficient != 0.0f) {
        Vector3 lift_dir = v_rel.cross(unit_normal).cross(v_rel);
        real_t lift_dir_len = lift_dir.length();
        if (lift_dir_len > CMP_EPSILON) {
            lift_dir /= lift_dir_len;
            lift_force = 0.5f * air_density * lift_coefficient * area * speed * lift_dir;
        }
    }

    // Total force applied to the face.
    return drag_force + lift_force;
}

// ---------------------------------------------------------------------------
// Apply forces to all triangles, updating per‑vertex velocities.
// ---------------------------------------------------------------------------
void UnifiedWindSystem::apply_forces(LocalVector<Vector3> &p_vertices,
                                      LocalVector<Vector3> &p_velocities,
                                      const LocalVector<real_t> &p_masses,
                                      const LocalVector<int> &p_triangles,
                                      real_t p_dt,
                                      const LocalVector<uint8_t> *p_pinned_mask) const {
    int n_verts = p_vertices.size();
    if (n_verts == 0 || p_triangles.is_empty()) return;

    // Accumulated impulses per vertex (force * dt).
    LocalVector<Vector3> impulses(n_verts, Vector3());

    int tri_count = p_triangles.size() / 3;
    for (int t = 0; t < tri_count; ++t) {
        int i0 = p_triangles[t * 3];
        int i1 = p_triangles[t * 3 + 1];
        int i2 = p_triangles[t * 3 + 2];
        if (i0 >= n_verts || i1 >= n_verts || i2 >= n_verts) continue;

        // Skip pinned triangles if mask provided and any vertex pinned.
        if (p_pinned_mask) {
            if ((*p_pinned_mask)[i0] || (*p_pinned_mask)[i1] || (*p_pinned_mask)[i2])
                continue;
        }

        Vector3 force = compute_face_force(p_vertices[i0], p_vertices[i1], p_vertices[i2],
                                           p_velocities[i0], p_velocities[i1], p_velocities[i2]);
        Vector3 impulse = force * p_dt;
        // Distribute equally to the three vertices.
        Vector3 per_vertex = impulse / 3.0f;
        impulses[i0] += per_vertex;
        impulses[i1] += per_vertex;
        impulses[i2] += per_vertex;
    }

    // Apply impulses as velocity changes.
    for (int i = 0; i < n_verts; ++i) {
        if (p_pinned_mask && (*p_pinned_mask)[i]) continue;
        real_t mass = (i < p_masses.size()) ? p_masses[i] : 1.0f;
        if (mass > CMP_EPSILON) {
            p_velocities[i] += impulses[i] / mass;
        }
    }
}

} // namespace unified