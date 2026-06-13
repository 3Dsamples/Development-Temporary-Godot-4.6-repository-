// File 346: modules/wicked/src/vehicles/wicked_raycast_vehicle.h
// High‑performance raycast vehicle for WickedEngine. Simulates wheels via
// ray‑casts against the ground (using Gaia BVH), suspension forces, tire
// friction (lateral and longitudinal), engine torque, steering, and braking.
// All hot‑path helpers are defined inline for maximum solver throughput.

#ifndef WICKED_VEHICLES_RAYCAST_VEHICLE_H
#define WICKED_VEHICLES_RAYCAST_VEHICLE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "../core/wicked_types.h"
#include "../core/wicked_constants.h"
#include "../bodies/wicked_body.h"
#include "../world/wicked_world.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/query.h"

namespace wicked {

class WickedRaycastVehicle : public RefCounted {
    GDCLASS(WickedRaycastVehicle, RefCounted);

public:
    struct WheelInfo {
        vec3 chassis_connection_point;      // local to chassis
        vec3 wheel_direction;               // suspension direction (down)
        vec3 wheel_axle;                    // lateral axis (right)
        real_t suspension_rest_length;      // max extension
        real_t suspension_max_compression;  // how far it can compress
        real_t suspension_stiffness;        // spring stiffness (N/m)
        real_t suspension_damping;          // damping coefficient (Ns/m)
        real_t wheel_radius;
        real_t friction_slip;               // lateral friction coefficient
        real_t roll_influence;              // rolling resistance influence
        bool is_front_wheel;                // steers
        bool is_drive_wheel;                // receives engine torque
        // runtime state
        vec3 world_contact_point;
        vec3 world_contact_normal;
        real_t suspension_length;           // current length
        bool is_in_contact;
    };

    WickedRaycastVehicle();
    void set_chassis_body(Ref<WickedBody> p_chassis);
    Ref<WickedBody> get_chassis_body() const { return chassis_body; }

    // Wheel management
    void add_wheel(const WheelInfo &p_wheel);
    int get_wheel_count() const { return wheels.size(); }
    WheelInfo &get_wheel(int p_idx) { return wheels[p_idx]; }
    void clear_wheels();

    // Control inputs
    void set_throttle(real_t p_throttle);   // 0..1
    void set_brake(real_t p_brake);         // 0..1
    void set_steering(real_t p_steering);   // radians
    real_t get_throttle() const { return throttle; }
    real_t get_brake() const { return brake; }
    real_t get_steering() const { return steering; }

    // Engine
    void set_engine_max_force(real_t p_force);
    void set_engine_max_speed(real_t p_speed);
    real_t get_engine_max_force() const { return engine_max_force; }
    real_t get_engine_max_speed() const { return engine_max_speed; }

    // Update vehicle physics for time dt (called each substep by world)
    void update(real_t dt, WickedWorld *p_world);

protected:
    static void _bind_methods();

private:
    // Ray‑casts a single wheel against the ground BVH, returns hit distance.
    real_t cast_wheel(const WheelInfo &p_wheel, const mat4 &p_chassis_xform,
                      const gaia::bvh::BVH &p_ground_bvh,
                      const LocalVector<body_id> &p_static_ids,
                      const WickedWorld *p_world,
                      vec3 &r_hit_point, vec3 &r_hit_normal) const;

    Ref<WickedBody> chassis_body;
    LocalVector<WheelInfo> wheels;
    real_t throttle;
    real_t brake;
    real_t steering;
    real_t engine_max_force;
    real_t engine_max_speed;
};

} // namespace wicked

#endif // WICKED_VEHICLES_RAYCAST_VEHICLE_H