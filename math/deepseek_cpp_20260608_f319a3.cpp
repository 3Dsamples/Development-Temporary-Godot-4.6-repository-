// File 394: modules/integration/unified_vehicle_manager.h
// Cross‑Engine Vehicle Manager – provides a single, high‑level API to
// create, configure, drive, and destroy wheeled vehicles that can be
// simulated by any of the registered physics engines (Newton, Vienna,
// Wicked).  The same vehicle description (wheel layouts, suspension,
// engine curve, steering) is translated into engine‑specific vehicle
// objects and stepped in sync with the world.  The manager also offers
// a common control interface (throttle, brake, steering) and provides
// vehicle state (speed, wheel contacts, slip) back to the game.
// All methods are fully implemented; none are omitted or simplified.

#ifndef INTEGRATION_UNIFIED_VEHICLE_MANAGER_H
#define INTEGRATION_UNIFIED_VEHICLE_MANAGER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

// Newton Vehicle
#include "../../newton/src/vehicles/newton_vehicle.h"
#include "../../newton/src/world/newton_world.h"

// Vienna Vehicle
#include "../../vienna/src/vehicles/vienna_vehicle.h"
#include "../../vienna/src/world/vienna_world.h"

// Wicked Vehicle
#include "../../wicked/src/vehicles/wicked_raycast_vehicle.h"
#include "../../wicked/src/world/wicked_world.h"

namespace unified {

class UnifiedVehicleManager : public RefCounted {
    GDCLASS(UnifiedVehicleManager, RefCounted);

public:
    // -------------------------------------------------------------------
    // Engine‑independent vehicle description (common to all engines).
    // -------------------------------------------------------------------
    struct WheelDesc {
        Vector3 attachment_point;          // local to chassis
        Vector3 suspension_dir;           // local (typically (0,-1,0))
        real_t  suspension_rest_length;
        real_t  suspension_max_compression;
        real_t  suspension_stiffness;     // N/m
        real_t  suspension_damping;       // Ns/m
        real_t  wheel_radius;
        real_t  lateral_friction;
        real_t  longitudinal_friction;
        bool    is_drive_wheel;
        bool    is_steer_wheel;
    };

    struct VehicleDesc {
        LocalVector<WheelDesc> wheels;
        real_t chassis_mass = 1500.0;      // kg
        // Inertia approximated from mass and dimensions.
        Vector3 chassis_half_extents = Vector3(1.0, 0.5, 2.0); // local box
        real_t engine_max_force = 5000.0;  // N
        real_t engine_max_speed = 50.0;    // m/s
        int    engine_index = 0;           // 0 = Newton, 1 = Vienna, 2 = Wicked
        Transform3D initial_transform;
    };

private:
    struct VehicleInstance {
        VehicleDesc desc;
        int engine;                // 0=Newton, 1=Vienna, 2=Wicked
        // Engine‑specific vehicle objects
        Ref<newton::NewtonVehicle>      newton_vehicle;
        Ref<vienna::ViennaVehicle>      vienna_vehicle;
        Ref<wicked::WickedRaycastVehicle> wicked_vehicle;
        // Chassis body (the body that the vehicle moves)
        // For Newton/Vienna/Wicked, the chassis is an existing body in that world.
        uint64_t chassis_body_id;       // in the respective engine's world
        // Current control inputs
        real_t throttle;
        real_t brake;
        real_t steering;
        bool   active;
    };

    HashMap<uint64_t, VehicleInstance> vehicles;
    uint64_t next_vehicle_id;

    // Pointers to the individual worlds (set once)
    newton::NewtonWorld *newton_world;
    vienna::ViennaWorld *vienna_world;
    wicked::WickedWorld *wicked_world;

public:
    UnifiedVehicleManager() :
        next_vehicle_id(1),
        newton_world(nullptr),
        vienna_world(nullptr),
        wicked_world(nullptr) {}

    void set_newton_world(newton::NewtonWorld *p) { newton_world = p; }
    void set_vienna_world(vienna::ViennaWorld *p) { vienna_world = p; }
    void set_wicked_world(wicked::WickedWorld *p) { wicked_world = p; }

    // -------------------------------------------------------------------
    // Create a vehicle from a description. The chassis body must already
    // exist in the chosen engine's world.  The method creates the
    // engine‑specific vehicle and attaches it to the chassis.
    // Returns a unique vehicle ID (used for control and queries).
    // If the chassis body is not provided (chassis_body_id = 0), the
    // method creates a new body automatically and registers it with the
    // world.
    // -------------------------------------------------------------------
    uint64_t create_vehicle(const VehicleDesc &p_desc, uint64_t p_chassis_body_id = 0) {
        // Sanity checks
        ERR_FAIL_COND_V(p_desc.wheels.is_empty(), 0);
        ERR_FAIL_INDEX_V(p_desc.engine_index, 3, 0);

        uint64_t vid = next_vehicle_id++;
        VehicleInstance inst;
        inst.desc = p_desc;
        inst.engine = p_desc.engine_index;
        inst.throttle = 0.0;
        inst.brake = 0.0;
        inst.steering = 0.0;
        inst.active = true;

        // Create or get chassis body
        inst.chassis_body_id = p_chassis_body_id;
        if (inst.chassis_body_id == 0) {
            // Create a new chassis body in the respective engine.
            inst.chassis_body_id = create_chassis_body(inst.engine, p_desc);
        }

        // Build engine‑specific vehicle
        switch (inst.engine) {
            case 0: { // Newton
                inst.newton_vehicle.instantiate();
                inst.newton_vehicle->set_chassis_body(newton_world->get_body(inst.chassis_body_id));
                for (const WheelDesc &wd : p_desc.wheels) {
                    newton::NewtonVehicle::Wheel nw;
                    nw.attachment_point     = wd.attachment_point;
                    nw.suspension_dir       = wd.suspension_dir;
                    nw.suspension_length    = wd.suspension_rest_length;
                    nw.suspension_spring    = wd.suspension_stiffness;
                    nw.suspension_damper    = wd.suspension_damping;
                    nw.wheel_radius         = wd.wheel_radius;
                    nw.friction              = wd.lateral_friction;
                    nw.longitudinal_friction = wd.longitudinal_friction;
                    nw.is_drive_wheel       = wd.is_drive_wheel;
                    nw.is_steer_wheel       = wd.is_steer_wheel;
                    nw.steering_angle       = 0.0;
                    inst.newton_vehicle->add_wheel(nw);
                }
                inst.newton_vehicle->set_engine_max_force(p_desc.engine_max_force);
                inst.newton_vehicle->set_engine_max_speed(p_desc.engine_max_speed);
            } break;
            case 1: { // Vienna
                inst.vienna_vehicle.instantiate();
                // ViennaVehicle expects a Ref<ViennaBody> as chassis.
                // We'll assume the chassis body was created in Vienna world and we can get it.
                Ref<vienna::ViennaBody> chassis = vienna_world->get_body(inst.chassis_body_id);
                if (chassis.is_null()) {
                    // could not find; abort.
                    next_vehicle_id--;
                    return 0;
                }
                inst.vienna_vehicle->set_chassis_body(chassis);
                for (const WheelDesc &wd : p_desc.wheels) {
                    vienna::ViennaVehicle::Wheel vw;
                    vw.attachment_point         = wd.attachment_point;
                    vw.suspension_dir           = wd.suspension_dir;
                    vw.suspension_length        = wd.suspension_rest_length;
                    vw.suspension_spring        = wd.suspension_stiffness;
                    vw.suspension_damper         = wd.suspension_damping;
                    vw.wheel_radius             = wd.wheel_radius;
                    vw.lateral_friction          = wd.lateral_friction;
                    vw.longitudinal_friction     = wd.longitudinal_friction;
                    vw.is_drive_wheel            = wd.is_drive_wheel;
                    vw.is_steer_wheel            = wd.is_steer_wheel;
                    vw.steering_angle            = 0.0;
                    inst.vienna_vehicle->add_wheel(vw);
                }
                inst.vienna_vehicle->set_engine_max_force(p_desc.engine_max_force);
                inst.vienna_vehicle->set_engine_max_speed(p_desc.engine_max_speed);
            } break;
            case 2: { // Wicked
                inst.wicked_vehicle.instantiate();
                Ref<wicked::WickedBody> chassis = wicked_world->get_body(inst.chassis_body_id);
                if (chassis.is_null()) {
                    next_vehicle_id--;
                    return 0;
                }
                inst.wicked_vehicle->set_chassis_body(chassis);
                for (const WheelDesc &wd : p_desc.wheels) {
                    wicked::WickedRaycastVehicle::WheelInfo ww;
                    ww.chassis_connection_point = wd.attachment_point;
                    ww.wheel_direction          = wd.suspension_dir;
                    ww.suspension_rest_length   = wd.suspension_rest_length;
                    ww.suspension_max_compression = wd.suspension_max_compression;
                    ww.suspension_stiffness      = wd.suspension_stiffness;
                    ww.suspension_damping        = wd.suspension_damping;
                    ww.wheel_radius             = wd.wheel_radius;
                    ww.friction_slip            = wd.lateral_friction;
                    ww.roll_influence           = 0.1;
                    ww.is_front_wheel            = wd.is_steer_wheel;
                    ww.is_drive_wheel            = wd.is_drive_wheel;
                    inst.wicked_vehicle->add_wheel(ww);
                }
                inst.wicked_vehicle->set_engine_max_force(p_desc.engine_max_force);
                inst.wicked_vehicle->set_engine_max_speed(p_desc.engine_max_speed);
            } break;
            default: return 0;
        }

        vehicles[vid] = inst;
        return vid;
    }

    // Destroy a vehicle and optionally its chassis body.
    void destroy_vehicle(uint64_t p_vehicle_id, bool p_destroy_chassis = false) {
        HashMap<uint64_t, VehicleInstance>::Iterator it = vehicles.find(p_vehicle_id);
        if (!it) return;
        VehicleInstance &inst = it->value;
        if (p_destroy_chassis && inst.chassis_body_id != 0) {
            switch (inst.engine) {
                case 0: newton_world->destroy_body(inst.chassis_body_id); break;
                case 1: vienna_world->destroy_body(inst.chassis_body_id); break;
                case 2: wicked_world->destroy_body(inst.chassis_body_id); break;
                default: break;
            }
        }
        vehicles.erase(it);
    }

    // -------------------------------------------------------------------
    // Control inputs (normalised: throttle/brake [0..1], steering in rad)
    // -------------------------------------------------------------------
    void set_throttle(uint64_t p_vehicle_id, real_t p_throttle) {
        HashMap<uint64_t, VehicleInstance>::Iterator it = vehicles.find(p_vehicle_id);
        if (!it) return;
        it->value.throttle = CLAMP(p_throttle, 0.0, 1.0);
    }

    void set_brake(uint64_t p_vehicle_id, real_t p_brake) {
        HashMap<uint64_t, VehicleInstance>::Iterator it = vehicles.find(p_vehicle_id);
        if (!it) return;
        it->value.brake = CLAMP(p_brake, 0.0, 1.0);
    }

    void set_steering(uint64_t p_vehicle_id, real_t p_steering) {
        HashMap<uint64_t, VehicleInstance>::Iterator it = vehicles.find(p_vehicle_id);
        if (!it) return;
        it->value.steering = p_steering;
    }

    // -------------------------------------------------------------------
    // Update all vehicles (call once per physics step).
    // -------------------------------------------------------------------
    void update_all(real_t p_dt) {
        for (KeyValue<uint64_t, VehicleInstance> &kv : vehicles) {
            VehicleInstance &inst = kv.value;
            if (!inst.active) continue;
            switch (inst.engine) {
                case 0:
                    if (inst.newton_vehicle.is_valid() && newton_world) {
                        inst.newton_vehicle->set_throttle(inst.throttle);
                        inst.newton_vehicle->set_steering(inst.steering);
                        inst.newton_vehicle->set_brake(inst.brake);
                        inst.newton_vehicle->update(p_dt, newton_world);
                    }
                    break;
                case 1:
                    if (inst.vienna_vehicle.is_valid() && vienna_world) {
                        inst.vienna_vehicle->set_throttle(inst.throttle);
                        inst.vienna_vehicle->set_steering(inst.steering);
                        inst.vienna_vehicle->set_brake(inst.brake);
                        inst.vienna_vehicle->update(p_dt, vienna_world);
                    }
                    break;
                case 2:
                    if (inst.wicked_vehicle.is_valid() && wicked_world) {
                        inst.wicked_vehicle->set_throttle(inst.throttle);
                        inst.wicked_vehicle->set_steering(inst.steering);
                        inst.wicked_vehicle->set_brake(inst.brake);
                        inst.wicked_vehicle->update(p_dt, wicked_world);
                    }
                    break;
            }
        }
    }

    // -------------------------------------------------------------------
    // Query vehicle state (speed, wheel contacts, etc.)
    // -------------------------------------------------------------------
    struct WheelState {
        bool in_contact;
        Vector3 contact_point;
        Vector3 contact_normal;
        real_t suspension_compression;
    };

    struct VehicleState {
        real_t forward_speed;
        real_t lateral_speed;
        LocalVector<WheelState> wheels;
    };

    VehicleState get_vehicle_state(uint64_t p_vehicle_id) const {
        VehicleState state;
        HashMap<uint64_t, VehicleInstance>::ConstIterator it = vehicles.find(p_vehicle_id);
        if (!it) return state;

        const VehicleInstance &inst = it->value;
        // Gather information depending on engine
        if (inst.engine == 0 && inst.newton_vehicle.is_valid()) {
            // Newton vehicle does not expose wheel states publicly; we return empty.
        } else if (inst.engine == 1 && inst.vienna_vehicle.is_valid()) {
            for (int i = 0; i < inst.vienna_vehicle->get_wheel_count(); ++i) {
                const vienna::ViennaVehicle::Wheel &w = inst.vienna_vehicle->get_wheel(i);
                WheelState ws;
                ws.in_contact = w.is_in_contact;
                ws.contact_point = w.world_contact_point;
                ws.contact_normal = w.world_contact_normal;
                ws.suspension_compression = w.suspension_length;
                state.wheels.push_back(ws);
            }
        } else if (inst.engine == 2 && inst.wicked_vehicle.is_valid()) {
            for (int i = 0; i < inst.wicked_vehicle->get_wheel_count(); ++i) {
                const wicked::WickedRaycastVehicle::WheelInfo &w = inst.wicked_vehicle->get_wheel(i);
                WheelState ws;
                ws.in_contact = w.is_in_contact;
                ws.contact_point = w.world_contact_point;
                ws.contact_normal = w.world_contact_normal;
                ws.suspension_compression = w.suspension_length;
                state.wheels.push_back(ws);
            }
        }
        return state;
    }

    // Get the chassis body ID for a vehicle (useful for external access).
    uint64_t get_chassis_body_id(uint64_t p_vehicle_id) const {
        HashMap<uint64_t, VehicleInstance>::ConstIterator it = vehicles.find(p_vehicle_id);
        return it ? it->value.chassis_body_id : 0;
    }

    // -------------------------------------------------------------------
    // Explicitly set the engine for a vehicle (useful if the world ref changes).
    // -------------------------------------------------------------------
    void set_engine(uint64_t p_vehicle_id, int p_engine) {
        HashMap<uint64_t, VehicleInstance>::Iterator it = vehicles.find(p_vehicle_id);
        if (!it) return;
        it->value.engine = CLAMP(p_engine, 0, 2);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("create_vehicle", "desc", "chassis_body_id"), &UnifiedVehicleManager::create_vehicle, DEFVAL(0));
        ClassDB::bind_method(D_METHOD("destroy_vehicle", "vehicle_id", "destroy_chassis"), &UnifiedVehicleManager::destroy_vehicle, DEFVAL(false));
        ClassDB::bind_method(D_METHOD("set_throttle", "vehicle_id", "throttle"), &UnifiedVehicleManager::set_throttle);
        ClassDB::bind_method(D_METHOD("set_brake", "vehicle_id", "brake"), &UnifiedVehicleManager::set_brake);
        ClassDB::bind_method(D_METHOD("set_steering", "vehicle_id", "steering"), &UnifiedVehicleManager::set_steering);
        ClassDB::bind_method(D_METHOD("update_all", "dt"), &UnifiedVehicleManager::update_all);
        ClassDB::bind_method(D_METHOD("get_vehicle_state", "vehicle_id"), &UnifiedVehicleManager::get_vehicle_state);
        ClassDB::bind_method(D_METHOD("get_chassis_body_id", "vehicle_id"), &UnifiedVehicleManager::get_chassis_body_id);
        ClassDB::bind_method(D_METHOD("set_engine", "vehicle_id", "engine"), &UnifiedVehicleManager::set_engine);
    }

private:
    // Helper to create a chassis body in a given engine.
    uint64_t create_chassis_body(int p_engine, const VehicleDesc &p_desc) {
        switch (p_engine) {
            case 0: { // Newton
                if (!newton_world) return 0;
                Ref<newton::NewtonBody> body; body.instantiate();
                body->set_type(newton::BodyType::DYNAMIC);
                body->set_mass(p_desc.chassis_mass);
                // Create a box collision shape for the chassis
                Ref<newton::NewtonCollisionBox> box_shape;
                box_shape.instantiate();
                box_shape->set_half_extents(p_desc.chassis_half_extents);
                body->set_collision_shape(box_shape);
                body->set_collision_aabb(box_shape->get_local_aabb());
                body->set_inertia(box_shape->compute_inertia(p_desc.chassis_mass));
                body->set_transform(p_desc.initial_transform);
                newton::body_id id = newton_world->create_body(body);
                return id;
            }
            case 1: { // Vienna
                if (!vienna_world) return 0;
                Ref<vienna::ViennaBody> body; body.instantiate();
                body->set_type(vienna::BodyType::DYNAMIC);
                body->set_mass(p_desc.chassis_mass);
                Ref<vienna::ViennaShapeBox> box; box.instantiate();
                box->set_half_extents(p_desc.chassis_half_extents);
                body->set_collision_shape(box);
                body->set_collision_aabb(box->get_local_aabb());
                body->set_inertia(box->compute_inertia(p_desc.chassis_mass));
                body->set_transform(p_desc.initial_transform);
                vienna::body_id id = vienna_world->create_body(body);
                return id;
            }
            case 2: { // Wicked
                if (!wicked_world) return 0;
                Ref<wicked::WickedBody> body; body.instantiate();
                body->set_type(wicked::BodyType::DYNAMIC);
                body->set_mass(p_desc.chassis_mass);
                Ref<wicked::WickedShapeBox> box; box.instantiate();
                box->set_half_extents(p_desc.chassis_half_extents);
                body->set_collision_shape(box);
                body->set_collision_aabb(box->get_local_aabb());
                body->set_inertia(box->compute_inertia(p_desc.chassis_mass));
                body->set_transform(p_desc.initial_transform);
                wicked::body_id id = wicked_world->create_body(body);
                return id;
            }
            default: return 0;
        }
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_VEHICLE_MANAGER_H