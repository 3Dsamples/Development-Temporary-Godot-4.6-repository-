// File 316: modules/vienna/src/vehicles/vienna_vehicle.cpp
// ViennaVehicle implementation – ray‑casts wheels via Gaia BVH, applies
// suspension, friction, engine torque, steering, and braking forces.

#include "vienna_vehicle.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../collision/vienna_shape.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/query.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace vienna {

void ViennaVehicle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_chassis_body", "chassis"), &ViennaVehicle::set_chassis_body);
	ClassDB::bind_method(D_METHOD("get_chassis_body"), &ViennaVehicle::get_chassis_body);
	ClassDB::bind_method(D_METHOD("add_wheel", "wheel"), &ViennaVehicle::add_wheel);
	ClassDB::bind_method(D_METHOD("get_wheel_count"), &ViennaVehicle::get_wheel_count);
	ClassDB::bind_method(D_METHOD("get_wheel", "index"), &ViennaVehicle::get_wheel);
	ClassDB::bind_method(D_METHOD("clear_wheels"), &ViennaVehicle::clear_wheels);
	ClassDB::bind_method(D_METHOD("set_throttle", "throttle"), &ViennaVehicle::set_throttle);
	ClassDB::bind_method(D_METHOD("get_throttle"), &ViennaVehicle::get_throttle);
	ClassDB::bind_method(D_METHOD("set_steering", "steering"), &ViennaVehicle::set_steering);
	ClassDB::bind_method(D_METHOD("get_steering"), &ViennaVehicle::get_steering);
	ClassDB::bind_method(D_METHOD("set_brake", "brake"), &ViennaVehicle::set_brake);
	ClassDB::bind_method(D_METHOD("get_brake"), &ViennaVehicle::get_brake);
	ClassDB::bind_method(D_METHOD("set_engine_max_force", "force"), &ViennaVehicle::set_engine_max_force);
	ClassDB::bind_method(D_METHOD("get_engine_max_force"), &ViennaVehicle::get_engine_max_force);
	ClassDB::bind_method(D_METHOD("set_engine_max_speed", "speed"), &ViennaVehicle::set_engine_max_speed);
	ClassDB::bind_method(D_METHOD("get_engine_max_speed"), &ViennaVehicle::get_engine_max_speed);
	ClassDB::bind_method(D_METHOD("update", "dt", "world"), &ViennaVehicle::update);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "throttle"), "set_throttle", "get_throttle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "steering"), "set_steering", "get_steering");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brake"), "set_brake", "get_brake");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "engine_max_force"), "set_engine_max_force", "get_engine_max_force");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "engine_max_speed"), "set_engine_max_speed", "get_engine_max_speed");
}

ViennaVehicle::ViennaVehicle() :
	throttle(0.0), steering(0.0), brake(0.0),
	engine_max_force(5000.0), engine_max_speed(50.0) {}

void ViennaVehicle::set_chassis_body(Ref<ViennaBody> p_chassis) { chassis = p_chassis; }

void ViennaVehicle::add_wheel(const Wheel &p_wheel) { wheels.push_back(p_wheel); }

void ViennaVehicle::clear_wheels() { wheels.clear(); }

void ViennaVehicle::set_throttle(real_t p_throttle) { throttle = CLAMP(p_throttle, 0.0, 1.0); }
void ViennaVehicle::set_steering(real_t p_steering) { steering = p_steering; }
void ViennaVehicle::set_brake(real_t p_brake) { brake = CLAMP(p_brake, 0.0, 1.0); }
void ViennaVehicle::set_engine_max_force(real_t p_force) { engine_max_force = MAX(p_force, 0.0); }
void ViennaVehicle::set_engine_max_speed(real_t p_speed) { engine_max_speed = MAX(p_speed, 0.0); }

void ViennaVehicle::update(real_t dt, ViennaWorld *p_world) {
	if (chassis.is_null() || wheels.is_empty() || !p_world) return;
	if (chassis->get_type() != BodyType::DYNAMIC) return;

	// Collect AABBs of all static bodies (for ground) and build a Gaia BVH.
	gaia::bvh::BVH ground_bvh;
	LocalVector<body_id> body_ids = p_world->get_body_ids();
	LocalVector<AABB> static_aabbs;
	LocalVector<body_id> static_ids;
	for (body_id id : body_ids) {
		Ref<ViennaBody> body = p_world->get_body(id);
		if (body.is_valid() && body->get_type() == BodyType::STATIC && body->is_active()) {
			static_aabbs.push_back(body->get_aabb());
			static_ids.push_back(id);
		}
	}
	if (static_aabbs.is_empty()) return;
	ground_bvh.build_final(static_aabbs);

	const mat4 &chassisXform = chassis->get_transform();
	const vec3 &chassisVel = chassis->get_linear_velocity();
	const vec3 &chassisOmega = chassis->get_angular_velocity();

	for (Wheel &wheel : wheels) {
		// World‑space attachment point and suspension direction.
		vec3 worldAttach = chassisXform.xform(wheel.attachment_point);
		vec3 worldSuspDir = chassisXform.basis.xform(wheel.suspension_dir).normalized();

		// Ray length = rest length + wheel radius.
		real_t rayLen = wheel.suspension_length + wheel.wheel_radius;
		vec3 rayStart = worldAttach;
		vec3 rayEnd   = worldAttach + worldSuspDir * rayLen;

		// Ray‑test against ground BVH.
		real_t best_t = rayLen;
		AABB rayAABB(rayStart, Vector3());
		rayAABB.expand_to(rayEnd);
		ground_bvh.query_intersect(rayAABB, [&](int prim) {
			const AABB &box = static_aabbs[prim];
			real_t t_entry, t_exit;
			if (gaia::bvh::intersect_ray_aabb(rayStart, worldSuspDir, box, 0.0, best_t, t_entry, t_exit)) {
				if (t_entry < best_t) best_t = t_entry;
			}
		});

		// No ground contact? wheel in the air → skip.
		if (best_t >= rayLen) continue;

		vec3 hitPoint = rayStart + worldSuspDir * best_t;
		real_t compression = rayLen - best_t;
		if (compression < 0.0) compression = 0.0;

		// Velocity of chassis at attachment point.
		vec3 r = worldAttach - chassisXform.origin;
		vec3 attachVel = chassisVel + chassisOmega.cross(r);

		// Suspension force (spring + damper)
		real_t springForce = wheel.suspension_spring * compression;
		real_t damperForce = wheel.suspension_damper * attachVel.dot(worldSuspDir);
		real_t totalSpringForce = springForce + damperForce;
		if (totalSpringForce < 0.0) totalSpringForce = 0.0;

		// Apply upward force on chassis.
		chassis->apply_force(worldSuspDir * totalSpringForce, worldAttach);

		// Wheel local directions.
		vec3 chassisForward = chassisXform.basis.get_column(2).normalized(); // Z‑forward
		vec3 chassisRight   = chassisXform.basis.get_column(0).normalized();

		// Apply steering.
		vec3 wheelForward = chassisForward * Math::cos(wheel.steering_angle * steering) +
							chassisRight * Math::sin(wheel.steering_angle * steering);
		// Project onto ground plane.
		wheelForward = (wheelForward - worldSuspDir * wheelForward.dot(worldSuspDir)).normalized();
		vec3 wheelRight = worldSuspDir.cross(wheelForward).normalized();

		// Contact velocity.
		vec3 contactVel = attachVel + chassisOmega.cross(hitPoint - chassisXform.origin);
		real_t forwardVel = contactVel.dot(wheelForward);
		real_t lateralVel = contactVel.dot(wheelRight);

		// Normal load approximation.
		real_t normalLoad = MAX(totalSpringForce, 1.0);

		// Lateral friction (Coulomb).
		real_t maxLat = wheel.lateral_friction * normalLoad;
		real_t lateralForce = -lateralVel * (maxLat / (Math::abs(lateralVel) + 0.01));
		lateralForce = CLAMP(lateralForce, -maxLat, maxLat);

		// Longitudinal force: engine + brake.
		real_t longForce = 0.0;
		if (wheel.is_drive_wheel) {
			real_t targetSpeed = throttle * engine_max_speed;
			real_t speedError = targetSpeed - forwardVel;
			real_t driveForce = speedError * (engine_max_force / (engine_max_speed + 0.1));
			longForce += CLAMP(driveForce, -engine_max_force, engine_max_force);
		}
		if (brake > 0.0 && Math::abs(forwardVel) > 0.01) {
			real_t brakeForce = -forwardVel * brake * (engine_max_force * 0.5);
			longForce += brakeForce;
		}
		real_t maxLong = wheel.longitudinal_friction * normalLoad;
		longForce = CLAMP(longForce, -maxLong, maxLong);

		// Apply total friction force at hit point.
		vec3 frictionForce = wheelForward * longForce + wheelRight * lateralForce;
		chassis->apply_force(frictionForce, hitPoint);
	}
}

} // namespace vienna