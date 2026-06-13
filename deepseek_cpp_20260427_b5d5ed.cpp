// File 256: modules/newton/src/vehicles/newton_vehicle.cpp
// NewtonVehicle implementation – updates suspension, wheel forces, steering,
// engine torque, and braking. Uses ray‑casts against the world's static bodies
// to determine ground contact and applies forces to the chassis.

#include "newton_vehicle.h"
#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../collision/newton_collision.h"
#include "../core/newton_types.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/aabb.h"
#include "../../../gaia/src/bvh/query.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace newton {

void NewtonVehicle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_chassis_body", "body"), &NewtonVehicle::set_chassis_body);
	ClassDB::bind_method(D_METHOD("get_chassis_body"), &NewtonVehicle::get_chassis_body);
	ClassDB::bind_method(D_METHOD("add_wheel", "wheel"), &NewtonVehicle::add_wheel);
	ClassDB::bind_method(D_METHOD("get_wheel_count"), &NewtonVehicle::get_wheel_count);
	ClassDB::bind_method(D_METHOD("get_wheel", "idx"), &NewtonVehicle::get_wheel);
	ClassDB::bind_method(D_METHOD("clear_wheels"), &NewtonVehicle::clear_wheels);
	ClassDB::bind_method(D_METHOD("set_throttle", "throttle"), &NewtonVehicle::set_throttle);
	ClassDB::bind_method(D_METHOD("get_throttle"), &NewtonVehicle::get_throttle);
	ClassDB::bind_method(D_METHOD("set_steering", "steering"), &NewtonVehicle::set_steering);
	ClassDB::bind_method(D_METHOD("get_steering"), &NewtonVehicle::get_steering);
	ClassDB::bind_method(D_METHOD("set_brake", "brake"), &NewtonVehicle::set_brake);
	ClassDB::bind_method(D_METHOD("get_brake"), &NewtonVehicle::get_brake);
	ClassDB::bind_method(D_METHOD("set_engine_max_force", "force"), &NewtonVehicle::set_engine_max_force);
	ClassDB::bind_method(D_METHOD("get_engine_max_force"), &NewtonVehicle::get_engine_max_force);
	ClassDB::bind_method(D_METHOD("set_engine_max_speed", "speed"), &NewtonVehicle::set_engine_max_speed);
	ClassDB::bind_method(D_METHOD("get_engine_max_speed"), &NewtonVehicle::get_engine_max_speed);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "throttle"), "set_throttle", "get_throttle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "steering"), "set_steering", "get_steering");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brake"), "set_brake", "get_brake");
}

NewtonVehicle::NewtonVehicle() : chassis(nullptr), throttle(0.0), steering(0.0), brake(0.0),
	engine_max_force(5000.0), engine_max_speed(50.0) {}

void NewtonVehicle::set_chassis_body(NewtonBody *p_body) {
	chassis = p_body;
}

void NewtonVehicle::add_wheel(const Wheel &p_wheel) {
	wheels.push_back(p_wheel);
}

NewtonVehicle::Wheel &NewtonVehicle::get_wheel(int p_idx) {
	return wheels[p_idx];
}

void NewtonVehicle::clear_wheels() {
	wheels.clear();
}

void NewtonVehicle::set_throttle(real_t p_throttle) {
	throttle = CLAMP(p_throttle, 0.0, 1.0);
}

void NewtonVehicle::set_steering(real_t p_steering) {
	steering = p_steering;
}

void NewtonVehicle::set_brake(real_t p_brake) {
	brake = CLAMP(p_brake, 0.0, 1.0);
}

void NewtonVehicle::update(NewtonWorld *world, real_t dt) {
	if (!chassis || wheels.is_empty()) return;
	if (chassis->get_type() != BodyType::DYNAMIC) return;

	// Collect all static body AABBs and build a Gaia BVH for fast ray-casts
	LocalVector<body_id> all_ids = world->get_body_ids();
	LocalVector<AABB> static_aabbs;
	LocalVector<body_id> static_body_ids;
	for (body_id id : all_ids) {
		Ref<NewtonBody> body = world->get_body(id);
		if (body.is_valid() && body->get_type() == BodyType::STATIC && body->is_active()) {
			static_aabbs.push_back(body->get_aabb());
			static_body_ids.push_back(id);
		}
	}
	if (static_aabbs.is_empty()) return;

	gaia::bvh::BVH ground_bvh;
	ground_bvh.build_final(static_aabbs);

	const mat4 &chassisXform = chassis->get_transform();
	const vec3 &chassisVel = chassis->get_linear_velocity();
	const vec3 &chassisOmega = chassis->get_angular_velocity();

	for (Wheel &wheel : wheels) {
		// World‑space attachment point on chassis
		vec3 worldAttach = chassisXform.xform(wheel.attachment_point);
		// World suspension direction (downward)
		vec3 worldSuspDir = chassisXform.basis.xform(wheel.suspension_dir).normalized();

		// Ray length: rest suspension + wheel radius
		real_t rayLength = wheel.suspension_length + wheel.wheel_radius;
		vec3 rayStart = worldAttach;
		vec3 rayEnd   = worldAttach + worldSuspDir * rayLength;

		// Ray-test against ground BVH
		real_t best_t = rayLength;
		AABB rayAABB(rayStart, Vector3());
		rayAABB.expand_to(rayEnd);
		ground_bvh.query_intersect(rayAABB, [&](int prim) {
			const AABB &box = static_aabbs[prim];
			real_t t_entry = 0.0, t_exit = 0.0;
			if (gaia::bvh::intersect_ray_aabb(rayStart, worldSuspDir, box, 0.0, best_t, t_entry, t_exit)) {
				if (t_entry < best_t) best_t = t_entry;
			}
		});

		// No ground contact → wheel in the air → skip
		if (best_t >= rayLength) continue;

		vec3 hitPoint = rayStart + worldSuspDir * best_t;
		real_t compression = rayLength - best_t;
		if (compression < 0.0) compression = 0.0;

		// Velocity of the chassis at the wheel attachment point
		vec3 r = worldAttach - chassisXform.origin;
		vec3 attachVel = chassisVel + chassisOmega.cross(r);

		// Suspension spring‑damper force
		real_t springForce = wheel.suspension_spring * compression;
		real_t damperForce = wheel.suspension_damper * (attachVel.dot(worldSuspDir));
		real_t totalSpringForce = springForce + damperForce;
		if (totalSpringForce < 0.0) totalSpringForce = 0.0;

		// Apply upward force on chassis (opposite to suspension direction)
		chassis->apply_force(worldSuspDir * totalSpringForce, worldAttach);

		// Compute wheel local directions (forward and right) in world space
		vec3 chassisForward = chassisXform.basis.get_column(2).normalized();
		vec3 chassisRight   = chassisXform.basis.get_column(0).normalized();

		// Apply steering angle around the suspension axis
		vec3 wheelForward = chassisForward * Math::cos(wheel.steering_angle) +
							chassisRight * Math::sin(wheel.steering_angle);
		// Project onto ground plane (perpendicular to worldSuspDir)
		wheelForward = (wheelForward - worldSuspDir * wheelForward.dot(worldSuspDir)).normalized();
		vec3 wheelRight = worldSuspDir.cross(wheelForward).normalized();

		// Velocity of the wheel contact point
		vec3 contactVel = attachVel + chassisOmega.cross(hitPoint - chassisXform.origin);
		real_t forwardVel = contactVel.dot(wheelForward);
		real_t lateralVel = contactVel.dot(wheelRight);

		// Normal load approximation (spring force)
		real_t normalLoad = totalSpringForce;
		if (normalLoad < 1.0) normalLoad = 1.0;

		// Lateral friction force (Coulomb)
		real_t maxLateralFriction = wheel.friction * normalLoad;
		real_t lateralForce = -lateralVel * (maxLateralFriction / (Math::abs(lateralVel) + 0.01));
		lateralForce = CLAMP(lateralForce, -maxLateralFriction, maxLateralFriction);

		// Longitudinal force (engine + brake)
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
		// Clamp to traction limit
		real_t maxLongFriction = wheel.longitudinal_friction * normalLoad;
		longForce = CLAMP(longForce, -maxLongFriction, maxLongFriction);

		// Apply friction forces at the hit point
		vec3 totalFrictionForce = wheelForward * longForce + wheelRight * lateralForce;
		chassis->apply_force(totalFrictionForce, hitPoint);
	}
}

} // namespace newton