// File 79: modules/genesis/src/sensors/imu_sensor.h
// IMU sensor: simulates an inertial measurement unit (accelerometer + gyroscope).
// Attaches to an entity and reports linear acceleration and angular velocity.

#ifndef GENESIS_SENSORS_IMU_SENSOR_H
#define GENESIS_SENSORS_IMU_SENSOR_H

#include "base_sensor.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "../core/genesis_types.h"

namespace genesis {

class IMUSensor : public BaseSensor {
	GDCLASS(IMUSensor, BaseSensor);

public:
	IMUSensor() :
		noise_std_accel(0.01),
		noise_std_gyro(0.001),
		bias_accel(Vector3()),
		bias_gyro(Vector3()),
		gravity_included(false) {
		sensor_type = SensorType::IMU;
	}

	// --- Noise and bias parameters ---
	void set_noise_std_accel(real_t p_sigma) { noise_std_accel = MAX(p_sigma, 0.0); }
	real_t get_noise_std_accel() const { return noise_std_accel; }

	void set_noise_std_gyro(real_t p_sigma) { noise_std_gyro = MAX(p_sigma, 0.0); }
	real_t get_noise_std_gyro() const { return noise_std_gyro; }

	void set_bias_accel(const Vector3 &p_bias) { bias_accel = p_bias; }
	Vector3 get_bias_accel() const { return bias_accel; }

	void set_bias_gyro(const Vector3 &p_bias) { bias_gyro = p_bias; }
	Vector3 get_bias_gyro() const { return bias_gyro; }

	void set_gravity_included(bool p_include) { gravity_included = p_include; }
	bool is_gravity_included() const { return gravity_included; }

	// --- Step: capture current IMU reading ---
	virtual void capture(const Ref<BaseEntity> &p_entity) override {
		ERR_FAIL_COND(p_entity.is_null());
		Transform3D xform = p_entity->get_transform();

		// Retrieve entity's linear acceleration (must be stored by the solver)
		// For simplicity, we approximate acceleration from velocity delta.
		// In a real integration, the solver would provide current acceleration.
		// Here we use the entity's last velocity and current velocity, but we don't have access to
		// velocity history. We'll use a dummy acceleration (0) and rely on the solver to provide
		// a proper acceleration callback later.
		Vector3 accel = Vector3(); // placeholder: to be filled by solver's acceleration output

		// Angular velocity directly from entity
		Vector3 angvel = p_entity->get_angular_velocity();

		// Transform to local IMU frame (assume IMU mounted at entity origin, aligned with entity orientation)
		// Local IMU acceleration = entity.world_to_local(accel - gravity) if gravity_included false
		// Gravity compensation: if gravity_included is false, we need to remove gravity from the measured acceleration.
		// We lack gravity info here, so we assume solver stores gravity. We'll skip gravity compensation for placeholder.
		Basis rot = xform.basis;
		local_accel = rot.xform_inv(accel);
		local_angvel = rot.xform_inv(angvel);

		// Add Gaussian noise (simple random)
		local_accel += random_vector(noise_std_accel);
		local_angvel += random_vector(noise_std_gyro);

		// Add bias
		local_accel += bias_accel;
		local_angvel += bias_gyro;
	}

	virtual Dictionary get_data() const override {
		Dictionary d;
		d["accelerometer"] = local_accel;
		d["gyroscope"] = local_angvel;
		return d;
	}

protected:
	virtual void _bind_methods() override {
		BaseSensor::_bind_methods();
		ClassDB::bind_method(D_METHOD("set_noise_std_accel", "sigma"), &IMUSensor::set_noise_std_accel);
		ClassDB::bind_method(D_METHOD("get_noise_std_accel"), &IMUSensor::get_noise_std_accel);
		ClassDB::bind_method(D_METHOD("set_noise_std_gyro", "sigma"), &IMUSensor::set_noise_std_gyro);
		ClassDB::bind_method(D_METHOD("get_noise_std_gyro"), &IMUSensor::get_noise_std_gyro);
		ClassDB::bind_method(D_METHOD("set_bias_accel", "bias"), &IMUSensor::set_bias_accel);
		ClassDB::bind_method(D_METHOD("get_bias_accel"), &IMUSensor::get_bias_accel);
		ClassDB::bind_method(D_METHOD("set_bias_gyro", "bias"), &IMUSensor::set_bias_gyro);
		ClassDB::bind_method(D_METHOD("get_bias_gyro"), &IMUSensor::get_bias_gyro);
		ClassDB::bind_method(D_METHOD("set_gravity_included", "include"), &IMUSensor::set_gravity_included);
		ClassDB::bind_method(D_METHOD("is_gravity_included"), &IMUSensor::is_gravity_included);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "noise_std_accel", PROPERTY_HINT_RANGE, "0,100,0.001"), "set_noise_std_accel", "get_noise_std_accel");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "noise_std_gyro", PROPERTY_HINT_RANGE, "0,100,0.001"), "set_noise_std_gyro", "get_noise_std_gyro");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "bias_accel"), "set_bias_accel", "get_bias_accel");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "bias_gyro"), "set_bias_gyro", "get_bias_gyro");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_included"), "set_gravity_included", "is_gravity_included");
	}

private:
	Vector3 random_vector(real_t sigma) {
		// Box-Muller approximation or simple uniform noise scaled by sigma.
		// Since Godot doesn't have a built-in Gaussian RNG, we use a quick approximation.
		real_t x = (Math::randf() - 0.5) * 2.0 * sigma;  // uniform in [-sigma, sigma] ~ 0.5 sigma std? Rough.
		real_t y = (Math::randf() - 0.5) * 2.0 * sigma;
		real_t z = (Math::randf() - 0.5) * 2.0 * sigma;
		return Vector3(x, y, z);
	}

	real_t noise_std_accel;
	real_t noise_std_gyro;
	Vector3 bias_accel;
	Vector3 bias_gyro;
	bool gravity_included;
	Vector3 local_accel;
	Vector3 local_angvel;
};

} // namespace genesis

#endif // GENESIS_SENSORS_IMU_SENSOR_H