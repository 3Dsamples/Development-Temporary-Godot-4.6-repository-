// File 76: modules/genesis/src/sensors/base_sensor.h
// Base class for all sensors. Sensors capture data from entities and the world.

#ifndef GENESIS_SENSORS_BASE_SENSOR_H
#define GENESIS_SENSORS_BASE_SENSOR_H

#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "../entities/base_entity.h"
#include "../core/genesis_types.h"

namespace genesis {

class BaseSensor : public Resource {
	GDCLASS(BaseSensor, Resource);

public:
	BaseSensor() :
		sensor_type(SensorType::CUSTOM_SENSOR),
		enabled(true),
		update_rate(60),
		entity_uid(0),
		time_accumulated(0.0) {}

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	bool is_enabled() const { return enabled; }

	void set_update_rate(int p_hz) { update_rate = MAX(p_hz, 1); }
	int get_update_rate() const { return update_rate; }

	void set_entity_uid(entity_id_t p_id) { entity_uid = p_id; }
	entity_id_t get_entity_uid() const { return entity_uid; }

	SensorType get_sensor_type() const { return sensor_type; }

	// Called every physics step; returns true if data was updated this step.
	virtual bool step(real_t p_dt, const Ref<BaseEntity> &p_entity) {
		if (!enabled) return false;
		time_accumulated += p_dt;
		real_t interval = 1.0 / update_rate;
		if (time_accumulated >= interval) {
			time_accumulated -= interval;
			capture(p_entity);
			return true;
		}
		return false;
	}

	// Return captured data as a Dictionary (overridden by specific sensors).
	virtual Dictionary get_data() const { return Dictionary(); }

	// Reset internal buffers
	virtual void reset() { time_accumulated = 0.0; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &BaseSensor::set_enabled);
		ClassDB::bind_method(D_METHOD("is_enabled"), &BaseSensor::is_enabled);
		ClassDB::bind_method(D_METHOD("set_update_rate", "hz"), &BaseSensor::set_update_rate);
		ClassDB::bind_method(D_METHOD("get_update_rate"), &BaseSensor::get_update_rate);
		ClassDB::bind_method(D_METHOD("set_entity_uid", "id"), &BaseSensor::set_entity_uid);
		ClassDB::bind_method(D_METHOD("get_entity_uid"), &BaseSensor::get_entity_uid);
		ClassDB::bind_method(D_METHOD("get_data"), &BaseSensor::get_data);
		ClassDB::bind_method(D_METHOD("reset"), &BaseSensor::reset);

		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "update_rate"), "set_update_rate", "get_update_rate");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "entity_uid"), "set_entity_uid", "get_entity_uid");
	}

	virtual void capture(const Ref<BaseEntity> &p_entity) = 0;

	SensorType sensor_type;
	bool enabled;
	int update_rate;
	entity_id_t entity_uid;
	real_t time_accumulated;
};

} // namespace genesis

#endif // GENESIS_SENSORS_BASE_SENSOR_H