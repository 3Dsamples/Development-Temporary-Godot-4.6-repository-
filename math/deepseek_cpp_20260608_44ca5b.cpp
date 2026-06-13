// File 127: modules/genesis/src/sensors/temperature_grid_sensor.h
// Temperature grid sensor: samples a scalar temperature field on a regular
// 3D grid within a defined world region. The temperature values are computed
// from material thermal properties and solver coupling (heat conduction,
// convection, radiation). For now, it provides a read‑back interface that
// other modules can fill (e.g., the SF solver sets the temperature field).

#ifndef GENESIS_SENSORS_TEMPERATURE_GRID_SENSOR_H
#define GENESIS_SENSORS_TEMPERATURE_GRID_SENSOR_H

#include "base_sensor.h"
#include "../core/genesis_types.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/templates/local_vector.h"

namespace genesis {

class TemperatureGridSensor : public BaseSensor {
	GDCLASS(TemperatureGridSensor, BaseSensor);

public:
	TemperatureGridSensor() :
		grid_resolution(32, 32, 32),
		grid_origin(-1.0, -1.0, -1.0),
		grid_size(2.0, 2.0, 2.0),
		ambient_temperature(300.0)
	{
		sensor_type = SensorType::CUSTOM_SENSOR;
		allocate_grid();
	}

	// --- Grid geometry ---
	void set_grid_resolution(const Vector3i &p_res) {
		grid_resolution = p_res;
		grid_resolution.x = MAX(grid_resolution.x, 1);
		grid_resolution.y = MAX(grid_resolution.y, 1);
		grid_resolution.z = MAX(grid_resolution.z, 1);
		allocate_grid();
	}
	Vector3i get_grid_resolution() const { return grid_resolution; }

	void set_grid_origin(const Vector3 &p_origin) { grid_origin = p_origin; }
	Vector3 get_grid_origin() const { return grid_origin; }

	void set_grid_size(const Vector3 &p_size) {
		grid_size = p_size.abs();
	}
	Vector3 get_grid_size() const { return grid_size; }

	void set_ambient_temperature(real_t p_T) { ambient_temperature = p_T; }
	real_t get_ambient_temperature() const { return ambient_temperature; }

	// --- Direct access to the temperature array (for solver to write into) ---
	LocalVector<real_t> &get_temperature_array() { return temperature; }
	const LocalVector<real_t> &get_temperature_array() const { return temperature; }

	// Index mapping
	inline int index(int x, int y, int z) const {
		return z * (grid_resolution.y * grid_resolution.x) + y * grid_resolution.x + x;
	}

	inline Vector3 cell_center(int x, int y, int z) const {
		Vector3 delta = grid_size / Vector3(grid_resolution);
		return grid_origin + delta * Vector3(x + 0.5, y + 0.5, z + 0.5);
	}

	// --- Capture step: reads the temperature from an attached entity's material ---
	virtual void capture(const Ref<BaseEntity> &p_entity) override {
		// In a thermally coupled simulation, the solver would write into
		// temperature[] after each step. This sensor simply keeps the data
		// already present and optionally resets to ambient.
		// If no solver writes, we keep the previous values.
	}

	virtual Dictionary get_data() const override {
		Dictionary dict;
		PackedRealArray temp_packed;
		temp_packed.resize(temperature.size());
		for (int i = 0; i < temperature.size(); ++i) temp_packed.set(i, temperature[i]);
		dict["temperature"] = temp_packed;
		dict["resolution"] = grid_resolution;
		dict["origin"] = grid_origin;
		dict["size"] = grid_size;
		return dict;
	}

	virtual void reset() override {
		BaseSensor::reset();
		for (int i = 0; i < temperature.size(); ++i) temperature[i] = ambient_temperature;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_grid_resolution", "resolution"), &TemperatureGridSensor::set_grid_resolution);
		ClassDB::bind_method(D_METHOD("get_grid_resolution"), &TemperatureGridSensor::get_grid_resolution);
		ClassDB::bind_method(D_METHOD("set_grid_origin", "origin"), &TemperatureGridSensor::set_grid_origin);
		ClassDB::bind_method(D_METHOD("get_grid_origin"), &TemperatureGridSensor::get_grid_origin);
		ClassDB::bind_method(D_METHOD("set_grid_size", "size"), &TemperatureGridSensor::set_grid_size);
		ClassDB::bind_method(D_METHOD("get_grid_size"), &TemperatureGridSensor::get_grid_size);
		ClassDB::bind_method(D_METHOD("set_ambient_temperature", "T"), &TemperatureGridSensor::set_ambient_temperature);
		ClassDB::bind_method(D_METHOD("get_ambient_temperature"), &TemperatureGridSensor::get_ambient_temperature);
		ClassDB::bind_method(D_METHOD("get_temperature_array"), &TemperatureGridSensor::get_temperature_array);
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "resolution"), "set_grid_resolution", "get_grid_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "origin"), "set_grid_origin", "get_grid_origin");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_grid_size", "get_grid_size");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ambient_temperature"), "set_ambient_temperature", "get_ambient_temperature");
	}

private:
	void allocate_grid() {
		int total = grid_resolution.x * grid_resolution.y * grid_resolution.z;
		temperature.resize(total);
		for (int i = 0; i < total; ++i) temperature[i] = ambient_temperature;
	}

	Vector3i grid_resolution;
	Vector3 grid_origin;
	Vector3 grid_size;
	real_t ambient_temperature;
	LocalVector<real_t> temperature;
};

} // namespace genesis

#endif // GENESIS_SENSORS_TEMPERATURE_GRID_SENSOR_H