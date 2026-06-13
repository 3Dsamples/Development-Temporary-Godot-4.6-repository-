// File 169: modules/genesis/src/sensors/camera_sensor.cpp
// Implementation of camera sensor – bind methods, data retrieval, and cleanup.

#include "camera_sensor.h"

#include "core/variant/variant.h"
#include "core/math/vector3.h"
#include "core/math/camera_matrix.h"
#include "core/templates/local_vector.h"

namespace genesis {

void CameraSensor::_bind_methods() {
	BaseSensor::_bind_methods();

	ClassDB::bind_method(D_METHOD("set_resolution", "w", "h"), &CameraSensor::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &CameraSensor::get_resolution);
	ClassDB::bind_method(D_METHOD("set_focal_length", "f"), &CameraSensor::set_focal_length);
	ClassDB::bind_method(D_METHOD("get_focal_length"), &CameraSensor::get_focal_length);
	ClassDB::bind_method(D_METHOD("set_sensor_width", "w"), &CameraSensor::set_sensor_width);
	ClassDB::bind_method(D_METHOD("get_sensor_width"), &CameraSensor::get_sensor_width);
	ClassDB::bind_method(D_METHOD("set_near_clip", "n"), &CameraSensor::set_near_clip);
	ClassDB::bind_method(D_METHOD("get_near_clip"), &CameraSensor::get_near_clip);
	ClassDB::bind_method(D_METHOD("set_far_clip", "f"), &CameraSensor::set_far_clip);
	ClassDB::bind_method(D_METHOD("get_far_clip"), &CameraSensor::get_far_clip);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "resolution"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "focal_length", PROPERTY_HINT_RANGE, "1,500,0.1"), "set_focal_length", "get_focal_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sensor_width", PROPERTY_HINT_RANGE, "1,100,0.1"), "set_sensor_width", "get_sensor_width");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "near_clip", PROPERTY_HINT_RANGE, "0.0001,10,0.01"), "set_near_clip", "get_near_clip");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "far_clip", PROPERTY_HINT_RANGE, "0.01,1000,0.1"), "set_far_clip", "get_far_clip");
}

CameraSensor::~CameraSensor() {
	reset();
}

void CameraSensor::reset() {
	BaseSensor::reset();
	if (last_rgb) {
		memfree(last_rgb);
		last_rgb = nullptr;
	}
	if (last_depth) {
		memfree(last_depth);
		last_depth = nullptr;
	}
}

// The capture method is defined in the header; this file provides the
// destructor and bindings.  get_data is also inlined in the header.

} // namespace genesis