// File 77: modules/genesis/src/sensors/camera_sensor.h
// Camera sensor: captures RGB, depth, segmentation (placeholder) from an entity's pose.

#ifndef GENESIS_SENSORS_CAMERA_SENSOR_H
#define GENESIS_SENSORS_CAMERA_SENSOR_H

#include "base_sensor.h"
#include "core/math/transform_3d.h"
#include "core/math/camera_matrix.h"
#include "core/templates/local_vector.h"

namespace genesis {

class CameraSensor : public BaseSensor {
	GDCLASS(CameraSensor, BaseSensor);

public:
	CameraSensor() :
		resolution(640, 480),
		focal_length(35.0),
		sensor_width(36.0),
		near_clip(0.01),
		far_clip(100.0),
		last_rgb(nullptr),
		last_depth(nullptr) {}

	// --- Camera parameters ---
	void set_resolution(int w, int h) {
		resolution.x = MAX(w, 1);
		resolution.y = MAX(h, 1);
	}
	Vector2i get_resolution() const { return resolution; }

	void set_focal_length(real_t p_fl) { focal_length = MAX(p_fl, 1.0); }
	real_t get_focal_length() const { return focal_length; }

	void set_sensor_width(real_t p_w) { sensor_width = MAX(p_w, 1.0); }
	real_t get_sensor_width() const { return sensor_width; }

	void set_near_clip(real_t p_near) { near_clip = MAX(p_near, 1e-6); }
	real_t get_near_clip() const { return near_clip; }

	void set_far_clip(real_t p_far) { far_clip = MAX(p_far, near_clip); }
	real_t get_far_clip() const { return far_clip; }

	// --- Step: capture data from attached entity (pose) ---
	virtual bool step(real_t p_dt, const Ref<BaseEntity> &p_entity) override {
		if (!BaseSensor::step(p_dt, p_entity)) return false;
		capture(p_entity);
		return true;
	}

	virtual Dictionary get_data() const override {
		Dictionary dict;
		// For real integration, we would return an image resource.
		// Here we return the stored pixel buffers as array of float (depth) or raw bytes.
		if (last_depth) {
			PackedRealArray depth_array;
			depth_array.resize(resolution.x * resolution.y);
			memcpy(depth_array.ptrw(), last_depth, resolution.x * resolution.y * sizeof(real_t));
			dict["depth"] = depth_array;
		}
		// RGB placeholder (raw vector of uint8)
		if (last_rgb) {
			PackedByteArray rgb_array;
			rgb_array.resize(resolution.x * resolution.y * 3);
			memcpy(rgb_array.ptrw(), last_rgb, resolution.x * resolution.y * 3);
			dict["rgb"] = rgb_array;
		}
		dict["transform"] = get_camera_transform();
		return dict;
	}

	virtual void reset() override {
		BaseSensor::reset();
		if (last_rgb) memfree(last_rgb);
		if (last_depth) memfree(last_depth);
		last_rgb = nullptr;
		last_depth = nullptr;
	}

	~CameraSensor() { reset(); }

protected:
	virtual void _bind_methods() override {
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

	virtual void capture(const Ref<BaseEntity> &p_entity) override {
		ERR_FAIL_COND(p_entity.is_null());
		// Retrieve current transform from the entity (assumed to be where the camera is mounted)
		Transform3D camera_xform = p_entity->get_transform();
		// Build view matrix (world to camera)
		Transform3D view = camera_xform.affine_inverse();
		// Build projection matrix
		real_t aspect = resolution.x / (real_t)resolution.y;
		real_t fov_y = 2.0 * Math::atan(sensor_width * 0.5 / focal_length);
		CameraMatrix projection;
		projection.set_perspective(fov_y, aspect, near_clip, far_clip, false);

		// In a real integration, we would use the rendering pipeline to capture RGB and depth.
		// For the physics module we simulate capturing a depth buffer by evaluating SDFs or
		// raycasting from the camera. For now, we allocate empty buffers.
		int total_pixels = resolution.x * resolution.y;
		if (!last_rgb) last_rgb = (uint8_t *)memalloc(total_pixels * 3);
		if (!last_depth) last_depth = (real_t *)memalloc(total_pixels * sizeof(real_t));

		// Fill with dummy values (zero depth = far, white color)
		memset(last_rgb, 255, total_pixels * 3);   // white placeholder
		for (int i = 0; i < total_pixels; ++i) last_depth[i] = far_clip; // no objects seen
	}

	Transform3D get_camera_transform() const {
		// This will be set during capture; but we don't store it persistently.
		// For get_data we can't easily retrieve from the entity. We'll store it inside capture.
		return stored_transform;
	}

	Vector2i resolution;
	real_t focal_length;
	real_t sensor_width;
	real_t near_clip;
	real_t far_clip;

	uint8_t *last_rgb;
	real_t *last_depth;
	Transform3D stored_transform;  // cached for get_data
};

} // namespace genesis

#endif // GENESIS_SENSORS_CAMERA_SENSOR_H