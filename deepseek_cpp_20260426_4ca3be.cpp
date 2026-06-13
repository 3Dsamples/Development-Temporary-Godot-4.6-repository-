// File 125: modules/genesis/src/sensors/kinematic_tactile_sensor.h
// Kinematic Tactile Sensor: collects contact forces on a robot limb (tool entity)
// and maps them onto a surface grid to simulate tactile skin patches.
// Designed for use with kinematic chains and joint-level force sensing.

#ifndef GENESIS_SENSORS_KINEMATIC_TACTILE_SENSOR_H
#define GENESIS_SENSORS_KINEMATIC_TACTILE_SENSOR_H

#include "base_sensor.h"
#include "../entities/tool_entity.h"
#include "../core/genesis_types.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace genesis {

struct TactileCell {
	Vector3 world_point;          // centre of the tactile patch in world frame
	Vector3 local_point;          // in the local frame of the attached entity
	Vector3 normal;               // outward normal of the patch (world)
	Vector3 accumulated_force;    // total force accumulated over the last sensor interval
	int     contact_count;        // number of contact events this interval
};

class KinematicTactileSensor : public BaseSensor {
	GDCLASS(KinematicTactileSensor, BaseSensor);

public:
	KinematicTactileSensor() {
		sensor_type = SensorType::CUSTOM_SENSOR; // or JOINT, for tactile we could add TACTILE
	}

	// --- Tactile grid layout (simple rectangular patch array on a local surface) ---
	void set_grid_resolution(int p_cols, int p_rows) {
		cols = MAX(p_cols, 1);
		rows = MAX(p_rows, 1);
		rebuild_grid();
	}

	// --- Define the local surface rectangle (origin at corner, axes along local u and v) ---
	void set_surface_origin(const Vector3 &p_local_origin) { local_origin = p_local_origin; rebuild_grid(); }
	void set_surface_u(const Vector3 &p_u_axis) { local_u = p_u_axis; rebuild_grid(); }
	void set_surface_v(const Vector3 &p_v_axis) { local_v = p_v_axis; rebuild_grid(); }

	// --- Remove all accumulated forces at the start of each frame ---
	virtual void reset() override {
		BaseSensor::reset();
		for (TactileCell &cell : cells) {
			cell.accumulated_force = Vector3();
			cell.contact_count = 0;
		}
	}

	// --- Feed a contact event from the physics solver ---
	void add_contact(const Vector3 &p_world_point, const Vector3 &p_force, const Vector3 &p_normal) {
		// Map world point to local frame of the attached entity (done in step/capture)
		// For real-time use we need the entity's transform at the moment of contact,
		// which we do not have here (the sensor's capture runs at its own rate).
		// Instead, we store the raw contact and process in capture(). This allows
		// the solver to call add_contact at sub-stepping rate.
		RawContact c;
		c.world_point = p_world_point;
		c.force = p_force;
		c.normal = p_normal;
		raw_contacts.push_back(c);
	}

	// --- Clear raw contacts (call after processing) ---
	void clear_raw_contacts() { raw_contacts.clear(); }

	// --- Override capture to process raw contacts into grid cells ---
	virtual void capture(const Ref<BaseEntity> &p_entity) override {
		ERR_FAIL_COND(p_entity.is_null());
		Transform3D xform = p_entity->get_transform();

		// Reset grid
		for (TactileCell &cell : cells) {
			cell.accumulated_force = Vector3();
			cell.contact_count = 0;
		}

		// Transform grid cell centres to world space for easy binning
		LocalVector<Vector3> world_centers;
		world_centers.resize(cells.size());
		for (int i = 0; i < cells.size(); ++i) {
			world_centers[i] = xform.xform(cells[i].local_point);
		}

		// For each raw contact, assign to nearest grid cell
		for (const RawContact &rc : raw_contacts) {
			int best_idx = -1;
			real_t best_dist = INFINITY;
			for (int i = 0; i < world_centers.size(); ++i) {
				real_t d2 = world_centers[i].distance_squared_to(rc.world_point);
				if (d2 < best_dist) {
					best_dist = d2;
					best_idx = i;
				}
			}
			if (best_idx >= 0 && best_dist < max_bin_distance_sq) {
				cells[best_idx].accumulated_force += rc.force;
				cells[best_idx].contact_count++;
			}
		}
	}

	virtual Dictionary get_data() const override {
		Dictionary dict;
		Array info;
		for (int i = 0; i < cells.size(); ++i) {
			Dictionary cell_dict;
			cell_dict["local_point"] = cells[i].local_point;
			cell_dict["force"] = cells[i].accumulated_force;
			cell_dict["count"] = cells[i].contact_count;
			info.push_back(cell_dict);
		}
		dict["cells"] = info;
		dict["cols"] = cols;
		dict["rows"] = rows;
		return dict;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_grid_resolution", "cols", "rows"), &KinematicTactileSensor::set_grid_resolution);
		ClassDB::bind_method(D_METHOD("get_grid_resolution"), &KinematicTactileSensor::get_grid_resolution);
		ClassDB::bind_method(D_METHOD("set_surface_origin", "origin"), &KinematicTactileSensor::set_surface_origin);
		ClassDB::bind_method(D_METHOD("get_surface_origin"), &KinematicTactileSensor::get_surface_origin);
		ClassDB::bind_method(D_METHOD("set_surface_u", "axis"), &KinematicTactileSensor::set_surface_u);
		ClassDB::bind_method(D_METHOD("get_surface_u"), &KinematicTactileSensor::get_surface_u);
		ClassDB::bind_method(D_METHOD("set_surface_v", "axis"), &KinematicTactileSensor::set_surface_v);
		ClassDB::bind_method(D_METHOD("get_surface_v"), &KinematicTactileSensor::get_surface_v);
		ClassDB::bind_method(D_METHOD("add_contact", "world_point", "force", "normal"), &KinematicTactileSensor::add_contact);
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "resolution"), "set_grid_resolution", "get_grid_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "surface_origin"), "set_surface_origin", "get_surface_origin");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "surface_u"), "set_surface_u", "get_surface_u");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "surface_v"), "set_surface_v", "get_surface_v");
	}

private:
	void rebuild_grid() {
		int total = cols * rows;
		cells.resize(total);
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				// Parameteric coordinates in [0,1]
				real_t u = (real_t(c) + 0.5) / real_t(cols);
				real_t v = (real_t(r) + 0.5) / real_t(rows);
				cells[r * cols + c].local_point = local_origin + local_u * u + local_v * v;
				cells[r * cols + c].normal = local_u.cross(local_v).normalized(); // approximate outward normal
				cells[r * cols + c].accumulated_force = Vector3();
				cells[r * cols + c].contact_count = 0;
			}
		}
	}

	struct RawContact {
		Vector3 world_point;
		Vector3 force;
		Vector3 normal;
	};

	LocalVector<TactileCell> cells;
	LocalVector<RawContact> raw_contacts;

	int cols = 4;
	int rows = 4;
	Vector3 local_origin = Vector3();
	Vector3 local_u = Vector3(1, 0, 0);
	Vector3 local_v = Vector3(0, 0, 1);
	real_t max_bin_distance_sq = 0.01 * 0.01; // max distance to assign contact to cell
};

} // namespace genesis

#endif // GENESIS_SENSORS_KINEMATIC_TACTILE_SENSOR_H