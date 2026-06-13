// File 78: modules/genesis/src/sensors/contact_force_sensor.h
// Contact force sensor: accumulates and stores forces/torques applied to an entity.
// Integrates with the collision system via a callback interface.

#ifndef GENESIS_SENSORS_CONTACT_FORCE_SENSOR_H
#define GENESIS_SENSORS_CONTACT_FORCE_SENSOR_H

#include "base_sensor.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "../core/genesis_types.h"

namespace genesis {

struct ContactRecord {
	real_t timestamp;
	Vector3 force;      // world-space force vector
	Vector3 torque;     // world-space torque (about entity origin)
	Vector3 point;      // contact point in world
	Vector3 normal;     // world normal (pointing away from entity? depends)
};

class ContactForceSensor : public BaseSensor {
	GDCLASS(ContactForceSensor, BaseSensor);

public:
	ContactForceSensor() :
		max_history(1000),
		total_force(Vector3()),
		total_torque(Vector3()) {
		sensor_type = SensorType::CONTACT_FORCE;
	}

	void set_max_history(int p_max) { max_history = MAX(p_max, 1); }
	int get_max_history() const { return max_history; }

	// Feed a contact event from the solver. Called externally (e.g., by RigidSolver).
	void add_contact(const Vector3 &p_force, const Vector3 &p_torque,
					 const Vector3 &p_point, const Vector3 &p_normal) {
		ContactRecord rec;
		rec.timestamp = time_accumulated;   // current simulation time
		rec.force = p_force;
		rec.torque = p_torque;
		rec.point = p_point;
		rec.normal = p_normal;
		if (history.size() >= max_history) {
			history.remove_at(0);
		}
		history.push_back(rec);
		total_force += p_force;
		total_torque += p_torque;
	}

	// Override step to accumulate over a frame (optional)
	virtual bool step(real_t p_dt, const Ref<BaseEntity> &p_entity) override {
		if (!BaseSensor::step(p_dt, p_entity)) return false;
		capture(p_entity);
		return true;
	}

	virtual Dictionary get_data() const override {
		Dictionary dict;
		dict["count"] = history.size();
		Array forces;
		for (const ContactRecord &rec : history) {
			Dictionary entry;
			entry["t"] = rec.timestamp;
			entry["force"] = rec.force;
			entry["torque"] = rec.torque;
			entry["point"] = rec.point;
			entry["normal"] = rec.normal;
			forces.push_back(entry);
		}
		dict["history"] = forces;
		dict["total_force"] = total_force;
		dict["total_torque"] = total_torque;
		return dict;
	}

	virtual void reset() override {
		BaseSensor::reset();
		history.clear();
		total_force = Vector3();
		total_torque = Vector3();
	}

protected:
	virtual void capture(const Ref<BaseEntity> &p_entity) override {
		// The actual contact data is fed via add_contact(), not computed here.
		// We could clear the total accumulation if per‑step reset is desired.
		// For now, total is cumulative unless reset() is called.
	}

	virtual void _bind_methods() override {
		BaseSensor::_bind_methods();
		ClassDB::bind_method(D_METHOD("set_max_history", "max"), &ContactForceSensor::set_max_history);
		ClassDB::bind_method(D_METHOD("get_max_history"), &ContactForceSensor::get_max_history);
		ClassDB::bind_method(D_METHOD("add_contact", "force", "torque", "point", "normal"), &ContactForceSensor::add_contact);
		ADD_PROPERTY(PropertyInfo(Variant::INT, "max_history", PROPERTY_HINT_RANGE, "1,10000,1"), "set_max_history", "get_max_history");
	}

private:
	LocalVector<ContactRecord> history;
	int max_history;
	Vector3 total_force;
	Vector3 total_torque;
};

} // namespace genesis

#endif // GENESIS_SENSORS_CONTACT_FORCE_SENSOR_H