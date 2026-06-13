// File 207: modules/newton/src/utils/newton_serializer.h
// Serialization utilities for Newton Dynamics bodies, joints, materials,
// and the entire world state. Produces JSON or binary representations
// suitable for saving / loading game scenes or physics simulation checkpoints.

#ifndef NEWTON_UTILS_SERIALIZER_H
#define NEWTON_UTILS_SERIALIZER_H

#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/variant/variant.h"
#include "core/templates/hash_map.h"
#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"
#include "../materials/newton_material.h"
#include "../collision/newton_collision.h"

namespace newton::serializer {

class NewtonSerializer {
public:
	// Save the entire world state to a JSON file.
	static Error save_world(const NewtonWorld *p_world, const String &p_path) {
		ERR_FAIL_COND_V(!p_world, ERR_INVALID_PARAMETER);
		Dictionary dict;
		// Serialize bodies
		Array bodies_arr;
		LocalVector<body_id> body_ids = p_world->get_body_ids();
		for (body_id id : body_ids) {
			Ref<NewtonBody> body = p_world->get_body(id);
			if (body.is_null()) continue;
			Dictionary body_dict;
			body_dict["id"] = id;
			body_dict["type"] = (int)body->get_type();
			body_dict["transform"] = body->get_transform();
			body_dict["linear_velocity"] = body->get_linear_velocity();
			body_dict["angular_velocity"] = body->get_angular_velocity();
			body_dict["mass"] = body->get_mass();
			body_dict["inertia"] = body->get_inertia_local();
			bodies_arr.push_back(body_dict);
		}
		dict["bodies"] = bodies_arr;

		// Serialize joints
		Array joints_arr;
		LocalVector<joint_id> joint_ids = p_world->get_joint_ids();
		for (joint_id jid : joint_ids) {
			Ref<NewtonJoint> joint = p_world->get_joint(jid);
			if (joint.is_null()) continue;
			Dictionary joint_dict;
			joint_dict["id"] = jid;
			joint_dict["type"] = (int)joint->get_joint_type();
			joint_dict["body_a"] = joint->get_body_a();
			joint_dict["body_b"] = joint->get_body_b();
			joint_dict["enabled"] = joint->is_enabled();
			joints_arr.push_back(joint_dict);
		}
		dict["joints"] = joints_arr;

		// Serialize materials
		Array materials_arr;
		LocalVector<material_id> material_ids = p_world->get_material_ids();
		for (material_id mid : material_ids) {
			Ref<NewtonMaterial> mat = p_world->get_material(mid);
			if (mat.is_null()) continue;
			Dictionary mat_dict;
			mat_dict["id"] = mid;
			mat_dict["static_friction"] = mat->get_static_friction();
			mat_dict["dynamic_friction"] = mat->get_dynamic_friction();
			mat_dict["restitution"] = mat->get_restitution();
			mat_dict["softness"] = mat->get_softness();
			materials_arr.push_back(mat_dict);
		}
		dict["materials"] = materials_arr;

		// World parameters
		dict["gravity"] = p_world->get_gravity();
		dict["solver_iterations"] = p_world->get_solver_iterations();
		dict["time"] = p_world->get_time();

		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
		if (f.is_null()) return ERR_FILE_CANT_WRITE;
		JSON json;
		String text = json.stringify(dict, "\t");
		f->store_string(text);
		return OK;
	}

	// Load world state from JSON and apply to an existing world.
	static Error load_world(NewtonWorld *p_world, const String &p_path) {
		ERR_FAIL_COND_V(!p_world, ERR_INVALID_PARAMETER);
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
		if (f.is_null()) return ERR_FILE_CANT_OPEN;
		String text = f->get_as_utf8_string();
		JSON json;
		Error err = json.parse(text);
		if (err != OK) return err;
		Dictionary dict = json.get_data();

		// Restore bodies
		if (dict.has("bodies")) {
			Array bodies_arr = dict["bodies"];
			for (int i = 0; i < bodies_arr.size(); ++i) {
				Dictionary bd = bodies_arr[i];
				body_id id = bd["id"];
				Ref<NewtonBody> body = p_world->get_body(id);
				if (body.is_null()) continue;
				body->set_type((BodyType)(int)bd["type"]);
				body->set_transform(bd["transform"]);
				body->set_linear_velocity(bd["linear_velocity"]);
				body->set_angular_velocity(bd["angular_velocity"]);
				body->set_mass(bd["mass"]);
				body->set_inertia(bd["inertia"]);
			}
		}

		// Restore joints
		if (dict.has("joints")) {
			Array joints_arr = dict["joints"];
			for (int i = 0; i < joints_arr.size(); ++i) {
				Dictionary jd = joints_arr[i];
				joint_id jid = jd["id"];
				Ref<NewtonJoint> joint = p_world->get_joint(jid);
				if (joint.is_null()) continue;
				joint->set_body_a(jd["body_a"]);
				joint->set_body_b(jd["body_b"]);
				joint->set_enabled(jd["enabled"]);
			}
		}

		// Restore materials
		if (dict.has("materials")) {
			Array mats_arr = dict["materials"];
			for (int i = 0; i < mats_arr.size(); ++i) {
				Dictionary md = mats_arr[i];
				material_id mid = md["id"];
				Ref<NewtonMaterial> mat = p_world->get_material(mid);
				if (mat.is_null()) continue;
				mat->set_static_friction(md["static_friction"]);
				mat->set_dynamic_friction(md["dynamic_friction"]);
				mat->set_restitution(md["restitution"]);
				mat->set_softness(md["softness"]);
			}
		}

		// Restore world globals
		if (dict.has("gravity")) {
			p_world->set_gravity(dict["gravity"]);
		}
		if (dict.has("solver_iterations")) {
			p_world->set_solver_iterations(dict["solver_iterations"]);
		}

		return OK;
	}
};

} // namespace newton::serializer

#endif // NEWTON_UTILS_SERIALIZER_H