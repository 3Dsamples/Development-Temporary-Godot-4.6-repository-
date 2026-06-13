// File 85: modules/genesis/src/solvers/kinematic_solver.h
// Kinematic (avatar) solver – drives entities via prescribed trajectories,
// keyframes, or external animation data. Used for animated characters and robots.

#ifndef GENESIS_SOLVERS_KINEMATIC_SOLVER_H
#define GENESIS_SOLVERS_KINEMATIC_SOLVER_H

#include "base_solver.h"
#include "../entities/base_entity.h"
#include "../entities/rigid_entity.h"
#include "../entities/tool_entity.h"
#include "../core/genesis_types.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"

namespace genesis {

class KinematicSolver : public BaseSolver {
	GDCLASS(KinematicSolver, BaseSolver);

public:
	KinematicSolver() : BaseSolver() {}

	struct Keyframe {
		real_t timestamp;
		Vector3 position;
		Vector3 orientation_euler; // euler angles in radians
		Vector3 linear_velocity;
		Vector3 angular_velocity;
	};

	// Attach a kinematic trajectory to an entity.
	void set_keyframes(entity_id_t p_uid, const LocalVector<Keyframe> &p_keyframes) {
		keyframe_data[p_uid] = p_keyframes;
		keyframe_index[p_uid] = 0;
	}

	// Add a single keyframe at the end of the trajectory.
	void add_keyframe(entity_id_t p_uid, const Keyframe &p_kf) {
		keyframe_data[p_uid].push_back(p_kf);
	}

	// Clear all keyframes for an entity.
	void clear_keyframes(entity_id_t p_uid) {
		keyframe_data.erase(p_uid);
		keyframe_index.erase(p_uid);
	}

	virtual void step() override {
		real_t sub_dt = dt / real_t(sub_steps);
		for (int substep = 0; substep < sub_steps; ++substep) {
			interpolate_entities(sub_dt);
			time += sub_dt;
		}
	}

	virtual void solve(real_t p_sub_dt) override {
		// Interpolation is done in step; nothing extra needed here.
	}

private:
	void interpolate_entities(real_t p_sub_dt) {
		for (KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
			entity_id_t uid = kv.key;
			Ref<BaseEntity> entity = kv.value;
			if (entity.is_null() || !entity->is_active()) continue;

			// Only affect kinematic-type entities or tool entities
			if (!keyframe_data.has(uid)) continue;

			LocalVector<Keyframe> &kfs = keyframe_data[uid];
			int &idx = keyframe_index[uid];
			if (kfs.is_empty()) continue;

			// Advance index to current time
			while (idx + 1 < kfs.size() && kfs[idx + 1].timestamp <= time) {
				idx++;
			}
			// If at last keyframe, hold
			if (idx >= kfs.size() - 1) {
				apply_keyframe(entity, kfs[kfs.size() - 1]);
				continue;
			}
			// Interpolate between idx and idx+1
			const Keyframe &kf0 = kfs[idx];
			const Keyframe &kf1 = kfs[idx + 1];
			real_t t = (time - kf0.timestamp) / MAX(kf1.timestamp - kf0.timestamp, CMP_EPSILON);
			t = CLAMP(t, 0.0, 1.0);

			Keyframe interp;
			interp.position = kf0.position.lerp(kf1.position, t);
			interp.orientation_euler = kf0.orientation_euler.lerp(kf1.orientation_euler, t);
			interp.linear_velocity = kf0.linear_velocity.lerp(kf1.linear_velocity, t);
			interp.angular_velocity = kf0.angular_velocity.lerp(kf1.angular_velocity, t);
			apply_keyframe(entity, interp);
		}
	}

	void apply_keyframe(const Ref<BaseEntity> &entity, const Keyframe &kf) {
		Basis rot = Basis::from_euler(kf.orientation_euler);
		entity->set_position(kf.position);
		entity->set_rotation(rot);
		entity->set_linear_velocity(kf.linear_velocity);
		entity->set_angular_velocity(kf.angular_velocity);
	}

	HashMap<entity_id_t, LocalVector<Keyframe>> keyframe_data;
	HashMap<entity_id_t, int> keyframe_index;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_keyframes", "uid", "keyframes"), &KinematicSolver::set_keyframes);
		ClassDB::bind_method(D_METHOD("add_keyframe", "uid", "keyframe"), &KinematicSolver::add_keyframe);
		ClassDB::bind_method(D_METHOD("clear_keyframes", "uid"), &KinematicSolver::clear_keyframes);
	}
};

} // namespace genesis

#endif // GENESIS_SOLVERS_KINEMATIC_SOLVER_H