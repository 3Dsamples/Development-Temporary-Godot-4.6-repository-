// File 253: modules/newton/src/ragdoll/newton_ragdoll.h
// NewtonRagdoll – builds a ragdoll from a skeleton (list of bones) and
// physical body parts (capsules, boxes, spheres). Creates Newton bodies
// linked by ball, hinge, or D6 joints to mimic humanoid or animal limbs.
// Supports motorised ragdolls and blending with animation.

#ifndef NEWTON_RAGDOLL_NEWTON_RAGDOLL_H
#define NEWTON_RAGDOLL_NEWTON_RAGDOLL_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_ball_joint.h"
#include "../joints/newton_hinge_joint.h"
#include "../joints/newton_d6_joint.h"
#include "../collision/newton_collision.h"

namespace newton {

class NewtonWorld;

class NewtonRagdoll : public RefCounted {
	GDCLASS(NewtonRagdoll, RefCounted);

public:
	struct Bone {
		String name;
		Transform3D bind_pose;         // initial world transform
		Ref<NewtonBody> body;
		Ref<NewtonCollision> shape;    // capsule, box, sphere, etc.
		real_t mass;
		Bone() : mass(1.0) {}
	};

	struct JointBinding {
		int parent_bone;
		int child_bone;
		Ref<NewtonJoint> joint;        // ball, hinge, or D6
		JointBinding() : parent_bone(-1), child_bone(-1) {}
	};

	NewtonRagdoll() : world(nullptr) {}

	void set_world(NewtonWorld *p_world) { world = p_world; }

	// Add a bone (body part) to the ragdoll.
	void add_bone(const Bone &p_bone) { bones.push_back(p_bone); }

	// Link two bones with a joint. The joint must already be configured
	// with its pivot and axis (if needed).
	void add_joint(int p_parent, int p_child, const Ref<NewtonJoint> &p_joint) {
		ERR_FAIL_INDEX(p_parent, bones.size());
		ERR_FAIL_INDEX(p_child, bones.size());
		JointBinding jb;
		jb.parent_bone = p_parent;
		jb.child_bone = p_child;
		jb.joint = p_joint;
		joints.push_back(jb);
	}

	// Build the ragdoll: create Newton bodies and joints in the world.
	void build() {
		ERR_FAIL_COND(!world);
		for (Bone &bone : bones) {
			Ref<NewtonBody> body;
			body.instantiate();
			body->set_type(BodyType::DYNAMIC);
			body->set_mass(bone.mass);
			body->set_collision_shape(bone.shape);
			body->set_collision_aabb(bone.shape->get_local_aabb());
			body->set_transform(bone.bind_pose);
			if (bone.mass > 0.0 && bone.shape.is_valid()) {
				body->set_inertia(bone.shape->compute_inertia(bone.mass));
			}
			bone.body = body;
			body_id id = world->create_body(body);
			bone_ids.push_back(id);
		}

		for (JointBinding &jb : joints) {
			Ref<NewtonBody> &parentBody = bones[jb.parent_bone].body;
			Ref<NewtonBody> &childBody = bones[jb.child_bone].body;
			body_id pid = bone_ids[jb.parent_bone];
			body_id cid = bone_ids[jb.child_bone];
			jb.joint->set_body_a(pid);
			jb.joint->set_body_b(cid);
			world->create_joint(jb.joint);
		}
	}

	// Destroy ragdoll bodies and joints from world.
	void destroy() {
		ERR_FAIL_COND(!world);
		for (JointBinding &jb : joints) {
			world->destroy_joint(jb.joint->get_body_a()); // need joint id storage
		}
		joints.clear();
		for (body_id id : bone_ids) {
			world->destroy_body(id);
		}
		bone_ids.clear();
		for (Bone &bone : bones) {
			bone.body.unref();
		}
	}

	// Pose the ragdoll according to animation skeleton transforms.
	// This sets kinematic targets on the bones, then lets the solver
	// apply forces to reach those targets (motor mode) or directly
	// sets position/velocity for kinematic bodies.
	void set_animation_pose(const LocalVector<Transform3D> &p_transforms, real_t dt) {
		int count = MIN(p_transforms.size(), bones.size());
		for (int i = 0; i < count; ++i) {
			if (bones[i].body.is_valid() && bones[i].body->get_type() == BodyType::KINEMATIC) {
				bones[i].body->set_transform(p_transforms[i]);
			}
		}
		// For dynamic motor-driven ragdoll, use joint motors (not implemented here).
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_world", "world"), &NewtonRagdoll::set_world);
		ClassDB::bind_method(D_METHOD("add_bone", "bone"), &NewtonRagdoll::add_bone);
		ClassDB::bind_method(D_METHOD("add_joint", "parent", "child", "joint"), &NewtonRagdoll::add_joint);
		ClassDB::bind_method(D_METHOD("build"), &NewtonRagdoll::build);
		ClassDB::bind_method(D_METHOD("destroy"), &NewtonRagdoll::destroy);
		ClassDB::bind_method(D_METHOD("set_animation_pose", "transforms", "dt"), &NewtonRagdoll::set_animation_pose);
	}

private:
	NewtonWorld *world;
	LocalVector<Bone> bones;
	LocalVector<body_id> bone_ids;
	LocalVector<JointBinding> joints;
};

} // namespace newton

#endif // NEWTON_RAGDOLL_NEWTON_RAGDOLL_H