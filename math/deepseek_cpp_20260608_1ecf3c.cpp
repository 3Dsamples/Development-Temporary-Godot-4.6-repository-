// File 318: modules/vienna/src/ragdoll/vienna_ragdoll.cpp
// Implementation of ViennaRagdoll – creates rigid bodies and joints for a
// ragdoll, supports kinematic posing, force application, and destruction.

#include "vienna_ragdoll.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../joints/vienna_ball_joint.h"
#include "../joints/vienna_hinge_joint.h"
#include "../joints/vienna_fixed_joint.h"
#include "../collision/vienna_shape.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

namespace vienna {

void ViennaRagdoll::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_world", "world"), &ViennaRagdoll::set_world);
	ClassDB::bind_method(D_METHOD("get_world"), &ViennaRagdoll::get_world);
	ClassDB::bind_method(D_METHOD("add_bone", "bone"), &ViennaRagdoll::add_bone);
	ClassDB::bind_method(D_METHOD("add_joint", "parent", "child", "joint"), &ViennaRagdoll::add_joint);
	ClassDB::bind_method(D_METHOD("build"), &ViennaRagdoll::build);
	ClassDB::bind_method(D_METHOD("destroy"), &ViennaRagdoll::destroy);
	ClassDB::bind_method(D_METHOD("set_animation_pose", "transforms"), &ViennaRagdoll::set_animation_pose);
	ClassDB::bind_method(D_METHOD("set_kinematic", "kinematic"), &ViennaRagdoll::set_kinematic);
	ClassDB::bind_method(D_METHOD("is_kinematic"), &ViennaRagdoll::is_kinematic);
	ClassDB::bind_method(D_METHOD("apply_force_to_all", "force"), &ViennaRagdoll::apply_force_to_all);
	ClassDB::bind_method(D_METHOD("get_bone_count"), &ViennaRagdoll::get_bone_count);
	ClassDB::bind_method(D_METHOD("get_bone", "index"), &ViennaRagdoll::get_bone);
	ClassDB::bind_method(D_METHOD("get_body_id", "index"), &ViennaRagdoll::get_body_id);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "kinematic"), "set_kinematic", "is_kinematic");
}

ViennaRagdoll::ViennaRagdoll() : world(nullptr), built(false), kinematic(false) {}

void ViennaRagdoll::set_world(ViennaWorld *p_world) { world = p_world; }

void ViennaRagdoll::add_bone(const Bone &p_bone) {
	ERR_FAIL_COND(built); // cannot add bones after build
	bones.push_back(p_bone);
}

void ViennaRagdoll::add_joint(int p_parent, int p_child, const JointBinding &p_joint) {
	ERR_FAIL_COND(built);
	ERR_FAIL_INDEX(p_parent, bones.size());
	ERR_FAIL_INDEX(p_child, bones.size());
	joints.push_back(p_joint);
}

void ViennaRagdoll::build() {
	ERR_FAIL_COND(!world);
	ERR_FAIL_COND(built);
	int nb = bones.size();
	bone_ids.resize(nb);
	// Create a ViennaBody for each bone
	for (int i = 0; i < nb; ++i) {
		Bone &bone = bones[i];
		Ref<ViennaBody> body;
		body.instantiate();
		body->set_type(kinematic ? BodyType::KINEMATIC : BodyType::DYNAMIC);
		body->set_mass(bone.mass);
		// Attach the collision shape (must be sphere, capsule, or box)
		if (bone.shape.is_valid()) {
			body->set_collision_shape(bone.shape);
			body->set_collision_aabb(bone.shape->get_local_aabb());
			body->set_inertia(bone.shape->compute_inertia(bone.mass));
		}
		body->set_transform(bone.bind_pose);
		bone.body = body;
		body_id id = world->create_body(body);
		bone_ids[i] = id;
	}

	// Create joints between bones
	joint_ids.resize(joints.size());
	for (int j = 0; j < joints.size(); ++j) {
		const JointBinding &jb = joints[j];
		Ref<ViennaJoint> joint;
		switch (jb.joint_type) {
			case JointType::BALL: {
				Ref<ViennaBallJoint> ball = memnew(ViennaBallJoint);
				ball->set_pivot(jb.pivot);
				// Optionally set cone/twist limits from jb if available; omitted for brevity.
				joint = ball;
			} break;
			case JointType::HINGE: {
				Ref<ViennaHingeJoint> hinge = memnew(ViennaHingeJoint);
				hinge->set_pivot(jb.pivot);
				hinge->set_axis(jb.axis);
				if (jb.min_angle != 0.0f || jb.max_angle != 0.0f) {
					hinge->set_limit_enabled(true);
					hinge->set_limit_angle(jb.min_angle, jb.max_angle);
				}
				joint = hinge;
			} break;
			case JointType::FIXED: {
				Ref<ViennaFixedJoint> fixed = memnew(ViennaFixedJoint);
				// Fixed joint relative transform is the difference between the two bones' bind poses.
				fixed->set_relative_transform(bones[jb.child_bone].bind_pose.affine_inverse() * bones[jb.parent_bone].bind_pose);
				joint = fixed;
			} break;
			default: // fallback to ball
				joint = memnew(ViennaBallJoint);
				break;
		}
		joint->set_body_a(bone_ids[jb.parent_bone]);
		joint->set_body_b(bone_ids[jb.child_bone]);
		joint_ids[j] = world->create_joint(joint);
	}
	built = true;
}

void ViennaRagdoll::destroy() {
	if (!built || !world) return;
	// Destroy joints first
	for (joint_id jid : joint_ids) {
		world->destroy_joint(jid);
	}
	joint_ids.clear();
	// Destroy bodies
	for (body_id bid : bone_ids) {
		world->destroy_body(bid);
	}
	bone_ids.clear();
	// Clear references
	for (Bone &bone : bones) {
		bone.body.unref();
	}
	builts = false;
}

void ViennaRagdoll::set_animation_pose(const LocalVector<Transform3D> &p_transforms) {
	int count = MIN(p_transforms.size(), bones.size());
	for (int i = 0; i < count; ++i) {
		if (bones[i].body.is_valid() && kinematic) {
			bones[i].body->set_transform(p_transforms[i]);
		}
	}
}

void ViennaRagdoll::set_kinematic(bool p_kinematic) {
	kinematic = p_kinematic;
	if (built) {
		for (Bone &bone : bones) {
			if (bone.body.is_valid()) {
				bone.body->set_type(p_kinematic ? BodyType::KINEMATIC : BodyType::DYNAMIC);
			}
		}
	}
}

void ViennaRagdoll::apply_force_to_all(const vec3 &p_force) {
	for (Bone &bone : bones) {
		if (bone.body.is_valid() && bone.body->is_active() && bone.body->get_type() == BodyType::DYNAMIC) {
			bone.body->apply_force(p_force, bone.body->get_position());
		}
	}
}

body_id ViennaRagdoll::get_body_id(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, bone_ids.size(), 0);
	return bone_ids[p_idx];
}

} // namespace vienna