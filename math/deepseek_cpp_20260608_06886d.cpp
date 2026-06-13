// File 317: modules/vienna/src/ragdoll/vienna_ragdoll.h
// ViennaRagdoll – builds a ragdoll from a skeleton of bones and physical
// body parts. Creates ViennaBody instances linked by ball, hinge, or D6
// joints to simulate realistic limb physics. Supports kinematic animation
// blending and motorised joints.

#ifndef VIENNA_RAGDOLL_VIENNA_RAGDOLL_H
#define VIENNA_RAGDOLL_VIENNA_RAGDOLL_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/transform_3d.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../bodies/vienna_body.h"
#include "../joints/vienna_ball_joint.h"
#include "../joints/vienna_hinge_joint.h"
#include "../joints/vienna_fixed_joint.h"
#include "../collision/vienna_shape.h"
#include "../world/vienna_world.h"

namespace vienna {

class ViennaRagdoll : public RefCounted {
	GDCLASS(ViennaRagdoll, RefCounted);

public:
	struct Bone {
		String name;
		Transform3D bind_pose;         // initial world transform
		Ref<ViennaBody> body;
		Ref<ViennaShape> shape;        // capsule, box, or sphere
		real_t mass;
		Bone() : mass(1.0) {}
	};

	struct JointBinding {
		int parent_bone;
		int child_bone;
		JointType joint_type;          // BALL, HINGE, or FIXED
		vec3 pivot;                    // in parent's local frame
		vec3 axis;                     // for hinge joints
		real_t min_angle;              // limit (hinge)
		real_t max_angle;
	};

	ViennaRagdoll();

	void set_world(ViennaWorld *p_world);
	ViennaWorld *get_world() const { return world; }

	// Add a bone (body part)
	void add_bone(const Bone &p_bone);
	// Link two bones with a joint
	void add_joint(int p_parent, int p_child, const JointBinding &p_joint);

	// Build all bones and joints in the world.
	void build();
	// Destroy the ragdoll from the world.
	void destroy();

	// Set the pose of all bones from an animation (kinematic mode).
	void set_animation_pose(const LocalVector<Transform3D> &p_transforms);

	// Enable/disable kinematic mode (driven by animation).
	void set_kinematic(bool p_kinematic);
	bool is_kinematic() const { return kinematic; }

	// Apply forces to all bones at once (e.g., explosion).
	void apply_force_to_all(const vec3 &p_force);

	// Access bones.
	int get_bone_count() const { return bones.size(); }
	const Bone &get_bone(int p_idx) const { return bones[p_idx]; }
	Bone &get_bone(int p_idx) { return bones[p_idx]; }

	// Get body ID for a bone (0 if not built).
	body_id get_body_id(int p_idx) const;

protected:
	static void _bind_methods();

private:
	ViennaWorld *world;
	LocalVector<Bone> bones;
	LocalVector<body_id> bone_ids;
	LocalVector<JointBinding> joints;
	LocalVector<joint_id> joint_ids;
	bool built;
	bool kinematic;
};

} // namespace vienna

#endif // VIENNA_RAGDOLL_VIENNA_RAGDOLL_H