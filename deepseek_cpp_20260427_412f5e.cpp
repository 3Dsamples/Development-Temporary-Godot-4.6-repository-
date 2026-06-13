// File 298: modules/vienna/src/utils/vienna_debug_draw.h
// ViennaDebugDraw – real‑time wireframe overlay for all bodies, joints,
// cloth, cloth, and particles in a ViennaWorld. Renders AABBs, collision shapes,
// velocity vectors, contact points, and joint pivots/axes using Godot's
// ImmediateMesh with per‑vertex colour support.

#ifndef VIENNA_UTILS_DEBUG_DRAW_H
#define VIENNA_UTILS_DEBUG_DRAW_H

#include "scene/3d/node_3d.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../joints/vienna_joint.h"
#include "../joints/vienna_ball_joint.h"
#include "../joints/vienna_hinge_joint.h"
#include "../joints/vienna_slider_joint.h"
#include "../joints/vienna_fixed_joint.h"
#include "../joints/vienna_distance_joint.h"
#include "../joints/vienna_rope_joint.h"
#include "../cloth/vienna_cloth.h"
#include "../particles/vienna_particle_system.h"

namespace vienna {

class ViennaDebugDraw : public Node3D {
	GDCLASS(ViennaDebugDraw, Node3D);

public:
	ViennaDebugDraw();

	void set_world(ViennaWorld *p_world);
	ViennaWorld *get_world() const { return world; }
	void set_show_bodies(bool p_show);
	void set_show_joints(bool p_show);
	void set_show_contacts(bool p_show);
	void set_show_cloth(bool p_show);
	void set_show_particles(bool p_show);

	void _notification(int p_what);

protected:
	static void _bind_methods();

private:
	void _create_display_mesh();
	void _redraw();
	void draw_aabb(ImmediateMesh *im, const aabb &box, const Color &color);
	void draw_sphere(ImmediateMesh *im, const vec3 &center, real_t radius, const Color &color);
	void draw_arrow(ImmediateMesh *im, const vec3 &from, const vec3 &to, const Color &color);
	void draw_joint_pivot(ImmediateMesh *im, const vec3 &pivot, const Color &color);

	ViennaWorld *world;
	Ref<ImmediateMesh> debug_mesh;
	MeshInstance3D *mesh_instance;
	bool show_bodies;
	bool show_joints;
	bool show_contacts;
	bool show_cloth;
	bool show_particles;
};

} // namespace vienna

#endif // VIENNA_UTILS_DEBUG_DRAW_H