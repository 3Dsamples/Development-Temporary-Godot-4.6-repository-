// File 311: modules/vienna/src/nodes/vienna_soft_body_3d.h
// ViennaSoftBody3D – a Godot Node3D that simulates a deformable cloth mesh
// powered by ViennaCloth. It creates the cloth, updates vertex positions each
// frame from the simulation, and drives a MeshInstance3D child for rendering.

#ifndef VIENNA_NODES_SOFT_BODY_3D_H
#define VIENNA_NODES_SOFT_BODY_3D_H

#include "scene/3d/node_3d.h"
#include "scene/resources/mesh.h"
#include "scene/resources/array_mesh.h"
#include "../cloth/vienna_cloth.h"
#include "../cloth/vienna_cloth_solver.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaSoftBody3D : public Node3D {
	GDCLASS(ViennaSoftBody3D, Node3D);

public:
	ViennaSoftBody3D();

	// Cloth generation parameters
	void set_resolution(int p_rx, int p_ry);
	int get_resolution_x() const;
	int get_resolution_y() const;

	void set_width(real_t p_w);
	real_t get_width() const;
	void set_height(real_t p_h);
	real_t get_height() const;

	// Physics parameters (forwarded to the internal ViennaCloth)
	void set_structural_stiffness(real_t p_k);
	real_t get_structural_stiffness() const;
	void set_shear_stiffness(real_t p_k);
	real_t get_shear_stiffness() const;
	void set_bending_stiffness(real_t p_k);
	real_t get_bending_stiffness() const;
	void set_damping(real_t p_d);
	real_t get_damping() const;
	void set_gravity(const vec3 &p_g);
	vec3 get_gravity() const;
	void set_wind(const vec3 &p_w);
	vec3 get_wind() const;
	void set_solver_type(int p_type);           // 0 = mass‑spring, 1 = XPBD
	int get_solver_type() const;
	void set_iteration_count(int p_iter);
	int get_iteration_count() const;

	// Pin vertices (e.g., top row)
	void pin_vertex(int p_x, int p_y, bool p_pin = true);

	// Collision with rigid bodies
	void set_collision_enabled(bool p_en);
	bool is_collision_enabled() const;

	// Access internal cloth (advanced)
	Ref<ViennaCloth> get_cloth() const { return cloth; }

protected:
	void _notification(int p_what);
	static void _bind_methods();

private:
	void _create_cloth();
	void _build_display_mesh();
	void _update_display_mesh();

	Ref<ViennaCloth> cloth;
	Ref<ViennaClothSolver> cloth_solver;      // optional solver (can be shared)

	// Cloth parameters (used when creating the cloth)
	int resolution_x;
	int resolution_y;
	real_t width;
	real_t height;
	real_t structural_stiffness;
	real_t shear_stiffness;
	real_t bending_stiffness;
	real_t damping;
	vec3 gravity;
	vec3 wind;
	int solver_type;
	int iteration_count;
	bool collision_enabled;

	// Rendering
	MeshInstance3D *render_mesh_instance;
	Ref<ArrayMesh> mesh;
};

} // namespace vienna

#endif // VIENNA_NODES_SOFT_BODY_3D_H