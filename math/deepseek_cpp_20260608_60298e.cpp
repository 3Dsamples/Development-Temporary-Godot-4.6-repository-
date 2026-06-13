// File 312: modules/vienna/src/nodes/vienna_soft_body_3d.cpp
// Implements ViennaSoftBody3D – generates a cloth mesh, steps simulation via
// ViennaCloth (and optional ViennaClothSolver for rigid collisions), updates
// an ArrayMesh to visualise the deformed cloth in the Godot scene.

#include "vienna_soft_body_3d.h"
#include "../cloth/vienna_cloth.h"
#include "../cloth/vienna_cloth_solver.h"
#include "../world/vienna_world.h"
#include "../nodes/vienna_world_node_3d.h"
#include "scene/resources/array_mesh.h"
#include "scene/resources/standard_material_3d.h"
#include "scene/main/scene_tree.h"
#include "core/typedefs.h"

namespace vienna {

void ViennaSoftBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_resolution", "resolution_x", "resolution_y"), &ViennaSoftBody3D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution_x"), &ViennaSoftBody3D::get_resolution_x);
	ClassDB::bind_method(D_METHOD("get_resolution_y"), &ViennaSoftBody3D::get_resolution_y);
	ClassDB::bind_method(D_METHOD("set_width", "width"), &ViennaSoftBody3D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &ViennaSoftBody3D::get_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &ViennaSoftBody3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &ViennaSoftBody3D::get_height);
	ClassDB::bind_method(D_METHOD("set_structural_stiffness", "k"), &ViennaSoftBody3D::set_structural_stiffness);
	ClassDB::bind_method(D_METHOD("get_structural_stiffness"), &ViennaSoftBody3D::get_structural_stiffness);
	ClassDB::bind_method(D_METHOD("set_shear_stiffness", "k"), &ViennaSoftBody3D::set_shear_stiffness);
	ClassDB::bind_method(D_METHOD("get_shear_stiffness"), &ViennaSoftBody3D::get_shear_stiffness);
	ClassDB::bind_method(D_METHOD("set_bending_stiffness", "k"), &ViennaSoftBody3D::set_bending_stiffness);
	ClassDB::bind_method(D_METHOD("get_bending_stiffness"), &ViennaSoftBody3D::get_bending_stiffness);
	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &ViennaSoftBody3D::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &ViennaSoftBody3D::get_damping);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &ViennaSoftBody3D::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &ViennaSoftBody3D::get_gravity);
	ClassDB::bind_method(D_METHOD("set_wind", "wind"), &ViennaSoftBody3D::set_wind);
	ClassDB::bind_method(D_METHOD("get_wind"), &ViennaSoftBody3D::get_wind);
	ClassDB::bind_method(D_METHOD("set_solver_type", "type"), &ViennaSoftBody3D::set_solver_type);
	ClassDB::bind_method(D_METHOD("get_solver_type"), &ViennaSoftBody3D::get_solver_type);
	ClassDB::bind_method(D_METHOD("set_iteration_count", "iterations"), &ViennaSoftBody3D::set_iteration_count);
	ClassDB::bind_method(D_METHOD("get_iteration_count"), &ViennaSoftBody3D::get_iteration_count);
	ClassDB::bind_method(D_METHOD("pin_vertex", "x", "y", "pin"), &ViennaSoftBody3D::pin_vertex, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_collision_enabled", "enabled"), &ViennaSoftBody3D::set_collision_enabled);
	ClassDB::bind_method(D_METHOD("is_collision_enabled"), &ViennaSoftBody3D::is_collision_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "resolution"), "set_resolution", "get_resolution_x");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "structural_stiffness"), "set_structural_stiffness", "get_structural_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "shear_stiffness"), "set_shear_stiffness", "get_shear_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bending_stiffness"), "set_bending_stiffness", "get_bending_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "wind"), "set_wind", "get_wind");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_type", PROPERTY_HINT_ENUM, "MassSpring,XPBD"), "set_solver_type", "get_solver_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "iteration_count"), "set_iteration_count", "get_iteration_count");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_enabled"), "set_collision_enabled", "is_collision_enabled");
}

ViennaSoftBody3D::ViennaSoftBody3D() :
	resolution_x(16), resolution_y(16),
	width(2.0), height(2.0),
	structural_stiffness(1000.0), shear_stiffness(100.0), bending_stiffness(200.0),
	damping(0.01), gravity(0.0, -9.81, 0.0), wind(0.0, 0.0, 0.0),
	solver_type(1), iteration_count(5), collision_enabled(false),
	render_mesh_instance(nullptr) {
	cloth.instantiate();
	cloth_solver.instantiate();
	set_physics_process(true);
}

void ViennaSoftBody3D::set_resolution(int p_rx, int p_ry) {
	resolution_x = MAX(p_rx, 2);
	resolution_y = MAX(p_ry, 2);
	if (cloth.is_valid()) {
		cloth->set_resolution(resolution_x, resolution_y);
		_create_cloth(); // regenerate mesh
	}
}
int ViennaSoftBody3D::get_resolution_x() const { return resolution_x; }
int ViennaSoftBody3D::get_resolution_y() const { return resolution_y; }

void ViennaSoftBody3D::set_width(real_t p_w) { width = MAX(p_w, 0.01); if (cloth.is_valid()) cloth->set_width(width); }
real_t ViennaSoftBody3D::get_width() const { return width; }
void ViennaSoftBody3D::set_height(real_t p_h) { height = MAX(p_h, 0.01); if (cloth.is_valid()) cloth->set_height(height); }
real_t ViennaSoftBody3D::get_height() const { return height; }

void ViennaSoftBody3D::set_structural_stiffness(real_t p_k) { structural_stiffness = MAX(p_k, 0.0); if (cloth.is_valid()) cloth->set_structural_stiffness(structural_stiffness); }
real_t ViennaSoftBody3D::get_structural_stiffness() const { return structural_stiffness; }
void ViennaSoftBody3D::set_shear_stiffness(real_t p_k) { shear_stiffness = MAX(p_k, 0.0); if (cloth.is_valid()) cloth->set_shear_stiffness(shear_stiffness); }
real_t ViennaSoftBody3D::get_shear_stiffness() const { return shear_stiffness; }
void ViennaSoftBody3D::set_bending_stiffness(real_t p_k) { bending_stiffness = MAX(p_k, 0.0); if (cloth.is_valid()) cloth->set_bending_stiffness(bending_stiffness); }
real_t ViennaSoftBody3D::get_bending_stiffness() const { return bending_stiffness; }
void ViennaSoftBody3D::set_damping(real_t p_d) { damping = CLAMP(p_d, 0.0, 1.0); if (cloth.is_valid()) cloth->set_damping(damping); }
real_t ViennaSoftBody3D::get_damping() const { return damping; }
void ViennaSoftBody3D::set_gravity(const vec3 &p_g) { gravity = p_g; if (cloth.is_valid()) cloth->set_gravity(gravity); }
vec3 ViennaSoftBody3D::get_gravity() const { return gravity; }
void ViennaSoftBody3D::set_wind(const vec3 &p_w) { wind = p_w; if (cloth.is_valid()) cloth->set_wind(wind); }
vec3 ViennaSoftBody3D::get_wind() const { return wind; }
void ViennaSoftBody3D::set_solver_type(int p_type) { solver_type = CLAMP(p_type, 0, 1); if (cloth.is_valid()) cloth->set_solver_type((ClothSolverType)solver_type); }
int ViennaSoftBody3D::get_solver_type() const { return solver_type; }
void ViennaSoftBody3D::set_iteration_count(int p_iter) { iteration_count = MAX(p_iter, 1); if (cloth.is_valid()) cloth->set_iterations(iteration_count); }
int ViennaSoftBody3D::get_iteration_count() const { return iteration_count; }

void ViennaSoftBody3D::pin_vertex(int p_x, int p_y, bool p_pin) {
	if (cloth.is_valid()) cloth->pin_vertex(p_x, p_y, p_pin);
}

void ViennaSoftBody3D::set_collision_enabled(bool p_en) { collision_enabled = p_en; }
bool ViennaSoftBody3D::is_collision_enabled() const { return collision_enabled; }

void ViennaSoftBody3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_create_cloth();                      // generate the cloth mesh and initialise physics
		_build_display_mesh();                // create the Godot mesh for visualisation
	}
	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
		if (cloth.is_valid()) {
			real_t dt = get_physics_process_delta_time();
			// If collision is enabled, attempt to find the world and feed rigid bodies
			if (collision_enabled) {
				// Locate the ViennaWorld through an ancestor ViennaWorldNode3D (or similar)
				Node *p = get_parent();
				while (p) {
					if (p->has_method("get_vienna_world")) {
						Variant ret = p->call("get_vienna_world");
						ViennaWorld *w = Object::cast_to<ViennaWorld>(ret);
						if (w) {
							// Build a list of rigid bodies (active, with collision shape)
							LocalVector<Ref<ViennaBody>> bodies;
							LocalVector<body_id> ids = w->get_body_ids();
							for (body_id id : ids) {
								Ref<ViennaBody> body = w->get_body(id);
								if (body.is_valid() && body->is_active() && body->get_collision_shape().is_valid()) {
									bodies.push_back(body);
								}
							}
							cloth_solver->set_rigid_bodies(bodies);
							break;
						}
					}
					ViennaWorldNode3D *wn = Object::cast_to<ViennaWorldNode3D>(p);
					if (wn) {
						ViennaWorld *w = wn->get_vienna_world();
						LocalVector<Ref<ViennaBody>> bodies;
						LocalVector<body_id> ids = w->get_body_ids();
						for (body_id id : ids) {
							Ref<ViennaBody> body = w->get_body(id);
							if (body.is_valid() && body->is_active() && body->get_collision_shape().is_valid()) {
								bodies.push_back(body);
							}
						}
						cloth_solver->set_rigid_bodies(bodies);
						break;
					}
					p = p->get_parent();
				}
				// Use the cloth solver to step collision and cloth
				cloth_solver->add_cloth(cloth);
				cloth_solver->step(dt);
				cloth_solver->clear_cloths();
			} else {
				// Simple standalone cloth step
				cloth->step(dt);
			}
			// Update the Godot mesh with current vertex positions
			_update_display_mesh();
		}
	}
}

void ViennaSoftBody3D::_create_cloth() {
	if (cloth.is_null()) return;
	cloth->set_resolution(resolution_x, resolution_y);
	cloth->set_width(width);
	cloth->set_height(height);
	cloth->set_structural_stiffness(structural_stiffness);
	cloth->set_shear_stiffness(shear_stiffness);
	cloth->set_bending_stiffness(bending_stiffness);
	cloth->set_damping(damping);
	cloth->set_gravity(gravity);
	cloth->set_wind(wind);
	cloth->set_solver_type((ClothSolverType)solver_type);
	cloth->set_iterations(iteration_count);
	cloth->generate();
	// Pin the top row by default
	for (int x = 0; x < resolution_x; ++x) pin_vertex(x, 0, true);
}

void ViennaSoftBody3D::_build_display_mesh() {
	// Create a MeshInstance3D child that will hold the cloth geometry
	render_mesh_instance = get_node_or_null<MeshInstance3D>(NodePath("ClothMesh"));
	if (!render_mesh_instance) {
		render_mesh_instance = memnew(MeshInstance3D);
		render_mesh_instance->set_name("ClothMesh");
		add_child(render_mesh_instance);
	}
	mesh.instantiate();
	render_mesh_instance->set_mesh(mesh);
	// Apply a basic material
	Ref<StandardMaterial3D> mat; mat.instantiate();
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, false);
	render_mesh_instance->set_material_override(mat);
	_update_display_mesh(); // set initial vertices
}

void ViennaSoftBody3D::_update_display_mesh() {
	if (mesh.is_null() || cloth.is_null()) return;

	int rx = resolution_x;
	int ry = resolution_y;
	int total_verts = rx * ry;
	// Two triangles per quad
	int total_tris = (rx - 1) * (ry - 1) * 2;

	PackedVector3Array vertices;
	vertices.resize(total_verts);
	for (int y = 0; y < ry; ++y) {
		for (int x = 0; x < rx; ++x) {
			int idx = y * rx + x;
			vertices.set(idx, cloth->get_vertex(idx).position);
		}
	}

	PackedInt32Array indices;
	indices.resize(total_tris * 3);
	int tri = 0;
	for (int y = 0; y < ry - 1; ++y) {
		for (int x = 0; x < rx - 1; ++x) {
			int i0 = y * rx + x;
			int i1 = i0 + 1;
			int i2 = i0 + rx;
			int i3 = i2 + 1;
			// Triangle 1: i0, i1, i2
			indices.set(tri * 3, i0);
			indices.set(tri * 3 + 1, i1);
			indices.set(tri * 3 + 2, i2);
			tri++;
			// Triangle 2: i1, i3, i2
			indices.set(tri * 3, i1);
			indices.set(tri * 3 + 1, i3);
			indices.set(tri * 3 + 2, i2);
			tri++;
		}
	}

	// Compute normals (quick average of face normals)
	PackedVector3Array normals;
	normals.resize(total_verts);
	LocalVector<vec3> normals_temp(total_verts, vec3());
	for (int i = 0; i < tri; ++i) {
		int i0 = indices[i * 3];
		int i1 = indices[i * 3 + 1];
		int i2 = indices[i * 3 + 2];
		vec3 v0 = vertices[i0];
		vec3 v1 = vertices[i1];
		vec3 v2 = vertices[i2];
		vec3 face_n = (v1 - v0).cross(v2 - v0);
		normals_temp[i0] += face_n;
		normals_temp[i1] += face_n;
		normals_temp[i2] += face_n;
	}
	for (int i = 0; i < total_verts; ++i) {
		vec3 n = normals_temp[i].normalized();
		normals.set(i, n);
	}

	// Build surface arrays
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_INDEX] = indices;

	mesh->clear_surfaces();
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
}

} // namespace vienna