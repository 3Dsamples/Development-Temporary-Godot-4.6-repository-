// File 156: modules/genesis/src/nodes/genesis_cloth_3d.cpp
// Implements the GenesisCloth3D node: generates a triangulated cloth mesh,
// builds distance and bending constraints, steps the PBD solver, and
// updates the debug display mesh every frame.

#include "genesis_cloth_3d.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"

#include "../solvers/pbd_solver.h"
#include "../materials/pbd_material.h"
#include "../../../gaia/src/mesh/tri_mesh.h"
#include "../../../gaia/src/pbd/distance_constraint.h"
#include "../../../gaia/src/pbd/bending_constraint.h"
#include "../../../gaia/src/pbd/collision_constraint.h"

namespace genesis {

void GenesisCloth3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &GenesisCloth3D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &GenesisCloth3D::get_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &GenesisCloth3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &GenesisCloth3D::get_height);
	ClassDB::bind_method(D_METHOD("set_resolution", "res"), &GenesisCloth3D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &GenesisCloth3D::get_resolution);
	ClassDB::bind_method(D_METHOD("set_structural_compliance", "c"), &GenesisCloth3D::set_structural_compliance);
	ClassDB::bind_method(D_METHOD("get_structural_compliance"), &GenesisCloth3D::get_structural_compliance);
	ClassDB::bind_method(D_METHOD("set_bending_compliance", "c"), &GenesisCloth3D::set_bending_compliance);
	ClassDB::bind_method(D_METHOD("get_bending_compliance"), &GenesisCloth3D::get_bending_compliance);
	ClassDB::bind_method(D_METHOD("set_damping_compliance", "c"), &GenesisCloth3D::set_damping_compliance);
	ClassDB::bind_method(D_METHOD("get_damping_compliance"), &GenesisCloth3D::get_damping_compliance);
	ClassDB::bind_method(D_METHOD("set_gravity_scale", "scale"), &GenesisCloth3D::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &GenesisCloth3D::get_gravity_scale);
	ClassDB::bind_method(D_METHOD("pin_vertex", "index", "pin"), &GenesisCloth3D::pin_vertex, DEFVAL(true));
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution", PROPERTY_HINT_RANGE, "2,100,1"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "structural_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_structural_compliance", "get_structural_compliance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bending_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_bending_compliance", "get_bending_compliance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_damping_compliance", "get_damping_compliance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale"), "set_gravity_scale", "get_gravity_scale");
}

void GenesisCloth3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_initialize();                     // build mesh, constraints, display
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_update_display();                 // redraw debug triangles
	}
}

void GenesisCloth3D::_initialize() {
	if (!_find_world()) {
		ERR_PRINT("GenesisCloth3D: no GenesisWorld found in scene tree.");
		return;
	}

	// 1. Generate the triangular cloth mesh (regular grid)
	_build_grid_mesh();

	// 2. Create a PBD material and solver
	pbd_material.instantiate();
	pbd_material->set_compliance(structural_compliance);
	pbd_material->set_bending_compliance(bending_compliance);
	pbd_material->set_damping_compliance(damping_compliance);

	solver.instantiate();
	solver->set_material(pbd_material);

	// 3. Wrap the mesh in a Gaia SoftBody (required by PBD constraints)
	soft_body = memnew(gaia::SoftBody);
	soft_body->resize(mesh.vertex_count());
	Transform3D base_xform = get_global_transform();
	for (int i = 0; i < mesh.vertex_count(); ++i) {
		Vector3 world_rest = base_xform.xform(mesh.get_vertex(i));
		soft_body->positions[i] = world_rest;          // initial position
		soft_body->rest_positions[i] = world_rest;     // rest shape
		soft_body->velocities[i] = Vector3();
	}

	// 4. Build distance (structural) constraints
	_add_distance_constraints();
	// 5. Build bending constraints from shared edges
	_add_bending_constraints();

	// 6. Pin the top row if needed (already set in _build_grid_mesh) – but we need to copy pinned flags
	pinned_flags.resize(mesh.vertex_count());
	// The top row was pinned during mesh construction; we respect that.
	for (int i = 0; i < resolution; ++i) pinned_flags[i] = true;

	// 7. Create a MeshInstance3D child for immediate‑mode debug display
	_display_mesh_instance = get_node_or_null<MeshInstance3D>(NodePath("ClothDisplay"));
	if (!_display_mesh_instance) {
		_display_mesh_instance = memnew(MeshInstance3D);
		_display_mesh_instance->set_name("ClothDisplay");
		add_child(_display_mesh_instance);
	}
	_debug_mesh.instantiate();
	_display_mesh_instance->set_mesh(_debug_mesh);

	// Use a simple unlit vertex‑color material so the wireframe is visible
	Ref<StandardMaterial3D> mat; mat.instantiate();
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	_display_mesh_instance->set_material_override(mat);

	set_physics_process(true);            // simulation driven by physics ticks
}

void GenesisCloth3D::_physics_process(real_t p_dt) {
	if (solver.is_null() || soft_body == nullptr) return;

	// Semi‑implicit Euler: predict positions using current velocities
	for (int i = 0; i < soft_body->positions.size(); ++i) {
		if (pinned_flags[i]) {
			soft_body->velocities[i] = Vector3(); // reset velocity for pinned vertices
			continue;
		}
		// Apply scaled gravity
		soft_body->velocities[i] += world->get_gravity() * gravity_scale * p_dt;
		// Predict position
		soft_body->positions[i] += soft_body->velocities[i] * p_dt;
	}

	// XPBD solve (iterative projection)
	solver->set_dt(p_dt);
	solver->solve(p_dt);

	// Update velocities from the corrected positions (post‑solve)
	for (int i = 0; i < soft_body->velocities.size(); ++i) {
		if (pinned_flags[i]) continue;
		Vector3 delta = soft_body->positions[i] - mesh.get_vertex(i); // original position before step
		soft_body->velocities[i] = delta / p_dt;
	}

	// Copy positions back to the TriMesh (for rendering and rest state recovery)
	for (int i = 0; i < mesh.vertex_count(); ++i) {
		mesh.get_vertex(i) = soft_body->positions[i];
	}
}

void GenesisCloth3D::_update_display() {
	if (_debug_mesh.is_null()) return;
	ImmediateMesh *im = _debug_mesh.ptr();
	im->clear_surfaces();
	im->surface_begin(Mesh::PRIMITIVE_TRIANGLES);

	int tri_count = mesh.triangle_count();
	for (int t = 0; t < tri_count; ++t) {
		gaia::mesh::TriMesh::Triangle tri = mesh.get_triangle(t);
		Vector3 v0 = mesh.get_vertex(tri.v0);
		Vector3 v1 = mesh.get_vertex(tri.v1);
		Vector3 v2 = mesh.get_vertex(tri.v2);
		im->surface_add_vertex(v0);
		im->surface_add_vertex(v1);
		im->surface_add_vertex(v2);
	}
	im->surface_end();
}

void GenesisCloth3D::_build_grid_mesh() {
	int w = resolution;
	int h = resolution;
	mesh.clear();
	pinned_flags.clear();

	real_t dx = width / (w - 1);
	real_t dy = height / (h - 1);

	// Generate vertices (XZ plane, Y up by default)
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			Vector3 pos(i * dx - width * 0.5f, 0.0f, j * dy - height * 0.5f);
			mesh.add_vertex(pos);
		}
	}

	// Generate triangles (two per quad)
	for (int j = 0; j < h - 1; ++j) {
		for (int i = 0; i < w - 1; ++i) {
			int a = j * w + i;
			int b = a + w;
			int c = a + 1;
			int d = b + 1;
			mesh.add_triangle(a, b, c);    // lower‑left
			mesh.add_triangle(c, b, d);    // upper‑right
		}
	}

	// Initialize pinned flags (top row fixed)
	pinned_flags.resize(mesh.vertex_count(), false);
	for (int i = 0; i < w; ++i) pinned_flags[i] = true;
}

void GenesisCloth3D::_add_distance_constraints() {
	int tri_count = mesh.triangle_count();
	for (int t = 0; t < tri_count; ++t) {
		auto tri = mesh.get_triangle(t);
		_add_distance_edge(tri.v0, tri.v1);
		_add_distance_edge(tri.v1, tri.v2);
		_add_distance_edge(tri.v2, tri.v0);
	}
}

void GenesisCloth3D::_add_distance_edge(int idx0, int idx1) {
	Ref<gaia::DistanceConstraint> dc; dc.instantiate();
	dc->set_body(soft_body);
	dc->set_indices(idx0, idx1);
	dc->init_from_positions();                          // computes rest_length from current positions
	dc->set_compliance(solver->get_material().is_valid() ?
						 solver->get_material()->get_compliance() : 1e-6f);
	solver->add_distance_constraint(dc.ptr());
}

void GenesisCloth3D::_add_bending_constraints() {
	// Build map from edge (min, max) to list of triangle indices sharing it.
	HashMap<uint64_t, int> edge_owner;        // first triangle seen
	HashMap<uint64_t, int> edge_opposite;     // opposite vertex in first triangle
	HashMap<uint64_t, int> edge_second;       // second triangle index

	int tri_count = mesh.triangle_count();
	for (int t = 0; t < tri_count; ++t) {
		auto tri = mesh.get_triangle(t);
		int v[3] = { tri.v0, tri.v1, tri.v2 };
		for (int i = 0; i < 3; ++i) {
			int a = v[i], b = v[(i+1)%3];
			if (a > b) SWAP(a, b);
			uint64_t key = (uint64_t(a) << 32) | b;
			if (!edge_owner.has(key)) {
				edge_owner[key] = t;
				edge_opposite[key] = v[(i+2)%3]; // opposite vertex
			} else {
				// Second triangle sharing this edge
				int other_tri = edge_owner[key];
				int opp_other = edge_opposite[key];
				int opp_this = v[(i+2)%3];
				// Create bending constraint between the two triangles
				Ref<gaia::BendingConstraint> bc; bc.instantiate();
				bc->set_body(soft_body);
				bc->set_indices(a, b, opp_other, opp_this);
				bc->init_from_positions();        // computes rest dihedral angle
				bc->set_compliance(solver->get_material().is_valid() ?
								 solver->get_material()->get_bending_compliance() : 1e-4f);
				solver->add_bending_constraint(bc.ptr());
			}
		}
	}
}

bool GenesisCloth3D::_find_world() {
	Node *parent = get_parent();
	while (parent) {
		world = Object::cast_to<GenesisWorld>(parent);
		if (world) return true;
		parent = parent->get_parent();
	}
	return false;
}

} // namespace genesis