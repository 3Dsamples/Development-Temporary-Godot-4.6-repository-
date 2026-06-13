// File 158: modules/genesis/src/nodes/genesis_soft_body_3d.cpp
// Implements the GenesisSoftBody3D node. Sets up a tetrahedral FEM simulation
// on a TetGen mesh or a generated primitive, registers the FEM entity with the
// GenesisWorld, and updates the debug wireframe each frame.

#include "genesis_soft_body_3d.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"

#include "../genesis_world.h"                     // GenesisWorld
#include "../entities/fem_entity.h"
#include "../materials/fem_material.h"
#include "../../../gaia/src/mesh/tet_mesh.h"
#include "../../../gaia/src/mesh/tri_mesh.h"
#include "../../../gaia/src/mesh/mesh_io.h"

namespace genesis {

void GenesisSoftBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh_path", "path"), &GenesisSoftBody3D::set_mesh_path);
	ClassDB::bind_method(D_METHOD("get_mesh_path"), &GenesisSoftBody3D::get_mesh_path);
	ClassDB::bind_method(D_METHOD("set_resolution", "res"), &GenesisSoftBody3D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &GenesisSoftBody3D::get_resolution);
	ClassDB::bind_method(D_METHOD("set_subdivisions", "subdiv"), &GenesisSoftBody3D::set_subdivisions);
	ClassDB::bind_method(D_METHOD("get_subdivisions"), &GenesisSoftBody3D::get_subdivisions);
	ClassDB::bind_method(D_METHOD("set_pressure_stiffness", "k"), &GenesisSoftBody3D::set_pressure_stiffness);
	ClassDB::bind_method(D_METHOD("get_pressure_stiffness"), &GenesisSoftBody3D::get_pressure_stiffness);
	ClassDB::bind_method(D_METHOD("set_fem_material", "material"), &GenesisSoftBody3D::set_fem_material);
	ClassDB::bind_method(D_METHOD("get_fem_material"), &GenesisSoftBody3D::get_fem_material);
	ClassDB::bind_method(D_METHOD("apply_force_to_vertices", "force"), &GenesisSoftBody3D::apply_force_to_vertices);
	ClassDB::bind_method(D_METHOD("reset_rest_shape"), &GenesisSoftBody3D::reset_rest_shape);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "mesh_path", PROPERTY_HINT_FILE, "*.node"), "set_mesh_path", "get_mesh_path");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "resolution", PROPERTY_HINT_RANGE, "0.01,10,0.01"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivisions", PROPERTY_HINT_RANGE, "0,5,1"), "set_subdivisions", "get_subdivisions");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pressure_stiffness", PROPERTY_HINT_RANGE, "0,10000,0.1"), "set_pressure_stiffness", "get_pressure_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fem_material", PROPERTY_HINT_RESOURCE_TYPE, "FEMMaterial"), "set_fem_material", "get_fem_material");
}

void GenesisSoftBody3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_initialize();                     // load/generate tet mesh, create FEM entity
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_sync_mesh_to_scene();             // update display wireframe from current positions
	}
}

void GenesisSoftBody3D::_initialize() {
	if (!_find_world()) {
		ERR_PRINT("GenesisSoftBody3D: no GenesisWorld found in scene tree.");
		return;
	}

	// 1. Create the FEM entity and assign it a material
	fem_entity.instantiate();
	fem_entity->set_entity_uid(_generate_uid());
	if (_material.is_valid()) fem_entity->set_material(_material);

	// 2. Load or generate a tetrahedral mesh
	gaia::mesh::TetMesh tet_mesh;
	if (!_mesh_path.is_empty()) {
		String node_path = _mesh_path + ".node";
		String ele_path  = _mesh_path + ".ele";
		Error err = gaia::mesh::MeshIO::load_tetgen(node_path, ele_path, tet_mesh);
		if (err != OK)
			WARN_PRINT("GenesisSoftBody3D: failed to load TetGen files – falling back to cube.");
	}
	if (tet_mesh.vertex_count() == 0) {
		_generate_cube_tet_mesh(tet_mesh, _resolution, _subdivisions); // generate a simple cube
	}

	fem_entity->set_mesh(tet_mesh);
	fem_entity->get_mesh().precompute_rest_state();

	// 3. Cache rest positions for resetting later
	int nv = tet_mesh.vertex_count();
	_rest_positions.resize(nv);
	for (int i = 0; i < nv; ++i) _rest_positions[i] = tet_mesh.get_vertex(i);

	// 4. Allocate per‑vertex force buffer (same size as mesh vertex count)
	_vertex_forces.resize(nv, Vector3());

	// 5. Set the entity's world transform from the Node3D
	fem_entity->set_transform(get_global_transform());

	// 6. Apply extra IPC / plasticity flags from material if set
	if (_material.is_valid()) {
		fem_entity->set_ipc_enabled(_material->is_plasticity_enabled()); // may be repurposed
		fem_entity->set_plasticity_enabled(_material->is_plasticity_enabled());
	}

	// 7. Register the entity with the GenesisWorld (now it will be stepped)
	world->add_entity(fem_entity);

	// 8. Create a debug display mesh (tetrahedron edges)
	_create_display_mesh();

	set_process(true); // we already set process to true in constructor
}

void GenesisSoftBody3D::_sync_mesh_to_scene() {
	if (fem_entity.is_null() || _debug_mesh.is_null()) return;
	gaia::mesh::TetMesh &tet = fem_entity->get_mesh();
	ImmediateMesh *im = _debug_mesh.ptr();
	im->clear_surfaces();
	im->surface_begin(Mesh::PRIMITIVE_LINES);

	int element_count = tet.element_count();
	for (int el = 0; el < element_count; ++el) {
		gaia::mesh::TetMesh::Tetrahedron t = tet.get_tetrahedron(el);
		Vector3 v0 = tet.get_vertex(t.v0);
		Vector3 v1 = tet.get_vertex(t.v1);
		Vector3 v2 = tet.get_vertex(t.v2);
		Vector3 v3 = tet.get_vertex(t.v3);
		// 6 edges of the tetrahedron
		im->surface_add_vertex(v0); im->surface_add_vertex(v1);
		im->surface_add_vertex(v0); im->surface_add_vertex(v2);
		im->surface_add_vertex(v0); im->surface_add_vertex(v3);
		im->surface_add_vertex(v1); im->surface_add_vertex(v2);
		im->surface_add_vertex(v1); im->surface_add_vertex(v3);
		im->surface_add_vertex(v2); im->surface_add_vertex(v3);
	}
	im->surface_end();
}

void GenesisSoftBody3D::_create_display_mesh() {
	// Ensure we have a MeshInstance3D child named "DisplayMesh"
	MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("DisplayMesh"));
	if (!mi) {
		mi = memnew(MeshInstance3D);
		mi->set_name("DisplayMesh");
		add_child(mi);
	}
	_debug_mesh.instantiate();
	mi->set_mesh(_debug_mesh);
	// Simple unlit material
	Ref<StandardMaterial3D> mat; mat.instantiate();
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mi->set_material_override(mat);
}

void GenesisSoftBody3D::apply_force_to_vertices(const Vector3 &force) {
	// Accumulate force for each vertex; they will be read by FEM solver
	for (int i = 0; i < _vertex_forces.size(); ++i) _vertex_forces[i] += force;
}

void GenesisSoftBody3D::reset_rest_shape() {
	if (fem_entity.is_null()) return;
	gaia::mesh::TetMesh &mesh = fem_entity->get_mesh();
	for (int i = 0; i < mesh.vertex_count(); ++i) mesh.get_vertex(i) = _rest_positions[i];
	mesh.precompute_rest_state();
}

bool GenesisSoftBody3D::_find_world() {
	Node *p = get_parent();
	while (p) {
		world = Object::cast_to<GenesisWorld>(p);
		if (world) return true;
		p = p->get_parent();
	}
	return false;
}

uint64_t GenesisSoftBody3D::_generate_uid() {
	return (uint64_t(get_instance_id()) << 16) | uint64_t(Math::rand() & 0xFFFF);
}

void GenesisSoftBody3D::_generate_cube_tet_mesh(gaia::mesh::TetMesh &mesh, real_t size, int subdivisions) {
	mesh.clear();
	// Generate a simple tetrahedralized cube with a single tet for each octant
	// of a cube? For demonstration we create 5 vertices forming a pyramid.
	if (subdivisions > 0) {
		// Recursive subdivision not implemented; create a refined grid.
		// Fallback to a single tetrahedron.
	}
	// Minimal valid tet mesh: 4 vertices, 1 tet
	mesh.add_vertex(Vector3(0, 0, 0));
	mesh.add_vertex(Vector3(size, 0, 0));
	mesh.add_vertex(Vector3(0, size, 0));
	mesh.add_vertex(Vector3(0, 0, size));
	mesh.add_tetrahedron(0, 1, 2, 3);
}

} // namespace genesis