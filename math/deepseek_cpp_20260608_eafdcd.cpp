// File 168: modules/genesis/src/nodes/genesis_mpm_3d.cpp
// GenesisMPM3D node implementation: initialises the MPM entity and solver,
// generates particles from a box or TetGen mesh, steps the MPM solver in
// physics process, and updates the debug point cloud display each frame.

#include "genesis_mpm_3d.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"

#include "../genesis_world.h"                  // GenesisWorld (for gravity & registration)
#include "../entities/mpm_entity.h"
#include "../solvers/mpm_solver.h"
#include "../materials/mpm_material.h"
#include "../../../gaia/src/mesh/tet_mesh.h"
#include "../../../gaia/src/mesh/mesh_io.h"

namespace genesis {

void GenesisMPM3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_grid_resolution", "res"), &GenesisMPM3D::set_grid_resolution);
	ClassDB::bind_method(D_METHOD("get_grid_resolution"), &GenesisMPM3D::get_grid_resolution);
	ClassDB::bind_method(D_METHOD("set_cell_size", "dx"), &GenesisMPM3D::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &GenesisMPM3D::get_cell_size);
	ClassDB::bind_method(D_METHOD("set_particle_spacing", "spacing"), &GenesisMPM3D::set_particle_spacing);
	ClassDB::bind_method(D_METHOD("get_particle_spacing"), &GenesisMPM3D::get_particle_spacing);
	ClassDB::bind_method(D_METHOD("set_mpm_material", "material"), &GenesisMPM3D::set_mpm_material);
	ClassDB::bind_method(D_METHOD("get_mpm_material"), &GenesisMPM3D::get_mpm_material);
	ClassDB::bind_method(D_METHOD("emit_particles_box", "box"), &GenesisMPM3D::emit_particles_box);
	ClassDB::bind_method(D_METHOD("load_tetmesh_as_particles", "node_path", "ele_path"), &GenesisMPM3D::load_tetmesh_as_particles);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "grid_resolution", PROPERTY_HINT_RANGE, "4,256,1"), "set_grid_resolution", "get_grid_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cell_size", PROPERTY_HINT_RANGE, "0.001,10,0.001"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "particle_spacing", PROPERTY_HINT_RANGE, "0.001,1,0.001"), "set_particle_spacing", "get_particle_spacing");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mpm_material", PROPERTY_HINT_RESOURCE_TYPE, "MPMMaterial"), "set_mpm_material", "get_mpm_material");
}

void GenesisMPM3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_initialize();
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_update_display();
	}
}

void GenesisMPM3D::_initialize() {
	// Locate the GenesisWorld node in the scene tree.
	if (!_find_world()) {
		ERR_PRINT("GenesisMPM3D: no GenesisWorld found in the scene tree.");
		return;
	}

	// Create the MPM entity and assign it an unique handle.
	_mpm_entity.instantiate();
	_mpm_entity->set_entity_uid(_generate_uid());
	_mpm_entity->set_grid_resolution(_grid_resolution);
	_mpm_entity->set_cell_size(_cell_size);

	// Create the MPM solver, link entity and material.
	_mpm_solver.instantiate();
	_mpm_solver->set_grid_resolution(_grid_resolution, _grid_resolution, _grid_resolution);
	_mpm_solver->set_grid_cell_size(_cell_size);
	_mpm_solver->add_entity(_mpm_entity);
	if (_material.is_valid()) {
		_mpm_solver->set_material(_material);
	}

	// Register entity with the world (solver stepping is handled independently).
	_world->add_entity(_mpm_entity);

	// Create the MeshInstance3D child for point cloud display.
	MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("MPMDisplay"));
	if (!mi) {
		mi = memnew(MeshInstance3D);
		mi->set_name("MPMDisplay");
		add_child(mi);
	}
	_debug_mesh.instantiate();
	mi->set_mesh(_debug_mesh);
	Ref<StandardMaterial3D> mat; mat.instantiate();
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mi->set_material_override(mat);

	// MPM solver is stepped in physics process.
	set_physics_process(true);
}

void GenesisMPM3D::_physics_process(real_t p_dt) {
	if (_mpm_solver.is_null()) return;
	// Set gravity from the world for consistency.
	_mpm_solver->set_gravity(_world ? _world->get_gravity() : Vector3(0.0f, -9.81f, 0.0f));
	_mpm_solver->set_dt(p_dt);
	_mpm_solver->step();
}

void GenesisMPM3D::_update_display() {
	if (_debug_mesh.is_null() || _mpm_entity.is_null()) return;
	ImmediateMesh *im = _debug_mesh.ptr();
	im->clear_surfaces();
	im->surface_begin(Mesh::PRIMITIVE_POINTS);

	const LocalVector<MPMEntity::Particle> &parts = _mpm_entity->get_particles();
	for (const MPMEntity::Particle &p : parts) {
		im->surface_add_vertex(p.position);
	}
	im->surface_end();
}

void GenesisMPM3D::emit_particles_box(const AABB &p_box) {
	ERR_FAIL_COND(_mpm_entity.is_null());
	_mpm_entity->clear_particles();
	real_t sp = _particle_spacing;
	real_t mass_per_particle = (_material.is_valid() ? _material->get_density() : 1000.0f) * sp * sp * sp;

	for (real_t x = p_box.position.x; x <= p_box.position.x + p_box.size.x; x += sp) {
		for (real_t y = p_box.position.y; y <= p_box.position.y + p_box.size.y; y += sp) {
			for (real_t z = p_box.position.z; z <= p_box.position.z + p_box.size.z; z += sp) {
				_mpm_entity->add_particle(Vector3(x, y, z), Vector3(), mass_per_particle, sp * sp * sp);
			}
		}
	}
}

void GenesisMPM3D::load_tetmesh_as_particles(const String &p_node_path, const String &p_ele_path) {
	ERR_FAIL_COND(_mpm_entity.is_null());
	gaia::mesh::TetMesh tm;
	if (gaia::mesh::MeshIO::load_tetgen(p_node_path, p_ele_path, tm) != OK) {
		ERR_PRINT("GenesisMPM3D: failed to load TetGen mesh.");
		return;
	}
	_mpm_entity->clear_particles();
	real_t total_density = _material.is_valid() ? _material->get_density() : 1000.0f;
	int tet_count = tm.element_count();
	real_t total_vol = 0.0f;
	for (int i = 0; i < tet_count; ++i) total_vol += tm.get_rest_volume(i);
	real_t mass_per_tet = total_density * total_vol / tet_count;

	for (int i = 0; i < tet_count; ++i) {
		gaia::mesh::TetMesh::Tetrahedron tet = tm.get_tetrahedron(i);
		Vector3 centroid = (tm.get_vertex(tet.v0) + tm.get_vertex(tet.v1) + tm.get_vertex(tet.v2) + tm.get_vertex(tet.v3)) * 0.25f;
		_mpm_entity->add_particle(centroid, Vector3(), mass_per_tet, tm.get_rest_volume(i));
	}
}

bool GenesisMPM3D::_find_world() {
	Node *p = get_parent();
	while (p) {
		_world = Object::cast_to<GenesisWorld>(p);
		if (_world) return true;
		p = p->get_parent();
	}
	return false;
}

uint64_t GenesisMPM3D::_generate_uid() {
	return (uint64_t(get_instance_id()) << 16) | uint64_t(Math::rand() & 0xFFFF);
}

} // namespace genesis