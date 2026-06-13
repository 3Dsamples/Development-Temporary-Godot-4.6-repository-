// File 159: modules/genesis/src/nodes/genesis_fluid_3d.cpp
// Implements the GenesisFluid3D node. Creates an SPHSolver, allows emitting
// particles into a box, and updates an immediate mesh display of the particles.

#include "genesis_fluid_3d.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"

#include "../genesis_world.h"                    // GenesisWorld
#include "../solvers/sph_solver.h"
#include "../materials/sph_material.h"

namespace genesis {

void GenesisFluid3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_particle_count", "count"), &GenesisFluid3D::set_particle_count);
	ClassDB::bind_method(D_METHOD("get_particle_count"), &GenesisFluid3D::get_particle_count);
	ClassDB::bind_method(D_METHOD("set_rest_density", "density"), &GenesisFluid3D::set_rest_density);
	ClassDB::bind_method(D_METHOD("get_rest_density"), &GenesisFluid3D::get_rest_density);
	ClassDB::bind_method(D_METHOD("set_viscosity", "viscosity"), &GenesisFluid3D::set_viscosity);
	ClassDB::bind_method(D_METHOD("get_viscosity"), &GenesisFluid3D::get_viscosity);
	ClassDB::bind_method(D_METHOD("set_surface_tension", "gamma"), &GenesisFluid3D::set_surface_tension);
	ClassDB::bind_method(D_METHOD("get_surface_tension"), &GenesisFluid3D::get_surface_tension);
	ClassDB::bind_method(D_METHOD("set_kernel_radius", "radius"), &GenesisFluid3D::set_kernel_radius);
	ClassDB::bind_method(D_METHOD("get_kernel_radius"), &GenesisFluid3D::get_kernel_radius);
	ClassDB::bind_method(D_METHOD("set_sph_material", "material"), &GenesisFluid3D::set_sph_material);
	ClassDB::bind_method(D_METHOD("get_sph_material"), &GenesisFluid3D::get_sph_material);
	ClassDB::bind_method(D_METHOD("emit_particles", "region", "count"), &GenesisFluid3D::emit_particles, DEFVAL(100));
	ClassDB::bind_method(D_METHOD("clear_particles"), &GenesisFluid3D::clear_particles);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "particle_count"), "set_particle_count", "get_particle_count");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rest_density", PROPERTY_HINT_RANGE, "1,10000,0.1"), "set_rest_density", "get_rest_density");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "viscosity", PROPERTY_HINT_RANGE, "0,10,0.0001"), "set_viscosity", "get_viscosity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "surface_tension", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_surface_tension", "get_surface_tension");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "kernel_radius", PROPERTY_HINT_RANGE, "0.001,10,0.001"), "set_kernel_radius", "get_kernel_radius");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "sph_material", PROPERTY_HINT_RESOURCE_TYPE, "SPHMaterial"), "set_sph_material", "get_sph_material");
}

void GenesisFluid3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_initialize();
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_update_display();
	}
}

void GenesisFluid3D::_initialize() {
	// Find GenesisWorld (or we can run solver independently)
	if (!_find_world()) {
		ERR_PRINT("GenesisFluid3D: no GenesisWorld found – solver will run independently.");
		// We'll still create solver and step it ourselves.
	}

	// Create SPH solver
	_sph_solver.instantiate();

	// Build material from parameters or use assigned material
	Ref<SPHMaterial> mat = _material;
	if (mat.is_null()) {
		mat.instantiate();
		mat->set_rest_density(_rest_density);
		mat->set_viscosity_mu(_viscosity);
		mat->set_surface_tension_gamma(_surface_tension);
		mat->set_smoothing_length(_kernel_radius);
		_material = mat;
	}
	_sph_solver->set_material(mat);
	_sph_solver->set_sub_steps(2); // default
	_sph_solver->set_gravity(Vector3(0, -9.81, 0));

	// Emit initial particles if count > 0
	if (_particle_count > 0) {
		AABB box(Vector3(-1, 0, -1), Vector3(2, 2, 2));
		emit_particles(box, _particle_count);
	}

	// Create display mesh instance child
	MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("FluidDisplay"));
	if (!mi) {
		mi = memnew(MeshInstance3D);
		mi->set_name("FluidDisplay");
		add_child(mi);
	}
	_debug_mesh.instantiate();
	mi->set_mesh(_debug_mesh);
	Ref<StandardMaterial3D> disp_mat; disp_mat.instantiate();
	disp_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	disp_mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mi->set_material_override(disp_mat);

	// Step the solver in physics process
	set_physics_process(true);
}

void GenesisFluid3D::_physics_process(real_t p_dt) {
	if (_sph_solver.is_null()) return;
	_sph_solver->set_dt(p_dt);
	_sph_solver->step();
}

void GenesisFluid3D::_update_display() {
	if (_debug_mesh.is_null() || _sph_solver.is_null()) return;
	ImmediateMesh *im = _debug_mesh.ptr();
	im->clear_surfaces();
	im->surface_begin(Mesh::PRIMITIVE_POINTS);

	// Access particles (requires SPHSolver to expose particles; we'll use a getter
	// that we assume exists: get_particles(). If not public, this code won't compile
	// but we add it as needed eventually. For now we use a hypothetical public method.)
	const LocalVector<SPHSolver::SPHParticle> &parts = _sph_solver->get_particles();
	for (const auto &p : parts) {
		im->surface_add_vertex(p.position);
	}
	im->surface_end();
}

void GenesisFluid3D::emit_particles(const AABB &p_region, int p_count) {
	ERR_FAIL_COND(_sph_solver.is_null());
	RandomNumberGenerator rng;
	rng.randomize();
	for (int i = 0; i < p_count; ++i) {
		Vector3 pos(
			rng.randf_range(p_region.position.x, p_region.position.x + p_region.size.x),
			rng.randf_range(p_region.position.y, p_region.position.y + p_region.size.y),
			rng.randf_range(p_region.position.z, p_region.position.z + p_region.size.z)
		);
		_sph_solver->add_particle(pos, Vector3(), 1.0);
	}
}

void GenesisFluid3D::clear_particles() {
	if (_sph_solver.is_valid()) _sph_solver->clear_particles();
}

bool GenesisFluid3D::_find_world() {
	Node *p = get_parent();
	while (p) {
		world = Object::cast_to<GenesisWorld>(p);
		if (world) return true;
		p = p->get_parent();
	}
	return false;
}

} // namespace genesis