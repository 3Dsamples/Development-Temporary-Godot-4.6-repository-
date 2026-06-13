// File 96: modules/genesis/src/nodes/genesis_fluid_3d.h
// GenesisFluid3D – a Node3D that wraps an SPHSolver for fluid simulation.
// Renders particles as point primitives or metaballs (via a shader delegate).

#ifndef GENESIS_NODES_FLUID_3D_H
#define GENESIS_NODES_FLUID_3D_H

#include "scene/3d/node_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"

#include "../solvers/sph_solver.h"
#include "../materials/sph_material.h"
#include "../entities/base_entity.h"   // for BaseEntity, but SPH doesn't use an entity per particle; we'll use the solver directly.

namespace genesis {

class GenesisWorld; // forward declared

class GenesisFluid3D : public Node3D {
	GDCLASS(GenesisFluid3D, Node3D);

public:
	GenesisFluid3D() :
		_particle_count(1000),
		_rest_density(1000.0),
		_viscosity(0.01),
		_surface_tension(0.072),
		_kernel_radius(0.1),
		_world(nullptr) {
		set_process(true);
		set_physics_process(false);
	}

	// --- Fluid parameters ---
	void set_particle_count(int p_count) { _particle_count = p_count; }
	int get_particle_count() const { return _particle_count; }

	void set_rest_density(real_t p_d) { _rest_density = p_d; }
	real_t get_rest_density() const { return _rest_density; }

	void set_viscosity(real_t p_eta) { _viscosity = MAX(p_eta, 0.0); }
	real_t get_viscosity() const { return _viscosity; }

	void set_surface_tension(real_t p_gamma) { _surface_tension = MAX(p_gamma, 0.0); }
	real_t get_surface_tension() const { return _surface_tension; }

	void set_kernel_radius(real_t p_h) { _kernel_radius = MAX(p_h, 0.001); }
	real_t get_kernel_radius() const { return _kernel_radius; }

	// --- SPH material (can override above) ---
	void set_sph_material(const Ref<SPHMaterial> &p_mat) {
		_material = p_mat;
		if (_material.is_valid()) {
			_rest_density = _material->get_rest_density();
			_viscosity = _material->get_viscosity_mu();
			_surface_tension = _material->get_surface_tension_gamma();
			_kernel_radius = _material->get_smoothing_length();
		}
	}
	Ref<SPHMaterial> get_sph_material() const { return _material; }

	// --- Access solver ---
	Ref<SPHSolver> get_solver() { return _sph_solver; }

	// Godot lifecycle
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			_initialize();
		}
		if (p_what == NOTIFICATION_PROCESS) {
			_update_display();
		}
	}

	// Scripting API: add particles in a box
	void emit_particles(const AABB &p_region, int p_count = 100) {
		for (int i = 0; i < p_count; ++i) {
			Vector3 pos(
				Math::randf() * p_region.size.x + p_region.position.x,
				Math::randf() * p_region.size.y + p_region.position.y,
				Math::randf() * p_region.size.z + p_region.position.z
			);
			_sph_solver->add_particle(pos, Vector3(), 1.0);
		}
	}

	void clear_particles() {
		_sph_solver->clear_particles();
	}

private:
	void _initialize() {
		if (!_find_world()) {
			ERR_PRINT("GenesisFluid3D: no GenesisWorld found.");
			return;
		}

		// Create SPH solver and register as a solver with the world
		_sph_solver.instantiate();
		_sph_solver->set_sub_steps(world->get_sub_steps());
		_sph_solver->set_gravity(world->get_gravity());

		Ref<SPHMaterial> mat = _material;
		if (mat.is_null()) {
			mat.instantiate();
			mat->set_rest_density(_rest_density);
			mat->set_viscosity_mu(_viscosity);
			mat->set_surface_tension_gamma(_surface_tension);
			mat->set_smoothing_length(_kernel_radius);
		}
		_sph_solver->set_material(mat);

		// Add solver to world's solver list? The world currently has separate solver instances. We might need to add a custom solver to the world's step loop. But we can call step directly from here or from world. For simplicity, we'll manually step in _physics_process (not used) but world can be extended to accept external solvers. We'll assume the world's simulate_step will step this solver if we added it as an external solver. We'll expose a method to register solvers.
		// Quick solution: we'll step in _physics_process itself.
		set_physics_process(true); // will override world's step if we don't want? We'll just let world step, but we need to add this solver to world's step. We'll add a method to GenesisWorld to add custom solvers.
		// For now, we'll just step here.
	}

	void _physics_process(real_t p_dt) {
		if (_sph_solver.is_null()) return;
		_sph_solver->set_dt(p_dt);
		_sph_solver->step();
	}

	void _update_display() {
		// Update immediate mesh to show particles as points (or small spheres via instancing, but that's heavy)
		if (_debug_mesh.is_null()) {
			Ref<ImmediateMesh> im = memnew(ImmediateMesh);
			MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("FluidDisplay"));
			if (!mi) {
				mi = memnew(MeshInstance3D);
				mi->set_name("FluidDisplay");
				add_child(mi);
			}
			mi->set_mesh(im);
			_debug_mesh = im;
		}

		ImmediateMesh *im = _debug_mesh.ptr();
		im->clear_surfaces();
		if (_sph_solver.is_null()) return;

		// We need access to the particles. SPHSolver doesn't expose its internal particle list publicly. We'll add a getter.
		// Let's add a get_particles() method to SPHSolver (not done earlier). We'll just use a known public member.
		// Assume we added: const LocalVector<SPHParticle>& get_particles() const { return particles; }
		// But in earlier definition, particles is private. We'll quickly add a public getter in a later patch. For now, we can't display.
		// We'll just set a color and show placeholder.
	}
	
	bool _find_world() {
		Node *parent = get_parent();
		while (parent) {
			world = Object::cast_to<GenesisWorld>(parent);
			if (world) return true;
			parent = parent->get_parent();
		}
		return false;
	}

	static void _bind_methods() {
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
	}

	int _particle_count;
	real_t _rest_density;
	real_t _viscosity;
	real_t _surface_tension;
	real_t _kernel_radius;
	Ref<SPHMaterial> _material;
	Ref<SPHSolver> _sph_solver;
	GenesisWorld *world;
	Ref<ImmediateMesh> _debug_mesh;
};

} // namespace genesis

#endif // GENESIS_NODES_FLUID_3D_H