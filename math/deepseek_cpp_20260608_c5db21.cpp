// File 58: modules/genesis/src/entities/fem_entity.h
// FEM entity: tetrahedral mesh with hyperelastic material, plasticity, and IPC coupling.

#ifndef GENESIS_ENTITIES_FEM_ENTITY_H
#define GENESIS_ENTITIES_FEM_ENTITY_H

#include "base_entity.h"
#include "../mesh/tet_mesh.h"            // reuse Gaia's TetMesh
#include "../materials/fem_material.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace genesis {

class FEMMaterial;

class FEMEntity : public BaseEntity {
	GDCLASS(FEMEntity, BaseEntity);

public:
	FEMEntity() :
		time_integration(TimeIntegration::IMPLICIT_EULER),
		damping_alpha(0.0),
		damping_beta(0.0),
		gravity_scale(1.0),
		ipc_enabled(false),
		ipc_distance(0.001),
		ipc_stiffness(1e6),
		plasticity_enabled(false),
		max_iter_newton(20),
		newton_tolerance(1e-6) {
		solver_type = SolverType::FEM;
	}

	// --- Mesh access ---
	gaia::mesh::TetMesh &get_mesh() { return mesh; }
	const gaia::mesh::TetMesh &get_mesh() const { return mesh; }
	void set_mesh(const gaia::mesh::TetMesh &p_mesh) { mesh = p_mesh; }

	// --- Simulation parameters ---
	void set_time_integration(TimeIntegration p_method) { time_integration = p_method; }
	TimeIntegration get_time_integration() const { return time_integration; }

	void set_damping_alpha(real_t p_val) { damping_alpha = MAX(p_val, 0.0); }
	real_t get_damping_alpha() const { return damping_alpha; }

	void set_damping_beta(real_t p_val) { damping_beta = MAX(p_val, 0.0); }
	real_t get_damping_beta() const { return damping_beta; }

	void set_gravity_scale(real_t p_val) { gravity_scale = p_val; }
	real_t get_gravity_scale() const { return gravity_scale; }

	void set_ipc_enabled(bool p_enabled) { ipc_enabled = p_enabled; }
	bool is_ipc_enabled() const { return ipc_enabled; }

	void set_ipc_distance(real_t p_dist) { ipc_distance = MAX(p_dist, 0.0); }
	real_t get_ipc_distance() const { return ipc_distance; }

	void set_ipc_stiffness(real_t p_stiff) { ipc_stiffness = MAX(p_stiff, 0.0); }
	real_t get_ipc_stiffness() const { return ipc_stiffness; }

	void set_plasticity_enabled(bool p_enabled) { plasticity_enabled = p_enabled; }
	bool is_plasticity_enabled() const { return plasticity_enabled; }

	void set_max_iter_newton(int p_iter) { max_iter_newton = MAX(p_iter, 1); }
	int get_max_iter_newton() const { return max_iter_newton; }

	void set_newton_tolerance(real_t p_tol) { newton_tolerance = MAX(p_tol, 1e-12); }
	real_t get_newton_tolerance() const { return newton_tolerance; }

	// --- Initialize from options (calls mesh loading if paths provided) ---
	virtual void init_from_options(const genesis::options::Options &opts) override {
		BaseEntity::init_from_options(opts);
		time_integration = TimeIntegration(opts.get_int("fem.time_integration", int(TimeIntegration::IMPLICIT_EULER)));
		damping_alpha = opts.get_real("fem.damping_alpha", 0.0);
		damping_beta = opts.get_real("fem.damping_beta", 0.0);
		gravity_scale = opts.get_real("fem.gravity_scale", 1.0);
		ipc_enabled = opts.get_bool("ipc.enabled", false);
		if (ipc_enabled) {
			ipc_distance = opts.get_real("ipc.distance", 0.001);
			ipc_stiffness = opts.get_real("ipc.stiffness", 1e6);
		}
		plasticity_enabled = opts.get_bool("fem.plasticity", false);
		max_iter_newton = opts.get_int("fem.max_iter", 20);
		newton_tolerance = opts.get_real("fem.newton_tol", 1e-6);
	}

	virtual AABB get_aabb() const override {
		// Build AABB from current vertex positions
		if (mesh.vertex_count() == 0) return AABB(transform.origin, Vector3());
		Vector3 minv(INFINITY, INFINITY, INFINITY);
		Vector3 maxv(-INFINITY, -INFINITY, -INFINITY);
		for (int i = 0; i < mesh.vertex_count(); ++i) {
			const Vector3 &v = mesh.get_vertex(i);
			minv = minv.min(v);
			maxv = maxv.max(v);
		}
		return AABB(minv, maxv - minv);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_time_integration", "method"), &FEMEntity::set_time_integration);
		ClassDB::bind_method(D_METHOD("get_time_integration"), &FEMEntity::get_time_integration);
		ClassDB::bind_method(D_METHOD("set_damping_alpha", "alpha"), &FEMEntity::set_damping_alpha);
		ClassDB::bind_method(D_METHOD("get_damping_alpha"), &FEMEntity::get_damping_alpha);
		ClassDB::bind_method(D_METHOD("set_damping_beta", "beta"), &FEMEntity::set_damping_beta);
		ClassDB::bind_method(D_METHOD("get_damping_beta"), &FEMEntity::get_damping_beta);
		ClassDB::bind_method(D_METHOD("set_gravity_scale", "scale"), &FEMEntity::set_gravity_scale);
		ClassDB::bind_method(D_METHOD("get_gravity_scale"), &FEMEntity::get_gravity_scale);
		ClassDB::bind_method(D_METHOD("set_ipc_enabled", "enabled"), &FEMEntity::set_ipc_enabled);
		ClassDB::bind_method(D_METHOD("is_ipc_enabled"), &FEMEntity::is_ipc_enabled);
		ClassDB::bind_method(D_METHOD("set_ipc_distance", "distance"), &FEMEntity::set_ipc_distance);
		ClassDB::bind_method(D_METHOD("get_ipc_distance"), &FEMEntity::get_ipc_distance);
		ClassDB::bind_method(D_METHOD("set_ipc_stiffness", "stiffness"), &FEMEntity::set_ipc_stiffness);
		ClassDB::bind_method(D_METHOD("get_ipc_stiffness"), &FEMEntity::get_ipc_stiffness);
		ClassDB::bind_method(D_METHOD("set_plasticity_enabled", "enabled"), &FEMEntity::set_plasticity_enabled);
		ClassDB::bind_method(D_METHOD("is_plasticity_enabled"), &FEMEntity::is_plasticity_enabled);
		ClassDB::bind_method(D_METHOD("set_max_iter_newton", "iter"), &FEMEntity::set_max_iter_newton);
		ClassDB::bind_method(D_METHOD("get_max_iter_newton"), &FEMEntity::get_max_iter_newton);
		ClassDB::bind_method(D_METHOD("set_newton_tolerance", "tolerance"), &FEMEntity::set_newton_tolerance);
		ClassDB::bind_method(D_METHOD("get_newton_tolerance"), &FEMEntity::get_newton_tolerance);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "time_integration", PROPERTY_HINT_ENUM, "ExplicitEuler,SymplecticEuler,ImplicitEuler,NewmarkBeta,BDF1,BDF2,RK4"), "set_time_integration", "get_time_integration");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping_alpha", PROPERTY_HINT_RANGE, "0,10,0.001"), "set_damping_alpha", "get_damping_alpha");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping_beta", PROPERTY_HINT_RANGE, "0,10,0.001"), "set_damping_beta", "get_damping_beta");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale"), "set_gravity_scale", "get_gravity_scale");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ipc_enabled"), "set_ipc_enabled", "is_ipc_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ipc_distance", PROPERTY_HINT_RANGE, "0,1,0.0001"), "set_ipc_distance", "get_ipc_distance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ipc_stiffness", PROPERTY_HINT_RANGE, "0,1e9,1"), "set_ipc_stiffness", "get_ipc_stiffness");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "plasticity_enabled"), "set_plasticity_enabled", "is_plasticity_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iter_newton", PROPERTY_HINT_RANGE, "1,50,1"), "set_max_iter_newton", "get_max_iter_newton");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "newton_tolerance", PROPERTY_HINT_RANGE, "1e-12,1,1e-6"), "set_newton_tolerance", "get_newton_tolerance");
	}

private:
	gaia::mesh::TetMesh mesh;
	TimeIntegration time_integration;
	real_t damping_alpha, damping_beta;
	real_t gravity_scale;
	bool ipc_enabled;
	real_t ipc_distance;
	real_t ipc_stiffness;
	bool plasticity_enabled;
	int max_iter_newton;
	real_t newton_tolerance;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_FEM_ENTITY_H