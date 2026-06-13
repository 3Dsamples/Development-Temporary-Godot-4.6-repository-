// File 139: modules/gaia/src/vbd_cloth/vbd_cloth_physics_parameters.h
// Stores all modifiable parameters for VBD cloth simulation.
// Exposed as a Godot Resource so it can be saved, loaded and edited in the inspector.

#ifndef GAIA_VBD_CLOTH_PHYSICS_PARAMETERS_H
#define GAIA_VBD_CLOTH_PHYSICS_PARAMETERS_H

#include "core/io/resource.h"
#include "core/typedefs.h"

namespace gaia::vbd_cloth {

class VBDClothPhysicsParameters : public Resource {
	GDCLASS(VBDClothPhysicsParameters, Resource);

public:
	// --- Time stepping ---
	real_t dt = 1.0 / 60.0;
	int    sub_steps = 2;
	int    solver_iterations = 5;

	// --- Gravity ---
	Vector3 gravity = Vector3(0.0, -9.81, 0.0);

	// --- Material (linear elasticity) ---
	real_t young_modulus = 1e5;         // Pa
	real_t poisson_ratio = 0.3;
	real_t density = 1000.0;            // kg/m³
	real_t thickness = 0.001;           // m

	// --- Damping ---
	real_t velocity_damping = 0.01;
	real_t collision_damping = 0.0;

	// --- Constraints ---
	real_t stretch_compliance = 1e-7;
	real_t shear_compliance = 1e-6;
	real_t bending_compliance = 1e-5;

	// --- Collision ---
	bool   collision_enabled = true;
	real_t collision_margin = 0.005;
	real_t friction_coefficient = 0.5;
	real_t restitution_coefficient = 0.0;

	// --- Bending model ---
	bool   use_quadratic_bending = false; // if true, use quadratic bending energy

	// --- Derived Lame constants ---
	real_t get_lame_mu() const {
		return young_modulus / (2.0 * (1.0 + poisson_ratio));
	}
	real_t get_lame_lambda() const {
		real_t nu = poisson_ratio;
		return young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
	}

	// --- Load from dictionary (useful for JSON config) ---
	void load_from_dict(const Dictionary &p_dict) {
		dt = p_dict.get("dt", dt);
		sub_steps = p_dict.get("sub_steps", sub_steps);
		solver_iterations = p_dict.get("iterations", solver_iterations);
		if (p_dict.has("gravity")) {
			Array g = p_dict["gravity"];
			if (g.size() == 3) gravity = Vector3(g[0], g[1], g[2]);
		}
		young_modulus = p_dict.get("young_modulus", young_modulus);
		poisson_ratio = p_dict.get("poisson_ratio", poisson_ratio);
		density = p_dict.get("density", density);
		thickness = p_dict.get("thickness", thickness);
		velocity_damping = p_dict.get("velocity_damping", velocity_damping);
		collision_damping = p_dict.get("collision_damping", collision_damping);
		stretch_compliance = p_dict.get("stretch_compliance", stretch_compliance);
		shear_compliance = p_dict.get("shear_compliance", shear_compliance);
		bending_compliance = p_dict.get("bending_compliance", bending_compliance);
		collision_enabled = p_dict.get("collision_enabled", collision_enabled);
		collision_margin = p_dict.get("collision_margin", collision_margin);
		friction_coefficient = p_dict.get("friction_coefficient", friction_coefficient);
		restitution_coefficient = p_dict.get("restitution_coefficient", restitution_coefficient);
		use_quadratic_bending = p_dict.get("quadratic_bending", use_quadratic_bending);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_dt", "dt"), &VBDClothPhysicsParameters::set_dt);
		ClassDB::bind_method(D_METHOD("get_dt"), &VBDClothPhysicsParameters::get_dt);
		ClassDB::bind_method(D_METHOD("set_sub_steps", "sub_steps"), &VBDClothPhysicsParameters::set_sub_steps);
		ClassDB::bind_method(D_METHOD("get_sub_steps"), &VBDClothPhysicsParameters::get_sub_steps);
		ClassDB::bind_method(D_METHOD("set_solver_iterations", "iter"), &VBDClothPhysicsParameters::set_solver_iterations);
		ClassDB::bind_method(D_METHOD("get_solver_iterations"), &VBDClothPhysicsParameters::get_solver_iterations);
		ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &VBDClothPhysicsParameters::set_gravity);
		ClassDB::bind_method(D_METHOD("get_gravity"), &VBDClothPhysicsParameters::get_gravity);
		ClassDB::bind_method(D_METHOD("set_young_modulus", "E"), &VBDClothPhysicsParameters::set_young_modulus);
		ClassDB::bind_method(D_METHOD("get_young_modulus"), &VBDClothPhysicsParameters::get_young_modulus);
		ClassDB::bind_method(D_METHOD("set_poisson_ratio", "nu"), &VBDClothPhysicsParameters::set_poisson_ratio);
		ClassDB::bind_method(D_METHOD("get_poisson_ratio"), &VBDClothPhysicsParameters::get_poisson_ratio);
		ClassDB::bind_method(D_METHOD("set_density", "rho"), &VBDClothPhysicsParameters::set_density);
		ClassDB::bind_method(D_METHOD("get_density"), &VBDClothPhysicsParameters::get_density);
		ClassDB::bind_method(D_METHOD("set_thickness", "thickness"), &VBDClothPhysicsParameters::set_thickness);
		ClassDB::bind_method(D_METHOD("get_thickness"), &VBDClothPhysicsParameters::get_thickness);
		ClassDB::bind_method(D_METHOD("set_velocity_damping", "damping"), &VBDClothPhysicsParameters::set_velocity_damping);
		ClassDB::bind_method(D_METHOD("get_velocity_damping"), &VBDClothPhysicsParameters::get_velocity_damping);
		ClassDB::bind_method(D_METHOD("set_collision_damping", "damping"), &VBDClothPhysicsParameters::set_collision_damping);
		ClassDB::bind_method(D_METHOD("get_collision_damping"), &VBDClothPhysicsParameters::get_collision_damping);
		ClassDB::bind_method(D_METHOD("set_stretch_compliance", "c"), &VBDClothPhysicsParameters::set_stretch_compliance);
		ClassDB::bind_method(D_METHOD("get_stretch_compliance"), &VBDClothPhysicsParameters::get_stretch_compliance);
		ClassDB::bind_method(D_METHOD("set_shear_compliance", "c"), &VBDClothPhysicsParameters::set_shear_compliance);
		ClassDB::bind_method(D_METHOD("get_shear_compliance"), &VBDClothPhysicsParameters::get_shear_compliance);
		ClassDB::bind_method(D_METHOD("set_bending_compliance", "c"), &VBDClothPhysicsParameters::set_bending_compliance);
		ClassDB::bind_method(D_METHOD("get_bending_compliance"), &VBDClothPhysicsParameters::get_bending_compliance);
		ClassDB::bind_method(D_METHOD("set_collision_enabled", "enabled"), &VBDClothPhysicsParameters::set_collision_enabled);
		ClassDB::bind_method(D_METHOD("is_collision_enabled"), &VBDClothPhysicsParameters::is_collision_enabled);
		ClassDB::bind_method(D_METHOD("set_collision_margin", "margin"), &VBDClothPhysicsParameters::set_collision_margin);
		ClassDB::bind_method(D_METHOD("get_collision_margin"), &VBDClothPhysicsParameters::get_collision_margin);
		ClassDB::bind_method(D_METHOD("set_friction_coefficient", "mu"), &VBDClothPhysicsParameters::set_friction_coefficient);
		ClassDB::bind_method(D_METHOD("get_friction_coefficient"), &VBDClothPhysicsParameters::get_friction_coefficient);
		ClassDB::bind_method(D_METHOD("set_restitution_coefficient", "e"), &VBDClothPhysicsParameters::set_restitution_coefficient);
		ClassDB::bind_method(D_METHOD("get_restitution_coefficient"), &VBDClothPhysicsParameters::get_restitution_coefficient);
		ClassDB::bind_method(D_METHOD("set_use_quadratic_bending", "enabled"), &VBDClothPhysicsParameters::set_use_quadratic_bending);
		ClassDB::bind_method(D_METHOD("is_quadratic_bending"), &VBDClothPhysicsParameters::is_quadratic_bending);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dt"), "set_dt", "get_dt");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "sub_steps"), "set_sub_steps", "get_sub_steps");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_iterations"), "set_solver_iterations", "get_solver_iterations");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "young_modulus"), "set_young_modulus", "get_young_modulus");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "poisson_ratio"), "set_poisson_ratio", "get_poisson_ratio");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "density"), "set_density", "get_density");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "thickness"), "set_thickness", "get_thickness");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "velocity_damping"), "set_velocity_damping", "get_velocity_damping");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_damping"), "set_collision_damping", "get_collision_damping");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stretch_compliance"), "set_stretch_compliance", "get_stretch_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "shear_compliance"), "set_shear_compliance", "get_shear_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bending_compliance"), "set_bending_compliance", "get_bending_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_enabled"), "set_collision_enabled", "is_collision_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_margin"), "set_collision_margin", "get_collision_margin");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction_coefficient"), "set_friction_coefficient", "get_friction_coefficient");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution_coefficient"), "set_restitution_coefficient", "get_restitution_coefficient");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "quadratic_bending"), "set_use_quadratic_bending", "is_quadratic_bending");
	}

private:
	// Setters/getters for property binding
	void set_dt(real_t v) { dt = v; }
	real_t get_dt() const { return dt; }
	void set_sub_steps(int v) { sub_steps = v; }
	int get_sub_steps() const { return sub_steps; }
	void set_solver_iterations(int v) { solver_iterations = v; }
	int get_solver_iterations() const { return solver_iterations; }
	void set_gravity(const Vector3 &v) { gravity = v; }
	Vector3 get_gravity() const { return gravity; }
	void set_young_modulus(real_t v) { young_modulus = v; }
	real_t get_young_modulus() const { return young_modulus; }
	void set_poisson_ratio(real_t v) { poisson_ratio = v; }
	real_t get_poisson_ratio() const { return poisson_ratio; }
	void set_density(real_t v) { density = v; }
	real_t get_density() const { return density; }
	void set_thickness(real_t v) { thickness = v; }
	real_t get_thickness() const { return thickness; }
	void set_velocity_damping(real_t v) { velocity_damping = v; }
	real_t get_velocity_damping() const { return velocity_damping; }
	void set_collision_damping(real_t v) { collision_damping = v; }
	real_t get_collision_damping() const { return collision_damping; }
	void set_stretch_compliance(real_t v) { stretch_compliance = v; }
	real_t get_stretch_compliance() const { return stretch_compliance; }
	void set_shear_compliance(real_t v) { shear_compliance = v; }
	real_t get_shear_compliance() const { return shear_compliance; }
	void set_bending_compliance(real_t v) { bending_compliance = v; }
	real_t get_bending_compliance() const { return bending_compliance; }
	void set_collision_enabled(bool v) { collision_enabled = v; }
	bool is_collision_enabled() const { return collision_enabled; }
	void set_collision_margin(real_t v) { collision_margin = v; }
	real_t get_collision_margin() const { return collision_margin; }
	void set_friction_coefficient(real_t v) { friction_coefficient = v; }
	real_t get_friction_coefficient() const { return friction_coefficient; }
	void set_restitution_coefficient(real_t v) { restitution_coefficient = v; }
	real_t get_restitution_coefficient() const { return restitution_coefficient; }
	void set_use_quadratic_bending(bool v) { use_quadratic_bending = v; }
	bool is_quadratic_bending() const { return use_quadratic_bending; }
};

} // namespace gaia::vbd_cloth

#endif // GAIA_VBD_CLOTH_PHYSICS_PARAMETERS_H