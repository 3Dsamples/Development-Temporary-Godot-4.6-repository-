// File 18: modules/gaia/src/materials/material.h

#ifndef GAIA_MATERIALS_MATERIAL_H
#define GAIA_MATERIALS_MATERIAL_H

#include "core/typedefs.h"
#include "core/string/ustring.h"

namespace gaia {

/**
 * Physical material properties used by rigid bodies, soft bodies,
 * and collision response.
 */
struct Material {
	String name;
	real_t density;             // kg/m³
	real_t young_modulus;       // Pa (for soft bodies)
	real_t poisson_ratio;       // 0..0.5
	real_t damping_coefficient; // position/velocity damping (PBD/XPBD)
	real_t friction;            // Coulomb friction coefficient
	real_t restitution;         // coefficient of restitution

	Material() :
		name("default"),
		density(1000.0f),
		young_modulus(1e6f),
		poisson_ratio(0.3f),
		damping_coefficient(0.0f),
		friction(0.5f),
		restitution(0.0f) {}

	// Derived Lame parameters (linear elasticity)
	real_t get_lambda() const {
		real_t nu = CLAMP(poisson_ratio, 0.0f, 0.49f);
		return young_modulus * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
	}
	real_t get_mu() const {
		real_t nu = CLAMP(poisson_ratio, 0.0f, 0.49f);
		return young_modulus / (2.0f * (1.0f + nu));
	}

	// Stiffness for XPBD distance constraints (compliance = 1/stiffness)
	real_t get_distance_stiffness() const {
		return young_modulus;
	}

	bool operator==(const Material &other) const {
		return density == other.density &&
			   young_modulus == other.young_modulus &&
			   poisson_ratio == other.poisson_ratio &&
			   damping_coefficient == other.damping_coefficient &&
			   friction == other.friction &&
			   restitution == other.restitution;
	}
	bool operator!=(const Material &other) const { return !(*this == other); }
};

} // namespace gaia

#endif // GAIA_MATERIALS_MATERIAL_H