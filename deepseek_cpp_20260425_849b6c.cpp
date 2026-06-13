// File 12: modules/gaia/src/framework/constraint.h

#ifndef GAIA_FRAMEWORK_CONSTRAINT_H
#define GAIA_FRAMEWORK_CONSTRAINT_H

#include "core/typedefs.h"
#include "core/math/vector3.h"

namespace gaia {

// Forward declarations
class RigidBody;
class SoftBody;

/**
 * Base class for all simulation constraints (XPBD, PBD, etc.).
 * Derived classes implement the position-level projection and
 * optionally velocity-level corrections.
 */
class Constraint {
public:
	enum Type {
		DISTANCE,
		BENDING,
		VOLUME,
		COLLISION,
		CUSTOM
	};

	Constraint() : type(CUSTOM), compliance(0.0), damping(0.0) {}
	virtual ~Constraint() {}

	Type get_type() const { return type; }

	// Position-based solve: correct positions to satisfy constraint.
	virtual void solve_position(real_t dt) = 0;

	// Velocity-based correction (optional, e.g. for damping).
	virtual void solve_velocity(real_t dt) {}

	// XPBD compliance (inverse stiffness). 0 = hard constraint.
	void set_compliance(real_t c) { compliance = MAX(c, 0.0); }
	real_t get_compliance() const { return compliance; }

	// Damping coefficient (0 = none, 1 = critical).
	void set_damping(real_t d) { damping = CLAMP(d, 0.0, 1.0); }
	real_t get_damping() const { return damping; }

protected:
	Type type;
	real_t compliance;
	real_t damping;
};

} // namespace gaia

#endif // GAIA_FRAMEWORK_CONSTRAINT_H