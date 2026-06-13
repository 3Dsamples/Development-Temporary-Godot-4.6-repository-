// File 328: modules/wicked/src/core/wicked_constants.h
// Global constants, enumerations, and default parameters for the WickedEngine physics module.

#ifndef WICKED_CORE_CONSTANTS_H
#define WICKED_CORE_CONSTANTS_H

#include "core/typedefs.h"
#include "core/math/vector3.h"

namespace wicked {

// ---------------------------------------------------------------------------
// Body type
// ---------------------------------------------------------------------------
enum class BodyType : uint8_t {
	STATIC    = 0,
	DYNAMIC   = 1,
	KINEMATIC = 2
};

// ---------------------------------------------------------------------------
// Broad‑phase algorithm
// ---------------------------------------------------------------------------
enum class BroadPhaseAlgorithm : uint8_t {
	BRUTE_FORCE  = 0,
	SAP           = 1,      // Sweep and Prune
	DBVT          = 2       // Dynamic Bounding Volume Tree (Bullet style)
};

// ---------------------------------------------------------------------------
// Solver method
// ---------------------------------------------------------------------------
enum class SolverMethod : uint8_t {
	SEQUENTIAL_IMPULSES     = 0,
	PROJECTED_GAUSS_SEIDEL  = 1,
	ACCELERATED_PGS         = 2
};

// ---------------------------------------------------------------------------
// Joint types
// ---------------------------------------------------------------------------
enum class JointType : uint8_t {
	BALL            = 0,
	HINGE           = 1,
	SLIDER          = 2,
	FIXED           = 3,
	CONE_TWIST      = 4,
	GENERIC_6DOF    = 5,
	POINT_TO_POINT  = 6,
	CUSTOM          = 7
};

// ---------------------------------------------------------------------------
// Collision shape type
// ---------------------------------------------------------------------------
enum class ShapeType : uint8_t {
	SPHERE        = 0,
	BOX           = 1,
	CAPSULE       = 2,
	CYLINDER      = 3,
	CONE          = 4,
	CONVEX_HULL   = 5,
	TRIANGLE_MESH = 6,
	HEIGHTFIELD   = 7,
	COMPOUND      = 8
};

// ---------------------------------------------------------------------------
// Activation state
// ---------------------------------------------------------------------------
enum class ActivationState : uint8_t {
	ACTIVE_TAG        = 0,
	ISLAND_SLEEPING   = 1,
	WANTS_DEACTIVATION = 2,
	DISABLE_DEACTIVATION = 3,
	DISABLE_SIMULATION = 4
};

// ---------------------------------------------------------------------------
// Default physical quantities (SI units)
// ---------------------------------------------------------------------------
constexpr real DEFAULT_GRAVITY           = -9.80665;
constexpr real DEFAULT_SLEEP_LINEAR      = 0.8;
constexpr real DEFAULT_SLEEP_ANGULAR     = 1.0;
constexpr int  DEFAULT_SLEEP_FRAMES      = 10;
constexpr int  DEFAULT_SOLVER_ITERATIONS = 10;
constexpr real DEFAULT_STEP_SIZE         = 1.0 / 60.0;
constexpr real DEFAULT_CONTACT_BREAKING_THRESHOLD = 0.02;

// ---------------------------------------------------------------------------
// Material defaults
// ---------------------------------------------------------------------------
constexpr real DEFAULT_FRICTION          = 0.5;
constexpr real DEFAULT_RESTITUTION       = 0.0;
constexpr real DEFAULT_ROLLING_FRICTION   = 0.0;
constexpr real DEFAULT_SPINNING_FRICTION  = 0.0;

// ---------------------------------------------------------------------------
// Solver relaxation
// ---------------------------------------------------------------------------
constexpr real DEFAULT_ERP               = 0.2;   // error reduction parameter
constexpr real DEFAULT_ERP2              = 0.8;   // error reduction parameter for split impulse
constexpr real DEFAULT_TAU               = 0.6;   // constraint force mixing (CFM)

// ---------------------------------------------------------------------------
// CCD motion threshold (fraction of body size)
// ---------------------------------------------------------------------------
constexpr real DEFAULT_CCD_MOTION_THRESHOLD = 0.0;
constexpr real DEFAULT_CCD_SWEPT_SPHERE_RADIUS = 0.2;

// ---------------------------------------------------------------------------
// Vehicle defaults
// ---------------------------------------------------------------------------
constexpr real DEFAULT_SUSPENSION_STIFFNESS  = 5.88;
constexpr real DEFAULT_SUSPENSION_COMPRESSION = 0.83;
constexpr real DEFAULT_SUSPENSION_DAMPING     = 0.88;
constexpr real DEFAULT_WHEEL_FRICTION         = 1000.0;
constexpr real DEFAULT_ROLL_INFLUENCE         = 0.1;

} // namespace wicked

#endif // WICKED_CORE_CONSTANTS_H