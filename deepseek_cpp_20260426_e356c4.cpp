// File 174: modules/newton/src/core/newton_constants.h
// Global constants, enumerations, and default parameters for Newton Dynamics.

#ifndef NEWTON_CORE_CONSTANTS_H
#define NEWTON_CORE_CONSTANTS_H

#include "core/typedefs.h"
#include "core/math/vector3.h"

namespace newton {

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
	BRUTE_FORCE      = 0,
	SAP              = 1,    // sweep and prune
	BVH              = 2     // bounding volume hierarchy (Gaia)
};

// ---------------------------------------------------------------------------
// Solver method
// ---------------------------------------------------------------------------
enum class SolverMethod : uint8_t {
	ITERATIVE_GAUSS_SEIDEL = 0,
	ITERATIVE_ACCELERATED  = 1,   // Newton's own accelerated iterative
	DIRECT_PARDISO         = 2    // for small systems
};

// ---------------------------------------------------------------------------
// Joint types
// ---------------------------------------------------------------------------
enum class JointType : uint8_t {
	BALL          = 0,
	HINGE         = 1,
	SLIDER        = 2,
	UNIVERSAL     = 3,
	CORKSCREW     = 4,
	FIXED_DISTANCE = 5,
	KINEMATIC     = 6,
	CUSTOM        = 7
};

// ---------------------------------------------------------------------------
// Collision shape type
// ---------------------------------------------------------------------------
enum class ShapeType : uint8_t {
	SPHERE       = 0,
	BOX          = 1,
	CAPSULE      = 2,
	CYLINDER     = 3,
	CONE         = 4,
	CONVEX_HULL  = 5,
	HEIGHTFIELD  = 6,
	BVH_TRI_MESH = 7
};

// ---------------------------------------------------------------------------
// Default physical quantities (SI)
// ---------------------------------------------------------------------------
constexpr real DEFAULT_GRAVITY          = -9.80665;
constexpr real DEFAULT_SLEEP_LINEAR     = 0.01;
constexpr real DEFAULT_SLEEP_ANGULAR    = 0.01;
constexpr int  DEFAULT_SLEEP_FRAMES     = 10;
constexpr int  DEFAULT_SOLVER_ITERATIONS = 16;

// ---------------------------------------------------------------------------
// Material defaults
// ---------------------------------------------------------------------------
constexpr real DEFAULT_FRICTION         = 0.5;
constexpr real DEFAULT_RESTITUTION      = 0.0;
constexpr real DEFAULT_SOFTNESS         = 0.001;

// ---------------------------------------------------------------------------
// Island activation thresholds
// ---------------------------------------------------------------------------
constexpr real ISLAND_ACTIVATION_SPEED  = 0.005;

} // namespace newton

#endif // NEWTON_CORE_CONSTANTS_H