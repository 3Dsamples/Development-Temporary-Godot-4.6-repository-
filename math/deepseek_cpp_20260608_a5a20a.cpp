// File 269: modules/vienna/src/core/vienna_constants.h
// Global constants, enumerations, and default parameters for ViennaPhysicsEngine.

#ifndef VIENNA_CORE_CONSTANTS_H
#define VIENNA_CORE_CONSTANTS_H

#include "core/typedefs.h"
#include "core/math/vector3.h"

namespace vienna {

// ---------------------------------------------------------------------------
// Body type
// ---------------------------------------------------------------------------
enum class BodyType : uint8_t {
	STATIC    = 0,
	DYNAMIC   = 1,
	KINEMATIC = 2
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
	TRI_MESH     = 7
};

// ---------------------------------------------------------------------------
// Joint types
// ---------------------------------------------------------------------------
enum class JointType : uint8_t {
	BALL      = 0,
	HINGE     = 1,
	SLIDER    = 2,
	FIXED     = 3,
	DISTANCE  = 4,
	ROPE      = 5,
	CUSTOM    = 6
};

// ---------------------------------------------------------------------------
// Solver method
// ---------------------------------------------------------------------------
enum class SolverMethod : uint8_t {
	SEQUENTIAL_IMPULSES = 0,
	PROJECTED_GAUSS_SEIDEL = 1,
	DIRECT_PARDISO = 2
};

// ---------------------------------------------------------------------------
// Broad‑phase algorithm
// ---------------------------------------------------------------------------
enum class BroadPhaseAlgorithm : uint8_t {
	BRUTE_FORCE = 0,
	SAP         = 1,       // sweep and prune
	BVH         = 2        // bounding volume hierarchy (Gaia)
};

// ---------------------------------------------------------------------------
// Cloth solver
// ---------------------------------------------------------------------------
enum class ClothSolverType : uint8_t {
	MASS_SPRING   = 0,
	XPBD          = 1,
	VBD           = 2
};

// ---------------------------------------------------------------------------
// Particle system type
// ---------------------------------------------------------------------------
enum class ParticleSystemType : uint8_t {
	SPH           = 0,
	GRANULAR      = 1,
	PBD           = 2
};

// ---------------------------------------------------------------------------
// Default physical quantities (SI)
// ---------------------------------------------------------------------------
constexpr real DEFAULT_GRAVITY           = -9.80665;
constexpr real DEFAULT_SLEEP_LINEAR      = 0.01;
constexpr real DEFAULT_SLEEP_ANGULAR     = 0.01;
constexpr int  DEFAULT_SLEEP_FRAMES      = 10;
constexpr int  DEFAULT_SOLVER_ITERATIONS = 16;
constexpr real DEFAULT_DAMPING           = 0.005;
constexpr real DEFAULT_FRICTION          = 0.5;
constexpr real DEFAULT_RESTITUTION       = 0.0;
constexpr real DEFAULT_SOFTNESS          = 0.001;
constexpr real DEFAULT_CLOTH_DAMPING     = 0.01;
constexpr real DEFAULT_CLOTH_STIFFNESS   = 1000.0;

} // namespace vienna

#endif // VIENNA_CORE_CONSTANTS_H