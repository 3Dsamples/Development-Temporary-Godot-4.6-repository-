// File 52: modules/genesis/src/core/genesis_constants.h
// Physics constants and enumerations used across all Genesis solvers.

#ifndef GENESIS_CORE_GENESIS_CONSTANTS_H
#define GENESIS_CORE_GENESIS_CONSTANTS_H

#include "genesis_types.h"

namespace genesis {

// --- Solver categories (mirrors Genesis' SolverType enum) ---
enum class SolverType : uint8_t {
	RIGID = 0,
	FEM   = 1,
	MPM   = 2,
	PBD   = 3,
	SPH   = 4,
	SF    = 5,   // stable fluids (Eulerian)
	TOOL  = 6,
	CUSTOM = 255
};

// --- Time integration schemes ---
enum class TimeIntegration : uint8_t {
	EXPLICIT_EULER = 0,
	SYMPLECTIC_EULER,
	IMPLICIT_EULER,
	NEWMARK_BETA,
	BDF1,
	BDF2,
	RUNGE_KUTTA_4
};

// --- Collision geometry types ---
enum class GeometryType : uint8_t {
	SPHERE       = 0,
	BOX          = 1,
	CAPSULE      = 2,
	CYLINDER     = 3,
	CONVEX_MESH  = 4,
	HEIGHTFIELD  = 5,
	TRI_MESH     = 6,
	SDF          = 7,
	POINT_CLOUD  = 8
};

// --- Boundary condition types ---
enum class BCType : uint8_t {
	DIRICHLET = 0,   // prescribed position / velocity
	NEUMANN   = 1,   // prescribed force / traction
	CONTACT   = 2,   // contact inequality constraint
	FRICTION  = 3
};

// --- Sensor types ---
enum class SensorType : uint8_t {
	CAMERA          = 0,
	CONTACT_FORCE   = 1,
	IMU             = 2,
	LIDAR           = 3,
	PRESSURE        = 4,
	JOINT           = 5,
	CUSTOM_SENSOR   = 255
};

// --- Physical quantities (unit system – SI) ---
constexpr real GRAVITY_EARTH = 9.80665;
constexpr real IDEAL_GAS_CONSTANT = 8.31446261815324;

// --- Default solver parameters ---
constexpr int   DEFAULT_SOLVER_ITERATIONS = 10;
constexpr real DEFAULT_DT                 = 1.0 / 60.0;
constexpr real DEFAULT_DAMPING            = 0.001;
constexpr int   DEFAULT_MPM_GRID_RES      = 64;

} // namespace genesis

#endif // GENESIS_CORE_GENESIS_CONSTANTS_H