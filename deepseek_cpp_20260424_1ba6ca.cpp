// genesis/constants.h

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace genesis {
namespace constants {

//------------------------------------------------------------------------------
// Numerical constants
//------------------------------------------------------------------------------

// Epsilon values for different precisions
constexpr double EPS = 1e-12;
constexpr double EPS_SQ = EPS * EPS;
constexpr double EPS_CUBE = EPS * EPS * EPS;
constexpr double EPS_SQRT = 1e-6;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
constexpr double HALF_PI = 0.5 * PI;
constexpr double QUARTER_PI = 0.25 * PI;
constexpr double INV_PI = 1.0 / PI;
constexpr double INV_TWO_PI = 1.0 / TWO_PI;
constexpr double DEG_TO_RAD = PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / PI;
constexpr double GRAVITY_EARTH = 9.80665;
constexpr double SPEED_OF_LIGHT = 299792458.0;
constexpr double BOLTZMANN = 1.380649e-23;
constexpr double AVOGADRO = 6.02214076e23;

//------------------------------------------------------------------------------
// Solver types
//------------------------------------------------------------------------------

enum class SolverType : uint8_t {
    PBD = 0,          // Position-Based Dynamics
    MPM = 1,          // Material Point Method
    SPH = 2,          // Smoothed Particle Hydrodynamics
    FEM = 3,          // Finite Element Method
    SF = 4,           // Signed Distance Field / Fluid
    KINEMATIC = 5,    // Kinematic (no physics)
    TOOL = 6,         // Tool/rigid body solver
    HYBRID = 7        // Hybrid solver
};

inline const char* solver_type_name(SolverType type) {
    switch (type) {
        case SolverType::PBD: return "PBD";
        case SolverType::MPM: return "MPM";
        case SolverType::SPH: return "SPH";
        case SolverType::FEM: return "FEM";
        case SolverType::SF:  return "SF";
        case SolverType::KINEMATIC: return "KINEMATIC";
        case SolverType::TOOL: return "TOOL";
        case SolverType::HYBRID: return "HYBRID";
        default: return "UNKNOWN";
    }
}

//------------------------------------------------------------------------------
// Particle / grid constants
//------------------------------------------------------------------------------

constexpr size_t MAX_PARTICLES_PER_CELL = 64;
constexpr size_t MAX_NEIGHBORS = 128;
constexpr size_t DEFAULT_GRID_SIZE = 64;
constexpr size_t MAX_GRID_SIZE = 1024;
constexpr double DEFAULT_CELL_SIZE = 0.1;
constexpr double DEFAULT_KERNEL_RADIUS = 0.2;
constexpr double DEFAULT_PARTICLE_RADIUS = 0.05;
constexpr double DEFAULT_REST_DENSITY = 1000.0;  // water density kg/m^3

//------------------------------------------------------------------------------
// Material types
//------------------------------------------------------------------------------

enum class MaterialType : uint8_t {
    RIGID = 0,
    ELASTIC = 1,
    PLASTIC = 2,
    FLUID = 3,
    GRANULAR = 4,
    CLOTH = 5,
    ROPE = 6,
    SOFT_BODY = 7,
    SAND = 8,
    SNOW = 9,
    FOAM = 10
};

// Default material parameters
struct MaterialParams {
    double youngs_modulus = 1e6;    // Pa
    double poisson_ratio = 0.3;
    double density = 1000.0;        // kg/m^3
    double friction_angle = 30.0;   // degrees
    double cohesion = 0.0;
    double tensile_strength = 0.0;
    double compressive_strength = 1e9;
    double hardening = 0.0;
};

//------------------------------------------------------------------------------
// Constraint types
//------------------------------------------------------------------------------

enum class ConstraintType : uint8_t {
    DISTANCE = 0,
    BENDING = 1,
    VOLUME = 2,
    COLLISION = 3,
    ATTACHMENT = 4,
    ANGLE = 5,
    TETRAHEDRAL = 6,
    PIN = 7,
    SLIDER = 8,
    HINGE = 9,
    BALL_JOINT = 10,
    FIXED = 11,
    CONTACT = 12,
    FRICTION = 13
};

//------------------------------------------------------------------------------
// Visualization constants
//------------------------------------------------------------------------------

namespace vis {
    constexpr size_t MAX_MESH_VERTICES = 1000000;
    constexpr size_t MAX_MESH_INDICES = 3000000;
    constexpr size_t MAX_LINE_SEGMENTS = 100000;
    constexpr size_t MAX_POINTS = 1000000;
    
    // Colors (RGBA as uint32_t)
    constexpr uint32_t COLOR_WHITE    = 0xFFFFFFFF;
    constexpr uint32_t COLOR_BLACK    = 0x000000FF;
    constexpr uint32_t COLOR_RED      = 0xFF0000FF;
    constexpr uint32_t COLOR_GREEN    = 0x00FF00FF;
    constexpr uint32_t COLOR_BLUE     = 0x0000FFFF;
    constexpr uint32_t COLOR_YELLOW   = 0xFFFF00FF;
    constexpr uint32_t COLOR_CYAN     = 0x00FFFFFF;
    constexpr uint32_t COLOR_MAGENTA  = 0xFF00FFFF;
    constexpr uint32_t COLOR_GRAY     = 0x808080FF;
    constexpr uint32_t COLOR_ORANGE   = 0xFFA500FF;
    
    // Camera defaults
    constexpr double CAMERA_FOV = 45.0;
    constexpr double CAMERA_NEAR = 0.01;
    constexpr double CAMERA_FAR = 1000.0;
    constexpr double CAMERA_SPEED = 1.0;
    constexpr double CAMERA_SENSITIVITY = 0.1;
}

//------------------------------------------------------------------------------
// Simulation defaults
//------------------------------------------------------------------------------

namespace sim {
    constexpr double DEFAULT_TIMESTEP = 0.01;
    constexpr size_t DEFAULT_SUBSTEPS = 10;
    constexpr size_t MAX_SUBSTEPS = 100;
    constexpr size_t MIN_SUBSTEPS = 1;
    constexpr double DEFAULT_GRAVITY[3] = {0.0, 0.0, -9.80665};
    constexpr double DEFAULT_DAMPING = 0.995;
    constexpr double DEFAULT_VELOCITY_DAMPING = 0.99;
    constexpr size_t DEFAULT_SOLVER_ITERATIONS = 5;
    constexpr size_t MAX_SOLVER_ITERATIONS = 100;
    constexpr double DEFAULT_CONTACT_OFFSET = 0.005;
    constexpr double DEFAULT_RESTITUTION = 0.5;
    constexpr double DEFAULT_STATIC_FRICTION = 0.5;
    constexpr double DEFAULT_DYNAMIC_FRICTION = 0.3;
}

//------------------------------------------------------------------------------
// File format magic numbers / extensions
//------------------------------------------------------------------------------

namespace file {
    constexpr uint32_t MESH_MAGIC = 0x47454E53; // "GENS"
    constexpr uint32_t SCENE_MAGIC = 0x5343454E; // "SCEN"
    constexpr uint32_t ANIM_MAGIC = 0x414E494D;  // "ANIM"
    
    inline const std::string MESH_EXTENSION = ".gmesh";
    inline const std::string SCENE_EXTENSION = ".gscene";
    inline const std::string ANIM_EXTENSION = ".ganim";
    inline const std::string URDF_EXTENSION = ".urdf";
    inline const std::string MJCF_EXTENSION = ".xml";
    inline const std::string OBJ_EXTENSION = ".obj";
    inline const std::string STL_EXTENSION = ".stl";
}

//------------------------------------------------------------------------------
// Threading / parallel constants
//------------------------------------------------------------------------------

namespace threading {
    constexpr size_t DEFAULT_NUM_THREADS = 0; // 0 means auto-detect
    constexpr size_t MAX_THREADS = 256;
    constexpr size_t MIN_BLOCK_SIZE = 1024;
    constexpr size_t CACHE_LINE_SIZE = 64;
}

//------------------------------------------------------------------------------
// Backend string constants
//------------------------------------------------------------------------------

namespace backend_str {
    inline const std::string CPU = "cpu";
    inline const std::string GPU = "gpu";
    inline const std::string CUDA = "cuda";
    inline const std::string AMDGPU = "amdgpu";
    inline const std::string METAL = "metal";
}

//------------------------------------------------------------------------------
// Logging levels (as strings)
//------------------------------------------------------------------------------

namespace log_level {
    inline const std::string DEBUG = "debug";
    inline const std::string INFO = "info";
    inline const std::string WARNING = "warning";
    inline const std::string ERROR = "error";
}

} // namespace constants
} // namespace genesis