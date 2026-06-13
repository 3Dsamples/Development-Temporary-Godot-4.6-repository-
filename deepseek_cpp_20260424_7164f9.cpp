// genesis/engine/solvers/__init__.h

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>
#include "genesis/engine/solvers/base_solver.h"

namespace genesis {
namespace engine {
namespace solvers {

// Forward declare all solver types
class PBDSolver;
class MPMSolver;
class SPHSolver;
class FEMSolver;
class SFSolver;
class KinematicSolver;
class ToolSolver;

//------------------------------------------------------------------------------
// Solver types enumeration
//------------------------------------------------------------------------------
enum class SolverType : uint8_t {
    PBD = 0,
    MPM = 1,
    SPH = 2,
    FEM = 3,
    SF = 4,
    KINEMATIC = 5,
    TOOL = 6,
    CUSTOM = 7
};

// Convert string to solver type
SolverType solver_type_from_string(const std::string& str);
std::string solver_type_to_string(SolverType type);

//------------------------------------------------------------------------------
// Solver creation function type
//------------------------------------------------------------------------------
using SolverCreator = std::function<std::shared_ptr<BaseSolver>(const SolverConfig&)>;

//------------------------------------------------------------------------------
// Solver factory: registry of available solvers
//------------------------------------------------------------------------------
class SolverFactory {
public:
    static SolverFactory& instance();

    // Register a solver type with its creator function
    void register_solver(SolverType type, SolverCreator creator);
    void register_solver(const std::string& name, SolverCreator creator);

    // Create a solver instance by type
    std::shared_ptr<BaseSolver> create(SolverType type, const SolverConfig& config = SolverConfig{}) const;
    std::shared_ptr<BaseSolver> create(const std::string& name, const SolverConfig& config = SolverConfig{}) const;

    // Check if a solver type is registered
    bool is_registered(SolverType type) const;
    bool is_registered(const std::string& name) const;

    // Get list of registered solver names
    std::vector<std::string> registered_names() const;

    // Clear all registrations (mostly for testing)
    void clear();

private:
    SolverFactory() = default;
    std::unordered_map<SolverType, SolverCreator> creators_by_type_;
    std::unordered_map<std::string, SolverCreator> creators_by_name_;
};

//------------------------------------------------------------------------------
// Convenience creation functions
//------------------------------------------------------------------------------
inline std::shared_ptr<PBDSolver> create_pbd_solver(const SolverConfig& config = SolverConfig{}) {
    return std::dynamic_pointer_cast<PBDSolver>(SolverFactory::instance().create(SolverType::PBD, config));
}

inline std::shared_ptr<MPMSolver> create_mpm_solver(const SolverConfig& config = SolverConfig{}) {
    return std::dynamic_pointer_cast<MPMSolver>(SolverFactory::instance().create(SolverType::MPM, config));
}

inline std::shared_ptr<SPHSolver> create_sph_solver(const SolverConfig& config = SolverConfig{}) {
    return std::dynamic_pointer_cast<SPHSolver>(SolverFactory::instance().create(SolverType::SPH, config));
}

inline std::shared_ptr<FEMSolver> create_fem_solver(const SolverConfig& config = SolverConfig{}) {
    return std::dynamic_pointer_cast<FEMSolver>(SolverFactory::instance().create(SolverType::FEM, config));
}

inline std::shared_ptr<SFSolver> create_sf_solver(const SolverConfig& config = SolverConfig{}) {
    return std::dynamic_pointer_cast<SFSolver>(SolverFactory::instance().create(SolverType::SF, config));
}

inline std::shared_ptr<KinematicSolver> create_kinematic_solver(const SolverConfig& config = SolverConfig{}) {
    return std::dynamic_pointer_cast<KinematicSolver>(SolverFactory::instance().create(SolverType::KINEMATIC, config));
}

inline std::shared_ptr<ToolSolver> create_tool_solver(const SolverConfig& config = SolverConfig{}) {
    return std::dynamic_pointer_cast<ToolSolver>(SolverFactory::instance().create(SolverType::TOOL, config));
}

//------------------------------------------------------------------------------
// Auto-registration helpers (to be used in solver source files)
//------------------------------------------------------------------------------
struct SolverRegistrar {
    SolverRegistrar(SolverType type, const std::string& name, SolverCreator creator) {
        SolverFactory::instance().register_solver(type, creator);
        SolverFactory::instance().register_solver(name, creator);
    }
};

// Macro to register a solver at static initialization time
#define REGISTER_SOLVER(SolverClass, TypeEnum, TypeName) \
    static SolverRegistrar _registrar_##SolverClass( \
        TypeEnum, TypeName, \
        [](const SolverConfig& cfg) -> std::shared_ptr<BaseSolver> { \
            return std::make_shared<SolverClass>(cfg); \
        })

} // namespace solvers
} // namespace engine
} // namespace genesis