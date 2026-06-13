// genesis/engine/solvers/base_solver.cpp

#include "genesis/engine/solvers/base_solver.h"
#include <algorithm>
#include <cmath>
#include <mutex>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// BaseSolver implementation
//------------------------------------------------------------------------------
BaseSolver::BaseSolver(const std::string& name)
    : name_(name)
{
}

BaseSolver::~BaseSolver() = default;

void BaseSolver::set_config(const SolverConfig& config) {
    config_ = config;
}

void BaseSolver::set_param(const std::string& key, double value) {
    config_.custom_params[key] = value;
}

double BaseSolver::get_param(const std::string& key, double default_value) const {
    auto it = config_.custom_params.find(key);
    if (it != config_.custom_params.end()) {
        return it->second;
    }
    return default_value;
}

void BaseSolver::reset() {
    reset_stats();
    initialized_ = false;
}

void BaseSolver::reset_stats() {
    stats_ = Stats{};
}

std::vector<std::shared_ptr<BaseEntity>> BaseSolver::filter_dynamic(
    const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    std::vector<std::shared_ptr<BaseEntity>> result;
    result.reserve(entities.size());
    for (const auto& e : entities) {
        if (e && e->is_dynamic()) {
            result.push_back(e);
        }
    }
    return result;
}

std::vector<std::shared_ptr<BaseEntity>> BaseSolver::filter_static(
    const std::vector<std::shared_ptr<BaseEntity>>& entities) {
    std::vector<std::shared_ptr<BaseEntity>> result;
    result.reserve(entities.size());
    for (const auto& e : entities) {
        if (e && !e->is_dynamic()) {
            result.push_back(e);
        }
    }
    return result;
}

} // namespace engine
} // namespace genesis