// genesis/typing.h

#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <memory>
#include <variant>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <map>
#include <set>
#include <tuple>

namespace genesis {
namespace typing {

//------------------------------------------------------------------------------
// Type aliases for common numeric types (already in datatypes, but re-exported)
//------------------------------------------------------------------------------
using int8   = int8_t;
using int16  = int16_t;
using int32  = int32_t;
using int64  = int64_t;
using uint8  = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using float32 = float;
using float64 = double;

#ifdef GENESIS_USE_DOUBLE_PRECISION
using real = float64;
#else
using real = float32;
#endif

//------------------------------------------------------------------------------
// Index types for particles, elements, nodes, etc.
//------------------------------------------------------------------------------
using particle_idx = int64_t;
using element_idx  = int64_t;
using node_idx     = int64_t;
using cell_idx     = int64_t;
using entity_idx   = int32_t;
using constraint_idx = int64_t;
using material_idx = int32_t;

//------------------------------------------------------------------------------
// Generic array/list aliases (mirror Python list/typing.List)
//------------------------------------------------------------------------------
template<typename T>
using List = std::vector<T>;

template<typename K, typename V>
using Dict = std::unordered_map<K, V>;

template<typename K, typename V>
using OrderedDict = std::map<K, V>;

template<typename T>
using Set = std::set<T>;

template<typename T>
using Optional = std::optional<T>;

//------------------------------------------------------------------------------
// Tuple types (fixed-size heterogeneous containers)
//------------------------------------------------------------------------------
template<typename... Ts>
using Tuple = std::tuple<Ts...>;

// Common specific tuples
using PairII = std::pair<int, int>;
using PairIF = std::pair<int, float>;
using PairIS = std::pair<int, std::string>;
using TripleIII = std::tuple<int, int, int>;
using TripleFFF = std::tuple<float, float, float>;

//------------------------------------------------------------------------------
// Union type (std::variant) for functions that accept multiple types
//------------------------------------------------------------------------------
template<typename... Ts>
using Union = std::variant<Ts...>;

// Common unions for simulation parameters
using ScalarOrVec3 = std::variant<real, std::array<real, 3>>;
using IntOrStr     = std::variant<int, std::string>;
using BoolOrReal   = std::variant<bool, real>;

//------------------------------------------------------------------------------
// Callable types (function objects)
//------------------------------------------------------------------------------
template<typename R, typename... Args>
using Callable = std::function<R(Args...)>;

// Common callback signatures
using VoidCallback       = std::function<void()>;
using StepCallback       = std::function<void(int)>;
using CollisionCallback  = std::function<void(entity_idx, entity_idx, const std::array<real, 3>&)>;
using UpdateCallback     = std::function<void(real)>;

//------------------------------------------------------------------------------
// Forward declarations for entity types (used in type hints)
//------------------------------------------------------------------------------
class BaseEntity;
class RigidEntity;
class SoftEntity;
class FluidEntity;
class ClothEntity;
class RopeEntity;
class DroneEntity;
class MPMParticles;
class SPHParticles;

// Smart pointer aliases
using EntityPtr      = std::shared_ptr<BaseEntity>;
using RigidEntityPtr = std::shared_ptr<RigidEntity>;
using SoftEntityPtr  = std::shared_ptr<SoftEntity>;
using FluidEntityPtr = std::shared_ptr<FluidEntity>;
using ClothEntityPtr = std::shared_ptr<ClothEntity>;
using RopeEntityPtr  = std::shared_ptr<RopeEntity>;
using DroneEntityPtr = std::shared_ptr<DroneEntity>;

//------------------------------------------------------------------------------
// Scene-related type hints
//------------------------------------------------------------------------------
class Scene;
using ScenePtr = std::shared_ptr<Scene>;

//------------------------------------------------------------------------------
// Solver-related type hints
//------------------------------------------------------------------------------
class BaseSolver;
class PBDSolver;
class MPMSolver;
class SPHSolver;
class FEMSolver;
class SFSolver;
class KinematicSolver;
class ToolSolver;

using SolverPtr      = std::shared_ptr<BaseSolver>;
using PBDSolverPtr   = std::shared_ptr<PBDSolver>;
using MPMSolverPtr   = std::shared_ptr<MPMSolver>;
using SPHSolverPtr   = std::shared_ptr<SPHSolver>;
using FEMSolverPtr   = std::shared_ptr<FEMSolver>;
using SFSolverPtr    = std::shared_ptr<SFSolver>;
using KinematicSolverPtr = std::shared_ptr<KinematicSolver>;
using ToolSolverPtr  = std::shared_ptr<ToolSolver>;

//------------------------------------------------------------------------------
// Mesh and geometry types
//------------------------------------------------------------------------------
class Mesh;
class BVH;
class SDF;

using MeshPtr = std::shared_ptr<Mesh>;
using BVHPtr  = std::shared_ptr<BVH>;
using SDFPtr  = std::shared_ptr<SDF>;

//------------------------------------------------------------------------------
// Tensor types (for grad module)
//------------------------------------------------------------------------------
template<typename T>
class Tensor;

using TensorF = Tensor<float>;
using TensorD = Tensor<double>;
using TensorR = Tensor<real>;

using TensorPtr = std::shared_ptr<TensorR>;

//------------------------------------------------------------------------------
// Type traits and concept-like checks (C++17 compatible)
//------------------------------------------------------------------------------
template<typename T>
struct is_numeric : std::disjunction<std::is_arithmetic<T>> {};

template<typename T>
struct is_container : std::false_type {};

template<typename T, typename A>
struct is_container<std::vector<T, A>> : std::true_type {};

template<typename T, std::size_t N>
struct is_container<std::array<T, N>> : std::true_type {};

template<typename T>
struct is_shared_ptr : std::false_type {};

template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

// Helper variable templates
template<typename T>
inline constexpr bool is_numeric_v = is_numeric<T>::value;

template<typename T>
inline constexpr bool is_container_v = is_container<T>::value;

template<typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

//------------------------------------------------------------------------------
// Type conversion utilities
//------------------------------------------------------------------------------
template<typename To, typename From>
std::optional<To> safe_cast(const From& value) {
    if constexpr (std::is_convertible_v<From, To>) {
        return static_cast<To>(value);
    } else {
        return std::nullopt;
    }
}

template<typename T>
std::string type_name() {
    return demangle_type_name(typeid(T));
}

// Helper to demangle (reused from repr_base but included for completeness)
std::string demangle_type_name(const std::type_info& ti);

//------------------------------------------------------------------------------
// Literal operator for real numbers (e.g., 1.0_r)
//------------------------------------------------------------------------------
inline namespace literals {
    constexpr real operator"" _r(long double val) {
        return static_cast<real>(val);
    }
    constexpr real operator"" _r(unsigned long long val) {
        return static_cast<real>(val);
    }
}

//------------------------------------------------------------------------------
// Type-checking helpers (compile-time and runtime)
//------------------------------------------------------------------------------
template<typename T, typename U>
constexpr bool is_same_v = std::is_same_v<T, U>;

template<typename Base, typename Derived>
constexpr bool is_base_of_v = std::is_base_of_v<Base, Derived>;

// Runtime type check for polymorphic types
template<typename Base, typename Derived>
bool instanceof(const Derived* ptr) {
    return dynamic_cast<const Base*>(ptr) != nullptr;
}

template<typename Base, typename Derived>
bool instanceof(const std::shared_ptr<Derived>& ptr) {
    return std::dynamic_pointer_cast<Base>(ptr) != nullptr;
}

//------------------------------------------------------------------------------
// Range type for iteration (Python-like range)
//------------------------------------------------------------------------------
class Range {
public:
    class Iterator {
    public:
        Iterator(int64_t value, int64_t step) : value_(value), step_(step) {}
        int64_t operator*() const { return value_; }
        Iterator& operator++() { value_ += step_; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; value_ += step_; return tmp; }
        bool operator!=(const Iterator& other) const {
            return step_ > 0 ? value_ < other.value_ : value_ > other.value_;
        }
    private:
        int64_t value_;
        int64_t step_;
    };

    Range(int64_t stop) : start_(0), stop_(stop), step_(1) {}
    Range(int64_t start, int64_t stop, int64_t step = 1)
        : start_(start), stop_(stop), step_(step) {}

    Iterator begin() const { return Iterator(start_, step_); }
    Iterator end() const { return Iterator(stop_, step_); }

private:
    int64_t start_, stop_, step_;
};

//------------------------------------------------------------------------------
// Enumerate helper for index+value iteration
//------------------------------------------------------------------------------
template<typename Container>
class Enumerate {
public:
    using value_type = typename Container::value_type;
    using iterator = typename Container::const_iterator;
    using size_type = typename Container::size_type;

    class EnumerateIterator {
    public:
        EnumerateIterator(size_type idx, iterator it) : idx_(idx), it_(it) {}
        std::pair<size_type, const value_type&> operator*() const { return {idx_, *it_}; }
        EnumerateIterator& operator++() { ++idx_; ++it_; return *this; }
        bool operator!=(const EnumerateIterator& other) const { return it_ != other.it_; }
    private:
        size_type idx_;
        iterator it_;
    };

    Enumerate(const Container& cont) : cont_(cont) {}
    EnumerateIterator begin() const { return EnumerateIterator(0, cont_.begin()); }
    EnumerateIterator end() const { return EnumerateIterator(cont_.size(), cont_.end()); }

private:
    const Container& cont_;
};

template<typename Container>
Enumerate<Container> enumerate(const Container& cont) {
    return Enumerate<Container>(cont);
}

//------------------------------------------------------------------------------
// Zip iterator for parallel iteration
//------------------------------------------------------------------------------
template<typename... Containers>
class Zip {
public:
    using iterator_tuple = std::tuple<typename Containers::const_iterator...>;
    using value_tuple = std::tuple<typename Containers::const_reference...>;

    class ZipIterator {
    public:
        ZipIterator(iterator_tuple its, bool at_end = false) : its_(its), at_end_(at_end) {}
        value_tuple operator*() const {
            return std::apply([](auto... it) { return std::tie(*it...); }, its_);
        }
        ZipIterator& operator++() {
            std::apply([](auto&... it) { (++it, ...); }, its_);
            return *this;
        }
        bool operator!=(const ZipIterator& other) const {
            return at_end_ != other.at_end_;
        }
    private:
        iterator_tuple its_;
        bool at_end_;
    };

    Zip(const Containers&... conts) : conts_(conts...) {}
    ZipIterator begin() const {
        auto begins = std::apply([](const auto&... c) { return std::make_tuple(c.begin()...); }, conts_);
        bool at_end = std::apply([](const auto&... c) { return ((c.begin() == c.end()) || ...); }, conts_);
        return ZipIterator(begins, at_end);
    }
    ZipIterator end() const {
        auto ends = std::apply([](const auto&... c) { return std::make_tuple(c.end()...); }, conts_);
        return ZipIterator(ends, true);
    }

private:
    std::tuple<const Containers&...> conts_;
};

template<typename... Containers>
Zip<Containers...> zip(const Containers&... conts) {
    return Zip<Containers...>(conts...);
}

} // namespace typing
} // namespace genesis