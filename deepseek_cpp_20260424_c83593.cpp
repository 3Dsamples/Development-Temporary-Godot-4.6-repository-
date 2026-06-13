// genesis/engine/force_fields.h

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include <cmath>
#include "genesis/datatypes.h"

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Base ForceField class. Force fields apply external forces to particles,
// rigid bodies, or mesh nodes. They are evaluated per simulation step.
//------------------------------------------------------------------------------

class ForceField {
public:
    virtual ~ForceField() = default;

    // Get the name/type of the force field
    virtual std::string type_name() const = 0;

    // Apply forces to a set of positions/velocities. The derived class should
    // compute forces for each element and add them to the force array.
    // Positions and velocities are provided as arrays of Vector3.
    virtual void apply(const std::vector<datatypes::Vector3>& positions,
                       const std::vector<datatypes::Vector3>& velocities,
                       const std::vector<datatypes::real>& masses,
                       std::vector<datatypes::Vector3>& forces,
                       datatypes::real dt) const = 0;

    // Check if this force field is time-varying
    virtual bool is_time_varying() const { return false; }

    // Update any internal state (for time-varying fields). Called at the beginning of each step.
    virtual void update(datatypes::real time, datatypes::real dt) {}

    // Enable/disable the force field
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

    // Set the magnitude multiplier
    void set_strength(datatypes::real strength) { strength_ = strength; }
    datatypes::real get_strength() const { return strength_; }

protected:
    bool enabled_ = true;
    datatypes::real strength_ = 1.0;
};

//------------------------------------------------------------------------------
// Gravity force field (uniform acceleration)
//------------------------------------------------------------------------------
class GravityForce : public ForceField {
public:
    explicit GravityForce(const datatypes::Vector3& acceleration = datatypes::Vector3(0, 0, -9.80665))
        : acceleration_(acceleration) {}

    std::string type_name() const override { return "Gravity"; }

    void apply(const std::vector<datatypes::Vector3>& positions,
               const std::vector<datatypes::Vector3>& velocities,
               const std::vector<datatypes::real>& masses,
               std::vector<datatypes::Vector3>& forces,
               datatypes::real dt) const override;

    void set_acceleration(const datatypes::Vector3& acc) { acceleration_ = acc; }
    datatypes::Vector3 get_acceleration() const { return acceleration_; }

private:
    datatypes::Vector3 acceleration_;
};

//------------------------------------------------------------------------------
// Drag force (velocity-dependent damping)
//------------------------------------------------------------------------------
class DragForce : public ForceField {
public:
    enum class Mode { LINEAR, QUADRATIC, BOTH };

    DragForce(datatypes::real linear_coef = 0.0,
              datatypes::real quadratic_coef = 0.0,
              Mode mode = Mode::BOTH)
        : linear_coef_(linear_coef), quadratic_coef_(quadratic_coef), mode_(mode) {}

    std::string type_name() const override { return "Drag"; }

    void apply(const std::vector<datatypes::Vector3>& positions,
               const std::vector<datatypes::Vector3>& velocities,
               const std::vector<datatypes::real>& masses,
               std::vector<datatypes::Vector3>& forces,
               datatypes::real dt) const override;

    void set_linear_coefficient(datatypes::real coef) { linear_coef_ = coef; }
    void set_quadratic_coefficient(datatypes::real coef) { quadratic_coef_ = coef; }
    void set_mode(Mode mode) { mode_ = mode; }

private:
    datatypes::real linear_coef_;
    datatypes::real quadratic_coef_;
    Mode mode_;
};

//------------------------------------------------------------------------------
// Point attractor / repulsor
//------------------------------------------------------------------------------
class PointForce : public ForceField {
public:
    enum class Type { ATTRACT, REPULSE };

    PointForce(const datatypes::Vector3& position = datatypes::Vector3(0),
               datatypes::real magnitude = 1000.0,
               datatypes::real max_distance = std::numeric_limits<datatypes::real>::max(),
               Type type = Type::ATTRACT,
               datatypes::real exponent = 2.0)
        : position_(position), magnitude_(magnitude), max_distance_(max_distance),
          type_(type), exponent_(exponent) {}

    std::string type_name() const override { return "PointForce"; }

    void apply(const std::vector<datatypes::Vector3>& positions,
               const std::vector<datatypes::Vector3>& velocities,
               const std::vector<datatypes::real>& masses,
               std::vector<datatypes::Vector3>& forces,
               datatypes::real dt) const override;

    void set_position(const datatypes::Vector3& pos) { position_ = pos; }
    void set_magnitude(datatypes::real mag) { magnitude_ = mag; }
    void set_max_distance(datatypes::real dist) { max_distance_ = dist; }
    void set_type(Type type) { type_ = type; }
    void set_exponent(datatypes::real exp) { exponent_ = exp; }

private:
    datatypes::Vector3 position_;
    datatypes::real magnitude_;
    datatypes::real max_distance_;
    Type type_;
    datatypes::real exponent_;
};

//------------------------------------------------------------------------------
// Noise / turbulence field
//------------------------------------------------------------------------------
class TurbulenceForce : public ForceField {
public:
    TurbulenceForce(datatypes::real strength = 1.0,
                    datatypes::real scale = 1.0,
                    datatypes::real frequency = 1.0)
        : noise_strength_(strength), noise_scale_(scale), frequency_(frequency),
          time_(0) {}

    std::string type_name() const override { return "Turbulence"; }
    bool is_time_varying() const override { return true; }

    void update(datatypes::real time, datatypes::real dt) override;
    void apply(const std::vector<datatypes::Vector3>& positions,
               const std::vector<datatypes::Vector3>& velocities,
               const std::vector<datatypes::real>& masses,
               std::vector<datatypes::Vector3>& forces,
               datatypes::real dt) const override;

    void set_strength(datatypes::real s) { noise_strength_ = s; }
    void set_scale(datatypes::real s) { noise_scale_ = s; }
    void set_frequency(datatypes::real f) { frequency_ = f; }

private:
    datatypes::real noise_strength_;
    datatypes::real noise_scale_;
    datatypes::real frequency_;
    datatypes::real time_;

    // Simple Perlin-like noise (actual implementation uses 3D noise)
    datatypes::Vector3 noise3(const datatypes::Vector3& p) const;
};

//------------------------------------------------------------------------------
// Wind force (directional with optional noise)
//------------------------------------------------------------------------------
class WindForce : public ForceField {
public:
    WindForce(const datatypes::Vector3& direction = datatypes::Vector3(1, 0, 0),
              datatypes::real speed = 1.0,
              datatypes::real gustiness = 0.0)
        : direction_(direction.normalized()), base_speed_(speed), gustiness_(gustiness),
          time_(0) {}

    std::string type_name() const override { return "Wind"; }
    bool is_time_varying() const override { return gustiness_ > 0; }

    void update(datatypes::real time, datatypes::real dt) override;
    void apply(const std::vector<datatypes::Vector3>& positions,
               const std::vector<datatypes::Vector3>& velocities,
               const std::vector<datatypes::real>& masses,
               std::vector<datatypes::Vector3>& forces,
               datatypes::real dt) const override;

    void set_direction(const datatypes::Vector3& dir) { direction_ = dir.normalized(); }
    void set_speed(datatypes::real speed) { base_speed_ = speed; }
    void set_gustiness(datatypes::real gust) { gustiness_ = gust; }

private:
    datatypes::Vector3 direction_;
    datatypes::real base_speed_;
    datatypes::real gustiness_;
    datatypes::real time_;
    datatypes::real current_gust_factor_ = 1.0;
};

//------------------------------------------------------------------------------
// ForceFieldManager: container for multiple force fields
//------------------------------------------------------------------------------
class ForceFieldManager {
public:
    ForceFieldManager() = default;

    // Add a force field (takes ownership)
    void add(std::shared_ptr<ForceField> field);
    void remove(const std::string& name); // name is not part of base, but we can use index or store with name
    void clear();

    // Apply all enabled force fields to a set of positions/forces
    void apply_all(const std::vector<datatypes::Vector3>& positions,
                   const std::vector<datatypes::Vector3>& velocities,
                   const std::vector<datatypes::real>& masses,
                   std::vector<datatypes::Vector3>& forces,
                   datatypes::real dt) const;

    // Update time-varying fields
    void update_all(datatypes::real time, datatypes::real dt);

    // Get list of fields
    const std::vector<std::shared_ptr<ForceField>>& get_fields() const { return fields_; }

    // Find field by type name
    std::shared_ptr<ForceField> find_by_type(const std::string& type) const;

private:
    std::vector<std::shared_ptr<ForceField>> fields_;
};

//------------------------------------------------------------------------------
// Built-in force field presets
//------------------------------------------------------------------------------
namespace force_presets {
    std::shared_ptr<GravityForce> earth_gravity();
    std::shared_ptr<GravityForce> moon_gravity();
    std::shared_ptr<DragForce> air_drag_standard();
    std::shared_ptr<DragForce> water_drag();
    std::shared_ptr<TurbulenceForce> light_turbulence();
    std::shared_ptr<WindForce> gentle_breeze(const datatypes::Vector3& direction = datatypes::Vector3(1, 0, 0));
}

} // namespace engine
} // namespace genesis