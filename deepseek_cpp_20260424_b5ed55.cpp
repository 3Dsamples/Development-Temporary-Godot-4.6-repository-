// genesis/engine/entities/drone_entity.h

#pragma once

//------------------------------------------------------------------------------
// Drone entity class - represents a multirotor aerial vehicle (quadcopter, etc.)
// Includes motor simulation, PID control, and basic flight dynamics.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/rigid_entity.h"   // Base class: rigid body dynamics
#include "genesis/datatypes.h"                      // Vector3, Quat, etc.
#include <vector>                                   // std::vector for motors
#include <functional>                               // std::function for control callbacks

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Motor configuration for a single rotor
//------------------------------------------------------------------------------
struct MotorConfig {
    int index = 0;                                 // Motor index (0-based)
    datatypes::Vector3 position;                   // Position relative to drone COM
    datatypes::Vector3 direction;                  // Thrust direction in body frame (usually (0,0,1) or (0,0,-1))
    int spin_direction = 1;                        // 1 for CCW, -1 for CW (affects torque)
    double thrust_coefficient = 1.0e-5;            // k_f: thrust = k_f * ω² (N / (rad/s)²)
    double torque_coefficient = 1.0e-7;            // k_m: torque = k_m * ω² (Nm / (rad/s)²)
    double min_rpm = 0.0;                          // Minimum motor speed (rad/s)
    double max_rpm = 2000.0;                       // Maximum motor speed (rad/s)
    double time_constant = 0.02;                   // Motor response time constant (s)
    double current_rpm = 0.0;                      // Current motor speed (rad/s)
    double target_rpm = 0.0;                       // Target motor speed (rad/s)
};

//------------------------------------------------------------------------------
// PID controller structure
//------------------------------------------------------------------------------
struct PIDController {
    double kp = 0.0;                               // Proportional gain
    double ki = 0.0;                               // Integral gain
    double kd = 0.0;                               // Derivative gain
    double integral = 0.0;                         // Accumulated integral error
    double prev_error = 0.0;                       // Previous error for derivative
    double integral_limit = 10.0;                  // Clamp for integral windup prevention
    double output_limit = 100.0;                   // Clamp for controller output

    // Compute control output given error and time step
    double update(double error, double dt);
    // Reset integral and previous error
    void reset();
};

//------------------------------------------------------------------------------
// Drone configuration parameters
//------------------------------------------------------------------------------
struct DroneConfig {
    // Physical properties
    double mass = 0.5;                             // Total mass (kg)
    datatypes::Vector3 inertia = {0.01, 0.01, 0.02}; // Moments of inertia (kg·m²)
    double arm_length = 0.2;                       // Distance from COM to motor (m)
    
    // Motor arrangement (default X configuration for quadcopter)
    std::vector<MotorConfig> motors;
    
    // Control parameters
    PIDController roll_pid;                        // Roll angle controller
    PIDController pitch_pid;                       // Pitch angle controller
    PIDController yaw_rate_pid;                    // Yaw rate controller
    PIDController altitude_pid;                    // Altitude (z) controller
    PIDController velocity_x_pid;                  // Velocity X controller (optional)
    PIDController velocity_y_pid;                  // Velocity Y controller (optional)
    
    // Control limits
    double max_roll_angle = 0.5;                   // Max roll angle (rad) ~30 deg
    double max_pitch_angle = 0.5;                  // Max pitch angle (rad)
    double max_yaw_rate = 1.0;                     // Max yaw rate (rad/s)
    double max_climb_rate = 3.0;                   // Max vertical speed (m/s)
    double max_horizontal_speed = 5.0;             // Max horizontal speed (m/s)
    
    // Motor mixing matrix (from control inputs to motor commands)
    // For X quad: roll, pitch, yaw, thrust allocation
    std::vector<std::vector<double>> motor_mixing;
    
    // Default constructor sets up quadcopter X configuration
    DroneConfig();
};

//------------------------------------------------------------------------------
// DroneEntity class - multirotor simulation entity
//------------------------------------------------------------------------------
class DroneEntity : public RigidEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    DroneEntity();                                 // Default constructor
    explicit DroneEntity(const DroneConfig& config); // Constructor with custom config
    explicit DroneEntity(const std::string& name);   // Named constructor
    virtual ~DroneEntity();                        // Destructor

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_config(const DroneConfig& config);    // Apply drone configuration
    const DroneConfig& config() const { return drone_config_; } // Get configuration

    //----------------------------------------------------------------------
    // Motor control
    //----------------------------------------------------------------------
    void set_motor_speed(int index, double rpm);   // Set target RPM for a specific motor
    double get_motor_speed(int index) const;       // Get current RPM of a motor
    size_t motor_count() const { return drone_config_.motors.size(); } // Number of motors

    // Apply raw motor commands (RPM array)
    void set_motor_speeds(const std::vector<double>& rpms);

    // Apply normalized thrust command [0,1] to all motors simultaneously
    void set_normalized_thrust(double thrust);

    //----------------------------------------------------------------------
    // High-level control commands (position/attitude setpoints)
    //----------------------------------------------------------------------
    // Set desired attitude (roll, pitch, yaw) in radians (body frame)
    void set_attitude_setpoint(double roll, double pitch, double yaw);
    // Set desired angular rates (roll, pitch, yaw) in rad/s
    void set_angular_rate_setpoint(double roll_rate, double pitch_rate, double yaw_rate);
    // Set desired altitude (world Z)
    void set_altitude_setpoint(double altitude);
    // Set desired velocity in world frame
    void set_velocity_setpoint(const datatypes::Vector3& velocity);
    // Set desired position in world frame (requires position controller)
    void set_position_setpoint(const datatypes::Vector3& position);

    // Enable/disable automatic control loops
    void enable_attitude_control(bool enable) { attitude_control_enabled_ = enable; }
    void enable_altitude_control(bool enable) { altitude_control_enabled_ = enable; }
    void enable_position_control(bool enable) { position_control_enabled_ = enable; }

    //----------------------------------------------------------------------
    // Control loop update (called each simulation step)
    //----------------------------------------------------------------------
    virtual void update_control(double dt);        // Compute motor commands from setpoints

    //----------------------------------------------------------------------
    // Override physics integration to include motor forces
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;    // Apply motor forces and integrate
    virtual void integrate_velocity(double dt) override; // Add motor forces before base integration

    //----------------------------------------------------------------------
    // State queries
    //----------------------------------------------------------------------
    double get_battery_voltage() const { return battery_voltage_; } // Simulated battery voltage
    void set_battery_voltage(double v) { battery_voltage_ = v; }
    double get_total_thrust() const { return total_thrust_; } // Current total thrust (N)
    datatypes::Vector3 get_body_angular_velocity() const; // Angular velocity in body frame

    //----------------------------------------------------------------------
    // Reset to initial state
    //----------------------------------------------------------------------
    virtual void reset() override;

    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "DroneEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    DroneConfig drone_config_;                     // Drone configuration parameters
    
    // Control setpoints
    double roll_setpoint_ = 0.0;                   // Desired roll angle (rad)
    double pitch_setpoint_ = 0.0;                  // Desired pitch angle (rad)
    double yaw_setpoint_ = 0.0;                    // Desired yaw angle (rad)
    double roll_rate_setpoint_ = 0.0;              // Desired roll rate (rad/s)
    double pitch_rate_setpoint_ = 0.0;             // Desired pitch rate (rad/s)
    double yaw_rate_setpoint_ = 0.0;               // Desired yaw rate (rad/s)
    double altitude_setpoint_ = 0.0;               // Desired altitude (m)
    datatypes::Vector3 velocity_setpoint_;         // Desired world velocity (m/s)
    datatypes::Vector3 position_setpoint_;         // Desired world position (m)
    
    // Control mode flags
    bool attitude_control_enabled_ = true;         // Enable roll/pitch/yaw control
    bool altitude_control_enabled_ = true;         // Enable altitude control
    bool position_control_enabled_ = false;        // Enable position (x,y) control
    bool rate_control_mode_ = false;               // Use angular rate setpoints instead of angles
    
    // Motor state
    std::vector<double> motor_speeds_;             // Current motor RPMs
    
    // Computed forces/torques for current step
    double total_thrust_ = 0.0;                    // Total thrust force (N) in body -Z? Actually body +Z usually down
    datatypes::Vector3 body_torque_;               // Torque in body frame (Nm)
    
    // Battery simulation
    double battery_voltage_ = 11.1;                // 3S LiPo nominal voltage
    
    // Internal helper methods
    void compute_motor_forces();                   // Calculate thrust and torque from motor speeds
    void apply_motor_dynamics(double dt);          // Update motor speeds with first-order lag
    void compute_mixer_matrix();                   // Set up default mixing matrix if not provided
    void run_attitude_control(double dt);          // Compute attitude control outputs
    void run_altitude_control(double dt);          // Compute altitude control output
    void run_position_control(double dt);           // Compute position control outputs
    void run_rate_control(double dt);              // Compute rate control outputs
};

} // namespace engine
} // namespace genesis