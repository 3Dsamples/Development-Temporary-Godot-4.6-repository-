// genesis/engine/entities/drone_entity.cpp

#include "genesis/engine/entities/drone_entity.h" // Include corresponding header
#include <cmath>                                   // std::sin, std::cos, std::abs, std::sqrt
#include <algorithm>                               // std::clamp, std::max, std::min
#include <sstream>                                 // std::ostringstream for repr
#include <iomanip>                                 // std::setprecision

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// PIDController implementation
//------------------------------------------------------------------------------
double PIDController::update(double error, double dt) {
    // Compute proportional term
    double p_term = kp * error;                    // P = Kp * e
    
    // Compute integral term with anti-windup
    integral += error * dt;                        // Accumulate integral: ∫e dt
    integral = std::clamp(integral, -integral_limit, integral_limit); // Clamp to prevent windup
    double i_term = ki * integral;                 // I = Ki * ∫e dt
    
    // Compute derivative term (avoid derivative kick on setpoint changes)
    double derivative = (error - prev_error) / dt; // de/dt using backward difference
    double d_term = kd * derivative;               // D = Kd * de/dt
    prev_error = error;                            // Store error for next step
    
    // Compute total output and clamp
    double output = p_term + i_term + d_term;      // PID output sum
    output = std::clamp(output, -output_limit, output_limit); // Clamp to output limits
    return output;                                 // Return control signal
}

void PIDController::reset() {
    // Reset internal state of PID controller
    integral = 0.0;                                // Clear integral accumulator
    prev_error = 0.0;                              // Clear previous error
}

//------------------------------------------------------------------------------
// DroneConfig default constructor (quadcopter X configuration)
//------------------------------------------------------------------------------
DroneConfig::DroneConfig() {
    // Set up default quadcopter X configuration with 4 motors
    
    // Motor 0: front-right, CCW
    MotorConfig m0;                                // Create motor config
    m0.index = 0;                                  // Motor index 0
    m0.position = datatypes::Vector3( arm_length,  arm_length, 0.0); // Position relative to COM
    m0.direction = datatypes::Vector3(0.0, 0.0, 1.0); // Thrust upward in body frame (Z up)
    m0.spin_direction = 1;                         // CCW rotation
    m0.thrust_coefficient = 1.0e-5;                // Thrust coefficient k_f
    m0.torque_coefficient = 1.0e-7;                // Torque coefficient k_m
    m0.min_rpm = 0.0;                              // Minimum RPM
    m0.max_rpm = 2000.0;                           // Maximum RPM
    motors.push_back(m0);                          // Add to motors list
    
    // Motor 1: rear-left, CCW
    MotorConfig m1;                                // Create motor config
    m1.index = 1;                                  // Motor index 1
    m1.position = datatypes::Vector3(-arm_length, -arm_length, 0.0);
    m1.direction = datatypes::Vector3(0.0, 0.0, 1.0);
    m1.spin_direction = 1;                         // CCW
    m1.thrust_coefficient = 1.0e-5;
    m1.torque_coefficient = 1.0e-7;
    m1.min_rpm = 0.0;
    m1.max_rpm = 2000.0;
    motors.push_back(m1);
    
    // Motor 2: front-left, CW
    MotorConfig m2;                                // Create motor config
    m2.index = 2;                                  // Motor index 2
    m2.position = datatypes::Vector3(-arm_length,  arm_length, 0.0);
    m2.direction = datatypes::Vector3(0.0, 0.0, 1.0);
    m2.spin_direction = -1;                        // CW rotation
    m2.thrust_coefficient = 1.0e-5;
    m2.torque_coefficient = 1.0e-7;
    m2.min_rpm = 0.0;
    m2.max_rpm = 2000.0;
    motors.push_back(m2);
    
    // Motor 3: rear-right, CW
    MotorConfig m3;                                // Create motor config
    m3.index = 3;                                  // Motor index 3
    m3.position = datatypes::Vector3( arm_length, -arm_length, 0.0);
    m3.direction = datatypes::Vector3(0.0, 0.0, 1.0);
    m3.spin_direction = -1;                        // CW
    m3.thrust_coefficient = 1.0e-5;
    m3.torque_coefficient = 1.0e-7;
    m3.min_rpm = 0.0;
    m3.max_rpm = 2000.0;
    motors.push_back(m3);
    
    // Default PID gains for a typical quadcopter
    roll_pid.kp = 5.0;                             // Roll proportional gain
    roll_pid.ki = 0.1;                             // Roll integral gain
    roll_pid.kd = 0.5;                             // Roll derivative gain
    roll_pid.integral_limit = 2.0;                 // Roll integral limit
    roll_pid.output_limit = 2.0;                   // Roll output limit (rad/s rate command)
    
    pitch_pid.kp = 5.0;                            // Pitch proportional gain
    pitch_pid.ki = 0.1;                            // Pitch integral gain
    pitch_pid.kd = 0.5;                            // Pitch derivative gain
    pitch_pid.integral_limit = 2.0;                // Pitch integral limit
    pitch_pid.output_limit = 2.0;                  // Pitch output limit
    
    yaw_rate_pid.kp = 2.0;                         // Yaw rate proportional gain
    yaw_rate_pid.ki = 0.05;                        // Yaw rate integral gain
    yaw_rate_pid.kd = 0.1;                         // Yaw rate derivative gain
    yaw_rate_pid.integral_limit = 1.0;             // Yaw rate integral limit
    yaw_rate_pid.output_limit = 1.0;               // Yaw rate output limit
    
    altitude_pid.kp = 10.0;                        // Altitude proportional gain
    altitude_pid.ki = 1.0;                         // Altitude integral gain
    altitude_pid.kd = 2.0;                         // Altitude derivative gain
    altitude_pid.integral_limit = 5.0;             // Altitude integral limit
    altitude_pid.output_limit = 10.0;              // Altitude output limit (thrust command)
    
    velocity_x_pid.kp = 2.0;                       // Velocity X proportional gain
    velocity_x_pid.ki = 0.2;                       // Velocity X integral gain
    velocity_x_pid.kd = 0.5;                       // Velocity X derivative gain
    velocity_x_pid.integral_limit = 1.0;           // Velocity X integral limit
    velocity_x_pid.output_limit = max_pitch_angle; // Velocity X output limit (pitch angle)
    
    velocity_y_pid.kp = 2.0;                       // Velocity Y proportional gain
    velocity_y_pid.ki = 0.2;                       // Velocity Y integral gain
    velocity_y_pid.kd = 0.5;                       // Velocity Y derivative gain
    velocity_y_pid.integral_limit = 1.0;           // Velocity Y integral limit
    velocity_y_pid.output_limit = max_roll_angle;  // Velocity Y output limit (roll angle)
    
    // Set up motor mixing matrix for X quadcopter
    // Matrix format: [roll, pitch, yaw, thrust] contributions for each motor
    motor_mixing = {
        { 1.0,  1.0,  1.0, 1.0}, // Motor 0: +roll, +pitch, +yaw, +thrust
        {-1.0, -1.0,  1.0, 1.0}, // Motor 1: -roll, -pitch, +yaw, +thrust
        {-1.0,  1.0, -1.0, 1.0}, // Motor 2: -roll, +pitch, -yaw, +thrust
        { 1.0, -1.0, -1.0, 1.0}  // Motor 3: +roll, -pitch, -yaw, +thrust
    };
}

//------------------------------------------------------------------------------
// DroneEntity construction
//------------------------------------------------------------------------------
DroneEntity::DroneEntity()
    : RigidEntity("Drone")                         // Call base class constructor with name
    , drone_config_()                              // Default drone configuration
{
    // Initialize from default config
    set_config(drone_config_);                     // Apply configuration to set up mass, inertia, etc.
    motor_speeds_.resize(drone_config_.motors.size(), 0.0); // Allocate motor speed array
}

DroneEntity::DroneEntity(const DroneConfig& config)
    : RigidEntity("Drone")                         // Base constructor
    , drone_config_(config)                        // Copy configuration
{
    set_config(drone_config_);                     // Apply configuration
    motor_speeds_.resize(drone_config_.motors.size(), 0.0); // Allocate motor speeds
}

DroneEntity::DroneEntity(const std::string& name)
    : RigidEntity(name)                            // Base constructor with custom name
    , drone_config_()                              // Default config
{
    set_config(drone_config_);                     // Apply configuration
    motor_speeds_.resize(drone_config_.motors.size(), 0.0);
}

DroneEntity::~DroneEntity() {
    // Virtual destructor (cleanup handled by base class)
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void DroneEntity::set_config(const DroneConfig& config) {
    // Store configuration
    drone_config_ = config;                        // Copy config
    
    // Apply physical properties to rigid body base
    set_mass(drone_config_.mass);                  // Set mass from config
    set_inertia(drone_config_.inertia);            // Set inertia tensor
    
    // Ensure motor speeds array is sized correctly
    motor_speeds_.resize(drone_config_.motors.size(), 0.0); // Resize to match motor count
    
    // Initialize motor speeds from config current values
    for (size_t i = 0; i < drone_config_.motors.size(); ++i) {
        motor_speeds_[i] = drone_config_.motors[i].current_rpm; // Copy initial RPM
    }
    
    // Build mixer matrix if not provided
    if (drone_config_.motor_mixing.empty()) {
        compute_mixer_matrix();                    // Generate default mixing matrix
    }
}

//------------------------------------------------------------------------------
// Motor control
//------------------------------------------------------------------------------
void DroneEntity::set_motor_speed(int index, double rpm) {
    // Set target RPM for a specific motor with bounds checking
    if (index < 0 || index >= static_cast<int>(drone_config_.motors.size())) {
        return;                                    // Invalid index, ignore
    }
    // Clamp RPM to motor limits
    rpm = std::clamp(rpm, drone_config_.motors[index].min_rpm, drone_config_.motors[index].max_rpm);
    drone_config_.motors[index].target_rpm = rpm;  // Store target RPM
}

double DroneEntity::get_motor_speed(int index) const {
    // Return current RPM of specified motor
    if (index < 0 || index >= static_cast<int>(drone_config_.motors.size())) {
        return 0.0;                                // Invalid index returns 0
    }
    return drone_config_.motors[index].current_rpm; // Return current RPM
}

void DroneEntity::set_motor_speeds(const std::vector<double>& rpms) {
    // Set target RPMs for all motors
    size_t n = std::min(rpms.size(), drone_config_.motors.size()); // Number of motors to set
    for (size_t i = 0; i < n; ++i) {
        set_motor_speed(static_cast<int>(i), rpms[i]); // Set each motor's target
    }
}

void DroneEntity::set_normalized_thrust(double thrust) {
    // Apply same normalized thrust [0,1] to all motors
    thrust = std::clamp(thrust, 0.0, 1.0);         // Clamp to valid range
    for (size_t i = 0; i < drone_config_.motors.size(); ++i) {
        double max_rpm = drone_config_.motors[i].max_rpm; // Get motor's max RPM
        double target = thrust * max_rpm;          // Compute target RPM
        set_motor_speed(static_cast<int>(i), target); // Set motor target
    }
}

//------------------------------------------------------------------------------
// High-level control setpoints
//------------------------------------------------------------------------------
void DroneEntity::set_attitude_setpoint(double roll, double pitch, double yaw) {
    // Set desired Euler angles (radians)
    roll_setpoint_ = std::clamp(roll, -drone_config_.max_roll_angle, drone_config_.max_roll_angle);
    pitch_setpoint_ = std::clamp(pitch, -drone_config_.max_pitch_angle, drone_config_.max_pitch_angle);
    yaw_setpoint_ = yaw;                           // Yaw angle (not clamped, continuous)
    rate_control_mode_ = false;                    // Use angle control mode
}

void DroneEntity::set_angular_rate_setpoint(double roll_rate, double pitch_rate, double yaw_rate) {
    // Set desired angular rates (rad/s)
    roll_rate_setpoint_ = roll_rate;               // Store roll rate setpoint
    pitch_rate_setpoint_ = pitch_rate;             // Store pitch rate setpoint
    yaw_rate_setpoint_ = std::clamp(yaw_rate, -drone_config_.max_yaw_rate, drone_config_.max_yaw_rate);
    rate_control_mode_ = true;                     // Enable rate control mode
}

void DroneEntity::set_altitude_setpoint(double altitude) {
    // Set desired altitude (world Z)
    altitude_setpoint_ = altitude;                 // Store altitude setpoint
}

void DroneEntity::set_velocity_setpoint(const datatypes::Vector3& velocity) {
    // Set desired world velocity
    velocity_setpoint_ = velocity;                 // Store velocity setpoint
    // Clamp horizontal speed
    double horiz_speed = std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1]);
    if (horiz_speed > drone_config_.max_horizontal_speed) {
        double scale = drone_config_.max_horizontal_speed / horiz_speed;
        velocity_setpoint_[0] *= scale;            // Scale X component
        velocity_setpoint_[1] *= scale;            // Scale Y component
    }
    velocity_setpoint_[2] = std::clamp(velocity[2], -drone_config_.max_climb_rate, drone_config_.max_climb_rate);
}

void DroneEntity::set_position_setpoint(const datatypes::Vector3& position) {
    // Set desired world position
    position_setpoint_ = position;                 // Store position setpoint
}

//------------------------------------------------------------------------------
// Control loop update
//------------------------------------------------------------------------------
void DroneEntity::update_control(double dt) {
    // Update motor dynamics (first-order lag)
    apply_motor_dynamics(dt);                      // Apply motor response lag
    
    // Run control loops based on enabled flags
    if (position_control_enabled_) {
        run_position_control(dt);                  // Position control generates velocity setpoints
    }
    if (altitude_control_enabled_) {
        run_altitude_control(dt);                  // Altitude control computes thrust command
    }
    if (attitude_control_enabled_) {
        if (rate_control_mode_) {
            run_rate_control(dt);                  // Rate control mode
        } else {
            run_attitude_control(dt);              // Angle control mode
        }
    }
    
    // Compute motor forces and torques from current RPMs
    compute_motor_forces();                        // Updates total_thrust_ and body_torque_
}

void DroneEntity::run_attitude_control(double dt) {
    // Get current attitude from entity transform
    datatypes::Vector3 euler = transform().rotation.toEulerAngles(); // Roll, pitch, yaw (rad)
    double current_roll = euler[0];                // Current roll angle
    double current_pitch = euler[1];               // Current pitch angle
    double current_yaw = euler[2];                 // Current yaw angle
    
    // Compute yaw error (with wrap-around handling)
    double yaw_error = yaw_setpoint_ - current_yaw; // Raw difference
    while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI; // Wrap to [-pi, pi]
    while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;
    
    // Run PID controllers
    double roll_output = drone_config_.roll_pid.update(roll_setpoint_ - current_roll, dt);
    double pitch_output = drone_config_.pitch_pid.update(pitch_setpoint_ - current_pitch, dt);
    double yaw_output = drone_config_.yaw_rate_pid.update(yaw_error, dt);
    
    // Map PID outputs to motor mixing (roll, pitch, yaw rate commands)
    // These are added to the base thrust command (which comes from altitude control)
    // For now, store them as control commands that will be used in compute_motor_forces
    // In this implementation, we'll directly set target RPMs via mixing matrix
    
    // Get base thrust from altitude control (or default hover thrust)
    double base_thrust = 0.0;
    if (altitude_control_enabled_) {
        // Altitude control will provide thrust
        base_thrust = drone_config_.altitude_pid.prev_error; // Placeholder
    } else {
        // Use current average RPM as base
        double avg_rpm = 0.0;
        for (double rpm : motor_speeds_) avg_rpm += rpm;
        avg_rpm /= motor_speeds_.size();
        base_thrust = avg_rpm;
    }
    
    // Apply mixing to compute motor targets
    const auto& mix = drone_config_.motor_mixing;  // Reference mixing matrix
    for (size_t i = 0; i < drone_config_.motors.size(); ++i) {
        double cmd = base_thrust 
                     + mix[i][0] * roll_output 
                     + mix[i][1] * pitch_output 
                     + mix[i][2] * yaw_output;
        cmd = std::max(0.0, cmd);                  // RPM cannot be negative
        set_motor_speed(static_cast<int>(i), cmd); // Set target RPM
    }
}

void DroneEntity::run_rate_control(double dt) {
    // Get current body angular rates
    datatypes::Vector3 body_rates = get_body_angular_velocity(); // [p, q, r] in body frame
    double current_p = body_rates[0];              // Roll rate
    double current_q = body_rates[1];              // Pitch rate
    double current_r = body_rates[2];              // Yaw rate
    
    // Compute errors
    double roll_rate_error = roll_rate_setpoint_ - current_p;
    double pitch_rate_error = pitch_rate_setpoint_ - current_q;
    double yaw_rate_error = yaw_rate_setpoint_ - current_r;
    
    // Run PID controllers for rates (simpler than angle control)
    double roll_output = drone_config_.roll_pid.update(roll_rate_error, dt);
    double pitch_output = drone_config_.pitch_pid.update(pitch_rate_error, dt);
    double yaw_output = drone_config_.yaw_rate_pid.update(yaw_rate_error, dt);
    
    // Base thrust (from altitude or current)
    double base_thrust = 0.0;
    for (double rpm : motor_speeds_) base_thrust += rpm;
    base_thrust /= motor_speeds_.size();
    
    const auto& mix = drone_config_.motor_mixing;
    for (size_t i = 0; i < drone_config_.motors.size(); ++i) {
        double cmd = base_thrust 
                     + mix[i][0] * roll_output 
                     + mix[i][1] * pitch_output 
                     + mix[i][2] * yaw_output;
        cmd = std::max(0.0, cmd);
        set_motor_speed(static_cast<int>(i), cmd);
    }
}

void DroneEntity::run_altitude_control(double dt) {
    // Get current altitude (world Z position)
    double current_altitude = position()[2];       // World Z coordinate
    double altitude_error = altitude_setpoint_ - current_altitude; // Error positive when below setpoint
    
    // Run PID to get desired vertical acceleration/thrust command
    double thrust_cmd = drone_config_.altitude_pid.update(altitude_error, dt);
    
    // Convert thrust command to base RPM (approximate)
    // Thrust per motor at hover: T = k_f * ω²
    double hover_rpm = std::sqrt(mass() * 9.81 / (drone_config_.motors.size() * drone_config_.motors[0].thrust_coefficient));
    double base_rpm = hover_rpm + thrust_cmd * 100.0; // Simple scaling
    
    // Set all motors to this base RPM (attitude control will add differential)
    for (size_t i = 0; i < drone_config_.motors.size(); ++i) {
        set_motor_speed(static_cast<int>(i), base_rpm);
    }
}

void DroneEntity::run_position_control(double dt) {
    // Get current position and velocity
    datatypes::Vector3 pos = position();           // Current world position
    datatypes::Vector3 vel = velocity();           // Current world velocity
    
    // Compute position error
    datatypes::Vector3 pos_error = position_setpoint_ - pos;
    
    // Run PID on X axis (forward direction)
    double pitch_cmd = drone_config_.velocity_x_pid.update(pos_error[0], dt);
    // Run PID on Y axis (lateral direction)
    double roll_cmd = -drone_config_.velocity_y_pid.update(pos_error[1], dt); // Negative because roll right produces -Y force?
    
    // Clamp to angle limits
    pitch_cmd = std::clamp(pitch_cmd, -drone_config_.max_pitch_angle, drone_config_.max_pitch_angle);
    roll_cmd = std::clamp(roll_cmd, -drone_config_.max_roll_angle, drone_config_.max_roll_angle);
    
    // Set attitude setpoints for inner loop
    roll_setpoint_ = roll_cmd;
    pitch_setpoint_ = pitch_cmd;
    // Yaw remains as is (could add yaw position control)
}

//------------------------------------------------------------------------------
// Internal motor dynamics and force computation
//------------------------------------------------------------------------------
void DroneEntity::apply_motor_dynamics(double dt) {
    // Apply first-order lag to motor speeds: dω/dt = (ω_target - ω_current) / τ
    for (auto& motor : drone_config_.motors) {
        double error = motor.target_rpm - motor.current_rpm; // Speed error
        double alpha = dt / motor.time_constant;    // Exponential smoothing factor
        motor.current_rpm += error * alpha;         // Update current RPM
        // Clamp to motor limits
        motor.current_rpm = std::clamp(motor.current_rpm, motor.min_rpm, motor.max_rpm);
        // Store in motor_speeds_ array for external access
        if (motor.index < static_cast<int>(motor_speeds_.size())) {
            motor_speeds_[motor.index] = motor.current_rpm;
        }
    }
}

void DroneEntity::compute_motor_forces() {
    // Reset accumulated thrust and torque
    total_thrust_ = 0.0;                           // Initialize total thrust
    body_torque_ = datatypes::Vector3(0.0);        // Initialize body torque
    
    for (const auto& motor : drone_config_.motors) {
        double omega = motor.current_rpm;           // Current motor speed (rad/s)
        if (omega < 1e-6) continue;                // Skip if not spinning
        
        // Compute thrust force: F = k_f * ω² (direction along motor direction)
        double thrust = motor.thrust_coefficient * omega * omega;
        datatypes::Vector3 force_body = motor.direction * thrust; // Force in body frame
        total_thrust_ += thrust;                   // Accumulate total thrust magnitude
        
        // Compute torque from motor: two components
        // 1. Torque due to thrust acting at motor position (r × F)
        datatypes::Vector3 r = motor.position;      // Position relative to COM
        body_torque_ += r.cross(force_body);       // Torque from thrust offset
        
        // 2. Reaction torque from propeller drag: τ = spin_direction * k_m * ω²
        double drag_torque = motor.spin_direction * motor.torque_coefficient * omega * omega;
        body_torque_ += motor.direction * drag_torque; // Torque along motor axis
    }
}

void DroneEntity::compute_mixer_matrix() {
    // Generate default mixing matrix for standard configurations
    // This assumes motors are arranged symmetrically around COM
    size_t n = drone_config_.motors.size();        // Number of motors
    drone_config_.motor_mixing.clear();
    drone_config_.motor_mixing.resize(n, std::vector<double>(4, 0.0)); // [roll, pitch, yaw, thrust]
    
    if (n == 4) {
        // Quadcopter X configuration
        // Motor order: 0=FR,1=RL,2=FL,3=RR
        drone_config_.motor_mixing = {
            { 1.0,  1.0,  1.0, 1.0},
            {-1.0, -1.0,  1.0, 1.0},
            {-1.0,  1.0, -1.0, 1.0},
            { 1.0, -1.0, -1.0, 1.0}
        };
    } else {
        // Generic: compute based on motor positions
        for (size_t i = 0; i < n; ++i) {
            const auto& m = drone_config_.motors[i];
            drone_config_.motor_mixing[i][0] = m.position[1] / drone_config_.arm_length; // Roll: Y position
            drone_config_.motor_mixing[i][1] = m.position[0] / drone_config_.arm_length; // Pitch: X position
            drone_config_.motor_mixing[i][2] = m.spin_direction * 1.0;                   // Yaw: spin direction
            drone_config_.motor_mixing[i][3] = 1.0;                                      // Thrust: always 1
        }
    }
}

//------------------------------------------------------------------------------
// Integration override
//------------------------------------------------------------------------------
void DroneEntity::integrate(double dt) {
    // First run control update to compute motor forces
    update_control(dt);                            // Compute motor forces and torques
    
    // Apply motor forces to rigid body accumulators
    // Convert body frame force to world frame
    datatypes::Vector3 world_force = rotation().rotate(datatypes::Vector3(0.0, 0.0, total_thrust_));
    apply_force(world_force);                      // Add thrust force at COM
    
    // Convert body torque to world torque
    datatypes::Vector3 world_torque = rotation().rotate(body_torque_);
    apply_torque(world_torque);                    // Add body torque
    
    // Call base class integration (applies forces and updates position)
    RigidEntity::integrate(dt);                    // Base integration handles velocity/position
}

void DroneEntity::integrate_velocity(double dt) {
    // Override to add motor forces before base integration
    // (Forces are already applied in integrate() before calling base)
    RigidEntity::integrate_velocity(dt);           // Call base implementation
}

//------------------------------------------------------------------------------
// State queries
//------------------------------------------------------------------------------
datatypes::Vector3 DroneEntity::get_body_angular_velocity() const {
    // Convert world angular velocity to body frame
    datatypes::Quat q = rotation();                // World to body rotation
    datatypes::Quat q_conj = q.conjugate();        // Inverse rotation
    // Rotate angular velocity vector: ω_body = q⁻¹ * ω_world * q
    datatypes::Vector3 world_omega = angular_velocity();
    datatypes::Quat omega_quat(0.0, world_omega[0], world_omega[1], world_omega[2]);
    datatypes::Quat body_omega_quat = q_conj * omega_quat * q;
    return datatypes::Vector3(body_omega_quat.x, body_omega_quat.y, body_omega_quat.z);
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
void DroneEntity::reset() {
    // Reset rigid body state
    RigidEntity::reset();                          // Base class reset
    
    // Reset PID controllers
    drone_config_.roll_pid.reset();                // Reset roll PID
    drone_config_.pitch_pid.reset();               // Reset pitch PID
    drone_config_.yaw_rate_pid.reset();            // Reset yaw PID
    drone_config_.altitude_pid.reset();            // Reset altitude PID
    drone_config_.velocity_x_pid.reset();          // Reset velocity X PID
    drone_config_.velocity_y_pid.reset();          // Reset velocity Y PID
    
    // Reset motor speeds to zero
    for (auto& motor : drone_config_.motors) {
        motor.current_rpm = 0.0;                   // Reset current RPM
        motor.target_rpm = 0.0;                    // Reset target RPM
    }
    std::fill(motor_speeds_.begin(), motor_speeds_.end(), 0.0); // Clear speed array
    
    // Reset setpoints
    roll_setpoint_ = pitch_setpoint_ = yaw_setpoint_ = 0.0;
    roll_rate_setpoint_ = pitch_rate_setpoint_ = yaw_rate_setpoint_ = 0.0;
    altitude_setpoint_ = 0.0;
    velocity_setpoint_ = datatypes::Vector3(0.0);
    position_setpoint_ = datatypes::Vector3(0.0);
    
    total_thrust_ = 0.0;
    body_torque_ = datatypes::Vector3(0.0);
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string DroneEntity::repr() const {
    std::ostringstream oss;
    oss << "DroneEntity(id=" << id() << ", name=\"" << name() << "\", pos=" << position()
        << ", thrust=" << total_thrust_ << ", motors=[";
    for (size_t i = 0; i < motor_speeds_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(0) << motor_speeds_[i];
    }
    oss << "])";
    return oss.str();
}

std::string DroneEntity::str() const {
    return "Drone " + name() + " (thrust=" + std::to_string(static_cast<int>(total_thrust_)) + "N)";
}

} // namespace engine
} // namespace genesis