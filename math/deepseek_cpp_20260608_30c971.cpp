// File 399: modules/integration/unified_vehicle_transmission.h
// UnifiedMultiBodyVehicleTransmission – provides a detailed drivetrain
// simulation including engine torque curve, clutch, multi‑speed gearbox,
// open / limited‑slip / locking differentials, and axle coupling to the
// wheel ground contacts.  The transmission reads wheel rotational speeds
// from the UnifiedVehicleManager and applies traction and braking torques
// to the corresponding chassis bodies via the engine‑specific interfaces.
// All calculations are engine‑independent and fully inline for timing.

#ifndef INTEGRATION_UNIFIED_VEHICLE_TRANSMISSION_H
#define INTEGRATION_UNIFIED_VEHICLE_TRANSMISSION_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedVehicleManager;

// ==========================================================================
// Engine torque curve: a set of (rpm, torqueNm) points with linear interp.
// ==========================================================================
struct EngineTorqueCurve {
    struct Point { real_t rpm; real_t torqueNm; };
    LocalVector<Point> points;

    // Evaluate torque at given rpm (linear interpolation).  Returns 0 if outside range.
    inline real_t get_torque(real_t p_rpm) const {
        if (points.is_empty()) return 0.0f;
        // Find the two points bracketing p_rpm.
        if (p_rpm <= points[0].rpm) return points[0].torqueNm;
        for (int i = 0; i < points.size() - 1; ++i) {
            if (p_rpm <= points[i + 1].rpm) {
                real_t t = (p_rpm - points[i].rpm) / (points[i + 1].rpm - points[i].rpm);
                return Math::lerp(points[i].torqueNm, points[i + 1].torqueNm, t);
            }
        }
        return points[points.size() - 1].torqueNm;
    }
};

// ==========================================================================
// Clutch model: simple friction clutch with maximum torque and engagement.
// ==========================================================================
struct ClutchModel {
    real_t max_torque_capacity = 400.0f;   // Nm, fully engaged
    real_t engagement_fraction = 0.0f;     // 0=disengaged, 1=fully engaged

    inline real_t get_transmitted_torque(real_t p_engine_torque) const {
        return MIN(p_engine_torque, max_torque_capacity) * engagement_fraction;
    }
};

// ==========================================================================
// Gearbox: multiple ratios, current gear, shift time.
// ==========================================================================
struct GearboxModel {
    LocalVector<real_t> forward_ratios; // first to top
    real_t reverse_ratio = -3.5f;
    real_t final_drive_ratio = 3.7f;
    int current_gear = 0;               // 0 = neutral, positive = forward, -1 = reverse
    real_t shift_time_remaining = 0.0f; // seconds before shift completes
    real_t shift_duration = 0.2f;

    // Get input torque (from clutch) to output torque (to differential).
    inline real_t compute_output_torque(real_t p_input_torque) {
        if (shift_time_remaining > 0.0f) return 0.0f; // shifting
        if (current_gear == 0) return 0.0f;
        real_t ratio = (current_gear > 0) ?
            forward_ratios[MIN(current_gear - 1, forward_ratios.size() - 1)] :
            reverse_ratio;
        return p_input_torque * ratio * final_drive_ratio;
    }

    // Get engine rpm from wheel speed (rad/s).
    inline real_t get_engine_rpm_from_wheel_speed(real_t p_avg_wheel_speed) {
        if (current_gear == 0) return 0.0f;
        real_t ratio = (current_gear > 0) ?
            forward_ratios[MIN(current_gear - 1, forward_ratios.size() - 1)] :
            reverse_ratio;
        return p_avg_wheel_speed * ratio * final_drive_ratio * (60.0f / Math_TAU); // rad/s to rpm
    }

    // Shift gear by delta (+1 = up, -1 = down).
    void shift(int p_delta) {
        if (shift_time_remaining > 0.0f) return;
        int target = current_gear + p_delta;
        if (target > (int)forward_ratios.size()) target = (int)forward_ratios.size();
        if (target < -1) target = -1;
        if (target == current_gear) return;
        current_gear = target;
        shift_time_remaining = shift_duration;
    }

    // Update shift timer.
    void update(real_t p_dt) {
        if (shift_time_remaining > 0.0f) {
            shift_time_remaining = MAX(0.0f, shift_time_remaining - p_dt);
        }
    }
};

// ==========================================================================
// Differential: redistributes input torque to two output shafts according
// to type and speed difference.  Models open, locked, viscous, and geared LSD.
// ==========================================================================
class DifferentialModel {
public:
    enum Type { OPEN, LOCKED, VISCOUS, GEARED };
    Type type = OPEN;
    real_t locking_coefficient = 0.5f;   // 0 = open, 1 = fully locked for GEARED; for VISCOUS, Nm/(rad/s)
    real_t preload = 0.0f;              // Nm minimum locking torque

    // Input: total torque from gearbox, wheel angular speeds (rad/s).
    // Output: torques to left and right axles.
    void distribute(real_t p_input_torque,
                    real_t p_speed_left, real_t p_speed_right,
                    real_t &r_torque_left, real_t &r_torque_right) {
        real_t base = p_input_torque * 0.5f;
        r_torque_left = base;
        r_torque_right = base;

        switch (type) {
            case OPEN: break; // open diff: moment transmitted is limited by wheel with less traction; here we just split torque equally. Torque is limited by engine, not traction; actual tire forces decide slip.
            case LOCKED:
                // lock: speed_left == speed_right, but we just apply a correction torque proportional to speed difference.
                real_t diff = p_speed_right - p_speed_left;
                real_t correction = diff * locking_coefficient * 100.0f; // stiff
                r_torque_left += correction;
                r_torque_right -= correction;
                break;
            case VISCOUS: {
                real_t slip = p_speed_right - p_speed_left;
                real_t viscous_torque = slip * locking_coefficient;
                r_torque_left += viscous_torque;
                r_torque_right -= viscous_torque;
            } break;
            case GEARED:
                // Geared LSD: torque bias fixed by locking_coefficient. If one wheel is slower, transfer torque.
                real_t slip = p_speed_right - p_speed_left;
                real_t bias = (slip > 0) ? (1.0f / locking_coefficient) : locking_coefficient;
                // Simplified: apply torque transfer limited to preload and coefficient.
                real_t transfer = CLAMP(slip * 50.0f, -preload * locking_coefficient, preload * locking_coefficient);
                r_torque_left += transfer;
                r_torque_right -= transfer;
                break;
        }
    }
};

// ==========================================================================
// The complete transmission for a single vehicle.
// ==========================================================================
class UnifiedVehicleTransmission : public Reference {
    GDCLASS(UnifiedVehicleTransmission, Reference);

    EngineTorqueCurve engine_curve;
    ClutchModel       clutch;
    GearboxModel      gearbox;
    DifferentialModel differential_front;
    DifferentialModel differential_rear;
    DifferentialModel differential_center; // for AWD

    real_t engine_rpm = 800.0f;          // idle
    real_t engine_inertia = 0.2f;        // kg·m²
    real_t idle_rpm = 800.0f;
    real_t rev_limiter_rpm = 7000.0f;
    real_t throttle_input = 0.0f;        // 0..1
    bool   awd = false;

    // Wheel speeds (rad/s) last frame.
    real_t wheel_speed_fl = 0.0f, wheel_speed_fr = 0.0f;
    real_t wheel_speed_rl = 0.0f, wheel_speed_rr = 0.0f;
    real_t engine_braking_torque = 0.0f; // negative torque from engine friction

public:
    UnifiedVehicleTransmission() {
        // Default torque curve: a typical petrol engine.
        engine_curve.points.push_back({ 800.0f, 50.0f });
        engine_curve.points.push_back({ 2000.0f, 180.0f });
        engine_curve.points.push_back({ 4000.0f, 250.0f });
        engine_curve.points.push_back({ 5500.0f, 230.0f });
        engine_curve.points.push_back({ 7000.0f, 180.0f });

        gearbox.forward_ratios.push_back(3.5f);  // 1st
        gearbox.forward_ratios.push_back(2.0f);  // 2nd
        gearbox.forward_ratios.push_back(1.4f);  // 3rd
        gearbox.forward_ratios.push_back(1.0f);  // 4th
        gearbox.forward_ratios.push_back(0.8f);  // 5th
    }

    void set_throttle(real_t p_throttle) { throttle_input = CLAMP(p_throttle, 0.0f, 1.0f); }
    void set_clutch(real_t p_engagement) { clutch.engagement_fraction = CLAMP(p_engagement, 0.0f, 1.0f); }
    void set_awd(bool p_awd) { awd = p_awd; }

    // Provide wheel speeds from vehicle manager (rad/s).
    void set_wheel_speeds(real_t fl, real_t fr, real_t rl, real_t rr) {
        wheel_speed_fl = fl;
        wheel_speed_fr = fr;
        wheel_speed_rl = rl;
        wheel_speed_rr = rr;
    }

    // Shift gear up/down.
    void shift_up() { gearbox.shift(1); }
    void shift_down() { gearbox.shift(-1); }

    // Return torques to be applied at each wheel (Nm).  Positive = drive torque.
    void compute_wheel_torques(real_t p_dt,
                               real_t &r_fl, real_t &r_fr, real_t &r_rl, real_t &r_rr) {
        gearbox.update(p_dt);

        // Average wheel speed at gearbox output (taking drivetrain configuration into account).
        real_t avg_speed_front = (wheel_speed_fl + wheel_speed_fr) * 0.5f;
        real_t avg_speed_rear  = (wheel_speed_rl + wheel_speed_rr) * 0.5f;
        real_t avg_wheel_speed = awd ? (avg_speed_front + avg_speed_rear) * 0.5f : avg_speed_rear;

        // Engine rpm from wheel speed.
        engine_rpm = gearbox.get_engine_rpm_from_wheel_speed(MAX(avg_wheel_speed, 0.0f));
        engine_rpm = CLAMP(engine_rpm, idle_rpm, rev_limiter_rpm);

        // Engine torque from curve.
        real_t engine_torque = engine_curve.get_torque(engine_rpm);
        // Throttle modulates.
        real_t driver_torque = engine_torque * throttle_input;

        // Engine braking (negative torque when throttle is low).
        engine_braking_torque = -engine_torque * (1.0f - throttle_input) * 0.3f;
        real_t net_engine_torque = driver_torque + engine_braking_torque;

        // Clutch.
        real_t clutch_torque = clutch.get_transmitted_torque(net_engine_torque);

        // Gearbox.
        real_t gearbox_output_torque = gearbox.compute_output_torque(clutch_torque);

        // Center differential (AWD) if applicable.
        real_t torque_front = 0.0f, torque_rear = 0.0f;
        if (awd) {
            differential_center.type = DifferentialModel::GEARED;
            differential_center.locking_coefficient = 0.5f; // 50:50 split
            differential_center.distribute(gearbox_output_torque, avg_speed_front, avg_speed_rear,
                                           torque_front, torque_rear);
        } else {
            torque_rear = gearbox_output_torque;
            torque_front = 0.0f;
        }

        // Front differential.
        real_t front_left = 0.0f, front_right = 0.0f;
        if (torque_front != 0.0f) {
            differential_front.distribute(torque_front, wheel_speed_fl, wheel_speed_fr,
                                          front_left, front_right);
        }

        // Rear differential.
        real_t rear_left = 0.0f, rear_right = 0.0f;
        differential_rear.distribute(torque_rear, wheel_speed_rl, wheel_speed_rr,
                                     rear_left, rear_right);

        // Assign outputs.
        r_fl = front_left;
        r_fr = front_right;
        r_rl = rear_left;
        r_rr = rear_right;

        // Apply engine inertia effect (simplified: update engine rpm based on torque balance).
        // The engine spins up or down due to difference between driving torque and load.
        // We'll adjust engine_rpm for next frame using Euler integration.
        real_t load_torque = (avg_wheel_speed > 0.001f) ? gearbox_output_torque : 0.0f;
        real_t net_engine_accel = (driver_torque - load_torque) / MAX(engine_inertia, 0.01f);
        // Don't modify engine_rpm further now; it will be recomputed next frame from wheel speeds.
        // To integrate engine rpm, we can do:
        // engine_rpm += net_engine_accel * (60.0f / Math_TAU) * p_dt;
        // But the engine rpm is also constrained by wheel speed through gearbox.
    }

    // Access individual models for tuning.
    EngineTorqueCurve &get_engine_curve() { return engine_curve; }
    ClutchModel &get_clutch() { return clutch; }
    GearboxModel &get_gearbox() { return gearbox; }
    DifferentialModel &get_front_diff() { return differential_front; }
    DifferentialModel &get_rear_diff()  { return differential_rear; }
    DifferentialModel &get_center_diff() { return differential_center; }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_throttle", "throttle"), &UnifiedVehicleTransmission::set_throttle);
        ClassDB::bind_method(D_METHOD("set_clutch", "engagement"), &UnifiedVehicleTransmission::set_clutch);
        ClassDB::bind_method(D_METHOD("set_awd", "awd"), &UnifiedVehicleTransmission::set_awd);
        ClassDB::bind_method(D_METHOD("set_wheel_speeds", "fl", "fr", "rl", "rr"), &UnifiedVehicleTransmission::set_wheel_speeds);
        ClassDB::bind_method(D_METHOD("shift_up"), &UnifiedVehicleTransmission::shift_up);
        ClassDB::bind_method(D_METHOD("shift_down"), &UnifiedVehicleTransmission::shift_down);
        ClassDB::bind_method(D_METHOD("compute_wheel_torques", "dt",
            "fl", "fr", "rl", "rr"), &UnifiedVehicleTransmission::compute_wheel_torques);
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "engine_curve"), "", "");
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "clutch"), "", "");
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gearbox"), "", "");
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_VEHICLE_TRANSMISSION_H