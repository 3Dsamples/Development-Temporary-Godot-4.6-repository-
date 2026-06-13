// File 421: modules/integration/unified_physics_project_settings.h
// Centralised, serialisable configuration resource for the complete
// unified multi‑engine physics pipeline.  Reads from and writes to
// Godot's ProjectSettings, allowing users to tune engine selection,
// solver parameters, CCD, sleep thresholds, adaptive quality distances,
// and vehicle default properties from one place.  No visual editor or
// UI is included; the resource exposes properties that can be saved to
// disk and loaded at runtime.

#ifndef INTEGRATION_UNIFIED_PHYSICS_PROJECT_SETTINGS_H
#define INTEGRATION_UNIFIED_PHYSICS_PROJECT_SETTINGS_H

#include "core/io/resource.h"
#include "core/config/project_settings.h"
#include "core/math/vector3.h"
#include "core/string/ustring.h"

namespace unified {

class UnifiedPhysicsProjectSettings : public Resource {
    GDCLASS(UnifiedPhysicsProjectSettings, Resource);

public:
    // -------------------------------------------------------------------
    // General
    // -------------------------------------------------------------------
    String primary_engine = "Newton";
    int    num_worker_threads = 4;
    bool   enable_ccd = true;

    // -------------------------------------------------------------------
    // Solver
    // -------------------------------------------------------------------
    int    solver_iterations = 16;
    real_t erp = 0.2;            // error reduction parameter
    real_t erp2 = 0.8;           // secondary erp (split impulse)
    real_t cfm = 0.001;          // constraint force mixing

    // -------------------------------------------------------------------
    // Sleep / deactivation
    // -------------------------------------------------------------------
    real_t sleep_linear_threshold = 0.01;
    real_t sleep_angular_threshold = 0.01;
    int    sleep_frames = 10;

    // -------------------------------------------------------------------
    // Adaptive quality (LOD)
    // -------------------------------------------------------------------
    bool   enable_adaptive_quality = true;
    real_t adaptive_tier2_distance = 30.0;
    real_t adaptive_tier1_distance = 80.0;
    real_t adaptive_tier0_distance = 150.0;
    real_t adaptive_max_physics_budget_ms = 4.0;

    // -------------------------------------------------------------------
    // Vehicles
    // -------------------------------------------------------------------
    bool   enable_vehicles = true;
    real_t default_vehicle_engine_max_force = 5000.0;
    real_t default_vehicle_engine_max_speed = 50.0;
    real_t default_vehicle_suspension_stiffness = 30000.0;
    real_t default_vehicle_suspension_damping = 3000.0;

    // -------------------------------------------------------------------
    // Cloth
    // -------------------------------------------------------------------
    bool   enable_cloth = true;
    real_t default_cloth_structural_stiffness = 1000.0;
    real_t default_cloth_shear_stiffness = 100.0;
    real_t default_cloth_bending_stiffness = 200.0;
    real_t default_cloth_damping = 0.01;

    // -------------------------------------------------------------------
    // Particles
    // -------------------------------------------------------------------
    bool   enable_particles = true;
    real_t default_particle_radius = 0.05;
    int    default_max_particles = 1000;

    // -------------------------------------------------------------------
    // Debug / Profiling
    // -------------------------------------------------------------------
    bool   enable_profiler = false;
    int    profiler_export_interval_frames = 120;

    // -------------------------------------------------------------------
    // Constructor – fills defaults from current ProjectSettings.
    // -------------------------------------------------------------------
    UnifiedPhysicsProjectSettings();

    // -------------------------------------------------------------------
    // Read all values from the current ProjectSettings store.
    // -------------------------------------------------------------------
    void load_from_project_settings();

    // -------------------------------------------------------------------
    // Write all current values into the ProjectSettings store.
    // -------------------------------------------------------------------
    void save_to_project_settings() const;

    // -------------------------------------------------------------------
    // Accessors (for ClassDB binding).
    // -------------------------------------------------------------------
    void set_primary_engine(const String &p_val);
    String get_primary_engine() const;
    void set_num_worker_threads(int p_val);
    int get_num_worker_threads() const;
    void set_enable_ccd(bool p_val);
    bool get_enable_ccd() const;
    void set_solver_iterations(int p_val);
    int get_solver_iterations() const;
    void set_erp(real_t p_val);
    real_t get_erp() const;
    void set_erp2(real_t p_val);
    real_t get_erp2() const;
    void set_cfm(real_t p_val);
    real_t get_cfm() const;
    void set_sleep_linear_threshold(real_t p_val);
    real_t get_sleep_linear_threshold() const;
    void set_sleep_angular_threshold(real_t p_val);
    real_t get_sleep_angular_threshold() const;
    void set_sleep_frames(int p_val);
    int get_sleep_frames() const;
    void set_enable_adaptive_quality(bool p_val);
    bool get_enable_adaptive_quality() const;
    void set_adaptive_tier2_distance(real_t p_val);
    real_t get_adaptive_tier2_distance() const;
    void set_adaptive_tier1_distance(real_t p_val);
    real_t get_adaptive_tier1_distance() const;
    void set_adaptive_tier0_distance(real_t p_val);
    real_t get_adaptive_tier0_distance() const;
    void set_adaptive_max_physics_budget_ms(real_t p_val);
    real_t get_adaptive_max_physics_budget_ms() const;
    void set_enable_vehicles(bool p_val);
    bool get_enable_vehicles() const;
    void set_default_vehicle_engine_max_force(real_t p_val);
    real_t get_default_vehicle_engine_max_force() const;
    void set_default_vehicle_engine_max_speed(real_t p_val);
    real_t get_default_vehicle_engine_max_speed() const;
    void set_default_vehicle_suspension_stiffness(real_t p_val);
    real_t get_default_vehicle_suspension_stiffness() const;
    void set_default_vehicle_suspension_damping(real_t p_val);
    real_t get_default_vehicle_suspension_damping() const;
    void set_enable_cloth(bool p_val);
    bool get_enable_cloth() const;
    void set_default_cloth_structural_stiffness(real_t p_val);
    real_t get_default_cloth_structural_stiffness() const;
    void set_default_cloth_shear_stiffness(real_t p_val);
    real_t get_default_cloth_shear_stiffness() const;
    void set_default_cloth_bending_stiffness(real_t p_val);
    real_t get_default_cloth_bending_stiffness() const;
    void set_default_cloth_damping(real_t p_val);
    real_t get_default_cloth_damping() const;
    void set_enable_particles(bool p_val);
    bool get_enable_particles() const;
    void set_default_particle_radius(real_t p_val);
    real_t get_default_particle_radius() const;
    void set_default_max_particles(int p_val);
    int get_default_max_particles() const;
    void set_enable_profiler(bool p_val);
    bool get_enable_profiler() const;
    void set_profiler_export_interval_frames(int p_val);
    int get_profiler_export_interval_frames() const;

protected:
    static void _bind_methods();
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_PROJECT_SETTINGS_H