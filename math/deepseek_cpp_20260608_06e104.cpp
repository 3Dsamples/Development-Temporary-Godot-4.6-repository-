// File 422: modules/integration/unified_physics_project_settings.cpp
// Implementation of UnifiedPhysicsProjectSettings.  All ProjectSettings
// keys are defined with the "physics/unified/" prefix.  The resource
// provides a save/load round‑trip so that users can edit values in
// the inspector and have them persist across editor restarts.
// Every property has a dedicated getter/setter that directly accesses
// the ProjectSettings singleton.

#include "unified_physics_project_settings.h"
#include "core/config/project_settings.h"
#include "core/variant/variant.h"

namespace unified {

// -------------------------------------------------------------------
// ProjectSettings keys (centralised, easy to search).
// -------------------------------------------------------------------
#define PS_KEY(a) String("physics/unified/" a)

UnifiedPhysicsProjectSettings::UnifiedPhysicsProjectSettings() {
    // Initialise defaults before loading from project settings.
    primary_engine = "Newton";
    num_worker_threads = 4;
    enable_ccd = true;
    solver_iterations = 16;
    erp = 0.2;
    erp2 = 0.8;
    cfm = 0.001;
    sleep_linear_threshold = 0.01;
    sleep_angular_threshold = 0.01;
    sleep_frames = 10;
    enable_adaptive_quality = true;
    adaptive_tier2_distance = 30.0;
    adaptive_tier1_distance = 80.0;
    adaptive_tier0_distance = 150.0;
    adaptive_max_physics_budget_ms = 4.0;
    enable_vehicles = true;
    default_vehicle_engine_max_force = 5000.0;
    default_vehicle_engine_max_speed = 50.0;
    default_vehicle_suspension_stiffness = 30000.0;
    default_vehicle_suspension_damping = 3000.0;
    enable_cloth = true;
    default_cloth_structural_stiffness = 1000.0;
    default_cloth_shear_stiffness = 100.0;
    default_cloth_bending_stiffness = 200.0;
    default_cloth_damping = 0.01;
    enable_particles = true;
    default_particle_radius = 0.05;
    default_max_particles = 1000;
    enable_profiler = false;
    profiler_export_interval_frames = 120;

    load_from_project_settings();
}

void UnifiedPhysicsProjectSettings::load_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    ERR_FAIL_COND(!ps);

    primary_engine = GLOBAL_GET(PS_KEY("primary_engine"));
    num_worker_threads = GLOBAL_GET(PS_KEY("threads"));
    enable_ccd = GLOBAL_GET(PS_KEY("enable_ccd"));
    solver_iterations = GLOBAL_GET(PS_KEY("solver_iterations"));
    erp = GLOBAL_GET(PS_KEY("erp"));
    erp2 = GLOBAL_GET(PS_KEY("erp2"));
    cfm = GLOBAL_GET(PS_KEY("cfm"));
    sleep_linear_threshold = GLOBAL_GET(PS_KEY("sleep_linear_threshold"));
    sleep_angular_threshold = GLOBAL_GET(PS_KEY("sleep_angular_threshold"));
    sleep_frames = GLOBAL_GET(PS_KEY("sleep_frames"));
    enable_adaptive_quality = GLOBAL_GET(PS_KEY("adaptive_quality_enabled"));
    adaptive_tier2_distance = GLOBAL_GET(PS_KEY("adaptive_tier2_distance"));
    adaptive_tier1_distance = GLOBAL_GET(PS_KEY("adaptive_tier1_distance"));
    adaptive_tier0_distance = GLOBAL_GET(PS_KEY("adaptive_tier0_distance"));
    adaptive_max_physics_budget_ms = GLOBAL_GET(PS_KEY("adaptive_max_budget_ms"));
    enable_vehicles = GLOBAL_GET(PS_KEY("enable_vehicles"));
    default_vehicle_engine_max_force = GLOBAL_GET(PS_KEY("vehicle_engine_max_force"));
    default_vehicle_engine_max_speed = GLOBAL_GET(PS_KEY("vehicle_engine_max_speed"));
    default_vehicle_suspension_stiffness = GLOBAL_GET(PS_KEY("vehicle_suspension_stiffness"));
    default_vehicle_suspension_damping = GLOBAL_GET(PS_KEY("vehicle_suspension_damping"));
    enable_cloth = GLOBAL_GET(PS_KEY("enable_cloth"));
    default_cloth_structural_stiffness = GLOBAL_GET(PS_KEY("cloth_structural_stiffness"));
    default_cloth_shear_stiffness = GLOBAL_GET(PS_KEY("cloth_shear_stiffness"));
    default_cloth_bending_stiffness = GLOBAL_GET(PS_KEY("cloth_bending_stiffness"));
    default_cloth_damping = GLOBAL_GET(PS_KEY("cloth_damping"));
    enable_particles = GLOBAL_GET(PS_KEY("enable_particles"));
    default_particle_radius = GLOBAL_GET(PS_KEY("particle_radius"));
    default_max_particles = GLOBAL_GET(PS_KEY("particle_max_count"));
    enable_profiler = GLOBAL_GET(PS_KEY("enable_profiler"));
    profiler_export_interval_frames = GLOBAL_GET(PS_KEY("profiler_export_interval_frames"));
}

void UnifiedPhysicsProjectSettings::save_to_project_settings() const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    ERR_FAIL_COND(!ps);

    ps->set_setting(PS_KEY("primary_engine"), primary_engine);
    ps->set_setting(PS_KEY("threads"), num_worker_threads);
    ps->set_setting(PS_KEY("enable_ccd"), enable_ccd);
    ps->set_setting(PS_KEY("solver_iterations"), solver_iterations);
    ps->set_setting(PS_KEY("erp"), erp);
    ps->set_setting(PS_KEY("erp2"), erp2);
    ps->set_setting(PS_KEY("cfm"), cfm);
    ps->set_setting(PS_KEY("sleep_linear_threshold"), sleep_linear_threshold);
    ps->set_setting(PS_KEY("sleep_angular_threshold"), sleep_angular_threshold);
    ps->set_setting(PS_KEY("sleep_frames"), sleep_frames);
    ps->set_setting(PS_KEY("adaptive_quality_enabled"), enable_adaptive_quality);
    ps->set_setting(PS_KEY("adaptive_tier2_distance"), adaptive_tier2_distance);
    ps->set_setting(PS_KEY("adaptive_tier1_distance"), adaptive_tier1_distance);
    ps->set_setting(PS_KEY("adaptive_tier0_distance"), adaptive_tier0_distance);
    ps->set_setting(PS_KEY("adaptive_max_budget_ms"), adaptive_max_physics_budget_ms);
    ps->set_setting(PS_KEY("enable_vehicles"), enable_vehicles);
    ps->set_setting(PS_KEY("vehicle_engine_max_force"), default_vehicle_engine_max_force);
    ps->set_setting(PS_KEY("vehicle_engine_max_speed"), default_vehicle_engine_max_speed);
    ps->set_setting(PS_KEY("vehicle_suspension_stiffness"), default_vehicle_suspension_stiffness);
    ps->set_setting(PS_KEY("vehicle_suspension_damping"), default_vehicle_suspension_damping);
    ps->set_setting(PS_KEY("enable_cloth"), enable_cloth);
    ps->set_setting(PS_KEY("cloth_structural_stiffness"), default_cloth_structural_stiffness);
    ps->set_setting(PS_KEY("cloth_shear_stiffness"), default_cloth_shear_stiffness);
    ps->set_setting(PS_KEY("cloth_bending_stiffness"), default_cloth_bending_stiffness);
    ps->set_setting(PS_KEY("cloth_damping"), default_cloth_damping);
    ps->set_setting(PS_KEY("enable_particles"), enable_particles);
    ps->set_setting(PS_KEY("particle_radius"), default_particle_radius);
    ps->set_setting(PS_KEY("particle_max_count"), default_max_particles);
    ps->set_setting(PS_KEY("enable_profiler"), enable_profiler);
    ps->set_setting(PS_KEY("profiler_export_interval_frames"), profiler_export_interval_frames);

    ps->save();
}

// -------------------------------------------------------------------
// Getters / Setters
// -------------------------------------------------------------------
#define IMPL_PROPERTY(type, name, ps_key) \
    void UnifiedPhysicsProjectSettings::set_##name(type p_val) { name = p_val; } \
    type UnifiedPhysicsProjectSettings::get_##name() const { return name; }

IMPL_PROPERTY(String, primary_engine, "primary_engine");
IMPL_PROPERTY(int, num_worker_threads, "threads");
IMPL_PROPERTY(bool, enable_ccd, "enable_ccd");
IMPL_PROPERTY(int, solver_iterations, "solver_iterations");
IMPL_PROPERTY(real_t, erp, "erp");
IMPL_PROPERTY(real_t, erp2, "erp2");
IMPL_PROPERTY(real_t, cfm, "cfm");
IMPL_PROPERTY(real_t, sleep_linear_threshold, "sleep_linear_threshold");
IMPL_PROPERTY(real_t, sleep_angular_threshold, "sleep_angular_threshold");
IMPL_PROPERTY(int, sleep_frames, "sleep_frames");
IMPL_PROPERTY(bool, enable_adaptive_quality, "adaptive_quality_enabled");
IMPL_PROPERTY(real_t, adaptive_tier2_distance, "adaptive_tier2_distance");
IMPL_PROPERTY(real_t, adaptive_tier1_distance, "adaptive_tier1_distance");
IMPL_PROPERTY(real_t, adaptive_tier0_distance, "adaptive_tier0_distance");
IMPL_PROPERTY(real_t, adaptive_max_physics_budget_ms, "adaptive_max_budget_ms");
IMPL_PROPERTY(bool, enable_vehicles, "enable_vehicles");
IMPL_PROPERTY(real_t, default_vehicle_engine_max_force, "vehicle_engine_max_force");
IMPL_PROPERTY(real_t, default_vehicle_engine_max_speed, "vehicle_engine_max_speed");
IMPL_PROPERTY(real_t, default_vehicle_suspension_stiffness, "vehicle_suspension_stiffness");
IMPL_PROPERTY(real_t, default_vehicle_suspension_damping, "vehicle_suspension_damping");
IMPL_PROPERTY(bool, enable_cloth, "enable_cloth");
IMPL_PROPERTY(real_t, default_cloth_structural_stiffness, "cloth_structural_stiffness");
IMPL_PROPERTY(real_t, default_cloth_shear_stiffness, "cloth_shear_stiffness");
IMPL_PROPERTY(real_t, default_cloth_bending_stiffness, "cloth_bending_stiffness");
IMPL_PROPERTY(real_t, default_cloth_damping, "cloth_damping");
IMPL_PROPERTY(bool, enable_particles, "enable_particles");
IMPL_PROPERTY(real_t, default_particle_radius, "particle_radius");
IMPL_PROPERTY(int, default_max_particles, "particle_max_count");
IMPL_PROPERTY(bool, enable_profiler, "enable_profiler");
IMPL_PROPERTY(int, profiler_export_interval_frames, "profiler_export_interval_frames");

#undef IMPL_PROPERTY

// -------------------------------------------------------------------
// Godot bindings.
// -------------------------------------------------------------------
void UnifiedPhysicsProjectSettings::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_from_project_settings"), &UnifiedPhysicsProjectSettings::load_from_project_settings);
    ClassDB::bind_method(D_METHOD("save_to_project_settings"), &UnifiedPhysicsProjectSettings::save_to_project_settings);

    ClassDB::bind_method(D_METHOD("set_primary_engine", "engine"), &UnifiedPhysicsProjectSettings::set_primary_engine);
    ClassDB::bind_method(D_METHOD("get_primary_engine"), &UnifiedPhysicsProjectSettings::get_primary_engine);
    ClassDB::bind_method(D_METHOD("set_num_worker_threads", "threads"), &UnifiedPhysicsProjectSettings::set_num_worker_threads);
    ClassDB::bind_method(D_METHOD("get_num_worker_threads"), &UnifiedPhysicsProjectSettings::get_num_worker_threads);
    ClassDB::bind_method(D_METHOD("set_enable_ccd", "enabled"), &UnifiedPhysicsProjectSettings::set_enable_ccd);
    ClassDB::bind_method(D_METHOD("get_enable_ccd"), &UnifiedPhysicsProjectSettings::get_enable_ccd);
    ClassDB::bind_method(D_METHOD("set_solver_iterations", "iterations"), &UnifiedPhysicsProjectSettings::set_solver_iterations);
    ClassDB::bind_method(D_METHOD("get_solver_iterations"), &UnifiedPhysicsProjectSettings::get_solver_iterations);
    ClassDB::bind_method(D_METHOD("set_erp", "erp"), &UnifiedPhysicsProjectSettings::set_erp);
    ClassDB::bind_method(D_METHOD("get_erp"), &UnifiedPhysicsProjectSettings::get_erp);
    ClassDB::bind_method(D_METHOD("set_erp2", "erp2"), &UnifiedPhysicsProjectSettings::set_erp2);
    ClassDB::bind_method(D_METHOD("get_erp2"), &UnifiedPhysicsProjectSettings::get_erp2);
    ClassDB::bind_method(D_METHOD("set_cfm", "cfm"), &UnifiedPhysicsProjectSettings::set_cfm);
    ClassDB::bind_method(D_METHOD("get_cfm"), &UnifiedPhysicsProjectSettings::get_cfm);

    ClassDB::bind_method(D_METHOD("set_sleep_linear_threshold", "threshold"), &UnifiedPhysicsProjectSettings::set_sleep_linear_threshold);
    ClassDB::bind_method(D_METHOD("get_sleep_linear_threshold"), &UnifiedPhysicsProjectSettings::get_sleep_linear_threshold);
    ClassDB::bind_method(D_METHOD("set_sleep_angular_threshold", "threshold"), &UnifiedPhysicsProjectSettings::set_sleep_angular_threshold);
    ClassDB::bind_method(D_METHOD("get_sleep_angular_threshold"), &UnifiedPhysicsProjectSettings::get_sleep_angular_threshold);
    ClassDB::bind_method(D_METHOD("set_sleep_frames", "frames"), &UnifiedPhysicsProjectSettings::set_sleep_frames);
    ClassDB::bind_method(D_METHOD("get_sleep_frames"), &UnifiedPhysicsProjectSettings::get_sleep_frames);

    ClassDB::bind_method(D_METHOD("set_enable_adaptive_quality", "enabled"), &UnifiedPhysicsProjectSettings::set_enable_adaptive_quality);
    ClassDB::bind_method(D_METHOD("get_enable_adaptive_quality"), &UnifiedPhysicsProjectSettings::get_enable_adaptive_quality);
    ClassDB::bind_method(D_METHOD("set_adaptive_tier2_distance", "distance"), &UnifiedPhysicsProjectSettings::set_adaptive_tier2_distance);
    ClassDB::bind_method(D_METHOD("get_adaptive_tier2_distance"), &UnifiedPhysicsProjectSettings::get_adaptive_tier2_distance);
    ClassDB::bind_method(D_METHOD("set_adaptive_tier1_distance", "distance"), &UnifiedPhysicsProjectSettings::set_adaptive_tier1_distance);
    ClassDB::bind_method(D_METHOD("get_adaptive_tier1_distance"), &UnifiedPhysicsProjectSettings::get_adaptive_tier1_distance);
    ClassDB::bind_method(D_METHOD("set_adaptive_tier0_distance", "distance"), &UnifiedPhysicsProjectSettings::set_adaptive_tier0_distance);
    ClassDB::bind_method(D_METHOD("get_adaptive_tier0_distance"), &UnifiedPhysicsProjectSettings::get_adaptive_tier0_distance);
    ClassDB::bind_method(D_METHOD("set_adaptive_max_physics_budget_ms", "budget_ms"), &UnifiedPhysicsProjectSettings::set_adaptive_max_physics_budget_ms);
    ClassDB::bind_method(D_METHOD("get_adaptive_max_physics_budget_ms"), &UnifiedPhysicsProjectSettings::get_adaptive_max_physics_budget_ms);

    ClassDB::bind_method(D_METHOD("set_enable_vehicles", "enabled"), &UnifiedPhysicsProjectSettings::set_enable_vehicles);
    ClassDB::bind_method(D_METHOD("get_enable_vehicles"), &UnifiedPhysicsProjectSettings::get_enable_vehicles);
    ClassDB::bind_method(D_METHOD("set_default_vehicle_engine_max_force", "force"), &UnifiedPhysicsProjectSettings::set_default_vehicle_engine_max_force);
    ClassDB::bind_method(D_METHOD("get_default_vehicle_engine_max_force"), &UnifiedPhysicsProjectSettings::get_default_vehicle_engine_max_force);
    ClassDB::bind_method(D_METHOD("set_default_vehicle_engine_max_speed", "speed"), &UnifiedPhysicsProjectSettings::set_default_vehicle_engine_max_speed);
    ClassDB::bind_method(D_METHOD("get_default_vehicle_engine_max_speed"), &UnifiedPhysicsProjectSettings::get_default_vehicle_engine_max_speed);
    ClassDB::bind_method(D_METHOD("set_default_vehicle_suspension_stiffness", "stiffness"), &UnifiedPhysicsProjectSettings::set_default_vehicle_suspension_stiffness);
    ClassDB::bind_method(D_METHOD("get_default_vehicle_suspension_stiffness"), &UnifiedPhysicsProjectSettings::get_default_vehicle_suspension_stiffness);
    ClassDB::bind_method(D_METHOD("set_default_vehicle_suspension_damping", "damping"), &UnifiedPhysicsProjectSettings::set_default_vehicle_suspension_damping);
    ClassDB::bind_method(D_METHOD("get_default_vehicle_suspension_damping"), &UnifiedPhysicsProjectSettings::get_default_vehicle_suspension_damping);

    ClassDB::bind_method(D_METHOD("set_enable_cloth", "enabled"), &UnifiedPhysicsProjectSettings::set_enable_cloth);
    ClassDB::bind_method(D_METHOD("get_enable_cloth"), &UnifiedPhysicsProjectSettings::get_enable_cloth);
    ClassDB::bind_method(D_METHOD("set_default_cloth_structural_stiffness", "stiffness"), &UnifiedPhysicsProjectSettings::set_default_cloth_structural_stiffness);
    ClassDB::bind_method(D_METHOD("get_default_cloth_structural_stiffness"), &UnifiedPhysicsProjectSettings::get_default_cloth_structural_stiffness);
    ClassDB::bind_method(D_METHOD("set_default_cloth_shear_stiffness", "stiffness"), &UnifiedPhysicsProjectSettings::set_default_cloth_shear_stiffness);
    ClassDB::bind_method(D_METHOD("get_default_cloth_shear_stiffness"), &UnifiedPhysicsProjectSettings::get_default_cloth_shear_stiffness);
    ClassDB::bind_method(D_METHOD("set_default_cloth_bending_stiffness", "stiffness"), &UnifiedPhysicsProjectSettings::set_default_cloth_bending_stiffness);
    ClassDB::bind_method(D_METHOD("get_default_cloth_bending_stiffness"), &UnifiedPhysicsProjectSettings::get_default_cloth_bending_stiffness);
    ClassDB::bind_method(D_METHOD("set_default_cloth_damping", "damping"), &UnifiedPhysicsProjectSettings::set_default_cloth_damping);
    ClassDB::bind_method(D_METHOD("get_default_cloth_damping"), &UnifiedPhysicsProjectSettings::get_default_cloth_damping);

    ClassDB::bind_method(D_METHOD("set_enable_particles", "enabled"), &UnifiedPhysicsProjectSettings::set_enable_particles);
    ClassDB::bind_method(D_METHOD("get_enable_particles"), &UnifiedPhysicsProjectSettings::get_enable_particles);
    ClassDB::bind_method(D_METHOD("set_default_particle_radius", "radius"), &UnifiedPhysicsProjectSettings::set_default_particle_radius);
    ClassDB::bind_method(D_METHOD("get_default_particle_radius"), &UnifiedPhysicsProjectSettings::get_default_particle_radius);
    ClassDB::bind_method(D_METHOD("set_default_max_particles", "max"), &UnifiedPhysicsProjectSettings::set_default_max_particles);
    ClassDB::bind_method(D_METHOD("get_default_max_particles"), &UnifiedPhysicsProjectSettings::get_default_max_particles);

    ClassDB::bind_method(D_METHOD("set_enable_profiler", "enabled"), &UnifiedPhysicsProjectSettings::set_enable_profiler);
    ClassDB::bind_method(D_METHOD("get_enable_profiler"), &UnifiedPhysicsProjectSettings::get_enable_profiler);
    ClassDB::bind_method(D_METHOD("set_profiler_export_interval_frames", "frames"), &UnifiedPhysicsProjectSettings::set_profiler_export_interval_frames);
    ClassDB::bind_method(D_METHOD("get_profiler_export_interval_frames"), &UnifiedPhysicsProjectSettings::get_profiler_export_interval_frames);

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "primary_engine", PROPERTY_HINT_ENUM, "Newton,Genesis,Vienna,Wicked,Unified"), "set_primary_engine", "get_primary_engine");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_worker_threads", PROPERTY_HINT_RANGE, "1,16,1"), "set_num_worker_threads", "get_num_worker_threads");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_ccd"), "set_enable_ccd", "get_enable_ccd");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_iterations", PROPERTY_HINT_RANGE, "1,256,1"), "set_solver_iterations", "get_solver_iterations");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "erp", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_erp", "get_erp");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "erp2", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_erp2", "get_erp2");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cfm", PROPERTY_HINT_RANGE, "0,1,0.0001"), "set_cfm", "get_cfm");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sleep_linear_threshold"), "set_sleep_linear_threshold", "get_sleep_linear_threshold");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sleep_angular_threshold"), "set_sleep_angular_threshold", "get_sleep_angular_threshold");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "sleep_frames", PROPERTY_HINT_RANGE, "1,100,1"), "set_sleep_frames", "get_sleep_frames");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_adaptive_quality"), "set_enable_adaptive_quality", "get_enable_adaptive_quality");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adaptive_tier2_distance"), "set_adaptive_tier2_distance", "get_adaptive_tier2_distance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adaptive_tier1_distance"), "set_adaptive_tier1_distance", "get_adaptive_tier1_distance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adaptive_tier0_distance"), "set_adaptive_tier0_distance", "get_adaptive_tier0_distance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adaptive_max_physics_budget_ms"), "set_adaptive_max_physics_budget_ms", "get_adaptive_max_physics_budget_ms");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_vehicles"), "set_enable_vehicles", "get_enable_vehicles");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_vehicle_engine_max_force"), "set_default_vehicle_engine_max_force", "get_default_vehicle_engine_max_force");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_vehicle_engine_max_speed"), "set_default_vehicle_engine_max_speed", "get_default_vehicle_engine_max_speed");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_vehicle_suspension_stiffness"), "set_default_vehicle_suspension_stiffness", "get_default_vehicle_suspension_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_vehicle_suspension_damping"), "set_default_vehicle_suspension_damping", "get_default_vehicle_suspension_damping");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_cloth"), "set_enable_cloth", "get_enable_cloth");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_cloth_structural_stiffness"), "set_default_cloth_structural_stiffness", "get_default_cloth_structural_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_cloth_shear_stiffness"), "set_default_cloth_shear_stiffness", "get_default_cloth_shear_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_cloth_bending_stiffness"), "set_default_cloth_bending_stiffness", "get_default_cloth_bending_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_cloth_damping"), "set_default_cloth_damping", "get_default_cloth_damping");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_particles"), "set_enable_particles", "get_enable_particles");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_particle_radius"), "set_default_particle_radius", "get_default_particle_radius");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "default_max_particles", PROPERTY_HINT_RANGE, "0,100000,1"), "set_default_max_particles", "get_default_max_particles");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_profiler"), "set_enable_profiler", "get_enable_profiler");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "profiler_export_interval_frames", PROPERTY_HINT_RANGE, "10,600,10"), "set_profiler_export_interval_frames", "get_profiler_export_interval_frames");
}

} // namespace unified