// File 352: modules/wicked/src/world/wicked_world.cpp (complete)
// Full implementation of WickedWorld including body/joint/material/vehicle
// management, as well as the step pipeline already defined.

#include "wicked_world.h"
#include "../bodies/wicked_body.h"
#include "../joints/wicked_joint.h"
#include "../materials/wicked_material.h"
#include "../solver/wicked_solver.h"
#include "../solver/wicked_island.h"
#include "../vehicles/wicked_raycast_vehicle.h"

#include "../../../gaia/src/collision_detector/narrow_phase.h"
#include "../../../gaia/src/collision_detector/contact.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace wicked {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
WickedWorld::WickedWorld() :
    world_time(0.0),
    solver_iterations(DEFAULT_SOLVER_ITERATIONS),
    solver_method(SolverMethod::SEQUENTIAL_IMPULSES),
    erp(DEFAULT_ERP),
    erp2(DEFAULT_ERP2),
    cfm(DEFAULT_TAU),
    sleep_linear_threshold(DEFAULT_SLEEP_LINEAR),
    sleep_angular_threshold(DEFAULT_SLEEP_ANGULAR),
    sleep_frames(DEFAULT_SLEEP_FRAMES) {
    solver.instantiate();
    island_manager.instantiate();
}

WickedWorld::~WickedWorld() {
    bodies.clear();
    joints.clear();
    materials.clear();
    vehicles.clear();
}

// ---------------------------------------------------------------------------
// Global settings
// ---------------------------------------------------------------------------
void WickedWorld::set_gravity(const vec3 &p_gravity) { gravity = p_gravity; }
void WickedWorld::set_solver_iterations(int p_iter) { solver_iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS); }
void WickedWorld::set_solver_method(SolverMethod p_method) { solver_method = p_method; }
void WickedWorld::set_sleep_linear_threshold(real_t p_thres) { sleep_linear_threshold = MAX(p_thres, 0.0); }
void WickedWorld::set_sleep_angular_threshold(real_t p_thres) { sleep_angular_threshold = MAX(p_thres, 0.0); }
void WickedWorld::set_sleep_frames(int p_frames) { sleep_frames = MAX(p_frames, 1); }

// ---------------------------------------------------------------------------
// Body management
// ---------------------------------------------------------------------------
body_id WickedWorld::create_body(const Ref<WickedBody> &p_body) {
    ERR_FAIL_COND_V(p_body.is_null(), 0);
    body_id id = next_body_id++;
    bodies[id] = p_body;
    gaia_broad_phase.add_object(id, p_body->get_aabb());
    return id;
}

void WickedWorld::destroy_body(body_id p_id) {
    bodies.erase(p_id);
    gaia_broad_phase.remove_object(p_id);
}

Ref<WickedBody> WickedWorld::get_body(body_id p_id) const {
    HashMap<body_id, Ref<WickedBody>>::ConstIterator it = bodies.find(p_id);
    return it ? it->value : Ref<WickedBody>();
}

int WickedWorld::get_body_count() const {
    return bodies.size();
}

void WickedWorld::add_body_with_id(body_id p_id, const Ref<WickedBody> &p_body) {
    bodies[p_id] = p_body;
    gaia_broad_phase.add_object(p_id, p_body->get_aabb());
}

// ---------------------------------------------------------------------------
// Joint management
// ---------------------------------------------------------------------------
joint_id WickedWorld::create_joint(const Ref<WickedJoint> &p_joint) {
    ERR_FAIL_COND_V(p_joint.is_null(), 0);
    joint_id id = next_joint_id++;
    joints[id] = p_joint;
    return id;
}

void WickedWorld::destroy_joint(joint_id p_id) {
    joints.erase(p_id);
}

Ref<WickedJoint> WickedWorld::get_joint(joint_id p_id) const {
    HashMap<joint_id, Ref<WickedJoint>>::ConstIterator it = joints.find(p_id);
    return it ? it->value : Ref<WickedJoint>();
}

// ---------------------------------------------------------------------------
// Material management
// ---------------------------------------------------------------------------
material_id WickedWorld::create_material(const Ref<WickedMaterial> &p_material) {
    ERR_FAIL_COND_V(p_material.is_null(), 0);
    material_id id = next_material_id++;
    materials[id] = p_material;
    return id;
}

void WickedWorld::destroy_material(material_id p_id) {
    materials.erase(p_id);
}

Ref<WickedMaterial> WickedWorld::get_material(material_id p_id) const {
    HashMap<material_id, Ref<WickedMaterial>>::ConstIterator it = materials.find(p_id);
    return it ? it->value : Ref<WickedMaterial>();
}

// ---------------------------------------------------------------------------
// Vehicle management
// ---------------------------------------------------------------------------
vehicle_id WickedWorld::create_vehicle(const Ref<WickedRaycastVehicle> &p_vehicle) {
    ERR_FAIL_COND_V(p_vehicle.is_null(), 0);
    vehicle_id id = next_vehicle_id++;
    vehicles[id] = p_vehicle;
    return id;
}

void WickedWorld::destroy_vehicle(vehicle_id p_id) {
    vehicles.erase(p_id);
}

Ref<WickedRaycastVehicle> WickedWorld::get_vehicle(vehicle_id p_id) const {
    HashMap<vehicle_id, Ref<WickedRaycastVehicle>>::ConstIterator it = vehicles.find(p_id);
    return it ? it->value : Ref<WickedRaycastVehicle>();
}

// ---------------------------------------------------------------------------
// Terrain / filtering (not fully implemented)
// ---------------------------------------------------------------------------
void WickedWorld::set_body_layer(body_id p_body, uint32_t p_layer) {}
uint32_t WickedWorld::get_body_layer(body_id p_body) const { return 1; }
void WickedWorld::set_body_mask(body_id p_body, uint32_t p_mask) {}
uint32_t WickedWorld::get_body_mask(body_id p_body) const { return 0xFFFFFFFF; }
void WickedWorld::set_ccd_enabled(body_id p_body, bool p_enabled) {}
bool WickedWorld::is_ccd_enabled(body_id p_body) const { return false; }

} // namespace wicked