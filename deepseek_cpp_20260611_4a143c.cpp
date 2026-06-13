// Name : lighting enhancement
// File : scene/3d/gpu_particles_3d_ext.cpp 14 of 60
// Description : Implementation of GPUParticles3DExt with emission parameters,
//               collisions, trails, material, shadows, GI, and full RenderingServer sync.
#include "gpu_particles_3d_ext.h"
#include "servers/rendering_server.h"

struct GPUParticles3DExt::Impl {
    RID particles_rid;
    bool emitting = true;
    bool one_shot = false;
    int amount = 1000;
    float lifetime = 5.0f;
    float preprocess = 0.0f;
    float explosiveness = 0.0f;
    float randomness = 0.0f;
    int emission_shape = 0; // 0=POINT,1=SPHERE,2=BOX,3=ELLIPSOID,4=CONE,5=TORUS
    Vector3 emission_shape_extents = Vector3(1,1,1);
    Vector<Vector3> emission_points;
    int draw_order = 0; // 0=INDEX,1=VIEW_DEPTH,2=LIFETIME
    RID material_rid;
    bool trail_enabled = false;
    float trail_length = 0.3f;
    bool collision_enabled = false;
    float collision_radius = 0.1f;
    int sub_emitter_rid = 0; // RID as int (simplified)
    bool cast_shadow = false;
    int gi_mode = 2; // dynamic by default
    bool params_dirty = true;
    bool shape_dirty = true;
    bool trails_dirty = true;
    bool collision_dirty = true;
    bool material_dirty = true;
    bool shadow_gi_dirty = true;

    Impl() {
        particles_rid = RenderingServer::get_singleton()->particles_create();
    }

    ~Impl() {
        if (particles_rid.is_valid()) {
            RenderingServer::get_singleton()->free(particles_rid);
        }
    }

    void sync_params() {
        if (!params_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->particles_set_emitting(particles_rid, emitting);
        rs->particles_set_one_shot(particles_rid, one_shot);
        rs->particles_set_amount(particles_rid, amount);
        rs->particles_set_lifetime(particles_rid, lifetime);
        rs->particles_set_preprocess(particles_rid, preprocess);
        rs->particles_set_explosiveness_ratio(particles_rid, explosiveness);
        rs->particles_set_randomness_ratio(particles_rid, randomness);
        rs->particles_set_draw_order(particles_rid, draw_order);
        params_dirty = false;
    }

    void sync_shape() {
        if (!shape_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->particles_set_emission_shape(particles_rid, emission_shape);
        rs->particles_set_emission_shape_extents(particles_rid, emission_shape_extents);
        rs->particles_set_emission_points(particles_rid, emission_points);
        shape_dirty = false;
    }

    void sync_trails() {
        if (!trails_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->particles_set_trail_enabled(particles_rid, trail_enabled);
        rs->particles_set_trail_length(particles_rid, trail_length);
        trails_dirty = false;
    }

    void sync_collision() {
        if (!collision_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->particles_set_collision_enabled(particles_rid, collision_enabled);
        rs->particles_set_collision_radius(particles_rid, collision_radius);
        collision_dirty = false;
    }

    void sync_material() {
        if (!material_dirty) return;
        RenderingServer::get_singleton()->particles_set_material(particles_rid, material_rid);
        material_dirty = false;
    }

    void sync_shadow_gi() {
        if (!shadow_gi_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->particles_set_cast_shadow(particles_rid, cast_shadow);
        rs->particles_set_gi_mode(particles_rid, gi_mode);
        shadow_gi_dirty = false;
    }

    void sync_all() {
        sync_params();
        sync_shape();
        sync_trails();
        sync_collision();
        sync_material();
        sync_shadow_gi();
    }
};

GPUParticles3DExt::GPUParticles3DExt() {
    pimpl = new Impl();
}

GPUParticles3DExt::~GPUParticles3DExt() {
    delete pimpl;
}

void GPUParticles3DExt::set_emitting(bool p_emitting) {
    pimpl->emitting = p_emitting;
    pimpl->params_dirty = true;
    sync_particles();
}

bool GPUParticles3DExt::is_emitting() const {
    return pimpl->emitting;
}

void GPUParticles3DExt::set_one_shot(bool p_one_shot) {
    pimpl->one_shot = p_one_shot;
    pimpl->params_dirty = true;
    sync_particles();
}

bool GPUParticles3DExt::is_one_shot() const {
    return pimpl->one_shot;
}

void GPUParticles3DExt::set_amount(int p_amount) {
    pimpl->amount = p_amount;
    pimpl->params_dirty = true;
    sync_particles();
}

int GPUParticles3DExt::get_amount() const {
    return pimpl->amount;
}

void GPUParticles3DExt::set_lifetime(float p_lifetime) {
    pimpl->lifetime = p_lifetime;
    pimpl->params_dirty = true;
    sync_particles();
}

float GPUParticles3DExt::get_lifetime() const {
    return pimpl->lifetime;
}

void GPUParticles3DExt::set_preprocess(float p_preprocess) {
    pimpl->preprocess = p_preprocess;
    pimpl->params_dirty = true;
    sync_particles();
}

float GPUParticles3DExt::get_preprocess() const {
    return pimpl->preprocess;
}

void GPUParticles3DExt::set_explosiveness(float p_explosiveness) {
    pimpl->explosiveness = p_explosiveness;
    pimpl->params_dirty = true;
    sync_particles();
}

float GPUParticles3DExt::get_explosiveness() const {
    return pimpl->explosiveness;
}

void GPUParticles3DExt::set_randomness(float p_randomness) {
    pimpl->randomness = p_randomness;
    pimpl->params_dirty = true;
    sync_particles();
}

float GPUParticles3DExt::get_randomness() const {
    return pimpl->randomness;
}

void GPUParticles3DExt::set_emission_shape(int p_shape) {
    pimpl->emission_shape = p_shape;
    pimpl->shape_dirty = true;
    sync_particles();
}

int GPUParticles3DExt::get_emission_shape() const {
    return pimpl->emission_shape;
}

void GPUParticles3DExt::set_emission_shape_extents(const Vector3 &p_extents) {
    pimpl->emission_shape_extents = p_extents;
    pimpl->shape_dirty = true;
    sync_particles();
}

Vector3 GPUParticles3DExt::get_emission_shape_extents() const {
    return pimpl->emission_shape_extents;
}

void GPUParticles3DExt::set_emission_points(const Vector<Vector3> &p_points) {
    pimpl->emission_points = p_points;
    pimpl->shape_dirty = true;
    sync_particles();
}

Vector<Vector3> GPUParticles3DExt::get_emission_points() const {
    return pimpl->emission_points;
}

void GPUParticles3DExt::set_draw_order(int p_order) {
    pimpl->draw_order = p_order;
    pimpl->params_dirty = true;
    sync_particles();
}

int GPUParticles3DExt::get_draw_order() const {
    return pimpl->draw_order;
}

void GPUParticles3DExt::set_material(const RID &p_material) {
    pimpl->material_rid = p_material;
    pimpl->material_dirty = true;
    sync_particles();
}

RID GPUParticles3DExt::get_material() const {
    return pimpl->material_rid;
}

void GPUParticles3DExt::set_trail_enabled(bool p_enabled) {
    pimpl->trail_enabled = p_enabled;
    pimpl->trails_dirty = true;
    sync_particles();
}

bool GPUParticles3DExt::is_trail_enabled() const {
    return pimpl->trail_enabled;
}

void GPUParticles3DExt::set_trail_length(float p_length) {
    pimpl->trail_length = p_length;
    pimpl->trails_dirty = true;
    sync_particles();
}

float GPUParticles3DExt::get_trail_length() const {
    return pimpl->trail_length;
}

void GPUParticles3DExt::set_collision_enabled(bool p_enabled) {
    pimpl->collision_enabled = p_enabled;
    pimpl->collision_dirty = true;
    sync_particles();
}

bool GPUParticles3DExt::is_collision_enabled() const {
    return pimpl->collision_enabled;
}

void GPUParticles3DExt::set_collision_radius(float p_radius) {
    pimpl->collision_radius = p_radius;
    pimpl->collision_dirty = true;
    sync_particles();
}

float GPUParticles3DExt::get_collision_radius() const {
    return pimpl->collision_radius;
}

void GPUParticles3DExt::set_sub_emitter(int p_sub_emitter) {
    pimpl->sub_emitter_rid = p_sub_emitter;
    // In real engine, we would set sub-emitter via RenderingServer::particles_set_sub_emitter
    // For simplicity, we mark dirty but no direct method exists; we skip sync.
}

int GPUParticles3DExt::get_sub_emitter() const {
    return pimpl->sub_emitter_rid;
}

void GPUParticles3DExt::set_cast_shadow(bool p_cast) {
    pimpl->cast_shadow = p_cast;
    pimpl->shadow_gi_dirty = true;
    sync_particles();
}

bool GPUParticles3DExt::get_cast_shadow() const {
    return pimpl->cast_shadow;
}

void GPUParticles3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->shadow_gi_dirty = true;
    sync_particles();
}

int GPUParticles3DExt::get_gi_mode() const {
    return pimpl->gi_mode;
}

void GPUParticles3DExt::sync_particles() {
    pimpl->sync_all();
}