// cpu_particles_3d.cpp
#include "cpu_particles_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <numeric>

namespace lighting {

// ============================================================================
// Helper random number generation (thread‑local)
// ============================================================================
static thread_local std::mt19937 rng(std::random_device{}());
static thread_local std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
static inline float rand_float() { return dist01(rng); }

// ============================================================================
// CPUParticles3D implementation
// ============================================================================
struct CPUParticles3D::Impl {
    // Parameters
    bool emitting = true;
    bool one_shot = false;
    int amount = 1000;
    float lifetime = 5.0f;
    float preprocess = 0.0f;

    ParticleEmissionShape emission_shape = ParticleEmissionShape::POINT;
    double shape_extents[3] = {1.0, 1.0, 1.0};
    double direction[3] = {0.0, 1.0, 0.0};
    float spread = 0.0f;
    float flatness = 0.0f;

    double gravity[3] = {0.0, -9.8, 0.0};
    float linear_damping = 0.0f;
    float angular_damping = 0.0f;
    float initial_vel_min = 1.0f;
    float initial_vel_max = 1.0f;
    float initial_ang_vel_min = 0.0f;
    float initial_ang_vel_max = 0.0f;

    float scale_min = 0.05f;
    float scale_max = 0.05f;
    float color_rgba[4] = {1.0f,1.0f,1.0f,1.0f};
    int color_ramp_tex = -1;
    int size_ramp_tex = -1;
    char material_path[256] = {0};

    bool trail_enabled = false;
    float trail_length = 0.3f;
    float trail_section_length = 0.1f;

    bool collision_enabled = false;
    float collision_radius = 0.05f;
    uint32_t collision_mask = 0xFFFFFFFF;
    float bounce = 0.5f;
    float friction = 0.2f;

    CPUParticles3D* sub_emitter = nullptr;
    float sub_emitter_at_end = 0.0f;

    bool cast_shadow = true;
    bool receive_shadow = false;
    int gi_mode = 2;          // dynamic
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    ParticleDrawOrder draw_order = ParticleDrawOrder::INDICES;
    bool sort_using_aabb = false;

    // Internal state
    std::vector<CPUParticle> particles;
    std::vector<uint32_t> particle_order;  // for view‑depth sorting
    double time_accum = 0.0;
    uint32_t next_seed = 123456789;
    bool particles_dirty = true;
    int active_count = 0;
    double bounds_min[3] = {-10,-10,-10};
    double bounds_max[3] = {10,10,10};
    bool bounds_dirty = true;

    ~Impl() {
        // cleanup
    }

    void init_particles();
    void emit_particle(double dt);
    void update_bounds();
};

CPUParticles3D::CPUParticles3D() : pimpl(std::make_unique<Impl>()) {}
CPUParticles3D::~CPUParticles3D() = default;

void CPUParticles3D::set_emitting(bool emit) { pimpl->emitting = emit; }
bool CPUParticles3D::is_emitting() const { return pimpl->emitting; }
void CPUParticles3D::restart() {
    pimpl->particles_dirty = true;
    pimpl->time_accum = 0.0;
}
void CPUParticles3D::set_one_shot(bool one_shot) { pimpl->one_shot = one_shot; }
bool CPUParticles3D::is_one_shot() const { return pimpl->one_shot; }
void CPUParticles3D::set_amount(int amount) { pimpl->amount = amount; pimpl->particles_dirty = true; }
int CPUParticles3D::get_amount() const { return pimpl->amount; }
void CPUParticles3D::set_lifetime(float seconds) { pimpl->lifetime = seconds; pimpl->particles_dirty = true; }
float CPUParticles3D::get_lifetime() const { return pimpl->lifetime; }
void CPUParticles3D::set_preprocess(float seconds) { pimpl->preprocess = seconds; pimpl->particles_dirty = true; }
float CPUParticles3D::get_preprocess() const { return pimpl->preprocess; }

void CPUParticles3D::set_emission_shape(ParticleEmissionShape shape) { pimpl->emission_shape = shape; }
ParticleEmissionShape CPUParticles3D::get_emission_shape() const { return pimpl->emission_shape; }
void CPUParticles3D::set_emission_shape_extents(const double* extents) { memcpy(pimpl->shape_extents, extents, 3*sizeof(double)); }
void CPUParticles3D::get_emission_shape_extents(double* out_extents) const { memcpy(out_extents, pimpl->shape_extents, 3*sizeof(double)); }
void CPUParticles3D::set_direction(const double* dir) { memcpy(pimpl->direction, dir, 3*sizeof(double)); }
void CPUParticles3D::get_direction(double* out_dir) const { memcpy(out_dir, pimpl->direction, 3*sizeof(double)); }
void CPUParticles3D::set_spread(float spread) { pimpl->spread = spread; }
float CPUParticles3D::get_spread() const { return pimpl->spread; }
void CPUParticles3D::set_flatness(float flatness) { pimpl->flatness = flatness; }
float CPUParticles3D::get_flatness() const { return pimpl->flatness; }

void CPUParticles3D::set_gravity(const double* gravity) { memcpy(pimpl->gravity, gravity, 3*sizeof(double)); }
void CPUParticles3D::get_gravity(double* out_gravity) const { memcpy(out_gravity, pimpl->gravity, 3*sizeof(double)); }
void CPUParticles3D::set_linear_damping(float damping) { pimpl->linear_damping = damping; }
float CPUParticles3D::get_linear_damping() const { return pimpl->linear_damping; }
void CPUParticles3D::set_angular_damping(float damping) { pimpl->angular_damping = damping; }
float CPUParticles3D::get_angular_damping() const { return pimpl->angular_damping; }
void CPUParticles3D::set_initial_velocity_min(float vel) { pimpl->initial_vel_min = vel; }
float CPUParticles3D::get_initial_velocity_min() const { return pimpl->initial_vel_min; }
void CPUParticles3D::set_initial_velocity_max(float vel) { pimpl->initial_vel_max = vel; }
float CPUParticles3D::get_initial_velocity_max() const { return pimpl->initial_vel_max; }
void CPUParticles3D::set_initial_angular_velocity_min(float vel) { pimpl->initial_ang_vel_min = vel; }
float CPUParticles3D::get_initial_angular_velocity_min() const { return pimpl->initial_ang_vel_min; }
void CPUParticles3D::set_initial_angular_velocity_max(float vel) { pimpl->initial_ang_vel_max = vel; }
float CPUParticles3D::get_initial_angular_velocity_max() const { return pimpl->initial_ang_vel_max; }

void CPUParticles3D::set_scale_min(float scale) { pimpl->scale_min = scale; }
float CPUParticles3D::get_scale_min() const { return pimpl->scale_min; }
void CPUParticles3D::set_scale_max(float scale) { pimpl->scale_max = scale; }
float CPUParticles3D::get_scale_max() const { return pimpl->scale_max; }
void CPUParticles3D::set_color(const float* rgba) { memcpy(pimpl->color_rgba, rgba, 4*sizeof(float)); }
void CPUParticles3D::get_color(float* out_rgba) const { memcpy(out_rgba, pimpl->color_rgba, 4*sizeof(float)); }
void CPUParticles3D::set_color_ramp(int texture_id) { pimpl->color_ramp_tex = texture_id; }
void CPUParticles3D::set_size_ramp(int texture_id) { pimpl->size_ramp_tex = texture_id; }
void CPUParticles3D::set_material(const char* material_path) { strncpy(pimpl->material_path, material_path, 255); pimpl->material_path[255]=0; }
const char* CPUParticles3D::get_material() const { return pimpl->material_path; }

void CPUParticles3D::set_trail_enabled(bool enabled) { pimpl->trail_enabled = enabled; }
bool CPUParticles3D::is_trail_enabled() const { return pimpl->trail_enabled; }
void CPUParticles3D::set_trail_length(float seconds) { pimpl->trail_length = seconds; }
float CPUParticles3D::get_trail_length() const { return pimpl->trail_length; }
void CPUParticles3D::set_trail_section_length(float length) { pimpl->trail_section_length = length; }
float CPUParticles3D::get_trail_section_length() const { return pimpl->trail_section_length; }

void CPUParticles3D::set_collision_enabled(bool enabled) { pimpl->collision_enabled = enabled; }
bool CPUParticles3D::is_collision_enabled() const { return pimpl->collision_enabled; }
void CPUParticles3D::set_collision_radius(float radius) { pimpl->collision_radius = radius; }
float CPUParticles3D::get_collision_radius() const { return pimpl->collision_radius; }
void CPUParticles3D::set_collision_mask(uint32_t mask) { pimpl->collision_mask = mask; }
uint32_t CPUParticles3D::get_collision_mask() const { return pimpl->collision_mask; }
void CPUParticles3D::set_bounce(float bounce) { pimpl->bounce = bounce; }
float CPUParticles3D::get_bounce() const { return pimpl->bounce; }
void CPUParticles3D::set_friction(float friction) { pimpl->friction = friction; }
float CPUParticles3D::get_friction() const { return pimpl->friction; }

void CPUParticles3D::set_sub_emitter(CPUParticles3D* emitter, float at_end) {
    pimpl->sub_emitter = emitter;
    pimpl->sub_emitter_at_end = at_end;
}
void CPUParticles3D::clear_sub_emitter() { pimpl->sub_emitter = nullptr; }

void CPUParticles3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void CPUParticles3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; GeometryInstance3D::set_receive_shadow(receive); }
void CPUParticles3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void CPUParticles3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; GeometryInstance3D::set_gi_contribution(amount); }
void CPUParticles3D::set_emissive(const float* color, float intensity) { memcpy(pimpl->emissive_color, color, 3*sizeof(float)); pimpl->emissive_intensity = intensity; }
void CPUParticles3D::get_emissive(float* out_color, float& out_intensity) const { memcpy(out_color, pimpl->emissive_color, 3*sizeof(float)); out_intensity = pimpl->emissive_intensity; }

void CPUParticles3D::set_draw_order(ParticleDrawOrder order) { pimpl->draw_order = order; }
ParticleDrawOrder CPUParticles3D::get_draw_order() const { return pimpl->draw_order; }
void CPUParticles3D::set_sorting_use_aabb_center(bool use) { pimpl->sort_using_aabb = use; }
bool CPUParticles3D::is_sorting_use_aabb_center() const { return pimpl->sort_using_aabb; }

void CPUParticles3D::Impl::init_particles() {
    particles.resize(amount);
    particle_order.resize(amount);
    std::iota(particle_order.begin(), particle_order.end(), 0);
    active_count = 0;
    for (int i = 0; i < amount; ++i) {
        particles[i].active = false;
        particles[i].lifetime = 0.0f;
        particles[i].seed = next_seed++;
    }
    // simulate preprocess
    if (preprocess > 0.0f) {
        double dt = 0.016f;
        int steps = (int)(preprocess / dt);
        for (int step = 0; step < steps; ++step) {
            // emit particles at random times (simplified)
            if (emitting) {
                emit_particle(dt);
            }
            simulate_particles(dt);
        }
    }
    particles_dirty = false;
    update_bounds();
}

void CPUParticles3D::Impl::emit_particle(double dt) {
    // emission rate based on dt and lifetime
    float emission_rate = (float)amount / lifetime;
    float expected = emission_rate * dt;
    int to_emit = (int)expected;
    if (rand_float() < expected - to_emit) ++to_emit;
    if (to_emit == 0) return;

    for (int e = 0; e < to_emit && active_count < amount; ++e) {
        // find first inactive particle
        int idx = -1;
        for (int i = 0; i < amount; ++i) {
            if (!particles[i].active) { idx = i; break; }
        }
        if (idx == -1) break;

        CPUParticle& p = particles[idx];
        p.active = true;
        p.lifetime = 0.0f;
        p.initial_lifetime = lifetime;
        double dir[3] = {direction[0], direction[1], direction[2]};
        // apply spread and flatness
        float r1 = rand_float();
        float r2 = rand_float();
        float spread_angle = spread * r1;
        float azimuth = 2.0f * M_PI * r2;
        // modify direction
        double x = sin(spread_angle) * cos(azimuth);
        double y = sin(spread_angle) * sin(azimuth);
        double z = cos(spread_angle);
        // flattening
        y *= (1.0 - flatness);
        double len = sqrt(x*x + y*y + z*z);
        if (len > 1e-6) { x /= len; y /= len; z /= len; }
        // rotate base direction to align with emission direction
        // ... (simplified: use spherical interpolation)
        p.velocity[0] = (x + direction[0]) * (initial_vel_min + rand_float() * (initial_vel_max - initial_vel_min));
        p.velocity[1] = (y + direction[1]) * (initial_vel_min + rand_float() * (initial_vel_max - initial_vel_min));
        p.velocity[2] = (z + direction[2]) * (initial_vel_min + rand_float() * (initial_vel_max - initial_vel_min));

        p.angular_velocity = initial_ang_vel_min + rand_float() * (initial_ang_vel_max - initial_ang_vel_min);
        p.size = scale_min + rand_float() * (scale_max - scale_min);
        p.rotation = 0.0f;
        // position within emission shape
        switch (emission_shape) {
            case ParticleEmissionShape::POINT:
                p.position[0] = 0; p.position[1] = 0; p.position[2] = 0; break;
            case ParticleEmissionShape::SPHERE: {
                double theta = acos(2.0 * rand_float() - 1.0);
                double phi = 2.0 * M_PI * rand_float();
                double r = shape_extents[0] * cbrt(rand_float()); // volume distribution
                p.position[0] = r * sin(theta) * cos(phi);
                p.position[1] = r * sin(theta) * sin(phi);
                p.position[2] = r * cos(theta);
                break;
            }
            case ParticleEmissionShape::BOX:
                p.position[0] = (rand_float() * 2.0 - 1.0) * shape_extents[0];
                p.position[1] = (rand_float() * 2.0 - 1.0) * shape_extents[1];
                p.position[2] = (rand_float() * 2.0 - 1.0) * shape_extents[2];
                break;
            default: // point
                p.position[0]=p.position[1]=p.position[2]=0;
        }
        // color and material overrides via ramp would be sampled later
        memcpy(p.color, color_rgba, 4*sizeof(double));
        ++active_count;
    }
}

void CPUParticles3D::Impl::simulate_particles(double dt) {
    double dt_clamped = std::min(dt, 0.033);
    for (int i = 0; i < amount; ++i) {
        CPUParticle& p = particles[i];
        if (!p.active) continue;

        // update lifetime
        p.lifetime += dt_clamped;
        if (p.lifetime >= p.initial_lifetime) {
            p.active = false;
            --active_count;
            // sub‑emitter on death
            if (sub_emitter && sub_emitter_at_end > 0.0f) {
                // spawn sub‑emitter at particle position
                sub_emitter->set_emitting(true);
                // quickly emit few particles at death position (simplified)
            }
            continue;
        }

        // apply gravity
        p.velocity[0] += gravity[0] * dt_clamped;
        p.velocity[1] += gravity[1] * dt_clamped;
        p.velocity[2] += gravity[2] * dt_clamped;
        // damping
        p.velocity[0] *= (1.0 - linear_damping * dt_clamped);
        p.velocity[1] *= (1.0 - linear_damping * dt_clamped);
        p.velocity[2] *= (1.0 - linear_damping * dt_clamped);
        p.angular_velocity *= (1.0 - angular_damping * dt_clamped);
        // integrate position
        p.position[0] += p.velocity[0] * dt_clamped;
        p.position[1] += p.velocity[1] * dt_clamped;
        p.position[2] += p.velocity[2] * dt_clamped;
        p.rotation += p.angular_velocity * dt_clamped;

        // collision (simplified sphere with world ground)
        if (collision_enabled && p.position[1] - collision_radius < 0.0) {
            p.position[1] = collision_radius;
            p.velocity[1] = -p.velocity[1] * bounce;
            p.velocity[0] *= (1.0 - friction);
            p.velocity[2] *= (1.0 - friction);
        }
        // update color over lifetime (ramp not implemented)
        float life = p.lifetime / p.initial_lifetime;
        // placeholder
    }
    update_bounds();
}

void CPUParticles3D::Impl::update_bounds() {
    if (active_count == 0) return;
    bounds_min[0] = bounds_min[1] = bounds_min[2] = 1e30;
    bounds_max[0] = bounds_max[1] = bounds_max[2] = -1e30;
    for (int i = 0; i < amount; ++i) {
        if (!particles[i].active) continue;
        for (int j = 0; j < 3; ++j) {
            bounds_min[j] = std::min(bounds_min[j], particles[i].position[j] - particles[i].size);
            bounds_max[j] = std::max(bounds_max[j], particles[i].position[j] + particles[i].size);
        }
    }
}

void CPUParticles3D::update_particles(double delta_time) {
    if (pimpl->particles_dirty) {
        pimpl->init_particles();
    }
    pimpl->time_accum += delta_time;
    const double MAX_DT = 0.033;
    while (pimpl->time_accum > 0.0) {
        double step = std::min(MAX_DT, pimpl->time_accum);
        if (pimpl->emitting) {
            pimpl->emit_particle(step);
        }
        pimpl->simulate_particles(step);
        pimpl->time_accum -= step;
    }
}

void CPUParticles3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    update_particles(delta);

    // Update render server with particle positions, colors, sizes, rotations
    // For each active particle, send data.
    // Also update bounding box for frustum culling
    double aabb_min[3] = {pimpl->bounds_min[0], pimpl->bounds_min[1], pimpl->bounds_min[2]};
    double aabb_max[3] = {pimpl->bounds_max[0], pimpl->bounds_max[1], pimpl->bounds_max[2]};
    set_aabb(aabb_min, aabb_max);
    double dx = aabb_max[0]-aabb_min[0];
    double dy = aabb_max[1]-aabb_min[1];
    double dz = aabb_max[2]-aabb_min[2];
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);

    // GI contribution: if emissive intensity > 0, inject as light source
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // For each active particle, add to light propagation volume
        // (simplified: average over particles)
    }
}

} // namespace lighting