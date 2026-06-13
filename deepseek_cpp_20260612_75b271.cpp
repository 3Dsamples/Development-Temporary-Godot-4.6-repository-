// Name : lighting enhancement
// File : scene/3d/cpu_particles_3d_ext.cpp 16 of 60
// Description : Full CPU particle simulation with emission shapes, Euler physics,
//               sphere/box/cone/torus emission, collision with plane, color ramp,
//               dynamic quad mesh generation, and RenderingServer integration.
#include "cpu_particles_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/random_pcg.h"
#include "core/math/transform_3d.h"
#include <cmath>
#include <vector>
#include <cstring>

// ------------------------------------------------------------------------
// Particle data structure
// ------------------------------------------------------------------------
struct CPUParticleData {
    Vector3 position;
    Vector3 velocity;
    Vector3 start_position;
    float lifetime;
    float initial_lifetime;
    float size;
    float rotation;
    float angular_velocity;
    Color color;
    bool active;
};

struct CPUParticles3DExt::Impl {
    // Rendering server handles
    RID particles_rid;
    RID mesh_rid;          // quad mesh for particles
    RID material_rid;
    RID color_ramp_rid;

    // Parameters
    bool emitting = true;
    bool one_shot = false;
    int amount = 1000;
    float lifetime = 5.0f;
    float preprocess = 0.0f;
    float explosiveness = 0.0f;
    float randomness = 0.0f;
    int emission_shape = 0;          // 0=POINT,1=SPHERE,2=BOX,3=ELLIPSOID,4=CONE,5=TORUS
    Vector3 emission_shape_extents = Vector3(1,1,1);
    Vector<Vector3> emission_points;
    int draw_order = 0;
    Color color = Color(1,1,1,1);
    bool trail_enabled = false;
    float trail_length = 0.3f;
    bool collision_enabled = false;
    float collision_radius = 0.1f;
    int sub_emitter_rid = 0;
    bool cast_shadow = false;
    int gi_mode = 2;

    // Physics
    Vector3 gravity = Vector3(0, -9.8f, 0);
    float linear_damping = 0.0f;
    float angular_damping = 0.0f;
    float initial_velocity_min = 1.0f;
    float initial_velocity_max = 1.0f;
    float initial_angular_velocity_min = 0.0f;
    float initial_angular_velocity_max = 0.0f;
    float scale_min = 0.05f;
    float scale_max = 0.05f;
    float bounce = 0.5f;
    float friction = 0.2f;

    // Internal state
    std::vector<CPUParticleData> particles;
    std::vector<float> vertex_buffer; // position+color+uv (interleaved)
    float time_accum = 0.0f;
    RandomPCG rng;
    bool mesh_dirty = true;

    Impl() {
        particles_rid = RenderingServer::get_singleton()->particles_create();
        mesh_rid = _create_quad_mesh();
        material_rid = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(material_rid, "vertex_color", true);
        RenderingServer::get_singleton()->particles_set_mesh(particles_rid, mesh_rid);
        RenderingServer::get_singleton()->particles_set_material(particles_rid, material_rid);
        rng.randomize();
    }

    ~Impl() {
        if (particles_rid.is_valid()) RenderingServer::get_singleton()->free(particles_rid);
        if (mesh_rid.is_valid()) RenderingServer::get_singleton()->free(mesh_rid);
        if (material_rid.is_valid()) RenderingServer::get_singleton()->free(material_rid);
    }

    // Create a unit quad mesh (centered, facing Z)
    RID _create_quad_mesh() {
        RID mesh = RenderingServer::get_singleton()->mesh_create();
        Vector<Vector3> vertices;
        vertices.push_back(Vector3(-0.5, -0.5, 0));
        vertices.push_back(Vector3( 0.5, -0.5, 0));
        vertices.push_back(Vector3( 0.5,  0.5, 0));
        vertices.push_back(Vector3(-0.5,  0.5, 0));
        Vector<Vector2> uvs;
        uvs.push_back(Vector2(0,0));
        uvs.push_back(Vector2(1,0));
        uvs.push_back(Vector2(1,1));
        uvs.push_back(Vector2(0,1));
        Vector<int> indices;
        indices.push_back(0); indices.push_back(1); indices.push_back(2);
        indices.push_back(0); indices.push_back(2); indices.push_back(3);
        RenderingServer::get_singleton()->mesh_add_surface(mesh, RS::PRIMITIVE_TRIANGLES, vertices, indices, uvs, Vector<Vector3>());
        return mesh;
    }

    void init_particles() {
        particles.resize(amount);
        for (int i = 0; i < amount; ++i) {
            particles[i].active = false;
            particles[i].lifetime = 0.0f;
            particles[i].initial_lifetime = lifetime;
            particles[i].size = scale_min + rng.random() * (scale_max - scale_min);
            particles[i].rotation = 0.0f;
            particles[i].angular_velocity = 0.0f;
            particles[i].color = color;
        }
        if (preprocess > 0.0f) {
            double dt = 0.016;
            int steps = int(preprocess / dt);
            for (int step = 0; step < steps; ++step) {
                if (emitting) emit_particles(dt);
                simulate(dt);
            }
        }
        mesh_dirty = true;
    }

    void emit_particles(float dt) {
        if (!emitting) return;
        float rate = amount / lifetime;
        float expected = rate * dt;
        int to_emit = int(expected);
        if (rng.random() < (expected - to_emit)) ++to_emit;
        to_emit = MIN(to_emit, amount);

        for (int e = 0; e < to_emit; ++e) {
            int idx = -1;
            for (int i = 0; i < amount; ++i) {
                if (!particles[i].active) {
                    idx = i;
                    break;
                }
            }
            if (idx == -1) break;

            CPUParticleData &p = particles[idx];
            p.active = true;
            p.lifetime = 0.0f;
            p.initial_lifetime = lifetime;
            p.position = _get_emission_position();
            p.start_position = p.position;
            p.velocity = _get_initial_velocity();
            p.angular_velocity = _get_initial_angular_velocity();
            p.size = scale_min + rng.random() * (scale_max - scale_min);
            p.rotation = 0.0f;
            p.color = color;
        }
    }

    Vector3 _get_initial_velocity() {
        float vel = initial_velocity_min + rng.random() * (initial_velocity_max - initial_velocity_min);
        // Spherical uniform direction
        float theta = 2.0f * Math_PI * rng.random();
        float phi = acos(1.0f - 2.0f * rng.random());
        float x = sin(phi) * cos(theta);
        float y = sin(phi) * sin(theta);
        float z = cos(phi);
        return Vector3(x, y, z) * vel;
    }

    float _get_initial_angular_velocity() {
        return initial_angular_velocity_min + rng.random() * (initial_angular_velocity_max - initial_angular_velocity_min);
    }

    Vector3 _get_emission_position() {
        Vector3 pos(0,0,0);
        float r1 = rng.random();
        float r2 = rng.random();
        switch (emission_shape) {
            case 0: // POINT
                break;
            case 1: // SPHERE
                {
                    float r = emission_shape_extents.x * cbrt(r1);
                    float theta = 2.0f * Math_PI * r2;
                    float phi = acos(2.0f * rng.random() - 1.0f);
                    pos.x = r * sin(phi) * cos(theta);
                    pos.y = r * sin(phi) * sin(theta);
                    pos.z = r * cos(phi);
                }
                break;
            case 2: // BOX
                pos.x = (r1 * 2.0f - 1.0f) * emission_shape_extents.x;
                pos.y = (r2 * 2.0f - 1.0f) * emission_shape_extents.y;
                pos.z = (rng.random() * 2.0f - 1.0f) * emission_shape_extents.z;
                break;
            case 3: // ELLIPSOID
                {
                    float r = cbrt(r1);
                    float theta = 2.0f * Math_PI * r2;
                    float phi = acos(2.0f * rng.random() - 1.0f);
                    pos.x = emission_shape_extents.x * r * sin(phi) * cos(theta);
                    pos.y = emission_shape_extents.y * r * sin(phi) * sin(theta);
                    pos.z = emission_shape_extents.z * r * cos(phi);
                }
                break;
            case 4: // CONE (pointing up Y)
                {
                    float radius = emission_shape_extents.x;
                    float height = emission_shape_extents.y;
                    float r = radius * sqrt(r1);
                    float theta = 2.0f * Math_PI * r2;
                    float y = height * rng.random();
                    pos.x = r * cos(theta);
                    pos.y = y - height * 0.5f;
                    pos.z = r * sin(theta);
                }
                break;
            case 5: // TORUS (XZ plane)
                {
                    float major = emission_shape_extents.x;
                    float minor = emission_shape_extents.y;
                    float u = 2.0f * Math_PI * r1;
                    float v = 2.0f * Math_PI * r2;
                    float r = major + minor * cos(v);
                    pos.x = r * cos(u);
                    pos.y = minor * sin(v);
                    pos.z = r * sin(u);
                }
                break;
        }
        return pos;
    }

    void simulate(float dt) {
        dt = MIN(dt, 0.033f);
        for (int i = 0; i < amount; ++i) {
            CPUParticleData &p = particles[i];
            if (!p.active) continue;

            p.lifetime += dt;
            if (p.lifetime >= p.initial_lifetime) {
                p.active = false;
                continue;
            }

            // Euler integration
            p.velocity += gravity * dt;
            p.velocity *= (1.0f - linear_damping * dt);
            p.position += p.velocity * dt;

            p.angular_velocity *= (1.0f - angular_damping * dt);
            p.rotation += p.angular_velocity * dt;

            // Simple collision with ground plane Y=0
            if (collision_enabled) {
                float half_size = p.size * 0.5f;
                if (p.position.y - half_size < 0.0f) {
                    p.position.y = half_size;
                    p.velocity.y = -p.velocity.y * bounce;
                    p.velocity.x *= (1.0f - friction);
                    p.velocity.z *= (1.0f - friction);
                }
            }

            // Color over lifetime (using ramp texture if provided)
            float t = p.lifetime / p.initial_lifetime;
            if (color_ramp_rid.is_valid()) {
                // Simulate color ramp: sample texture (simplified: use a 1D gradient)
                float r = 1.0f - t; // fade out
                float g = 1.0f - t * 0.5f;
                float b = 1.0f;
                p.color = Color(r, g, b, 1.0f - t);
            } else {
                p.color = color;
            }
        }
        mesh_dirty = true;
    }

    void update_vertex_buffer() {
        // Count active particles
        int active = 0;
        for (int i = 0; i < amount; ++i) if (particles[i].active) ++active;
        if (active == 0) return;

        // Each particle = 4 vertices, each vertex = 3 pos + 4 color + 2 uv = 9 floats
        vertex_buffer.resize(active * 4 * 9);
        int idx = 0;
        for (int i = 0; i < amount; ++i) {
            const CPUParticleData &p = particles[i];
            if (!p.active) continue;

            // Quad corners (local space)
            float s = p.size;
            Vector3 corners[4] = {
                Vector3(-s, -s, 0),
                Vector3( s, -s, 0),
                Vector3( s,  s, 0),
                Vector3(-s,  s, 0)
            };
            // Apply rotation around Z
            float c = cos(p.rotation);
            float sc = sin(p.rotation);
            for (int j = 0; j < 4; ++j) {
                Vector3 v = corners[j];
                float x = v.x * c - v.y * sc;
                float y = v.x * sc + v.y * c;
                Vector3 world = p.position + Vector3(x, y, 0);
                // Write position
                vertex_buffer[idx++] = world.x;
                vertex_buffer[idx++] = world.y;
                vertex_buffer[idx++] = world.z;
                // Write color
                vertex_buffer[idx++] = p.color.r;
                vertex_buffer[idx++] = p.color.g;
                vertex_buffer[idx++] = p.color.b;
                vertex_buffer[idx++] = p.color.a;
                // Write UV
                float u = (j == 0 || j == 3) ? 0.0f : 1.0f;
                float vv = (j == 0 || j == 1) ? 0.0f : 1.0f;
                vertex_buffer[idx++] = u;
                vertex_buffer[idx++] = vv;
            }
        }

        // Rebuild mesh surface with dynamic vertex data
        RenderingServer::get_singleton()->mesh_clear(mesh_rid);
        Vector<Vector3> verts;
        Vector<Vector2> uvs;
        Vector<int> indices;
        for (int i = 0; i < active; ++i) {
            int base = i * 4;
            for (int j = 0; j < 4; ++j) {
                int off = base * 9 + j * 9;
                verts.push_back(Vector3(vertex_buffer[off], vertex_buffer[off+1], vertex_buffer[off+2]));
                uvs.push_back(Vector2(vertex_buffer[off+7], vertex_buffer[off+8]));
            }
            indices.push_back(base+0); indices.push_back(base+1); indices.push_back(base+2);
            indices.push_back(base+0); indices.push_back(base+2); indices.push_back(base+3);
        }
        RenderingServer::get_singleton()->mesh_add_surface(mesh_rid, RS::PRIMITIVE_TRIANGLES, verts, indices, uvs, Vector<Vector3>());
        // Reapply material (to keep vertex color flag)
        RenderingServer::get_singleton()->particles_set_material(particles_rid, material_rid);
        mesh_dirty = false;
    }

    void sync_all() {
        if (mesh_dirty) update_vertex_buffer();
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->particles_set_emitting(particles_rid, emitting);
        rs->particles_set_amount(particles_rid, amount);
        rs->particles_set_lifetime(particles_rid, lifetime);
        rs->particles_set_cast_shadow(particles_rid, cast_shadow);
        rs->particles_set_gi_mode(particles_rid, gi_mode);
    }
};

// ------------------------------------------------------------------------
// CPUParticles3DExt public methods
// ------------------------------------------------------------------------
CPUParticles3DExt::CPUParticles3DExt() {
    pimpl = new Impl();
    pimpl->init_particles();
}

CPUParticles3DExt::~CPUParticles3DExt() {
    delete pimpl;
}

void CPUParticles3DExt::set_emitting(bool p_emitting) { pimpl->emitting = p_emitting; }
bool CPUParticles3DExt::is_emitting() const { return pimpl->emitting; }
void CPUParticles3DExt::set_one_shot(bool p_one_shot) { pimpl->one_shot = p_one_shot; }
bool CPUParticles3DExt::is_one_shot() const { return pimpl->one_shot; }
void CPUParticles3DExt::set_amount(int p_amount) { pimpl->amount = p_amount; pimpl->init_particles(); }
int CPUParticles3DExt::get_amount() const { return pimpl->amount; }
void CPUParticles3DExt::set_lifetime(float p_lifetime) { pimpl->lifetime = p_lifetime; pimpl->init_particles(); }
float CPUParticles3DExt::get_lifetime() const { return pimpl->lifetime; }
void CPUParticles3DExt::set_preprocess(float p_preprocess) { pimpl->preprocess = p_preprocess; pimpl->init_particles(); }
float CPUParticles3DExt::get_preprocess() const { return pimpl->preprocess; }
void CPUParticles3DExt::set_explosiveness(float p_explosiveness) { pimpl->explosiveness = p_explosiveness; }
float CPUParticles3DExt::get_explosiveness() const { return pimpl->explosiveness; }
void CPUParticles3DExt::set_randomness(float p_randomness) { pimpl->randomness = p_randomness; }
float CPUParticles3DExt::get_randomness() const { return pimpl->randomness; }
void CPUParticles3DExt::set_emission_shape(int p_shape) { pimpl->emission_shape = p_shape; }
int CPUParticles3DExt::get_emission_shape() const { return pimpl->emission_shape; }
void CPUParticles3DExt::set_emission_shape_extents(const Vector3 &p_extents) { pimpl->emission_shape_extents = p_extents; }
Vector3 CPUParticles3DExt::get_emission_shape_extents() const { return pimpl->emission_shape_extents; }
void CPUParticles3DExt::set_emission_points(const Vector<Vector3> &p_points) { pimpl->emission_points = p_points; }
Vector<Vector3> CPUParticles3DExt::get_emission_points() const { return pimpl->emission_points; }
void CPUParticles3DExt::set_draw_order(int p_order) { pimpl->draw_order = p_order; }
int CPUParticles3DExt::get_draw_order() const { return pimpl->draw_order; }
void CPUParticles3DExt::set_material(const RID &p_material) { pimpl->material_rid = p_material; }
RID CPUParticles3DExt::get_material() const { return pimpl->material_rid; }
void CPUParticles3DExt::set_color(const Color &p_color) { pimpl->color = p_color; }
Color CPUParticles3DExt::get_color() const { return pimpl->color; }
void CPUParticles3DExt::set_color_ramp(const RID &p_color_ramp) { pimpl->color_ramp_rid = p_color_ramp; }
RID CPUParticles3DExt::get_color_ramp() const { return pimpl->color_ramp_rid; }
void CPUParticles3DExt::set_trail_enabled(bool p_enabled) { pimpl->trail_enabled = p_enabled; }
bool CPUParticles3DExt::is_trail_enabled() const { return pimpl->trail_enabled; }
void CPUParticles3DExt::set_trail_length(float p_length) { pimpl->trail_length = p_length; }
float CPUParticles3DExt::get_trail_length() const { return pimpl->trail_length; }
void CPUParticles3DExt::set_collision_enabled(bool p_enabled) { pimpl->collision_enabled = p_enabled; }
bool CPUParticles3DExt::is_collision_enabled() const { return pimpl->collision_enabled; }
void CPUParticles3DExt::set_collision_radius(float p_radius) { pimpl->collision_radius = p_radius; }
float CPUParticles3DExt::get_collision_radius() const { return pimpl->collision_radius; }
void CPUParticles3DExt::set_sub_emitter(int p_sub_emitter) { pimpl->sub_emitter_rid = p_sub_emitter; }
int CPUParticles3DExt::get_sub_emitter() const { return pimpl->sub_emitter_rid; }
void CPUParticles3DExt::set_cast_shadow(bool p_cast) { pimpl->cast_shadow = p_cast; }
bool CPUParticles3DExt::get_cast_shadow() const { return pimpl->cast_shadow; }
void CPUParticles3DExt::set_gi_mode(int p_mode) { pimpl->gi_mode = p_mode; }
int CPUParticles3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void CPUParticles3DExt::sync_particles() {
    // Simulate one frame at 60 FPS delta
    float dt = 1.0f / 60.0f;
    pimpl->emit_particles(dt);
    pimpl->simulate(dt);
    pimpl->sync_all();
}