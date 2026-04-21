// physics/xparticles.hpp
#ifndef XTENSOR_XPARTICLES_HPP
#define XTENSOR_XPARTICLES_HPP

// ----------------------------------------------------------------------------
// xparticles.hpp – High‑performance particle systems
// ----------------------------------------------------------------------------
// Provides CPU and GPU‑friendly particle simulation:
//   - Particle state storage (SoA layout for SIMD)
//   - Emitters: point, box, sphere, mesh surface, texture‑driven
//   - Forces: gravity, drag, wind, turbulence, vortex, attractors
//   - Collisions: sphere, plane, box, mesh (signed distance field)
//   - Neighbor search: uniform grid, spatial hashing, BVH
//   - Rendering: billboard, mesh instancing, ribbon trails
//   - GPU offload via compute shader abstraction
//   - Integration with BigNumber for high‑precision physics
//
// Targets 120 fps with SIMD and parallel task scheduling.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmesh.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace physics {
namespace particles {

// ========================================================================
// Particle Structure (Structure of Arrays for SIMD efficiency)
// ========================================================================
template <class T>
class particle_system {
public:
    particle_system(size_t max_particles);

    // Core data (SoA layout)
    xarray_container<T>& positions();      // (N, 3)
    xarray_container<T>& velocities();     // (N, 3)
    xarray_container<T>& accelerations();  // (N, 3)
    xarray_container<T>& lifetimes();      // (N) current age
    xarray_container<T>& max_lifetimes();  // (N) total lifespan
    xarray_container<T>& sizes();          // (N)
    xarray_container<T>& colors();         // (N, 4) RGBA
    xarray_container<uint32_t>& flags();   // (N) bitmask (active, collidable, etc.)

    size_t active_count() const;
    size_t capacity() const;

    // Spawn/remove
    size_t spawn(const xarray_container<T>& pos, const xarray_container<T>& vel,
                 T lifetime, T size = T(1));
    void kill(size_t index);
    void clear();

    // Update (integrates forces, lifetimes, collisions)
    void update(T dt);

    // GPU offload (if available)
    void update_gpu(T dt);
    void sync_from_gpu();

private:
    size_t m_capacity;
    size_t m_active;
    // SoA buffers
    xarray_container<T> m_pos, m_vel, m_acc;
    xarray_container<T> m_life, m_max_life, m_size;
    xarray_container<T> m_color;
    xarray_container<uint32_t> m_flags;
    // GPU buffers (opaque)
    void* m_gpu_pos;
    void* m_gpu_vel;
    bool m_gpu_dirty;
};

// ========================================================================
// Emitters
// ========================================================================
template <class T>
class particle_emitter {
public:
    virtual ~particle_emitter() = default;
    virtual void emit(particle_system<T>& sys, T dt, size_t max_new) = 0;
    virtual void set_rate(T particles_per_sec) = 0;
    virtual void set_enabled(bool enabled) = 0;
};

// Point emitter
template <class T>
class point_emitter : public particle_emitter<T> {
public:
    point_emitter(const xarray_container<T>& position);
    void emit(particle_system<T>& sys, T dt, size_t max_new) override;
    void set_rate(T rate) override;
    void set_enabled(bool enabled) override;
    void set_velocity(const xarray_container<T>& base_vel, T random_spread);
private:
    xarray_container<T> m_pos, m_vel_base;
    T m_rate, m_spread, m_accumulator;
    bool m_enabled;
};

// Box emitter
template <class T>
class box_emitter : public particle_emitter<T> {
public:
    box_emitter(const xarray_container<T>& min_corner, const xarray_container<T>& max_corner);
    void emit(particle_system<T>& sys, T dt, size_t max_new) override;
    void set_rate(T rate) override;
    void set_enabled(bool enabled) override;
private:
    xarray_container<T> m_min, m_max;
    T m_rate, m_accumulator;
    bool m_enabled;
};

// Mesh surface emitter (uses mesh sampling)
template <class T>
class mesh_emitter : public particle_emitter<T> {
public:
    mesh_emitter(const mesh::mesh<T>& mesh);
    void emit(particle_system<T>& sys, T dt, size_t max_new) override;
    void set_rate(T rate) override;
    void set_enabled(bool enabled) override;
    void set_emission_mode(bool from_vertices, bool from_triangles);
private:
    mesh::mesh<T> m_mesh;
    T m_rate, m_accumulator;
    bool m_enabled, m_use_verts, m_use_tris;
};

// ========================================================================
// Force Affectors
// ========================================================================
template <class T>
class force_affector {
public:
    virtual ~force_affector() = default;
    virtual void apply(particle_system<T>& sys, T dt) = 0;
};

template <class T>
class gravity_force : public force_affector<T> {
public:
    gravity_force(const xarray_container<T>& gravity);
    void apply(particle_system<T>& sys, T dt) override;
private:
    xarray_container<T> m_gravity;
};

template <class T>
class drag_force : public force_affector<T> {
public:
    drag_force(T linear_drag, T quadratic_drag);
    void apply(particle_system<T>& sys, T dt) override;
private:
    T m_linear, m_quadratic;
};

template <class T>
class wind_force : public force_affector<T> {
public:
    wind_force(const xarray_container<T>& direction, T strength, T turbulence);
    void apply(particle_system<T>& sys, T dt) override;
    void set_noise_scale(T scale);
private:
    xarray_container<T> m_dir;
    T m_strength, m_turbulence, m_noise_scale;
    xarray_container<T> m_noise_offsets;
};

template <class T>
class vortex_force : public force_affector<T> {
public:
    vortex_force(const xarray_container<T>& center, const xarray_container<T>& axis,
                 T strength, T falloff);
    void apply(particle_system<T>& sys, T dt) override;
private:
    xarray_container<T> m_center, m_axis;
    T m_strength, m_falloff;
};

template <class T>
class point_attractor : public force_affector<T> {
public:
    point_attractor(const xarray_container<T>& position, T strength, T radius);
    void apply(particle_system<T>& sys, T dt) override;
private:
    xarray_container<T> m_pos;
    T m_strength, m_radius;
};

// ========================================================================
// Collision Constraints
// ========================================================================
template <class T>
class collision_constraint {
public:
    virtual ~collision_constraint() = default;
    virtual void resolve(particle_system<T>& sys, T dt) = 0;
    virtual void set_bounce(T restitution, T friction) = 0;
};

template <class T>
class plane_collision : public collision_constraint<T> {
public:
    plane_collision(const xarray_container<T>& point, const xarray_container<T>& normal);
    void resolve(particle_system<T>& sys, T dt) override;
    void set_bounce(T restitution, T friction) override;
private:
    xarray_container<T> m_point, m_normal;
    T m_restitution, m_friction;
};

template <class T>
class sphere_collision : public collision_constraint<T> {
public:
    sphere_collision(const xarray_container<T>& center, T radius);
    void resolve(particle_system<T>& sys, T dt) override;
    void set_bounce(T restitution, T friction) override;
private:
    xarray_container<T> m_center;
    T m_radius, m_restitution, m_friction;
};

template <class T>
class mesh_collision : public collision_constraint<T> {
public:
    mesh_collision(const mesh::mesh<T>& mesh);
    void resolve(particle_system<T>& sys, T dt) override;
    void set_bounce(T restitution, T friction) override;
    void build_sdf();  // signed distance field for fast queries
private:
    mesh::mesh<T> m_mesh;
    xarray_container<T> m_sdf;
    T m_restitution, m_friction;
    bool m_use_sdf;
};

// ========================================================================
// Neighbor Search (for SPH / flocking)
// ========================================================================
template <class T>
class neighbor_search {
public:
    virtual ~neighbor_search() = default;
    virtual void build(const xarray_container<T>& positions) = 0;
    virtual std::vector<size_t> query(const xarray_container<T>& point, T radius) = 0;
    virtual void query_all(T radius, std::vector<std::vector<size_t>>& neighbors) = 0;
};

template <class T>
class uniform_grid_search : public neighbor_search<T> {
public:
    uniform_grid_search(T cell_size, size_t max_particles);
    void build(const xarray_container<T>& positions) override;
    std::vector<size_t> query(const xarray_container<T>& point, T radius) override;
    void query_all(T radius, std::vector<std::vector<size_t>>& neighbors) override;
private:
    T m_cell_size;
    std::unordered_map<uint64_t, std::vector<size_t>> m_grid;
    xarray_container<T> m_positions;
};

template <class T>
class spatial_hash_search : public neighbor_search<T> {
public:
    spatial_hash_search(T cell_size, size_t table_size = 65536);
    void build(const xarray_container<T>& positions) override;
    std::vector<size_t> query(const xarray_container<T>& point, T radius) override;
    void query_all(T radius, std::vector<std::vector<size_t>>& neighbors) override;
private:
    T m_cell_size;
    size_t m_table_size;
    std::vector<std::vector<size_t>> m_table;
    xarray_container<T> m_positions;
};

// ========================================================================
// Flocking (Boids)
// ========================================================================
template <class T>
class flocking_behavior {
public:
    flocking_behavior(T separation_weight, T alignment_weight, T cohesion_weight,
                      T neighbor_radius, T separation_radius);

    void apply(particle_system<T>& sys, neighbor_search<T>& search);

private:
    T m_sep_w, m_align_w, m_coh_w, m_neighbor_r, m_sep_r;
};

// ========================================================================
// Rendering helpers
// ========================================================================
template <class T>
class particle_renderer {
public:
    // Extract billboard quads (positions, sizes, colors)
    void build_billboards(const particle_system<T>& sys,
                          xarray_container<T>& out_vertices,
                          xarray_container<T>& out_uvs,
                          xarray_container<T>& out_colors);

    // Extract mesh instances (for instanced rendering)
    void build_instances(const particle_system<T>& sys,
                         xarray_container<T>& out_transforms,
                         xarray_container<T>& out_colors);

    // Generate trail ribbons (for connected particles)
    void build_trails(const particle_system<T>& sys,
                      size_t trail_length,
                      xarray_container<T>& out_vertices,
                      xarray_container<uint32_t>& out_indices);
};

// ========================================================================
// GPU Offload (abstraction)
// ========================================================================
template <class T>
class gpu_particle_system {
public:
    gpu_particle_system(size_t max_particles);
    ~gpu_particle_system();

    void upload(const particle_system<T>& sys);
    void update(T dt, const std::vector<force_affector<T>*>& forces,
                const std::vector<collision_constraint<T>*>& collisions);
    void download(particle_system<T>& sys);

    bool is_available() const;

private:
    void* m_compute_shader;
    void* m_buffer_pos;
    void* m_buffer_vel;
    size_t m_capacity;
    bool m_available;
};

} // namespace particles

using particles::particle_system;
using particles::point_emitter;
using particles::box_emitter;
using particles::mesh_emitter;
using particles::gravity_force;
using particles::drag_force;
using particles::wind_force;
using particles::vortex_force;
using particles::point_attractor;
using particles::plane_collision;
using particles::sphere_collision;
using particles::mesh_collision;
using particles::flocking_behavior;
using particles::particle_renderer;
using particles::gpu_particle_system;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comments)
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace particles {

// particle_system
template <class T> particle_system<T>::particle_system(size_t max_particles) : m_capacity(max_particles), m_active(0) { /* TODO: allocate SoA buffers */ }
template <class T> xarray_container<T>& particle_system<T>::positions() { return m_pos; }
template <class T> xarray_container<T>& particle_system<T>::velocities() { return m_vel; }
template <class T> xarray_container<T>& particle_system<T>::accelerations() { return m_acc; }
template <class T> xarray_container<T>& particle_system<T>::lifetimes() { return m_life; }
template <class T> xarray_container<T>& particle_system<T>::max_lifetimes() { return m_max_life; }
template <class T> xarray_container<T>& particle_system<T>::sizes() { return m_size; }
template <class T> xarray_container<T>& particle_system<T>::colors() { return m_color; }
template <class T> xarray_container<uint32_t>& particle_system<T>::flags() { return m_flags; }
template <class T> size_t particle_system<T>::active_count() const { return m_active; }
template <class T> size_t particle_system<T>::capacity() const { return m_capacity; }
template <class T> size_t particle_system<T>::spawn(const xarray_container<T>& pos, const xarray_container<T>& vel, T lifetime, T size) { /* TODO: allocate particle */ return 0; }
template <class T> void particle_system<T>::kill(size_t index) { /* TODO: deactivate particle */ }
template <class T> void particle_system<T>::clear() { m_active = 0; }
template <class T> void particle_system<T>::update(T dt) { /* TODO: integrate, apply forces, collisions */ }
template <class T> void particle_system<T>::update_gpu(T dt) { /* TODO: dispatch compute shader */ }
template <class T> void particle_system<T>::sync_from_gpu() { /* TODO: download buffers */ }

// emitters
template <class T> point_emitter<T>::point_emitter(const xarray_container<T>& position) : m_pos(position), m_rate(0), m_spread(0), m_accumulator(0), m_enabled(true) {}
template <class T> void point_emitter<T>::emit(particle_system<T>& sys, T dt, size_t max_new) { /* TODO: spawn at point */ }
template <class T> void point_emitter<T>::set_rate(T rate) { m_rate = rate; }
template <class T> void point_emitter<T>::set_enabled(bool enabled) { m_enabled = enabled; }
template <class T> void point_emitter<T>::set_velocity(const xarray_container<T>& base_vel, T random_spread) { m_vel_base = base_vel; m_spread = random_spread; }

template <class T> box_emitter<T>::box_emitter(const xarray_container<T>& min_corner, const xarray_container<T>& max_corner) : m_min(min_corner), m_max(max_corner), m_rate(0), m_accumulator(0), m_enabled(true) {}
template <class T> void box_emitter<T>::emit(particle_system<T>& sys, T dt, size_t max_new) { /* TODO: random inside box */ }
template <class T> void box_emitter<T>::set_rate(T rate) { m_rate = rate; }
template <class T> void box_emitter<T>::set_enabled(bool enabled) { m_enabled = enabled; }

template <class T> mesh_emitter<T>::mesh_emitter(const mesh::mesh<T>& mesh) : m_mesh(mesh), m_rate(0), m_accumulator(0), m_enabled(true), m_use_verts(true), m_use_tris(true) {}
template <class T> void mesh_emitter<T>::emit(particle_system<T>& sys, T dt, size_t max_new) { /* TODO: sample mesh surface */ }
template <class T> void mesh_emitter<T>::set_rate(T rate) { m_rate = rate; }
template <class T> void mesh_emitter<T>::set_enabled(bool enabled) { m_enabled = enabled; }
template <class T> void mesh_emitter<T>::set_emission_mode(bool from_vertices, bool from_triangles) { m_use_verts = from_vertices; m_use_tris = from_triangles; }

// forces
template <class T> gravity_force<T>::gravity_force(const xarray_container<T>& gravity) : m_gravity(gravity) {}
template <class T> void gravity_force<T>::apply(particle_system<T>& sys, T dt) { /* TODO: add m_gravity * dt to velocity */ }
template <class T> drag_force<T>::drag_force(T linear_drag, T quadratic_drag) : m_linear(linear_drag), m_quadratic(quadratic_drag) {}
template <class T> void drag_force<T>::apply(particle_system<T>& sys, T dt) { /* TODO: apply drag */ }
template <class T> wind_force<T>::wind_force(const xarray_container<T>& direction, T strength, T turbulence) : m_dir(direction), m_strength(strength), m_turbulence(turbulence), m_noise_scale(1) {}
template <class T> void wind_force<T>::apply(particle_system<T>& sys, T dt) { /* TODO: apply wind with Perlin turbulence */ }
template <class T> void wind_force<T>::set_noise_scale(T scale) { m_noise_scale = scale; }
template <class T> vortex_force<T>::vortex_force(const xarray_container<T>& center, const xarray_container<T>& axis, T strength, T falloff) : m_center(center), m_axis(axis), m_strength(strength), m_falloff(falloff) {}
template <class T> void vortex_force<T>::apply(particle_system<T>& sys, T dt) { /* TODO: tangential acceleration */ }
template <class T> point_attractor<T>::point_attractor(const xarray_container<T>& position, T strength, T radius) : m_pos(position), m_strength(strength), m_radius(radius) {}
template <class T> void point_attractor<T>::apply(particle_system<T>& sys, T dt) { /* TODO: attract towards point */ }

// collisions
template <class T> plane_collision<T>::plane_collision(const xarray_container<T>& point, const xarray_container<T>& normal) : m_point(point), m_normal(normal), m_restitution(0.5), m_friction(0) {}
template <class T> void plane_collision<T>::resolve(particle_system<T>& sys, T dt) { /* TODO: reflect and slide */ }
template <class T> void plane_collision<T>::set_bounce(T restitution, T friction) { m_restitution = restitution; m_friction = friction; }
template <class T> sphere_collision<T>::sphere_collision(const xarray_container<T>& center, T radius) : m_center(center), m_radius(radius), m_restitution(0.5), m_friction(0) {}
template <class T> void sphere_collision<T>::resolve(particle_system<T>& sys, T dt) { /* TODO: push out of sphere */ }
template <class T> void sphere_collision<T>::set_bounce(T restitution, T friction) { m_restitution = restitution; m_friction = friction; }
template <class T> mesh_collision<T>::mesh_collision(const mesh::mesh<T>& mesh) : m_mesh(mesh), m_restitution(0.5), m_friction(0), m_use_sdf(false) {}
template <class T> void mesh_collision<T>::resolve(particle_system<T>& sys, T dt) { /* TODO: collide with mesh triangles or SDF */ }
template <class T> void mesh_collision<T>::set_bounce(T restitution, T friction) { m_restitution = restitution; m_friction = friction; }
template <class T> void mesh_collision<T>::build_sdf() { /* TODO: compute signed distance field */ m_use_sdf = true; }

// neighbor search
template <class T> uniform_grid_search<T>::uniform_grid_search(T cell_size, size_t max_particles) : m_cell_size(cell_size) {}
template <class T> void uniform_grid_search<T>::build(const xarray_container<T>& positions) { /* TODO: hash positions to grid cells */ m_positions = positions; }
template <class T> std::vector<size_t> uniform_grid_search<T>::query(const xarray_container<T>& point, T radius) { /* TODO: query cells within radius */ return {}; }
template <class T> void uniform_grid_search<T>::query_all(T radius, std::vector<std::vector<size_t>>& neighbors) { /* TODO: build neighbor lists */ }
template <class T> spatial_hash_search<T>::spatial_hash_search(T cell_size, size_t table_size) : m_cell_size(cell_size), m_table_size(table_size) {}
template <class T> void spatial_hash_search<T>::build(const xarray_container<T>& positions) { /* TODO: spatial hashing */ m_positions = positions; }
template <class T> std::vector<size_t> spatial_hash_search<T>::query(const xarray_container<T>& point, T radius) { return {}; }
template <class T> void spatial_hash_search<T>::query_all(T radius, std::vector<std::vector<size_t>>& neighbors) {}

// flocking
template <class T> flocking_behavior<T>::flocking_behavior(T separation_weight, T alignment_weight, T cohesion_weight, T neighbor_radius, T separation_radius) : m_sep_w(separation_weight), m_align_w(alignment_weight), m_coh_w(cohesion_weight), m_neighbor_r(neighbor_radius), m_sep_r(separation_radius) {}
template <class T> void flocking_behavior<T>::apply(particle_system<T>& sys, neighbor_search<T>& search) { /* TODO: compute boids rules */ }

// renderer
template <class T> void particle_renderer<T>::build_billboards(const particle_system<T>& sys, xarray_container<T>& out_vertices, xarray_container<T>& out_uvs, xarray_container<T>& out_colors) { /* TODO: generate quad vertices */ }
template <class T> void particle_renderer<T>::build_instances(const particle_system<T>& sys, xarray_container<T>& out_transforms, xarray_container<T>& out_colors) { /* TODO: generate instance matrices */ }
template <class T> void particle_renderer<T>::build_trails(const particle_system<T>& sys, size_t trail_length, xarray_container<T>& out_vertices, xarray_container<uint32_t>& out_indices) { /* TODO: generate trail ribbon */ }

// GPU
template <class T> gpu_particle_system<T>::gpu_particle_system(size_t max_particles) : m_compute_shader(nullptr), m_buffer_pos(nullptr), m_buffer_vel(nullptr), m_capacity(max_particles), m_available(false) { /* TODO: check GPU support */ }
template <class T> gpu_particle_system<T>::~gpu_particle_system() { /* TODO: release GPU resources */ }
template <class T> void gpu_particle_system<T>::upload(const particle_system<T>& sys) { /* TODO: copy to GPU */ }
template <class T> void gpu_particle_system<T>::update(T dt, const std::vector<force_affector<T>*>& forces, const std::vector<collision_constraint<T>*>& collisions) { /* TODO: dispatch compute shader */ }
template <class T> void gpu_particle_system<T>::download(particle_system<T>& sys) { /* TODO: copy back to CPU */ }
template <class T> bool gpu_particle_system<T>::is_available() const { return m_available; }

} // namespace particles
} // namespace physics
} // namespace xt

#endif // XTENSOR_XPARTICLES_HPP