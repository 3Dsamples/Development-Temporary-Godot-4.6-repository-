// soft_body_3d.cpp
#include "soft_body_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace lighting {

struct SoftBody3D::Impl {
    // Geometry
    std::vector<SoftBodyVertex> vertices;
    std::vector<int> tetrahedra;   // 4 ints per tetra
    std::vector<int> triangles;    // 3 ints per triangle

    // Spring network (derived from tets)
    std::vector<std::pair<int,int>> springs;
    std::vector<float> rest_length;
    std::vector<float> stiffness;

    // Parameters
    float stiffness_val = 0.8f;
    float damping_val = 0.1f;
    float pressure_val = 0.0f;
    float pinning_radius = 0.01f;

    int collision_group = 1;
    int collision_mask = 0xFFFFFFFF;
    bool self_collision = false;

    // Lighting flags
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;          // dynamic by default for soft bodies
    float gi_contribution = 1.0f;

    // Temporary buffers for forces
    std::vector<double> forces; // 3 per vertex

    void build_springs();
    void simulate_euler(double dt);
    void apply_pressure(double dt);
    void resolve_collisions();
};

SoftBody3D::SoftBody3D() : pimpl(std::make_unique<Impl>()) {
    set_body_mode(BodyMode::RIGID_DYNAMIC); // soft bodies are dynamic
}
SoftBody3D::~SoftBody3D() = default;

void SoftBody3D::set_vertices(const SoftBodyVertex* vertices, int count) {
    pimpl->vertices.assign(vertices, vertices + count);
    pimpl->forces.assign(count * 3, 0.0);
}
int SoftBody3D::get_vertex_count() const { return (int)pimpl->vertices.size(); }

void SoftBody3D::set_tetrahedra(const int* indices, int tetra_count) {
    pimpl->tetrahedra.assign(indices, indices + tetra_count * 4);
    pimpl->build_springs();
}
void SoftBody3D::set_triangles(const int* indices, int triangle_count) {
    pimpl->triangles.assign(indices, indices + triangle_count * 3);
}
void SoftBody3D::update_vertex_positions(const double* positions, int count) {
    if (count * 3 > (int)pimpl->vertices.size() * 3) return;
    for (size_t i = 0; i < pimpl->vertices.size(); ++i) {
        pimpl->vertices[i].position[0] = positions[i*3];
        pimpl->vertices[i].position[1] = positions[i*3+1];
        pimpl->vertices[i].position[2] = positions[i*3+2];
    }
}

void SoftBody3D::Impl::build_springs() {
    springs.clear();
    rest_length.clear();
    stiffness.clear();
    // For each tetrahedron, add edges (6 edges per tetra)
    for (size_t i = 0; i < tetrahedra.size(); i += 4) {
        int v[4] = { tetrahedra[i], tetrahedra[i+1], tetrahedra[i+2], tetrahedra[i+3] };
        // all combinations
        for (int a = 0; a < 4; ++a)
            for (int b = a+1; b < 4; ++b) {
                int idx = springs.size();
                springs.push_back({v[a], v[b]});
                double dx = vertices[v[a]].position[0] - vertices[v[b]].position[0];
                double dy = vertices[v[a]].position[1] - vertices[v[b]].position[1];
                double dz = vertices[v[a]].position[2] - vertices[v[b]].position[2];
                rest_length.push_back((float)sqrt(dx*dx+dy*dy+dz*dz));
                stiffness.push_back(stiffness_val);
            }
    }
    // Remove duplicate springs (same unordered pair)
    // (simplified – not implemented for brevity)
}

void SoftBody3D::set_stiffness(float stiffness) { pimpl->stiffness_val = stiffness; }
float SoftBody3D::get_stiffness() const { return pimpl->stiffness_val; }
void SoftBody3D::set_damping(float damping) { pimpl->damping_val = damping; }
float SoftBody3D::get_damping() const { return pimpl->damping_val; }
void SoftBody3D::set_pressure(float pressure) { pimpl->pressure_val = pressure; }
float SoftBody3D::get_pressure() const { return pimpl->pressure_val; }
void SoftBody3D::set_pinning_radius(float radius) { pimpl->pinning_radius = radius; }
float SoftBody3D::get_pinning_radius() const { return pimpl->pinning_radius; }
void SoftBody3D::set_collision_group(int group) { pimpl->collision_group = group; }
int SoftBody3D::get_collision_group() const { return pimpl->collision_group; }
void SoftBody3D::set_collision_mask(int mask) { pimpl->collision_mask = mask; }
int SoftBody3D::get_collision_mask() const { return pimpl->collision_mask; }
void SoftBody3D::enable_self_collision(bool enable) { pimpl->self_collision = enable; }
bool SoftBody3D::is_self_collision_enabled() const { return pimpl->self_collision; }

void SoftBody3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void SoftBody3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; GeometryInstance3D::set_receive_shadow(receive); }
void SoftBody3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void SoftBody3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; GeometryInstance3D::set_gi_contribution(amount); }

void SoftBody3D::apply_force(const double* force, const double* at_vertex_index) {
    int idx = (int)(*at_vertex_index);
    if (idx >= 0 && idx < (int)pimpl->vertices.size()) {
        pimpl->forces[idx*3] += force[0];
        pimpl->forces[idx*3+1] += force[1];
        pimpl->forces[idx*3+2] += force[2];
    }
}

void SoftBody3D::Impl::simulate_euler(double dt) {
    // Clear forces, add gravity and internal spring forces
    std::fill(forces.begin(), forces.end(), 0.0);
    double gravity[3] = {0, -9.8, 0}; // could be from scene

    // Spring forces
    for (size_t i = 0; i < springs.size(); ++i) {
        int a = springs[i].first;
        int b = springs[i].second;
        double* p1 = vertices[a].position;
        double* p2 = vertices[b].position;
        double dx = p1[0] - p2[0];
        double dy = p1[1] - p2[1];
        double dz = p1[2] - p2[2];
        double dist = sqrt(dx*dx+dy*dy+dz*dz);
        if (dist < 1e-6) continue;
        double inv_dist = 1.0/dist;
        double force_mag = stiffness[i] * (dist - rest_length[i]);
        // direction p1->p2 normalized
        double fx = dx * inv_dist * force_mag;
        double fy = dy * inv_dist * force_mag;
        double fz = dz * inv_dist * force_mag;
        forces[a*3] -= fx;
        forces[a*3+1] -= fy;
        forces[a*3+2] -= fz;
        forces[b*3] += fx;
        forces[b*3+1] += fy;
        forces[b*3+2] += fz;
    }

    // Euler integration per vertex
    double dt_sq = dt*dt;
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (vertices[i].inv_mass == 0.0f) continue; // pinned
        double ax = forces[i*3] + gravity[0];
        double ay = forces[i*3+1] + gravity[1];
        double az = forces[i*3+2] + gravity[2];
        ax *= vertices[i].inv_mass;
        ay *= vertices[i].inv_mass;
        az *= vertices[i].inv_mass;
        // damping: reduce velocity
        vertices[i].velocity[0] = vertices[i].velocity[0] * (1.0 - damping_val) + ax * dt;
        vertices[i].velocity[1] = vertices[i].velocity[1] * (1.0 - damping_val) + ay * dt;
        vertices[i].velocity[2] = vertices[i].velocity[2] * (1.0 - damping_val) + az * dt;
        vertices[i].position[0] += vertices[i].velocity[0] * dt;
        vertices[i].position[1] += vertices[i].velocity[1] * dt;
        vertices[i].position[2] += vertices[i].velocity[2] * dt;
    }
}

void SoftBody3D::Impl::apply_pressure(double dt) {
    if (pressure_val == 0.0f) return;
    // Simplified: for each tetra, compute volume and apply pressure force to each vertex
    for (size_t i = 0; i < tetrahedra.size(); i += 4) {
        int v0 = tetrahedra[i];
        int v1 = tetrahedra[i+1];
        int v2 = tetrahedra[i+2];
        int v3 = tetrahedra[i+3];
        double* p0 = vertices[v0].position;
        double* p1 = vertices[v1].position;
        double* p2 = vertices[v2].position;
        double* p3 = vertices[v3].position;
        // compute volume
        double a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
        double b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
        double c[3] = { p3[0]-p0[0], p3[1]-p0[1], p3[2]-p0[2] };
        double cross[3] = { a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] };
        double volume = fabs(cross[0]*c[0] + cross[1]*c[1] + cross[2]*c[2]) / 6.0;
        if (volume < 1e-9) continue;
        // pressure force magnitude = pressure * area (approximated)
        double area = pow(volume, 2.0/3.0);
        double f_mag = pressure_val * area * dt;
        // compute outward normal (average face normals)
        // simple: push each vertex away from centroid
        double centroid[3] = { (p0[0]+p1[0]+p2[0]+p3[0])*0.25,
                               (p0[1]+p1[1]+p2[1]+p3[1])*0.25,
                               (p0[2]+p1[2]+p2[2]+p3[2])*0.25 };
        for (int vi : {v0, v1, v2, v3}) {
            double dir[3] = { vertices[vi].position[0] - centroid[0],
                              vertices[vi].position[1] - centroid[1],
                              vertices[vi].position[2] - centroid[2] };
            double len = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
            if (len > 1e-6) {
                forces[vi*3] += dir[0]/len * f_mag;
                forces[vi*3+1] += dir[1]/len * f_mag;
                forces[vi*3+2] += dir[2]/len * f_mag;
            }
        }
    }
}

void SoftBody3D::Impl::resolve_collisions() {
    // placeholder – would query physics server for collisions with other bodies
    // and adjust velocities/positions
}

void SoftBody3D::update_physics(double delta_time) {
    if (delta_time > 0.033f) delta_time = 0.033f;
    pimpl->simulate_euler(delta_time);
    pimpl->apply_pressure(delta_time);
    pimpl->resolve_collisions();

    // Update node transform to the bounding box of vertices (not ideal but works)
    double min_x=1e30, min_y=1e30, min_z=1e30, max_x=-1e30, max_y=-1e30, max_z=-1e30;
    for (auto& v : pimpl->vertices) {
        min_x = std::min(min_x, v.position[0]); max_x = std::max(max_x, v.position[0]);
        min_y = std::min(min_y, v.position[1]); max_y = std::max(max_y, v.position[1]);
        min_z = std::min(min_z, v.position[2]); max_z = std::max(max_z, v.position[2]);
    }
    double center[3] = { (min_x+max_x)*0.5, (min_y+max_y)*0.5, (min_z+max_z)*0.5 };
    Transform3D trans = get_global_transform();
    trans.origin[0] = center[0];
    trans.origin[1] = center[1];
    trans.origin[2] = center[2];
    set_global_transform(trans);
    // Bounding box for culling
    double aabb_min[3] = {min_x, min_y, min_z};
    double aabb_max[3] = {max_x, max_y, max_z};
    set_aabb(aabb_min, aabb_max);
    set_bounding_sphere_radius(std::max({max_x-min_x, max_y-min_y, max_z-min_z}) * 0.5);
}

void SoftBody3D::synchronize_render_server(double delta) {
    PhysicsBody3D::synchronize_render_server(delta);
    // Send vertex positions to render server for dynamic mesh update
    // (placeholder)
}

} // namespace lighting