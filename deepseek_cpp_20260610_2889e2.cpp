// gpu_particles_collision_3d.cpp
#include "gpu_particles_collision_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>

namespace lighting {

// ============================================================================
// Implementation
// ============================================================================
struct GPUParticlesCollision3D::Impl {
    CollisionShape3DType shape_type = CollisionShape3DType::SPHERE;
    double extents[3] = {1.0, 1.0, 1.0};   // half extents for box/sphere
    double radius = 1.0;

    // Heightfield
    bool has_heightfield = false;
    int heightfield_width = 0;
    int heightfield_depth = 0;
    std::vector<float> heightfield_data;
    double heightfield_min = -10.0;
    double heightfield_max = 10.0;
    double heightfield_origin[3] = {0,0,0};
    double heightfield_cell_size = 1.0;

    // SDF texture
    int64_t sdf_texture = -1;
    double sdf_scale = 1.0;

    // Collision parameters
    float bounce = 0.5f;
    float friction = 0.2f;
    float attenuation = 1.0f;
    CollisionFalloff falloff = CollisionFalloff::LINEAR;
    float max_distance = 100.0f;
    uint32_t cull_mask = 0xFFFFFFFF;

    // Visual
    bool visible = true;
    bool cast_shadow = false;
    int gi_mode = 0;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Render server
    int64_t collision_rid = -1;
    bool dirty = true;

    // Helper: evaluate distance and normal for a given world point
    double evaluate_distance(const double* point, double* out_normal) const;
};

GPUParticlesCollision3D::GPUParticlesCollision3D() : pimpl(std::make_unique<Impl>()) {}
GPUParticlesCollision3D::~GPUParticlesCollision3D() = default;

void GPUParticlesCollision3D::set_shape_type(CollisionShape3DType type) {
    pimpl->shape_type = type;
    pimpl->dirty = true;
}
CollisionShape3DType GPUParticlesCollision3D::get_shape_type() const { return pimpl->shape_type; }

void GPUParticlesCollision3D::set_extents(const double* extents) {
    memcpy(pimpl->extents, extents, 3*sizeof(double));
    if (pimpl->shape_type == CollisionShape3DType::SPHERE) {
        // sphere uses max extents as radius
        pimpl->radius = std::max({extents[0], extents[1], extents[2]});
    }
    pimpl->dirty = true;
}
void GPUParticlesCollision3D::get_extents(double* out_extents) const { memcpy(out_extents, pimpl->extents, 3*sizeof(double)); }
void GPUParticlesCollision3D::set_radius(double radius) {
    pimpl->radius = radius;
    if (pimpl->shape_type == CollisionShape3DType::SPHERE) {
        pimpl->extents[0] = pimpl->extents[1] = pimpl->extents[2] = radius;
    }
    pimpl->dirty = true;
}
double GPUParticlesCollision3D::get_radius() const { return pimpl->radius; }

void GPUParticlesCollision3D::set_heightfield_data(int width, int depth,
                                                   const std::vector<float>& heights,
                                                   double min_height, double max_height,
                                                   const double* origin, double cell_size) {
    pimpl->heightfield_width = width;
    pimpl->heightfield_depth = depth;
    pimpl->heightfield_data = heights;
    pimpl->heightfield_min = min_height;
    pimpl->heightfield_max = max_height;
    if (origin) memcpy(pimpl->heightfield_origin, origin, 3*sizeof(double));
    else { pimpl->heightfield_origin[0]=pimpl->heightfield_origin[1]=pimpl->heightfield_origin[2]=0.0; }
    pimpl->heightfield_cell_size = cell_size;
    pimpl->has_heightfield = true;
    pimpl->shape_type = CollisionShape3DType::HEIGHTFIELD;
    pimpl->dirty = true;
}
void GPUParticlesCollision3D::clear_heightfield() {
    pimpl->has_heightfield = false;
    pimpl->heightfield_data.clear();
    pimpl->dirty = true;
}
void GPUParticlesCollision3D::set_sdf_texture(int64_t texture_rid, double texture_to_world_scale) {
    pimpl->sdf_texture = texture_rid;
    pimpl->sdf_scale = texture_to_world_scale;
    pimpl->shape_type = CollisionShape3DType::SDF;
    pimpl->dirty = true;
}
int64_t GPUParticlesCollision3D::get_sdf_texture() const { return pimpl->sdf_texture; }

void GPUParticlesCollision3D::set_bounce(float bounce) { pimpl->bounce = bounce; pimpl->dirty = true; }
float GPUParticlesCollision3D::get_bounce() const { return pimpl->bounce; }
void GPUParticlesCollision3D::set_friction(float friction) { pimpl->friction = friction; pimpl->dirty = true; }
float GPUParticlesCollision3D::get_friction() const { return pimpl->friction; }
void GPUParticlesCollision3D::set_attenuation(float attenuation) { pimpl->attenuation = attenuation; pimpl->dirty = true; }
float GPUParticlesCollision3D::get_attenuation() const { return pimpl->attenuation; }
void GPUParticlesCollision3D::set_falloff(CollisionFalloff falloff) { pimpl->falloff = falloff; pimpl->dirty = true; }
CollisionFalloff GPUParticlesCollision3D::get_falloff() const { return pimpl->falloff; }
void GPUParticlesCollision3D::set_max_distance(float max_dist) { pimpl->max_distance = max_dist; pimpl->dirty = true; }
float GPUParticlesCollision3D::get_max_distance() const { return pimpl->max_distance; }
void GPUParticlesCollision3D::set_cull_mask(uint32_t mask) { pimpl->cull_mask = mask; }
uint32_t GPUParticlesCollision3D::get_cull_mask() const { return pimpl->cull_mask; }

void GPUParticlesCollision3D::set_visible(bool visible) { pimpl->visible = visible; }
bool GPUParticlesCollision3D::is_visible() const { return pimpl->visible; }
void GPUParticlesCollision3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool GPUParticlesCollision3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void GPUParticlesCollision3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int GPUParticlesCollision3D::get_gi_mode() const { return pimpl->gi_mode; }
void GPUParticlesCollision3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void GPUParticlesCollision3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

double GPUParticlesCollision3D::Impl::evaluate_distance(const double* point, double* out_normal) const {
    Transform3D global = get_global_transform(); // need parent node's transform
    // Transform point to local space
    double local[3] = {
        point[0] - global.origin[0],
        point[1] - global.origin[1],
        point[2] - global.origin[2]
    };
    // orientation not handled for simplicity – in production we rotate.
    double dist = 0.0;
    double normal[3] = {0,0,0};
    switch (shape_type) {
        case CollisionShape3DType::SPHERE: {
            double len = std::sqrt(local[0]*local[0] + local[1]*local[1] + local[2]*local[2]);
            dist = len - radius;
            if (len > 1e-6) {
                normal[0] = local[0] / len;
                normal[1] = local[1] / len;
                normal[2] = local[2] / len;
            } else normal[2] = 1.0;
            break;
        }
        case CollisionShape3DType::BOX: {
            // compute signed distance to box
            double dx = std::abs(local[0]) - extents[0];
            double dy = std::abs(local[1]) - extents[1];
            double dz = std::abs(local[2]) - extents[2];
            double max_axis = std::max({dx, dy, dz});
            if (max_axis < 0) {
                // inside: distance to nearest face
                double mind = std::min({-dx, -dy, -dz});
                dist = -mind;
                // normal: direction of closest face
                if (-dx == mind) { normal[0] = (local[0] > 0) ? 1.0 : -1.0; normal[1]=0; normal[2]=0; }
                else if (-dy == mind) { normal[0]=0; normal[1]=(local[1] > 0) ? 1.0 : -1.0; normal[2]=0; }
                else { normal[0]=0; normal[1]=0; normal[2]=(local[2] > 0) ? 1.0 : -1.0; }
            } else {
                // outside
                dist = std::sqrt(std::max(0.0, dx)*std::max(0.0, dx) +
                                 std::max(0.0, dy)*std::max(0.0, dy) +
                                 std::max(0.0, dz)*std::max(0.0, dz));
                normal[0] = (local[0] > 0) ? 1.0 : -1.0;
                normal[1] = (local[1] > 0) ? 1.0 : -1.0;
                normal[2] = (local[2] > 0) ? 1.0 : -1.0;
            }
            break;
        }
        case CollisionShape3DType::HEIGHTFIELD: {
            if (!has_heightfield || heightfield_width == 0) return 1e30;
            // transform to heightfield local space
            double x = local[0] - heightfield_origin[0];
            double z = local[2] - heightfield_origin[2];
            int ix = (int)(x / heightfield_cell_size);
            int iz = (int)(z / heightfield_cell_size);
            if (ix < 0 || ix >= heightfield_width-1 || iz < 0 || iz >= heightfield_depth-1) {
                return 1e30; // outside
            }
            float h00 = heightfield_data[iz * heightfield_width + ix];
            float h10 = heightfield_data[iz * heightfield_width + ix + 1];
            float h01 = heightfield_data[(iz+1) * heightfield_width + ix];
            float h11 = heightfield_data[(iz+1) * heightfield_width + ix + 1];
            double fx = (x - ix * heightfield_cell_size) / heightfield_cell_size;
            double fz = (z - iz * heightfield_cell_size) / heightfield_cell_size;
            double h = (1-fx)*(1-fz)*h00 + fx*(1-fz)*h10 + (1-fx)*fz*h01 + fx*fz*h11;
            dist = local[1] - h;
            normal[0] = 0; normal[1] = (dist > 0) ? 1.0 : -1.0; normal[2] = 0;
            break;
        }
        case CollisionShape3DType::SDF:
            // sample from texture – simplified
            dist = 1e30;
            break;
    }
    if (out_normal) {
        out_normal[0] = normal[0];
        out_normal[1] = normal[1];
        out_normal[2] = normal[2];
    }
    return dist;
}

void GPUParticlesCollision3D::process(double delta) {
    Node3D::process(delta);
}

void GPUParticlesCollision3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (!pimpl->dirty) return;
    if (pimpl->collision_rid == -1) {
        // create in render server
        // pimpl->collision_rid = RenderingServer::particles_collision_create();
    }
    // Update parameters:
    // - shape type, extents, heightfield data, SDF texture
    // - bounce, friction, attenuation, falloff, max_distance
    // - cull mask
    // - visual representation if visible
    if (pimpl->visible && (pimpl->gi_mode > 0 || pimpl->cast_shadow)) {
        // generate debug mesh and set as visual instance
        // For GI: if emissive, add to GI system
        if (pimpl->gi_mode > 0 && pimpl->emissive_intensity > 0.0f) {
            // inject as emissive source
        }
    }
    // Also compute world AABB for frustum culling
    Transform3D global = get_global_transform();
    double min_x = -pimpl->extents[0], max_x = pimpl->extents[0];
    double min_y = -pimpl->extents[1], max_y = pimpl->extents[1];
    double min_z = -pimpl->extents[2], max_z = pimpl->extents[2];
    if (pimpl->shape_type == CollisionShape3DType::SPHERE) {
        min_x = min_y = min_z = -pimpl->radius;
        max_x = max_y = max_z = pimpl->radius;
    }
    double corners[8][3] = {{min_x,min_y,min_z},{max_x,min_y,min_z},{min_x,max_y,min_z},{max_x,max_y,min_z},
                            {min_x,min_y,max_z},{max_x,min_y,max_z},{min_x,max_y,max_z},{max_x,max_y,max_z}};
    double world_min[3] = {1e30,1e30,1e30};
    double world_max[3] = {-1e30,-1e30,-1e30};
    for (int i=0;i<8;++i) {
        Vector3 p(corners[i][0], corners[i][1], corners[i][2]);
        Vector3 wp = global * p;
        world_min[0] = std::min(world_min[0], wp.x);
        world_min[1] = std::min(world_min[1], wp.y);
        world_min[2] = std::min(world_min[2], wp.z);
        world_max[0] = std::max(world_max[0], wp.x);
        world_max[1] = std::max(world_max[1], wp.y);
        world_max[2] = std::max(world_max[2], wp.z);
    }
    set_aabb(world_min, world_max);
    double dx = world_max[0]-world_min[0], dy=world_max[1]-world_min[1], dz=world_max[2]-world_min[2];
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
    pimpl->dirty = false;
}

} // namespace lighting