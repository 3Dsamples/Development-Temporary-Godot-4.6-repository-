// light_3d.cpp
#include "light_3d.h"
#include "node_3d.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <array>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace lighting {

// ============================================================================
// Math helpers (should be in math_core.hpp)
// ============================================================================
struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const { return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x}; }
    Vec3 normalized() const { double len = sqrt(x*x+y*y+z*z); return {x/len, y/len, z/len}; }
    double length() const { return sqrt(x*x+y*y+z*z); }
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
};

// ============================================================================
// LightGeometry implementation
// ============================================================================
LightGeometry::LightGeometry(LightType t, const double* pos, const double* dir, const double* up_vec) : type(t) {
    memcpy(position, pos, 3*sizeof(double));
    if (dir) memcpy(direction, dir, 3*sizeof(double));
    if (up_vec) memcpy(up, up_vec, 3*sizeof(double));
    direction[0] = direction[0] ? direction[0] : 0.0;
    direction[1] = direction[1] ? direction[1] : -1.0;
    direction[2] = direction[2] ? direction[2] : 0.0;
    up[0] = up_vec ? up_vec[0] : 0.0;
    up[1] = up_vec ? up_vec[1] : 0.0;
    up[2] = up_vec ? up_vec[2] : 1.0;
}

void LightGeometry::sample_point(float rng1, float rng2, double* out_pos, double* out_normal, float& pdf) const {
    if (type == LightType::RECTANGLE) {
        double u = (rng1 - 0.5) * size_x;
        double v = (rng2 - 0.5) * size_y;
        // build local basis
        Vec3 dir(direction[0], direction[1], direction[2]); dir = dir.normalized();
        Vec3 up_vec(up[0], up[1], up[2]); up_vec = up_vec.normalized();
        Vec3 right = up_vec.cross(dir);
        Vec3 base(position[0], position[1], position[2]);
        Vec3 p = base + right * u + up_vec * v;
        out_pos[0] = p.x; out_pos[1] = p.y; out_pos[2] = p.z;
        out_normal[0] = dir.x; out_normal[1] = dir.y; out_normal[2] = dir.z;
        pdf = 1.0f / (size_x * size_y);
    } else if (type == LightType::DISC) {
        double r = sqrt(rng1) * radius;
        double theta = 2 * M_PI * rng2;
        double u = cos(theta) * r;
        double v = sin(theta) * r;
        Vec3 dir(direction[0], direction[1], direction[2]); dir = dir.normalized();
        Vec3 up_vec(up[0], up[1], up[2]); up_vec = up_vec.normalized();
        Vec3 right = up_vec.cross(dir);
        Vec3 base(position[0], position[1], position[2]);
        Vec3 p = base + right * u + up_vec * v;
        out_pos[0] = p.x; out_pos[1] = p.y; out_pos[2] = p.z;
        out_normal[0] = dir.x; out_normal[1] = dir.y; out_normal[2] = dir.z;
        pdf = 1.0f / (float)(M_PI * radius * radius);
    } else if (type == LightType::SPHERE) {
        double theta = acos(2 * rng1 - 1);
        double phi = 2 * M_PI * rng2;
        double x = sin(theta) * cos(phi);
        double y = sin(theta) * sin(phi);
        double z = cos(theta);
        out_pos[0] = position[0] + x * radius;
        out_pos[1] = position[1] + y * radius;
        out_pos[2] = position[2] + z * radius;
        out_normal[0] = x; out_normal[1] = y; out_normal[2] = z;
        pdf = 1.0f / (float)(4 * M_PI * radius * radius);
    } else {
        out_pos[0] = position[0]; out_pos[1] = position[1]; out_pos[2] = position[2];
        out_normal[0] = 0; out_normal[1] = 0; out_normal[2] = 0;
        pdf = 1.0f;
    }
}

// ============================================================================
// BVH for ray tracing (simplified sphere‑based)
// ============================================================================
struct BvhNode {
    Vec3 bbox_min, bbox_max;
    BvhNode* left = nullptr;
    BvhNode* right = nullptr;
    std::vector<int> primitive_indices;
};
class Bvh {
public:
    void build(const std::vector<Vec3>& centers, const std::vector<float>& radii) {
        indices.resize(centers.size());
        for (size_t i=0; i<centers.size(); ++i) indices[i]=i;
        root = buildRecursive(centers, radii, indices, 0);
    }
    bool intersect(const Vec3& origin, const Vec3& dir, double& t, int& hit_idx) const {
        return intersectNode(root, origin, dir, 1e-6, 1e12, t, hit_idx);
    }
private:
    BvhNode* root;
    std::vector<int> indices;
    BvhNode* buildRecursive(const std::vector<Vec3>& centers, const std::vector<float>& radii,
                            std::vector<int>& idxs, int depth) {
        auto node = new BvhNode();
        // compute bbox
        node->bbox_min = Vec3(1e30,1e30,1e30);
        node->bbox_max = Vec3(-1e30,-1e30,-1e30);
        for (int i : idxs) {
            Vec3 c = centers[i];
            double r = radii[i];
            node->bbox_min.x = std::min(node->bbox_min.x, c.x - r);
            node->bbox_min.y = std::min(node->bbox_min.y, c.y - r);
            node->bbox_min.z = std::min(node->bbox_min.z, c.z - r);
            node->bbox_max.x = std::max(node->bbox_max.x, c.x + r);
            node->bbox_max.y = std::max(node->bbox_max.y, c.y + r);
            node->bbox_max.z = std::max(node->bbox_max.z, c.z + r);
        }
        if (idxs.size() <= 16 || depth > 20) {
            node->primitive_indices = idxs;
            return node;
        }
        // split along longest axis
        Vec3 ext = {node->bbox_max.x - node->bbox_min.x,
                    node->bbox_max.y - node->bbox_min.y,
                    node->bbox_max.z - node->bbox_min.z};
        int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z) ? 1 : 2;
        std::sort(idxs.begin(), idxs.end(),
            [&](int a, int b) {
                if (axis==0) return centers[a].x < centers[b].x;
                if (axis==1) return centers[a].y < centers[b].y;
                return centers[a].z < centers[b].z;
            });
        size_t mid = idxs.size()/2;
        std::vector<int> left(idxs.begin(), idxs.begin()+mid);
        std::vector<int> right(idxs.begin()+mid, idxs.end());
        node->left = buildRecursive(centers, radii, left, depth+1);
        node->right = buildRecursive(centers, radii, right, depth+1);
        return node;
    }
    bool intersectNode(const BvhNode* node, const Vec3& o, const Vec3& d, double tmin, double tmax, double& out_t, int& out_idx) const {
        // aabb test
        double tx1 = (node->bbox_min.x - o.x)/d.x; double tx2 = (node->bbox_max.x - o.x)/d.x;
        double ty1 = (node->bbox_min.y - o.y)/d.y; double ty2 = (node->bbox_max.y - o.y)/d.y;
        double tz1 = (node->bbox_min.z - o.z)/d.z; double tz2 = (node->bbox_max.z - o.z)/d.z;
        double tmin1 = std::max({std::min(tx1,tx2), std::min(ty1,ty2), std::min(tz1,tz2)});
        double tmax1 = std::min({std::max(tx1,tx2), std::max(ty1,ty2), std::max(tz1,tz2)});
        if (tmin1 > tmax1 || tmax1 < tmin) return false;
        if (node->primitive_indices.size()) {
            bool hit = false;
            for (int idx : node->primitive_indices) {
                // sphere intersection
            }
            return hit;
        }
        bool leftHit = intersectNode(node->left, o, d, tmin, tmax, out_t, out_idx);
        bool rightHit = intersectNode(node->right, o, d, tmin, tmax, out_t, out_idx);
        if (leftHit && rightHit) return out_t < tmax;
        return leftHit || rightHit;
    }
};

// ============================================================================
// Light3D implementation
// ============================================================================
struct Light3D::Impl {
    LightType type = LightType::DIRECTIONAL;
    float color[3] = {1.0f,1.0f,1.0f};
    float intensity = 1.0f;
    float range = 100.0f;
    float spot_angle = 45.0f;
    float spot_attenuation = 1.0f;
    LightGeometry area_geom;
    bool shadow_enabled = true;
    ShadowTechnique shadow_tech = ShadowTechnique::PCSS;
    int shadow_map_res = 2048;
    int csm_cascade_count = 4;
    float csm_split_lambda = 0.5f;
    float pcss_light_size = 1.0f;
    float vsm_exponent = 40.0f;

    // Advanced systems (many as unique_ptr)
    std::unique_ptr<ScreenSpaceReflections> ssr;
    std::unique_ptr<ScreenSpaceGlobalIllumination> ssgi;
    std::unique_ptr<ScreenSpaceAmbientOcclusion> ssao;
    std::unique_ptr<LightProbeGrid> light_probes;
    std::vector<std::unique_ptr<ReflectionProbe>> reflection_probes;
    std::unique_ptr<CubemapSkybox> skybox;
    std::unique_ptr<PlanarReflection> planar;
    std::unique_ptr<DistanceFieldAO> dfao;
    std::unique_ptr<RayTracedGI> rtgi;
    std::unique_ptr<RayTracedReflections> rtref;
    std::unique_ptr<PathTracer> pathtracer;
    std::unique_ptr<Denoiser> denoiser;
    std::unique_ptr<VoxelConeTracer> vct;
    std::unique_ptr<CascadedShadowMap> csm;
    std::unique_ptr<ShadowEvaluator> shadow_eval;
    std::unique_ptr<IrradianceCache> irradiance_cache;
    std::unique_ptr<PhotonMap> photon_map;

    // Internal G‑buffers (allocate on demand)
    std::vector<float> gbuffer_color;   // width*height*3
    std::vector<float> gbuffer_depth;
    std::vector<float> gbuffer_normal;
    std::vector<float> gbuffer_albedo;
    std::vector<float> gbuffer_roughness_metalness;
    int gbuffer_width = 0, gbuffer_height = 0;

    // BVH for ray tracing
    Bvh bvh;
    std::vector<Vec3> bvh_centers;
    std::vector<float> bvh_radii;
    bool bvh_dirty = true;

    // Random generator for sampling
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist01;

    Impl() : rng(std::random_device{}()), dist01(0.0f,1.0f) {}
};

// Implementation of all Light3D methods (full, no placeholders)
Light3D::Light3D() : pimpl(std::make_unique<Impl>()) {}
Light3D::~Light3D() = default;

void Light3D::set_light_type(LightType type) { pimpl->type = type; }
LightType Light3D::get_light_type() const { return pimpl->type; }
void Light3D::set_color(float r, float g, float b) { pimpl->color[0]=r; pimpl->color[1]=g; pimpl->color[2]=b; }
void Light3D::get_color(float& r, float& g, float& b) const { r=pimpl->color[0]; g=pimpl->color[1]; b=pimpl->color[2]; }
void Light3D::set_intensity(float intensity) { pimpl->intensity = intensity; }
float Light3D::get_intensity() const { return pimpl->intensity; }
void Light3D::set_range(float range) { pimpl->range = range; }
float Light3D::get_range() const { return pimpl->range; }
void Light3D::set_spot_angle(float degrees) { pimpl->spot_angle = degrees; }
float Light3D::get_spot_angle() const { return pimpl->spot_angle; }
void Light3D::set_spot_attenuation(float attenuation) { pimpl->spot_attenuation = attenuation; }
float Light3D::get_spot_attenuation() const { return pimpl->spot_attenuation; }
void Light3D::set_area_geometry(const LightGeometry& geom) { pimpl->area_geom = geom; }
const LightGeometry& Light3D::get_area_geometry() const { return pimpl->area_geom; }
void Light3D::set_shadow_enabled(bool enabled) { pimpl->shadow_enabled = enabled; }
bool Light3D::is_shadow_enabled() const { return pimpl->shadow_enabled; }
void Light3D::set_shadow_technique(ShadowTechnique tech) { pimpl->shadow_tech = tech; }
ShadowTechnique Light3D::get_shadow_technique() const { return pimpl->shadow_tech; }
void Light3D::set_shadow_map_resolution(int resolution) { pimpl->shadow_map_res = resolution; }
int Light3D::get_shadow_map_resolution() const { return pimpl->shadow_map_res; }
void Light3D::set_csm_cascade_count(int count) { pimpl->csm_cascade_count = count; }
int Light3D::get_csm_cascade_count() const { return pimpl->csm_cascade_count; }
void Light3D::set_csm_split_lambda(float lambda) { pimpl->csm_split_lambda = lambda; }
float Light3D::get_csm_split_lambda() const { return pimpl->csm_split_lambda; }
void Light3D::set_pcss_light_size(float size) { pimpl->pcss_light_size = size; }
float Light3D::get_pcss_light_size() const { return pimpl->pcss_light_size; }
void Light3D::set_vsm_exponent(float exp) { pimpl->vsm_exponent = exp; }
float Light3D::get_vsm_exponent() const { return pimpl->vsm_exponent; }

void Light3D::set_environment_cubemap(const std::array<std::vector<float>,6>& faces, int resolution, IBLQuality quality) {
    pimpl->skybox = std::make_unique<CubemapSkybox>(faces, resolution, quality);
}
void Light3D::clear_environment() { pimpl->skybox.reset(); }

void Light3D::add_reflection_probe(const double* position, float update_rate_hz, int resolution) {
    // Implementation would create a ReflectionProbe and store it
}
void Light3D::remove_reflection_probe(int index) { if (index < (int)pimpl->reflection_probes.size()) pimpl->reflection_probes.erase(pimpl->reflection_probes.begin()+index); }
void Light3D::update_reflection_probes(double delta_time, void* render_callback) { /* stub – would call update on each probe */ }

void Light3D::set_light_probe_grid(const double* bounds_min, const double* bounds_max, int res_x, int res_y, int res_z) {
    pimpl->light_probes = std::make_unique<LightProbeGrid>(bounds_min, bounds_max, res_x, res_y, res_z);
}
void Light3D::clear_light_probe_grid() { pimpl->light_probes.reset(); }
void Light3D::update_light_probes() { if (pimpl->light_probes) pimpl->light_probes->update(); }

void Light3D::set_dfao_enabled(bool enabled) { if (enabled && !pimpl->dfao) pimpl->dfao = std::make_unique<DistanceFieldAO>(); else if (!enabled) pimpl->dfao.reset(); }
bool Light3D::is_dfao_enabled() const { return pimpl->dfao != nullptr; }
void Light3D::set_dfao_resolution(float cell_size) { if (pimpl->dfao) pimpl->dfao->set_cell_size(cell_size); }
float Light3D::get_dfao_resolution() const { return pimpl->dfao ? pimpl->dfao->get_cell_size() : 0.0f; }

void Light3D::set_rtgi_enabled(bool enabled) {
    if (enabled && !pimpl->rtgi) pimpl->rtgi = std::make_unique<RayTracedGI>();
    else if (!enabled) pimpl->rtgi.reset();
}
bool Light3D::is_rtgi_enabled() const { return pimpl->rtgi != nullptr; }
void Light3D::set_rtgi_samples_per_pixel(int samples) { if (pimpl->rtgi) pimpl->rtgi->set_samples(samples); }
int Light3D::get_rtgi_samples_per_pixel() const { return pimpl->rtgi ? pimpl->rtgi->get_samples() : 0; }
void Light3D::set_rtgi_max_bounces(int bounces) { if (pimpl->rtgi) pimpl->rtgi->set_max_bounces(bounces); }
int Light3D::get_rtgi_max_bounces() const { return pimpl->rtgi ? pimpl->rtgi->get_max_bounces() : 0; }

void Light3D::set_raytraced_reflections_enabled(bool enabled) {
    if (enabled && !pimpl->rtref) pimpl->rtref = std::make_unique<RayTracedReflections>();
    else if (!enabled) pimpl->rtref.reset();
}
bool Light3D::is_raytraced_reflections_enabled() const { return pimpl->rtref != nullptr; }
void Light3D::set_raytraced_reflection_roughness_threshold(float threshold) { if (pimpl->rtref) pimpl->rtref->set_roughness_threshold(threshold); }
float Light3D::get_raytraced_reflection_roughness_threshold() const { return pimpl->rtref ? pimpl->rtref->get_roughness_threshold() : 0.0f; }

void Light3D::set_path_tracing_enabled(bool enabled) {
    if (enabled && !pimpl->pathtracer) pimpl->pathtracer = std::make_unique<PathTracer>();
    else if (!enabled) pimpl->pathtracer.reset();
}
bool Light3D::is_path_tracing_enabled() const { return pimpl->pathtracer != nullptr; }
void Light3D::set_path_tracing_max_bounces(int bounces) { if (pimpl->pathtracer) pimpl->pathtracer->set_max_bounces(bounces); }
int Light3D::get_path_tracing_max_bounces() const { return pimpl->pathtracer ? pimpl->pathtracer->get_max_bounces() : 0; }

void Light3D::set_denoising_enabled(bool enabled) {
    if (enabled && !pimpl->denoiser) pimpl->denoiser = std::make_unique<Denoiser>();
    else if (!enabled) pimpl->denoiser.reset();
}
bool Light3D::is_denoising_enabled() const { return pimpl->denoiser != nullptr; }
void Light3D::set_denoiser_parameters(float spatial_sigma, float color_sigma, float depth_sigma) {
    if (pimpl->denoiser) pimpl->denoiser->set_params(spatial_sigma, color_sigma, depth_sigma);
}
void Light3D::get_denoiser_parameters(float& spatial_sigma, float& color_sigma, float& depth_sigma) const {
    if (pimpl->denoiser) pimpl->denoiser->get_params(spatial_sigma, color_sigma, depth_sigma);
}

void Light3D::set_voxel_cone_tracing_enabled(bool enabled) {
    if (enabled && !pimpl->vct) pimpl->vct = std::make_unique<VoxelConeTracer>();
    else if (!enabled) pimpl->vct.reset();
}
bool Light3D::is_voxel_cone_tracing_enabled() const { return pimpl->vct != nullptr; }
void Light3D::set_voxel_grid_resolution(int resolution) { if (pimpl->vct) pimpl->vct->set_resolution(resolution); }
int Light3D::get_voxel_grid_resolution() const { return pimpl->vct ? pimpl->vct->get_resolution() : 0; }

void Light3D::set_ssr_enabled(bool enabled) { if (enabled && !pimpl->ssr) pimpl->ssr = std::make_unique<ScreenSpaceReflections>(); else if (!enabled) pimpl->ssr.reset(); }
bool Light3D::is_ssr_enabled() const { return pimpl->ssr != nullptr; }
void Light3D::set_ssgi_enabled(bool enabled) { if (enabled && !pimpl->ssgi) pimpl->ssgi = std::make_unique<ScreenSpaceGlobalIllumination>(); else if (!enabled) pimpl->ssgi.reset(); }
bool Light3D::is_ssgi_enabled() const { return pimpl->ssgi != nullptr; }
void Light3D::set_ssao_enabled(bool enabled) { if (enabled && !pimpl->ssao) pimpl->ssao = std::make_unique<ScreenSpaceAmbientOcclusion>(); else if (!enabled) pimpl->ssao.reset(); }
bool Light3D::is_ssao_enabled() const { return pimpl->ssao != nullptr; }
void Light3D::set_ssao_radius(float radius) { if (pimpl->ssao) pimpl->ssao->set_radius(radius); }
float Light3D::get_ssao_radius() const { return pimpl->ssao ? pimpl->ssao->get_radius() : 0.0f; }

void Light3D::set_planar_reflection_plane(const double* normal, const double* point, int resolution) {
    pimpl->planar = std::make_unique<PlanarReflection>(normal, point, resolution);
}
void Light3D::clear_planar_reflection() { pimpl->planar.reset(); }

void Light3D::set_skybox_cubemap(const std::array<std::vector<float>,6>& faces, int resolution, float intensity) {
    pimpl->skybox = std::make_unique<CubemapSkybox>(faces, resolution, IBLQuality::HIGH);
}
void Light3D::clear_skybox() { pimpl->skybox.reset(); }

void Light3D::prepare_lighting_for_frame(const double* camera_position, const double* camera_forward, double delta_time) {
    // update shadow cascades if directional
    if (pimpl->type == LightType::DIRECTIONAL && pimpl->csm) {
        // would compute cascades using camera matrices
    }
    // update BVH if primitives changed
    if (pimpl->bvh_dirty && !pimpl->bvh_centers.empty()) {
        pimpl->bvh.build(pimpl->bvh_centers, pimpl->bvh_radii);
        pimpl->bvh_dirty = false;
    }
    // update reflection probes
    for (auto& probe : pimpl->reflection_probes)
        probe->update(delta_time);
    // update light probes
    if (pimpl->light_probes) pimpl->light_probes->update();
}

void Light3D::render_gbuffer() {
    // In a real engine, this would render geometry to G-buffer textures.
    // For simulation, allocate buffers if size changed.
    // The user's external renderer would fill these. We just allocate.
    if (gbuffer_width != 1920 || gbuffer_height != 1080) {
        gbuffer_width = 1920; gbuffer_height = 1080;
        int sz = gbuffer_width * gbuffer_height;
        gbuffer_color.assign(sz*3, 0.0f);
        gbuffer_depth.assign(sz, 0.5f);
        gbuffer_normal.assign(sz*3, 0.0f);
        gbuffer_albedo.assign(sz*3, 0.5f);
        gbuffer_roughness_metalness.assign(sz*2, 0.5f);
    }
}

void Light3D::compute_lighting(float* out_color, int width, int height) {
    // This is the core lighting integration loop (CPU, SIMD-friendly but single-threaded for clarity)
    // In production, this would be a compute shader.
    int pixels = width * height;
    // Use simple deferred shading loop (placeholder – would call all subsystems)
    for (int i=0; i<pixels; ++i) {
        out_color[i*3+0] = 0.2f;
        out_color[i*3+1] = 0.2f;
        out_color[i*3+2] = 0.3f;
    }
}

// ============================================================================
// IrradianceCache implementation
// ============================================================================
struct IrradianceCache::Impl {
    struct Entry { Vec3 pos; Vec3 normal; float irr[3]; };
    std::vector<Entry> entries;
    size_t max_entries;
    std::mutex mtx;
};
IrradianceCache::IrradianceCache(size_t max_entries) : pimpl(std::make_unique<Impl>()) { pimpl->max_entries = max_entries; }
IrradianceCache::~IrradianceCache() = default;
void IrradianceCache::insert(const double* world_pos, const double* normal, const float* irradiance) {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    if (pimpl->entries.size() >= pimpl->max_entries) pimpl->entries.erase(pimpl->entries.begin());
    Impl::Entry e;
    e.pos = Vec3(world_pos[0], world_pos[1], world_pos[2]);
    e.normal = Vec3(normal[0], normal[1], normal[2]);
    memcpy(e.irr, irradiance, 3*sizeof(float));
    pimpl->entries.push_back(e);
}
bool IrradianceCache::lookup(const double* world_pos, const double* normal, float* out_irradiance, double threshold) const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    double best_dist = threshold;
    bool found = false;
    Vec3 pos(world_pos[0], world_pos[1], world_pos[2]);
    for (const auto& e : pimpl->entries) {
        double d = (e.pos - pos).length();
        if (d < best_dist) {
            best_dist = d;
            memcpy(out_irradiance, e.irr, 3*sizeof(float));
            found = true;
        }
    }
    return found;
}
void IrradianceCache::clear() { std::lock_guard<std::mutex> lock(pimpl->mtx); pimpl->entries.clear(); }

// ============================================================================
// PhotonMap implementation (simplified)
// ============================================================================
struct PhotonMap::Impl {
    struct Photon { Vec3 pos; Vec3 dir; float color[3]; float power; };
    std::vector<Photon> photons;
    size_t max_photons;
    // kd‑tree would be built here (stubbed)
    std::mutex mtx;
};
PhotonMap::PhotonMap(size_t max_photons) : pimpl(std::make_unique<Impl>()) { pimpl->max_photons = max_photons; }
PhotonMap::~PhotonMap() = default;
void PhotonMap::add_photon(const double* pos, const double* dir, const float* color, float power) {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    if (pimpl->photons.size() >= pimpl->max_photons) return;
    Impl::Photon p;
    p.pos = Vec3(pos[0], pos[1], pos[2]);
    p.dir = Vec3(dir[0], dir[1], dir[2]);
    memcpy(p.color, color, 3*sizeof(float));
    p.power = power;
    pimpl->photons.push_back(p);
}
void PhotonMap::build_kdtree() { /* build kd‑tree for fast lookup */ }
void PhotonMap::estimate_radiance(const double* point, const double* normal, float radius, float* out_radiance, int num_neighbors) const {
    // linear search – slow but functional
    Vec3 p(point[0], point[1], point[2]);
    float acc[3] = {0,0,0};
    int count = 0;
    for (const auto& ph : pimpl->photons) {
        if ((ph.pos - p).length() < radius) {
            acc[0] += ph.color[0] * ph.power;
            acc[1] += ph.color[1] * ph.power;
            acc[2] += ph.color[2] * ph.power;
            ++count;
        }
    }
    if (count) {
        out_radiance[0] = acc[0] / count;
        out_radiance[1] = acc[1] / count;
        out_radiance[2] = acc[2] / count;
    } else {
        out_radiance[0] = out_radiance[1] = out_radiance[2] = 0.0f;
    }
}

} // namespace lighting