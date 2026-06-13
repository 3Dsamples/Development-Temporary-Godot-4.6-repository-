// lightmap_gi.cpp
#include "lightmap_gi.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <queue>

namespace lighting {

// ============================================================================
// Helper: AABB for mesh
// ============================================================================
struct AABB {
    double min[3], max[3];
    AABB() { min[0]=min[1]=min[2]=1e30; max[0]=max[1]=max[2]=-1e30; }
    void expand(const double* p) {
        for (int i=0;i<3;++i) {
            min[i] = std::min(min[i], p[i]);
            max[i] = std::max(max[i], p[i]);
        }
    }
    double get_surface_area() const {
        double dx = max[0]-min[0], dy = max[1]-min[1], dz = max[2]-min[2];
        return 2.0*(dx*dy + dy*dz + dz*dx);
    }
};

// ============================================================================
// Per‑mesh lightmap UV generation (simplified: planar projection)
// ============================================================================
static void generate_lightmap_uvs(const std::vector<double>& vertices,
                                  const std::vector<int>& indices,
                                  const AABB& bounds,
                                  std::vector<float>& out_uvs,
                                  int texels_per_unit) {
    out_uvs.resize(vertices.size() / 3 * 2);
    double texel_size = 1.0 / texels_per_unit;
    // planar projection from top (Y axis) for simplicity
    for (size_t i = 0; i < vertices.size(); i += 3) {
        double x = vertices[i];
        double z = vertices[i+2];
        double u = (x - bounds.min[0]) * texel_size;
        double v = (z - bounds.min[2]) * texel_size;
        out_uvs[i/3*2] = (float)u;
        out_uvs[i/3*2+1] = (float)v;
    }
}

// ============================================================================
// Simple path tracer for lightmap baking (naive, not optimized for production)
// ============================================================================
class LightmapBaker {
public:
    struct MeshData {
        std::vector<double> vertices;
        std::vector<int> indices;
        std::vector<float> uvs;          // lightmap UVs (generated)
        int material_id;
        AABB bounds;
    };

    std::vector<MeshData> meshes;
    int texels_per_unit = 64;
    int bounce_count = 2;
    bool bake_shadows = true;
    bool bake_emissive = true;

    std::vector<LightmapData> lightmaps; // per‑mesh or per‑atlas
    std::atomic<float> progress{0.0f};
    std::atomic<bool> cancel{false};
    std::thread worker;

    void start_bake() {
        cancel = false;
        worker = std::thread(&LightmapBaker::bake_thread, this);
    }

    void join() { if (worker.joinable()) worker.join(); }

private:
    void bake_thread() {
        int total_texels = 0;
        for (auto& mesh : meshes) {
            // compute resolution from bounds and texels_per_unit
            double width = mesh.bounds.max[0] - mesh.bounds.min[0];
            double depth = mesh.bounds.max[2] - mesh.bounds.min[2];
            int res_u = std::max(1, (int)(width * texels_per_unit));
            int res_v = std::max(1, (int)(depth * texels_per_unit));
            LightmapData lm;
            lm.width = res_u;
            lm.height = res_v;
            lm.data_r.assign(res_u * res_v, 0.0f);
            lm.data_g.assign(res_u * res_v, 0.0f);
            lm.data_b.assign(res_u * res_v, 0.0f);
            // For each texel, trace a ray from surface point into hemisphere
            // For simplicity, we assume a uniform ambient + direct light only.
            // Real implementation would do path tracing.
            for (int y = 0; y < res_v && !cancel; ++y) {
                for (int x = 0; x < res_u; ++x) {
                    // reconstruct world position from UV and bounds (simplified)
                    double u = (x + 0.5) / res_u;
                    double v = (y + 0.5) / res_v;
                    double wx = mesh.bounds.min[0] + u * width;
                    double wz = mesh.bounds.min[2] + v * depth;
                    double wy = mesh.bounds.min[1] + 0.1; // approximate height
                    // evaluate direct lighting from a directional light (sun)
                    double light_dir[3] = {0.5, -1.0, 0.2};
                    double light_intensity[3] = {1.8, 1.8, 1.6};
                    double normal[3] = {0,1,0}; // assume ground plane
                    double ndotl = -light_dir[1]; // simplified
                    if (ndotl > 0) {
                        lm.data_r[y*res_u + x] = light_intensity[0] * ndotl;
                        lm.data_g[y*res_u + x] = light_intensity[1] * ndotl;
                        lm.data_b[y*res_u + x] = light_intensity[2] * ndotl;
                    }
                }
                progress = (float)(y+1) / (float)res_v / (float)meshes.size();
            }
            lightmaps.push_back(lm);
        }
        progress = 1.0f;
    }
};

// ============================================================================
// LightmapGI implementation
// ============================================================================
struct LightmapGI::Impl {
    LightmapQuality quality = LightmapQuality::MEDIUM;
    int texels_per_unit = 64;                 // overrides quality if >0
    int bounce_count = 2;
    bool bake_shadows = true;
    bool bake_emissive = true;

    bool baking = false;
    float bake_progress = 0.0f;
    std::function<void()> bake_callback;
    std::unique_ptr<LightmapBaker> baker;
    std::thread bake_thread;

    // Lightmap atlases
    std::vector<LightmapData> lightmaps;
    std::vector<int64_t> texture_rids;        // RenderingServer texture IDs

    // Light probes (for dynamic GI)
    bool use_light_probes = true;
    float probe_update_freq = 5.0f;
    int probe_resolution = 32;                // per axis
    std::vector<float> probe_data;            // SH coefficients for each probe

    float gi_contribution = 1.0f;

    bool dirty = true;

    void start_bake();
    void finish_bake();
};

LightmapGI::LightmapGI() : pimpl(std::make_unique<Impl>()) {}
LightmapGI::~LightmapGI() {
    if (pimpl->baking) cancel_bake();
}

void LightmapGI::set_quality(LightmapQuality quality) {
    pimpl->quality = quality;
    pimpl->texels_per_unit = static_cast<int>(quality);
}
LightmapQuality LightmapGI::get_quality() const { return pimpl->quality; }
void LightmapGI::set_texel_per_unit(int texels) {
    pimpl->texels_per_unit = texels;
    if (texels <= 0) pimpl->texels_per_unit = static_cast<int>(pimpl->quality);
}
int LightmapGI::get_texel_per_unit() const { return pimpl->texels_per_unit; }
void LightmapGI::set_bounce_count(int bounces) { pimpl->bounce_count = std::max(0, bounces); }
int LightmapGI::get_bounce_count() const { return pimpl->bounce_count; }
void LightmapGI::set_bake_shadows(bool shadows) { pimpl->bake_shadows = shadows; }
bool LightmapGI::get_bake_shadows() const { return pimpl->bake_shadows; }
void LightmapGI::set_bake_emissive(bool emissive) { pimpl->bake_emissive = emissive; }
bool LightmapGI::get_bake_emissive() const { return pimpl->bake_emissive; }

void LightmapGI::Impl::start_bake() {
    if (baking) return;
    baker = std::make_unique<LightmapBaker>();
    // Gather static geometry from scene (in real engine, query all GeometryInstance3D with gi_mode = static)
    // For demo, we add a dummy plane.
    LightmapBaker::MeshData mesh;
    mesh.vertices = {
        -10,0,-10,  10,0,-10,  10,0,10,  -10,0,10
    };
    mesh.indices = {0,1,2, 0,2,3};
    mesh.bounds.expand(&mesh.vertices[0]);
    mesh.bounds.expand(&mesh.vertices[3]);
    mesh.bounds.expand(&mesh.vertices[6]);
    mesh.bounds.expand(&mesh.vertices[9]);
    generate_lightmap_uvs(mesh.vertices, mesh.indices, mesh.bounds, mesh.uvs, texels_per_unit);
    baker->meshes.push_back(mesh);
    baker->texels_per_unit = texels_per_unit;
    baker->bounce_count = bounce_count;
    baker->bake_shadows = bake_shadows;
    baker->bake_emissive = bake_emissive;
    baker->start_bake();
    baking = true;
    bake_thread = std::thread([this]() {
        baker->join();
        finish_bake();
    });
}

void LightmapGI::Impl::finish_bake() {
    if (baker) {
        lightmaps = baker->lightmaps;
        bake_progress = baker->progress;
        baking = false;
        // Create rendering server textures
        texture_rids.clear();
        for (auto& lm : lightmaps) {
            int64_t tid = -1; // RenderingServer::texture_create_2d(lm.width, lm.height, lm.data_r.data(), ...)
            texture_rids.push_back(tid);
        }
        dirty = true;
    }
    if (bake_callback) bake_callback();
    baker.reset();
}

void LightmapGI::bake() {
    if (pimpl->baking) return;
    pimpl->start_bake();
}
bool LightmapGI::is_baking() const { return pimpl->baking; }
void LightmapGI::cancel_bake() {
    if (pimpl->baker) pimpl->baker->cancel = true;
    if (pimpl->bake_thread.joinable()) pimpl->bake_thread.join();
    pimpl->baking = false;
}
float LightmapGI::get_bake_progress() const { return pimpl->bake_progress; }
void LightmapGI::set_bake_completed_callback(std::function<void()> callback) { pimpl->bake_callback = callback; }

int LightmapGI::get_lightmap_atlas_count() const { return (int)pimpl->lightmaps.size(); }
int64_t LightmapGI::get_lightmap_texture_id(int atlas_index) const {
    if (atlas_index >= 0 && atlas_index < (int)pimpl->texture_rids.size())
        return pimpl->texture_rids[atlas_index];
    return -1;
}
void LightmapGI::get_lightmap_uv_scale(int atlas_index, double* out_scale) const {
    // return UV scale for meshes assigned to this atlas (simplified)
    out_scale[0] = 1.0; out_scale[1] = 1.0;
}

void LightmapGI::set_probe_update_frequency(float fps) { pimpl->probe_update_freq = fps; }
float LightmapGI::get_probe_update_frequency() const { return pimpl->probe_update_freq; }
void LightmapGI::set_use_light_probes(bool use) { pimpl->use_light_probes = use; }
bool LightmapGI::is_using_light_probes() const { return pimpl->use_light_probes; }
int LightmapGI::get_probe_grid_resolution() const { return pimpl->probe_resolution; }

void LightmapGI::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float LightmapGI::get_gi_contribution() const { return pimpl->gi_contribution; }

void LightmapGI::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        // Upload lightmap textures to rendering server and assign to static meshes
        pimpl->dirty = false;
    }
    // Update light probes (if enabled and not baking) – interpolation from baked lightmaps
    if (pimpl->use_light_probes && !pimpl->baking) {
        // For each probe in grid, sample nearest lightmap texels and compute SH
    }
}

void LightmapGI::process(double delta) {
    Node3D::process(delta);
    if (pimpl->baking && pimpl->baker) {
        pimpl->bake_progress = pimpl->baker->progress;
        if (pimpl->baker->cancel) cancel_bake();
    }
}

} // namespace lighting