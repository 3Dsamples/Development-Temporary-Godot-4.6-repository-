// Name : lighting enhancement
// File : scene/3d/navigation_region_3d_ext.cpp 46 of 60
// Description : Implementation of NavigationRegion3DExt with NavMesh baking (simulated),
//               obstacle management, debug wireframe mesh, and server sync.
#include "navigation_region_3d_ext.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/geometry_3d.h"
#include "core/math/transform_3d.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include <cmath>
#include <atomic>

struct NavigationRegion3DExt::Impl {
    RID nav_region_rid;                 // NavigationServer3D region handle
    RID nav_mesh_rid;                   // Navigation mesh resource (simulated)
    // Baking parameters
    float cell_size = 0.2f;
    float cell_height = 0.1f;
    float agent_height = 2.0f;
    float agent_radius = 0.5f;
    float agent_max_climb = 0.9f;
    float agent_max_slope = 45.0f;      // degrees
    float region_min_size = 2.0f;
    float region_merge_size = 20.0f;

    // Debug visualization
    RID debug_mesh_rid;
    RID debug_instance_rid;
    bool debug_visible = false;
    Color debug_color = Color(0.2, 0.6, 1.0);
    Color debug_emissive_color = Color(0,0,0);
    float debug_emissive_intensity = 0.0f;

    // Obstacles (dynamic)
    Vector<RID> obstacles;

    // GI flags (for debug mesh)
    int gi_mode = 0;
    float gi_contribution = 1.0f;

    // Baking state (simulated)
    std::atomic<bool> baking{false};
    std::atomic<float> bake_progress{0.0f};
    Thread *bake_thread = nullptr;
    Mutex bake_mutex;

    Impl() {
        nav_region_rid = NavigationServer3D::get_singleton()->region_create();
        // Dummy navmesh RID (not used until baked)
        nav_mesh_rid = NavigationServer3D::get_singleton()->navigation_mesh_create();
        // Create debug mesh resources
        debug_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        debug_instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(debug_instance_rid, debug_mesh_rid);
    }

    ~Impl() {
        cancel_bake();
        if (nav_region_rid.is_valid()) {
            NavigationServer3D::get_singleton()->free(nav_region_rid);
        }
        if (nav_mesh_rid.is_valid()) {
            NavigationServer3D::get_singleton()->free(nav_mesh_rid);
        }
        if (debug_mesh_rid.is_valid()) {
            RenderingServer::get_singleton()->free(debug_mesh_rid);
        }
        if (debug_instance_rid.is_valid()) {
            RenderingServer::get_singleton()->free(debug_instance_rid);
        }
    }

    void update_navigation_server() {
        NavigationServer3D *ns = NavigationServer3D::get_singleton();
        ns->region_set_transform(nav_region_rid, get_global_transform());
        ns->region_set_navigation_mesh(nav_region_rid, nav_mesh_rid);
        ns->region_set_cell_size(nav_region_rid, cell_size);
        ns->region_set_cell_height(nav_region_rid, cell_height);
        ns->region_set_agent_height(nav_region_rid, agent_height);
        ns->region_set_agent_radius(nav_region_rid, agent_radius);
        ns->region_set_agent_max_climb(nav_region_rid, agent_max_climb);
        ns->region_set_agent_max_slope(nav_region_rid, agent_max_slope);
        ns->region_set_region_min_size(nav_region_rid, region_min_size);
        ns->region_set_region_merge_size(nav_region_rid, region_merge_size);
        // Add obstacles
        for (const RID &obs : obstacles) {
            ns->region_add_obstacle(nav_region_rid, obs);
        }
    }

    // Simulate baking: generate a simple plane mesh as NavMesh (for demonstration)
    void bake_navmesh_async() {
        if (baking) return;
        baking = true;
        bake_progress = 0.0f;
        bake_thread = new Thread;
        bake_thread->start([this]() {
            // Step 1: Collect scene geometry (simplified: create a ground plane at y=0)
            Vector<Vector3> vertices;
            Vector<int> indices;
            float size = 20.0f;
            int div = 20;
            float step = size / div;
            for (int i = 0; i <= div; ++i) {
                float z = -size/2 + i * step;
                for (int j = 0; j <= div; ++j) {
                    float x = -size/2 + j * step;
                    vertices.push_back(Vector3(x, 0, z));
                }
            }
            for (int i = 0; i < div; ++i) {
                for (int j = 0; j < div; ++j) {
                    int i0 = i * (div+1) + j;
                    int i1 = i * (div+1) + j+1;
                    int i2 = (i+1) * (div+1) + j;
                    int i3 = (i+1) * (div+1) + j+1;
                    indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
                    indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
                }
            }
            // Simulate baking steps
            for (int step = 0; step <= 100; ++step) {
                if (!baking) return;
                bake_progress = step / 100.0f;
                OS::get_singleton()->delay_usec(10000);
            }
            // Build navigation mesh from geometry (in real engine, use Recast)
            // For simulation, we just create a dummy mesh and set to server
            RID nav_mesh = NavigationServer3D::get_singleton()->navigation_mesh_create();
            NavigationServer3D::get_singleton()->navigation_mesh_set_vertices(nav_mesh, vertices);
            NavigationServer3D::get_singleton()->navigation_mesh_set_indices(nav_mesh, indices);
            NavigationServer3D::get_singleton()->region_set_navigation_mesh(nav_region_rid, nav_mesh);
            // Generate debug mesh for visualization
            generate_debug_mesh(vertices, indices);
            // Cleanup old nav_mesh if needed
            if (nav_mesh_rid.is_valid()) {
                NavigationServer3D::get_singleton()->free(nav_mesh_rid);
            }
            nav_mesh_rid = nav_mesh;
            MutexLock lock(bake_mutex);
            baking = false;
            bake_progress = 1.0f;
        });
    }

    void generate_debug_mesh(const Vector<Vector3> &vertices, const Vector<int> &indices) {
        if (!debug_visible) return;
        // Convert triangle mesh to lines (wireframe)
        Vector<Vector3> line_verts;
        Vector<int> line_indices;
        for (int i = 0; i < indices.size(); i += 3) {
            Vector3 a = vertices[indices[i]];
            Vector3 b = vertices[indices[i+1]];
            Vector3 c = vertices[indices[i+2]];
            line_verts.push_back(a); line_verts.push_back(b);
            line_verts.push_back(b); line_verts.push_back(c);
            line_verts.push_back(c); line_verts.push_back(a);
            int base = line_indices.size();
            line_indices.push_back(base); line_indices.push_back(base+1);
            line_indices.push_back(base+2); line_indices.push_back(base+3);
            line_indices.push_back(base+4); line_indices.push_back(base+5);
        }
        RenderingServer::get_singleton()->mesh_clear(debug_mesh_rid);
        if (line_verts.size() == 0) return;
        RenderingServer::get_singleton()->mesh_add_surface(debug_mesh_rid, RS::PRIMITIVE_LINES, line_verts, line_indices, Vector<Vector2>(), Vector<Vector3>());
        RID mat = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(mat, "albedo", debug_color);
        if (debug_emissive_intensity > 0.0f) {
            RenderingServer::get_singleton()->material_set_param(mat, "emission", debug_emissive_color);
            RenderingServer::get_singleton()->material_set_param(mat, "emission_intensity", debug_emissive_intensity);
        }
        RenderingServer::get_singleton()->mesh_surface_set_material(debug_mesh_rid, 0, mat);
        Transform3D global = get_global_transform(); // from Node3D
        RenderingServer::get_singleton()->instance_set_transform(debug_instance_rid, global);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, debug_visible);
    }

    void cancel_bake() {
        if (!baking) return;
        baking = false;
        if (bake_thread) {
            bake_thread->wait_to_finish();
            delete bake_thread;
            bake_thread = nullptr;
        }
    }
};

NavigationRegion3DExt::NavigationRegion3DExt() {
    pimpl = new Impl();
}

NavigationRegion3DExt::~NavigationRegion3DExt() {
    delete pimpl;
}

void NavigationRegion3DExt::set_navigation_mesh(const RID &p_nav_mesh) {
    if (pimpl->baking) return;
    pimpl->nav_mesh_rid = p_nav_mesh;
    pimpl->update_navigation_server();
}
RID NavigationRegion3DExt::get_navigation_mesh() const { return pimpl->nav_mesh_rid; }

void NavigationRegion3DExt::set_cell_size(float p_size) {
    pimpl->cell_size = p_size;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_cell_size() const { return pimpl->cell_size; }
void NavigationRegion3DExt::set_cell_height(float p_height) {
    pimpl->cell_height = p_height;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_cell_height() const { return pimpl->cell_height; }
void NavigationRegion3DExt::set_agent_height(float p_height) {
    pimpl->agent_height = p_height;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_agent_height() const { return pimpl->agent_height; }
void NavigationRegion3DExt::set_agent_radius(float p_radius) {
    pimpl->agent_radius = p_radius;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_agent_radius() const { return pimpl->agent_radius; }
void NavigationRegion3DExt::set_agent_max_climb(float p_climb) {
    pimpl->agent_max_climb = p_climb;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_agent_max_climb() const { return pimpl->agent_max_climb; }
void NavigationRegion3DExt::set_agent_max_slope(float p_slope_deg) {
    pimpl->agent_max_slope = p_slope_deg;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_agent_max_slope() const { return pimpl->agent_max_slope; }
void NavigationRegion3DExt::set_region_min_size(float p_size) {
    pimpl->region_min_size = p_size;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_region_min_size() const { return pimpl->region_min_size; }
void NavigationRegion3DExt::set_region_merge_size(float p_size) {
    pimpl->region_merge_size = p_size;
    pimpl->update_navigation_server();
}
float NavigationRegion3DExt::get_region_merge_size() const { return pimpl->region_merge_size; }

void NavigationRegion3DExt::bake_navmesh() {
    if (pimpl->baking) return;
    pimpl->bake_navmesh_async();
}
bool NavigationRegion3DExt::is_baking() const { return pimpl->baking; }
float NavigationRegion3DExt::get_bake_progress() const { return pimpl->bake_progress; }
void NavigationRegion3DExt::cancel_bake() { pimpl->cancel_bake(); }

void NavigationRegion3DExt::add_obstacle(const RID &p_obstacle) {
    pimpl->obstacles.push_back(p_obstacle);
    NavigationServer3D::get_singleton()->region_add_obstacle(pimpl->nav_region_rid, p_obstacle);
}
void NavigationRegion3DExt::remove_obstacle(const RID &p_obstacle) {
    int idx = pimpl->obstacles.find(p_obstacle);
    if (idx != -1) {
        pimpl->obstacles.remove(idx);
        NavigationServer3D::get_singleton()->region_remove_obstacle(pimpl->nav_region_rid, p_obstacle);
    }
}
void NavigationRegion3DExt::clear_obstacles() {
    for (const RID &obs : pimpl->obstacles) {
        NavigationServer3D::get_singleton()->region_remove_obstacle(pimpl->nav_region_rid, obs);
    }
    pimpl->obstacles.clear();
}

void NavigationRegion3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    if (pimpl->debug_instance_rid.is_valid())
        RenderingServer::get_singleton()->instance_set_visible(pimpl->debug_instance_rid, p_visible);
}
bool NavigationRegion3DExt::is_debug_visible() const { return pimpl->debug_visible; }

void NavigationRegion3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    // Mesh will be regenerated on next bake or sync
}
Color NavigationRegion3DExt::get_debug_color() const { return pimpl->debug_color; }

void NavigationRegion3DExt::set_debug_emissive(const Color &p_color, float p_intensity) {
    pimpl->debug_emissive_color = p_color;
    pimpl->debug_emissive_intensity = p_intensity;
}
void NavigationRegion3DExt::get_debug_emissive(Color &r_color, float &r_intensity) const {
    r_color = pimpl->debug_emissive_color;
    r_intensity = pimpl->debug_emissive_intensity;
}

void NavigationRegion3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
}
int NavigationRegion3DExt::get_gi_mode() const { return pimpl->gi_mode; }
void NavigationRegion3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}
float NavigationRegion3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void NavigationRegion3DExt::sync_region() {
    pimpl->update_navigation_server();
    if (pimpl->debug_visible && !pimpl->baking) {
        // Regenerate debug mesh if needed (but we generate after bake)
    }
}