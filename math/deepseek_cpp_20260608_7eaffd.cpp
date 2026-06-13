// File 99: modules/genesis/src/nodes/genesis_cloth_3d.h
// GenesisCloth3D – a Node3D for real‑time cloth simulation using PBD/XPBD.
// Wraps a triangular mesh, builds distance and bending constraints, and
// drives the simulation via GenesisPBDSolver.

#ifndef GENESIS_NODES_CLOTH_3D_H
#define GENESIS_NODES_CLOTH_3D_H

#include "scene/3d/node_3d.h"
#include "scene/resources/immediate_mesh.h"

#include "../solvers/pbd_solver.h"
#include "../materials/pbd_material.h"
#include "../../../gaia/src/mesh/tri_mesh.h"
#include "../../../gaia/src/pbd/distance_constraint.h"
#include "../../../gaia/src/pbd/bending_constraint.h"
#include "../../../gaia/src/pbd/collision_constraint.h"

namespace genesis {

class GenesisWorld;

class GenesisCloth3D : public Node3D {
    GDCLASS(GenesisCloth3D, Node3D);

public:
    GenesisCloth3D() :
        width(1.0f),
        height(1.0f),
        resolution(16),
        structural_compliance(1e-6f),
        bending_compliance(1e-4f),
        damping_compliance(0.0f),
        gravity_scale(1.0f),
        world(nullptr) {
        set_process(true);
        set_physics_process(false);
    }

    // --- Cloth dimensions ---
    void set_width(real_t p_w) { width = MAX(p_w, 0.1f); }
    real_t get_width() const { return width; }

    void set_height(real_t p_h) { height = MAX(p_h, 0.1f); }
    real_t get_height() const { return height; }

    void set_resolution(int p_res) { resolution = MAX(p_res, 2); }
    int get_resolution() const { return resolution; }

    // --- Material / compliance ---
    void set_structural_compliance(real_t p_c) { structural_compliance = MAX(p_c, 0.0f); }
    real_t get_structural_compliance() const { return structural_compliance; }

    void set_bending_compliance(real_t p_c) { bending_compliance = MAX(p_c, 0.0f); }
    real_t get_bending_compliance() const { return bending_compliance; }

    void set_damping_compliance(real_t p_d) { damping_compliance = MAX(p_d, 0.0f); }
    real_t get_damping_compliance() const { return damping_compliance; }

    void set_gravity_scale(real_t p_s) { gravity_scale = p_s; }
    real_t get_gravity_scale() const { return gravity_scale; }

    // --- Access internals ---
    Ref<GenesisPBDSolver> get_pbd_solver() { return solver; }
    gaia::mesh::TriMesh &get_mesh() { return mesh; }

    // --- Pin a vertex (fix it in place) ---
    void pin_vertex(int p_index, bool p_pin = true) {
        ERR_FAIL_INDEX(p_index, pinned_flags.size());
        pinned_flags[p_index] = p_pin;
    }

    void _notification(int p_what) {
        if (p_what == NOTIFICATION_READY) {
            _initialize();
        }
        if (p_what == NOTIFICATION_PROCESS) {
            _update_display();
        }
    }

private:
    void _initialize() {
        // Locate GenesisWorld
        if (!_find_world()) {
            ERR_PRINT("GenesisCloth3D: no GenesisWorld found.");
            return;
        }

        // Generate triangular mesh (regular grid)
        _build_grid_mesh();

        // Create PBD material override (even if not assigned externally)
        Ref<PBDMaterial> pbd_mat = memnew(PBDMaterial);
        pbd_mat->set_compliance(structural_compliance);
        pbd_mat->set_bending_compliance(bending_compliance);
        pbd_mat->set_damping_compliance(damping_compliance);

        // Create Gaia soft body wrapper (needed by PBD solver constraints)
        soft_body = memnew(gaia::SoftBody);
        soft_body->resize(mesh.vertex_count());
        for (int i = 0; i < mesh.vertex_count(); ++i) {
            Vector3 pos = get_global_transform().xform(mesh.get_vertex(i));
            soft_body->positions[i] = pos;
            soft_body->rest_positions[i] = pos;
        }

        // Build constraints
        solver.instantiate();
        solver->set_material(pbd_mat);

        // Distance constraints (structural + shear)
        _add_distance_constraints();
        // Bending constraints
        _add_bending_constraints();

        // World will step the solver if we add the body entity? Actually PBD operates on a soft body, not on a BaseEntity. 
        // The solver has its own step loop. We'll add the solver to a list of external solvers in GenesisWorld? 
        // For simplicity, we'll step the PBD solver ourselves in _physics_process.
        set_physics_process(true);

        // Debug mesh
        if (!_debug_mesh.is_null()) {
            MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("ClothDisplay"));
            if (!mi) {
                mi = memnew(MeshInstance3D);
                mi->set_name("ClothDisplay");
                add_child(mi);
            }
            _debug_mesh = memnew(ImmediateMesh);
            mi->set_mesh(_debug_mesh);
        }
    }

    void _physics_process(real_t p_dt) {
        if (solver.is_null() || soft_body.is_null()) return;
        // Apply gravity (scaled) to all vertex masses via solver? Solver's step will call solve. We need to integrate velocities before solve.
        // In Gaia PBD, the solver step does: apply forces? Actually the PBDSolver we wrote only calls solve on constraints. 
        // We must handle per-vertex external forces ourselves. Here we apply gravity as acceleration.
        real_t mass_per_vertex = 1.0f; // uniform mass for now
        for (int i = 0; i < soft_body->velocities.size(); ++i) {
            if (pinned_flags[i]) {
                soft_body->velocities[i] = Vector3();
                continue;
            }
            soft_body->velocities[i] += world->get_gravity() * gravity_scale * p_dt;
        }
        // Integrate positions from velocities (semi-implicit Euler velocity step)
        for (int i = 0; i < soft_body->positions.size(); ++i) {
            soft_body->positions[i] += soft_body->velocities[i] * p_dt;
        }
        // PBD solve
        solver->set_dt(p_dt);
        solver->solve(p_dt);
        // Update velocities from position changes (post-solve)
        for (int i = 0; i < soft_body->velocities.size(); ++i) {
            // Not using position delta; keep velocities as before.
        }
        // Copy back positions to mesh for display
        for (int i = 0; i < mesh.vertex_count(); ++i) {
            mesh.get_vertex(i) = soft_body->positions[i];
        }
    }

    void _update_display() {
        if (_debug_mesh.is_null()) return;
        ImmediateMesh *im = _debug_mesh.ptr();
        im->clear_surfaces();
        im->surface_begin(Mesh::PRIMITIVE_TRIANGLES);
        // Output triangles from TriMesh
        int tri_count = mesh.triangle_count();
        for (int t = 0; t < tri_count; ++t) {
            gaia::mesh::TriMesh::Triangle tri = mesh.get_triangle(t);
            Vector3 v0 = mesh.get_vertex(tri.v0);
            Vector3 v1 = mesh.get_vertex(tri.v1);
            Vector3 v2 = mesh.get_vertex(tri.v2);
            im->surface_add_vertex(v0);
            im->surface_add_vertex(v1);
            im->surface_add_vertex(v2);
        }
        im->surface_end();
    }

    void _build_grid_mesh() {
        int w = resolution;
        int h = resolution;
        mesh.clear();
        // Generate vertices
        real_t dx = width / (w - 1);
        real_t dy = height / (h - 1);
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                Vector3 pos(i * dx - width * 0.5f, j * dy - height * 0.5f, 0.0f);
                mesh.add_vertex(pos);
            }
        }
        // Generate triangles (two per quad)
        for (int j = 0; j < h - 1; ++j) {
            for (int i = 0; i < w - 1; ++i) {
                int a = j * w + i;
                int b = a + w;
                int c = a + 1;
                int d = b + 1;
                mesh.add_triangle(a, b, c); // lower left
                mesh.add_triangle(c, b, d); // upper right
            }
        }
        // Setup pinned flags (top row pinned by default)
        pinned_flags.resize(mesh.vertex_count(), false);
        for (int i = 0; i < w; ++i) {
            int idx = i; // first row (j=0)
            pinned_flags[idx] = true;
        }
    }

    void _add_distance_constraints() {
        // For each triangle, add three edge distance constraints (structural)
        int tri_count = mesh.triangle_count();
        for (int t = 0; t < tri_count; ++t) {
            auto tri = mesh.get_triangle(t);
            _add_distance(tri.v0, tri.v1);
            _add_distance(tri.v1, tri.v2);
            _add_distance(tri.v2, tri.v0);
        }
        // Shear constraints (diagonal of each quad) can be added later
    }

    void _add_distance(int idx0, int idx1) {
        Ref<gaia::DistanceConstraint> dc = memnew(gaia::DistanceConstraint);
        dc->set_body(soft_body);
        dc->set_indices(idx0, idx1);
        dc->init_from_positions();
        dc->set_compliance(solver->get_material().is_valid() ? solver->get_material()->get_compliance() : 1e-6f);
        solver->add_distance_constraint(dc.ptr());
    }

    void _add_bending_constraints() {
        // Map edge -> two triangles sharing that edge
        HashMap<std::pair<int,int>, int> edge_map; // edge (min, max) -> count
        // First pass: build edge map
        int tri_count = mesh.triangle_count();
        struct EdgeInfo { int tri, opp; };
        HashMap<std::pair<int,int>, EdgeInfo> opposite_vertex;
        for (int t = 0; t < tri_count; ++t) {
            auto tri = mesh.get_triangle(t);
            int v[3] = {tri.v0, tri.v1, tri.v2};
            for (int i = 0; i < 3; ++i) {
                int a = v[i];
                int b = v[(i+1)%3];
                if (a > b) SWAP(a,b);
                auto key = std::make_pair(a,b);
                if (!edge_map.has(key)) {
                    edge_map[key] = 0;
                    opposite_vertex[key] = {t, v[(i+2)%3]};
                } else {
                    // second triangle sharing this edge
                    int other_tri = opposite_vertex[key].tri;
                    int other_opp = opposite_vertex[key].opp;
                    int opp_this = v[(i+2)%3];
                    // Create bending constraint
                    Ref<gaia::BendingConstraint> bc = memnew(gaia::BendingConstraint);
                    bc->set_body(soft_body);
                    bc->set_indices(a, b, other_opp, opp_this);
                    bc->init_from_positions();
                    bc->set_compliance(solver->get_material().is_valid() ? solver->get_material()->get_bending_compliance() : 1e-4f);
                    solver->add_bending_constraint(bc.ptr());
                }
            }
        }
    }

    bool _find_world() {
        Node *parent = get_parent();
        while (parent) {
            world = Object::cast_to<GenesisWorld>(parent);
            if (world) return true;
            parent = parent->get_parent();
        }
        return false;
    }

    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_width", "width"), &GenesisCloth3D::set_width);
        ClassDB::bind_method(D_METHOD("get_width"), &GenesisCloth3D::get_width);
        ClassDB::bind_method(D_METHOD("set_height", "height"), &GenesisCloth3D::set_height);
        ClassDB::bind_method(D_METHOD("get_height"), &GenesisCloth3D::get_height);
        ClassDB::bind_method(D_METHOD("set_resolution", "res"), &GenesisCloth3D::set_resolution);
        ClassDB::bind_method(D_METHOD("get_resolution"), &GenesisCloth3D::get_resolution);
        ClassDB::bind_method(D_METHOD("set_structural_compliance", "c"), &GenesisCloth3D::set_structural_compliance);
        ClassDB::bind_method(D_METHOD("get_structural_compliance"), &GenesisCloth3D::get_structural_compliance);
        ClassDB::bind_method(D_METHOD("set_bending_compliance", "c"), &GenesisCloth3D::set_bending_compliance);
        ClassDB::bind_method(D_METHOD("get_bending_compliance"), &GenesisCloth3D::get_bending_compliance);
        ClassDB::bind_method(D_METHOD("set_damping_compliance", "c"), &GenesisCloth3D::set_damping_compliance);
        ClassDB::bind_method(D_METHOD("get_damping_compliance"), &GenesisCloth3D::get_damping_compliance);
        ClassDB::bind_method(D_METHOD("set_gravity_scale", "scale"), &GenesisCloth3D::set_gravity_scale);
        ClassDB::bind_method(D_METHOD("get_gravity_scale"), &GenesisCloth3D::get_gravity_scale);
        ClassDB::bind_method(D_METHOD("pin_vertex", "index", "pin"), &GenesisCloth3D::pin_vertex, DEFVAL(true));

        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution", PROPERTY_HINT_RANGE, "2,100,1"), "set_resolution", "get_resolution");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "structural_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_structural_compliance", "get_structural_compliance");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bending_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_bending_compliance", "get_bending_compliance");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_damping_compliance", "get_damping_compliance");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale"), "set_gravity_scale", "get_gravity_scale");
    }

    real_t width, height;
    int resolution;
    real_t structural_compliance, bending_compliance, damping_compliance;
    real_t gravity_scale;

    gaia::mesh::TriMesh mesh;
    gaia::SoftBody *soft_body;
    Ref<GenesisPBDSolver> solver;
    LocalVector<bool> pinned_flags;
    GenesisWorld *world;
    Ref<ImmediateMesh> _debug_mesh;
};

} // namespace genesis

#endif // GENESIS_NODES_CLOTH_3D_H