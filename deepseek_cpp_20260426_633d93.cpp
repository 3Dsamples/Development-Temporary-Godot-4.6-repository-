// File 98: modules/genesis/src/nodes/genesis_mpm_3d.h
// GenesisMPM3D – a Node3D that hosts an MPMEntity and drives its simulation
// using MPMSolver. It can load a tetrahedral mesh (converted to particles)
// or generate particles from a geometric primitive.

#ifndef GENESIS_NODES_MPM_3D_H
#define GENESIS_NODES_MPM_3D_H

#include "scene/3d/node_3d.h"
#include "scene/resources/immediate_mesh.h"
#include "../entities/mpm_entity.h"
#include "../solvers/mpm_solver.h"
#include "../materials/mpm_material.h"
#include "../../../gaia/src/mesh/tet_mesh.h"
#include "../../../gaia/src/mesh/mesh_io.h"

namespace genesis {

class GenesisWorld;

class GenesisMPM3D : public Node3D {
    GDCLASS(GenesisMPM3D, Node3D);

public:
    GenesisMPM3D() :
        _grid_resolution(32),
        _cell_size(0.05),
        _particle_spacing(0.02),
        _world(nullptr) {
        set_process(true);
        set_physics_process(false); // solver steered by world
    }

    // --- Grid / particle settings ---
    void set_grid_resolution(int p_res) { _grid_resolution = MAX(p_res, 4); }
    int get_grid_resolution() const { return _grid_resolution; }

    void set_cell_size(real_t p_dx) { _cell_size = MAX(p_dx, 1e-6); }
    real_t get_cell_size() const { return _cell_size; }

    void set_particle_spacing(real_t p_sp) { _particle_spacing = MAX(p_sp, 1e-6); }
    real_t get_particle_spacing() const { return _particle_spacing; }

    // --- Material ---
    void set_mpm_material(const Ref<MPMMaterial> &p_mat) {
        _material = p_mat;
        if (_material.is_valid()) {
            // sync with solver if already created
            if (_mpm_solver.is_valid()) _mpm_solver->set_material(_material);
        }
    }
    Ref<MPMMaterial> get_mpm_material() const { return _material; }

    // --- Entity and solver access ---
    Ref<MPMEntity> get_mpm_entity() { return _mpm_entity; }
    Ref<MPMSolver> get_mpm_solver() { return _mpm_solver; }

    // --- Generate particles inside a box ---
    void emit_particles_box(const AABB &p_box) {
        // Fill box with particles at `particle_spacing` intervals
        Vector3 origin = p_box.position;
        Vector3 size = p_box.size;
        _mpm_entity->clear_particles();
        real_t mass_per_particle = 1.0; // density will set later
        for (real_t x = origin.x; x < origin.x + size.x; x += _particle_spacing) {
            for (real_t y = origin.y; y < origin.y + size.y; y += _particle_spacing) {
                for (real_t z = origin.z; z < origin.z + size.z; z += _particle_spacing) {
                    _mpm_entity->add_particle(Vector3(x, y, z), Vector3(), mass_per_particle, _particle_spacing * _particle_spacing * _particle_spacing);
                }
            }
        }
    }

    // --- Load tetrahedral mesh and convert to particles (each tetrahedron → 1 particle at centroid?) ---
    void load_tetmesh_as_particles(const String &p_node_path, const String &p_ele_path) {
        gaia::mesh::TetMesh tm;
        gaia::mesh::MeshIO::load_tetgen(p_node_path, p_ele_path, tm);
        _mpm_entity->clear_particles();
        real_t total_volume = 0;
        for (int i = 0; i < tm.element_count(); ++i) total_volume += tm.get_rest_volume(i);
        real_t mass_per_tet = (_material.is_valid() ? _material->get_density() * total_volume : total_volume) / tm.element_count();
        for (int i = 0; i < tm.element_count(); ++i) {
            TetMesh::Tetrahedron tet = tm.get_tetrahedron(i);
            Vector3 centroid = (tm.get_vertex(tet.v0) + tm.get_vertex(tet.v1) + tm.get_vertex(tet.v2) + tm.get_vertex(tet.v3)) / 4.0;
            _mpm_entity->add_particle(centroid, Vector3(), mass_per_tet, tm.get_rest_volume(i));
        }
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
        if (!_find_world()) {
            ERR_PRINT("GenesisMPM3D: no GenesisWorld found.");
            return;
        }
        // Create MPM entity
        _mpm_entity.instantiate();
        _mpm_entity->set_entity_uid(_generate_uid());
        _mpm_entity->set_grid_resolution(_grid_resolution);
        _mpm_entity->set_cell_size(_cell_size);

        // Create MPM solver and attach to entity
        _mpm_solver.instantiate();
        _mpm_solver->set_grid_resolution(_grid_resolution, _grid_resolution, _grid_resolution);
        _mpm_solver->set_grid_cell_size(_cell_size);
        _mpm_solver->add_entity(_mpm_entity);
        if (_material.is_valid()) _mpm_solver->set_material(_material);

        // Register entity with world (solvers will be stepped separately, or we can add solver to world's external list – for now we'll step internally)
        _world->add_entity(_mpm_entity);
        // World doesn't step MPM solver by default; we'll step it in our own _physics_process
        set_physics_process(true);

        // Create display mesh (points or small cubes)
        if (!_debug_mesh.is_null()) {
            MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("MPMDisplay"));
            if (!mi) {
                mi = memnew(MeshInstance3D);
                mi->set_name("MPMDisplay");
                add_child(mi);
            }
            _debug_mesh = memnew(ImmediateMesh);
            mi->set_mesh(_debug_mesh);
        }
    }

    void _physics_process(real_t p_dt) {
        if (_mpm_solver.is_null()) return;
        _mpm_solver->set_dt(p_dt);
        _mpm_solver->step();
    }

    void _update_display() {
        if (_debug_mesh.is_null() || _mpm_entity.is_null()) return;
        ImmediateMesh *im = _debug_mesh.ptr();
        im->clear_surfaces();
        const LocalVector<MPMEntity::Particle> &parts = _mpm_entity->get_particles();
        im->surface_begin(Mesh::PRIMITIVE_POINTS);
        for (const MPMEntity::Particle &p : parts) {
            im->surface_add_vertex(p.position);
            // No per-vertex color in POINTS mode? We can set a global color later via material.
        }
        im->surface_end();
    }

    bool _find_world() {
        Node *parent = get_parent();
        while (parent) {
            _world = Object::cast_to<GenesisWorld>(parent);
            if (_world) return true;
            parent = parent->get_parent();
        }
        return false;
    }

    uint64_t _generate_uid() { return (uint64_t(get_instance_id()) << 16) | uint64_t(Math::rand() & 0xFFFF); }

    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_grid_resolution", "res"), &GenesisMPM3D::set_grid_resolution);
        ClassDB::bind_method(D_METHOD("get_grid_resolution"), &GenesisMPM3D::get_grid_resolution);
        ClassDB::bind_method(D_METHOD("set_cell_size", "dx"), &GenesisMPM3D::set_cell_size);
        ClassDB::bind_method(D_METHOD("get_cell_size"), &GenesisMPM3D::get_cell_size);
        ClassDB::bind_method(D_METHOD("set_particle_spacing", "spacing"), &GenesisMPM3D::set_particle_spacing);
        ClassDB::bind_method(D_METHOD("get_particle_spacing"), &GenesisMPM3D::get_particle_spacing);
        ClassDB::bind_method(D_METHOD("set_mpm_material", "material"), &GenesisMPM3D::set_mpm_material);
        ClassDB::bind_method(D_METHOD("get_mpm_material"), &GenesisMPM3D::get_mpm_material);
        ClassDB::bind_method(D_METHOD("emit_particles_box", "box"), &GenesisMPM3D::emit_particles_box);
        ClassDB::bind_method(D_METHOD("load_tetmesh_as_particles", "node_path", "ele_path"), &GenesisMPM3D::load_tetmesh_as_particles);

        ADD_PROPERTY(PropertyInfo(Variant::INT, "grid_resolution", PROPERTY_HINT_RANGE, "4,256,1"), "set_grid_resolution", "get_grid_resolution");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cell_size", PROPERTY_HINT_RANGE, "0.001,10,0.001"), "set_cell_size", "get_cell_size");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "particle_spacing", PROPERTY_HINT_RANGE, "0.001,1,0.001"), "set_particle_spacing", "get_particle_spacing");
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mpm_material", PROPERTY_HINT_RESOURCE_TYPE, "MPMMaterial"), "set_mpm_material", "get_mpm_material");
    }

    int _grid_resolution;
    real_t _cell_size;
    real_t _particle_spacing;
    Ref<MPMMaterial> _material;
    Ref<MPMEntity> _mpm_entity;
    Ref<MPMSolver> _mpm_solver;
    GenesisWorld *_world;
    Ref<ImmediateMesh> _debug_mesh;
};

} // namespace genesis

#endif // GENESIS_NODES_MPM_3D_H