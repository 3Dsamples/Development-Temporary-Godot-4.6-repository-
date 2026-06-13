// File 95: modules/genesis/src/nodes/genesis_soft_body_3d.h
// GenesisSoftBody3D – a Node3D that wraps a FEMEntity for deformable solid simulation.
// Provides an interface similar to Godot's SoftBody3D but powered by Genesis FEM solver.

#ifndef GENESIS_NODES_SOFT_BODY_3D_H
#define GENESIS_NODES_SOFT_BODY_3D_H

#include "scene/3d/node_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/mesh.h"
#include "scene/resources/immediate_mesh.h"

#include "../entities/fem_entity.h"
#include "../materials/fem_material.h"
#include "../../../gaia/src/mesh/tet_mesh.h"
#include "../../../gaia/src/mesh/tri_mesh.h"
#include "../../../gaia/src/mesh/mesh_io.h"
#include "../core/genesis_types.h"

namespace genesis {

class GenesisWorld; // forward declared

class GenesisSoftBody3D : public Node3D {
	GDCLASS(GenesisSoftBody3D, Node3D);

public:
	GenesisSoftBody3D() :
		_mesh_path(""),
		_resolution(1.0),
		_subdivisions(0),
		_pressure_stiffness(0.0),
		_world(nullptr) {
		set_process(true);
	}

	// --- Mesh setup (load from .node/.ele or generate from a Godot Mesh) ---
	void set_mesh_path(const String &p_path) { _mesh_path = p_path; }
	String get_mesh_path() const { return _mesh_path; }

	void set_resolution(real_t p_res) { _resolution = MAX(p_res, 0.01); }
	real_t get_resolution() const { return _resolution; }

	void set_subdivisions(int p_subdiv) { _subdivisions = MAX(p_subdiv, 0); }
	int get_subdivisions() const { return _subdivisions; }

	void set_pressure_stiffness(real_t p_k) { _pressure_stiffness = MAX(p_k, 0.0); }
	real_t get_pressure_stiffness() const { return _pressure_stiffness; }

	// --- FEM material ---
	void set_fem_material(const Ref<FEMMaterial> &p_mat) { _material = p_mat; if (fem_entity.is_valid()) fem_entity->set_material(p_mat); }
	Ref<FEMMaterial> get_fem_material() const { return _material; }

	// --- Access FEM entity ---
	Ref<FEMEntity> get_fem_entity() { return fem_entity; }

	// --- Godot lifecycle ---
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			_initialize();
		}
		if (p_what == NOTIFICATION_PROCESS) {
			_sync_mesh_to_scene();
		}
	}

	// Scripting API: apply force to all vertices (gravity already handled by solver)
	void apply_force_to_vertices(const Vector3 &force) {
		if (fem_entity.is_valid()) {
			for (int i = 0; i < fem_entity->get_mesh().vertex_count(); ++i) {
				fem_entity->get_mesh().get_vertex(i); // can't apply force here directly; forces are accumulated via BaseEntity. But BaseEntity's apply_force is for rigid; FEM uses per-vertex forces. We'll need a separate force array. We'll add a method to FEMEntity later. For now, we store in a member and apply during step callback.
				_vertex_forces[i] += force;
			}
		}
	}

	void reset_rest_shape() {
		if (fem_entity.is_valid()) {
			gaia::mesh::TetMesh &mesh = fem_entity->get_mesh();
			for (int i = 0; i < mesh.vertex_count(); ++i) {
				mesh.get_vertex(i) = _rest_positions[i]; // restore
			}
			mesh.precompute_rest_state();
		}
	}

private:
	void _initialize() {
		// Find GenesisWorld
		if (!_find_world()) {
			ERR_PRINT("GenesisSoftBody3D: no GenesisWorld found.");
			return;
		}

		// Create FEM entity
		fem_entity.instantiate();
		fem_entity->set_entity_uid(_generate_uid());
		if (_material.is_valid()) fem_entity->set_material(_material);

		// Load or generate tet mesh
		gaia::mesh::TetMesh tet_mesh;
		if (!_mesh_path.is_empty()) {
			// Load TetGen .node/.ele pair
			gaia::mesh::MeshIO::load_tetgen(_mesh_path + ".node", _mesh_path + ".ele", tet_mesh);
		} else {
			// Generate a regular tetrahedral mesh from a cube (for testing)
			_generate_cube_tet_mesh(tet_mesh, _resolution, _subdivisions);
		}
		fem_entity->set_mesh(tet_mesh);
		fem_entity->get_mesh().precompute_rest_state();
		_rest_positions.resize(tet_mesh.vertex_count());
		for (int i = 0; i < tet_mesh.vertex_count(); ++i)
			_rest_positions[i] = tet_mesh.get_vertex(i);
		_vertex_forces.resize(tet_mesh.vertex_count(), Vector3());

		fem_entity->set_transform(get_global_transform());
		fem_entity->set_ipc_enabled(false); // can be set later
		fem_entity->set_plasticity_enabled(_material.is_valid() && _material->is_plasticity_enabled());

		// Register with world
		world->add_entity(fem_entity);

		// Create a debug mesh to display
		_create_display_mesh();
	}

	void _sync_mesh_to_scene() {
		if (fem_entity.is_null()) return;
		// Update display mesh vertex positions from current FEM state
		gaia::mesh::TetMesh &tet = fem_entity->get_mesh();
		if (_debug_mesh.is_null()) _create_display_mesh();
		// We need to update the ImmediateMesh (or ArrayMesh) vertices.
		// We'll rebuild the immediate mesh each frame (inefficient but fine for debug).
		ImmediateMesh *im = _debug_mesh.ptr();
		im->clear_surfaces();
		// Render triangles from tetrahedron faces? For now, just render edges of all tetrahedra using lines.
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		for (int el = 0; el < tet.element_count(); ++el) {
			TetMesh::Tetrahedron t = tet.get_tetrahedron(el);
			Vector3 v0 = tet.get_vertex(t.v0);
			Vector3 v1 = tet.get_vertex(t.v1);
			Vector3 v2 = tet.get_vertex(t.v2);
			Vector3 v3 = tet.get_vertex(t.v3);
			// 6 edges
			im->surface_add_vertex(v0); im->surface_add_vertex(v1);
			im->surface_add_vertex(v0); im->surface_add_vertex(v2);
			im->surface_add_vertex(v0); im->surface_add_vertex(v3);
			im->surface_add_vertex(v1); im->surface_add_vertex(v2);
			im->surface_add_vertex(v1); im->surface_add_vertex(v3);
			im->surface_add_vertex(v2); im->surface_add_vertex(v3);
		}
		im->surface_end();
	}

	void _create_display_mesh() {
		// Create a MeshInstance3D child if not present, or we can set the node's own mesh.
		// For simplicity, we will use the node's own mesh (Node3D can't have mesh, so we need a MeshInstance3D child).
		// We'll assume there's a MeshInstance3D child named "DisplayMesh".
		MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("DisplayMesh"));
		if (!mi) {
			mi = memnew(MeshInstance3D);
			mi->set_name("DisplayMesh");
			add_child(mi);
		}
		Ref<ImmediateMesh> im = memnew(ImmediateMesh);
		mi->set_mesh(im);
		_debug_mesh = im;
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

	uint64_t _generate_uid() { return (uint64_t(get_instance_id()) << 16) | uint64_t(Math::rand() & 0xFFFF); }

	// Generate a cube tetrahedral mesh
	void _generate_cube_tet_mesh(gaia::mesh::TetMesh &mesh, real_t size, int subdivisions) {
		// Not fully implemented; create a simple TET cube (5 points) for demonstration.
		// Better: use Gaia's mesh generation (not defined) so we'll just create a single tetrahedron as placeholder.
		mesh.clear();
		mesh.add_vertex(Vector3(0, 0, 0));
		mesh.add_vertex(Vector3(size, 0, 0));
		mesh.add_vertex(Vector3(0, size, 0));
		mesh.add_vertex(Vector3(0, 0, size));
		mesh.add_tetrahedron(0, 1, 2, 3);
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_mesh_path", "path"), &GenesisSoftBody3D::set_mesh_path);
		ClassDB::bind_method(D_METHOD("get_mesh_path"), &GenesisSoftBody3D::get_mesh_path);
		ClassDB::bind_method(D_METHOD("set_resolution", "res"), &GenesisSoftBody3D::set_resolution);
		ClassDB::bind_method(D_METHOD("get_resolution"), &GenesisSoftBody3D::get_resolution);
		ClassDB::bind_method(D_METHOD("set_subdivisions", "subdiv"), &GenesisSoftBody3D::set_subdivisions);
		ClassDB::bind_method(D_METHOD("get_subdivisions"), &GenesisSoftBody3D::get_subdivisions);
		ClassDB::bind_method(D_METHOD("set_pressure_stiffness", "k"), &GenesisSoftBody3D::set_pressure_stiffness);
		ClassDB::bind_method(D_METHOD("get_pressure_stiffness"), &GenesisSoftBody3D::get_pressure_stiffness);
		ClassDB::bind_method(D_METHOD("set_fem_material", "material"), &GenesisSoftBody3D::set_fem_material);
		ClassDB::bind_method(D_METHOD("get_fem_material"), &GenesisSoftBody3D::get_fem_material);
		ClassDB::bind_method(D_METHOD("apply_force_to_vertices", "force"), &GenesisSoftBody3D::apply_force_to_vertices);
		ClassDB::bind_method(D_METHOD("reset_rest_shape"), &GenesisSoftBody3D::reset_rest_shape);

		ADD_PROPERTY(PropertyInfo(Variant::STRING, "mesh_path", PROPERTY_HINT_FILE, "*.node"), "set_mesh_path", "get_mesh_path");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "resolution", PROPERTY_HINT_RANGE, "0.01,10,0.01"), "set_resolution", "get_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "subdivisions", PROPERTY_HINT_RANGE, "0,5,1"), "set_subdivisions", "get_subdivisions");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pressure_stiffness", PROPERTY_HINT_RANGE, "0,10000,0.1"), "set_pressure_stiffness", "get_pressure_stiffness");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fem_material", PROPERTY_HINT_RESOURCE_TYPE, "FEMMaterial"), "set_fem_material", "get_fem_material");
	}

	Ref<FEMEntity> fem_entity;
	Ref<FEMMaterial> _material;
	String _mesh_path;
	real_t _resolution;
	int _subdivisions;
	real_t _pressure_stiffness;
	GenesisWorld *world;
	LocalVector<Vector3> _rest_positions;
	LocalVector<Vector3> _vertex_forces;
	Ref<ImmediateMesh> _debug_mesh;
};

} // namespace genesis

#endif // GENESIS_NODES_SOFT_BODY_3D_H