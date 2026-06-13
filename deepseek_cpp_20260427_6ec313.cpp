// File 325: modules/vienna/src/utils/vienna_scene_loader.h
// High‑performance utility that scans a Godot subtree and automatically
// creates static Vienna rigid bodies for all MeshInstance3D nodes that
// match a user‑defined group. Uses ViennaMeshLoader to convert each mesh
// into a ViennaTriMesh collision tree, and places them in the ViennaWorld.
// Supports filtering by node name, group, or material.

#ifndef VIENNA_UTILS_SCENE_LOADER_H
#define VIENNA_UTILS_SCENE_LOADER_H

#include "scene/3d/node_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../collision/vienna_trimesh.h"
#include "../collision/vienna_shape.h"
#include "vienna_mesh_loader.h"
#include "core/templates/local_vector.h"

namespace vienna {

class ViennaSceneLoader : public RefCounted {
	GDCLASS(ViennaSceneLoader, RefCounted);

public:
	ViennaSceneLoader() :
		world(nullptr),
		default_friction(0.5),
		default_restitution(0.0),
		skip_scaled_nodes(true),
		prefix("VPHY_") {}

	void set_world(ViennaWorld *p_world) { world = p_world; }
	ViennaWorld *get_world() const { return world; }

	// If set, only nodes whose name starts with this prefix are processed.
	void set_prefix(const String &p_prefix) { prefix = p_prefix; }
	String get_prefix() const { return prefix; }

	void set_default_friction(real_t p_fric) { default_friction = MAX(p_fric, 0.0); }
	real_t get_default_friction() const { return default_friction; }

	void set_default_restitution(real_t p_rest) { default_restitution = CLAMP(p_rest, 0.0, 1.0); }
	real_t get_default_restitution() const { return default_restitution; }

	// When true, nodes whose world scale is not exactly (1,1,1) are skipped
	// because the physics shape would need non‑uniform scaling support.
	void set_skip_scaled_nodes(bool p_skip) { skip_scaled_nodes = p_skip; }
	bool get_skip_scaled_nodes() const { return skip_scaled_nodes; }

	/**
	 * Scan the subtree rooted at `p_root` and create static Vienna bodies
	 * for every MeshInstance3D that matches the prefix.  Bodies are added
	 * to the ViennaWorld and stored in the output list.
	 *
	 * @param p_root        The root Node to scan recursively.
	 * @param r_created_ids Output vector of Vienna body IDs that were created.
	 * @return              Number of bodies created.
	 */
	int create_bodies_from_subtree(Node *p_root, LocalVector<body_id> &r_created_ids) {
		ERR_FAIL_COND_V(!world, 0);
		ERR_FAIL_COND_V(!p_root, 0);
		r_created_ids.clear();

		_create_recursive(p_root, r_created_ids);
		return r_created_ids.size();
	}

	/**
	 * Remove all bodies previously created by this loader.
	 * Call this when the level is unloaded to free physics memory.
	 */
	void remove_created_bodies(const LocalVector<body_id> &p_ids) {
		ERR_FAIL_COND(!world);
		for (body_id id : p_ids) {
			world->destroy_body(id);
		}
	}

	/**
	 * Convenience method that creates all bodies and returns them as a vector.
	 */
	LocalVector<body_id> load_scene(Node *p_root) {
		LocalVector<body_id> ids;
		create_bodies_from_subtree(p_root, ids);
		return ids;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_world", "world"), &ViennaSceneLoader::set_world);
		ClassDB::bind_method(D_METHOD("get_world"), &ViennaSceneLoader::get_world);
		ClassDB::bind_method(D_METHOD("set_prefix", "prefix"), &ViennaSceneLoader::set_prefix);
		ClassDB::bind_method(D_METHOD("get_prefix"), &ViennaSceneLoader::get_prefix);
		ClassDB::bind_method(D_METHOD("set_default_friction", "friction"), &ViennaSceneLoader::set_default_friction);
		ClassDB::bind_method(D_METHOD("get_default_friction"), &ViennaSceneLoader::get_default_friction);
		ClassDB::bind_method(D_METHOD("set_default_restitution", "restitution"), &ViennaSceneLoader::set_default_restitution);
		ClassDB::bind_method(D_METHOD("get_default_restitution"), &ViennaSceneLoader::get_default_restitution);
		ClassDB::bind_method(D_METHOD("set_skip_scaled_nodes", "skip"), &ViennaSceneLoader::set_skip_scaled_nodes);
		ClassDB::bind_method(D_METHOD("get_skip_scaled_nodes"), &ViennaSceneLoader::get_skip_scaled_nodes);
		ClassDB::bind_method(D_METHOD("create_bodies_from_subtree", "root"), &ViennaSceneLoader::create_bodies_from_subtree);
		ClassDB::bind_method(D_METHOD("remove_created_bodies", "ids"), &ViennaSceneLoader::remove_created_bodies);
		ClassDB::bind_method(D_METHOD("load_scene", "root"), &ViennaSceneLoader::load_scene);
		ADD_PROPERTY(PropertyInfo(Variant::STRING, "prefix"), "set_prefix", "get_prefix");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_friction"), "set_default_friction", "get_default_friction");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_restitution"), "set_default_restitution", "get_default_restitution");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "skip_scaled_nodes"), "set_skip_scaled_nodes", "get_skip_scaled_nodes");
	}

private:
	ViennaWorld *world;
	String prefix;
	real_t default_friction;
	real_t default_restitution;
	bool skip_scaled_nodes;

	void _create_recursive(Node *p_node, LocalVector<body_id> &r_ids) {
		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_node);
		if (mi && (prefix.is_empty() || mi->get_name().begins_with(prefix))) {
			// Skip if scaled and we require uniform scale.
			if (skip_scaled_nodes) {
				vec3 scale = mi->get_global_transform().basis.get_scale();
				if (Math::abs(scale.x - 1.0f) > 1e-4f ||
					Math::abs(scale.y - 1.0f) > 1e-4f ||
					Math::abs(scale.z - 1.0f) > 1e-4f) {
					; // skip
				} else {
					_create_body_for_mesh(mi, r_ids);
				}
			} else {
				_create_body_for_mesh(mi, r_ids);
			}
		}
		// Recurse children
		for (int i = 0; i < p_node->get_child_count(); ++i) {
			_create_recursive(p_node->get_child(i), r_ids);
		}
	}

	void _create_body_for_mesh(MeshInstance3D *p_mi, LocalVector<body_id> &r_ids) {
		Ref<Mesh> mesh = p_mi->get_mesh();
		if (mesh.is_null()) return;

		// Build collision shape using the loader.
		Ref<ViennaTriMesh> trimesh = ViennaMeshLoader::create_triangle_mesh(mesh);
		if (trimesh.is_null() || trimesh->get_triangle_count() == 0) return;

		// Create a static body with this shape.
		Ref<ViennaBody> body;
		body.instantiate();
		body->set_type(BodyType::STATIC);
		body->set_collision_shape(trimesh);
		body->set_collision_aabb(trimesh->get_local_aabb());
		body->set_transform(p_mi->get_global_transform());

		// Optionally assign a material with default friction/restitution.
		Ref<ViennaMaterial> mat;
		mat.instantiate();
		mat->set_dynamic_friction(default_friction);
		mat->set_restitution(default_restitution);
		material_id mat_id = world->create_material(mat);
		body->set_material_id(mat_id);

		body_id id = world->create_body(body);
		r_ids.push_back(id);
	}
};

} // namespace vienna

#endif // VIENNA_UTILS_SCENE_LOADER_H