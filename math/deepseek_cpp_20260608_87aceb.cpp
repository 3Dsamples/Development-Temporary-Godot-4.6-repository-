// File 306: modules/vienna/src/nodes/vienna_world_node_3d.h
// ViennaWorldNode3D – a Godot Node3D that owns and steps a ViennaWorld
// each physics frame.  Provides convenient access to add bodies, joints,
// cloth, particles, and debug drawing.  Suitable for standalone Vienna physics.

#ifndef VIENNA_NODES_WORLD_NODE_3D_H
#define VIENNA_NODES_WORLD_NODE_3D_H

#include "scene/3d/node_3d.h"
#include "../world/vienna_world.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaWorldNode3D : public Node3D {
	GDCLASS(ViennaWorldNode3D, Node3D);

public:
	ViennaWorldNode3D();
	virtual ~ViennaWorldNode3D();

	// Access the ViennaWorld (for adding bodies, joints, etc.)
	ViennaWorld *get_vienna_world() const { return world; }

	// Physics parameters
	void set_gravity(const vec3 &p_gravity);
	vec3 get_gravity() const;

	void set_solver_iterations(int p_iters);
	int get_solver_iterations() const;

	void set_sleep_frames(int p_frames);
	int get_sleep_frames() const;

	void set_active(bool p_active);
	bool is_active() const;

	// Expose configuration resource (future extension)
	void set_config(const Ref<Resource> &p_config);
	Ref<Resource> get_config() const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

private:
	ViennaWorld *world;
	bool active;
	vec3 gravity;
	int solver_iterations;
	int sleep_frames;
	Ref<Resource> config;
};

} // namespace vienna

#endif // VIENNA_NODES_WORLD_NODE_3D_H