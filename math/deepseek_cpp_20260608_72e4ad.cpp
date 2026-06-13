// File 254: modules/newton/src/nodes/newton_world_node.h
// NewtonWorldNode – a Godot Node3D that owns a NewtonWorld instance,
// adds it to the scene tree, and steps the physics every frame in
// _physics_process.  This allows easy drop‑in usage without a separate
// physics server.

#ifndef NEWTON_NODES_WORLD_NODE_H
#define NEWTON_NODES_WORLD_NODE_H

#include "scene/3d/node_3d.h"
#include "../world/newton_world.h"
#include "../world/newton_world_config.h"

namespace newton {

class NewtonWorldNode : public Node3D {
	GDCLASS(NewtonWorldNode, Node3D);

public:
	NewtonWorldNode();
	~NewtonWorldNode();

	// Access the underlying NewtonWorld (for adding bodies etc.).
	NewtonWorld *get_newton_world() const { return world; }

	// Set configuration resource.
	void set_config(const Ref<NewtonWorldConfig> &p_config);
	Ref<NewtonWorldConfig> get_config() const;

	// Start / stop physics.
	void set_active(bool p_active);
	bool is_active() const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

private:
	void _apply_config();

	NewtonWorld *world;
	Ref<NewtonWorldConfig> config;
	bool active;
};

} // namespace newton

#endif // NEWTON_NODES_WORLD_NODE_H