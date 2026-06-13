// File 266: modules/integration/unified_physics_server_factory.cpp
// Factory that registers the UnifiedPhysicsServer3D as the default physics
// server for Godot.  It reads the project setting "physics/3d/physics_engine"
// and creates the appropriate server.  If "Unified" is selected, it creates
// an instance of UnifiedPhysicsServer3D.

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "servers/physics_server_3d.h"
#include "unified_physics_server_3d.h"

static PhysicsServer3D *_create_unified_server() {
	return memnew(unified::UnifiedPhysicsServer3D);
}

void register_unified_physics_server() {
	// Register as a known engine
	GLOBAL_DEF("physics/3d/physics_engine", "GodotPhysics3D");
	ProjectSettings::get_singleton()->set_custom_property_info(
		"physics/3d/physics_engine",
		PropertyInfo(Variant::STRING, "physics/3d/physics_engine", PROPERTY_HINT_ENUM, "GodotPhysics3D,UnifiedPhysics3D"));

	// If the project setting is "UnifiedPhysics3D", override the factory
	String engine = GLOBAL_GET("physics/3d/physics_engine");
	if (engine == "UnifiedPhysics3D") {
		PhysicsServer3DManager::get_singleton()->register_server("UnifiedPhysics3D", _create_unified_server);
		PhysicsServer3DManager::get_singleton()->set_default_server("UnifiedPhysics3D");
	}
}