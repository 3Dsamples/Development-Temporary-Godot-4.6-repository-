// File 320: modules/vienna/src/servers/vienna_physics_server_factory.cpp
// Factory function that registers ViennaPhysicsServer3D as an alternative
// physics engine in Godot 4.6. When the project setting
// "physics/3d/physics_engine" is set to "ViennaPhysics3D", the engine will
// use the Vienna physics backend instead of the default GodotPhysics3D.

#include "vienna_physics_server_3d.h"
#include "servers/physics_server_3d.h"
#include "core/config/project_settings.h"

static PhysicsServer3D *_create_vienna_server() {
    return memnew(vienna::ViennaPhysicsServer3D);
}

void register_vienna_physics_server() {
    // Define the project setting if not already present.
    GLOBAL_DEF("physics/3d/physics_engine", "GodotPhysics3D");
    ProjectSettings::get_singleton()->set_custom_property_info(
        "physics/3d/physics_engine",
        PropertyInfo(Variant::STRING,
                     "physics/3d/physics_engine",
                     PROPERTY_HINT_ENUM,
                     "GodotPhysics3D,UnifiedPhysics3D,ViennaPhysics3D"));

    // Register the Vienna server with the engine.
    PhysicsServer3DManager *manager = PhysicsServer3DManager::get_singleton();
    manager->register_server("ViennaPhysics3D", _create_vienna_server);

    String selected = GLOBAL_GET("physics/3d/physics_engine");
    if (selected == "ViennaPhysics3D") {
        manager->set_default_server("ViennaPhysics3D");
    }
}