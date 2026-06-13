# File 217: modules/newton/SCsub
# SCons build script for the Newton Dynamics 4.0 module.
# Adds all source files to the Godot build.

Import('env')

# Core
env.add_source_files(env.modules_sources, "*.cpp")
env.add_source_files(env.modules_sources, "src/core/*.cpp")

# World
env.add_source_files(env.modules_sources, "src/world/*.cpp")

# Bodies
env.add_source_files(env.modules_sources, "src/bodies/*.cpp")

# Collision
env.add_source_files(env.modules_sources, "src/collision/*.cpp")

# Joints
env.add_source_files(env.modules_sources, "src/joints/*.cpp")

# Solver
env.add_source_files(env.modules_sources, "src/solver/*.cpp")

# Materials
env.add_source_files(env.modules_sources, "src/materials/*.cpp")

# Vehicles
env.add_source_files(env.modules_sources, "src/vehicles/*.cpp")

# Servers (PhysicsServer3D extension)
env.add_source_files(env.modules_sources, "src/servers/*.cpp")

# Utils
env.add_source_files(env.modules_sources, "src/utils/*.cpp")