# File 304: modules/vienna/SCsub
# SCons build script for the ViennaPhysicsEngine module.
# Compiles all .cpp files in the module’s source tree.

Import('env')

env.add_source_files(env.modules_sources, "*.cpp")
env.add_source_files(env.modules_sources, "src/core/*.cpp")
env.add_source_files(env.modules_sources, "src/world/*.cpp")
env.add_source_files(env.modules_sources, "src/bodies/*.cpp")
env.add_source_files(env.modules_sources, "src/collision/*.cpp")
env.add_source_files(env.modules_sources, "src/joints/*.cpp")
env.add_source_files(env.modules_sources, "src/solver/*.cpp")
env.add_source_files(env.modules_sources, "src/materials/*.cpp")
env.add_source_files(env.modules_sources, "src/cloth/*.cpp")
env.add_source_files(env.modules_sources, "src/particles/*.cpp")
env.add_source_files(env.modules_sources, "src/utils/*.cpp")
env.add_source_files(env.modules_sources, "src/servers/*.cpp")