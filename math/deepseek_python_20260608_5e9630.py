# File 355: modules/wicked/SCsub
Import('env')

env.add_source_files(env.modules_sources, "*.cpp")
env.add_source_files(env.modules_sources, "src/core/*.cpp")
env.add_source_files(env.modules_sources, "src/world/*.cpp")
env.add_source_files(env.modules_sources, "src/bodies/*.cpp")
env.add_source_files(env.modules_sources, "src/collision/*.cpp")
env.add_source_files(env.modules_sources, "src/joints/*.cpp")
env.add_source_files(env.modules_sources, "src/solver/*.cpp")
env.add_source_files(env.modules_sources, "src/materials/*.cpp")
env.add_source_files(env.modules_sources, "src/vehicles/*.cpp")
env.add_source_files(env.modules_sources, "src/servers/*.cpp")
env.add_source_files(env.modules_sources, "src/utils/*.cpp")