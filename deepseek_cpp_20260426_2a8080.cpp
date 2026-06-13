// File 89: modules/genesis/SCsub

Import('env')

env.add_source_files(env.modules_sources, "*.cpp")
env.add_source_files(env.modules_sources, "src/*.cpp")
env.add_source_files(env.modules_sources, "src/core/*.cpp")
env.add_source_files(env.modules_sources, "src/options/*.cpp")
env.add_source_files(env.modules_sources, "src/materials/*.cpp")
env.add_source_files(env.modules_sources, "src/entities/*.cpp")
env.add_source_files(env.modules_sources, "src/solvers/*.cpp")
env.add_source_files(env.modules_sources, "src/collision/*.cpp")
env.add_source_files(env.modules_sources, "src/constraints/*.cpp")
env.add_source_files(env.modules_sources, "src/boundaries/*.cpp")
env.add_source_files(env.modules_sources, "src/sensors/*.cpp")
env.add_source_files(env.modules_sources, "src/states/*.cpp")
env.add_source_files(env.modules_sources, "src/grad/*.cpp")