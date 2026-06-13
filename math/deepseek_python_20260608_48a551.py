# File 46: modules/gaia/SCsub

Import('env')

env.add_source_files(env.modules_sources, "*.cpp")
env.add_source_files(env.modules_sources, "src/*.cpp")
env.add_source_files(env.modules_sources, "src/bvh/*.cpp")
env.add_source_files(env.modules_sources, "src/collision_detector/*.cpp")
env.add_source_files(env.modules_sources, "src/framework/*.cpp")
env.add_source_files(env.modules_sources, "src/io/*.cpp")
env.add_source_files(env.modules_sources, "src/json/*.cpp")
env.add_source_files(env.modules_sources, "src/materials/*.cpp")
env.add_source_files(env.modules_sources, "src/pbd/*.cpp")
env.add_source_files(env.modules_sources, "src/vbd/*.cpp")
env.add_source_files(env.modules_sources, "src/parallelization/*.cpp")
env.add_source_files(env.modules_sources, "src/spatial_query/*.cpp")
env.add_source_files(env.modules_sources, "src/mesh/*.cpp")
env.add_source_files(env.modules_sources, "src/types/*.cpp")
env.add_source_files(env.modules_sources, "src/utility/*.cpp")
env.add_source_files(env.modules_sources, "src/viewer/*.cpp")

# Optionally enable CUDA support
if env['cuda_enabled']:
    env.add_source_files(env.modules_sources, "src/parallelization/*.cu")

# Link libraries if needed (e.g., CUDA)
if env['cuda_enabled']:
    env.Append(LIBS=['cudart'])