# File 264: modules/integration/SCsub
# SCons build script for the integration module that ties together
# Gaia, Genesis, and Newton Dynamics into one unified module.

Import('env')

env.add_source_files(env.modules_sources, "*.cpp")