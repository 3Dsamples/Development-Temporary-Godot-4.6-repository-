# File 225: modules/newton/config.py
# Configuration for building the Newton Dynamics module.
# Enables the module by default and allows optional settings.

def can_build(env, platform):
    # Newton is pure C++ with no external dependencies (uses Gaia narrow-phase)
    return True

def configure(env):
    # No special configuration required for basic compilation.
    pass