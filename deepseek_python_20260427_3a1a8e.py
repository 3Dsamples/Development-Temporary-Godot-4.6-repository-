# File 308: modules/vienna/config.py
# Configuration for the ViennaPhysicsEngine module.
# This module is a pure C++ implementation with no external dependencies,
# relying only on Godot's built‑in math and threading.

def can_build(env, platform):
    # Vienna runs on all platforms supported by Godot.
    return True

def configure(env):
    # No special compiler flags required.
    pass