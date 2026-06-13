// File 50: modules/genesis/genesis_macros.h
// Configuration macros for integration with Godot 4.6

#ifndef GENESIS_MACROS_H
#define GENESIS_MACROS_H

// Enable logging for the genesis module (using Godot's print_verbose)
#define GENESIS_VERBOSE

// Optional: enable double precision for Genesis solvers (inherits Godot's real_t)
// If Godot is built with precision=double, all Genesis types will follow suit.

// Enable differentiable physics (requires custom tensor type)
// #define GENESIS_ENABLE_DIFFERENTIABLE

// Enable IPC coupling for deformable-rigid interactions
#define GENESIS_ENABLE_IPC_COUPLING

// Enable GPU-accelerated MPM (requires CUDA; handled by gaia/cuda_utilities)
#ifdef CUDA_ENABLED
#  define GENESIS_ENABLE_CUDA_MPM
#endif

#endif // GENESIS_MACROS_H