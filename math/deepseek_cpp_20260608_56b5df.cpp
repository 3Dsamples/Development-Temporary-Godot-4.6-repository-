// File 47: modules/gaia/gaia.h

#ifndef GAIA_H
#define GAIA_H

// Umbrella header that includes the entire Gaia physics library.
// Add this single include to your Godot project to use all Gaia modules.

// BVH
#include "src/bvh/aabb.h"
#include "src/bvh/bvh.h"
#include "src/bvh/morton_code.h"
#include "src/bvh/query.h"

// Collision detection
#include "src/collision_detector/broad_phase.h"
#include "src/collision_detector/narrow_phase.h"
#include "src/collision_detector/contact.h"
#include "src/collision_detector/collision_object.h"

// Framework
#include "src/framework/sim_framework.h"
#include "src/framework/world.h"
#include "src/framework/body.h"
#include "src/framework/constraint.h"
#include "src/framework/solver.h"

// IO
#include "src/io/io.h"
#include "src/io/parameter_reader.h"
#include "src/io/parameter_writer.h"

// JSON
#include "src/json/json_parser.h"

// Materials
#include "src/materials/material.h"
#include "src/materials/material_library.h"

// PBD
#include "src/pbd/pbd_solver.h"
#include "src/pbd/distance_constraint.h"
#include "src/pbd/bending_constraint.h"
#include "src/pbd/volume_constraint.h"
#include "src/pbd/collision_constraint.h"

// VBD
#include "src/vbd/vbd_solver.h"
#include "src/vbd/vbd_constraint.h"
#include "src/vbd/vbd_element.h"

// Parallelization
#include "src/parallelization/thread_pool.h"
#include "src/parallelization/cuda_utilities.h"
#include "src/parallelization/parallel_sort.h"

// Spatial Query
#include "src/spatial_query/spatial_hash.h"
#include "src/spatial_query/neighbour_query.h"

// Mesh
#include "src/mesh/tet_mesh.h"
#include "src/mesh/tri_mesh.h"
#include "src/mesh/mesh_io.h"
#include "src/mesh/mesh_quality.h"

// Types
#include "src/types/vector_types.h"
#include "src/types/matrix_types.h"
#include "src/types/constants.h"

// Utility
#include "src/utility/math_utils.h"
#include "src/utility/logger.h"
#include "src/utility/timer.h"

// Viewer
#include "src/viewer/debug_draw.h"

#endif // GAIA_H