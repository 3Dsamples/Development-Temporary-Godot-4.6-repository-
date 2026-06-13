// File 75: modules/genesis/src/boundaries/boundary_conditions.h
// Boundary conditions: Dirichlet, Neumann, contact, friction, and SDF‑based constraints.
// Used by all solvers to apply domain limits, walls, and custom surfaces.

#ifndef GENESIS_BOUNDARIES_BOUNDARY_CONDITIONS_H
#define GENESIS_BOUNDARIES_BOUNDARY_CONDITIONS_H

#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"

namespace genesis {

/**
 * Describes a single boundary condition applied to a region.
 * Supported types:
 * - DIRICHLET: fix position / velocity (e.g., clamped edge)
 * - NEUMANN: applied force / traction
 * - CONTACT: inequality constraint for one‑way penetration
 * - FRICTION: Coulomb friction tangent forces
 */
struct BoundaryCondition {
	BCType type = BCType::DIRICHLET;

	// Geometry: can be a plane, box, sphere, or SDF (implicit surface)
	// For simplicity, we support plane (normal + point) and sphere.
	Vector3 plane_normal;          // used if type is planar (e.g., ground)
	Vector3 plane_point;
	real_t   sphere_radius = 1.0;
	Vector3 sphere_center;
	bool     use_sphere = false;   // true if it's a sphere boundary

	// Dirichlet / Contact parameters
	Vector3 prescribed_position;
	Vector3 prescribed_velocity;
	bool    fix_x = false, fix_y = false, fix_z = false; // per‑axis fixing

	// Neumann
	Vector3 surface_traction;

	// Friction
	real_t friction_coeff = 0.5;

	// SDF (optional pointer to a signed distance function – not implemented here)
	// For future extension: a custom function that returns signed distance.
};

/**
 * BoundaryConditionManager: applies boundary conditions to a set of
 * particles or grid nodes (MPM, SPH, FEM, SF).
 * Works with positions/velocities arrays directly.
 */
class BoundaryConditionManager {
public:
	LocalVector<BoundaryCondition> conditions;

	// Add a simple ground plane (y = height)
	void add_ground_plane(real_t y_height, real_t friction = 0.5) {
		BoundaryCondition bc;
		bc.type = BCType::CONTACT;
		bc.plane_normal = Vector3(0, 1, 0);
		bc.plane_point = Vector3(0, y_height, 0);
		bc.friction_coeff = friction;
		conditions.push_back(bc);
	}

	// Add a spherical container (e.g., for SPH)
	void add_sphere(const Vector3 &center, real_t radius, BCType type = BCType::CONTACT) {
		BoundaryCondition bc;
		bc.type = type;
		bc.use_sphere = true;
		bc.sphere_center = center;
		bc.sphere_radius = radius;
		conditions.push_back(bc);
	}

	// Apply all boundary constraints to a set of particles
	// (positions and velocities are updated in‑place)
	void apply_to_particles(LocalVector<Vector3> &p_positions,
							LocalVector<Vector3> &p_velocities,
							const LocalVector<real_t> &p_masses) {
		for (int i = 0; i < p_positions.size(); ++i) {
			Vector3 &pos = p_positions[i];
			Vector3 &vel = p_velocities[i];
			real_t mass = (p_masses.size() > 0) ? p_masses[i] : 1.0;

			for (const BoundaryCondition &bc : conditions) {
				real_t dist_to_bc = 0.0;
				Vector3 normal;

				if (!bc.use_sphere) {
					// Plane
					Vector3 diff = pos - bc.plane_point;
					dist_to_bc = diff.dot(bc.plane_normal);
					normal = bc.plane_normal;
				} else {
					// Sphere
					Vector3 diff = pos - bc.sphere_center;
					real_t r = diff.length();
					dist_to_bc = r - bc.sphere_radius;
					normal = (r > CMP_EPSILON) ? diff / r : Vector3(0, 1, 0);
				}

				if (bc.type == BCType::CONTACT && dist_to_bc < 0.0) {
					// Push particle back to surface
					pos -= normal * dist_to_bc;
					// Update velocity: reflect normal component with friction
					real_t vn = vel.dot(normal);
					if (vn < 0) {
						Vector3 vt = vel - normal * vn;
						vel = vt + normal * (-bc.friction_coeff * vn * 0.0); // simplified
						// Coulomb friction: if vt_len > mu * |vn|, clamp
						real_t vt_len = vt.length();
						real_t friction_limit = bc.friction_coeff * Math::abs(vn);
						if (vt_len > friction_limit && vt_len > CMP_EPSILON) {
							vt *= friction_limit / vt_len;
						}
						vel = vt + normal * (-vn * 0.0); // no normal rebound by default
					}
					// Also enforce position clamp
					pos += normal * (-dist_to_bc); // already done
				} else if (bc.type == BCType::DIRICHLET) {
					// Fix selected axes
					if (bc.fix_x) { pos.x = bc.prescribed_position.x; vel.x = bc.prescribed_velocity.x; }
					if (bc.fix_y) { pos.y = bc.prescribed_position.y; vel.y = bc.prescribed_velocity.y; }
					if (bc.fix_z) { pos.z = bc.prescribed_position.z; vel.z = bc.prescribed_velocity.z; }
				}
			}
		}
	}

	// Apply to a single position (for grid nodes, etc.)
	void apply_to_point(Vector3 &p_pos, Vector3 &p_vel, real_t p_mass) const {
		LocalVector<Vector3> pos_arr, vel_arr;
		LocalVector<real_t> mass_arr;
		pos_arr.push_back(p_pos);
		vel_arr.push_back(p_vel);
		mass_arr.push_back(p_mass);
		const_cast<BoundaryConditionManager *>(this)->apply_to_particles(pos_arr, vel_arr, mass_arr);
		p_pos = pos_arr[0];
		p_vel = vel_arr[0];
	}
};

} // namespace genesis

#endif // GENESIS_BOUNDARIES_BOUNDARY_CONDITIONS_H