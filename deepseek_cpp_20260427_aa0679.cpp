// File 289: modules/vienna/src/cloth/vienna_cloth.h
// ViennaCloth — a rectangular grid‑based cloth mesh simulated with mass‑spring
// or XPBD. Supports structural, shear, and bending springs, wind, gravity,
// damping, pinned vertices, and collision spheres.

#ifndef VIENNA_CLOTH_VIENNA_CLOTH_H
#define VIENNA_CLOTH_VIENNA_CLOTH_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaCloth : public RefCounted {
	GDCLASS(ViennaCloth, RefCounted);

public:
	struct Vertex {
		vec3 position;
		vec3 prev_position;   // for Verlet
		vec3 velocity;
		real_t mass;
		bool pinned;
		Vertex() : position(), prev_position(), velocity(), mass(1.0), pinned(false) {}
	};

	ViennaCloth() :
		resolution_x(32), resolution_y(32),
		width(2.0), height(2.0),
		structural_stiffness(1000.0),
		shear_stiffness(100.0),
		bending_stiffness(200.0),
		damping(0.01),
		gravity(vec3(0.0, -9.81, 0.0)),
		wind(vec3()),
		solver(ClothSolverType::MASS_SPRING),
		iteration_count(5),
		collision_enabled(false) {}

	// --- Generation ---
	void set_resolution(int p_rx, int p_ry) {
		resolution_x = MAX(p_rx, 2);
		resolution_y = MAX(p_ry, 2);
	}
	void set_width(real_t p_w) { width = MAX(p_w, 0.01); }
	void set_height(real_t p_h) { height = MAX(p_h, 0.01); }

	// --- Physics parameters ---
	void set_structural_stiffness(real_t p_k) { structural_stiffness = MAX(p_k, 0.0); }
	void set_shear_stiffness(real_t p_k) { shear_stiffness = MAX(p_k, 0.0); }
	void set_bending_stiffness(real_t p_k) { bending_stiffness = MAX(p_k, 0.0); }
	void set_damping(real_t p_d) { damping = CLAMP(p_d, 0.0, 1.0); }
	void set_gravity(const vec3 &p_g) { gravity = p_g; }
	void set_wind(const vec3 &p_w) { wind = p_w; }
	void set_solver_type(ClothSolverType p_type) { solver = p_type; }
	void set_iterations(int p_iter) { iteration_count = MAX(p_iter, 1); }
	void set_collision_enabled(bool p_en) { collision_enabled = p_en; }

	// --- Pin vertices ---
	void pin_vertex(int p_x, int p_y, bool p_pin = true) {
		int idx = index(p_x, p_y);
		if (idx < 0 || idx >= vertices.size()) return;
		vertices[idx].pinned = p_pin;
	}

	// --- Collision spheres (centre, radius, push‑out factor) ---
	void add_collision_sphere(const vec3 &p_center, real_t p_radius) {
		CollisionSphere cs;
		cs.center = p_center;
		cs.radius = p_radius;
		collision_spheres.push_back(cs);
	}
	void clear_collision_spheres() { collision_spheres.clear(); }

	// --- Generate the cloth mesh (must be called before first step) ---
	void generate() {
		vertices.resize(resolution_x * resolution_y);
		real_t dx = width / (resolution_x - 1);
		real_t dz = height / (resolution_y - 1);
		vec3 start(-width * 0.5, 0.0, -height * 0.5);
		for (int y = 0; y < resolution_y; ++y) {
			for (int x = 0; x < resolution_x; ++x) {
				int idx = index(x, y);
				vec3 pos = start + vec3(x * dx, 0.0, y * dz);
				vertices[idx].position = pos;
				vertices[idx].prev_position = pos;
				vertices[idx].velocity = vec3();
				vertices[idx].mass = 1.0;
				vertices[idx].pinned = false;
			}
		}
		// Pin top row by default
		for (int x = 0; x < resolution_x; ++x) pin_vertex(x, 0, true);
	}

	// --- Step the cloth simulation by dt seconds ---
	void step(real_t dt) {
		if (vertices.is_empty()) return;

		// Semi‑implicit Euler or Verlet (we use Verlet for mass‑spring, XPBD for XPBD)
		if (solver == ClothSolverType::MASS_SPRING) {
			step_verlet(dt);
		} else {
			step_xpbd(dt);
		}
		// Resolve collision spheres
		if (collision_enabled) {
			for (Vertex &v : vertices) {
				for (const CollisionSphere &cs : collision_spheres) {
					vec3 diff = v.position - cs.center;
					real_t dist = diff.length();
					if (dist < cs.radius && dist > 0.0) {
						vec3 normal = diff / dist;
						v.position = cs.center + normal * cs.radius;
						// Reflect velocity
						real_t vn = v.velocity.dot(normal);
						if (vn < 0.0) v.velocity -= normal * vn * 1.5; // damped reflection
					}
				}
			}
		}
	}

	// --- Access vertices for rendering ---
	int get_vertex_count() const { return vertices.size(); }
	const Vertex &get_vertex(int p_idx) const { return vertices[p_idx]; }
	int get_resolution_x() const { return resolution_x; }
	int get_resolution_y() const { return resolution_y; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_resolution","rx","ry"), &ViennaCloth::set_resolution);
		ClassDB::bind_method(D_METHOD("set_width","w"), &ViennaCloth::set_width);
		ClassDB::bind_method(D_METHOD("set_height","h"), &ViennaCloth::set_height);
		ClassDB::bind_method(D_METHOD("set_structural_stiffness","k"), &ViennaCloth::set_structural_stiffness);
		ClassDB::bind_method(D_METHOD("set_shear_stiffness","k"), &ViennaCloth::set_shear_stiffness);
		ClassDB::bind_method(D_METHOD("set_bending_stiffness","k"), &ViennaCloth::set_bending_stiffness);
		ClassDB::bind_method(D_METHOD("set_damping","d"), &ViennaCloth::set_damping);
		ClassDB::bind_method(D_METHOD("set_gravity","g"), &ViennaCloth::set_gravity);
		ClassDB::bind_method(D_METHOD("set_wind","w"), &ViennaCloth::set_wind);
		ClassDB::bind_method(D_METHOD("set_solver_type","t"), &ViennaCloth::set_solver_type);
		ClassDB::bind_method(D_METHOD("set_iterations","i"), &ViennaCloth::set_iterations);
		ClassDB::bind_method(D_METHOD("set_collision_enabled","en"), &ViennaCloth::set_collision_enabled);
		ClassDB::bind_method(D_METHOD("pin_vertex","x","y","pin"), &ViennaCloth::pin_vertex, DEFVAL(true));
		ClassDB::bind_method(D_METHOD("add_collision_sphere","center","radius"), &ViennaCloth::add_collision_sphere);
		ClassDB::bind_method(D_METHOD("clear_collision_spheres"), &ViennaCloth::clear_collision_spheres);
		ClassDB::bind_method(D_METHOD("generate"), &ViennaCloth::generate);
		ClassDB::bind_method(D_METHOD("step","dt"), &ViennaCloth::step);
	}

private:
	int resolution_x;
	int resolution_y;
	real_t width;
	real_t height;
	real_t structural_stiffness;
	real_t shear_stiffness;
	real_t bending_stiffness;
	real_t damping;
	vec3 gravity;
	vec3 wind;
	ClothSolverType solver;
	int iteration_count;
	bool collision_enabled;

	struct CollisionSphere {
		vec3 center;
		real_t radius;
	};
	LocalVector<Vertex> vertices;
	LocalVector<CollisionSphere> collision_spheres;

	inline int index(int x, int y) const { return y * resolution_x + x; }

	// Verlet integration with mass‑spring
	void step_verlet(real_t dt) {
		real_t dt2 = dt * dt;
		for (Vertex &v : vertices) {
			if (v.pinned) continue;
			vec3 temp = v.position;
			vec3 accel = gravity + wind;
			// Verlet: pos = 2*pos - prev_pos + a*dt^2
			v.position = v.position * 2.0 - v.prev_position + accel * dt2;
			v.prev_position = temp;
			// Damping: reduce velocity
			vec3 vel = (v.position - v.prev_position) / dt;
			vel *= (1.0 - damping);
			v.prev_position = v.position - vel * dt;
		}
		// Springs iterations
		for (int iter = 0; iter < iteration_count; ++iter) {
			apply_springs();
		}
		// Update velocities for external use
		for (Vertex &v : vertices) {
			v.velocity = (v.position - v.prev_position) / dt;
		}
	}

	// XPBD solver
	void step_xpbd(real_t dt) {
		// Semi‑implicit Euler velocity update
		for (Vertex &v : vertices) {
			if (v.pinned) { v.velocity = vec3(); continue; }
			v.velocity += (gravity + wind) * dt;
			v.velocity *= (1.0 - damping);
			v.prev_position = v.position;
			v.position += v.velocity * dt;
		}
		// XPBD constraint iterations
		for (int iter = 0; iter < iteration_count; ++iter) {
			apply_xpbd_springs(dt);
		}
		// Update velocities
		for (Vertex &v : vertices) {
			if (v.pinned) continue;
			v.velocity = (v.position - v.prev_position) / dt;
		}
	}

	// Structural (horizontal, vertical), shear (diagonals), bending (skip‑1) springs
	void apply_springs() {
		real_t kStruct = structural_stiffness;
		real_t kShear  = shear_stiffness;
		real_t kBend   = bending_stiffness;

		// Structural: horizontal edges
		for (int y = 0; y < resolution_y; ++y) {
			for (int x = 0; x < resolution_x - 1; ++x) {
				int i0 = index(x, y), i1 = index(x+1, y);
				relax_edge(i0, i1, width/(resolution_x-1), kStruct);
			}
		}
		// Structural: vertical edges
		for (int y = 0; y < resolution_y - 1; ++y) {
			for (int x = 0; x < resolution_x; ++x) {
				int i0 = index(x, y), i1 = index(x, y+1);
				relax_edge(i0, i1, height/(resolution_y-1), kStruct);
			}
		}
		// Shear: diagonal edges
		real_t diag_len = Math::sqrt((width/(resolution_x-1))*(width/(resolution_x-1)) + (height/(resolution_y-1))*(height/(resolution_y-1)));
		for (int y = 0; y < resolution_y - 1; ++y) {
			for (int x = 0; x < resolution_x - 1; ++x) {
				// diagonal /
				int i0 = index(x, y+1), i1 = index(x+1, y);
				relax_edge(i0, i1, diag_len, kShear);
				// diagonal \
				i0 = index(x, y); i1 = index(x+1, y+1);
				relax_edge(i0, i1, diag_len, kShear);
			}
		}
		// Bending: horizontal skip‑1
		real_t bend_len_h = 2.0 * width / (resolution_x-1);
		for (int y = 0; y < resolution_y; ++y) {
			for (int x = 0; x < resolution_x - 2; ++x) {
				int i0 = index(x, y), i1 = index(x+2, y);
				relax_edge(i0, i1, bend_len_h, kBend);
			}
		}
		// Bending: vertical skip‑1
		real_t bend_len_v = 2.0 * height / (resolution_y-1);
		for (int x = 0; x < resolution_x; ++x) {
			for (int y = 0; y < resolution_y - 2; ++y) {
				int i0 = index(x, y), i1 = index(x, y+2);
				relax_edge(i0, i1, bend_len_v, kBend);
			}
		}
	}

	// Relax a single edge towards rest length using positional correction (Verlet)
	void relax_edge(int i0, int i1, real_t rest_len, real_t stiffness) {
		Vertex &v0 = vertices[i0];
		Vertex &v1 = vertices[i1];
		vec3 delta = v1.position - v0.position;
		real_t len = delta.length();
		if (len < CMP_EPSILON) return;
		vec3 dir = delta / len;
		real_t correction = (len - rest_len) * stiffness * 0.5;
		// If both pinned, skip
		if (v0.pinned && v1.pinned) return;
		// Move each vertex half the correction
		if (!v0.pinned) v0.position += dir * correction;
		if (!v1.pinned) v1.position -= dir * correction;
	}

	// XPBD edge constraint
	struct XPBDEdge { int i0, i1; real_t rest_len; real_t compliance; };
	void apply_xpbd_springs(real_t dt) {
		// Prepare list of constraints (lazy built each frame)
		LocalVector<XPBDEdge> constraints;
		auto add_edge = [&](int a, int b, real_t rest, real_t k) {
			XPBDEdge e;
			e.i0 = a; e.i1 = b; e.rest_len = rest;
			e.compliance = (k > 0.0) ? 1.0 / k : 0.0;
			constraints.push_back(e);
		};

		real_t dx = width / (resolution_x - 1);
		real_t dz = height / (resolution_y - 1);
		real_t diag = Math::sqrt(dx*dx + dz*dz);
		real_t bend_h = 2.0*dx;
		real_t bend_v = 2.0*dz;

		for (int y=0; y<resolution_y; ++y)
			for (int x=0; x<resolution_x-1; ++x)
				add_edge(index(x,y), index(x+1,y), dx, structural_stiffness);
		for (int y=0; y<resolution_y-1; ++y)
			for (int x=0; x<resolution_x; ++x)
				add_edge(index(x,y), index(x,y+1), dz, structural_stiffness);
		for (int y=0; y<resolution_y-1; ++y) {
			for (int x=0; x<resolution_x-1; ++x) {
				add_edge(index(x,y+1), index(x+1,y), diag, shear_stiffness);
				add_edge(index(x,y), index(x+1,y+1), diag, shear_stiffness);
			}
		}
		for (int y=0; y<resolution_y; ++y)
			for (int x=0; x<resolution_x-2; ++x)
				add_edge(index(x,y), index(x+2,y), bend_h, bending_stiffness);
		for (int x=0; x<resolution_x; ++x)
			for (int y=0; y<resolution_y-2; ++y)
				add_edge(index(x,y), index(x,y+2), bend_v, bending_stiffness);

		for (const XPBDEdge &e : constraints) {
			Vertex &v0 = vertices[e.i0];
			Vertex &v1 = vertices[e.i1];
			vec3 delta = v1.position - v0.position;
			real_t len = delta.length();
			if (len < CMP_EPSILON) continue;
			vec3 dir = delta / len;
			real_t C = len - e.rest_len;
			real_t w0 = v0.pinned ? 0.0 : 1.0 / v0.mass;
			real_t w1 = v1.pinned ? 0.0 : 1.0 / v1.mass;
			real_t wsum = w0 + w1;
			if (wsum < CMP_EPSILON) continue;
			real_t alpha_tilde = e.compliance / (dt * dt);
			// Lagrange multiplier update (simplified XPBD)
			real_t delta_lambda = -C / (wsum + alpha_tilde);
			if (!v0.pinned) v0.position -= delta_lambda * w0 * dir;
			if (!v1.pinned) v1.position += delta_lambda * w1 * dir;
		}
	}
};

} // namespace vienna

#endif // VIENNA_CLOTH_VIENNA_CLOTH_H