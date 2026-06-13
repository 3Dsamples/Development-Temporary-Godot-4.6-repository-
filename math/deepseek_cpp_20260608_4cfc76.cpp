// File 64: modules/genesis/src/collision/ipc_coupler.h
// IPC (Incremental Potential Contact) coupling between FEM/MPM deformable bodies and rigid bodies.
// Implements barrier-based contact forces that are smooth and differentiable.

#ifndef GENESIS_COLLISION_IPC_COUPLER_H
#define GENESIS_COLLISION_IPC_COUPLER_H

#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "../core/genesis_types.h"

namespace genesis {

/**
 * IPC contact pair between a point (vertex) and a surface (triangle or analytical plane).
 * The barrier potential is defined as:
 *   b(d) = - d_hat * ln(d / d_hat) for d < d_hat, else 0.
 * Force: f = kappa * b'(d) * grad(d).
 *
 * This implementation handles FEM tetrahedral meshes and rigid body shapes.
 */
class IPCCoupler {
public:
	// Contact pair descriptor (vertex vs. triangle)
	struct ContactPair {
		int32_t body_a;        // index of body A (0: rigid, 1+: deformable entity ID)
		int32_t body_b;        // index of body B
		int32_t idx_a;         // vertex index in body A (if rigid, point in local space? we store world point for rigid)
		int32_t idx_b0, idx_b1, idx_b2; // triangle vertices in body B (or -1 for rigid triangle)
		bool is_rigid_a;
		bool is_rigid_b;
		real_t d_hat;          // IPC distance threshold
		real_t kappa;          // contact stiffness
		real_t friction;       // Coulomb friction coefficient (0 = no friction)
	};

	// Precomputed barrier energy and gradient for a single contact pair.
	// The caller provides the current world positions of the involved vertices.
	struct ContactForces {
		real_t energy;
		Vector3 force_a;            // on vertex a (body A)
		Vector3 force_b0, force_b1, force_b2; // on triangle vertices of body B
		// For rigid body, forces will be converted to impulses.
	};

	// Set up CPC parameters.
	IPCCoupler() : default_d_hat(0.001), default_kappa(1e6) {}

	void set_default_d_hat(real_t p_d) { default_d_hat = MAX(p_d, 1e-6); }
	real_t get_default_d_hat() const { return default_d_hat; }

	void set_default_kappa(real_t p_k) { default_kappa = MAX(p_k, 0.0); }
	real_t get_default_kappa() const { return default_kappa; }

	// Evaluate barrier forces for a single contact pair given current world positions.
	ContactForces evaluate_pair(const ContactPair &cp,
								const LocalVector<Vector3> &pos_a,       // positions for body A (if deformable)
								const LocalVector<Vector3> &pos_b,       // positions for body B
								const Vector3 *rigid_point_a = nullptr,  // if body A is rigid
								const Vector3 *rigid_point_b = nullptr) const {
		ContactForces res;
		res.energy = 0.0;
		res.force_a = Vector3();
		res.force_b0 = res.force_b1 = res.force_b2 = Vector3();

		Vector3 p;
		if (cp.is_rigid_a) {
			if (!rigid_point_a) return res;
			p = *rigid_point_a;
		} else {
			if (cp.idx_a < 0 || cp.idx_a >= pos_a.size()) return res;
			p = pos_a[cp.idx_a];
		}

		Vector3 t0, t1, t2;
		if (cp.is_rigid_b) {
			if (!rigid_point_b) return res;
			t0 = *rigid_point_b; // treat as point vs point? For rigid surface, we'd need a plane or triangle. This is simplified to point-point.
			// We'll assume a point contact for rigid-rigid? Actually IPC coupler typically works between deformable and rigid.
			// For this stub, we handle deformable vs deformable triangle; rigid uses a single contact point.
			// Then distance is point-point, barrier between two points.
			real_t d = p.distance_to(t0);
			real_t d_hat = cp.d_hat > 0 ? cp.d_hat : default_d_hat;
			if (d >= d_hat) return res;
			real_t kappa = cp.kappa > 0 ? cp.kappa : default_kappa;
			real_t b_prime = -d_hat / MAX(d, CMP_EPSILON);
			real_t grad_factor = kappa * b_prime;
			Vector3 n = (p - t0).normalized();
			res.energy = kappa * (-d_hat * Math::log(d / d_hat) + (d - d_hat));
			res.force_a = grad_factor * n;
			// Only point A, point B will receive opposite force (handled externally)
			res.force_b0 = -grad_factor * n;
			return res;
		}

		// Deformable triangle: compute closest point and distance
		if (cp.idx_b0 < 0 || cp.idx_b0 >= pos_b.size() ||
			cp.idx_b1 < 0 || cp.idx_b1 >= pos_b.size() ||
			cp.idx_b2 < 0 || cp.idx_b2 >= pos_b.size()) return res;

		t0 = pos_b[cp.idx_b0];
		t1 = pos_b[cp.idx_b1];
		t2 = pos_b[cp.idx_b2];

		// Use triangle closest point function from gaia::bvh::query (assuming it's accessible)
		// We'll implement a local version.
		auto closest_point_triangle = [](const Vector3 &p, const Vector3 &a, const Vector3 &b, const Vector3 &c,
										   real_t &u, real_t &v) -> Vector3 {
			Vector3 ab = b - a, ac = c - a, ap = p - a;
			real_t d1 = ab.dot(ap), d2 = ac.dot(ap);
			if (d1 <= 0 && d2 <= 0) { u=0; v=0; return a; }
			Vector3 bp = p - b;
			real_t d3 = ab.dot(bp), d4 = ac.dot(bp);
			if (d3 >= 0 && d4 <= d3) { u=1; v=0; return b; }
			real_t vc = d1*d4 - d3*d2;
			if (vc <= 0 && d1>=0 && d3<=0) { real_t vv = d1/(d1-d3); u=vv; v=0; return a+ab*vv; }
			Vector3 cp = p - c;
			real_t d5 = ab.dot(cp), d6 = ac.dot(cp);
			if (d6>=0 && d5<=d6) { u=0; v=1; return c; }
			real_t vb = d5*d2 - d1*d6;
			if (vb<=0 && d2>=0 && d6<=0) { real_t w = d2/(d2-d6); u=0; v=w; return a+ac*w; }
			real_t va = d3*d6 - d5*d4;
			if (va<=0 && (d4-d3)>=0 && (d5-d6)>=0) {
				real_t w = (d4-d3)/((d4-d3)+(d5-d6));
				u=1-w; v=w; return b+(c-b)*w;
			}
			real_t denom = 1.0/(va+vb+vc);
			real_t vv = vb*denom, ww = vc*denom;
			u=vv; v=ww;
			return a + ab*vv + ac*ww;
		};

		real_t u, v;
		Vector3 closest = closest_point_triangle(p, t0, t1, t2, u, v);
		Vector3 delta = p - closest;
		real_t d = delta.length();
		real_t d_hat = cp.d_hat > 0 ? cp.d_hat : default_d_hat;
		if (d >= d_hat) return res;
		real_t kappa = cp.kappa > 0 ? cp.kappa : default_kappa;
		real_t b_prime = -d_hat / MAX(d, CMP_EPSILON);
		real_t grad_factor = kappa * b_prime;
		Vector3 n = (d > CMP_EPSILON) ? (delta / d) : Vector3(0,1,0); // normal direction

		res.energy = kappa * (-d_hat * Math::log(d / d_hat) + d - d_hat);
		res.force_a = grad_factor * n;
		// Triangle vertices forces: -grad_factor * (partial closest / partial p, but symmetric)
		// We approximate by distributing force to triangle vertices by barycentric coordinates.
		// Actually force on triangle vertices = -force_a * (1-u-v) on v0, u on v1, v on v2
		real_t w = 1.0 - u - v;
		Vector3 f_total = -grad_factor * n;
		res.force_b0 = f_total * w;
		res.force_b1 = f_total * u;
		res.force_b2 = f_total * v;
		return res;
	}

	// Utility to build contact pairs from proximity of two meshes (simplified brute-force).
	// In practice, a broad-phase (Gaia BVH) should be used.
	void build_pairs_deformable_rigid(const LocalVector<Vector3> &deform_vertices,
									  const Collider *rigid_collider,
									  const Transform3D &rigid_transform,
									  LocalVector<ContactPair> &pairs,
									  real_t d_hat_override = -1.0) {
		// For each vertex, if inside the rigid shape's proximity, add pair.
		// We'll approximate by checking distance from vertex to rigid collider surface.
		// Since we lack SDF, we can use the GJK distance (expensive). Instead we'll use a simple sphere check if collider is sphere.
		// Not full implementation; placeholder.
	}

private:
	real_t default_d_hat;
	real_t default_kappa;
};

} // namespace genesis

#endif // GENESIS_COLLISION_IPC_COUPLER_H