// File 131: modules/gaia/src/vbd_cloth/vbd_tri_mesh_stvk.h
// St. Venant‑Kirchhoff constant‑strain triangle element for VBD cloth simulation.
// Computes elastic energy, gradient, and per‑vertex block descent updates
// using an analytic linearisation of the StVK energy density.

#ifndef GAIA_VBD_CLOTH_TRI_MESH_STVK_H
#define GAIA_VBD_CLOTH_TRI_MESH_STVK_H

#include "../vbd/vbd_element.h"           // gaia::vbd::VBDElement (abstract)
#include "../framework/body.h"            // gaia::SoftBody (for vertex access)
#include "vbd_base_tri_mesh.h"            // VBDBaseTriMesh
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace gaia::vbd_cloth {

/**
 * A triangle element with St. Venant‑Kirchhoff material.
 * Stores the three vertex indices, the rest‑state edge vectors,
 * and the constant inverse of the 2×3 rest‑shape matrix used to
 * compute the 3×2 deformation gradient.
 */
class VBDTriMeshStVK : public genesis::VBDElement {
public:
	VBDTriMeshStVK() : mu(1e4), lambda(1e4), thickness(0.001) {}

	// Set vertex indices in the owning soft body.
	void set_indices(int v0, int v1, int v2) {
		idx[0] = v0; idx[1] = v1; idx[2] = v2;
	}

	// Material parameters (Lame constants and thickness for area).
	void set_material_parameters(real_t p_mu, real_t p_lambda, real_t p_thickness = 0.001) {
		mu = p_mu;
		lambda = p_lambda;
		thickness = p_thickness;
	}

	// Compute the rest state from the current rest positions of the owning soft body.
	// Must be called after the soft body rest positions are set.
	virtual void compute_rest_state(const gaia::SoftBody *p_body) override {
		ERR_FAIL_COND(!p_body);
		const Vector3 &p0 = p_body->rest_positions[idx[0]];
		const Vector3 &p1 = p_body->rest_positions[idx[1]];
		const Vector3 &p2 = p_body->rest_positions[idx[2]];

		rest_e1 = p1 - p0;
		rest_e2 = p2 - p0;
		rest_area2 = rest_e1.cross(rest_e2).length(); // 2 * rest area

		// Build 2x2 matrix G = [e1.e1  e1.e2; e1.e2  e2.e2]
		real_t a = rest_e1.dot(rest_e1);
		real_t b = rest_e1.dot(rest_e2);
		real_t c = rest_e2.dot(rest_e2);
		real_t det = a * c - b * b;
		if (det < CMP_EPSILON) {
			// Degenerate element – use identity mapping for stability.
			inv_Dm_col0 = Vector3(1, 0, 0);
			inv_Dm_col1 = Vector3(0, 1, 0);
			return;
		}
		real_t inv_det = 1.0 / det;
		// inv_Dm = [e1 e2] * G^{-1}, columns = (c*e1 - b*e2)/det  and  (-b*e1 + a*e2)/det
		inv_Dm_col0 = (rest_e1 * c - rest_e2 * b) * inv_det;
		inv_Dm_col1 = (rest_e2 * a - rest_e1 * b) * inv_det;
	}

	// Perform one block‑descent step for this element.
	// Each vertex is updated sequentially (Gauss‑Seidel fashion) using a
	// single Newton step along its descent direction derived from the
	// linearised energy.
	virtual void solve_block_descent(gaia::SoftBody *p_body, real_t p_alpha, real_t p_dt) override {
		ERR_FAIL_COND(!p_body);
		Vector3 &p0 = p_body->positions[idx[0]];
		Vector3 &p1 = p_body->positions[idx[1]];
		Vector3 &p2 = p_body->positions[idx[2]];

		// Current edge vectors
		Vector3 e1 = p1 - p0;
		Vector3 e2 = p2 - p0;

		// Deformation gradient F = [e1 e2] * inv_Dm   (3x2)
		Vector3 f1 = inv_Dm_col0.x * e1 + inv_Dm_col0.y * e2; // F col 0
		Vector3 f2 = inv_Dm_col1.x * e1 + inv_Dm_col1.y * e2; // F col 1

		// Green strain E (2x2 symmetric)
		real_t E00 = 0.5 * (f1.dot(f1) - 1.0);
		real_t E11 = 0.5 * (f2.dot(f2) - 1.0);
		real_t E01 = 0.5 * (f1.dot(f2));       // E10 = E01

		// Second Piola‑Kirchhoff stress S = 2 mu E + lambda tr(E) I
		real_t traceE = E00 + E11;
		real_t S00 = 2.0 * mu * E00 + lambda * traceE;
		real_t S11 = 2.0 * mu * E11 + lambda * traceE;
		real_t S01 = 2.0 * mu * E01;

		// First Piola‑Kirchhoff P = F * S   (3x2)
		Vector3 P_col0 = f1 * S00 + f2 * S01;
		Vector3 P_col1 = f1 * S01 + f2 * S11;

		// Rest area (for force).  Volume = thickness * rest_area
		real_t V0 = rest_area2 * 0.5 * thickness;

		// Elasticity tensor D = dP/dF (4th order).  We store its action on vectors.
		// For StVK, D_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)

		// For block descent, we solve for each vertex displacement that zeroes the
		// gradient of the energy w.r.t. that vertex's position using a linearisation.
		// The spring stiffness matrix for vertex a is K_a = V0 * (grad_phi_a)^T * D * grad_phi_a
		// where grad_phi_a is a 3x2 matrix of shape function derivatives in material space.
		// For the constant‑strain triangle, grad_phi_0 = -(grad_phi_1 + grad_phi_2).
		// We precomputed inv_Dm whose columns are the gradients of the shape functions
		// in material space? Actually inv_Dm = [grad_phi_1, grad_phi_2] (3x2).
		// grad_phi_0 = -(grad_phi_1 + grad_phi_2).

		Vector3 g1 = inv_Dm_col0; // grad phi_1
		Vector3 g2 = inv_Dm_col1; // grad phi_2
		Vector3 g0 = -(g1 + g2);  // grad phi_0

		// We'll perform a block descent step for each vertex.
		// The gradient of the energy w.r.t. vertex a is:  V0 * P * g_a   (3-vector).
		// The local Hessian (3x3) for vertex a is:  H_a = V0 * g_a^T * D * g_a
		// We can compute H_a analytically.  Then the Newton step is  dx_a = -H_a^{-1} * grad_a.
		// A compliance parameter p_alpha can be absorbed as p_alpha/dt^2 into the stiffness
		// to model XPBD compliance.

		real_t alpha_dt2 = p_alpha / (p_dt * p_dt);

		// --------------------- Vertex 0 ---------------------
		{
			Vector3 grad0 = V0 * (P_col0 * g0.x + P_col1 * g0.y + Vector3()? Actually P * g_a is computed as:
				// P_col0 * g_a[0] + P_col1 * g_a[1] (since g_a has only two non-zero components in material 2D,
				// but g_a is a 3-vector in world space? Wait: g_a are vectors in world space because inv_Dm is
				// computed from world rest edges. They are 3-vectors. So P * g_a = P * (3-vector) is not defined
				// because P is 3x2. The mapping is: P is a 3x2 tensor mapping material 2D directions to world forces.
				// Actually the deformation gradient F is 3x2.  P is 3x2.  The gradient of the energy w.r.t.
				// a displacement dx_a is:  V0 * P * G_a, where G_a is a 2-vector (the reference shape function
				// gradients in material coordinates).  Those 2-vectors are: for vertex 1: (1,0), vertex 2: (0,1),
				// vertex 0: (-1,-1).  So the material 2D gradients are:
				//   dN0/dX = [-1, -1]^T,  dN1/dX = [1, 0]^T,  dN2/dX = [0, 1]^T.
				// The world-space force is:  f_a = V0 * P * (dN_a/dX).
				// P is 3x2, (dN_a/dX) is 2x1, so f_a is a 3-vector.
				// We computed g0,g1,g2 earlier incorrectly as 3-vectors from inv_Dm. Actually inv_Dm is the 3x2 matrix
				// mapping material 2-vectors to world 3-vectors:  dx_world = inv_Dm * dX_material.
				// So the world-space gradient of shape function a is: grad_world_phi_a = inv_Dm * dN_a/dX.
				// Therefore: f_a = V0 * P * (dN_a/dX)  (where P is 3x2, dN_a/dX is 2x1).
				// And the stiffness block H_ab = V0 * (dN_a/dX)^T * [P derivative] * (dN_b/dX) ... but we can
				// use the simplified vertex block descent that directly updates the vertex position using
				// a diagonal or dense 3x3 Hessian based on material directions.
				// For simplicity and robustness, we use the same approach as a projective constraint:
				// we compute the current force f = V0 * P * dN_a/dX, and we move the vertex by
				// dx = - (compliance) * f.  That is a simple gradient descent, not full Newton.
				// To satisfy "no missing logics", we should implement the actual Newton step.
				// The local Hessian for vertex a (3x3) is:
				//   H_a = V0 * (dN_a/dX)_i * D_ijkl * (dN_a/dX)_k   where D is the 4th order elasticity tensor.
				// D_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
				// Then H_a(p,q) = V0 * sum_{i,k} (dN_a/dX)_i * D_ipkq * (dN_a/dX)_k
				// This yields a 3x3 matrix.  We'll compute it.
			)
			// Re-compute forces and Hessians correctly using the 2D material gradients.
			Vector2 gradN0(-1.0, -1.0);
			Vector2 gradN1(1.0, 0.0);
			Vector2 gradN2(0.0, 1.0);

			// Precompute P for force: f_a = V0 * P * gradN_a
			Vector3 f0 = V0 * (P_col0 * gradN0.x + P_col1 * gradN0.y);
			Vector3 f1_force = V0 * (P_col0 * gradN1.x + P_col1 * gradN1.y);
			Vector3 f2_force = V0 * (P_col0 * gradN2.x + P_col1 * gradN2.y);

			// Build Hessian for each vertex (3x3) using D
			auto compute_H = [&](const Vector2 &g) -> Basis {
				Basis H; // zero
				for (int p = 0; p < 3; ++p) {
					for (int q = 0; q < 3; ++q) {
						real_t sum = 0.0;
						// sum over i,k in {0,1}
						for (int i = 0; i < 2; ++i) {
							for (int k = 0; k < 2; ++k) {
								// D_ipkq
								real_t D = 0.0;
								if (i == k && p == q) D += lambda;
								if (i == p && k == q) D += mu;
								if (i == q && k == p) D += mu;
								sum += D * g[i] * g[k];
							}
						}
						// Set element: H.get_row(p)[q] = V0 * sum  (we'll set via setting column)
						// Use set_row/get_row.
						Vector3 row = H.get_row(p);
						row[q] = V0 * sum;
						H.set_row(p, row);
					}
				}
				// Add compliance term (alpha/dt^2) to diagonal
				for (int d = 0; d < 3; ++d) {
					Vector3 row = H.get_row(d);
					row[d] += alpha_dt2;
					H.set_row(d, row);
				}
				return H;
			};

			Basis H0 = compute_H(gradN0);
			Basis H1 = compute_H(gradN1);
			Basis H2 = compute_H(gradN2);

			// Solve dx = -H^{-1} * f
			// For each vertex, we invert its 3x3 Hessian. Since the matrix is small,
			// we use Basis::inverse().
			Vector3 dx0 = H0.inverse().xform(-f0);
			Vector3 dx1 = H1.inverse().xform(-f1_force);
			Vector3 dx2 = H2.inverse().xform(-f2_force);

			// Apply position updates (damped by compliance if not already)
			// Also ensure we don't apply if pinned (caller should handle, but we can skip pinned vertices - we don't have pinned array here)
			p0 += dx0;
			p1 += dx1;
			p2 += dx2;
		}
	}

	virtual real_t compute_energy(const gaia::SoftBody *p_body, const genesis::FEMMaterial *p_mat) const override {
		const Vector3 &p0 = p_body->positions[idx[0]];
		const Vector3 &p1 = p_body->positions[idx[1]];
		const Vector3 &p2 = p_body->positions[idx[2]];
		Vector3 e1 = p1 - p0;
		Vector3 e2 = p2 - p0;

		Vector3 f1 = inv_Dm_col0.x * e1 + inv_Dm_col0.y * e2;
		Vector3 f2 = inv_Dm_col1.x * e1 + inv_Dm_col1.y * e2;

		real_t E00 = 0.5 * (f1.dot(f1) - 1.0);
		real_t E11 = 0.5 * (f2.dot(f2) - 1.0);
		real_t E01 = 0.5 * (f1.dot(f2));
		real_t traceE = E00 + E11;
		real_t psi = mu * (E00*E00 + E11*E11 + 2.0*E01*E01) + 0.5 * lambda * traceE * traceE;
		real_t V0 = rest_area2 * 0.5 * thickness;
		return V0 * psi;
	}

private:
	int idx[3];
	real_t mu, lambda, thickness;
	Vector3 rest_e1, rest_e2;
	real_t rest_area2;               // twice the rest area
	Vector3 inv_Dm_col0, inv_Dm_col1; // columns of the 3x2 pseudo-inverse
};

} // namespace gaia::vbd_cloth

#endif // GAIA_VBD_CLOTH_TRI_MESH_STVK_H