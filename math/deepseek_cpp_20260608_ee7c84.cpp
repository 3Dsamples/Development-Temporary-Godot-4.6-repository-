// File 359: modules/gaia/src/vbd_physics/vbd_neohookean.h
// Neo-Hookean hyperelastic energy for VBD (Vertex Block Descent) tetrahedral elements.
// Provides energy, gradient, and Hessian for implicit integration.
// Rewritten from Gaia's VBD_NeoHookean.h for Godot 4.6.

#ifndef GAIA_VBD_NEOHOOKEAN_H
#define GAIA_VBD_NEOHOOKEAN_H

#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/typedefs.h"
#include "../mesh/tet_mesh.h"

namespace gaia::vbd {

class VBDNeoHookean {
public:
    VBDNeoHookean(real_t p_mu = 1e4, real_t p_lambda = 1e4) : mu(p_mu), lambda(p_lambda) {}

    void set_mu(real_t p_mu) { mu = MAX(p_mu, 0.0); }
    real_t get_mu() const { return mu; }
    void set_lambda(real_t p_lambda) { lambda = MAX(p_lambda, 0.0); }
    real_t get_lambda() const { return lambda; }

    /**
     * Compute elastic energy density Ψ(F) for a tetrahedron.
     * F is the deformation gradient (3x3 matrix).
     * Neo-Hookean: Ψ = (μ/2)(tr(C) - 3) - μ ln(J) + (λ/2)(ln(J))^2
     * where C = F^T F, J = det(F).
     */
    real_t energy(const Basis &F) const {
        real_t J = F.determinant();
        if (J <= CMP_EPSILON) return 1e10; // degenerate element penalty
        Basis C = F.transposed() * F;
        real_t traceC = C[0][0] + C[1][1] + C[2][2];
        real_t logJ = Math::log(J);
        return (mu * 0.5) * (traceC - 3.0) - mu * logJ + (lambda * 0.5) * logJ * logJ;
    }

    /**
     * Compute first Piola-Kirchhoff stress P = ∂Ψ/∂F.
     * P = μ (F - F^{-T}) + λ ln(J) F^{-T}
     */
    Basis pk1_stress(const Basis &F) const {
        real_t J = F.determinant();
        if (J <= CMP_EPSILON) return Basis().scaled(Vector3(0, 0, 0));
        Basis F_inv_T = F.inverse().transposed();
        return mu * (F - F_inv_T) + lambda * Math::log(J) * F_inv_T;
    }

    /**
     * Compute the element's contribution to the block descent update.
     * For vertex `local_idx` (0..3), returns the local Hessian (3x3) and
     * gradient (3‑vector) with respect to that vertex's position.
     * Uses the rest shape stored in the tetrahedral mesh.
     *
     * @param mesh               The tetrahedral mesh (must have precomputed rest state).
     * @param element_idx        Index of the tetrahedron.
     * @param positions          Current positions of all vertices.
     * @param local_vertex_idx   Which vertex of the tet (0,1,2,3).
     * @param out_hessian        Output 3x3 Hessian block.
     * @param out_gradient       Output 3‑vector gradient.
     */
    void compute_local_block(const mesh::TetMesh &mesh, int element_idx,
                             const LocalVector<Vector3> &positions, int local_vertex_idx,
                             Basis &out_hessian, Vector3 &out_gradient) const {
        mesh::TetMesh::Tetrahedron tet = mesh.get_tetrahedron(element_idx);
        int ids[4] = { tet.v0, tet.v1, tet.v2, tet.v3 };
        const Vector3 &p0 = positions[ids[0]];
        const Vector3 &p1 = positions[ids[1]];
        const Vector3 &p2 = positions[ids[2]];
        const Vector3 &p3 = positions[ids[3]];

        // Retrieve rest shape matrix Dm inverse columns.
        Vector3 inv_Dm0, inv_Dm1, inv_Dm2;
        mesh.get_inv_Dm(element_idx, inv_Dm0, inv_Dm1, inv_Dm2);

        // Deformation gradient F = [e1 e2 e3] * inv_Dm
        Vector3 e1 = p1 - p0;
        Vector3 e2 = p2 - p0;
        Vector3 e3 = p3 - p0;
        Basis edges(e1, e2, e3);
        Basis Dm_inv(inv_Dm0, inv_Dm1, inv_Dm2); // columns
        Basis F = edges * Dm_inv;

        real_t J = F.determinant();
        if (J <= CMP_EPSILON) {
            out_hessian = Basis().scaled(Vector3(1e6, 1e6, 1e6)); // large stiffness
            out_gradient = Vector3();
            return;
        }
        Basis F_inv_T = F.inverse().transposed();
        real_t logJ = Math::log(J);

        // P = μ(F - F^{-T}) + λ logJ F^{-T}
        Basis P = mu * (F - F_inv_T) + lambda * logJ * F_inv_T;

        // Shape function gradients in material space (2D for tetrahedron).
        // dN0/dX = (-1,-1,-1)^T, dN1 = (1,0,0), dN2 = (0,1,0), dN3 = (0,0,1).
        static const real_t dN[4][3] = {
            { -1, -1, -1 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }
        };

        Vector3 grad_N_local(dN[local_vertex_idx][0], dN[local_vertex_idx][1], dN[local_vertex_idx][2]);

        // Rest volume of the element.
        real_t V0 = mesh.get_rest_volume(element_idx);
        if (V0 < CMP_EPSILON) V0 = 1e-6;

        // Gradient: f = V0 * P * (Dm^{-T} * grad_N_local)
        Basis Dm_inv_T = Dm_inv.transposed(); // Dm^{-T}
        Vector3 grad_phi = Dm_inv_T.xform(grad_N_local); // world space gradient
        out_gradient = V0 * P.xform(grad_N_local); // P * grad_N_local (material gradient)
        // Actually P is 3x3, grad_N_local is 3‑vector. P * grad_N_local gives world force.
        out_gradient = V0 * P.xform(grad_N_local);

        // Hessian: linearisation of P w.r.t. F.
        // For block descent, approximate with a diagonal stiffness.
        // Full 3x3 Hessian for this vertex: K = V0 * grad_phi^T * D * grad_phi
        // where D is the 4th‑order elasticity tensor. We use the simplified
        // approach: K ≈ μ * V0 * grad_phi·grad_phi^T + λ * V0 * grad_phi·grad_phi^T? Actually
        // the derivative of P w.r.t. F for Neo‑Hookean gives a formula.
        // We'll use the approximate stiffness: K = μ * V0 * (grad_outer) + λ * V0 * (grad_outer)
        // where grad_outer = grad_phi * grad_phi^T.
        out_hessian = Basis(); // zero
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                real_t sum = 0.0;
                // Contribution from μ and λ terms.
                // μ * V0 * (grad_phi_i * grad_phi_j)
                // λ * V0 * (grad_phi_i * grad_phi_j)  (approximation)
                sum = (mu + lambda) * V0 * grad_phi[i] * grad_phi[j];
                out_hessian[i][j] = sum;
            }
        }
        // Add diagonal stabilisation to avoid zero eigenvalues.
        for (int d = 0; d < 3; ++d) {
            out_hessian[d][d] += 1e-6;
        }
    }

private:
    real_t mu;
    real_t lambda;
};

} // namespace gaia::vbd

#endif // GAIA_VBD_NEOHOOKEAN_H