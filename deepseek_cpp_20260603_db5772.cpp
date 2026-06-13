// system name : Octree Spatial Master
//File 0029 : core/math/fixed_fem_assembly.h
//Finite element assembly for tetrahedral meshes: shape functions, Gauss quadrature, mass/stiffness matrices, Voigt tensor operations, sparse matrix assembly
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "core/math/fixed_sparse_solver.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Linear tetrahedron element with global node indices for assembly
// ---------------------------------------------------------------------------
struct Tet4 {
    fvec3 nodes[4];   // position of the 4 vertices
    int   idx[4];     // global node index (0‑based) for each vertex

    // Compute 6 * signed volume
    fixed64_t six_volume() const noexcept {
        fvec3 a = fvec3_sub(nodes[1], nodes[0]);
        fvec3 b = fvec3_sub(nodes[2], nodes[0]);
        fvec3 c = fvec3_sub(nodes[3], nodes[0]);
        fvec3 cross = fvec3_cross(b, c);
        return fvec3_dot(a, cross);
    }

    // Shape function gradients (constant) w.r.t. global coordinates
    // grad[0..3] = (dNi/dx, dNi/dy, dNi/dz)  (gradients of the four linear shape functions)
    void shape_gradients(fvec3 grad[4]) const noexcept {
        fvec3 a = fvec3_sub(nodes[1], nodes[0]);
        fvec3 b = fvec3_sub(nodes[2], nodes[0]);
        fvec3 c = fvec3_sub(nodes[3], nodes[0]);
        fixed64_t V6 = fvec3_dot(a, fvec3_cross(b, c)); // 6*volume (signed)
        if (V6 == 0) {
            for (int i=0;i<4;++i) grad[i] = {0,0,0};
            return;
        }
        fixed64_t invV6 = fixed_rcp(fixed_abs(V6));
        // Standard formulas for a linear tet:
        // N1: - (b × c) / (6V)
        // N2:   (a × c) / (6V)
        // N3: - (a × b) / (6V)
        // N4:   (b × a) / (6V)   (i.e. (b-a)×(c-a)?? Actually using derived formulas)
        fvec3 n0 = fvec3_cross(b, c);
        grad[0] = fvec3_scale(n0, -invV6);
        grad[1] = fvec3_scale(fvec3_cross(c, a), invV6);
        grad[2] = fvec3_scale(fvec3_cross(a, b), -invV6);
        // The last gradient is the negative sum of the first three (since sum Ni =1 → sum grad Ni =0)
        grad[3] = fvec3_neg(fvec3_add(fvec3_add(grad[0], grad[1]), grad[2]));
    }

    // Element stiffness matrix (12x12 row‑major) for linear isotropic elasticity
    // D : 6x6 Voigt stiffness matrix, Ke : pointer to 144 fixed64_t values
    void stiffness_matrix(const StiffnessMatrix6x6& D, fixed64_t* Ke) const noexcept {
        fixed64_t V = fixed_abs(six_volume()) / 6; // absolute volume
        fvec3 grad[4];
        shape_gradients(grad);
        std::memset(Ke, 0, 144 * sizeof(fixed64_t));

        // Helper to extract B(k,i) for a given gradient vector g
        auto B_val = [](int k, int i, const fvec3& g) -> fixed64_t {
            switch (k) {
                case 0: return (i==0) ? g.x : 0;                  // ε_xx
                case 1: return (i==1) ? g.y : 0;                  // ε_yy
                case 2: return (i==2) ? g.z : 0;                  // ε_zz
                case 3: return ((i==1)?g.z:0) + ((i==2)?g.y:0);  // 2*ε_yz
                case 4: return ((i==0)?g.z:0) + ((i==2)?g.x:0);  // 2*ε_xz
                case 5: return ((i==0)?g.y:0) + ((i==1)?g.x:0);  // 2*ε_xy
                default: return 0;
            }
        };

        for (int a=0; a<4; ++a) {
            const fvec3& ga = grad[a];
            for (int b=0; b<4; ++b) {
                const fvec3& gb = grad[b];
                // 3x3 block at (a*3, b*3) within the 12x12 matrix (row‑major)
                fixed64_t* block = Ke + (a * 3 * 12) + (b * 3); // row = a*3, col = b*3
                for (int i=0; i<3; ++i) {
                    for (int j=0; j<3; ++j) {
                        fixed64_t sum = 0;
                        for (int k=0; k<6; ++k) {
                            fixed64_t Ba_ki = B_val(k, i, ga);
                            if (Ba_ki == 0) continue;
                            for (int l=0; l<6; ++l) {
                                fixed64_t Bb_lj = B_val(l, j, gb);
                                if (Bb_lj == 0) continue;
                                // sum += B_ki * D_kl * B_lj
                                sum += fixed_mul(fixed_mul(Ba_ki, D(k,l)), Bb_lj);
                            }
                        }
                        block[i * 12 + j] = fixed_mul(sum, V);
                    }
                }
            }
        }
    }

    // Consistent mass matrix (12x12 row‑major)
    void consistent_mass_matrix(fixed64_t* Me) const noexcept {
        fixed64_t V = fixed_abs(six_volume()) / 6;
        std::memset(Me, 0, 144 * sizeof(fixed64_t));
        // ∫ N_a N_b dV = (V/20) for a≠b, (V/10) for a=b
        for (int a=0; a<4; ++a) {
            for (int b=0; b<4; ++b) {
                fixed64_t val = (a == b) ? (V / 10) : (V / 20);
                fixed64_t* block = Me + (a * 3 * 12) + (b * 3);
                for (int i=0; i<3; ++i) {
                    block[i * 12 + i] = val; // identity scaled
                }
            }
        }
    }

    // Lumped mass vector (length 12) – each dof gets V/4
    void lumped_mass(fixed64_t* diag) const noexcept {
        fixed64_t V = fixed_abs(six_volume()) / 6;
        fixed64_t m = V / 4;
        for (int a=0; a<4; ++a)
            for (int i=0; i<3; ++i)
                diag[a*3 + i] = m;
    }
};

// ---------------------------------------------------------------------------
// Gauss quadrature for tetrahedron (4‑point rule, degree 2)
// ---------------------------------------------------------------------------
struct GaussPoint {
    fixed64_t weight;      // quadrature weight multiplied by element volume
    fvec3    coord;        // physical coordinate of the integration point
};

inline void tet_gauss_points_degree2(const Tet4& tet, std::array<GaussPoint,4>& gp) noexcept {
    fixed64_t V = fixed_abs(tet.six_volume()) / 6;
    // Pre‑compute sqrt(5) in fixed point
    fixed64_t sqrt5 = fixed_sqrt(5 * FIXED64_ONE);
    // b = (5 - sqrt5) / 20,  c = (5 + 3*sqrt5) / 20
    fixed64_t b = fixed_div(5 * FIXED64_ONE - sqrt5, 20 * FIXED64_ONE);
    fixed64_t c = fixed_div(5 * FIXED64_ONE + 3 * sqrt5, 20 * FIXED64_ONE);
    fixed64_t w = V / 4;

    // Helper: map barycentric (L0,L1,L2,L3) to physical point
    auto phys = [&](fixed64_t l0, fixed64_t l1, fixed64_t l2, fixed64_t l3) -> fvec3 {
        return fvec3_add(fvec3_add(fvec3_scale(tet.nodes[0], l0), fvec3_scale(tet.nodes[1], l1)),
                         fvec3_add(fvec3_scale(tet.nodes[2], l2), fvec3_scale(tet.nodes[3], l3)));
    };

    // The 4 points are permutations of (c, b, b, b)
    gp[0] = { w, phys(c, b, b, b) };
    gp[1] = { w, phys(b, c, b, b) };
    gp[2] = { w, phys(b, b, c, b) };
    gp[3] = { w, phys(b, b, b, c) };
}

// ---------------------------------------------------------------------------
// Sparse assembly helper: accumulate contributions into a CRS builder
// ---------------------------------------------------------------------------
class CRSBuilder {
public:
    CRSBuilder(int rows, int cols) : m_rows(rows), m_cols(cols) {
        m_row_entries.resize(rows);
    }

    // Add a value to global matrix entry (i, j). Multiple calls for the same (i,j) are summed.
    void add(int i, int j, fixed64_t val) noexcept {
        if (val == 0) return;
        auto& vec = m_row_entries[i];
        // Linear search (OK for typical small row contributions)
        for (auto& p : vec) {
            if (p.first == j) {
                p.second += val;
                return;
            }
        }
        vec.emplace_back(j, val);
    }

    // Build the final SparseMatrixCRS
    void build(SparseMatrixCRS& A) const noexcept {
        A.rows = m_rows;
        A.cols = m_cols;
        A.row_ptr.resize(m_rows + 1);
        size_t nnz = 0;
        for (int i=0; i<m_rows; ++i) {
            A.row_ptr[i] = (int32_t)nnz;
            nnz += m_row_entries[i].size();
        }
        A.row_ptr[m_rows] = (int32_t)nnz;
        A.values.resize(nnz);
        A.col_idx.resize(nnz);
        size_t pos = 0;
        for (int i=0; i<m_rows; ++i) {
            for (const auto& p : m_row_entries[i]) {
                A.values[pos] = p.second;
                A.col_idx[pos] = p.first;
                ++pos;
            }
        }
    }

private:
    int m_rows, m_cols;
    std::vector<std::vector<std::pair<int, fixed64_t>>> m_row_entries;
};

// ---------------------------------------------------------------------------
// Assemble global stiffness matrix from tetrahedral mesh into CRS format
// ---------------------------------------------------------------------------
inline void assemble_stiffness_crs(const std::vector<Tet4>& elements,
                                   const std::vector<fvec3>& nodes,
                                   const StiffnessMatrix6x6& D,
                                   SparseMatrixCRS& global_K) noexcept {
    int num_nodes = (int)nodes.size();
    int dofs = num_nodes * 3;
    CRSBuilder builder(dofs, dofs);

    for (const auto& tet : elements) {
        fixed64_t Ke[144];
        tet.stiffness_matrix(D, Ke);
        for (int a=0; a<4; ++a) {
            int global_a = tet.idx[a];
            for (int b=0; b<4; ++b) {
                int global_b = tet.idx[b];
                for (int i=0; i<3; ++i) {
                    for (int j=0; j<3; ++j) {
                        fixed64_t val = Ke[(a*3 + i) * 12 + (b*3 + j)];
                        builder.add(global_a*3 + i, global_b*3 + j, val);
                    }
                }
            }
        }
    }
    builder.build(global_K);
}

// ---------------------------------------------------------------------------
// Assemble global mass matrix (consistent) into CRS format
// ---------------------------------------------------------------------------
inline void assemble_mass_crs(const std::vector<Tet4>& elements,
                              const std::vector<fvec3>& nodes,
                              SparseMatrixCRS& global_M) noexcept {
    int num_nodes = (int)nodes.size();
    int dofs = num_nodes * 3;
    CRSBuilder builder(dofs, dofs);

    for (const auto& tet : elements) {
        fixed64_t Me[144];
        tet.consistent_mass_matrix(Me);
        for (int a=0; a<4; ++a) {
            int global_a = tet.idx[a];
            for (int b=0; b<4; ++b) {
                int global_b = tet.idx[b];
                for (int i=0; i<3; ++i) {
                    for (int j=0; j<3; ++j) {
                        fixed64_t val = Me[(a*3 + i) * 12 + (b*3 + j)];
                        builder.add(global_a*3 + i, global_b*3 + j, val);
                    }
                }
            }
        }
    }
    builder.build(global_M);
}

// ---------------------------------------------------------------------------
// Assemble lumped mass vector (diagonal) into a dense array of length num_nodes*3
// ---------------------------------------------------------------------------
inline void assemble_lumped_mass(const std::vector<Tet4>& elements,
                                 const std::vector<fvec3>& nodes,
                                 fixed64_t* lumped) noexcept {
    int dofs = (int)nodes.size() * 3;
    std::memset(lumped, 0, dofs * sizeof(fixed64_t));
    for (const auto& tet : elements) {
        fixed64_t local_diag[12];
        tet.lumped_mass(local_diag);
        for (int a=0; a<4; ++a) {
            int global_a = tet.idx[a];
            for (int i=0; i<3; ++i) {
                lumped[global_a*3 + i] += local_diag[a*3 + i];
            }
        }
    }
}

} // namespace fixed_math