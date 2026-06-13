// genesis/engine/solvers/soft_tissue_solver.cpp
// ... (includes unchanged) ...

void SoftTissueSolver::compute_viscoelastic_forces(EntityState& st, double dt) {
    size_t n_elems = st.entity->element_count();
    if (n_elems == 0) return;

    double mu = soft_config_.youngs_modulus / (2.0 * (1.0 + soft_config_.poisson_ratio));
    double lambda = soft_config_.youngs_modulus * soft_config_.poisson_ratio /
                    ((1.0 + soft_config_.poisson_ratio) * (1.0 - 2.0 * soft_config_.poisson_ratio));
    double tau = soft_config_.relaxation_time;
    if (tau <= 0.0) return;

    double alpha = std::exp(-dt / tau);
    const auto& tets = st.entity->tetrahedra();

    for (size_t e = 0; e < n_elems; ++e) {
        const auto& tet = tets[e];
        if (tet.empty()) continue;

        // Total deformation gradient
        datatypes::Matrix3r Fe = st.entity->deformation_gradient(e);

        // Update viscous strain
        st.viscous_strain[e] = st.viscous_strain[e] * alpha + Fe * (1.0 - alpha);

        // Elastic part for viscous branch
        datatypes::Matrix3r Fv = st.viscous_strain[e];
        datatypes::Matrix3r Fe_v = Fe * Fv.inverse();

        // First Piola-Kirchhoff stress from Fe_v (Neo-Hookean)
        double Je = Fe_v.determinant();
        datatypes::Matrix3r Fe_v_inv_T = Fe_v.inverse().transpose();
        datatypes::Matrix3r P = mu * (Fe_v - Fe_v_inv_T) + lambda * std::log(Je) * Fe_v_inv_T;

        // --- Compute nodal forces from P using shape function gradients ---
        // Obtain rest positions of the element's nodes
        datatypes::Vector3 X[4];
        for (int i = 0; i < 4; ++i) X[i] = st.rest_positions[tet[i]];

        // Shape matrix D_m = [ X1-X0, X2-X0, X3-X0 ]
        datatypes::Matrix3r Dm;
        for (int col = 0; col < 3; ++col) {
            Dm(0, col) = X[col+1][0] - X[0][0];
            Dm(1, col) = X[col+1][1] - X[0][1];
            Dm(2, col) = X[col+1][2] - X[0][2];
        }
        double volume = std::abs(Dm.determinant()) / 6.0;
        if (volume < 1e-12) continue;

        // Gradient matrix H = -volume * P * Dm^{-T}
        datatypes::Matrix3r Dm_inv_T = Dm.inverse().transpose();
        datatypes::Matrix3r H = -volume * P * Dm_inv_T;

        // Forces: f_i = H * (δ_i0? Actually standard FEM gives:
        // f_0 = -H * (1,1,1)^T, and f_i = H * e_i (i=1,2,3)
        datatypes::Vector3 f[4];
        f[0] = -H * datatypes::Vector3(1.0, 1.0, 1.0);
        for (int i = 1; i < 4; ++i) {
            f[i] = H.col(i-1);
        }

        // Apply forces to the four nodes
        for (int i = 0; i < 4; ++i) {
            st.external_nodal_forces[tet[i]] += f[i];
        }
    }
}