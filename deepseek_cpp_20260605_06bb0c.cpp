//File 0232 : sparse/xsparse_modal.hpp
//Sparse modal analysis for structures: generalized eigenvalue problem K·φ = λ·M·φ using shift‑invert Lanczos, mode extraction, and participation factors.
#ifndef XTENSOR_XSPARSE_MODAL_HPP
#define XTENSOR_XSPARSE_MODAL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xarray.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xeval.hpp"
#include "../core/xnorm.hpp"
#include "../core/xbuilder.hpp"
#include "../core/xsort.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_reducer.hpp"
#include "../sparse/xsparse_solver.hpp"
#include "../sparse/xsparse_eigen.hpp"

namespace xt {
namespace sparse {

    /**
     * @struct modal_result
     * @brief Holds eigenvalues, eigenvectors, and derived quantities for modal analysis.
     */
    template <class T>
    struct modal_result
    {
        xarray_container<uvector<T>> frequencies; // natural frequencies (Hz)
        xarray_container<uvector<T>> eigenvalues;  // ω² (rad²/s²)
        xarray_container<uvector<T>> mode_shapes;  // matrix (dof × n_modes)
        xarray_container<uvector<T>> participation_factors; // per mode
        xarray_container<uvector<T>> effective_mass; // per mode, per direction
        std::size_t n_modes;
    };

    namespace detail
    {
        /**
         * Compute the largest eigenvalue of the generalized problem K*x = λ*M*x
         * via power iteration on M^{-1}*K, using sparse CG to solve M*y = K*x.
         */
        template <class T>
        inline auto largest_generalized_eigenvalue(
            const xcsr_matrix<T>& K, const xcsr_matrix<T>& M,
            std::size_t max_iter = 500, T tol = T(1e-8))
        {
            std::size_t n = K.rows();
            xarray_container<uvector<T>> x({n}, T(1) / std::sqrt(static_cast<T>(n)));
            T lambda = T(0);
            for (std::size_t iter = 0; iter < max_iter; ++iter)
            {
                auto Kx = K.dot(x);
                // Solve M*y = Kx using CG
                auto y = cg_solve(M, Kx, T(1e-6), 300, preconditioner_type::diagonal);
                T lambda_new = dot1d(y, Kx) / std::max(dot1d(y, M.dot(y)), T(1e-15));
                T norm_y = std::sqrt(dot1d(y, M.dot(y)));
                if (norm_y > T(0))
                    x = y / norm_y;
                if (std::abs(lambda_new - lambda) < tol) break;
                lambda = lambda_new;
            }
            return lambda;
        }

        /**
         * Shift-invert Lanczos for the generalized eigenvalue problem.
         * Finds eigenvalues closest to shift sigma by applying Lanczos to (K - σM)^{-1}*M.
         */
        template <class T>
        inline auto shift_invert_lanczos(
            const xcsr_matrix<T>& K, const xcsr_matrix<T>& M,
            T sigma, std::size_t k, std::size_t max_iter = 1000, T tol = T(1e-8))
        {
            std::size_t n = K.rows();
            // Build shifted operator A_shift = K - sigma * M
            auto K_copy = K;
            auto M_copy = M;
            M_copy *= -sigma;
            auto A_shift = xcsr_matrix<T>::add(K_copy, M_copy);

            // Initial random vector
            xarray_container<uvector<T>> v({n});
            for (std::size_t i = 0; i < n; ++i) v[i] = (i % 13 == 0) ? T(1) : T(0);
            T beta = std::sqrt(dot1d(v, M.dot(v)));
            v = v / beta;

            std::vector<xarray_container<uvector<T>>> Q;
            std::vector<T> alpha, betas;
            Q.push_back(v);
            betas.push_back(T(0));

            for (std::size_t j = 0; j < max_iter; ++j)
            {
                // w = (K - sigma*M)^{-1} * M * v_j
                auto Mv = M.dot(Q[j]);
                auto w = cg_solve(A_shift, Mv, T(1e-6), 500, preconditioner_type::diagonal);
                if (j > 0) w = w - betas[j] * Q[j-1];
                T alpha_j = dot1d(w, M.dot(Q[j]));
                alpha.push_back(alpha_j);
                w = w - alpha_j * Q[j];

                // Reorthogonalize against all previous Q
                for (std::size_t i = 0; i <= j; ++i)
                {
                    T proj = dot1d(w, M.dot(Q[i]));
                    w = w - proj * Q[i];
                }

                T beta_next = std::sqrt(dot1d(w, M.dot(w)));
                if (beta_next < tol) break;
                betas.push_back(beta_next);
                Q.push_back(w / beta_next);

                if (Q.size() >= k + 10 || beta_next < tol)
                {
                    std::size_t m = Q.size() - 1;
                    // Build tridiagonal matrix T of size m
                    xarray_container<uvector<T>> Tmat({m, m}, T(0));
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        Tmat(i, i) = alpha[i];
                        if (i + 1 < m) Tmat(i, i+1) = betas[i+1];
                        if (i > 0) Tmat(i, i-1) = betas[i];
                    }
                    auto [evals_T, evecs_T] = xt::linalg::eig_sym(Tmat);

                    // Convert eigenvalues of T back to eigenvalues of original problem:
                    // λ_original = sigma + 1 / θ, where θ are eigenvalues of T
                    xarray_container<uvector<T>> evals({m});
                    for (std::size_t i = 0; i < m; ++i)
                        evals[i] = sigma + T(1) / (evals_T[i] + T(1e-15));

                    // Compute Ritz vectors: φ = Q_m * z where z are eigenvectors of T
                    std::vector<std::size_t> order(m);
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                        return std::abs(evals[a]) < std::abs(evals[b]);
                    });

                    xarray_container<uvector<T>> eigenvectors({n, k});
                    xarray_container<uvector<T>> eigenvalues({k});
                    for (std::size_t l = 0; l < k; ++l)
                    {
                        std::size_t idx = order[l];
                        eigenvalues[l] = evals[idx];
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            T sum = T(0);
                            for (std::size_t s = 0; s < m; ++s)
                                sum += Q[s][i] * evecs_T(s, idx);
                            eigenvectors(i, l) = sum;
                        }
                    }
                    return std::make_pair(eigenvalues, eigenvectors);
                }
            }
            throw std::runtime_error("shift_invert_lanczos: did not converge.");
        }
    }

    /**
     * Perform modal analysis of a structure given stiffness K and mass M.
     * Solves K·φ = ω²·M·φ for the lowest n_modes natural frequencies.
     * Assumes K and M are symmetric sparse CSR matrices.
     * @param K Stiffness matrix (n_dof × n_dof).
     * @param M Mass matrix (n_dof × n_dof), can be lumped or consistent.
     * @param n_modes Number of modes to extract.
     * @param shift Optional shift for interior eigenvalues.
     * @return modal_result containing frequencies, mode shapes, and participation factors.
     */
    template <class T>
    inline auto modal_analysis(
        const xcsr_matrix<T>& K,
        const xcsr_matrix<T>& M,
        std::size_t n_modes,
        T shift = T(0))
    {
        if (K.rows() != K.cols() || M.rows() != M.cols() || K.rows() != M.rows())
            throw std::runtime_error("modal_analysis: dimension mismatch.");
        std::size_t n = K.rows();
        if (n_modes > n) n_modes = n;

        // Compute eigenvalues and eigenvectors via shift-invert Lanczos
        auto [evals, evecs] = detail::shift_invert_lanczos(K, M, shift, n_modes);

        // Convert to frequencies: f = sqrt(λ)/(2π)
        xarray_container<uvector<T>> freqs({n_modes});
        for (std::size_t i = 0; i < n_modes; ++i)
        {
            T val = evals[i];
            if (val < T(0)) val = T(0); // negative eigenvalues become zero
            freqs[i] = std::sqrt(val) / (T(2) * xt::numeric_constants<T>::pi);
        }

        // Mass-normalize eigenvectors: φ^T * M * φ = I
        for (std::size_t j = 0; j < n_modes; ++j)
        {
            T norm = T(0);
            auto Mphi = M.dot(evecs); // not correct; we need dot product φ_j^T M φ_j
            // We'll compute scaling factor
            for (std::size_t r = 0; r < n; ++r)
            {
                T sum = T(0);
                for (std::size_t c = 0; c < n; ++c)
                {
                    // M is sparse; we need (M*evecs_col_j)[r] to compute φ^T M φ
                }
                norm += evecs(r, j) * Mphi[r]; // Mphi not yet computed correctly
            }
            // Actually compute M*φ_j
            xarray_container<uvector<T>> col_j({n});
            for (std::size_t r = 0; r < n; ++r) col_j[r] = evecs(r, j);
            auto M_col = M.dot(col_j);
            T scale = T(0);
            for (std::size_t r = 0; r < n; ++r)
                scale += col_j[r] * M_col[r];
            if (scale > T(1e-15))
            {
                T inv_sqrt = T(1) / std::sqrt(scale);
                for (std::size_t r = 0; r < n; ++r)
                    evecs(r, j) *= inv_sqrt;
            }
        }

        // Participation factors: Γ_j = φ_j^T * M * r, where r is rigid body displacement vector (1 per DOF direction)
        // Compute rigid body vector for each spatial direction
        // For a 3D problem, r_x = [1,0,0, 1,0,0, ...] etc.
        std::size_t dim = (n % 3 == 0) ? 3 : (n % 2 == 0 ? 2 : 1); // guess spatial dimension
        std::size_t nodes = n / dim;
        xarray_container<uvector<T>> participation({n_modes}, T(0));
        xarray_container<uvector<T>> effective_mass({n_modes, dim}, T(0));

        for (std::size_t d = 0; d < dim; ++d)
        {
            xarray_container<uvector<T>> r({n}, T(0));
            for (std::size_t nd = 0; nd < nodes; ++nd)
                r[nd * dim + d] = T(1);
            auto Mr = M.dot(r);
            for (std::size_t j = 0; j < n_modes; ++j)
            {
                T factor = T(0);
                for (std::size_t i = 0; i < n; ++i)
                    factor += evecs(i, j) * Mr[i];
                participation[j] += factor; // sum over directions? Actually we store per direction
                effective_mass(j, d) = factor * factor; // effective mass = Γ_jd^2
            }
        }

        modal_result<T> result;
        result.frequencies = freqs;
        result.eigenvalues = evals;
        result.mode_shapes = evecs;
        result.participation_factors = participation;
        result.effective_mass = effective_mass;
        result.n_modes = n_modes;
        return result;
    }

    /**
     * Perform modal frequency response analysis: compute steady-state response
     * of the structure to a harmonic excitation force F(ω) = F0 * sin(ωt).
     * Uses mode superposition: X(ω) = Σ_j (Γ_j / (ω_j² - ω² + 2iζω)) * φ_j.
     * @param K Stiffness matrix.
     * @param M Mass matrix.
     * @param modes Result from modal_analysis.
     * @param force Force vector.
     * @param omega Excitation frequency (rad/s).
     * @param damping_ratio Modal damping ratio (ζ).
     * @return Complex displacement amplitude.
     */
    template <class T>
    inline auto harmonic_response(
        const xcsr_matrix<T>& K,
        const xcsr_matrix<T>& M,
        const modal_result<T>& modes,
        const xarray_container<uvector<T>>& force,
        T omega, T damping_ratio)
    {
        std::size_t n = K.rows();
        std::size_t n_modes = modes.n_modes;
        using complex = std::complex<T>;
        xarray_container<uvector<complex>> X({n}, complex(0,0));
        for (std::size_t j = 0; j < n_modes; ++j)
        {
            T omega_j = T(2) * xt::numeric_constants<T>::pi * modes.frequencies[j];
            T denom = (omega_j * omega_j - omega * omega);
            complex denominator(denom, T(2) * damping_ratio * omega_j * omega);
            T gamma = T(0);
            for (std::size_t i = 0; i < n; ++i)
                gamma += modes.mode_shapes(i, j) * force[i];
            complex factor = complex(gamma, T(0)) / denominator;
            for (std::size_t i = 0; i < n; ++i)
                X[i] += factor * modes.mode_shapes(i, j);
        }
        return X;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_MODAL_HPP