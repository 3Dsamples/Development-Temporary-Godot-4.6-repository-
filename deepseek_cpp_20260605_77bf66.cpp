//File 0212 : sparse/xsparse_utils.hpp
//Sparse utility functions: sparsity statistics, pattern extraction, reordering, diagonal extraction, identity creation, and memory estimation.
#ifndef XTENSOR_XSPARSE_UTILS_HPP
#define XTENSOR_XSPARSE_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xsort.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcsc.hpp"
#include "../sparse/xsparse_config.hpp"

namespace xt {
namespace sparse {

    /**
     * Compute the sparsity ratio (nnz / total_elements) of a sparse matrix.
     */
    template <class SparseMat>
    inline double sparsity_ratio(const SparseMat& mat) noexcept
    {
        std::size_t total = static_cast<std::size_t>(mat.rows()) * mat.cols();
        if (total == 0) return 1.0;
        return static_cast<double>(mat.nnz()) / static_cast<double>(total);
    }

    /**
     * Compute the density (1 - sparsity_ratio).
     */
    template <class SparseMat>
    inline double density(const SparseMat& mat) noexcept
    {
        return 1.0 - sparsity_ratio(mat);
    }

    /**
     * Count non-zeros per row, returning a 1D array.
     */
    template <class T>
    inline auto row_nnz(const xcsr_matrix<T>& mat)
    {
        std::size_t nrows = mat.rows();
        std::vector<std::size_t> nnz_per_row(nrows);
        for (std::size_t r = 0; r < nrows; ++r)
            nnz_per_row[r] = mat.row_ptr()[r + 1] - mat.row_ptr()[r];
        xarray_container<uvector<std::size_t>> result({nrows});
        std::copy(nnz_per_row.begin(), nnz_per_row.end(), result.data());
        return result;
    }

    /**
     * Count non-zeros per column (requires scanning CSR, returns dense array).
     */
    template <class T>
    inline auto col_nnz(const xcsr_matrix<T>& mat)
    {
        std::size_t ncols = mat.cols();
        std::vector<std::size_t> nnz_per_col(ncols, 0);
        for (std::size_t r = 0; r < mat.rows(); ++r)
        {
            for (std::size_t i = mat.row_ptr()[r]; i < mat.row_ptr()[r + 1]; ++i)
                nnz_per_col[mat.col_idx()[i]]++;
        }
        xarray_container<uvector<std::size_t>> result({ncols});
        std::copy(nnz_per_col.begin(), nnz_per_col.end(), result.data());
        return result;
    }

    /**
     * Extract the diagonal of a sparse matrix as a dense 1D array.
     */
    template <class T>
    inline auto diagonal(const xcsr_matrix<T>& mat)
    {
        std::size_t n = std::min(mat.rows(), mat.cols());
        xarray_container<uvector<T>> result({n}, T(0));
        for (std::size_t i = 0; i < n; ++i)
            result[i] = mat(i, i);
        return result;
    }

    /**
     * Create a sparse identity matrix of size n x n.
     */
    template <class T>
    inline xcsr_matrix<T> eye_sparse(std::size_t n)
    {
        xcoo_matrix<T> coo(n, n);
        coo.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            coo.append(i, i, T(1));
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Create a sparse diagonal matrix from a dense 1D vector.
     */
    template <class T>
    inline xcsr_matrix<T> diag_sparse(const xarray_container<uvector<T>>& diag_vals)
    {
        std::size_t n = diag_vals.size();
        xcoo_matrix<T> coo(n, n);
        coo.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            if (diag_vals[i] != T(0))
                coo.append(i, i, diag_vals[i]);
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Compute the Frobenius norm of a sparse matrix (sqrt(sum of squares of non-zeros)).
     */
    template <class T>
    inline T frobenius_norm(const xcsr_matrix<T>& mat)
    {
        T sum_sq = 0;
        const T* vals = mat.values().data();
        std::size_t nnz = mat.nnz();
        if constexpr (is_simd_enabled_v<T>)
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            std::size_t vec_count = nnz / simd_size;
            simd_type vsum(0);
            for (std::size_t i = 0; i < vec_count; ++i)
            {
                simd_type v = simd_type::load_unaligned(vals + i * simd_size);
                vsum = vsum + v * v;
            }
            T tmp[simd_size];
            vsum.store_unaligned(tmp);
            for (std::size_t k = 0; k < simd_size; ++k) sum_sq += tmp[k];
            for (std::size_t i = vec_count * simd_size; i < nnz; ++i)
                sum_sq += vals[i] * vals[i];
        }
        else
        {
            for (std::size_t i = 0; i < nnz; ++i)
                sum_sq += vals[i] * vals[i];
        }
        return std::sqrt(sum_sq);
    }

    /**
     * Estimate memory usage of a CSR matrix in bytes.
     */
    template <class T>
    inline std::size_t memory_usage_bytes(const xcsr_matrix<T>& mat) noexcept
    {
        std::size_t row_ptr_bytes = (mat.rows() + 1) * sizeof(std::size_t);
        std::size_t col_idx_bytes = mat.nnz() * sizeof(std::size_t);
        std::size_t values_bytes = mat.nnz() * sizeof(T);
        return row_ptr_bytes + col_idx_bytes + values_bytes + sizeof(mat);
    }

    /**
     * Check if a sparse matrix is symmetric (A == A^T).
     */
    template <class T>
    inline bool is_symmetric(const xcsr_matrix<T>& mat)
    {
        if (mat.rows() != mat.cols()) return false;
        auto trans = mat.transpose();
        // Compare element-wise with tolerance
        for (std::size_t r = 0; r < mat.rows(); ++r)
        {
            for (std::size_t i = mat.row_ptr()[r]; i < mat.row_ptr()[r + 1]; ++i)
            {
                std::size_t c = mat.col_idx()[i];
                T val = mat.values()[i];
                T val_t = trans(r, c);
                if (std::abs(val - val_t) > default_zero_tol<T>)
                    return false;
            }
        }
        return true;
    }

    /**
     * Check if a sparse matrix is structurally symmetric (same sparsity pattern).
     */
    template <class T>
    inline bool is_structurally_symmetric(const xcsr_matrix<T>& mat)
    {
        if (mat.rows() != mat.cols()) return false;
        // For each non-zero A(i,j), check that A(j,i) exists
        for (std::size_t r = 0; r < mat.rows(); ++r)
        {
            for (std::size_t i = mat.row_ptr()[r]; i < mat.row_ptr()[r + 1]; ++i)
            {
                std::size_t c = mat.col_idx()[i];
                // Look for (c, r) in row c
                bool found = false;
                for (std::size_t j = mat.row_ptr()[c]; j < mat.row_ptr()[c + 1]; ++j)
                    if (mat.col_idx()[j] == r) { found = true; break; }
                if (!found) return false;
            }
        }
        return true;
    }

    /**
     * Compute a permutation that reduces matrix bandwidth using Cuthill-McKee ordering.
     * Returns a vector of new indices, such that A(perm, perm) has reduced bandwidth.
     */
    template <class T>
    inline auto reverse_cuthill_mckee(const xcsr_matrix<T>& mat)
    {
        // Simple implementation: degree-based BFS ordering
        std::size_t n = mat.rows();
        // Compute degrees of each node
        std::vector<std::size_t> degree(n, 0);
        for (std::size_t r = 0; r < n; ++r)
            degree[r] = mat.row_ptr()[r + 1] - mat.row_ptr()[r];

        // Find starting node: node with minimum degree? Actually algorithm picks peripheral node.
        // We'll start from node with smallest degree as a heuristic.
        std::size_t start = static_cast<std::size_t>(std::distance(degree.begin(),
            std::min_element(degree.begin(), degree.end())));

        // BFS with queue, selecting neighbors in increasing order of degree
        std::vector<bool> visited(n, false);
        std::vector<std::size_t> perm;
        perm.reserve(n);
        std::vector<std::size_t> queue;
        queue.push_back(start);
        visited[start] = true;

        while (!queue.empty())
        {
            std::size_t u = queue.front();
            queue.erase(queue.begin());
            perm.push_back(u);

            // Collect neighbors and sort by degree
            std::vector<std::pair<std::size_t, std::size_t>> neighbors;
            for (std::size_t i = mat.row_ptr()[u]; i < mat.row_ptr()[u + 1]; ++i)
            {
                std::size_t v = mat.col_idx()[i];
                if (!visited[v])
                    neighbors.emplace_back(degree[v], v);
            }
            std::sort(neighbors.begin(), neighbors.end());
            for (auto& [deg, v] : neighbors)
            {
                if (!visited[v])
                {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }

        // Any unvisited nodes (disconnected components)
        for (std::size_t i = 0; i < n; ++i)
            if (!visited[i])
                perm.push_back(i);

        // Reverse for RCM
        std::reverse(perm.begin(), perm.end());

        xarray_container<uvector<std::size_t>> result({n});
        std::copy(perm.begin(), perm.end(), result.data());
        return result;
    }

    /**
     * Apply a row/column permutation to a sparse matrix.
     * Returns A_perm where A_perm(i,j) = A(perm[i], perm[j]).
     */
    template <class T>
    inline auto permute(const xcsr_matrix<T>& mat, const std::vector<std::size_t>& perm)
    {
        std::size_t n = mat.rows();
        if (perm.size() != n) throw std::runtime_error("permute: permutation size mismatch.");
        // Build inverse permutation
        std::vector<std::size_t> inv_perm(n);
        for (std::size_t i = 0; i < n; ++i) inv_perm[perm[i]] = i;
        xcoo_matrix<T> coo(n, n);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t i = mat.row_ptr()[r]; i < mat.row_ptr()[r + 1]; ++i)
            {
                std::size_t c = mat.col_idx()[i];
                std::size_t new_r = inv_perm[r];
                std::size_t new_c = inv_perm[c];
                coo.append(new_r, new_c, mat.values()[i]);
            }
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_UTILS_HPP