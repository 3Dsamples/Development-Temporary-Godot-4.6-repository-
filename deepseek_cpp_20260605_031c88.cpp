//File 0211 : sparse/xsparse_config.hpp
//Sparse-specific configuration: sparse index types, default storage formats, zero tolerance, and SIMD alignment for sparse operations.
#ifndef XTENSOR_XSPARSE_CONFIG_HPP
#define XTENSOR_XSPARSE_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"

namespace xt {
namespace sparse {

    /**
     * Default index type for sparse matrices (row/column indices).
     */
    using index_type = std::size_t;

    /**
     * Default zero tolerance: values with absolute value <= zero_tol are treated as zero.
     */
    template <class T>
    constexpr T default_zero_tol = T(1e-12);

    /**
     * Default sparse storage order (row-major: CSR, column-major: CSC).
     */
    constexpr layout_type default_sparse_layout = layout_type::row_major;

    /**
     * Threshold for switching from linear search to binary search within rows/columns.
     */
    constexpr std::size_t sparse_binary_search_threshold = 16;

    /**
     * Block size for SIMD processing of sparse rows.
     */
    constexpr std::size_t sparse_simd_block_size = 64;

    /**
     * Minimum non-zeros per thread for parallel reduction.
     */
    constexpr std::size_t sparse_parallel_threshold = 1024;

    /**
     * Alias for aligned allocator suitable for sparse value storage.
     */
    template <class T>
    using sparse_allocator = aligned_allocator<T, SIMD_ALIGNMENT>;

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_CONFIG_HPP