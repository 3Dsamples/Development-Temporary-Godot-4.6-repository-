//File 0221 : sparse/xsparse_block.hpp
//Block sparse matrix operations: block CSR format, block-sparse arithmetic, SIMD block multiplication, and conversion from/to dense blocks.
#ifndef XTENSOR_XSPARSE_BLOCK_HPP
#define XTENSOR_XSPARSE_BLOCK_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
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
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_expression.hpp"

namespace xt {
namespace sparse {

    /**
     * @class xblock_csr_matrix
     * @brief Block sparse matrix in CSR format.
     *
     * Each non-zero entry is a dense matrix (block) of size block_rows x block_cols.
     * The matrix logically has dimensions (nrows*block_rows) x (ncols*block_cols).
     * The sparsity pattern is stored in CSR form, with block indices and block data.
     */
    template <class T>
    class xblock_csr_matrix
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using block_type = xarray_container<uvector<T>>;

        /**
         * Construct a block CSR matrix.
         * @param nrows Number of block rows.
         * @param ncols Number of block columns.
         * @param block_rows Number of scalar rows per block.
         * @param block_cols Number of scalar cols per block.
         */
        xblock_csr_matrix(size_type nrows, size_type ncols,
                          size_type block_rows, size_type block_cols)
            : m_nrows(nrows), m_ncols(ncols)
            , m_block_rows(block_rows), m_block_cols(block_cols)
            , m_row_ptr(nrows + 1, 0)
        {
        }

        size_type block_rows() const noexcept { return m_nrows; }
        size_type block_cols() const noexcept { return m_ncols; }
        size_type scalar_rows() const noexcept { return m_nrows * m_block_rows; }
        size_type scalar_cols() const noexcept { return m_ncols * m_block_cols; }
        size_type nnz_blocks() const noexcept { return m_col_idx.size(); }

        const std::vector<size_type>& row_ptr() const noexcept { return m_row_ptr; }
        const std::vector<size_type>& col_idx() const noexcept { return m_col_idx; }
        const std::vector<block_type>& blocks() const noexcept { return m_blocks; }

        /**
         * Append a dense block at position (row, col).
         * Must be called in row-major order for correct row_ptr construction.
         */
        void append_block(size_type row, size_type col, const block_type& block)
        {
            if (block.shape()[0] != m_block_rows || block.shape()[1] != m_block_cols)
                throw std::runtime_error("xblock_csr_matrix::append_block: block dimension mismatch.");
            // Row pointer is updated; we assume sequential row building.
            if (row >= m_nrows || col >= m_ncols)
                throw std::out_of_range("xblock_csr_matrix::append_block: index out of bounds.");
            m_col_idx.push_back(col);
            m_blocks.push_back(block);
            m_row_ptr[row + 1] = m_col_idx.size();
        }

        /**
         * Finalize row pointers (call after all blocks have been appended).
         */
        void finalize()
        {
            for (size_type r = 0; r < m_nrows; ++r)
                m_row_ptr[r + 1] = m_col_idx.size(); // already set if appended properly
        }

        /**
         * Multiply block sparse matrix by a dense vector (scalar dimensions).
         * y = A * x, where A is block sparse.
         */
        auto dot(const xarray_container<uvector<T>>& x) const
        {
            if (x.dimension() != 1 || x.size() != scalar_cols())
                throw std::runtime_error("xblock_csr_matrix::dot: dimension mismatch.");
            xarray_container<uvector<T>> y({scalar_rows()}, T(0));
            T* y_data = y.data();
            const T* x_data = x.data();
            for (size_type r = 0; r < m_nrows; ++r)
            {
                size_type beg = m_row_ptr[r];
                size_type end = m_row_ptr[r + 1];
                for (size_type i = beg; i < end; ++i)
                {
                    size_type c = m_col_idx[i];
                    const block_type& block = m_blocks[i];
                    // y[r*block_rows : (r+1)*block_rows] += block * x[c*block_cols : (c+1)*block_cols]
                    const T* block_data = block.data();
                    const T* x_seg = x_data + c * m_block_cols;
                    T* y_seg = y_data + r * m_block_rows;
                    // Dense matrix-vector multiply (small block)
                    for (size_type br = 0; br < m_block_rows; ++br)
                    {
                        T sum = 0;
                        for (size_type bc = 0; bc < m_block_cols; ++bc)
                            sum += block_data[br * m_block_cols + bc] * x_seg[bc];
                        y_seg[br] += sum;
                    }
                }
            }
            return y;
        }

        /**
         * Multiply block sparse matrix by a dense matrix (scalar dimensions).
         * Y = A * X.
         */
        auto matmul(const xarray_container<uvector<T>>& X) const
        {
            if (X.dimension() != 2 || X.shape()[0] != scalar_cols())
                throw std::runtime_error("xblock_csr_matrix::matmul: dimension mismatch.");
            size_type k = X.shape()[1];
            xarray_container<uvector<T>> Y({scalar_rows(), k}, T(0));
            T* Y_data = Y.data();
            const T* X_data = X.data();
            for (size_type r = 0; r < m_nrows; ++r)
            {
                size_type beg = m_row_ptr[r];
                size_type end = m_row_ptr[r + 1];
                for (size_type i = beg; i < end; ++i)
                {
                    size_type c = m_col_idx[i];
                    const block_type& block = m_blocks[i];
                    // Y[r*block_rows : +block_rows, :] += block * X[c*block_cols : +block_cols, :]
                    const T* block_data = block.data();
                    const T* X_seg = X_data + c * m_block_cols * k;
                    T* Y_seg = Y_data + r * m_block_rows * k;
                    for (size_type br = 0; br < m_block_rows; ++br)
                    {
                        for (size_type j = 0; j < k; ++j)
                        {
                            T sum = 0;
                            for (size_type bc = 0; bc < m_block_cols; ++bc)
                                sum += block_data[br * m_block_cols + bc] * X_seg[bc * k + j];
                            Y_seg[br * k + j] += sum;
                        }
                    }
                }
            }
            return Y;
        }

        /**
         * Convert to a scalar CSR matrix.
         */
        auto to_scalar_csr() const
        {
            size_type total_rows = scalar_rows();
            size_type total_cols = scalar_cols();
            xcoo_matrix<T> coo(total_rows, total_cols);
            for (size_type r = 0; r < m_nrows; ++r)
            {
                for (size_type i = m_row_ptr[r]; i < m_row_ptr[r + 1]; ++i)
                {
                    size_type c = m_col_idx[i];
                    const block_type& blk = m_blocks[i];
                    for (size_type br = 0; br < m_block_rows; ++br)
                    {
                        for (size_type bc = 0; bc < m_block_cols; ++bc)
                        {
                            T val = blk(br, bc);
                            if (val != T(0))
                                coo.append(r * m_block_rows + br, c * m_block_cols + bc, val);
                        }
                    }
                }
            }
            return xcsr_matrix<T>::from_coo(coo);
        }

        /**
         * Block diagonal extraction: returns a block consisting of diagonal blocks.
         */
        auto block_diagonal(size_type block_index) const
        {
            size_type beg = m_row_ptr[block_index];
            size_type end = m_row_ptr[block_index + 1];
            for (size_type i = beg; i < end; ++i)
                if (m_col_idx[i] == block_index)
                    return m_blocks[i];
            return block_type({m_block_rows, m_block_cols}, T(0));
        }

    private:
        size_type m_nrows, m_ncols;
        size_type m_block_rows, m_block_cols;
        std::vector<size_type> m_row_ptr;
        std::vector<size_type> m_col_idx;
        std::vector<block_type> m_blocks;
    };

    /**
     * Build a block sparse Laplacian matrix for a 2D grid using blocks.
     * Each block corresponds to a row of grid points.
     */
    template <class T>
    inline auto block_laplacian2d(std::size_t nx, std::size_t ny)
    {
        // Each row of grid corresponds to a block of size ny x ny.
        xblock_csr_matrix<T> mat(nx, nx, ny, ny);
        // Diagonal block: tridiagonal of size ny (Laplacian along y)
        xarray_container<uvector<T>> diag_block({ny, ny}, T(0));
        for (std::size_t j = 0; j < ny; ++j)
        {
            diag_block(j, j) = T(4);
            if (j > 0) diag_block(j, j-1) = T(-1);
            if (j + 1 < ny) diag_block(j, j+1) = T(-1);
        }
        // Off-diagonal block: -I
        xarray_container<uvector<T>> off_block({ny, ny}, T(0));
        for (std::size_t j = 0; j < ny; ++j)
            off_block(j, j) = T(-1);

        for (std::size_t i = 0; i < nx; ++i)
        {
            mat.append_block(i, i, diag_block);
            if (i > 0) mat.append_block(i, i-1, off_block);
            if (i + 1 < nx) mat.append_block(i, i+1, off_block);
        }
        mat.finalize();
        return mat;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_BLOCK_HPP