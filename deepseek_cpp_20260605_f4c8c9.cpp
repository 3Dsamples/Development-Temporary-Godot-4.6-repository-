//File 0214 : sparse/xsparse.hpp
//Top-level include for the xtensor-sparse module: aggregates all sparse matrix formats, solvers, views, and utilities with full C++17 support.
#ifndef XTENSOR_XSPARSE_HPP
#define XTENSOR_XSPARSE_HPP

// Sparse configuration and base
#include "xsparse_config.hpp"
#include "xsparse_expression.hpp"

// Sparse matrix formats
#include "xcoo.hpp"
#include "xcsr.hpp"
#include "xcsc.hpp"

// Sparse containers and views
#include "xsparse_array.hpp"
#include "xsparse_tensor.hpp"
#include "xsparse_view.hpp"

// Sparse operations
#include "xsparse_operation.hpp"
#include "xsparse_assign.hpp"
#include "xsparse_reducer.hpp"

// Sparse linear algebra and solvers
#include "xsparse_linalg.hpp"
#include "xsparse_solver.hpp"

// Utilities
#include "xsparse_utils.hpp"

namespace xt
{
    // Bring sparse types into xt namespace for convenience
    using sparse::xcoo_matrix;
    using sparse::xcsr_matrix;
    using sparse::xcsc_matrix;
    using sparse::xsparse_array;
    using sparse::xsparse_tensor;

    // Bring sparse functions into xt namespace
    using sparse::to_dense;
    using sparse::spdot;
    using sparse::spsum;
    using sparse::spmax;
    using sparse::spmin;
    using sparse::eye_sparse;
    using sparse::diag_sparse;
    using sparse::cg_solve;
    using sparse::bicgstab_solve;
    using sparse::gmres_solve;
    using sparse::lu_solve;
    using sparse::sparse_slice;
    using sparse::sparse_diagonal;
    using sparse::sparse_row;
    using sparse::sparse_col;
}

#endif // XTENSOR_XSPARSE_HPP