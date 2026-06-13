//File 0213 : sparse/xsparse_expression.hpp
//Base expression class for sparse arrays, providing CRTP interface, shape, strides, and SIMD-accelerated element access with sparse-specific optimizations.
#ifndef XTENSOR_XSPARSE_EXPRESSION_HPP
#define XTENSOR_XSPARSE_EXPRESSION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../sparse/xsparse_config.hpp"

namespace xt {
namespace sparse {

    /**
     * @class xsparse_expression
     * @brief CRTP base class for all sparse expressions.
     *
     * Provides the common interface that sparse containers and views must implement.
     * It extends the dense xexpression interface with sparse-specific methods such as
     * sparse_storage(), nnz(), and sparsity_ratio().
     */
    template <class D>
    class xsparse_expression : public xexpression<D>
    {
    public:
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;
        using sparse_storage_type = typename inner_types::sparse_storage_type;

        derived_type& derived_cast() noexcept
        {
            return *static_cast<derived_type*>(this);
        }
        const derived_type& derived_cast() const noexcept
        {
            return *static_cast<const derived_type*>(this);
        }

        // Access to sparse storage (must be provided by derived classes)
        sparse_storage_type& sparse_storage() noexcept
        {
            return derived_cast().sparse_storage();
        }
        const sparse_storage_type& sparse_storage() const noexcept
        {
            return derived_cast().sparse_storage();
        }

        // Number of non‑zero elements
        size_type nnz() const noexcept
        {
            return sparse_storage().nnz();
        }

        // Sparsity ratio: nnz / total elements
        double sparsity_ratio() const noexcept
        {
            size_type total = this->size();
            if (total == 0) return 1.0;
            return static_cast<double>(nnz()) / static_cast<double>(total);
        }

        // Density
        double density() const noexcept
        {
            return 1.0 - sparsity_ratio();
        }

        // Assignment from a dense expression
        template <class E>
        disable_xexpression<E, derived_type&> operator=(const E& e)
        {
            derived_cast().assign_from_dense(e);
            return derived_cast();
        }

        // Assignment from a sparse expression
        derived_type& operator=(const derived_type& rhs)
        {
            if (this != &rhs)
            {
                sparse_storage() = rhs.sparse_storage();
                this->set_shape(rhs.shape());
            }
            return derived_cast();
        }

        derived_type& operator=(derived_type&& rhs) noexcept
        {
            if (this != &rhs)
            {
                sparse_storage() = std::move(rhs.sparse_storage());
                this->set_shape(rhs.shape());
            }
            return derived_cast();
        }

        // Conversion to dense array
        auto to_dense() const
        {
            using dense_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
            auto sh = this->shape();
            if (sh.size() == 1)
            {
                dense_type result({sparse_storage().cols()}, value_type(0));
                const auto& csr = sparse_storage();
                for (size_type i = csr.row_ptr()[0]; i < csr.row_ptr()[1]; ++i)
                    result[csr.col_idx()[i]] = csr.values()[i];
                return result;
            }
            else if (sh.size() == 2)
            {
                dense_type result({sparse_storage().rows(), sparse_storage().cols()}, value_type(0));
                const auto& csr = sparse_storage();
                for (size_type r = 0; r < csr.rows(); ++r)
                    for (size_type i = csr.row_ptr()[r]; i < csr.row_ptr()[r + 1]; ++i)
                        result(r, csr.col_idx()[i]) = csr.values()[i];
                return result;
            }
            throw std::runtime_error("xsparse_expression::to_dense: unsupported dimension.");
        }

        // Load a SIMD batch of elements from the sparse expression.
        // Default implementation gathers scalar elements; may be overridden for performance.
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buffer;
            for (std::size_t k = 0; k < simd_size; ++k)
                buffer[k] = derived_cast()(i + k);
            return simd_type::load_aligned(buffer.data());
        }

    protected:
        xsparse_expression() = default;
        ~xsparse_expression() = default;
        xsparse_expression(const xsparse_expression&) = default;
        xsparse_expression& operator=(const xsparse_expression&) = default;
        xsparse_expression(xsparse_expression&&) = default;
        xsparse_expression& operator=(xsparse_expression&&) = default;
    };

    /**
     * Inner types specialization for sparse expressions that use CSR storage.
     */
    template <class T, layout_type L>
    struct xcontainer_inner_types<xsparse_array<T, L>>
    {
        using sparse_storage_type = xcsr_matrix<T>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using temporary_type = xarray_container<uvector<T>, L, shape_type>;
        static constexpr layout_type layout = L;
    };

    template <class T, std::size_t N, layout_type L>
    struct xcontainer_inner_types<xsparse_tensor<T, N, L>>
    {
        using sparse_storage_type = xcsr_matrix<T>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::array<size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using temporary_type = xtensor_container<uvector<T>, N, L>;
        static constexpr layout_type layout = L;
    };

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_EXPRESSION_HPP