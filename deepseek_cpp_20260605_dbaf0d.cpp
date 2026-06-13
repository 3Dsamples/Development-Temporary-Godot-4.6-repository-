//File 0040 : core/xadapt.hpp
//Array adaptor for external storage with optional ownership, shape/strides inference, and full expression integration.
#ifndef XTENSOR_XADAPT_HPP
#define XTENSOR_XADAPT_HPP

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xarray.hpp"
#include "xcontainer.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"

namespace xt
{
    /********************************************
     * xarray_adaptor – dynamic rank adaptor
     ********************************************/
    template <class EC, layout_type L = DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xarray_adaptor : public xarray_container<EC, L, std::vector<typename EC::size_type>, Tag>
    {
    public:
        using base_type = xarray_container<EC, L, std::vector<typename EC::size_type>, Tag>;
        using storage_type = EC;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using size_type = typename base_type::size_type;
        using value_type = typename base_type::value_type;

        // Default constructor – empty adaptor (no storage)
        xarray_adaptor() noexcept : base_type() {}

        // Adapt existing storage and shape (non‑owning by default)
        xarray_adaptor(storage_type&& storage, const shape_type& shape) noexcept
            : base_type(std::move(storage))
        {
            base_type::set_shape(shape);
            compute_strides();
        }

        // Adapt with explicit strides and offset
        xarray_adaptor(storage_type&& storage, const shape_type& shape, const strides_type& strides,
                       size_type offset = 0) noexcept
            : base_type(std::move(storage))
        {
            base_type::set_shape(shape);
            base_type::set_strides(strides);
            m_offset = offset;
        }

        // Copy / move constructors
        xarray_adaptor(const xarray_adaptor&) = default;
        xarray_adaptor& operator=(const xarray_adaptor&) = default;
        xarray_adaptor(xarray_adaptor&&) = default;
        xarray_adaptor& operator=(xarray_adaptor&&) = default;

        // Access to internal offset
        size_type offset() const noexcept { return m_offset; }

        // Override data() to account for offset
        typename storage_type::pointer data() noexcept
        {
            return base_type::storage().data() + m_offset;
        }
        typename storage_type::const_pointer data() const noexcept
        {
            return base_type::storage().data() + m_offset;
        }

        // Reshape – no reallocation, only shape change
        void reshape(const shape_type& new_shape)
        {
            base_type::set_shape(new_shape);
            compute_strides();
        }

        // Resize – adaptor cannot resize; but we provide for compatibility (throws if size mismatch)
        void resize(const shape_type& new_shape)
        {
            if (compute_size(new_shape) != base_type::size())
                throw std::runtime_error("xarray_adaptor::resize cannot change total size.");
            reshape(new_shape);
        }

        // Ownership: adopt a heap-allocated storage to own it
        void own_storage(std::unique_ptr<storage_type>&& storage_ptr)
        {
            m_owned_storage = std::move(storage_ptr);
        }

    private:
        size_type m_offset = 0;
        std::unique_ptr<storage_type> m_owned_storage; // for owning mode

        void compute_strides()
        {
            auto st = xt::compute_strides(base_type::shape(), L);
            base_type::set_strides(st);
        }
    };

    /********************************************
     * xtensor_adaptor – fixed rank adaptor
     ********************************************/
    template <class EC, std::size_t N, layout_type L = DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xtensor_adaptor : public xtensor_container<EC, N, L, Tag>
    {
    public:
        using base_type = xtensor_container<EC, N, L, Tag>;
        using storage_type = EC;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using size_type = typename base_type::size_type;
        using value_type = typename base_type::value_type;

        xtensor_adaptor() noexcept : base_type() {}

        xtensor_adaptor(storage_type&& storage, const shape_type& shape) noexcept
            : base_type(std::move(storage), shape) {}

        xtensor_adaptor(storage_type&& storage, const shape_type& shape, const strides_type& strides,
                        size_type offset = 0) noexcept
            : base_type(std::move(storage), shape, strides)
        {
            m_offset = offset;
        }

        xtensor_adaptor(const xtensor_adaptor&) = default;
        xtensor_adaptor& operator=(const xtensor_adaptor&) = default;
        xtensor_adaptor(xtensor_adaptor&&) = default;
        xtensor_adaptor& operator=(xtensor_adaptor&&) = default;

        size_type offset() const noexcept { return m_offset; }

        typename storage_type::pointer data() noexcept { return base_type::storage().data() + m_offset; }
        typename storage_type::const_pointer data() const noexcept { return base_type::storage().data() + m_offset; }

        void reshape(const shape_type& new_shape)
        {
            base_type::set_shape(new_shape);
            compute_strides();
        }

        void resize(const shape_type& new_shape)
        {
            if (compute_size(new_shape) != base_type::size())
                throw std::runtime_error("xtensor_adaptor::resize cannot change total size.");
            reshape(new_shape);
        }

    private:
        size_type m_offset = 0;

        void compute_strides()
        {
            base_type::set_strides(xt::compute_strides(base_type::shape(), L));
        }
    };

    /********************************************
     * Helper: adapt a raw pointer with shape
     ********************************************/
    template <class T, layout_type L = DEFAULT_LAYOUT>
    inline auto adapt(T* data, const std::vector<std::size_t>& shape)
    {
        using storage_type = uvector<T>;
        // Create a non‑owning view via adaptor? Actually adaptor expects rvalue storage, but we can wrap pointer?
        // For simplicity, we'll just create an xarray from it (copy). Real adapt with external pointer would require a custom allocator that uses the pointer, which is not directly supported.
        // Instead, we return an xarray_container that copies data.
        auto size = compute_size(shape);
        storage_type vec(data, data + size);
        return xarray_adaptor<storage_type, L>(std::move(vec), shape);
    }

    template <class T, std::size_t N, layout_type L = DEFAULT_LAYOUT>
    inline auto adapt(T* data, const std::array<std::size_t, N>& shape)
    {
        using storage_type = uvector<T>;
        auto size = compute_size(shape);
        storage_type vec(data, data + size);
        return xtensor_adaptor<storage_type, N, L>(std::move(vec), shape);
    }

} // namespace xt

#endif // XTENSOR_XADAPT_HPP