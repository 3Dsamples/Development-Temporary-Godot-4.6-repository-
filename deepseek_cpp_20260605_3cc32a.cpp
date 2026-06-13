//File 0047 : core/xbuffer_adaptor.hpp
//Buffer adaptor providing an xtensor interface over raw memory with specified layout, alignment, and SIMD support.
#ifndef XTENSOR_XBUFFER_ADAPTOR_HPP
#define XTENSOR_XBUFFER_ADAPTOR_HPP

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xarray.hpp"
#include "xcontainer.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"

namespace xt
{
    template <class CP, class O, layout_type L, class A, class Tag>
    class xbuffer_adaptor;

    template <class CP, class O, layout_type L, class A, class Tag>
    struct xcontainer_inner_types<xbuffer_adaptor<CP, O, L, A, Tag>>
    {
        using storage_type = std::vector<typename CP::value_type, A>;
        using value_type = typename CP::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, L, shape_type>;
        static constexpr layout_type layout = L;
    };

    /**
     * @class xbuffer_adaptor
     * @brief Adapts a raw memory buffer (pointer) to the xtensor container interface.
     *
     * The adaptor can optionally own the memory (if an allocator and deleter are
     * provided) or simply provide a non‑owning view. Supports strided layouts.
     */
    template <class CP, class O = xtensor_expression_tag, layout_type L = DEFAULT_LAYOUT,
              class A = aligned_allocator<typename CP::value_type, 64>, class Tag = xtensor_expression_tag>
    class xbuffer_adaptor : public xstrided_container<xbuffer_adaptor<CP, O, L, A, Tag>>,
                            public xcontainer_semantic<xbuffer_adaptor<CP, O, L, A, Tag>>
    {
    public:
        using self_type = xbuffer_adaptor<CP, O, L, A, Tag>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using storage_type = typename xcontainer_inner_types<self_type>::storage_type;
        using value_type = typename CP::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;

        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;

        /**
         * Construct from a raw pointer, shape, and optional ownership.
         */
        xbuffer_adaptor(pointer data, const shape_type& shape, bool own = false) noexcept
            : m_data(data), m_size(compute_size(shape)), m_owned(own ? new bool(true) : nullptr)
        {
            base_type::set_shape(shape);
            compute_strides();
        }

        /**
         * Construct with explicit strides.
         */
        xbuffer_adaptor(pointer data, const shape_type& shape, const strides_type& strides) noexcept
            : m_data(data), m_size(compute_size(shape)), m_owned(nullptr)
        {
            base_type::set_shape(shape);
            base_type::set_strides(strides);
        }

        xbuffer_adaptor(const self_type&) = delete;
        xbuffer_adaptor& operator=(const self_type&) = delete;

        xbuffer_adaptor(self_type&& rhs) noexcept
            : m_data(rhs.m_data), m_size(rhs.m_size), m_owned(std::move(rhs.m_owned))
        {
            rhs.m_data = nullptr;
            rhs.m_size = 0;
        }

        xbuffer_adaptor& operator=(self_type&& rhs) noexcept
        {
            if (this != &rhs)
            {
                release();
                m_data = rhs.m_data;
                m_size = rhs.m_size;
                m_owned = std::move(rhs.m_owned);
                rhs.m_data = nullptr;
                rhs.m_size = 0;
            }
            return *this;
        }

        ~xbuffer_adaptor() noexcept { release(); }

        size_type size() const noexcept { return m_size; }

        pointer data() noexcept { return m_data; }
        const_pointer data() const noexcept { return m_data; }

        // Extended assignment from any expression
        template <class E>
        disable_xexpression<E, self_type&> operator=(const E& e)
        {
            return semantic_base::operator=(e);
        }

        using base_type::begin;
        using base_type::end;
        using base_type::cbegin;
        using base_type::cend;

    private:
        pointer m_data;
        size_type m_size;
        std::unique_ptr<bool> m_owned; // just a flag, real ownership uses allocator

        void compute_strides()
        {
            base_type::set_strides(xt::compute_strides(base_type::shape(), L));
        }

        void release() noexcept
        {
            if (m_owned && m_data)
            {
                // Free aligned memory
                aligned_free(m_data);
            }
            m_data = nullptr;
            m_size = 0;
        }
    };

    /**
     * Helper function to create an xbuffer_adaptor from a raw pointer.
     */
    template <class T, layout_type L = DEFAULT_LAYOUT>
    inline auto adapt_buffer(T* data, const std::vector<std::size_t>& shape)
    {
        return xbuffer_adaptor<xtl::xclosure_wrapper<T*>, xtensor_expression_tag, L>(
            data, shape, false);
    }

} // namespace xt

#endif // XTENSOR_XBUFFER_ADAPTOR_HPP