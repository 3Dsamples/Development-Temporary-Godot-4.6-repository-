//File 0043 : containers/xfixed.hpp
//Fixed-shape compile-time multidimensional arrays with SIMD-accelerated operations, C++17 constexpr support, and full expression integration.
#ifndef XTENSOR_XFIXED_HPP
#define XTENSOR_XFIXED_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xreducer.hpp"

namespace xt
{
    /**
     * @class xfixed_container
     * @brief Multidimensional array with compile-time fixed shape.
     *
     * Shape is encoded as template parameters, enabling stack allocation and
     * compiler optimizations. Supports full expression template interface.
     */
    template <class ET, class S, layout_type L, class Tag>
    class xfixed_container;

    template <class ET, class S, layout_type L, class Tag>
    struct xcontainer_inner_types<xfixed_container<ET, S, L, Tag>>
    {
        using storage_type = std::array<ET, std::tuple_size<S>::value == 0 ? 1 : compute_size(S{})>;
        using value_type = ET;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = S;
        using strides_type = S;
        using backstrides_type = S;
        using inner_shape_type = S;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xfixed_container<ET, S, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class ET, class S, layout_type L = DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xfixed_container : public xstrided_container<xfixed_container<ET, S, L, Tag>>,
                             public xcontainer_semantic<xfixed_container<ET, S, L, Tag>>
    {
    public:
        using self_type = xfixed_container<ET, S, L, Tag>;
        using semantic_base = xcontainer_semantic<self_type>;
        using base_type = xstrided_container<self_type>;
        using storage_type = typename xcontainer_inner_types<self_type>::storage_type;
        using value_type = ET;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = S;
        using strides_type = S;
        using backstrides_type = S;
        using container_iterator = typename storage_type::iterator;
        using const_container_iterator = typename storage_type::const_iterator;
        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;

        static constexpr std::size_t static_size = compute_size(S{});

        /**
         * Default constructor – zero-initializes all elements.
         */
        xfixed_container() noexcept : m_storage{} {}

        /**
         * Construct with a single value fill.
         */
        explicit xfixed_container(value_type val) noexcept
        {
            m_storage.fill(val);
        }

        /**
         * Construct from an initializer list (flat layout).
         */
        xfixed_container(std::initializer_list<value_type> init) noexcept
        {
            std::copy(init.begin(), init.end(), m_storage.begin());
        }

        /**
         * Construct from a shape array and optional value.
         */
        explicit xfixed_container(const shape_type& shape, value_type val = value_type()) noexcept
        {
            m_storage.fill(val);
        }

        /**
         * Construct from a shape array and strides (for view-like usage).
         */
        xfixed_container(const shape_type& shape, const strides_type& strides,
                        value_type val = value_type()) noexcept
        {
            m_storage.fill(val);
        }

        xfixed_container(const self_type&) = default;
        xfixed_container& operator=(const self_type&) = default;
        xfixed_container(self_type&&) = default;
        xfixed_container& operator=(self_type&&) = default;

        template <class E>
        xfixed_container(const xexpression<E>& e)
        {
            semantic_base::operator=(e);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }

        size_type size() const noexcept { return static_size; }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; compute_strides(); }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = detail::compute_backstrides(st, m_shape); }

        /**
         * Access operators.
         */
        template <class... Args>
        reference operator()(Args... args)
        {
            return element(args...);
        }
        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return element(args...);
        }

        reference operator[](size_type i) { return m_storage[i]; }
        const_reference operator[](size_type i) const { return m_storage[i]; }

        template <class It>
        reference element(It first, It last)
        {
            auto idx = shape_type(first, last);
            return m_storage[ravel_index(idx, m_strides)];
        }
        template <class It>
        const_reference element(It first, It last) const
        {
            auto idx = shape_type(first, last);
            return m_storage[ravel_index(idx, m_strides)];
        }

        pointer data() noexcept { return m_storage.data(); }
        const_pointer data() const noexcept { return m_storage.data(); }

        storage_type& storage() noexcept { return m_storage; }
        const storage_type& storage() const noexcept { return m_storage; }

        /**
         * Reshape – only allowed if total size remains unchanged.
         */
        template <class NewShape>
        void reshape(const NewShape& new_shape)
        {
            static_assert(std::tuple_size<NewShape>::value == std::tuple_size<S>::value,
                          "Reshape must maintain rank.");
            if (compute_size(new_shape) != static_size)
                throw std::runtime_error("xfixed reshape: total size must not change.");
            m_shape = new_shape;
            compute_strides();
        }

        /**
         * Resize is not allowed for fixed-size containers.
         */
        void resize(const shape_type&) = delete;

        using base_type::begin;
        using base_type::end;
        using base_type::cbegin;
        using base_type::cend;

    private:
        storage_type m_storage;
        shape_type m_shape = S{};
        strides_type m_strides = compute_strides(m_shape, L);
        backstrides_type m_backstrides = detail::compute_backstrides(m_strides, m_shape);

        void compute_strides()
        {
            m_strides = xt::compute_strides(m_shape, L);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }
    };

    /**
     * Convenience alias: xfixed<T, N, M, ...> for 1D, 2D, 3D, ... arrays.
     */
    template <class T, std::size_t... Dims>
    using xfixed = xfixed_container<T, std::array<std::size_t, sizeof...(Dims)>, DEFAULT_LAYOUT>;

} // namespace xt

#endif // XTENSOR_XFIXED_HPP