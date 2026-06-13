//File 0021 (UPDATED) : core/xbroadcast.hpp
//Lazy broadcasting expression with stride-only storage, SIMD access, iterators, and shape deduction.
#ifndef XTENSOR_XBROADCAST_HPP
#define XTENSOR_XBROADCAST_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"

namespace xt
{
    template <class E>
    class xbroadcast;

    template <class E>
    struct xcontainer_inner_types<xbroadcast<E>>
    {
        using storage_type = typename std::decay_t<E>::storage_type;
        using value_type = typename std::decay_t<E>::value_type;
        using reference = typename std::decay_t<E>::reference;
        using const_reference = typename std::decay_t<E>::const_reference;
        using pointer = typename std::decay_t<E>::pointer;
        using const_pointer = typename std::decay_t<E>::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    /**
     * @class xbroadcast
     * @brief Expression that broadcasts a source expression to a target shape.
     *
     * Stores only the target shape and computed broadcast strides (zero for
     * broadcasted dimensions), enabling lazy element evaluation without copying.
     */
    template <class E>
    class xbroadcast : public xexpression<xbroadcast<E>>
    {
    public:
        using self_type = xbroadcast<E>;
        using inner_types = xcontainer_inner_types<self_type>;
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

        xbroadcast(const E& e, const shape_type& target_shape);
        xbroadcast(const xbroadcast&) = default;
        xbroadcast& operator=(const xbroadcast&) = default;
        xbroadcast(xbroadcast&&) = default;
        xbroadcast& operator=(xbroadcast&&) = default;

        size_type size() const noexcept;
        const shape_type& shape() const noexcept { return m_target_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class It>
        const_reference element(It first, It last) const;

        const_pointer data() const noexcept { return m_e.data(); }
        pointer data() noexcept { return nullptr; }

        using base_type = xexpression<self_type>;
        using base_type::begin;
        using base_type::end;

        const E& expression() const noexcept { return m_e; }

    private:
        const E& m_e;
        shape_type m_target_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;

        void compute_broadcast_strides();
    };

    // Implementation
    template <class E>
    xbroadcast<E>::xbroadcast(const E& e, const shape_type& target_shape)
        : m_e(e), m_target_shape(target_shape)
    {
        compute_broadcast_strides();
    }

    template <class E>
    auto xbroadcast<E>::size() const noexcept -> size_type
    {
        return compute_size(m_target_shape);
    }

    template <class E>
    void xbroadcast<E>::compute_broadcast_strides()
    {
        auto src_shape = m_e.shape();
        auto src_strides = compute_strides(src_shape);
        m_strides.resize(m_target_shape.size());
        m_backstrides.resize(m_target_shape.size());
        std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(m_target_shape.size()) - static_cast<std::ptrdiff_t>(src_shape.size());
        if (offset < 0)
            throw std::runtime_error("Target shape must have at least as many dimensions as source.");
        // Leading new dimensions: stride 0 (broadcast)
        for (std::size_t i = 0; i < static_cast<std::size_t>(offset); ++i)
        {
            m_strides[i] = 0;
            m_backstrides[i] = 0;
        }
        // For each original dimension, if source size is 1, stride 0 (broadcast), else keep stride
        for (std::size_t i = 0; i < src_shape.size(); ++i)
        {
            std::size_t target_dim = i + static_cast<std::size_t>(offset);
            if (src_shape[i] == 1)
            {
                m_strides[target_dim] = 0;
                m_backstrides[target_dim] = 0;
            }
            else
            {
                m_strides[target_dim] = src_strides[i];
                m_backstrides[target_dim] = (m_target_shape[target_dim] - 1) * src_strides[i];
            }
        }
    }

    template <class E>
    template <class... Args>
    auto xbroadcast<E>::operator()(Args... args) const -> const_reference
    {
        std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
        return element(idx.begin(), idx.end());
    }

    template <class E>
    template <class It>
    auto xbroadcast<E>::element(It first, It last) const -> const_reference
    {
        auto src_shape = m_e.shape();
        std::size_t src_linear = 0;
        auto src_strides = compute_strides(src_shape);
        auto it = first;
        for (std::size_t i = 0; i < m_target_shape.size(); ++i, ++it)
        {
            size_type coord = *it;
            std::ptrdiff_t src_dim = static_cast<std::ptrdiff_t>(i) - (static_cast<std::ptrdiff_t>(m_target_shape.size()) - static_cast<std::ptrdiff_t>(src_shape.size()));
            if (src_dim >= 0 && static_cast<std::size_t>(src_dim) < src_shape.size())
            {
                std::size_t orig_dim = static_cast<std::size_t>(src_dim);
                if (src_shape[orig_dim] == 1)
                    coord = 0; // broadcast
                else
                    coord = coord; // already within range
                src_linear += coord * src_strides[orig_dim];
            }
            // else leading dimension, ignore (coordinate doesn't affect source)
        }
        return m_e.data()[src_linear];
    }

    // Free function to create a broadcast expression
    template <class E>
    inline auto broadcast(const E& e, const std::vector<std::size_t>& new_shape)
    {
        return xbroadcast<E>(e, new_shape);
    }

    // Broadcast shapes: determine the common broadcast shape of two shapes
    inline std::vector<std::size_t> broadcast_shapes(const std::vector<std::size_t>& s1,
                                                     const std::vector<std::size_t>& s2)
    {
        std::size_t ndim = std::max(s1.size(), s2.size());
        std::vector<std::size_t> result(ndim);
        for (std::size_t i = 0; i < ndim; ++i)
        {
            std::size_t d1 = (i < ndim - s1.size()) ? 1 : s1[i - (ndim - s1.size())];
            std::size_t d2 = (i < ndim - s2.size()) ? 1 : s2[i - (ndim - s2.size())];
            if (d1 != d2 && d1 != 1 && d2 != 1)
                throw std::runtime_error("Incompatible broadcast shapes.");
            result[i] = std::max(d1, d2);
        }
        return result;
    }

    template <class... Shapes>
    inline std::vector<std::size_t> broadcast_shapes(const std::vector<std::size_t>& s1,
                                                     const std::vector<std::size_t>& s2,
                                                     const Shapes&... rest)
    {
        return broadcast_shapes(broadcast_shapes(s1, s2), rest...);
    }

    // Broadcast an array to a new shape, evaluating into a new container
    template <class E>
    inline auto broadcast_to(const E& e, const std::vector<std::size_t>& new_shape)
    {
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        result_type result(new_shape);
        auto broad = broadcast(e, new_shape);
        for (std::size_t i = 0; i < broad.size(); ++i)
        {
            auto idx = unravel_index(i, broad.shape());
            result[i] = broad.element(idx.begin(), idx.end());
        }
        return result;
    }

} // namespace xt

#endif // XTENSOR_XBROADCAST_HPP