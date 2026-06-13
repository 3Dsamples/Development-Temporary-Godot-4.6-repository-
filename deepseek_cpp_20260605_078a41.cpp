//File 0019 : core/xcontainer.hpp
//Base container class with concrete shape, strides, and data storage management.
#ifndef XTENSOR_XCONTAINER_HPP
#define XTENSOR_XCONTAINER_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    /**
     * @class xcontainer_base
     * @brief Mixin that provides concrete storage for shape, strides, and backstrides.
     *
     * Inherits from xstrided_container to get element access, and adds management
     * of dynamic or static shape arrays. This class assumes the derived type also
     * inherits from xcontainer_semantic and owns a data buffer.
     */
    template <class D, class S = std::vector<std::size_t>>
    class xcontainer_base : public xstrided_container<D>
    {
    public:
        using self_type = D;
        using base_type = xstrided_container<self_type>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename inner_types::value_type;
        using size_type = typename inner_types::size_type;
        using shape_type = S;
        using strides_type = S;
        using backstrides_type = S;
        using storage_type = typename inner_types::storage_type;

        // Constructors
        xcontainer_base() noexcept = default;

        explicit xcontainer_base(const shape_type& shape, layout_type l = DEFAULT_LAYOUT)
        {
            set_shape(shape);
            compute_strides_from_shape(l);
        }

        explicit xcontainer_base(const shape_type& shape, const strides_type& strides,
                                 layout_type l = DEFAULT_LAYOUT) noexcept
            : m_shape(shape)
            , m_strides(strides)
            , m_backstrides(detail::compute_backstrides(strides, shape))
        {
        }

        // Copy/Move
        xcontainer_base(const xcontainer_base&) = default;
        xcontainer_base(xcontainer_base&&) = default;
        xcontainer_base& operator=(const xcontainer_base&) = default;
        xcontainer_base& operator=(xcontainer_base&&) = default;

        // Shape and strides access
        const shape_type& shape() const noexcept override { return m_shape; }
        const strides_type& strides() const noexcept override { return m_strides; }
        const backstrides_type& backstrides() const noexcept override { return m_backstrides; }

        void set_shape(const shape_type& shape)
        {
            m_shape = shape;
            m_strides.resize(shape.size());
            m_backstrides.resize(shape.size());
        }

        void set_strides(const strides_type& strides)
        {
            m_strides = strides;
            m_backstrides = detail::compute_backstrides(strides, m_shape);
        }

        // Resize: allocate new storage if size changes
        void resize(const shape_type& new_shape, bool force = false)
        {
            size_type new_size = compute_size(new_shape);
            self_type& self = *static_cast<self_type*>(this);
            if (force || new_size != self.data().size())
            {
                self.storage().resize(new_size, value_type());
            }
            set_shape(new_shape);
            compute_strides_from_shape(DEFAULT_LAYOUT);
        }

        // Reshape: only change shape, no reallocation
        void reshape(const shape_type& new_shape, layout_type l = DEFAULT_LAYOUT)
        {
            size_type new_size = compute_size(new_shape);
            self_type& self = *static_cast<self_type*>(this);
            if (new_size != self.storage().size())
            {
                throw std::runtime_error("Reshape cannot change total number of elements.");
            }
            set_shape(new_shape);
            compute_strides_from_shape(l);
        }

    protected:
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;

        void compute_strides_from_shape(layout_type l)
        {
            m_strides = xt::compute_strides(m_shape, l);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }
    };

} // namespace xt

#endif // XTENSOR_XCONTAINER_HPP