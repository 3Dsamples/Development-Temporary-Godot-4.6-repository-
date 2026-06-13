//File 0107 : numdot/broadcast.h
//Broadcasting engine: lazy broadcast expression, shape promotion, and SIMD-accelerated element mapping with automatic stride computation.
#ifndef NUMDOT_BROADCAST_H
#define NUMDOT_BROADCAST_H

#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>
#include <stdexcept>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"
#include "elementwise.h"

namespace numdot
{
    /**
     * @class broadcast_expression
     * @brief Lazy expression that broadcasts a source array to a larger shape.
     *
     * Dimensions of size 1 in the source are repeated along the corresponding
     * dimension of the target shape; new leading dimensions are also broadcast.
     * The strides of broadcasted dimensions are set to 0, so element lookup
     * automatically maps to the correct source element.
     */
    template <class E>
    class broadcast_expression : public expression<broadcast_expression<E>>
    {
    public:
        using self_type = broadcast_expression<E>;
        using value_type = typename E::value_type;
        using const_reference = typename E::const_reference;
        using reference = typename E::reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        static constexpr layout layout = default_layout;

        /**
         * Construct a broadcast expression.
         * @param source The source expression.
         * @param target_shape The desired broadcast shape.
         */
        broadcast_expression(const E& source, const shape_type& target_shape)
            : m_source(source), m_target_shape(target_shape)
        {
            auto src_shape = m_source.shape();
            if (target_shape.size() < src_shape.size())
                throw std::runtime_error("Target shape must have at least as many dimensions as source.");

            // Align shapes by prepending 1s to source shape
            shape_type aligned_src = target_shape;
            std::fill(aligned_src.begin(), aligned_src.begin() + (target_shape.size() - src_shape.size()), 1);
            std::copy(src_shape.begin(), src_shape.end(), aligned_src.begin() + (target_shape.size() - src_shape.size()));

            // Compute strides: for each dimension, if source size is 1, stride = 0; otherwise copy from source
            auto src_strides = m_source.strides();
            shape_type aligned_src_strides(target_shape.size(), 0);
            std::copy(src_strides.begin(), src_strides.end(),
                      aligned_src_strides.begin() + (target_shape.size() - src_shape.size()));

            m_strides.resize(target_shape.size());
            for (size_type d = 0; d < target_shape.size(); ++d)
            {
                if (aligned_src[d] == 1)
                    m_strides[d] = 0;
                else
                    m_strides[d] = aligned_src_strides[d];
            }
            m_backstrides = compute_backstrides(m_strides, m_target_shape);
        }

        size_type size() const noexcept { return compute_size(m_target_shape); }
        const shape_type& shape() const noexcept { return m_target_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        pointer data() noexcept { return m_source.data(); }
        const_pointer data() const noexcept { return m_source.data(); }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(args...));
        }

        const_reference operator[](size_type i) const
        {
            return operator()(i);
        }

        reference operator[](size_type i)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)[i]);
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            // Map target index to source index using source shape broadcasting
            auto src_shape = m_source.shape();
            auto src_strides = m_source.strides();
            std::size_t linear_src = 0;
            std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(m_target_shape.size()) - static_cast<std::ptrdiff_t>(src_shape.size());
            for (size_type d = 0; d < m_target_shape.size(); ++d)
            {
                size_type coord = *first++;
                std::ptrdiff_t src_dim = static_cast<std::ptrdiff_t>(d) - offset;
                if (src_dim >= 0 && static_cast<size_type>(src_dim) < src_shape.size())
                {
                    if (src_shape[src_dim] == 1)
                        coord = 0;  // broadcast
                    else
                        coord = coord; // already valid
                    linear_src += coord * src_strides[src_dim];
                }
                // else leading dimension: ignore (maps to 0 stride contribution)
            }
            return m_source.data()[linear_src];
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        // SIMD load not efficient for broadcast (non‑contiguous), fallback to scalar.

    private:
        const E& m_source;
        shape_type m_target_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
    };

    /**
     * Free function to create a broadcast expression.
     */
    template <class E>
    inline auto broadcast_to(const E& e, const std::vector<std::size_t>& target_shape)
    {
        return broadcast_expression<E>(e, target_shape);
    }

    /**
     * Helper: automatically broadcast two expressions to a common shape.
     * Returns a pair of broadcast expressions if needed.
     */
    template <class E1, class E2>
    inline auto auto_broadcast(const E1& a, const E2& b)
    {
        auto s1 = a.shape();
        auto s2 = b.shape();
        if (s1 == s2)
        {
            // No broadcasting needed
            return std::make_pair(std::cref(a), std::cref(b));
        }
        else
        {
            auto common = broadcast_shapes(s1, s2);
            return std::make_pair(broadcast_to(a, common), broadcast_to(b, common));
        }
    }

} // namespace numdot

#endif // NUMDOT_BROADCAST_H