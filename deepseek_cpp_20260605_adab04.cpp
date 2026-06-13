//File 0020 : core/xslice.hpp
//Slice and range utilities (xslice, xrange, xall, xnewaxis) with stepping and integration into views.
#ifndef XTENSOR_XSLICE_HPP
#define XTENSOR_XSLICE_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"

namespace xt
{
    // Placeholder for "all elements" slice
    struct xall_tag {};
    constexpr xall_tag xall = xall_tag{};

    // Placeholder for newaxis
    struct xnewaxis_tag {};
    constexpr xnewaxis_tag xnewaxis = xnewaxis_tag{};

    /**
     * @class xslice
     * @brief Represents a slice with start, stop, and step.
     *
     * Negative start/stop are resolved against a given size later.
     */
    template <class T = std::ptrdiff_t>
    class xslice
    {
    public:
        using size_type = T;

        xslice() noexcept : m_start(0), m_stop(0), m_step(1) {}
        xslice(T start, T stop, T step = 1) noexcept
            : m_start(start), m_stop(stop), m_step(step)
        {
        }

        T start() const noexcept { return m_start; }
        T stop() const noexcept { return m_stop; }
        T step() const noexcept { return m_step; }

        // Resolve this slice against a dimension size (handles negatives and bounds)
        void normalize(std::size_t dim_size) noexcept
        {
            T size = static_cast<T>(dim_size);
            if (m_start < 0) m_start += size;
            if (m_stop < 0) m_stop += size;
            if (m_step > 0)
            {
                if (m_start < 0) m_start = 0;
                if (m_stop > size) m_stop = size;
            }
            else if (m_step < 0)
            {
                if (m_start >= size) m_start = size - 1;
                if (m_stop < -1) m_stop = -1;
            }
        }

        // Number of elements in this slice after normalization
        std::size_t size() const noexcept
        {
            if (m_step > 0 && m_start >= m_stop) return 0;
            if (m_step < 0 && m_start <= m_stop) return 0;
            T diff = m_stop - m_start;
            T abs_step = m_step > 0 ? m_step : -m_step;
            T count = diff / m_step;
            if (diff % m_step != 0) count += 1;
            return static_cast<std::size_t>(count > 0 ? count : 0);
        }

        // Step size for stride computation
        std::ptrdiff_t step_size() const noexcept
        {
            return static_cast<std::ptrdiff_t>(m_step);
        }

    private:
        T m_start;
        T m_stop;
        T m_step;
    };

    // Range constructors for xslice
    template <class T>
    inline xslice<T> range(T start, T stop, T step = 1)
    {
        return xslice<T>(start, stop, step);
    }

    // Single index slice (results in size 1)
    template <class T>
    inline xslice<T> range(T index)
    {
        return xslice<T>(index, index + 1, 1);
    }

    // Slice from start to end with step (no stop)
    template <class T>
    inline xslice<T> range_from(T start, T step = 1)
    {
        return xslice<T>(start, T(0), step); // stop 0 will be resolved later to max
    }

    // Slice from 0 to stop with step
    template <class T>
    inline xslice<T> range_to(T stop, T step = 1)
    {
        return xslice<T>(T(0), stop, step);
    }

    // xrange object that can be used in slices
    class xrange
    {
    public:
        xrange(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1)
            : m_slice(start, stop, step) {}

        xrange(std::ptrdiff_t stop) : m_slice(0, stop, 1) {}

        const xslice<std::ptrdiff_t>& slice() const noexcept { return m_slice; }

        std::ptrdiff_t start() const noexcept { return m_slice.start(); }
        std::ptrdiff_t stop() const noexcept { return m_slice.stop(); }
        std::ptrdiff_t step() const noexcept { return m_slice.step(); }
        std::size_t size() const noexcept { return m_slice.size(); }

    private:
        xslice<std::ptrdiff_t> m_slice;
    };

    // xall as range (just a tag that will be replaced by full dimension later)
    struct xall_range
    {
        xall_range() = default;
        // Will be expanded to xrange(0, size, 1) when dimension size known
    };

    // xnewaxis range that adds a dimension of size 1
    struct xnewaxis_range
    {
        xnewaxis_range() = default;
    };

    // is_xslice trait
    template <class T>
    struct is_xslice : std::false_type {};
    template <class T>
    struct is_xslice<xslice<T>> : std::true_type {};
    template <>
    struct is_xslice<xall_tag> : std::true_type {};
    template <>
    struct is_xslice<xnewaxis_tag> : std::true_type {};
    template <>
    struct is_xslice<xrange> : std::true_type {};
    template <>
    struct is_xslice<xall_range> : std::true_type {};
    template <>
    struct is_xslice<xnewaxis_range> : std::true_type {};
    template <class T>
    constexpr bool is_xslice_v = is_xslice<T>::value;

    // Normalize a slice against dimension size and return the corresponding sub-view shape/stride.
    template <class Slice>
    std::pair<std::size_t, std::ptrdiff_t> normalize_slice(const Slice& sl, std::size_t dim_size)
    {
        if constexpr (std::is_same_v<Slice, xall_tag> || std::is_same_v<Slice, xall_range>)
        {
            return {dim_size, 1};
        }
        else if constexpr (std::is_same_v<Slice, xnewaxis_tag> || std::is_same_v<Slice, xnewaxis_range>)
        {
            return {1, 0}; // new axis: size 1, stride 0 (broadcast)
        }
        else if constexpr (std::is_same_v<Slice, std::ptrdiff_t> || std::is_same_v<Slice, int>)
        {
            // integer index: dimension removed, stride 0
            std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(sl);
            if (idx < 0) idx += static_cast<std::ptrdiff_t>(dim_size);
            if (idx < 0 || static_cast<std::size_t>(idx) >= dim_size)
                throw std::out_of_range("Index out of range in slice");
            // size 0 indicates dimension is dropped; stride is irrelevant
            return {0, 0};
        }
        else
        {
            // Must be xslice or xrange
            auto slc = [&]() -> xslice<std::ptrdiff_t> {
                if constexpr (std::is_same_v<Slice, xslice<std::ptrdiff_t>>) return sl;
                else if constexpr (std::is_same_v<Slice, xrange>) return sl.slice();
                else return xslice<std::ptrdiff_t>(0, dim_size, 1); // fallback
            }();
            slc.normalize(dim_size);
            return {slc.size(), slc.step_size()};
        }
    }

    // Compute the new shape and strides from a set of slices and the original shape/strides.
    // The slices parameter is a tuple of slice descriptors. Returns a pair of vectors (new_shape, new_strides).
    template <class SlicesTuple, class ShapeType, class StridesType>
    auto compute_sliced_view(const SlicesTuple& slices, const ShapeType& old_shape,
                             const StridesType& old_strides)
    {
        using size_type = typename ShapeType::value_type;
        std::vector<size_type> new_shape;
        std::vector<size_type> new_strides;
        std::size_t offset = 0;
        std::size_t current_dim = 0;

        auto process_slice = [&](const auto& sl) {
            if constexpr (std::is_same_v<std::decay_t<decltype(sl)>, xnewaxis_tag> ||
                          std::is_same_v<std::decay_t<decltype(sl)>, xnewaxis_range>)
            {
                new_shape.push_back(1);
                new_strides.push_back(0);
                return; // no current_dim advance
            }
            else
            {
                if (current_dim >= old_shape.size())
                    throw std::runtime_error("Too many slices for tensor dimensionality.");
                auto [size, stride_multiplier] = normalize_slice(sl, old_shape[current_dim]);
                if (size > 0)
                {
                    new_shape.push_back(size);
                    new_strides.push_back(static_cast<size_type>(stride_multiplier * old_strides[current_dim]));
                }
                // integer index: dimension dropped, no shape entry
                ++current_dim;
            }
        };

        std::apply([&](auto&&... args) { (process_slice(args), ...); }, slices);

        // Remaining dimensions (if slices didn't cover all) are kept as is
        while (current_dim < old_shape.size())
        {
            new_shape.push_back(old_shape[current_dim]);
            new_strides.push_back(old_strides[current_dim]);
            ++current_dim;
        }

        return std::make_pair(std::move(new_shape), std::move(new_strides));
    }

} // namespace xt

#endif // XTENSOR_XSLICE_HPP