//File 0113 : numdot/slicing.h
//Slice descriptors and helpers: slice, range, all, newaxis, normalization, and sliced view shape/stride computation.
#ifndef NUMDOT_SLICING_H
#define NUMDOT_SLICING_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"

namespace numdot
{
    // Tags for all and newaxis
    struct all_tag {};
    constexpr all_tag all = all_tag{};

    struct newaxis_tag {};
    constexpr newaxis_tag newaxis = newaxis_tag{};

    /**
     * @class slice
     * @brief Represents a slice with start, stop, and step.
     */
    template <class T = std::ptrdiff_t>
    class slice
    {
    public:
        using size_type = T;

        slice() noexcept : m_start(0), m_stop(0), m_step(1) {}
        slice(T start, T stop, T step = 1) noexcept : m_start(start), m_stop(stop), m_step(step) {}

        T start() const noexcept { return m_start; }
        T stop() const noexcept { return m_stop; }
        T step() const noexcept { return m_step; }

        /**
         * Normalize negative indices against a given dimension size.
         */
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

        /**
         * Number of elements in this slice after normalization.
         */
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

        std::ptrdiff_t step_size() const noexcept { return static_cast<std::ptrdiff_t>(m_step); }

    private:
        T m_start;
        T m_stop;
        T m_step;
    };

    /**
     * Range constructors for slices.
     */
    template <class T>
    inline slice<T> range(T start, T stop, T step = 1) { return slice<T>(start, stop, step); }

    template <class T>
    inline slice<T> range(T index) { return slice<T>(index, index + 1, 1); }

    template <class T>
    inline slice<T> range_from(T start, T step = 1) { return slice<T>(start, T(0), step); }

    template <class T>
    inline slice<T> range_to(T stop, T step = 1) { return slice<T>(T(0), stop, step); }

    /**
     * @class range
     * @brief Convenience class wrapping a slice with int64 step/stop.
     */
    class range
    {
    public:
        range(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1)
            : m_slice(start, stop, step) {}
        range(std::ptrdiff_t stop) : m_slice(0, stop, 1) {}

        const slice<std::ptrdiff_t>& slice() const noexcept { return m_slice; }
        std::ptrdiff_t start() const noexcept { return m_slice.start(); }
        std::ptrdiff_t stop() const noexcept { return m_slice.stop(); }
        std::ptrdiff_t step() const noexcept { return m_slice.step(); }
        std::size_t size() const noexcept { return m_slice.size(); }

    private:
        slice<std::ptrdiff_t> m_slice;
    };

    // Concrete range types for all and newaxis
    class all_range { public: all_range() = default; };
    class newaxis_range { public: newaxis_range() = default; };

    // Type traits for detecting slice types
    template <class T> struct is_slice : std::false_type {};
    template <class T> struct is_slice<slice<T>> : std::true_type {};
    template <> struct is_slice<all_tag> : std::true_type {};
    template <> struct is_slice<newaxis_tag> : std::true_type {};
    template <> struct is_slice<range> : std::true_type {};
    template <> struct is_slice<all_range> : std::true_type {};
    template <> struct is_slice<newaxis_range> : std::true_type {};
    template <class T> constexpr bool is_slice_v = is_slice<T>::value;

    /**
     * Normalize a slice against a dimension size.
     * Returns a pair (new_size, stride_multiplier).
     */
    template <class Slice>
    inline std::pair<std::size_t, std::ptrdiff_t> normalize_slice(const Slice& sl, std::size_t dim_size)
    {
        if constexpr (std::is_same_v<Slice, all_tag> || std::is_same_v<Slice, all_range>)
        {
            return {dim_size, 1};
        }
        else if constexpr (std::is_same_v<Slice, newaxis_tag> || std::is_same_v<Slice, newaxis_range>)
        {
            return {1, 0}; // new axis: size 1, stride 0
        }
        else if constexpr (std::is_same_v<Slice, std::ptrdiff_t> || std::is_same_v<Slice, int> ||
                           std::is_same_v<Slice, long> || std::is_same_v<Slice, long long>)
        {
            std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(sl);
            if (idx < 0) idx += static_cast<std::ptrdiff_t>(dim_size);
            if (idx < 0 || static_cast<std::size_t>(idx) >= dim_size)
                throw std::out_of_range("Index out of range in slice.");
            return {0, 0}; // size 0 means dimension removed
        }
        else
        {
            // Must be slice or range
            auto slc = [&]() -> slice<std::ptrdiff_t> {
                if constexpr (std::is_same_v<Slice, slice<std::ptrdiff_t>>) return sl;
                else if constexpr (std::is_same_v<Slice, range>) return sl.slice();
                else return slice<std::ptrdiff_t>(0, dim_size, 1);
            }();
            slc.normalize(dim_size);
            return {slc.size(), slc.step_size()};
        }
    }

    /**
     * Compute new shape and strides from a set of slices and the original shape/strides.
     */
    template <class SlicesTuple, class ShapeType, class StridesType>
    inline auto compute_sliced_view(const SlicesTuple& slices, const ShapeType& old_shape,
                                     const StridesType& old_strides)
    {
        using size_type = typename ShapeType::value_type;
        std::vector<size_type> new_shape;
        std::vector<size_type> new_strides;
        std::size_t current_dim = 0;

        auto process_slice = [&](const auto& sl) {
            if constexpr (std::is_same_v<std::decay_t<decltype(sl)>, newaxis_tag> ||
                          std::is_same_v<std::decay_t<decltype(sl)>, newaxis_range>)
            {
                new_shape.push_back(1);
                new_strides.push_back(0);
                return;
            }
            else
            {
                if (current_dim >= old_shape.size())
                    throw std::runtime_error("Too many slices for tensor dimensionality.");
                auto [sz, stride_mult] = normalize_slice(sl, old_shape[current_dim]);
                if (sz > 0)
                {
                    new_shape.push_back(sz);
                    new_strides.push_back(static_cast<size_type>(stride_mult * old_strides[current_dim]));
                }
                ++current_dim;
            }
        };

        std::apply([&](auto&&... args) { (process_slice(args), ...); }, slices);

        // Remaining dimensions not covered by slices are kept as is
        while (current_dim < old_shape.size())
        {
            new_shape.push_back(old_shape[current_dim]);
            new_strides.push_back(old_strides[current_dim]);
            ++current_dim;
        }

        return std::make_pair(std::move(new_shape), std::move(new_strides));
    }

} // namespace numdot

#endif // NUMDOT_SLICING_H