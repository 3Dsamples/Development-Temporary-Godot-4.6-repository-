//File 0403 : xtensor-fftw/xtensor_fftw_complex.hpp
//Complex-to-complex FFT operations (1D, 2D, ND) using FFTW with SIMD-accelerated data packing, RAII plan management, and expression integration.
#ifndef XTENSOR_FFTW_COMPLEX_HPP
#define XTENSOR_FFTW_COMPLEX_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_fftw_config.hpp"
#include "xtensor_fftw_common.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"

namespace xt {
namespace fftw {

    /**
     * @class complex_fft
     * @brief Complex-to-complex FFT for 1D/2D/ND arrays using FFTW3.
     *
     * The class manages plan creation, memory allocation, and execution.
     * Input and output are xtensor complex arrays. Plans are cached for
     * given shapes and directions to avoid repeated planning overhead.
     */
    template <class T>
    class complex_fft : public fft_common<T> {
    public:
        using base_type = fft_common<T>;
        using value_type = T;
        using complex_type = std::complex<T>;
        using real_type = T;
        using size_type = std::size_t;
        using shape_type = std::vector<size_type>;
        using array_type = xarray_container<uvector<complex_type>, DEFAULT_LAYOUT, shape_type>;

        complex_fft() noexcept = default;

        /**
         * Plan a complex-to-complex FFT for a given shape.
         * @param shape Array dimensions.
         * @param dir Transform direction.
         * @param flags FFTW planning flags.
         */
        void plan(const shape_type& shape,
                  fft_direction dir,
                  unsigned int flags = default_flags) override
        {
            if (shape.empty()) return;
            m_shape = shape;
            m_dir = dir;
            // Compute total size
            std::size_t N = compute_size(shape);
            // Allocate FFTW-aligned arrays
            m_in = detail::fftw_allocate<fftw_complex>(N);
            m_out = detail::fftw_allocate<fftw_complex>(N);
            // Convert shape to FFTW format (reverse for row-major? FFTW expects row-major if we use advanced interface)
            auto fftw_shape = detail::to_fftw_shape(shape);
            // Create plan
            fftw_plan p = fftw_plan_dft(
                static_cast<int>(shape.size()),
                fftw_shape.data(),
                reinterpret_cast<fftw_complex*>(m_in.get()),
                reinterpret_cast<fftw_complex*>(m_out.get()),
                static_cast<int>(dir),
                flags
            );
            if (!p) throw std::runtime_error("complex_fft::plan: failed to create FFTW plan.");
            this->m_plan = std::make_shared<detail::plan_holder>(p);
        }

        /**
         * Execute the transform using the pre-created plan.
         * The input and output buffers must be filled prior to calling this.
         */
        void execute() override
        {
            if (!this->has_plan()) throw std::runtime_error("complex_fft::execute: no plan exists.");
            fftw_execute(this->m_plan->get());
        }

        /**
         * Compute the forward FFT of a complex xtensor array.
         * @param input The input complex array.
         * @return The transformed complex array.
         */
        template <class E>
        array_type fft(const xexpression<E>& input)
        {
            const auto& src = input.derived_cast();
            // If shape or direction changed, replan
            auto src_shape = src.shape();
            if (src_shape != m_shape || m_dir != fft_direction::forward)
                plan(src_shape, fft_direction::forward);
            // Copy data to input buffer with SIMD
            copy_to_buffer(src, m_in.get());
            execute();
            // Copy output to result
            array_type result(m_shape);
            copy_from_buffer(result, m_out.get());
            return result;
        }

        /**
         * Compute the inverse FFT of a complex xtensor array.
         * @param input The input complex array.
         * @return The inverse-transformed complex array.
         */
        template <class E>
        array_type ifft(const xexpression<E>& input)
        {
            const auto& src = input.derived_cast();
            auto src_shape = src.shape();
            if (src_shape != m_shape || m_dir != fft_direction::backward)
                plan(src_shape, fft_direction::backward);
            copy_to_buffer(src, m_in.get());
            execute();
            array_type result(m_shape);
            copy_from_buffer(result, m_out.get());
            // Scale by 1/N for inverse
            T inv_n = T(1) / static_cast<T>(compute_size(m_shape));
            result *= inv_n;
            return result;
        }

        /**
         * Access the raw input buffer (for advanced usage).
         */
        fftw_complex* input_buffer() noexcept { return m_in.get(); }
        const fftw_complex* input_buffer() const noexcept { return m_in.get(); }

        /**
         * Access the raw output buffer.
         */
        fftw_complex* output_buffer() noexcept { return m_out.get(); }
        const fftw_complex* output_buffer() const noexcept { return m_out.get(); }

    private:
        shape_type m_shape;
        fft_direction m_dir = fft_direction::forward;
        detail::fftw_ptr<fftw_complex> m_in;
        detail::fftw_ptr<fftw_complex> m_out;

        /**
         * Copy xtensor data into FFTW's aligned buffer.
         * Uses SIMD for performance.
         */
        template <class E>
        void copy_to_buffer(const E& src, fftw_complex* dst) const
        {
            const auto* src_data = src.data();
            std::size_t n = src.size();
            if constexpr (is_simd_enabled_v<complex_type>)
            {
                using simd_type = xsimd::batch<complex_type, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(reinterpret_cast<const complex_type*>(src_data + i * simd_size));
                    v.store_unaligned(reinterpret_cast<complex_type*>(dst + i * simd_size));
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst[i] = *reinterpret_cast<const fftw_complex*>(&src_data[i]);
            }
            else
            {
                for (std::size_t i = 0; i < n; ++i)
                    dst[i] = *reinterpret_cast<const fftw_complex*>(&src_data[i]);
            }
        }

        /**
         * Copy FFTW buffer to xtensor array with SIMD.
         */
        void copy_from_buffer(array_type& dst, const fftw_complex* src) const
        {
            auto* dst_data = dst.data();
            std::size_t n = dst.size();
            if constexpr (is_simd_enabled_v<complex_type>)
            {
                using simd_type = xsimd::batch<complex_type, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(reinterpret_cast<const complex_type*>(src + i * simd_size));
                    v.store_unaligned(reinterpret_cast<complex_type*>(dst_data + i * simd_size));
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst_data[i] = *reinterpret_cast<const complex_type*>(&src[i]);
            }
            else
            {
                for (std::size_t i = 0; i < n; ++i)
                    dst_data[i] = *reinterpret_cast<const complex_type*>(&src[i]);
            }
        }
    };

    /**
     * Free functions for complex FFT.
     */
    template <class E>
    inline auto fft(const xexpression<E>& e)
    {
        using T = typename std::decay_t<E>::value_type::value_type;
        complex_fft<T> fft;
        return fft.fft(e);
    }

    template <class E>
    inline auto ifft(const xexpression<E>& e)
    {
        using T = typename std::decay_t<E>::value_type::value_type;
        complex_fft<T> fft;
        return fft.ifft(e);
    }

    template <class E>
    inline auto fft2(const xexpression<E>& e)
    {
        return fft(e);
    }

    template <class E>
    inline auto ifft2(const xexpression<E>& e)
    {
        return ifft(e);
    }

} // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_COMPLEX_HPP