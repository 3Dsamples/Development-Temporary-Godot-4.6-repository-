//File 0404 : xtensor-fftw/xtensor_fftw_real.hpp
//Real-to-complex and complex-to-real FFT operations with SIMD-accelerated data packing, half-spectrum handling, and RAII plan management.
#ifndef XTENSOR_FFTW_REAL_HPP
#define XTENSOR_FFTW_REAL_HPP

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
     * @class real_fft
     * @brief Real-to-complex and complex-to-real FFT using FFTW3.
     *
     * For real input, the output is the half-spectrum (n/2+1 complex values).
     * For complex input (half-spectrum), the output is a real array.
     * Plans are cached for given shapes and directions.
     */
    template <class T>
    class real_fft : public fft_common<T> {
    public:
        using base_type = fft_common<T>;
        using value_type = T;
        using complex_type = std::complex<T>;
        using real_type = T;
        using size_type = std::size_t;
        using shape_type = std::vector<size_type>;
        using real_array_type = xarray_container<uvector<T>, DEFAULT_LAYOUT, shape_type>;
        using complex_array_type = xarray_container<uvector<complex_type>, DEFAULT_LAYOUT, shape_type>;

        real_fft() noexcept = default;

        /**
         * Plan a real-to-complex FFT.
         * @param shape The shape of the real input array (n0, n1, ...).
         * @param dir Must be forward (real->complex) or backward (complex->real).
         * @param flags FFTW planning flags.
         */
        void plan(const shape_type& shape,
                  fft_direction dir,
                  unsigned int flags = default_flags) override
        {
            if (shape.empty()) return;
            m_shape = shape;
            m_dir = dir;
            std::size_t N = compute_size(shape);
            // Output half-spectrum size: last dimension becomes n/2+1
            m_half_shape = shape;
            m_half_shape.back() = shape.back() / 2 + 1;
            std::size_t Nh = compute_size(m_half_shape);
            if (dir == fft_direction::forward)
            {
                m_real_in = detail::fftw_allocate<T>(N);
                m_complex_out = detail::fftw_allocate<fftw_complex>(Nh);
                m_real_out = nullptr;
                m_complex_in = nullptr;
            }
            else
            {
                m_complex_in = detail::fftw_allocate<fftw_complex>(Nh);
                m_real_out = detail::fftw_allocate<T>(N);
                m_real_in = nullptr;
                m_complex_out = nullptr;
            }
            auto fftw_shape = detail::to_fftw_shape(shape);
            fftw_plan p;
            if (dir == fft_direction::forward)
            {
                p = fftw_plan_dft_r2c(
                    static_cast<int>(shape.size()),
                    fftw_shape.data(),
                    m_real_in.get(),
                    reinterpret_cast<fftw_complex*>(m_complex_out.get()),
                    flags
                );
            }
            else
            {
                p = fftw_plan_dft_c2r(
                    static_cast<int>(shape.size()),
                    fftw_shape.data(),
                    reinterpret_cast<fftw_complex*>(m_complex_in.get()),
                    m_real_out.get(),
                    flags
                );
            }
            if (!p) throw std::runtime_error("real_fft::plan: failed to create FFTW plan.");
            this->m_plan = std::make_shared<detail::plan_holder>(p);
        }

        /**
         * Execute the transform.
         */
        void execute() override
        {
            if (!this->has_plan()) throw std::runtime_error("real_fft::execute: no plan exists.");
            fftw_execute(this->m_plan->get());
        }

        /**
         * Forward real-to-complex transform.
         * @param input Real-valued array (shape matches planned shape).
         * @return Complex half-spectrum with last dimension size = n/2+1.
         */
        template <class E>
        complex_array_type rfft(const xexpression<E>& input)
        {
            const auto& src = input.derived_cast();
            auto src_shape = src.shape();
            if (src_shape != m_shape || m_dir != fft_direction::forward)
                plan(src_shape, fft_direction::forward);
            // Copy real input to FFTW buffer
            copy_real_to_buffer(src, m_real_in.get());
            execute();
            complex_array_type result(m_half_shape);
            copy_complex_from_buffer(result, m_complex_out.get());
            return result;
        }

        /**
         * Inverse complex-to-real transform.
         * @param input Complex half-spectrum (last dimension size = n/2+1).
         * @param n_out The desired real output size (must be <= last_dim*2-2? Usually the original n).
         * @return Real-valued array of length n_out.
         */
        template <class E>
        real_array_type irfft(const xexpression<E>& input, std::size_t n_out = 0)
        {
            const auto& src = input.derived_cast();
            auto src_shape = src.shape();
            // Infer original real length from half-spectrum: n_original = 2*(n_half-1)
            std::size_t n_original = (src_shape.back() - 1) * 2;
            if (n_out == 0) n_out = n_original;
            if (n_out > n_original) n_out = n_original;
            shape_type real_shape = src_shape;
            real_shape.back() = n_out;
            if (real_shape != m_shape || m_dir != fft_direction::backward)
                plan(real_shape, fft_direction::backward);
            // Copy complex input to FFTW buffer
            copy_complex_to_buffer(src, m_complex_in.get());
            execute();
            real_array_type result(real_shape);
            copy_real_from_buffer(result, m_real_out.get());
            // Scale by 1/N for inverse
            T inv_n = T(1) / static_cast<T>(compute_size(real_shape));
            result *= inv_n;
            return result;
        }

    private:
        shape_type m_shape;
        shape_type m_half_shape;
        fft_direction m_dir = fft_direction::forward;
        detail::fftw_ptr<T> m_real_in;
        detail::fftw_ptr<fftw_complex> m_complex_out;
        detail::fftw_ptr<fftw_complex> m_complex_in;
        detail::fftw_ptr<T> m_real_out;

        template <class E>
        void copy_real_to_buffer(const E& src, T* dst) const
        {
            const auto* src_data = src.data();
            std::size_t n = src.size();
            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(src_data + i * simd_size);
                    v.store_unaligned(dst + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst[i] = src_data[i];
            }
            else
            {
                std::copy(src_data, src_data + n, dst);
            }
        }

        void copy_complex_from_buffer(complex_array_type& dst, const fftw_complex* src) const
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

        template <class E>
        void copy_complex_to_buffer(const E& src, fftw_complex* dst) const
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

        void copy_real_from_buffer(real_array_type& dst, const T* src) const
        {
            auto* dst_data = dst.data();
            std::size_t n = dst.size();
            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(src + i * simd_size);
                    v.store_unaligned(dst_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst_data[i] = src[i];
            }
            else
            {
                std::copy(src, src + n, dst_data);
            }
        }
    };

    /**
     * Free functions for real FFT.
     */
    template <class E>
    inline auto rfft(const xexpression<E>& e)
    {
        using T = typename std::decay_t<E>::value_type;
        real_fft<T> fft;
        return fft.rfft(e);
    }

    template <class E>
    inline auto irfft(const xexpression<E>& e, std::size_t n_out = 0)
    {
        using T = typename std::decay_t<E>::value_type::value_type;
        real_fft<T> fft;
        return fft.irfft(e, n_out);
    }

    template <class E>
    inline auto rfft2(const xexpression<E>& e)
    {
        return rfft(e);
    }

    template <class E>
    inline auto irfft2(const xexpression<E>& e)
    {
        return irfft(e);
    }

} // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_REAL_HPP