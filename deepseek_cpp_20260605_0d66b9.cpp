//File 0405 : xtensor-fftw/xtensor_fftw_multidim.hpp
//Multi-dimensional complex and real FFT (n-D) with FFTW advanced interface, SIMD data packing, stride handling, and RAII plan management.
#ifndef XTENSOR_FFTW_MULTIDIM_HPP
#define XTENSOR_FFTW_MULTIDIM_HPP

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
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xtensor_simd.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"

namespace xt {
namespace fftw {

    namespace detail {
        /**
         * Compute strides for a given shape in row-major (C order) layout.
         */
        inline std::vector<std::size_t> compute_fftw_strides(const std::vector<std::size_t>& shape) {
            if (shape.empty()) return {};
            std::vector<std::size_t> strides(shape.size(), 1);
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2; i >= 0; --i)
                strides[static_cast<std::size_t>(i)] = strides[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
            return strides;
        }
    }

    /**
     * @class multidim_fft
     * @brief Multi-dimensional FFT (complex-to-complex and real-to-complex) using FFTW advanced interface.
     *
     * This class plans and executes n-dimensional transforms of arbitrary rank.
     * It supports both complex and real input. For real input, the last dimension
     * is transformed to half-spectrum.
     */
    template <class T>
    class multidim_fft : public fft_common<T> {
    public:
        using base_type = fft_common<T>;
        using value_type = T;
        using complex_type = std::complex<T>;
        using real_type = T;
        using size_type = std::size_t;
        using shape_type = std::vector<size_type>;
        using complex_array_type = xarray_container<uvector<complex_type>, DEFAULT_LAYOUT, shape_type>;
        using real_array_type = xarray_container<uvector<T>, DEFAULT_LAYOUT, shape_type>;

        multidim_fft() noexcept = default;

        /**
         * Plan a complex-to-complex n-D transform.
         * @param shape The shape of the array.
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
            std::size_t N = compute_size(shape);
            m_in = detail::fftw_allocate<fftw_complex>(N);
            m_out = detail::fftw_allocate<fftw_complex>(N);

            auto fftw_dims = detail::to_fftw_shape(shape);
            fftw_plan p = fftw_plan_dft(
                static_cast<int>(shape.size()),
                fftw_dims.data(),
                reinterpret_cast<fftw_complex*>(m_in.get()),
                reinterpret_cast<fftw_complex*>(m_out.get()),
                static_cast<int>(dir),
                flags
            );
            if (!p) throw std::runtime_error("multidim_fft::plan: failed to create FFTW plan.");
            this->m_plan = std::make_shared<detail::plan_holder>(p);
        }

        /**
         * Plan a real-to-complex n-D transform (forward only) or complex-to-real (backward).
         * For real transforms, the last dimension's output is n/2+1.
         */
        void plan_real(const shape_type& shape,
                       fft_direction dir,
                       unsigned int flags = default_flags)
        {
            if (shape.empty()) return;
            m_shape = shape;
            m_dir = dir;
            std::size_t N = compute_size(shape);
            if (dir == fft_direction::forward)
            {
                m_real_in = detail::fftw_allocate<T>(N);
                m_half_shape = shape;
                m_half_shape.back() = shape.back() / 2 + 1;
                std::size_t Nh = compute_size(m_half_shape);
                m_complex_out = detail::fftw_allocate<fftw_complex>(Nh);
                m_real_out = nullptr;
                m_complex_in = nullptr;
            }
            else
            {
                m_half_shape = shape;
                m_half_shape.back() = shape.back() / 2 + 1;
                std::size_t Nh = compute_size(m_half_shape);
                m_complex_in = detail::fftw_allocate<fftw_complex>(Nh);
                m_real_out = detail::fftw_allocate<T>(N);
                m_real_in = nullptr;
                m_complex_out = nullptr;
            }

            auto fftw_dims = detail::to_fftw_shape(shape);
            fftw_plan p;
            if (dir == fft_direction::forward)
            {
                p = fftw_plan_dft_r2c(
                    static_cast<int>(shape.size()),
                    fftw_dims.data(),
                    m_real_in.get(),
                    reinterpret_cast<fftw_complex*>(m_complex_out.get()),
                    flags
                );
            }
            else
            {
                p = fftw_plan_dft_c2r(
                    static_cast<int>(shape.size()),
                    fftw_dims.data(),
                    reinterpret_cast<fftw_complex*>(m_complex_in.get()),
                    m_real_out.get(),
                    flags
                );
            }
            if (!p) throw std::runtime_error("multidim_fft::plan_real: failed to create FFTW plan.");
            this->m_plan = std::make_shared<detail::plan_holder>(p);
        }

        void execute() override
        {
            if (!this->has_plan()) throw std::runtime_error("multidim_fft::execute: no plan.");
            fftw_execute(this->m_plan->get());
        }

        /**
         * Forward complex n-D FFT.
         */
        template <class E>
        complex_array_type fftn(const xexpression<E>& input)
        {
            const auto& src = input.derived_cast();
            auto src_shape = src.shape();
            if (src_shape != m_shape || m_dir != fft_direction::forward)
                plan(src_shape, fft_direction::forward);
            copy_complex_to_buffer(src, m_in.get());
            execute();
            complex_array_type result(m_shape);
            copy_complex_from_buffer(result, m_out.get());
            return result;
        }

        /**
         * Inverse complex n-D FFT.
         */
        template <class E>
        complex_array_type ifftn(const xexpression<E>& input)
        {
            const auto& src = input.derived_cast();
            auto src_shape = src.shape();
            if (src_shape != m_shape || m_dir != fft_direction::backward)
                plan(src_shape, fft_direction::backward);
            copy_complex_to_buffer(src, m_in.get());
            execute();
            complex_array_type result(m_shape);
            copy_complex_from_buffer(result, m_out.get());
            T inv_n = T(1) / static_cast<T>(compute_size(m_shape));
            result *= inv_n;
            return result;
        }

        /**
         * Forward real n-D FFT (returns half-spectrum in last dimension).
         */
        template <class E>
        complex_array_type rfftn(const xexpression<E>& input)
        {
            const auto& src = input.derived_cast();
            auto src_shape = src.shape();
            if (src_shape != m_shape || m_dir != fft_direction::forward)
                plan_real(src_shape, fft_direction::forward);
            copy_real_to_buffer(src, m_real_in.get());
            execute();
            complex_array_type result(m_half_shape);
            copy_complex_from_buffer(result, m_complex_out.get());
            return result;
        }

        /**
         * Inverse complex n-D FFT to real (from half-spectrum).
         */
        template <class E>
        real_array_type irfftn(const xexpression<E>& input, const shape_type& real_shape)
        {
            const auto& src = input.derived_cast();
            if (real_shape != m_shape || m_dir != fft_direction::backward)
                plan_real(real_shape, fft_direction::backward);
            copy_complex_to_buffer(src, m_complex_in.get());
            execute();
            real_array_type result(real_shape);
            copy_real_from_buffer(result, m_real_out.get());
            T inv_n = T(1) / static_cast<T>(compute_size(real_shape));
            result *= inv_n;
            return result;
        }

    private:
        shape_type m_shape;
        shape_type m_half_shape;
        fft_direction m_dir = fft_direction::forward;
        detail::fftw_ptr<fftw_complex> m_in;
        detail::fftw_ptr<fftw_complex> m_out;
        detail::fftw_ptr<T> m_real_in;
        detail::fftw_ptr<fftw_complex> m_complex_out;
        detail::fftw_ptr<fftw_complex> m_complex_in;
        detail::fftw_ptr<T> m_real_out;

        template <class E>
        void copy_complex_to_buffer(const E& src, fftw_complex* dst) const {
            const auto* src_data = src.data();
            std::size_t n = src.size();
            if constexpr (is_simd_enabled_v<complex_type>) {
                using simd_type = xsimd::batch<complex_type, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i) {
                    simd_type v = simd_type::load_unaligned(reinterpret_cast<const complex_type*>(src_data + i * simd_size));
                    v.store_unaligned(reinterpret_cast<complex_type*>(dst + i * simd_size));
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst[i] = *reinterpret_cast<const fftw_complex*>(&src_data[i]);
            } else {
                for (std::size_t i = 0; i < n; ++i)
                    dst[i] = *reinterpret_cast<const fftw_complex*>(&src_data[i]);
            }
        }

        void copy_complex_from_buffer(complex_array_type& dst, const fftw_complex* src) const {
            auto* dst_data = dst.data();
            std::size_t n = dst.size();
            if constexpr (is_simd_enabled_v<complex_type>) {
                using simd_type = xsimd::batch<complex_type, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i) {
                    simd_type v = simd_type::load_unaligned(reinterpret_cast<const complex_type*>(src + i * simd_size));
                    v.store_unaligned(reinterpret_cast<complex_type*>(dst_data + i * simd_size));
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst_data[i] = *reinterpret_cast<const complex_type*>(&src[i]);
            } else {
                for (std::size_t i = 0; i < n; ++i)
                    dst_data[i] = *reinterpret_cast<const complex_type*>(&src[i]);
            }
        }

        template <class E>
        void copy_real_to_buffer(const E& src, T* dst) const {
            const auto* src_data = src.data();
            std::size_t n = src.size();
            if constexpr (is_simd_enabled_v<T>) {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i) {
                    simd_type v = simd_type::load_unaligned(src_data + i * simd_size);
                    v.store_unaligned(dst + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst[i] = src_data[i];
            } else {
                std::copy(src_data, src_data + n, dst);
            }
        }

        void copy_real_from_buffer(real_array_type& dst, const T* src) const {
            auto* dst_data = dst.data();
            std::size_t n = dst.size();
            if constexpr (is_simd_enabled_v<T>) {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i) {
                    simd_type v = simd_type::load_unaligned(src + i * simd_size);
                    v.store_unaligned(dst_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst_data[i] = src[i];
            } else {
                std::copy(src, src + n, dst_data);
            }
        }
    };

    // Free functions for multi-dimensional FFT
    template <class E>
    inline auto fftn(const xexpression<E>& e) {
        using T = typename std::decay_t<E>::value_type::value_type;
        multidim_fft<T> fft;
        return fft.fftn(e);
    }

    template <class E>
    inline auto ifftn(const xexpression<E>& e) {
        using T = typename std::decay_t<E>::value_type::value_type;
        multidim_fft<T> fft;
        return fft.ifftn(e);
    }

    template <class E>
    inline auto rfftn(const xexpression<E>& e) {
        using T = typename std::decay_t<E>::value_type;
        multidim_fft<T> fft;
        return fft.rfftn(e);
    }

    template <class E>
    inline auto irfftn(const xexpression<E>& e, const std::vector<std::size_t>& real_shape) {
        using T = typename std::decay_t<E>::value_type::value_type;
        multidim_fft<T> fft;
        return fft.irfftn(e, real_shape);
    }

} // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_MULTIDIM_HPP