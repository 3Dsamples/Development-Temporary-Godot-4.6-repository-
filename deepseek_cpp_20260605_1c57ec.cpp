//File 0401 : xtensor-fftw/xtensor_fftw_common.hpp
//Core FFTW wrapper with planning, execution, memory management, SIMD-accelerated data transfer, and RAII-based plan lifetime management.
#ifndef XTENSOR_FFTW_COMMON_HPP
#define XTENSOR_FFTW_COMMON_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_fftw_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xtensor_simd.hpp"
#include "xtensor/xcomplex.hpp"

namespace xt {
namespace fftw {

    namespace detail {

        /**
         * RAII wrapper around fftw_complex* allocated by FFTW.
         * Automatically calls fftw_free on destruction.
         */
        struct fftw_malloc_deleter {
            void operator()(void* ptr) const noexcept {
                if (ptr) fftw_free(ptr);
            }
        };

        template <class T>
        using fftw_ptr = std::unique_ptr<T, fftw_malloc_deleter>;

        /**
         * Allocate aligned memory using FFTW's aligned allocator.
         */
        template <class T>
        inline auto fftw_allocate(std::size_t count) {
            void* ptr = fftw_malloc(count * sizeof(T));
            if (!ptr) throw std::bad_alloc();
            return fftw_ptr<T>(static_cast<T*>(ptr));
        }

        /**
         * RAII wrapper around fftw_plan.
         * Automatically calls fftw_destroy_plan on destruction.
         */
        class plan_holder {
        public:
            plan_holder() noexcept : m_plan(nullptr) {}
            explicit plan_holder(fftw_plan p) noexcept : m_plan(p) {}
            plan_holder(const plan_holder&) = delete;
            plan_holder& operator=(const plan_holder&) = delete;

            plan_holder(plan_holder&& other) noexcept : m_plan(other.m_plan) {
                other.m_plan = nullptr;
            }

            plan_holder& operator=(plan_holder&& other) noexcept {
                if (this != &other) {
                    if (m_plan) fftw_destroy_plan(m_plan);
                    m_plan = other.m_plan;
                    other.m_plan = nullptr;
                }
                return *this;
            }

            ~plan_holder() noexcept {
                if (m_plan) fftw_destroy_plan(m_plan);
            }

            fftw_plan get() const noexcept { return m_plan; }
            explicit operator bool() const noexcept { return m_plan != nullptr; }

        private:
            fftw_plan m_plan;
        };

        /**
         * Thread-local wisdom management for plan reuse.
         */
        inline void import_wisdom_from_string(const std::string& wisdom) {
            fftw_import_wisdom_from_string(wisdom.c_str());
        }

        inline std::string export_wisdom_to_string() {
            char* w = fftw_export_wisdom_to_string();
            if (!w) return {};
            std::string result(w);
            std::free(w);
            return result;
        }

        /**
         * Set the number of threads for FFTW planning and execution.
         */
        inline void set_num_threads(std::size_t n) {
            fftw_plan_with_nthreads(static_cast<int>(n));
        }

        inline std::size_t get_num_threads() noexcept {
            return static_cast<std::size_t>(std::max(1, fftw_planner_nthreads()));
        }

        /**
         * Convert xtensor layout to FFTW stride convention.
         * FFTW expects the strides in its own format (usually row-major).
         */
        template <class Shape, class Strides>
        inline std::vector<int> to_fftw_strides(const Shape& shape, const Strides& strides) {
            std::vector<int> fftw_strides(shape.size());
            for (std::size_t i = 0; i < shape.size(); ++i)
                fftw_strides[i] = static_cast<int>(strides[i]);
            return fftw_strides;
        }

        template <class Shape>
        inline std::vector<int> to_fftw_shape(const Shape& shape) {
            std::vector<int> fftw_shape(shape.size());
            for (std::size_t i = 0; i < shape.size(); ++i)
                fftw_shape[i] = static_cast<int>(shape[i]);
            return fftw_shape;
        }

        /**
         * Check if a type is complex (std::complex<T>).
         */
        template <class T> struct is_complex : std::false_type {};
        template <class T> struct is_complex<std::complex<T>> : std::true_type {};
        template <class T> inline constexpr bool is_complex_v = is_complex<T>::value;

        /**
         * Get the corresponding real type for complex, or identity for real.
         */
        template <class T> struct real_type_of { using type = T; };
        template <class T> struct real_type_of<std::complex<T>> { using type = T; };
        template <class T> using real_type_of_t = typename real_type_of<T>::type;

    } // namespace detail

    /**
     * @class fft_common
     * @brief Common base for all FFT operations, managing plan creation and execution.
     *
     * Stores a shared pointer to the plan holder so multiple views of the same
     * plan can share its lifetime. Provides forward/backward transform methods.
     */
    template <class T>
    class fft_common {
    public:
        using value_type = T;
        using real_type = detail::real_type_of_t<T>;
        using size_type = std::size_t;

        fft_common() noexcept = default;
        virtual ~fft_common() = default;

        /**
         * Set the number of FFTW threads.
         */
        static void set_num_threads(std::size_t n) {
            detail::set_num_threads(n);
        }

        static std::size_t get_num_threads() noexcept {
            return detail::get_num_threads();
        }

        /**
         * Import wisdom from a string to reuse existing plans.
         */
        static void import_wisdom(const std::string& w) {
            detail::import_wisdom_from_string(w);
        }

        /**
         * Export current wisdom to a string.
         */
        static std::string export_wisdom() {
            return detail::export_wisdom_to_string();
        }

        /**
         * Forget accumulated wisdom.
         */
        static void forget_wisdom() noexcept {
            fftw_forget_wisdom();
        }

        /**
         * Create a new plan for the given shape and direction.
         * @param shape The dimensions.
         * @param dir FFT direction (forward/backward).
         * @param flags Planning flags.
         */
        virtual void plan(const std::vector<size_type>& shape,
                          fft_direction dir,
                          unsigned int flags = default_flags) = 0;

        /**
         * Execute the transform.
         */
        virtual void execute() = 0;

        /**
         * Check if a plan has been created.
         */
        bool has_plan() const noexcept { return m_plan.get() != nullptr; }

    protected:
        std::shared_ptr<detail::plan_holder> m_plan;
    };

} // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_COMMON_HPP