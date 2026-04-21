// core/xexpression.hpp
#ifndef XTENSOR_XEXPRESSION_HPP
#define XTENSOR_XEXPRESSION_HPP

// ----------------------------------------------------------------------------
// xexpression.hpp – Base classes for the xtensor expression system
// ----------------------------------------------------------------------------
// This header defines the CRTP base classes for all xtensor expressions:
//   - xexpression<D>: root of the expression hierarchy
//   - xcontainer_expression<D>: base for containers with storage
//   - xview_expression<D>: base for non‑owning views
//
// It also provides core type traits and helper functions used throughout
// the library, enhanced with BigNumber and FFT awareness for high‑performance
// scientific simulation, real‑time animation, and multi‑scale physics.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <iterator>
#include <initializer_list>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ========================================================================
    // xexpression – Base class for all expressions (CRTP)
    // ========================================================================
    template <class D>
    class xexpression
    {
    public:
        using derived_type = D;

        // --------------------------------------------------------------------
        // Access to the derived object
        // --------------------------------------------------------------------
        const derived_type& derived_cast() const noexcept;
        derived_type& derived_cast() noexcept;

        // --------------------------------------------------------------------
        // Shape and size queries (forwarded to derived)
        // --------------------------------------------------------------------
        shape_type shape() const;
        size_type size() const;
        size_type dimension() const;
        bool empty() const noexcept;

        // --------------------------------------------------------------------
        // Element access (forwarded to derived)
        // --------------------------------------------------------------------
        template <class... Args>
        auto operator()(Args... args) const -> decltype(std::declval<const derived_type&>()(args...));
        template <class... Args>
        auto operator()(Args... args) -> decltype(std::declval<derived_type&>()(args...));
        template <class S>
        auto operator[](const S& index) const -> decltype(std::declval<const derived_type&>()[index]);
        template <class S>
        auto operator[](const S& index) -> decltype(std::declval<derived_type&>()[index]);
        auto operator[](size_type i) const -> decltype(std::declval<const derived_type&>()[i]);
        auto operator[](size_type i) -> decltype(std::declval<derived_type&>()[i]);

        // --------------------------------------------------------------------
        // Flat indexing (used internally by expression evaluators)
        // --------------------------------------------------------------------
        auto flat(size_type i) const -> decltype(std::declval<const derived_type&>().flat(i));

        // --------------------------------------------------------------------
        // Begin/end iterators (if supported by derived)
        // --------------------------------------------------------------------
        auto begin() const -> decltype(std::declval<const derived_type&>().begin());
        auto begin() -> decltype(std::declval<derived_type&>().begin());
        auto end() const -> decltype(std::declval<const derived_type&>().end());
        auto end() -> decltype(std::declval<derived_type&>().end());
        auto cbegin() const -> decltype(std::declval<const derived_type&>().cbegin());
        auto cend() const -> decltype(std::declval<const derived_type&>().cend());

        // --------------------------------------------------------------------
        // Storage order (if supported)
        // --------------------------------------------------------------------
        layout_type layout() const noexcept;

    protected:
        xexpression() = default;
        ~xexpression() = default;
    };

    // ========================================================================
    // xcontainer_expression – Base for owning containers
    // ========================================================================
    template <class D>
    class xcontainer_expression : public xexpression<D>
    {
    public:
        using base_type = xexpression<D>;
        using derived_type = D;
        using value_type = typename D::value_type;
        using reference = typename D::reference;
        using const_reference = typename D::const_reference;
        using pointer = typename D::pointer;
        using const_pointer = typename D::const_pointer;
        using size_type = xt::size_type;

        // --------------------------------------------------------------------
        // Data access
        // --------------------------------------------------------------------
        pointer data() noexcept;
        const_pointer data() const noexcept;

        // --------------------------------------------------------------------
        // Reshape / resize (forwarded)
        // --------------------------------------------------------------------
        void resize(const shape_type& shape);
        void resize(const shape_type& shape, layout_type l);
        void reshape(const shape_type& shape);
        void reshape(const shape_type& shape, layout_type l);

        // --------------------------------------------------------------------
        // Assignment from expressions
        // --------------------------------------------------------------------
        template <class E> derived_type& operator=(const xexpression<E>& e);
        template <class E> void assign(const xexpression<E>& e);

        // --------------------------------------------------------------------
        // Compound assignment operators
        // --------------------------------------------------------------------
        template <class E> derived_type& operator+=(const xexpression<E>& e);
        template <class E> derived_type& operator-=(const xexpression<E>& e);
        template <class E> derived_type& operator*=(const xexpression<E>& e);
        template <class E> derived_type& operator/=(const xexpression<E>& e);

        // --------------------------------------------------------------------
        // Fill with scalar
        // --------------------------------------------------------------------
        void fill(const value_type& value);

    protected:
        xcontainer_expression() = default;
        ~xcontainer_expression() = default;
    };

    // ========================================================================
    // xview_expression – Base for non‑owning views
    // ========================================================================
    template <class D>
    class xview_expression : public xexpression<D>
    {
    public:
        using base_type = xexpression<D>;
        using derived_type = D;

        // --------------------------------------------------------------------
        // Assignment from expressions (if view is writable)
        // --------------------------------------------------------------------
        template <class E> derived_type& operator=(const xexpression<E>& e);
        template <class E> void assign(const xexpression<E>& e);

        // --------------------------------------------------------------------
        // Compound assignment (if writable)
        // --------------------------------------------------------------------
        template <class E> derived_type& operator+=(const xexpression<E>& e);
        template <class E> derived_type& operator-=(const xexpression<E>& e);
        template <class E> derived_type& operator*=(const xexpression<E>& e);
        template <class E> derived_type& operator/=(const xexpression<E>& e);

    protected:
        xview_expression() = default;
        ~xview_expression() = default;
    };

    // ========================================================================
    // Type traits for expressions
    // ========================================================================
    template <class E> struct is_xexpression : std::is_base_of<xexpression<typename E::derived_type>, E> {};
    template <class E> inline constexpr bool is_xexpression_v = is_xexpression<E>::value;

    template <class E> struct is_xcontainer_expression : std::is_base_of<xcontainer_expression<typename E::derived_type>, E> {};
    template <class E> inline constexpr bool is_xcontainer_expression_v = is_xcontainer_expression<E>::value;

    template <class E> struct is_xview_expression : std::is_base_of<xview_expression<typename E::derived_type>, E> {};
    template <class E> inline constexpr bool is_xview_expression_v = is_xview_expression<E>::value;

    // ------------------------------------------------------------------------
    // Disable xexpression for non‑expression types (SFINAE helper)
    // ------------------------------------------------------------------------
    template <class E, class R = void>
    using disable_xexpression = std::enable_if_t<!is_xexpression<E>::value, R>;
    template <class... E>
    using enable_xexpression = std::enable_if_t<(is_xexpression<E>::value && ...)>;

    // ========================================================================
    // BigNumber expression traits (FFT integration)
    // ========================================================================
    template <class E>
    struct is_bignumber_expression : std::conjunction<is_xexpression<E>, std::is_same<typename E::value_type, bignumber::BigNumber>> {};
    template <class E> inline constexpr bool is_bignumber_expression_v = is_bignumber_expression<E>::value;

    template <class E1, class E2>
    struct use_fft_multiplication : std::conjunction<is_bignumber_expression<E1>, is_bignumber_expression<E2>, std::bool_constant<config::use_fft_multiply>> {};
    template <class E1, class E2> inline constexpr bool use_fft_multiplication_v = use_fft_multiplication<E1, E2>::value;

    // ========================================================================
    // Arithmetic functors (used by xfunction)
    // ========================================================================
    namespace detail
    {
        struct plus { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a + b) { return a + b; } };
        struct minus { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a - b) { return a - b; } };
        struct multiplies { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a * b) { return a * b; } };
        struct divides { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a / b) { return a / b; } };
        struct modulus { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a % b) { return a % b; } };
        struct bignumber_fft_multiply { template <class T> std::enable_if_t<std::is_same_v<T, bignumber::BigNumber>, T> operator()(const T& a, const T& b) const; };
        struct logical_and { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a && b) { return a && b; } };
        struct logical_or { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a || b) { return a || b; } };
        struct logical_not { template <class T> constexpr auto operator()(const T& a) const -> decltype(!a) { return !a; } };
        struct equal_to { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a == b) { return a == b; } };
        struct not_equal_to { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a != b) { return a != b; } };
        struct less { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a < b) { return a < b; } };
        struct less_equal { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a <= b) { return a <= b; } };
        struct greater { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a > b) { return a > b; } };
        struct greater_equal { template <class T> constexpr auto operator()(const T& a, const T& b) const -> decltype(a >= b) { return a >= b; } };
        struct identity { template <class T> constexpr T&& operator()(T&& t) const noexcept { return std::forward<T>(t); } };
        struct negate { template <class T> constexpr auto operator()(const T& a) const -> decltype(-a) { return -a; } };
        struct abs { template <class T> constexpr auto operator()(const T& a) const -> decltype(a < T(0) ? -a : a) { return a < T(0) ? -a : a; } };
        struct square { template <class T> constexpr auto operator()(const T& a) const -> decltype(a * a) { return a * a; } };
        struct cube { template <class T> constexpr auto operator()(const T& a) const -> decltype(a * a * a) { return a * a * a; } };
        struct sqrt { template <class T> auto operator()(const T& a) const -> decltype(std::sqrt(a)) { return std::sqrt(a); } };
        struct exp { template <class T> auto operator()(const T& a) const -> decltype(std::exp(a)) { return std::exp(a); } };
        struct log { template <class T> auto operator()(const T& a) const -> decltype(std::log(a)) { return std::log(a); } };
        struct power { template <class T> auto operator()(const T& base, const T& exponent) const -> decltype(std::pow(base, exponent)) { return std::pow(base, exponent); } };
    } // namespace detail

    // ========================================================================
    // Utility functions for expression evaluation
    // ========================================================================
    namespace detail
    {
        template <class... Shapes> shape_type broadcast_shapes(const Shapes&... shapes);
        template <class E> shape_type get_expression_shape(const xexpression<E>& e);
        template <class T, XTL_REQUIRES(!is_xexpression<T>::value)> shape_type get_expression_shape(const T&);
        bool can_broadcast(const shape_type& s1, const shape_type& s2);
    } // namespace detail

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (empty with TODO comments)
// ----------------------------------------------------------------------------
namespace xt
{
    // xexpression
    template <class D> inline auto xexpression<D>::derived_cast() const noexcept -> const derived_type& { return *static_cast<const derived_type*>(this); }
    template <class D> inline auto xexpression<D>::derived_cast() noexcept -> derived_type& { return *static_cast<derived_type*>(this); }
    template <class D> inline shape_type xexpression<D>::shape() const { return derived_cast().shape(); }
    template <class D> inline size_type xexpression<D>::size() const { return derived_cast().size(); }
    template <class D> inline size_type xexpression<D>::dimension() const { return derived_cast().dimension(); }
    template <class D> inline bool xexpression<D>::empty() const noexcept { return derived_cast().empty(); }
    template <class D> template <class... Args> inline auto xexpression<D>::operator()(Args... args) const -> decltype(std::declval<const derived_type&>()(args...)) { return derived_cast()(args...); }
    template <class D> template <class... Args> inline auto xexpression<D>::operator()(Args... args) -> decltype(std::declval<derived_type&>()(args...)) { return derived_cast()(args...); }
    template <class D> template <class S> inline auto xexpression<D>::operator[](const S& index) const -> decltype(std::declval<const derived_type&>()[index]) { return derived_cast()[index]; }
    template <class D> template <class S> inline auto xexpression<D>::operator[](const S& index) -> decltype(std::declval<derived_type&>()[index]) { return derived_cast()[index]; }
    template <class D> inline auto xexpression<D>::operator[](size_type i) const -> decltype(std::declval<const derived_type&>()[i]) { return derived_cast()[i]; }
    template <class D> inline auto xexpression<D>::operator[](size_type i) -> decltype(std::declval<derived_type&>()[i]) { return derived_cast()[i]; }
    template <class D> inline auto xexpression<D>::flat(size_type i) const -> decltype(std::declval<const derived_type&>().flat(i)) { return derived_cast().flat(i); }
    template <class D> inline auto xexpression<D>::begin() const -> decltype(std::declval<const derived_type&>().begin()) { return derived_cast().begin(); }
    template <class D> inline auto xexpression<D>::begin() -> decltype(std::declval<derived_type&>().begin()) { return derived_cast().begin(); }
    template <class D> inline auto xexpression<D>::end() const -> decltype(std::declval<const derived_type&>().end()) { return derived_cast().end(); }
    template <class D> inline auto xexpression<D>::end() -> decltype(std::declval<derived_type&>().end()) { return derived_cast().end(); }
    template <class D> inline auto xexpression<D>::cbegin() const -> decltype(std::declval<const derived_type&>().cbegin()) { return derived_cast().cbegin(); }
    template <class D> inline auto xexpression<D>::cend() const -> decltype(std::declval<const derived_type&>().cend()) { return derived_cast().cend(); }
    template <class D> inline layout_type xexpression<D>::layout() const noexcept { return derived_cast().layout(); }

    // xcontainer_expression
    template <class D> inline auto xcontainer_expression<D>::data() noexcept -> pointer { return this->derived_cast().data(); }
    template <class D> inline auto xcontainer_expression<D>::data() const noexcept -> const_pointer { return this->derived_cast().data(); }
    template <class D> inline void xcontainer_expression<D>::resize(const shape_type& shape) { this->derived_cast().resize(shape); }
    template <class D> inline void xcontainer_expression<D>::resize(const shape_type& shape, layout_type l) { this->derived_cast().resize(shape, l); }
    template <class D> inline void xcontainer_expression<D>::reshape(const shape_type& shape) { this->derived_cast().reshape(shape); }
    template <class D> inline void xcontainer_expression<D>::reshape(const shape_type& shape, layout_type l) { this->derived_cast().reshape(shape, l); }
    template <class D> template <class E> inline auto xcontainer_expression<D>::operator=(const xexpression<E>& e) -> derived_type& { this->derived_cast().assign(e); return this->derived_cast(); }
    template <class D> template <class E> inline void xcontainer_expression<D>::assign(const xexpression<E>& e) { this->derived_cast().assign(e); }
    template <class D> template <class E> inline auto xcontainer_expression<D>::operator+=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator+=(e); return this->derived_cast(); }
    template <class D> template <class E> inline auto xcontainer_expression<D>::operator-=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator-=(e); return this->derived_cast(); }
    template <class D> template <class E> inline auto xcontainer_expression<D>::operator*=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator*=(e); return this->derived_cast(); }
    template <class D> template <class E> inline auto xcontainer_expression<D>::operator/=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator/=(e); return this->derived_cast(); }
    template <class D> inline void xcontainer_expression<D>::fill(const value_type& value) { std::fill(this->derived_cast().begin(), this->derived_cast().end(), value); }

    // xview_expression
    template <class D> template <class E> inline auto xview_expression<D>::operator=(const xexpression<E>& e) -> derived_type& { this->derived_cast().assign(e); return this->derived_cast(); }
    template <class D> template <class E> inline void xview_expression<D>::assign(const xexpression<E>& e) { this->derived_cast().assign(e); }
    template <class D> template <class E> inline auto xview_expression<D>::operator+=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator+=(e); return this->derived_cast(); }
    template <class D> template <class E> inline auto xview_expression<D>::operator-=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator-=(e); return this->derived_cast(); }
    template <class D> template <class E> inline auto xview_expression<D>::operator*=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator*=(e); return this->derived_cast(); }
    template <class D> template <class E> inline auto xview_expression<D>::operator/=(const xexpression<E>& e) -> derived_type& { this->derived_cast().operator/=(e); return this->derived_cast(); }

    // detail functors
    namespace detail {
        template <class T> inline std::enable_if_t<std::is_same_v<T, bignumber::BigNumber>, T> bignumber_fft_multiply::operator()(const T& a, const T& b) const { return bignumber::fft_multiply(a, b); }
    }

} // namespace xt

#endif // XTENSOR_XEXPRESSION_HPP decltype(std::log(a))
            {
                return std::log(a);
            }

            bignumber::BigNumber operator()(const bignumber::BigNumber& a) const
            {
                return bignumber::log(a);
            }
        };

        // Power
        struct power
        {
            template <class T>
            auto operator()(const T& base, const T& exponent) const -> decltype(std::pow(base, exponent))
            {
                return std::pow(base, exponent);
            }

            bignumber::BigNumber operator()(const bignumber::BigNumber& base,
                                            const bignumber::BigNumber& exponent) const
            {
                return bignumber::pow(base, exponent);
            }
        };

    } // namespace detail

    // ========================================================================
    // Utility functions for expression evaluation
    // ========================================================================
    namespace detail
    {
        // Compute broadcasted shape for multiple expressions
        template <class... Shapes>
        shape_type broadcast_shapes(const Shapes&... shapes)
        {
            return broadcast_shapes_impl(shapes...);
        }

        // Helper to get the shape of an expression, or treat scalar as shape {}
        template <class E>
        shape_type get_expression_shape(const xexpression<E>& e)
        {
            return e.derived_cast().shape();
        }

        template <class T, XTL_REQUIRES(!is_xexpression<T>::value)>
        shape_type get_expression_shape(const T&)
        {
            return shape_type{};
        }

        // Check if two shapes can be broadcast together
        inline bool can_broadcast(const shape_type& s1, const shape_type& s2)
        {
            size_type dim1 = s1.size();
            size_type dim2 = s2.size();
            size_type max_dim = std::max(dim1, dim2);
            for (size_type i = 0; i < max_dim; ++i)
            {
                size_type idx1 = (i < dim1) ? s1[dim1 - 1 - i] : 1;
                size_type idx2 = (i < dim2) ? s2[dim2 - 1 - i] : 1;
                if (idx1 != idx2 && idx1 != 1 && idx2 != 1)
                    return false;
            }
            return true;
        }

    } // namespace detail

} // namespace xt

#endif // XTENSOR_XEXPRESSION_HPP