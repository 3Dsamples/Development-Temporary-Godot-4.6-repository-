//File 0102 : numdot/forward.h
//Forward declarations, expression tags, and type traits for all NumDot components, enabling decoupled headers.
#ifndef NUMDOT_FORWARD_H
#define NUMDOT_FORWARD_H

#include <cstddef>
#include <type_traits>
#include <vector>
#include <array>
#include "config.h"

namespace numdot
{
    // Expression tag for CRTP
    struct expression_tag {};

    // Base expression
    template <class D>
    class expression;

    // Array container
    template <class T, std::size_t N = dynamic_rank>
    class array;

    // Fixed‑rank array
    template <class T, std::size_t N>
    class fixed_array;

    // Adaptor for external storage
    template <class T>
    class array_adaptor;

    // Function expression node
    template <class F, class... CT>
    class function;

    // Scalar constant
    template <class T>
    class scalar;

    // Broadcast expression
    template <class E>
    class broadcast;

    // Reduction
    template <class F, class E, class X>
    class reducer;

    // Accumulator
    template <class F, class E>
    class accumulator;

    // Strided view
    template <class CT, class S>
    class strided_view;

    // Slice descriptor
    template <class T = std::ptrdiff_t>
    class slice;

    // Range
    class range;

    // All and newaxis tags
    struct all_tag {};
    struct newaxis_tag {};
    constexpr all_tag all = all_tag{};
    constexpr newaxis_tag newaxis = newaxis_tag{};

    // Iterator
    template <class E>
    class iterator;

    // Stepper
    template <class E>
    class stepper;

    // Dynamic rank
    constexpr std::size_t dynamic_rank = std::numeric_limits<std::size_t>::max();

    // Type traits
    template <class T>
    struct type_traits
    {
        using value_type = typename T::value_type;
        using reference = typename T::reference;
        using const_reference = typename T::const_reference;
        using pointer = typename T::pointer;
        using const_pointer = typename T::const_pointer;
        using size_type = typename T::size_type;
        using difference_type = typename T::difference_type;
    };

    // Default shape type
    template <class T>
    using default_shape = std::vector<T>;

    // Default strides type
    template <class T>
    using default_strides = std::vector<T>;

    // Is expression trait
    template <class T, class = void>
    struct is_expression : std::false_type {};

    template <class T>
    struct is_expression<T, std::void_t<decltype(std::declval<const T&>().derived())>>
        : std::is_base_of<expression_tag, T> {};

    template <class T>
    inline constexpr bool is_expression_v = is_expression<T>::value;

    // Disable overload for expressions
    template <class T>
    using disable_if_expression_t = std::enable_if_t<!is_expression_v<std::decay_t<T>>>;

    // Numeric constants
    template <class T>
    struct constants;

    // Math functors forward
    namespace math
    {
        struct abs_fun;
        struct sqrt_fun;
        struct exp_fun;
        struct log_fun;
        struct sin_fun;
        struct cos_fun;
        struct tan_fun;
        struct asin_fun;
        struct acos_fun;
        struct atan_fun;
        struct sinh_fun;
        struct cosh_fun;
        struct tanh_fun;
        struct ceil_fun;
        struct floor_fun;
        struct round_fun;
        struct trunc_fun;
        struct pow_fun;
        struct atan2_fun;
    }

    // Linear algebra forward
    namespace linalg
    {
        template <class E1, class E2> auto dot(const E1& a, const E2& b);
        template <class E1, class E2> auto matmul(const E1& a, const E2& b);
        template <class E> auto inv(const E& a);
        template <class E> auto det(const E& a);
        template <class E1, class E2> auto solve(const E1& a, const E2& b);
    }

    // Reduction functions
    template <class E> auto sum(const E& e);
    template <class E> auto prod(const E& e);
    template <class E> auto mean(const E& e);
    template <class E> auto max(const E& e);
    template <class E> auto min(const E& e);

    // Element‑wise operations
    template <class E1, class E2> auto operator+(const expression<E1>&, const expression<E2>&);
    template <class E1, class E2> auto operator-(const expression<E1>&, const expression<E2>&);
    template <class E1, class E2> auto operator*(const expression<E1>&, const expression<E2>&);
    template <class E1, class E2> auto operator/(const expression<E1>&, const expression<E2>&);
    template <class E> auto operator-(const expression<E>&);

    // Broadcasting
    template <class E> auto broadcast_to(const E& e, const default_shape<std::size_t>& shape);

    // Slice helpers
    slice<std::ptrdiff_t> range(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1);

    // I/O
    template <class E> void save(const std::string& filename, const expression<E>& e);
    template <class T> auto load(const std::string& filename);

    // Random
    template <class S> auto rand(const S& shape);
    template <class S> auto randn(const S& shape);

} // namespace numdot

#endif // NUMDOT_FORWARD_H