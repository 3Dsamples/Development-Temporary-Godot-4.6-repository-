// system name : onetbb-warp
// File 0031 : core/math/tensor.h
// Description : Arbitrary‑rank tensor class with contraction, outer product, and index notation.

#ifndef __TBB_WARP_CORE_MATH_TENSOR_H
#define __TBB_WARP_CORE_MATH_TENSOR_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Tensor shape descriptor (dimensions)
// ============================================================

template<std::size_t Rank>
struct tensor_shape {
    std::array<std::size_t, Rank> dims;
    constexpr tensor_shape() noexcept : dims{} {}
    constexpr tensor_shape(const std::array<std::size_t, Rank>& d) noexcept : dims(d) {}
    constexpr std::size_t total_elements() const noexcept {
        std::size_t prod = 1;
        for (std::size_t i = 0; i < Rank; ++i) prod *= dims[i];
        return prod;
    }
    constexpr bool operator==(const tensor_shape& o) const noexcept { return dims == o.dims; }
    constexpr bool operator!=(const tensor_shape& o) const noexcept { return !(*this == o); }
};

// ============================================================
// Tensor class template
// ============================================================

template<typename T, std::size_t Rank>
class tensor {
public:
    using value_type = T;
    using shape_type = tensor_shape<Rank>;
    using size_type = std::size_t;

    static constexpr std::size_t rank = Rank;

    // ---- Constructors ----
    constexpr tensor() noexcept : m_shape(), m_data() {}
    explicit tensor(const shape_type& shape) : m_shape(shape), m_data(shape.total_elements(), T(0)) {}
    explicit tensor(const shape_type& shape, const T& init) : m_shape(shape), m_data(shape.total_elements(), init) {}
    tensor(const shape_type& shape, std::initializer_list<T> il) : m_shape(shape), m_data(il) {
        if (m_data.size() != shape.total_elements()) throw std::runtime_error("tensor initializer size mismatch");
    }
    explicit tensor(const shape_type& shape, std::vector<T>&& data) : m_shape(shape), m_data(std::move(data)) {
        if (m_data.size() != shape.total_elements()) throw std::runtime_error("tensor data size mismatch");
    }
    template<typename... Dims, std::enable_if_t<sizeof...(Dims)==Rank, int> = 0>
    explicit tensor(Dims... dims) : m_shape{{static_cast<std::size_t>(dims)...}}, m_data(m_shape.total_elements(), T(0)) {}

    // ---- Access ----
    constexpr const shape_type& shape() const noexcept { return m_shape; }
    constexpr size_type size() const noexcept { return m_data.size(); }
    T* data() noexcept { return m_data.data(); }
    const T* data() const noexcept { return m_data.data(); }

    T& operator()(const std::array<std::size_t, Rank>& indices) {
        return m_data[linear_index(indices)];
    }
    const T& operator()(const std::array<std::size_t, Rank>& indices) const {
        return m_data[linear_index(indices)];
    }

    template<typename... Idx, std::enable_if_t<sizeof...(Idx)==Rank, int> = 0>
    T& operator()(Idx... indices) {
        return m_data[linear_index({{static_cast<std::size_t>(indices)...}})];
    }
    template<typename... Idx, std::enable_if_t<sizeof...(Idx)==Rank, int> = 0>
    const T& operator()(Idx... indices) const {
        return m_data[linear_index({{static_cast<std::size_t>(indices)...}})];
    }

    // ---- Linear index ----
    constexpr size_type linear_index(const std::array<std::size_t, Rank>& indices) const noexcept {
        size_type idx = 0;
        size_type stride = 1;
        for (std::size_t i = 0; i < Rank; ++i) {
            idx += indices[i] * stride;
            stride *= m_shape.dims[i];
        }
        return idx;
    }

    // ---- Unary operations ----
    tensor operator+() const { return *this; }
    tensor operator-() const {
        tensor result(m_shape);
        for (std::size_t i = 0; i < m_data.size(); ++i) result.m_data[i] = -m_data[i];
        return result;
    }

    // ---- Compound assignment ----
    tensor& operator+=(const tensor& o) {
        if (m_shape != o.m_shape) throw std::runtime_error("tensor shape mismatch");
        for (std::size_t i = 0; i < m_data.size(); ++i) m_data[i] += o.m_data[i];
        return *this;
    }
    tensor& operator-=(const tensor& o) {
        if (m_shape != o.m_shape) throw std::runtime_error("tensor shape mismatch");
        for (std::size_t i = 0; i < m_data.size(); ++i) m_data[i] -= o.m_data[i];
        return *this;
    }
    tensor& operator*=(const T& s) {
        for (auto& v : m_data) v *= s;
        return *this;
    }
    tensor& operator/=(const T& s) {
        T inv = T(1) / s;
        for (auto& v : m_data) v *= inv;
        return *this;
    }

    // ---- Comparison ----
    bool operator==(const tensor& o) const {
        return m_shape == o.m_shape && m_data == o.m_data;
    }
    bool operator!=(const tensor& o) const { return !(*this == o); }

    // ---- Norms ----
    T frobenius_norm() const noexcept {
        T sum = T(0);
        for (const auto& v : m_data) sum += v * v;
        return std::sqrt(sum);
    }

    // ---- Iterators ----
    auto begin() noexcept { return m_data.begin(); }
    auto end() noexcept { return m_data.end(); }
    auto begin() const noexcept { return m_data.begin(); }
    auto end() const noexcept { return m_data.end(); }

private:
    shape_type m_shape;
    std::vector<T> m_data;
};

// ============================================================
// Binary arithmetic operators
// ============================================================

template<typename T, std::size_t R>
tensor<T,R> operator+(const tensor<T,R>& a, const tensor<T,R>& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("tensor shape mismatch");
    tensor<T,R> result(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i) result.data()[i] = a.data()[i] + b.data()[i];
    return result;
}

template<typename T, std::size_t R>
tensor<T,R> operator-(const tensor<T,R>& a, const tensor<T,R>& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("tensor shape mismatch");
    tensor<T,R> result(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i) result.data()[i] = a.data()[i] - b.data()[i];
    return result;
}

template<typename T, std::size_t R>
tensor<T,R> operator*(const tensor<T,R>& a, const T& s) {
    tensor<T,R> result(a.shape());
    for (std::size_t i = 0; i < a.size(); ++i) result.data()[i] = a.data()[i] * s;
    return result;
}

template<typename T, std::size_t R>
tensor<T,R> operator*(const T& s, const tensor<T,R>& a) {
    return a * s;
}

template<typename T, std::size_t R>
tensor<T,R> operator/(const tensor<T,R>& a, const T& s) {
    T inv = T(1) / s;
    return a * inv;
}

// ============================================================
// Outer product (concatenates shapes)
// ============================================================

template<typename T, std::size_t R1, std::size_t R2>
tensor<T, R1+R2> outer_product(const tensor<T,R1>& a, const tensor<T,R2>& b) {
    using result_shape = tensor_shape<R1+R2>;
    std::array<std::size_t, R1+R2> dims;
    for (std::size_t i = 0; i < R1; ++i) dims[i] = a.shape().dims[i];
    for (std::size_t i = 0; i < R2; ++i) dims[R1+i] = b.shape().dims[i];
    result_shape shape(dims);
    tensor<T, R1+R2> result(shape);
    for (std::size_t ia = 0; ia < a.size(); ++ia) {
        for (std::size_t ib = 0; ib < b.size(); ++ib) {
            // Build combined linear index (simple flatten of a then b)
            std::size_t idx = ia * b.size() + ib;
            result.data()[idx] = a.data()[ia] * b.data()[ib];
        }
    }
    return result;
}

// ============================================================
// Contraction (sum over two indices of the same tensor)
// ============================================================

template<typename T, std::size_t R>
tensor<T, R-2> contract(const tensor<T,R>& a, std::size_t axis1, std::size_t axis2) {
    static_assert(R >= 2, "Tensor must have rank >= 2 to contract");
    if (axis1 >= R || axis2 >= R || axis1 == axis2) throw std::runtime_error("invalid contraction axes");
    if (a.shape().dims[axis1] != a.shape().dims[axis2]) throw std::runtime_error("contraction dimensions mismatch");

    const auto& shape = a.shape();
    std::size_t N = shape.dims[axis1];  // shared dimension

    // Result shape: remove axis1 and axis2 (order preserved)
    std::array<std::size_t, R-2> new_dims;
    int idx = 0;
    for (std::size_t i = 0; i < R; ++i) {
        if (i != axis1 && i != axis2) new_dims[idx++] = shape.dims[i];
    }
    tensor<T, R-2> result(tensor_shape<R-2>(new_dims));

    // Brute‑force contraction (can be optimised)
    std::array<std::size_t, R-2> res_idx;
    std::array<std::size_t, R> src_idx;
    std::fill(res_idx.begin(), res_idx.end(), 0);
    std::fill(src_idx.begin(), src_idx.end(), 0);

    std::function<void(std::size_t, std::size_t)> iterate = [&](std::size_t dim, std::size_t res_dim) {
        if (dim == R) {
            // Accumulate
            T sum = T(0);
            for (std::size_t k = 0; k < N; ++k) {
                src_idx[axis1] = k;
                src_idx[axis2] = k;
                sum += a(src_idx);
            }
            result(res_idx) = sum;
            return;
        }
        if (dim == axis1 || dim == axis2) {
            iterate(dim + 1, res_dim);
        } else {
            for (std::size_t v = 0; v < shape.dims[dim]; ++v) {
                src_idx[dim] = v;
                res_idx[res_dim] = v;
                iterate(dim + 1, res_dim + 1);
            }
        }
    };
    iterate(0, 0);
    return result;
}

// ============================================================
// Inner product (contract on last axis of a and first axis of b)
// ============================================================

template<typename T, std::size_t R1, std::size_t R2>
tensor<T, R1+R2-2> inner_product(const tensor<T,R1>& a, const tensor<T,R2>& b) {
    static_assert(R1 >= 1 && R2 >= 1, "Both tensors must have rank >= 1");
    const auto& sa = a.shape();
    const auto& sb = b.shape();
    std::size_t n = sa.dims[R1-1];
    if (n != sb.dims[0]) throw std::runtime_error("inner product dimension mismatch");
    // Result shape: remove last dim of a, remove first dim of b
    std::array<std::size_t, R1+R2-2> new_dims;
    for (std::size_t i = 0; i < R1-1; ++i) new_dims[i] = sa.dims[i];
    for (std::size_t i = 0; i < R2-1; ++i) new_dims[R1-1+i] = sb.dims[i+1];
    tensor<T, R1+R2-2> result(tensor_shape<R1+R2-2>(new_dims));
    // Brute‑force (nested loops)
    std::array<std::size_t, R1+R2-2> res_idx;
    std::array<std::size_t, R1> a_idx;
    std::array<std::size_t, R2> b_idx;
    std::fill(res_idx.begin(), res_idx.end(), 0);
    std::function<void(std::size_t)> iterate_result = [&](std::size_t dim) {
        if (dim == R1+R2-2) {
            T sum = T(0);
            for (std::size_t k = 0; k < n; ++k) {
                a_idx[R1-1] = k;
                b_idx[0] = k;
                sum += a(a_idx) * b(b_idx);
            }
            result(res_idx) = sum;
            return;
        }
        std::size_t limit;
        if (dim < R1-1) {
            limit = sa.dims[dim];
            for (std::size_t v = 0; v < limit; ++v) {
                a_idx[dim] = v;
                res_idx[dim] = v;
                iterate_result(dim + 1);
            }
        } else {
            std::size_t bdim = dim - (R1-1) + 1;
            limit = sb.dims[bdim];
            for (std::size_t v = 0; v < limit; ++v) {
                b_idx[bdim] = v;
                res_idx[dim] = v;
                iterate_result(dim + 1);
            }
        }
    };
    iterate_result(0);
    return result;
}

// ============================================================
// Tensor transpose (permute axes)
// ============================================================

template<typename T, std::size_t R>
tensor<T,R> transpose(const tensor<T,R>& a, const std::array<std::size_t, R>& perm) {
    const auto& shape = a.shape();
    std::array<std::size_t, R> new_dims;
    for (std::size_t i = 0; i < R; ++i) new_dims[i] = shape.dims[perm[i]];
    tensor<T,R> result(tensor_shape<R>(new_dims));
    std::array<std::size_t, R> src_idx, dst_idx;
    std::fill(src_idx.begin(), src_idx.end(), 0);
    std::function<void(std::size_t)> iterate = [&](std::size_t dim) {
        if (dim == R) {
            for (std::size_t i = 0; i < R; ++i) dst_idx[i] = src_idx[perm[i]];
            result(dst_idx) = a(src_idx);
            return;
        }
        for (std::size_t v = 0; v < shape.dims[dim]; ++v) {
            src_idx[dim] = v;
            iterate(dim + 1);
        }
    };
    iterate(0);
    return result;
}

// ============================================================
// Type alias for 2nd‑rank tensor (matrix), 3rd‑rank, 4th‑rank
// ============================================================

template<typename T> using tensor2 = tensor<T, 2>;
template<typename T> using tensor3 = tensor<T, 3>;
template<typename T> using tensor4 = tensor<T, 4>;

// ============================================================
// Common tensor constructors for physics (stress, strain)
// ============================================================

template<typename T>
tensor2<T> identity_tensor2(std::size_t dim = 3) {
    tensor2<T> I(tensor_shape<2>{{dim, dim}});
    for (std::size_t i = 0; i < dim; ++i) I({i,i}) = T(1);
    return I;
}

template<typename T>
tensor4<T> isotropic_stiffness_tensor(const T& lambda, const T& mu) {
    // C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
    tensor4<T> C(tensor_shape<4>{{3,3,3,3}});
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k)
                for (std::size_t l = 0; l < 3; ++l) {
                    T val = T(0);
                    if (i == j && k == l) val += lambda;
                    if (i == k && j == l) val += mu;
                    if (i == l && j == k) val += mu;
                    C({i,j,k,l}) = val;
                }
    return C;
}

// ============================================================
// Double contraction of 4th and 2nd rank tensors (C : epsilon)
// ============================================================

template<typename T>
tensor2<T> double_contract(const tensor4<T>& C, const tensor2<T>& eps) {
    if (C.shape().dims[2] != eps.shape().dims[0] || C.shape().dims[3] != eps.shape().dims[1])
        throw std::runtime_error("double contraction dimension mismatch");
    std::size_t a = C.shape().dims[0], b = C.shape().dims[1];
    tensor2<T> result(tensor_shape<2>{{a,b}});
    for (std::size_t i = 0; i < a; ++i)
        for (std::size_t j = 0; j < b; ++j) {
            T sum = T(0);
            for (std::size_t k = 0; k < C.shape().dims[2]; ++k)
                for (std::size_t l = 0; l < C.shape().dims[3]; ++l)
                    sum += C({i,j,k,l}) * eps({k,l});
            result({i,j}) = sum;
        }
    return result;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_TENSOR_H