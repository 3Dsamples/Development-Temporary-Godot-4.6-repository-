// containers/xtensor.hpp

#ifndef XTENSOR_XTENSOR_HPP
#define XTENSOR_XTENSOR_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../core/xbroadcast.hpp"
#include "../core/xview.hpp"
#include "../core/xreducer.hpp"
#include "xarray.hpp"  // for some utilities

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // xtensor_container - fixed-rank dense multidimensional array
        // --------------------------------------------------------------------
        template <class T, std::size_t N, layout_type L, class Tag>
        class xtensor_container : public xexpression<xtensor_container<T, N, L, Tag>>
        {
        public:
            using self_type = xtensor_container<T, N, L, Tag>;
            using base_type = xexpression<self_type>;
            using tag = Tag;
            
            using value_type = T;
            using reference = T&;
            using const_reference = const T&;
            using pointer = T*;
            using const_pointer = const T*;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            
            using shape_type = std::array<size_type, N>;
            using strides_type = std::array<size_type, N>;
            
            using storage_type = std::vector<T>;
            using allocator_type = typename storage_type::allocator_type;
            
            using iterator = typename storage_type::iterator;
            using const_iterator = typename storage_type::const_iterator;
            using reverse_iterator = typename storage_type::reverse_iterator;
            using const_reverse_iterator = typename storage_type::const_reverse_iterator;
            
            using temporary_type = self_type;
            using expression_tag = xtensor_expression_tag;
            
            static constexpr layout_type layout = L;
            static constexpr std::size_t rank = N;
            static constexpr bool is_const = false;
            
            // Construction and destruction
            xtensor_container() = default;
            
            explicit xtensor_container(const allocator_type& alloc);
            
            explicit xtensor_container(const shape_type& shape,
                                      layout_type l = L,
                                      const allocator_type& alloc = allocator_type());
            
            xtensor_container(const shape_type& shape,
                             const_reference value,
                             layout_type l = L,
                             const allocator_type& alloc = allocator_type());
            
            xtensor_container(const shape_type& shape,
                             const strides_type& strides,
                             const allocator_type& alloc = allocator_type());
            
            xtensor_container(const shape_type& shape,
                             const strides_type& strides,
                             const_reference value,
                             const allocator_type& alloc = allocator_type());
            
            explicit xtensor_container(const T& t);
            
            // Nested initializer list constructors for up to 3D
            template <std::size_t D = N, typename std::enable_if_t<D == 1, int> = 0>
            xtensor_container(std::initializer_list<T> list);
            
            template <std::size_t D = N, typename std::enable_if_t<D == 2, int> = 0>
            xtensor_container(std::initializer_list<std::initializer_list<T>> list);
            
            template <std::size_t D = N, typename std::enable_if_t<D == 3, int> = 0>
            xtensor_container(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list);
            
            template <class E>
            explicit xtensor_container(const xexpression<E>& e);
            
            template <class E>
            xtensor_container(const xexpression<E>& e,
                             layout_type l = L,
                             const allocator_type& alloc = allocator_type());
            
            // Copy and move constructors
            xtensor_container(const self_type& rhs);
            xtensor_container(const self_type& rhs, const allocator_type& alloc);
            xtensor_container(self_type&& rhs) noexcept;
            xtensor_container(self_type&& rhs, const allocator_type& alloc) noexcept;
            
            // Destructor
            ~xtensor_container() = default;
            
            // Assignment operators
            self_type& operator=(const self_type& rhs);
            self_type& operator=(self_type&& rhs) noexcept;
            
            template <class E>
            self_type& operator=(const xexpression<E>& e);
            
            template <class E>
            disable_xexpression<E, self_type&> operator=(const E& e);
            
            // Size and shape
            size_type size() const noexcept;
            static constexpr std::size_t dimension() noexcept;
            const shape_type& shape() const noexcept;
            const strides_type& strides() const noexcept;
            const strides_type& backstrides() const noexcept;
            layout_type layout() const noexcept;
            
            // Element access
            template <class... Args>
            reference operator()(Args... args);
            
            template <class... Args>
            const_reference operator()(Args... args) const;
            
            template <class... Args>
            reference unchecked(Args... args);
            
            template <class... Args>
            const_reference unchecked(Args... args) const;
            
            reference operator[](size_type i);
            const_reference operator[](size_type i) const;
            
            reference at(size_type i);
            const_reference at(size_type i) const;
            
            template <class S>
            reference element(const S& index);
            
            template <class S>
            const_reference element(const S& index) const;
            
            reference flat(size_type i);
            const_reference flat(size_type i) const;
            
            // Element access with periodicity
            template <class... Args>
            reference periodic(Args... args);
            
            template <class... Args>
            const_reference periodic(Args... args) const;
            
            // Data access
            pointer data() noexcept;
            const_pointer data() const noexcept;
            storage_type& storage() noexcept;
            const storage_type& storage() const noexcept;
            
            // Iterators
            iterator begin() noexcept;
            iterator end() noexcept;
            const_iterator begin() const noexcept;
            const_iterator end() const noexcept;
            const_iterator cbegin() const noexcept;
            const_iterator cend() const noexcept;
            
            reverse_iterator rbegin() noexcept;
            reverse_iterator rend() noexcept;
            const_reverse_iterator rbegin() const noexcept;
            const_reverse_iterator rend() const noexcept;
            const_reverse_iterator crbegin() const noexcept;
            const_reverse_iterator crend() const noexcept;
            
            // Shape modification
            void reshape(const shape_type& new_shape);
            void reshape(const shape_type& new_shape, layout_type l);
            
            void transpose();
            
            template <class P>
            void transpose(const P& permutation);
            
            template <class P>
            self_type transpose(const P& permutation) const;
            
            // Broadcasting
            template <class S>
            auto broadcast(const S& new_shape) const;
            
            template <class S>
            auto broadcast_shape(const S& new_shape) const;
            
            // Comparison
            bool operator==(const self_type& rhs) const;
            bool operator!=(const self_type& rhs) const;
            
            template <class E>
            bool operator==(const xexpression<E>& e) const;
            
            template <class E>
            bool operator!=(const xexpression<E>& e) const;
            
            // Fill
            void fill(const_reference value);
            
            // Swap
            void swap(self_type& rhs) noexcept;
            
            // Allocator
            allocator_type get_allocator() const noexcept;
            
        private:
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            storage_type m_storage;
            size_type m_size;
            layout_type m_layout;
            
            void init_from_shape(const shape_type& shape);
            void init_from_shape(const shape_type& shape, layout_type l);
            void compute_strides();
            void compute_backstrides();
            
            template <class E>
            void assign_expression(const xexpression<E>& e);
            
            template <class S>
            size_type compute_index(const S& index) const;
            
            template <class... Args>
            size_type compute_index_impl(Args... args) const;
            
            template <std::size_t... I, class... Args>
            size_type compute_index_impl(std::index_sequence<I...>, Args... args) const;
        };
        
        // --------------------------------------------------------------------
        // xtensor_adaptor - adapts external memory as an xtensor
        // --------------------------------------------------------------------
        template <class EC, std::size_t N, layout_type L, class Tag>
        class xtensor_adaptor : public xexpression<xtensor_adaptor<EC, N, L, Tag>>
        {
        public:
            using self_type = xtensor_adaptor<EC, N, L, Tag>;
            using base_type = xexpression<self_type>;
            using container_type = EC;
            using tag = Tag;
            
            using value_type = typename container_type::value_type;
            using reference = typename container_type::reference;
            using const_reference = typename container_type::const_reference;
            using pointer = typename container_type::pointer;
            using const_pointer = typename container_type::const_pointer;
            using size_type = typename container_type::size_type;
            using difference_type = typename container_type::difference_type;
            
            using shape_type = std::array<size_type, N>;
            using strides_type = std::array<size_type, N>;
            
            using storage_type = container_type;
            using iterator = typename container_type::iterator;
            using const_iterator = typename container_type::const_iterator;
            using reverse_iterator = typename container_type::reverse_iterator;
            using const_reverse_iterator = typename container_type::const_reverse_iterator;
            
            using temporary_type = xtensor_container<value_type, N, L, std::allocator<value_type>, Tag>;
            using expression_tag = xtensor_expression_tag;
            
            static constexpr layout_type layout = L;
            static constexpr std::size_t rank = N;
            static constexpr bool is_const = std::is_const<container_type>::value;
            
            // Construction
            xtensor_adaptor() = delete;
            
            explicit xtensor_adaptor(container_type& data);
            
            xtensor_adaptor(container_type& data, const shape_type& shape);
            
            xtensor_adaptor(container_type& data,
                           const shape_type& shape,
                           const strides_type& strides);
            
            xtensor_adaptor(const self_type& rhs) = default;
            self_type& operator=(const self_type& rhs) = default;
            
            template <class E>
            self_type& operator=(const xexpression<E>& e);
            
            // Size and shape
            size_type size() const noexcept;
            static constexpr std::size_t dimension() noexcept;
            const shape_type& shape() const noexcept;
            const strides_type& strides() const noexcept;
            const strides_type& backstrides() const noexcept;
            layout_type layout() const noexcept;
            
            // Element access
            template <class... Args>
            reference operator()(Args... args);
            
            template <class... Args>
            const_reference operator()(Args... args) const;
            
            reference operator[](size_type i);
            const_reference operator[](size_type i) const;
            
            reference at(size_type i);
            const_reference at(size_type i) const;
            
            template <class S>
            reference element(const S& index);
            
            template <class S>
            const_reference element(const S& index) const;
            
            reference flat(size_type i);
            const_reference flat(size_type i) const;
            
            // Data access
            pointer data() noexcept;
            const_pointer data() const noexcept;
            container_type& data_container() noexcept;
            const container_type& data_container() const noexcept;
            
            // Iterators
            iterator begin() noexcept;
            iterator end() noexcept;
            const_iterator begin() const noexcept;
            const_iterator end() const noexcept;
            const_iterator cbegin() const noexcept;
            const_iterator cend() const noexcept;
            
            reverse_iterator rbegin() noexcept;
            reverse_iterator rend() noexcept;
            const_reverse_iterator rbegin() const noexcept;
            const_reverse_iterator rend() const noexcept;
            
            // Broadcasting
            template <class S>
            auto broadcast(const S& new_shape) const;
            
            // Reshape
            void reshape(const shape_type& new_shape);
            void reshape(const shape_type& new_shape, const strides_type& new_strides);
            
            // Fill
            void fill(const_reference value);
            
        private:
            container_type* m_data;
            shape_type m_shape;
            strides_type m_strides;
            strides_type m_backstrides;
            size_type m_size;
            bool m_owner;
            
            void compute_size();
            void compute_backstrides();
            
            template <class S>
            size_type compute_index(const S& index) const;
            
            template <class... Args>
            size_type compute_index_impl(Args... args) const;
        };
        
        // --------------------------------------------------------------------
        // xtensor_container implementation
        // --------------------------------------------------------------------
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const allocator_type& alloc)
            : base_type()
            , m_shape{}
            , m_strides{}
            , m_backstrides{}
            , m_storage(alloc)
            , m_size(0)
            , m_layout(L)
        {
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const shape_type& shape,
                                                                  layout_type l,
                                                                  const allocator_type& alloc)
            : base_type()
            , m_layout(l)
            , m_storage(alloc)
        {
            init_from_shape(shape, l);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const shape_type& shape,
                                                                  const_reference value,
                                                                  layout_type l,
                                                                  const allocator_type& alloc)
            : base_type()
            , m_layout(l)
            , m_storage(alloc)
        {
            init_from_shape(shape, l);
            std::fill(m_storage.begin(), m_storage.end(), value);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const shape_type& shape,
                                                                  const strides_type& strides,
                                                                  const allocator_type& alloc)
            : base_type()
            , m_shape(shape)
            , m_strides(strides)
            , m_layout(L)
            , m_storage(alloc)
        {
            compute_backstrides();
            m_size = 1;
            for (size_type s : m_shape) m_size *= s;
            m_storage.resize(m_size);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const shape_type& shape,
                                                                  const strides_type& strides,
                                                                  const_reference value,
                                                                  const allocator_type& alloc)
            : base_type()
            , m_shape(shape)
            , m_strides(strides)
            , m_layout(L)
            , m_storage(alloc)
        {
            compute_backstrides();
            m_size = 1;
            for (size_type s : m_shape) m_size *= s;
            m_storage.resize(m_size, value);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const T& t)
            : base_type()
            , m_shape{}
            , m_strides{}
            , m_backstrides{}
            , m_storage(1, t)
            , m_size(1)
            , m_layout(L)
        {
            if constexpr (N != 0)
            {
                // scalar tensor with rank>0 not allowed, but we can handle rank 0 specially?
                // Actually, xtensor with N=0 is scalar-like. We'll support it.
            }
            if constexpr (N == 0)
            {
                // do nothing
            }
            else
            {
                XTENSOR_THROW(std::runtime_error, "Cannot construct non-scalar xtensor from single value");
            }
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <std::size_t D, typename std::enable_if_t<D == 1, int>>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(std::initializer_list<T> list)
            : base_type()
            , m_layout(L)
        {
            m_shape[0] = list.size();
            compute_strides();
            compute_backstrides();
            m_size = list.size();
            m_storage.assign(list.begin(), list.end());
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <std::size_t D, typename std::enable_if_t<D == 2, int>>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(
            std::initializer_list<std::initializer_list<T>> list)
            : base_type()
            , m_layout(L)
        {
            if (list.size() == 0)
            {
                m_shape = {0, 0};
                m_size = 0;
                compute_strides();
                compute_backstrides();
                return;
            }
            
            size_type rows = list.size();
            size_type cols = list.begin()->size();
            
            for (const auto& row : list)
            {
                if (row.size() != cols)
                {
                    XTENSOR_THROW(std::runtime_error, "Inconsistent row sizes in initializer list");
                }
            }
            
            m_shape = {rows, cols};
            compute_strides();
            compute_backstrides();
            m_size = rows * cols;
            m_storage.reserve(m_size);
            
            for (const auto& row : list)
            {
                m_storage.insert(m_storage.end(), row.begin(), row.end());
            }
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <std::size_t D, typename std::enable_if_t<D == 3, int>>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(
            std::initializer_list<std::initializer_list<std::initializer_list<T>>> list)
            : base_type()
            , m_layout(L)
        {
            if (list.size() == 0)
            {
                m_shape = {0, 0, 0};
                m_size = 0;
                compute_strides();
                compute_backstrides();
                return;
            }
            
            size_type depth = list.size();
            size_type rows = list.begin()->size();
            size_type cols = list.begin()->begin()->size();
            
            for (const auto& slice : list)
            {
                if (slice.size() != rows)
                {
                    XTENSOR_THROW(std::runtime_error, "Inconsistent row sizes in 3D initializer list");
                }
                for (const auto& row : slice)
                {
                    if (row.size() != cols)
                    {
                        XTENSOR_THROW(std::runtime_error, "Inconsistent column sizes in 3D initializer list");
                    }
                }
            }
            
            m_shape = {depth, rows, cols};
            compute_strides();
            compute_backstrides();
            m_size = depth * rows * cols;
            m_storage.reserve(m_size);
            
            for (const auto& slice : list)
            {
                for (const auto& row : slice)
                {
                    m_storage.insert(m_storage.end(), row.begin(), row.end());
                }
            }
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const xexpression<E>& e)
            : base_type()
            , m_layout(L)
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() != N)
            {
                XTENSOR_THROW(std::runtime_error, "Dimension mismatch in xtensor construction");
            }
            std::copy(expr.shape().begin(), expr.shape().end(), m_shape.begin());
            init_from_shape(m_shape, L);
            assign_expression(e);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const xexpression<E>& e,
                                                                  layout_type l,
                                                                  const allocator_type& alloc)
            : base_type()
            , m_layout(l)
            , m_storage(alloc)
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() != N)
            {
                XTENSOR_THROW(std::runtime_error, "Dimension mismatch in xtensor construction");
            }
            std::copy(expr.shape().begin(), expr.shape().end(), m_shape.begin());
            init_from_shape(m_shape, l);
            assign_expression(e);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const self_type& rhs)
            : base_type()
            , m_shape(rhs.m_shape)
            , m_strides(rhs.m_strides)
            , m_backstrides(rhs.m_backstrides)
            , m_storage(rhs.m_storage)
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(const self_type& rhs,
                                                                  const allocator_type& alloc)
            : base_type()
            , m_shape(rhs.m_shape)
            , m_strides(rhs.m_strides)
            , m_backstrides(rhs.m_backstrides)
            , m_storage(rhs.m_storage, alloc)
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(self_type&& rhs) noexcept
            : base_type()
            , m_shape(std::move(rhs.m_shape))
            , m_strides(std::move(rhs.m_strides))
            , m_backstrides(std::move(rhs.m_backstrides))
            , m_storage(std::move(rhs.m_storage))
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
            rhs.m_size = 0;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline xtensor_container<T, N, L, Tag>::xtensor_container(self_type&& rhs,
                                                                  const allocator_type& alloc) noexcept
            : base_type()
            , m_shape(std::move(rhs.m_shape))
            , m_strides(std::move(rhs.m_strides))
            , m_backstrides(std::move(rhs.m_backstrides))
            , m_storage(std::move(rhs.m_storage), alloc)
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
            rhs.m_size = 0;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::operator=(const self_type& rhs) -> self_type&
        {
            if (this != &rhs)
            {
                m_shape = rhs.m_shape;
                m_strides = rhs.m_strides;
                m_backstrides = rhs.m_backstrides;
                m_storage = rhs.m_storage;
                m_size = rhs.m_size;
                m_layout = rhs.m_layout;
            }
            return *this;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::operator=(self_type&& rhs) noexcept -> self_type&
        {
            if (this != &rhs)
            {
                m_shape = std::move(rhs.m_shape);
                m_strides = std::move(rhs.m_strides);
                m_backstrides = std::move(rhs.m_backstrides);
                m_storage = std::move(rhs.m_storage);
                m_size = rhs.m_size;
                m_layout = rhs.m_layout;
                rhs.m_size = 0;
            }
            return *this;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline auto xtensor_container<T, N, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
        {
            const auto& expr = e.derived_cast();
            
            if (expr.dimension() == N)
            {
                shape_type new_shape;
                std::copy(expr.shape().begin(), expr.shape().end(), new_shape.begin());
                
                if (new_shape == m_shape)
                {
                    // Same shape, just copy values
                    if (expr.layout() == m_layout)
                    {
                        std::copy(expr.begin(), expr.end(), m_storage.begin());
                    }
                    else
                    {
                        for (size_type i = 0; i < m_size; ++i)
                        {
                            flat(i) = expr.flat(i);
                        }
                    }
                }
                else
                {
                    // Different shape, need to reshape
                    m_shape = new_shape;
                    init_from_shape(m_shape, m_layout);
                    assign_expression(e);
                }
            }
            else
            {
                XTENSOR_THROW(std::runtime_error, "Dimension mismatch in assignment");
            }
            return *this;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline auto xtensor_container<T, N, L, Tag>::operator=(const E& e) -> disable_xexpression<E, self_type&>
        {
            std::fill(m_storage.begin(), m_storage.end(), e);
            return *this;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::size() const noexcept -> size_type
        {
            return m_size;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline constexpr std::size_t xtensor_container<T, N, L, Tag>::dimension() noexcept
        {
            return N;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::shape() const noexcept -> const shape_type&
        {
            return m_shape;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::strides() const noexcept -> const strides_type&
        {
            return m_strides;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::backstrides() const noexcept -> const strides_type&
        {
            return m_backstrides;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline layout_type xtensor_container<T, N, L, Tag>::layout() const noexcept
        {
            return m_layout;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_container<T, N, L, Tag>::operator()(Args... args) -> reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_container<T, N, L, Tag>::operator()(Args... args) const -> const_reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_container<T, N, L, Tag>::unchecked(Args... args) -> reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_container<T, N, L, Tag>::unchecked(Args... args) const -> const_reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::operator[](size_type i) -> reference
        {
            return m_storage[i];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::operator[](size_type i) const -> const_reference
        {
            return m_storage[i];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::at(size_type i) -> reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xtensor::at");
            }
            return m_storage[i];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::at(size_type i) const -> const_reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xtensor::at");
            }
            return m_storage[i];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_container<T, N, L, Tag>::element(const S& index) -> reference
        {
            return m_storage[compute_index(index)];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_container<T, N, L, Tag>::element(const S& index) const -> const_reference
        {
            return m_storage[compute_index(index)];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::flat(size_type i) -> reference
        {
            return m_storage[i];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::flat(size_type i) const -> const_reference
        {
            return m_storage[i];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_container<T, N, L, Tag>::periodic(Args... args) -> reference
        {
            std::array<size_type, sizeof...(Args)> indices;
            size_t idx = 0;
            ((indices[idx++] = static_cast<size_type>(
                args >= 0 ? args % static_cast<ptrdiff_t>(m_shape[idx]) 
                          : (m_shape[idx] + (args % static_cast<ptrdiff_t>(m_shape[idx])))
            )), ...);
            return m_storage[compute_index(indices)];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_container<T, N, L, Tag>::periodic(Args... args) const -> const_reference
        {
            std::array<size_type, sizeof...(Args)> indices;
            size_t idx = 0;
            ((indices[idx++] = static_cast<size_type>(
                args >= 0 ? args % static_cast<ptrdiff_t>(m_shape[idx]) 
                          : (m_shape[idx] + (args % static_cast<ptrdiff_t>(m_shape[idx])))
            )), ...);
            return m_storage[compute_index(indices)];
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::data() noexcept -> pointer
        {
            return m_storage.data();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::data() const noexcept -> const_pointer
        {
            return m_storage.data();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::storage() noexcept -> storage_type&
        {
            return m_storage;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::storage() const noexcept -> const storage_type&
        {
            return m_storage;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::begin() noexcept -> iterator
        {
            return m_storage.begin();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::end() noexcept -> iterator
        {
            return m_storage.end();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::begin() const noexcept -> const_iterator
        {
            return m_storage.begin();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::end() const noexcept -> const_iterator
        {
            return m_storage.end();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::cbegin() const noexcept -> const_iterator
        {
            return m_storage.cbegin();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::cend() const noexcept -> const_iterator
        {
            return m_storage.cend();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::rbegin() noexcept -> reverse_iterator
        {
            return m_storage.rbegin();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::rend() noexcept -> reverse_iterator
        {
            return m_storage.rend();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::rbegin() const noexcept -> const_reverse_iterator
        {
            return m_storage.rbegin();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::rend() const noexcept -> const_reverse_iterator
        {
            return m_storage.rend();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::crbegin() const noexcept -> const_reverse_iterator
        {
            return m_storage.crbegin();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::crend() const noexcept -> const_reverse_iterator
        {
            return m_storage.crend();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::reshape(const shape_type& new_shape)
        {
            reshape(new_shape, m_layout);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::reshape(const shape_type& new_shape, layout_type l)
        {
            size_type new_size = 1;
            for (size_type s : new_shape) new_size *= s;
            
            if (new_size != m_size)
            {
                XTENSOR_THROW(std::runtime_error, "Reshape operation changes total size");
            }
            
            m_shape = new_shape;
            m_layout = l;
            compute_strides();
            compute_backstrides();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::transpose()
        {
            if constexpr (N == 2)
            {
                shape_type new_shape = {m_shape[1], m_shape[0]};
                reshape(new_shape, m_layout == layout_type::row_major ? 
                       layout_type::column_major : layout_type::row_major);
            }
            else
            {
                std::array<size_type, N> perm;
                for (std::size_t i = 0; i < N; ++i)
                    perm[i] = N - 1 - i;
                transpose(perm);
            }
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class P>
        inline void xtensor_container<T, N, L, Tag>::transpose(const P& permutation)
        {
            static_assert(std::tuple_size<P>::value == N, "Permutation size must match rank");
            shape_type new_shape;
            strides_type new_strides;
            
            for (std::size_t i = 0; i < N; ++i)
            {
                new_shape[i] = m_shape[permutation[i]];
                new_strides[i] = m_strides[permutation[i]];
            }
            
            m_shape = new_shape;
            m_strides = new_strides;
            compute_backstrides();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class P>
        inline auto xtensor_container<T, N, L, Tag>::transpose(const P& permutation) const -> self_type
        {
            self_type result(*this);
            result.transpose(permutation);
            return result;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_container<T, N, L, Tag>::broadcast(const S& new_shape) const
        {
            using broadcast_type = xbroadcast<const self_type, S>;
            return broadcast_type(*this, new_shape);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_container<T, N, L, Tag>::broadcast_shape(const S& new_shape) const
        {
            return broadcast(new_shape);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline bool xtensor_container<T, N, L, Tag>::operator==(const self_type& rhs) const
        {
            if (m_shape != rhs.m_shape)
            {
                return false;
            }
            return std::equal(m_storage.begin(), m_storage.end(), rhs.m_storage.begin());
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline bool xtensor_container<T, N, L, Tag>::operator!=(const self_type& rhs) const
        {
            return !(*this == rhs);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline bool xtensor_container<T, N, L, Tag>::operator==(const xexpression<E>& e) const
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() != N)
            {
                return false;
            }
            if (!std::equal(m_shape.begin(), m_shape.end(), expr.shape().begin()))
            {
                return false;
            }
            return std::equal(begin(), end(), expr.begin());
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline bool xtensor_container<T, N, L, Tag>::operator!=(const xexpression<E>& e) const
        {
            return !(*this == e);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::fill(const_reference value)
        {
            std::fill(m_storage.begin(), m_storage.end(), value);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::swap(self_type& rhs) noexcept
        {
            using std::swap;
            swap(m_shape, rhs.m_shape);
            swap(m_strides, rhs.m_strides);
            swap(m_backstrides, rhs.m_backstrides);
            swap(m_storage, rhs.m_storage);
            swap(m_size, rhs.m_size);
            swap(m_layout, rhs.m_layout);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_container<T, N, L, Tag>::get_allocator() const noexcept -> allocator_type
        {
            return m_storage.get_allocator();
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::init_from_shape(const shape_type& shape)
        {
            init_from_shape(shape, m_layout);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::init_from_shape(const shape_type& shape, layout_type l)
        {
            m_shape = shape;
            m_layout = l;
            compute_strides();
            compute_backstrides();
            m_size = 1;
            for (size_type s : m_shape) m_size *= s;
            m_storage.resize(m_size);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::compute_strides()
        {
            if constexpr (N == 0) return;
            
            if (m_layout == layout_type::row_major)
            {
                m_strides[N - 1] = 1;
                for (std::size_t i = N - 1; i > 0; --i)
                {
                    m_strides[i - 1] = m_strides[i] * m_shape[i];
                }
            }
            else // column_major
            {
                m_strides[0] = 1;
                for (std::size_t i = 0; i < N - 1; ++i)
                {
                    m_strides[i + 1] = m_strides[i] * m_shape[i];
                }
            }
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        inline void xtensor_container<T, N, L, Tag>::compute_backstrides()
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
            }
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline void xtensor_container<T, N, L, Tag>::assign_expression(const xexpression<E>& e)
        {
            const auto& expr = e.derived_cast();
            if (expr.layout() == m_layout)
            {
                std::copy(expr.begin(), expr.end(), m_storage.begin());
            }
            else
            {
                for (size_type i = 0; i < m_size; ++i)
                {
                    m_storage[i] = expr.flat(i);
                }
            }
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_container<T, N, L, Tag>::compute_index(const S& index) const -> size_type
        {
            size_type result = 0;
            for (std::size_t i = 0; i < N; ++i)
            {
                result += static_cast<size_type>(index[i]) * m_strides[i];
            }
            return result;
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_container<T, N, L, Tag>::compute_index_impl(Args... args) const -> size_type
        {
            return compute_index_impl(std::index_sequence_for<Args...>(), args...);
        }
        
        template <class T, std::size_t N, layout_type L, class Tag>
        template <std::size_t... I, class... Args>
        inline auto xtensor_container<T, N, L, Tag>::compute_index_impl(std::index_sequence<I...>,
                                                                        Args... args) const -> size_type
        {
            std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...};
            size_type result = 0;
            ((result += indices[I] * m_strides[I]), ...);
            return result;
        }
        
        // --------------------------------------------------------------------
        // xtensor_adaptor implementation
        // --------------------------------------------------------------------
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(container_type& data)
            : m_data(&data)
            , m_shape{}
            , m_strides{}
            , m_backstrides{}
            , m_size(data.size())
            , m_owner(false)
        {
            // For 1D adaptor, shape is {size}
            static_assert(N == 1, "Default adaptor construction only for rank 1");
            m_shape[0] = data.size();
            m_strides[0] = 1;
            m_backstrides[0] = data.size() - 1;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(container_type& data, const shape_type& shape)
            : m_data(&data)
            , m_shape(shape)
            , m_owner(false)
        {
            compute_size();
            if (m_size > data.size())
            {
                XTENSOR_THROW(std::runtime_error, "Container size too small for shape");
            }
            
            if (L == layout_type::row_major)
            {
                m_strides[N - 1] = 1;
                for (std::size_t i = N - 1; i > 0; --i)
                {
                    m_strides[i - 1] = m_strides[i] * m_shape[i];
                }
            }
            else
            {
                m_strides[0] = 1;
                for (std::size_t i = 0; i < N - 1; ++i)
                {
                    m_strides[i + 1] = m_strides[i] * m_shape[i];
                }
            }
            compute_backstrides();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(container_type& data,
                                                               const shape_type& shape,
                                                               const strides_type& strides)
            : m_data(&data)
            , m_shape(shape)
            , m_strides(strides)
            , m_owner(false)
        {
            compute_size();
            compute_backstrides();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class E>
        inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() == N && std::equal(m_shape.begin(), m_shape.end(), expr.shape().begin()))
            {
                if (expr.layout() == L)
                {
                    std::copy(expr.begin(), expr.end(), begin());
                }
                else
                {
                    for (size_type i = 0; i < m_size; ++i)
                    {
                        flat(i) = expr.flat(i);
                    }
                }
            }
            else
            {
                XTENSOR_THROW(std::runtime_error, "Cannot assign expression with different shape");
            }
            return *this;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::size() const noexcept -> size_type
        {
            return m_size;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline constexpr std::size_t xtensor_adaptor<EC, N, L, Tag>::dimension() noexcept
        {
            return N;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::shape() const noexcept -> const shape_type&
        {
            return m_shape;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::strides() const noexcept -> const strides_type&
        {
            return m_strides;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::backstrides() const noexcept -> const strides_type&
        {
            return m_backstrides;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline layout_type xtensor_adaptor<EC, N, L, Tag>::layout() const noexcept
        {
            return L;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_adaptor<EC, N, L, Tag>::operator()(Args... args) -> reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return (*m_data)[index];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_adaptor<EC, N, L, Tag>::operator()(Args... args) const -> const_reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return (*m_data)[index];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::operator[](size_type i) -> reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::operator[](size_type i) const -> const_reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::at(size_type i) -> reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xtensor_adaptor::at");
            }
            return (*m_data)[i];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::at(size_type i) const -> const_reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xtensor_adaptor::at");
            }
            return (*m_data)[i];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_adaptor<EC, N, L, Tag>::element(const S& index) -> reference
        {
            return (*m_data)[compute_index(index)];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_adaptor<EC, N, L, Tag>::element(const S& index) const -> const_reference
        {
            return (*m_data)[compute_index(index)];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::flat(size_type i) -> reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::flat(size_type i) const -> const_reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::data() noexcept -> pointer
        {
            return m_data->data();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::data() const noexcept -> const_pointer
        {
            return m_data->data();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::data_container() noexcept -> container_type&
        {
            return *m_data;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::data_container() const noexcept -> const container_type&
        {
            return *m_data;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::begin() noexcept -> iterator
        {
            return m_data->begin();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::end() noexcept -> iterator
        {
            return m_data->begin() + static_cast<difference_type>(m_size);
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::begin() const noexcept -> const_iterator
        {
            return m_data->begin();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::end() const noexcept -> const_iterator
        {
            return m_data->begin() + static_cast<difference_type>(m_size);
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::cbegin() const noexcept -> const_iterator
        {
            return m_data->cbegin();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::cend() const noexcept -> const_iterator
        {
            return m_data->cbegin() + static_cast<difference_type>(m_size);
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::rbegin() noexcept -> reverse_iterator
        {
            return reverse_iterator(end());
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::rend() noexcept -> reverse_iterator
        {
            return reverse_iterator(begin());
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::rbegin() const noexcept -> const_reverse_iterator
        {
            return const_reverse_iterator(end());
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline auto xtensor_adaptor<EC, N, L, Tag>::rend() const noexcept -> const_reverse_iterator
        {
            return const_reverse_iterator(begin());
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_adaptor<EC, N, L, Tag>::broadcast(const S& new_shape) const
        {
            return xbroadcast<const self_type, S>(*this, new_shape);
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline void xtensor_adaptor<EC, N, L, Tag>::reshape(const shape_type& new_shape)
        {
            size_type new_size = 1;
            for (size_type s : new_shape) new_size *= s;
            if (new_size > m_data->size())
            {
                XTENSOR_THROW(std::runtime_error, "New shape requires more elements than available");
            }
            m_shape = new_shape;
            m_size = new_size;
            
            if (L == layout_type::row_major)
            {
                m_strides[N - 1] = 1;
                for (std::size_t i = N - 1; i > 0; --i)
                {
                    m_strides[i - 1] = m_strides[i] * m_shape[i];
                }
            }
            else
            {
                m_strides[0] = 1;
                for (std::size_t i = 0; i < N - 1; ++i)
                {
                    m_strides[i + 1] = m_strides[i] * m_shape[i];
                }
            }
            compute_backstrides();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline void xtensor_adaptor<EC, N, L, Tag>::reshape(const shape_type& new_shape,
                                                            const strides_type& new_strides)
        {
            m_shape = new_shape;
            m_strides = new_strides;
            compute_size();
            compute_backstrides();
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline void xtensor_adaptor<EC, N, L, Tag>::fill(const_reference value)
        {
            std::fill(begin(), end(), value);
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline void xtensor_adaptor<EC, N, L, Tag>::compute_size()
        {
            m_size = 1;
            for (size_type s : m_shape) m_size *= s;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        inline void xtensor_adaptor<EC, N, L, Tag>::compute_backstrides()
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
            }
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class S>
        inline auto xtensor_adaptor<EC, N, L, Tag>::compute_index(const S& index) const -> size_type
        {
            size_type result = 0;
            for (std::size_t i = 0; i < N; ++i)
            {
                result += static_cast<size_type>(index[i]) * m_strides[i];
            }
            return result;
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <class... Args>
        inline auto xtensor_adaptor<EC, N, L, Tag>::compute_index_impl(Args... args) const -> size_type
        {
            return compute_index_impl(std::index_sequence_for<Args...>(), args...);
        }
        
        template <class EC, std::size_t N, layout_type L, class Tag>
        template <std::size_t... I, class... Args>
        inline auto xtensor_adaptor<EC, N, L, Tag>::compute_index_impl(std::index_sequence<I...>,
                                                                       Args... args) const -> size_type
        {
            std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...};
            size_type result = 0;
            ((result += indices[I] * m_strides[I]), ...);
            return result;
        }
        
        // --------------------------------------------------------------------
        // Helper functions for xtensor
        // --------------------------------------------------------------------
        
        template <class T, std::size_t N, layout_type L = config::default_layout>
        inline xtensor_container<T, N, L> zeros(const std::array<std::size_t, N>& shape)
        {
            return xtensor_container<T, N, L>(shape, T(0));
        }
        
        template <class T, std::size_t N, layout_type L = config::default_layout>
        inline xtensor_container<T, N, L> ones(const std::array<std::size_t, N>& shape)
        {
            return xtensor_container<T, N, L>(shape, T(1));
        }
        
        template <class T, std::size_t N, layout_type L = config::default_layout>
        inline xtensor_container<T, N, L> empty(const std::array<std::size_t, N>& shape)
        {
            return xtensor_container<T, N, L>(shape);
        }
        
        template <class T, std::size_t N, layout_type L = config::default_layout>
        inline xtensor_container<T, N, L> full(const std::array<std::size_t, N>& shape, const T& value)
        {
            return xtensor_container<T, N, L>(shape, value);
        }
        
        // For N=2 matrix specializations
        template <class T, layout_type L = config::default_layout>
        using matrix = xtensor_container<T, 2, L>;
        
        template <class T, layout_type L = config::default_layout>
        using vector = xtensor_container<T, 1, L>;
        
        template <class T>
        using row_vector = xtensor_container<T, 1, layout_type::row_major>;
        
        template <class T>
        using col_vector = xtensor_container<T, 1, layout_type::column_major>;
        
        template <class T, layout_type L = config::default_layout>
        inline matrix<T, L> eye(std::size_t n)
        {
            std::array<std::size_t, 2> shape = {n, n};
            auto result = zeros<T, 2, L>(shape);
            for (std::size_t i = 0; i < n; ++i)
            {
                result(i, i) = T(1);
            }
            return result;
        }
        
        template <class T, layout_type L = config::default_layout>
        inline matrix<T, L> diagonal(const matrix<T, L>& mat, std::ptrdiff_t offset = 0)
        {
            std::size_t rows = mat.shape()[0];
            std::size_t cols = mat.shape()[1];
            
            std::size_t diag_size = 0;
            if (offset >= 0)
            {
                diag_size = std::min(cols - static_cast<std::size_t>(offset), rows);
            }
            else
            {
                diag_size = std::min(rows - static_cast<std::size_t>(-offset), cols);
            }
            
            std::array<std::size_t, 2> shape = {diag_size, 1};
            matrix<T, L> result = zeros<T, 2, L>(shape);
            
            for (std::size_t i = 0; i < diag_size; ++i)
            {
                if (offset >= 0)
                {
                    result(i, 0) = mat(i, i + static_cast<std::size_t>(offset));
                }
                else
                {
                    result(i, 0) = mat(i + static_cast<std::size_t>(-offset), i);
                }
            }
            return result;
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XTENSOR_HPP