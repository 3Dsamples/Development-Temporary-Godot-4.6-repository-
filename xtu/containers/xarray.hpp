// containers/xarray.hpp

#ifndef XTENSOR_XARRAY_HPP
#define XTENSOR_XARRAY_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../core/xbroadcast.hpp"
#include "../core/xview.hpp"
#include "../core/xreducer.hpp"

#include <algorithm>
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
        // xarray_container - dynamically shaped dense multidimensional array
        // --------------------------------------------------------------------
        template <class T, layout_type L, class A, class Tag>
        class xarray_container : public xexpression<xarray_container<T, L, A, Tag>>,
                                 private A
        {
        public:
            using self_type = xarray_container<T, L, A, Tag>;
            using base_type = xexpression<self_type>;
            using allocator_type = A;
            using tag = Tag;
            
            using value_type = T;
            using reference = T&;
            using const_reference = const T&;
            using pointer = T*;
            using const_pointer = const T*;
            using size_type = typename A::size_type;
            using difference_type = typename A::difference_type;
            
            using shape_type = svector<size_type>;
            using strides_type = svector<size_type>;
            
            using inner_types = xcontainer_inner_types<self_type>;
            using storage_type = typename inner_types::storage_type;
            using iterator = typename storage_type::iterator;
            using const_iterator = typename storage_type::const_iterator;
            using reverse_iterator = typename storage_type::reverse_iterator;
            using const_reverse_iterator = typename storage_type::const_reverse_iterator;
            
            using temporary_type = self_type;
            using expression_tag = xcontainer_tag;
            
            static constexpr layout_type layout = L;
            static constexpr bool is_const = false;
            
            // Construction and destruction
            xarray_container() noexcept(std::is_nothrow_default_constructible<allocator_type>::value);
            
            explicit xarray_container(const allocator_type& alloc) noexcept;
            
            explicit xarray_container(const shape_type& shape, 
                                     layout_type l = L,
                                     const allocator_type& alloc = allocator_type());
            
            explicit xarray_container(const shape_type& shape, 
                                     const_reference value,
                                     layout_type l = L,
                                     const allocator_type& alloc = allocator_type());
            
            xarray_container(const shape_type& shape,
                            const strides_type& strides,
                            const allocator_type& alloc = allocator_type());
            
            xarray_container(const shape_type& shape,
                            const strides_type& strides,
                            const_reference value,
                            const allocator_type& alloc = allocator_type());
            
            explicit xarray_container(const T& t);
            
            xarray_container(std::initializer_list<T> list);
            
            xarray_container(std::initializer_list<std::initializer_list<T>> list);
            
            xarray_container(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list);
            
            template <class E>
            explicit xarray_container(const xexpression<E>& e);
            
            template <class E>
            xarray_container(const xexpression<E>& e,
                            layout_type l = L,
                            const allocator_type& alloc = allocator_type());
            
            // Copy and move constructors
            xarray_container(const self_type& rhs);
            xarray_container(const self_type& rhs, const allocator_type& alloc);
            xarray_container(self_type&& rhs) noexcept;
            xarray_container(self_type&& rhs, const allocator_type& alloc) noexcept;
            
            // Destructor
            ~xarray_container() = default;
            
            // Assignment operators
            self_type& operator=(const self_type& rhs);
            self_type& operator=(self_type&& rhs) noexcept;
            
            template <class E>
            self_type& operator=(const xexpression<E>& e);
            
            template <class E>
            disable_xexpression<E, self_type&> operator=(const E& e);
            
            // Size and shape
            size_type size() const noexcept;
            size_type dimension() const noexcept;
            const shape_type& shape() const noexcept;
            const strides_type& strides() const noexcept;
            const strides_type& backstrides() const noexcept;
            layout_type layout() const noexcept;
            
            // Element access
            reference operator()();
            const_reference operator()() const;
            
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
            template <class S>
            void reshape(const S& new_shape);
            
            template <class S>
            void reshape(const S& new_shape, layout_type l);
            
            void resize(const shape_type& new_shape);
            void resize(const shape_type& new_shape, layout_type l);
            void resize(const shape_type& new_shape, const_reference value);
            void resize(const shape_type& new_shape, layout_type l, const_reference value);
            
            void transpose();
            
            template <class P>
            void transpose(const P& permutation);
            
            void transpose(const std::vector<size_type>& permutation);
            
            template <class P>
            self_type transpose(const P& permutation) const;
            
            template <class S>
            self_type reshape_view(const S& new_shape) const;
            
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
        // xarray_adaptor - adapts external memory as an xarray
        // --------------------------------------------------------------------
        template <class EC, layout_type L, class Tag>
        class xarray_adaptor : public xexpression<xarray_adaptor<EC, L, Tag>>
        {
        public:
            using self_type = xarray_adaptor<EC, L, Tag>;
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
            
            using shape_type = svector<size_type>;
            using strides_type = svector<size_type>;
            
            using storage_type = container_type;
            using iterator = typename container_type::iterator;
            using const_iterator = typename container_type::const_iterator;
            using reverse_iterator = typename container_type::reverse_iterator;
            using const_reverse_iterator = typename container_type::const_reverse_iterator;
            
            using temporary_type = xarray_container<value_type, L, std::allocator<value_type>, Tag>;
            using expression_tag = xcontainer_tag;
            
            static constexpr layout_type layout = L;
            static constexpr bool is_const = std::is_const<container_type>::value;
            
            // Construction
            xarray_adaptor() = delete;
            
            explicit xarray_adaptor(container_type& data);
            
            xarray_adaptor(container_type& data, const shape_type& shape);
            
            xarray_adaptor(container_type& data, 
                          const shape_type& shape,
                          const strides_type& strides);
            
            xarray_adaptor(const self_type& rhs) = default;
            self_type& operator=(const self_type& rhs) = default;
            
            template <class E>
            self_type& operator=(const xexpression<E>& e);
            
            // Size and shape
            size_type size() const noexcept;
            size_type dimension() const noexcept;
            const shape_type& shape() const noexcept;
            const strides_type& strides() const noexcept;
            const strides_type& backstrides() const noexcept;
            layout_type layout() const noexcept;
            
            // Element access
            reference operator()();
            const_reference operator()() const;
            
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
        // xarray_container implementation
        // --------------------------------------------------------------------
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container() noexcept(
            std::is_nothrow_default_constructible<allocator_type>::value)
            : base_type(), A()
            , m_shape()
            , m_strides()
            , m_backstrides()
            , m_storage()
            , m_size(0)
            , m_layout(L)
        {
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const allocator_type& alloc) noexcept
            : base_type(), A(alloc)
            , m_shape()
            , m_strides()
            , m_backstrides()
            , m_storage(alloc)
            , m_size(0)
            , m_layout(L)
        {
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const shape_type& shape,
                                                                layout_type l,
                                                                const allocator_type& alloc)
            : base_type(), A(alloc)
            , m_layout(l)
        {
            init_from_shape(shape, l);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const shape_type& shape,
                                                                const_reference value,
                                                                layout_type l,
                                                                const allocator_type& alloc)
            : base_type(), A(alloc)
            , m_layout(l)
        {
            init_from_shape(shape, l);
            std::fill(m_storage.begin(), m_storage.end(), value);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const shape_type& shape,
                                                                const strides_type& strides,
                                                                const allocator_type& alloc)
            : base_type(), A(alloc)
            , m_shape(shape)
            , m_strides(strides)
            , m_layout(L)
        {
            compute_backstrides();
            m_size = m_shape.empty() ? 0 : std::accumulate(m_shape.begin(), m_shape.end(),
                                                           size_type(1), std::multiplies<size_type>());
            m_storage.resize(m_size);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const shape_type& shape,
                                                                const strides_type& strides,
                                                                const_reference value,
                                                                const allocator_type& alloc)
            : base_type(), A(alloc)
            , m_shape(shape)
            , m_strides(strides)
            , m_layout(L)
        {
            compute_backstrides();
            m_size = m_shape.empty() ? 0 : std::accumulate(m_shape.begin(), m_shape.end(),
                                                           size_type(1), std::multiplies<size_type>());
            m_storage.resize(m_size, value);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const T& t)
            : base_type(), A()
            , m_shape()
            , m_strides()
            , m_backstrides()
            , m_storage(1, t)
            , m_size(1)
            , m_layout(L)
        {
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(std::initializer_list<T> list)
            : base_type(), A()
            , m_shape({list.size()})
            , m_layout(L)
        {
            compute_strides();
            compute_backstrides();
            m_size = list.size();
            m_storage.assign(list.begin(), list.end());
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(
            std::initializer_list<std::initializer_list<T>> list)
            : base_type(), A()
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
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(
            std::initializer_list<std::initializer_list<std::initializer_list<T>>> list)
            : base_type(), A()
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
        
        template <class T, layout_type L, class A, class Tag>
        template <class E>
        inline xarray_container<T, L, A, Tag>::xarray_container(const xexpression<E>& e)
            : base_type(), A()
            , m_layout(L)
        {
            const auto& expr = e.derived_cast();
            m_shape = expr.shape();
            init_from_shape(m_shape, L);
            assign_expression(e);
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class E>
        inline xarray_container<T, L, A, Tag>::xarray_container(const xexpression<E>& e,
                                                                layout_type l,
                                                                const allocator_type& alloc)
            : base_type(), A(alloc)
            , m_layout(l)
        {
            const auto& expr = e.derived_cast();
            m_shape = expr.shape();
            init_from_shape(m_shape, l);
            assign_expression(e);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const self_type& rhs)
            : base_type(), A(rhs.get_allocator())
            , m_shape(rhs.m_shape)
            , m_strides(rhs.m_strides)
            , m_backstrides(rhs.m_backstrides)
            , m_storage(rhs.m_storage, rhs.get_allocator())
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(const self_type& rhs,
                                                                const allocator_type& alloc)
            : base_type(), A(alloc)
            , m_shape(rhs.m_shape)
            , m_strides(rhs.m_strides)
            , m_backstrides(rhs.m_backstrides)
            , m_storage(rhs.m_storage, alloc)
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(self_type&& rhs) noexcept
            : base_type(), A(std::move(rhs.get_allocator()))
            , m_shape(std::move(rhs.m_shape))
            , m_strides(std::move(rhs.m_strides))
            , m_backstrides(std::move(rhs.m_backstrides))
            , m_storage(std::move(rhs.m_storage))
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
            rhs.m_size = 0;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline xarray_container<T, L, A, Tag>::xarray_container(self_type&& rhs,
                                                                const allocator_type& alloc) noexcept
            : base_type(), A(alloc)
            , m_shape(std::move(rhs.m_shape))
            , m_strides(std::move(rhs.m_strides))
            , m_backstrides(std::move(rhs.m_backstrides))
            , m_storage(std::move(rhs.m_storage), alloc)
            , m_size(rhs.m_size)
            , m_layout(rhs.m_layout)
        {
            rhs.m_size = 0;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::operator=(const self_type& rhs) -> self_type&
        {
            if (this != &rhs)
            {
                if (XTENSOR_LIKELY(get_allocator() == rhs.get_allocator()))
                {
                    m_shape = rhs.m_shape;
                    m_strides = rhs.m_strides;
                    m_backstrides = rhs.m_backstrides;
                    m_storage = rhs.m_storage;
                    m_size = rhs.m_size;
                    m_layout = rhs.m_layout;
                }
                else
                {
                    m_shape = rhs.m_shape;
                    m_strides = rhs.m_strides;
                    m_backstrides = rhs.m_backstrides;
                    m_storage.assign(rhs.m_storage.begin(), rhs.m_storage.end());
                    m_size = rhs.m_size;
                    m_layout = rhs.m_layout;
                }
            }
            return *this;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::operator=(self_type&& rhs) noexcept -> self_type&
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
        
        template <class T, layout_type L, class A, class Tag>
        template <class E>
        inline auto xarray_container<T, L, A, Tag>::operator=(const xexpression<E>& e) -> self_type&
        {
            const auto& expr = e.derived_cast();
            
            if (expr.dimension() == dimension() && expr.shape() == m_shape)
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
                m_shape = expr.shape();
                init_from_shape(m_shape, m_layout);
                assign_expression(e);
            }
            return *this;
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class E>
        inline auto xarray_container<T, L, A, Tag>::operator=(const E& e) -> disable_xexpression<E, self_type&>
        {
            std::fill(m_storage.begin(), m_storage.end(), e);
            return *this;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::size() const noexcept -> size_type
        {
            return m_size;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::dimension() const noexcept -> size_type
        {
            return m_shape.size();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::shape() const noexcept -> const shape_type&
        {
            return m_shape;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::strides() const noexcept -> const strides_type&
        {
            return m_strides;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::backstrides() const noexcept -> const strides_type&
        {
            return m_backstrides;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline layout_type xarray_container<T, L, A, Tag>::layout() const noexcept
        {
            return m_layout;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::operator()() -> reference
        {
            XTENSOR_ASSERT(dimension() == 0);
            return m_storage[0];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::operator()() const -> const_reference
        {
            XTENSOR_ASSERT(dimension() == 0);
            return m_storage[0];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class... Args>
        inline auto xarray_container<T, L, A, Tag>::operator()(Args... args) -> reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class... Args>
        inline auto xarray_container<T, L, A, Tag>::operator()(Args... args) const -> const_reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class... Args>
        inline auto xarray_container<T, L, A, Tag>::unchecked(Args... args) -> reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class... Args>
        inline auto xarray_container<T, L, A, Tag>::unchecked(Args... args) const -> const_reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return m_storage[index];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::operator[](size_type i) -> reference
        {
            return m_storage[i];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::operator[](size_type i) const -> const_reference
        {
            return m_storage[i];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::at(size_type i) -> reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xarray::at");
            }
            return m_storage[i];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::at(size_type i) const -> const_reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xarray::at");
            }
            return m_storage[i];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline auto xarray_container<T, L, A, Tag>::element(const S& index) -> reference
        {
            return m_storage[compute_index(index)];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline auto xarray_container<T, L, A, Tag>::element(const S& index) const -> const_reference
        {
            return m_storage[compute_index(index)];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::flat(size_type i) -> reference
        {
            return m_storage[i];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::flat(size_type i) const -> const_reference
        {
            return m_storage[i];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class... Args>
        inline auto xarray_container<T, L, A, Tag>::periodic(Args... args) -> reference
        {
            std::array<size_type, sizeof...(Args)> indices = {
                static_cast<size_type>(args >= 0 ? args % static_cast<ptrdiff_t>(m_shape[sizeof...(Args) - 1 - (&args - args)]) 
                                                : (m_shape[sizeof...(Args) - 1 - (&args - args)] + (args % static_cast<ptrdiff_t>(m_shape[sizeof...(Args) - 1 - (&args - args)]))))...
            };
            size_type index = compute_index(indices);
            return m_storage[index];
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class... Args>
        inline auto xarray_container<T, L, A, Tag>::periodic(Args... args) const -> const_reference
        {
            std::array<size_type, sizeof...(Args)> indices = {
                static_cast<size_type>(args >= 0 ? args % static_cast<ptrdiff_t>(m_shape[sizeof...(Args) - 1 - (&args - args)]) 
                                                : (m_shape[sizeof...(Args) - 1 - (&args - args)] + (args % static_cast<ptrdiff_t>(m_shape[sizeof...(Args) - 1 - (&args - args)]))))...
            };
            size_type index = compute_index(indices);
            return m_storage[index];
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::data() noexcept -> pointer
        {
            return m_storage.data();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::data() const noexcept -> const_pointer
        {
            return m_storage.data();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::storage() noexcept -> storage_type&
        {
            return m_storage;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::storage() const noexcept -> const storage_type&
        {
            return m_storage;
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::begin() noexcept -> iterator
        {
            return m_storage.begin();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::end() noexcept -> iterator
        {
            return m_storage.end();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::begin() const noexcept -> const_iterator
        {
            return m_storage.begin();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::end() const noexcept -> const_iterator
        {
            return m_storage.end();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::cbegin() const noexcept -> const_iterator
        {
            return m_storage.cbegin();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::cend() const noexcept -> const_iterator
        {
            return m_storage.cend();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::rbegin() noexcept -> reverse_iterator
        {
            return m_storage.rbegin();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::rend() noexcept -> reverse_iterator
        {
            return m_storage.rend();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::rbegin() const noexcept -> const_reverse_iterator
        {
            return m_storage.rbegin();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::rend() const noexcept -> const_reverse_iterator
        {
            return m_storage.rend();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::crbegin() const noexcept -> const_reverse_iterator
        {
            return m_storage.crbegin();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::crend() const noexcept -> const_reverse_iterator
        {
            return m_storage.crend();
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline void xarray_container<T, L, A, Tag>::reshape(const S& new_shape)
        {
            reshape(new_shape, m_layout);
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline void xarray_container<T, L, A, Tag>::reshape(const S& new_shape, layout_type l)
        {
            shape_type shape(new_shape.begin(), new_shape.end());
            size_type new_size = std::accumulate(shape.begin(), shape.end(),
                                                 size_type(1), std::multiplies<size_type>());
            
            if (new_size != m_size)
            {
                XTENSOR_THROW(std::runtime_error, "Reshape operation changes total size");
            }
            
            m_shape = std::move(shape);
            m_layout = l;
            compute_strides();
            compute_backstrides();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::resize(const shape_type& new_shape)
        {
            resize(new_shape, m_layout, T());
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::resize(const shape_type& new_shape, layout_type l)
        {
            resize(new_shape, l, T());
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::resize(const shape_type& new_shape, const_reference value)
        {
            resize(new_shape, m_layout, value);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::resize(const shape_type& new_shape, layout_type l,
                                                          const_reference value)
        {
            size_type new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                                 size_type(1), std::multiplies<size_type>());
            
            if (new_shape == m_shape)
            {
                return;
            }
            
            storage_type new_storage(new_size, value, get_allocator());
            
            if (m_size > 0 && new_size > 0)
            {
                shape_type broadcast_shape = m_shape;
                // Compute overlapping region and copy data
                for (size_type d = 0; d < std::min(dimension(), new_shape.size()); ++d)
                {
                    broadcast_shape[d] = std::min(m_shape[d], new_shape[d]);
                }
                
                strides_type old_strides = m_strides;
                strides_type new_strides = compute_strides(new_shape, l);
                
                // Copy data element by element within the overlap
                for (size_type i = 0; i < std::min(m_size, new_size); ++i)
                {
                    // Compute multi-index and check bounds
                    size_type old_idx = 0;
                    size_type temp = i;
                    for (size_type d = 0; d < new_shape.size(); ++d)
                    {
                        size_type coord = temp / new_strides[d];
                        temp %= new_strides[d];
                        if (d < dimension() && coord < m_shape[d])
                        {
                            old_idx += coord * old_strides[d];
                        }
                        else
                        {
                            old_idx = std::numeric_limits<size_type>::max();
                            break;
                        }
                    }
                    if (old_idx < m_size)
                    {
                        new_storage[i] = m_storage[old_idx];
                    }
                }
            }
            
            m_shape = new_shape;
            m_layout = l;
            m_storage = std::move(new_storage);
            m_size = new_size;
            compute_strides();
            compute_backstrides();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::transpose()
        {
            if (dimension() == 2)
            {
                shape_type new_shape = {m_shape[1], m_shape[0]};
                reshape(new_shape, m_layout == layout_type::row_major ? 
                       layout_type::column_major : layout_type::row_major);
            }
            else
            {
                std::vector<size_type> perm(dimension());
                std::iota(perm.rbegin(), perm.rend(), 0);
                transpose(perm);
            }
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class P>
        inline void xarray_container<T, L, A, Tag>::transpose(const P& permutation)
        {
            shape_type new_shape(dimension());
            strides_type new_strides(dimension());
            
            for (size_type i = 0; i < dimension(); ++i)
            {
                new_shape[i] = m_shape[permutation[i]];
                new_strides[i] = m_strides[permutation[i]];
            }
            
            m_shape = std::move(new_shape);
            m_strides = std::move(new_strides);
            compute_backstrides();
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::transpose(const std::vector<size_type>& permutation)
        {
            shape_type new_shape(dimension());
            strides_type new_strides(dimension());
            
            for (size_type i = 0; i < dimension(); ++i)
            {
                new_shape[i] = m_shape[permutation[i]];
                new_strides[i] = m_strides[permutation[i]];
            }
            
            m_shape = std::move(new_shape);
            m_strides = std::move(new_strides);
            compute_backstrides();
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class P>
        inline auto xarray_container<T, L, A, Tag>::transpose(const P& permutation) const -> self_type
        {
            self_type result(*this);
            result.transpose(permutation);
            return result;
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline auto xarray_container<T, L, A, Tag>::reshape_view(const S& new_shape) const
        {
            shape_type shape(new_shape.begin(), new_shape.end());
            size_type new_size = std::accumulate(shape.begin(), shape.end(),
                                                 size_type(1), std::multiplies<size_type>());
            if (new_size != m_size)
            {
                XTENSOR_THROW(std::runtime_error, "Reshape view changes total size");
            }
            return xview<const self_type, decltype(shape)>(*this, std::move(shape));
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline auto xarray_container<T, L, A, Tag>::broadcast(const S& new_shape) const
        {
            shape_type shape(new_shape.begin(), new_shape.end());
            return xbroadcast<const self_type, decltype(shape)>(*this, std::move(shape));
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline auto xarray_container<T, L, A, Tag>::broadcast_shape(const S& new_shape) const
        {
            return broadcast(new_shape);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline bool xarray_container<T, L, A, Tag>::operator==(const self_type& rhs) const
        {
            if (m_shape != rhs.m_shape)
            {
                return false;
            }
            return std::equal(m_storage.begin(), m_storage.end(), rhs.m_storage.begin());
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline bool xarray_container<T, L, A, Tag>::operator!=(const self_type& rhs) const
        {
            return !(*this == rhs);
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class E>
        inline bool xarray_container<T, L, A, Tag>::operator==(const xexpression<E>& e) const
        {
            const auto& expr = e.derived_cast();
            if (m_shape.size() != expr.dimension())
            {
                return false;
            }
            if (!std::equal(m_shape.begin(), m_shape.end(), expr.shape().begin()))
            {
                return false;
            }
            return std::equal(begin(), end(), expr.begin());
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class E>
        inline bool xarray_container<T, L, A, Tag>::operator!=(const xexpression<E>& e) const
        {
            return !(*this == e);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::fill(const_reference value)
        {
            std::fill(m_storage.begin(), m_storage.end(), value);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::swap(self_type& rhs) noexcept
        {
            using std::swap;
            swap(m_shape, rhs.m_shape);
            swap(m_strides, rhs.m_strides);
            swap(m_backstrides, rhs.m_backstrides);
            swap(m_storage, rhs.m_storage);
            swap(m_size, rhs.m_size);
            swap(m_layout, rhs.m_layout);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline auto xarray_container<T, L, A, Tag>::get_allocator() const noexcept -> allocator_type
        {
            return static_cast<const A&>(*this);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::init_from_shape(const shape_type& shape)
        {
            init_from_shape(shape, m_layout);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::init_from_shape(const shape_type& shape, layout_type l)
        {
            m_shape = shape;
            m_layout = l;
            compute_strides();
            compute_backstrides();
            m_size = m_shape.empty() ? 0 : std::accumulate(m_shape.begin(), m_shape.end(),
                                                           size_type(1), std::multiplies<size_type>());
            m_storage.resize(m_size);
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::compute_strides()
        {
            m_strides.resize(dimension());
            if (dimension() == 0) return;
            
            if (m_layout == layout_type::row_major)
            {
                m_strides.back() = 1;
                for (size_type i = dimension() - 1; i > 0; --i)
                {
                    m_strides[i - 1] = m_strides[i] * m_shape[i];
                }
            }
            else // column_major
            {
                m_strides.front() = 1;
                for (size_type i = 0; i < dimension() - 1; ++i)
                {
                    m_strides[i + 1] = m_strides[i] * m_shape[i];
                }
            }
        }
        
        template <class T, layout_type L, class A, class Tag>
        inline void xarray_container<T, L, A, Tag>::compute_backstrides()
        {
            m_backstrides.resize(dimension());
            for (size_type i = 0; i < dimension(); ++i)
            {
                m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
            }
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class E>
        inline void xarray_container<T, L, A, Tag>::assign_expression(const xexpression<E>& e)
        {
            const auto& expr = e.derived_cast();
            if (expr.layout() == m_layout && expr.strides() == m_strides)
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
        
        template <class T, layout_type L, class A, class Tag>
        template <class S>
        inline auto xarray_container<T, L, A, Tag>::compute_index(const S& index) const -> size_type
        {
            size_type result = 0;
            for (size_type i = 0; i < dimension(); ++i)
            {
                result += static_cast<size_type>(index[i]) * m_strides[i];
            }
            return result;
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <class... Args>
        inline auto xarray_container<T, L, A, Tag>::compute_index_impl(Args... args) const -> size_type
        {
            return compute_index_impl(std::index_sequence_for<Args...>(), args...);
        }
        
        template <class T, layout_type L, class A, class Tag>
        template <std::size_t... I, class... Args>
        inline auto xarray_container<T, L, A, Tag>::compute_index_impl(std::index_sequence<I...>,
                                                                       Args... args) const -> size_type
        {
            std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...};
            size_type result = 0;
            ((result += indices[I] * m_strides[I]), ...);
            return result;
        }
        
        // --------------------------------------------------------------------
        // xarray_adaptor implementation
        // --------------------------------------------------------------------
        
        template <class EC, layout_type L, class Tag>
        inline xarray_adaptor<EC, L, Tag>::xarray_adaptor(container_type& data)
            : m_data(&data)
            , m_shape({data.size()})
            , m_strides({1})
            , m_backstrides({data.size() - 1})
            , m_size(data.size())
            , m_owner(false)
        {
        }
        
        template <class EC, layout_type L, class Tag>
        inline xarray_adaptor<EC, L, Tag>::xarray_adaptor(container_type& data, const shape_type& shape)
            : m_data(&data)
            , m_shape(shape)
            , m_owner(false)
        {
            compute_size();
            if (m_size > data.size())
            {
                XTENSOR_THROW(std::runtime_error, "Container size too small for shape");
            }
            
            m_strides.resize(dimension());
            if (L == layout_type::row_major)
            {
                m_strides.back() = 1;
                for (size_type i = dimension() - 1; i > 0; --i)
                {
                    m_strides[i - 1] = m_strides[i] * m_shape[i];
                }
            }
            else
            {
                m_strides.front() = 1;
                for (size_type i = 0; i < dimension() - 1; ++i)
                {
                    m_strides[i + 1] = m_strides[i] * m_shape[i];
                }
            }
            compute_backstrides();
        }
        
        template <class EC, layout_type L, class Tag>
        inline xarray_adaptor<EC, L, Tag>::xarray_adaptor(container_type& data,
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
        
        template <class EC, layout_type L, class Tag>
        template <class E>
        inline auto xarray_adaptor<EC, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() == dimension() && expr.shape() == m_shape)
            {
                if (expr.layout() == L && expr.strides() == m_strides)
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
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::size() const noexcept -> size_type
        {
            return m_size;
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::dimension() const noexcept -> size_type
        {
            return m_shape.size();
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::shape() const noexcept -> const shape_type&
        {
            return m_shape;
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::strides() const noexcept -> const strides_type&
        {
            return m_strides;
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::backstrides() const noexcept -> const strides_type&
        {
            return m_backstrides;
        }
        
        template <class EC, layout_type L, class Tag>
        inline layout_type xarray_adaptor<EC, L, Tag>::layout() const noexcept
        {
            return L;
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::operator()() -> reference
        {
            XTENSOR_ASSERT(dimension() == 0);
            return (*m_data)[0];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::operator()() const -> const_reference
        {
            XTENSOR_ASSERT(dimension() == 0);
            return (*m_data)[0];
        }
        
        template <class EC, layout_type L, class Tag>
        template <class... Args>
        inline auto xarray_adaptor<EC, L, Tag>::operator()(Args... args) -> reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return (*m_data)[index];
        }
        
        template <class EC, layout_type L, class Tag>
        template <class... Args>
        inline auto xarray_adaptor<EC, L, Tag>::operator()(Args... args) const -> const_reference
        {
            size_type index = compute_index_impl(std::index_sequence_for<Args...>(), 
                                                 static_cast<size_type>(args)...);
            return (*m_data)[index];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::operator[](size_type i) -> reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::operator[](size_type i) const -> const_reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::at(size_type i) -> reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xarray_adaptor::at");
            }
            return (*m_data)[i];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::at(size_type i) const -> const_reference
        {
            if (i >= m_size)
            {
                XTENSOR_THROW(std::out_of_range, "Index out of range in xarray_adaptor::at");
            }
            return (*m_data)[i];
        }
        
        template <class EC, layout_type L, class Tag>
        template <class S>
        inline auto xarray_adaptor<EC, L, Tag>::element(const S& index) -> reference
        {
            return (*m_data)[compute_index(index)];
        }
        
        template <class EC, layout_type L, class Tag>
        template <class S>
        inline auto xarray_adaptor<EC, L, Tag>::element(const S& index) const -> const_reference
        {
            return (*m_data)[compute_index(index)];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::flat(size_type i) -> reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::flat(size_type i) const -> const_reference
        {
            return (*m_data)[i];
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::data() noexcept -> pointer
        {
            return m_data->data();
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::data() const noexcept -> const_pointer
        {
            return m_data->data();
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::data_container() noexcept -> container_type&
        {
            return *m_data;
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::data_container() const noexcept -> const container_type&
        {
            return *m_data;
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::begin() noexcept -> iterator
        {
            return m_data->begin();
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::end() noexcept -> iterator
        {
            return m_data->begin() + static_cast<difference_type>(m_size);
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::begin() const noexcept -> const_iterator
        {
            return m_data->begin();
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::end() const noexcept -> const_iterator
        {
            return m_data->begin() + static_cast<difference_type>(m_size);
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::cbegin() const noexcept -> const_iterator
        {
            return m_data->cbegin();
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::cend() const noexcept -> const_iterator
        {
            return m_data->cbegin() + static_cast<difference_type>(m_size);
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::rbegin() noexcept -> reverse_iterator
        {
            return reverse_iterator(end());
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::rend() noexcept -> reverse_iterator
        {
            return reverse_iterator(begin());
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::rbegin() const noexcept -> const_reverse_iterator
        {
            return const_reverse_iterator(end());
        }
        
        template <class EC, layout_type L, class Tag>
        inline auto xarray_adaptor<EC, L, Tag>::rend() const noexcept -> const_reverse_iterator
        {
            return const_reverse_iterator(begin());
        }
        
        template <class EC, layout_type L, class Tag>
        template <class S>
        inline auto xarray_adaptor<EC, L, Tag>::broadcast(const S& new_shape) const
        {
            shape_type shape(new_shape.begin(), new_shape.end());
            return xbroadcast<const self_type, decltype(shape)>(*this, std::move(shape));
        }
        
        template <class EC, layout_type L, class Tag>
        inline void xarray_adaptor<EC, L, Tag>::reshape(const shape_type& new_shape)
        {
            size_type new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                                 size_type(1), std::multiplies<size_type>());
            if (new_size > m_data->size())
            {
                XTENSOR_THROW(std::runtime_error, "New shape requires more elements than available");
            }
            m_shape = new_shape;
            m_size = new_size;
            
            m_strides.resize(dimension());
            if (L == layout_type::row_major)
            {
                m_strides.back() = 1;
                for (size_type i = dimension() - 1; i > 0; --i)
                {
                    m_strides[i - 1] = m_strides[i] * m_shape[i];
                }
            }
            else
            {
                m_strides.front() = 1;
                for (size_type i = 0; i < dimension() - 1; ++i)
                {
                    m_strides[i + 1] = m_strides[i] * m_shape[i];
                }
            }
            compute_backstrides();
        }
        
        template <class EC, layout_type L, class Tag>
        inline void xarray_adaptor<EC, L, Tag>::reshape(const shape_type& new_shape,
                                                        const strides_type& new_strides)
        {
            m_shape = new_shape;
            m_strides = new_strides;
            compute_size();
            compute_backstrides();
        }
        
        template <class EC, layout_type L, class Tag>
        inline void xarray_adaptor<EC, L, Tag>::fill(const_reference value)
        {
            std::fill(begin(), end(), value);
        }
        
        template <class EC, layout_type L, class Tag>
        inline void xarray_adaptor<EC, L, Tag>::compute_size()
        {
            m_size = m_shape.empty() ? 0 : std::accumulate(m_shape.begin(), m_shape.end(),
                                                           size_type(1), std::multiplies<size_type>());
        }
        
        template <class EC, layout_type L, class Tag>
        inline void xarray_adaptor<EC, L, Tag>::compute_backstrides()
        {
            m_backstrides.resize(dimension());
            for (size_type i = 0; i < dimension(); ++i)
            {
                m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
            }
        }
        
        template <class EC, layout_type L, class Tag>
        template <class S>
        inline auto xarray_adaptor<EC, L, Tag>::compute_index(const S& index) const -> size_type
        {
            size_type result = 0;
            for (size_type i = 0; i < dimension(); ++i)
            {
                result += static_cast<size_type>(index[i]) * m_strides[i];
            }
            return result;
        }
        
        template <class EC, layout_type L, class Tag>
        template <class... Args>
        inline auto xarray_adaptor<EC, L, Tag>::compute_index_impl(Args... args) const -> size_type
        {
            return compute_index_impl(std::index_sequence_for<Args...>(), args...);
        }
        
        template <class EC, layout_type L, class Tag>
        template <std::size_t... I, class... Args>
        inline auto xarray_adaptor<EC, L, Tag>::compute_index_impl(std::index_sequence<I...>,
                                                                   Args... args) const -> size_type
        {
            std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...};
            size_type result = 0;
            ((result += indices[I] * m_strides[I]), ...);
            return result;
        }
        
        // --------------------------------------------------------------------
        // Helper functions
        // --------------------------------------------------------------------
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> zeros(const svector<typename A::size_type>& shape)
        {
            return xarray_container<T, L, A>(shape, T(0));
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> ones(const svector<typename A::size_type>& shape)
        {
            return xarray_container<T, L, A>(shape, T(1));
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> empty(const svector<typename A::size_type>& shape)
        {
            return xarray_container<T, L, A>(shape);
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> full(const svector<typename A::size_type>& shape, const T& value)
        {
            return xarray_container<T, L, A>(shape, value);
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> eye(typename A::size_type n)
        {
            svector<typename A::size_type> shape = {n, n};
            auto result = zeros<T, L, A>(shape);
            for (typename A::size_type i = 0; i < n; ++i)
            {
                result(i, i) = T(1);
            }
            return result;
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> linspace(T start, T stop, typename A::size_type num,
                                                  bool endpoint = true)
        {
            svector<typename A::size_type> shape = {num};
            xarray_container<T, L, A> result(shape);
            
            T step = endpoint ? (stop - start) / static_cast<T>(num - 1)
                              : (stop - start) / static_cast<T>(num);
            
            for (typename A::size_type i = 0; i < num; ++i)
            {
                result[i] = start + step * static_cast<T>(i);
            }
            
            if (endpoint && num > 0)
            {
                result[num - 1] = stop;
            }
            
            return result;
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> logspace(T start, T stop, typename A::size_type num,
                                                  T base = 10, bool endpoint = true)
        {
            auto powers = linspace<T, L, A>(start, stop, num, endpoint);
            xarray_container<T, L, A> result(powers.shape());
            std::transform(powers.begin(), powers.end(), result.begin(),
                           [base](T x) { return std::pow(base, x); });
            return result;
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> arange(T start, T stop, T step = 1)
        {
            typename A::size_type num = static_cast<typename A::size_type>(
                std::ceil((stop - start) / step));
            svector<typename A::size_type> shape = {num};
            xarray_container<T, L, A> result(shape);
            
            for (typename A::size_type i = 0; i < num; ++i)
            {
                result[i] = start + step * static_cast<T>(i);
            }
            
            return result;
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> arange(T stop)
        {
            return arange<T, L, A>(T(0), stop, T(1));
        }
        
        template <class T, layout_type L = config::default_layout, class A = default_allocator<T>>
        inline xarray_container<T, L, A> diagonal(const xarray_container<T, L, A>& arr,
                                                  ptrdiff_t offset = 0)
        {
            if (arr.dimension() != 2)
            {
                XTENSOR_THROW(std::runtime_error, "diagonal requires 2D array");
            }
            
            size_type rows = arr.shape()[0];
            size_type cols = arr.shape()[1];
            
            size_type diag_size = 0;
            if (offset >= 0)
            {
                diag_size = std::min(cols - static_cast<size_type>(offset), rows);
            }
            else
            {
                diag_size = std::min(rows - static_cast<size_type>(-offset), cols);
            }
            
            svector<size_type> shape = {diag_size};
            xarray_container<T, L, A> result(shape);
            
            for (size_type i = 0; i < diag_size; ++i)
            {
                if (offset >= 0)
                {
                    result[i] = arr(i, i + static_cast<size_type>(offset));
                }
                else
                {
                    result[i] = arr(i + static_cast<size_type>(-offset), i);
                }
            }
            
            return result;
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XARRAY_HPP