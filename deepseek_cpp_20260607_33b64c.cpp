/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_DETAIL_ZIP_VIEW_H_INCLUDED
#define ORTHOTREE_DETAIL_ZIP_VIEW_H_INCLUDED

#include <iterator>
#include <tuple>
#include <type_traits>
#include <cstddef>
#include <utility>

namespace OrthoTree {
namespace detail {

template<bool is_const, typename It1, typename It2>
class zip_iterator;

template<bool is_const, typename It1, typename It2>
class zip_proxy_reference {
public:
    using value_type = std::pair<
        typename std::iterator_traits<It1>::value_type,
        typename std::iterator_traits<It2>::value_type
    >;

    zip_proxy_reference() = default;
    zip_proxy_reference(It1 it1, It2 it2) noexcept : it1_(it1), it2_(it2) {}

    template<bool other_const>
    zip_proxy_reference(const zip_proxy_reference<other_const, It1, It2>& other) noexcept
        : it1_(other.it1_), it2_(other.it2_) {}

    zip_proxy_reference& operator=(const zip_proxy_reference& other) noexcept {
        *it1_ = *other.it1_;
        *it2_ = *other.it2_;
        return *this;
    }

    zip_proxy_reference& operator=(zip_proxy_reference&& other) noexcept {
        *it1_ = std::move(*other.it1_);
        *it2_ = std::move(*other.it2_);
        return *this;
    }

    zip_proxy_reference& operator=(const value_type& val) noexcept {
        *it1_ = val.first;
        *it2_ = val.second;
        return *this;
    }

    zip_proxy_reference& operator=(value_type&& val) noexcept {
        *it1_ = std::move(val.first);
        *it2_ = std::move(val.second);
        return *this;
    }

    operator value_type() const& noexcept { return {*it1_, *it2_}; }
    operator value_type() && noexcept { return {std::move(*it1_), std::move(*it2_)}; }

    template<bool other_const>
    void swap(zip_proxy_reference<other_const, It1, It2> other) noexcept {
        using std::swap;
        swap(*it1_, *other.it1_);
        swap(*it2_, *other.it2_);
    }

    friend void swap(zip_proxy_reference a, zip_proxy_reference b) noexcept {
        a.swap(b);
    }

private:
    It1 it1_;
    It2 it2_;

    template<bool, typename, typename>
    friend class zip_iterator;
};

template<bool is_const, typename It1, typename It2>
class zip_iterator {
public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::pair<
        typename std::iterator_traits<It1>::value_type,
        typename std::iterator_traits<It2>::value_type
    >;
    using reference = zip_proxy_reference<is_const, It1, It2>;
    using pointer = void;

    zip_iterator() = default;
    zip_iterator(It1 it1, It2 it2) noexcept : it1_(it1), it2_(it2) {}
    zip_iterator(const zip_iterator&) = default;
    zip_iterator(zip_iterator&&) = default;

    template<bool other_const>
    zip_iterator(const zip_iterator<other_const, It1, It2>& other) noexcept
        : it1_(other.it1_), it2_(other.it2_) {}

    reference operator*() const noexcept { return reference(it1_, it2_); }

    zip_iterator& operator++() noexcept { ++it1_; ++it2_; return *this; }
    zip_iterator operator++(int) noexcept { auto tmp = *this; ++(*this); return tmp; }

    zip_iterator& operator--() noexcept { --it1_; --it2_; return *this; }
    zip_iterator operator--(int) noexcept { auto tmp = *this; --(*this); return tmp; }

    zip_iterator& operator+=(difference_type n) noexcept { it1_ += n; it2_ += n; return *this; }
    zip_iterator& operator-=(difference_type n) noexcept { it1_ -= n; it2_ -= n; return *this; }

    zip_iterator operator+(difference_type n) const noexcept {
        return zip_iterator(it1_ + n, it2_ + n);
    }
    zip_iterator operator-(difference_type n) const noexcept {
        return zip_iterator(it1_ - n, it2_ - n);
    }

    difference_type operator-(const zip_iterator& other) const noexcept {
        return it1_ - other.it1_;
    }

    reference operator[](difference_type n) const noexcept {
        return *(*this + n);
    }

    bool operator==(const zip_iterator& other) const noexcept {
        return it1_ == other.it1_;
    }
    bool operator!=(const zip_iterator& other) const noexcept {
        return !(*this == other);
    }
    bool operator<(const zip_iterator& other) const noexcept {
        return it1_ < other.it1_;
    }
    bool operator>(const zip_iterator& other) const noexcept {
        return it1_ > other.it1_;
    }
    bool operator<=(const zip_iterator& other) const noexcept {
        return it1_ <= other.it1_;
    }
    bool operator>=(const zip_iterator& other) const noexcept {
        return it1_ >= other.it1_;
    }

private:
    It1 it1_;
    It2 it2_;

    template<bool, typename, typename>
    friend class zip_iterator;
};

template<typename T1, typename T2>
class zip_view {
public:
    using iterator = zip_iterator<false,
        typename T1::iterator,
        typename T2::iterator
    >;
    using const_iterator = zip_iterator<true,
        typename T1::const_iterator,
        typename T2::const_iterator
    >;
    using size_type = std::size_t;

    zip_view(T1& data1, T2& data2) noexcept : data1_(data1), data2_(data2) {}

    iterator begin() noexcept { return iterator(data1_.begin(), data2_.begin()); }
    const_iterator begin() const noexcept { return const_iterator(data1_.begin(), data2_.begin()); }
    iterator end() noexcept { return iterator(data1_.end(), data2_.end()); }
    const_iterator end() const noexcept { return const_iterator(data1_.end(), data2_.end()); }

    size_type size() const noexcept { return data1_.size(); }
    bool empty() const noexcept { return data1_.empty(); }

private:
    T1& data1_;
    T2& data2_;
};

template<typename T1, typename T2>
zip_view<T1, T2> make_zip_view(T1& data1, T2& data2) noexcept {
    return zip_view<T1, T2>(data1, data2);
}

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_ZIP_VIEW_H_INCLUDED