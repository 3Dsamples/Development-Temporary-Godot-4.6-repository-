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

#ifndef ORTHOTREE_SERIALIZATION_STL_COMMON_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_STL_COMMON_H_INCLUDED

#include "../../../core/build_config.h"
#include "../../../core/types.h"
#include "../../../core/math/numerical_methods.h"
#include "../../../detail/common.h"
#include "../../../detail/simd_utils.h"
#include "../nvp.h"
#include "../traits.h"
#include "../binary_archive.h"
#include "../msgpack_archive.h"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <utility>

namespace OrthoTree {
namespace serialization {
namespace stl {

// ============================================================================
//  Common helpers for STL container serialization.
//  Provides SFINAE detection, batch processing hints, and dynamic environment
//  controls for containers (e.g., vector, map, set).
// ============================================================================

// ----------------------------------------------------------------------------
//  Trait to detect if a container supports .data() (contiguous)
// ----------------------------------------------------------------------------
template<typename T, typename = void>
struct has_data_method : std::false_type {};

template<typename T>
struct has_data_method<T, std::void_t<decltype(std::declval<T>().data())>>
    : std::true_type {};

template<typename T>
inline constexpr bool has_data_method_v = has_data_method<T>::value;

// ----------------------------------------------------------------------------
//  Trait to detect if a container supports .size()
// ----------------------------------------------------------------------------
template<typename T, typename = void>
struct has_size_method : std::false_type {};

template<typename T>
struct has_size_method<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

template<typename T>
inline constexpr bool has_size_method_v = has_size_method<T>::value;

// ----------------------------------------------------------------------------
//  Helper: get raw pointer to underlying data if contiguous, else nullptr.
// ----------------------------------------------------------------------------
template<typename Container>
auto get_contiguous_data(Container& c) -> decltype(c.data()) {
    if constexpr (has_data_method_v<Container>) {
        return c.data();
    } else {
        return nullptr;
    }
}

// ----------------------------------------------------------------------------
//  Batch serialization helper for contiguous containers.
//  If the container is contiguous and the value type is trivially serializable,
//  we can use archive.write_batch / read_batch for efficiency.
// ----------------------------------------------------------------------------
template<typename Archive, typename Container>
bool try_batch_serialize_container(Archive& ar, Container& c, size_t size) {
    using T = typename Container::value_type;
    if constexpr (has_data_method_v<Container> && is_trivially_serializable_v<T> &&
                  (is_binary_archive_v<Archive> || is_msgpack_archive_v<Archive>)) {
        auto* data = c.data();
        if constexpr (is_output_archive_v<Archive>) {
            ar.write_batch(data, size);
        } else {
            ar.read_batch(data, size);
        }
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for STL serialization
//  Allows runtime adjustment of batch thresholds and compression.
// ----------------------------------------------------------------------------
class STLCommonEnvironment {
public:
    static STLCommonEnvironment& instance() {
        static STLCommonEnvironment env;
        return env;
    }

    // Minimum container size to use batch serialisation (default: 64)
    void setBatchThreshold(size_t threshold) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_batchThreshold = threshold;
    }
    size_t batchThreshold() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_batchThreshold;
    }

    // Enable compression for large containers (future)
    void setEnableCompression(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableCompression = enable;
    }
    bool enableCompression() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableCompression;
    }

private:
    STLCommonEnvironment() : m_batchThreshold(64), m_enableCompression(false) {}
    mutable std::mutex m_mutex;
    size_t m_batchThreshold;
    bool m_enableCompression;
};

// ----------------------------------------------------------------------------
//  Helper: serialize container size (with versioning and compression hint)
// ----------------------------------------------------------------------------
template<typename Archive>
void serialize_container_size(Archive& ar, size_t& size) {
    ar & make_nvp("size", size);
}

// ----------------------------------------------------------------------------
//  Helper: resize container to given size (handles reserve for contiguous)
// ----------------------------------------------------------------------------
template<typename Container>
void container_resize(Container& c, size_t new_size) {
    c.resize(new_size);
    if constexpr (has_data_method_v<Container>) {
        // Optionally reserve capacity to avoid reallocation during reading
        if (c.capacity() < new_size) c.reserve(new_size);
    }
}

// ----------------------------------------------------------------------------
//  SIMD batch processing for multiple container sizes (used in archives)
// ----------------------------------------------------------------------------
inline void batch_serialize_sizes(BinaryOutputArchive& ar, const size_t* sizes, size_t count) {
    ar.write_batch(sizes, count);
}

inline void batch_serialize_sizes(BinaryInputArchive& ar, size_t* sizes, size_t count) {
    ar.read_batch(sizes, count);
}

// ----------------------------------------------------------------------------
//  Helper: compute total serialized size of a container (for pre‑allocation)
//  Not used in runtime serialisation, but useful for buffer sizing.
// ----------------------------------------------------------------------------
template<typename Container>
size_t serialized_size_estimate(const Container& c) {
    using T = typename Container::value_type;
    size_t elem_size = is_trivially_serializable_v<T> ? sizeof(T) : 0;
    return sizeof(size_t) + c.size() * elem_size;
}

} // namespace stl
} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_STL_COMMON_H_INCLUDED