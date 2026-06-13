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

#ifndef ORTHOTREE_SERIALIZATION_BINARY_ARCHIVE_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_BINARY_ARCHIVE_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/numerical_methods.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"
#include "nvp.h"
#include "traits.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <atomic>
#include <mutex>

namespace OrthoTree {
namespace serialization {

// ============================================================================
//  Binary archive base: handles endianness conversion, versioning,
//  and low‑level read/write of fundamental types with SIMD hints.
// ============================================================================
class binary_archive_base {
public:
    using size_type = std::size_t;
    using version_type = uint32_t;

    // ------------------------------------------------------------------------
    //  Configuration for dynamic environment (endianness, version tolerance)
    // ------------------------------------------------------------------------
    struct Config {
        bool bigEndian = false;           // output big‑endian (default: native)
        bool useVersioning = true;
        version_type archiveVersion = 1;
        bool enableCompression = false;   // future extension
        bool enableSIMD = true;
        bool verifyChecksum = false;
    };

    binary_archive_base() : m_config(), m_error(false) {}
    explicit binary_archive_base(const Config& cfg) : m_config(cfg), m_error(false) {}

    virtual ~binary_archive_base() = default;

    // ------------------------------------------------------------------------
    //  Error handling
    // ------------------------------------------------------------------------
    bool is_error() const { return m_error; }
    void clear_error() { m_error = false; }

protected:
    // ------------------------------------------------------------------------
    //  Endianness helpers (compile‑time optimized)
    // ------------------------------------------------------------------------
    template<typename T>
    T maybe_swap(T value) const {
        static_assert(std::is_arithmetic_v<T>, "Only arithmetic types");
        if (!needs_swap()) return value;
        if constexpr (sizeof(T) == 2) {
            uint16_t v = static_cast<uint16_t>(value);
            v = (v >> 8) | (v << 8);
            return static_cast<T>(v);
        } else if constexpr (sizeof(T) == 4) {
            uint32_t v = static_cast<uint32_t>(value);
            v = (v >> 24) | ((v >> 8) & 0xFF00) | ((v << 8) & 0xFF0000) | (v << 24);
            return static_cast<T>(v);
        } else if constexpr (sizeof(T) == 8) {
            uint64_t v = static_cast<uint64_t>(value);
            v = (v >> 56) | ((v >> 40) & 0xFF00) | ((v >> 24) & 0xFF0000) |
                ((v >> 8) & 0xFF000000) | ((v << 8) & 0xFF00000000ULL) |
                ((v << 24) & 0xFF0000000000ULL) | ((v << 40) & 0xFF000000000000ULL) |
                (v << 56);
            return static_cast<T>(v);
        }
        return value;
    }

    bool needs_swap() const {
#if ORTHOTREE_LITTLE_ENDIAN
        return m_config.bigEndian;
#else
        return !m_config.bigEndian;
#endif
    }

    Config m_config;
    bool m_error;
};

// ============================================================================
//  Binary output archive: writes data to a stream or buffer.
//  Supports SIMD batch writes for arrays of fundamental types.
// ============================================================================
class BinaryOutputArchive : public binary_archive_base {
public:
    using buffer_type = std::vector<uint8_t>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    BinaryOutputArchive() : binary_archive_base(), m_buffer() {}
    explicit BinaryOutputArchive(const Config& cfg) : binary_archive_base(cfg), m_buffer() {}

    // ------------------------------------------------------------------------
    //  Write primitive types (arithmetic, enums)
    // ------------------------------------------------------------------------
    template<typename T>
    BinaryOutputArchive& operator&(const T& value) {
        write(value);
        return *this;
    }

    template<typename T>
    void write(const T& value) {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>,
                      "BinaryOutputArchive only supports arithmetic and enum types directly");
        if (m_error) return;
        T v = maybe_swap(value);
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&v);
        m_buffer.insert(m_buffer.end(), ptr, ptr + sizeof(T));
    }

    // Write a string (length + data)
    void write(const std::string& str) {
        uint32_t len = static_cast<uint32_t>(str.size());
        write(len);
        write_bytes(reinterpret_cast<const uint8_t*>(str.data()), len);
    }

    // Write raw bytes (for custom serialization)
    void write_bytes(const uint8_t* data, size_type count) {
        if (m_error) return;
        m_buffer.insert(m_buffer.end(), data, data + count);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch write for arrays of arithmetic types (e.g., float, double)
    // ------------------------------------------------------------------------
    template<typename T>
    void write_batch(const T* data, size_type count) {
        if (count == 0 || m_error) return;
        if (m_config.enableSIMD && count >= 4 && sizeof(T) >= 4) {
            // Pre‑allocate space
            size_type oldSize = m_buffer.size();
            m_buffer.resize(oldSize + count * sizeof(T));
            uint8_t* dst = m_buffer.data() + oldSize;
            // If no endian swap needed, copy directly (SIMD‑friendly)
            if (!needs_swap()) {
                std::memcpy(dst, data, count * sizeof(T));
            } else {
                for (size_type i = 0; i < count; ++i) {
                    T swapped = maybe_swap(data[i]);
                    std::memcpy(dst + i * sizeof(T), &swapped, sizeof(T));
                }
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                write(data[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Named value wrapper (ignores name, writes value)
    // ------------------------------------------------------------------------
    template<typename T>
    BinaryOutputArchive& operator&(const nvp<T>& nv) {
        write(nv.value());
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Save to file / get buffer
    // ------------------------------------------------------------------------
    bool saveToFile(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) return false;
        ofs.write(reinterpret_cast<const char*>(m_buffer.data()), m_buffer.size());
        return !!ofs;
    }

    const buffer_type& buffer() const { return m_buffer; }
    buffer_type release_buffer() { return std::move(m_buffer); }

    void clear() { m_buffer.clear(); m_error = false; }

private:
    buffer_type m_buffer;
};

// ============================================================================
//  Binary input archive: reads data from a stream or buffer.
//  Supports SIMD batch reads for arrays of fundamental types.
// ============================================================================
class BinaryInputArchive : public binary_archive_base {
public:
    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    BinaryInputArchive() : binary_archive_base(), m_buffer(), m_pos(0) {}
    explicit BinaryInputArchive(const Config& cfg) : binary_archive_base(cfg), m_buffer(), m_pos(0) {}

    // Load from file
    bool loadFromFile(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
        if (!ifs) return false;
        std::streamsize size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        m_buffer.resize(static_cast<size_type>(size));
        ifs.read(reinterpret_cast<char*>(m_buffer.data()), size);
        m_pos = 0;
        m_error = false;
        return !!ifs;
    }

    // Load from memory buffer
    void loadFromMemory(const uint8_t* data, size_type size) {
        m_buffer.assign(data, data + size);
        m_pos = 0;
        m_error = false;
    }

    // ------------------------------------------------------------------------
    //  Read primitive types
    // ------------------------------------------------------------------------
    template<typename T>
    BinaryInputArchive& operator&(T& value) {
        read(value);
        return *this;
    }

    template<typename T>
    void read(T& value) {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>,
                      "BinaryInputArchive only supports arithmetic and enum types directly");
        if (m_error) return;
        if (m_pos + sizeof(T) > m_buffer.size()) {
            m_error = true;
            return;
        }
        T v;
        std::memcpy(&v, m_buffer.data() + m_pos, sizeof(T));
        m_pos += sizeof(T);
        value = maybe_swap(v);
    }

    void read(std::string& str) {
        uint32_t len = 0;
        read(len);
        str.resize(len);
        read_bytes(reinterpret_cast<uint8_t*>(&str[0]), len);
    }

    void read_bytes(uint8_t* data, size_type count) {
        if (m_error) return;
        if (m_pos + count > m_buffer.size()) {
            m_error = true;
            return;
        }
        std::memcpy(data, m_buffer.data() + m_pos, count);
        m_pos += count;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch read for arrays
    // ------------------------------------------------------------------------
    template<typename T>
    void read_batch(T* out, size_type count) {
        if (count == 0 || m_error) return;
        if (m_config.enableSIMD && count >= 4 && sizeof(T) >= 4 && !needs_swap()) {
            size_type bytes = count * sizeof(T);
            if (m_pos + bytes > m_buffer.size()) {
                m_error = true;
                return;
            }
            std::memcpy(out, m_buffer.data() + m_pos, bytes);
            m_pos += bytes;
        } else {
            for (size_type i = 0; i < count; ++i) {
                read(out[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Named value wrapper
    // ------------------------------------------------------------------------
    template<typename T>
    BinaryInputArchive& operator&(const nvp<T>& nv) {
        read(const_cast<T&>(nv.value()));
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Position and size
    // ------------------------------------------------------------------------
    size_type position() const { return m_pos; }
    size_type size() const { return m_buffer.size(); }
    bool eof() const { return m_pos >= m_buffer.size(); }

    // Skip bytes
    void skip(size_type bytes) {
        if (m_pos + bytes <= m_buffer.size()) m_pos += bytes;
        else m_error = true;
    }

private:
    std::vector<uint8_t> m_buffer;
    size_type m_pos;
};

// ============================================================================
//  Archive traits (for serialization framework)
// ============================================================================
template<>
struct is_binary_archive<BinaryOutputArchive> : std::true_type {};
template<>
struct is_binary_archive<BinaryInputArchive> : std::true_type {};
template<>
struct is_output_archive<BinaryOutputArchive> : std::true_type {};
template<>
struct is_input_archive<BinaryInputArchive> : std::true_type {};

// ============================================================================
//  Dynamic environment controller for binary archives
// ============================================================================
class BinaryArchiveEnvironment {
public:
    static BinaryArchiveEnvironment& instance() {
        static BinaryArchiveEnvironment env;
        return env;
    }

    void setDefaultBigEndian(bool bigEndian) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultBigEndian = bigEndian;
    }
    bool defaultBigEndian() const { return m_defaultBigEndian; }

    void setDefaultVersion(uint32_t version) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultVersion = version;
    }
    uint32_t defaultVersion() const { return m_defaultVersion; }

    void setEnableChecksum(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableChecksum = enable;
    }
    bool enableChecksum() const { return m_enableChecksum; }

private:
    BinaryArchiveEnvironment() : m_defaultBigEndian(false), m_defaultVersion(1), m_enableChecksum(false) {}
    std::mutex m_mutex;
    bool m_defaultBigEndian;
    uint32_t m_defaultVersion;
    bool m_enableChecksum;
};

// ----------------------------------------------------------------------------
//  Helper: create archive with environment settings
// ----------------------------------------------------------------------------
inline BinaryOutputArchive make_binary_output_archive() {
    BinaryOutputArchive::Config cfg;
    cfg.bigEndian = BinaryArchiveEnvironment::instance().defaultBigEndian();
    cfg.archiveVersion = BinaryArchiveEnvironment::instance().defaultVersion();
    cfg.verifyChecksum = BinaryArchiveEnvironment::instance().enableChecksum();
    return BinaryOutputArchive(cfg);
}

inline BinaryInputArchive make_binary_input_archive() {
    BinaryInputArchive::Config cfg;
    cfg.bigEndian = BinaryArchiveEnvironment::instance().defaultBigEndian();
    cfg.archiveVersion = BinaryArchiveEnvironment::instance().defaultVersion();
    cfg.verifyChecksum = BinaryArchiveEnvironment::instance().enableChecksum();
    return BinaryInputArchive(cfg);
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_BINARY_ARCHIVE_H_INCLUDED