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

#ifndef ORTHOTREE_SERIALIZATION_MSGPACK_ARCHIVE_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_MSGPACK_ARCHIVE_H_INCLUDED

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
//  MsgPack format helpers (small subset for serialization)
//  Based on MessagePack specification (https://github.com/msgpack/msgpack/blob/master/spec.md)
// ============================================================================
namespace msgpack_format {
    constexpr uint8_t POSITIVE_FIXINT_START = 0x00;
    constexpr uint8_t POSITIVE_FIXINT_END   = 0x7f;
    constexpr uint8_t FIXMAP_START          = 0x80;
    constexpr uint8_t FIXMAP_END            = 0x8f;
    constexpr uint8_t FIXARRAY_START        = 0x90;
    constexpr uint8_t FIXARRAY_END          = 0x9f;
    constexpr uint8_t FIXSTR_START          = 0xa0;
    constexpr uint8_t FIXSTR_END            = 0xbf;
    constexpr uint8_t NIL                   = 0xc0;
    constexpr uint8_t FALSE                 = 0xc2;
    constexpr uint8_t TRUE                  = 0xc3;
    constexpr uint8_t BIN8                  = 0xc4;
    constexpr uint8_t BIN16                 = 0xc5;
    constexpr uint8_t BIN32                 = 0xc6;
    constexpr uint8_t EXT8                  = 0xc7;
    constexpr uint8_t EXT16                 = 0xc8;
    constexpr uint8_t EXT32                 = 0xc9;
    constexpr uint8_t FLOAT32               = 0xca;
    constexpr uint8_t FLOAT64               = 0xcb;
    constexpr uint8_t UINT8                 = 0xcc;
    constexpr uint8_t UINT16                = 0xcd;
    constexpr uint8_t UINT32                = 0xce;
    constexpr uint8_t UINT64                = 0xcf;
    constexpr uint8_t INT8                  = 0xd0;
    constexpr uint8_t INT16                 = 0xd1;
    constexpr uint8_t INT32                 = 0xd2;
    constexpr uint8_t INT64                 = 0xd3;
    constexpr uint8_t FIXEXT1               = 0xd4;
    constexpr uint8_t FIXEXT2               = 0xd5;
    constexpr uint8_t FIXEXT4               = 0xd6;
    constexpr uint8_t FIXEXT8               = 0xd7;
    constexpr uint8_t FIXEXT16              = 0xd8;
    constexpr uint8_t STR8                  = 0xd9;
    constexpr uint8_t STR16                 = 0xda;
    constexpr uint8_t STR32                 = 0xdb;
    constexpr uint8_t ARRAY16               = 0xdc;
    constexpr uint8_t ARRAY32               = 0xdd;
    constexpr uint8_t MAP16                 = 0xde;
    constexpr uint8_t MAP32                 = 0xdf;
    constexpr uint8_t NEGATIVE_FIXINT_START = 0xe0;
    constexpr uint8_t NEGATIVE_FIXINT_END   = 0xff;
} // namespace msgpack_format

// ============================================================================
//  MsgPack archive base (common functionality)
// ============================================================================
class msgpack_archive_base {
public:
    using size_type = std::size_t;
    using version_type = uint32_t;

    struct Config {
        bool useVersioning = true;
        version_type archiveVersion = 1;
        bool enableCompression = false;
        bool enableSIMD = true;
        bool verifyChecksum = false;
    };

    msgpack_archive_base() : m_config(), m_error(false) {}
    explicit msgpack_archive_base(const Config& cfg) : m_config(cfg), m_error(false) {}

    virtual ~msgpack_archive_base() = default;

    bool is_error() const { return m_error; }
    void clear_error() { m_error = false; }

protected:
    // ------------------------------------------------------------------------
    //  MsgPack primitive writers (used by output archive)
    // ------------------------------------------------------------------------
    static void write_nil(std::vector<uint8_t>& out) {
        out.push_back(msgpack_format::NIL);
    }

    static void write_bool(std::vector<uint8_t>& out, bool value) {
        out.push_back(value ? msgpack_format::TRUE : msgpack_format::FALSE);
    }

    static void write_uint8(std::vector<uint8_t>& out, uint8_t value) {
        if (value <= 0x7f) {
            out.push_back(static_cast<uint8_t>(value));
        } else {
            out.push_back(msgpack_format::UINT8);
            out.push_back(value);
        }
    }

    static void write_uint16(std::vector<uint8_t>& out, uint16_t value) {
        if (value <= 0x7f) {
            out.push_back(static_cast<uint8_t>(value));
        } else if (value <= 0xff) {
            out.push_back(msgpack_format::UINT8);
            out.push_back(static_cast<uint8_t>(value));
        } else {
            out.push_back(msgpack_format::UINT16);
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        }
    }

    static void write_uint32(std::vector<uint8_t>& out, uint32_t value) {
        if (value <= 0x7f) {
            out.push_back(static_cast<uint8_t>(value));
        } else if (value <= 0xff) {
            out.push_back(msgpack_format::UINT8);
            out.push_back(static_cast<uint8_t>(value));
        } else if (value <= 0xffff) {
            out.push_back(msgpack_format::UINT16);
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        } else {
            out.push_back(msgpack_format::UINT32);
            out.push_back(static_cast<uint8_t>(value >> 24));
            out.push_back(static_cast<uint8_t>(value >> 16));
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        }
    }

    static void write_uint64(std::vector<uint8_t>& out, uint64_t value) {
        if (value <= 0x7f) {
            out.push_back(static_cast<uint8_t>(value));
        } else if (value <= 0xff) {
            out.push_back(msgpack_format::UINT8);
            out.push_back(static_cast<uint8_t>(value));
        } else if (value <= 0xffff) {
            out.push_back(msgpack_format::UINT16);
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        } else if (value <= 0xffffffff) {
            out.push_back(msgpack_format::UINT32);
            out.push_back(static_cast<uint8_t>(value >> 24));
            out.push_back(static_cast<uint8_t>(value >> 16));
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        } else {
            out.push_back(msgpack_format::UINT64);
            out.push_back(static_cast<uint8_t>(value >> 56));
            out.push_back(static_cast<uint8_t>(value >> 48));
            out.push_back(static_cast<uint8_t>(value >> 40));
            out.push_back(static_cast<uint8_t>(value >> 32));
            out.push_back(static_cast<uint8_t>(value >> 24));
            out.push_back(static_cast<uint8_t>(value >> 16));
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        }
    }

    static void write_int8(std::vector<uint8_t>& out, int8_t value) {
        if (value >= 0) {
            write_uint8(out, static_cast<uint8_t>(value));
        } else if (value >= -32) {
            out.push_back(static_cast<uint8_t>(0xe0 | (value + 32)));
        } else {
            out.push_back(msgpack_format::INT8);
            out.push_back(static_cast<uint8_t>(value));
        }
    }

    static void write_int16(std::vector<uint8_t>& out, int16_t value) {
        if (value >= 0) {
            write_uint16(out, static_cast<uint16_t>(value));
        } else if (value >= -32) {
            out.push_back(static_cast<uint8_t>(0xe0 | (value + 32)));
        } else if (value >= -128) {
            out.push_back(msgpack_format::INT8);
            out.push_back(static_cast<uint8_t>(value));
        } else {
            out.push_back(msgpack_format::INT16);
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        }
    }

    static void write_int32(std::vector<uint8_t>& out, int32_t value) {
        if (value >= 0) {
            write_uint32(out, static_cast<uint32_t>(value));
        } else if (value >= -32) {
            out.push_back(static_cast<uint8_t>(0xe0 | (value + 32)));
        } else if (value >= -128) {
            out.push_back(msgpack_format::INT8);
            out.push_back(static_cast<uint8_t>(value));
        } else if (value >= -32768) {
            out.push_back(msgpack_format::INT16);
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        } else {
            out.push_back(msgpack_format::INT32);
            out.push_back(static_cast<uint8_t>(value >> 24));
            out.push_back(static_cast<uint8_t>(value >> 16));
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        }
    }

    static void write_int64(std::vector<uint8_t>& out, int64_t value) {
        if (value >= 0) {
            write_uint64(out, static_cast<uint64_t>(value));
        } else if (value >= -32) {
            out.push_back(static_cast<uint8_t>(0xe0 | (value + 32)));
        } else if (value >= -128) {
            out.push_back(msgpack_format::INT8);
            out.push_back(static_cast<uint8_t>(value));
        } else if (value >= -32768) {
            out.push_back(msgpack_format::INT16);
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        } else if (value >= -2147483648LL) {
            out.push_back(msgpack_format::INT32);
            out.push_back(static_cast<uint8_t>(value >> 24));
            out.push_back(static_cast<uint8_t>(value >> 16));
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        } else {
            out.push_back(msgpack_format::INT64);
            out.push_back(static_cast<uint8_t>(value >> 56));
            out.push_back(static_cast<uint8_t>(value >> 48));
            out.push_back(static_cast<uint8_t>(value >> 40));
            out.push_back(static_cast<uint8_t>(value >> 32));
            out.push_back(static_cast<uint8_t>(value >> 24));
            out.push_back(static_cast<uint8_t>(value >> 16));
            out.push_back(static_cast<uint8_t>(value >> 8));
            out.push_back(static_cast<uint8_t>(value));
        }
    }

    static void write_float32(std::vector<uint8_t>& out, float value) {
        out.push_back(msgpack_format::FLOAT32);
        uint32_t u;
        std::memcpy(&u, &value, sizeof(u));
        out.push_back(static_cast<uint8_t>(u >> 24));
        out.push_back(static_cast<uint8_t>(u >> 16));
        out.push_back(static_cast<uint8_t>(u >> 8));
        out.push_back(static_cast<uint8_t>(u));
    }

    static void write_float64(std::vector<uint8_t>& out, double value) {
        out.push_back(msgpack_format::FLOAT64);
        uint64_t u;
        std::memcpy(&u, &value, sizeof(u));
        out.push_back(static_cast<uint8_t>(u >> 56));
        out.push_back(static_cast<uint8_t>(u >> 48));
        out.push_back(static_cast<uint8_t>(u >> 40));
        out.push_back(static_cast<uint8_t>(u >> 32));
        out.push_back(static_cast<uint8_t>(u >> 24));
        out.push_back(static_cast<uint8_t>(u >> 16));
        out.push_back(static_cast<uint8_t>(u >> 8));
        out.push_back(static_cast<uint8_t>(u));
    }

    static void write_string(std::vector<uint8_t>& out, const std::string& str) {
        uint32_t len = static_cast<uint32_t>(str.size());
        if (len <= 31) {
            out.push_back(static_cast<uint8_t>(msgpack_format::FIXSTR_START | len));
        } else if (len <= 0xff) {
            out.push_back(msgpack_format::STR8);
            out.push_back(static_cast<uint8_t>(len));
        } else if (len <= 0xffff) {
            out.push_back(msgpack_format::STR16);
            out.push_back(static_cast<uint8_t>(len >> 8));
            out.push_back(static_cast<uint8_t>(len));
        } else {
            out.push_back(msgpack_format::STR32);
            out.push_back(static_cast<uint8_t>(len >> 24));
            out.push_back(static_cast<uint8_t>(len >> 16));
            out.push_back(static_cast<uint8_t>(len >> 8));
            out.push_back(static_cast<uint8_t>(len));
        }
        out.insert(out.end(), str.begin(), str.end());
    }

    static void write_binary(std::vector<uint8_t>& out, const uint8_t* data, size_type len) {
        if (len <= 0xff) {
            out.push_back(msgpack_format::BIN8);
            out.push_back(static_cast<uint8_t>(len));
        } else if (len <= 0xffff) {
            out.push_back(msgpack_format::BIN16);
            out.push_back(static_cast<uint8_t>(len >> 8));
            out.push_back(static_cast<uint8_t>(len));
        } else {
            out.push_back(msgpack_format::BIN32);
            out.push_back(static_cast<uint8_t>(len >> 24));
            out.push_back(static_cast<uint8_t>(len >> 16));
            out.push_back(static_cast<uint8_t>(len >> 8));
            out.push_back(static_cast<uint8_t>(len));
        }
        out.insert(out.end(), data, data + len);
    }

    // ------------------------------------------------------------------------
    //  MsgPack primitive readers (used by input archive)
    // ------------------------------------------------------------------------
    struct read_result {
        enum type_t { NIL, BOOL, UINT, INT, FLOAT32, FLOAT64, STR, BIN, EXT, ARRAY, MAP };
        type_t type;
        uint64_t uint_value;
        int64_t int_value;
        double float_value;
        std::string str_value;
        std::vector<uint8_t> bin_value;
    };

    static bool read_next(const uint8_t*& ptr, const uint8_t* end, read_result& out) {
        if (ptr >= end) return false;
        uint8_t b = *ptr++;
        // positive fixint
        if (b <= msgpack_format::POSITIVE_FIXINT_END) {
            out.type = read_result::UINT;
            out.uint_value = b;
            return true;
        }
        // negative fixint
        if (b >= msgpack_format::NEGATIVE_FIXINT_START) {
            out.type = read_result::INT;
            out.int_value = static_cast<int8_t>(b);
            return true;
        }
        // fixmap
        if (b >= msgpack_format::FIXMAP_START && b <= msgpack_format::FIXMAP_END) {
            out.type = read_result::MAP;
            out.uint_value = b & 0x0f;
            return true;
        }
        // fixarray
        if (b >= msgpack_format::FIXARRAY_START && b <= msgpack_format::FIXARRAY_END) {
            out.type = read_result::ARRAY;
            out.uint_value = b & 0x0f;
            return true;
        }
        // fixstr
        if (b >= msgpack_format::FIXSTR_START && b <= msgpack_format::FIXSTR_END) {
            out.type = read_result::STR;
            uint32_t len = b & 0x1f;
            if (ptr + len > end) return false;
            out.str_value.assign(reinterpret_cast<const char*>(ptr), len);
            ptr += len;
            return true;
        }
        switch (b) {
            case msgpack_format::NIL:
                out.type = read_result::NIL;
                return true;
            case msgpack_format::FALSE:
                out.type = read_result::BOOL;
                out.uint_value = 0;
                return true;
            case msgpack_format::TRUE:
                out.type = read_result::BOOL;
                out.uint_value = 1;
                return true;
            case msgpack_format::BIN8: {
                if (ptr + 1 > end) return false;
                uint32_t len = *ptr++;
                if (ptr + len > end) return false;
                out.type = read_result::BIN;
                out.bin_value.assign(ptr, ptr + len);
                ptr += len;
                return true;
            }
            case msgpack_format::BIN16: {
                if (ptr + 2 > end) return false;
                uint32_t len = (static_cast<uint32_t>(ptr[0]) << 8) | ptr[1];
                ptr += 2;
                if (ptr + len > end) return false;
                out.type = read_result::BIN;
                out.bin_value.assign(ptr, ptr + len);
                ptr += len;
                return true;
            }
            case msgpack_format::BIN32: {
                if (ptr + 4 > end) return false;
                uint32_t len = (static_cast<uint32_t>(ptr[0]) << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
                ptr += 4;
                if (ptr + len > end) return false;
                out.type = read_result::BIN;
                out.bin_value.assign(ptr, ptr + len);
                ptr += len;
                return true;
            }
            case msgpack_format::FLOAT32: {
                if (ptr + 4 > end) return false;
                uint32_t u = (static_cast<uint32_t>(ptr[0]) << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
                ptr += 4;
                float f;
                std::memcpy(&f, &u, 4);
                out.type = read_result::FLOAT32;
                out.float_value = f;
                return true;
            }
            case msgpack_format::FLOAT64: {
                if (ptr + 8 > end) return false;
                uint64_t u = (static_cast<uint64_t>(ptr[0]) << 56) | (static_cast<uint64_t>(ptr[1]) << 48) |
                             (static_cast<uint64_t>(ptr[2]) << 40) | (static_cast<uint64_t>(ptr[3]) << 32) |
                             (static_cast<uint64_t>(ptr[4]) << 24) | (static_cast<uint64_t>(ptr[5]) << 16) |
                             (static_cast<uint64_t>(ptr[6]) << 8) | ptr[7];
                ptr += 8;
                double d;
                std::memcpy(&d, &u, 8);
                out.type = read_result::FLOAT64;
                out.float_value = d;
                return true;
            }
            case msgpack_format::UINT8: {
                if (ptr + 1 > end) return false;
                out.type = read_result::UINT;
                out.uint_value = *ptr++;
                return true;
            }
            case msgpack_format::UINT16: {
                if (ptr + 2 > end) return false;
                out.uint_value = (static_cast<uint32_t>(ptr[0]) << 8) | ptr[1];
                ptr += 2;
                out.type = read_result::UINT;
                return true;
            }
            case msgpack_format::UINT32: {
                if (ptr + 4 > end) return false;
                out.uint_value = (static_cast<uint64_t>(ptr[0]) << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
                ptr += 4;
                out.type = read_result::UINT;
                return true;
            }
            case msgpack_format::UINT64: {
                if (ptr + 8 > end) return false;
                out.uint_value = (static_cast<uint64_t>(ptr[0]) << 56) | (static_cast<uint64_t>(ptr[1]) << 48) |
                                 (static_cast<uint64_t>(ptr[2]) << 40) | (static_cast<uint64_t>(ptr[3]) << 32) |
                                 (static_cast<uint64_t>(ptr[4]) << 24) | (static_cast<uint64_t>(ptr[5]) << 16) |
                                 (static_cast<uint64_t>(ptr[6]) << 8) | ptr[7];
                ptr += 8;
                out.type = read_result::UINT;
                return true;
            }
            case msgpack_format::INT8: {
                if (ptr + 1 > end) return false;
                out.type = read_result::INT;
                out.int_value = static_cast<int8_t>(*ptr++);
                return true;
            }
            case msgpack_format::INT16: {
                if (ptr + 2 > end) return false;
                out.int_value = static_cast<int16_t>((static_cast<uint16_t>(ptr[0]) << 8) | ptr[1]);
                ptr += 2;
                out.type = read_result::INT;
                return true;
            }
            case msgpack_format::INT32: {
                if (ptr + 4 > end) return false;
                out.int_value = static_cast<int32_t>((static_cast<uint32_t>(ptr[0]) << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3]);
                ptr += 4;
                out.type = read_result::INT;
                return true;
            }
            case msgpack_format::INT64: {
                if (ptr + 8 > end) return false;
                out.int_value = static_cast<int64_t>((static_cast<uint64_t>(ptr[0]) << 56) | (static_cast<uint64_t>(ptr[1]) << 48) |
                                                     (static_cast<uint64_t>(ptr[2]) << 40) | (static_cast<uint64_t>(ptr[3]) << 32) |
                                                     (static_cast<uint64_t>(ptr[4]) << 24) | (static_cast<uint64_t>(ptr[5]) << 16) |
                                                     (static_cast<uint64_t>(ptr[6]) << 8) | ptr[7]);
                ptr += 8;
                out.type = read_result::INT;
                return true;
            }
            case msgpack_format::STR8: {
                if (ptr + 1 > end) return false;
                uint32_t len = *ptr++;
                if (ptr + len > end) return false;
                out.type = read_result::STR;
                out.str_value.assign(reinterpret_cast<const char*>(ptr), len);
                ptr += len;
                return true;
            }
            case msgpack_format::STR16: {
                if (ptr + 2 > end) return false;
                uint32_t len = (static_cast<uint32_t>(ptr[0]) << 8) | ptr[1];
                ptr += 2;
                if (ptr + len > end) return false;
                out.type = read_result::STR;
                out.str_value.assign(reinterpret_cast<const char*>(ptr), len);
                ptr += len;
                return true;
            }
            case msgpack_format::STR32: {
                if (ptr + 4 > end) return false;
                uint32_t len = (static_cast<uint32_t>(ptr[0]) << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
                ptr += 4;
                if (ptr + len > end) return false;
                out.type = read_result::STR;
                out.str_value.assign(reinterpret_cast<const char*>(ptr), len);
                ptr += len;
                return true;
            }
            case msgpack_format::ARRAY16: {
                if (ptr + 2 > end) return false;
                out.type = read_result::ARRAY;
                out.uint_value = (static_cast<uint32_t>(ptr[0]) << 8) | ptr[1];
                ptr += 2;
                return true;
            }
            case msgpack_format::ARRAY32: {
                if (ptr + 4 > end) return false;
                out.type = read_result::ARRAY;
                out.uint_value = (static_cast<uint64_t>(ptr[0]) << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
                ptr += 4;
                return true;
            }
            case msgpack_format::MAP16: {
                if (ptr + 2 > end) return false;
                out.type = read_result::MAP;
                out.uint_value = (static_cast<uint32_t>(ptr[0]) << 8) | ptr[1];
                ptr += 2;
                return true;
            }
            case msgpack_format::MAP32: {
                if (ptr + 4 > end) return false;
                out.type = read_result::MAP;
                out.uint_value = (static_cast<uint64_t>(ptr[0]) << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
                ptr += 4;
                return true;
            }
            default:
                // Unknown type
                return false;
        }
    }

    Config m_config;
    bool m_error;
};

// ============================================================================
//  MsgPack output archive
// ============================================================================
class MsgPackOutputArchive : public msgpack_archive_base {
public:
    using buffer_type = std::vector<uint8_t>;

    MsgPackOutputArchive() : msgpack_archive_base(), m_buffer() {}
    explicit MsgPackOutputArchive(const Config& cfg) : msgpack_archive_base(cfg), m_buffer() {}

    // ------------------------------------------------------------------------
    //  Public write interface
    // ------------------------------------------------------------------------
    template<typename T>
    MsgPackOutputArchive& operator&(const T& value) {
        write(value);
        return *this;
    }

    template<typename T>
    MsgPackOutputArchive& operator&(const nvp<T>& nv) {
        write(nv.value());
        return *this;
    }

    void write(bool value) { write_bool(m_buffer, value); }
    void write(uint8_t value) { write_uint8(m_buffer, value); }
    void write(uint16_t value) { write_uint16(m_buffer, value); }
    void write(uint32_t value) { write_uint32(m_buffer, value); }
    void write(uint64_t value) { write_uint64(m_buffer, value); }
    void write(int8_t value) { write_int8(m_buffer, value); }
    void write(int16_t value) { write_int16(m_buffer, value); }
    void write(int32_t value) { write_int32(m_buffer, value); }
    void write(int64_t value) { write_int64(m_buffer, value); }
    void write(float value) { write_float32(m_buffer, value); }
    void write(double value) { write_float64(m_buffer, value); }
    void write(const std::string& value) { write_string(m_buffer, value); }
    void write(const char* value) { write(std::string(value)); }

    // Write raw bytes (as binary MsgPack)
    void write_bytes(const uint8_t* data, size_type len) {
        write_binary(m_buffer, data, len);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch write for arithmetic arrays (float, double, int)
    //  Writes as a MsgPack array of the appropriate type.
    // ------------------------------------------------------------------------
    template<typename T>
    void write_batch(const T* data, size_type count) {
        static_assert(std::is_arithmetic_v<T>, "write_batch only for arithmetic types");
        if (count == 0) {
            // write empty array (fixarray of size 0)
            m_buffer.push_back(msgpack_format::FIXARRAY_START | 0);
            return;
        }
        // Write array header
        if (count <= 15) {
            m_buffer.push_back(static_cast<uint8_t>(msgpack_format::FIXARRAY_START | count));
        } else if (count <= 0xffff) {
            m_buffer.push_back(msgpack_format::ARRAY16);
            m_buffer.push_back(static_cast<uint8_t>(count >> 8));
            m_buffer.push_back(static_cast<uint8_t>(count));
        } else {
            m_buffer.push_back(msgpack_format::ARRAY32);
            m_buffer.push_back(static_cast<uint8_t>(count >> 24));
            m_buffer.push_back(static_cast<uint8_t>(count >> 16));
            m_buffer.push_back(static_cast<uint8_t>(count >> 8));
            m_buffer.push_back(static_cast<uint8_t>(count));
        }
        // Write each element
        if (m_config.enableSIMD && count >= 4) {
            // For arrays, we can't use a single SIMD write because each element is separate MsgPack object.
            // We still loop but hint compiler.
            for (size_type i = 0; i < count; ++i) {
                write(data[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                write(data[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Container serialisation helpers
    // ------------------------------------------------------------------------
    void begin_array(size_type size) {
        if (size <= 15) {
            m_buffer.push_back(static_cast<uint8_t>(msgpack_format::FIXARRAY_START | size));
        } else if (size <= 0xffff) {
            m_buffer.push_back(msgpack_format::ARRAY16);
            m_buffer.push_back(static_cast<uint8_t>(size >> 8));
            m_buffer.push_back(static_cast<uint8_t>(size));
        } else {
            m_buffer.push_back(msgpack_format::ARRAY32);
            m_buffer.push_back(static_cast<uint8_t>(size >> 24));
            m_buffer.push_back(static_cast<uint8_t>(size >> 16));
            m_buffer.push_back(static_cast<uint8_t>(size >> 8));
            m_buffer.push_back(static_cast<uint8_t>(size));
        }
    }

    void begin_map(size_type size) {
        if (size <= 15) {
            m_buffer.push_back(static_cast<uint8_t>(msgpack_format::FIXMAP_START | size));
        } else if (size <= 0xffff) {
            m_buffer.push_back(msgpack_format::MAP16);
            m_buffer.push_back(static_cast<uint8_t>(size >> 8));
            m_buffer.push_back(static_cast<uint8_t>(size));
        } else {
            m_buffer.push_back(msgpack_format::MAP32);
            m_buffer.push_back(static_cast<uint8_t>(size >> 24));
            m_buffer.push_back(static_cast<uint8_t>(size >> 16));
            m_buffer.push_back(static_cast<uint8_t>(size >> 8));
            m_buffer.push_back(static_cast<uint8_t>(size));
        }
    }

    // ------------------------------------------------------------------------
    //  I/O and buffer access
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
//  MsgPack input archive
// ============================================================================
class MsgPackInputArchive : public msgpack_archive_base {
public:
    MsgPackInputArchive() : msgpack_archive_base(), m_buffer(), m_pos(0) {}
    explicit MsgPackInputArchive(const Config& cfg) : msgpack_archive_base(cfg), m_buffer(), m_pos(0) {}

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

    void loadFromMemory(const uint8_t* data, size_type size) {
        m_buffer.assign(data, data + size);
        m_pos = 0;
        m_error = false;
    }

    // ------------------------------------------------------------------------
    //  Read interface
    // ------------------------------------------------------------------------
    template<typename T>
    MsgPackInputArchive& operator&(T& value) {
        read(value);
        return *this;
    }

    template<typename T>
    MsgPackInputArchive& operator&(const nvp<T>& nv) {
        read(const_cast<T&>(nv.value()));
        return *this;
    }

    void read(bool& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::BOOL) {
            value = (res.uint_value != 0);
            advance();
        } else if (res.type == read_result::UINT) {
            value = (res.uint_value != 0);
            advance();
        } else if (res.type == read_result::INT) {
            value = (res.int_value != 0);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(uint8_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::UINT) {
            value = static_cast<uint8_t>(res.uint_value);
            advance();
        } else if (res.type == read_result::INT && res.int_value >= 0) {
            value = static_cast<uint8_t>(res.int_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(uint16_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::UINT) {
            value = static_cast<uint16_t>(res.uint_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(uint32_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::UINT) {
            value = static_cast<uint32_t>(res.uint_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(uint64_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::UINT) {
            value = res.uint_value;
            advance();
        } else if (res.type == read_result::INT && res.int_value >= 0) {
            value = static_cast<uint64_t>(res.int_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(int8_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::INT) {
            value = static_cast<int8_t>(res.int_value);
            advance();
        } else if (res.type == read_result::UINT && res.uint_value <= 127) {
            value = static_cast<int8_t>(res.uint_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(int16_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::INT) {
            value = static_cast<int16_t>(res.int_value);
            advance();
        } else if (res.type == read_result::UINT && res.uint_value <= 32767) {
            value = static_cast<int16_t>(res.uint_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(int32_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::INT) {
            value = static_cast<int32_t>(res.int_value);
            advance();
        } else if (res.type == read_result::UINT && res.uint_value <= 2147483647) {
            value = static_cast<int32_t>(res.uint_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(int64_t& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::INT) {
            value = res.int_value;
            advance();
        } else if (res.type == read_result::UINT && res.uint_value <= 9223372036854775807ULL) {
            value = static_cast<int64_t>(res.uint_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(float& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::FLOAT32) {
            value = static_cast<float>(res.float_value);
            advance();
        } else if (res.type == read_result::FLOAT64) {
            value = static_cast<float>(res.float_value);
            advance();
        } else if (res.type == read_result::UINT) {
            value = static_cast<float>(res.uint_value);
            advance();
        } else if (res.type == read_result::INT) {
            value = static_cast<float>(res.int_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(double& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::FLOAT64) {
            value = res.float_value;
            advance();
        } else if (res.type == read_result::FLOAT32) {
            value = static_cast<double>(res.float_value);
            advance();
        } else if (res.type == read_result::UINT) {
            value = static_cast<double>(res.uint_value);
            advance();
        } else if (res.type == read_result::INT) {
            value = static_cast<double>(res.int_value);
            advance();
        } else {
            m_error = true;
        }
    }

    void read(std::string& value) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::STR) {
            value = std::move(res.str_value);
            advance();
        } else if (res.type == read_result::BIN) {
            value.assign(reinterpret_cast<const char*>(res.bin_value.data()), res.bin_value.size());
            advance();
        } else {
            m_error = true;
        }
    }

    // Read raw binary (as MsgPack binary object)
    void read_bytes(std::vector<uint8_t>& out) {
        read_result res;
        if (!peek_next(res)) { m_error = true; return; }
        if (res.type == read_result::BIN) {
            out = std::move(res.bin_value);
            advance();
        } else {
            m_error = true;
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch read for arithmetic arrays (assumes the next item is an array)
    // ------------------------------------------------------------------------
    template<typename T>
    void read_batch(T* out, size_type expected_count) {
        static_assert(std::is_arithmetic_v<T>, "read_batch only for arithmetic types");
        // Read array header
        read_result res;
        if (!peek_next(res) || (res.type != read_result::ARRAY && res.type != read_result::MAP)) {
            m_error = true;
            return;
        }
        size_type count = static_cast<size_type>(res.uint_value);
        if (count != expected_count) {
            m_error = true;
            return;
        }
        advance(); // consume array header
        for (size_type i = 0; i < count; ++i) {
            read(out[i]);
        }
    }

    // ------------------------------------------------------------------------
    //  Container reading
    // ------------------------------------------------------------------------
    size_type begin_array() {
        read_result res;
        if (!peek_next(res) || res.type != read_result::ARRAY) {
            m_error = true;
            return 0;
        }
        size_type size = static_cast<size_type>(res.uint_value);
        advance();
        return size;
    }

    size_type begin_map() {
        read_result res;
        if (!peek_next(res) || res.type != read_result::MAP) {
            m_error = true;
            return 0;
        }
        size_type size = static_cast<size_type>(res.uint_value);
        advance();
        return size;
    }

    // ------------------------------------------------------------------------
    //  Position and size
    // ------------------------------------------------------------------------
    size_type position() const { return m_pos; }
    size_type size() const { return m_buffer.size(); }
    bool eof() const { return m_pos >= m_buffer.size(); }

private:
    bool peek_next(read_result& res) const {
        const uint8_t* ptr = m_buffer.data() + m_pos;
        const uint8_t* end = m_buffer.data() + m_buffer.size();
        return read_next(ptr, end, res);
    }

    void advance() {
        // Re-parse the current object to skip it (we already have the result from peek, but need to advance m_pos).
        // Since we don't store the parsed result's size, we need to call read_next again and update m_pos.
        const uint8_t* ptr = m_buffer.data() + m_pos;
        const uint8_t* end = m_buffer.data() + m_buffer.size();
        read_result dummy;
        if (read_next(ptr, end, dummy)) {
            m_pos = static_cast<size_type>(ptr - m_buffer.data());
        } else {
            m_error = true;
        }
    }

    std::vector<uint8_t> m_buffer;
    size_type m_pos;
};

// ============================================================================
//  Archive traits
// ============================================================================
template<>
struct is_msgpack_archive<MsgPackOutputArchive> : std::true_type {};
template<>
struct is_msgpack_archive<MsgPackInputArchive> : std::true_type {};
template<>
struct is_output_archive<MsgPackOutputArchive> : std::true_type {};
template<>
struct is_input_archive<MsgPackInputArchive> : std::true_type {};

// ============================================================================
//  Dynamic environment controller
// ============================================================================
class MsgPackArchiveEnvironment {
public:
    static MsgPackArchiveEnvironment& instance() {
        static MsgPackArchiveEnvironment env;
        return env;
    }

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
    MsgPackArchiveEnvironment() : m_defaultVersion(1), m_enableChecksum(false) {}
    std::mutex m_mutex;
    uint32_t m_defaultVersion;
    bool m_enableChecksum;
};

inline MsgPackOutputArchive make_msgpack_output_archive() {
    MsgPackOutputArchive::Config cfg;
    cfg.archiveVersion = MsgPackArchiveEnvironment::instance().defaultVersion();
    cfg.verifyChecksum = MsgPackArchiveEnvironment::instance().enableChecksum();
    return MsgPackOutputArchive(cfg);
}

inline MsgPackInputArchive make_msgpack_input_archive() {
    MsgPackInputArchive::Config cfg;
    cfg.archiveVersion = MsgPackArchiveEnvironment::instance().defaultVersion();
    cfg.verifyChecksum = MsgPackArchiveEnvironment::instance().enableChecksum();
    return MsgPackInputArchive(cfg);
}

} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_MSGPACK_ARCHIVE_H_INCLUDED