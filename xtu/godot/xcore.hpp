// include/xtu/godot/xcore.hpp
// xtensor-unified - Core utilities for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XCORE_HPP
#define XTU_GODOT_XCORE_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace core {

// #############################################################################
// Forward declarations
// #############################################################################
class CharString;
class String;
class StringName;
class NodePath;
class FileAccess;
class DirAccess;
class OS;
class Thread;
class Mutex;
class Semaphore;
class RWLock;

// #############################################################################
// CharString - 8-bit string representation
// #############################################################################
class CharString {
private:
    std::string m_data;

public:
    CharString() = default;
    CharString(const char* str) : m_data(str) {}
    CharString(const std::string& str) : m_data(str) {}
    CharString(const CharString&) = default;
    CharString(CharString&&) = default;
    CharString& operator=(const CharString&) = default;
    CharString& operator=(CharString&&) = default;

    const char* get_data() const { return m_data.c_str(); }
    const char* c_str() const { return m_data.c_str(); }
    size_t length() const { return m_data.size(); }
    bool empty() const { return m_data.empty(); }

    char operator[](size_t idx) const { return m_data[idx]; }
    char& operator[](size_t idx) { return m_data[idx]; }

    bool operator==(const CharString& other) const { return m_data == other.m_data; }
    bool operator!=(const CharString& other) const { return m_data != other.m_data; }
    bool operator<(const CharString& other) const { return m_data < other.m_data; }

    std::string to_std_string() const { return m_data; }
};

// #############################################################################
// String - UTF-8 string with reference counting and copy-on-write
// #############################################################################
class String {
private:
    struct SharedData {
        std::string utf8;
        mutable std::atomic<int32_t> ref_count;
        SharedData(const std::string& str) : utf8(str), ref_count(1) {}
    };
    SharedData* m_data = nullptr;

    void release() {
        if (m_data && --m_data->ref_count == 0) {
            delete m_data;
            m_data = nullptr;
        }
    }

    void copy_on_write() {
        if (m_data && m_data->ref_count > 1) {
            auto* new_data = new SharedData(m_data->utf8);
            --m_data->ref_count;
            m_data = new_data;
        }
    }

public:
    String() = default;
    String(const char* str) : m_data(new SharedData(str)) {}
    String(const std::string& str) : m_data(new SharedData(str)) {}
    String(const String& other) : m_data(other.m_data) {
        if (m_data) ++m_data->ref_count;
    }
    String(String&& other) noexcept : m_data(other.m_data) {
        other.m_data = nullptr;
    }
    ~String() { release(); }

    String& operator=(const String& other) {
        if (this != &other) {
            release();
            m_data = other.m_data;
            if (m_data) ++m_data->ref_count;
        }
        return *this;
    }

    String& operator=(String&& other) noexcept {
        if (this != &other) {
            release();
            m_data = other.m_data;
            other.m_data = nullptr;
        }
        return *this;
    }

    String& operator=(const char* str) {
        release();
        m_data = new SharedData(str);
        return *this;
    }

    String& operator=(const std::string& str) {
        release();
        m_data = new SharedData(str);
        return *this;
    }

    // Length and emptiness
    size_t length() const { return m_data ? m_data->utf8.size() : 0; }
    bool empty() const { return !m_data || m_data->utf8.empty(); }
    bool is_empty() const { return empty(); }

    // Access
    const char* utf8() const { return m_data ? m_data->utf8.c_str() : ""; }
    const char* c_str() const { return utf8(); }
    std::string to_std_string() const { return m_data ? m_data->utf8 : ""; }

    // Operators
    String operator+(const String& other) const {
        if (!m_data) return other;
        if (!other.m_data) return *this;
        return String(m_data->utf8 + other.m_data->utf8);
    }

    String operator+(const char* other) const {
        if (!m_data) return String(other);
        return String(m_data->utf8 + other);
    }

    String& operator+=(const String& other) {
        if (!other.m_data) return *this;
        copy_on_write();
        if (!m_data) m_data = new SharedData("");
        m_data->utf8 += other.m_data->utf8;
        return *this;
    }

    String& operator+=(const char* other) {
        copy_on_write();
        if (!m_data) m_data = new SharedData("");
        m_data->utf8 += other;
        return *this;
    }

    bool operator==(const String& other) const {
        if (m_data == other.m_data) return true;
        if (!m_data || !other.m_data) return false;
        return m_data->utf8 == other.m_data->utf8;
    }

    bool operator!=(const String& other) const { return !(*this == other); }
    bool operator<(const String& other) const {
        if (!m_data) return other.m_data != nullptr;
        if (!other.m_data) return false;
        return m_data->utf8 < other.m_data->utf8;
    }

    // Substring and find
    String substr(size_t pos, size_t len = std::string::npos) const {
        if (!m_data) return String();
        return String(m_data->utf8.substr(pos, len));
    }

    size_t find(const String& what, size_t from = 0) const {
        if (!m_data || !what.m_data) return std::string::npos;
        return m_data->utf8.find(what.m_data->utf8, from);
    }

    size_t find(const char* what, size_t from = 0) const {
        if (!m_data) return std::string::npos;
        return m_data->utf8.find(what, from);
    }

    size_t rfind(const String& what, size_t from = std::string::npos) const {
        if (!m_data || !what.m_data) return std::string::npos;
        return m_data->utf8.rfind(what.m_data->utf8, from);
    }

    bool begins_with(const String& prefix) const {
        if (!m_data || !prefix.m_data) return false;
        return m_data->utf8.compare(0, prefix.m_data->utf8.size(), prefix.m_data->utf8) == 0;
    }

    bool ends_with(const String& suffix) const {
        if (!m_data || !suffix.m_data) return false;
        if (suffix.m_data->utf8.size() > m_data->utf8.size()) return false;
        return m_data->utf8.compare(m_data->utf8.size() - suffix.m_data->utf8.size(),
                                    suffix.m_data->utf8.size(), suffix.m_data->utf8) == 0;
    }

    // Conversion
    CharString ascii() const { return CharString(m_data ? m_data->utf8 : ""); }
    CharString utf8_buffer() const { return ascii(); }

    int64_t to_int() const {
        if (!m_data) return 0;
        return std::stoll(m_data->utf8);
    }

    double to_float() const {
        if (!m_data) return 0.0;
        return std::stod(m_data->utf8);
    }

    // Static constructors
    static String num(int64_t value, int base = 10) {
        if (base == 10) return String(std::to_string(value));
        // Handle other bases
        return String(std::to_string(value));
    }

    static String num(double value, int decimals = -1) {
        char buf[64];
        if (decimals >= 0) {
            snprintf(buf, sizeof(buf), "%.*f", decimals, value);
        } else {
            snprintf(buf, sizeof(buf), "%g", value);
        }
        return String(buf);
    }

    static String num_scientific(double value) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%e", value);
        return String(buf);
    }

    // Split
    std::vector<String> split(const String& delimiter, bool allow_empty = true, int maxsplit = 0) const {
        std::vector<String> result;
        if (!m_data) return result;
        size_t pos = 0;
        int count = 0;
        while (pos <= m_data->utf8.size()) {
            size_t end = m_data->utf8.find(delimiter.m_data->utf8, pos);
            if (end == std::string::npos) {
                if (allow_empty || pos < m_data->utf8.size()) {
                    result.push_back(substr(pos));
                }
                break;
            }
            if (allow_empty || end > pos) {
                result.push_back(substr(pos, end - pos));
            }
            pos = end + delimiter.length();
            if (maxsplit > 0 && ++count >= maxsplit) {
                if (pos < m_data->utf8.size() || allow_empty) {
                    result.push_back(substr(pos));
                }
                break;
            }
        }
        return result;
    }

    // Replace
    String replace(const String& what, const String& with) const {
        if (!m_data || !what.m_data) return *this;
        std::string result = m_data->utf8;
        size_t pos = 0;
        while ((pos = result.find(what.m_data->utf8, pos)) != std::string::npos) {
            result.replace(pos, what.m_data->utf8.size(), with.m_data ? with.m_data->utf8 : "");
            pos += with.length();
        }
        return String(result);
    }
};

// #############################################################################
// StringName - Interned string for fast comparison
// #############################################################################
class StringName {
private:
    static std::unordered_map<uint64_t, String> s_interned_strings;
    static std::mutex s_mutex;
    uint64_t m_hash = 0;
    const String* m_ptr = nullptr;

    static uint64_t compute_hash(const String& str) {
        const char* c = str.utf8();
        uint64_t hash = 14695981039346656037ULL;
        while (*c) {
            hash ^= static_cast<uint64_t>(*c++);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

public:
    StringName() = default;
    StringName(const String& str) : m_hash(compute_hash(str)) {
        std::lock_guard<std::mutex> lock(s_mutex);
        auto it = s_interned_strings.find(m_hash);
        if (it == s_interned_strings.end()) {
            it = s_interned_strings.emplace(m_hash, str).first;
        }
        m_ptr = &it->second;
    }

    StringName(const char* str) : StringName(String(str)) {}
    StringName(const StringName&) = default;
    StringName(StringName&&) = default;
    StringName& operator=(const StringName&) = default;
    StringName& operator=(StringName&&) = default;

    const String& string() const {
        static String empty;
        return m_ptr ? *m_ptr : empty;
    }

    const char* c_str() const { return m_ptr ? m_ptr->utf8() : ""; }
    uint64_t hash() const { return m_hash; }
    size_t length() const { return m_ptr ? m_ptr->length() : 0; }
    bool empty() const { return m_ptr == nullptr; }

    bool operator==(const StringName& other) const { return m_hash == other.m_hash; }
    bool operator!=(const StringName& other) const { return m_hash != other.m_hash; }
    bool operator<(const StringName& other) const { return m_hash < other.m_hash; }

    explicit operator bool() const { return m_ptr != nullptr; }
};

inline std::unordered_map<uint64_t, String> StringName::s_interned_strings;
inline std::mutex StringName::s_mutex;

// #############################################################################
// NodePath - Path to a node in the scene tree
// #############################################################################
class NodePath {
private:
    std::vector<StringName> m_names;
    std::vector<StringName> m_subnames;
    bool m_absolute = false;

public:
    NodePath() = default;
    NodePath(const String& path) { parse(path); }
    NodePath(const char* path) : NodePath(String(path)) {}

    void parse(const String& path) {
        m_names.clear();
        m_subnames.clear();
        std::string p = path.to_std_string();
        m_absolute = !p.empty() && p[0] == '/';
        size_t pos = m_absolute ? 1 : 0;
        while (pos < p.size()) {
            size_t end = p.find('/', pos);
            if (end == std::string::npos) end = p.size();
            std::string segment = p.substr(pos, end - pos);
            size_t colon = segment.find(':');
            if (colon != std::string::npos) {
                m_names.push_back(StringName(segment.substr(0, colon).c_str()));
                m_subnames.push_back(StringName(segment.substr(colon + 1).c_str()));
            } else {
                m_names.push_back(StringName(segment.c_str()));
                m_subnames.push_back(StringName());
            }
            pos = end + 1;
        }
    }

    String to_string() const {
        std::string result = m_absolute ? "/" : "";
        for (size_t i = 0; i < m_names.size(); ++i) {
            if (i > 0) result += "/";
            result += m_names[i].string().to_std_string();
            if (!m_subnames[i].empty()) {
                result += ":" + m_subnames[i].string().to_std_string();
            }
        }
        return String(result);
    }

    size_t get_name_count() const { return m_names.size(); }
    StringName get_name(size_t idx) const { return idx < m_names.size() ? m_names[idx] : StringName(); }
    StringName get_subname(size_t idx) const { return idx < m_subnames.size() ? m_subnames[idx] : StringName(); }
    bool is_absolute() const { return m_absolute; }
    bool is_empty() const { return m_names.empty(); }
};

// #############################################################################
// FileAccess - File I/O abstraction
// #############################################################################
class FileAccess {
public:
    enum ModeFlags : uint32_t {
        READ = 1 << 0,
        WRITE = 1 << 1,
        READ_WRITE = READ | WRITE,
        WRITE_READ = WRITE | READ
    };

    enum CompressionMode : uint8_t {
        COMPRESSION_FASTLZ = 0,
        COMPRESSION_DEFLATE = 1,
        COMPRESSION_ZSTD = 2,
        COMPRESSION_GZIP = 3
    };

private:
    std::fstream m_file;
    std::string m_path;
    ModeFlags m_mode = READ;
    bool m_open = false;
    size_t m_position = 0;

public:
    FileAccess() = default;
    ~FileAccess() { close(); }

    static Ref<FileAccess> open(const String& path, ModeFlags flags) {
        Ref<FileAccess> fa;
        fa.instance();
        if (fa->open_internal(path, flags)) {
            return fa;
        }
        return Ref<FileAccess>();
    }

    static Ref<FileAccess> open_compressed(const String& path, ModeFlags flags, CompressionMode compression) {
        // Compressed file support would require zlib integration
        return open(path, flags);
    }

    static Ref<FileAccess> open_encrypted(const String& path, ModeFlags flags, const std::vector<uint8_t>& key) {
        // Encrypted file support
        return open(path, flags);
    }

    static bool file_exists(const String& path) {
        return std::filesystem::exists(path.to_std_string());
    }

    static uint64_t get_modified_time(const String& path) {
        auto ftime = std::filesystem::last_write_time(path.to_std_string());
        return std::chrono::duration_cast<std::chrono::seconds>(
            ftime.time_since_epoch()).count();
    }

    bool open_internal(const String& path, ModeFlags flags) {
        close();
        m_path = path.to_std_string();
        m_mode = flags;
        std::ios::openmode mode = std::ios::binary;
        if (flags & READ) mode |= std::ios::in;
        if (flags & WRITE) mode |= std::ios::out;
        m_file.open(m_path, mode);
        m_open = m_file.is_open();
        if (m_open) {
            m_file.seekg(0, std::ios::end);
            m_position = 0;
        }
        return m_open;
    }

    void close() {
        if (m_open) {
            m_file.close();
            m_open = false;
        }
    }

    bool is_open() const { return m_open; }
    const String& get_path() const { static String p; p = m_path; return p; }

    void seek(size_t pos) {
        if (m_open) {
            m_file.seekg(pos);
            m_position = pos;
        }
    }

    void seek_end(int64_t pos = 0) {
        if (m_open) {
            m_file.seekg(pos, std::ios::end);
            m_position = m_file.tellg();
        }
    }

    size_t get_position() const { return m_position; }
    size_t get_length() const {
        if (!m_open) return 0;
        auto current = m_file.tellg();
        m_file.seekg(0, std::ios::end);
        size_t len = m_file.tellg();
        m_file.seekg(current);
        return len;
    }

    bool eof_reached() const { return m_open && m_file.eof(); }

    // Reading
    uint8_t get_8() {
        uint8_t val = 0;
        if (m_open && (m_mode & READ)) {
            m_file.read(reinterpret_cast<char*>(&val), 1);
            ++m_position;
        }
        return val;
    }

    uint16_t get_16() {
        uint16_t val = 0;
        if (m_open && (m_mode & READ)) {
            m_file.read(reinterpret_cast<char*>(&val), 2);
            m_position += 2;
        }
        return val;
    }

    uint32_t get_32() {
        uint32_t val = 0;
        if (m_open && (m_mode & READ)) {
            m_file.read(reinterpret_cast<char*>(&val), 4);
            m_position += 4;
        }
        return val;
    }

    uint64_t get_64() {
        uint64_t val = 0;
        if (m_open && (m_mode & READ)) {
            m_file.read(reinterpret_cast<char*>(&val), 8);
            m_position += 8;
        }
        return val;
    }

    float get_float() {
        float val = 0;
        if (m_open && (m_mode & READ)) {
            m_file.read(reinterpret_cast<char*>(&val), 4);
            m_position += 4;
        }
        return val;
    }

    double get_double() {
        double val = 0;
        if (m_open && (m_mode & READ)) {
            m_file.read(reinterpret_cast<char*>(&val), 8);
            m_position += 8;
        }
        return val;
    }

    String get_line() {
        std::string line;
        if (m_open && (m_mode & READ)) {
            std::getline(m_file, line);
            m_position = m_file.tellg();
        }
        return String(line);
    }

    std::vector<uint8_t> get_buffer(size_t len) {
        std::vector<uint8_t> buf(len);
        if (m_open && (m_mode & READ)) {
            m_file.read(reinterpret_cast<char*>(buf.data()), len);
            m_position += len;
        }
        return buf;
    }

    String get_as_text() {
        if (!m_open) return String();
        seek(0);
        std::string content((std::istreambuf_iterator<char>(m_file)),
                            std::istreambuf_iterator<char>());
        return String(content);
    }

    // Writing
    void store_8(uint8_t val) {
        if (m_open && (m_mode & WRITE)) {
            m_file.write(reinterpret_cast<const char*>(&val), 1);
            ++m_position;
        }
    }

    void store_16(uint16_t val) {
        if (m_open && (m_mode & WRITE)) {
            m_file.write(reinterpret_cast<const char*>(&val), 2);
            m_position += 2;
        }
    }

    void store_32(uint32_t val) {
        if (m_open && (m_mode & WRITE)) {
            m_file.write(reinterpret_cast<const char*>(&val), 4);
            m_position += 4;
        }
    }

    void store_64(uint64_t val) {
        if (m_open && (m_mode & WRITE)) {
            m_file.write(reinterpret_cast<const char*>(&val), 8);
            m_position += 8;
        }
    }

    void store_float(float val) {
        if (m_open && (m_mode & WRITE)) {
            m_file.write(reinterpret_cast<const char*>(&val), 4);
            m_position += 4;
        }
    }

    void store_double(double val) {
        if (m_open && (m_mode & WRITE)) {
            m_file.write(reinterpret_cast<const char*>(&val), 8);
            m_position += 8;
        }
    }

    void store_string(const String& str) {
        store_32(static_cast<uint32_t>(str.length()));
        if (m_open && (m_mode & WRITE)) {
            m_file.write(str.utf8(), str.length());
            m_position += str.length();
        }
    }

    void store_buffer(const std::vector<uint8_t>& buf) {
        if (m_open && (m_mode & WRITE)) {
            m_file.write(reinterpret_cast<const char*>(buf.data()), buf.size());
            m_position += buf.size();
        }
    }

    void flush() { if (m_open) m_file.flush(); }
};

// #############################################################################
// DirAccess - Directory operations
// #############################################################################
class DirAccess {
private:
    std::filesystem::path m_current_path;
    std::filesystem::directory_iterator m_iterator;
    bool m_listing = false;

public:
    static Ref<DirAccess> open(const String& path) {
        Ref<DirAccess> da;
        da.instance();
        if (da->change_dir(path) == OK) {
            return da;
        }
        return Ref<DirAccess>();
    }

    static Ref<DirAccess> create_for_path(const String& path) {
        return open(path);
    }

    int change_dir(const String& path) {
        std::filesystem::path p(path.to_std_string());
        if (std::filesystem::exists(p) && std::filesystem::is_directory(p)) {
            m_current_path = p;
            return OK;
        }
        return ERR_FILE_NOT_FOUND;
    }

    String get_current_dir() const {
        return String(m_current_path.string());
    }

    int make_dir(const String& path) {
        std::filesystem::path p(path.to_std_string());
        if (std::filesystem::create_directory(p)) {
            return OK;
        }
        return ERR_CANT_CREATE;
    }

    int make_dir_recursive(const String& path) {
        std::filesystem::path p(path.to_std_string());
        if (std::filesystem::create_directories(p)) {
            return OK;
        }
        return ERR_CANT_CREATE;
    }

    int remove(const String& path) {
        std::filesystem::path p(path.to_std_string());
        if (std::filesystem::remove(p)) {
            return OK;
        }
        return ERR_FILE_NOT_FOUND;
    }

    int rename(const String& from, const String& to) {
        std::filesystem::path p_from(from.to_std_string());
        std::filesystem::path p_to(to.to_std_string());
        std::error_code ec;
        std::filesystem::rename(p_from, p_to, ec);
        return ec ? ERR_FILE_CANT_WRITE : OK;
    }

    bool dir_exists(const String& path) {
        std::filesystem::path p(path.to_std_string());
        return std::filesystem::exists(p) && std::filesystem::is_directory(p);
    }

    bool file_exists(const String& path) {
        std::filesystem::path p(path.to_std_string());
        return std::filesystem::exists(p) && std::filesystem::is_regular_file(p);
    }

    void list_dir_begin() {
        m_iterator = std::filesystem::directory_iterator(m_current_path);
        m_listing = true;
    }

    String get_next() {
        if (!m_listing) return String();
        if (m_iterator == std::filesystem::end(m_iterator)) {
            m_listing = false;
            return String();
        }
        String name = String(m_iterator->path().filename().string());
        ++m_iterator;
        return name;
    }

    bool current_is_dir() const {
        return m_listing && m_iterator != std::filesystem::end(m_iterator) &&
               std::filesystem::is_directory(m_iterator->path());
    }

    void list_dir_end() {
        m_listing = false;
    }

    std::vector<String> get_directories() {
        std::vector<String> dirs;
        for (const auto& entry : std::filesystem::directory_iterator(m_current_path)) {
            if (entry.is_directory()) {
                dirs.push_back(String(entry.path().filename().string()));
            }
        }
        return dirs;
    }

    std::vector<String> get_files() {
        std::vector<String> files;
        for (const auto& entry : std::filesystem::directory_iterator(m_current_path)) {
            if (entry.is_regular_file()) {
                files.push_back(String(entry.path().filename().string()));
            }
        }
        return files;
    }

    int copy(const String& from, const String& to) {
        std::filesystem::path p_from(from.to_std_string());
        std::filesystem::path p_to(to.to_std_string());
        std::error_code ec;
        std::filesystem::copy(p_from, p_to, std::filesystem::copy_options::recursive, ec);
        return ec ? ERR_FILE_CANT_WRITE : OK;
    }
};

// #############################################################################
// OS - Operating system abstraction
// #############################################################################
class OS {
public:
    enum SystemDir {
        SYSTEM_DIR_DESKTOP,
        SYSTEM_DIR_DOCUMENTS,
        SYSTEM_DIR_DOWNLOADS,
        SYSTEM_DIR_MUSIC,
        SYSTEM_DIR_PICTURES,
        SYSTEM_DIR_VIDEOS
    };

    static OS* get_singleton() {
        static OS instance;
        return &instance;
    }

    String get_name() const {
#ifdef XTU_OS_WINDOWS
        return "Windows";
#elif defined(XTU_OS_LINUX)
        return "Linux";
#elif defined(XTU_OS_MACOS)
        return "macOS";
#else
        return "Unknown";
#endif
    }

    String get_version() const { return "1.0"; }
    String get_distribution_name() const { return ""; }
    String get_locale() const { return "en_US"; }
    String get_latin_keyboard_variant() const { return "QWERTY"; }

    String get_model_name() const { return ""; }
    String get_unique_id() const { return ""; }

    bool is_debug_build() const {
#ifdef NDEBUG
        return false;
#else
        return true;
#endif
    }

    bool is_stdout_verbose() const { return false; }
    bool is_userfs_persistent() const { return true; }

    int get_exit_code() const { return 0; }
    void set_exit_code(int code) {}

    int get_processor_count() const {
        return static_cast<int>(std::thread::hardware_concurrency());
    }

    String get_executable_path() const {
        return String(std::filesystem::current_path().string());
    }

    String get_user_data_dir() const {
        std::string home;
#ifdef XTU_OS_WINDOWS
        home = std::getenv("APPDATA");
#else
        home = std::getenv("HOME");
#endif
        return String(home + "/.godot");
    }

    String get_system_dir(SystemDir dir) const {
        std::string home;
#ifdef XTU_OS_WINDOWS
        home = std::getenv("USERPROFILE");
#else
        home = std::getenv("HOME");
#endif
        switch (dir) {
            case SYSTEM_DIR_DESKTOP: return String(home + "/Desktop");
            case SYSTEM_DIR_DOCUMENTS: return String(home + "/Documents");
            case SYSTEM_DIR_DOWNLOADS: return String(home + "/Downloads");
            default: return String(home);
        }
    }

    void delay_usec(uint32_t usec) const {
        std::this_thread::sleep_for(std::chrono::microseconds(usec));
    }

    void delay_msec(uint32_t msec) const {
        std::this_thread::sleep_for(std::chrono::milliseconds(msec));
    }

    uint64_t get_ticks_msec() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    uint64_t get_ticks_usec() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    uint64_t get_unix_time() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void set_time_scale(float scale) { m_time_scale = scale; }
    float get_time_scale() const { return m_time_scale; }

    void set_low_processor_usage_mode(bool enable) { m_low_processor_mode = enable; }
    bool is_low_processor_usage_mode() const { return m_low_processor_mode; }

    String get_environment(const String& var) const {
        const char* val = std::getenv(var.utf8());
        return val ? String(val) : String();
    }

    bool set_environment(const String& var, const String& value) const {
#ifdef XTU_OS_WINDOWS
        return _putenv_s(var.utf8(), value.utf8()) == 0;
#else
        return setenv(var.utf8(), value.utf8(), 1) == 0;
#endif
    }

    void print(const String& str) { std::cout << str.utf8(); }
    void print_error(const String& str) { std::cerr << "ERROR: " << str.utf8() << std::endl; }
    void print_warning(const String& str) { std::cerr << "WARNING: " << str.utf8() << std::endl; }

    void alert(const String& text, const String& title = "Alert") {}
    void crash(const String& message) {
        std::cerr << "CRASH: " << message.utf8() << std::endl;
        std::abort();
    }

private:
    float m_time_scale = 1.0f;
    bool m_low_processor_mode = false;
};

// #############################################################################
// Mutex - Mutual exclusion lock
// #############################################################################
class Mutex {
private:
    std::mutex m_mutex;

public:
    void lock() { m_mutex.lock(); }
    void unlock() { m_mutex.unlock(); }
    bool try_lock() { return m_mutex.try_lock(); }
};

// #############################################################################
// Semaphore - Counting semaphore
// #############################################################################
class Semaphore {
private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    int32_t m_count = 0;

public:
    Semaphore(int32_t initial = 0) : m_count(initial) {}

    void post() {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_count;
        m_cv.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] { return m_count > 0; });
        --m_count;
    }

    bool try_wait() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_count > 0) {
            --m_count;
            return true;
        }
        return false;
    }
};

// #############################################################################
// RWLock - Read-write lock
// #############################################################################
class RWLock {
private:
    std::shared_mutex m_mutex;

public:
    void read_lock() { m_mutex.lock_shared(); }
    void read_unlock() { m_mutex.unlock_shared(); }
    void write_lock() { m_mutex.lock(); }
    void write_unlock() { m_mutex.unlock(); }
};

// #############################################################################
// Thread - Platform thread wrapper
// #############################################################################
class Thread {
private:
    std::thread m_thread;
    StringName m_name;
    std::function<void()> m_callback;
    bool m_started = false;

public:
    Thread() = default;
    ~Thread() { wait_to_finish(); }

    int start(const std::function<void()>& callback) {
        if (m_started) return ERR_ALREADY_IN_USE;
        m_callback = callback;
        m_thread = std::thread([this]() {
            if (m_callback) m_callback();
        });
        m_started = true;
        return OK;
    }

    void wait_to_finish() {
        if (m_started && m_thread.joinable()) {
            m_thread.join();
            m_started = false;
        }
    }

    bool is_started() const { return m_started; }
    bool is_alive() const { return m_started && m_thread.joinable(); }

    void set_name(const StringName& name) { m_name = name; }
    StringName get_name() const { return m_name; }

    static uint64_t get_caller_id() {
        return std::hash<std::thread::id>{}(std::this_thread::get_id());
    }

    static uint64_t get_main_id() {
        static uint64_t main_id = get_caller_id();
        return main_id;
    }
};

// #############################################################################
// Error macros and return codes
// #############################################################################
enum Error : int {
    OK = 0,
    FAILED = 1,
    ERR_UNAVAILABLE = 2,
    ERR_UNCONFIGURED = 3,
    ERR_UNAUTHORIZED = 4,
    ERR_PARAMETER_RANGE_ERROR = 5,
    ERR_OUT_OF_MEMORY = 6,
    ERR_FILE_NOT_FOUND = 7,
    ERR_FILE_BAD_DRIVE = 8,
    ERR_FILE_BAD_PATH = 9,
    ERR_FILE_NO_PERMISSION = 10,
    ERR_FILE_ALREADY_IN_USE = 11,
    ERR_FILE_CANT_OPEN = 12,
    ERR_FILE_CANT_WRITE = 13,
    ERR_FILE_CANT_READ = 14,
    ERR_FILE_UNRECOGNIZED = 15,
    ERR_FILE_CORRUPT = 16,
    ERR_FILE_MISSING_DEPENDENCIES = 17,
    ERR_FILE_EOF = 18,
    ERR_CANT_OPEN = 19,
    ERR_CANT_CREATE = 20,
    ERR_PARSE_ERROR = 21,
    ERR_QUERY_FAILED = 22,
    ERR_ALREADY_IN_USE = 23,
    ERR_LOCKED = 24,
    ERR_TIMEOUT = 25,
    ERR_CANT_CONNECT = 26,
    ERR_CANT_RESOLVE = 27,
    ERR_CONNECTION_ERROR = 28,
    ERR_CANT_ACQUIRE_RESOURCE = 29,
    ERR_INVALID_DATA = 30,
    ERR_INVALID_PARAMETER = 31,
    ERR_ALREADY_EXISTS = 32,
    ERR_DOES_NOT_EXIST = 33,
    ERR_DATABASE_CANT_READ = 34,
    ERR_DATABASE_CANT_WRITE = 35,
    ERR_COMPILATION_FAILED = 36,
    ERR_METHOD_NOT_FOUND = 37,
    ERR_LINK_FAILED = 38,
    ERR_SCRIPT_FAILED = 39,
    ERR_CYCLIC_LINK = 40,
    ERR_INVALID_DECLARATION = 41,
    ERR_DUPLICATE_SYMBOL = 42,
    ERR_PARSE_ERROR_TOKEN = 43,
    ERR_BUSY = 44,
    ERR_HELP = 45,
    ERR_BUG = 46,
    ERR_PRINTER_ON_FIRE = 47
};

#define ERR_FAIL_COND(cond) do { if (cond) return ERR_FAILED; } while(0)
#define ERR_FAIL_COND_V(cond, ret) do { if (cond) return ret; } while(0)
#define ERR_FAIL_NULL(cond) do { if (cond) return nullptr; } while(0)
#define ERR_FAIL_INDEX(idx, size) do { if (idx < 0 || idx >= size) return ERR_PARAMETER_RANGE_ERROR; } while(0)

// #############################################################################
// Vector - Fast dynamic array
// #############################################################################
template <typename T>
class Vector {
private:
    std::vector<T> m_data;

public:
    Vector() = default;
    Vector(std::initializer_list<T> init) : m_data(init) {}
    explicit Vector(size_t size) : m_data(size) {}

    void push_back(const T& value) { m_data.push_back(value); }
    void push_back(T&& value) { m_data.push_back(std::move(value)); }
    void pop_back() { m_data.pop_back(); }
    void resize(size_t size) { m_data.resize(size); }
    void clear() { m_data.clear(); }
    void reserve(size_t size) { m_data.reserve(size); }

    size_t size() const { return m_data.size(); }
    bool empty() const { return m_data.empty(); }

    T& operator[](size_t idx) { return m_data[idx]; }
    const T& operator[](size_t idx) const { return m_data[idx]; }

    T* data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }

    typename std::vector<T>::iterator begin() { return m_data.begin(); }
    typename std::vector<T>::iterator end() { return m_data.end(); }
    typename std::vector<T>::const_iterator begin() const { return m_data.begin(); }
    typename std::vector<T>::const_iterator end() const { return m_data.end(); }

    int find(const T& value) const {
        auto it = std::find(m_data.begin(), m_data.end(), value);
        return it != m_data.end() ? static_cast<int>(std::distance(m_data.begin(), it)) : -1;
    }

    bool has(const T& value) const { return find(value) != -1; }
    void erase(const T& value) {
        auto it = std::find(m_data.begin(), m_data.end(), value);
        if (it != m_data.end()) m_data.erase(it);
    }

    void remove_at(size_t idx) {
        if (idx < m_data.size()) m_data.erase(m_data.begin() + idx);
    }

    void sort() { std::sort(m_data.begin(), m_data.end()); }
    template <typename Compare>
    void sort_custom(Compare comp) { std::sort(m_data.begin(), m_data.end(), comp); }
};

// #############################################################################
// HashMap - Fast hash map
// #############################################################################
template <typename K, typename V, typename Hash = std::hash<K>>
class HashMap {
private:
    std::unordered_map<K, V, Hash> m_map;

public:
    void insert(const K& key, const V& value) { m_map[key] = value; }
    bool has(const K& key) const { return m_map.find(key) != m_map.end(); }
    V& get(const K& key) { return m_map[key]; }
    const V& get(const K& key) const { return m_map.at(key); }
    V* getptr(const K& key) {
        auto it = m_map.find(key);
        return it != m_map.end() ? &it->second : nullptr;
    }
    void erase(const K& key) { m_map.erase(key); }
    void clear() { m_map.clear(); }
    size_t size() const { return m_map.size(); }
    bool empty() const { return m_map.empty(); }

    auto begin() { return m_map.begin(); }
    auto end() { return m_map.end(); }
    auto begin() const { return m_map.begin(); }
    auto end() const { return m_map.end(); }
};

// #############################################################################
// HashSet - Fast hash set
// #############################################################################
template <typename T, typename Hash = std::hash<T>>
class HashSet {
private:
    std::unordered_set<T, Hash> m_set;

public:
    void insert(const T& value) { m_set.insert(value); }
    bool has(const T& value) const { return m_set.find(value) != m_set.end(); }
    void erase(const T& value) { m_set.erase(value); }
    void clear() { m_set.clear(); }
    size_t size() const { return m_set.size(); }
    bool empty() const { return m_set.empty(); }

    auto begin() { return m_set.begin(); }
    auto end() { return m_set.end(); }
    auto begin() const { return m_set.begin(); }
    auto end() const { return m_set.end(); }
};

} // namespace core

// Bring into godot namespace
using core::String;
using core::StringName;
using core::CharString;
using core::NodePath;
using core::FileAccess;
using core::DirAccess;
using core::OS;
using core::Thread;
using core::Mutex;
using core::Semaphore;
using core::RWLock;
using core::Vector;
using core::HashMap;
using core::HashSet;
using core::Error;

} // namespace godot
XTU_NAMESPACE_END

#endif // XTU_GODOT_XCORE_HPP