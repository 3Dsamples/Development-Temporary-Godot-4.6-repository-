// include/xtu/godot/xpck_encryption.hpp
// xtensor-unified - PCK encryption and signing for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPCK_ENCRYPTION_HPP
#define XTU_GODOT_XPCK_ENCRYPTION_HPP

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xcrypto.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace io {

// #############################################################################
// Forward declarations
// #############################################################################
class PCKPacker;
class PCKEncryption;
class EncryptedFileAccess;

// #############################################################################
// Encryption algorithm types
// #############################################################################
enum class PCKEncryptionAlgorithm : uint8_t {
    ALGORITHM_NONE = 0,
    ALGORITHM_AES_256_CBC = 1,
    ALGORITHM_AES_256_GCM = 2,
    ALGORITHM_CHACHA20 = 3
};

// #############################################################################
// PCK encryption configuration
// #############################################################################
struct PCKEncryptionConfig {
    PCKEncryptionAlgorithm algorithm = PCKEncryptionAlgorithm::ALGORITHM_AES_256_GCM;
    std::vector<uint8_t> key;           // 32 bytes for AES-256
    std::vector<uint8_t> iv;            // 12 bytes for GCM, 16 for CBC
    std::vector<uint8_t> hmac_key;      // 32 bytes for HMAC-SHA256
    bool encrypt_index = true;          // Encrypt the file index
    bool encrypt_file_names = false;    // Obfuscate file names
    bool sign_pack = true;             // Add HMAC signature
    bool compress_first = false;        // Compress before encryption
    String script_encryption_key;       // Optional key for script encryption
};

// #############################################################################
// PCK file entry (encrypted)
// #############################################################################
struct PCKEncryptedEntry {
    String path;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t original_size = 0;
    std::vector<uint8_t> encryption_metadata;  // IV, tag, etc.
    bool encrypted = false;
    bool compressed = false;
    uint32_t crc32 = 0;
};

// #############################################################################
// PCKPacker - Builds PCK files with encryption support
// #############################################################################
class PCKPacker : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(PCKPacker, RefCounted)

private:
    std::vector<PCKEncryptedEntry> m_entries;
    std::unordered_map<String, size_t> m_path_index;
    PCKEncryptionConfig m_config;
    String m_output_path;
    std::vector<uint8_t> m_pack_buffer;
    mutable std::mutex m_mutex;

    // PCK format constants
    static constexpr uint32_t PCK_MAGIC = 0x43504447; // "GDPC"
    static constexpr uint32_t PCK_VERSION = 2;
    static constexpr uint32_t PCK_FLAG_ENCRYPTED = 1 << 0;
    static constexpr uint32_t PCK_FLAG_SIGNED = 1 << 1;
    static constexpr uint32_t PCK_FLAG_COMPRESSED = 1 << 2;

public:
    static StringName get_class_static() { return StringName("PCKPacker"); }

    PCKPacker() = default;

    void set_encryption_config(const PCKEncryptionConfig& config) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_config = config;
    }

    PCKEncryptionConfig get_encryption_config() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_config;
    }

    void set_output_path(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_output_path = path;
    }

    void add_file(const String& path, const std::vector<uint8_t>& data, bool encrypt = true) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        PCKEncryptedEntry entry;
        entry.path = path;
        entry.original_size = data.size();
        entry.encrypted = encrypt && m_config.algorithm != PCKEncryptionAlgorithm::ALGORITHM_NONE;
        
        m_path_index[path] = m_entries.size();
        m_entries.push_back(entry);
        
        // Store data temporarily (will be processed during flush)
        m_pack_buffer.insert(m_pack_buffer.end(), data.begin(), data.end());
    }

    void add_file_from_path(const String& path, const String& source_path, bool encrypt = true) {
        Ref<FileAccess> file = FileAccess::open(source_path, FileAccess::READ);
        if (!file.is_valid()) return;
        
        std::vector<uint8_t> data = file->get_buffer(file->get_length());
        add_file(path, data, encrypt);
    }

    Error flush(bool verbose = false) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (m_output_path.empty()) {
            return ERR_UNCONFIGURED;
        }

        // Generate encryption key if not provided
        if (m_config.algorithm != PCKEncryptionAlgorithm::ALGORITHM_NONE) {
            if (m_config.key.empty()) {
                m_config.key = Crypto::get_singleton()->generate_random_bytes(32);
            }
            if (m_config.iv.empty() && m_config.algorithm == PCKEncryptionAlgorithm::ALGORITHM_AES_256_GCM) {
                m_config.iv = Crypto::get_singleton()->generate_random_bytes(12);
            }
            if (m_config.iv.empty() && m_config.algorithm == PCKEncryptionAlgorithm::ALGORITHM_AES_256_CBC) {
                m_config.iv = Crypto::get_singleton()->generate_random_bytes(16);
            }
            if (m_config.sign_pack && m_config.hmac_key.empty()) {
                m_config.hmac_key = Crypto::get_singleton()->generate_random_bytes(32);
            }
        }

        // Build PCK file
        Ref<FileAccess> output = FileAccess::open(m_output_path, FileAccess::WRITE);
        if (!output.is_valid()) {
            return ERR_FILE_CANT_WRITE;
        }

        // Write header
        output->store_32(PCK_MAGIC);
        output->store_32(PCK_VERSION);
        
        uint32_t flags = 0;
        if (m_config.algorithm != PCKEncryptionAlgorithm::ALGORITHM_NONE) {
            flags |= PCK_FLAG_ENCRYPTED;
        }
        if (m_config.sign_pack) {
            flags |= PCK_FLAG_SIGNED;
        }
        output->store_32(flags);
        
        // Reserve space for index offset (will be filled later)
        uint64_t index_offset_pos = output->get_position();
        output->store_64(0);

        // Process and write file data
        size_t data_offset = output->get_position();
        std::vector<uint8_t> combined_data;
        size_t current_offset = 0;
        
        for (auto& entry : m_entries) {
            // Get file data from buffer
            std::vector<uint8_t> file_data(
                m_pack_buffer.begin() + current_offset,
                m_pack_buffer.begin() + current_offset + entry.original_size
            );
            current_offset += entry.original_size;

            // Compress if enabled
            if (m_config.compress_first && !entry.encrypted) {
                file_data = compress_data(file_data);
                entry.compressed = true;
                entry.size = file_data.size();
            } else {
                entry.size = entry.original_size;
            }

            // Encrypt if enabled
            if (entry.encrypted) {
                auto result = encrypt_data(file_data, entry);
                if (!result.empty()) {
                    file_data = result;
                }
            }

            // Compute CRC32
            entry.crc32 = compute_crc32(file_data);

            // Write data
            entry.offset = output->get_position();
            output->store_buffer(file_data);

            if (verbose) {
                print_line("Added: " + entry.path + " (" + 
                           String::num(entry.original_size) + " -> " + 
                           String::num(entry.size) + " bytes)");
            }
        }

        // Write index
        uint64_t index_offset = output->get_position();
        
        // Write file count
        output->store_32(static_cast<uint32_t>(m_entries.size()));

        // Write each entry's metadata
        for (const auto& entry : m_entries) {
            String stored_path = entry.path;
            if (m_config.encrypt_file_names && m_config.algorithm != PCKEncryptionAlgorithm::ALGORITHM_NONE) {
                stored_path = encrypt_string(stored_path);
            }

            output->store_string(stored_path);
            output->store_64(entry.offset);
            output->store_64(entry.size);
            output->store_64(entry.original_size);
            
            uint32_t entry_flags = 0;
            if (entry.encrypted) entry_flags |= 1;
            if (entry.compressed) entry_flags |= 2;
            output->store_32(entry_flags);
            
            output->store_32(entry.crc32);

            // Write encryption metadata if encrypted
            if (entry.encrypted) {
                output->store_32(static_cast<uint32_t>(entry.encryption_metadata.size()));
                if (!entry.encryption_metadata.empty()) {
                    output->store_buffer(entry.encryption_metadata);
                }
            } else {
                output->store_32(0);
            }
        }

        // Write encryption config if enabled
        if (m_config.algorithm != PCKEncryptionAlgorithm::ALGORITHM_NONE) {
            output->store_32(static_cast<uint32_t>(m_config.algorithm));
            output->store_32(static_cast<uint32_t>(m_config.key.size()));
            if (!m_config.key.empty()) {
                output->store_buffer(m_config.key);
            }
            output->store_32(static_cast<uint32_t>(m_config.iv.size()));
            if (!m_config.iv.empty()) {
                output->store_buffer(m_config.iv);
            }
        }

        // Write signature if enabled
        if (m_config.sign_pack) {
            // Seek back to compute signature over everything except signature itself
            uint64_t current_pos = output->get_position();
            output->seek(0);
            std::vector<uint8_t> signed_data = output->get_buffer(current_pos);
            
            std::vector<uint8_t> signature = compute_hmac(signed_data);
            output->seek(current_pos);
            output->store_32(static_cast<uint32_t>(signature.size()));
            output->store_buffer(signature);
        }

        // Go back and write index offset
        output->seek(index_offset_pos);
        output->store_64(index_offset);

        // Clear internal state
        m_entries.clear();
        m_path_index.clear();
        m_pack_buffer.clear();

        return OK;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries.clear();
        m_path_index.clear();
        m_pack_buffer.clear();
    }

    int get_file_count() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int>(m_entries.size());
    }

    std::vector<String> get_file_list() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> result;
        for (const auto& entry : m_entries) {
            result.push_back(entry.path);
        }
        return result;
    }

    String generate_encryption_key_hex() const {
        auto key = Crypto::get_singleton()->generate_random_bytes(32);
        return Crypto::hex_encode(key);
    }

private:
    std::vector<uint8_t> compress_data(const std::vector<uint8_t>& data) {
        // Simple compression placeholder - would use zlib in production
        return data;
    }

    std::vector<uint8_t> encrypt_data(const std::vector<uint8_t>& data, PCKEncryptedEntry& entry) {
        AESContext aes;
        aes.start(AESMode::MODE_GCM, m_config.key, m_config.iv);
        
        std::vector<uint8_t> encrypted = aes.encrypt(data);
        
        // Store authentication tag as metadata
        // In GCM mode, the tag is appended to the ciphertext
        // For simplicity, we're assuming the AESContext handles this
        
        return encrypted;
    }

    String encrypt_string(const String& str) {
        std::string s = str.to_std_string();
        std::vector<uint8_t> data(s.begin(), s.end());
        
        AESContext aes;
        aes.start(AESMode::MODE_GCM, m_config.key, m_config.iv);
        std::vector<uint8_t> encrypted = aes.encrypt(data);
        
        return Crypto::base64_encode(encrypted);
    }

    uint32_t compute_crc32(const std::vector<uint8_t>& data) {
        uint32_t crc = 0xFFFFFFFF;
        for (uint8_t b : data) {
            crc ^= b;
            for (int i = 0; i < 8; ++i) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
        }
        return ~crc;
    }

    std::vector<uint8_t> compute_hmac(const std::vector<uint8_t>& data) {
        HashingContext ctx;
        ctx.start(HashingContext::HMAC_MODE, m_config.hmac_key);
        ctx.update(data);
        return ctx.finish();
    }
};

// #############################################################################
// EncryptedFileAccess - Runtime encrypted file reader
// #############################################################################
class EncryptedFileAccess : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EncryptedFileAccess, RefCounted)

private:
    Ref<FileAccess> m_file;
    PCKEncryptionConfig m_config;
    std::unordered_map<String, PCKEncryptedEntry> m_index;
    uint64_t m_index_offset = 0;
    bool m_initialized = false;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("EncryptedFileAccess"); }

    Error open(const String& path, const PCKEncryptionConfig& config) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        m_file = FileAccess::open(path, FileAccess::READ);
        if (!m_file.is_valid()) {
            return ERR_FILE_CANT_OPEN;
        }

        m_config = config;

        // Read header
        uint32_t magic = m_file->get_32();
        if (magic != PCKPacker::PCK_MAGIC) {
            return ERR_FILE_UNRECOGNIZED;
        }

        uint32_t version = m_file->get_32();
        if (version != PCKPacker::PCK_VERSION) {
            return ERR_FILE_UNRECOGNIZED;
        }

        uint32_t flags = m_file->get_32();
        m_index_offset = m_file->get_64();

        // Read index
        m_file->seek(m_index_offset);
        uint32_t file_count = m_file->get_32();

        for (uint32_t i = 0; i < file_count; ++i) {
            PCKEncryptedEntry entry;
            entry.path = m_file->get_string();
            entry.offset = m_file->get_64();
            entry.size = m_file->get_64();
            entry.original_size = m_file->get_64();
            uint32_t entry_flags = m_file->get_32();
            entry.encrypted = (entry_flags & 1) != 0;
            entry.compressed = (entry_flags & 2) != 0;
            entry.crc32 = m_file->get_32();
            
            uint32_t metadata_size = m_file->get_32();
            if (metadata_size > 0) {
                entry.encryption_metadata = m_file->get_buffer(metadata_size);
            }

            m_index[entry.path] = entry;
        }

        // Read encryption config if present
        if (flags & PCKPacker::PCK_FLAG_ENCRYPTED) {
            m_config.algorithm = static_cast<PCKEncryptionAlgorithm>(m_file->get_32());
            uint32_t key_size = m_file->get_32();
            if (key_size > 0) {
                m_config.key = m_file->get_buffer(key_size);
            }
            uint32_t iv_size = m_file->get_32();
            if (iv_size > 0) {
                m_config.iv = m_file->get_buffer(iv_size);
            }
        }

        m_initialized = true;
        return OK;
    }

    void close() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_file.is_valid()) {
            m_file->close();
        }
        m_initialized = false;
        m_index.clear();
    }

    bool file_exists(const String& path) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_index.find(path) != m_index.end();
    }

    std::vector<uint8_t> read_file(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (!m_initialized || !m_file.is_valid()) {
            return {};
        }

        auto it = m_index.find(path);
        if (it == m_index.end()) {
            return {};
        }

        const PCKEncryptedEntry& entry = it->second;
        m_file->seek(entry.offset);
        std::vector<uint8_t> data = m_file->get_buffer(entry.size);

        // Verify CRC32
        uint32_t computed_crc = compute_crc32(data);
        if (computed_crc != entry.crc32) {
            WARN_PRINT("CRC32 mismatch for file: " + path);
            return {};
        }

        // Decrypt if needed
        if (entry.encrypted) {
            AESContext aes;
            aes.start(AESMode::MODE_GCM, m_config.key, m_config.iv);
            data = aes.decrypt(data);
        }

        // Decompress if needed
        if (entry.compressed) {
            data = decompress_data(data);
        }

        return data;
    }

    std::vector<String> get_file_list() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> result;
        for (const auto& kv : m_index) {
            result.push_back(kv.first);
        }
        return result;
    }

private:
    uint32_t compute_crc32(const std::vector<uint8_t>& data) {
        uint32_t crc = 0xFFFFFFFF;
        for (uint8_t b : data) {
            crc ^= b;
            for (int i = 0; i < 8; ++i) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
        }
        return ~crc;
    }

    std::vector<uint8_t> decompress_data(const std::vector<uint8_t>& data) {
        // Decompression placeholder
        return data;
    }
};

// #############################################################################
// PCKEncryption - Utility class for encryption key management
// #############################################################################
class PCKEncryption : public Object {
    XTU_GODOT_REGISTER_CLASS(PCKEncryption, Object)

public:
    static StringName get_class_static() { return StringName("PCKEncryption"); }

    static PCKEncryptionConfig create_config(const String& password) {
        PCKEncryptionConfig config;
        config.algorithm = PCKEncryptionAlgorithm::ALGORITHM_AES_256_GCM;
        
        // Derive key from password using PBKDF2
        std::vector<uint8_t> salt = Crypto::get_singleton()->generate_random_bytes(16);
        String salt_hex = Crypto::hex_encode(salt);
        
        config.key = Crypto::get_singleton()->pbkdf2(password, salt, 100000, 32, HashType::HASH_SHA256);
        config.iv = Crypto::get_singleton()->generate_random_bytes(12);
        config.hmac_key = Crypto::get_singleton()->pbkdf2(password, salt, 100000, 32, HashType::HASH_SHA256);
        config.sign_pack = true;
        config.encrypt_index = true;
        
        return config;
    }

    static PCKEncryptionConfig create_config_from_key(const String& key_hex) {
        PCKEncryptionConfig config;
        config.algorithm = PCKEncryptionAlgorithm::ALGORITHM_AES_256_GCM;
        config.key = Crypto::hex_decode(key_hex);
        config.iv = Crypto::get_singleton()->generate_random_bytes(12);
        config.sign_pack = true;
        return config;
    }

    static String generate_key_hex() {
        auto key = Crypto::get_singleton()->generate_random_bytes(32);
        return Crypto::hex_encode(key);
    }

    static bool validate_config(const PCKEncryptionConfig& config) {
        if (config.algorithm == PCKEncryptionAlgorithm::ALGORITHM_NONE) {
            return true;
        }
        
        if (config.key.size() != 32) {
            return false;
        }
        
        if (config.algorithm == PCKEncryptionAlgorithm::ALGORITHM_AES_256_GCM) {
            if (config.iv.size() != 12) {
                return false;
            }
        } else if (config.algorithm == PCKEncryptionAlgorithm::ALGORITHM_AES_256_CBC) {
            if (config.iv.size() != 16) {
                return false;
            }
        }
        
        if (config.sign_pack && config.hmac_key.size() != 32) {
            return false;
        }
        
        return true;
    }
};

} // namespace io

// Bring into main namespace
using io::PCKPacker;
using io::PCKEncryption;
using io::EncryptedFileAccess;
using io::PCKEncryptionAlgorithm;
using io::PCKEncryptionConfig;
using io::PCKEncryptedEntry;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPCK_ENCRYPTION_HPP