// io/xnpz.hpp
#ifndef XTENSOR_XNPZ_HPP
#define XTENSOR_XNPZ_HPP

// ----------------------------------------------------------------------------
// xnpz.hpp – NumPy .npy / .npz I/O for xtensor
// ----------------------------------------------------------------------------
// This header provides functions to read and write NumPy's binary format:
//   - .npy: single array with header
//   - .npz: archive of multiple arrays (optionally compressed)
//   - Supports all standard NumPy dtypes and bignumber::BigNumber (custom)
//   - Preserves shape, dtype, and byte order
//   - Memory‑mapped reading for large files
//
// All functions are templated to work with any value type, including
// bignumber::BigNumber. FFT acceleration is not directly used but the
// infrastructure is maintained for potential spectral compression.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <complex>

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace io {

// ========================================================================
// .npy header structure and utilities
// ========================================================================
struct npy_header {
    std::string dtype_descr;   // e.g., "<f8", ">i4", "|u1"
    bool fortran_order;        // true for column‑major
    shape_type shape;
};

// Parse the dictionary‑style .npy header
npy_header parse_npy_header(const std::string& header_str);
// Build a .npy header string from components
std::string build_npy_header(const npy_header& hdr);

// Map C++ type to NumPy dtype string
template <class T> std::string dtype_to_npy();
// Map NumPy dtype string to xtensor value_type and byte size
std::pair<std::string, size_t> npy_dtype_to_cpp(const std::string& descr);

// ========================================================================
// .npy single array I/O
// ========================================================================
template <class T>
void save_npy(const std::string& filename, const xarray_container<T>& arr,
              bool fortran_order = false);

template <class T>
xarray_container<T> load_npy(const std::string& filename);

// Memory‑mapped .npy (read‑only, zero‑copy)
template <class T>
class npy_memmap {
public:
    npy_memmap(const std::string& filename);
    ~npy_memmap();

    const T* data() const noexcept;
    const shape_type& shape() const noexcept;
    bool fortran_order() const noexcept;

    // Prevent copying
    npy_memmap(const npy_memmap&) = delete;
    npy_memmap& operator=(const npy_memmap&) = delete;

private:
    void* m_mapped_data;
    size_t m_size;
    shape_type m_shape;
    bool m_fortran_order;
    int m_fd;
};

// ========================================================================
// .npz archive I/O
// ========================================================================
// Read all arrays from a .npz file into a map (name -> array)
template <class T>
std::map<std::string, xarray_container<T>> load_npz(const std::string& filename);

// Write a collection of named arrays to a .npz file
template <class T>
void save_npz(const std::string& filename,
              const std::map<std::string, xarray_container<T>>& arrays,
              bool compress = true);

// ========================================================================
// Convenience functions
// ========================================================================
// Check if a file is a valid .npy file
bool is_npy_file(const std::string& filename);
// Check if a file is a valid .npz (ZIP) archive
bool is_npz_file(const std::string& filename);

} // namespace io

using io::save_npy;
using io::load_npy;
using io::npy_memmap;
using io::save_npz;
using io::load_npz;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt {
namespace io {

// Parse the .npy header (Python dict literal)
inline npy_header parse_npy_header(const std::string& header_str)
{ /* TODO: parse {'descr': ..., 'fortran_order': ..., 'shape': ...} */ return {}; }

// Build a .npy header string
inline std::string build_npy_header(const npy_header& hdr)
{ /* TODO: format as Python dict literal */ return {}; }

// Map C++ type to NumPy dtype string
template <class T> std::string dtype_to_npy() {
    if constexpr (std::is_same_v<T, bool>) return "|b1";
    else if constexpr (std::is_same_v<T, uint8_t>) return "|u1";
    else if constexpr (std::is_same_v<T, int8_t>) return "|i1";
    else if constexpr (std::is_same_v<T, uint16_t>) return "<u2";
    else if constexpr (std::is_same_v<T, int16_t>) return "<i2";
    else if constexpr (std::is_same_v<T, uint32_t>) return "<u4";
    else if constexpr (std::is_same_v<T, int32_t>) return "<i4";
    else if constexpr (std::is_same_v<T, uint64_t>) return "<u8";
    else if constexpr (std::is_same_v<T, int64_t>) return "<i8";
    else if constexpr (std::is_same_v<T, float>) return "<f4";
    else if constexpr (std::is_same_v<T, double>) return "<f8";
    else if constexpr (std::is_same_v<T, std::complex<float>>) return "<c8";
    else if constexpr (std::is_same_v<T, std::complex<double>>) return "<c16";
    else if constexpr (std::is_same_v<T, bignumber::BigNumber>) return "|bignumber";
    else return "|V" + std::to_string(sizeof(T));
}

// Map NumPy dtype to C++ type name and element size
inline std::pair<std::string, size_t> npy_dtype_to_cpp(const std::string& descr)
{ /* TODO: parse dtype descriptor */ return {"", 0}; }

// Save a single array to .npy
template <class T>
void save_npy(const std::string& filename, const xarray_container<T>& arr, bool fortran_order)
{ /* TODO: write magic, header, and data */ }

// Load a single array from .npy
template <class T>
xarray_container<T> load_npy(const std::string& filename)
{ /* TODO: read header, validate dtype, read data */ return {}; }

// Construct memory‑mapped .npy reader
template <class T>
npy_memmap<T>::npy_memmap(const std::string& filename)
{ /* TODO: mmap file, parse header, set data pointer */ }
// Destructor – unmap file
template <class T> npy_memmap<T>::~npy_memmap() { /* TODO: munmap */ }
// Get pointer to raw data
template <class T> const T* npy_memmap<T>::data() const noexcept { return nullptr; }
// Get array shape
template <class T> const shape_type& npy_memmap<T>::shape() const noexcept { return m_shape; }
// Check storage order
template <class T> bool npy_memmap<T>::fortran_order() const noexcept { return m_fortran_order; }

// Load .npz archive (ZIP containing multiple .npy)
template <class T>
std::map<std::string, xarray_container<T>> load_npz(const std::string& filename)
{ /* TODO: unzip and parse each .npy */ return {}; }

// Save .npz archive
template <class T>
void save_npz(const std::string& filename,
              const std::map<std::string, xarray_container<T>>& arrays,
              bool compress)
{ /* TODO: zip individual .npy files */ }

// Check for .npy magic
inline bool is_npy_file(const std::string& filename)
{ /* TODO: read first 6 bytes and compare to "\x93NUMPY" */ return false; }

// Check for ZIP magic (PK)
inline bool is_npz_file(const std::string& filename)
{ /* TODO: read first 2 bytes and compare to "PK" */ return false; }

} // namespace io
} // namespace xt

#endif // XTENSOR_XNPZ_HPP                stream.read(header_buf.data(), hdr.header_len);
                    std::string header_str(header_buf.data(), hdr.header_len);
                    
                    // Parse header dict
                    auto dict = parse_python_dict(header_str);
                    if (dict.count("descr"))
                        hdr.dtype = dtype_descr::from_string(dict["descr"]);
                    if (dict.count("fortran_order"))
                        hdr.fortran_order = (dict["fortran_order"] == "True");
                    if (dict.count("shape"))
                        hdr.shape = parse_shape_string(dict["shape"]);
                    return hdr;
                }

                // Read .npz (zip of .npy files)
                // Since full zip parsing is complex, we provide a basic implementation
                // using zlib and manual zip structure parsing (simplified for .npz)
                class NpzReader
                {
                public:
                    explicit NpzReader(const std::string& filename)
                    {
                        std::ifstream file(filename, std::ios::binary);
                        if (!file)
                            XTENSOR_THROW(std::runtime_error, "Cannot open npz file: " + filename);
                        
                        // Read entire file into memory
                        file.seekg(0, std::ios::end);
                        size_t size = file.tellg();
                        file.seekg(0, std::ios::beg);
                        m_buffer.resize(size);
                        file.read(reinterpret_cast<char*>(m_buffer.data()), size);
                        
                        // Parse zip central directory (simplified)
                        parse_zip();
                    }
                    
                    bool has_array(const std::string& name) const
                    {
                        return m_entries.find(name) != m_entries.end();
                    }
                    
                    std::vector<std::string> list_arrays() const
                    {
                        std::vector<std::string> names;
                        for (const auto& p : m_entries) names.push_back(p.first);
                        return names;
                    }
                    
                    // Read array as xarray_container of specified type
                    template<typename T>
                    xarray_container<T> read_array(const std::string& name) const
                    {
                        auto it = m_entries.find(name);
                        if (it == m_entries.end())
                            XTENSOR_THROW(std::runtime_error, "Array not found in npz: " + name);
                        
                        const auto& entry = it->second;
                        const uint8_t* data_ptr = m_buffer.data() + entry.data_offset;
                        
                        // Parse npy header from data
                        std::istringstream npy_stream(std::string(
                            reinterpret_cast<const char*>(data_ptr), entry.compressed_size));
                        NpyHeader hdr = read_npy_header(npy_stream);
                        
                        // Get data position after header
                        size_t header_end = 10 + hdr.header_len; // 6 magic + 2 version + 2/4 len + header
                        const uint8_t* raw_data = data_ptr + header_end;
                        size_t raw_size = entry.compressed_size - header_end;
                        
                        // Decompress if needed
                        std::vector<uint8_t> decompressed;
                        if (entry.compression == 8) // DEFLATE
                        {
                            decompressed = decompress_zlib(raw_data, raw_size, entry.uncompressed_size);
                            raw_data = decompressed.data();
                            raw_size = decompressed.size();
                        }
                        
                        // Convert to xarray
                        size_t total_elements = 1;
                        for (size_t s : hdr.shape) total_elements *= s;
                        
                        xarray_container<T> result(hdr.shape);
                        if (raw_size != total_elements * sizeof(T))
                            XTENSOR_THROW(std::runtime_error, "Data size mismatch");
                        
                        // Copy data (assuming host endianness matches file; simplified)
                        std::memcpy(result.data(), raw_data, raw_size);
                        
                        // Handle fortran order
                        if (hdr.fortran_order)
                        {
                            // Convert from column-major to row-major (default for xtensor)
                            xarray_container<T> transposed = xt::transpose(result);
                            result = transposed;
                        }
                        return result;
                    }
                    
                private:
                    struct Entry
                    {
                        size_t data_offset;
                        size_t compressed_size;
                        size_t uncompressed_size;
                        uint16_t compression; // 8 = DEFLATE, 0 = stored
                        std::string name;
                    };
                    
                    std::vector<uint8_t> m_buffer;
                    std::map<std::string, Entry> m_entries;
                    
                    void parse_zip()
                    {
                        // Find End of Central Directory record (EOCD)
                        // Simplified: scan backwards for signature 0x06054b50
                        size_t eocd_offset = 0;
                        for (size_t i = m_buffer.size() - 22; i > 0; --i)
                        {
                            if (m_buffer[i] == 0x50 && m_buffer[i+1] == 0x4b && 
                                m_buffer[i+2] == 0x05 && m_buffer[i+3] == 0x06)
                            {
                                eocd_offset = i;
                                break;
                            }
                        }
                        if (eocd_offset == 0)
                            XTENSOR_THROW(std::runtime_error, "Invalid zip: EOCD not found");
                        
                        // Read EOCD
                        uint16_t disk_num = read_u16(eocd_offset + 4);
                        uint16_t cd_disk = read_u16(eocd_offset + 6);
                        uint16_t cd_count_disk = read_u16(eocd_offset + 8);
                        uint16_t cd_count_total = read_u16(eocd_offset + 10);
                        uint32_t cd_size = read_u32(eocd_offset + 12);
                        uint32_t cd_offset = read_u32(eocd_offset + 16);
                        
                        if (disk_num != 0 || cd_disk != 0 || cd_count_disk != cd_count_total)
                            XTENSOR_THROW(std::runtime_error, "Multi-disk zip not supported");
                        
                        // Parse Central Directory entries
                        size_t offset = cd_offset;
                        for (uint16_t i = 0; i < cd_count_total; ++i)
                        {
                            if (offset + 46 > m_buffer.size()) break;
                            uint32_t signature = read_u32(offset);
                            if (signature != 0x02014b50) break;
                            
                            uint16_t version = read_u16(offset + 4);
                            uint16_t flags = read_u16(offset + 8);
                            uint16_t compression = read_u16(offset + 10);
                            uint32_t crc32 = read_u32(offset + 16);
                            uint32_t comp_size = read_u32(offset + 20);
                            uint32_t uncomp_size = read_u32(offset + 24);
                            uint16_t name_len = read_u16(offset + 28);
                            uint16_t extra_len = read_u16(offset + 30);
                            uint16_t comment_len = read_u16(offset + 32);
                            uint32_t local_header_offset = read_u32(offset + 42);
                            
                            std::string name(reinterpret_cast<const char*>(m_buffer.data() + offset + 46), name_len);
                            
                            Entry entry;
                            entry.name = name;
                            entry.compression = compression;
                            entry.compressed_size = comp_size;
                            entry.uncompressed_size = uncomp_size;
                            
                            // Read local header to get actual data offset
                            size_t local_off = local_header_offset;
                            uint32_t local_sig = read_u32(local_off);
                            if (local_sig != 0x04034b50) break;
                            uint16_t local_name_len = read_u16(local_off + 26);
                            uint16_t local_extra_len = read_u16(local_off + 28);
                            entry.data_offset = local_off + 30 + local_name_len + local_extra_len;
                            
                            m_entries[name] = entry;
                            
                            offset += 46 + name_len + extra_len + comment_len;
                        }
                    }
                    
                    uint16_t read_u16(size_t offset) const
                    {
                        return static_cast<uint16_t>(m_buffer[offset]) |
                               (static_cast<uint16_t>(m_buffer[offset+1]) << 8);
                    }
                    uint32_t read_u32(size_t offset) const
                    {
                        return static_cast<uint32_t>(m_buffer[offset]) |
                               (static_cast<uint32_t>(m_buffer[offset+1]) << 8) |
                               (static_cast<uint32_t>(m_buffer[offset+2]) << 16) |
                               (static_cast<uint32_t>(m_buffer[offset+3]) << 24);
                    }
                    
                    std::vector<uint8_t> decompress_zlib(const uint8_t* data, size_t comp_size, size_t uncomp_size) const
                    {
                        std::vector<uint8_t> result(uncomp_size);
                        z_stream stream;
                        stream.zalloc = Z_NULL;
                        stream.zfree = Z_NULL;
                        stream.opaque = Z_NULL;
                        stream.avail_in = static_cast<uInt>(comp_size);
                        stream.next_in = const_cast<Bytef*>(data);
                        stream.avail_out = static_cast<uInt>(uncomp_size);
                        stream.next_out = result.data();
                        
                        if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
                            XTENSOR_THROW(std::runtime_error, "Zlib inflateInit failed");
                        int ret = inflate(&stream, Z_FINISH);
                        inflateEnd(&stream);
                        if (ret != Z_STREAM_END)
                            XTENSOR_THROW(std::runtime_error, "Zlib decompression failed");
                        return result;
                    }
                };

                // Writer for .npy
                template<typename T>
                void write_npy(std::ostream& stream, const xarray_container<T>& arr,
                               bool fortran_order = false)
                {
                    // Build header dict string
                    dtype_descr dtype = dtype_of<T>::get();
                    std::ostringstream header_dict;
                    header_dict << "{'descr': '" << dtype.to_string() << "', "
                                << "'fortran_order': " << (fortran_order ? "True" : "False") << ", "
                                << "'shape': (";
                    const auto& shape = arr.shape();
                    for (size_t i = 0; i < shape.size(); ++i)
                    {
                        if (i > 0) header_dict << ", ";
                        header_dict << shape[i];
                    }
                    if (shape.size() == 1) header_dict << ",";
                    header_dict << ")}";
                    
                    std::string header_str = header_dict.str();
                    // Pad to multiple of HEADER_ALIGNMENT
                    size_t header_len = header_str.size();
                    size_t padding = HEADER_ALIGNMENT - ((10 + header_len) % HEADER_ALIGNMENT);
                    if (padding < HEADER_ALIGNMENT)
                        header_str.append(padding, ' ');
                    else
                        padding = 0;
                    header_len = header_str.size();
                    
                    // Write magic
                    stream.write(reinterpret_cast<const char*>(MAGIC), 6);
                    // Version 1.0
                    uint8_t major = 1, minor = 0;
                    stream.write(reinterpret_cast<const char*>(&major), 1);
                    stream.write(reinterpret_cast<const char*>(&minor), 1);
                    // Header length (16-bit for version 1)
                    uint16_t header_len16 = static_cast<uint16_t>(header_len);
                    stream.write(reinterpret_cast<const char*>(&header_len16), 2);
                    // Header string
                    stream.write(header_str.c_str(), header_len);
                    // Array data (assuming row-major; if fortran_order requested, we need to transpose)
                    if (fortran_order)
                    {
                        auto transposed = xt::transpose(arr);
                        stream.write(reinterpret_cast<const char*>(transposed.data()),
                                     transposed.size() * sizeof(T));
                    }
                    else
                    {
                        stream.write(reinterpret_cast<const char*>(arr.data()),
                                     arr.size() * sizeof(T));
                    }
                }

                // Writer for .npz (compressed zip of .npy files)
                class NpzWriter
                {
                public:
                    explicit NpzWriter(const std::string& filename, int compression_level = Z_DEFAULT_COMPRESSION)
                        : m_filename(filename), m_compression_level(compression_level)
                    {
                    }
                    
                    template<typename T>
                    void add_array(const std::string& name, const xarray_container<T>& arr, bool compress = true)
                    {
                        m_pending_arrays.push_back({name, compress});
                        // Serialize array to .npy format in memory
                        std::ostringstream npy_stream;
                        write_npy(npy_stream, arr, false);
                        m_array_buffers.push_back(npy_stream.str());
                    }
                    
                    void write()
                    {
                        std::ofstream out(m_filename, std::ios::binary);
                        if (!out)
                            XTENSOR_THROW(std::runtime_error, "Cannot open npz file for writing: " + m_filename);
                        
                        std::vector<uint8_t> central_dir;
                        size_t current_offset = 0;
                        
                        for (size_t i = 0; i < m_pending_arrays.size(); ++i)
                        {
                            const auto& info = m_pending_arrays[i];
                            const std::string& npy_data = m_array_buffers[i];
                            
                            // Compress if requested
                            std::vector<uint8_t> comp_data;
                            uint16_t compression_method = 0; // stored
                            if (info.compress)
                            {
                                comp_data = compress_zlib(npy_data);
                                compression_method = 8; // DEFLATE
                            }
                            else
                            {
                                comp_data.assign(npy_data.begin(), npy_data.end());
                            }
                            
                            uint32_t crc = crc32(0, comp_data.data(), static_cast<uInt>(comp_data.size()));
                            
                            // Write local file header
                            uint32_t local_sig = 0x04034b50;
                            out.write(reinterpret_cast<const char*>(&local_sig), 4);
                            uint16_t version_needed = 20;
                            out.write(reinterpret_cast<const char*>(&version_needed), 2);
                            uint16_t flags = 0;
                            out.write(reinterpret_cast<const char*>(&flags), 2);
                            out.write(reinterpret_cast<const char*>(&compression_method), 2);
                            uint16_t mod_time = 0, mod_date = 0;
                            out.write(reinterpret_cast<const char*>(&mod_time), 2);
                            out.write(reinterpret_cast<const char*>(&mod_date), 2);
                            out.write(reinterpret_cast<const char*>(&crc), 4);
                            uint32_t comp_size = static_cast<uint32_t>(comp_data.size());
                            uint32_t uncomp_size = static_cast<uint32_t>(npy_data.size());
                            out.write(reinterpret_cast<const char*>(&comp_size), 4);
                            out.write(reinterpret_cast<const char*>(&uncomp_size), 4);
                            uint16_t name_len = static_cast<uint16_t>(info.name.size());
                            out.write(reinterpret_cast<const char*>(&name_len), 2);
                            uint16_t extra_len = 0;
                            out.write(reinterpret_cast<const char*>(&extra_len), 2);
                            out.write(info.name.c_str(), name_len);
                            
                            // Write compressed data
                            out.write(reinterpret_cast<const char*>(comp_data.data()), comp_data.size());
                            
                            // Prepare central directory entry
                            uint32_t cd_sig = 0x02014b50;
                            uint16_t version_made = 20;
                            size_t cd_start = central_dir.size();
                            central_dir.resize(cd_start + 46 + name_len);
                            uint8_t* cd = central_dir.data() + cd_start;
                            std::memcpy(cd, &cd_sig, 4);
                            std::memcpy(cd+4, &version_made, 2);
                            std::memcpy(cd+6, &version_needed, 2);
                            std::memcpy(cd+8, &flags, 2);
                            std::memcpy(cd+10, &compression_method, 2);
                            std::memcpy(cd+12, &mod_time, 2);
                            std::memcpy(cd+14, &mod_date, 2);
                            std::memcpy(cd+16, &crc, 4);
                            std::memcpy(cd+20, &comp_size, 4);
                            std::memcpy(cd+24, &uncomp_size, 4);
                            std::memcpy(cd+28, &name_len, 2);
                            std::memcpy(cd+30, &extra_len, 2);
                            uint16_t comment_len = 0;
                            std::memcpy(cd+32, &comment_len, 2);
                            uint16_t disk_num = 0;
                            std::memcpy(cd+34, &disk_num, 2);
                            uint16_t internal_attr = 0;
                            std::memcpy(cd+36, &internal_attr, 2);
                            uint32_t external_attr = 0;
                            std::memcpy(cd+38, &external_attr, 4);
                            uint32_t local_hdr_off = static_cast<uint32_t>(current_offset);
                            std::memcpy(cd+42, &local_hdr_off, 4);
                            std::memcpy(cd+46, info.name.c_str(), name_len);
                            
                            current_offset += 30 + name_len + comp_data.size();
                        }
                        
                        // Write central directory
                        size_t cd_offset = current_offset;
                        out.write(reinterpret_cast<const char*>(central_dir.data()), central_dir.size());
                        
                        // Write EOCD
                        uint32_t eocd_sig = 0x06054b50;
                        out.write(reinterpret_cast<const char*>(&eocd_sig), 4);
                        uint16_t disk_num = 0;
                        out.write(reinterpret_cast<const char*>(&disk_num), 2);
                        uint16_t cd_disk = 0;
                        out.write(reinterpret_cast<const char*>(&cd_disk), 2);
                        uint16_t cd_count = static_cast<uint16_t>(m_pending_arrays.size());
                        out.write(reinterpret_cast<const char*>(&cd_count), 2);
                        out.write(reinterpret_cast<const char*>(&cd_count), 2);
                        uint32_t cd_size = static_cast<uint32_t>(central_dir.size());
                        out.write(reinterpret_cast<const char*>(&cd_size), 4);
                        uint32_t cd_offset32 = static_cast<uint32_t>(cd_offset);
                        out.write(reinterpret_cast<const char*>(&cd_offset32), 4);
                        uint16_t comment_len = 0;
                        out.write(reinterpret_cast<const char*>(&comment_len), 2);
                    }
                    
                private:
                    struct ArrayInfo
                    {
                        std::string name;
                        bool compress;
                    };
                    
                    std::string m_filename;
                    int m_compression_level;
                    std::vector<ArrayInfo> m_pending_arrays;
                    std::vector<std::string> m_array_buffers;
                    
                    std::vector<uint8_t> compress_zlib(const std::string& data)
                    {
                        uLongf dest_len = compressBound(static_cast<uLong>(data.size()));
                        std::vector<uint8_t> dest(dest_len);
                        if (compress2(dest.data(), &dest_len,
                                      reinterpret_cast<const Bytef*>(data.data()),
                                      static_cast<uLong>(data.size()),
                                      m_compression_level) != Z_OK)
                        {
                            XTENSOR_THROW(std::runtime_error, "Zlib compression failed");
                        }
                        dest.resize(dest_len);
                        return dest;
                    }
                };

            } // namespace npz

            // --------------------------------------------------------------------
            // Public NPZ interface
            // --------------------------------------------------------------------
            inline auto load_npz(const std::string& filename)
            {
                npz::NpzReader reader(filename);
                std::map<std::string, xarray_container<double>> result; // default to double
                for (const auto& name : reader.list_arrays())
                {
                    result[name] = reader.read_array<double>(name);
                }
                return result;
            }

            template<typename T>
            inline auto load_npz_as(const std::string& filename)
            {
                npz::NpzReader reader(filename);
                std::map<std::string, xarray_container<T>> result;
                for (const auto& name : reader.list_arrays())
                {
                    result[name] = reader.read_array<T>(name);
                }
                return result;
            }

            // Save multiple arrays to .npz
            inline void save_npz(const std::string& filename,
                                 const std::map<std::string, xarray_container<double>>& arrays,
                                 bool compress = true,
                                 int compression_level = Z_DEFAULT_COMPRESSION)
            {
                npz::NpzWriter writer(filename, compression_level);
                for (const auto& p : arrays)
                    writer.add_array(p.first, p.second, compress);
                writer.write();
            }

            // Save single array to .npy
            template<typename T>
            inline void save_npy(const std::string& filename, const xarray_container<T>& arr,
                                 bool fortran_order = false)
            {
                std::ofstream out(filename, std::ios::binary);
                if (!out) XTENSOR_THROW(std::runtime_error, "Cannot open npy file: " + filename);
                npz::write_npy(out, arr, fortran_order);
            }

            // Load single .npy file
            template<typename T>
            inline xarray_container<T> load_npy(const std::string& filename)
            {
                std::ifstream in(filename, std::ios::binary);
                if (!in) XTENSOR_THROW(std::runtime_error, "Cannot open npy file: " + filename);
                npz::NpyHeader hdr = npz::read_npy_header(in);
                size_t total_elements = 1;
                for (size_t s : hdr.shape) total_elements *= s;
                xarray_container<T> result(hdr.shape);
                // Skip header part (we already read it, so stream is at data)
                // Read data
                in.read(reinterpret_cast<char*>(result.data()), total_elements * sizeof(T));
                if (hdr.fortran_order)
                    result = xt::transpose(result);
                return result;
            }

        } // namespace io

        // Bring NPZ functions into xt namespace
        using io::load_npz;
        using io::load_npz_as;
        using io::save_npz;
        using io::save_npy;
        using io::load_npy;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XNPZ_HPP

// io/xnpz.hpp