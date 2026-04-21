// io/xio_avro.hpp
#ifndef XTENSOR_XIO_AVRO_HPP
#define XTENSOR_XIO_AVRO_HPP

// ----------------------------------------------------------------------------
// xio_avro.hpp – Apache Avro I/O for xtensor
// ----------------------------------------------------------------------------
// This header provides row‑oriented serialization using Apache Avro:
//   - Read/Write Avro Object Container Files
//   - Schema evolution (reader/writer schema resolution)
//   - Support for Avro primitive and complex types (records, arrays, maps, unions)
//   - Compression: NULL, DEFLATE, SNAPPY, ZSTD
//   - Code generation from Avro schema (JSON) or dynamic schema building
//   - Binary and JSON encoding
//   - Logical types: decimal, uuid, date, time‑millis, timestamp‑millis
//   - Projection and predicate pushdown (column selection)
//
// All numeric types are supported, including bignumber::BigNumber (stored as
// bytes or fixed decimal logical type). FFT acceleration is not directly used.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>
#include <functional>
#include <type_traits>
#include <fstream>

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace io {
namespace avro {

// ========================================================================
// Avro Schema
// ========================================================================
enum class avro_type {
    NULL_TYPE, BOOLEAN, INT, LONG, FLOAT, DOUBLE, BYTES, STRING,
    RECORD, ENUM, ARRAY, MAP, UNION, FIXED
};

struct avro_schema_node {
    avro_type type;
    std::string name;
    std::string doc;
    std::vector<std::string> aliases;
    std::optional<std::string> logical_type;
    int precision = 0;  // for decimal
    int scale = 0;      // for decimal
    size_t size = 0;    // for fixed

    // For RECORD
    std::vector<std::pair<std::string, avro_schema_node>> fields;

    // For ENUM
    std::vector<std::string> symbols;

    // For ARRAY
    avro_schema_node items;

    // For MAP
    avro_schema_node values;

    // For UNION
    std::vector<avro_schema_node> branches;

    // Parse from JSON string
    static avro_schema_node from_json(const std::string& json);
    std::string to_json(bool pretty = false) const;
};

// ========================================================================
// Generic Avro Value (for dynamic data)
// ========================================================================
class avro_value {
public:
    avro_value() = default;
    explicit avro_value(std::nullptr_t);
    explicit avro_value(bool b);
    explicit avro_value(int32_t i);
    explicit avro_value(int64_t l);
    explicit avro_value(float f);
    explicit avro_value(double d);
    explicit avro_value(const std::string& s);
    explicit avro_value(const std::vector<uint8_t>& bytes);

    avro_type type() const noexcept;
    bool is_null() const noexcept;
    bool as_bool() const;
    int32_t as_int32() const;
    int64_t as_int64() const;
    float as_float() const;
    double as_double() const;
    const std::string& as_string() const;
    const std::vector<uint8_t>& as_bytes() const;

    // Array operations
    size_t array_size() const;
    void append(const avro_value& val);
    const avro_value& operator[](size_t idx) const;

    // Map operations
    size_t map_size() const;
    bool contains(const std::string& key) const;
    const avro_value& operator[](const std::string& key) const;
    void insert(const std::string& key, const avro_value& val);

    // Record operations
    const avro_value& field(const std::string& name) const;
    void set_field(const std::string& name, const avro_value& val);
    std::vector<std::string> field_names() const;

    // Union operations
    size_t union_branch() const noexcept;
    const avro_value& union_value() const;
};

// ========================================================================
// Writer
// ========================================================================
struct avro_writer_options {
    std::string codec = "null";  // "null", "deflate", "snappy", "zstd"
    int deflate_level = 6;
    size_t block_size = 64 * 1024;  // objects per block
    bool write_header = true;
    std::map<std::string, std::string> metadata;
};

class avro_writer {
public:
    avro_writer();
    ~avro_writer();

    void open(const std::string& filename, const avro_schema_node& schema,
              const avro_writer_options& opts = {});
    void close();

    // Write a single record (as avro_value)
    void write(const avro_value& record);

    // Write batch of records (from xtensor array, assuming tabular data)
    template <class T>
    void write_batch(const xarray_container<T>& data,
                     const std::vector<std::string>& column_names);

    // Schema access
    const avro_schema_node& schema() const;

    // Flush current block
    void flush();

    // Number of records written
    size_t record_count() const noexcept;

private:
    void* m_writer;
    avro_schema_node m_schema;
    size_t m_count;
};

// ========================================================================
// Reader
// ========================================================================
struct avro_reader_options {
    avro_schema_node reader_schema;  // optional, for schema evolution
    std::vector<std::string> columns;  // projection (empty = all)
    size_t max_records = 0;  // 0 = unlimited
    bool use_threads = false;
    int num_threads = 1;
};

class avro_reader {
public:
    avro_reader();
    ~avro_reader();

    void open(const std::string& filename, const avro_reader_options& opts = {});
    void close();

    // Read next record, returns false at EOF
    bool read(avro_value& record);

    // Read all records into vector
    std::vector<avro_value> read_all();

    // Read into xtensor array (tabular data)
    template <class T>
    xarray_container<T> read_to_array(const std::vector<std::string>& column_names = {});

    // Schema access
    const avro_schema_node& writer_schema() const;
    const avro_schema_node& effective_schema() const;

    // Metadata
    std::map<std::string, std::string> metadata() const;
    size_t total_records() const noexcept;
    size_t current_record() const noexcept;

    // Seek to record (if file supports)
    void seek(size_t record_index);

private:
    void* m_reader;
    avro_schema_node m_writer_schema;
    avro_schema_node m_effective_schema;
    size_t m_total;
    size_t m_current;
};

// ========================================================================
// Schema Builder (helper)
// ========================================================================
class schema_builder {
public:
    schema_builder& set_name(const std::string& name);
    schema_builder& set_namespace(const std::string& ns);
    schema_builder& set_doc(const std::string& doc);

    schema_builder& add_field(const std::string& name, avro_type type);
    schema_builder& add_nullable_field(const std::string& name, avro_type type);
    schema_builder& add_string_field(const std::string& name);
    schema_builder& add_int_field(const std::string& name);
    schema_builder& add_long_field(const std::string& name);
    schema_builder& add_float_field(const std::string& name);
    schema_builder& add_double_field(const std::string& name);
    schema_builder& add_boolean_field(const std::string& name);
    schema_builder& add_bytes_field(const std::string& name);
    schema_builder& add_decimal_field(const std::string& name, int precision, int scale);
    schema_builder& add_bignumber_field(const std::string& name);
    schema_builder& add_array_field(const std::string& name, avro_type items);
    schema_builder& add_map_field(const std::string& name, avro_type values);

    avro_schema_node build() const;

private:
    avro_schema_node m_record;
};

// ========================================================================
// Convenience Functions
// ========================================================================
template <class T>
void save_avro(const std::string& filename, const xarray_container<T>& data,
               const avro_schema_node& schema = {},
               const avro_writer_options& opts = {});

template <class T>
xarray_container<T> load_avro(const std::string& filename,
                              const avro_reader_options& opts = {});

// Check if file is a valid Avro Object Container File
bool is_avro_file(const std::string& filename);

// Convert xtensor array to Avro record batch
avro_value array_to_avro(const xarray_container<double>& data,
                         const std::vector<std::string>& column_names);

// Convert Avro record batch to xtensor array
xarray_container<double> avro_to_array(const std::vector<avro_value>& records,
                                       const std::vector<std::string>& column_names);

} // namespace avro

using avro::avro_writer;
using avro::avro_reader;
using avro::save_avro;
using avro::load_avro;
using avro::is_avro_file;
using avro::avro_schema_node;
using avro::avro_value;
using avro::schema_builder;

} // namespace io
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt {
namespace io {
namespace avro {

// avro_schema_node
inline avro_schema_node avro_schema_node::from_json(const std::string& json)
{ /* TODO: parse JSON schema */ return {}; }
inline std::string avro_schema_node::to_json(bool pretty) const
{ /* TODO: serialize to JSON */ return ""; }

// avro_value
inline avro_value::avro_value(std::nullptr_t) { /* TODO: set null */ }
inline avro_value::avro_value(bool b) { /* TODO: set boolean */ }
inline avro_value::avro_value(int32_t i) { /* TODO: set int */ }
inline avro_value::avro_value(int64_t l) { /* TODO: set long */ }
inline avro_value::avro_value(float f) { /* TODO: set float */ }
inline avro_value::avro_value(double d) { /* TODO: set double */ }
inline avro_value::avro_value(const std::string& s) { /* TODO: set string */ }
inline avro_value::avro_value(const std::vector<uint8_t>& bytes) { /* TODO: set bytes */ }
inline avro_type avro_value::type() const noexcept { return avro_type::NULL_TYPE; }
inline bool avro_value::is_null() const noexcept { return type() == avro_type::NULL_TYPE; }
inline bool avro_value::as_bool() const { return false; }
inline int32_t avro_value::as_int32() const { return 0; }
inline int64_t avro_value::as_int64() const { return 0; }
inline float avro_value::as_float() const { return 0.0f; }
inline double avro_value::as_double() const { return 0.0; }
inline const std::string& avro_value::as_string() const { static std::string empty; return empty; }
inline const std::vector<uint8_t>& avro_value::as_bytes() const { static std::vector<uint8_t> empty; return empty; }
inline size_t avro_value::array_size() const { return 0; }
inline void avro_value::append(const avro_value& val) { /* TODO: append */ }
inline const avro_value& avro_value::operator[](size_t idx) const { return *this; }
inline size_t avro_value::map_size() const { return 0; }
inline bool avro_value::contains(const std::string& key) const { return false; }
inline const avro_value& avro_value::operator[](const std::string& key) const { return *this; }
inline void avro_value::insert(const std::string& key, const avro_value& val) { /* TODO: insert */ }
inline const avro_value& avro_value::field(const std::string& name) const { return *this; }
inline void avro_value::set_field(const std::string& name, const avro_value& val) { /* TODO: set */ }
inline std::vector<std::string> avro_value::field_names() const { return {}; }
inline size_t avro_value::union_branch() const noexcept { return 0; }
inline const avro_value& avro_value::union_value() const { return *this; }

// avro_writer
inline avro_writer::avro_writer() : m_writer(nullptr), m_count(0) {}
inline avro_writer::~avro_writer() { close(); }
inline void avro_writer::open(const std::string& filename, const avro_schema_node& schema, const avro_writer_options& opts)
{ /* TODO: create file and write header */ m_schema = schema; }
inline void avro_writer::close() { /* TODO: flush and close */ }
inline void avro_writer::write(const avro_value& record)
{ /* TODO: append to current block */ ++m_count; }
template <class T>
void avro_writer::write_batch(const xarray_container<T>& data, const std::vector<std::string>& column_names)
{ /* TODO: convert array to records and write */ }
inline const avro_schema_node& avro_writer::schema() const { return m_schema; }
inline void avro_writer::flush() { /* TODO: write current block */ }
inline size_t avro_writer::record_count() const noexcept { return m_count; }

// avro_reader
inline avro_reader::avro_reader() : m_reader(nullptr), m_total(0), m_current(0) {}
inline avro_reader::~avro_reader() { close(); }
inline void avro_reader::open(const std::string& filename, const avro_reader_options& opts)
{ /* TODO: open file, read header and schema */ }
inline void avro_reader::close() { /* TODO: release resources */ }
inline bool avro_reader::read(avro_value& record)
{ /* TODO: read next record */ return false; }
inline std::vector<avro_value> avro_reader::read_all()
{ /* TODO: read all records */ return {}; }
template <class T>
xarray_container<T> avro_reader::read_to_array(const std::vector<std::string>& column_names)
{ /* TODO: read and convert to array */ return {}; }
inline const avro_schema_node& avro_reader::writer_schema() const { return m_writer_schema; }
inline const avro_schema_node& avro_reader::effective_schema() const { return m_effective_schema; }
inline std::map<std::string, std::string> avro_reader::metadata() const { return {}; }
inline size_t avro_reader::total_records() const noexcept { return m_total; }
inline size_t avro_reader::current_record() const noexcept { return m_current; }
inline void avro_reader::seek(size_t record_index) { /* TODO: seek to sync marker */ }

// schema_builder
inline schema_builder& schema_builder::set_name(const std::string& name) { return *this; }
inline schema_builder& schema_builder::set_namespace(const std::string& ns) { return *this; }
inline schema_builder& schema_builder::set_doc(const std::string& doc) { return *this; }
inline schema_builder& schema_builder::add_field(const std::string& name, avro_type type)
{ /* TODO: add field */ return *this; }
inline schema_builder& schema_builder::add_nullable_field(const std::string& name, avro_type type)
{ /* TODO: add union with null */ return *this; }
inline schema_builder& schema_builder::add_string_field(const std::string& name)
{ return add_field(name, avro_type::STRING); }
inline schema_builder& schema_builder::add_int_field(const std::string& name)
{ return add_field(name, avro_type::INT); }
inline schema_builder& schema_builder::add_long_field(const std::string& name)
{ return add_field(name, avro_type::LONG); }
inline schema_builder& schema_builder::add_float_field(const std::string& name)
{ return add_field(name, avro_type::FLOAT); }
inline schema_builder& schema_builder::add_double_field(const std::string& name)
{ return add_field(name, avro_type::DOUBLE); }
inline schema_builder& schema_builder::add_boolean_field(const std::string& name)
{ return add_field(name, avro_type::BOOLEAN); }
inline schema_builder& schema_builder::add_bytes_field(const std::string& name)
{ return add_field(name, avro_type::BYTES); }
inline schema_builder& schema_builder::add_decimal_field(const std::string& name, int precision, int scale)
{ /* TODO: add decimal logical type */ return *this; }
inline schema_builder& schema_builder::add_bignumber_field(const std::string& name)
{ return add_bytes_field(name); }
inline schema_builder& schema_builder::add_array_field(const std::string& name, avro_type items)
{ /* TODO: add array */ return *this; }
inline schema_builder& schema_builder::add_map_field(const std::string& name, avro_type values)
{ /* TODO: add map */ return *this; }
inline avro_schema_node schema_builder::build() const { return m_record; }

// Convenience
template <class T>
void save_avro(const std::string& filename, const xarray_container<T>& data,
               const avro_schema_node& schema, const avro_writer_options& opts)
{ avro_writer w; w.open(filename, schema, opts); w.write_batch(data, {}); w.close(); }
template <class T>
xarray_container<T> load_avro(const std::string& filename, const avro_reader_options& opts)
{ avro_reader r; r.open(filename, opts); return r.read_to_array<T>(); }
inline bool is_avro_file(const std::string& filename)
{ /* TODO: check magic "Obj" + 1 */ return false; }
inline avro_value array_to_avro(const xarray_container<double>& data, const std::vector<std::string>& column_names)
{ /* TODO: convert */ return {}; }
inline xarray_container<double> avro_to_array(const std::vector<avro_value>& records, const std::vector<std::string>& column_names)
{ /* TODO: convert */ return {}; }

} // namespace avro
} // namespace io
} // namespace xt

#endif // XTENSOR_XIO_AVRO_HPPshape")->type());
                    for (auto s : arr.shape())
                        shape_arr.value().push_back(avro::GenericDatum(static_cast<int32_t>(s)));
                    rec.field("shape") = shape_arr;
                    // data
                    avro::GenericArray data_arr(schema.field("data")->type());
                    for (size_t i = 0; i < arr.size(); ++i)
                    {
                        if constexpr (std::is_same_v<T, double>)
                            data_arr.value().push_back(avro::GenericDatum(static_cast<double>(arr.flat(i))));
                        else if constexpr (std::is_same_v<T, float>)
                            data_arr.value().push_back(avro::GenericDatum(static_cast<float>(arr.flat(i))));
                        else if constexpr (std::is_same_v<T, int64_t>)
                            data_arr.value().push_back(avro::GenericDatum(static_cast<int64_t>(arr.flat(i))));
                        else
                            data_arr.value().push_back(avro::GenericDatum(static_cast<double>(arr.flat(i))));
                    }
                    rec.field("data") = data_arr;
                    return avro::GenericDatum(rec);
                }

                // Avro record to xarray
                template<typename T>
                xarray_container<T> avro_to_array(const avro::GenericDatum& datum)
                {
                    if (datum.type() != avro::AVRO_RECORD)
                        throw std::runtime_error("Expected record datum");
                    const avro::GenericRecord& rec = datum.value<avro::GenericRecord>();
                    
                    // Get shape
                    const avro::GenericArray& shape_arr = rec.field("shape").value<avro::GenericArray>();
                    std::vector<size_t> shape;
                    for (const auto& d : shape_arr.value())
                        shape.push_back(static_cast<size_t>(d.value<int32_t>()));
                    
                    // Get data
                    const avro::GenericArray& data_arr = rec.field("data").value<avro::GenericArray>();
                    xarray_container<T> result(shape);
                    for (size_t i = 0; i < data_arr.value().size(); ++i)
                    {
                        const auto& d = data_arr.value()[i];
                        if constexpr (std::is_same_v<T, double>)
                            result.flat(i) = static_cast<T>(d.value<double>());
                        else if constexpr (std::is_same_v<T, float>)
                            result.flat(i) = static_cast<T>(d.value<float>());
                        else if constexpr (std::is_same_v<T, int64_t>)
                            result.flat(i) = static_cast<T>(d.value<int64_t>());
                        else
                            result.flat(i) = static_cast<T>(d.value<double>());
                    }
                    return result;
                }
            }
#endif // XTENSOR_HAS_AVRO

            // --------------------------------------------------------------------
            // Avro Writer
            // --------------------------------------------------------------------
            class AvroWriter
            {
            public:
                AvroWriter() = default;

                void open(const std::string& filename, const AvroField& schema = {})
                {
#if XTENSOR_HAS_AVRO
                    m_filename = filename;
                    if (schema.type != AvroType::NULL_TYPE)
                    {
                        m_schema = avro_detail::build_schema(schema);
                    }
                    // Writer will be created on first write or explicitly
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                void open_with_json_schema(const std::string& filename, const std::string& json_schema)
                {
#if XTENSOR_HAS_AVRO
                    m_filename = filename;
                    m_schema = avro::compileJsonSchemaFromString(json_schema);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                template<typename T>
                void write(const xarray_container<T>& data)
                {
#if XTENSOR_HAS_AVRO
                    if (!m_writer)
                    {
                        if (!m_schema.valid())
                        {
                            // Auto-generate schema for tensor
                            m_schema = avro_detail::array_to_avro(data).schema();
                        }
                        m_writer = std::make_unique<avro::DataFileWriter<avro::GenericDatum>>(
                            m_filename.c_str(), m_schema);
                    }
                    auto datum = avro_detail::array_to_avro(data);
                    m_writer->write(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                void write_generic(const JsonValue& record)
                {
#if XTENSOR_HAS_AVRO
                    if (!m_writer)
                        throw std::runtime_error("Writer not initialized with schema");
                    auto datum = avro_detail::json_to_datum(record, m_schema);
                    m_writer->write(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                void close()
                {
#if XTENSOR_HAS_AVRO
                    if (m_writer)
                    {
                        m_writer->close();
                        m_writer.reset();
                    }
#endif
                }

                ~AvroWriter()
                {
                    close();
                }

            private:
                std::string m_filename;
#if XTENSOR_HAS_AVRO
                avro::ValidSchema m_schema;
                std::unique_ptr<avro::DataFileWriter<avro::GenericDatum>> m_writer;
#endif
            };

            // --------------------------------------------------------------------
            // Avro Reader
            // --------------------------------------------------------------------
            class AvroReader
            {
            public:
                AvroReader() = default;

                void open(const std::string& filename)
                {
#if XTENSOR_HAS_AVRO
                    m_filename = filename;
                    m_reader = std::make_unique<avro::DataFileReader<avro::GenericDatum>>(filename.c_str());
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
#endif
                }

                bool has_next() const
                {
#if XTENSOR_HAS_AVRO
                    return m_reader && m_reader->hasMore();
#else
                    return false;
#endif
                }

                template<typename T>
                xarray_container<T> read_next()
                {
#if XTENSOR_HAS_AVRO
                    if (!m_reader || !m_reader->hasMore())
                        throw std::runtime_error("No more records");
                    avro::GenericDatum datum;
                    m_reader->read(datum);
                    return avro_detail::avro_to_array<T>(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
                    return {};
#endif
                }

                JsonValue read_next_generic()
                {
#if XTENSOR_HAS_AVRO
                    if (!m_reader || !m_reader->hasMore())
                        throw std::runtime_error("No more records");
                    avro::GenericDatum datum;
                    m_reader->read(datum);
                    return avro_detail::datum_to_json(datum);
#else
                    XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
                    return JsonValue();
#endif
                }

                template<typename T>
                std::vector<xarray_container<T>> read_all()
                {
                    std::vector<xarray_container<T>> result;
                    while (has_next())
                        result.push_back(read_next<T>());
                    return result;
                }

                avro::ValidSchema get_schema() const
                {
#if XTENSOR_HAS_AVRO
                    if (!m_reader) throw std::runtime_error("Reader not open");
                    return m_reader->readerSchema();
#else
                    throw std::runtime_error("Avro support not compiled");
#endif
                }

                void close()
                {
#if XTENSOR_HAS_AVRO
                    if (m_reader)
                    {
                        m_reader->close();
                        m_reader.reset();
                    }
#endif
                }

                ~AvroReader()
                {
                    close();
                }

            private:
                std::string m_filename;
#if XTENSOR_HAS_AVRO
                std::unique_ptr<avro::DataFileReader<avro::GenericDatum>> m_reader;
#endif
            };

            // --------------------------------------------------------------------
            // Convenience functions
            // --------------------------------------------------------------------
            template<typename T>
            inline void save_avro(const std::string& filename, const xarray_container<T>& data)
            {
                AvroWriter writer;
                writer.open(filename);
                writer.write(data);
                writer.close();
            }

            template<typename T>
            inline void save_avro_batch(const std::string& filename, const std::vector<xarray_container<T>>& batches)
            {
                AvroWriter writer;
                writer.open(filename);
                for (const auto& arr : batches)
                    writer.write(arr);
                writer.close();
            }

            template<typename T>
            inline std::vector<xarray_container<T>> load_avro(const std::string& filename)
            {
                AvroReader reader;
                reader.open(filename);
                return reader.read_all<T>();
            }

            inline JsonValue load_avro_schema(const std::string& filename)
            {
#if XTENSOR_HAS_AVRO
                AvroReader reader;
                reader.open(filename);
                auto schema = reader.get_schema();
                return JsonParser::parse(schema.toJson());
#else
                XTENSOR_THROW(std::runtime_error, "Avro support not compiled");
                return JsonValue();
#endif
            }

            // Write multiple named arrays as Avro records (each record is a map)
            inline void save_avro_dict(const std::string& filename,
                                       const std::map<std::string, xarray_container<double>>& dict)
            {
                // Build schema: record with fields for each key (all arrays)
                AvroField record_schema{"", AvroType::RECORD};
                for (const auto& p : dict)
                {
                    AvroField field;
                    field.name = p.first;
                    field.type = AvroType::ARRAY;
                    field.items_type = AvroType::DOUBLE;
                    record_schema.fields.push_back(field);
                }
                AvroWriter writer;
                writer.open(filename, record_schema);
                // Create a JSON record
                JsonValue::Object jobj;
                for (const auto& p : dict)
                {
                    JsonValue::Array jarr;
                    for (size_t i = 0; i < p.second.size(); ++i)
                        jarr.push_back(JsonValue(p.second.flat(i)));
                    jobj[p.first] = JsonValue(jarr);
                }
                writer.write_generic(JsonValue(jobj));
                writer.close();
            }

            inline std::map<std::string, xarray_container<double>> load_avro_dict(const std::string& filename)
            {
                std::map<std::string, xarray_container<double>> result;
#if XTENSOR_HAS_AVRO
                AvroReader reader;
                reader.open(filename);
                if (!reader.has_next())
                    return result;
                JsonValue rec = reader.read_next_generic();
                if (rec.is_object())
                {
                    for (const auto& p : rec.as_object())
                    {
                        if (p.second.is_array())
                        {
                            const auto& arr = p.second.as_array();
                            xarray_container<double> vec({arr.size()});
                            for (size_t i = 0; i < arr.size(); ++i)
                                vec(i) = arr[i].as_double();
                            result[p.first] = vec;
                        }
                    }
                }
#endif
                return result;
            }

        } // namespace io

        // Bring Avro functions into xt namespace
        using io::AvroWriter;
        using io::AvroReader;
        using io::save_avro;
        using io::save_avro_batch;
        using io::load_avro;
        using io::load_avro_schema;
        using io::save_avro_dict;
        using io::load_avro_dict;
        using io::AvroField;
        using io::AvroType;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIO_AVRO_HPP

// io/xio_avro.hpp